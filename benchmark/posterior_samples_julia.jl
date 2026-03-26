#!/usr/bin/env julia
# Save posterior samples from Julia MCMC for cross-language ECDF comparison
#
# Run: julia --project=. benchmark/posterior_samples_julia.jl

using Odin
using CSV, DataFrames
using Statistics
using Random

println("Generating Julia posterior samples for ECDF comparison...")

sir = @odin begin
    update(S) = S - n_SI
    update(I) = I + n_SI - n_IR
    update(R) = R + n_IR
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    initial(incidence, zero_every = 1) = 0
    update(incidence) = incidence + n_SI
    p_SI = 1 - exp(-beta * I / N * dt)
    p_IR = 1 - exp(-gamma * dt)
    n_SI = Binomial(S, p_SI)
    n_IR = Binomial(I, p_IR)
    cases = data()
    cases ~ Poisson(incidence + 1e-6)
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10)
    N = parameter(1000)
end

# Load the SAME data that R generated (ensures identical observations)
shared_data = CSV.read("benchmark/shared_data.csv", DataFrame)
println("  Loaded shared_data.csv ($(nrow(shared_data)) obs)")

true_pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)

fdata = Odin.ObservedData(
    [(time=Float64(row.time), cases=Float64(row.cases)) for row in eachrow(shared_data)]
)

# Run MCMC with same configuration as R
filter = Likelihood(sir, fdata; n_particles=500, dt=1.0, seed=42)
packer = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))
likelihood = as_model(filter, packer)
prior = @prior begin
    beta ~ Gamma(2.0, 0.25)
    gamma ~ Gamma(2.0, 0.05)
end
posterior = likelihood + prior
sampler = random_walk([0.005 0.0; 0.0 0.001])
initial_mat = repeat([0.4, 0.08], 1, 4)

println("  Running MCMC: 5000 steps × 4 chains, 500 particles...")
t0 = time()
samples = sample(posterior, sampler, 5000;
                       initial=initial_mat, n_chains=4)
elapsed = time() - t0
println("  Done in $(round(elapsed, digits=1)) seconds")

# Extract samples: samples.pars is [n_pars, n_steps, n_chains]
burnin = 1000
n_steps = size(samples.pars, 2)
n_chains = size(samples.pars, 3)
keep = (burnin+1):n_steps
beta_post = vec(samples.pars[1, keep, :])
gamma_post = vec(samples.pars[2, keep, :])

println("  Post burn-in samples: $(length(beta_post)) (per parameter)")
println("  beta:  mean=$(round(mean(beta_post), digits=4))  " *
        "sd=$(round(std(beta_post), digits=4))  " *
        "[$(round(quantile(beta_post, 0.025), digits=4)), " *
        "$(round(quantile(beta_post, 0.975), digits=4))]")
println("  gamma: mean=$(round(mean(gamma_post), digits=4))  " *
        "sd=$(round(std(gamma_post), digits=4))  " *
        "[$(round(quantile(gamma_post, 0.025), digits=4)), " *
        "$(round(quantile(gamma_post, 0.975), digits=4))]")

# Save posterior samples
CSV.write("benchmark/posterior_julia.csv",
          DataFrame(beta=beta_post, gamma=gamma_post))
println("  Saved posterior_julia.csv")

# Also save log-likelihood distribution (use unseeded filter for variability)
println("  Computing log-likelihood distribution (500 particles × 100 runs)...")
ll_vals = Float64[]
for i in 1:100
    f_tmp = Likelihood(sir, fdata; n_particles=500, dt=1.0, seed=nothing)
    push!(ll_vals, loglik(f_tmp, true_pars))
end
CSV.write("benchmark/ll_dist_julia.csv", DataFrame(ll=ll_vals))
println("  LL: mean=$(round(mean(ll_vals), digits=2))  sd=$(round(std(ll_vals), digits=2))")
println("  Saved ll_dist_julia.csv")
println("Done.")
