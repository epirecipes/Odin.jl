#!/usr/bin/env julia
# Odin.jl benchmarks — comparable to the R benchmarks in benchmark_r.R
#
# Run: julia --project=. benchmark/benchmark_julia.jl

using Odin
using BenchmarkTools
using Statistics
using Printf

BenchmarkTools.DEFAULT_PARAMETERS.samples = 50
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

println("=" ^ 70)
println("Odin.jl Benchmarks")
println("=" ^ 70)

# ── 1. ODE SIR simulation ──────────────────────────────────────

sir_ode = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10)
    N = parameter(1000)
end

pars_ode = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
times_ode = collect(0.0:1.0:365.0)

# Warmup
simulate(sir_ode, pars_ode; times=times_ode, seed=1)

println("\n1. ODE SIR — simulate 365 days, 1 particle")
b1 = @benchmark simulate($sir_ode, $pars_ode; times=$times_ode, seed=1)
display(b1)

# ── 2. Stochastic SIR simulation ───────────────────────────────

sir_stoch = @odin begin
    update(S) = S - n_SI
    update(I) = I + n_SI - n_IR
    update(R) = R + n_IR
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    p_SI = 1 - exp(-beta * I / N * dt)
    p_IR = 1 - exp(-gamma * dt)
    n_SI = Binomial(S, p_SI)
    n_IR = Binomial(I, p_IR)
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10)
    N = parameter(1000)
end

pars_stoch = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
times_stoch = collect(0.0:1.0:365.0)

# Warmup
simulate(sir_stoch, pars_stoch; times=times_stoch, dt=1.0, seed=1, n_particles=100)

println("\n2. Stochastic SIR — 365 days, 100 particles")
b2 = @benchmark simulate($sir_stoch, $pars_stoch;
    times=$times_stoch, dt=1.0, seed=1, n_particles=100)
display(b2)

# ── 3. Age-structured ODE SIR ──────────────────────────────────

sir_age = @odin begin
    n_age = parameter(10)
    dim(S) = n_age
    dim(I) = n_age
    dim(R) = n_age
    dim(beta) = n_age
    dim(S0) = n_age
    dim(I0) = n_age

    deriv(S[i]) = -beta[i] * S[i] * total_I / N
    deriv(I[i]) = beta[i] * S[i] * total_I / N - gamma * I[i]
    deriv(R[i]) = gamma * I[i]

    total_I = sum(I)

    initial(S[i]) = S0[i]
    initial(I[i]) = I0[i]
    initial(R[i]) = 0

    S0 = parameter()
    I0 = parameter()
    beta = parameter()
    gamma = parameter(0.1)
    N = parameter(10000)
end

n_age = 10
pars_age = (
    n_age = Float64(n_age),
    S0 = fill(990.0, n_age),
    I0 = fill(10.0, n_age),
    beta = range(0.2, 0.6, length=n_age) |> collect |> Vector{Float64},
    gamma = 0.1,
    N = 10000.0,
)
times_age = collect(0.0:1.0:365.0)

# Warmup
simulate(sir_age, pars_age; times=times_age, seed=1)

println("\n3. Age-structured ODE SIR (10 groups) — 365 days")
b3 = @benchmark simulate($sir_age, $pars_age; times=$times_age, seed=1)
display(b3)

# ── 4. Particle filter ─────────────────────────────────────────

sir_pf = @odin begin
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

# Generate data
pars_pf = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
times_pf = collect(0.0:1.0:100.0)
sim_data = simulate(sir_pf, pars_pf; times=times_pf, dt=1.0, seed=1)
obs = round.(sim_data[4, 1, 2:end])

fdata = Odin.ObservedData(
    [(time=Float64(t), cases=Float64(c)) for (t, c) in zip(times_pf[2:end], obs)]
)

filter = Likelihood(sir_pf, fdata; n_particles=200, dt=1.0, seed=42)

# Warmup
loglik(filter, pars_pf)

println("\n4. Particle filter — 100 days, 200 particles")
b4 = @benchmark loglik($filter, $pars_pf)
display(b4)

# ── 5. MCMC sampling (short chain) ─────────────────────────────

packer = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))
likelihood = as_model(filter, packer)
prior = @prior begin
    beta ~ Gamma(2.0, 0.25)
    gamma ~ Gamma(2.0, 0.05)
end
posterior = likelihood + prior
sampler = random_walk([0.005 0.0; 0.0 0.001])
initial_mat = collect([0.4 0.08]')

# Warmup
sample(posterior, sampler, 50; initial=initial_mat, n_chains=1)

println("\n5. MCMC — 500 iterations, RW sampler, particle filter likelihood")
b5 = @benchmark sample($posterior, $sampler, 500;
    initial=$initial_mat, n_chains=1) samples=10
display(b5)

# ── Summary table ───────────────────────────────────────────────

println("\n" * "=" ^ 70)
println("Summary (median times)")
println("=" ^ 70)
results = [
    ("ODE SIR (365d, 1 part.)",     median(b1).time / 1e6),
    ("Stochastic SIR (365d, 100p)", median(b2).time / 1e6),
    ("Age-struct ODE (10grp, 365d)", median(b3).time / 1e6),
    ("Particle filter (100d, 200p)", median(b4).time / 1e6),
    ("MCMC 500 iter (PF + RW)",     median(b5).time / 1e6),
]
for (name, ms) in results
    @printf("  %-35s %10.2f ms\n", name, ms)
end
println("=" ^ 70)

# Save results for comparison
open("benchmark/results_julia.csv", "w") do f
    println(f, "benchmark,median_ms,min_ms,max_ms")
    benchmarks = [b1, b2, b3, b4, b5]
    names = ["ode_sir", "stoch_sir", "age_sir", "particle_filter", "mcmc_500"]
    for (n, b) in zip(names, benchmarks)
        med = median(b).time / 1e6
        mn = minimum(b).time / 1e6
        mx = maximum(b).time / 1e6
        println(f, "$n,$med,$mn,$mx")
    end
end
println("\nResults saved to benchmark/results_julia.csv")
