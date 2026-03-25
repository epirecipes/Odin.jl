#!/usr/bin/env julia
# Benchmark: Julia Odin.jl symbolic adjoint HMC/NUTS vs R odin2/dust2/monty
#
# Compares:
#   1. Julia NUTS with symbolic adjoint VJP
#   2. Julia NUTS without symbolic (ForwardDiff fallback)
#   3. Julia HMC with symbolic adjoint
#   4. R HMC with compiled C++ adjoint
#   5. R NUTS with compiled C++ adjoint
#
# All use the same SIR ODE model fit to the same synthetic data.

using Odin
using Random
using Statistics
using Printf
using Distributions
using DelimitedFiles

# ──────────────────────────────────────────────────────────────
# 1. Define models (with and without symbolic diff)
# ──────────────────────────────────────────────────────────────

sir_symbolic = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    N = parameter(1000.0)
    I0 = parameter(10.0)
    beta = parameter(0.5, differentiate = true)
    gamma = parameter(0.1, differentiate = true)
    rho = parameter(0.3, differentiate = true)
    cases_lambda = I > 0 ? rho * I : 1e-10
    cases ~ Poisson(cases_lambda)
end

sir_nosym = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    N = parameter(1000.0)
    I0 = parameter(10.0)
    beta = parameter(0.5)
    gamma = parameter(0.1)
    rho = parameter(0.3)
    cases_lambda = I > 0 ? rho * I : 1e-10
    cases ~ Poisson(cases_lambda)
end

println("Symbolic Jacobian (sir_symbolic): ",
        Odin._odin_has_symbolic_jacobian(sir_symbolic.model))
println("Symbolic Jacobian (sir_nosym):    ",
        Odin._odin_has_symbolic_jacobian(sir_nosym.model))

# ──────────────────────────────────────────────────────────────
# 2. Generate synthetic data
# ──────────────────────────────────────────────────────────────

true_pars = (beta=0.5, gamma=0.1, rho=0.3, I0=10.0, N=1000.0)
times = collect(1.0:1.0:50.0)

sys = dust_system_create(sir_symbolic, true_pars; n_particles=1, seed=42)
dust_system_set_state_initial!(sys)
sim = dust_system_simulate(sys, times)

# Extract I compartment and simulate Poisson observations
rng = MersenneTwister(123)
I_trajectory = [sim[2, 1, i] for i in 1:length(times)]
obs_cases = [rand(rng, Poisson(max(1e-10, 0.3 * I_trajectory[i]))) for i in 1:length(times)]
obs_cases = max.(obs_cases, 1)

data_rows = [(time=times[i], cases=Float64(obs_cases[i])) for i in 1:length(times)]
fdata = dust_filter_data(data_rows; time_field=:time)

# Save data for R benchmark
open(joinpath(@__DIR__, "benchmark_hmc_data.csv"), "w") do io
    println(io, "time,cases")
    for i in 1:length(times)
        println(io, "$(Int(times[i])),$(obs_cases[i])")
    end
end

println("\nData summary: $(length(times)) time points, ",
        "mean cases=$(round(mean(obs_cases), digits=1))")

# ──────────────────────────────────────────────────────────────
# 3. Setup posterior
# ──────────────────────────────────────────────────────────────

function make_posterior(sir_gen, fdata)
    uf = dust_unfilter_create(sir_gen, fdata)
    packer = monty_packer([:beta, :gamma, :rho]; fixed=(N=1000.0, I0=10.0))
    ll = dust_likelihood_monty(uf, packer)

    prior = monty_model(
        x -> begin
            b, g, r = x
            (b <= 0 || g <= 0 || r <= 0 || r >= 1) && return -Inf
            logpdf(Exponential(1.0), b) + logpdf(Exponential(1.0), g) +
                logpdf(Beta(2, 5), r)
        end;
        parameters=["beta", "gamma", "rho"],
        gradient=x -> begin
            b, g, r = x
            (b <= 0 || g <= 0 || r <= 0 || r >= 1) && return [0.0, 0.0, 0.0]
            # d/dx log(Exp(1)) = -1, d/dx log(Beta(2,5)) = 1/x - 4/(1-x)
            [-1.0/b + 1.0/b - 1.0,  # d/db logpdf(Exp(1), b) = 1/b - 1 ... wait
             -1.0/g + 1.0/g - 1.0,  # same
             (2-1)/r - (5-1)/(1-r)]   # Beta(2,5) gradient
        end,
        domain=[0.0 Inf; 0.0 Inf; 0.0 1.0],
    )

    return ll + prior
end

# Fix the prior gradient — Exponential(rate=1) has logpdf = -x, gradient = -1
function make_posterior_v2(sir_gen, fdata)
    uf = dust_unfilter_create(sir_gen, fdata)
    packer = monty_packer([:beta, :gamma, :rho]; fixed=(N=1000.0, I0=10.0))
    ll = dust_likelihood_monty(uf, packer)

    prior = monty_model(
        x -> begin
            b, g, r = x
            (b <= 0 || g <= 0 || r <= 0 || r >= 1) && return -Inf
            logpdf(Exponential(1.0), b) + logpdf(Exponential(1.0), g) +
                logpdf(Beta(2, 5), r)
        end;
        parameters=["beta", "gamma", "rho"],
        gradient=x -> begin
            b, g, r = x
            (b <= 0 || g <= 0 || r <= 0 || r >= 1) && return [0.0, 0.0, 0.0]
            [-1.0, -1.0, (2-1)/r - (5-1)/(1-r)]
        end,
        domain=[0.0 Inf; 0.0 Inf; 0.0 1.0],
    )

    return ll + prior
end

post_sym = make_posterior_v2(sir_symbolic, fdata)
post_nosym = make_posterior_v2(sir_nosym, fdata)

# ──────────────────────────────────────────────────────────────
# 4. Benchmark settings
# ──────────────────────────────────────────────────────────────

n_steps = 2000
n_chains = 4
n_burnin = 500

initial_mat = [0.4 0.45 0.55 0.6;
               0.08 0.09 0.11 0.12;
               0.25 0.28 0.32 0.35]

# ──────────────────────────────────────────────────────────────
# 5. Run benchmarks
# ──────────────────────────────────────────────────────────────

results = Dict{String, Any}()

function compute_ess(samples_arr, n_burnin)
    # Simple batch-means ESS
    n_total, n_chains = size(samples_arr, 2), size(samples_arr, 3)
    post = samples_arr[:, (n_burnin+1):end, :]
    n_post = size(post, 2)
    n_params = size(post, 1)
    ess = zeros(n_params)
    for p in 1:n_params
        all_vals = vec(post[p, :, :])
        n = length(all_vals)
        m = mean(all_vals)
        v = var(all_vals)
        if v < 1e-20
            ess[p] = 0.0
            continue
        end
        # Batch means with batch size sqrt(n)
        batch_size = max(1, Int(floor(sqrt(n))))
        n_batches = div(n, batch_size)
        batch_means = [mean(all_vals[((b-1)*batch_size+1):(b*batch_size)]) for b in 1:n_batches]
        var_bm = var(batch_means) * batch_size
        ess[p] = n * v / max(var_bm, 1e-20)
    end
    return ess
end

# ── NUTS with symbolic adjoint ──
println("\n=== Julia NUTS (symbolic adjoint) ===")
nuts_sym = monty_sampler_nuts(target_acceptance=0.8, n_adaption=n_burnin, metric=:dense)
# Warmup run
_ = monty_sample(post_sym, nuts_sym, 50; n_chains=1,
                 initial=initial_mat[:, 1:1], n_burnin=20, seed=1)

t_sym = @elapsed begin
    res_nuts_sym = monty_sample(post_sym, nuts_sym, n_steps; n_chains=n_chains,
                                initial=initial_mat, n_burnin=n_burnin, seed=42)
end
ess_sym = compute_ess(res_nuts_sym.pars, n_burnin)
println("  Time: $(round(t_sym, digits=2))s")
println("  ESS:  beta=$(round(ess_sym[1], digits=1)), gamma=$(round(ess_sym[2], digits=1)), rho=$(round(ess_sym[3], digits=1))")
println("  ESS/s: $(round.(ess_sym ./ t_sym, digits=1))")
results["NUTS_symbolic"] = (time=t_sym, ess=ess_sym)

# ── NUTS without symbolic (ForwardDiff fallback) ──
println("\n=== Julia NUTS (ForwardDiff fallback) ===")
nuts_nosym = monty_sampler_nuts(target_acceptance=0.8, n_adaption=n_burnin, metric=:dense)
_ = monty_sample(post_nosym, nuts_nosym, 50; n_chains=1,
                 initial=initial_mat[:, 1:1], n_burnin=20, seed=1)

t_nosym = @elapsed begin
    res_nuts_nosym = monty_sample(post_nosym, nuts_nosym, n_steps; n_chains=n_chains,
                                  initial=initial_mat, n_burnin=n_burnin, seed=42)
end
ess_nosym = compute_ess(res_nuts_nosym.pars, n_burnin)
println("  Time: $(round(t_nosym, digits=2))s")
println("  ESS:  beta=$(round(ess_nosym[1], digits=1)), gamma=$(round(ess_nosym[2], digits=1)), rho=$(round(ess_nosym[3], digits=1))")
println("  ESS/s: $(round.(ess_nosym ./ t_nosym, digits=1))")
results["NUTS_forwarddiff"] = (time=t_nosym, ess=ess_nosym)

# ── HMC with symbolic adjoint ──
println("\n=== Julia HMC (symbolic adjoint) ===")
hmc_sym = monty_sampler_hmc(0.005, 20)
_ = monty_sample(post_sym, hmc_sym, 50; n_chains=1,
                 initial=initial_mat[:, 1:1], n_burnin=20, seed=1)

t_hmc = @elapsed begin
    res_hmc = monty_sample(post_sym, hmc_sym, n_steps; n_chains=n_chains,
                            initial=initial_mat, n_burnin=n_burnin, seed=42)
end
ess_hmc = compute_ess(res_hmc.pars, n_burnin)
println("  Time: $(round(t_hmc, digits=2))s")
println("  ESS:  beta=$(round(ess_hmc[1], digits=1)), gamma=$(round(ess_hmc[2], digits=1)), rho=$(round(ess_hmc[3], digits=1))")
println("  ESS/s: $(round.(ess_hmc ./ t_hmc, digits=1))")
results["HMC_symbolic"] = (time=t_hmc, ess=ess_hmc)

# ── Random Walk (baseline) ──
println("\n=== Julia Random Walk (baseline) ===")
vcv = [0.002 0.0 0.0; 0.0 0.0005 0.0; 0.0 0.0 0.003]
rw = monty_sampler_random_walk(vcv)

t_rw = @elapsed begin
    res_rw = monty_sample(post_sym, rw, n_steps; n_chains=n_chains,
                           initial=initial_mat, n_burnin=n_burnin, seed=42)
end
ess_rw = compute_ess(res_rw.pars, n_burnin)
println("  Time: $(round(t_rw, digits=2))s")
println("  ESS:  beta=$(round(ess_rw[1], digits=1)), gamma=$(round(ess_rw[2], digits=1)), rho=$(round(ess_rw[3], digits=1))")
println("  ESS/s: $(round.(ess_rw ./ t_rw, digits=1))")
results["RW"] = (time=t_rw, ess=ess_rw)

# ──────────────────────────────────────────────────────────────
# 6. Summary table
# ──────────────────────────────────────────────────────────────

println("\n" * "="^80)
println("BENCHMARK SUMMARY: SIR ODE — $(n_steps) steps × $(n_chains) chains")
println("="^80)
@printf("%-25s %8s %8s %8s %8s %8s %8s\n",
        "Sampler", "Time(s)", "ESS_β", "ESS_γ", "ESS_ρ", "ESS/s_β", "ESS/s_γ")
println("-"^80)

for (name, r) in sort(collect(results), by=x->x.second.time)
    @printf("%-25s %8.2f %8.1f %8.1f %8.1f %8.1f %8.1f\n",
            name, r.time, r.ess[1], r.ess[2], r.ess[3],
            r.ess[1]/r.time, r.ess[2]/r.time)
end

# ──────────────────────────────────────────────────────────────
# 7. Save results
# ──────────────────────────────────────────────────────────────

outfile = joinpath(@__DIR__, "results_hmc_benchmark_julia.csv")
open(outfile, "w") do io
    println(io, "sampler,time_sec,param,ess,ess_per_sec")
    for (name, r) in results
        for (i, pname) in enumerate(["beta", "gamma", "rho"])
            @printf(io, "%s,%.4f,%s,%.2f,%.2f\n",
                    name, r.time, pname, r.ess[i], r.ess[i]/r.time)
        end
    end
end
println("\nJulia results saved to: $outfile")

# Also save posterior means for comparison
println("\nPosterior means (NUTS symbolic):")
post_samples = res_nuts_sym.pars[:, (n_burnin+1):end, :]
for (i, pname) in enumerate(["beta", "gamma", "rho"])
    vals = vec(post_samples[i, :, :])
    @printf("  %s: mean=%.4f, std=%.4f (true=%.1f)\n",
            pname, mean(vals), std(vals),
            [0.5, 0.1, 0.3][i])
end
