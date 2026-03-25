#!/usr/bin/env julia
# Odin.jl inference benchmarks — particle filter & MCMC
#
# Run: julia --project=. benchmark/benchmark_inference_julia.jl

using Odin
using BenchmarkTools
using Statistics
using Printf
using Random

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 15

println("=" ^ 72)
println("Odin.jl Inference Benchmarks")
println("=" ^ 72)

# ── Model definition ────────────────────────────────────────────

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

# ── Generate synthetic data ─────────────────────────────────────

true_pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
n_days = 100
times = collect(0.0:1.0:Float64(n_days))
sim = dust_system_simulate(sir, true_pars; times=times, dt=1.0, seed=1)
obs = round.(sim[4, 1, 2:end])

data_rows = [(time=Float64(t), cases=Float64(obs[t])) for t in 1:n_days]
fdata = Odin.dust_filter_data(data_rows)

# ══════════════════════════════════════════════════════════════════
# PART 1: PARTICLE FILTER — vary n_particles
# ══════════════════════════════════════════════════════════════════

println("\n" * "─" ^ 72)
println("PART 1: Particle Filter  (100 days, varying n_particles)")
println("─" ^ 72)

pf_results = []
for np in [50, 100, 200, 500, 1000, 2000]
    filter = dust_filter_create(sir, fdata; n_particles=np, dt=1.0, seed=42)
    # warmup
    dust_likelihood_run!(filter, true_pars)
    dust_likelihood_run!(filter, true_pars)

    b = @benchmark dust_likelihood_run!($filter, $true_pars) samples=30
    med = median(b).time / 1e6
    mn = minimum(b).time / 1e6
    mx = maximum(b).time / 1e6
    allocs = median(b).allocs
    @printf("  n_particles=%5d  →  %8.2f ms  (min %7.2f, max %8.2f)  allocs=%d\n",
            np, med, mn, mx, allocs)
    push!(pf_results, (n_particles=np, median_ms=med, min_ms=mn, max_ms=mx, allocs=allocs))
end

# ── Verify filter correctness ───────────────────────────────────

filter_check = dust_filter_create(sir, fdata; n_particles=1000, dt=1.0, seed=42)
lls = [dust_likelihood_run!(filter_check, true_pars) for _ in 1:20]
println("\n  Log-likelihood check (1000 particles, 20 runs):")
@printf("    mean = %.2f  ±  %.2f\n", mean(lls), std(lls))

# ══════════════════════════════════════════════════════════════════
# PART 2: MCMC — vary n_steps, n_chains
# ══════════════════════════════════════════════════════════════════

println("\n" * "─" ^ 72)
println("PART 2: MCMC Sampling  (RW sampler + particle filter likelihood)")
println("─" ^ 72)

packer = monty_packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))
filter_mcmc = dust_filter_create(sir, fdata; n_particles=200, dt=1.0, seed=42)
likelihood = dust_likelihood_monty(filter_mcmc, packer)
prior = @monty_prior begin
    beta ~ Gamma(2.0, 0.25)
    gamma ~ Gamma(2.0, 0.05)
end
posterior = likelihood + prior
sampler = monty_sampler_random_walk([0.005 0.0; 0.0 0.001])
initial_mat = reshape([0.4, 0.08], 2, 1)

# warmup
monty_sample(posterior, sampler, 10; initial=initial_mat, n_chains=1)

mcmc_results = []

# 2a: Vary n_steps (1 chain)
println("\n  2a: Vary n_steps (1 chain, 200 particles)")
for ns in [100, 500, 1000, 2000]
    b = @benchmark monty_sample($posterior, $sampler, $ns;
        initial=$initial_mat, n_chains=1) samples=5 seconds=60
    med = median(b).time / 1e6
    mn = minimum(b).time / 1e6
    mx = maximum(b).time / 1e6
    @printf("    n_steps=%5d  →  %10.1f ms  (min %9.1f, max %10.1f)\n",
            ns, med, mn, mx)
    push!(mcmc_results, (config="steps_$(ns)_1chain", n_steps=ns, n_chains=1,
                         median_ms=med, min_ms=mn, max_ms=mx))
end

# 2b: Vary n_chains (500 steps each)
println("\n  2b: Vary n_chains (500 steps, 200 particles)")
for nc in [1, 2, 4]
    init = repeat([0.4, 0.08], 1, nc)
    b = @benchmark monty_sample($posterior, $sampler, 500;
        initial=$init, n_chains=$nc) samples=5 seconds=60
    med = median(b).time / 1e6
    mn = minimum(b).time / 1e6
    mx = maximum(b).time / 1e6
    @printf("    n_chains=%d  →  %10.1f ms  (min %9.1f, max %10.1f)\n",
            nc, med, mn, mx)
    push!(mcmc_results, (config="500steps_$(nc)chain", n_steps=500, n_chains=nc,
                         median_ms=med, min_ms=mn, max_ms=mx))
end

# 2c: Effect of n_particles on MCMC
println("\n  2c: Vary n_particles in MCMC (500 steps, 1 chain)")
for np in [50, 200, 500, 1000]
    f = dust_filter_create(sir, fdata; n_particles=np, dt=1.0, seed=42)
    ll = dust_likelihood_monty(f, packer)
    post = ll + prior
    # warmup
    monty_sample(post, sampler, 10; initial=initial_mat, n_chains=1)
    b = @benchmark monty_sample($post, $sampler, 500;
        initial=$initial_mat, n_chains=1) samples=5 seconds=60
    med = median(b).time / 1e6
    @printf("    n_particles=%5d  →  %10.1f ms\n", np, med)
    push!(mcmc_results, (config="500steps_$(np)part", n_steps=500, n_chains=1,
                         median_ms=med, min_ms=minimum(b).time/1e6,
                         max_ms=maximum(b).time/1e6))
end

# ── Summary ──────────────────────────────────────────────────────

println("\n" * "=" ^ 72)
println("SUMMARY")
println("=" ^ 72)

println("\nParticle Filter (100 days):")
@printf("  %-20s  %10s  %10s\n", "n_particles", "median_ms", "ms/particle")
for r in pf_results
    @printf("  %-20d  %10.2f  %10.4f\n", r.n_particles, r.median_ms,
            r.median_ms / r.n_particles)
end

println("\nMCMC:")
@printf("  %-25s  %10s\n", "config", "median_ms")
for r in mcmc_results
    @printf("  %-25s  %10.1f\n", r.config, r.median_ms)
end

# Save results
open("benchmark/results_inference_julia.csv", "w") do f
    println(f, "type,config,median_ms,min_ms,max_ms")
    for r in pf_results
        println(f, "pf,np_$(r.n_particles),$(r.median_ms),$(r.min_ms),$(r.max_ms)")
    end
    for r in mcmc_results
        println(f, "mcmc,$(r.config),$(r.median_ms),$(r.min_ms),$(r.max_ms)")
    end
end
println("\nResults saved to benchmark/results_inference_julia.csv")
