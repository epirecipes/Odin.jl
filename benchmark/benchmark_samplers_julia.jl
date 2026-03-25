#!/usr/bin/env julia
# Odin.jl sampler ESS benchmarks — compare RW, HMC, NUTS with ESS/time metrics
#
# Run: julia --project=. benchmark/benchmark_samplers_julia.jl

using Odin
using MCMCChains
using Statistics
using Printf
using Random
using LinearAlgebra

"""Convert MontySamples to MCMCChains.Chains, recording wall time for ESS/sec."""
function to_chains(samples::MontySamples; elapsed_sec::Float64=NaN)
    n_pars, n_samples, n_chains = size(samples.pars)
    # MCMCChains expects (n_samples × n_pars × n_chains)
    vals = permutedims(samples.pars, (2, 1, 3))
    param_names = Symbol.(samples.parameter_names)
    info = isnan(elapsed_sec) ? NamedTuple() : (start_time=0.0, stop_time=elapsed_sec)
    return Chains(vals, param_names; info=info)
end

println("=" ^ 72)
println("Odin.jl Sampler ESS Benchmarks")
println("=" ^ 72)

# ── ODE SIR Model definition ──────────────────────────────────

sir = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0.0
    N = parameter(1000.0)
    I0 = parameter(10.0)
    beta = parameter(0.5)
    gamma = parameter(0.1)
    cases_lambda = I > 0 ? rho * I : 1e-10
    cases ~ Poisson(cases_lambda)
    rho = parameter(0.3)
end

# ── Generate synthetic data ──────────────────────────────────

true_pars = (beta=0.5, gamma=0.1, rho=0.3, I0=10.0, N=1000.0)
sys = dust_system_create(sir, true_pars; n_particles=1, seed=42)
dust_system_set_state_initial!(sys)
times = collect(1.0:1.0:50.0)
sol = dust_system_simulate(sys, times)
data_rows = [(time=t, cases=max(1.0, round(sol[2,1,i] * 0.3))) for (i,t) in enumerate(times)]
fdata = dust_filter_data(data_rows; time_field=:time)

# ── Unfilter + packer ────────────────────────────────────────

uf = dust_unfilter_create(sir, fdata)
packer = monty_packer([:beta, :gamma, :rho]; fixed=(N=1000.0, I0=10.0))
ll_model = dust_likelihood_monty(uf, packer)

prior = monty_model(
    x -> begin
        b, g, r = x
        (b <= 0 || g <= 0 || r <= 0 || r > 1) && return -Inf
        return -log(b) - log(g)  # flat improper prior on log scale
    end;
    parameters=["beta", "gamma", "rho"],
    gradient=x -> [-1.0/x[1], -1.0/x[2], 0.0],
    domain=[0.0 Inf; 0.0 Inf; 0.0 1.0],
)

posterior = ll_model + prior

# ── Settings ─────────────────────────────────────────────────

n_steps = 2000
n_burnin = 500
n_chains = 4
n_repeats = 1   # single run for timing

initial_mat = [0.4 0.45 0.55 0.6;
               0.08 0.09 0.11 0.12;
               0.25 0.28 0.32 0.35]

# ── Helper ───────────────────────────────────────────────────

function benchmark_sampler(name, sampler, model, n_steps, n_burnin, n_chains, initial, n_repeats)
    println("\n  Sampling with $name...")
    # Warmup
    try
        monty_sample(model, sampler, 50;
            n_chains=1, initial=initial[:, 1:1], seed=1)
    catch e
        println("    Warmup failed: $e")
        return nothing
    end

    # Run multiple times, keep best timing
    best_time = Inf
    best_samples = nothing
    for rep in 1:n_repeats
        t0 = time_ns()
        samples = monty_sample(model, sampler, n_steps;
            n_chains=n_chains, initial=initial, n_burnin=n_burnin, seed=42+rep)
        elapsed = (time_ns() - t0) / 1e9
        if elapsed < best_time
            best_time = elapsed
            best_samples = samples
        end
    end

    # Convert to MCMCChains and compute ESS
    chn = to_chains(best_samples; elapsed_sec=best_time)
    ess_df = ess(chn)
    ess_vals = ess_df[:, :ess]
    ess_per_sec = ess_vals ./ best_time

    return (name=name, time_sec=best_time, chains=chn,
            ess=ess_vals, ess_per_sec=ess_per_sec, samples=best_samples)
end

# ══════════════════════════════════════════════════════════════════
# PART 1: Julia samplers on ODE SIR
# ══════════════════════════════════════════════════════════════════

println("\n" * "─" ^ 72)
println("PART 1: Julia Samplers on SIR ODE ($(n_steps) steps, $(n_chains) chains)")
println("─" ^ 72)

results = []

# 1. Random Walk
vcv = [0.002 0.0 0.0;
       0.0 0.0005 0.0;
       0.0 0.0 0.003]
rw = monty_sampler_random_walk(vcv)
r = benchmark_sampler("RW", rw, posterior, n_steps, n_burnin, n_chains, initial_mat, n_repeats)
r !== nothing && push!(results, r)

# 2. HMC
hmc = monty_sampler_hmc(0.005, 20)
r = benchmark_sampler("HMC", hmc, posterior, n_steps, n_burnin, n_chains, initial_mat, n_repeats)
r !== nothing && push!(results, r)

# 3. Adaptive RW
adaptive = monty_sampler_adaptive(vcv)
r = benchmark_sampler("Adaptive", adaptive, posterior, n_steps, n_burnin, n_chains, initial_mat, n_repeats)
r !== nothing && push!(results, r)

# 4. NUTS
nuts = monty_sampler_nuts(target_acceptance=0.8, n_adaption=n_burnin)
r = benchmark_sampler("NUTS", nuts, posterior, n_steps, n_burnin, n_chains, initial_mat, n_repeats)
r !== nothing && push!(results, r)

# 5. NUTS with dense metric
nuts_dense = monty_sampler_nuts(target_acceptance=0.8, n_adaption=n_burnin, metric=:dense)
r = benchmark_sampler("NUTS-dense", nuts_dense, posterior, n_steps, n_burnin, n_chains, initial_mat, n_repeats)
r !== nothing && push!(results, r)

# ── Print results table ─────────────────────────────────────

println("\n" * "=" ^ 72)
println("RESULTS: ESS and ESS/sec")
println("=" ^ 72)

param_names = ["beta", "gamma", "rho"]
@printf("\n  %-12s  %8s  ", "Sampler", "Time(s)")
for p in param_names
    @printf(" %8s  %10s", "ESS_$p", "ESS/s_$p")
end
println()
println("  " * "─" ^ 90)

for r in results
    @printf("  %-12s  %8.2f  ", r.name, r.time_sec)
    for (i, p) in enumerate(param_names)
        @printf(" %8.1f  %10.1f", r.ess[i], r.ess_per_sec[i])
    end
    println()
end

# ── Summary statistics ──────────────────────────────────────

println("\n" * "─" ^ 72)
println("MIN ESS / sec (worst parameter, lower bound on efficiency):")
println("─" ^ 72)
for r in results
    min_ess = minimum(r.ess)
    min_ess_per_sec = minimum(r.ess_per_sec)
    @printf("  %-12s  min_ESS = %8.1f   min_ESS/sec = %8.1f\n",
            r.name, min_ess, min_ess_per_sec)
end

# ── MCMCChains diagnostics ──────────────────────────────────

println("\n" * "─" ^ 72)
println("R-hat diagnostics (should be close to 1.0):")
println("─" ^ 72)
for r in results
    rhat_df = rhat(r.chains)
    rhat_vals = rhat_df.nt.rhat
    @printf("  %-12s  rhat = [%s]\n", r.name,
            join([@sprintf("%.4f", v) for v in rhat_vals], ", "))
end

# ══════════════════════════════════════════════════════════════════
# PART 2: Stochastic model (particle filter) — RW only
# ══════════════════════════════════════════════════════════════════

println("\n" * "─" ^ 72)
println("PART 2: Stochastic SIR (Particle Filter, RW sampler)")
println("─" ^ 72)

sir_stoch = @odin begin
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

stoch_pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
stoch_sim = dust_system_simulate(sir_stoch, stoch_pars; times=collect(0.0:1.0:100.0), dt=1.0, seed=1)
obs_stoch = round.(stoch_sim[4, 1, 2:end])
stoch_data = [(time=Float64(t), cases=Float64(obs_stoch[t])) for t in 1:100]
stoch_fdata = dust_filter_data(stoch_data; time_field=:time)

packer_stoch = monty_packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))

for np in [200, 500]
    filter = dust_filter_create(sir_stoch, stoch_fdata; n_particles=np, dt=1.0, seed=42)
    stoch_ll = dust_likelihood_monty(filter, packer_stoch)
    stoch_prior = @monty_prior begin
        beta ~ Gamma(2.0, 0.25)
        gamma ~ Gamma(2.0, 0.05)
    end
    stoch_posterior = stoch_ll + stoch_prior

    stoch_vcv = [0.005 0.0; 0.0 0.001]
    stoch_rw = monty_sampler_random_walk(stoch_vcv)
    stoch_init = [0.4 0.45 0.55 0.6; 0.08 0.09 0.11 0.12]

    println("\n  RW + PF ($(np) particles)...")
    # warmup
    monty_sample(stoch_posterior, stoch_rw, 50; n_chains=1, initial=stoch_init[:,1:1], seed=1)

    t0 = time_ns()
    stoch_samples = monty_sample(stoch_posterior, stoch_rw, 2000;
        n_chains=4, initial=stoch_init, n_burnin=500, seed=42)
    elapsed = (time_ns() - t0) / 1e9

    chn = to_chains(stoch_samples; elapsed_sec=elapsed)
    ess_df = ess(chn)
    ess_vals = ess_df.nt.ess
    ess_per_sec = ess_vals ./ elapsed
    rhat_df = rhat(chn)
    rhat_vals = rhat_df.nt.rhat

    @printf("    Time: %.2f s | ESS: [%.1f, %.1f] | ESS/s: [%.1f, %.1f] | rhat: [%.4f, %.4f]\n",
            elapsed, ess_vals..., ess_per_sec..., rhat_vals...)
end

# ── Save CSV ────────────────────────────────────────────────

open("benchmark/results_sampler_ess_julia.csv", "w") do f
    println(f, "sampler,time_sec,param,ess,ess_per_sec,rhat")
    for r in results
        ess_df = ess(r.chains)
        rhat_df = rhat(r.chains)
        ess_vals = ess_df.nt.ess
        rhat_vals = rhat_df.nt.rhat
        for (i, p) in enumerate(param_names)
            @printf(f, "%s,%.4f,%s,%.2f,%.2f,%.4f\n",
                    r.name, r.time_sec, p, r.ess[i], r.ess_per_sec[i], rhat_vals[i])
        end
    end
end
println("\nResults saved to benchmark/results_sampler_ess_julia.csv")
