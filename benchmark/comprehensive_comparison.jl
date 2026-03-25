#!/usr/bin/env julia
# Comprehensive Julia vs R comparison and benchmark
# Covers: ODE simulation, stochastic simulation, particle filter, 
#         unfilter, MCMC, NUTS, posterior ECDF correlation

using Odin
using Distributions
using Statistics
using LinearAlgebra
using Random
using Printf
using RCall
using BenchmarkTools
using DataFrames
using CSV

const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════

function ecdf_correlation(x::AbstractVector, y::AbstractVector)
    # Compute correlation between empirical CDFs
    all_vals = sort(unique(vcat(x, y)))
    ecdf_x = [count(xi -> xi <= v, x) / length(x) for v in all_vals]
    ecdf_y = [count(yi -> yi <= v, y) / length(y) for v in all_vals]
    return cor(ecdf_x, ecdf_y)
end

function ks_statistic(x::AbstractVector, y::AbstractVector)
    all_vals = sort(unique(vcat(x, y)))
    ecdf_x = [count(xi -> xi <= v, x) / length(x) for v in all_vals]
    ecdf_y = [count(yi -> yi <= v, y) / length(y) for v in all_vals]
    return maximum(abs.(ecdf_x .- ecdf_y))
end

macro timed_block(name, expr)
    quote
        local t0 = time()
        local result = $(esc(expr))
        local elapsed = time() - t0
        @printf("  %-40s %8.3f ms\n", $(esc(name)), elapsed * 1000)
        (result=result, time_ms=elapsed * 1000)
    end
end

println("=" ^ 72)
println("  COMPREHENSIVE Julia vs R COMPARISON AND BENCHMARK")
println("=" ^ 72)
println()

# ═══════════════════════════════════════════════════════════════
# 1. ODE SIMULATION COMPARISON
# ═══════════════════════════════════════════════════════════════
println("━" ^ 72)
println("  1. ODE SIMULATION: SIR model")
println("━" ^ 72)

sir_ode = @odin begin
    deriv(S) = -beta * S * Inf2 / N
    deriv(Inf2) = beta * S * Inf2 / N - gamma * Inf2
    deriv(R) = gamma * Inf2
    initial(S) = N - I0
    initial(Inf2) = I0
    initial(R) = 0
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10.0)
    N = parameter(1000.0)
end

pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
times = collect(0.0:1.0:100.0)

jl_result = dust_system_simulate(sir_ode, pars; times=times, seed=1)
jl_S = jl_result[1, 1, :]
jl_I = jl_result[2, 1, :]
jl_R = jl_result[3, 1, :]

R"""
library(odin2)
library(dust2)
library(monty)

sir <- odin2::odin({
    deriv(S) <- -beta * S * I / N
    deriv(I) <- beta * S * I / N - gamma * I
    deriv(R) <- gamma * I
    initial(S) <- N - I0
    initial(I) <- I0
    initial(R) <- 0
    beta <- parameter(0.5)
    gamma <- parameter(0.1)
    I0 <- parameter(10)
    N <- parameter(1000)
})

sys_r <- dust2::dust_system_create(sir, list(beta=0.5, gamma=0.1, I0=10, N=1000), seed=1L)
dust2::dust_system_set_state_initial(sys_r)
times_r <- seq(0, 100, by=1)
r_result <- dust2::dust_system_simulate(sys_r, times_r)
"""

r_S = rcopy(R"r_result[1, ]")
r_I = rcopy(R"r_result[2, ]")
r_R = rcopy(R"r_result[3, ]")

# Compare
max_diff_S = maximum(abs.(jl_S .- r_S))
max_diff_I = maximum(abs.(jl_I .- r_I))
max_diff_R = maximum(abs.(jl_R .- r_R))
corr_I = cor(jl_I, r_I)

@printf("  Max |S_jl - S_r|: %.2e\n", max_diff_S)
@printf("  Max |I_jl - I_r|: %.2e\n", max_diff_I)
@printf("  Max |R_jl - R_r|: %.2e\n", max_diff_R)
@printf("  Correlation(I):   %.10f\n", corr_I)
println()

# Benchmark
print("  Julia ODE simulate: ")
jl_ode_bench = @benchmark dust_system_simulate($sir_ode, $pars; times=$times, seed=1) samples=50
jl_ode_ms = median(jl_ode_bench.times) / 1e6
@printf("%.3f ms (median)\n", jl_ode_ms)

R"""
library(microbenchmark)
r_ode_bench <- microbenchmark::microbenchmark({
    sys <- dust2::dust_system_create(sir, list(beta=0.5, gamma=0.1, I0=10, N=1000), seed=1L)
    dust2::dust_system_set_state_initial(sys)
    dust2::dust_system_simulate(sys, times_r)
}, times=50)
r_ode_ms <- median(r_ode_bench$time) / 1e6
"""
r_ode_ms = rcopy(R"r_ode_ms")
@printf("  R ODE simulate:     %.3f ms (median)\n", r_ode_ms)
@printf("  Speedup (R/Julia):  %.1fx\n", r_ode_ms / jl_ode_ms)
println()

# ═══════════════════════════════════════════════════════════════
# 2. STOCHASTIC SIMULATION COMPARISON
# ═══════════════════════════════════════════════════════════════
println("━" ^ 72)
println("  2. STOCHASTIC SIMULATION: SIR discrete")
println("━" ^ 72)

sir_stoch = @odin begin
    update(S) = S - n_SI
    update(Inf2) = Inf2 + n_SI - n_IR
    update(R) = R + n_IR
    initial(S) = N - I0
    initial(Inf2) = I0
    initial(R) = 0
    p_SI = 1 - exp(-beta * Inf2 / N * dt)
    p_IR = 1 - exp(-gamma * dt)
    n_SI = Binomial(S, p_SI)
    n_IR = Binomial(Inf2, p_IR)
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10.0)
    N = parameter(1000.0)
end

stimes = collect(0.0:1.0:100.0)
n_particles = 1000

jl_stoch = dust_system_simulate(sir_stoch, pars; times=stimes, n_particles=n_particles, dt=1.0, seed=42)
jl_mean_I = mean(jl_stoch[2, :, :], dims=1)[1, :]
jl_var_I = var(jl_stoch[2, :, :], dims=1)[1, :]

R"""
sir_stoch_r <- odin2::odin({
    update(S) <- S - n_SI
    update(I) <- I + n_SI - n_IR
    update(R) <- R + n_IR
    initial(S) <- N - I0
    initial(I) <- I0
    initial(R) <- 0
    p_SI <- 1 - exp(-beta * I / N * dt)
    p_IR <- 1 - exp(-gamma * dt)
    n_SI <- Binomial(S, p_SI)
    n_IR <- Binomial(I, p_IR)
    beta <- parameter(0.5)
    gamma <- parameter(0.1)
    I0 <- parameter(10)
    N <- parameter(1000)
})

sys_stoch_r <- dust2::dust_system_create(sir_stoch_r, list(beta=0.5, gamma=0.1, I0=10, N=1000),
                                          n_particles=1000L, dt=1, seed=42L)
dust2::dust_system_set_state_initial(sys_stoch_r)
r_stoch <- dust2::dust_system_simulate(sys_stoch_r, seq(0, 100, by=1))
r_mean_I <- apply(r_stoch[2, , ], 2, mean)
r_var_I <- apply(r_stoch[2, , ], 2, var)
"""

r_mean_I = rcopy(R"r_mean_I")
r_var_I = rcopy(R"r_var_I")

# Distribution comparison (not identical due to different RNG, but moments should be close)
mean_rel_diff = mean(abs.(jl_mean_I[2:end] .- r_mean_I[2:end]) ./ max.(abs.(r_mean_I[2:end]), 1.0))
var_rel_diff = mean(abs.(jl_var_I[2:end] .- r_var_I[2:end]) ./ max.(abs.(r_var_I[2:end]), 1.0))
corr_mean = cor(jl_mean_I[2:end], r_mean_I[2:end])

@printf("  Mean relative diff (means): %.4f\n", mean_rel_diff)
@printf("  Mean relative diff (vars):  %.4f\n", var_rel_diff)
@printf("  Correlation(mean I):        %.6f\n", corr_mean)
println()

# Benchmark
print("  Julia stochastic (1000 particles): ")
jl_stoch_bench = @benchmark dust_system_simulate($sir_stoch, $pars; times=$stimes, n_particles=1000, dt=1.0, seed=42) samples=30
jl_stoch_ms = median(jl_stoch_bench.times) / 1e6
@printf("%.3f ms\n", jl_stoch_ms)

R"""
r_stoch_bench <- microbenchmark::microbenchmark({
    sys <- dust2::dust_system_create(sir_stoch_r, list(beta=0.5, gamma=0.1, I0=10, N=1000),
                                     n_particles=1000L, dt=1, seed=42L)
    dust2::dust_system_set_state_initial(sys)
    dust2::dust_system_simulate(sys, seq(0, 100, by=1))
}, times=30)
r_stoch_ms <- median(r_stoch_bench$time) / 1e6
"""
r_stoch_ms = rcopy(R"r_stoch_ms")
@printf("  R stochastic (1000 particles):     %.3f ms\n", r_stoch_ms)
@printf("  Speedup (R/Julia):                 %.1fx\n", r_stoch_ms / jl_stoch_ms)
println()

# ═══════════════════════════════════════════════════════════════
# 3. PARTICLE FILTER COMPARISON
# ═══════════════════════════════════════════════════════════════
println("━" ^ 72)
println("  3. PARTICLE FILTER: Log-likelihood distribution")
println("━" ^ 72)

# Generate shared data for filtering
sir_obs = @odin begin
    update(S) = S - n_SI
    update(Inf2) = Inf2 + n_SI - n_IR
    update(R) = R + n_IR
    initial(S) = N - I0
    initial(Inf2) = I0
    initial(R) = 0
    initial(incidence, zero_every=1) = 0
    update(incidence) = incidence + n_SI
    p_SI = 1 - exp(-beta * Inf2 / N * dt)
    p_IR = 1 - exp(-gamma * dt)
    n_SI = Binomial(S, p_SI)
    n_IR = Binomial(Inf2, p_IR)
    cases = data()
    cases ~ Poisson(incidence + 1e-6)
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10.0)
    N = parameter(1000.0)
end

# Generate data
Random.seed!(1)
sys_data = dust_system_create(sir_obs, pars; n_particles=1, dt=1.0, seed=1)
dust_system_set_state_initial!(sys_data)
sim = dust_system_simulate(sys_data, collect(0.0:1.0:50.0))
incidence_data = sim[4, 1, 2:end]
obs_data = [rand(Poisson(max(x, 1.0))) for x in incidence_data]

data_records = [(time=Float64(t), cases=Float64(c)) for (t, c) in zip(1:50, obs_data)]
fdata = dust_filter_data(data_records; time_field=:time)

# Save shared data
shared_csv_path = joinpath(@__DIR__, "shared_data_comparison.csv")
CSV.write(shared_csv_path, DataFrame(time=1:50, cases=obs_data))

# Julia particle filter - run 100 times for LL distribution
n_pf_runs = 100
n_pf_particles = 200
jl_lls = zeros(n_pf_runs)
for i in 1:n_pf_runs
    pf = dust_filter_create(sir_obs, fdata; n_particles=n_pf_particles, dt=1.0, seed=i)
    jl_lls[i] = dust_likelihood_run!(pf, pars)
end

# R particle filter
R"""
sir_obs_r <- odin2::odin({
    update(S) <- S - n_SI
    update(I) <- I + n_SI - n_IR
    update(R) <- R + n_IR
    initial(S) <- N - I0
    initial(I) <- I0
    initial(R) <- 0
    initial(incidence, zero_every = 1) <- 0
    update(incidence) <- incidence + n_SI
    p_SI <- 1 - exp(-beta * I / N * dt)
    p_IR <- 1 - exp(-gamma * dt)
    n_SI <- Binomial(S, p_SI)
    n_IR <- Binomial(I, p_IR)
    cases <- data()
    cases ~ Poisson(incidence + 1e-6)
    beta <- parameter(0.5)
    gamma <- parameter(0.1)
    I0 <- parameter(10)
    N <- parameter(1000)
})

shared_data <- read.csv($shared_csv_path)
data_r <- data.frame(time=shared_data$time, cases=shared_data$cases)

r_lls <- numeric(100)
for (i in 1:100) {
    pf_r <- dust2::dust_filter_create(sir_obs_r, time_start=0, data=data_r,
                                       n_particles=200L, dt=1, seed=i)
    r_lls[i] <- dust2::dust_likelihood_run(pf_r, list(beta=0.5, gamma=0.1, I0=10, N=1000))
}
"""

r_lls = rcopy(R"r_lls")

ll_ecdf_corr = ecdf_correlation(jl_lls, r_lls)
ll_ks = ks_statistic(jl_lls, r_lls)

@printf("  Julia LL: mean=%.2f, sd=%.4f\n", mean(jl_lls), std(jl_lls))
@printf("  R LL:     mean=%.2f, sd=%.4f\n", mean(r_lls), std(r_lls))
@printf("  ECDF correlation:   %.6f\n", ll_ecdf_corr)
@printf("  KS statistic:       %.4f\n", ll_ks)
println()

# Benchmark PF
print("  Julia PF (200 particles, 50 steps): ")
pf_jl = dust_filter_create(sir_obs, fdata; n_particles=n_pf_particles, dt=1.0, seed=42)
jl_pf_bench = @benchmark dust_likelihood_run!($pf_jl, $pars) samples=50
jl_pf_ms = median(jl_pf_bench.times) / 1e6
@printf("%.3f ms\n", jl_pf_ms)

R"""
r_pf_bench <- microbenchmark::microbenchmark({
    pf <- dust2::dust_filter_create(sir_obs_r, time_start=0, data=data_r,
                                     n_particles=200L, dt=1, seed=42L)
    dust2::dust_likelihood_run(pf, list(beta=0.5, gamma=0.1, I0=10, N=1000))
}, times=50)
r_pf_ms <- median(r_pf_bench$time) / 1e6
"""
r_pf_ms = rcopy(R"r_pf_ms")
@printf("  R PF (200 particles, 50 steps):     %.3f ms\n", r_pf_ms)
@printf("  Speedup (R/Julia):                  %.1fx\n", r_pf_ms / jl_pf_ms)
println()

# ═══════════════════════════════════════════════════════════════
# 4. DETERMINISTIC LIKELIHOOD (UNFILTER) COMPARISON
# ═══════════════════════════════════════════════════════════════
println("━" ^ 72)
println("  4. UNFILTER: Deterministic ODE likelihood")
println("━" ^ 72)

sir_ode_obs = @odin begin
    deriv(S) = -beta * S * Inf2 / N
    deriv(Inf2) = beta * S * Inf2 / N - gamma * Inf2
    deriv(R) = gamma * Inf2
    initial(S) = N - I0
    initial(Inf2) = I0
    initial(R) = 0
    cases = data()
    cases ~ Poisson(Inf2 + 1e-6)
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10.0)
    N = parameter(1000.0)
end

uf_jl = dust_unfilter_create(sir_ode_obs, fdata; time_start=0.0)
jl_uf_ll = dust_unfilter_run!(uf_jl, pars)

R"""
sir_ode_obs_r <- odin2::odin({
    deriv(S) <- -beta * S * I / N
    deriv(I) <- beta * S * I / N - gamma * I
    deriv(R) <- gamma * I
    initial(S) <- N - I0
    initial(I) <- I0
    initial(R) <- 0
    cases <- data()
    cases ~ Poisson(I + 1e-6)
    beta <- parameter(0.5)
    gamma <- parameter(0.1)
    I0 <- parameter(10)
    N <- parameter(1000)
})

uf_r <- dust2::dust_unfilter_create(sir_ode_obs_r, time_start=0, data=data_r)
r_uf_ll <- dust2::dust_likelihood_run(uf_r, list(beta=0.5, gamma=0.1, I0=10, N=1000))
"""
r_uf_ll = rcopy(R"r_uf_ll")

@printf("  Julia unfilter LL: %.6f\n", jl_uf_ll)
@printf("  R unfilter LL:     %.6f\n", r_uf_ll)
@printf("  Absolute diff:     %.2e\n", abs(jl_uf_ll - r_uf_ll))
println()

# Benchmark
print("  Julia unfilter: ")
jl_uf_bench = @benchmark dust_unfilter_run!($uf_jl, $pars) samples=100
jl_uf_ms = median(jl_uf_bench.times) / 1e6
@printf("%.3f ms\n", jl_uf_ms)

R"""
r_uf_bench <- microbenchmark::microbenchmark({
    uf <- dust2::dust_unfilter_create(sir_ode_obs_r, time_start=0, data=data_r)
    dust2::dust_likelihood_run(uf, list(beta=0.5, gamma=0.1, I0=10, N=1000))
}, times=100)
r_uf_ms <- median(r_uf_bench$time) / 1e6
"""
r_uf_ms = rcopy(R"r_uf_ms")
@printf("  R unfilter:     %.3f ms\n", r_uf_ms)
@printf("  Speedup:        %.1fx\n", r_uf_ms / jl_uf_ms)
println()

# ═══════════════════════════════════════════════════════════════
# 5. MCMC COMPARISON (Random Walk + Adaptive)
# ═══════════════════════════════════════════════════════════════
println("━" ^ 72)
println("  5. MCMC: Random walk on deterministic SIR")
println("━" ^ 72)

pk = monty_packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))
ll_jl = dust_likelihood_monty(uf_jl, pk)
prior_jl = @monty_prior begin
    beta ~ Gamma(2.0, 0.25)
    gamma ~ Gamma(2.0, 0.05)
end
posterior_jl = ll_jl + prior_jl

n_mcmc = 5000
n_burnin = 1000
vcv = diagm([0.005, 0.001])
sampler_rw = monty_sampler_random_walk(vcv)

print("  Julia MCMC (5000 steps, RW): ")
t0 = time()
samples_jl = monty_sample(posterior_jl, sampler_rw, n_mcmc;
    initial=reshape([0.4, 0.08], 2, 1), n_chains=1, n_burnin=n_burnin, seed=42)
jl_mcmc_ms = (time() - t0) * 1000
@printf("%.1f ms\n", jl_mcmc_ms)

R"""
uf_r2 <- dust2::dust_unfilter_create(sir_ode_obs_r, time_start=0, data=data_r)
pk_r <- monty::monty_packer(c("beta", "gamma"), fixed=list(I0=10, N=1000))
ll_r <- dust2::dust_likelihood_monty(uf_r2, pk_r)
prior_r <- monty::monty_dsl({
    beta ~ Gamma(shape=2, rate=4)
    gamma ~ Gamma(shape=2, rate=20)
})
posterior_r <- ll_r + prior_r

vcv_r <- matrix(c(0.005, 0, 0, 0.001), 2, 2)
sampler_r <- monty::monty_sampler_random_walk(vcv_r)

r_t0 <- proc.time()
samples_r <- monty::monty_sample(posterior_r, sampler_r, 5000L,
    initial=c(0.4, 0.08), n_chains=1L, burnin=1000L)
r_mcmc_time <- (proc.time() - r_t0)[["elapsed"]] * 1000

r_beta <- samples_r$pars[1, , 1]
r_gamma <- samples_r$pars[2, , 1]
"""

r_beta = rcopy(R"r_beta")
r_gamma = rcopy(R"r_gamma")
r_mcmc_ms = rcopy(R"r_mcmc_time")

jl_beta = samples_jl.pars[1, :, 1]
jl_gamma = samples_jl.pars[2, :, 1]

ecdf_beta = ecdf_correlation(jl_beta, r_beta)
ecdf_gamma = ecdf_correlation(jl_gamma, r_gamma)
ks_beta = ks_statistic(jl_beta, r_beta)
ks_gamma = ks_statistic(jl_gamma, r_gamma)

@printf("  R MCMC (5000 steps, RW):     %.1f ms\n", r_mcmc_ms)
@printf("  Speedup (R/Julia):           %.1fx\n", r_mcmc_ms / jl_mcmc_ms)
println()
@printf("  Julia posterior β: mean=%.4f, sd=%.4f\n", mean(jl_beta), std(jl_beta))
@printf("  R posterior β:     mean=%.4f, sd=%.4f\n", mean(r_beta), std(r_beta))
@printf("  ECDF correlation (β): %.6f\n", ecdf_beta)
@printf("  KS statistic (β):     %.4f\n", ks_beta)
println()
@printf("  Julia posterior γ: mean=%.4f, sd=%.4f\n", mean(jl_gamma), std(jl_gamma))
@printf("  R posterior γ:     mean=%.4f, sd=%.4f\n", mean(r_gamma), std(r_gamma))
@printf("  ECDF correlation (γ): %.6f\n", ecdf_gamma)
@printf("  KS statistic (γ):     %.4f\n", ks_gamma)
println()

# ═══════════════════════════════════════════════════════════════
# 6. ADAPTIVE MCMC COMPARISON
# ═══════════════════════════════════════════════════════════════
println("━" ^ 72)
println("  6. ADAPTIVE MCMC: Spencer 2021 accelerated shaping")
println("━" ^ 72)

sampler_adapt = monty_sampler_adaptive(vcv)

print("  Julia Adaptive MCMC (5000 steps): ")
t0 = time()
samples_adapt_jl = monty_sample(posterior_jl, sampler_adapt, n_mcmc;
    initial=repeat([0.4, 0.08], 1, 4), n_chains=4, n_burnin=n_burnin, seed=42)
jl_adapt_ms = (time() - t0) * 1000
@printf("%.1f ms\n", jl_adapt_ms)

R"""
sampler_adapt_r <- monty::monty_sampler_adaptive(vcv_r)

r_t0 <- proc.time()
samples_adapt_r <- monty::monty_sample(posterior_r, sampler_adapt_r, 5000L,
    initial=c(0.4, 0.08), n_chains=4L, burnin=1000L)
r_adapt_time <- (proc.time() - r_t0)[["elapsed"]] * 1000

r_beta_adapt <- as.vector(samples_adapt_r$pars[1, , ])
r_gamma_adapt <- as.vector(samples_adapt_r$pars[2, , ])
"""

r_beta_adapt = rcopy(R"r_beta_adapt")
r_gamma_adapt = rcopy(R"r_gamma_adapt")
r_adapt_ms = rcopy(R"r_adapt_time")

jl_beta_adapt = vec(samples_adapt_jl.pars[1, :, :])
jl_gamma_adapt = vec(samples_adapt_jl.pars[2, :, :])

ecdf_beta_a = ecdf_correlation(jl_beta_adapt, r_beta_adapt)
ecdf_gamma_a = ecdf_correlation(jl_gamma_adapt, r_gamma_adapt)

@printf("  R Adaptive MCMC (5000 steps):     %.1f ms\n", r_adapt_ms)
@printf("  Speedup (R/Julia):                %.1fx\n", r_adapt_ms / jl_adapt_ms)
println()
@printf("  Julia posterior β: mean=%.4f, sd=%.4f\n", mean(jl_beta_adapt), std(jl_beta_adapt))
@printf("  R posterior β:     mean=%.4f, sd=%.4f\n", mean(r_beta_adapt), std(r_beta_adapt))
@printf("  ECDF correlation (β): %.6f\n", ecdf_beta_a)
println()
@printf("  Julia posterior γ: mean=%.4f, sd=%.4f\n", mean(jl_gamma_adapt), std(jl_gamma_adapt))
@printf("  R posterior γ:     mean=%.4f, sd=%.4f\n", mean(r_gamma_adapt), std(r_gamma_adapt))
@printf("  ECDF correlation (γ): %.6f\n", ecdf_gamma_a)
println()

# ═══════════════════════════════════════════════════════════════
# 7. NUTS SAMPLER (Julia only — R doesn't have NUTS in monty)
# ═══════════════════════════════════════════════════════════════
println("━" ^ 72)
println("  7. NUTS SAMPLER (Julia only — not available in R monty)")
println("━" ^ 72)

print("  Julia NUTS (1000 steps): ")
sampler_nuts = monty_sampler_nuts()
t0 = time()
samples_nuts = monty_sample(posterior_jl, sampler_nuts, 1000;
    initial=repeat([0.4, 0.08], 1, 4), n_chains=4, n_burnin=500, seed=42)
jl_nuts_ms = (time() - t0) * 1000
@printf("%.1f ms\n", jl_nuts_ms)

jl_beta_nuts = vec(samples_nuts.pars[1, :, :])
jl_gamma_nuts = vec(samples_nuts.pars[2, :, :])

# Compare NUTS vs RW posteriors
ecdf_nuts_rw_beta = ecdf_correlation(jl_beta_nuts, jl_beta_adapt)
ecdf_nuts_rw_gamma = ecdf_correlation(jl_gamma_nuts, jl_gamma_adapt)

@printf("  NUTS posterior β: mean=%.4f, sd=%.4f\n", mean(jl_beta_nuts), std(jl_beta_nuts))
@printf("  NUTS posterior γ: mean=%.4f, sd=%.4f\n", mean(jl_gamma_nuts), std(jl_gamma_nuts))
@printf("  ECDF corr NUTS vs Adaptive (β): %.6f\n", ecdf_nuts_rw_beta)
@printf("  ECDF corr NUTS vs Adaptive (γ): %.6f\n", ecdf_nuts_rw_gamma)
println()

# ═══════════════════════════════════════════════════════════════
# 8. PARTICLE FILTER MCMC COMPARISON
# ═══════════════════════════════════════════════════════════════
println("━" ^ 72)
println("  8. PARTICLE FILTER MCMC: Stochastic SIR")
println("━" ^ 72)

pf_jl2 = dust_filter_create(sir_obs, fdata; n_particles=500, dt=1.0, seed=42)
ll_pf_jl = dust_likelihood_monty(pf_jl2, pk)
posterior_pf_jl = ll_pf_jl + prior_jl

print("  Julia PF-MCMC (10000 steps, 4 chains): ")
t0 = time()
samples_pf_jl = monty_sample(posterior_pf_jl, sampler_rw, 10000;
    initial=repeat([0.4, 0.08], 1, 4), n_chains=4, n_burnin=2000, seed=42)
jl_pfmcmc_ms = (time() - t0) * 1000
@printf("%.1f ms\n", jl_pfmcmc_ms)

R"""
pf_r2 <- dust2::dust_filter_create(sir_obs_r, time_start=0, data=data_r,
                                    n_particles=500L, dt=1, seed=42L)
ll_pf_r <- dust2::dust_likelihood_monty(pf_r2, pk_r)
posterior_pf_r <- ll_pf_r + prior_r

sampler_rw_r <- monty::monty_sampler_random_walk(vcv_r)
r_t0 <- proc.time()
samples_pf_r <- monty::monty_sample(posterior_pf_r, sampler_rw_r, 10000L,
    initial=c(0.4, 0.08), n_chains=4L, burnin=2000L)
r_pfmcmc_time <- (proc.time() - r_t0)[["elapsed"]] * 1000

r_beta_pf <- as.vector(samples_pf_r$pars[1, , ])
r_gamma_pf <- as.vector(samples_pf_r$pars[2, , ])
"""

r_beta_pf = rcopy(R"r_beta_pf")
r_gamma_pf = rcopy(R"r_gamma_pf")
r_pfmcmc_ms = rcopy(R"r_pfmcmc_time")

jl_beta_pf = vec(samples_pf_jl.pars[1, :, :])
jl_gamma_pf = vec(samples_pf_jl.pars[2, :, :])

ecdf_pf_beta = ecdf_correlation(jl_beta_pf, r_beta_pf)
ecdf_pf_gamma = ecdf_correlation(jl_gamma_pf, r_gamma_pf)

@printf("  R PF-MCMC (10000 steps, 4 chains): %.1f ms\n", r_pfmcmc_ms)
@printf("  Speedup (R/Julia):                 %.1fx\n", r_pfmcmc_ms / jl_pfmcmc_ms)
println()
@printf("  Julia posterior β: mean=%.4f, sd=%.4f\n", mean(jl_beta_pf), std(jl_beta_pf))
@printf("  R posterior β:     mean=%.4f, sd=%.4f\n", mean(r_beta_pf), std(r_beta_pf))
@printf("  ECDF correlation (β): %.6f\n", ecdf_pf_beta)
println()
@printf("  Julia posterior γ: mean=%.4f, sd=%.4f\n", mean(jl_gamma_pf), std(jl_gamma_pf))
@printf("  R posterior γ:     mean=%.4f, sd=%.4f\n", mean(r_gamma_pf), std(r_gamma_pf))
@printf("  ECDF correlation (γ): %.6f\n", ecdf_pf_gamma)
println()

# ═══════════════════════════════════════════════════════════════
# 9. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
println("━" ^ 72)
println("  SUMMARY")
println("━" ^ 72)

results = DataFrame(
    Task = [
        "ODE simulation (100 steps)",
        "Stochastic sim (1000 particles, 100 steps)",
        "Particle filter (200 particles, 50 steps)",
        "Unfilter (ODE likelihood, 50 steps)",
        "MCMC RW (5000 steps, 1 chain)",
        "Adaptive MCMC (5000 steps, 4 chains)",
        "NUTS (1000 steps, 4 chains)",
        "PF-MCMC (10000 steps, 4 chains, 500 particles)",
    ],
    Julia_ms = [jl_ode_ms, jl_stoch_ms, jl_pf_ms, jl_uf_ms, jl_mcmc_ms, jl_adapt_ms, jl_nuts_ms, jl_pfmcmc_ms],
    R_ms = [r_ode_ms, r_stoch_ms, r_pf_ms, r_uf_ms, r_mcmc_ms, r_adapt_ms, NaN, r_pfmcmc_ms],
    Speedup = [r_ode_ms/jl_ode_ms, r_stoch_ms/jl_stoch_ms, r_pf_ms/jl_pf_ms, r_uf_ms/jl_uf_ms,
               r_mcmc_ms/jl_mcmc_ms, r_adapt_ms/jl_adapt_ms, NaN, r_pfmcmc_ms/jl_pfmcmc_ms],
)

println()
@printf("  %-45s %10s %10s %8s\n", "Task", "Julia (ms)", "R (ms)", "Speedup")
@printf("  %s\n", "-" ^ 75)
for row in eachrow(results)
    if isnan(row.R_ms)
        @printf("  %-45s %10.2f %10s %8s\n", row.Task, row.Julia_ms, "N/A", "N/A")
    else
        @printf("  %-45s %10.2f %10.2f %7.1fx\n", row.Task, row.Julia_ms, row.R_ms, row.Speedup)
    end
end

println()
println("━" ^ 72)
println("  POSTERIOR ECDF CORRELATIONS (Julia vs R)")
println("━" ^ 72)
println()

ecdf_results = DataFrame(
    Comparison = [
        "RW MCMC β", "RW MCMC γ",
        "Adaptive MCMC β", "Adaptive MCMC γ",
        "PF-MCMC β", "PF-MCMC γ",
        "NUTS vs Adaptive β (Julia)", "NUTS vs Adaptive γ (Julia)",
    ],
    ECDF_Correlation = [ecdf_beta, ecdf_gamma, ecdf_beta_a, ecdf_gamma_a,
                        ecdf_pf_beta, ecdf_pf_gamma,
                        ecdf_nuts_rw_beta, ecdf_nuts_rw_gamma],
    KS_Statistic = [ks_beta, ks_gamma,
                    ks_statistic(jl_beta_adapt, r_beta_adapt),
                    ks_statistic(jl_gamma_adapt, r_gamma_adapt),
                    ks_statistic(jl_beta_pf, r_beta_pf),
                    ks_statistic(jl_gamma_pf, r_gamma_pf),
                    ks_statistic(jl_beta_nuts, jl_beta_adapt),
                    ks_statistic(jl_gamma_nuts, jl_gamma_adapt)],
)

@printf("  %-35s %18s %15s\n", "Comparison", "ECDF Correlation", "KS Statistic")
@printf("  %s\n", "-" ^ 70)
for row in eachrow(ecdf_results)
    @printf("  %-35s %18.6f %15.4f\n", row.Comparison, row.ECDF_Correlation, row.KS_Statistic)
end

# Save results
CSV.write(joinpath(RESULTS_DIR, "benchmark_summary.csv"), results)
CSV.write(joinpath(RESULTS_DIR, "ecdf_comparison.csv"), ecdf_results)

println()
println("=" ^ 72)
println("  Results saved to benchmark/results/")
println("=" ^ 72)
