#!/usr/bin/env julia
"""
Benchmark the three advanced models (Mpox SEIR, Malaria, SARS-CoV-2 Multi-region)
against R implementations, comparing simulation speed and inference accuracy.
"""

using Odin
using Random
using Statistics
using Distributions
using BenchmarkTools
using Printf
using DelimitedFiles

println("="^70)
println("ADVANCED MODEL BENCHMARKS — Odin.jl vs R (odin2/dust2/monty)")
println("="^70)

results = Dict{String, Any}()

# ─── 1. Mpox SEIR (Stochastic, 4 age × 3 vax, particle filter) ────────────

println("\n── 1. Mpox SEIR (age-structured stochastic) ──")

mpox = @odin begin
    n_age = parameter(4)
    n_vax = parameter(3)
    dim(S) = c(n_age, n_vax)
    dim(E) = c(n_age, n_vax)
    dim(I) = c(n_age, n_vax)
    dim(R) = c(n_age, n_vax)
    dim(D) = c(n_age, n_vax)
    dim(cases_inc) = c(n_age, n_vax)
    dim(n_SE) = c(n_age, n_vax)
    dim(n_EI) = c(n_age, n_vax)
    dim(n_IR) = c(n_age, n_vax)
    dim(n_ID) = c(n_age, n_vax)
    dim(p_SE) = c(n_age, n_vax)
    dim(contact) = c(n_age, n_age)
    dim(N0) = n_age
    dim(S0) = c(n_age, n_vax)
    dim(I0) = c(n_age, n_vax)
    dim(lambda) = n_age
    dim(ve) = n_vax
    dim(I_age) = n_age

    contact = parameter()
    N0 = parameter()
    S0 = parameter()
    I0 = parameter()
    beta = parameter(0.3)
    sigma = parameter(0.1)
    gamma = parameter(0.14)
    mu = parameter(0.01)
    ve1 = parameter(0.5)
    ve2 = parameter(0.85)
    alpha_cases = parameter(0.1)

    ve[1] = 0.0
    ve[2] = ve1
    ve[3] = ve2

    I_age[i] = I[i, 1] + I[i, 2] + I[i, 3]
    lambda[i] = beta * (contact[i, 1] * I_age[1] / N0[1] +
                         contact[i, 2] * I_age[2] / N0[2] +
                         contact[i, 3] * I_age[3] / N0[3] +
                         contact[i, 4] * I_age[4] / N0[4])

    p_SE[i, j] = 1 - exp(-lambda[i] * (1 - ve[j]) * dt)
    p_EI = 1 - exp(-sigma * dt)
    p_IR = 1 - exp(-gamma * (1 - mu) * dt)
    p_ID = 1 - exp(-gamma * mu * dt)

    n_SE[i, j] = Binomial(S[i, j], p_SE[i, j])
    n_EI[i, j] = Binomial(E[i, j], p_EI)
    n_IR[i, j] = Binomial(I[i, j], p_IR)
    n_ID[i, j] = Binomial(I[i, j] - n_IR[i, j], min(p_ID / max(1 - p_IR, 1e-10), 1.0))

    update(S[i, j]) = S[i, j] - n_SE[i, j]
    update(E[i, j]) = E[i, j] + n_SE[i, j] - n_EI[i, j]
    update(I[i, j]) = I[i, j] + n_EI[i, j] - n_IR[i, j] - n_ID[i, j]
    update(R[i, j]) = R[i, j] + n_IR[i, j]
    update(D[i, j]) = D[i, j] + n_ID[i, j]
    update(cases_inc[i, j]) = cases_inc[i, j] + n_EI[i, j]

    initial(S[i, j]) = S0[i, j]
    initial(E[i, j]) = 0
    initial(I[i, j]) = I0[i, j]
    initial(R[i, j]) = 0
    initial(D[i, j]) = 0
    initial(cases_inc[i, j]) = 0

    total_cases = sum(cases_inc)
    cases_data = data()
    cases_data ~ NegativeBinomial(1.0 / alpha_cases, max(total_cases, 1e-6) * alpha_cases / (1 + max(total_cases, 1e-6) * alpha_cases))
end

contact = [2.0 0.5 0.3 0.1;
           0.5 3.0 1.0 0.3;
           0.3 1.0 2.5 0.5;
           0.1 0.3 0.5 1.5]
N0_vals = [200000.0, 300000.0, 350000.0, 150000.0]
S0_mat = zeros(4, 3)
I0_mat = zeros(4, 3)
for i in 1:4
    S0_mat[i, 1] = N0_vals[i] - (i == 2 ? 5.0 : 0.0)
    I0_mat[i, 1] = (i == 2 ? 5.0 : 0.0)
end

mpox_pars = (
    n_age=4.0, n_vax=3.0,
    contact=contact, N0=N0_vals,
    S0=S0_mat, I0=I0_mat,
    beta=0.3, sigma=0.1, gamma=0.14, mu=0.01,
    ve1=0.5, ve2=0.85, alpha_cases=0.1,
)

# Benchmark simulation (200 particles, 180 days)
sys = dust_system_create(mpox, mpox_pars; n_particles=200, dt=0.25, seed=42)
dust_system_set_state_initial!(sys)
sim_times = collect(0.0:1.0:180.0)

# Warmup
dust_system_simulate(sys, sim_times)

# Benchmark
_reset_sys = Odin._reset_system!
b_mpox_sim = @benchmark begin
    $_reset_sys($sys, $mpox_pars, 42)
    dust_system_simulate($sys, $sim_times)
end samples=20 evals=1
t_mpox_sim = median(b_mpox_sim).time / 1e6
println(@sprintf("  Simulation (200 ptcl, 180 days): %.1f ms", t_mpox_sim))
results["mpox_sim_julia_ms"] = t_mpox_sim

# ─── 2. Malaria Ross-Macdonald (ODE, seasonal forcing) ─────────────────────

println("\n── 2. Malaria Simple (Ross-Macdonald ODE) ──")

malaria = @odin begin
    a = parameter(0.3)
    b = parameter(0.55)
    c_par = parameter(0.15)
    phi = parameter(0.5)
    delta_h = parameter(0.0833)
    delta_m = parameter(0.1)
    gamma_h = parameter(0.2)
    gamma_a = parameter(0.005)
    omega = parameter(0.00274)
    rho_relapse = parameter(0.00137)
    p_asymp = parameter(0.5)
    mu_h = parameter(5.479e-5)
    mu_m = parameter(0.1)
    K_m0 = parameter(20.0)
    amp = parameter(0.5)
    phase = parameter(60.0)
    N_h = parameter(10000.0)
    I_h0 = parameter(100.0)
    itn_cov = parameter(0.0)
    itn_eff = parameter(0.5)

    K_m = K_m0 * N_h * (1 + amp * sin(2 * 3.14159265 * (time - phase) / 365))
    itn_effect = 1 - itn_cov * itn_eff

    lambda_h = a * b * I_m / N_h
    lambda_m = a * c_par * (I_h + phi * A_h) / N_h

    deriv(S_h) = -lambda_h * S_h + omega * R_h + mu_h * N_h - mu_h * S_h
    deriv(E_h) = lambda_h * S_h - delta_h * E_h - mu_h * E_h
    deriv(I_h) = delta_h * E_h * (1 - p_asymp) - gamma_h * I_h - mu_h * I_h
    deriv(A_h) = delta_h * E_h * p_asymp + rho_relapse * R_h - gamma_a * A_h - mu_h * A_h
    deriv(R_h) = gamma_h * I_h + gamma_a * A_h - omega * R_h - rho_relapse * R_h - mu_h * R_h

    emergence = mu_m * K_m
    deriv(S_m) = emergence - lambda_m * S_m * itn_effect - mu_m * S_m
    deriv(E_m) = lambda_m * S_m * itn_effect - delta_m * E_m - mu_m * E_m
    deriv(I_m) = delta_m * E_m - mu_m * I_m

    initial(S_h) = N_h - I_h0
    initial(E_h) = 0
    initial(I_h) = I_h0 * (1 - p_asymp)
    initial(A_h) = I_h0 * p_asymp
    initial(R_h) = 0
    initial(S_m) = K_m0 * N_h * 0.7
    initial(E_m) = K_m0 * N_h * 0.1
    initial(I_m) = K_m0 * N_h * 0.2
end

malaria_pars = (
    a=0.3, b=0.55, c_par=0.15, phi=0.5,
    delta_h=1/12, delta_m=0.1, gamma_h=0.2, gamma_a=0.005,
    omega=1/365, rho_relapse=1/730, p_asymp=0.5,
    mu_h=1/(50*365), mu_m=0.1, K_m0=20.0,
    amp=0.5, phase=60.0, N_h=10000.0, I_h0=100.0,
    itn_cov=0.0, itn_eff=0.5,
)

sim_times_malaria = collect(0.0:1.0:1095.0)  # 3 years daily

# Warmup
dust_system_simulate(malaria, malaria_pars; times=sim_times_malaria, seed=1)

b_malaria_sim = @benchmark dust_system_simulate($malaria, $malaria_pars;
    times=$sim_times_malaria, seed=1) samples=50 evals=1
t_malaria_sim = median(b_malaria_sim).time / 1e6
println(@sprintf("  Simulation (3 years daily): %.2f ms", t_malaria_sim))
results["malaria_sim_julia_ms"] = t_malaria_sim

# ─── 3. SARS-CoV-2 Multi-Region (ODE, 3 regions, interpolation) ───────────

println("\n── 3. SARS-CoV-2 Multi-Region (3-region ODE) ──")

Rt_times = Float64[0, 30, 60, 90, 120, 180, 250, 365]
Rt_r1 = [2.5, 2.5, 1.2, 0.8, 1.1, 1.3, 1.5, 1.2]
Rt_r2 = [1.0, 1.8, 2.0, 1.0, 0.9, 1.2, 1.4, 1.1]
Rt_r3 = [1.0, 1.2, 1.5, 1.3, 0.9, 1.0, 1.1, 1.0]

covid = @odin begin
    n_Rt_times = parameter()
    dim(Rt_t) = n_Rt_times
    dim(Rt_v1) = n_Rt_times
    dim(Rt_v2) = n_Rt_times
    dim(Rt_v3) = n_Rt_times
    Rt_t = parameter()
    Rt_v1 = parameter()
    Rt_v2 = parameter()
    Rt_v3 = parameter()
    Rt_1 = interpolate(Rt_t, Rt_v1, :linear)
    Rt_2 = interpolate(Rt_t, Rt_v2, :linear)
    Rt_3 = interpolate(Rt_t, Rt_v3, :linear)

    gamma = parameter(0.2)
    sigma = parameter(0.333)
    ifr = parameter(0.005)
    rho = parameter(0.3)
    epsilon = parameter(0.05)
    N1 = parameter(5e6)
    N2 = parameter(3e6)
    N3 = parameter(1e6)
    I0_1 = parameter(100.0)
    I0_2 = parameter(10.0)
    I0_3 = parameter(1.0)
    c12 = parameter(0.1)
    c13 = parameter(0.05)
    c23 = parameter(0.1)

    foi_1 = Rt_1 * gamma * (I1 / N1 + epsilon * (c12 * I2 / N2 + c13 * I3 / N3))
    foi_2 = Rt_2 * gamma * (I2 / N2 + epsilon * (c12 * I1 / N1 + c23 * I3 / N3))
    foi_3 = Rt_3 * gamma * (I3 / N3 + epsilon * (c13 * I1 / N1 + c23 * I2 / N2))

    deriv(S1) = -foi_1 * S1
    deriv(E1) = foi_1 * S1 - sigma * E1
    deriv(I1) = sigma * E1 - gamma * I1
    deriv(R1) = (1 - ifr) * gamma * I1
    deriv(D1) = ifr * gamma * I1
    deriv(cum1) = foi_1 * S1
    deriv(S2) = -foi_2 * S2
    deriv(E2) = foi_2 * S2 - sigma * E2
    deriv(I2) = sigma * E2 - gamma * I2
    deriv(R2) = (1 - ifr) * gamma * I2
    deriv(D2) = ifr * gamma * I2
    deriv(cum2) = foi_2 * S2
    deriv(S3) = -foi_3 * S3
    deriv(E3) = foi_3 * S3 - sigma * E3
    deriv(I3) = sigma * E3 - gamma * I3
    deriv(R3) = (1 - ifr) * gamma * I3
    deriv(D3) = ifr * gamma * I3
    deriv(cum3) = foi_3 * S3

    initial(S1) = N1 - I0_1
    initial(E1) = 0
    initial(I1) = I0_1
    initial(R1) = 0
    initial(D1) = 0
    initial(cum1) = 0
    initial(S2) = N2 - I0_2
    initial(E2) = 0
    initial(I2) = I0_2
    initial(R2) = 0
    initial(D2) = 0
    initial(cum2) = 0
    initial(S3) = N3 - I0_3
    initial(E3) = 0
    initial(I3) = I0_3
    initial(R3) = 0
    initial(D3) = 0
    initial(cum3) = 0

    output(daily_cases_1) = rho * sigma * E1
    output(daily_cases_2) = rho * sigma * E2
    output(daily_cases_3) = rho * sigma * E3

    cases1 = data()
    cases2 = data()
    cases3 = data()
    cases1 ~ Poisson(rho * sigma * E1 * 7 + 1e-6)
    cases2 ~ Poisson(rho * sigma * E2 * 7 + 1e-6)
    cases3 ~ Poisson(rho * sigma * E3 * 7 + 1e-6)
end

covid_pars = (
    n_Rt_times=8.0, Rt_t=Rt_times,
    Rt_v1=Rt_r1, Rt_v2=Rt_r2, Rt_v3=Rt_r3,
    gamma=0.2, sigma=1/3, ifr=0.005, rho=0.3, epsilon=0.05,
    N1=5e6, N2=3e6, N3=1e6, I0_1=100.0, I0_2=10.0, I0_3=1.0,
    c12=0.1, c13=0.05, c23=0.1,
)

sim_times_covid = collect(0.0:1.0:365.0)

# Warmup
dust_system_simulate(covid, covid_pars; times=sim_times_covid, seed=1)

b_covid_sim = @benchmark dust_system_simulate($covid, $covid_pars;
    times=$sim_times_covid, seed=1) samples=50 evals=1
t_covid_sim = median(b_covid_sim).time / 1e6
println(@sprintf("  Simulation (365 days, 18 states): %.2f ms", t_covid_sim))
results["covid_sim_julia_ms"] = t_covid_sim

# Unfilter benchmark
# Generate synthetic data
Random.seed!(42)
sim = dust_system_simulate(covid, covid_pars; times=collect(0.0:7.0:364.0), seed=42)
println("  Sim shape: ", size(sim))
n_times = size(sim)[end]  # times is last dimension
n_weeks = n_times - 1  # exclude t=0
println("  Generating synthetic data: ", n_weeks, " weeks")
fdata_vec = NamedTuple{(:time,:cases1,:cases2,:cases3), NTuple{4,Float64}}[]
for w in 1:n_weeks
    push!(fdata_vec, (
        time = 7.0 * w,
        cases1 = Float64(rand(Poisson(0.3 * (1/3) * max(sim[2, 1, w+1], 0) * 7 + 1e-6))),
        cases2 = Float64(rand(Poisson(0.3 * (1/3) * max(sim[8, 1, w+1], 0) * 7 + 1e-6))),
        cases3 = Float64(rand(Poisson(0.3 * (1/3) * max(sim[14, 1, w+1], 0) * 7 + 1e-6))),
    ))
end
fdata = dust_filter_data(fdata_vec)

uf_covid = dust_unfilter_create(covid, fdata; time_start=0.0)
# Warmup
Odin.dust_unfilter_run!(uf_covid, covid_pars)

b_covid_uf = @benchmark Odin.dust_unfilter_run!($uf_covid, $covid_pars) samples=100 evals=1
t_covid_uf = median(b_covid_uf).time / 1e6
println(@sprintf("  Unfilter (52 weeks, 18 states): %.3f ms", t_covid_uf))
results["covid_unfilter_julia_ms"] = t_covid_uf

# ─── Summary ───────────────────────────────────────────────────────────────

println("\n" * "="^70)
println("SUMMARY — Julia Benchmark Results")
println("="^70)
println(@sprintf("  Mpox SEIR sim (200 ptcl, 180 days):    %8.1f ms", results["mpox_sim_julia_ms"]))
println(@sprintf("  Malaria ODE sim (3 years daily):       %8.2f ms", results["malaria_sim_julia_ms"]))
println(@sprintf("  SARS-CoV-2 ODE sim (365 days):         %8.2f ms", results["covid_sim_julia_ms"]))
println(@sprintf("  SARS-CoV-2 unfilter (52 weeks):        %8.3f ms", results["covid_unfilter_julia_ms"]))

# Save results
outdir = joinpath(@__DIR__, "results")
mkpath(outdir)
open(joinpath(outdir, "advanced_models_julia.csv"), "w") do io
    println(io, "model,task,julia_ms")
    println(io, "mpox_seir,sim_200ptcl_180d,", results["mpox_sim_julia_ms"])
    println(io, "malaria_simple,sim_3yr_daily,", results["malaria_sim_julia_ms"])
    println(io, "sarscov2_3region,sim_365d,", results["covid_sim_julia_ms"])
    println(io, "sarscov2_3region,unfilter_52wk,", results["covid_unfilter_julia_ms"])
end
println("\nResults saved to benchmark/results/advanced_models_julia.csv")
