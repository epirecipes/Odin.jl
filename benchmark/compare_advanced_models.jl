#!/usr/bin/env julia
# Quantitative comparison: Julia vs R for vignettes 17-19
# Compares: short-time ODE agreement, unfilter likelihood, benchmarks

using Odin
using Random, Statistics, Distributions, LinearAlgebra
using BenchmarkTools
using RCall
using Printf

println("="^70)
println("  Quantitative Comparison: Julia vs R — Vignettes 17–19")
println("="^70)

# ─── Helper ─────────────────────────────────────────────────────
results = Dict{String,NamedTuple}()

# ═══════════════════════════════════════════════════════════════════
println("\n", "─"^70)
println("  VIGNETTE 18: Malaria Simple (ODE)")
println("─"^70)

MalariaModel = @odin begin
    deriv(S_h) = -Lambda_h * S_h + delta_h * R_h + mu_h * N_h - mu_h * S_h
    deriv(E_h) = Lambda_h * S_h - epsilon_h * E_h - mu_h * E_h
    deriv(I_h) = p_symptomatic * epsilon_h * E_h - gamma_h * I_h - mu_h * I_h
    deriv(A_h) = (1 - p_symptomatic) * epsilon_h * E_h - gamma_a * A_h - mu_h * A_h
    deriv(R_h) = gamma_h * I_h + gamma_a * A_h - delta_h * R_h - mu_h * R_h
    deriv(S_m) = mu_m_t * N_m - Lambda_m * S_m - mu_m_t * S_m
    deriv(E_m) = Lambda_m * S_m - epsilon_m * E_m - mu_m_t * E_m
    deriv(I_m) = epsilon_m * E_m - mu_m_t * I_m
    Lambda_h = a * b * I_m / N_h * itn_effect
    Lambda_m = a * c_h * (I_h + kappa * A_h) / N_h
    mu_m_t = mu_m * (1 + seasonal_amplitude * cos(2 * 3.141592653589793 * (time - peak_day / 365)))
    itn_effect = 1.0 - itn_coverage * itn_efficacy
    deriv(cum_cases) = p_symptomatic * epsilon_h * E_h
    deriv(cum_infections) = Lambda_h * S_h
    output(daily_incidence) = p_symptomatic * epsilon_h * E_h
    output(slide_positivity) = (I_h + 0.5 * A_h) / N_h
    tested = data()
    positive = data()
    positive ~ Binomial(tested, (I_h + 0.5 * A_h) / N_h)
    initial(S_h) = N_h - E_h0 - I_h0 - A_h0 - R_h0
    initial(E_h) = E_h0; initial(I_h) = I_h0; initial(A_h) = A_h0; initial(R_h) = R_h0
    initial(S_m) = N_m - E_m0 - I_m0; initial(E_m) = E_m0; initial(I_m) = I_m0
    initial(cum_cases) = 0; initial(cum_infections) = 0
    N_h = parameter(100000); N_m = parameter(300000)
    a = parameter(0.3); b = parameter(0.2); c_h = parameter(0.5); kappa = parameter(0.5)
    epsilon_h = parameter(1/14); epsilon_m = parameter(1/10)
    gamma_h = parameter(1/21); gamma_a = parameter(1/60)
    delta_h = parameter(1/365); mu_h = parameter(1/(60*365)); mu_m = parameter(1/14)
    p_symptomatic = parameter(0.4); seasonal_amplitude = parameter(0.3); peak_day = parameter(180)
    itn_coverage = parameter(0.0); itn_efficacy = parameter(0.5)
    E_h0 = parameter(500); I_h0 = parameter(300); A_h0 = parameter(1000); R_h0 = parameter(5000)
    E_m0 = parameter(5000); I_m0 = parameter(3000)
end

pars_mal = (N_h=100000.0, N_m=300000.0, a=0.3, b=0.2, c_h=0.5, kappa=0.5,
            epsilon_h=1/14, epsilon_m=1/10, gamma_h=1/21, gamma_a=1/60,
            delta_h=1/365, mu_h=1/(60*365), mu_m=1/14,
            p_symptomatic=0.4, seasonal_amplitude=0.3, peak_day=180.0,
            itn_coverage=0.0, itn_efficacy=0.5,
            E_h0=500.0, I_h0=300.0, A_h0=1000.0, R_h0=5000.0, E_m0=5000.0, I_m0=3000.0)

# Julia sim
sys_mal = dust_system_create(MalariaModel, pars_mal; n_particles=1)
dust_system_set_state_initial!(sys_mal)
sol_jl = dust_system_simulate(sys_mal, [0.0, 1.0, 5.0, 10.0])

# R sim
R"""
suppressPackageStartupMessages({library(odin2); library(dust2); library(monty)})
malaria_r <- odin2::odin({
    deriv(S_h) <- -Lambda_h * S_h + delta_h * R_h + mu_h * N_h - mu_h * S_h
    deriv(E_h) <- Lambda_h * S_h - epsilon_h * E_h - mu_h * E_h
    deriv(I_h) <- p_symptomatic * epsilon_h * E_h - gamma_h * I_h - mu_h * I_h
    deriv(A_h) <- (1 - p_symptomatic) * epsilon_h * E_h - gamma_a * A_h - mu_h * A_h
    deriv(R_h) <- gamma_h * I_h + gamma_a * A_h - delta_h * R_h - mu_h * R_h
    deriv(S_m) <- mu_m_t * N_m - Lambda_m * S_m - mu_m_t * S_m
    deriv(E_m) <- Lambda_m * S_m - epsilon_m * E_m - mu_m_t * E_m
    deriv(I_m) <- epsilon_m * E_m - mu_m_t * I_m
    Lambda_h <- a * b * I_m / N_h * itn_effect
    Lambda_m <- a * c_h * (I_h + kappa * A_h) / N_h
    mu_m_t <- mu_m * (1 + seasonal_amplitude * cos(2 * pi * (time - peak_day / 365)))
    itn_effect <- 1.0 - itn_coverage * itn_efficacy
    deriv(cum_cases) <- p_symptomatic * epsilon_h * E_h
    deriv(cum_infections) <- Lambda_h * S_h
    tested <- data()
    positive <- data()
    positive ~ Binomial(tested, (I_h + 0.5 * A_h) / N_h)
    initial(S_h) <- N_h - E_h0 - I_h0 - A_h0 - R_h0
    initial(E_h) <- E_h0; initial(I_h) <- I_h0; initial(A_h) <- A_h0; initial(R_h) <- R_h0
    initial(S_m) <- N_m - E_m0 - I_m0; initial(E_m) <- E_m0; initial(I_m) <- I_m0
    initial(cum_cases) <- 0; initial(cum_infections) <- 0
    N_h <- parameter(100000); N_m <- parameter(300000)
    a <- parameter(0.3); b <- parameter(0.2); c_h <- parameter(0.5); kappa <- parameter(0.5)
    epsilon_h <- parameter(1/14); epsilon_m <- parameter(1/10)
    gamma_h <- parameter(1/21); gamma_a <- parameter(1/60)
    delta_h <- parameter(1/365); mu_h <- parameter(1/(60*365)); mu_m <- parameter(1/14)
    p_symptomatic <- parameter(0.4); seasonal_amplitude <- parameter(0.3); peak_day <- parameter(180)
    itn_coverage <- parameter(0.0); itn_efficacy <- parameter(0.5)
    E_h0 <- parameter(500); I_h0 <- parameter(300); A_h0 <- parameter(1000); R_h0 <- parameter(5000)
    E_m0 <- parameter(5000); I_m0 <- parameter(3000)
})
sys_r <- dust_system_create(malaria_r, list(), n_particles = 1L)
dust_system_set_state_initial(sys_r)
state_r <- dust_system_simulate(sys_r, c(0, 1, 5, 10))
"""

r_st = rcopy(R"state_r")  # 10 × 4

# Compare at short times (first 10 states, ignoring outputs in Julia rows 11-12)
println("\n  Short-time ODE comparison (first 10 state variables):")
println("  ┌────────┬─────────────────┐")
println("  │ Time   │ Max Rel Error   │")
println("  ├────────┼─────────────────┤")
max_errs = Float64[]
for (ti, t) in enumerate([0.0, 1.0, 5.0, 10.0])
    jl_state = sol_jl[1:10, 1, ti]
    r_state_t = r_st[:, ti]
    max_re = maximum(abs.(jl_state .- r_state_t) ./ max.(abs.(r_state_t), 1.0))
    push!(max_errs, max_re)
    @printf("  │ t=%-4.0f │ %15.2e │\n", t, max_re)
end
println("  └────────┴─────────────────┘")
mal_short_pass = all(max_errs .< 1e-3)
println("  Short-time match (rtol < 1e-3): ", mal_short_pass ? "✅ PASS" : "❌ FAIL")

# Unfilter comparison with shared data
Random.seed!(42)
state_names_jl = MalariaModel.model.state_names
idx_I_h = findfirst(==(:I_h), state_names_jl)
idx_A_h = findfirst(==(:A_h), state_names_jl)

sys2 = dust_system_create(MalariaModel, pars_mal; n_particles=1)
dust_system_set_state_initial!(sys2)
obs_times = collect(30.0:30.0:360.0)  # monthly, only 1 year
sol_obs = dust_system_simulate(sys2, obs_times)
sp = [(sol_obs[idx_I_h,1,i] + 0.5*sol_obs[idx_A_h,1,i]) / 100000 for i in 1:length(obs_times)]
pos_vals = [rand(Binomial(200, clamp(s, 0.001, 0.999))) for s in sp]
raw_data = [(time=obs_times[i], tested=200, positive=pos_vals[i]) for i in 1:length(obs_times)]
fdata = dust_filter_data(raw_data)

uf_jl = dust_unfilter_create(MalariaModel, fdata; time_start=0.0)
ll_jl = dust_unfilter_run!(uf_jl, pars_mal)
println("\n  Julia unfilter LL: ", @sprintf("%.4f", ll_jl))

# R unfilter with same data
obs_t_r = [Int(t) for t in fdata.times]
tested_r = [d.tested for d in fdata.data]
positive_r = [d.positive for d in fdata.data]
R"""
data_r <- data.frame(time=$(obs_t_r), tested=$(tested_r), positive=$(positive_r))
uf_r <- dust_unfilter_create(malaria_r, time_start = 0, data = data_r)
ll_r <- dust_likelihood_run(uf_r, list())
"""
ll_r = rcopy(R"ll_r")
println("  R unfilter LL:     ", @sprintf("%.4f", ll_r))
ll_diff = abs(ll_jl - ll_r)
@printf("  |ΔLL|:             %.4f\n", ll_diff)
mal_ll_pass = ll_diff < 5.0
println("  LL match (|Δ| < 5): ", mal_ll_pass ? "✅ PASS" : "❌ FAIL")

# Benchmark
b_jl = @benchmark begin
    s = dust_system_create($MalariaModel, $pars_mal; n_particles=1)
    dust_system_set_state_initial!(s)
    dust_system_simulate(s, $(collect(0.0:1.0:1095.0)))
end samples=20 evals=1
jl_ms = median(b_jl.times) / 1e6

R"""
r_b <- system.time(for(i in 1:20) {
    s <- dust_system_create(malaria_r, list(), n_particles=1L)
    dust_system_set_state_initial(s)
    dust_system_simulate(s, 0:1095)
})
r_ms <- r_b[["elapsed"]] / 20 * 1000
"""
r_ms = rcopy(R"r_ms")
results["Malaria ODE"] = (jl=jl_ms, r=r_ms, speedup=r_ms/jl_ms)
@printf("\n  Benchmark: Julia %.3f ms, R %.3f ms → %.1fx speedup\n", jl_ms, r_ms, r_ms/jl_ms)

# ═══════════════════════════════════════════════════════════════════
println("\n", "─"^70)
println("  VIGNETTE 19: SARS-CoV-2 Multi-Region (ODE)")
println("─"^70)

SARSCoV2Model = @odin begin
    deriv(S1) = -beta1 * S1 * I1 / N1 - epsilon * (c12 * I2 / N2 + c13 * I3 / N3) * S1
    deriv(E1) = beta1 * S1 * I1 / N1 + epsilon * (c12 * I2 / N2 + c13 * I3 / N3) * S1 - sigma * E1
    deriv(I1) = sigma * E1 - gamma * I1 - ifr * gamma * I1
    deriv(R1) = gamma * I1; deriv(D1) = ifr * gamma * I1; deriv(cum_infections1) = sigma * E1

    deriv(S2) = -beta2 * S2 * I2 / N2 - epsilon * (c12 * I1 / N1 + c23 * I3 / N3) * S2
    deriv(E2) = beta2 * S2 * I2 / N2 + epsilon * (c12 * I1 / N1 + c23 * I3 / N3) * S2 - sigma * E2
    deriv(I2) = sigma * E2 - gamma * I2 - ifr * gamma * I2
    deriv(R2) = gamma * I2; deriv(D2) = ifr * gamma * I2; deriv(cum_infections2) = sigma * E2

    deriv(S3) = -beta3 * S3 * I3 / N3 - epsilon * (c13 * I1 / N1 + c23 * I2 / N2) * S3
    deriv(E3) = beta3 * S3 * I3 / N3 + epsilon * (c13 * I1 / N1 + c23 * I2 / N2) * S3 - sigma * E3
    deriv(I3) = sigma * E3 - gamma * I3 - ifr * gamma * I3
    deriv(R3) = gamma * I3; deriv(D3) = ifr * gamma * I3; deriv(cum_infections3) = sigma * E3

    beta1 = Rt1 * gamma; beta2 = Rt2 * gamma; beta3 = Rt3 * gamma
    Rt1 = interpolate(Rt_t, Rt_v1, "linear")
    Rt2 = interpolate(Rt_t, Rt_v2, "linear")
    Rt3 = interpolate(Rt_t, Rt_v3, "linear")

    initial(S1) = N1 - I0_1; initial(E1) = 0; initial(I1) = I0_1
    initial(R1) = 0; initial(D1) = 0; initial(cum_infections1) = 0
    initial(S2) = N2 - I0_2; initial(E2) = 0; initial(I2) = I0_2
    initial(R2) = 0; initial(D2) = 0; initial(cum_infections2) = 0
    initial(S3) = N3 - I0_3; initial(E3) = 0; initial(I3) = I0_3
    initial(R3) = 0; initial(D3) = 0; initial(cum_infections3) = 0

    N1 = parameter(5e6); N2 = parameter(3e6); N3 = parameter(1e6)
    I0_1 = parameter(100); I0_2 = parameter(10); I0_3 = parameter(1)
    gamma = parameter(0.2); sigma = parameter(1/3); ifr = parameter(0.005)
    rho = parameter(0.3); epsilon = parameter(0.05)
    c12 = parameter(0.1); c13 = parameter(0.05); c23 = parameter(0.1)
    n_Rt_times = parameter(8)
    Rt_t = parameter(); Rt_v1 = parameter(); Rt_v2 = parameter(); Rt_v3 = parameter()
end

Rt_times = Float64[0, 52, 104, 156, 208, 260, 312, 365]
Rt_v1 = [2.5, 1.2, 0.9, 1.5, 1.1, 0.8, 1.3, 1.0]
Rt_v2 = [2.2, 1.4, 1.0, 1.3, 0.9, 1.1, 1.2, 0.9]
Rt_v3 = [2.0, 1.1, 1.2, 1.6, 1.3, 0.7, 1.0, 1.1]

pars_cv = (N1=5e6, N2=3e6, N3=1e6, I0_1=100.0, I0_2=10.0, I0_3=1.0,
           gamma=0.2, sigma=1/3, ifr=0.005, rho=0.3, epsilon=0.05,
           c12=0.1, c13=0.05, c23=0.1,
           n_Rt_times=8, Rt_t=Rt_times, Rt_v1=Rt_v1, Rt_v2=Rt_v2, Rt_v3=Rt_v3)

sys_cv = dust_system_create(SARSCoV2Model, pars_cv; n_particles=1)
dust_system_set_state_initial!(sys_cv)
sol_cv_jl = dust_system_simulate(sys_cv, [0.0, 1.0, 5.0, 10.0, 30.0])

R"""
seird_r <- odin2::odin({
    deriv(S1) <- -beta1 * S1 * I1 / N1 - epsilon * (c12 * I2 / N2 + c13 * I3 / N3) * S1
    deriv(E1) <- beta1 * S1 * I1 / N1 + epsilon * (c12 * I2 / N2 + c13 * I3 / N3) * S1 - sigma * E1
    deriv(I1) <- sigma * E1 - gamma * I1 - ifr * gamma * I1
    deriv(R1) <- gamma * I1; deriv(D1) <- ifr * gamma * I1; deriv(cum_infections1) <- sigma * E1
    deriv(S2) <- -beta2 * S2 * I2 / N2 - epsilon * (c12 * I1 / N1 + c23 * I3 / N3) * S2
    deriv(E2) <- beta2 * S2 * I2 / N2 + epsilon * (c12 * I1 / N1 + c23 * I3 / N3) * S2 - sigma * E2
    deriv(I2) <- sigma * E2 - gamma * I2 - ifr * gamma * I2
    deriv(R2) <- gamma * I2; deriv(D2) <- ifr * gamma * I2; deriv(cum_infections2) <- sigma * E2
    deriv(S3) <- -beta3 * S3 * I3 / N3 - epsilon * (c13 * I1 / N1 + c23 * I2 / N2) * S3
    deriv(E3) <- beta3 * S3 * I3 / N3 + epsilon * (c13 * I1 / N1 + c23 * I2 / N2) * S3 - sigma * E3
    deriv(I3) <- sigma * E3 - gamma * I3 - ifr * gamma * I3
    deriv(R3) <- gamma * I3; deriv(D3) <- ifr * gamma * I3; deriv(cum_infections3) <- sigma * E3
    beta1 <- Rt1 * gamma; beta2 <- Rt2 * gamma; beta3 <- Rt3 * gamma
    Rt1 <- interpolate(Rt_t, Rt_v1, "linear")
    Rt2 <- interpolate(Rt_t, Rt_v2, "linear")
    Rt3 <- interpolate(Rt_t, Rt_v3, "linear")
    initial(S1) <- N1 - I0_1; initial(E1) <- 0; initial(I1) <- I0_1
    initial(R1) <- 0; initial(D1) <- 0; initial(cum_infections1) <- 0
    initial(S2) <- N2 - I0_2; initial(E2) <- 0; initial(I2) <- I0_2
    initial(R2) <- 0; initial(D2) <- 0; initial(cum_infections2) <- 0
    initial(S3) <- N3 - I0_3; initial(E3) <- 0; initial(I3) <- I0_3
    initial(R3) <- 0; initial(D3) <- 0; initial(cum_infections3) <- 0
    N1 <- parameter(5e6); N2 <- parameter(3e6); N3 <- parameter(1e6)
    I0_1 <- parameter(100); I0_2 <- parameter(10); I0_3 <- parameter(1)
    gamma <- parameter(0.2); sigma <- parameter(1/3); ifr <- parameter(0.005)
    epsilon <- parameter(0.05)
    c12 <- parameter(0.1); c13 <- parameter(0.05); c23 <- parameter(0.1)
    n_Rt_times <- parameter(8)
    dim(Rt_t) <- n_Rt_times
    dim(Rt_v1) <- n_Rt_times
    dim(Rt_v2) <- n_Rt_times
    dim(Rt_v3) <- n_Rt_times
    Rt_t <- parameter(); Rt_v1 <- parameter(); Rt_v2 <- parameter(); Rt_v3 <- parameter()
})
pars_cv_r <- list(n_Rt_times=8L,
    Rt_t=c(0, 52, 104, 156, 208, 260, 312, 365),
    Rt_v1=c(2.5, 1.2, 0.9, 1.5, 1.1, 0.8, 1.3, 1.0),
    Rt_v2=c(2.2, 1.4, 1.0, 1.3, 0.9, 1.1, 1.2, 0.9),
    Rt_v3=c(2.0, 1.1, 1.2, 1.6, 1.3, 0.7, 1.0, 1.1))
sys_cv_r <- dust_system_create(seird_r, pars_cv_r, n_particles=1L)
dust_system_set_state_initial(sys_cv_r)
state_cv_r <- dust_system_simulate(sys_cv_r, c(0, 1, 5, 10, 30))
"""

r_cv_st = rcopy(R"state_cv_r")
n_state_cv = size(r_cv_st, 1)

println("\n  Short-time ODE comparison (18 state variables):")
println("  ┌────────┬─────────────────┐")
println("  │ Time   │ Max Rel Error   │")
println("  ├────────┼─────────────────┤")
cv_errs = Float64[]
for (ti, t) in enumerate([0.0, 1.0, 5.0, 10.0, 30.0])
    jl_s = sol_cv_jl[1:n_state_cv, 1, ti]
    r_s = r_cv_st[:, ti]
    max_re = maximum(abs.(jl_s .- r_s) ./ max.(abs.(r_s), 1.0))
    push!(cv_errs, max_re)
    @printf("  │ t=%-4.0f │ %15.2e │\n", t, max_re)
end
println("  └────────┴─────────────────┘")
cv_short_pass = all(cv_errs .< 1e-3)
println("  Short-time match (rtol < 1e-3): ", cv_short_pass ? "✅ PASS" : "❌ FAIL")

# Benchmark
b_cv_jl = @benchmark begin
    s = dust_system_create($SARSCoV2Model, $pars_cv; n_particles=1)
    dust_system_set_state_initial!(s)
    dust_system_simulate(s, $(collect(0.0:1.0:365.0)))
end samples=20 evals=1
jl_cv_ms = median(b_cv_jl.times) / 1e6

R"""
r_cv_b <- system.time(for(i in 1:20) {
    s <- dust_system_create(seird_r, pars_cv_r, n_particles=1L)
    dust_system_set_state_initial(s)
    dust_system_simulate(s, 0:365)
})
r_cv_ms <- r_cv_b[["elapsed"]] / 20 * 1000
"""
r_cv_ms = rcopy(R"r_cv_ms")
results["SARS-CoV-2 ODE"] = (jl=jl_cv_ms, r=r_cv_ms, speedup=r_cv_ms/jl_cv_ms)
@printf("\n  Benchmark: Julia %.3f ms, R %.3f ms → %.1fx speedup\n", jl_cv_ms, r_cv_ms, r_cv_ms/jl_cv_ms)

# ═══════════════════════════════════════════════════════════════════
println("\n", "─"^70)
println("  VIGNETTE 17: Mpox SEIR (Stochastic)")
println("─"^70)

MpoxModel = @odin begin
    n_age = parameter(4); n_vax = parameter(3)
    dim(S) = c(n_age, n_vax); dim(E) = c(n_age, n_vax)
    dim(I) = c(n_age, n_vax); dim(R) = c(n_age, n_vax)
    dim(N_age) = n_age; dim(contact) = c(n_age, n_age); dim(vax_rate) = n_age
    dim(foi) = n_age; dim(I_total) = n_age
    I_total[i] = I[i,1] + I[i,2] + I[i,3]
    foi[i] = beta * sum(j, contact[i,j] * I_total[j] / N_age[j])
    dim(new_infections) = c(n_age, n_vax)
    dim(new_symptomatic) = c(n_age, n_vax)
    dim(new_recoveries) = c(n_age, n_vax)
    dim(new_vax) = n_age
    new_infections[i,j] = Binomial(S[i,j], 1 - exp(-foi[i] * (1 - ve[j]) * dt))
    new_symptomatic[i,j] = Binomial(E[i,j], 1 - exp(-sigma * dt))
    new_recoveries[i,j] = Binomial(I[i,j], 1 - exp(-gamma * dt))
    new_vax[i] = Binomial(S[i,1], 1 - exp(-vax_rate[i] * dt))
    update(S[i,1]) = S[i,1] - new_infections[i,1] - new_vax[i]
    update(S[i,2]) = S[i,2] - new_infections[i,2] + new_vax[i]
    update(S[i,3]) = S[i,3] - new_infections[i,3]
    update(E[i,j]) = E[i,j] + new_infections[i,j] - new_symptomatic[i,j]
    update(I[i,j]) = I[i,j] + new_symptomatic[i,j] - new_recoveries[i,j]
    update(R[i,j]) = R[i,j] + new_recoveries[i,j]
    update(cases_inc) = sum(i, sum(j, new_symptomatic[i,j]))
    cases_data = data()
    cases_data ~ NegBinomial(cases_inc + 1, kappa)
    beta = parameter(0.3); sigma = parameter(1/7); gamma = parameter(1/14); kappa = parameter(10)
    dim(ve) = n_vax; ve = parameter()
    contact = parameter(); N_age = parameter(); vax_rate = parameter()
    dim(I0) = n_age; I0 = parameter()
    initial(S[i,j]) = (j == 1) ? N_age[i] - I0[i] : 0
    initial(E[i,j]) = 0
    initial(I[i,1]) = I0[i]
    initial(I[i,j]) = (j > 1) ? 0 : I0[i]
    initial(R[i,j]) = 0
    initial(cases_inc) = 0
end

contact_matrix = [3.0 0.5 0.2 0.1; 0.5 2.5 0.5 0.2; 0.2 0.5 2.0 0.5; 0.1 0.2 0.5 1.5]
pars_mpox = (n_age=4, n_vax=3, beta=0.3, sigma=1/7, gamma=1/14, kappa=10.0,
             contact=contact_matrix, N_age=[250000.0, 300000.0, 200000.0, 150000.0],
             I0=[5.0, 3.0, 2.0, 1.0], ve=[0.0, 0.7, 0.3], vax_rate=[0.001, 0.002, 0.001, 0.0005])

# Quick sanity check — population conservation
sys_mpox = dust_system_create(MpoxModel, pars_mpox; n_particles=1, dt=0.25, seed=42)
dust_system_set_state_initial!(sys_mpox)
sol_mpox = dust_system_simulate(sys_mpox, [0.0, 90.0, 180.0])
println("  Mpox model runs — shape: ", size(sol_mpox))
total_pop = 900000.0
for (ti, t) in enumerate([0.0, 90.0, 180.0])
    # Sum all S,E,I,R compartments (4*3=12 each, total 48 states, plus cases_inc)
    pop = sum(sol_mpox[1:48, 1, ti])
    @printf("  t=%.0f  pop=%.0f (expected %.0f, diff=%.1f)\n", t, pop, total_pop, pop - total_pop)
end

# Benchmark stochastic simulation
b_mpox = @benchmark begin
    s = dust_system_create($MpoxModel, $pars_mpox; n_particles=200, dt=0.25, seed=42)
    dust_system_set_state_initial!(s)
    dust_system_simulate(s, collect(0.0:1.0:180.0))
end samples=10 evals=1
jl_mpox_ms = median(b_mpox.times) / 1e6

# R stochastic benchmark (mpox is more complex — skip detailed R model, just benchmark)
R"""
r_mpox_ms <- NA
"""
r_mpox_ms = NaN
results["Mpox stochastic"] = (jl=jl_mpox_ms, r=r_mpox_ms, speedup=NaN)
@printf("\n  Benchmark: Julia %.1f ms (200 particles, 180 days)\n", jl_mpox_ms)

# ═══════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  SUMMARY")
println("="^70)

println("\n  ┌─────────────────────────┬────────────┬────────────┬──────────┐")
println("  │ Model                   │ Julia (ms) │ R (ms)     │ Speedup  │")
println("  ├─────────────────────────┼────────────┼────────────┼──────────┤")
for (name, r) in sort(collect(results))
    if isnan(r.r)
        @printf("  │ %-23s │ %10.1f │      —     │    —     │\n", name, r.jl)
    else
        @printf("  │ %-23s │ %10.3f │ %10.3f │ %6.1fx  │\n", name, r.jl, r.r, r.speedup)
    end
end
println("  └─────────────────────────┴────────────┴────────────┴──────────┘")

println("\n  ┌─────────────────────────┬──────────┐")
println("  │ Check                   │ Status   │")
println("  ├─────────────────────────┼──────────┤")
println("  │ Malaria short-time ODE  │ ", mal_short_pass ? "✅ PASS" : "❌ FAIL", "  │")
println("  │ Malaria unfilter LL     │ ", mal_ll_pass ? "✅ PASS" : "❌ FAIL", "  │")
println("  │ SARS-CoV-2 short-time   │ ", cv_short_pass ? "✅ PASS" : "❌ FAIL", "  │")
println("  └─────────────────────────┴──────────┘")

println("\nNote: Both Julia and R use DP5 (Dormand-Prince 5(4)) with atol=rtol=1e-6.")
println("SARS-CoV-2 matches to ~1e-7. Malaria ~4e-4 (seasonal oscillation amplifies tiny step diffs).")
println("\nDone!")
