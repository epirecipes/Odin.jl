# Deterministic Mpox Inference Benchmark


## Introduction

Vignette 17 uses a stochastic age-structured mpox model. This companion
vignette adds a **deterministic** mpox benchmark model that is small
enough to run quickly, but still exercises the main inference path:

1.  piecewise transmission before and after an intervention,
2.  multiple infectious severity strata,
3.  deterministic likelihood evaluation,
4.  forward- and adjoint-based gradients.

The model follows the same compartment layout as the simple
deterministic mpox reference used elsewhere in this workspace.

``` julia
using BenchmarkTools
using DelimitedFiles
using Distributions
using LinearAlgebra
using Odin
using Random
using Statistics
```

## Model definition

``` julia
mpox_det = @odin begin
    total_I = I_mild + I_hosp + I_ICU
    beta = ifelse(time < intervention_day, beta_1, beta_2)
    contact = ifelse(time < intervention_day, contact_1, contact_2)

    deriv(S) = -beta * contact * S * total_I / N
    deriv(E) = beta * contact * S * total_I / N - gamma * E
    deriv(I_mild) = p_mild * gamma * E - sigma * I_mild - mu * I_mild
    deriv(I_hosp) = p_hosp * gamma * E - sigma * I_hosp - mu * I_hosp
    deriv(I_ICU) = p_ICU * gamma * E - sigma * I_ICU - mu * I_ICU
    deriv(R) = sigma * total_I
    deriv(D) = mu * total_I

    N = S + E + I_mild + I_hosp + I_ICU + R + D

    cases = data()
    cases ~ Poisson(max(total_I, 1e-6))

    initial(S) = S0
    initial(E) = E0
    initial(I_mild) = I_mild0
    initial(I_hosp) = I_hosp0
    initial(I_ICU) = I_ICU0
    initial(R) = R0
    initial(D) = D0

    S0 = parameter(9_969.0)
    E0 = parameter(20.0)
    I_mild0 = parameter(8.0)
    I_hosp0 = parameter(2.0)
    I_ICU0 = parameter(1.0)
    R0 = parameter(0.0)
    D0 = parameter(0.0)

    gamma = parameter(1 / 10)
    sigma = parameter(1 / 7)
    mu = parameter(0.01)
    p_mild = parameter(0.85)
    p_hosp = parameter(0.10)
    p_ICU = parameter(0.05)

    beta_1 = parameter(0.9, differentiate = true)
    beta_2 = parameter(0.45, differentiate = true)
    contact_1 = parameter(1.0, differentiate = true)
    contact_2 = parameter(0.65, differentiate = true)
    intervention_day = parameter(40.0)
end
```

    Odin.DustSystemGenerator{var"##OdinModel#277"}(var"##OdinModel#277"(7, [:S, :E, :I_mild, :I_hosp, :I_ICU, :R, :D], [:S0, :E0, :I_mild0, :I_hosp0, :I_ICU0, :R0, :D0, :gamma, :sigma, :mu, :p_mild, :p_hosp, :p_ICU, :beta_1, :beta_2, :contact_1, :contact_2, :intervention_day], (S0 = 9969.0, E0 = 20.0, I_mild0 = 8.0, I_hosp0 = 2.0, I_ICU0 = 1.0, R0 = 0.0, D0 = 0.0, gamma = 0.1, sigma = 0.14285714285714285, mu = 0.01, p_mild = 0.85, p_hosp = 0.1, p_ICU = 0.05, beta_1 = 0.9, beta_2 = 0.45, contact_1 = 1.0, contact_2 = 0.65, intervention_day = 40.0), true, false, true, false, false, false, Dict{Symbol, Array}()))

## Simulation setup

``` julia
true_pars = (
    S0=9_969.0, E0=20.0, I_mild0=8.0, I_hosp0=2.0, I_ICU0=1.0, R0=0.0, D0=0.0,
    gamma=1 / 10, sigma=1 / 7, mu=0.01, p_mild=0.85, p_hosp=0.10, p_ICU=0.05,
    beta_1=0.9, beta_2=0.45, contact_1=1.0, contact_2=0.65, intervention_day=40.0,
)

times = collect(0.0:1.0:90.0)
sim = simulate(mpox_det, true_pars; times=times, seed=123)

obs_times = collect(10.0:5.0:80.0)
obs_idx = [findfirst(==(t), times) for t in obs_times]

rng = Xoshiro(99)
obs_cases = [
    rand(rng, Poisson(max(sum(sim[3:5, 1, i]), 1e-6)))
    for i in obs_idx
]

obs = ObservedData([(time=t, cases=y) for (t, y) in zip(obs_times, obs_cases)])
obs
```

    Odin.FilterData{@NamedTuple{cases::Int64}}([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0], [(cases = 41,), (cases = 103,), (cases = 253,), (cases = 571,), (cases = 1061,), (cases = 1602,), (cases = 1868,), (cases = 1857,), (cases = 1496,), (cases = 1194,), (cases = 834,), (cases = 587,), (cases = 441,), (cases = 264,), (cases = 194,)])

The daily infectious burden is the sum of the mild, hospitalised, and
ICU compartments, which occupy rows 3:5 in this model.

## Deterministic likelihood

``` julia
candidate_pars = merge(true_pars, (
    beta_1=0.82,
    beta_2=0.52,
    contact_1=1.05,
    contact_2=0.60,
))

lik = Likelihood(mpox_det, obs; time_start=0.0)
ll = loglik(lik, candidate_pars)
ll
```

    -106.25000075402961

## Gradient benchmark

``` julia
packer = Packer([:beta_1, :beta_2, :contact_1, :contact_2])

grad_forward = loglik_gradient(lik, candidate_pars, packer; method=:forward)
grad_adjoint = loglik_gradient(lik, candidate_pars, packer; method=:adjoint)

(;
    loglik_forward = grad_forward.log_likelihood,
    loglik_adjoint = grad_adjoint.log_likelihood,
    gradient_forward = grad_forward.gradient,
    gradient_adjoint = grad_adjoint.gradient,
)
```

    (loglik_forward = -106.25000075402961, loglik_adjoint = -106.25000075402961, gradient_forward = [2071.527663880288, -56.75873810040799, 1617.7644017915723, -49.19090633415651], gradient_adjoint = [2071.556956697429, -56.764823572687355, 1617.787310525903, -49.196180270874315])

The forward and adjoint gradients agree closely on this compact
deterministic problem, with the adjoint path giving the same derivative
information without carrying all parameter sensitivities through the
solve.

``` julia
bench_ms(f; samples=20) = median(@benchmark $f() samples=samples evals=1).time / 1e6

function fit_logspace(lik, packer, fixed_pars, initial_free; step=0.2, max_iter=40, min_step=1e-6)
    theta = log.(Odin.pack(packer, initial_free))
    current_free = Odin.unpack(packer, exp.(theta))
    current_pars = merge(fixed_pars, current_free)
    current_ll = loglik(lik, current_pars)
    current_grad = loglik_gradient(lik, current_pars, packer; method=:adjoint).gradient
    history = Float64[current_ll]

    for _ in 1:max_iter
        grad_theta = current_grad .* exp.(theta)
        grad_norm = norm(grad_theta)
        grad_norm == 0 && break

        accepted = false
        trial_step = step
        while trial_step >= min_step
            theta_trial = theta .+ trial_step .* grad_theta ./ max(grad_norm, 1.0)
            free_trial = Odin.unpack(packer, exp.(theta_trial))
            pars_trial = merge(fixed_pars, free_trial)
            ll_trial = loglik(lik, pars_trial)
            if isfinite(ll_trial) && ll_trial > current_ll + 1e-6
                theta = theta_trial
                current_free = free_trial
                current_pars = pars_trial
                current_ll = ll_trial
                current_grad = loglik_gradient(lik, current_pars, packer; method=:adjoint).gradient
                push!(history, current_ll)
                accepted = true
                break
            end
            trial_step *= 0.5
        end

        accepted || break
    end

    (; free=current_free, pars=current_pars, log_likelihood=current_ll, history=history)
end

simulation_ms = bench_ms(() -> simulate(mpox_det, true_pars; times=times, seed=123))
likelihood_ms = bench_ms(() -> loglik(lik, candidate_pars))
forward_ms = bench_ms(() -> loglik_gradient(lik, candidate_pars, packer; method=:forward))
adjoint_ms = bench_ms(() -> loglik_gradient(lik, candidate_pars, packer; method=:adjoint))

benchmark_summary = (
    simulation_ms = simulation_ms,
    likelihood_ms = likelihood_ms,
    forward_gradient_ms = forward_ms,
    adjoint_gradient_ms = adjoint_ms,
    forward_over_likelihood = forward_ms / likelihood_ms,
    adjoint_over_likelihood = adjoint_ms / likelihood_ms,
)

benchmark_summary
```

    (simulation_ms = 2.3876875, likelihood_ms = 0.0199165, forward_gradient_ms = 3.587917, adjoint_gradient_ms = 17.0810625, forward_over_likelihood = 180.14796776542062, adjoint_over_likelihood = 857.633745889087)

## DZA real-data slice

The synthetic benchmark above is a fast smoke test. To move one step
closer to the production `mpoxseir` workflow, this vignette also vendors
a processed example DZA data slice and the corresponding intervention
multiplier:

``` julia
dza_daily, _ = readdlm("data/dza_daily.csv", ',', Float64, '\n', header=true)
dza_intervention, _ = readdlm("data/dza_intervention.csv", ',', Float64, '\n', header=true)

dza_obs = ObservedData([
    (time=row[1], cases=round(Int, row[2]), deaths=round(Int, row[3]))
    for row in eachrow(dza_daily)
])

(;
    n_observations = size(dza_daily, 1),
    first_day = dza_daily[1, 1],
    last_day = dza_daily[end, 1],
    peak_cases = maximum(dza_daily[:, 2]),
    peak_deaths = maximum(dza_daily[:, 3]),
)
```

    (n_observations = 47, first_day = 0.0, last_day = 56.0, peak_cases = 185.0, peak_deaths = 42.0)

## DZA interpolation model

``` julia
mpox_det_dza = @odin begin
    contact_multiplier = interpolate(contact_time, contact_value, :constant)
    total_I = I_mild + I_hosp + I_ICU
    force = beta * contact_multiplier * S * total_I / N
    cases_rate = case_scale * gamma * E
    deaths_rate = death_scale * mu * total_I

    deriv(S) = -force
    deriv(E) = force - gamma * E
    deriv(I_mild) = p_mild * gamma * E - sigma * I_mild - mu * I_mild
    deriv(I_hosp) = p_hosp * gamma * E - sigma * I_hosp - mu * I_hosp
    deriv(I_ICU) = p_ICU * gamma * E - sigma * I_ICU - mu * I_ICU
    deriv(R) = sigma * total_I
    deriv(D) = deaths_rate

    N = S + E + I_mild + I_hosp + I_ICU + R + D

    output(cases_rate_out) = cases_rate
    output(deaths_rate_out) = deaths_rate

    cases = data()
    deaths = data()
    cases ~ Poisson(max(cases_rate, 1e-6))
    deaths ~ Poisson(max(deaths_rate, 1e-6))

    initial(S) = S0
    initial(E) = E0
    initial(I_mild) = I_mild0
    initial(I_hosp) = I_hosp0
    initial(I_ICU) = I_ICU0
    initial(R) = R0
    initial(D) = D0

    contact_time = parameter(rank=1)
    contact_value = parameter(rank=1)
    S0 = parameter(500_000.0)
    E0 = parameter(40.0)
    I_mild0 = parameter(10.0)
    I_hosp0 = parameter(3.0)
    I_ICU0 = parameter(1.0)
    R0 = parameter(0.0)
    D0 = parameter(0.0)
    gamma = parameter(1 / 7)
    sigma = parameter(1 / 10)
    mu = parameter(0.004, differentiate = true)
    p_mild = parameter(0.85)
    p_hosp = parameter(0.10)
    p_ICU = parameter(0.05)
    beta = parameter(0.38, differentiate = true)
    case_scale = parameter(8.0, differentiate = true)
    death_scale = parameter(2.5, differentiate = true)
end
```

    Odin.DustSystemGenerator{var"##OdinModel#283"}(var"##OdinModel#283"(7, [:S, :E, :I_mild, :I_hosp, :I_ICU, :R, :D], [:contact_time, :contact_value, :S0, :E0, :I_mild0, :I_hosp0, :I_ICU0, :R0, :D0, :gamma, :sigma, :mu, :p_mild, :p_hosp, :p_ICU, :beta, :case_scale, :death_scale], (S0 = 500000.0, E0 = 40.0, I_mild0 = 10.0, I_hosp0 = 3.0, I_ICU0 = 1.0, R0 = 0.0, D0 = 0.0, gamma = 0.14285714285714285, sigma = 0.1, mu = 0.004, p_mild = 0.85, p_hosp = 0.1, p_ICU = 0.05, beta = 0.38, case_scale = 8.0, death_scale = 2.5), true, false, true, true, true, false, Dict{Symbol, Array}()))

This is still deliberately compact, but it adds two useful parity
features:

1.  a time-varying contact multiplier via `interpolate(..., :constant)`,
    and
2.  a two-stream deterministic likelihood over cases and deaths.

## DZA likelihood and gradient benchmark

``` julia
dza_pars = (
    contact_time=dza_intervention[:, 1],
    contact_value=dza_intervention[:, 2],
    S0=500_000.0, E0=40.0, I_mild0=10.0, I_hosp0=3.0, I_ICU0=1.0, R0=0.0, D0=0.0,
    gamma=1 / 7, sigma=1 / 10, mu=0.004, p_mild=0.85, p_hosp=0.10, p_ICU=0.05,
    beta=0.38, case_scale=8.0, death_scale=2.5,
)

dza_lik = Likelihood(mpox_det_dza, dza_obs; time_start=-1.0)
dza_packer = Packer([:beta, :mu, :case_scale, :death_scale])

dza_forward = loglik_gradient(dza_lik, dza_pars, dza_packer; method=:forward)
dza_adjoint = loglik_gradient(dza_lik, dza_pars, dza_packer; method=:adjoint)

(;
    loglik_forward = dza_forward.log_likelihood,
    loglik_adjoint = dza_adjoint.log_likelihood,
    gradient_forward = dza_forward.gradient,
    gradient_adjoint = dza_adjoint.gradient,
)
```

    (loglik_forward = -8060.6208644317785, loglik_adjoint = -8060.6208644317785, gradient_forward = [-98666.92198857598, 225692.83842888684, -1395.1527263156295, 99.94176253249357], gradient_adjoint = [-96129.82572786299, 221229.0627217178, -1395.1527263156295, 99.93686455002258])

``` julia
dza_simulation_ms = bench_ms(() -> simulate(
    mpox_det_dza,
    dza_pars;
    times=collect(0.0:1.0:maximum(dza_daily[:, 1])),
    seed=123,
))
dza_likelihood_ms = bench_ms(() -> loglik(dza_lik, dza_pars))
dza_forward_ms = bench_ms(() -> loglik_gradient(dza_lik, dza_pars, dza_packer; method=:forward))
dza_adjoint_ms = bench_ms(() -> loglik_gradient(dza_lik, dza_pars, dza_packer; method=:adjoint))

(;
    simulation_ms = dza_simulation_ms,
    likelihood_ms = dza_likelihood_ms,
    forward_gradient_ms = dza_forward_ms,
    adjoint_gradient_ms = dza_adjoint_ms,
    forward_over_likelihood = dza_forward_ms / dza_likelihood_ms,
    adjoint_over_likelihood = dza_adjoint_ms / dza_likelihood_ms,
)
```

    (simulation_ms = 2.598229, likelihood_ms = 0.0525625, forward_gradient_ms = 25.778312, adjoint_gradient_ms = 104.491271, forward_over_likelihood = 490.43161950059454, adjoint_over_likelihood = 1987.9433246135552)

## DZA point fit

The corrected deterministic gradients are now good enough to support a
small real-data point fit. To keep this compact example well-behaved, we
fit `beta`, `case_scale`, and `death_scale` in log-space while keeping
`mu` fixed; in this benchmark, `mu` and `death_scale` are otherwise too
confounded to make the fit especially informative.

``` julia
dza_fit_fixed = (
    contact_time=dza_intervention[:, 1],
    contact_value=dza_intervention[:, 2],
    S0=500_000.0, E0=40.0, I_mild0=10.0, I_hosp0=3.0, I_ICU0=1.0, R0=0.0, D0=0.0,
    gamma=1 / 7, sigma=1 / 10, mu=0.004, p_mild=0.85, p_hosp=0.10, p_ICU=0.05,
)
dza_fit_initial = (beta=0.24, case_scale=5.5, death_scale=1.6)
dza_fit_packer = Packer([:beta, :case_scale, :death_scale])
dza_fit_start_ll = loglik(dza_lik, merge(dza_fit_fixed, dza_fit_initial))
dza_fit = fit_logspace(dza_lik, dza_fit_packer, dza_fit_fixed, dza_fit_initial)

(;
    start_loglik = dza_fit_start_ll,
    final_loglik = dza_fit.log_likelihood,
    fitted = dza_fit.free,
    n_iterations = length(dza_fit.history) - 1,
)
```

    (start_loglik = -2856.048657807718, final_loglik = -1378.805775532576, fitted = (beta = 0.3813026036101985, case_scale = 1.5553437994787547, death_scale = 2.7254183082178907), n_iterations = 40)

``` julia
dza_fit_ms = bench_ms(
    () -> fit_logspace(dza_lik, dza_fit_packer, dza_fit_fixed, dza_fit_initial);
    samples=5,
)

(;
    point_fit_ms = dza_fit_ms,
    loglik_gain = dza_fit.log_likelihood - dza_fit_start_ll,
)
```

    (point_fit_ms = 3012.666667, loglik_gain = 1477.242882275142)

## Summary

This deterministic mpox model is intentionally much smaller than the
full `mpoxseir` application, but it is already large enough to show the
main runtime trade-off:

1.  simulation is cheap,
2.  deterministic likelihood evaluation is still fast,
3.  gradients cost more than a single likelihood evaluation,
4.  adjoint mode gives a practical route to gradient-based inference for
    richer deterministic mpox models, including a simple real-data point
    fit on the DZA slice.

The R companion vignette in `R/38_mpox_deterministic_benchmark.qmd` now
uses the same compact synthetic benchmark and the same processed DZA
real-data slice, so the trajectory, likelihood, and intervention-driven
benchmark outputs remain directly comparable across Julia and R.

## Age-structured extension

The benchmarks above are scalar (no array dimensions). We now add a
richer slice: a **3-age-group SEIRD** model with split latent stages,
age-specific susceptibility and case-fatality ratios, and per-age
Poisson case/death observation streams. This exercises the runtime-sized
array ODE solver, the deterministic unfilter for parameter-dependent
dimensions, and forward/adjoint gradients on a 21-state system.

``` julia
mpox_age = @odin begin
    n_age = parameter(3)

    # Age-specific susceptibility and CFR (parameter arrays)
    dim(susc) = n_age;  susc = parameter(rank = 1)
    dim(cfr)  = n_age;  cfr  = parameter(rank = 1)

    # Force of infection (uniform mixing, age-specific susceptibility)
    dim(I_total) = n_age
    I_total[i] = Ir[i] + Id[i]
    total_I = sum(I_total)
    dim(lambda) = n_age
    lambda[i] = beta * susc[i] * total_I / N

    # 7 compartments × n_age groups
    dim(S) = n_age; dim(Ea) = n_age; dim(Eb) = n_age
    dim(Ir) = n_age; dim(Id) = n_age
    dim(R) = n_age; dim(D) = n_age

    deriv(S[i])  = -lambda[i] * S[i]
    deriv(Ea[i]) =  lambda[i] * S[i] - 2 * gamma_E * Ea[i]
    deriv(Eb[i]) =  2 * gamma_E * Ea[i] - 2 * gamma_E * Eb[i]
    deriv(Ir[i]) = (1 - cfr[i]) * 2 * gamma_E * Eb[i] - sigma * Ir[i]
    deriv(Id[i]) =  cfr[i]      * 2 * gamma_E * Eb[i] - mu * Id[i]
    deriv(R[i])  =  sigma * Ir[i]
    deriv(D[i])  =  mu * Id[i]

    N = sum(S) + sum(Ea) + sum(Eb) + sum(Ir) + sum(Id) + sum(R) + sum(D)

    # Observation rates
    dim(cases_rate) = n_age; dim(deaths_rate) = n_age
    cases_rate[i]  = case_scale  * 2 * gamma_E * Eb[i]
    deaths_rate[i] = death_scale * mu * Id[i]

    output(cases_out)  = sum(cases_rate)
    output(deaths_out) = sum(deaths_rate)

    # Per-age data / likelihood
    dim(cases) = n_age; dim(deaths) = n_age
    cases[i]  = data();  deaths[i] = data()
    cases[i]  ~ Poisson(max(cases_rate[i], 1e-6))
    deaths[i] ~ Poisson(max(deaths_rate[i], 1e-6))

    # Initial conditions
    dim(S0)  = n_age; S0  = parameter(rank = 1)
    dim(Ea0) = n_age; Ea0 = parameter(rank = 1)
    initial(S[i])  = S0[i];  initial(Ea[i]) = Ea0[i]
    initial(Eb[i]) = 0; initial(Ir[i]) = 0; initial(Id[i]) = 0
    initial(R[i])  = 0; initial(D[i])  = 0

    beta        = parameter(0.065, differentiate = true)
    gamma_E     = parameter(1 / 10)
    sigma       = parameter(1 / 14)
    mu          = parameter(1 / 7)
    case_scale  = parameter(6.0, differentiate = true)
    death_scale = parameter(2.0, differentiate = true)
end
```

    Odin.DustSystemGenerator{var"##OdinModel#284"}(var"##OdinModel#284"(0, [:S, :Ea, :Eb, :Ir, :Id, :R, :D], [:n_age, :susc, :cfr, :S0, :Ea0, :beta, :gamma_E, :sigma, :mu, :case_scale, :death_scale], (n_age = 3, beta = 0.065, gamma_E = 0.1, sigma = 0.07142857142857142, mu = 0.14285714285714285, case_scale = 6.0, death_scale = 2.0), true, false, true, true, false, false, Dict{Symbol, Array}()))

### Simulation and synthetic data

``` julia
age_pars = (
    n_age = 3,
    susc = [1.0, 1.2, 0.8],
    cfr  = [0.005, 0.02, 0.08],
    S0   = [4000.0, 3500.0, 2500.0],
    Ea0  = [8.0, 6.0, 4.0],
    beta = 0.065, gamma_E = 1/10, sigma = 1/14, mu = 1/7,
    case_scale = 6.0, death_scale = 2.0,
)

age_sim_times = collect(0.0:1.0:120.0)
age_sim = simulate(mpox_age, age_pars; times=age_sim_times, seed=42)
(; n_state=size(age_sim, 1), n_times=size(age_sim, 3))
```

    (n_state = 23, n_times = 121)

``` julia
snames = string.(Odin._odin_state_names(mpox_age.model, age_pars))
idx_Eb = [findfirst(==("Eb[$i]"), snames) for i in 1:3]
idx_Id = [findfirst(==("Id[$i]"), snames) for i in 1:3]

age_obs_times = collect(7.0:7.0:112.0)
rng = Xoshiro(77)
age_data = NamedTuple[]
for t in age_obs_times
    ti = findfirst(==(t), age_sim_times)
    c_rate = [age_pars.case_scale * 2 * age_pars.gamma_E * age_sim[idx_Eb[a], 1, ti] for a in 1:3]
    d_rate = [age_pars.death_scale * age_pars.mu * age_sim[idx_Id[a], 1, ti] for a in 1:3]
    push!(age_data, (
        time   = t,
        cases  = Float64[max(1.0, rand(rng, Poisson(max(r, 1e-6)))) for r in c_rate],
        deaths = Float64[max(1.0, rand(rng, Poisson(max(r, 1e-6)))) for r in d_rate],
    ))
end

(; n_obs=length(age_data), first_obs=age_data[1])
```

    (n_obs = 16, first_obs = (time = 7.0, cases = [4.0, 6.0, 2.0], deaths = [1.0, 1.0, 1.0]))

### Deterministic likelihood and gradients

``` julia
age_fdata = ObservedData(age_data)
age_lik = Likelihood(mpox_age, age_fdata; time_start=0.0)
age_ll = loglik(age_lik, age_pars)
age_ll_ms = bench_ms(() -> loglik(age_lik, age_pars); samples=20)

(; log_likelihood=age_ll, loglik_ms=age_ll_ms)
```

    (log_likelihood = -303.7312781743081, loglik_ms = 0.025604)

``` julia
age_packer = Packer([:beta, :case_scale, :death_scale])
age_fwd = loglik_gradient(age_lik, age_pars, age_packer; method=:forward)
age_adj = loglik_gradient(age_lik, age_pars, age_packer; method=:adjoint)

age_fwd_ms = bench_ms(
    () -> loglik_gradient(age_lik, age_pars, age_packer; method=:forward);
    samples=10,
)
age_adj_ms = bench_ms(
    () -> loglik_gradient(age_lik, age_pars, age_packer; method=:adjoint);
    samples=10,
)

(;
    forward_gradient  = age_fwd.gradient,
    adjoint_gradient  = age_adj.gradient,
    max_grad_diff     = maximum(abs.(age_fwd.gradient .- age_adj.gradient)),
    forward_ms        = age_fwd_ms,
    adjoint_ms        = age_adj_ms,
)
```

    (forward_gradient = [2131.226185339347, 4.1963196887688055, 23.73056942062135], adjoint_gradient = [2131.2276075008035, 4.1963196887688055, 23.73056942062135], max_grad_diff = 0.0014221614565030904, forward_ms = 0.7214795, adjoint_ms = 43.94025)

### Point fit

``` julia
age_fit_fixed = (
    n_age = 3,
    susc = [1.0, 1.2, 0.8],
    cfr  = [0.005, 0.02, 0.08],
    S0   = [4000.0, 3500.0, 2500.0],
    Ea0  = [8.0, 6.0, 4.0],
    gamma_E = 1/10, sigma = 1/14, mu = 1/7,
)
age_fit_initial = (beta=0.04, case_scale=4.0, death_scale=1.0)
age_fit_start_ll = loglik(age_lik, merge(age_fit_fixed, age_fit_initial))
age_fit = fit_logspace(age_lik, age_packer, age_fit_fixed, age_fit_initial;
                       step=0.15, max_iter=30)

(;
    start_loglik = age_fit_start_ll,
    final_loglik = age_fit.log_likelihood,
    fitted       = age_fit.free,
    improvement  = age_fit.log_likelihood - age_fit_start_ll,
    n_iterations = length(age_fit.history) - 1,
)
```

    (start_loglik = -448.42079128178085, final_loglik = -215.07121849846297, fitted = (beta = 0.09811875985221367, case_scale = 3.5295150456914164, death_scale = 8.560754295707737), improvement = 233.34957278331788, n_iterations = 30)

## Full summary

This vignette covers three progressively richer deterministic mpox
models:

1.  **Compact synthetic** — scalar SEIRD with piecewise intervention,
2.  **DZA real-data slice** — scalar SEIRD with interpolated contact
    multiplier and real weekly observations,
3.  **Age-structured slice** — 3-age-group SEIRD with split latent
    stages, age-specific susceptibility/CFR, and per-age Poisson
    case/death streams.

All three exercise the deterministic likelihood and gradient pipeline,
including both forward and adjoint modes. The age-structured model
additionally validates the runtime-sized array ODE solver and the
dynamic state-cache resizing in the deterministic unfilter.
