# Model Validation and Diagnostics


## Overview

After fitting a model to data, we need to assess whether the fit is
adequate. Odin.jl provides a suite of model validation tools:

1.  **Prior predictive checks** — are the priors reasonable?
2.  **Posterior predictive checks** — does the fitted model reproduce
    the data?
3.  **Residual diagnostics** — are there systematic patterns in the
    errors?
4.  **Calibration assessment** — are the prediction intervals
    well-calibrated?

We demonstrate these tools using an SIR model fitted to synthetic data,
then show how a misspecified model produces clearly bad diagnostics.

## Setup

``` julia
using Odin
using Random
using Statistics
using Distributions
```

## Define the SIR model

``` julia
sir = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    output(cases) = beta * S * I / N
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10)
    N = parameter(1000)
end
```

    Odin.DustSystemGenerator{var"##OdinModel#277"}(var"##OdinModel#277"(3, [:S, :I, :R], [:beta, :gamma, :I0, :N], true, false, false, true, false, Dict{Symbol, Array}()))

## Generate synthetic data

We simulate from known “true” parameters and add Poisson-like noise:

``` julia
true_pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
times = collect(1.0:1.0:50.0)
result = simulate(sir, true_pars; times=times, dt=0.25)

# Extract the cases output variable
sys = System(sir, true_pars; n_particles=1, dt=0.25)
cases_idx = findfirst(==(:cases), vcat(sys.state_names, sys.output_names))

Random.seed!(42)
data = [(time=times[i], cases=max(0.0, result[cases_idx, 1, i] + randn() * sqrt(abs(result[cases_idx, 1, i]) + 1))) for i in 1:length(times)]

println("Generated $(length(data)) data points")
println("Peak cases at time $(data[argmax([d.cases for d in data])].time)")
```

    Generated 50 data points
    Peak cases at time 11.0

## Set up parameter packer and prior

``` julia
packer = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))

prior = @prior begin
    beta ~ Exponential(1.0)
    gamma ~ Exponential(1.0)
end
```

    MontyModel{var"#10#11", var"#12#13"{var"#10#11"}, var"#14#15", Matrix{Float64}}(["beta", "gamma"], var"#10#11"(), var"#12#13"{var"#10#11"}(var"#10#11"()), var"#14#15"(), [0.0 Inf; 0.0 Inf], Odin.MontyModelProperties(true, true, false, false))

## 1. Prior predictive check

Before fitting, check whether the priors produce reasonable
trajectories:

``` julia
prior_pp = prior_predictive(prior, sir;
    times=times, packer=packer,
    n_draws=100, dt=0.25, seed=42)

println("Prior predictive draws: ", size(prior_pp.draws))
println("Variable names: ", prior_pp.variable_names)

# Summary statistics for the cases variable
vi = findfirst(==(:cases), prior_pp.variable_names)
println("\nPrior predictive cases at t=25:")
println("  Mean:   ", round(prior_pp.summary.mean[vi, 25], digits=1))
println("  Median: ", round(prior_pp.summary.median[vi, 25], digits=1))
println("  95% CI: [", round(prior_pp.summary.q025[vi, 25], digits=1), ", ",
        round(prior_pp.summary.q975[vi, 25], digits=1), "]")
```

    Prior predictive draws: (4, 50, 100)
    Variable names: [:S, :I, :R, :cases]

    Prior predictive cases at t=25:
      Mean:   1.7
      Median: 0.0
      95% CI: [-0.0, 17.3]

The prior predictive should cover a wide range of plausible
trajectories. If the observed data falls entirely outside the prior
predictive range, the priors may be too narrow or misspecified.

## 2. Fit the model (simulated posterior)

For this vignette, we simulate “posterior samples” by adding noise
around the true parameters (in practice, you would use `monty_sample`):

``` julia
n_samples = 200
n_chains = 2
n_pars = 2
pars_arr = zeros(n_pars, n_samples, n_chains)

Random.seed!(123)
for c in 1:n_chains, s in 1:n_samples
    pars_arr[1, s, c] = 0.45 + 0.1 * rand()   # beta ~ Uniform(0.45, 0.55)
    pars_arr[2, s, c] = 0.08 + 0.04 * rand()   # gamma ~ Uniform(0.08, 0.12)
end
density = zeros(n_samples, n_chains)
samples = Samples(pars_arr, density, pars_arr[:, 1, :],
                       ["beta", "gamma"], Dict{Symbol, Any}())

println("Posterior samples: $(n_samples) × $(n_chains) chains")
println("Beta range: [$(round(minimum(pars_arr[1,:,:]), digits=3)), $(round(maximum(pars_arr[1,:,:]), digits=3))]")
println("Gamma range: [$(round(minimum(pars_arr[2,:,:]), digits=3)), $(round(maximum(pars_arr[2,:,:]), digits=3))]")
```

    Posterior samples: 200 × 2 chains
    Beta range: [0.45, 0.549]
    Gamma range: [0.08, 0.12]

## 3. Posterior predictive check

``` julia
pp = posterior_predict(samples, sir;
    times=times, n_draws=100, dt=0.25,
    output_vars=[:cases],
    packer=packer, seed=42)

ppc = ppc_check(pp, data; pred_var=:cases, data_var=:cases)

println("Posterior Predictive Check Results:")
println("  Coverage (50% CI): ", round(ppc.coverage_50 * 100, digits=1), "%")
println("  Coverage (90% CI): ", round(ppc.coverage_90 * 100, digits=1), "%")
println("  Coverage (95% CI): ", round(ppc.coverage_95 * 100, digits=1), "%")
println("  Chi-squared:       ", round(ppc.chi_squared, digits=2))
println("  Mean p-value:      ", round(mean(ppc.p_values), digits=3))
```

    Posterior Predictive Check Results:
      Coverage (50% CI): 36.0%
      Coverage (90% CI): 50.0%
      Coverage (95% CI): 60.0%
      Chi-squared:       729.32
      Mean p-value:      0.524

For a well-specified model: - 50% CI coverage should be near 50% - 90%
CI coverage should be near 90% - 95% CI coverage should be near 95% -
Bayesian p-values should be roughly uniform

## 4. Residual diagnostics

``` julia
rd = residual_diagnostics(pp, data; pred_var=:cases, data_var=:cases)

println("Residual Diagnostics:")
println("  RMSE:              ", round(rd.rmse, digits=2))
println("  MAE:               ", round(rd.mae, digits=2))
println("  Bias:              ", round(rd.bias, digits=3))
println("  Ljung-Box p-value: ", round(rd.ljung_box_p, digits=3))
println()
println("Autocorrelation (lags 1-5):")
for lag in 1:min(5, length(rd.autocorrelation))
    println("  Lag $lag: ", round(rd.autocorrelation[lag], digits=3))
end
println()
println("Standardized residuals:")
println("  Mean: ", round(mean(rd.standardized_residuals), digits=3))
println("  Std:  ", round(std(rd.standardized_residuals), digits=3))
```

    Residual Diagnostics:
      RMSE:              3.66
      MAE:               2.37
      Bias:              0.21
      Ljung-Box p-value: 0.476

    Autocorrelation (lags 1-5):
      Lag 1: 0.048
      Lag 2: 0.034
      Lag 3: -0.075
      Lag 4: 0.35
      Lag 5: 0.115

    Standardized residuals:
      Mean: 0.799
      Std:  3.773

Good diagnostics show: - Bias near zero (no systematic error) -
Standardized residuals with mean ≈ 0 and std ≈ 1 - Low autocorrelation
(Ljung-Box p \> 0.05)

## 5. Calibration assessment

``` julia
cal = calibration_check(pp, data; pred_var=:cases, data_var=:cases)

println("Calibration Check:")
println("  Calibration error: ", round(cal.calibration_error, digits=3))
println("  Well-calibrated:   ", cal.is_well_calibrated)
println()
println("  Nominal → Empirical:")
for i in 1:length(cal.nominal_levels)
    nom = round(cal.nominal_levels[i], digits=1)
    emp = round(cal.empirical_levels[i], digits=2)
    diff = round(abs(cal.nominal_levels[i] - cal.empirical_levels[i]), digits=2)
    println("    $nom → $emp  (Δ = $diff)")
end
```

    Calibration Check:
      Calibration error: 0.076
      Well-calibrated:   true

      Nominal → Empirical:
        0.1 → 0.28  (Δ = 0.18)
        0.2 → 0.32  (Δ = 0.12)
        0.3 → 0.4  (Δ = 0.1)
        0.4 → 0.42  (Δ = 0.02)
        0.5 → 0.52  (Δ = 0.02)
        0.6 → 0.64  (Δ = 0.04)
        0.7 → 0.66  (Δ = 0.04)
        0.8 → 0.76  (Δ = 0.04)
        0.9 → 0.78  (Δ = 0.12)

A well-calibrated model has empirical levels close to nominal levels.
The calibration error (mean absolute difference) should be \< 0.1.

## 6. Misspecified model demonstration

Now let’s see what happens when we use wrong parameters (simulating a
misspecified model — e.g., fitting an SIR to data that was actually
generated with different dynamics):

``` julia
# "Posterior" concentrated on wrong parameters
wrong_pars_arr = zeros(n_pars, n_samples, n_chains)
for c in 1:n_chains, s in 1:n_samples
    wrong_pars_arr[1, s, c] = 0.15 + 0.02 * rand()   # beta too low
    wrong_pars_arr[2, s, c] = 0.3 + 0.05 * rand()     # gamma too high
end
wrong_density = zeros(n_samples, n_chains)
wrong_samples = Samples(wrong_pars_arr, wrong_density, wrong_pars_arr[:, 1, :],
                              ["beta", "gamma"], Dict{Symbol, Any}())

wrong_pp = posterior_predict(wrong_samples, sir;
    times=times, n_draws=100, dt=0.25,
    output_vars=[:cases],
    packer=packer, seed=42)
```

    PosteriorPredictive([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0], [1.3677941762857149 1.1691978973999473 … 0.000733241099027645 0.0006267791714972918;;; 1.4173345460007387 1.211756724760003 … 0.0007645910136475534 0.000653657340033169;;; 1.2511350307461742 1.0294614631742611 … 0.00010885278647253034 8.95892932344083e-5;;; … ;;; 1.4447938512573413 1.2405361799510222 … 0.00095486952952362 0.000819764130575228;;; 1.2421788371783504 1.0215135339875918 … 0.00010520525873456084 8.6538835003412e-5;;; 1.2396718440767625 1.0334567032861568 … 0.0002016170119116481 0.0001681153044713415], [:cases], (mean = [1.3398083672351684 1.1361637568818543 … 0.0006128935623760404 0.0005243088081787078], median = [1.3418841553108867 1.1392754103083262 … 0.0005473668545057421 0.00046514531052479385], q025 = [1.2350025893212844 1.0149008096133203 … 0.00010457797132112172 8.602285348117827e-5], q975 = [1.444816689542819 1.2443382188371013 … 0.001331212744398738 0.001152121544207473]))

### Bad PPC results

``` julia
wrong_ppc = ppc_check(wrong_pp, data; pred_var=:cases, data_var=:cases)

println("Misspecified Model — PPC:")
println("  Coverage (50% CI): ", round(wrong_ppc.coverage_50 * 100, digits=1), "% (expect ~50%)")
println("  Coverage (90% CI): ", round(wrong_ppc.coverage_90 * 100, digits=1), "% (expect ~90%)")
println("  Coverage (95% CI): ", round(wrong_ppc.coverage_95 * 100, digits=1), "% (expect ~95%)")
println("  Chi-squared:       ", round(wrong_ppc.chi_squared, digits=1), " (expect low)")
```

    Misspecified Model — PPC:
      Coverage (50% CI): 0.0% (expect ~50%)
      Coverage (90% CI): 0.0% (expect ~90%)
      Coverage (95% CI): 0.0% (expect ~95%)
      Chi-squared:       5.33531166e7 (expect low)

### Bad residual diagnostics

``` julia
wrong_rd = residual_diagnostics(wrong_pp, data; pred_var=:cases, data_var=:cases)

println("Misspecified Model — Residuals:")
println("  RMSE:  ", round(wrong_rd.rmse, digits=2), " (compare correct: $(round(rd.rmse, digits=2)))")
println("  Bias:  ", round(wrong_rd.bias, digits=2), " (compare correct: $(round(rd.bias, digits=3)))")
println("  Ljung-Box p: ", round(wrong_rd.ljung_box_p, digits=3))
```

    Misspecified Model — Residuals:
      RMSE:  34.58 (compare correct: 3.66)
      Bias:  19.59 (compare correct: 0.21)
      Ljung-Box p: 0.0

### Bad calibration

``` julia
wrong_cal = calibration_check(wrong_pp, data; pred_var=:cases, data_var=:cases)

println("Misspecified Model — Calibration:")
println("  Calibration error: ", round(wrong_cal.calibration_error, digits=3),
        " (compare correct: $(round(cal.calibration_error, digits=3)))")
println("  Well-calibrated:   ", wrong_cal.is_well_calibrated)
```

    Misspecified Model — Calibration:
      Calibration error: 0.369 (compare correct: 0.076)
      Well-calibrated:   false

## 7. Prior vs posterior width comparison

A key sanity check: the posterior predictive should be narrower than the
prior predictive (the data has constrained our uncertainty):

``` julia
vi = findfirst(==(:cases), prior_pp.variable_names)
vi_post = findfirst(==(:cases), pp.variable_names)

prior_width = prior_pp.summary.q975[vi, :] .- prior_pp.summary.q025[vi, :]
post_width = pp.summary.q975[vi_post, :] .- pp.summary.q025[vi_post, :]

println("95% CI width comparison at selected times:")
for t_idx in [5, 15, 25, 35, 45]
    t_idx > length(times) && continue
    println("  t=$(times[t_idx]): prior=$(round(prior_width[t_idx], digits=1)), ",
            "posterior=$(round(post_width[t_idx], digits=1))")
end

n_narrower = count(post_width .< prior_width)
println("\nPosterior narrower at $(n_narrower)/$(length(times)) time points")
```

    95% CI width comparison at selected times:
      t=5.0: prior=207.7, posterior=20.2
      t=15.0: prior=40.3, posterior=26.5
      t=25.0: prior=17.3, posterior=6.6
      t=35.0: prior=7.4, posterior=1.2
      t=45.0: prior=2.6, posterior=0.3

    Posterior narrower at 50/50 time points

## Summary

| Diagnostic        | Good Model | Bad Model  |
|-------------------|------------|------------|
| 95% Coverage      | ~95%       | Much lower |
| RMSE              | Low        | High       |
| Bias              | ~0         | Large      |
| Autocorrelation   | Low        | High       |
| Calibration error | \<0.1      | \>0.1      |

## R companion: bayesplot

For users who prefer R, the `bayesplot` package provides excellent PPC
visualisation. The `PosteriorPredictive` draws can be exported to R:

``` r
# R code (not executed)
library(bayesplot)

# Assuming `pp_draws` is a matrix (n_draws × n_times) and `y_obs` is observed data:
ppc_dens_overlay(y = y_obs, yrep = pp_draws[1:50, ])
ppc_intervals(y = y_obs, yrep = pp_draws, x = times)
ppc_stat(y = y_obs, yrep = pp_draws, stat = "mean")
ppc_ecdf_overlay(y = y_obs, yrep = pp_draws[1:20, ])

# Calibration plot
library(ggplot2)
cal_df <- data.frame(nominal = cal$nominal, empirical = cal$empirical)
ggplot(cal_df, aes(nominal, empirical)) +
  geom_point() + geom_abline(linetype = "dashed") +
  labs(x = "Nominal coverage", y = "Empirical coverage")
```
