# Model Selection and Comparison (R)


## Introduction

When fitting multiple candidate models to the same data, we need
principled tools to select the best model. This vignette demonstrates
model selection in R using information criteria (AIC, BIC, WAIC) and
leave-one-out cross-validation (LOO-CV), comparing an SIR and SEIR model
fitted to the same outbreak data.

### When to use each criterion

| Criterion    | Best for                          | R function      |
|--------------|-----------------------------------|-----------------|
| **AIC/AICc** | Prediction, small samples         | `AIC()`, manual |
| **BIC**      | Identifying true model            | `BIC()`, manual |
| **WAIC**     | Bayesian models (fully pointwise) | `loo::waic()`   |
| **LOO-CV**   | Robust Bayesian comparison        | `loo::loo()`    |

## Define Competing Models

We define SIR and SEIR models as ODE systems using `deSolve`:

### SIR Model

``` r
library(deSolve)
set.seed(42)

sir_model <- function(t, state, pars) {
  with(as.list(c(state, pars)), {
    dS <- -beta * S * I / N
    dI <- beta * S * I / N - gamma * I
    dR <- gamma * I
    list(c(dS, dI, dR))
  })
}
```

### SEIR Model

``` r
seir_model <- function(t, state, pars) {
  with(as.list(c(state, pars)), {
    dS <- -beta * S * I / N
    dE <- beta * S * I / N - sigma * E
    dI <- sigma * E - gamma * I
    dR <- gamma * I
    list(c(dS, dE, dI, dR))
  })
}
```

## Generate Synthetic Data

We generate data from the SIR model so we know the “true” model:

``` r
N_pop <- 1000
I0 <- 10
true_pars <- c(beta = 0.5, gamma = 0.1, N = N_pop)
state0_sir <- c(S = N_pop - I0, I = I0, R = 0)
times <- 1:40

sim <- ode(y = state0_sir, times = c(0, times), func = sir_model, parms = true_pars)
observed <- pmax(1, round(sim[-1, "I"] + rnorm(length(times)) * sqrt(abs(sim[-1, "I"]) + 1)))

cat("Generated", length(observed), "observations\n")
```

    Generated 40 observations

``` r
cat("Peak cases:", max(observed), "\n")
```

    Peak cases: 492 

## Compute Log-Likelihoods

We compute log-likelihoods under a Poisson observation model:
$\text{cases}_t \sim \text{Poisson}(I_t + \epsilon)$.

``` r
ll_poisson <- function(predicted, observed) {
  sum(dpois(round(observed), lambda = pmax(predicted, 1e-6), log = TRUE))
}

# SIR log-likelihood
sim_sir <- ode(y = state0_sir, times = c(0, times), func = sir_model, parms = true_pars)
pred_sir <- sim_sir[-1, "I"]
ll_sir <- ll_poisson(pred_sir, observed)

# SEIR log-likelihood
pars_seir <- c(beta = 0.8, gamma = 0.1, sigma = 0.2, N = N_pop)
state0_seir <- c(S = N_pop - I0, E = 0, I = I0, R = 0)
sim_seir <- ode(y = state0_seir, times = c(0, times), func = seir_model, parms = pars_seir)
pred_seir <- sim_seir[-1, "I"]
ll_seir <- ll_poisson(pred_seir, observed)

cat("SIR log-likelihood: ", ll_sir, "\n")
```

    SIR log-likelihood:  -169.4903 

``` r
cat("SEIR log-likelihood:", ll_seir, "\n")
```

    SEIR log-likelihood: -3253.15 

## AIC and BIC

AIC and BIC use the maximum log-likelihood and the number of free
parameters:

``` r
k_sir <- 3   # beta, gamma, I0
k_seir <- 4  # beta, gamma, sigma, I0
n_obs <- length(times)

aic_sir  <- -2 * ll_sir  + 2 * k_sir
aic_seir <- -2 * ll_seir + 2 * k_seir

bic_sir  <- -2 * ll_sir  + k_sir  * log(n_obs)
bic_seir <- -2 * ll_seir + k_seir * log(n_obs)

cat(sprintf("AIC — SIR: %.1f,  SEIR: %.1f\n", aic_sir, aic_seir))
```

    AIC — SIR: 345.0,  SEIR: 6514.3

``` r
cat(sprintf("BIC — SIR: %.1f,  SEIR: %.1f\n", bic_sir, bic_seir))
```

    BIC — SIR: 350.0,  SEIR: 6521.1

For small samples, use the corrected AICc:

``` r
aicc <- function(ll, k, n) -2 * ll + 2 * k + (2 * k * (k + 1)) / (n - k - 1)

aicc_sir  <- aicc(ll_sir, k_sir, n_obs)
aicc_seir <- aicc(ll_seir, k_seir, n_obs)
cat(sprintf("AICc — SIR: %.1f,  SEIR: %.1f\n", aicc_sir, aicc_seir))
```

    AICc — SIR: 345.6,  SEIR: 6515.4

## Akaike Weights

Akaike weights convert AIC differences into model probabilities:

``` r
akaike_weights <- function(aic_values) {
  delta <- aic_values - min(aic_values)
  w <- exp(-0.5 * delta)
  w / sum(w)
}

w <- akaike_weights(c(aic_sir, aic_seir))
cat(sprintf("Akaike weights — SIR: %.3f,  SEIR: %.3f\n", w[1], w[2]))
```

    Akaike weights — SIR: 1.000,  SEIR: 0.000

## WAIC via Pointwise Log-Likelihoods

WAIC requires per-observation log-likelihoods across posterior samples.
We use the `loo` package:

``` r
library(loo)
```

    This is loo version 2.8.0

    - Online documentation and vignettes at mc-stan.org/loo

    - As of v2.0.0 loo defaults to 1 core but we recommend using as many as possible. Use the 'cores' argument or set options(mc.cores = NUM_CORES) for an entire session. 

``` r
n_posterior <- 200

# Simulate a simple posterior by perturbing around MLEs
pw_sir <- matrix(NA, nrow = n_posterior, ncol = n_obs)
for (s in seq_len(n_posterior)) {
  p <- c(beta = 0.5 + rnorm(1, 0, 0.02),
         gamma = 0.1 + rnorm(1, 0, 0.005),
         N = N_pop)
  st <- c(S = N_pop - (I0 + rnorm(1)), I = I0 + rnorm(1), R = 0)
  st["S"] <- N_pop - st["I"]
  sim_s <- tryCatch(
    ode(y = st, times = c(0, times), func = sir_model, parms = p),
    error = function(e) NULL
  )
  if (is.null(sim_s)) {
    pw_sir[s, ] <- rep(-1e6, n_obs)
  } else {
    pred <- pmax(sim_s[-1, "I"], 1e-6)
    pw_sir[s, ] <- dpois(round(observed), lambda = pred, log = TRUE)
  }
}

pw_seir <- matrix(NA, nrow = n_posterior, ncol = n_obs)
for (s in seq_len(n_posterior)) {
  p <- c(beta = 0.8 + rnorm(1, 0, 0.02),
         gamma = 0.1 + rnorm(1, 0, 0.005),
         sigma = 0.2 + rnorm(1, 0, 0.01),
         N = N_pop)
  st <- c(S = N_pop - (I0 + rnorm(1)), E = 0, I = I0 + rnorm(1), R = 0)
  st["S"] <- N_pop - st["E"] - st["I"]
  sim_s <- tryCatch(
    ode(y = st, times = c(0, times), func = seir_model, parms = p),
    error = function(e) NULL
  )
  if (is.null(sim_s)) {
    pw_seir[s, ] <- rep(-1e6, n_obs)
  } else {
    pred <- pmax(sim_s[-1, "I"], 1e-6)
    pw_seir[s, ] <- dpois(round(observed), lambda = pred, log = TRUE)
  }
}

# loo::waic() expects log-lik matrix with rows=draws, cols=observations
waic_sir  <- suppressWarnings(waic(pw_sir))
waic_seir <- suppressWarnings(waic(pw_seir))

cat(sprintf("WAIC — SIR: %.1f,  SEIR: %.1f\n",
            waic_sir$estimates["waic", "Estimate"],
            waic_seir$estimates["waic", "Estimate"]))
```

    WAIC — SIR: 703.7,  SEIR: 32133.8

``` r
cat(sprintf("p_WAIC — SIR: %.2f,  SEIR: %.2f\n",
            waic_sir$estimates["p_waic", "Estimate"],
            waic_seir$estimates["p_waic", "Estimate"]))
```

    p_WAIC — SIR: 180.83,  SEIR: 13864.13

## LOO-CV

Pareto-smoothed importance sampling LOO provides a robust alternative:

``` r
loo_sir  <- suppressWarnings(loo(pw_sir))
loo_seir <- suppressWarnings(loo(pw_seir))

cat(sprintf("LOO — SIR: %.1f,  SEIR: %.1f\n",
            loo_sir$estimates["looic", "Estimate"],
            loo_seir$estimates["looic", "Estimate"]))
```

    LOO — SIR: 659.7,  SEIR: 8882.9

``` r
# Pareto k diagnostics
cat(sprintf("Max Pareto k — SIR: %.2f,  SEIR: %.2f\n",
            max(loo_sir$diagnostics$pareto_k),
            max(loo_seir$diagnostics$pareto_k)))
```

    Max Pareto k — SIR: 4.80,  SEIR: 14.62

A Pareto k \> 0.7 indicates unreliable LOO estimates for that
observation.

## LOO Model Comparison

The `loo` package provides a convenient `loo_compare()` function:

``` r
comp <- loo_compare(list(SIR = loo_sir, SEIR = loo_seir))
print(comp)
```

         elpd_diff se_diff
    SIR      0.0       0.0
    SEIR -4111.6     698.7

## Model Comparison Table

``` r
results <- data.frame(
  Model = c("SIR", "SEIR"),
  k = c(k_sir, k_seir),
  LogLik = round(c(ll_sir, ll_seir), 1),
  AIC = round(c(aic_sir, aic_seir), 1),
  AICc = round(c(aicc_sir, aicc_seir), 1),
  BIC = round(c(bic_sir, bic_seir), 1),
  WAIC = round(c(waic_sir$estimates["waic", "Estimate"],
                  waic_seir$estimates["waic", "Estimate"]), 1),
  LOO = round(c(loo_sir$estimates["looic", "Estimate"],
                 loo_seir$estimates["looic", "Estimate"]), 1),
  Weight = round(w, 3)
)
print(results)
```

      Model k  LogLik    AIC   AICc    BIC    WAIC    LOO Weight
    1   SIR 3  -169.5  345.0  345.6  350.0   703.7  659.7      1
    2  SEIR 4 -3253.1 6514.3 6515.4 6521.1 32133.8 8882.9      0

## Guidance on Criterion Choice

- **AIC/AICc**: Best for predictive accuracy. Use AICc when n/k \< 40.
- **BIC**: Consistent — selects the true model as n → ∞. Penalises
  complexity more heavily than AIC.
- **WAIC**: Fully Bayesian, uses the entire posterior via `loo::waic()`.
  More stable than DIC.
- **LOO-CV (PSIS-LOO)**: Gold standard for Bayesian model comparison via
  `loo::loo()`. Pareto k diagnostics flag unreliable estimates.

In practice, start with AIC/BIC for quick comparison, then use WAIC or
LOO-CV for rigorous Bayesian analysis. The `loo` package (Vehtari,
Gabry, Magnusson, Yao, Bürkner, Paananen, Gelman, 2024) provides the
most comprehensive R implementation.

## Summary

| Criterion  | R Package/Function          | Julia/Odin         |
|------------|-----------------------------|--------------------|
| AIC        | `-2*ll + 2*k` (manual)      | `compute_aic()`    |
| AICc       | Manual formula              | `compute_aicc()`   |
| BIC        | `-2*ll + k*log(n)` (manual) | `compute_bic()`    |
| WAIC       | `loo::waic()`               | `compute_waic()`   |
| LOO-CV     | `loo::loo()`                | `compute_loo()`    |
| Comparison | `loo::loo_compare()`        | `compare_models()` |
