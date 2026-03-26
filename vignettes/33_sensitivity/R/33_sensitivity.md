# Sensitivity Analysis for ODE Models (R)


## Overview

This vignette mirrors the Julia sensitivity analysis vignette (vignette
33) using R packages. We demonstrate four approaches to sensitivity
analysis:

| Method | R Package | Type | Best for |
|----|----|----|----|
| Forward sensitivity | `FME::sensFun` | Local | Few parameters, trajectory sensitivity |
| Finite-difference gradient | Base R | Local | Quick gradient check |
| Sobol indices | `sensitivity::sobolSalt` | Global, variance-based | Parameter importance ranking |
| Morris screening | `sensitivity::morris` | Global, screening | Cheap screening of many parameters |

``` r
library(deSolve)
library(FME)
```

    Loading required package: rootSolve

    Loading required package: coda

``` r
library(sensitivity)
```

    Registered S3 method overwritten by 'sensitivity':
      method    from 
      print.src dplyr

## SIR Model

We use the same SIR model as in the Julia vignette:

``` r
sir_ode <- function(t, state, pars) {
  with(as.list(c(state, pars)), {
    dS <- -beta * S * I / N
    dI <-  beta * S * I / N - gamma * I
    dR <-  gamma * I
    list(c(dS = dS, dI = dI, dR = dR))
  })
}

pars <- c(beta = 0.5, gamma = 0.1, N = 1000, I0 = 10)
state0 <- c(S = 990, I = 10, R = 0)
times <- seq(0, 100, by = 1)
```

### Baseline solution

``` r
sol <- ode(state0, times, sir_ode, pars)
peak_I <- max(sol[, "I"])
peak_t <- sol[which.max(sol[, "I"]), "time"]
cat(sprintf("Peak I = %.1f at t = %.0f\n", peak_I, peak_t))
```

    Peak I = 479.4 at t = 15

## Forward Sensitivity (FME)

The `FME::sensFun` function computes local sensitivities by solving the
variational equation numerically via finite differences. This is
analogous to Julia’s `dust_sensitivity_forward`:

``` r
sir_for_fme <- function(pars) {
  p <- as.list(pars)
  y0 <- c(S = p$N - p$I0, I = p$I0, R = 0)
  out <- ode(y0, times, sir_ode, pars)
  out
}

sf <- sensFun(func = sir_for_fme,
              parms = pars,
              sensvar = "I",
              varscale = 1,
              parscale = 1)
```

### Sensitivity of I to β and γ over time

``` r
sf_df <- as.data.frame(sf)
cat(sprintf("%-6s %8s %10s %10s\n", "Time", "I(t)", "dI/dbeta", "dI/dgamma"))
```

    Time       I(t)   dI/dbeta  dI/dgamma

``` r
cat(strrep("-", 38), "\n")
```

    -------------------------------------- 

``` r
for (ti in c(10, 20, 30, 40, 50)) {
  row <- sf_df[sf_df$x == ti, ]
  I_val <- sol[sol[, "time"] == ti, "I"]
  cat(sprintf("%-6d %8.1f %10.1f %10.1f\n", ti, I_val, row$beta, row$gamma))
}
```

    10        294.4     1647.5    -2229.5
    20        400.3     -452.6    -2937.7
    30        175.0     -493.6    -2323.8
    40         68.6     -234.0    -1439.3
    50         26.4      -99.9     -766.5

The sensitivity to β is positive (more transmission → more infected),
while the sensitivity to γ is negative (faster recovery → fewer
infected).

### Summary sensitivity

The `summary` method provides a condensed view of parameter importance
across the full time horizon:

``` r
summary(sf)
```

          value scale      L1      L2     Mean     Min    Max   N
    beta  5e-01     1 2.6e+02  468.34  7.1e+00  -623.5 1764.1 101
    gamma 1e-01     1 1.0e+03 1486.66 -1.0e+03 -3226.6    0.0 101
    N     1e+03     1 9.8e-02    0.18  9.8e-02     0.0    0.5 101
    I0    1e+01     1 2.4e+00    4.30  5.9e-03    -6.7   15.5 101

## Finite-Difference Gradient

For a scalar objective (e.g., peak infected count), a simple
finite-difference gradient gives the local sensitivity:

``` r
peak_infected <- function(p) {
  y0 <- c(S = unname(p["N"] - p["I0"]),
          I = unname(p["I0"]),
          R = 0)
  out <- ode(y0, times, sir_ode, p, method = "lsoda")
  max(out[, "I"])
}

fd_gradient <- function(f, p, eps = 1e-4) {
  g <- numeric(length(p))
  f0 <- f(p)
  for (i in seq_along(p)) {
    pp <- p
    pp[i] <- pp[i] + eps
    g[i] <- (f(pp) - f0) / eps
  }
  names(g) <- names(p)
  g
}

grad <- fd_gradient(peak_infected, pars)
cat("Gradient of peak I w.r.t. parameters:\n")
```

    Gradient of peak I w.r.t. parameters:

``` r
print(round(grad, 3))
```

         beta     gamma         N        I0 
      763.913 -3212.633     0.467     1.243 

## Sobol Sensitivity Indices

Sobol indices decompose output variance into contributions from each
parameter. The first-order index $S_i$ measures the direct effect; the
total-order index $S_{Ti}$ includes interactions.

We use the Saltelli sampling scheme via `sensitivity::sobolSalt`:

``` r
set.seed(42)

n_sobol <- 500
param_names <- c("beta", "gamma", "I0", "N")
param_ranges <- list(
  beta  = c(0.2, 1.0),
  gamma = c(0.05, 0.3),
  I0    = c(1, 50),
  N     = c(500, 2000)
)

# Generate quasi-random samples in [0,1] and scale to ranges
X1 <- matrix(runif(n_sobol * 4), ncol = 4)
X2 <- matrix(runif(n_sobol * 4), ncol = 4)
colnames(X1) <- colnames(X2) <- param_names
for (j in seq_along(param_names)) {
  nm <- param_names[j]
  lo <- param_ranges[[nm]][1]
  hi <- param_ranges[[nm]][2]
  X1[, j] <- lo + X1[, j] * (hi - lo)
  X2[, j] <- lo + X2[, j] * (hi - lo)
}

sobol_obj <- sobolSalt(model = NULL, X1 = X1, X2 = X2, scheme = "A",
                       nboot = 100)
```

### Model evaluation wrapper

``` r
sir_sobol_eval <- function(X) {
  apply(X, 1, function(row) {
    row <- unname(row)
    p <- c(beta = row[1], gamma = row[2], I0 = row[3], N = row[4])
    y0 <- c(S = p[["N"]] - p[["I0"]],
            I = p[["I0"]],
            R = 0)
    out <- tryCatch(
      suppressWarnings(ode(y0, times, sir_ode, p, method = "lsoda")),
      error = function(e) NULL
    )
    if (is.null(out) || any(is.na(out[, "I"]))) return(0)
    max(out[, "I"])
  })
}

y_sobol <- sir_sobol_eval(sobol_obj$X)
tell(sobol_obj, y_sobol)
```


    Call:
    sobolSalt(model = NULL, X1 = X1, X2 = X2, scheme = "A", nboot = 100)

    Model runs: 3000 

    Model variance: 97551.63 

    First order indices:
         original          bias std. error   min. c.i. max. c.i.
    X1 0.33850611  0.0021744858 0.04089512  0.25947324 0.4272613
    X2 0.39689028 -0.0005929949 0.03332237  0.33296697 0.4780974
    X3 0.02724788  0.0005981327 0.05085235 -0.08038225 0.1126793
    X4 0.27305378 -0.0035271067 0.05078327  0.14757740 0.3653888

    Total indices:
           original         bias   std. error    min. c.i.    max. c.i.
    X1 0.3343356805 8.215123e-04 2.647664e-02 0.2759529859 0.3840369344
    X2 0.3733141529 3.089781e-03 2.908004e-02 0.3194251703 0.4265457023
    X3 0.0004060329 1.026415e-05 5.188556e-05 0.0002868778 0.0004857238
    X4 0.3280325035 6.309068e-04 3.127610e-02 0.2570934035 0.3824433388

### Sobol results

``` r
cat(sprintf("%-10s %12s %12s\n", "Parameter", "First-order", "Total-order"))
```

    Parameter   First-order  Total-order

``` r
cat(strrep("-", 36), "\n")
```

    ------------------------------------ 

``` r
S <- sobol_obj$S$original
ST <- sobol_obj$T$original
for (i in seq_along(param_names)) {
  cat(sprintf("%-10s %12.3f %12.3f\n", param_names[i], S[i], ST[i]))
}
```

    beta              0.339        0.334
    gamma             0.397        0.373
    I0                0.027        0.000
    N                 0.273        0.328

Parameters with high total-order but low first-order indices have
important interactions with other parameters.

## Morris Screening

Morris screening computes elementary effects to rank parameter
importance. High $\mu^*$ means the parameter is influential; high
$\sigma$ means its effect depends on other parameters.

``` r
set.seed(42)

morris_result <- morris(
  model = sir_sobol_eval,
  factors = param_names,
  r = 30,
  design = list(type = "oat", levels = 6, grid.jump = 3),
  binf = sapply(param_ranges, `[`, 1),
  bsup = sapply(param_ranges, `[`, 2)
)
```

### Morris results

``` r
cat(sprintf("%-10s %10s %10s\n", "Parameter", "mu*", "sigma"))
```

    Parameter         mu*      sigma

``` r
cat(strrep("-", 32), "\n")
```

    -------------------------------- 

``` r
mu_star <- apply(abs(morris_result$ee), 2, mean, na.rm = TRUE)
sigma <- apply(morris_result$ee, 2, sd, na.rm = TRUE)
for (i in seq_along(param_names)) {
  cat(sprintf("%-10s %10.1f %10.1f\n", param_names[i],
              mu_star[i], sigma[i]))
}
```

    beta            538.1      249.9
    gamma           579.2      318.3
    I0               18.1       15.3
    N               527.5      359.6

### Interpretation

- **μ\* ≫ 0**: parameter has important influence on the output
- **σ/μ\* \> 1**: strong interactions or nonlinearity
- **σ/μ\* ≈ 0**: nearly linear, additive effect

## Using odin2/dust2 for sensitivity

The odin2 R package generates C++ models with adjoint support. When
combined with `dust2::dust_unfilter_create`, the log-likelihood gradient
can be computed efficiently:

``` r
sir <- odin2::odin({
  deriv(S) <- -beta * S * I / N
  deriv(I) <-  beta * S * I / N - gamma * I
  deriv(R) <-  gamma * I
  initial(S) <- N - I0
  initial(I) <- I0
  initial(R) <- 0
  beta <- parameter(0.5)
  gamma <- parameter(0.1)
  I0 <- parameter(10)
  N <- parameter(1000)
})
# adjoint sensitivity for log-likelihood gradients:
# ll <- Likelihood(sir, data = ...)
# grad <- dust_unfilter_run(ll, pars, adjoint = TRUE)
```

This is the R equivalent of Julia’s `dust_sensitivity_adjoint`.

## Summary

| Method | R function | Type | Cost |
|----|----|----|----|
| Forward sensitivity | `FME::sensFun` | Local, trajectory | O(n × k) ODE solves |
| Finite-difference gradient | Base R | Local, scalar | O(k) ODE solves |
| Sobol indices | `sensitivity::sobolSalt` | Global, variance | O(N × (k+2)) runs |
| Morris screening | `sensitivity::morris` | Global, screening | O(N × (k+1)) runs |
| Adjoint gradient | `dust2` (odin2) | Local, scalar | O(n) backward pass |

- `FME::sensFun` is the easiest route for trajectory-level local
  sensitivity
- `sensitivity::sobolSalt` and `sensitivity::morris` give global views
- For model fitting, odin2’s adjoint mode provides efficient gradients
