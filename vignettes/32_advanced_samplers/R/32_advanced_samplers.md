# Advanced MCMC Samplers (R)


## Overview

This vignette mirrors the Julia advanced samplers vignette (vignette 32)
using R packages. We compare several MCMC samplers on the same target
distributions:

| R Package | Sampler                     | Gradient? |
|-----------|-----------------------------|-----------|
| `monty`   | Random walk                 | No        |
| `monty`   | HMC                         | Yes       |
| `mcmc`    | Metropolis-within-Gibbs     | No        |
| `mcmc`    | Slice (via `metrop` tuning) | No        |

R’s `monty` package (from the odin2/dust2 ecosystem) provides random
walk and HMC samplers. The `mcmc` package provides a flexible Metropolis
sampler that can be configured for component-wise updates. For slice
sampling, we implement a simple univariate slice sampler following Neal
(2003).

``` r
library(monty)
library(mcmc)
library(coda)
library(MASS)
```

## Target distributions

### Banana-shaped distribution

A twisted Gaussian (“banana”) that challenges samplers with strong
nonlinear correlation:

``` r
banana_density <- function(x, b = 0.1) {
  -0.5 * (x[1]^2 + (x[2] - b * x[1]^2 + 100 * b)^2)
}

banana_gradient <- function(x, b = 0.1) {
  dx1 <- -x[1] - 2 * b * x[1] * (x[2] - b * x[1]^2 + 100 * b)
  dx2 <- -(x[2] - b * x[1]^2 + 100 * b)
  c(dx1, dx2)
}
```

### Correlated Gaussian

A strongly correlated 2D Gaussian to test ESS:

``` r
target_mean <- c(0, 0)
target_cov <- matrix(c(1, 0.95, 0.95, 1), 2, 2)
target_prec <- solve(target_cov)

gauss_density <- function(x) {
  d <- x - target_mean
  -0.5 * sum(d * (target_prec %*% d))
}

gauss_gradient <- function(x) {
  -target_prec %*% (x - target_mean)
}
```

## monty: Random walk sampler

The `monty` package provides a random walk Metropolis sampler with a
user-specified proposal covariance:

``` r
banana_model <- monty_model_function(
  function(x, y) banana_density(c(x, y)),
  packer = monty_packer(c("x", "y"))
)

sampler_rw <- monty_sampler_random_walk(vcv = diag(2) * 0.5)
samples_rw <- monty_sample(banana_model, sampler_rw, 5000, n_chains = 4,
                           initial = matrix(0, 2, 4))
```

    ⡀⠀ Sampling [▁▁▁▁] ■                                |   0% ETA:  2m

    ✔ Sampled 20000 steps across 4 chains in 512ms

``` r
cat("Random walk on banana target:\n")
```

    Random walk on banana target:

``` r
cat("  Mean x:", round(mean(samples_rw$pars["x", , ]), 2), "\n")
```

      Mean x: -0.02 

``` r
cat("  Mean y:", round(mean(samples_rw$pars["y", , ]), 2), "\n")
```

      Mean y: -9.83 

## monty: HMC sampler

HMC uses gradient information for more efficient exploration. It
requires a model with a gradient method:

``` r
banana_model_grad <- monty_model(
  list(
    parameters = c("x", "y"),
    density = function(x) banana_density(x),
    gradient = function(x) banana_gradient(x),
    domain = rbind(c(-Inf, Inf), c(-Inf, Inf))
  ),
  monty_model_properties(allow_multiple_parameters = FALSE,
                         has_gradient = TRUE)
)

sampler_hmc <- monty_sampler_hmc(epsilon = 0.05, n_integration_steps = 20)
samples_hmc <- monty_sample(banana_model_grad, sampler_hmc, 5000,
                            n_chains = 4,
                            initial = matrix(0, 2, 4))
```

    ⡀⠀ Sampling [██▃▁] ■■■■■■■■■■■■■■■■■■■              |  60% ETA:  1s

    ✔ Sampled 20000 steps across 4 chains in 3.2s

``` r
cat("HMC on banana target:\n")
```

    HMC on banana target:

``` r
cat("  Mean x:", round(mean(samples_hmc$pars["x", , ]), 2), "\n")
```

      Mean x: 0.01 

``` r
cat("  Mean y:", round(mean(samples_hmc$pars["y", , ]), 2), "\n")
```

      Mean y: -9.9 

## Slice sampler (Neal 2003)

R does not have a widely-used standalone slice sampler package, so we
implement a simple univariate slice sampler following Neal’s
stepping-out/shrinking procedure:

``` r
slice_sample_1d <- function(log_f, x0, w = 2.0, max_steps = 20) {
  y <- log_f(x0) - rexp(1)
  L <- x0 - runif(1) * w
  R <- L + w
  j <- floor(max_steps * runif(1))
  k <- max_steps - 1 - j
  while (j > 0 && log_f(L) > y) { L <- L - w; j <- j - 1 }
  while (k > 0 && log_f(R) > y) { R <- R + w; k <- k - 1 }
  repeat {
    x1 <- runif(1, L, R)
    if (log_f(x1) >= y) return(x1)
    if (x1 < x0) L <- x1 else R <- x1
  }
}

slice_sampler <- function(log_f, x0, n_iter, w = 2.0) {
  d <- length(x0)
  samples <- matrix(NA, n_iter, d)
  x <- x0
  for (i in seq_len(n_iter)) {
    for (j in seq_len(d)) {
      x[j] <- slice_sample_1d(
        function(v) { xp <- x; xp[j] <- v; log_f(xp) },
        x[j], w = w
      )
    }
    samples[i, ] <- x
  }
  samples
}
```

Run the slice sampler on the banana target:

``` r
set.seed(42)
samples_slice_raw <- slice_sampler(banana_density, c(0, 0), 5000, w = 2.0)

cat("Slice sampler on banana target:\n")
```

    Slice sampler on banana target:

``` r
cat("  Mean x:", round(mean(samples_slice_raw[-(1:1000), 1]), 2), "\n")
```

      Mean x: -0.04 

``` r
cat("  Mean y:", round(mean(samples_slice_raw[-(1:1000), 2]), 2), "\n")
```

      Mean y: -9.92 

## mcmc: Metropolis sampler (component-wise)

The `mcmc` package provides a general Metropolis sampler. With a
diagonal proposal, it mimics Gibbs-style component-wise updates:

``` r
set.seed(42)
mcmc_out <- metrop(banana_density, initial = c(0, 0), nbatch = 5000,
                   scale = c(0.5, 0.5))
cat("mcmc::metrop acceptance rate:", round(mcmc_out$accept, 3), "\n")
```

    mcmc::metrop acceptance rate: 0.757 

``` r
# Discard burn-in
mcmc_samples <- mcmc_out$batch[-(1:1000), ]
cat("mcmc::metrop on banana target:\n")
```

    mcmc::metrop on banana target:

``` r
cat("  Mean x:", round(mean(mcmc_samples[, 1]), 2), "\n")
```

      Mean x: 0 

``` r
cat("  Mean y:", round(mean(mcmc_samples[, 2]), 2), "\n")
```

      Mean y: -9.87 

## ESS comparison on correlated Gaussian

We compare effective sample sizes across samplers on the correlated
Gaussian target:

``` r
compute_ess <- function(chain) {
  if (is.matrix(chain)) {
    return(apply(chain, 2, compute_ess))
  }
  n <- length(chain)
  if (n < 10) return(n)
  v <- var(chain)
  if (v < 1e-12) return(n)
  m <- mean(chain)
  max_lag <- min(n - 1, 100)
  rho_sum <- 0
  for (k in seq_len(max_lag)) {
    acf_k <- sum((chain[1:(n - k)] - m) * (chain[(k + 1):n] - m)) /
             ((n - k) * v)
    if (acf_k < 0.05) break
    rho_sum <- rho_sum + acf_k
  }
  n / (1 + 2 * rho_sum)
}
```

``` r
n_steps <- 5000
n_burnin <- 1000

gauss_model_rw <- monty_model_function(
  function(x, y) gauss_density(c(x, y)),
  packer = monty_packer(c("x", "y"))
)

gauss_model_hmc <- monty_model(
  list(
    parameters = c("x", "y"),
    density = function(x) gauss_density(x),
    gradient = function(x) as.numeric(gauss_gradient(x)),
    domain = rbind(c(-Inf, Inf), c(-Inf, Inf))
  ),
  monty_model_properties(allow_multiple_parameters = FALSE,
                         has_gradient = TRUE)
)

results <- list()

# Random walk (monty)
t_rw <- system.time({
  s_rw <- monty_sample(gauss_model_rw,
                       monty_sampler_random_walk(vcv = diag(2) * 0.5),
                       n_steps, n_chains = 1,
                       initial = matrix(0, 2, 1))
})["elapsed"]
chain_rw <- t(s_rw$pars[, -(1:n_burnin), 1])
ess_rw <- compute_ess(chain_rw)
results$RW <- c(ess = min(ess_rw), time = t_rw)

# HMC (monty)
t_hmc <- system.time({
  s_hmc <- monty_sample(gauss_model_hmc,
                        monty_sampler_hmc(epsilon = 0.2,
                                          n_integration_steps = 10),
                        n_steps, n_chains = 1,
                        initial = matrix(0, 2, 1))
})["elapsed"]
chain_hmc <- t(s_hmc$pars[, -(1:n_burnin), 1])
ess_hmc <- compute_ess(chain_hmc)
results$HMC <- c(ess = min(ess_hmc), time = t_hmc)

# Slice sampler
set.seed(42)
t_slice <- system.time({
  chain_slice_raw <- slice_sampler(gauss_density, c(0, 0), n_steps, w = 1.5)
})["elapsed"]
chain_slice <- chain_slice_raw[-(1:n_burnin), ]
ess_slice <- compute_ess(chain_slice)
results$Slice <- c(ess = min(ess_slice), time = t_slice)

# mcmc::metrop
set.seed(42)
t_metrop <- system.time({
  m_out <- metrop(gauss_density, initial = c(0, 0), nbatch = n_steps,
                  scale = c(0.3, 0.3))
})["elapsed"]
chain_metrop <- m_out$batch[-(1:n_burnin), ]
ess_metrop <- compute_ess(chain_metrop)
results$Metrop <- c(ess = min(ess_metrop), time = t_metrop)

cat(sprintf("%-10s %8s %8s %10s\n", "Sampler", "min ESS", "Time(s)",
            "ESS/s"))
```

    Sampler     min ESS  Time(s)      ESS/s

``` r
cat(strrep("-", 40), "\n")
```

    ---------------------------------------- 

``` r
for (nm in names(results)) {
  r <- results[[nm]]
  cat(sprintf("%-10s %8.0f %8.3f %10.0f\n", nm, r["ess"], r["time"],
              r["ess"] / r["time"]))
}
```

    RW               79       NA         NA
    HMC            2741       NA         NA
    Slice           249       NA         NA
    Metrop           36       NA         NA

## Notes on sampler availability

The Julia Odin.jl package provides MALA (Metropolis-adjusted Langevin
algorithm) and Gibbs samplers with mixed sub-samplers. In R:

- **MALA** is not directly available in `monty` or `mcmc`. The
  `adaptMCMC` package provides MALA-like adaptive samplers. HMC with one
  leapfrog step is similar to MALA.
- **Gibbs** with mixed sub-samplers can be approximated by running
  component-wise updates with `mcmc::metrop` or custom code.
- **Slice sampling** is available in JAGS and Stan (NUTS uses a form of
  slice sampling for step-size selection), but no standalone R
  implementation is on CRAN — hence our implementation above.

## Summary

| Sampler     | R package     | Gradient? | Notes                            |
|-------------|---------------|-----------|----------------------------------|
| Random walk | `monty`       | No        | Baseline; tune proposal variance |
| HMC         | `monty`       | Yes       | Best ESS/step for smooth targets |
| Slice       | Custom / JAGS | No        | Auto-tuning, no rejections       |
| Metropolis  | `mcmc`        | No        | Flexible; component-wise updates |
| MALA        | `adaptMCMC`   | Yes       | Not shown; HMC(L=1) is similar   |
| Gibbs       | Custom / JAGS | Optional  | Block structure via manual code  |
