# Advanced MCMC Samplers
Odin.jl

## Overview

Odin.jl provides three advanced MCMC samplers beyond the standard random
walk and HMC:

| Sampler | Gradient? | Key advantage |
|----|----|----|
| **Slice** (`monty_sampler_slice`) | No | Auto-tuning, no rejections |
| **MALA** (`monty_sampler_mala`) | Yes | Gradient-guided proposals, simpler than HMC |
| **Gibbs** (`monty_sampler_gibbs`) | Optional | Block structure, mix-and-match sub-samplers |

``` julia
using Odin
using Distributions
using LinearAlgebra
using Statistics
using Random
```

## Target distributions

We compare all samplers on two challenging targets.

### Banana-shaped distribution

A twisted Gaussian (“banana”) that challenges samplers with strong
nonlinear correlation:

``` julia
function banana_density(x; b=0.1)
    return -0.5 * (x[1]^2 + (x[2] - b * x[1]^2 + 100 * b)^2)
end

function banana_gradient(x; b=0.1)
    dx1 = -x[1] - 2 * b * x[1] * (x[2] - b * x[1]^2 + 100 * b)
    dx2 = -(x[2] - b * x[1]^2 + 100 * b)
    return [dx1, dx2]
end

banana_no_grad = DensityModel(
    banana_density;
    parameters=["x", "y"],
)

banana_with_grad = DensityModel(
    banana_density;
    parameters=["x", "y"],
    gradient=banana_gradient,
)
```

    MontyModel{typeof(banana_density), typeof(banana_gradient), Nothing, Nothing}(["x", "y"], banana_density, banana_gradient, nothing, nothing, Odin.MontyModelProperties(true, false, false, false))

### Correlated Gaussian

A strongly correlated 2D Gaussian to test ESS:

``` julia
target_mean = [0.0, 0.0]
target_cov = [1.0 0.95; 0.95 1.0]
target_dist = MvNormal(target_mean, target_cov)

gauss_model = DensityModel(
    x -> logpdf(target_dist, x);
    parameters=["x", "y"],
    gradient=x -> -inv(target_cov) * (x .- target_mean),
)
```

    MontyModel{var"#5#6", var"#7#8", Nothing, Nothing}(["x", "y"], var"#5#6"(), var"#7#8"(), nothing, nothing, Odin.MontyModelProperties(true, false, false, false))

## Slice sampler

The slice sampler uses Neal’s stepping-out and shrinking procedure. It
requires no gradient and no tuning of proposal variance — only the
bracket width `w`:

``` julia
sl = slice(w=2.0, max_steps=20)
initial = zeros(Float64, 2, 4)

samples_slice = sample(banana_no_grad, sl, 5000;
    n_chains=4, initial=initial, n_burnin=1000, seed=42)

println("Slice sampler on banana target:")
println("  Mean: ", round.(mean(samples_slice.pars[:, :, :], dims=(2, 3))[:, 1, 1], digits=2))
println("  Acceptance rate: ", round.(samples_slice.details[:acceptance_rate], digits=3))
```

    Slice sampler on banana target:
      Mean: [-0.0, -9.9]
      Acceptance rate: [1.0, 1.0, 1.0, 1.0]

## MALA sampler

MALA uses gradient information for smarter proposals. It’s simpler than
HMC (one gradient evaluation per step) but still makes directed moves:

``` julia
ml = mala(0.3)
initial = zeros(Float64, 2, 4)

samples_ml = sample(banana_with_grad, ml, 5000;
    n_chains=4, initial=initial, n_burnin=1000, seed=42)

println("MALA on banana target:")
println("  Mean: ", round.(mean(samples_ml.pars[:, :, :], dims=(2, 3))[:, 1, 1], digits=2))
println("  Acceptance rate: ", round.(samples_ml.details[:acceptance_rate], digits=3))
```

    MALA on banana target:
      Mean: [0.1, -9.93]
      Acceptance rate: [0.96, 0.964, 0.97, 0.97]

### MALA with preconditioning

A mass matrix can improve MALA on poorly scaled targets:

``` julia
M = [1.0 0.0; 0.0 4.0]
ml_precond = mala(0.2; vcv=M)

samples_ml_p = sample(banana_with_grad, ml_precond, 5000;
    n_chains=4, initial=zeros(Float64, 2, 4), n_burnin=1000, seed=42)

println("Preconditioned MALA acceptance: ",
    round.(samples_ml_p.details[:acceptance_rate], digits=3))
```

    Preconditioned MALA acceptance: [0.881, 0.863, 0.875, 0.872]

## Gibbs sampler

The Gibbs sampler cycles through parameter blocks. Each block can use a
different sub-sampler — useful for hierarchical models or mixed
parameter types.

### Block-wise slice sampling

``` julia
blocks = [[1], [2]]
sub_samplers = [
    slice(w=2.0),
    slice(w=2.0),
]
gibbs_slice = gibbs(blocks, sub_samplers)

samples_gibbs = sample(banana_no_grad, gibbs_slice, 5000;
    n_chains=4, initial=zeros(Float64, 2, 4), n_burnin=1000, seed=42)

println("Gibbs (slice blocks) on banana:")
println("  Mean: ", round.(mean(samples_gibbs.pars[:, :, :], dims=(2, 3))[:, 1, 1], digits=2))
```

    Gibbs (slice blocks) on banana:
      Mean: [-0.0, -9.9]

### Mixed sub-samplers: MALA + Slice

Combine gradient-based and gradient-free samplers in one Gibbs sweep:

``` julia
mixed_blocks = [[1], [2]]
mixed_subs = Odin.AbstractMontySampler[
    mala(0.5; vcv=reshape([1.0], 1, 1)),
    slice(w=2.0),
]
gibbs_mixed = gibbs(mixed_blocks, mixed_subs)

samples_mixed = sample(banana_with_grad, gibbs_mixed, 5000;
    n_chains=4, initial=zeros(Float64, 2, 4), n_burnin=1000, seed=42)

println("Gibbs (MALA + Slice) on banana:")
println("  Mean: ", round.(mean(samples_mixed.pars[:, :, :], dims=(2, 3))[:, 1, 1], digits=2))
```

    Gibbs (MALA + Slice) on banana:
      Mean: [0.06, -9.9]

## ESS comparison

We compare effective sample sizes per second across all samplers on the
correlated Gaussian:

``` julia
n_steps = 5000
n_burnin = 1000
initial = zeros(Float64, 2, 4)

function simple_ess(chain::AbstractVector{Float64})
    n = length(chain)
    n < 10 && return Float64(n)
    m = mean(chain)
    v = var(chain)
    v < 1e-12 && return Float64(n)
    max_lag = min(n - 1, 100)
    rho_sum = 0.0
    for k in 1:max_lag
        acf = sum((chain[1:end-k] .- m) .* (chain[k+1:end] .- m)) / ((n - k) * v)
        acf < 0.05 && break
        rho_sum += acf
    end
    return n / (1.0 + 2.0 * rho_sum)
end

# Random Walk
rw = random_walk(Matrix{Float64}(0.5I, 2, 2))
t_rw = @elapsed s_rw = sample(gauss_model, rw, n_steps;
    n_chains=4, initial=initial, n_burnin=n_burnin, seed=42)

# Slice
sl = slice(w=1.5)
t_sl = @elapsed s_sl = sample(gauss_model, sl, n_steps;
    n_chains=4, initial=initial, n_burnin=n_burnin, seed=42)

# MALA
ml = mala(0.5)
t_ml = @elapsed s_ml = sample(gauss_model, ml, n_steps;
    n_chains=4, initial=initial, n_burnin=n_burnin, seed=42)

# HMC
hm = hmc(0.1, 10)
t_hm = @elapsed s_hm = sample(gauss_model, hm, n_steps;
    n_chains=4, initial=initial, n_burnin=n_burnin, seed=42)

for (name, samp, t) in [("RW", s_rw, t_rw), ("Slice", s_sl, t_sl),
                          ("MALA", s_ml, t_ml), ("HMC", s_hm, t_hm)]
    ess_x = simple_ess(samp.pars[1, :, 1])
    ess_y = simple_ess(samp.pars[2, :, 1])
    min_ess = min(ess_x, ess_y)
    acc = round(samp.details[:acceptance_rate][1], digits=3)
    println("$name: ESS(x)=$(round(ess_x, digits=0)), ESS(y)=$(round(ess_y, digits=0)), ",
            "min_ESS/s=$(round(min_ess/t, digits=0)), accept=$acc")
end
```

    RW: ESS(x)=73.0, ESS(y)=74.0, min_ESS/s=343.0, accept=0.323
    Slice: ESS(x)=213.0, ESS(y)=213.0, min_ESS/s=919.0, accept=1.0
    MALA: ESS(x)=58.0, ESS(y)=59.0, min_ESS/s=185.0, accept=0.39
    HMC: ESS(x)=570.0, ESS(y)=571.0, min_ESS/s=550.0, accept=0.985

## Summary

| Sampler | When to use |
|----|----|
| **Slice** | No gradient available; want automatic step size; low-dimensional models |
| **MALA** | Gradient available; want simpler alternative to HMC; moderate dimensions |
| **Gibbs** | Block structure (e.g. hierarchical models); mixed parameter types; combining different samplers for different parameter groups |
| **RW** | Simple baseline; stochastic likelihoods where gradient is unavailable |
| **HMC/NUTS** | High-dimensional smooth targets with gradient; maximum ESS/step |

## R companion

For comparison, here is equivalent R code using the monty R package:

``` r
# Slice-like behaviour in R monty via the random walk with adaptive step
library(monty)
library(dust2)

# Define target
m <- monty_model_function(
  function(x) -0.5 * (x[1]^2 + (x[2] - 0.1 * x[1]^2 + 10)^2),
  packer = Packer(c("x", "y"))
)

# Random walk baseline
sampler_rw <- random_walk(vcv = diag(2) * 0.5)
samples_rw <- sample(m, sampler_rw, 5000, n_chains = 4)

# HMC (requires gradient)
sampler_hmc <- hmc(epsilon = 0.1, n_integration_steps = 10)
# Note: R monty does not yet include slice, MALA, or Gibbs —
# these are new in Odin.jl's Monty module.
```
