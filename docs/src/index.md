# Odin.jl

*A Julia port of the R odin2/dust2/monty infectious disease modelling ecosystem.*

## Overview

Odin.jl provides a complete framework for building, simulating, and fitting compartmental models to data. It consists of five integrated modules:

| Module | R Equivalent | Purpose |
|--------|-------------|---------|
| **DSL** (`@odin`) | odin2 | Define models using a domain-specific language |
| **Dust** | dust2 | Simulate systems, particle filters, deterministic likelihood |
| **Monty** | monty | MCMC samplers, packers, inference loops |
| **GPU** | *(new)* | GPU-accelerated simulation and filtering |
| **Categorical** | *(new)* | Compose and stratify models via category theory |

## Quick Start

```julia
using Odin

# Define an SIR model
sir = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    N = parameter(1000)
    I0 = parameter(10)
    beta = parameter(0.5)
    gamma = parameter(0.1)
end

# Simulate
sys = System(sir, (beta=0.5, gamma=0.1, N=1000.0, I0=10.0))
reset!(sys)
times = 0.0:1.0:100.0
result = simulate(sys, times)
```

See the [Getting Started](@ref) guide for a more complete walkthrough.

## Package Features

- **Macro-based DSL**: [`@odin`](@ref) compiles model definitions to efficient Julia code at macro-expansion time
- **ODE and discrete-time models**: Continuous and stochastic discrete-time dynamics with built-in [DP5 and SDIRK4 solvers](@ref "ODE Solvers")
- **Arrays**: Multi-dimensional state variables with `dim()` declarations (up to 8D)
- **Interpolation**: Time-varying parameters via constant, linear, or spline interpolation
- **[Particle filter](@ref "Filtering & Likelihood")**: Bootstrap filter with systematic resampling for stochastic likelihood
- **Unfilter**: Deterministic ODE-based likelihood for continuous models
- **[Five MCMC samplers](@ref "Inference")**: Random Walk, HMC, NUTS, Adaptive, Parallel Tempering
- **DynamicPPL integration**: Use Turing.jl priors and samplers with Odin models
- **[GPU acceleration](@ref "GPU Acceleration")**: Metal/CUDA/AMDGPU backends for massively parallel filtering and simulation
- **[Categorical composition](@ref "Categorical Models")**: Build complex models by composing simple ones via category theory
- **Zero-allocation inner loops**: Custom fast random number generators, pre-allocated work buffers

## Vignettes

Progressive tutorials are available in the `vignettes/` directory:

| # | Topic | Key concepts |
|---|-------|-------------|
| 01 | Basic ODE | SIR model, `deriv`, `initial`, `parameter` |
| 02 | Stochastic models | `update`, `Binomial`, discrete-time dynamics |
| 03 | Observations | Incidence, `zero_every`, `Poisson` comparison |
| 04 | Time-varying parameters | `interpolate()`, time-dependent β |
| 05 | Arrays | Age-structured SIR, `dim()`, `sum()` |
| 06 | Vaccination | Multi-dimensional arrays, vaccination strata |
| 07 | Particle filter | Bootstrap PF, log-likelihood estimation |
| 08 | Inference | MCMC with `monty_sample`, posterior diagnostics |
| 09 | Projections | Counterfactual scenarios from posterior |
| 10 | Advanced model | Complex real-world epi model |
| 11 | Categorical models | Composition, stratification via `EpiNet` |
| 12 | HMC & NUTS | Gradient-based sampling with bijectors |
| 13 | DynamicPPL | Turing.jl integration, hierarchical priors |
| 14 | Delay & vaccination | Shift registers, age structure, spillover |
| 15 | Vector-borne disease | Human-mosquito coupling, rainfall forcing |
| 16 | Multi-stream inference | Cases + deaths joint fitting |

Each vignette has a matching R version for cross-language comparison.

## Contents

```@contents
Pages = [
    "getting_started.md",
    "dsl.md",
    "simulation.md",
    "filtering.md",
    "inference.md",
    "solvers.md",
    "gpu.md",
    "categorical.md",
    "api.md",
]
Depth = 2
```
