# API Reference

Complete index of all exported functions and types, organised by module.

## DSL — Model Definition

| Function / Macro | Description |
|:-----------------|:------------|
| `@odin` | Define an ODE or discrete-time epidemiological model |
| `@odin_model` | Alias for `@odin` |
| `@prior` | Define a prior distribution for MCMC |
| `validate_model` | Check a model definition for errors |
| `show_code` | Display the generated Julia code for a model |

```@docs
Odin.@odin
Odin.validate_model
Odin.show_code
```

## Simulation

| Function / Type | Description |
|:----------------|:------------|
| `simulate` | Run a model forward, returning state trajectories |
| `System` | Create a `DustSystem` from a model generator |
| `reset!` | Reset system state to initial conditions |
| `state` | Get the current state vector |
| `set_state!` | Set the state vector |
| `run_to!` | Advance the system to a given time |
| `compare_data` | Evaluate log-likelihood of data given current state |

```@docs
Odin.simulate
Odin.System
Odin.reset!
Odin.run_to!
Odin.compare_data
```

## Likelihood & Filtering

| Function / Type | Description |
|:----------------|:------------|
| `Likelihood` | Create a likelihood object (auto-selects filter/unfilter) |
| `loglik` | Evaluate the log-likelihood at given parameters |
| `loglik_gradient` | Compute log-likelihood and its gradient (forward or adjoint) |
| `loglik_pointwise` | Per-time-point log-likelihood contributions |
| `as_model` | Convert a likelihood + packer into a `MontyModel` for MCMC |
| `DustFilter` | Bootstrap particle filter for stochastic likelihoods |
| `DustUnfilter` | Deterministic ODE-based likelihood |

```@docs
Odin.Likelihood
Odin.loglik
Odin.loglik_gradient
Odin.loglik_pointwise
Odin.as_model
```

## Parameter Packing

| Function / Type | Description |
|:----------------|:------------|
| `Packer` | Create a parameter packer (scalar + array params → flat vector) |
| `GroupedPacker` | Packer for multi-group models (shared + varied params) |
| `MontyPacker` | Low-level packer type |
| `MontyPackerGrouped` | Low-level grouped packer type |

```@docs
Odin.Packer
Odin.GroupedPacker
```

## Inference — Samplers

| Function | Description |
|:---------|:------------|
| `sample` | Run MCMC sampling with a model, sampler, and runner |
| `sample_continue` | Continue a previous sampling run |
| `random_walk` | Random-walk Metropolis-Hastings sampler |
| `hmc` | Hamiltonian Monte Carlo sampler |
| `nuts` | No-U-Turn Sampler (NUTS) via DynamicPPL/Turing bridge |
| `adaptive_mh` | Adaptive Metropolis-Hastings (Spencer 2021) |
| `mala` | Metropolis-adjusted Langevin algorithm |
| `slice` | Slice sampler |
| `parallel_tempering` | Replica exchange / parallel tempering |
| `gibbs` | Gibbs sampler (component-wise) |

```@docs
Odin.sample
Odin.sample_continue
Odin.random_walk
Odin.hmc
Odin.adaptive_mh
Odin.parallel_tempering
```

## Inference — Runners

| Type | Description |
|:-----|:------------|
| `Serial` | Single-threaded runner |
| `Threaded` | Multi-threaded runner (one chain per thread) |
| `Simultaneous` | All chains advanced simultaneously |
| `DistributedRunner` | Distributed runner via `Distributed.jl` |

```@docs
Odin.Serial
Odin.Threaded
Odin.Simultaneous
```

## Inference — Models

| Function / Type | Description |
|:----------------|:------------|
| `DensityModel` | Create a model from a log-density function |
| `MontyModel` | Core model type wrapping density + gradient + domain |
| `Observer` | Attach observers to monitor MCMC chains |

```@docs
Odin.DensityModel
```

## Sensitivity Analysis

| Function / Type | Description |
|:----------------|:------------|
| `sensitivity` | Compute sensitivity indices (forward, adjoint, Sobol, Morris) |
| `ForwardSensitivityResult` | Result of forward sensitivity analysis |
| `AdjointSensitivityResult` | Result of adjoint sensitivity analysis |
| `SobolResult` | Sobol sensitivity indices (first-order + total) |
| `MorrisResult` | Morris screening results (μ* and σ) |

```@docs
Odin.sensitivity
```

## Model Validation & Selection

| Function | Description |
|:---------|:------------|
| `ppc_check` | Posterior predictive check |
| `residual_diagnostics` | Compute residuals and diagnostics |
| `calibration_check` | PIT-based calibration assessment |
| `prior_predictive` | Prior predictive simulation |
| `sbc_check` | Simulation-based calibration |
| `aic`, `aicc`, `bic`, `dic`, `waic`, `loo` | Information criteria |
| `compare` | Compare models by information criteria |

```@docs
Odin.ppc_check
Odin.residual_diagnostics
Odin.calibration_check
Odin.aic
Odin.compare
```

## Events

| Type | Description |
|:-----|:------------|
| `ContinuousEvent` | Event triggered by a continuous condition |
| `DiscreteEvent` | Event triggered at discrete time steps |
| `TimedEvent` | Event triggered at specified times |
| `EventSet` | Collection of events attached to a model |

```@docs
Odin.ContinuousEvent
Odin.DiscreteEvent
Odin.TimedEvent
Odin.EventSet
```

## ODE Solvers

| Function / Type | Description |
|:----------------|:------------|
| `ODEControl` | Configure ODE solver tolerances and options |
| `sdirk_solve!` | Solve with the SDIRK4 implicit method (stiff systems) |
| `sde_solve!` | Solve stochastic differential equations |
| `SDIRKWorkspace` | Pre-allocated workspace for SDIRK solver |
| `SDEWorkspace` | Pre-allocated workspace for SDE solver |

```@docs
Odin.ODEControl
```

## GPU Acceleration

| Function / Type | Description |
|:----------------|:------------|
| `GPUBackend` | Abstract GPU backend type |
| `gpu_backend` | Get or set the active GPU backend |
| `has_gpu` | Check if a GPU is available |
| `GPUDustFilter` | GPU-accelerated particle filter |

```@docs
Odin.gpu_backend
Odin.has_gpu
```

## DynamicPPL / Turing Integration

| Function | Description |
|:---------|:------------|
| `as_logdensity` | Wrap a `MontyModel` as a `LogDensityProblems` interface |
| `to_turing_model` | Convert dust likelihood + prior into a Turing `@model` |
| `turing_sample` | Sample using Turing.jl samplers (NUTS, HMC, etc.) |
| `dppl_prior` | Define a prior using DynamicPPL syntax |
| `to_chains` | Convert Odin samples to `MCMCChains.Chains` |
| `from_chains` | Convert `MCMCChains.Chains` to Odin sample format |

```@docs
Odin.to_turing_model
Odin.turing_sample
Odin.to_chains
```

## Category Theory — Compositional Modelling

| Function / Type | Description |
|:----------------|:------------|
| `EpiNet` | Epidemiological Petri net (species + transitions) |
| `add_species!` | Add a species (compartment) to a net |
| `add_transition!` | Add a transition (flow) to a net |
| `compose` | Compose two nets by merging shared species |
| `stratify` | Stratify a net by age, space, or other structure |
| `lower_expr` | Lower an `EpiNet` to `@odin`-compatible expressions |
| `compile` | Compile an `EpiNet` directly to a runnable model |

**Built-in templates:** `SIR`, `SEIR`, `SIS`, `SIRS`, `SEIRS`, `SIRVax`, `TwoStrainSIR`

```@docs
Odin.EpiNet
Odin.add_species!
Odin.add_transition!
Odin.compose
Odin.stratify
Odin.lower_expr
```

## Full Index

```@index
Pages = [
    "dsl.md",
    "simulation.md",
    "filtering.md",
    "inference.md",
    "solvers.md",
    "gpu.md",
    "categorical.md",
]
```
