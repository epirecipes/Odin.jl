# Odin.jl

[![CI](https://github.com/epirecipes/Odin.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/epirecipes/Odin.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://epirecipes.github.io/Odin.jl/stable)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://epirecipes.github.io/Odin.jl/dev)

**A Julia toolkit for defining, simulating, and fitting compartmental infectious disease models.**

> Julia port of the R ecosystem [mrc-ide/odin2](https://github.com/mrc-ide/odin2) + [mrc-ide/dust2](https://github.com/mrc-ide/dust2) + [mrc-ide/monty](https://github.com/mrc-ide/monty), with a category-theory extension for compositional model building.

---

## Overview

Odin.jl provides four tightly integrated components:

| Component | R equivalent | What it does |
|-----------|-------------|--------------|
| **`@odin` DSL** | odin2 | Define epidemiological models — continuous-time ODEs or stochastic discrete-time — using a concise domain-specific language |
| **Dust runtime** | dust2 | Simulate systems with multi-particle support, run bootstrap particle filters, and compute likelihoods |
| **Monty inference** | monty | Bayesian parameter inference via MCMC — random walk Metropolis–Hastings, HMC, adaptive, and parallel tempering samplers |
| **Categorical extension** | *(new)* | Compose and stratify models using Petri net algebra, built on [Catlab.jl](https://github.com/AlgebraicJulia/Catlab.jl) |

The pipeline flows: **DSL definition → code generation → simulation → likelihood → inference**.

---

## Installation

```julia
using Pkg
Pkg.add("Odin")
```

Or for the development version:

```julia
using Pkg
Pkg.add(url="https://github.com/epirecipes/Odin.jl")
```

### Dependencies

Core dependencies are installed automatically:

- [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) — ODE solvers
- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) — statistical distributions
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) — automatic differentiation for HMC/NUTS and prior gradients
- [DynamicPPL.jl](https://github.com/TuringLang/DynamicPPL.jl) — probabilistic programming (Turing.jl integration)
- [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) — NUTS sampler backend
- [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl) — chain diagnostics and ESS
- [Catlab.jl](https://github.com/AlgebraicJulia/Catlab.jl) — category theory / ACSets for compositional models
- [PoissonRandom.jl](https://github.com/SciML/PoissonRandom.jl) — fast Poisson sampling
- [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl) — time-varying parameter support

---

## Quick Start

Define an SIR model, simulate it, and plot the results:

```julia
using Odin

# 1. Define the model
sir = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    N = parameter(1000)
    I0 = parameter(10)
    beta = parameter(0.3)
    gamma = parameter(0.1)
end

# 2. Create a dust system and simulate
sys = System(sir, (N=1000, I0=10, beta=0.3, gamma=0.1))
reset!(sys)
result = simulate(sys, 0.0:1.0:100.0)

# 3. Plot (using any plotting package)
using Plots
times = [r.time for r in result]
S = [r.state[1] for r in result]
I = [r.state[2] for r in result]
R = [r.state[3] for r in result]
plot(times, [S I R], label=["S" "I" "R"], lw=2, xlabel="Time", ylabel="Count")
```

---

## Model Definition

The `@odin` macro accepts a block of equations that define compartmental models. Models can be continuous (ODE) or discrete-time (stochastic).

### Continuous-Time ODE

Use `deriv()` to specify the right-hand side of an ODE system:

```julia
sir_ode = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    N = parameter(1000)
    I0 = parameter(10)
    beta = parameter(0.3)
    gamma = parameter(0.1)
end
```

### Discrete-Time Stochastic

Use `update()` with stochastic draws. The built-in variable `dt` is the time step:

```julia
sir_stoch = @odin begin
    p_SI = 1 - exp(-beta * I / N * dt)
    p_IR = 1 - exp(-gamma * dt)
    n_SI = Binomial(S, p_SI)
    n_IR = Binomial(I, p_IR)
    update(S) = S - n_SI
    update(I) = I + n_SI - n_IR
    update(R) = R + n_IR
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    N = parameter(1000)
    I0 = parameter(10)
    beta = parameter(0.5)
    gamma = parameter(0.1)
end
```

**Supported distributions:** `Normal`, `Poisson`, `Binomial`, `Beta`, `Gamma`, `Uniform`, `Exponential`, `NegativeBinomial`, `Cauchy`, `LogNormal`, `Weibull`, `Hypergeometric`, `BetaBinomial`, `Multinomial`.

### Arrays (Age-Structured Models)

Declare dimensions with `dim()` and index with brackets:

```julia
sir_age = @odin begin
    dim(S) = n_age
    dim(I) = n_age
    dim(R) = n_age
    dim(lambda) = n_age
    dim(C) = c(n_age, n_age)

    lambda[i] = beta * sum(C[i, j] * I[j] for j in 1:n_age) / N
    deriv(S[i]) = -lambda[i] * S[i]
    deriv(I[i]) = lambda[i] * S[i] - gamma * I[i]
    deriv(R[i]) = gamma * I[i]

    initial(S[i]) = N / n_age - I0
    initial(I[i]) = I0
    initial(R[i]) = 0

    n_age = parameter(3)
    N = parameter(1000)
    I0 = parameter(5)
    beta = parameter(0.3)
    gamma = parameter(0.1)
    C = parameter()
end
```

### Interpolation (Time-Varying Parameters)

Support for step, linear, and spline interpolation:

```julia
model = @odin begin
    beta = interpolate(beta_time, beta_value, "linear")
    deriv(S) = -beta * S * I / N
    # ...
    beta_time = parameter()
    beta_value = parameter()
end
```

Modes: `"constant"` (step function), `"linear"`, `"spline"` (cubic).

### Data Comparison (Likelihood)

Use `~` to declare how observed data relates to model state:

```julia
model = @odin begin
    # ... state equations ...
    cases = data()
    cases ~ Poisson(incidence + 1e-6)
end
```

### Output Variables

Declare derived quantities to include in simulation output:

```julia
model = @odin begin
    # ... state equations ...
    output(prevalence) = I
    output(Rt) = beta * S / N / gamma
end
```

### Incidence Tracking with `zero_every`

Track cumulative counts that reset at regular intervals — essential for fitting to periodic case data:

```julia
model = @odin begin
    initial(incidence, zero_every=1) = 0
    update(incidence) = incidence + n_SI
    # ... other equations ...
    cases = data()
    cases ~ Poisson(incidence + 1e-6)
end
```

---

## Simulation

### Creating and Running Systems

```julia
# ODE model — single trajectory
sys = System(sir_ode, (N=1000, I0=10, beta=0.3, gamma=0.1))
reset!(sys)
result = simulate(sys, 0.0:1.0:365.0)

# Stochastic model — multiple particles
sys = System(sir_stoch, (N=1000, I0=10, beta=0.5, gamma=0.1);
                         n_particles=100, dt=0.25, seed=42)
reset!(sys)
result = simulate(sys, 0.0:1.0:365.0)
```

### State Management

```julia
reset!(sys)           # Reset to initial conditions
Odin.dust_system_set_state!(sys, state_matrix) # Set state directly (n_state × n_particles)
st = state(sys)                   # Get current state
run_to!(sys, 100.0)          # Advance without recording
```

### Data Comparison

```julia
ll = Odin.dust_system_compare_data(sys, (cases=15,))  # Log-likelihood per particle
```

---

## Inference

Odin.jl provides a complete Bayesian inference pipeline: particle filter → likelihood → prior → posterior → MCMC.

### Step 1: Create a Likelihood

For **stochastic models**, use a bootstrap particle filter:

```julia
filter = Likelihood(sir_stoch;
    data    = data,          # Vector of NamedTuples with :time and observed fields
    time_start   = 0,
    n_particles  = 200,
    dt           = 0.25,
    seed         = 42)
```

For **deterministic ODE models**, use the unfilter:

```julia
unfilter = Likelihood(sir_ode;
    data       = data,
    time_start = 0)
```

### Step 2: Bridge to Monty

Convert the dust likelihood into a `MontyModel` for MCMC:

```julia
packer = Packer([:beta, :gamma])
ll = as_model(filter, packer; fixed=(N=1000, I0=10))
```

### Step 3: Define Priors

```julia
prior = @prior begin
    beta ~ Exponential(0.5)
    gamma ~ Exponential(0.3)
end
```

The `@prior` macro automatically generates gradients for HMC.

### Step 4: Combine and Sample

```julia
posterior = Odin.monty_model_combine(ll, prior)

sampler = random_walk(; vcv=diagm([0.01, 0.01]))

samples = sample(posterior, sampler, 5000;
                       initial=[0.5, 0.1],
                       n_chains=4,
                       runner=Serial())
```

### Available Samplers

| Sampler | Constructor | Use case |
|---------|-------------|----------|
| **Random walk MH** | `random_walk(; vcv, boundaries)` | General purpose; supports `:reflect`/`:reject`/`:ignore` boundaries |
| **HMC** | `hmc(ε, L; vcv)` | Models with gradients (via `@prior` or ForwardDiff) |
| **NUTS** | `nuts(; max_depth)` | Adaptive HMC — no tuning needed, best for smooth posteriors |
| **Adaptive** | `adaptive_mh(; target_acceptance, initial_vcv)` | Auto-tuning proposal (Spencer 2021 accelerated shaping) |
| **Parallel tempering** | `parallel_tempering(temps, sampler)` | Multi-modal posteriors via replica exchange |

### Runners

```julia
Serial()      # Sequential chains
Threaded()    # Parallel chains via Julia threads
```

### Continuing Sampling

```julia
more_samples = sample_continue(samples, 5000)
```

---

## Categorical Composition

Build complex models from simple components using Petri net algebra.

### Define Components

```julia
using Odin

infection = EpiNet([:S => 990.0, :I => 10.0],
                   [:inf => ([:S, :I] => [:I, :I], :beta)])

recovery = EpiNet([:I => 10.0, :R => 0.0],
                  [:rec => ([:I] => [:R], :gamma)])
```

### Compose

Merge networks by identifying shared species:

```julia
sir = compose(infection, recovery)
```

### Stratify

Replicate a model across groups (e.g., age classes) with a contact matrix:

```julia
C = [2.0 0.5; 0.5 1.0]
sir_age = stratify(sir, [:young, :old]; contact=C)
```

### Compile to a Simulatable Model

Compile the Petri net into an `@odin`-compatible model:

```julia
gen = compile(sir_age; mode=:ode, frequency_dependent=true, N=:N,
            params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))
sys = System(gen, (beta=0.3, gamma=0.1, N=1000.0))
reset!(sys)
result = simulate(sys, 0.0:1.0:365.0)
```

### Pre-Built Networks

| Function | Model |
|----------|-------|
| `SIR()` | SIR (susceptible → infected → recovered) |
| `SEIR()` | SEIR (with exposed compartment) |
| `SIS()` | SIS (no recovered; re-susceptibility) |
| `SIRS()` | SIRS (with waning immunity) |
| `SEIRS()` | SEIRS (exposed + waning) |
| `SIRVax()` | SIR with vaccination |
| `two_strain_SIR()` | Two-strain SIR |

---

## Performance

Odin.jl uses type-stable generated functions, pre-allocated buffers, and custom inline samplers (BTPE binomial, PoissonRandom.jl) for zero-allocation inner loops.

### Benchmark Comparison: Julia vs R+C++

| Benchmark | Julia | R+C++ | Speedup |
|-----------|------:|------:|--------:|
| ODE SIR simulation (100 steps) | 0.16 ms | 0.24 ms | **1.5×** |
| Stochastic SIR (1K particles, 100 steps) | 0.41 ms* | 8.6 ms | **21×*** |
| Particle filter (200 particles, 50 steps) | 1.8 ms | 2.6 ms | **1.5×** |
| ODE unfilter (50 data points) | 0.023 ms | 0.031 ms | **1.3×** |
| RW MCMC (5K steps, 1 chain) | 182 ms | 393 ms | **2.2×** |
| Adaptive MCMC (5K steps, 4 chains) | 1335 ms | 2987 ms | **2.2×** |
| NUTS (1K steps, 4 chains) | 8.4 s | N/A | *Julia only* |
| PF-MCMC (10K steps, 500 particles, 4 chains) | 185 s | 146 s | 0.8× |

*\* Inner simulation loop with pre-created system; convenience function includes one-time system setup.*

> Benchmarked on the same machine. R+C++ uses odin2-generated C++ compiled with dust2.

### Key Performance Features

- **Custom BTPE Binomial sampler**: 2× faster than Distributions.jl, zero allocations
- **Cached ODEProblem with `remake()`**: eliminates solver allocation in MCMC inner loop
- **Pre-allocated work buffers**: simulation loop runs with ≤4 allocations (output array only)
- **Zero-allocation MVN sampling**: in-place triangular multiply for MCMC proposals
- **Lazy RNG snapshots**: avoids copying 1000+ Xoshiro states on system creation

### Sampler ESS Comparison

End-to-end inference efficiency across samplers (ODE SIR model, 5 000 steps × 4 chains):

| Language | Sampler | Time (s) | min ESS | min ESS/s | max R̂ |
|----------|---------|----------|---------|-----------|-------|
| R | RW | 3.0 | 97.7 | 32.5 | 1.093 |
| R | Adaptive | 6.9 | 594.1 | 85.8 | 1.004 |
| Julia | RW | 3.4 | 107.1 | 31.4 | 1.113 |
| Julia | Adaptive | 2.6 | 691.5 | 268.3 | 1.005 |
| Julia | NUTS (dense) | 47.6 | 974.1 | 20.5 | 1.004 |

Key takeaways:

- **Julia's Adaptive sampler achieves 268 ESS/s** (3.1× R's 85.8 ESS/s) — the best overall ESS efficiency
- **Julia's NUTS sampler** (not available in R) produces the highest absolute ESS with perfect convergence
- R's C++ ODE solver is faster per evaluation, but Julia's adaptive proposal tuning compensates
- Julia's unfilter uses a single ODE solve with `saveat` for efficiency (~40 μs vs R's ~30 μs)
- NUTS is unique to the Julia port — R's monty lacks gradient-based samplers for ODE models

The NUTS sampler uses [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) with automatic bijections for constrained parameters (e.g., log-transform for positive rates), enabling efficient exploration of the posterior geometry.

### Correctness

Posterior samples from Julia and R are statistically indistinguishable:

| Comparison | ECDF Correlation | KS Statistic |
|------------|:----------------:|:------------:|
| RW MCMC β | 0.992 | 0.122 |
| Adaptive MCMC β | 0.9996 | 0.038 |
| PF-MCMC β | 0.9994 | 0.038 |
| PF-MCMC γ | 0.9996 | 0.031 |
| NUTS vs Adaptive β (Julia) | 0.9995 | 0.051 |

---

## Vignettes

Progressive tutorials are in `vignettes/` as [Quarto](https://quarto.org/) documents with pre-rendered Markdown. Each has a matching R version in its `R/` subdirectory.

| # | Title | Description |
|--:|-------|-------------|
| 01 | [Basic ODE Model: SIR](vignettes/01_basic_ode/01_basic_ode.md) | Introduction to `deriv()` for continuous-time ODE models |
| 02 | [Stochastic Discrete-Time SIR](vignettes/02_stochastic/02_stochastic.md) | Discrete-time models with `update()` and `Binomial` transitions |
| 03 | [Incidence Tracking with `zero_every`](vignettes/03_observations/03_observations.md) | Periodic-reset counters for fitting to case count data |
| 04 | [Age-Structured SIR with Arrays](vignettes/04_arrays/04_arrays.md) | Multi-dimensional state via `dim()`, contact matrices |
| 05 | [Particle Filter and Likelihood](vignettes/05_particle_filter/05_particle_filter.md) | Bootstrap particle filter with systematic resampling |
| 06 | [Bayesian Inference with MCMC](vignettes/06_inference/06_inference.md) | Complete inference pipeline: filter → prior → MCMC sampling |
| 07 | [Compositional Model Building](vignettes/07_categorical/07_categorical.md) | Petri net composition, stratification, and lowering |
| 08 | [Time-Varying Parameters](vignettes/08_time_varying/08_time_varying.md) | Step, linear, and spline interpolation for interventions |
| 09 | [Advanced SEIR-V Model](vignettes/09_advanced/09_advanced.md) | SEIR with vaccination, waning immunity, and time-varying rates |
| 10 | [Projections](vignettes/10_projections/10_projections.md) | Counterfactual scenarios from posterior samples |
| 11 | [Delay Model](vignettes/11_delay_model/11_delay_model.md) | Erlang-distributed delays and gamma-distributed compartments |
| 12 | [Reactive Policy](vignettes/12_reactive_policy/12_reactive_policy.md) | Threshold-based interventions with feedback control |
| 13 | [DynamicPPL Integration](vignettes/13_dynamicppl/13_dynamicppl.md) | Using Turing.jl/DynamicPPL for priors and MCMC |
| 14 | [SEIR with Delay & Vaccination](vignettes/14_delay_vaccination/14_delay_vaccination.md) | Shift-register delays, age structure, spillover FOI |
| 15 | [Vector-Borne Disease Dynamics](vignettes/15_vector_borne/15_vector_borne.md) | Ross-Macdonald malaria with seasonal rainfall forcing |
| 16 | [Multi-Stream Outbreak Inference](vignettes/16_multi_stream/16_multi_stream.md) | Fitting to cases + deaths simultaneously |
| 17 | [Mpox SEIR](vignettes/17_mpox_seir/17_mpox_seir.md) | Age-structured stochastic with vaccination strata |
| 18 | [Malaria Simple](vignettes/18_malaria_simple/18_malaria_simple.md) | Ross-Macdonald with seasonal forcing and ITN |
| 19 | [SARS-CoV-2 Multi-Region](vignettes/19_sarscov2_multiregion/19_sarscov2_multiregion.md) | Coupled 3-region ODE with time-varying Rt |
| 20 | [Yellow Fever SEIRV](vignettes/20_yellowfever/20_yellowfever.md) | Age-structured model with spillover and vaccination |
| 21 | [SIS with School Closure](vignettes/21_school_closure/21_school_closure.md) | Threshold-based school closure interventions |
| 22 | [Beta Blocks](vignettes/22_beta_blocks/22_beta_blocks.md) | Piecewise-constant transmission rate estimation |
| 23 | [Oropouche (OROV)](vignettes/23_orov/23_orov.md) | Vector-borne model with relapse dynamics |
| 24 | [Yellow Fever with Erlang Delays](vignettes/24_yf_delay/24_yf_delay.md) | SEIRV with Erlang delay compartments |
| 25 | [Yellow Fever 2-Track Vaccination](vignettes/25_yf_vtrack/25_yf_vtrack.md) | SEIR with dual vaccination tracking |
| 26 | [Complete Fitting Workflow](vignettes/26_fitting_workflow/26_fitting_workflow.md) | End-to-end inference pipeline with diagnostics |
| 27 | [Spatial Composition](vignettes/27_spatial_composition/27_spatial_composition.md) | Composing models across spatial patches |
| 28 | [Age Stratification](vignettes/28_stratification/28_stratification.md) | Stratifying models by age groups |
| 29 | [Multi-Pathogen Composition](vignettes/29_multi_pathogen/29_multi_pathogen.md) | Composing models for co-circulating pathogens |
| 30 | [Stiff ODE Models](vignettes/30_stiff_ode/30_stiff_ode.md) | L-stable SDIRK4 solver for stiff systems |
| 31 | [GPU-Accelerated Particle Filter](vignettes/31_gpu_filter/31_gpu_filter.md) | GPU parallelisation for bootstrap particle filters |
| 32 | [Advanced MCMC Samplers](vignettes/32_advanced_samplers/32_advanced_samplers.md) | Slice, MALA, and Gibbs sampling |
| 33 | [Sensitivity Analysis](vignettes/33_sensitivity/33_sensitivity.md) | Adjoint and forward sensitivity for ODE models |
| 34 | [Event Handling](vignettes/34_events/34_events.md) | Discontinuities and callbacks for interventions |
| 35 | [Stochastic Differential Equations](vignettes/35_sde/35_sde.md) | SDE models with Euler-Maruyama integration |
| 36 | [Model Selection](vignettes/36_model_selection/36_model_selection.md) | Model comparison via WAIC, LOO-CV, and Bayes factors |
| 37 | [Model Validation](vignettes/37_model_validation/37_model_validation.md) | Posterior predictive checks, residuals, and calibration |

---

## API Reference

### DSL

| Symbol | Description |
|--------|-------------|
| `@odin` | Compile a model definition block into a `OdinModel` |

### Dust Runtime — System

| Function | Description |
|----------|-------------|
| `System(gen, pars; n_particles, dt, seed)` | Create a simulation system |
| `reset!(sys)` | Reset to initial conditions |
| `Odin.dust_system_set_state!(sys, state)` | Set state matrix directly |
| `state(sys)` | Get current state (`n_state × n_particles`) |
| `simulate(sys, times)` | Run simulation, recording at each time |
| `run_to!(sys, t)` | Advance to time `t` without recording |
| `Odin.dust_system_compare_data(sys, data)` | Compute log-likelihood against data |

### Dust Runtime — Filtering

| Function | Description |
|----------|-------------|
| `Likelihood(gen; data, time_start, n_particles, dt, seed)` | Bootstrap particle filter |
| `Likelihood(gen; data, time_start)` | Deterministic (ODE-based) likelihood |
| `loglik(filter, pars)` | Run filter, return log-likelihood |
| `as_model(filter, packer; fixed)` | Convert to `MontyModel` for MCMC |

### Monty Inference — Models & Packers

| Function | Description |
|----------|-------------|
| `DensityModel(density; parameters, gradient, domain)` | Wrap a density function as a model |
| `Odin.monty_model_combine(m1, m2)` | Sum two models (e.g., likelihood + prior) |
| `Packer(names; fixed)` | Map named parameters ↔ flat vectors |
| `GroupedPacker(names, groups; fixed)` | Grouped parameter packing |
| `@prior` | DSL for prior specification with automatic gradients |

### Monty Inference — Samplers & Sampling

| Function | Description |
|----------|-------------|
| `random_walk(; vcv, boundaries)` | Random walk Metropolis–Hastings |
| `hmc(ε, L; vcv)` | Hamiltonian Monte Carlo |
| `nuts(; max_depth)` | No-U-Turn Sampler (adaptive HMC via AdvancedHMC.jl) |
| `adaptive_mh(; target_acceptance, initial_vcv)` | Adaptive MCMC (Spencer 2021) |
| `parallel_tempering(temps, sampler)` | Parallel tempering / replica exchange |
| `Serial()` | Sequential multi-chain runner |
| `Threaded()` | Threaded multi-chain runner |
| `sample(model, sampler, n_steps; initial, n_chains, runner)` | Run MCMC |
| `sample_continue(samples, n_steps)` | Continue from previous samples |

### DynamicPPL / Turing Integration

| Function | Description |
|----------|-------------|
| `to_turing_model(gen, data; priors...)` | Create a DynamicPPL `@model` from an Odin system |
| `dppl_prior(block)` | Define priors using DynamicPPL syntax |
| `dppl_to_DensityModel(turing_model)` | Convert DynamicPPL model → `MontyModel` |
| `turing_sample(model, sampler, n; kwargs...)` | Sample using Turing.jl's MCMC infrastructure |
| `to_chains(samples)` | Convert `Samples` → `MCMCChains.Chains` |
| `@odin_model` | Convenience macro combining `@odin` + priors in one block |

### Categorical Extension

| Function | Description |
|----------|-------------|
| `EpiNet(species, transitions)` | Construct a labelled Petri net |
| `compose(net1, net2, ...)` | Merge networks on shared species |
| `compose_with_interface(nets, shared)` | Explicit interface composition |
| `stratify(net, groups; contact)` | Stratify across groups with contact matrix |
| `compile(net; mode, params, ...)` | Compile Petri net → `OdinModel` |
| `SIR()`, `SEIR()`, `SIS()`, ... | Pre-built epidemic networks |

---

## Comparison with R

| Concept | R (odin2 / dust2 / monty) | Julia (Odin.jl) |
|---------|---------------------------|------------------|
| Model definition | `odin({ ... })` | `@odin begin ... end` |
| System creation | `dust_system_create(gen, pars)` | `System(gen, pars)` |
| Set initial state | `dust_system_set_state_initial(sys)` | `reset!(sys)` |
| Simulate | `dust_system_simulate(sys, times)` | `simulate(sys, times)` |
| Particle filter | `dust_filter_create(gen, ...)` | `Likelihood(gen; ...)` |
| Deterministic LL | `dust_unfilter_create(gen, ...)` | `Likelihood(gen; ...)` |
| Monty bridge | `dust_likelihood_monty(filter, packer)` | `as_model(filter, packer)` |
| Packer | `monty_packer(c("beta", "gamma"))` | `Packer([:beta, :gamma])` |
| Prior | `monty_dsl({ beta ~ Exp(0.5) })` | `@prior begin beta ~ Exponential(0.5) end` |
| MCMC | `monty_sample(model, sampler, n)` | `sample(model, sampler, n)` |
| Random walk | `monty_sampler_random_walk(vcv=V)` | `random_walk(; vcv=V)` |
| HMC | `monty_sampler_hmc(eps, L)` | `hmc(ε, L)` |
| NUTS | *(not available)* | `nuts()` |
| DynamicPPL priors | *(not available)* | `dppl_prior`, `to_turing_model()` |
| Compositional models | *(not available)* | `compose()`, `stratify()`, `compile()` |
| Mutating fns | — | Use `!` suffix (Julia convention) |
| Parameters | `list(beta=0.3)` | `(beta=0.3,)` (NamedTuple) |

---

## Acknowledgements

Odin.jl is a Julia port of the R ecosystem developed by the [MRC Centre for Global Infectious Disease Analysis](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/) at Imperial College London:

- [odin2](https://github.com/mrc-ide/odin2) — DSL compiler (Rich FitzJohn et al.)
- [dust2](https://github.com/mrc-ide/dust2) — simulation runtime (Rich FitzJohn et al.)
- [monty](https://github.com/mrc-ide/monty) — inference toolbox (Rich FitzJohn et al.)

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

If you use Odin.jl in published work, please cite the original R packages:

> FitzJohn, R. et al. (2024). odin2: Next Generation ODE and Stochastic Compartmental Modelling. R package. https://github.com/mrc-ide/odin2
