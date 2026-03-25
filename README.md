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
sys = dust_system_create(sir, (N=1000, I0=10, beta=0.3, gamma=0.1))
dust_system_set_state_initial!(sys)
result = dust_system_simulate(sys, 0.0:1.0:100.0)

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
sys = dust_system_create(sir_ode, (N=1000, I0=10, beta=0.3, gamma=0.1))
dust_system_set_state_initial!(sys)
result = dust_system_simulate(sys, 0.0:1.0:365.0)

# Stochastic model — multiple particles
sys = dust_system_create(sir_stoch, (N=1000, I0=10, beta=0.5, gamma=0.1);
                         n_particles=100, dt=0.25, seed=42)
dust_system_set_state_initial!(sys)
result = dust_system_simulate(sys, 0.0:1.0:365.0)
```

### State Management

```julia
dust_system_set_state_initial!(sys)           # Reset to initial conditions
dust_system_set_state!(sys, state_matrix)     # Set state directly (n_state × n_particles)
state = dust_system_state(sys)                # Get current state
dust_system_run_to_time!(sys, 100.0)          # Advance without recording
```

### Data Comparison

```julia
ll = dust_system_compare_data(sys, (cases=15,))  # Log-likelihood per particle
```

---

## Inference

Odin.jl provides a complete Bayesian inference pipeline: particle filter → likelihood → prior → posterior → MCMC.

### Step 1: Create a Likelihood

For **stochastic models**, use a bootstrap particle filter:

```julia
filter = dust_filter_create(sir_stoch;
    data    = data,          # Vector of NamedTuples with :time and observed fields
    time_start   = 0,
    n_particles  = 200,
    dt           = 0.25,
    seed         = 42)
```

For **deterministic ODE models**, use the unfilter:

```julia
unfilter = dust_unfilter_create(sir_ode;
    data       = data,
    time_start = 0)
```

### Step 2: Bridge to Monty

Convert the dust likelihood into a `MontyModel` for MCMC:

```julia
packer = monty_packer([:beta, :gamma])
ll = dust_likelihood_monty(filter, packer; fixed=(N=1000, I0=10))
```

### Step 3: Define Priors

```julia
prior = @monty_prior begin
    beta ~ Exponential(0.5)
    gamma ~ Exponential(0.3)
end
```

The `@monty_prior` macro automatically generates gradients for HMC.

### Step 4: Combine and Sample

```julia
posterior = monty_model_combine(ll, prior)

sampler = monty_sampler_random_walk(; vcv=diagm([0.01, 0.01]))

samples = monty_sample(posterior, sampler, 5000;
                       initial=[0.5, 0.1],
                       n_chains=4,
                       runner=monty_runner_serial())
```

### Available Samplers

| Sampler | Constructor | Use case |
|---------|-------------|----------|
| **Random walk MH** | `monty_sampler_random_walk(; vcv, boundaries)` | General purpose; supports `:reflect`/`:reject`/`:ignore` boundaries |
| **HMC** | `monty_sampler_hmc(ε, L; vcv)` | Models with gradients (via `@monty_prior` or ForwardDiff) |
| **NUTS** | `monty_sampler_nuts(; max_depth)` | Adaptive HMC — no tuning needed, best for smooth posteriors |
| **Adaptive** | `monty_sampler_adaptive(; target_acceptance, initial_vcv)` | Auto-tuning proposal (Spencer 2021 accelerated shaping) |
| **Parallel tempering** | `monty_sampler_parallel_tempering(temps, sampler)` | Multi-modal posteriors via replica exchange |

### Runners

```julia
monty_runner_serial()      # Sequential chains
monty_runner_threaded()    # Parallel chains via Julia threads
```

### Continuing Sampling

```julia
more_samples = monty_sample_continue(samples, 5000)
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

### Lower to a Simulatable Model

Compile the Petri net into an `@odin`-compatible model:

```julia
gen = lower(sir_age; mode=:ode, frequency_dependent=true, N=:N,
            params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))
sys = dust_system_create(gen, (beta=0.3, gamma=0.1, N=1000.0))
dust_system_set_state_initial!(sys)
result = dust_system_simulate(sys, 0.0:1.0:365.0)
```

### Pre-Built Networks

| Function | Model |
|----------|-------|
| `sir_net()` | SIR (susceptible → infected → recovered) |
| `seir_net()` | SEIR (with exposed compartment) |
| `sis_net()` | SIS (no recovered; re-susceptibility) |
| `sirs_net()` | SIRS (with waning immunity) |
| `seirs_net()` | SEIRS (exposed + waning) |
| `sir_vax_net()` | SIR with vaccination |
| `two_strain_sir_net()` | Two-strain SIR |

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

Progressive tutorials are in `vignettes/` as [Quarto](https://quarto.org/) documents. Each has a matching R version in its `R/` subdirectory.

| # | Title | Description |
|--:|-------|-------------|
| 01 | [Basic ODE Model: SIR](vignettes/01_basic_ode/01_basic_ode.qmd) | Introduction to `deriv()` for continuous-time ODE models |
| 02 | [Stochastic Discrete-Time SIR](vignettes/02_stochastic/02_stochastic.qmd) | Discrete-time models with `update()` and `Binomial` transitions |
| 03 | [Incidence Tracking with `zero_every`](vignettes/03_observations/03_observations.qmd) | Periodic-reset counters for fitting to case count data |
| 04 | [Age-Structured SIR with Arrays](vignettes/04_arrays/04_arrays.qmd) | Multi-dimensional state via `dim()`, contact matrices |
| 05 | [Particle Filter and Likelihood](vignettes/05_particle_filter/05_particle_filter.qmd) | Bootstrap particle filter with systematic resampling |
| 06 | [Bayesian Inference with MCMC](vignettes/06_inference/06_inference.qmd) | Complete inference pipeline: filter → prior → MCMC sampling |
| 07 | [Compositional Model Building](vignettes/07_categorical/07_categorical.qmd) | Petri net composition, stratification, and lowering |
| 08 | [Time-Varying Parameters](vignettes/08_time_varying/08_time_varying.qmd) | Step, linear, and spline interpolation for interventions |
| 09 | [Advanced SEIR-V Model](vignettes/09_advanced/09_advanced.qmd) | SEIR with vaccination, waning immunity, and time-varying rates |
| 10 | [Projections](vignettes/10_projections/10_projections.qmd) | Counterfactual scenarios from posterior samples |
| 11 | [Delay Model](vignettes/11_delay_model/11_delay_model.qmd) | Erlang-distributed delays and gamma-distributed compartments |
| 12 | [Reactive Policy](vignettes/12_reactive_policy/12_reactive_policy.qmd) | Threshold-based interventions with feedback control |
| 13 | [DynamicPPL Integration](vignettes/13_dynamicppl/13_dynamicppl.qmd) | Using Turing.jl/DynamicPPL for priors and MCMC |
| 14 | [SEIR with Delay & Vaccination](vignettes/14_delay_vaccination/14_delay_vaccination.qmd) | Shift-register delays, age structure, spillover FOI |
| 15 | [Vector-Borne Disease Dynamics](vignettes/15_vector_borne/15_vector_borne.qmd) | Ross-Macdonald malaria with seasonal rainfall forcing |
| 16 | [Multi-Stream Outbreak Inference](vignettes/16_multi_stream/16_multi_stream.qmd) | Fitting to cases + deaths simultaneously |
| 17 | [Mpox SEIR](vignettes/17_mpox_seir/17_mpox_seir.qmd) | Age-structured stochastic with vaccination strata |
| 18 | [Malaria Simple](vignettes/18_malaria_simple/18_malaria_simple.qmd) | Ross-Macdonald with seasonal forcing and ITN |
| 19 | [SARS-CoV-2 Multi-Region](vignettes/19_sarscov2_multiregion/19_sarscov2_multiregion.qmd) | Coupled 3-region ODE with time-varying Rt |

---

## API Reference

### DSL

| Symbol | Description |
|--------|-------------|
| `@odin` | Compile a model definition block into a `DustSystemGenerator` |

### Dust Runtime — System

| Function | Description |
|----------|-------------|
| `dust_system_create(gen, pars; n_particles, dt, seed)` | Create a simulation system |
| `dust_system_set_state_initial!(sys)` | Reset to initial conditions |
| `dust_system_set_state!(sys, state)` | Set state matrix directly |
| `dust_system_state(sys)` | Get current state (`n_state × n_particles`) |
| `dust_system_simulate(sys, times)` | Run simulation, recording at each time |
| `dust_system_run_to_time!(sys, t)` | Advance to time `t` without recording |
| `dust_system_compare_data(sys, data)` | Compute log-likelihood against data |

### Dust Runtime — Filtering

| Function | Description |
|----------|-------------|
| `dust_filter_create(gen; data, time_start, n_particles, dt, seed)` | Bootstrap particle filter |
| `dust_unfilter_create(gen; data, time_start)` | Deterministic (ODE-based) likelihood |
| `dust_likelihood_run!(filter, pars)` | Run filter, return log-likelihood |
| `dust_likelihood_monty(filter, packer; fixed)` | Convert to `MontyModel` for MCMC |

### Monty Inference — Models & Packers

| Function | Description |
|----------|-------------|
| `monty_model(density; parameters, gradient, domain)` | Wrap a density function as a model |
| `monty_model_combine(m1, m2)` | Sum two models (e.g., likelihood + prior) |
| `monty_packer(names; fixed)` | Map named parameters ↔ flat vectors |
| `monty_packer_grouped(names, groups; fixed)` | Grouped parameter packing |
| `@monty_prior` | DSL for prior specification with automatic gradients |

### Monty Inference — Samplers & Sampling

| Function | Description |
|----------|-------------|
| `monty_sampler_random_walk(; vcv, boundaries)` | Random walk Metropolis–Hastings |
| `monty_sampler_hmc(ε, L; vcv)` | Hamiltonian Monte Carlo |
| `monty_sampler_nuts(; max_depth)` | No-U-Turn Sampler (adaptive HMC via AdvancedHMC.jl) |
| `monty_sampler_adaptive(; target_acceptance, initial_vcv)` | Adaptive MCMC (Spencer 2021) |
| `monty_sampler_parallel_tempering(temps, sampler)` | Parallel tempering / replica exchange |
| `monty_runner_serial()` | Sequential multi-chain runner |
| `monty_runner_threaded()` | Threaded multi-chain runner |
| `monty_sample(model, sampler, n_steps; initial, n_chains, runner)` | Run MCMC |
| `monty_sample_continue(samples, n_steps)` | Continue from previous samples |

### DynamicPPL / Turing Integration

| Function | Description |
|----------|-------------|
| `to_turing_model(gen, data; priors...)` | Create a DynamicPPL `@model` from an Odin system |
| `dppl_prior(block)` | Define priors using DynamicPPL syntax |
| `dppl_to_monty_model(turing_model)` | Convert DynamicPPL model → `MontyModel` |
| `turing_sample(model, sampler, n; kwargs...)` | Sample using Turing.jl's MCMC infrastructure |
| `to_chains(samples)` | Convert `MontySamples` → `MCMCChains.Chains` |
| `@odin_model` | Convenience macro combining `@odin` + priors in one block |

### Categorical Extension

| Function | Description |
|----------|-------------|
| `EpiNet(species, transitions)` | Construct a labelled Petri net |
| `compose(net1, net2, ...)` | Merge networks on shared species |
| `compose_with_interface(nets, shared)` | Explicit interface composition |
| `stratify(net, groups; contact)` | Stratify across groups with contact matrix |
| `lower(net; mode, params, ...)` | Compile Petri net → `DustSystemGenerator` |
| `sir_net()`, `seir_net()`, `sis_net()`, ... | Pre-built epidemic networks |

---

## Comparison with R

| Concept | R (odin2 / dust2 / monty) | Julia (Odin.jl) |
|---------|---------------------------|------------------|
| Model definition | `odin({ ... })` | `@odin begin ... end` |
| System creation | `dust_system_create(gen, pars)` | `dust_system_create(gen, pars)` |
| Set initial state | `dust_system_set_state_initial(sys)` | `dust_system_set_state_initial!(sys)` |
| Simulate | `dust_system_simulate(sys, times)` | `dust_system_simulate(sys, times)` |
| Particle filter | `dust_filter_create(gen, ...)` | `dust_filter_create(gen; ...)` |
| Deterministic LL | `dust_unfilter_create(gen, ...)` | `dust_unfilter_create(gen; ...)` |
| Monty bridge | `dust_likelihood_monty(filter, packer)` | `dust_likelihood_monty(filter, packer)` |
| Packer | `monty_packer(c("beta", "gamma"))` | `monty_packer([:beta, :gamma])` |
| Prior | `monty_dsl({ beta ~ Exp(0.5) })` | `@monty_prior begin beta ~ Exponential(0.5) end` |
| MCMC | `monty_sample(model, sampler, n)` | `monty_sample(model, sampler, n)` |
| Random walk | `monty_sampler_random_walk(vcv=V)` | `monty_sampler_random_walk(; vcv=V)` |
| HMC | `monty_sampler_hmc(eps, L)` | `monty_sampler_hmc(ε, L)` |
| NUTS | *(not available)* | `monty_sampler_nuts()` |
| DynamicPPL priors | *(not available)* | `dppl_prior`, `to_turing_model()` |
| Compositional models | *(not available)* | `compose()`, `stratify()`, `lower()` |
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
