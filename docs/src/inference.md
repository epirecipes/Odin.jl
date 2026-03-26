# Inference

The Monty module provides MCMC sampling infrastructure: models, parameter packers,
samplers, runners, and the main sampling loop. It also includes bridges to
DynamicPPL/Turing.jl.

## MontyModel

A [`MontyModel`](@ref) wraps a log-density function with optional gradient and domain information:

```julia
model = DensityModel(;
    density = θ -> -sum(θ.^2),
    parameters = [:x, :y],
    domain = [(-Inf, Inf), (-Inf, Inf)],
)
```

### Model Combination

Combine models (e.g., likelihood + prior) with `+`:

```julia
posterior = likelihood + prior
```

This creates a new model whose density is the sum of the component densities.
If both models have gradients, the combined model will also have a gradient.

## MontyPacker

Pack/unpack between named parameters and flat numeric vectors:

```julia
packer = Packer([:beta, :gamma];
    fixed = (N=1000.0, I0=10.0))

# Pack: named → vector
θ = packer(Dict(:beta => 0.5, :gamma => 0.1))  # [0.5, 0.1]

# Unpack: vector → named tuple with fixed values merged
pars = Odin.unpack(packer, [0.5, 0.1])
# (beta=0.5, gamma=0.1, N=1000.0, I0=10.0)
```

### Grouped Packer

For models with shared and per-group parameters (e.g., hierarchical models):

```julia
gpacker = GroupedPacker([:beta, :gamma]; n_groups=4)
```

## Samplers

### Random Walk Metropolis–Hastings

The simplest sampler — proposes from a multivariate normal centred on the current position:

```julia
sampler = random_walk(;
    vcv = [0.01 0.0; 0.0 0.01],
    boundaries = :reflect,  # :reflect, :reject, or :ignore
)
```

### Adaptive MCMC

Implements the accelerated shaping algorithm (Spencer 2021). The proposal
variance-covariance matrix is learned online from the chain history:

```julia
sampler = adaptive_mh(;
    initial_vcv = [0.01 0.0; 0.0 0.01],
    acceptance_target = 0.234,
    forget_rate = 0.2,
    log_scaling_update = true,
)
```

### Hamiltonian Monte Carlo

Requires a model with gradient support (e.g., via `@prior` or ForwardDiff through
an unfilter):

```julia
sampler = hmc(
    0.01,    # step size ε
    10;      # number of leapfrog steps L
    vcv = I(2),
)
```

### NUTS (No-U-Turn Sampler)

Automatic step size tuning and mass matrix adaptation via AdvancedHMC.jl:

```julia
sampler = nuts(;
    max_depth = 10,
    target_acceptance = 0.8,
    n_adaption = 1000,
)
```

### Parallel Tempering

Run multiple temperature-tempered chains with replica exchange:

```julia
base = random_walk(; vcv=V)
sampler = parallel_tempering(base, 4)
```

## Runners

Control how multiple chains are executed:

```julia
runner = Serial()     # sequential chains
runner = Threaded()   # parallel chains via Julia threads
```

## Sampling

### Main Loop

```julia
samples = sample(model, sampler, n_steps;
    initial = Dict(:beta => 0.3, :gamma => 0.1),
    n_chains = 4,
    runner = Serial(),
    burnin = 0,
    thinning = 1,
)
```

Returns a [`Samples`](@ref) object with:
- `samples.pars` — parameter array `(n_pars × n_steps × n_chains)`
- `samples.density` — log-density values
- `samples.details` — sampler-specific diagnostics

### Continuation

Continue sampling from the final state of a previous run:

```julia
more = sample_continue(samples, model, sampler, 5000)
```

## Prior Definition

### `@prior`

Define priors with automatic density, gradient, and direct sampling:

```julia
prior = @prior begin
    beta ~ Exponential(1.0)
    gamma ~ Gamma(2.0, 0.5)
end
```

Supports all distributions from Distributions.jl.

### DynamicPPL Priors

Use DynamicPPL syntax for priors, automatically converted to a `MontyModel`:

```julia
prior_model = dppl_prior() do
    beta ~ Exponential(1.0)
    gamma ~ Gamma(2.0, 0.5)
end
```

## DynamicPPL / Turing.jl Integration

### Full Turing Model

Wrap an Odin system + data into a DynamicPPL model:

```julia
turing_model = to_turing_model(gen, data;
    priors = (
        beta = Exponential(1.0),
        gamma = Gamma(2.0, 0.5),
    ),
    fixed = (N=1000.0, I0=10.0),
    time_start = 0.0,
)

# Sample with Turing or Odin samplers
chain = sample(turing_model, NUTS(), 5000)
samples = turing_sample(turing_model, adaptive_mh(), 5000; n_chains=4)
```

### LogDensityProblems Interface

Any `MontyModel` implements the `LogDensityProblems` interface, so it works with
AdvancedHMC.jl, AdvancedMH.jl, and other compatible samplers:

```julia
using LogDensityProblems
logdensity = as_logdensity(model)
LogDensityProblems.logdensity(logdensity, θ)
LogDensityProblems.dimension(logdensity)
```

### MCMCChains Conversion

Convert between Odin's `Samples` and `MCMCChains.Chains`:

```julia
using MCMCChains

# Odin → MCMCChains (for diagnostics and plotting)
chain = to_chains(samples)
summarystats(chain)

# MCMCChains → Odin
samples2 = from_chains(chain)
```

## `@odin_model` Convenience Macro

Combine model definition, prior specification, and parameter packing in a single block:

```julia
model = @odin_model begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10.0)
    N = parameter(1000.0)
    @prior begin
        beta ~ Gamma(2.0, 0.25)
        gamma ~ Gamma(2.0, 0.05)
    end
    @fixed I0 = 10.0 N = 1000.0
end
# Returns: model.system, model.prior, model.packer
```

## API Reference

### Models and Packers

```@docs
Odin.MontyModel
Odin.monty_model
Odin.monty_model_combine
Odin.MontyPacker
Odin.monty_packer
Odin.MontyPackerGrouped
Odin.monty_packer_grouped
```

### Samplers

```@docs
Odin.Odin.MontyRandomWalkSampler
Odin.monty_sampler_random_walk
Odin.Odin.MontyAdaptiveSampler
Odin.monty_sampler_adaptive
Odin.Odin.MontyHMCSampler
Odin.monty_sampler_hmc
Odin.Odin.MontyNUTSSampler
Odin.monty_sampler_nuts
Odin.Odin.MontyParallelTemperingSampler
Odin.monty_sampler_parallel_tempering
```

### Runners and Sampling

```@docs
Odin.Odin.MontySerialRunner
Odin.Odin.MontyThreadedRunner
Odin.monty_runner_serial
Odin.monty_runner_threaded
Odin.Samples
Odin.monty_sample
Odin.monty_sample_continue
Odin.@prior
```

### DynamicPPL Bridge

```@docs
Odin.as_logdensity
Odin.to_turing_model
Odin.turing_sample
Odin.dppl_prior
Odin.dppl_to_monty_model
Odin.to_chains
Odin.from_chains
```
