# Filtering & Likelihood

Odin.jl provides two approaches for computing likelihoods from time-series data:
the **particle filter** (stochastic models) and the **unfilter** (deterministic ODE models).
Both can be bridged to the [Monty inference engine](@ref "Inference") for MCMC.

## Preparing Data

Observation data must be converted to a [`FilterData`](@ref) object using [`dust_filter_data`](@ref):

```julia
# From keyword arguments
data = dust_filter_data(;
    time = [1.0, 2.0, 3.0, 4.0, 5.0],
    cases = [5, 12, 8, 15, 20],
    deaths = [0, 1, 0, 2, 1],
)

# From a DataFrame
using DataFrames
df = DataFrame(time=[1.0, 2.0, 3.0], cases=[5, 12, 8])
data = dust_filter_data(df)
```

The `time` column is extracted and stored separately; remaining columns become data
variables available to `compare_data` expressions in the model.

## Particle Filter (Stochastic Models)

The bootstrap particle filter estimates log-likelihood by running many
particles in parallel and resampling at each data time point.

```julia
# Create the filter
filt = dust_filter_create(gen;
    data = data,
    time_start = 0.0,
    n_particles = 200,
    dt = 0.25,
    seed = 42,
)

# Run with specific parameters — returns log-likelihood
ll = dust_likelihood_run!(filt, (beta=0.5, gamma=0.1, N=1000.0, I0=10.0))
```

### Features

- **Systematic resampling** at each data time point
- **Pre-allocated** work buffers — minimal allocations after warmup
- **Thread-safe** — multiple filters can run in parallel

## Unfilter (Deterministic Likelihood)

For ODE models without stochastic transitions, the unfilter integrates the system
deterministically and evaluates `compare_data` at each data time point:

```julia
uf = dust_unfilter_create(gen;
    data = data,
    time_start = 0.0,
    ode_control = DustODEControl(atol=1e-8, rtol=1e-8),
)

ll = dust_unfilter_run!(uf, pars)
```

The unfilter supports automatic differentiation via ForwardDiff, making it
compatible with gradient-based samplers like [HMC and NUTS](@ref "Inference").

## Monty Bridge

Convert a filter or unfilter to a [`MontyModel`](@ref) for use with MCMC samplers:

```julia
packer = monty_packer([:beta, :gamma]; fixed=(N=1000.0, I0=10.0))
likelihood = dust_likelihood_monty(filt, packer)

# Combine with prior
prior = @monty_prior begin
    beta ~ Exponential(1.0)
    gamma ~ Exponential(1.0)
end
posterior = likelihood + prior

# Sample with any monty sampler
samples = monty_sample(posterior, sampler, 5000)
```

- **Particle filter** → stochastic `MontyModel` (no gradient)
- **Unfilter** → deterministic `MontyModel` (with ForwardDiff gradient)

## API Reference

```@docs
Odin.FilterData
Odin.dust_filter_data
Odin.DustFilter
Odin.dust_filter_create
Odin.dust_likelihood_run!
Odin.DustUnfilter
Odin.dust_unfilter_create
Odin.dust_unfilter_run!
Odin.dust_likelihood_monty
```
