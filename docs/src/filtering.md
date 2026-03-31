# Filtering & Likelihood

Odin.jl provides two approaches for computing likelihoods from time-series data:
the **particle filter** (stochastic models) and the **unfilter** (deterministic ODE models).
Both can be bridged to the [Monty inference engine](@ref "Inference") for MCMC.

## Preparing Data

Observation data should be converted to a [`ObservedData`](@ref) object before likelihood evaluation:

```julia
data = ObservedData([
    (time=1.0, cases=5, deaths=0),
    (time=2.0, cases=12, deaths=1),
    (time=3.0, cases=8, deaths=0),
])

grouped = ObservedData([
    (time=1.0, group=:a, cases=5),
    (time=1.0, group=:b, cases=7),
    (time=2.0, group=:a, cases=6),
    (time=2.0, group=:b, cases=8),
]; group_field=:group)
```

The `time` column is extracted and stored separately; remaining columns become data
variables available to `compare_data` expressions in the model.

## Particle Filter (Stochastic Models)

The bootstrap particle filter estimates log-likelihood by running many
particles in parallel and resampling at each data time point.

```julia
# Create the filter
filt = Likelihood(gen, data;
    time_start = 0.0,
    n_particles = 200,
    dt = 0.25,
    seed = 42,
)

# Run with specific parameters — returns log-likelihood
ll = loglik(filt, (beta=0.5, gamma=0.1, N=1000.0, I0=10.0))
```

### Features

- **Systematic resampling** at each data time point
- **Pre-allocated** work buffers — minimal allocations after warmup
- **Independent likelihood objects can run in parallel** — avoid sharing one mutable filter instance across chains

## Unfilter (Deterministic Likelihood)

For ODE models without stochastic transitions, the unfilter integrates the system
deterministically and evaluates `compare_data` at each data time point:

```julia
uf = Likelihood(gen, data;
    time_start = 0.0,
    ode_control = ODEControl(atol=1e-8, rtol=1e-8),
)

ll = loglik(uf, pars)
```

The unfilter supports automatic differentiation via ForwardDiff, making it
compatible with gradient-based samplers like [HMC and NUTS](@ref "Inference").

## Monty Bridge

Convert a filter or unfilter to a [`MontyModel`](@ref) for use with MCMC samplers:

```julia
packer = Packer([:beta, :gamma]; fixed=(N=1000.0, I0=10.0))
likelihood = as_model(filt, packer)

# Combine with prior
prior = @prior begin
    beta ~ Exponential(1.0)
    gamma ~ Exponential(1.0)
end
posterior = likelihood + prior

# Sample with any monty sampler
samples = sample(posterior, sampler, 5000)
```

- **Particle filter** → stochastic `MontyModel` (no gradient)
- **Unfilter** → deterministic `MontyModel` (with ForwardDiff gradient)

## API Reference

```@docs
Odin.ObservedData
Odin.Likelihood
Odin.loglik
Odin.loglik_pointwise
Odin.loglik_gradient
Odin.as_model
Odin.DustFilter
Odin.DustUnfilter
```
