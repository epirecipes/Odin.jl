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

### Data Format Requirements

- **time**: Must be numeric (Float64) and strictly increasing (after sorting)
- **data columns**: Must match the `data()` declarations in your model
- **Missing data**: Use `NaN` to mark missing observations — these time points
  contribute zero to the log-likelihood

### Linking Data to Models

In the odin DSL, declare data variables and comparison distributions:

```julia
sir = @odin begin
    # ... state equations ...
    cases = data()                    # declares "cases" as observed data
    cases ~ Poisson(max(I, 1e-6))    # comparison distribution
end
```

The names must match exactly: `cases = data()` in the model pairs with the
`cases` column in `ObservedData`.

## Particle Filter (Stochastic Models)

The bootstrap particle filter estimates log-likelihood by running many
particles in parallel and resampling at each data time point.

```julia
filt = Likelihood(gen, data;
    time_start = 0.0,
    n_particles = 200,
    dt = 0.25,
    seed = 42,
)

ll = loglik(filt, (beta=0.5, gamma=0.1, N=1000.0, I0=10.0))
```

### How It Works

1. **Initialise** `n_particles` copies of the system at `time_start`
2. **Advance** all particles to the next data time point using the stochastic `update!`
3. **Evaluate** `compare_data` for each particle to get per-particle log-weights
4. **Resample** particles proportional to their weights (systematic resampling)
5. **Accumulate** log(mean weight) into the total log-likelihood
6. **Repeat** steps 2–5 for each data time point

### Features

- **Systematic resampling** at each data time point
- **Pre-allocated** work buffers — minimal allocations after warmup
- **Deterministic with seed** — same seed + same parameters = same log-likelihood
- **Trajectory saving** — optionally record full particle trajectories

### Saving Trajectories

```julia
filt = Likelihood(gen, data; n_particles=100, save_trajectories=true)
ll = loglik(filt, pars)
traj = last_trajectories(filt.inner)  # (n_state, n_particles, n_times)
```

### Tuning the Particle Filter

| Parameter | Effect | Guidance |
|-----------|--------|----------|
| `n_particles` | Variance of log-likelihood estimate | Start with 100–200; increase if MCMC mixing is poor |
| `dt` | Time step for stochastic update | Match the model's natural time scale |
| `seed` | RNG reproducibility | Fix for debugging; vary for variance estimation |

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

### Pointwise Log-Likelihood

Retrieve the per-observation log-likelihood contributions:

```julia
ll_vec = loglik_pointwise(uf, pars)  # Vector{Float64}, one per data point
```

### Gradients

```julia
ll, grad = loglik_gradient(uf, pars; method=:forward)   # ForwardDiff
ll, grad = loglik_gradient(uf, pars; method=:adjoint)    # Adjoint (symbolic Jacobian)
```

The `:forward` method uses ForwardDiff.jl through the ODE solve. The `:adjoint`
method uses the symbolic Jacobian (when available) for faster gradient computation
on large models.

## Filter vs Unfilter: When to Use Which

| | Particle Filter | Unfilter |
|---|---|---|
| **Model type** | Stochastic (discrete-time) | Deterministic (ODE) |
| **Gradients** | ✗ (use random walk / adaptive MH) | ✓ (ForwardDiff, adjoint) |
| **Cost** | O(n_particles × n_times) | O(n_times × ODE steps) |
| **Variance** | Stochastic (controlled by n_particles) | Deterministic |
| **Best samplers** | Random walk, adaptive MH | HMC, NUTS, MALA |

## Monty Bridge

Convert a filter or unfilter to a [`MontyModel`](@ref) for use with MCMC samplers:

```julia
packer = Packer([:beta, :gamma]; fixed=(N=1000.0, I0=10.0))
likelihood = as_model(filt, packer)

prior = @prior begin
    beta ~ Exponential(1.0)
    gamma ~ Exponential(1.0)
end
posterior = likelihood + prior

samples = sample(posterior, sampler, 5000)
```

- **Particle filter** → stochastic `MontyModel` (no gradient)
- **Unfilter** → deterministic `MontyModel` (with ForwardDiff gradient)

## Complete Example: ODE Fitting

```julia
using Odin

# Define model with comparison
sir = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    cases = data()
    cases ~ Poisson(max(I, 1e-6))
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10)
    N = parameter(1000)
end

# Observed data
data = ObservedData([
    (time=7.0,  cases=45),
    (time=14.0, cases=152),
    (time=21.0, cases=280),
    (time=28.0, cases=310),
    (time=35.0, cases=220),
])

# Create likelihood + packer
uf = Likelihood(sir, data)
packer = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))
ll_model = as_model(uf, packer)

# Add prior and sample
prior = @prior begin
    beta ~ Exponential(1.0)
    gamma ~ Exponential(1.0)
end
posterior = ll_model + prior
samples = sample(posterior, nuts(), 2000; n_chains=4)
```

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
