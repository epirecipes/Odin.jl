# Simulation

The Dust simulation engine manages system creation, time-stepping, state access, and multi-particle simulation.

## System Lifecycle

```julia
# 1. Create a generator from @odin
gen = @odin begin ... end

# 2. Create a system with parameters
sys = System(gen, pars; n_particles=1, dt=1.0, seed=42)

# 3. Set initial state
reset!(sys)

# 4. Simulate or step
result = simulate(sys, times)
# or step-by-step:
run_to!(sys, target_time)

# 5. Read state
state = state(sys)
```

## Creating Systems

A [`OdinModel`](@ref) is the compiled model factory returned by [`@odin`](@ref).
Pass it to [`System`](@ref) along with parameters to get a running [`DustSystem`](@ref):

```julia
pars = (beta=0.5, gamma=0.1, N=1000.0, I0=10.0)
sys = System(gen, pars; n_particles=100, dt=0.25, seed=42)
```

### Key options

| Keyword | Default | Description |
|---------|---------|-------------|
| `n_particles` | `1` | Number of independent stochastic realisations |
| `dt` | `1.0` | Time step for discrete-time models |
| `seed` | `nothing` | RNG seed for reproducibility |
| `ode_control` | `ODEControl()` | ODE solver settings (continuous models only) |

## Setting and Reading State

```julia
# Initialise all particles to initial conditions
reset!(sys)

# Read current state — returns n_state × n_particles matrix
state = state(sys)

# Manually set state (vector broadcasts to all particles)
set_state!(sys, [990.0, 10.0, 0.0])

# Or set per-particle state (matrix)
set_state!(sys, state_matrix)
```

## Running Simulations

### Full trajectory

[`simulate`](@ref) collects output at specified times:

```julia
times = 0.0:1.0:100.0
result = simulate(sys, times)
# Returns n_state × n_particles × n_times
```

For continuous models the built-in [DP5 or SDIRK4 solvers](@ref "ODE Solvers") are used;
for discrete-time models the system steps forward with `dt`.

### Step-by-step

Advance to a specific time without saving intermediate states:

```julia
run_to!(sys, 50.0)
state = state(sys)
```

## Data Comparison

If the model has `compare_data` expressions (e.g. `cases ~ Poisson(I)`), evaluate per-particle log-likelihoods at the current time:

```julia
ll = compare_data(sys, data_at_t)
# Returns a vector of length n_particles
```

## ODE Control

For continuous models, configure the ODE solver via [`ODEControl`](@ref):

```julia
ctrl = ODEControl(;
    atol = 1e-8,          # absolute tolerance
    rtol = 1e-8,          # relative tolerance
    max_steps = 10000,    # maximum integration steps
    solver = :dp5,        # :dp5 (default) or :sdirk (for stiff systems)
)

sys = System(gen, pars; ode_control=ctrl)
```

See [ODE Solvers](@ref) for details on the DP5 and SDIRK4 solvers.

## API Reference

```@docs
Odin.OdinModel
Odin.System
Odin.DustSystem
Odin.simulate
Odin.reset!
Odin.state
Odin.set_state!
Odin.run_to!
Odin.compare_data
```
