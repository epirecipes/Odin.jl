# Getting Started

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/epirecipes/Odin.jl")
```

Or in the Pkg REPL:

```
] add https://github.com/epirecipes/Odin.jl
```

## Your First Model

### 1. Define a model

Use the `@odin` macro to define a system of ODEs:

```julia
using Odin

sir = @odin begin
    # Differential equations
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I

    # Initial conditions
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0

    # Parameters with defaults
    N = parameter(1000)
    I0 = parameter(10)
    beta = parameter(0.5)
    gamma = parameter(0.1)
end
```

The `@odin` macro compiles this into a `OdinModel` — a factory for creating runnable systems.

### 2. Create a system and simulate

```julia
pars = (beta=0.5, gamma=0.1, N=1000.0, I0=10.0)
sys = System(sir, pars)
reset!(sys)

times = 0.0:1.0:100.0
result = simulate(sys, times)
# result is a 3×101 matrix (3 states × 101 time points)
```

### 3. Fit to data

```julia
# Observed incidence data
data = ObservedData(;
    time = 1.0:1.0:50.0,
    cases = [3, 5, 8, 12, ...]  # your observed counts
)

# Create an unfilter (deterministic likelihood)
uf = Likelihood(sir; data=data, time_start=0.0)

# Pack parameters for MCMC
packer = Packer([:beta, :gamma];
    fixed = (N=1000.0, I0=10.0))

# Define priors
prior = @prior begin
    beta ~ Exponential(1.0)
    gamma ~ Exponential(1.0)
end

# Bridge to monty model and combine with prior
likelihood = as_model(uf, packer)
posterior = likelihood + prior

# Sample with adaptive MCMC
sampler = adaptive_mh(;
    initial_vcv = [0.01 0.0; 0.0 0.01])
samples = sample(posterior, sampler, 5000;
    initial = Dict(:beta => 0.3, :gamma => 0.1),
    n_chains = 4)
```

### 4. Examine results

```julia
# Parameter estimates
println("β: ", mean(samples.pars[1, :, :]))
println("γ: ", mean(samples.pars[2, :, :]))
println("Acceptance: ", mean(samples.details["accept"]))
```

## Stochastic Models

Replace `deriv` with `update` and add stochastic transitions:

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

    beta = parameter(0.5)
    gamma = parameter(0.1)
    N = parameter(1000)
    I0 = parameter(10)
end

sys = System(sir_stoch, pars; n_particles=100, dt=0.25)
reset!(sys)
result = simulate(sys, 0.0:1.0:100.0)
# result is 3×101×100 (states × times × particles)
```

## Next Steps

- [DSL Reference](@ref) — full syntax guide for `@odin`
- [Simulation](@ref) — system creation, time-stepping, and output
- [Filtering & Likelihood](@ref) — particle filter, unfilter, and data fitting
- [Inference](@ref) — samplers, runners, and MCMC
- [ODE Solvers](@ref) — DP5 and SDIRK4 solver details
- [GPU Acceleration](@ref) — Metal/CUDA/AMDGPU backends
- [Categorical Models](@ref) — compositional model building
