# ODE Solvers

Odin.jl includes two lightweight ODE solvers designed for high-performance
inner loops in simulation and likelihood evaluation. They avoid the overhead of
DifferentialEquations.jl while providing adaptive step-size control and dense
output.

## Solver Selection

Choose the solver via [`ODEControl`](@ref):

```julia
# Non-stiff (default)
ctrl = ODEControl(solver=:dp5)

# Stiff systems
ctrl = ODEControl(solver=:sdirk)
```

| Solver | Symbol | Order | Stability | Best for |
|--------|--------|-------|-----------|----------|
| Dormand-Prince 5(4) | `:dp5` | 5th | Explicit | Non-stiff systems |
| SDIRK4 (Cash 1979) | `:sdirk` | 4th | L-stable | Stiff systems |

## DP5 — Dormand-Prince 5(4)

An explicit Runge-Kutta method with 7 stages and an embedded 4th-order error
estimator. Features:

- **Adaptive step-size control** using the standard embedded error formula
- **Dense output** via Hairer's free 4th-order interpolation for efficient `saveat`
- **Pre-allocated workspaces** — zero allocations in the inner loop after warmup
- **FSAL (First Same As Last)** — the 7th stage evaluation is reused as the first stage of the next step

The Butcher tableau follows Dormand & Prince (1980). The dense output
coefficients follow Hairer, Norsett & Wanner (*Solving ODEs I*, §II.6).

### When to use DP5

- Standard compartmental models (SIR, SEIR, etc.)
- Models without extremely different timescales
- When you need maximum throughput for non-stiff systems

### Example: Comparing DP5 with DifferentialEquations.jl

```julia
using Odin

sir = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10)
    N = parameter(1000)
end

pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
times = collect(0.0:1.0:365.0)

# DP5 (default — used automatically)
sys = System(sir, pars; ode_control=ODEControl(solver=:dp5, atol=1e-6, rtol=1e-6))
reset!(sys)
out_dp5 = simulate(sys, times)

# DifferentialEquations.jl fallback (used automatically when AD is needed)
sys2 = System(sir, pars; ode_control=ODEControl(solver=:diffeq, atol=1e-6, rtol=1e-6))
reset!(sys2)
out_diffeq = simulate(sys2, times)

# Results should agree to solver tolerance
maximum(abs.(out_dp5 .- out_diffeq))  # < 1e-4
```

## SDIRK4 — Singly Diagonally Implicit Runge-Kutta

A 5-stage, 4th-order, L-stable implicit method using the Cash (1979) Butcher
tableau (also given in Hairer & Wanner, *Solving ODEs II*, Table 6.2).

Features:

- **L-stability** — no spurious oscillations for very stiff problems
- **Adaptive step-size control** using embedded 3rd-order error estimator
- **Newton iteration** for the implicit stage equations (tolerance `0.01`,
  max 10 iterations)
- **Jacobian caching** — reuses LU factorisation for up to 20 steps, with
  automatic recomputation when convergence degrades
- **ForwardDiff compatible** — the Newton solver works with dual numbers for
  automatic differentiation through the ODE solve

### When to use SDIRK

- Models with widely separated timescales (e.g., fast immune dynamics + slow
  epidemiological dynamics)
- Systems where DP5 requires extremely small step sizes
- When using the unfilter with gradient-based samplers on stiff models

### Example: Stiff system

```julia
# Model with fast and slow dynamics
stiff_model = @odin begin
    deriv(x) = -1000 * (x - cos(time))
    deriv(y) = x - y
    initial(x) = 1.0
    initial(y) = 0.0
end

pars = NamedTuple()
ctrl = ODEControl(solver=:sdirk, atol=1e-8, rtol=1e-8, max_steps=50000)
sys = System(stiff_model, pars; ode_control=ctrl)
reset!(sys)
out = simulate(sys, collect(0.0:0.01:1.0))
```

## Workspace Reuse

Both solvers use pre-allocated workspace objects to avoid repeated heap
allocations in tight loops (e.g., inside a particle filter):

```julia
ws = SDIRKWorkspace(n_state)
result = sdirk_solve!(f!, u0, tspan, ws; atol=1e-6, rtol=1e-6)
```

The workspaces are created automatically by [`System`](@ref) and
cached on the [`DustSystem`](@ref) for reuse across time steps.

## Performance Tips

1. **Use DP5 for non-stiff models** — it's 2–10× faster than SDIRK and the
   DifferentialEquations.jl fallback for typical epidemiological models.

2. **Tighten tolerances for gradients** — when computing gradients via
   ForwardDiff (e.g., for HMC/NUTS), use `atol=1e-8, rtol=1e-8` to avoid
   noisy gradient estimates.

3. **Increase `max_steps` for long simulations** — the default 10,000 steps
   may not be sufficient for multi-year simulations with fine structure.

4. **Watch for step-size warnings** — if the solver hits `max_steps`, the
   solution may be inaccurate. This often indicates a stiff system that
   should use `:sdirk`.

## Configuration

Solver behaviour is controlled via [`ODEControl`](@ref):

```julia
ctrl = ODEControl(;
    atol = 1e-6,          # absolute tolerance
    rtol = 1e-6,          # relative tolerance
    max_steps = 10000,    # maximum steps per integration interval
    solver = :dp5,        # :dp5 or :sdirk
)
```

Pass this to [`System`](@ref) or [`Likelihood`](@ref):

```julia
sys = System(gen, pars; ode_control=ctrl)
uf  = Likelihood(gen, data; time_start=0.0, ode_control=ctrl)
```

## API Reference

```@docs
Odin.ODEControl
Odin.sdirk_solve!
Odin.SDIRKWorkspace
Odin.SDIRKResult
```
