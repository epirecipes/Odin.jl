# ODE Solvers

Odin.jl includes two lightweight ODE solvers designed for high-performance
inner loops in simulation and likelihood evaluation. They avoid the overhead of
DifferentialEquations.jl while providing adaptive step-size control and dense
output.

## Solver Selection

Choose the solver via [`DustODEControl`](@ref):

```julia
# Non-stiff (default)
ctrl = DustODEControl(solver=:dp5)

# Stiff systems
ctrl = DustODEControl(solver=:sdirk)
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

The Butcher tableau follows Dormand & Prince (1980). The dense output
coefficients follow Hairer, Norsett & Wanner (*Solving ODEs I*, §II.6).

### When to use DP5

- Standard compartmental models (SIR, SEIR, etc.)
- Models without extremely different timescales
- When you need maximum throughput for non-stiff systems

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

## Workspace Reuse

Both solvers use pre-allocated workspace objects to avoid repeated heap
allocations in tight loops (e.g., inside a particle filter):

```julia
ws = SDIRKWorkspace(n_state)
result = sdirk_solve!(f!, u0, tspan, ws; atol=1e-6, rtol=1e-6)
```

The workspaces are created automatically by [`dust_system_create`](@ref) and
cached on the [`DustSystem`](@ref) for reuse across time steps.

## Configuration

Solver behaviour is controlled via [`DustODEControl`](@ref):

```julia
ctrl = DustODEControl(;
    atol = 1e-6,          # absolute tolerance
    rtol = 1e-6,          # relative tolerance
    max_steps = 10000,    # maximum steps per integration interval
    solver = :dp5,        # :dp5 or :sdirk
)
```

Pass this to [`dust_system_create`](@ref) or [`dust_unfilter_create`](@ref):

```julia
sys = dust_system_create(gen, pars; ode_control=ctrl)
uf  = dust_unfilter_create(gen; data=data, time_start=0.0, ode_control=ctrl)
```

## API Reference

```@docs
Odin.DustODEControl
Odin.sdirk_solve!
Odin.SDIRKWorkspace
Odin.SDIRKResult
```
