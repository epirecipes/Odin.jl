# ODE solver control for continuous-time models.

"""
    DustODEControl

Configuration for the ODE solver used in continuous-time models and unfilters.

# Fields
- `atol::Float64` — absolute tolerance (default 1e-6)
- `rtol::Float64` — relative tolerance (default 1e-6)
- `max_steps::Int` — maximum integration steps (default 10000)
- `solver::Symbol` — `:dp5` (Dormand-Prince 5th order, default) or
  `:sdirk` (SDIRK4 L-stable 4th order, for stiff systems)
"""
struct DustODEControl
    atol::Float64
    rtol::Float64
    max_steps::Int
    solver::Symbol
end

DustODEControl(; atol=1e-6, rtol=1e-6, max_steps=10000, solver=:dp5) =
    DustODEControl(atol, rtol, max_steps, solver)
