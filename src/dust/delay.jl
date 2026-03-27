# Delay differential equation (DDE) support: history buffer and Hermite interpolation.
#
# Stores past ODE solver steps and provides cubic Hermite interpolation
# to evaluate state variables at past times (t - τ).

"""
    DDEHistoryStep{T}

A single recorded step of the ODE solver, storing enough data for
cubic Hermite interpolation of the solution within [t0, t1].
"""
mutable struct DDEHistoryStep{T}
    t0::T           # time at step start
    t1::T           # time at step end
    h::T            # step size (t1 - t0)
    y0::Vector{T}   # state at t0
    y1::Vector{T}   # state at t1
    k1::Vector{T}   # RHS at t0 (f(t0, y0))
    k7::Vector{T}   # RHS at t1 (f(t1, y1)), from FSAL
end

function DDEHistoryStep(n::Int, ::Type{T}=Float64) where T
    DDEHistoryStep{T}(
        zero(T), zero(T), zero(T),
        zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
    )
end

"""
    DDEHistory{T}

Circular buffer of past ODE solver steps for DDE delay interpolation.
Pre-allocated for a fixed maximum number of steps.
"""
mutable struct DDEHistory{T}
    steps::Vector{DDEHistoryStep{T}}
    initial_state::Vector{T}
    t0::T                   # simulation start time
    max_steps::Int
    head::Int               # write position (1-based)
    count::Int              # number of stored steps
    n::Int                  # number of state variables
end

"""
    DDEHistory(n_state, max_steps; T=Float64)

Create a DDE history buffer for `n_state` state variables,
storing up to `max_steps` past steps.
"""
function DDEHistory(n_state::Int, max_steps::Int=10000; T::Type=Float64)
    steps = [DDEHistoryStep(n_state, T) for _ in 1:max_steps]
    DDEHistory{T}(
        steps,
        zeros(T, n_state),
        zero(T),
        max_steps,
        0,
        0,
        n_state,
    )
end

"""
    dde_history_init!(hist, state0, t0)

Initialize the history buffer with initial state and start time.
"""
function dde_history_init!(hist::DDEHistory{T}, state0::AbstractVector, t0::T) where T
    copyto!(hist.initial_state, state0)
    hist.t0 = t0
    hist.count = 0
    hist.head = 0
    return nothing
end

"""
    dde_history_push!(hist, t0, t1, h, y0, y1, k1, k7)

Record a completed solver step into the history buffer.
Uses PRE-FSAL-swap k1 and k7.
"""
function dde_history_push!(hist::DDEHistory{T}, t0::T, t1::T, h::T,
                            y0::AbstractVector{T}, y1::AbstractVector{T},
                            k1::AbstractVector{T}, k7::AbstractVector{T}) where T
    hist.head = mod1(hist.head + 1, hist.max_steps)
    step = hist.steps[hist.head]
    step.t0 = t0
    step.t1 = t1
    step.h = h
    copyto!(step.y0, y0)
    copyto!(step.y1, y1)
    copyto!(step.k1, k1)
    copyto!(step.k7, k7)
    hist.count = min(hist.count + 1, hist.max_steps)
    return nothing
end

"""
    _dde_find_step(hist, t_query)

Find the history step containing time `t_query`. Returns the step index
or 0 if not found (query is before recorded history).
"""
function _dde_find_step(hist::DDEHistory{T}, t_query::T) where T
    hist.count == 0 && return 0

    # Search backwards from head (most recent step first)
    for k in 0:(hist.count - 1)
        idx = mod1(hist.head - k, hist.max_steps)
        step = hist.steps[idx]
        if t_query >= step.t0 - eps(step.t0) * 100 &&
           t_query <= step.t1 + eps(step.t1) * 100
            return idx
        end
    end
    return 0
end

"""
    dde_history_eval(hist, t_query, var_idx)

Evaluate a single state variable at time `t_query` using cubic Hermite
interpolation from stored solver steps.

Returns the interpolated value. For queries at or before t0, returns
the initial state.
"""
function dde_history_eval(hist::DDEHistory{T}, t_query::T, var_idx::Int) where T
    # Before simulation start: return initial state
    if t_query <= hist.t0 + eps(abs(hist.t0)) * 100
        return hist.initial_state[var_idx]
    end

    # Find the step containing t_query
    step_idx = _dde_find_step(hist, t_query)

    if step_idx == 0
        # Query is before recorded history — return initial state
        return hist.initial_state[var_idx]
    end

    step = hist.steps[step_idx]

    # At step boundaries, return exact values
    if abs(t_query - step.t0) < eps(step.t0) * 1000
        return step.y0[var_idx]
    end
    if abs(t_query - step.t1) < eps(step.t1) * 1000
        return step.y1[var_idx]
    end

    # Cubic Hermite interpolation
    h = step.h
    theta = (t_query - step.t0) / h
    theta1 = one(T) - theta

    i = var_idx
    dy = step.y1[i] - step.y0[i]
    val = theta1 * step.y0[i] + theta * step.y1[i] +
          theta * (theta - one(T)) * (
              (one(T) - 2 * theta) * dy +
              (theta - one(T)) * h * step.k1[i] +
              theta * h * step.k7[i]
          )
    return val
end

"""
    dde_history_eval_vec(hist, t_query, var_indices)

Evaluate multiple state variables at time `t_query`.
Returns a vector of interpolated values.
"""
function dde_history_eval_vec(hist::DDEHistory{T}, t_query::T,
                               var_indices::AbstractVector{Int}) where T
    return T[dde_history_eval(hist, t_query, idx) for idx in var_indices]
end
