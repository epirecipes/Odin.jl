# Event handling for ODE solvers — discontinuities and callbacks.
# Supports timed, discrete, and continuous (root-finding) events.

"""
    ContinuousEvent{C, A}

Triggered when `condition(u, pars, t)` crosses zero.
Uses bisection on dense output to locate the exact event time.

- `direction`: `:up` (negative→positive), `:down` (positive→negative), `:both`
- `rootfind`: if `true` (default), use Brent's method to find the exact crossing time
"""
struct ContinuousEvent{C, A}
    condition::C     # condition(u, pars, t) → Float64
    affect!::A       # affect!(u, pars, t) — modify state in-place
    direction::Symbol
    rootfind::Bool
end

function ContinuousEvent(condition, affect!; direction::Symbol=:both, rootfind::Bool=true)
    ContinuousEvent(condition, affect!, direction, rootfind)
end

"""
    DiscreteEvent{C, A}

Checked at every accepted step. Triggered when `condition(u, pars, t)` returns `true`.
"""
struct DiscreteEvent{C, A}
    condition::C     # condition(u, pars, t) → Bool
    affect!::A       # affect!(u, pars, t)
end

"""
    TimedEvent{A}

Triggered at specific pre-scheduled times.
"""
struct TimedEvent{A}
    times::Vector{Float64}
    affect!::A       # affect!(u, pars, t)
end

"""
    EventSet

Collection of events for a simulation. Pass to `dust_system_simulate` via the
`events` keyword argument.
"""
struct EventSet
    continuous::Vector{<:ContinuousEvent}
    discrete::Vector{<:DiscreteEvent}
    timed::Vector{<:TimedEvent}
end

function EventSet(;
    continuous::AbstractVector=ContinuousEvent[],
    discrete::AbstractVector=DiscreteEvent[],
    timed::AbstractVector=TimedEvent[],
)
    EventSet(collect(continuous), collect(discrete), collect(timed))
end

"""
    EventRecord

Stores information about a single triggered event.
"""
struct EventRecord
    time::Float64
    kind::Symbol    # :timed, :continuous, :discrete
    index::Int      # index into the corresponding event vector
end

_has_events(::Nothing) = false
_has_events(es::EventSet) = !(isempty(es.continuous) && isempty(es.discrete) && isempty(es.timed))

# ── Root finding via Brent's method ─────────────────────────────
"""
    _brent_root(g, a, b, ga, gb; atol=1e-12, maxiter=100)

Find root of scalar function `g` in `[a, b]` given `ga = g(a)` and `gb = g(b)`
with `sign(ga) ≠ sign(gb)`. Returns the root to within `atol`.
"""
function _brent_root(g, a::Float64, b::Float64, ga::Float64, gb::Float64;
                     atol::Float64=1e-12, maxiter::Int=100)
    # Ensure ga and gb have opposite signs
    if sign(ga) == sign(gb)
        return abs(ga) < abs(gb) ? a : b
    end

    c = a; gc = ga
    d = b - a; e = d

    for _ in 1:maxiter
        if sign(gb) == sign(gc)
            c = a; gc = ga
            d = b - a; e = d
        end
        if abs(gc) < abs(gb)
            a = b; b = c; c = a
            ga = gb; gb = gc; gc = ga
        end

        tol1 = 2.0 * eps(Float64) * abs(b) + 0.5 * atol
        m = 0.5 * (c - b)

        if abs(m) <= tol1 || gb == 0.0
            return b
        end

        if abs(e) >= tol1 && abs(ga) > abs(gb)
            # Try inverse quadratic interpolation
            s = gb / ga
            if a == c
                p = 2.0 * m * s
                q = 1.0 - s
            else
                q_val = ga / gc
                r = gb / gc
                p = s * (2.0 * m * q_val * (q_val - r) - (b - a) * (r - 1.0))
                q = (q_val - 1.0) * (r - 1.0) * (s - 1.0)
            end
            if p > 0.0
                q = -q
            else
                p = -p
            end
            if 2.0 * p < min(3.0 * m * q - abs(tol1 * q), abs(e * q))
                e = d
                d = p / q
            else
                d = m; e = m
            end
        else
            d = m; e = m
        end

        a = b; ga = gb
        if abs(d) > tol1
            b += d
        else
            b += m > 0.0 ? tol1 : -tol1
        end
        gb = g(b)
    end
    return b
end

# ── Dense output evaluation into a vector ───────────────────────
"""
    _dp5_dense_eval_vec!(out, y0, y1, k1, k3, k4, k5, k6, k7, h, theta, n)

Evaluate DP5 dense output at fractional step position `theta` ∈ [0, 1],
writing the interpolated state into vector `out`.
"""
function _dp5_dense_eval_vec!(out::AbstractVector{T},
                              y0::AbstractVector{T}, y1::AbstractVector{T},
                              k1, k3, k4, k5, k6, k7,
                              h::T, theta::T, n::Int) where T
    theta1 = one(T) - theta
    @inbounds for i in 1:n
        dy = y1[i] - y0[i]
        hermite = theta1 * y0[i] + theta * y1[i] +
                  theta * (theta - one(T)) * ((one(T) - 2*theta) * dy +
                  (theta - one(T)) * h * k1[i] + theta * h * k7[i])
        correction = theta^2 * theta1^2 * h *
                     (_DP5_D1 * k1[i] + _DP5_D3 * k3[i] + _DP5_D4 * k4[i] +
                      _DP5_D5 * k5[i] + _DP5_D6 * k6[i] + _DP5_D7 * k7[i])
        out[i] = hermite + correction
    end
    return nothing
end

# ── Check crossing direction ────────────────────────────────────
function _check_crossing(g_start::Float64, g_end::Float64, direction::Symbol)
    if direction === :both
        return sign(g_start) != sign(g_end) && g_start != 0.0
    elseif direction === :up
        return g_start < 0.0 && g_end >= 0.0
    elseif direction === :down
        return g_start > 0.0 && g_end <= 0.0
    end
    return false
end

# ── Collect sorted timed event schedule ─────────────────────────
function _build_timed_schedule(events::EventSet, t0::Float64, tf::Float64)
    schedule = Tuple{Float64, Int}[]
    for (idx, te) in enumerate(events.timed)
        for t in te.times
            if t > t0 && t <= tf
                push!(schedule, (t, idx))
            end
        end
    end
    sort!(schedule; by=first)
    return schedule
end

# ── DP5 solver with event handling ──────────────────────────────
"""
    _dp5_solve_events!(f!, u0, tspan, pars, saveat, w, result, abstol, reltol,
                       max_steps, events) -> Vector{EventRecord}

DP5 solve with event detection. Falls through to `_dp5_solve_core!` when
no events are active. Returns a vector of triggered event records.
"""
function _dp5_solve_events!(f!::F, u0::AbstractVector{T}, tspan::Tuple{T,T}, pars::P,
                            saveat::AbstractVector{T},
                            w::DP5Workspace{T},
                            result::Union{Nothing, Matrix{T}},
                            abstol::T, reltol::T,
                            max_steps::Int,
                            events::EventSet) where {F, T<:AbstractFloat, P}
    t0, tf = tspan
    n = length(u0)
    n_save = length(saveat)

    _ensure_workspace!(w, n)
    u = w.u; u_new = w.u_new; u_prev = w.u_prev
    k1 = w.k1; k2 = w.k2; k3 = w.k3
    k4 = w.k4; k5 = w.k5; k6 = w.k6; k7 = w.k7
    interp_buf = w.interp_buf

    @inbounds for i in 1:n; u[i] = u0[i]; end

    if result !== nothing
        result_u = result
    else
        _ensure_result_matrix!(w, n, n_save)
        result_u = w.result_matrix
    end

    sc = w.saveat_cache
    if length(sc) != n_save
        resize!(sc, n_save)
    end
    copyto!(sc, saveat)

    # Build timed event schedule
    timed_schedule = _build_timed_schedule(events, t0, tf)
    timed_idx = 1

    event_log = EventRecord[]

    # Evaluate initial conditions for continuous events
    n_cont = length(events.continuous)
    g_prev = Vector{Float64}(undef, n_cont)
    for (ci, ce) in enumerate(events.continuous)
        g_prev[ci] = ce.condition(u, pars, t0)
    end

    f!(k1, u, pars, t0)
    h = _dp5_initial_step(f!, u, k1, pars, t0, tf, abstol, reltol, n,
                           w.init_u1, w.init_f1)

    t = t0
    save_idx = 1

    while save_idx <= n_save && saveat[save_idx] <= t0 + eps(t0) * 10
        @inbounds for i in 1:n; result_u[i, save_idx] = u[i]; end
        save_idx += 1
    end

    total_steps = 0

    while t < tf - eps(tf) * 10 && save_idx <= n_save && total_steps < max_steps
        total_steps += 1

        # Limit step to next timed event
        h_max = tf - t
        if timed_idx <= length(timed_schedule)
            h_max = min(h_max, timed_schedule[timed_idx][1] - t)
        end
        h = min(h, h_max)
        if h < eps(t) * 10
            # We are at a timed event time — process it
            te_time, te_idx = timed_schedule[timed_idx]
            events.timed[te_idx].affect!(u, pars, te_time)
            push!(event_log, EventRecord(te_time, :timed, te_idx))
            timed_idx += 1
            # Recompute derivative after state change
            f!(k1, u, pars, t)
            # Recompute continuous event conditions
            for (ci, ce) in enumerate(events.continuous)
                g_prev[ci] = ce.condition(u, pars, t)
            end
            # Recompute initial step size
            h = _dp5_initial_step(f!, u, k1, pars, t, tf, abstol, reltol, n,
                                   w.init_u1, w.init_f1)
            continue
        end

        @inbounds for i in 1:n; u_prev[i] = u[i]; end
        t_start = t

        # ── RK stages (identical to _dp5_solve_core!) ──
        @inbounds for i in 1:n
            u_new[i] = u[i] + h * _DP5_A21 * k1[i]
        end
        f!(k2, u_new, pars, t + _DP5_C2 * h)

        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (_DP5_A31 * k1[i] + _DP5_A32 * k2[i])
        end
        f!(k3, u_new, pars, t + _DP5_C3 * h)

        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (_DP5_A41 * k1[i] + _DP5_A42 * k2[i] + _DP5_A43 * k3[i])
        end
        f!(k4, u_new, pars, t + _DP5_C4 * h)

        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (_DP5_A51 * k1[i] + _DP5_A52 * k2[i] + _DP5_A53 * k3[i] + _DP5_A54 * k4[i])
        end
        f!(k5, u_new, pars, t + _DP5_C5 * h)

        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (_DP5_A61 * k1[i] + _DP5_A62 * k2[i] + _DP5_A63 * k3[i] + _DP5_A64 * k4[i] + _DP5_A65 * k5[i])
        end
        f!(k6, u_new, pars, t + h)

        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (_DP5_A71 * k1[i] + _DP5_A73 * k3[i] + _DP5_A74 * k4[i] + _DP5_A75 * k5[i] + _DP5_A76 * k6[i])
        end
        f!(k7, u_new, pars, t + h)

        # ── Error estimate ──
        err = zero(T)
        @inbounds for i in 1:n
            sc_i = abstol + reltol * max(abs(u[i]), abs(u_new[i]))
            ei = h * (_DP5_E1 * k1[i] + _DP5_E3 * k3[i] + _DP5_E4 * k4[i] +
                      _DP5_E5 * k5[i] + _DP5_E6 * k6[i] + _DP5_E7 * k7[i])
            err += (ei / sc_i)^2
        end
        err = sqrt(err / n)

        if err <= one(T)
            # Step accepted — now check for continuous events
            t_new = t + h

            # Check continuous events for sign changes
            earliest_event_time = t_new + one(T)
            earliest_event_idx = 0

            for (ci, ce) in enumerate(events.continuous)
                g_end = ce.condition(u_new, pars, t_new)

                if _check_crossing(g_prev[ci], g_end, ce.direction)
                    if ce.rootfind
                        # Use Brent's method with dense output
                        g_start_val = g_prev[ci]
                        root_fn = function(τ)
                            θ = (τ - t_start) / h
                            _dp5_dense_eval_vec!(interp_buf, u_prev, u_new,
                                                  k1, k3, k4, k5, k6, k7, h, θ, n)
                            return ce.condition(interp_buf, pars, τ)
                        end
                        t_event = _brent_root(root_fn, t_start, t_new,
                                              g_start_val, g_end; atol=1e-12)
                    else
                        t_event = t_new
                    end
                    if t_event < earliest_event_time
                        earliest_event_time = t_event
                        earliest_event_idx = ci
                    end
                end
                # Update stored condition value (will be overwritten if event fires)
                g_prev[ci] = g_end
            end

            if earliest_event_idx > 0
                # An event was detected — interpolate state to event time
                if abs(earliest_event_time - t_new) < eps(t_new) * 100
                    # Event at step end — use u_new
                    @inbounds for i in 1:n; u[i] = u_new[i]; end
                else
                    θ_event = (earliest_event_time - t_start) / h
                    _dp5_dense_eval_vec!(u, u_prev, u_new,
                                          k1, k3, k4, k5, k6, k7, h, θ_event, n)
                end
                t = earliest_event_time

                # Save any saveat points before the event
                while save_idx <= n_save && saveat[save_idx] <= t + eps(t) * 100
                    ts = saveat[save_idx]
                    if ts < t_start + eps(t_start) * 10
                        save_idx += 1
                        continue
                    end
                    if abs(ts - t) < h * T(1e-12) + eps(t) * 100
                        @inbounds for i in 1:n; result_u[i, save_idx] = u[i]; end
                    else
                        θ_s = (ts - t_start) / h
                        _dp5_dense_eval!(result_u, save_idx, u_prev, u_new,
                                          k1, k3, k4, k5, k6, k7, h, θ_s, n)
                    end
                    save_idx += 1
                end

                # Apply the affect
                ce = events.continuous[earliest_event_idx]
                ce.affect!(u, pars, t)
                push!(event_log, EventRecord(t, :continuous, earliest_event_idx))

                # Recompute derivative and conditions after affect
                f!(k1, u, pars, t)
                for (ci2, ce2) in enumerate(events.continuous)
                    g_prev[ci2] = ce2.condition(u, pars, t)
                end

                # Reset step size after discontinuity
                h = _dp5_initial_step(f!, u, k1, pars, t, tf, abstol, reltol, n,
                                       w.init_u1, w.init_f1)
            else
                # No events — normal step acceptance
                t = t_new

                # Dense output saves
                while save_idx <= n_save && saveat[save_idx] <= t + eps(t) * 100
                    ts = saveat[save_idx]
                    if abs(ts - t) < h * T(1e-12) + eps(t) * 100
                        @inbounds for i in 1:n; result_u[i, save_idx] = u_new[i]; end
                    else
                        theta = (ts - t_start) / h
                        _dp5_dense_eval!(result_u, save_idx, u_prev, u_new,
                                          k1, k3, k4, k5, k6, k7, h, theta, n)
                    end
                    save_idx += 1
                end

                # FSAL swap
                k1, k7 = k7, k1
                w.k1 = k1; w.k7 = k7
                @inbounds for i in 1:n; u[i] = u_new[i]; end

                # Check discrete events at accepted step
                for (di, de) in enumerate(events.discrete)
                    if de.condition(u, pars, t)
                        de.affect!(u, pars, t)
                        push!(event_log, EventRecord(t, :discrete, di))
                        # Recompute derivative after affect
                        f!(k1, u, pars, t)
                        for (ci2, ce2) in enumerate(events.continuous)
                            g_prev[ci2] = ce2.condition(u, pars, t)
                        end
                    end
                end
            end

            # Step size control
            if err == zero(T)
                h *= T(5)
            else
                fac = T(0.9) * err^(T(-0.2))
                h *= clamp(fac, T(0.2), T(5.0))
            end
        else
            # Step rejected — shrink step
            if err == zero(T)
                h *= T(5)
            else
                fac = T(0.9) * err^(T(-0.2))
                h *= clamp(fac, T(0.2), T(5.0))
            end
        end
    end

    # Process remaining timed events at tf
    while timed_idx <= length(timed_schedule)
        te_time, te_idx = timed_schedule[timed_idx]
        if abs(te_time - tf) < eps(tf) * 100
            events.timed[te_idx].affect!(u, pars, te_time)
            push!(event_log, EventRecord(te_time, :timed, te_idx))
        end
        timed_idx += 1
    end

    # Fill remaining save points
    while save_idx <= n_save
        @inbounds for i in 1:n; result_u[i, save_idx] = u[i]; end
        save_idx += 1
    end

    return event_log
end

# ── Public solve with events ────────────────────────────────────
"""
    dp5_solve_events!(f!, u0, tspan, pars, saveat; ws, abstol, reltol, max_steps, events) -> (DP5Result, Vector{EventRecord})

Solve with event handling. Returns the result and a log of triggered events.
"""
function dp5_solve_events!(f!::F, u0::AbstractVector{T}, tspan::Tuple{T,T}, pars,
                           saveat::AbstractVector{T};
                           ws::Union{Nothing, DP5Workspace{T}}=nothing,
                           result::Union{Nothing, Matrix{T}}=nothing,
                           abstol::T=T(1e-6), reltol::T=T(1e-6),
                           max_steps::Int=100000,
                           events::EventSet=EventSet()) where {F, T<:AbstractFloat}
    w = ws === nothing ? DP5Workspace(length(u0), T) : ws
    log = _dp5_solve_events!(f!, u0, tspan, pars, saveat, w, result,
                              abstol, reltol, max_steps, events)
    return DP5Result(w.saveat_cache, w.result_matrix), log
end
