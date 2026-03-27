# Lightweight Dormand-Prince 5(4) ODE solver.
# Avoids DifferentialEquations.jl overhead for the inner simulation loop.
# Uses Hairer free 4th-order dense output for efficient saveat interpolation.

"""
    DP5Result{T}

Result of a lightweight DP5 solve.
`u` is a matrix of shape (n_state, n_save) — column `i` holds the state at `t[i]`.
"""
struct DP5Result{T}
    t::Vector{T}
    u::Matrix{T}
end

# ── Butcher tableau (Dormand & Prince, 1980) ─────────────────
const _DP5_A21 = 1/5
const _DP5_A31 = 3/40;       const _DP5_A32 = 9/40
const _DP5_A41 = 44/45;      const _DP5_A42 = -56/15;     const _DP5_A43 = 32/9
const _DP5_A51 = 19372/6561; const _DP5_A52 = -25360/2187; const _DP5_A53 = 64448/6561;  const _DP5_A54 = -212/729
const _DP5_A61 = 9017/3168;  const _DP5_A62 = -355/33;     const _DP5_A63 = 46732/5247;  const _DP5_A64 = 49/176;    const _DP5_A65 = -5103/18656
const _DP5_A71 = 35/384;     const _DP5_A73 = 500/1113;    const _DP5_A74 = 125/192;     const _DP5_A75 = -2187/6784; const _DP5_A76 = 11/84

# Error coefficients: e_i = b_i - b̂_i (5th order - 4th order)
const _DP5_E1 = 71/57600;  const _DP5_E3 = -71/16695; const _DP5_E4 = 71/1920
const _DP5_E5 = -17253/339200; const _DP5_E6 = 22/525; const _DP5_E7 = -1/40

const _DP5_C2 = 1/5; const _DP5_C3 = 3/10; const _DP5_C4 = 4/5; const _DP5_C5 = 8/9

# Dense output correction coefficients (Hairer, Norsett & Wanner, Solving ODEs I, II.6)
const _DP5_D1 = -12715105075.0 / 11282082432.0
const _DP5_D3 =  87487479700.0 / 32700410799.0
const _DP5_D4 = -10690763975.0 / 1880347072.0
const _DP5_D5 = 701980252875.0 / 199316789632.0
const _DP5_D6 =  -1453857185.0 / 822651844.0
const _DP5_D7 =    69997945.0 / 29380423.0

"""
    DP5Workspace{T}

Pre-allocated workspace for DP5 solver. Create once, reuse across solves.
"""
mutable struct DP5Workspace{T}
    u::Vector{T}
    u_new::Vector{T}
    u_prev::Vector{T}     # state at start of step (for dense output)
    k1::Vector{T}; k2::Vector{T}; k3::Vector{T}
    k4::Vector{T}; k5::Vector{T}; k6::Vector{T}; k7::Vector{T}
    interp_buf::Vector{T}  # buffer for dense output
    n::Int
    result_matrix::Matrix{T}  # pre-allocated n_state × n_save output
    init_u1::Vector{T}        # buffer for _dp5_initial_step
    init_f1::Vector{T}        # buffer for _dp5_initial_step
    saveat_cache::Vector{T}   # reusable buffer for saveat times
end

function DP5Workspace(n::Int, ::Type{T}=Float64) where T
    DP5Workspace{T}(
        Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
        Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
        Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
        Vector{T}(undef, n),
        n,
        Matrix{T}(undef, n, 0),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, 0))
end

function _ensure_workspace!(ws::DP5Workspace{T}, n::Int) where T
    ws.n == n && return
    for f in (:u, :u_new, :u_prev, :k1, :k2, :k3, :k4, :k5, :k6, :k7, :interp_buf, :init_u1, :init_f1)
        resize!(getfield(ws, f), n)
    end
    ws.n = n
end

"""Ensure the workspace result matrix has the right shape (n × n_save)."""
function _ensure_result_matrix!(ws::DP5Workspace{T}, n::Int, n_save::Int) where T
    if size(ws.result_matrix, 1) != n || size(ws.result_matrix, 2) != n_save
        ws.result_matrix = Matrix{T}(undef, n, n_save)
    end
end

"""
    dp5_solve!(f!, u0, tspan, pars, saveat; ws, abstol, reltol, max_steps) -> DP5Result

Solve `du/dt = f!(du, u, pars, t)` using adaptive Dormand-Prince 5(4).
Uses Hairer's free 4th-order dense output for saving at arbitrary times.
Pass a `DP5Workspace` to avoid allocations on repeated calls.
"""
function dp5_solve!(f!::F, u0::AbstractVector{T}, tspan::Tuple{T,T}, pars,
                    saveat::AbstractVector{T};
                    ws::Union{Nothing, DP5Workspace{T}}=nothing,
                    result::Union{Nothing, Matrix{T}}=nothing,
                    abstol::T=T(1e-6), reltol::T=T(1e-6),
                    max_steps::Int=100000) where {F, T<:AbstractFloat}
    if ws === nothing
        w = DP5Workspace(length(u0), T)
    else
        w = ws
    end
    _dp5_solve_core!(f!, u0, tspan, pars, saveat, w, result, abstol, reltol, max_steps)
    return DP5Result(w.saveat_cache, w.result_matrix)
end

"""
Zero-allocation core of the DP5 solver. All results written into workspace `w`.
After return, `w.result_matrix` holds the solution and `w.saveat_cache` holds the times.
Callers in hot paths should use this directly and read from `w` to avoid struct allocation.
"""
function _dp5_solve_core!(f!::F, u0::AbstractVector{T}, tspan::Tuple{T,T}, pars::P,
                          saveat::AbstractVector{T},
                          w::DP5Workspace{T},
                          result::Union{Nothing, Matrix{T}},
                          abstol::T, reltol::T,
                          max_steps::Int) where {F, T<:AbstractFloat, P}
    t0, tf = tspan
    n = length(u0)
    n_save = length(saveat)

    _ensure_workspace!(w, n)
    u = w.u; u_new = w.u_new; u_prev = w.u_prev
    k1 = w.k1; k2 = w.k2; k3 = w.k3
    k4 = w.k4; k5 = w.k5; k6 = w.k6; k7 = w.k7

    @inbounds for i in 1:n; u[i] = u0[i]; end

    # Pre-allocate or reuse result matrix (n × n_save)
    if result !== nothing
        result_u = result
    else
        _ensure_result_matrix!(w, n, n_save)
        result_u = w.result_matrix
    end

    # Copy saveat into workspace cache (no-op when length matches)
    sc = w.saveat_cache
    if length(sc) != n_save
        resize!(sc, n_save)
    end
    copyto!(sc, saveat)

    # Evaluate initial derivative (FSAL: k1 = f(t0, u0))
    f!(k1, u, pars, t0)
    h = _dp5_initial_step(f!, u, k1, pars, t0, tf, abstol, reltol, n,
                           w.init_u1, w.init_f1)

    t = t0
    save_idx = 1

    # Save initial point(s)
    while save_idx <= n_save && saveat[save_idx] <= t0 + eps(t0) * 10
        @inbounds for i in 1:n; result_u[i, save_idx] = u[i]; end
        save_idx += 1
    end

    total_steps = 0

    while t < tf - eps(tf) * 10 && save_idx <= n_save && total_steps < max_steps
        total_steps += 1
        h = min(h, tf - t)

        # Save state at step start for dense output
        @inbounds for i in 1:n; u_prev[i] = u[i]; end
        t_start = t

        # ── RK stages ──
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

        # 5th order solution
        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (_DP5_A71 * k1[i] + _DP5_A73 * k3[i] + _DP5_A74 * k4[i] + _DP5_A75 * k5[i] + _DP5_A76 * k6[i])
        end

        # k7 = f(t + h, u_new) — needed for error estimate and FSAL
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
            # Step accepted
            t += h

            # Dense output: save any saveat points in (t_start, t]
            # Must be done BEFORE FSAL swap (k1 = f_n, k7 = f_{n+1})
            while save_idx <= n_save && saveat[save_idx] <= t + eps(t) * 100
                ts = saveat[save_idx]
                if abs(ts - t) < h * T(1e-12) + eps(t) * 100
                    # At step end — use u_new directly
                    @inbounds for i in 1:n; result_u[i, save_idx] = u_new[i]; end
                else
                    # Interpolate within step using Hairer's free 4th-order formula
                    theta = (ts - t_start) / h
                    _dp5_dense_eval!(result_u, save_idx, u_prev, u_new, k1, k3, k4, k5, k6, k7, h, theta, n)
                end
                save_idx += 1
            end

            # FSAL: k7 becomes k1 for next step
            k1, k7 = k7, k1
            w.k1 = k1; w.k7 = k7
            @inbounds for i in 1:n; u[i] = u_new[i]; end
        end

        # Step size control (standard PI controller)
        if err == zero(T)
            h *= T(5)
        else
            fac = T(0.9) * err^(T(-0.2))
            h *= clamp(fac, T(0.2), T(5.0))
        end
    end

    # Fill remaining save points
    while save_idx <= n_save
        @inbounds for i in 1:n; result_u[i, save_idx] = u[i]; end
        save_idx += 1
    end

    return nothing
end

"""
Hairer's free 4th-order dense output for DP5 (in-place).
Writes interpolated state into column `col` of matrix `out`.

Arguments use PRE-FSAL-swap stages: k1 = f(t_n, y_n), k7 = f(t_{n+1}, y_{n+1}).
"""
function _dp5_dense_eval!(out::AbstractMatrix{T}, col::Int,
                           y0::AbstractVector{T}, y1::AbstractVector{T},
                           k1, k3, k4, k5, k6, k7,
                           h::T, theta::T, n::Int) where T
    theta1 = one(T) - theta
    @inbounds for i in 1:n
        # Hermite cubic
        dy = y1[i] - y0[i]
        hermite = theta1 * y0[i] + theta * y1[i] +
                  theta * (theta - one(T)) * ((one(T) - 2*theta) * dy + (theta - one(T)) * h * k1[i] + theta * h * k7[i])

        # 4th order correction
        correction = theta^2 * theta1^2 * h *
                     (_DP5_D1 * k1[i] + _DP5_D3 * k3[i] + _DP5_D4 * k4[i] +
                      _DP5_D5 * k5[i] + _DP5_D6 * k6[i] + _DP5_D7 * k7[i])

        out[i, col] = hermite + correction
    end
    return nothing
end

"""Estimate initial step size (Hairer, Norsett & Wanner, Solving ODEs I, II.4)."""
function _dp5_initial_step(f!::F, u0::AbstractVector{T}, f0::AbstractVector{T},
                           pars::P, t0::T, tf::T,
                           abstol::T, reltol::T, n::Int,
                           u1_buf::Vector{T}, f1_buf::Vector{T}) where {F, T, P}
    d0 = zero(T); d1 = zero(T)
    @inbounds for i in 1:n
        sc = abstol + reltol * abs(u0[i])
        d0 += (u0[i] / sc)^2
        d1 += (f0[i] / sc)^2
    end
    d0 = sqrt(d0 / n); d1 = sqrt(d1 / n)

    h0 = (d0 < T(1e-5) || d1 < T(1e-5)) ? T(1e-6) : T(0.01) * d0 / d1
    h0 = min(h0, tf - t0)

    @inbounds for i in 1:n; u1_buf[i] = u0[i] + h0 * f0[i]; end
    f!(f1_buf, u1_buf, pars, t0 + h0)

    d2 = zero(T)
    @inbounds for i in 1:n
        sc = abstol + reltol * abs(u0[i])
        d2 += ((f1_buf[i] - f0[i]) / sc)^2
    end
    d2 = sqrt(d2 / n) / h0

    h1 = max(d1, d2) <= T(1e-15) ? max(T(1e-6), h0 * T(1e-3)) : (T(0.01) / max(d1, d2))^(T(1)/T(5))
    return min(T(100) * h0, h1, tf - t0)
end

# Keyword-argument convenience overload (for external callers)
function _dp5_initial_step(f!::F, u0::AbstractVector{T}, f0::AbstractVector{T},
                           pars::P, t0::T, tf::T,
                           abstol::T, reltol::T, n::Int;
                           u1_buf::Union{Nothing, Vector{T}}=nothing,
                           f1_buf::Union{Nothing, Vector{T}}=nothing) where {F, T, P}
    u1 = u1_buf !== nothing ? u1_buf : similar(u0, T, n)
    f1 = f1_buf !== nothing ? f1_buf : similar(u0, T, n)
    return _dp5_initial_step(f!, u0, f0, pars, t0, tf, abstol, reltol, n, u1, f1)
end


# ── DDE variant of DP5 solver ──────────────────────────────────
# Records each step into DDEHistory and caps step size at h_max.

function _dp5_solve_dde_core!(
    f!::F, u0::AbstractVector{T}, tspan::Tuple{T,T}, pars::P,
    t_out::AbstractVector{T}, ws::DP5Workspace{T},
    hist::DDEHistory{T}, h_max::T,
    mass::Nothing, abstol::T, reltol::T, maxsteps::Int,
) where {F, T, P}
    n = length(u0)
    t0, tf = tspan
    nt_out = length(t_out)

    _ensure_workspace!(ws, n)
    _ensure_result_matrix!(ws, n, nt_out)

    u  = ws.u;  copyto!(u, u0)
    u_new = ws.u_new; u_prev = ws.u_prev
    k1 = ws.k1; k2 = ws.k2; k3 = ws.k3; k4 = ws.k4
    k5 = ws.k5; k6 = ws.k6; k7 = ws.k7

    f!(k1, u, pars, t0)
    dde_history_push!(hist, u, k1, t0)

    h = _dp5_initial_step(f!, u, k1, pars, t0, tf, abstol, reltol, n,
                          ws.init_u1, ws.init_f1)
    h = min(h, h_max)

    t = t0
    out_idx = 1
    step_count = 0

    a21 = T(1)/T(5)
    a31 = T(3)/T(40);    a32 = T(9)/T(40)
    a41 = T(44)/T(45);   a42 = T(-56)/T(15);   a43 = T(32)/T(9)
    a51 = T(19372)/T(6561); a52 = T(-25360)/T(2187)
    a53 = T(64448)/T(6561); a54 = T(-212)/T(729)
    a61 = T(9017)/T(3168);  a62 = T(-355)/T(33)
    a63 = T(46732)/T(5247); a64 = T(49)/T(176); a65 = T(-5103)/T(18656)
    b1  = T(35)/T(384);  b3  = T(500)/T(1113)
    b4  = T(125)/T(192);  b5 = T(-2187)/T(6784); b6 = T(11)/T(84)
    e1  = T(71)/T(57600); e3 = T(-71)/T(16695)
    e4  = T(71)/T(1920);  e5 = T(-17253)/T(339200); e6 = T(22)/T(525); e7 = T(-1)/T(40)

    while t < tf && step_count < maxsteps
        step_count += 1
        h = min(h, tf - t)
        h = min(h, h_max)

        # Save state for potential rollback
        @inbounds for i in 1:n; u_prev[i] = u[i]; end

        @inbounds for i in 1:n
            u_new[i] = u[i] + h * a21 * k1[i]
        end
        f!(k2, u_new, pars, t + T(1)/T(5)*h)

        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (a31*k1[i] + a32*k2[i])
        end
        f!(k3, u_new, pars, t + T(3)/T(10)*h)

        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (a41*k1[i] + a42*k2[i] + a43*k3[i])
        end
        f!(k4, u_new, pars, t + T(4)/T(5)*h)

        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (a51*k1[i] + a52*k2[i] + a53*k3[i] + a54*k4[i])
        end
        f!(k5, u_new, pars, t + T(8)/T(9)*h)

        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (a61*k1[i] + a62*k2[i] + a63*k3[i] + a64*k4[i] + a65*k5[i])
        end
        f!(k6, u_new, pars, t + h)

        @inbounds for i in 1:n
            u[i] = u[i] + h * (b1*k1[i] + b3*k3[i] + b4*k4[i] + b5*k5[i] + b6*k6[i])
        end

        t_new = t + h
        f!(k7, u, pars, t_new)

        err = T(0)
        @inbounds for i in 1:n
            sc = abstol + reltol * max(abs(u[i]), abs(u[i] - h*(b1*k1[i]+b3*k3[i]+b4*k4[i]+b5*k5[i]+b6*k6[i])))
            ei = h * (e1*k1[i] + e3*k3[i] + e4*k4[i] + e5*k5[i] + e6*k6[i] + e7*k7[i])
            err += (ei / sc)^2
        end
        err = sqrt(err / n)

        if err <= T(1)
            t = t_new
            dde_history_push!(hist, u, k7, t)
            copyto!(k1, k7)

            while out_idx <= nt_out && t_out[out_idx] <= t + 10*eps(t)
                if abs(t_out[out_idx] - t) < 10*eps(t)
                    @inbounds for j in 1:n
                        ws.result_matrix[j, out_idx] = u[j]
                    end
                else
                    t_target = t_out[out_idx]
                    @inbounds for j in 1:n
                        ws.result_matrix[j, out_idx] = dde_history_eval(hist, t_target, j)
                    end
                end
                out_idx += 1
            end

            factor = err > T(0) ? T(0.9) * err^(T(-1)/T(5)) : T(5)
            factor = clamp(factor, T(0.2), T(5))
            h = min(h * factor, h_max)
        else
            factor = T(0.9) * err^(T(-1)/T(5))
            factor = max(factor, T(0.2))
            h *= factor
            copyto!(u, u_prev)
        end
    end

    while out_idx <= nt_out
        @inbounds for j in 1:n
            ws.result_matrix[j, out_idx] = u[j]
        end
        out_idx += 1
    end

    return nothing
end
