# SDIRK4 — 5-stage, L-stable, 4th-order SDIRK solver.
# Cash (1979) / Hairer-Wanner (Solving ODEs II, Table 6.2) Butcher tableau.
# Suitable for stiff ODE systems with widely separated timescales.
# Uses the same interface as dp5_solve! for consistency.

"""
    SDIRKResult{T}

Result of SDIRK solve.
`u` is a matrix of shape (n_state, n_save) — column `i` holds the state at `t[i]`.
"""
struct SDIRKResult{T}
    t::Vector{T}
    u::Matrix{T}
end

# ── SDIRK4 Butcher tableau (Cash 1979 / Hairer-Wanner) ────────
# 5-stage, L-stable, 4th order with embedded 3rd order.
# γ = 1/4  (diagonal element)
#
#   1/4    | 1/4
#   3/4    | 1/2         1/4
#   11/20  | 17/50      -1/25        1/4
#   1/2    | 371/1360   -137/2720    15/544     1/4
#   1      | 25/24      -49/48       125/16    -85/12   1/4
#   -------+---------------------------------------------------
#   b (4th)| 25/24      -49/48       125/16    -85/12   1/4
#   b̂ (3rd)| 59/48      -17/96       225/32    -85/12   0

const _SDIRK_GAMMA = 0.25

const _SDIRK_C1 = 0.25
const _SDIRK_C2 = 0.75
const _SDIRK_C3 = 0.55        # 11/20
const _SDIRK_C4 = 0.5
const _SDIRK_C5 = 1.0

# Below-diagonal entries of A (a_ij for j < i)
const _SDIRK_A21 = 0.5
const _SDIRK_A31 = 17.0 / 50.0
const _SDIRK_A32 = -1.0 / 25.0
const _SDIRK_A41 = 371.0 / 1360.0
const _SDIRK_A42 = -137.0 / 2720.0
const _SDIRK_A43 = 15.0 / 544.0
const _SDIRK_A51 = 25.0 / 24.0
const _SDIRK_A52 = -49.0 / 48.0
const _SDIRK_A53 = 125.0 / 16.0
const _SDIRK_A54 = -85.0 / 12.0

# b weights (4th order, stiffly accurate = last row of A)
const _SDIRK_B1 = 25.0 / 24.0
const _SDIRK_B2 = -49.0 / 48.0
const _SDIRK_B3 = 125.0 / 16.0
const _SDIRK_B4 = -85.0 / 12.0
const _SDIRK_B5 = 0.25

# Error coefficients: e_i = b_i - b̂_i (4th order − 3rd order embedded)
const _SDIRK_E1 = 25.0/24.0 - 59.0/48.0    # -3/16
const _SDIRK_E2 = -49.0/48.0 + 17.0/96.0   # -27/32
const _SDIRK_E3 = 125.0/16.0 - 225.0/32.0  #  25/32
const _SDIRK_E4 = 0.0                        #  0
const _SDIRK_E5 = 0.25                       #  1/4

# Newton iteration parameters
const _SDIRK_NEWTON_TOL      = 0.01
const _SDIRK_MAX_NEWTON_ITER = 10
const _SDIRK_MAX_JAC_AGE     = 20

"""
    SDIRKWorkspace{T}

Pre-allocated workspace for SDIRK4 solver. Create once, reuse across solves.
"""
mutable struct SDIRKWorkspace{T}
    u::Vector{T}
    u_new::Vector{T}
    k1::Vector{T}
    k2::Vector{T}
    k3::Vector{T}
    k4::Vector{T}
    k5::Vector{T}
    residual::Vector{T}
    delta::Vector{T}
    jac::Matrix{T}           # n × n Jacobian
    jac_lu::Any               # LU factorization cache
    f_tmp::Vector{T}          # temporary for Jacobian estimation
    stage_rhs::Vector{T}      # right-hand side for implicit stage equation
    stage_val::Vector{T}      # current stage value (Newton iterate)
    n::Int
    jac_current::Bool         # is Jacobian up to date?
    jac_age::Int              # steps since last Jacobian recomputation
    result_matrix::Matrix{T}  # pre-allocated n_state × n_save output
    result_ncols::Int         # current allocated column count
    saveat_cache::Vector{T}   # reusable buffer for saveat times
end

function SDIRKWorkspace(n::Int, ::Type{T}=Float64) where T
    SDIRKWorkspace{T}(
        Vector{T}(undef, n),      # u
        Vector{T}(undef, n),      # u_new
        Vector{T}(undef, n),      # k1
        Vector{T}(undef, n),      # k2
        Vector{T}(undef, n),      # k3
        Vector{T}(undef, n),      # k4
        Vector{T}(undef, n),      # k5
        Vector{T}(undef, n),      # residual
        Vector{T}(undef, n),      # delta
        Matrix{T}(undef, n, n),   # jac
        nothing,                   # jac_lu
        Vector{T}(undef, n),      # f_tmp
        Vector{T}(undef, n),      # stage_rhs
        Vector{T}(undef, n),      # stage_val
        n,
        false,                     # jac_current
        0,                         # jac_age
        Matrix{T}(undef, n, 0),   # result_matrix
        0,                         # result_ncols
        Vector{T}(undef, 0))      # saveat_cache
end

function _ensure_workspace!(ws::SDIRKWorkspace{T}, n::Int) where T
    ws.n == n && return
    for f in (:u, :u_new, :k1, :k2, :k3, :k4, :k5,
              :residual, :delta, :f_tmp, :stage_rhs, :stage_val)
        resize!(getfield(ws, f), n)
    end
    ws.jac = Matrix{T}(undef, n, n)
    ws.jac_lu = nothing
    ws.jac_current = false
    ws.jac_age = 0
    ws.n = n
    ws.result_ncols = 0
end

"""Ensure the workspace result matrix has the right shape (n × n_save)."""
function _ensure_result_matrix!(ws::SDIRKWorkspace{T}, n::Int, n_save::Int) where T
    if size(ws.result_matrix, 1) != n || ws.result_ncols < n_save
        ws.result_matrix = Matrix{T}(undef, n, n_save)
        ws.result_ncols = n_save
    end
end

# ── Jacobian estimation and factorization ─────────────────────

"""
Compute Jacobian J[i,j] = ∂fᵢ/∂uⱼ via forward finite differences.
`f0` must contain `f!(_, u, pars, t)` on entry.
"""
function _sdirk_compute_jacobian_fd!(jac::Matrix{T}, f!::F, f0::AbstractVector{T},
                                     u::AbstractVector{T}, f_tmp::Vector{T},
                                     pars, t::T, n::Int) where {F, T}
    eps_jac = sqrt(eps(T))
    @inbounds for j in 1:n
        uj_save = u[j]
        delta_j = eps_jac * max(abs(uj_save), one(T))
        u[j] = uj_save + delta_j
        f!(f_tmp, u, pars, t)
        inv_delta = one(T) / delta_j
        for i in 1:n
            jac[i, j] = (f_tmp[i] - f0[i]) * inv_delta
        end
        u[j] = uj_save
    end
end

"""
Compute Jacobian using ForwardDiff automatic differentiation.
More accurate than finite differences, especially for ill-conditioned systems.
"""
function _sdirk_compute_jacobian_ad!(jac::Matrix{T}, f!::F,
                                     u::AbstractVector{T},
                                     pars, t::T, n::Int) where {F, T}
    f_oop = x -> begin
        dx = Vector{eltype(x)}(undef, n)
        f!(dx, x, pars, t)
        return dx
    end
    ForwardDiff.jacobian!(jac, f_oop, u)
end

"""
Compute Jacobian, dispatching on the method: `:finite_diff` (default),
`:autodiff` (ForwardDiff), or a user-provided function.
"""
function _sdirk_compute_jacobian!(jac::Matrix{T}, f!::F, f0::AbstractVector{T},
                                  u::AbstractVector{T}, f_tmp::Vector{T},
                                  pars, t::T, n::Int;
                                  jac_fn::Union{Nothing, Function}=nothing,
                                  autodiff::Bool=false) where {F, T}
    if jac_fn !== nothing
        jac_fn(jac, u, pars, t)
    elseif autodiff
        _sdirk_compute_jacobian_ad!(jac, f!, u, pars, t, n)
    else
        _sdirk_compute_jacobian_fd!(jac, f!, f0, u, f_tmp, pars, t, n)
    end
end

"""Form and factorize the iteration matrix W = I − h·γ·J."""
function _sdirk_factorize!(ws::SDIRKWorkspace{T}, h::T) where T
    n = ws.n
    jac = ws.jac
    hg = h * T(_SDIRK_GAMMA)
    W = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n
        for i in 1:n
            W[i, j] = -hg * jac[i, j]
        end
        W[j, j] += one(T)
    end
    ws.jac_lu = lu(W)
end

# ── Newton solver for implicit stages ─────────────────────────

"""
Solve one implicit stage: find Y such that  Y − h·γ·f(t_stage, Y) = rhs.
Uses simplified Newton with a pre-factorized (I − h·γ·J).
Returns `true` if Newton converged, `false` otherwise.
"""
function _sdirk_solve_stage!(f!::F, ws::SDIRKWorkspace{T}, pars, t_stage::T, h::T,
                              rhs::AbstractVector{T}, Y::Vector{T},
                              abstol::T, reltol::T) where {F, T}
    n = ws.n
    hg = h * T(_SDIRK_GAMMA)
    lu_fact = ws.jac_lu
    residual = ws.residual
    delta = ws.delta
    f_tmp = ws.f_tmp

    # Initial guess: Y = rhs  (i.e., stage derivative k = 0)
    @inbounds for i in 1:n; Y[i] = rhs[i]; end

    for _iter in 1:_SDIRK_MAX_NEWTON_ITER
        # Residual: G(Y) = Y − h·γ·f(t_stage, Y) − rhs
        f!(f_tmp, Y, pars, t_stage)
        @inbounds for i in 1:n
            residual[i] = Y[i] - hg * f_tmp[i] - rhs[i]
        end

        # Convergence check (scaled norm)
        res_norm = zero(T)
        @inbounds for i in 1:n
            sc = abstol + reltol * abs(Y[i])
            res_norm += (residual[i] / sc)^2
        end
        res_norm = sqrt(res_norm / n)

        if res_norm <= T(_SDIRK_NEWTON_TOL)
            return true
        end

        # Solve  W · Δ = −G(Y)  where  W = I − h·γ·J
        @inbounds for i in 1:n; delta[i] = -residual[i]; end
        ldiv!(lu_fact, delta)

        # Update: Y ← Y + Δ
        @inbounds for i in 1:n; Y[i] += delta[i]; end
    end

    return false
end

# ── Main solver ───────────────────────────────────────────────

"""
    sdirk_solve!(f!, u0, tspan, pars, saveat; ws, abstol, reltol, max_steps,
                 jac_fn, autodiff) -> SDIRKResult

Solve `du/dt = f!(du, u, pars, t)` using adaptive SDIRK4 (L-stable, 4th order).
5-stage method (Cash 1979 / Hairer-Wanner). Suitable for stiff ODE systems.
Uses simplified Newton iteration with configurable Jacobian:
- Default: finite-difference Jacobian
- `autodiff=true`: ForwardDiff automatic differentiation
- `jac_fn(J, u, pars, t)`: user-provided Jacobian function

Pass a `SDIRKWorkspace` to avoid allocations on repeated calls.
"""
function sdirk_solve!(f!::F, u0::AbstractVector{T}, tspan::Tuple{T,T}, pars,
                      saveat::AbstractVector{T};
                      ws::Union{Nothing, SDIRKWorkspace{T}}=nothing,
                      result::Union{Nothing, Matrix{T}}=nothing,
                      abstol::T=T(1e-6), reltol::T=T(1e-6),
                      max_steps::Int=100000,
                      jac_fn::Union{Nothing, Function}=nothing,
                      autodiff::Bool=false) where {F, T<:AbstractFloat}
    if ws === nothing
        w = SDIRKWorkspace(length(u0), T)
    else
        w = ws
    end
    _sdirk_solve_core!(f!, u0, tspan, pars, saveat, w, result,
                        abstol, reltol, max_steps, jac_fn, autodiff)
    ns = length(w.saveat_cache)
    return SDIRKResult(copy(w.saveat_cache), w.result_matrix[:, 1:ns])
end

"""
Zero-allocation core of the SDIRK4 solver. All results written into workspace `w`.
After return, `w.result_matrix` holds the solution and `w.saveat_cache` holds the times.
Callers in hot paths should use this directly and read from `w` to avoid struct allocation.
"""
function _sdirk_solve_core!(f!::F, u0::AbstractVector{T}, tspan::Tuple{T,T}, pars::P,
                            saveat::AbstractVector{T},
                            w::SDIRKWorkspace{T},
                            result::Union{Nothing, Matrix{T}},
                            abstol::T, reltol::T,
                            max_steps::Int,
                            jac_fn::Union{Nothing, Function}=nothing,
                            autodiff::Bool=false) where {F, T<:AbstractFloat, P}
    t0, tf = tspan
    n = length(u0)
    n_save = length(saveat)

    _ensure_workspace!(w, n)
    u = w.u; u_new = w.u_new
    k1 = w.k1; k2 = w.k2; k3 = w.k3; k4 = w.k4; k5 = w.k5
    stage_rhs = w.stage_rhs; stage_val = w.stage_val

    @inbounds for i in 1:n; u[i] = u0[i]; end

    # Pre-allocate or reuse result matrix (n × n_save)
    if result !== nothing
        result_u = result
    else
        _ensure_result_matrix!(w, n, n_save)
        result_u = w.result_matrix
    end

    # Copy saveat into workspace cache
    sc = w.saveat_cache
    if length(sc) != n_save
        resize!(sc, n_save)
    end
    copyto!(sc, saveat)

    # Evaluate initial derivative for step size estimation and Jacobian
    # f_eval must be separate from w.f_tmp (used in Newton iteration / Jacobian FD)
    f_eval = Vector{T}(undef, n)
    f!(f_eval, u, pars, t0)

    # Estimate initial step size (order p = 4)
    h = _sdirk_initial_step(f!, u, f_eval, pars, t0, tf, abstol, reltol, n)

    # Compute initial Jacobian and factorize
    _sdirk_compute_jacobian!(w.jac, f!, f_eval, u, w.f_tmp, pars, t0, n;
                              jac_fn=jac_fn, autodiff=autodiff)
    _sdirk_factorize!(w, h)
    w.jac_current = true
    w.jac_age = 0
    last_factor_h = h

    t = t0
    save_idx = 1

    # Save initial point(s)
    while save_idx <= n_save && saveat[save_idx] <= t0 + eps(t0) * 10
        @inbounds for i in 1:n; result_u[i, save_idx] = u[i]; end
        save_idx += 1
    end

    total_steps = 0
    n_reject = 0

    while t < tf - eps(tf) * 10 && save_idx <= n_save && total_steps < max_steps
        total_steps += 1
        h = min(h, tf - t)

        # Compute f(t, u) at step start — used for Jacobian and Hermite interpolation
        f!(f_eval, u, pars, t)

        # Refactorize if h changed significantly since last factorization
        if abs(h - last_factor_h) > T(0.01) * last_factor_h
            _sdirk_factorize!(w, h)
            last_factor_h = h
        end

        # Recompute Jacobian if too old
        if w.jac_age > _SDIRK_MAX_JAC_AGE
            _sdirk_compute_jacobian!(w.jac, f!, f_eval, u, w.f_tmp, pars, t, n;
                                      jac_fn=jac_fn, autodiff=autodiff)
            w.jac_current = true
            w.jac_age = 0
            _sdirk_factorize!(w, h)
            last_factor_h = h
        end

        t_start = t

        # ── Stage 1: Y₁ − h·γ·f(t + c₁·h, Y₁) = uₙ ──
        @inbounds for i in 1:n; stage_rhs[i] = u[i]; end
        t_stage = t + T(_SDIRK_C1) * h

        converged = _sdirk_solve_stage!(f!, w, pars, t_stage, h,
                                         stage_rhs, stage_val, abstol, reltol)
        if !converged
            @goto newton_fail
        end
        f!(k1, stage_val, pars, t_stage)

        # ── Stage 2: Y₂ − h·γ·f(t + c₂·h, Y₂) = uₙ + h·a₂₁·k₁ ──
        @inbounds for i in 1:n
            stage_rhs[i] = u[i] + h * T(_SDIRK_A21) * k1[i]
        end
        t_stage = t + T(_SDIRK_C2) * h

        converged = _sdirk_solve_stage!(f!, w, pars, t_stage, h,
                                         stage_rhs, stage_val, abstol, reltol)
        if !converged
            @goto newton_fail
        end
        f!(k2, stage_val, pars, t_stage)

        # ── Stage 3: Y₃ − h·γ·f(t + c₃·h, Y₃) = uₙ + h·(a₃₁·k₁ + a₃₂·k₂) ──
        @inbounds for i in 1:n
            stage_rhs[i] = u[i] + h * (T(_SDIRK_A31) * k1[i] + T(_SDIRK_A32) * k2[i])
        end
        t_stage = t + T(_SDIRK_C3) * h

        converged = _sdirk_solve_stage!(f!, w, pars, t_stage, h,
                                         stage_rhs, stage_val, abstol, reltol)
        if !converged
            @goto newton_fail
        end
        f!(k3, stage_val, pars, t_stage)

        # ── Stage 4: Y₄ − h·γ·f(t + c₄·h, Y₄) = uₙ + h·(a₄₁·k₁ + a₄₂·k₂ + a₄₃·k₃) ──
        @inbounds for i in 1:n
            stage_rhs[i] = u[i] + h * (T(_SDIRK_A41) * k1[i] + T(_SDIRK_A42) * k2[i] +
                                        T(_SDIRK_A43) * k3[i])
        end
        t_stage = t + T(_SDIRK_C4) * h

        converged = _sdirk_solve_stage!(f!, w, pars, t_stage, h,
                                         stage_rhs, stage_val, abstol, reltol)
        if !converged
            @goto newton_fail
        end
        f!(k4, stage_val, pars, t_stage)

        # ── Stage 5: Y₅ − h·γ·f(t + c₅·h, Y₅) = uₙ + h·(a₅₁·k₁ + a₅₂·k₂ + a₅₃·k₃ + a₅₄·k₄) ──
        @inbounds for i in 1:n
            stage_rhs[i] = u[i] + h * (T(_SDIRK_A51) * k1[i] + T(_SDIRK_A52) * k2[i] +
                                        T(_SDIRK_A53) * k3[i] + T(_SDIRK_A54) * k4[i])
        end
        t_stage = t + T(_SDIRK_C5) * h

        converged = _sdirk_solve_stage!(f!, w, pars, t_stage, h,
                                         stage_rhs, stage_val, abstol, reltol)
        if !converged
            @goto newton_fail
        end
        f!(k5, stage_val, pars, t_stage)

        # 4th order solution (stiffly accurate — equals stage 5 value)
        @inbounds for i in 1:n
            u_new[i] = u[i] + h * (T(_SDIRK_B1) * k1[i] + T(_SDIRK_B2) * k2[i] +
                                    T(_SDIRK_B3) * k3[i] + T(_SDIRK_B4) * k4[i] +
                                    T(_SDIRK_B5) * k5[i])
        end

        # ── Error estimate (4th order − 3rd order embedded) ──
        err = zero(T)
        @inbounds for i in 1:n
            sc_i = abstol + reltol * max(abs(u[i]), abs(u_new[i]))
            ei = h * (T(_SDIRK_E1) * k1[i] + T(_SDIRK_E2) * k2[i] +
                      T(_SDIRK_E3) * k3[i] + T(_SDIRK_E4) * k4[i] +
                      T(_SDIRK_E5) * k5[i])
            err += (ei / sc_i)^2
        end
        err = sqrt(err / n)

        if err <= one(T)
            # Step accepted
            t += h
            n_reject = 0
            w.jac_age += 1

            # Hermite cubic interpolation for saveat points in (t_start, t]
            # Uses f_eval = f(t_start, u_n) and k5 = f(t_start+h, u_{n+1})
            while save_idx <= n_save && saveat[save_idx] <= t + eps(t) * 100
                ts = saveat[save_idx]
                if abs(ts - t) < h * T(1e-12) + eps(t) * 100
                    @inbounds for i in 1:n; result_u[i, save_idx] = u_new[i]; end
                else
                    theta = (ts - t_start) / h
                    theta1 = one(T) - theta
                    @inbounds for i in 1:n
                        dy = u_new[i] - u[i]
                        result_u[i, save_idx] = theta1 * u[i] + theta * u_new[i] +
                            theta * (theta - one(T)) * ((one(T) - 2*theta) * dy +
                            (theta - one(T)) * h * f_eval[i] + theta * h * k5[i])
                    end
                end
                save_idx += 1
            end

            @inbounds for i in 1:n; u[i] = u_new[i]; end
            w.jac_current = false
        else
            n_reject += 1
            # Recompute Jacobian after repeated rejections
            if n_reject >= 3 && !w.jac_current
                _sdirk_compute_jacobian!(w.jac, f!, f_eval, u, w.f_tmp, pars, t, n;
                                          jac_fn=jac_fn, autodiff=autodiff)
                w.jac_current = true
                w.jac_age = 0
                _sdirk_factorize!(w, h)
                last_factor_h = h
            end
        end

        # Step size control (order p = 4 ⟹ exponent = -1/(p+1) = -1/5)
        if err == zero(T)
            h *= T(4)
        else
            fac = T(0.9) * err^(T(-1) / T(5))
            h *= clamp(fac, T(0.2), T(4.0))
        end

        continue

        @label newton_fail
        # Recompute Jacobian and retry with smaller step
        f!(f_eval, u, pars, t)
        _sdirk_compute_jacobian!(w.jac, f!, f_eval, u, w.f_tmp, pars, t, n;
                                  jac_fn=jac_fn, autodiff=autodiff)
        w.jac_current = true
        w.jac_age = 0
        h *= T(0.5)
        _sdirk_factorize!(w, h)
        last_factor_h = h
        n_reject += 1
    end

    # Fill remaining save points with last known state
    while save_idx <= n_save
        @inbounds for i in 1:n; result_u[i, save_idx] = u[i]; end
        save_idx += 1
    end

    return nothing
end

# ── Initial step size ─────────────────────────────────────────

"""Estimate initial step size (adapted from Hairer–Nørsett–Wanner, order p = 4)."""
function _sdirk_initial_step(f!::F, u0::AbstractVector{T}, f0::AbstractVector{T},
                             pars, t0::T, tf::T,
                             abstol::T, reltol::T, n::Int) where {F, T}
    d0 = zero(T); d1 = zero(T)
    @inbounds for i in 1:n
        sc = abstol + reltol * abs(u0[i])
        d0 += (u0[i] / sc)^2
        d1 += (f0[i] / sc)^2
    end
    d0 = sqrt(d0 / n); d1 = sqrt(d1 / n)

    h0 = (d0 < T(1e-5) || d1 < T(1e-5)) ? T(1e-6) : T(0.01) * d0 / d1
    h0 = min(h0, tf - t0)

    u1 = similar(u0); f1 = similar(u0)
    @inbounds for i in 1:n; u1[i] = u0[i] + h0 * f0[i]; end
    f!(f1, u1, pars, t0 + h0)

    d2 = zero(T)
    @inbounds for i in 1:n
        sc = abstol + reltol * abs(u0[i])
        d2 += ((f1[i] - f0[i]) / sc)^2
    end
    d2 = sqrt(d2 / n) / h0

    # 1/(p+1) = 1/5 for order p = 4
    h1 = max(d1, d2) <= T(1e-15) ? max(T(1e-6), h0 * T(1e-3)) :
                                    (T(0.01) / max(d1, d2))^(T(1) / T(5))
    return min(T(100) * h0, h1, tf - t0)
end
