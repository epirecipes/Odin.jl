# Stochastic Differential Equation (SDE) solvers.
# Implements Euler-Maruyama (order 0.5) and Milstein (order 1.0) schemes
# for systems of the form: dX = f(X,t) dt + σ(X,t) dW
# Follows the zero-allocation workspace pattern from DP5.

"""
    SDEWorkspace{T}

Pre-allocated workspace for SDE solvers. Create once, reuse across solves.
"""
mutable struct SDEWorkspace{T}
    drift::Vector{T}           # f(X, t) buffer
    diffusion_coeff::Vector{T} # σ(X, t) buffer
    diffusion_pert::Vector{T}  # σ(X+h, t) buffer for Milstein
    noise::Vector{T}           # Z ~ N(0,1) random vector
    u::Vector{T}               # current state
    u_new::Vector{T}           # next state
    u_pert::Vector{T}          # perturbed state for Milstein
    n::Int
    result_matrix::Matrix{T}   # pre-allocated n_state × n_save output
end

function SDEWorkspace(n::Int, ::Type{T}=Float64) where T
    SDEWorkspace{T}(
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        n,
        Matrix{T}(undef, n, 0),
    )
end

function _ensure_sde_workspace!(ws::SDEWorkspace{T}, n::Int) where T
    ws.n == n && return
    for f in (:drift, :diffusion_coeff, :diffusion_pert, :noise, :u, :u_new, :u_pert)
        resize!(getfield(ws, f), n)
    end
    ws.n = n
end

function _ensure_sde_result_matrix!(ws::SDEWorkspace{T}, n::Int, n_save::Int) where T
    if size(ws.result_matrix, 1) != n || size(ws.result_matrix, 2) != n_save
        ws.result_matrix = Matrix{T}(undef, n, n_save)
    end
end

"""
    SDEResult{T}

Result of an SDE solve.
"""
struct SDEResult{T}
    t::Vector{T}
    u::Matrix{T}
end

"""
    sde_solve!(rhs_fn!, diffusion_fn!, u0, tspan, pars, dt, saveat;
               ws, rng, method) -> SDEResult

Solve an SDE system using fixed time-stepping.
`method` can be `:euler_maruyama` or `:milstein`.
"""
function sde_solve!(rhs_fn!::F, diffusion_fn!::G,
                    u0::AbstractVector{T}, tspan::Tuple{T,T}, pars,
                    dt::T, saveat::AbstractVector{T};
                    ws::Union{Nothing, SDEWorkspace{T}}=nothing,
                    rng::Random.AbstractRNG=Random.default_rng(),
                    method::Symbol=:euler_maruyama) where {F, G, T<:AbstractFloat}
    if ws === nothing
        w = SDEWorkspace(length(u0), T)
    else
        w = ws
    end
    _sde_solve_core!(rhs_fn!, diffusion_fn!, u0, tspan, pars, dt, saveat, w, rng, method)
    return SDEResult(collect(saveat), w.result_matrix)
end

"""
Zero-allocation core of the SDE solver. All results written into workspace `w`.
After return, `w.result_matrix` holds the solution at the saveat times.
"""
function _sde_solve_core!(rhs_fn!::F, diffusion_fn!::G,
                          u0::AbstractVector{T}, tspan::Tuple{T,T}, pars::P,
                          dt::T, saveat::AbstractVector{T},
                          w::SDEWorkspace{T},
                          rng::Random.AbstractRNG,
                          method::Symbol) where {F, G, T<:AbstractFloat, P}
    t0, tf = tspan
    n = length(u0)
    n_save = length(saveat)

    _ensure_sde_workspace!(w, n)
    _ensure_sde_result_matrix!(w, n, n_save)

    u = w.u
    u_new = w.u_new
    drift = w.drift
    σ = w.diffusion_coeff
    z = w.noise
    result_u = w.result_matrix

    @inbounds for i in 1:n; u[i] = u0[i]; end

    t = t0
    save_idx = 1
    sqrt_dt = sqrt(dt)

    # Save initial point(s)
    while save_idx <= n_save && saveat[save_idx] <= t0 + eps(t0) * 10
        @inbounds for i in 1:n; result_u[i, save_idx] = u[i]; end
        save_idx += 1
    end

    while t < tf - dt / 2 && save_idx <= n_save
        # Compute drift: f(u, t)
        rhs_fn!(drift, u, pars, t)
        # Compute diffusion coefficients: σ(u, t)
        diffusion_fn!(σ, u, pars, t)
        # Generate random normal increments
        @inbounds for i in 1:n
            z[i] = randn(rng)
        end

        if method === :milstein
            _milstein_step!(u_new, u, drift, σ, z, dt, sqrt_dt, n,
                            diffusion_fn!, pars, t, w.u_pert, w.diffusion_pert)
        else
            _euler_maruyama_step!(u_new, u, drift, σ, z, dt, sqrt_dt, n)
        end

        # Copy u_new → u
        @inbounds for i in 1:n; u[i] = u_new[i]; end
        t += dt

        # Save at requested times
        while save_idx <= n_save && saveat[save_idx] <= t + eps(t) * 100
            @inbounds for i in 1:n; result_u[i, save_idx] = u[i]; end
            save_idx += 1
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
Euler-Maruyama step (strong order 0.5):
    X_{n+1} = X_n + f(X_n, t_n) * dt + σ(X_n, t_n) * √dt * Z_n
"""
function _euler_maruyama_step!(u_new::AbstractVector{T}, u::AbstractVector{T},
                               drift::AbstractVector{T}, σ::AbstractVector{T},
                               z::AbstractVector{T}, dt::T, sqrt_dt::T, n::Int) where T
    @inbounds for i in 1:n
        u_new[i] = u[i] + drift[i] * dt + σ[i] * sqrt_dt * z[i]
    end
    return nothing
end

"""
Milstein step (strong order 1.0, diagonal noise):
    X_{n+1} = X_n + f(X_n, t_n) * dt + σ(X_n, t_n) * dW
              + 0.5 * σ(X_n) * (dσ_i/dx_i) * (dW² - dt)

Uses forward finite differences for dσ_i/dx_i.
"""
function _milstein_step!(u_new::AbstractVector{T}, u::AbstractVector{T},
                         drift::AbstractVector{T}, σ::AbstractVector{T},
                         z::AbstractVector{T}, dt::T, sqrt_dt::T, n::Int,
                         diffusion_fn!::G, pars, t::T,
                         u_pert::AbstractVector{T},
                         σ_pert::AbstractVector{T}) where {T, G}
    # First do the Euler-Maruyama part
    @inbounds for i in 1:n
        dW_i = sqrt_dt * z[i]
        u_new[i] = u[i] + drift[i] * dt + σ[i] * dW_i
    end

    # Now add Milstein correction for each component
    @inbounds for i in 1:n
        # Forward finite difference for dσ_i/dx_i
        h = cbrt(eps(T)) * max(abs(u[i]), one(T))

        for j in 1:n; u_pert[j] = u[j]; end
        u_pert[i] += h
        diffusion_fn!(σ_pert, u_pert, pars, t)

        dσ_dx = (σ_pert[i] - σ[i]) / h
        dW_i = sqrt_dt * z[i]
        u_new[i] += T(0.5) * σ[i] * dσ_dx * (dW_i^2 - dt)
    end
    return nothing
end
