# Sensitivity analysis for ODE models: forward, adjoint, Sobol, and Morris.
# Pure Julia implementation using ForwardDiff for Jacobians.

using ForwardDiff: Dual, Tag, Partials, value, partials, jacobian, gradient

# ────────────────────────────────────────────────────────────────
# Result types
# ────────────────────────────────────────────────────────────────

"""
    ForwardSensitivityResult

Result of forward sensitivity analysis.
- `trajectory`: state trajectory, (n_state, n_times)
- `sensitivities`: (n_state, n_params, n_times) — `S[i,j,t] = ∂u_i/∂p_j` at time `t`
- `times`: output time points
- `param_names`: names of the parameters differentiated w.r.t.
"""
struct ForwardSensitivityResult
    trajectory::Matrix{Float64}
    sensitivities::Array{Float64, 3}
    times::Vector{Float64}
    param_names::Vector{Symbol}
end

"""
    AdjointSensitivityResult

Result of adjoint sensitivity analysis.
- `gradient`: vector `∂L/∂p` for each parameter
- `param_names`: names of the parameters
- `loss_value`: the scalar loss L(p)
"""
struct AdjointSensitivityResult
    gradient::Vector{Float64}
    param_names::Vector{Symbol}
    loss_value::Float64
end

"""
    SobolResult

Sobol sensitivity indices.
- `first_order`: Dict mapping parameter name → first-order index
- `total_order`: Dict mapping parameter name → total-order index
- `param_names`: parameter names
"""
struct SobolResult
    first_order::Dict{Symbol, Float64}
    total_order::Dict{Symbol, Float64}
    param_names::Vector{Symbol}
end

"""
    MorrisResult

Morris screening results.
- `mu_star`: Dict mapping parameter name → μ* (absolute mean of elementary effects)
- `sigma`: Dict mapping parameter name → σ (standard deviation)
- `param_names`: parameter names
"""
struct MorrisResult
    mu_star::Dict{Symbol, Float64}
    sigma::Dict{Symbol, Float64}
    param_names::Vector{Symbol}
end

# ────────────────────────────────────────────────────────────────
# Forward sensitivity analysis
# ────────────────────────────────────────────────────────────────

"""
    dust_sensitivity_forward(gen, pars; times, params_of_interest, solver=:dp5,
                             ode_control=DustODEControl())

Compute forward sensitivities ∂u/∂p by solving the augmented variational system.

The augmented system is `[u; vec(S)]` where `S = ∂u/∂p`. This gives the full
sensitivity matrix at every output time.

# Arguments
- `gen::DustSystemGenerator` — an ODE model generator
- `pars::NamedTuple` — nominal parameter values
- `times` — output time points
- `params_of_interest` — vector of parameter `Symbol`s to differentiate w.r.t.
- `solver` — `:dp5` or `:sdirk`

# Returns
`ForwardSensitivityResult` with trajectory and sensitivity arrays.
"""
function dust_sensitivity_forward(
    gen::DustSystemGenerator,
    pars::NamedTuple;
    times::AbstractVector{Float64},
    params_of_interest::Vector{Symbol},
    solver::Symbol=:dp5,
    ode_control::DustODEControl=DustODEControl(),
)
    model = gen.model
    @assert model.is_continuous "Forward sensitivity requires a continuous (ODE) model"

    full_pars = _merge_pars(model, pars, 1.0)
    if model.has_interpolation
        full_pars = _odin_setup_pars(model, full_pars)
    end

    n_state = _odin_n_state(model, full_pars)
    n_params = length(params_of_interest)
    n_aug = n_state + n_state * n_params  # [u; vec(S)]

    # Extract nominal parameter values as a vector
    p_vals = Float64[Float64(full_pars[k]) for k in params_of_interest]

    # Initial state
    u0 = zeros(Float64, n_state)
    _odin_initial!(model, u0, full_pars, Random.Xoshiro())

    # Check if initial conditions depend on parameters via finite differences
    S0 = zeros(Float64, n_state, n_params)
    eps_fd = 1e-7
    for j in 1:n_params
        pname = params_of_interest[j]
        pv_plus = Float64(full_pars[pname]) + eps_fd
        pars_plus = merge(full_pars, NamedTuple{(pname,)}((pv_plus,)))
        if model.has_interpolation
            pars_plus = _odin_setup_pars(model, pars_plus)
        end
        u0_plus = zeros(Float64, n_state)
        _odin_initial!(model, u0_plus, pars_plus, Random.Xoshiro())
        S0[:, j] .= (u0_plus .- u0) ./ eps_fd
    end

    # Build augmented initial condition: [u0; vec(S0)]
    aug0 = zeros(Float64, n_aug)
    aug0[1:n_state] .= u0
    aug0[n_state+1:end] .= vec(S0)

    # Build augmented RHS using ForwardDiff for Jacobians
    aug_rhs! = _build_forward_augmented_rhs(model, full_pars, params_of_interest,
                                             n_state, n_params)

    # Solve the augmented system
    t_start = 0.0
    t_end = last(times)

    if solver === :sdirk
        ws = SDIRKWorkspace(n_aug, Float64)
        _sdirk_solve_core!(aug_rhs!, aug0, (t_start, t_end), full_pars, collect(times),
                           ws, nothing, ode_control.atol, ode_control.rtol, ode_control.max_steps)
        result_mat = ws.result_matrix
    else
        ws = DP5Workspace(n_aug, Float64)
        _dp5_solve_core!(aug_rhs!, aug0, (t_start, t_end), full_pars, collect(times),
                         ws, nothing, ode_control.atol, ode_control.rtol, ode_control.max_steps)
        result_mat = ws.result_matrix
    end

    # Unpack results
    n_times = length(times)
    trajectory = zeros(Float64, n_state, n_times)
    sensitivities = zeros(Float64, n_state, n_params, n_times)

    for ti in 1:n_times
        trajectory[:, ti] .= result_mat[1:n_state, ti]
        S_flat = result_mat[n_state+1:end, ti]
        sensitivities[:, :, ti] .= reshape(S_flat, n_state, n_params)
    end

    return ForwardSensitivityResult(trajectory, sensitivities, collect(times), params_of_interest)
end

"""Build the augmented RHS for forward sensitivity: d[u;S]/dt = [f; df/du*S + df/dp]."""
function _build_forward_augmented_rhs(model, full_pars, params_of_interest, n_state, n_params)
    # Pre-allocate work buffers for Jacobian computation
    du_buf = zeros(Float64, n_state)
    du_buf2 = zeros(Float64, n_state)
    u_dual_buf = zeros(Float64, n_state)

    function aug_rhs!(d_aug, aug, pars, t)
        u = @view aug[1:n_state]
        S_flat = @view aug[n_state+1:end]

        # f(u, p, t) — the original RHS
        _odin_rhs!(model, du_buf, u, full_pars, t)
        @inbounds for i in 1:n_state
            d_aug[i] = du_buf[i]
        end

        # Compute Jacobian ∂f/∂u using ForwardDiff (column by column)
        # J_u[:, j] = ∂f/∂u_j
        # We compute J_u * S + J_p columnwise to avoid allocating the full Jacobian

        S = reshape(S_flat, n_state, n_params)

        # For each parameter column j, compute: dS[:,j]/dt = J_u * S[:,j] + J_p[:,j]
        # First compute J_u * S for all columns, then add J_p

        # Compute J_u via finite differences (more robust than ForwardDiff for generated code)
        eps_jac = 1e-8
        for jp in 1:n_params
            # Initialize dS column to zero
            offset = n_state + (jp - 1) * n_state
            @inbounds for i in 1:n_state
                d_aug[offset + i] = 0.0
            end

            # J_u * S[:, jp] via directional derivative: (f(u + eps*S[:,jp]) - f(u)) / eps
            norm_s = 0.0
            @inbounds for i in 1:n_state
                norm_s += S[i, jp]^2
            end
            norm_s = sqrt(norm_s)

            if norm_s > 0.0
                h_dir = eps_jac / max(norm_s, 1.0)
                @inbounds for i in 1:n_state
                    u_dual_buf[i] = u[i] + h_dir * S[i, jp]
                end
                _odin_rhs!(model, du_buf2, u_dual_buf, full_pars, t)
                @inbounds for i in 1:n_state
                    d_aug[offset + i] += (du_buf2[i] - du_buf[i]) / h_dir
                end
            end

            # J_p[:, jp] via finite difference on the parameter
            pname = params_of_interest[jp]
            pv = Float64(full_pars[pname])
            h_p = eps_jac * max(abs(pv), 1.0)
            pars_plus = merge(full_pars, NamedTuple{(pname,)}((pv + h_p,)))
            if model.has_interpolation
                pars_plus = _odin_setup_pars(model, pars_plus)
            end
            _odin_rhs!(model, du_buf2, u, pars_plus, t)
            @inbounds for i in 1:n_state
                d_aug[offset + i] += (du_buf2[i] - du_buf[i]) / h_p
            end
        end

        return nothing
    end

    return aug_rhs!
end

# ────────────────────────────────────────────────────────────────
# Adjoint sensitivity analysis
# ────────────────────────────────────────────────────────────────

"""
    dust_sensitivity_adjoint(gen, pars, loss_fn; times, params_of_interest,
                             solver=:dp5, ode_control=DustODEControl(),
                             n_checkpoints=50)

Compute the gradient ∂L/∂p of a scalar loss function using the adjoint method.

The loss is `L = Σ_t loss_fn(u(t), t)` where the sum is over `times`.
The adjoint equation `dλ/dt = -λ * ∂f/∂u` is solved backward, and the
parameter gradient is accumulated as `∂L/∂p += λ * ∂f/∂p`.

# Arguments
- `gen::DustSystemGenerator` — an ODE model generator
- `pars::NamedTuple` — nominal parameter values
- `loss_fn` — function `(state_vec, t) -> scalar` (e.g., log-likelihood contribution)
- `times` — time points at which loss is evaluated
- `params_of_interest` — parameter symbols to differentiate w.r.t.
- `n_checkpoints` — number of checkpoints for forward trajectory storage

# Returns
`AdjointSensitivityResult` with gradient vector and loss value.
"""
function dust_sensitivity_adjoint(
    gen::DustSystemGenerator,
    pars::NamedTuple,
    loss_fn::Function;
    times::AbstractVector{Float64},
    params_of_interest::Vector{Symbol},
    solver::Symbol=:dp5,
    adjoint_solver::Union{Symbol,Nothing}=nothing,
    ode_control::DustODEControl=DustODEControl(),
    n_checkpoints::Int=50,
)
    adj_solver = something(adjoint_solver, solver)
    model = gen.model
    @assert model.is_continuous "Adjoint sensitivity requires a continuous (ODE) model"

    full_pars = _merge_pars(model, pars, 1.0)
    if model.has_interpolation
        full_pars = _odin_setup_pars(model, full_pars)
    end

    n_state = _odin_n_state(model, full_pars)
    n_params = length(params_of_interest)
    n_times = length(times)
    n_aug = n_state + n_params  # augmented system: [λ; grad_p]

    # ── Step 1: Forward solve and store trajectory at data times ──
    # Include t=0 so we have forward state for the [0, times[1]] interval
    u0 = zeros(Float64, n_state)
    _odin_initial!(model, u0, full_pars, Random.Xoshiro())

    rhs_fn! = (du, u, p, t) -> _odin_rhs!(model, du, u, p, t)

    t_start = 0.0
    t_end = last(times)
    all_saveat = vcat(0.0, collect(times))
    # Remove duplicate if times[1] == 0
    unique!(all_saveat)
    sort!(all_saveat)
    n_all = length(all_saveat)

    if solver === :sdirk
        ws = SDIRKWorkspace(n_state, Float64)
        _sdirk_solve_core!(rhs_fn!, u0, (t_start, t_end), full_pars, all_saveat,
                           ws, nothing, ode_control.atol, ode_control.rtol, ode_control.max_steps)
        full_traj = copy(ws.result_matrix)
    else
        ws = DP5Workspace(n_state, Float64)
        _dp5_solve_core!(rhs_fn!, u0, (t_start, t_end), full_pars, all_saveat,
                         ws, nothing, ode_control.atol, ode_control.rtol, ode_control.max_steps)
        full_traj = copy(ws.result_matrix)
    end

    # Map data times to indices in full_traj
    # all_saveat = [0.0, times[1], times[2], ...]
    # If times[1] == 0, offset is 0; otherwise offset is 1
    offset = all_saveat[1] < times[1] - eps(times[1]) ? 1 : 0
    fwd_traj = @view full_traj[:, (1+offset):(n_times+offset)]

    # Evaluate total loss
    total_loss = 0.0
    for ti in 1:n_times
        state_t = @view fwd_traj[:, ti]
        total_loss += loss_fn(state_t, times[ti])
    end

    # ── Step 2: Backward adjoint solve using the same solver ──
    # Augmented state z = [u_1...u_n_state, λ_1...λ_n_state, g_1...g_n_params]
    # The forward state u is reconstructed alongside the adjoint to avoid
    # interpolation error. In reversed time τ = t_hi - t:
    #   du/dτ = -f(u, t_hi - τ)           (forward ODE, reversed)
    #   dλ/dτ = +(∂f/∂y)^T λ              (adjoint)
    #   dg/dτ = +(∂f/∂θ)^T λ              (param gradient accumulator)

    n_aug = 2 * n_state + n_params

    # Pre-allocate VJP buffers (shared across all intervals)
    vjp_state_buf = zeros(Float64, n_state)
    vjp_params_buf = zeros(Float64, n_params)
    du_fwd = zeros(Float64, n_state)

    eps_jac = 1e-8

    # Start with λ = 0 at t = T, then add terminal condition
    lambda = zeros(Float64, n_state)
    grad_p = zeros(Float64, n_params)

    _add_loss_gradient!(lambda, loss_fn, fwd_traj, n_times, times, n_state)

    # Store the current interval's t_hi for the RHS closure
    t_hi_ref = Ref(0.0)

    function adjoint_aug_rhs!(dz, z, _pars, tau)
        # τ runs from 0 to (t_hi - t_lo); real time t = t_hi - τ
        t_real = t_hi_ref[] - tau

        # Extract forward state and adjoint from augmented vector
        u_cur = @view z[1:n_state]
        lambda_cur = @view z[n_state+1:2*n_state]

        # du/dτ = -f(u, t)
        _odin_rhs!(model, du_fwd, u_cur, full_pars, t_real)
        @inbounds for i in 1:n_state
            dz[i] = -du_fwd[i]
        end

        # dλ/dτ = +(∂f/∂y)^T * λ
        fill!(vjp_state_buf, 0.0)
        compute_vjp_state!(model, vjp_state_buf, u_cur, lambda_cur,
                          full_pars, t_real)
        @inbounds for i in 1:n_state
            dz[n_state + i] = vjp_state_buf[i]
        end

        # dg/dτ = +(∂f/∂θ)^T * λ
        fill!(vjp_params_buf, 0.0)
        compute_vjp_params!(model, vjp_params_buf, u_cur, lambda_cur,
                           full_pars, t_real, params_of_interest)
        @inbounds for jp in 1:n_params
            dz[2*n_state + jp] = vjp_params_buf[jp]
        end
        return nothing
    end

    # Create workspace for the augmented backward system
    if adj_solver === :sdirk
        adj_ws = SDIRKWorkspace(n_aug, Float64)
    else
        adj_ws = DP5Workspace(n_aug, Float64)
    end
    z0 = zeros(Float64, n_aug)
    adj_saveat = Float64[0.0]  # only need endpoint

    # Walk backward through time intervals
    for ti in n_times:-1:2
        t_hi = times[ti]
        t_lo = times[ti - 1]
        t_hi_ref[] = t_hi

        # Initial condition: [u(t_hi), λ, 0 (param grad for this interval)]
        @inbounds for i in 1:n_state
            z0[i] = fwd_traj[i, ti]          # forward state at t_hi
            z0[n_state + i] = lambda[i]       # adjoint
        end
        @inbounds for jp in 1:n_params
            z0[2*n_state + jp] = 0.0          # param grad accumulator
        end

        # Integrate from τ=0 to τ=(t_hi - t_lo)
        tau_span = (0.0, t_hi - t_lo)
        adj_saveat[1] = t_hi - t_lo

        if adj_solver === :sdirk
            _sdirk_solve_core!(adjoint_aug_rhs!, z0, tau_span, nothing, adj_saveat,
                               adj_ws, nothing, ode_control.atol, ode_control.rtol,
                               ode_control.max_steps)
            z_end = @view adj_ws.result_matrix[:, 1]
        else
            _dp5_solve_core!(adjoint_aug_rhs!, z0, tau_span, nothing, adj_saveat,
                             adj_ws, nothing, ode_control.atol, ode_control.rtol,
                             ode_control.max_steps)
            z_end = @view adj_ws.result_matrix[:, 1]
        end

        # Extract λ and accumulate parameter gradient
        @inbounds for i in 1:n_state
            lambda[i] = z_end[n_state + i]
        end
        @inbounds for jp in 1:n_params
            grad_p[jp] += z_end[2*n_state + jp]
        end

        # Jump condition at times[ti-1]: add ∂loss/∂u
        _add_loss_gradient!(lambda, loss_fn, fwd_traj, ti - 1, times, n_state)
    end

    # ── Integrate from times[1] back to t=0 (if times[1] > 0) ──
    if times[1] > eps(1.0)
        t_hi = times[1]
        t_lo = 0.0
        t_hi_ref[] = t_hi

        # Forward state at t_hi from fwd_traj, at t=0 from full_traj column 1
        @inbounds for i in 1:n_state
            z0[i] = fwd_traj[i, 1]           # u(times[1])
            z0[n_state + i] = lambda[i]       # current λ
        end
        @inbounds for jp in 1:n_params
            z0[2*n_state + jp] = 0.0
        end

        tau_span = (0.0, t_hi - t_lo)
        adj_saveat[1] = t_hi - t_lo

        if adj_solver === :sdirk
            _sdirk_solve_core!(adjoint_aug_rhs!, z0, tau_span, nothing, adj_saveat,
                               adj_ws, nothing, ode_control.atol, ode_control.rtol,
                               ode_control.max_steps)
            z_end = @view adj_ws.result_matrix[:, 1]
        else
            _dp5_solve_core!(adjoint_aug_rhs!, z0, tau_span, nothing, adj_saveat,
                             adj_ws, nothing, ode_control.atol, ode_control.rtol,
                             ode_control.max_steps)
            z_end = @view adj_ws.result_matrix[:, 1]
        end

        @inbounds for i in 1:n_state
            lambda[i] = z_end[n_state + i]
        end
        @inbounds for jp in 1:n_params
            grad_p[jp] += z_end[2*n_state + jp]
        end
    end

    # Handle initial condition dependence on parameters
    for jp in 1:n_params
        pname = params_of_interest[jp]
        pv = Float64(full_pars[pname])
        h_p = eps_jac * max(abs(pv), 1.0)
        pars_plus = merge(full_pars, NamedTuple{(pname,)}((pv + h_p,)))
        if model.has_interpolation
            pars_plus = _odin_setup_pars(model, pars_plus)
        end
        u0_plus = zeros(Float64, n_state)
        _odin_initial!(model, u0_plus, pars_plus, Random.Xoshiro())
        u0_base = zeros(Float64, n_state)
        _odin_initial!(model, u0_base, full_pars, Random.Xoshiro())
        for i in 1:n_state
            grad_p[jp] += lambda[i] * (u0_plus[i] - u0_base[i]) / h_p
        end
    end

    return AdjointSensitivityResult(grad_p, params_of_interest, total_loss)
end

"""Add ∂loss/∂u to λ at time index `ti` (finite differences)."""
function _add_loss_gradient!(lambda, loss_fn, fwd_traj, ti, times, n_state)
    state_t = @view fwd_traj[:, ti]
    t = times[ti]
    L0 = loss_fn(state_t, t)
    eps_l = 1e-7
    state_pert = copy(state_t)
    for i in 1:n_state
        state_pert[i] += eps_l
        L_pert = loss_fn(state_pert, t)
        lambda[i] += (L_pert - L0) / eps_l
        state_pert[i] = state_t[i]
    end
end

"""Accumulate discrete parameter gradient contribution at time `ti`."""
function _accumulate_param_gradient!(grad_p, lambda, model, full_pars, fwd_traj, ti,
                                      times, params_of_interest, n_state, n_params,
                                      du_buf, du_buf2, eps_jac)
    # No continuous accumulation at a discrete point — handled by the integral
    return nothing
end

# ────────────────────────────────────────────────────────────────
# Sobol global sensitivity indices
# ────────────────────────────────────────────────────────────────

"""
    dust_sensitivity_sobol(gen, pars_ranges; n_samples=1000, times,
                           output_var=1, output_time=nothing,
                           solver=:dp5, ode_control=DustODEControl())

Compute first-order and total Sobol sensitivity indices via Saltelli's method.

# Arguments
- `gen` — ODE model generator
- `pars_ranges` — Dict{Symbol, Tuple{Float64,Float64}} mapping parameter name to (lo, hi)
- `n_samples` — base sample size (total model evaluations ≈ n_samples * (n_params + 2))
- `times` — output times for simulation
- `output_var` — state variable index (Int) or name (Symbol) to analyse
- `output_time` — if `nothing`, uses the last time; otherwise index into `times`
- `solver` — ODE solver

# Returns
`SobolResult` with first_order and total_order dictionaries.
"""
function dust_sensitivity_sobol(
    gen::DustSystemGenerator,
    pars_ranges::Dict{Symbol, Tuple{Float64, Float64}};
    n_samples::Int=1000,
    times::AbstractVector{Float64},
    output_var::Union{Int, Symbol}=1,
    output_time::Union{Nothing, Int}=nothing,
    solver::Symbol=:dp5,
    ode_control::DustODEControl=DustODEControl(),
)
    model = gen.model
    param_names = collect(keys(pars_ranges))
    n_params = length(param_names)
    lo = Float64[pars_ranges[k][1] for k in param_names]
    hi = Float64[pars_ranges[k][2] for k in param_names]

    t_idx = output_time === nothing ? length(times) : output_time

    # Resolve output_var to index
    var_idx = _resolve_var_idx(output_var, model, pars_ranges, param_names)

    # Generate two independent quasi-random matrices A and B
    A_mat = _sobol_sample(n_samples, n_params, lo, hi)
    B_mat = _sobol_sample(n_samples, n_params, lo, hi)

    # Evaluate model for A and B
    yA = _sobol_eval_batch(gen, A_mat, param_names, pars_ranges, times, var_idx, t_idx, solver, ode_control)
    yB = _sobol_eval_batch(gen, B_mat, param_names, pars_ranges, times, var_idx, t_idx, solver, ode_control)

    f0 = mean(yA)
    var_total = var(vcat(yA, yB))

    first_order = Dict{Symbol, Float64}()
    total_order = Dict{Symbol, Float64}()

    for j in 1:n_params
        # AB_j matrix: same as A except column j comes from B
        AB_j = copy(A_mat)
        AB_j[:, j] .= B_mat[:, j]
        yAB_j = _sobol_eval_batch(gen, AB_j, param_names, pars_ranges, times, var_idx, t_idx, solver, ode_control)

        # First order: S_j = V[E[Y|X_j]] / V[Y]
        # ≈ (1/N) Σ yB * (yAB_j - yA) / V[Y]
        s1 = 0.0
        for i in 1:n_samples
            s1 += yB[i] * (yAB_j[i] - yA[i])
        end
        s1 /= n_samples
        Si = var_total > 0.0 ? s1 / var_total : 0.0

        # Total order: ST_j = E[(Y_A - Y_AB_j)^2] / (2 * V[Y])
        st = 0.0
        for i in 1:n_samples
            st += (yA[i] - yAB_j[i])^2
        end
        st /= (2 * n_samples)
        STi = var_total > 0.0 ? st / var_total : 0.0

        first_order[param_names[j]] = clamp(Si, 0.0, 1.0)
        total_order[param_names[j]] = clamp(STi, 0.0, 1.0)
    end

    return SobolResult(first_order, total_order, param_names)
end

"""Generate a sample matrix of size (n_samples, n_params) with uniform sampling in [lo, hi]."""
function _sobol_sample(n_samples, n_params, lo, hi)
    mat = rand(n_samples, n_params)
    for j in 1:n_params
        mat[:, j] .= lo[j] .+ mat[:, j] .* (hi[j] - lo[j])
    end
    return mat
end

"""Evaluate model for each row of param_matrix, returning output values."""
function _sobol_eval_batch(gen, param_matrix, param_names, pars_ranges, times,
                            var_idx, t_idx, solver, ode_control)
    n_samples = size(param_matrix, 1)
    results = zeros(Float64, n_samples)

    model = gen.model
    n_state = 0
    # Pre-compute n_state from first sample
    test_pars = _build_pars_from_row(param_matrix, 1, param_names, pars_ranges)
    full_test = _merge_pars(model, test_pars, 1.0)
    if model.has_interpolation
        full_test = _odin_setup_pars(model, full_test)
    end
    n_state = _odin_n_state(model, full_test)

    ws = DP5Workspace(n_state, Float64)
    u0 = zeros(Float64, n_state)

    for i in 1:n_samples
        sample_pars = _build_pars_from_row(param_matrix, i, param_names, pars_ranges)
        full_pars = _merge_pars(model, sample_pars, 1.0)
        if model.has_interpolation
            full_pars = _odin_setup_pars(model, full_pars)
        end

        fill!(u0, 0.0)
        _odin_initial!(model, u0, full_pars, Random.Xoshiro())

        rhs_fn! = (du, u, p, t) -> _odin_rhs!(model, du, u, p, t)

        try
            if solver === :sdirk
                sws = SDIRKWorkspace(n_state, Float64)
                _sdirk_solve_core!(rhs_fn!, u0, (0.0, last(times)), full_pars,
                                   collect(times), sws, nothing,
                                   ode_control.atol, ode_control.rtol, ode_control.max_steps)
                results[i] = sws.result_matrix[var_idx, t_idx]
            else
                _dp5_solve_core!(rhs_fn!, u0, (0.0, last(times)), full_pars,
                                 collect(times), ws, nothing,
                                 ode_control.atol, ode_control.rtol, ode_control.max_steps)
                results[i] = ws.result_matrix[var_idx, t_idx]
            end
        catch
            results[i] = NaN
        end
    end
    return results
end

"""Build a NamedTuple of parameters from row `i` of the sample matrix."""
function _build_pars_from_row(param_matrix, i, param_names, pars_ranges)
    pairs = Pair{Symbol, Float64}[]
    for (j, pname) in enumerate(param_names)
        push!(pairs, pname => param_matrix[i, j])
    end
    return NamedTuple(pairs)
end

"""Resolve an output variable specifier to a state index."""
function _resolve_var_idx(output_var, model, pars_ranges, param_names)
    if output_var isa Int
        return output_var
    end
    # Try matching against state names
    test_pars = NamedTuple(Pair{Symbol,Float64}[k => (v[1]+v[2])/2 for (k,v) in pars_ranges])
    full_test = _merge_pars(model, test_pars, 1.0)
    if model.has_interpolation
        full_test = _odin_setup_pars(model, full_test)
    end
    snames = _odin_state_names(model, full_test)
    idx = findfirst(==(output_var), snames)
    idx === nothing && error("State variable $output_var not found in model. Available: $snames")
    return idx
end

# ────────────────────────────────────────────────────────────────
# Morris screening
# ────────────────────────────────────────────────────────────────

"""
    dust_sensitivity_morris(gen, pars_ranges; n_trajectories=20, times,
                            output_var=1, output_time=nothing, p_levels=4,
                            solver=:dp5, ode_control=DustODEControl())

Perform Morris elementary effects screening.

# Arguments
- `gen` — ODE model generator
- `pars_ranges` — Dict{Symbol, Tuple{Float64,Float64}} mapping parameter name to (lo, hi)
- `n_trajectories` — number of Morris trajectories (typically 10–50)
- `times` — output times
- `output_var` — state index or Symbol
- `output_time` — index into `times` (default: last)
- `p_levels` — number of grid levels (default 4)

# Returns
`MorrisResult` with μ* and σ for each parameter.
"""
function dust_sensitivity_morris(
    gen::DustSystemGenerator,
    pars_ranges::Dict{Symbol, Tuple{Float64, Float64}};
    n_trajectories::Int=20,
    times::AbstractVector{Float64},
    output_var::Union{Int, Symbol}=1,
    output_time::Union{Nothing, Int}=nothing,
    p_levels::Int=4,
    solver::Symbol=:dp5,
    ode_control::DustODEControl=DustODEControl(),
)
    model = gen.model
    param_names = collect(keys(pars_ranges))
    n_params = length(param_names)
    lo = Float64[pars_ranges[k][1] for k in param_names]
    hi = Float64[pars_ranges[k][2] for k in param_names]

    t_idx = output_time === nothing ? length(times) : output_time
    var_idx = _resolve_var_idx(output_var, model, pars_ranges, param_names)

    delta = 1.0 / (p_levels - 1)  # step size in [0,1] space

    # Collect elementary effects for each parameter
    effects = Dict{Symbol, Vector{Float64}}(k => Float64[] for k in param_names)

    for traj in 1:n_trajectories
        # Generate a random base point on the grid
        base = [lo[j] + rand(0:p_levels-1) * delta * (hi[j] - lo[j]) for j in 1:n_params]

        # Evaluate at base
        y_base = _morris_eval(gen, base, param_names, pars_ranges, times, var_idx, t_idx, solver, ode_control)

        # Random permutation of parameter indices
        perm = randperm(n_params)

        current = copy(base)
        y_current = y_base

        for j in perm
            # Perturb parameter j by ±delta
            old_val = current[j]
            step = delta * (hi[j] - lo[j])
            if current[j] + step > hi[j]
                current[j] -= step
            else
                current[j] += step
            end

            y_new = _morris_eval(gen, current, param_names, pars_ranges, times, var_idx, t_idx, solver, ode_control)

            # Elementary effect: (y_new - y_current) / delta (in normalised space)
            ee = (y_new - y_current) / delta
            push!(effects[param_names[j]], ee)

            y_current = y_new
        end
    end

    mu_star = Dict{Symbol, Float64}()
    sigma = Dict{Symbol, Float64}()
    for k in param_names
        ee = effects[k]
        mu_star[k] = mean(abs.(ee))
        sigma[k] = length(ee) > 1 ? std(ee) : 0.0
    end

    return MorrisResult(mu_star, sigma, param_names)
end

"""Evaluate model at a single parameter point for Morris screening."""
function _morris_eval(gen, param_vec, param_names, pars_ranges, times, var_idx, t_idx, solver, ode_control)
    model = gen.model
    pairs = Pair{Symbol, Float64}[]
    for (j, pname) in enumerate(param_names)
        push!(pairs, pname => param_vec[j])
    end
    sample_pars = NamedTuple(pairs)
    full_pars = _merge_pars(model, sample_pars, 1.0)
    if model.has_interpolation
        full_pars = _odin_setup_pars(model, full_pars)
    end

    n_state = _odin_n_state(model, full_pars)
    u0 = zeros(Float64, n_state)
    _odin_initial!(model, u0, full_pars, Random.Xoshiro())

    rhs_fn! = (du, u, p, t) -> _odin_rhs!(model, du, u, p, t)

    try
        if solver === :sdirk
            ws = SDIRKWorkspace(n_state, Float64)
            _sdirk_solve_core!(rhs_fn!, u0, (0.0, last(times)), full_pars,
                               collect(times), ws, nothing,
                               ode_control.atol, ode_control.rtol, ode_control.max_steps)
            return ws.result_matrix[var_idx, t_idx]
        else
            ws = DP5Workspace(n_state, Float64)
            _dp5_solve_core!(rhs_fn!, u0, (0.0, last(times)), full_pars,
                             collect(times), ws, nothing,
                             ode_control.atol, ode_control.rtol, ode_control.max_steps)
            return ws.result_matrix[var_idx, t_idx]
        end
    catch
        return NaN
    end
end

# ────────────────────────────────────────────────────────────────
# Integration with unfilter: gradient via forward/adjoint
# ────────────────────────────────────────────────────────────────

"""
    dust_unfilter_gradient(unfilter, pars, packer; method=:forward,
                           solver=:dp5, ode_control=DustODEControl())

Compute the gradient of the unfilter log-likelihood w.r.t. parameters using
forward or adjoint sensitivity analysis (avoiding ForwardDiff through the solver).

# Arguments
- `method` — `:forward` for forward sensitivity, `:adjoint` for adjoint method

# Returns
Named tuple `(log_likelihood, gradient)`.
"""
function dust_unfilter_gradient(
    unfilter::DustUnfilter,
    pars::NamedTuple,
    packer::MontyPacker;
    method::Symbol=:forward,
    solver::Symbol=:dp5,
    ode_control::DustODEControl=DustODEControl(),
)
    gen = unfilter.generator
    model = gen.model
    data = unfilter.data

    # Determine which parameters to differentiate
    param_names = packer.names

    full_pars = _merge_pars(model, pars, 1.0)
    if model.has_interpolation
        full_pars = _odin_setup_pars(model, full_pars)
    end

    if method === :forward
        return _unfilter_gradient_forward(gen, model, full_pars, data, param_names, solver, ode_control)
    elseif method === :adjoint
        return _unfilter_gradient_adjoint(gen, model, full_pars, data, param_names, solver, ode_control)
    else
        error("Unknown gradient method: $method. Use :forward or :adjoint.")
    end
end

function _unfilter_gradient_forward(gen, model, full_pars, data, param_names, solver, ode_control)
    fwd = dust_sensitivity_forward(gen, full_pars;
        times=data.times, params_of_interest=collect(param_names), solver=solver, ode_control=ode_control)

    n_state = size(fwd.trajectory, 1)
    n_params = length(param_names)
    n_times = length(data.times)

    # Compute log-likelihood and its gradient via chain rule
    ll = 0.0
    grad = zeros(Float64, n_params)

    eps_ll = 1e-7
    for ti in 1:n_times
        state_t = fwd.trajectory[:, ti]
        data_t = data.data[ti]
        t = data.times[ti]

        ll_t = _odin_compare_data(model, state_t, full_pars, data_t, t)
        ll += ll_t

        # ∂ll_t/∂u via finite differences
        dll_du = zeros(Float64, n_state)
        for i in 1:n_state
            state_pert = copy(state_t)
            state_pert[i] += eps_ll
            ll_pert = _odin_compare_data(model, state_pert, full_pars, data_t, t)
            dll_du[i] = (ll_pert - ll_t) / eps_ll
        end

        # Chain rule: ∂ll_t/∂p = ∂ll_t/∂u * ∂u/∂p
        for jp in 1:n_params
            for i in 1:n_state
                grad[jp] += dll_du[i] * fwd.sensitivities[i, jp, ti]
            end
        end
    end

    return (log_likelihood=ll, gradient=grad)
end

function _unfilter_gradient_adjoint(gen, model, full_pars, data, param_names, solver, ode_control)
    loss_fn = (state, t) -> begin
        # Find the data point closest to time t
        ti = findfirst(τ -> abs(τ - t) < 1e-10, data.times)
        ti === nothing && return 0.0
        return _odin_compare_data(model, state, full_pars, data.data[ti], t)
    end

    result = dust_sensitivity_adjoint(gen, full_pars, loss_fn;
        times=data.times, params_of_interest=collect(param_names),
        solver=solver, ode_control=ode_control)

    return (log_likelihood=result.loss_value, gradient=result.gradient)
end

# Helper: compute mean of a vector
function _sensitivity_mean(x::AbstractVector{Float64})
    s = 0.0
    for v in x; s += v; end
    return s / length(x)
end
