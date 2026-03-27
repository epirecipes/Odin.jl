# Deterministic unfilter for ODE-based likelihood computation.

using DifferentialEquations: ODEProblem, solve, DP5, ReturnCode, remake

"""
    DustUnfilter{M}

Deterministic likelihood evaluator for continuous-time models.
Runs the ODE forward and evaluates data likelihood at each data point.
"""
mutable struct DustUnfilter{M<:AbstractOdinModel, D<:NamedTuple}
    generator::DustSystemGenerator{M}
    data::FilterData{D}
    time_start::Float64
    ode_control::DustODEControl
    # Cached work buffers for repeated calls
    _state_cache::Vector{Float64}
    _rng_cache::Xoshiro
    _cached_prob::Any  # ODEProblem, lazily initialized
    _dp5_workspace::Union{Nothing, DP5Workspace{Float64}}
    _sdirk_workspace::Union{Nothing, SDIRKWorkspace{Float64}}
    _saveat_cache::Vector{Float64}
    # Multi-group support
    n_groups::Int
    group_data::Union{Nothing, Vector{FilterData{D}}}
    # Snapshot/trajectory storage
    _last_snapshots::Union{Nothing, Dict{Float64, Vector{Float64}}}
    _last_trajectories::Union{Nothing, Matrix{Float64}}
end

"""
    dust_unfilter_create(generator, data; time_start, ode_control)

Create a deterministic unfilter for an ODE model.
"""
function dust_unfilter_create(
    gen::DustSystemGenerator{M},
    data::FilterData{D};
    time_start::Float64=0.0,
    ode_control::DustODEControl=DustODEControl(),
) where {M, D}
    n_state = gen.model.n_state
    saveat_cache = collect(Float64, data.times)
    return DustUnfilter{M,D}(gen, data, time_start, ode_control,
                              zeros(Float64, n_state), Random.Xoshiro(), nothing,
                              nothing, nothing, saveat_cache,
                               1, nothing, nothing, nothing)
end

"""
    dust_unfilter_run!(unfilter, pars) -> Float64

Run the deterministic unfilter and return the log-likelihood.
Uses lightweight DP5 or SDIRK4 for Float64, DifferentialEquations.jl for AD (Dual numbers).
"""
function dust_unfilter_run!(unfilter::DustUnfilter, pars::NamedTuple;
                                save_snapshots::Union{Nothing, Vector{Float64}}=nothing)
    if unfilter.n_groups > 1 && unfilter.group_data !== nothing
        return _run_grouped_unfilter!(unfilter, pars; save_snapshots=save_snapshots)
    end
    return _unfilter_run_single!(unfilter, pars; save_snapshots=save_snapshots)
end

function _unfilter_run_single!(unfilter::DustUnfilter, pars::NamedTuple;
                                save_snapshots::Union{Nothing, Vector{Float64}}=nothing)
    model = unfilter.generator.model
    n_state = model.n_state
    n_data = length(unfilter.data.times)

    # Merge dt into pars (compare_data may reference it)
    full_pars = _merge_pars(model, pars, 1.0)
    if model.has_interpolation
        full_pars = _odin_setup_pars(model, full_pars)
    end

    # Determine element type from parameters (supports ForwardDiff Dual numbers)
    T_el = Float64
    for v in values(full_pars)
        if v isa Number
            T_el = promote_type(T_el, typeof(v))
        end
    end

    # Create initial state — reuse cache for Float64 path, allocate for AD
    if T_el === Float64
        state = unfilter._state_cache
        fill!(state, zero(Float64))
        rng = unfilter._rng_cache
    else
        state = zeros(T_el, n_state)
        rng = Random.Xoshiro()
    end
    _odin_initial!(model, state, full_pars, rng)

    ctrl = unfilter.ode_control
    data_times = unfilter.data.times
    t_start = unfilter.time_start
    t_end = data_times[end]

    rhs_fn! = (du, u, p, t) -> _odin_rhs!(model, du, u, p, t)

    # For Float64 path, use lightweight solvers (no DiffEq overhead)
    if T_el === Float64
        saveat = unfilter._saveat_cache

        if ctrl.solver === :sdirk
            # SDIRK4 path for stiff systems
            if unfilter._sdirk_workspace === nothing
                unfilter._sdirk_workspace = SDIRKWorkspace(n_state, Float64)
            end
            ws = unfilter._sdirk_workspace::SDIRKWorkspace{Float64}

            _sdirk_solve_core!(rhs_fn!, state, (t_start, t_end), full_pars, saveat,
                               ws, nothing, ctrl.atol, ctrl.rtol, ctrl.max_steps)

            log_likelihood = zero(Float64)
            for t_idx in 1:n_data
                state_t = @view ws.result_matrix[:, t_idx]
                data_t = unfilter.data.data[t_idx]
                ll_t = _odin_compare_data(model, state_t, full_pars, data_t, data_times[t_idx])
                log_likelihood += ll_t
            end
            return log_likelihood
        else
            # DP5 path (default)
            if unfilter._dp5_workspace === nothing
                unfilter._dp5_workspace = DP5Workspace(n_state, Float64)
            end
            ws = unfilter._dp5_workspace::DP5Workspace{Float64}

            _dp5_solve_core!(rhs_fn!, state, (t_start, t_end), full_pars, saveat,
                             ws, nothing, ctrl.atol, ctrl.rtol, ctrl.max_steps)

            log_likelihood = zero(Float64)
            for t_idx in 1:n_data
                state_t = @view ws.result_matrix[:, t_idx]
                data_t = unfilter.data.data[t_idx]
                ll_t = _odin_compare_data(model, state_t, full_pars, data_t, data_times[t_idx])
                log_likelihood += ll_t
            end
            return log_likelihood
        end
    else
        # AD path: use DifferentialEquations.jl (supports Dual numbers)
        prob = ODEProblem(rhs_fn!, state, (t_start, t_end), full_pars)
        sol = solve(prob, DP5();
            saveat=data_times,
            abstol=ctrl.atol, reltol=ctrl.rtol,
            maxiters=ctrl.max_steps,
            save_everystep=false,
            verbose=false,
        )

        if sol.retcode != ReturnCode.Success
            return T_el(-Inf)
        end

        log_likelihood = zero(T_el)
        for t_idx in 1:n_data
            state_t = sol.u[t_idx]
            data_t = unfilter.data.data[t_idx]
            ll_t = _odin_compare_data(model, state_t, full_pars, data_t, data_times[t_idx])
            log_likelihood += ll_t
        end
        return log_likelihood
    end
end


"""
    dust_unfilter_create(gen, group_data::Vector{FilterData}; kwargs...)

Create a multi-group deterministic unfilter.
"""
function dust_unfilter_create(
    gen::DustSystemGenerator{M},
    group_data::Vector{FilterData{D}};
    time_start::Float64=0.0,
    ode_control::DustODEControl=DustODEControl(),
) where {M, D}
    n_groups = length(group_data)
    n_state = gen.model.n_state
    saveat_cache = collect(Float64, group_data[1].times)
    return DustUnfilter{M,D}(gen, group_data[1], time_start, ode_control,
                              zeros(Float64, n_state), Random.Xoshiro(), nothing,
                              nothing, nothing, saveat_cache,
                              n_groups, group_data, nothing, nothing)
end

"""Run grouped unfilter: sum log-likelihoods across groups."""
function _run_grouped_unfilter!(unfilter::DustUnfilter, pars::NamedTuple;
                                save_snapshots=nothing)
    total_ll = 0.0
    orig_data = unfilter.data
    orig_saveat = unfilter._saveat_cache
    for g in 1:unfilter.n_groups
        gdata = unfilter.group_data[g]
        unfilter.data = gdata
        unfilter._saveat_cache = collect(Float64, gdata.times)
        ll = _unfilter_run_single!(unfilter, pars; save_snapshots=save_snapshots)
        total_ll += ll
    end
    unfilter.data = orig_data
    unfilter._saveat_cache = orig_saveat
    return total_ll
end

"""Retrieve saved snapshots from last unfilter run."""
last_snapshots(u::DustUnfilter) = u._last_snapshots

"""Retrieve saved trajectories from last unfilter run."""
last_trajectories(u::DustUnfilter) = u._last_trajectories
