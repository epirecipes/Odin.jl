# Bootstrap particle filter for stochastic models.

"""
    DustFilter{M}

A bootstrap particle filter for stochastic likelihood estimation.
"""
mutable struct DustFilter{M<:AbstractOdinModel, T<:AbstractFloat, D<:NamedTuple}
    generator::DustSystemGenerator{M}
    data::FilterData{D}
    time_start::T
    n_particles::Int
    dt::T
    seed::Union{Nothing, Int}
    save_trajectories::Bool
    # Cached system — reused across filter runs to avoid allocations
    _cached_sys::Union{Nothing, DustSystem}
    # Multi-group support
    n_groups::Int
    group_data::Union{Nothing, Vector{FilterData{D}}}
    # Snapshot/trajectory storage (populated after run)
    _last_snapshots::Union{Nothing, Dict{Float64, Matrix{Float64}}}
    _last_trajectories::Union{Nothing, Array{Float64, 3}}
end

"""
    dust_filter_create(generator, data; time_start, n_particles, dt, seed, save_trajectories)

Create a bootstrap particle filter.
"""
function dust_filter_create(
    gen::DustSystemGenerator{M},
    data::FilterData{D};
    time_start::Float64=0.0,
    n_particles::Int=100,
    dt::Float64=1.0,
    seed::Union{Nothing, Int}=nothing,
    save_trajectories::Bool=false,
) where {M, D}
    n_particles > 0 || throw(ArgumentError("dust_filter_create: n_particles must be positive, got $n_particles"))
    dt > 0 || throw(ArgumentError("dust_filter_create: dt must be positive, got $dt"))
    return DustFilter{M, Float64, D}(gen, data, time_start, n_particles, dt, seed, save_trajectories, nothing,
                                       1, nothing, nothing, nothing)
end

"""
    DustFilterState

State of a running particle filter (for reuse across likelihood evaluations).
"""
mutable struct DustFilterState{M, T, P}
    sys::DustSystem{M, T, P}
    log_likelihood::T
    trajectories::Union{Nothing, Array{T, 3}}
end

"""
    dust_likelihood_run!(filter, pars) -> Float64

Run the particle filter with given parameters and return the log-likelihood.
"""
function dust_likelihood_run!(filter::DustFilter, pars::NamedTuple;
                               save_snapshots::Union{Nothing, Vector{Float64}}=nothing)
    if filter.n_groups > 1 && filter.group_data !== nothing
        return _run_grouped_filter!(filter, pars; save_snapshots=save_snapshots)
    end
    sys = _get_or_create_sys!(filter, pars)
    dust_system_set_state_initial!(sys)
    np = filter.n_particles
    use_threads = np >= 4 && Threads.nthreads() > 1
    save_traj = filter.save_trajectories
    if use_threads
        result = _filter_inner_threaded!(sys, sys.pars, filter.data, np, save_traj, save_snapshots)
    else
        result = _filter_inner!(sys, sys.pars, filter.data, np, save_traj, save_snapshots)
    end
    filter._last_snapshots = result.snapshots
    filter._last_trajectories = result.trajectories
    return result.ll
end

function dust_likelihood_run!(filter::DustFilter, pars_vec::AbstractVector{<:NamedTuple};
                               save_snapshots::Union{Nothing, Vector{Float64}}=nothing)
    filter.n_groups > 1 && filter.group_data !== nothing ||
        throw(ArgumentError("Vector-of-parameters likelihoods require grouped filter data"))
    return _run_grouped_filter!(filter, pars_vec; save_snapshots=save_snapshots)
end

"""Type-stable inner loop of the particle filter."""
function _filter_inner!(sys::DustSystem{M,T}, pars::P, data::FilterData{D},
                        n_particles::Int, save_traj::Bool,
                        save_snapshots::Union{Nothing, Vector{Float64}}) where {M,T,P,D}
    n_data = length(data.times)
    n_state = sys.n_state
    log_likelihood = 0.0
    model = sys.generator.model
    dt = sys.dt

    # Extract ALL mutable struct fields into locals for type stability
    state = sys.state
    rngs = sys.rng
    log_weights = sys._work_weights
    indices = sys._work_indices
    state_tmp = sys._work_state_tmp
    state_col = sys._work_state_col
    state_next = sys._work_state_next
    zero_every = sys.zero_every

    trajectories = save_traj ?
        zeros(T, n_state, n_particles, n_data) : nothing
    snapshots = save_snapshots === nothing ? nothing : Dict{Float64, Matrix{T}}()

    for t_idx in 1:n_data
        target_time = data.times[t_idx]
        data_t = data.data[t_idx]

        # Inline _run_discrete! to avoid re-reading sys fields.
        # Tolerance `dt/2` guards against floating-point drift across many
        # accumulated `+= dt` steps: stop when within half a step of target.
        while sys.time < target_time - dt / 2
            # Inline _apply_zero_every!
            if !isempty(zero_every)
                @inbounds for entry in zero_every
                    period = entry.period
                    # `dt/4` (< dt/2) classifies the current step as the
                    # period boundary without double-firing on adjacent steps.
                    if period > 0 && abs(sys.time - round(sys.time / period) * period) < dt / 4
                        for p in 1:n_particles
                            for idx in entry.range
                                state[idx, p] = 0.0
                            end
                        end
                    end
                end
            end

            for p in 1:n_particles
                state_view = @view state[:, p]
                @inbounds for j in 1:n_state
                    state_next[j] = state_view[j]
                end
                _odin_update!(model, state_next, state_view, pars, sys.time, dt, rngs[p])
                @inbounds for j in 1:n_state
                    state_view[j] = state_next[j]
                end
            end
            sys.time += dt
        end

        # Compute weights
        @inbounds for p in 1:n_particles
            state_col_view = @view state[:, p]
            log_weights[p] = _odin_compare_data(model, state_col_view, pars, data_t, target_time)
        end

        ll_t = log_sum_exp(log_weights) - log(n_particles)
        log_likelihood += ll_t

        # Resample
        if t_idx < n_data
            _systematic_resample_inplace!(indices, log_weights, rngs[1])
            state_tmp .= state
            @inbounds for p in 1:n_particles
                src = indices[p]
                for j in 1:n_state
                    state[j, p] = state_tmp[j, src]
                end
            end
        end

        if trajectories !== nothing
            trajectories[:, :, t_idx] .= state
        end
        if snapshots !== nothing
            for st in save_snapshots
                if abs(target_time - st) < 1e-10
                    snapshots[st] = copy(state)
                end
            end
        end
    end

    return (; ll=log_likelihood, snapshots=snapshots, trajectories=trajectories)
end

"""Type-stable threaded inner loop of the particle filter."""
function _filter_inner_threaded!(sys::DustSystem{M,T}, pars::P, data::FilterData{D},
                                 n_particles::Int, save_traj::Bool,
                                 save_snapshots::Union{Nothing, Vector{Float64}}) where {M,T,P,D}
    n_data = length(data.times)
    n_state = sys.n_state
    log_likelihood = 0.0
    model = sys.generator.model
    dt = sys.dt
    nt = sys.n_threads

    # Extract ALL mutable struct fields into locals for type stability
    state = sys.state
    rngs = sys.rng
    log_weights = sys._work_weights
    indices = sys._work_indices
    state_tmp = sys._work_state_tmp
    zero_every = sys.zero_every

    # Per-thread resources
    thread_state_next = sys._thread_state_next
    thread_models = _make_thread_models(model, sys._thread_workspaces, nt)

    trajectories = save_traj ?
        zeros(T, n_state, n_particles, n_data) : nothing
    snapshots = save_snapshots === nothing ? nothing : Dict{Float64, Matrix{T}}()

    for t_idx in 1:n_data
        target_time = data.times[t_idx]
        data_t = data.data[t_idx]

        # Advance to target time
        while sys.time < target_time - dt / 2
            # Apply zero_every (sequential — modifies shared state)
            if !isempty(zero_every)
                @inbounds for entry in zero_every
                    period = entry.period
                    if period > 0 && abs(sys.time - round(sys.time / period) * period) < dt / 4
                        for p in 1:n_particles
                            for idx in entry.range
                                state[idx, p] = 0.0
                            end
                        end
                    end
                end
            end

            # SAFETY (thread): `t_now` is captured BEFORE the parallel region
            # so every thread reads a consistent value. `sys.time` must NOT be
            # mutated inside this loop. The single mutation below happens
            # after the implicit barrier at the end of `Threads.@threads`.
            t_now = sys.time
            Threads.@threads for p in 1:n_particles
                tid = Threads.threadid()
                sn = thread_state_next[tid]
                m = thread_models[tid]
                state_view = @view state[:, p]
                @inbounds for j in 1:n_state
                    sn[j] = state_view[j]
                end
                _odin_update!(m, sn, state_view, pars, t_now, dt, rngs[p])
                @inbounds for j in 1:n_state
                    state_view[j] = sn[j]
                end
            end
            sys.time += dt
        end

        # Compute weights (threaded — each particle writes to its own index `p`,
        # so there are no data races on `log_weights`).
        Threads.@threads for p in 1:n_particles
            tid = Threads.threadid()
            m = thread_models[tid]
            state_col_view = @view state[:, p]
            @inbounds log_weights[p] = _odin_compare_data(m, state_col_view, pars, data_t, target_time)
        end

        ll_t = log_sum_exp(log_weights) - log(n_particles)
        log_likelihood += ll_t

        # Resample (sequential — must remain single-threaded)
        if t_idx < n_data
            _systematic_resample_inplace!(indices, log_weights, rngs[1])
            state_tmp .= state
            @inbounds for p in 1:n_particles
                src = indices[p]
                for j in 1:n_state
                    state[j, p] = state_tmp[j, src]
                end
            end
        end

        if trajectories !== nothing
            trajectories[:, :, t_idx] .= state
        end
        if snapshots !== nothing
            for st in save_snapshots
                if abs(target_time - st) < 1e-10
                    snapshots[st] = copy(state)
                end
            end
        end
    end

    return (; ll=log_likelihood, snapshots=snapshots, trajectories=trajectories)
end

"""Get or create the cached system, resetting it for reuse."""
function _get_or_create_sys!(filter::DustFilter, pars::NamedTuple)
    if filter._cached_sys === nothing
        sys = dust_system_create(
            filter.generator, pars;
            n_particles=filter.n_particles,
            dt=filter.dt,
            time=filter.time_start,
            seed=filter.seed,
        )
        if filter.seed !== nothing
            sys._saved_rngs = [copy(r) for r in sys.rng]
        end
        filter._cached_sys = sys
        return sys
    else
        sys = filter._cached_sys
        _reset_system!(sys, pars, filter.seed; time=filter.time_start)
        return sys
    end
end


"""
    dust_filter_create(gen, group_data::Vector{FilterData}; kwargs...)

Create a multi-group particle filter.
"""
function dust_filter_create(
    gen::DustSystemGenerator{M},
    group_data::Vector{FilterData{D}};
    time_start::Float64=0.0,
    n_particles::Int=100,
    dt::Float64=1.0,
    seed::Union{Nothing, Int}=nothing,
    save_trajectories::Bool=false,
) where {M, D}
    n_particles > 0 || throw(ArgumentError("dust_filter_create: n_particles must be positive, got $n_particles"))
    dt > 0 || throw(ArgumentError("dust_filter_create: dt must be positive, got $dt"))
    !isempty(group_data) || throw(ArgumentError("dust_filter_create: group_data must be non-empty"))
    n_groups = length(group_data)
    filt = DustFilter{M, Float64, D}(gen, group_data[1], time_start, n_particles, dt, seed,
                                      save_trajectories, nothing, n_groups, group_data,
                                      nothing, nothing)
    return filt
end

"""Run grouped filter: sum log-likelihoods across groups."""
function _run_grouped_filter!(filter::DustFilter, pars::NamedTuple;
                              save_snapshots=nothing)
    total_ll = 0.0
    filter._last_snapshots = nothing
    filter._last_trajectories = nothing
    for g in 1:filter.n_groups
        gdata = filter.group_data[g]
        group_seed = filter.seed === nothing ? nothing : filter.seed + g - 1
        sys = dust_system_create(
            filter.generator, pars;
            n_particles=filter.n_particles,
            dt=filter.dt,
            seed=group_seed,
        )
        dust_system_set_state_initial!(sys)
        np = filter.n_particles
        result = _filter_inner!(sys, sys.pars, gdata, np, filter.save_trajectories, save_snapshots)
        total_ll += result.ll
        if result.snapshots !== nothing
            filter._last_snapshots = result.snapshots
        end
        if result.trajectories !== nothing
            filter._last_trajectories = result.trajectories
        end
    end
    return total_ll
end

function _run_grouped_filter!(filter::DustFilter, pars_vec::AbstractVector{<:NamedTuple};
                              save_snapshots=nothing)
    length(pars_vec) == filter.n_groups ||
        throw(ArgumentError("Expected $(filter.n_groups) parameter groups, got $(length(pars_vec))"))
    total_ll = 0.0
    filter._last_snapshots = nothing
    filter._last_trajectories = nothing
    for g in 1:filter.n_groups
        gdata = filter.group_data[g]
        group_seed = filter.seed === nothing ? nothing : filter.seed + g - 1
        sys = dust_system_create(
            filter.generator, pars_vec[g];
            n_particles=filter.n_particles,
            dt=filter.dt,
            seed=group_seed,
            time=filter.time_start,
        )
        dust_system_set_state_initial!(sys)
        np = filter.n_particles
        result = _filter_inner!(sys, sys.pars, gdata, np, filter.save_trajectories, save_snapshots)
        total_ll += result.ll
        if result.snapshots !== nothing
            filter._last_snapshots = result.snapshots
        end
        if result.trajectories !== nothing
            filter._last_trajectories = result.trajectories
        end
    end
    return total_ll
end

"""Retrieve saved snapshots from last filter/unfilter run."""
last_snapshots(f::DustFilter) = f._last_snapshots

"""Retrieve saved trajectories from last filter/unfilter run."""
last_trajectories(f::DustFilter) = f._last_trajectories
