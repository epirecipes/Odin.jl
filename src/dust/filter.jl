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
    return DustFilter{M, Float64, D}(gen, data, time_start, n_particles, dt, seed, save_trajectories, nothing)
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
function dust_likelihood_run!(filter::DustFilter, pars::NamedTuple)
    sys = _get_or_create_sys!(filter, pars)
    dust_system_set_state_initial!(sys)
    np = filter.n_particles
    use_threads = np >= 4 && Threads.nthreads() > 1
    # Pass pars explicitly to maintain type stability through the inner loop
    if use_threads
        return _filter_inner_threaded!(sys, sys.pars, filter.data, np, filter.save_trajectories)
    else
        return _filter_inner!(sys, sys.pars, filter.data, np, filter.save_trajectories)
    end
end

"""Type-stable inner loop of the particle filter."""
function _filter_inner!(sys::DustSystem{M,T}, pars::P, data::FilterData{D},
                        n_particles::Int, save_traj::Bool) where {M,T,P,D}
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

    for t_idx in 1:n_data
        target_time = data.times[t_idx]
        data_t = data.data[t_idx]

        # Inline _run_discrete! to avoid re-reading sys fields
        while sys.time < target_time - dt / 2
            # Inline _apply_zero_every!
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
    end

    return log_likelihood
end

"""Type-stable threaded inner loop of the particle filter."""
function _filter_inner_threaded!(sys::DustSystem{M,T}, pars::P, data::FilterData{D},
                                 n_particles::Int, save_traj::Bool) where {M,T,P,D}
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

        # Compute weights (threaded — each particle writes to its own index)
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
    end

    return log_likelihood
end

"""Get or create the cached system, resetting it for reuse."""
function _get_or_create_sys!(filter::DustFilter, pars::NamedTuple)
    if filter._cached_sys === nothing
        sys = dust_system_create(
            filter.generator, pars;
            n_particles=filter.n_particles,
            dt=filter.dt,
            seed=filter.seed,
        )
        filter._cached_sys = sys
        return sys
    else
        sys = filter._cached_sys
        _reset_system!(sys, pars, filter.seed)
        return sys
    end
end
