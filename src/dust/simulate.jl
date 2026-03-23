# Simulation interface: run systems and collect trajectories.

"""
    dust_system_simulate(sys, times; solver=:dp5, events=nothing) -> Array{Float64, 3}

Simulate the system, recording state at each time in `times`.
Returns array of shape (n_state, n_particles, n_times).

For continuous (ODE) models, `solver` selects the integration method:
- `:dp5` — Dormand-Prince 5(4), explicit, for non-stiff systems (default)
- `:sdirk` — SDIRK4, L-stable implicit, for stiff systems

Pass an `EventSet` via `events` to enable discontinuities and callbacks.
"""
function dust_system_simulate(sys::DustSystem, times::AbstractVector{Float64};
                              solver::Symbol=:dp5, events::Union{Nothing, EventSet}=nothing)
    n_times = length(times)
    n_rows = sys.n_state + sys.n_output
    output = zeros(Float64, n_rows, sys.n_particles, n_times)
    model = sys.generator.model

    if model.is_continuous
        return _simulate_continuous(sys, times, output, solver; events=events)
    else
        return _simulate_discrete(sys, times, output)
    end
end

function _simulate_discrete(sys::DustSystem{M, T, P}, times::AbstractVector{T}, output::Array{T, 3}) where {M, T, P}
    model = sys.generator.model
    dt = sys.dt
    pars = sys.pars
    n_times = length(times)
    n_state = sys.n_state
    n_output = sys.n_output
    np = sys.n_particles
    state = sys.state
    use_threads = np >= 4 && Threads.nthreads() > 1

    if use_threads
        return _simulate_discrete_threaded(sys, times, output)
    end

    # Serial path (original)
    state_next = sys._work_state_next
    output_buf = n_output > 0 ? Vector{T}(undef, n_output) : T[]
    time_idx = 1

    if time_idx <= n_times && abs(sys.time - times[time_idx]) < dt / 2
        @inbounds for p in 1:np
            for j in 1:n_state
                output[j, p, time_idx] = state[j, p]
            end
        end
        if n_output > 0
            for p in 1:np
                state_view = @view state[:, p]
                _odin_output!(model, output_buf, state_view, pars, sys.time)
                @inbounds for j in 1:n_output
                    output[n_state + j, p, time_idx] = output_buf[j]
                end
            end
        end
        time_idx += 1
    end

    while time_idx <= n_times
        target = times[time_idx]
        while sys.time < target - dt / 2
            _apply_zero_every!(sys, sys.time)
            @inbounds for p in 1:np
                state_view = @view state[:, p]
                for j in 1:n_state
                    state_next[j] = state_view[j]
                end
                _odin_update!(model, state_next, state_view, pars, sys.time, dt, sys.rng[p])
                for j in 1:n_state
                    state_view[j] = state_next[j]
                end
            end
            sys.time += dt
        end

        @inbounds for p in 1:np
            for j in 1:n_state
                output[j, p, time_idx] = state[j, p]
            end
        end
        if n_output > 0
            for p in 1:np
                state_view = @view state[:, p]
                _odin_output!(model, output_buf, state_view, pars, sys.time)
                @inbounds for j in 1:n_output
                    output[n_state + j, p, time_idx] = output_buf[j]
                end
            end
        end
        time_idx += 1
    end

    return output
end

function _simulate_discrete_threaded(sys::DustSystem{M, T, P}, times::AbstractVector{T}, output::Array{T, 3}) where {M, T, P}
    model = sys.generator.model
    dt = sys.dt
    pars = sys.pars
    n_times = length(times)
    n_state = sys.n_state
    n_output = sys.n_output
    np = sys.n_particles
    state = sys.state
    rngs = sys.rng
    thread_state_next = sys._thread_state_next
    thread_output_buf = sys._thread_output_buf
    time_idx = 1

    # Create per-thread model copies (shallow copy with separate workspace)
    nt = sys.n_threads
    thread_models = _make_thread_models(model, sys._thread_workspaces, nt)

    # Record initial state
    if time_idx <= n_times && abs(sys.time - times[time_idx]) < dt / 2
        @inbounds for p in 1:np
            for j in 1:n_state
                output[j, p, time_idx] = state[j, p]
            end
        end
        if n_output > 0
            Threads.@threads for p in 1:np
                tid = Threads.threadid()
                obuf = thread_output_buf[tid]
                sv = @view state[:, p]
                _odin_output!(thread_models[tid], obuf, sv, pars, sys.time)
                @inbounds for j in 1:n_output
                    output[n_state + j, p, time_idx] = obuf[j]
                end
            end
        end
        time_idx += 1
    end

    while time_idx <= n_times
        target = times[time_idx]
        while sys.time < target - dt / 2
            _apply_zero_every!(sys, sys.time)
            t_now = sys.time

            Threads.@threads for p in 1:np
                tid = Threads.threadid()
                sn = thread_state_next[tid]
                m = thread_models[tid]
                sv = @view state[:, p]
                @inbounds for j in 1:n_state
                    sn[j] = sv[j]
                end
                _odin_update!(m, sn, sv, pars, t_now, dt, rngs[p])
                @inbounds for j in 1:n_state
                    sv[j] = sn[j]
                end
            end
            sys.time += dt
        end

        @inbounds for p in 1:np
            for j in 1:n_state
                output[j, p, time_idx] = state[j, p]
            end
        end
        if n_output > 0
            Threads.@threads for p in 1:np
                tid = Threads.threadid()
                obuf = thread_output_buf[tid]
                sv = @view state[:, p]
                _odin_output!(thread_models[tid], obuf, sv, pars, sys.time)
                @inbounds for j in 1:n_output
                    output[n_state + j, p, time_idx] = obuf[j]
                end
            end
        end
        time_idx += 1
    end

    return output
end

"""Create per-thread shallow copies of model, each with its own workspace Dict."""
function _make_thread_models(model::M, thread_workspaces::Vector{Dict{Symbol, Array}}, nt::Int) where M <: AbstractOdinModel
    models = Vector{M}(undef, nt)
    nf = fieldcount(M)
    for t in 1:nt
        m = ccall(:jl_new_struct_uninit, Any, (Any,), M)::M
        for i in 1:nf
            if fieldname(M, i) === :_workspace
                setfield!(m, i, thread_workspaces[t])
            else
                setfield!(m, i, getfield(model, i))
            end
        end
        models[t] = m
    end
    return models
end

function _simulate_continuous(sys::DustSystem{M, T, P}, times::AbstractVector{T}, output::Array{T, 3}, solver::Symbol=:dp5; events::Union{Nothing, EventSet}=nothing) where {M, T, P}
    np = sys.n_particles
    use_threads = np >= 4 && Threads.nthreads() > 1 && !_has_events(events)

    if use_threads
        return _simulate_continuous_threaded(sys, times, output, solver)
    end

    model = sys.generator.model
    n_state = sys.n_state
    n_output = sys.n_output
    output_buf = sys._thread_output_buf[1]
    state_vec = sys._work_state_next

    rhs_fn! = (du, u, pars, t) -> _odin_rhs!(model, du, u, pars, t)

    if solver === :sdirk
        if sys._sdirk_workspace === nothing
            sys._sdirk_workspace = SDIRKWorkspace(n_state, T)
        end
        ws = sys._sdirk_workspace::SDIRKWorkspace{T}

        for p in 1:np
            @inbounds for j in 1:n_state
                state_vec[j] = sys.state[j, p]
            end

            _sdirk_solve_core!(rhs_fn!, state_vec, (sys.time, last(times)), sys.pars,
                               times, ws, nothing, T(1e-6), T(1e-6), 100000)

            for ti in 1:length(times)
                @inbounds for j in 1:n_state
                    output[j, p, ti] = ws.result_matrix[j, ti]
                end
                if n_output > 0
                    sv = @view ws.result_matrix[:, ti]
                    _odin_output!(model, output_buf, sv, sys.pars, times[ti])
                    @inbounds for j in 1:n_output
                        output[n_state + j, p, ti] = output_buf[j]
                    end
                end
            end
        end
    else
        if sys._dp5_workspace === nothing
            sys._dp5_workspace = DP5Workspace(n_state, T)
        end
        ws = sys._dp5_workspace::DP5Workspace{T}

        for p in 1:np
            @inbounds for j in 1:n_state
                state_vec[j] = sys.state[j, p]
            end

            if _has_events(events)
                _dp5_solve_events!(rhs_fn!, state_vec, (sys.time, last(times)), sys.pars,
                                   times, ws, nothing, T(1e-6), T(1e-6), 100000, events)
            else
                _dp5_solve_core!(rhs_fn!, state_vec, (sys.time, last(times)), sys.pars,
                                 times, ws, nothing, T(1e-6), T(1e-6), 100000)
            end

            for ti in 1:length(times)
                @inbounds for j in 1:n_state
                    output[j, p, ti] = ws.result_matrix[j, ti]
                end
                if n_output > 0
                    sv = @view ws.result_matrix[:, ti]
                    _odin_output!(model, output_buf, sv, sys.pars, times[ti])
                    @inbounds for j in 1:n_output
                        output[n_state + j, p, ti] = output_buf[j]
                    end
                end
            end
        end
    end

    sys.time = last(times)
    return output
end

function _simulate_continuous_threaded(sys::DustSystem{M, T, P}, times::AbstractVector{T}, output::Array{T, 3}, solver::Symbol=:dp5) where {M, T, P}
    model = sys.generator.model
    n_state = sys.n_state
    n_output = sys.n_output
    np = sys.n_particles
    nt = sys.n_threads

    # Per-thread resources
    thread_models = _make_thread_models(model, sys._thread_workspaces, nt)

    if solver === :sdirk
        thread_ws = sys._thread_sdirk_workspaces
        for tid in 1:nt
            if thread_ws[tid] === nothing
                thread_ws[tid] = SDIRKWorkspace(n_state, T)
            end
        end

        Threads.@threads for p in 1:np
            tid = Threads.threadid()
            ws = thread_ws[tid]::SDIRKWorkspace{T}
            m = thread_models[tid]
            state_vec = sys._thread_state_next[tid]
            obuf = sys._thread_output_buf[tid]

            rhs_fn! = (du, u, pars, t) -> _odin_rhs!(m, du, u, pars, t)

            @inbounds for j in 1:n_state
                state_vec[j] = sys.state[j, p]
            end

            _sdirk_solve_core!(rhs_fn!, state_vec, (sys.time, last(times)), sys.pars,
                               times, ws, nothing, T(1e-6), T(1e-6), 100000)

            for ti in 1:length(times)
                @inbounds for j in 1:n_state
                    output[j, p, ti] = ws.result_matrix[j, ti]
                end
                if n_output > 0
                    sv = @view ws.result_matrix[:, ti]
                    _odin_output!(m, obuf, sv, sys.pars, times[ti])
                    @inbounds for j in 1:n_output
                        output[n_state + j, p, ti] = obuf[j]
                    end
                end
            end
        end
    else
        thread_dp5 = sys._thread_dp5_workspaces
        for tid in 1:nt
            if thread_dp5[tid] === nothing
                thread_dp5[tid] = DP5Workspace(n_state, T)
            end
        end

        Threads.@threads for p in 1:np
            tid = Threads.threadid()
            ws = thread_dp5[tid]::DP5Workspace{T}
            m = thread_models[tid]
            state_vec = sys._thread_state_next[tid]
            obuf = sys._thread_output_buf[tid]

            rhs_fn! = (du, u, pars, t) -> _odin_rhs!(m, du, u, pars, t)

            @inbounds for j in 1:n_state
                state_vec[j] = sys.state[j, p]
            end

            _dp5_solve_core!(rhs_fn!, state_vec, (sys.time, last(times)), sys.pars,
                             times, ws, nothing, T(1e-6), T(1e-6), 100000)

            for ti in 1:length(times)
                @inbounds for j in 1:n_state
                    output[j, p, ti] = ws.result_matrix[j, ti]
                end
                if n_output > 0
                    sv = @view ws.result_matrix[:, ti]
                    _odin_output!(m, obuf, sv, sys.pars, times[ti])
                    @inbounds for j in 1:n_output
                        output[n_state + j, p, ti] = obuf[j]
                    end
                end
            end
        end
    end

    sys.time = last(times)
    return output
end

"""
    dust_system_simulate(gen::DustSystemGenerator, pars; times, dt, seed, n_particles, system, solver)

Convenience: create system (or reuse `system`), set initial state, simulate.
Pass `system` from a previous call to avoid system creation overhead.
"""
function dust_system_simulate(
    gen::DustSystemGenerator,
    pars::NamedTuple;
    times::AbstractVector{Float64},
    dt::Float64=1.0,
    seed::Union{Nothing, Int}=nothing,
    n_particles::Int=1,
    system::Union{Nothing, DustSystem}=nothing,
    solver::Symbol=:dp5,
)
    if system !== nothing && system.n_particles == n_particles
        _reset_system!(system, pars, seed)
        dust_system_set_state_initial!(system)
        return dust_system_simulate(system, times; solver=solver)
    else
        sys = dust_system_create(gen, pars; n_particles=n_particles, dt=dt, seed=seed)
        dust_system_set_state_initial!(sys)
        return dust_system_simulate(sys, times; solver=solver)
    end
end
