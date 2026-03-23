# Dust system types and interface — the core runtime engine.

"""
    DustSystemGenerator

A compiled model generator. Created by `@odin`, used to create `DustSystem` instances.
"""
struct DustSystemGenerator{M<:AbstractOdinModel}
    model::M
end

"""Stores range-to-period mapping for zero_every variables."""
struct ZeroEveryEntry
    range::UnitRange{Int}
    period::Int
end

"""
    DustSystem{M}

A running dust system with state, parameters, and RNG.
"""
mutable struct DustSystem{M<:AbstractOdinModel, T<:AbstractFloat, P}
    generator::DustSystemGenerator{M}
    state::Matrix{T}            # n_state × n_particles
    pars::P
    time::T
    dt::T
    n_particles::Int
    n_state::Int
    n_output::Int
    rng::Vector{Xoshiro}
    state_names::Vector{Symbol}
    output_names::Vector{Symbol}
    zero_every::Vector{ZeroEveryEntry}
    # Pre-allocated work buffers (avoids per-step allocations)
    _work_state_next::Vector{T}
    _work_state_col::Vector{T}
    _work_weights::Vector{T}     # for resampling
    _work_indices::Vector{Int}   # for resampling
    _work_state_tmp::Matrix{T}   # for resampling state copy
    _saved_rngs::Union{Nothing, Vector{Xoshiro}} # lazily initialized on first reset
    _dp5_workspace::Union{Nothing, DP5Workspace{T}}  # cached ODE solver workspace
    _thread_dp5_workspaces::Vector{Union{Nothing, DP5Workspace{T}}}  # per-thread ODE workspaces
    _sdirk_workspace::Union{Nothing, SDIRKWorkspace{T}}  # cached stiff ODE solver workspace
    _thread_sdirk_workspaces::Vector{Union{Nothing, SDIRKWorkspace{T}}}  # per-thread stiff ODE workspaces
    _sde_workspace::Union{Nothing, SDEWorkspace{T}}  # cached SDE solver workspace
    _thread_sde_workspaces::Vector{Union{Nothing, SDEWorkspace{T}}}  # per-thread SDE workspaces
    # Per-thread work buffers for threaded particle loops
    _thread_state_next::Vector{Vector{T}}
    _thread_output_buf::Vector{Vector{T}}
    _thread_workspaces::Vector{Dict{Symbol, Array}}
    n_threads::Int
end

"""
    dust_system_create(generator, pars; n_particles=1, dt=1.0, time=0.0, seed=nothing, n_groups=1)

Create a new dust system from a generator and parameters.
"""
function dust_system_create(
    gen::DustSystemGenerator{M},
    pars::NamedTuple;
    n_particles::Int=1,
    dt::Float64=1.0,
    time::Float64=0.0,
    seed::Union{Nothing, Int}=nothing,
) where {M}
    model = gen.model

    # Merge parameter defaults with user pars + dt
    full_pars = _merge_pars(model, pars, dt)

    # Build interpolators if model has interpolated variables
    if model.has_interpolation
        full_pars = _odin_setup_pars(model, full_pars)
    end

    # Compute n_state dynamically (supports array state variables)
    n_state = _odin_n_state(model, full_pars)

    # Compute state names dynamically
    state_names = _odin_state_names(model, full_pars)

    # Compute output count + names
    n_output = 0
    output_names = Symbol[]
    if model.has_output
        try
            n_output = _odin_n_output(model, full_pars)
            output_names = _odin_output_names(model, full_pars)
        catch
        end
    end

    # Compute zero_every info if available
    ze = ZeroEveryEntry[]
    try
        ze_ranges = _odin_zero_every(model, full_pars)
        for (k, v) in ze_ranges
            push!(ze, ZeroEveryEntry(k, v))
        end
    catch
        # No zero_every method defined — that's fine
    end

    # Initialise RNGs
    base_rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(seed)
    rngs = [Random.Xoshiro(rand(base_rng, UInt64)) for _ in 1:n_particles]

    state = zeros(Float64, n_state, n_particles)

    # Pre-allocate work buffers
    work_state_next = Vector{Float64}(undef, n_state)
    work_state_col  = Vector{Float64}(undef, n_state)
    work_weights    = Vector{Float64}(undef, n_particles)
    work_indices    = Vector{Int}(undef, n_particles)
    work_state_tmp  = Matrix{Float64}(undef, n_state, n_particles)

    # Pre-allocate per-thread work buffers for threaded particle loops.
    # Use maxthreadid() not nthreads(): threadid() can exceed nthreads()
    # due to interactive threads in Julia ≥1.9.
    nt = Threads.maxthreadid()
    thread_state_next = [Vector{Float64}(undef, n_state) for _ in 1:nt]
    thread_output_buf = [n_output > 0 ? Vector{Float64}(undef, n_output) : Float64[] for _ in 1:nt]
    thread_workspaces = [Dict{Symbol, Array}() for _ in 1:nt]
    thread_dp5_workspaces = Union{Nothing, DP5Workspace{Float64}}[nothing for _ in 1:nt]
    thread_sdirk_workspaces = Union{Nothing, SDIRKWorkspace{Float64}}[nothing for _ in 1:nt]
    thread_sde_workspaces = Union{Nothing, SDEWorkspace{Float64}}[nothing for _ in 1:nt]

    sys = DustSystem(
        gen, state, full_pars, time, dt, n_particles, n_state, n_output,
        rngs, state_names, output_names, ze,
        work_state_next, work_state_col, work_weights, work_indices, work_state_tmp,
        nothing,  # _saved_rngs: lazily initialized on first reset
        nothing,  # _dp5_workspace: lazily initialized for ODE models
        thread_dp5_workspaces,
        nothing,  # _sdirk_workspace: lazily initialized for stiff ODE models
        thread_sdirk_workspaces,
        nothing,  # _sde_workspace: lazily initialized for SDE models
        thread_sde_workspaces,
        thread_state_next, thread_output_buf, thread_workspaces, nt,
    )

    return sys
end

function _merge_pars(model::AbstractOdinModel, user_pars::NamedTuple, dt::Float64)
    # Add dt to pars if not present
    if !haskey(user_pars, :dt)
        user_pars = merge(user_pars, (dt=dt,))
    end
    return user_pars
end

"""Reset system in-place for reuse (avoids allocations in filter hot path)."""
function _reset_system!(sys::DustSystem, pars::NamedTuple, seed::Union{Nothing, Int})
    model = sys.generator.model
    full_pars = _merge_pars(model, pars, sys.dt)
    if model.has_interpolation
        full_pars = _odin_setup_pars(model, full_pars)
    end
    sys.pars = full_pars
    sys.time = zero(sys.time)

    # Restore RNG states from saved snapshot (zero-alloc via copy!)
    if seed !== nothing
        if sys._saved_rngs === nothing
            # First reset — snapshot current RNG states
            sys._saved_rngs = [copy(r) for r in sys.rng]
        end
        @inbounds for i in 1:sys.n_particles
            copy!(sys.rng[i], sys._saved_rngs[i])
        end
    end

    # Reset state
    sys.state .= 0.0
    return sys
end

"""
    dust_system_set_state_initial!(sys)

Set the state of all particles to the initial conditions.
"""
function dust_system_set_state_initial!(sys::DustSystem)
    model = sys.generator.model
    for p in 1:sys.n_particles
        state_view = @view sys.state[:, p]
        _odin_initial!(model, state_view, sys.pars, sys.rng[p])
    end
    return nothing
end

"""
    dust_system_state(sys) -> Matrix{Float64}

Get the current state matrix (n_state × n_particles).
"""
function dust_system_state(sys::DustSystem)
    return copy(sys.state)
end

"""
    dust_system_set_state!(sys, state)

Set the state matrix directly.
"""
function dust_system_set_state!(sys::DustSystem, state::AbstractMatrix)
    sys.state .= state
    return nothing
end

"""
    dust_system_set_state!(sys, state::AbstractVector)

Set state from a vector (broadcasts to all particles).
"""
function dust_system_set_state!(sys::DustSystem, state::AbstractVector)
    for p in 1:sys.n_particles
        sys.state[:, p] .= state
    end
    return nothing
end

"""
    dust_system_run_to_time!(sys, time)

Advance the system to the given time.
"""
function dust_system_run_to_time!(sys::DustSystem, time::Float64)
    model = sys.generator.model
    if model.is_continuous
        _run_continuous!(sys, time)
    else
        # Pass pars explicitly for type stability
        _run_discrete!(sys, sys.pars, time)
    end
    return nothing
end

function _run_discrete!(sys::DustSystem{M, T}, pars::P, target_time::T) where {M, T, P}
    model = sys.generator.model
    dt = sys.dt
    n_state = sys.n_state
    np = sys.n_particles
    state = sys.state
    state_next = sys._work_state_next

    while sys.time < target_time - dt / 2
        _apply_zero_every!(sys, sys.time)

        for p in 1:np
            state_view = @view state[:, p]
            @inbounds for j in 1:n_state
                state_next[j] = state_view[j]  # copy forward for partial updates
            end
            _odin_update!(model, state_next, state_view, pars, sys.time, dt, sys.rng[p])
            @inbounds for j in 1:n_state
                state_view[j] = state_next[j]
            end
        end
        sys.time += dt
    end
    return nothing
end

function _run_continuous!(sys::DustSystem{M, T, P}, target_time::T) where {M, T, P}
    model = sys.generator.model
    for p in 1:sys.n_particles
        state_vec = sys.state[:, p]
        prob = OrdinaryDiffEq.ODEProblem(
            (du, u, pars, t) -> _odin_rhs!(model, du, u, pars, t),
            state_vec,
            (sys.time, target_time),
            sys.pars,
        )
        sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(); save_everystep=false)
        sys.state[:, p] .= sol.u[end]
    end
    sys.time = target_time
    return nothing
end

function _apply_zero_every!(sys::DustSystem, time::Float64)
    isempty(sys.zero_every) && return nothing
    @inbounds for entry in sys.zero_every
        period = entry.period
        if period > 0 && abs(time - round(time / period) * period) < sys.dt / 4
            for p in 1:sys.n_particles
                for idx in entry.range
                    sys.state[idx, p] = 0.0
                end
            end
        end
    end
    return nothing
end

"""
    dust_system_compare_data(sys, data) -> Vector{Float64}

Compute log-likelihood for each particle given data at current time.
"""
function dust_system_compare_data(sys::DustSystem, data::NamedTuple)
    model = sys.generator.model
    ll = zeros(Float64, sys.n_particles)
    for p in 1:sys.n_particles
        state_view = @view sys.state[:, p]
        ll[p] = _odin_compare_data(model, state_view, sys.pars, data, sys.time)
    end
    return ll
end
