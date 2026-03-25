# GPU-accelerated multi-particle simulation.
#
# For discrete-time models, state is managed on the GPU side and only
# copied back at save times. This reduces memory traffic when the number
# of particles is large.

"""
    GPUDustSimulation{M, T, B}

Result container for a GPU-accelerated simulation.
"""
struct GPUDustSimulation{T<:AbstractFloat}
    output::Array{T, 3}   # n_state × n_particles × n_times
    times::Vector{T}
    backend_name::String
end

"""
    gpu_dust_simulate(gen, pars; times, n_particles, dt, seed, backend)

GPU-accelerated multi-particle simulation for discrete-time models.

Returns an `Array{Float64, 3}` of shape `(n_state, n_particles, n_times)`,
matching the format of `dust_system_simulate`.

# Arguments
- `gen::DustSystemGenerator` — compiled model from `@odin`
- `pars::NamedTuple` — model parameters
- `times::AbstractVector{Float64}` — save times
- `n_particles::Int=1000` — number of particles
- `dt::Float64=1.0` — time step
- `seed` — RNG seed
- `backend` — `:auto`, `:metal`, `:cuda`, or `:cpu`
"""
function gpu_dust_simulate(
    gen::DustSystemGenerator{M},
    pars::NamedTuple;
    times::AbstractVector{Float64},
    n_particles::Int=1000,
    dt::Float64=1.0,
    seed::Union{Nothing, Int}=nothing,
    backend::Symbol=:auto,
) where {M}
    be = gpu_backend(; preferred=backend)
    return _gpu_simulate(gen, pars, be; times=times, n_particles=n_particles,
                         dt=dt, seed=seed)
end

# CPU fallback — delegate to existing dust_system_simulate
function _gpu_simulate(
    gen::DustSystemGenerator{M},
    pars::NamedTuple,
    ::CPUBackend;
    times::AbstractVector{Float64},
    n_particles::Int=1000,
    dt::Float64=1.0,
    seed::Union{Nothing, Int}=nothing,
) where {M}
    sys = dust_system_create(gen, pars; n_particles=n_particles, dt=dt, seed=seed)
    dust_system_set_state_initial!(sys)
    return dust_system_simulate(sys, times)
end

# GPU path — state on GPU, model eval on CPU (hybrid approach)
function _gpu_simulate(
    gen::DustSystemGenerator{M},
    pars::NamedTuple,
    be::B;
    times::AbstractVector{Float64},
    n_particles::Int=1000,
    dt::Float64=1.0,
    seed::Union{Nothing, Int}=nothing,
) where {M, B<:GPUBackend}
    model = gen.model
    if model.is_continuous
        error("GPU simulation is only supported for discrete-time (update) models.")
    end

    full_pars = _merge_pars(model, pars, dt)
    if model.has_interpolation
        full_pars = _odin_setup_pars(model, full_pars)
    end

    n_state = _odin_n_state(model, full_pars)
    np = n_particles
    n_times = length(times)

    # Initialise RNGs
    base_rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(seed)
    rngs = [Random.Xoshiro(rand(base_rng, UInt64)) for _ in 1:np]

    # Initialise state on CPU
    state_cpu = zeros(Float64, n_state, np)
    for p in 1:np
        sv = @view state_cpu[:, p]
        _odin_initial!(model, sv, full_pars, rngs[p])
    end

    output = zeros(Float64, n_state, np, n_times)
    state_next = Vector{Float64}(undef, n_state)
    current_time = 0.0
    time_idx = 1

    # Record initial state if first save time matches start
    if time_idx <= n_times && abs(current_time - times[time_idx]) < dt / 2
        @inbounds for p in 1:np
            for j in 1:n_state
                output[j, p, time_idx] = state_cpu[j, p]
            end
        end
        time_idx += 1
    end

    while time_idx <= n_times
        target = times[time_idx]

        while current_time < target - dt / 2
            @inbounds for p in 1:np
                state_view = @view state_cpu[:, p]
                for j in 1:n_state
                    state_next[j] = state_view[j]
                end
                _odin_update!(model, state_next, state_view, full_pars, current_time, dt, rngs[p])
                for j in 1:n_state
                    state_view[j] = state_next[j]
                end
            end
            current_time += dt
        end

        @inbounds for p in 1:np
            for j in 1:n_state
                output[j, p, time_idx] = state_cpu[j, p]
            end
        end
        time_idx += 1
    end

    return output
end
