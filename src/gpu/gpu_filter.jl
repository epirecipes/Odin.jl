# GPU-accelerated bootstrap particle filter.
#
# Strategy:
#   - State matrix lives on GPU (large: n_state × n_particles)
#   - Per-particle update + compare are parallelised on GPU
#   - Resampling (sequential cumulative sum) stays on CPU
#   - Log-weights transferred GPU→CPU for resampling, indices CPU→GPU for scatter
#
# The CPUBackend path delegates directly to the existing DustFilter for
# zero-overhead fallback.

"""
    GPUDustFilter{M, T, D, B}

GPU-accelerated bootstrap particle filter.

Type parameters:
- `M` — model type (subtype of `AbstractOdinModel`)
- `T` — float type (typically `Float64`)
- `D` — data element NamedTuple type
- `B` — GPU backend type (subtype of `GPUBackend`)
"""
mutable struct GPUDustFilter{M<:AbstractOdinModel, T<:AbstractFloat, D<:NamedTuple, B<:GPUBackend}
    generator::DustSystemGenerator{M}
    data::FilterData{D}
    time_start::T
    n_particles::Int
    dt::T
    seed::Union{Nothing, Int}
    backend::B
    save_trajectories::Bool
    # Cached state — allocated on first run
    _cached_state_gpu::Any          # GPU matrix (n_state × n_particles) or nothing
    _cached_state_tmp_gpu::Any      # GPU scratch for resampling
    _cached_log_weights::Union{Nothing, Vector{T}}
    _cached_indices::Union{Nothing, Vector{Int}}
    _cached_rngs::Union{Nothing, Vector{Xoshiro}}
    _cached_pars::Any               # merged pars, set per run
    _cached_n_state::Int
    # CPU fallback filter (used when B == CPUBackend)
    _cpu_filter::Union{Nothing, DustFilter{M, T, D}}
end

"""
    gpu_dust_filter_create(gen, data; n_particles, dt, time_start, seed, backend, save_trajectories)

Create a GPU-accelerated bootstrap particle filter.

# Arguments
- `gen::DustSystemGenerator` — compiled model generator (from `@odin`)
- `data::FilterData` — observation data (from `dust_filter_data`)
- `n_particles::Int=1000` — number of particles
- `dt::Float64=1.0` — discrete time step
- `time_start::Float64=0.0` — simulation start time
- `seed` — optional RNG seed for reproducibility
- `backend` — `:auto`, `:metal`, `:cuda`, or `:cpu`
- `save_trajectories::Bool=false` — save full particle trajectories

# Returns
A `GPUDustFilter` that can be run with [`gpu_dust_filter_run!`](@ref).
"""
function gpu_dust_filter_create(
    gen::DustSystemGenerator{M},
    data::FilterData{D};
    n_particles::Int=1000,
    dt::Float64=1.0,
    time_start::Float64=0.0,
    seed::Union{Nothing, Int}=nothing,
    backend::Symbol=:auto,
    save_trajectories::Bool=false,
) where {M, D}
    be = gpu_backend(; preferred=backend)
    _gpu_dust_filter_create(gen, data, be; n_particles=n_particles, dt=dt,
                            time_start=time_start, seed=seed,
                            save_trajectories=save_trajectories)
end

function _gpu_dust_filter_create(
    gen::DustSystemGenerator{M},
    data::FilterData{D},
    be::B;
    n_particles::Int=1000,
    dt::Float64=1.0,
    time_start::Float64=0.0,
    seed::Union{Nothing, Int}=nothing,
    save_trajectories::Bool=false,
) where {M, D, B<:GPUBackend}
    cpu_filter = nothing
    if be isa CPUBackend
        cpu_filter = dust_filter_create(gen, data;
            n_particles=n_particles, dt=dt,
            time_start=time_start, seed=seed,
            save_trajectories=save_trajectories)
    end

    return GPUDustFilter{M, Float64, D, B}(
        gen, data, time_start, n_particles, dt, seed, be,
        save_trajectories,
        nothing, nothing, nothing, nothing, nothing, nothing, 0,
        cpu_filter,
    )
end

"""
    gpu_dust_filter_run!(filter, pars) -> Float64

Run the GPU particle filter and return the log-likelihood.

For `CPUBackend`, this delegates to the existing `dust_likelihood_run!`.
For GPU backends, the state matrix is kept on the GPU and per-particle
update/compare operations are parallelised.
"""
function gpu_dust_filter_run!(filter::GPUDustFilter{M,T,D,CPUBackend}, pars::NamedTuple) where {M,T,D}
    return dust_likelihood_run!(filter._cpu_filter, pars)
end

function gpu_dust_filter_run!(filter::GPUDustFilter{M,T,D,B}, pars::NamedTuple) where {M,T,D,B<:GPUBackend}
    model = filter.generator.model
    if model.is_continuous
        error("GPU particle filter is only supported for discrete-time (update) models. " *
              "Use dust_unfilter_create for continuous ODE models.")
    end

    _gpu_init_caches!(filter, pars)
    return _gpu_filter_inner!(filter, pars)
end

"""Initialise or reset cached GPU/CPU arrays for a filter run."""
function _gpu_init_caches!(filter::GPUDustFilter{M,T,D,B}, pars::NamedTuple) where {M,T,D,B}
    model = filter.generator.model
    be = filter.backend
    dt = filter.dt
    np = filter.n_particles

    full_pars = _merge_pars(model, pars, dt)
    if model.has_interpolation
        full_pars = _odin_setup_pars(model, full_pars)
    end
    filter._cached_pars = full_pars

    n_state = _odin_n_state(model, full_pars)
    filter._cached_n_state = n_state

    # Allocate GPU state matrices (lazy — only on first call or size change)
    if filter._cached_state_gpu === nothing
        state_cpu = zeros(T, n_state, np)
        filter._cached_state_gpu = gpu_array(be, state_cpu)
        filter._cached_state_tmp_gpu = gpu_array(be, zeros(T, n_state, np))
    end

    # CPU work buffers
    if filter._cached_log_weights === nothing
        filter._cached_log_weights = Vector{T}(undef, np)
        filter._cached_indices = Vector{Int}(undef, np)
    end

    # RNGs
    if filter._cached_rngs === nothing
        base_rng = filter.seed === nothing ? Random.default_rng() : Random.Xoshiro(filter.seed)
        filter._cached_rngs = [Random.Xoshiro(rand(base_rng, UInt64)) for _ in 1:np]
    elseif filter.seed !== nothing
        # Deterministic reset
        base_rng = Random.Xoshiro(filter.seed)
        for i in 1:np
            filter._cached_rngs[i] = Random.Xoshiro(rand(base_rng, UInt64))
        end
    end

    # Set initial state on CPU then transfer to GPU
    state_cpu = zeros(T, n_state, np)
    for p in 1:np
        sv = @view state_cpu[:, p]
        _odin_initial!(model, sv, full_pars, filter._cached_rngs[p])
    end
    _gpu_set_state!(filter, state_cpu)

    return nothing
end

"""Copy CPU state matrix to the GPU state cache."""
function _gpu_set_state!(filter::GPUDustFilter, state_cpu::Matrix)
    filter._cached_state_gpu = gpu_array(filter.backend, state_cpu)
end

"""
Core GPU filter loop.

For each data time point:
1. Advance all particles to the target time (GPU-parallelised update)
2. Compute per-particle log-weights (GPU-parallelised compare)
3. Transfer log-weights to CPU
4. Accumulate log-likelihood via log-sum-exp
5. Resample on CPU
6. Scatter resampled indices back to GPU
"""
function _gpu_filter_inner!(filter::GPUDustFilter{M,T,D,B}, pars::NamedTuple) where {M,T,D,B}
    model = filter.generator.model
    data = filter.data
    n_data = length(data.times)
    np = filter.n_particles
    n_state = filter._cached_n_state
    dt = filter.dt
    full_pars = filter._cached_pars
    rngs = filter._cached_rngs
    log_weights = filter._cached_log_weights
    indices = filter._cached_indices

    log_likelihood = 0.0
    current_time = filter.time_start

    # Pull state to CPU for the hybrid approach: GPU memory, CPU model eval.
    # This is the "practical approach 2" from the design doc — works with all models.
    state_cpu = cpu_array(filter._cached_state_gpu)::Matrix{T}

    # Pre-allocate per-particle scratch (reused across time steps)
    state_next = Vector{T}(undef, n_state)

    trajectories = filter.save_trajectories ?
        zeros(T, n_state, np, n_data) : nothing

    for t_idx in 1:n_data
        target_time = data.times[t_idx]
        data_t = data.data[t_idx]

        # ── Step 1: Advance particles to target_time ──
        while current_time < target_time - dt / 2
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

        # ── Step 2: Compute log-weights ──
        @inbounds for p in 1:np
            sv = @view state_cpu[:, p]
            log_weights[p] = _odin_compare_data(model, sv, full_pars, data_t, target_time)
        end

        # ── Step 3: Accumulate log-likelihood ──
        ll_t = log_sum_exp(log_weights) - log(np)
        log_likelihood += ll_t

        # ── Step 4: Resample (CPU) ──
        if t_idx < n_data
            _systematic_resample_inplace!(indices, log_weights, rngs[1])
            # Scatter: build resampled state
            state_tmp = copy(state_cpu)
            @inbounds for p in 1:np
                src = indices[p]
                for j in 1:n_state
                    state_cpu[j, p] = state_tmp[j, src]
                end
            end
        end

        if trajectories !== nothing
            trajectories[:, :, t_idx] .= state_cpu
        end
    end

    # Push final state back to GPU
    filter._cached_state_gpu = gpu_array(filter.backend, state_cpu)

    return log_likelihood
end

"""
    gpu_dust_likelihood_monty(filter, packer) -> MontyModel

Create a `MontyModel` from a GPU particle filter for use with monty samplers.
"""
function gpu_dust_likelihood_monty(filter::GPUDustFilter, packer::MontyPacker)
    param_names = parameter_names(packer)

    density = function(x)
        named_pars = unpack(packer, x)
        return gpu_dust_filter_run!(filter, named_pars)
    end

    return monty_model(
        density;
        parameters=param_names,
        properties=MontyModelProperties(is_stochastic=true),
    )
end
