# GPU Acceleration

Odin.jl supports GPU-accelerated simulation and particle filtering via a
backend abstraction layer. Currently supported backends are **Apple Metal**
(via Metal.jl), **NVIDIA CUDA**, and **AMD GPU**, plus a **CPU fallback**.

## Backend Selection

```julia
# Auto-detect best available backend
backend = gpu_backend()

# Prefer a specific backend
backend = gpu_backend(preferred=:metal)
backend = gpu_backend(preferred=:cuda)
backend = gpu_backend(preferred=:cpu)   # force CPU

# Query availability
has_gpu()                    # true if any GPU backend is registered
available_gpu_backends()     # e.g., [:metal]
backend_name(backend)        # e.g., "Metal"
```

GPU backends are registered automatically when the corresponding extension is
loaded (e.g., `using Metal` activates the `OdinMetalExt` extension).

## GPU Particle Filter

Run a bootstrap particle filter on the GPU for massively parallel likelihood
evaluation:

```julia
# Create GPU filter
gpu_filt = gpu_Likelihood(gen, data;
    time_start = 0.0,
    n_particles = 10_000,
    dt = 0.25,
    seed = 42,
    backend = gpu_backend(),
)

# Run filter — returns log-likelihood
ll = gpu_dust_filter_run!(gpu_filt, pars)
```

### Monty Bridge

Wrap the GPU filter as a [`MontyModel`](@ref) for MCMC:

```julia
packer = Packer([:beta, :gamma]; fixed=(N=1000.0, I0=10.0))
likelihood = gpu_as_model(gpu_filt, packer)
posterior = likelihood + prior
samples = sample(posterior, sampler, 5000)
```

## GPU Simulation

Run multi-particle simulation entirely on the GPU:

```julia
result = gpu_dust_simulate(gen, pars;
    times = 0.0:1.0:100.0,
    n_particles = 1000,
    dt = 0.25,
    seed = 42,
    backend = gpu_backend(),
)
# result.output — (n_state × n_times × n_particles) array
# result.times  — time points
```

## Array Transfer

Move arrays between CPU and GPU:

```julia
gpu_x = gpu_array(backend, x)   # CPU → GPU
cpu_x = cpu_array(gpu_x)        # GPU → CPU
T = gpu_array_type(backend)     # e.g., MtlArray
```

## Backend Types

| Type | Description |
|------|-------------|
| [`CPUBackend`](@ref) | CPU fallback (no GPU) |
| [`MetalBackend`](@ref) | Apple Metal via Metal.jl |
| [`CUDABackend`](@ref) | NVIDIA CUDA (requires CUDA.jl) |
| [`AMDGPUBackend`](@ref) | AMD GPU (requires AMDGPU.jl) |

## API Reference

```@docs
Odin.GPUBackend
Odin.CPUBackend
Odin.MetalBackend
Odin.CUDABackend
Odin.AMDGPUBackend
Odin.gpu_backend
Odin.has_gpu
Odin.available_gpu_backends
Odin.backend_name
Odin.gpu_array
Odin.cpu_array
Odin.gpu_array_type
Odin.GPUDustFilter
Odin.gpu_dust_filter_create
Odin.gpu_dust_filter_run!
Odin.gpu_dust_likelihood_monty
Odin.gpu_dust_simulate
```
