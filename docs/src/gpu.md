# GPU Acceleration

Odin.jl supports GPU-accelerated simulation and particle filtering via a
backend abstraction layer. Currently supported backends are **Apple Metal**
(via Metal.jl), **NVIDIA CUDA**, and **AMD GPU**, plus a **CPU fallback**.

## Quick Start

```julia
using Odin
using Metal  # or: using CUDA, using AMDGPU

sir = @odin begin
    update(S) = S - n_SI
    update(I) = I + n_SI - n_IR
    update(R) = R + n_IR
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    p_SI = 1 - exp(-beta * I / N * dt)
    p_IR = 1 - exp(-gamma * dt)
    n_SI = Binomial(S, p_SI)
    n_IR = Binomial(I, p_IR)
    N = parameter(1000)
    I0 = parameter(10)
    beta = parameter(0.3)
    gamma = parameter(0.1)
end

# Run 10,000 particles on GPU
result = gpu_dust_simulate(sir, (N=1000.0, I0=10.0, beta=0.3, gamma=0.1);
    times=0.0:1.0:100.0, n_particles=10_000, backend=gpu_backend())
```

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

### Setup by Platform

| Platform | Package | Install |
|----------|---------|---------|
| macOS (Apple Silicon) | Metal.jl | `Pkg.add("Metal")` |
| NVIDIA GPU | CUDA.jl | `Pkg.add("CUDA")` |
| AMD GPU | AMDGPU.jl | `Pkg.add("AMDGPU")` |

After installation, simply `using Metal` (or equivalent) before calling any
GPU functions — Odin.jl detects the backend automatically via Julia's package
extension mechanism.

## GPU Particle Filter

Run a bootstrap particle filter on the GPU for massively parallel likelihood
evaluation:

```julia
gpu_filt = gpu_Likelihood(gen, data;
    time_start = 0.0,
    n_particles = 10_000,
    dt = 0.25,
    seed = 42,
    backend = gpu_backend(),
)

ll = gpu_dust_filter_run!(gpu_filt, pars)
```

### When GPU Filtering Helps

The GPU particle filter is most beneficial when:

- **n_particles ≥ 1,000** — GPU overhead dominates for small particle counts
- **Model is compute-bound** — models with many state variables or complex
  update rules benefit most
- **Running many filter evaluations** — e.g., inside an MCMC loop

For simple models with < 500 particles, the CPU filter is often faster due
to lower kernel launch overhead.

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

## Performance Tips

1. **Use enough particles** — GPU overhead is fixed per kernel launch. With
   < 500 particles, CPU is often faster. Aim for ≥ 2,000 for clear GPU benefit.

2. **Minimise CPU↔GPU transfers** — each `gpu_array`/`cpu_array` call incurs
   transfer latency. Batch operations on the GPU where possible.

3. **Profile with `@time`** — the first call includes JIT compilation for the
   GPU kernel. Subsequent calls are much faster.

4. **Check memory** — large particle counts with many state variables can
   exhaust GPU memory. Monitor with `Metal.device()` or `CUDA.memory_status()`.

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
