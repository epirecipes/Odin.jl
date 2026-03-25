# GPU backend abstraction layer for Odin.jl
#
# Provides a unified interface for GPU acceleration, with backends for
# Metal (Apple Silicon), CUDA (NVIDIA), and a CPU fallback. Metal-specific
# code lives in ext/OdinMetalExt.jl and is loaded automatically when
# `using Metal` is active.

"""
    GPUBackend

Abstract type for GPU compute backends.
"""
abstract type GPUBackend end

"""
    MetalBackend <: GPUBackend

Apple Metal GPU backend (requires Metal.jl).
Activated via the OdinMetalExt package extension.
"""
struct MetalBackend <: GPUBackend end

"""
    CUDABackend <: GPUBackend

NVIDIA CUDA GPU backend (placeholder for future CUDA.jl support).
"""
struct CUDABackend <: GPUBackend end

"""
    AMDGPUBackend <: GPUBackend

AMD GPU backend (placeholder for future AMDGPU.jl support).
"""
struct AMDGPUBackend <: GPUBackend end

"""
    CPUBackend <: GPUBackend

CPU fallback backend. Wraps the existing CPU particle filter with the
GPU filter API so that code written for GPU filters works on any machine.
"""
struct CPUBackend <: GPUBackend end

# ── Backend registry ─────────────────────────────────────────

"""Global registry of available GPU backends, populated by extensions."""
const _GPU_BACKENDS = Dict{Symbol, GPUBackend}()

"""
    register_gpu_backend!(name::Symbol, backend::GPUBackend)

Register a GPU backend. Called by package extensions at load time.
"""
function register_gpu_backend!(name::Symbol, backend::GPUBackend)
    _GPU_BACKENDS[name] = backend
    return nothing
end

"""
    available_gpu_backends() -> Vector{Symbol}

Return names of all registered GPU backends (excluding :cpu).
"""
function available_gpu_backends()
    return collect(keys(_GPU_BACKENDS))
end

"""
    has_gpu() -> Bool

Return `true` if any GPU backend is registered and functional.
"""
function has_gpu()
    return !isempty(_GPU_BACKENDS)
end

"""
    gpu_backend(; preferred=:auto) -> GPUBackend

Select a GPU backend.

- `:auto` — pick the first available GPU backend, falling back to `CPUBackend()`
- `:metal` — require Metal backend (error if unavailable)
- `:cuda` — require CUDA backend (error if unavailable)
- `:amdgpu` — require AMD GPU backend (error if unavailable)
- `:cpu` — explicitly use the CPU fallback
"""
function gpu_backend(; preferred::Symbol=:auto)
    if preferred == :cpu
        return CPUBackend()
    elseif preferred == :auto
        for name in (:metal, :cuda, :amdgpu)
            if haskey(_GPU_BACKENDS, name)
                return _GPU_BACKENDS[name]
            end
        end
        return CPUBackend()
    else
        if haskey(_GPU_BACKENDS, preferred)
            return _GPU_BACKENDS[preferred]
        else
            avail = isempty(_GPU_BACKENDS) ? "none" : join(keys(_GPU_BACKENDS), ", ")
            error("GPU backend :$preferred is not available. " *
                  "Available backends: $avail. " *
                  "Load the required package (e.g., `using Metal`) first.")
        end
    end
end

# ── GPU array interface ──────────────────────────────────────

"""
    gpu_array(backend, x)

Transfer array `x` to the GPU managed by `backend`.
Returns the input unchanged for `CPUBackend`.
"""
gpu_array(::CPUBackend, x) = x

"""
    cpu_array(x)

Transfer a GPU array back to the CPU. Identity for regular arrays.
"""
cpu_array(x::Array) = x

"""
    gpu_array_type(backend)

Return the array type used by the given backend.
"""
gpu_array_type(::CPUBackend) = Array

# Extension points — overridden by OdinMetalExt, etc.
# Default implementations raise informative errors for unregistered backends.
function gpu_array(::MetalBackend, x)
    error("Metal.jl is not loaded. Run `using Metal` before using MetalBackend.")
end
function gpu_array(::CUDABackend, x)
    error("CUDA.jl is not loaded. Run `using CUDA` before using CUDABackend.")
end
function gpu_array(::AMDGPUBackend, x)
    error("AMDGPU.jl is not loaded. Run `using AMDGPU` before using AMDGPUBackend.")
end

"""
    backend_name(backend::GPUBackend) -> String

Human-readable name of the backend for display/logging.
"""
backend_name(::CPUBackend) = "CPU (fallback)"
backend_name(::MetalBackend) = "Metal (Apple Silicon)"
backend_name(::CUDABackend) = "CUDA (NVIDIA)"
backend_name(::AMDGPUBackend) = "AMDGPU (AMD)"
