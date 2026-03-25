# OdinMetalExt — package extension for Metal.jl GPU support on Apple Silicon.
#
# Loaded automatically when `using Metal` is active alongside `using Odin`.
# Registers the MetalBackend and provides Metal-specific GPU array operations.

module OdinMetalExt

using Odin
using Metal

function __init__()
    # Register Metal backend if a GPU device is available
    if Metal.functional()
        Odin.register_gpu_backend!(:metal, Odin.MetalBackend())
        @info "Odin: Metal GPU backend registered ($(Metal.current_device().name))"
    else
        @warn "Odin: Metal.jl loaded but no functional GPU device found"
    end
end

# ── GPU array operations ─────────────────────────────────────

Odin.gpu_array(::Odin.MetalBackend, x::AbstractArray) = MtlArray(x)
Odin.cpu_array(x::MtlArray) = Array(x)
Odin.gpu_array_type(::Odin.MetalBackend) = MtlArray

end # module
