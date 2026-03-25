# Shared sampler utilities.

"""
    reflect_proposal!(x, domain)

Reflect a proposal back into the domain. Modifies `x` in-place.
"""
function reflect_proposal!(x::AbstractVector{Float64}, domain::Matrix{Float64})
    for i in eachindex(x)
        lo = domain[i, 1]
        hi = domain[i, 2]
        isinf(lo) && isinf(hi) && continue

        max_iter = 100
        for _ in 1:max_iter
            if x[i] < lo
                x[i] = lo + (lo - x[i])
            elseif x[i] > hi
                x[i] = hi - (x[i] - hi)
            else
                break
            end
        end
        x[i] = clamp(x[i], lo, hi)
    end
    return nothing
end

"""
    in_domain(x, domain) -> Bool

Check if `x` is within the domain bounds.
"""
function in_domain(x::AbstractVector{Float64}, domain::Union{Nothing, Matrix{Float64}})
    domain === nothing && return true
    for i in eachindex(x)
        if x[i] < domain[i, 1] || x[i] > domain[i, 2]
            return false
        end
    end
    return true
end

"""
    mvn_sample!(result, mean, chol_vcv, rng)

Sample from a multivariate normal with given mean and Cholesky factor of VCV.
"""
function mvn_sample!(result::Vector{Float64}, mean::Vector{Float64}, chol_vcv::LowerTriangular{Float64}, rng::AbstractRNG)
    n = length(mean)
    @inbounds for i in 1:n
        result[i] = randn(rng)
    end
    # result = chol_vcv * z (in-place: use result as both z and output)
    # We need a temporary since mul! would overwrite z. Instead, do the
    # triangular multiply manually to avoid allocation.
    @inbounds for i in n:-1:1
        s = 0.0
        for j in 1:i
            s += chol_vcv[i, j] * result[j]
        end
        result[i] = s + mean[i]
    end
    return nothing
end


# ── Bijector infrastructure (shared by HMC and NUTS) ──────────

"""
    _build_bijector(domain, n_pars) -> (forward_transform, inverse_transform)

Build element-wise bijections from the model's domain (n_pars × 2 matrix of [lo, hi]).

Each parameter gets one of four transforms:
- `(-Inf, Inf)`: identity
- `(lo, Inf)`: `log(x - lo)` / `exp(y) + lo`
- `(-Inf, hi)`: `log(hi - x)` / `hi - exp(y)`
- `(lo, hi)`: scaled logit / scaled sigmoid
"""
function _build_bijector(domain::Nothing, n_pars::Int)
    return nothing, nothing
end

# Per-parameter transform specification
struct _ParamTransform
    kind::Symbol   # :identity, :log_shift, :reflected_log, :scaled_logit
    lo::Float64
    hi::Float64
end

function _build_bijector(domain::Matrix{Float64}, n_pars::Int)
    specs = Vector{_ParamTransform}(undef, n_pars)
    needs_transform = false

    for i in 1:n_pars
        lo, hi = domain[i, 1], domain[i, 2]
        if lo == -Inf && hi == Inf
            specs[i] = _ParamTransform(:identity, lo, hi)
        elseif lo > -Inf && hi == Inf
            specs[i] = _ParamTransform(:log_shift, lo, hi)
            needs_transform = true
        elseif lo == -Inf && hi < Inf
            specs[i] = _ParamTransform(:reflected_log, lo, hi)
            needs_transform = true
        else
            specs[i] = _ParamTransform(:scaled_logit, lo, hi)
            needs_transform = true
        end
    end

    if !needs_transform
        return nothing, nothing
    end

    specs_tuple = Tuple(specs)
    fwd = _StackedForward(specs_tuple)
    inv = _StackedInverse(specs_tuple)
    return fwd, inv
end


# ── Per-element transform functions (AD-compatible) ──────────

@inline function _fwd_one(s::_ParamTransform, x)
    if s.kind === :identity
        return x
    elseif s.kind === :log_shift
        return log(x - s.lo)
    elseif s.kind === :reflected_log
        return log(s.hi - x)
    else  # :scaled_logit
        t = (x - s.lo) / (s.hi - s.lo)
        return log(t / (one(t) - t))
    end
end

@inline function _inv_one(s::_ParamTransform, y)
    if s.kind === :identity
        return y
    elseif s.kind === :log_shift
        return exp(y) + s.lo
    elseif s.kind === :reflected_log
        return s.hi - exp(y)
    else  # :scaled_logit
        sig = one(y) / (one(y) + exp(-y))
        return s.lo + (s.hi - s.lo) * sig
    end
end

@inline function _inv_ladj_one(s::_ParamTransform, y)
    if s.kind === :identity
        return zero(y)
    elseif s.kind === :log_shift
        return y   # d(exp(y)+lo)/dy = exp(y), log|det| = y
    elseif s.kind === :reflected_log
        return y   # d(hi-exp(y))/dy = -exp(y), log|det| = y
    else  # :scaled_logit
        sig = one(y) / (one(y) + exp(-y))
        return log(s.hi - s.lo) + log(sig) + log(one(y) - sig)
    end
end


# ── Stacked transforms ─────────────────────────────────────

struct _StackedForward{S<:Tuple}
    specs::S
end

function (f::_StackedForward)(x::AbstractVector)
    y = similar(x)
    @inbounds for i in eachindex(x)
        y[i] = _fwd_one(f.specs[i], x[i])
    end
    return y
end

struct _StackedInverse{S<:Tuple}
    specs::S
end

function (inv::_StackedInverse)(y::AbstractVector)
    x = similar(y)
    @inbounds for i in eachindex(y)
        x[i] = _inv_one(inv.specs[i], y[i])
    end
    return x
end

function _stacked_inv_logabsdetjac(inv::_StackedInverse, y::AbstractVector)
    lj = zero(eltype(y))
    @inbounds for i in eachindex(y)
        lj += _inv_ladj_one(inv.specs[i], y[i])
    end
    return lj
end

# Fallback for no-transform case
_stacked_inv_logabsdetjac(::Nothing, y::AbstractVector) = zero(eltype(y))
