# Systematic resampling for particle filters.

"""
    _systematic_resample_inplace!(indices, log_weights, rng)

Systematic resampling that normalizes `log_weights` **in-place** (destructive).
Zero-allocation version for use in particle filter hot loop.
"""
function _systematic_resample_inplace!(indices::Vector{Int}, log_weights::Vector{Float64},
                                       rng::AbstractRNG)
    n = length(log_weights)

    # Normalize in-place: log_weights → normalized weights
    max_lw = maximum(log_weights)
    total = 0.0
    @inbounds for i in 1:n
        w = exp(log_weights[i] - max_lw)
        log_weights[i] = w
        total += w
    end
    inv_total = 1.0 / total
    @inbounds for i in 1:n
        log_weights[i] *= inv_total
    end

    # Systematic resampling
    u = rand(rng) / n
    cumw = 0.0
    j = 1
    @inbounds for i in 1:n
        cumw += log_weights[i]
        while u < cumw && j <= n
            indices[j] = i
            j += 1
            u += 1.0 / n
        end
    end
    @inbounds while j <= n
        indices[j] = n
        j += 1
    end
    return nothing
end

# Backward-compatible 3-arg version (allocating)
function systematic_resample!(indices::Vector{Int}, log_weights::Vector{Float64},
                              rng::AbstractRNG)
    systematic_resample!(indices, log_weights, similar(log_weights), rng)
end

"""
    log_sum_exp(x) -> Float64

Numerically stable log-sum-exp (allocation-free).
"""
function log_sum_exp(x::AbstractVector{Float64})
    mx = maximum(x)
    isinf(mx) && return -Inf
    s = 0.0
    @inbounds for i in eachindex(x)
        s += exp(x[i] - mx)
    end
    return mx + log(s)
end
