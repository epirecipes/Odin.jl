# Systematic resampling for particle filters.

"""
    _systematic_resample_inplace!(indices, log_weights, rng)

Systematic resampling (Kitagawa, 1996) with destructive in-place weight
normalization. Zero-allocation for use in the particle filter hot loop.

# Arguments
- `indices::Vector{Int}` — output buffer of length `n` for resampled indices.
- `log_weights::Vector{Float64}` — input log-weights; **mutated**: on return
  the contents are normalized weights, not log-weights. Caller must discard.
- `rng::AbstractRNG` — random number generator.

# Algorithm
1. Subtract `max(log_weights)` for numerical stability and exponentiate.
2. Normalize the resulting weights to sum to 1 (in place).
3. Draw `u ~ Uniform(0, 1/n)`; walk the cumulative weight curve, assigning
   `indices[j]` whenever `u` crosses the boundary at particle `i`.

# Degenerate handling
If all log-weights are `-Inf` (or otherwise non-finite, yielding a
non-normalizable distribution), the function falls back to **uniform
resampling** — every particle is selected once. This prevents NaN
propagation when MCMC explores impossible parameter regions and is
mathematically equivalent to "no information from this data point".

# Reference
Kitagawa, G. (1996). "Monte Carlo Filter and Smoother for Non-Gaussian
Nonlinear State Space Models." *Journal of Computational and Graphical
Statistics*, 5(1), 1–25.
"""
function _systematic_resample_inplace!(indices::Vector{Int}, log_weights::Vector{Float64},
                                       rng::AbstractRNG)
    n = length(log_weights)
    n > 0 || return nothing

    # Normalize in-place: log_weights → normalized weights
    max_lw = maximum(log_weights)

    # Degenerate case: all log-weights are -Inf (or NaN). Fall back to
    # uniform resampling so the filter does not corrupt state with NaN.
    if !isfinite(max_lw)
        @inbounds for i in 1:n
            log_weights[i] = 1.0 / n
            indices[i] = i
        end
        return nothing
    end

    total = 0.0
    @inbounds for i in 1:n
        w = exp(log_weights[i] - max_lw)
        log_weights[i] = w
        total += w
    end

    # Defensive guard: if total underflowed to 0 or is non-finite, fall
    # back to uniform resampling.
    if !(total > 0.0) || !isfinite(total)
        @inbounds for i in 1:n
            log_weights[i] = 1.0 / n
            indices[i] = i
        end
        return nothing
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
