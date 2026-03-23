# Slice sampler — Neal (2003) stepping-out and shrinking.
# Multivariate: coordinate-wise (cycle through dimensions).

"""
    MontySliceSampler

Slice sampler using Neal's stepping-out and shrinking procedure.
No gradient required — works with density only.
Multivariate targets are handled by cycling through dimensions (coordinate-wise).
"""
struct MontySliceSampler <: AbstractMontySampler
    w::Float64                  # initial bracket width
    max_steps::Int              # max stepping-out expansions per side
end

mutable struct SliceState <: AbstractSamplerState
    proposal::Vector{Float64}
    current_dim::Int            # current coordinate index (1-based)
end

"""
    monty_sampler_slice(; w=1.0, max_steps=10)

Create a slice sampler.

## Keyword arguments
- `w`: initial bracket width (default 1.0)
- `max_steps`: maximum number of stepping-out expansions per side (default 10)
"""
function monty_sampler_slice(; w::Float64=1.0, max_steps::Int=10)
    w > 0 || error("w must be positive")
    max_steps >= 1 || error("max_steps must be >= 1")
    return MontySliceSampler(w, max_steps)
end

function initialise(sampler::MontySliceSampler, chain::ChainState, model::MontyModel, rng::AbstractRNG)
    return SliceState(similar(chain.pars), 1)
end

function step!(sampler::MontySliceSampler, chain::ChainState, state::SliceState, model::MontyModel, rng::AbstractRNG)
    n = length(chain.pars)

    # Cycle through all dimensions in one "step"
    for _ in 1:n
        d = state.current_dim
        state.current_dim = (d % n) + 1

        _slice_sample_dim!(sampler, chain, state, model, rng, d)
    end

    # Slice sampling always "accepts" (no rejection step in the MH sense)
    return true
end

function _slice_sample_dim!(sampler::MontySliceSampler, chain::ChainState, state::SliceState,
                            model::MontyModel, rng::AbstractRNG, dim::Int)
    w = sampler.w
    x0 = chain.pars[dim]
    y = chain.density + log(rand(rng))  # slice level

    # Stepping out: find bracket [L, R]
    u = rand(rng)
    L = x0 - w * u
    R = L + w

    # Respect domain bounds if present
    lo = model.domain !== nothing ? model.domain[dim, 1] : -Inf
    hi = model.domain !== nothing ? model.domain[dim, 2] : Inf

    # Step out left
    state.proposal .= chain.pars
    j = sampler.max_steps
    while j > 0 && L > lo
        state.proposal[dim] = L
        if model(state.proposal) <= y
            break
        end
        L -= w
        j -= 1
    end
    L = max(L, lo)

    # Step out right
    state.proposal .= chain.pars
    k = sampler.max_steps
    while k > 0 && R < hi
        state.proposal[dim] = R
        if model(state.proposal) <= y
            break
        end
        R += w
        k -= 1
    end
    R = min(R, hi)

    # Shrinking: sample from bracket until above slice
    max_shrink = 200
    for _ in 1:max_shrink
        x1 = L + rand(rng) * (R - L)
        state.proposal .= chain.pars
        state.proposal[dim] = x1
        density_new = model(state.proposal)

        if density_new > y
            chain.pars[dim] = x1
            chain.density = density_new
            return nothing
        end

        # Shrink bracket
        if x1 < x0
            L = x1
        else
            R = x1
        end
    end

    # If we exhaust shrinking iterations, stay at current position
    return nothing
end
