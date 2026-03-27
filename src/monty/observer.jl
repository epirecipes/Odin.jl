# Observer pattern for collecting custom outputs during MCMC sampling.

"""
    MontyObserver

Observer for collecting custom outputs during MCMC sampling.
After each density evaluation, `observe(model, rng)` is called to
capture model state (e.g., particle filter trajectories).

# Fields
- `observe`: `(model, rng) -> observation` — called each step
- `finalise`: `(Vector{obs}) -> combined` — combines within a chain
- `combine`: `(Vector{chain_obs}) -> final` — combines across chains
- `append`: `(obs1, obs2) -> merged` — for `sample_continue`
"""
struct MontyObserver
    observe::Function
    finalise::Function
    combine::Function
    append::Function
end

function MontyObserver(observe::Function;
                       finalise::Function=_auto_finalise,
                       combine::Function=_auto_combine,
                       append::Function=_auto_append)
    MontyObserver(observe, finalise, combine, append)
end

"""Combine a vector of observations with matching NamedTuple structure."""
function _auto_finalise(observations::Vector)
    isempty(observations) && return nothing
    first_obs = first(observations)
    if first_obs isa NamedTuple
        ks = keys(first_obs)
        if all(obs isa NamedTuple && keys(obs) == ks for obs in observations)
            return NamedTuple{ks}(Tuple([
                _stack_values([obs[k] for obs in observations]) for k in ks
            ]))
        end
    end
    return observations
end

"""Combine across chains — concatenate along the chain dimension."""
function _auto_combine(chain_results::Vector)
    isempty(chain_results) && return nothing
    # Filter out nothing results
    valid = filter(!isnothing, chain_results)
    isempty(valid) && return nothing
    first_res = first(valid)
    if first_res isa NamedTuple
        ks = keys(first_res)
        if all(r isa NamedTuple && keys(r) == ks for r in valid)
            return NamedTuple{ks}(Tuple([
                _combine_chain_values([r[k] for r in valid]) for k in ks
            ]))
        end
    end
    return valid
end

"""Append observations for sample_continue."""
function _auto_append(obs1, obs2)
    if obs1 === nothing
        return obs2
    elseif obs2 === nothing
        return obs1
    end
    if obs1 isa NamedTuple && obs2 isa NamedTuple && keys(obs1) == keys(obs2)
        ks = keys(obs1)
        return NamedTuple{ks}(Tuple([
            _append_values(obs1[k], obs2[k]) for k in ks
        ]))
    end
    return vcat(obs1, obs2)
end

function _stack_values(vals::Vector)
    if all(v isa Number for v in vals)
        return collect(Float64, vals)
    elseif all(v isa AbstractVector for v in vals)
        if all(length(v) == length(first(vals)) for v in vals)
            return hcat(vals...)
        end
    elseif all(v isa AbstractMatrix for v in vals)
        if all(size(v) == size(first(vals)) for v in vals)
            return cat(vals...; dims=3)
        end
    end
    return vals
end

function _combine_chain_values(vals::Vector)
    if all(v isa AbstractVector for v in vals)
        return hcat(vals...)
    elseif all(v isa AbstractMatrix for v in vals)
        return cat(vals...; dims=3)
    elseif all(v isa AbstractArray{<:Any, 3} for v in vals)
        return cat(vals...; dims=4)
    end
    return vals
end

function _append_values(v1, v2)
    if v1 isa AbstractVector && v2 isa AbstractVector
        return vcat(v1, v2)
    elseif v1 isa AbstractMatrix && v2 isa AbstractMatrix
        return cat(v1, v2; dims=2)
    elseif v1 isa AbstractArray{<:Any, 3} && v2 isa AbstractArray{<:Any, 3}
        return cat(v1, v2; dims=3)
    end
    return vcat(v1, v2)
end
