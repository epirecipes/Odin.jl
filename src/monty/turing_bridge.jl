# DynamicPPL / Turing ecosystem bridge for Odin.jl
#
# Provides:
# 1. LogDensityProblems interface for MontyModel
# 2. MCMCChains conversion utilities
# 3. to_turing_model() — wraps dust likelihood + priors as a DynamicPPL @model
# 4. dppl_prior() — converts DynamicPPL @model to MontyModel for Odin samplers
# 5. dppl_to_monty_model() — converts full DynamicPPL model to MontyModel
# 6. turing_sample() — convenience wrapper

using DynamicPPL
using AbstractMCMC
using MCMCChains: Chains

import LogDensityProblems

# ═══════════════════════════════════════════════════════════════
# 1. LogDensityProblems interface for MontyModel
# ═══════════════════════════════════════════════════════════════

struct MontyLogDensityWrapper{M<:MontyModel}
    model::M
end

LogDensityProblems.logdensity(w::MontyLogDensityWrapper, θ::AbstractVector) = w.model(θ)
LogDensityProblems.dimension(w::MontyLogDensityWrapper) = length(w.model.parameters)
function LogDensityProblems.capabilities(::Type{<:MontyLogDensityWrapper})
    return LogDensityProblems.LogDensityOrder{0}()
end

struct MontyLogDensityGradWrapper{M<:MontyModel}
    model::M
end

LogDensityProblems.logdensity(w::MontyLogDensityGradWrapper, θ::AbstractVector) = w.model(θ)
LogDensityProblems.dimension(w::MontyLogDensityGradWrapper) = length(w.model.parameters)
function LogDensityProblems.capabilities(::Type{<:MontyLogDensityGradWrapper})
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.logdensity_and_gradient(w::MontyLogDensityGradWrapper, θ::AbstractVector)
    return w.model(θ), w.model.gradient(θ)
end

"""
    as_logdensity(model::MontyModel)

Wrap a `MontyModel` as a `LogDensityProblems`-compatible object.
Returns a gradient-aware wrapper if the model has a gradient.
"""
function as_logdensity(model::MontyModel)
    if model.properties.has_gradient && model.gradient !== nothing
        return MontyLogDensityGradWrapper(model)
    else
        return MontyLogDensityWrapper(model)
    end
end

# ═══════════════════════════════════════════════════════════════
# 2. MCMCChains conversion utilities
# ═══════════════════════════════════════════════════════════════

"""
    to_chains(samples::MontySamples) -> MCMCChains.Chains

Convert Odin `MontySamples` to an `MCMCChains.Chains` object for
diagnostics (ESS, R̂), plotting, and interoperability with the Turing ecosystem.
"""
function to_chains(samples::MontySamples)
    pars = permutedims(samples.pars, (2, 1, 3))
    cnames = Symbol.(samples.parameter_names)
    return Chains(pars, cnames)
end

"""
    from_chains(chain::Chains) -> MontySamples

Convert an `MCMCChains.Chains` object back to `MontySamples`.
"""
function from_chains(chain::Chains, parameter_names::Union{Nothing, Vector{String}}=nothing)
    arr = chain.value.data
    n_samples, n_pars, n_chains = size(arr)
    pars = permutedims(arr, (2, 1, 3))
    if parameter_names === nothing
        parameter_names = String.(names(chain, :parameters))
    end
    density = zeros(Float64, n_samples, n_chains)
    initial = pars[:, 1, :]
    return MontySamples(pars, density, initial, parameter_names, Dict{Symbol, Any}(), nothing)
end

# ═══════════════════════════════════════════════════════════════
# 3. to_turing_model — DynamicPPL model from dust likelihood
# ═══════════════════════════════════════════════════════════════

"""
    to_turing_model(filter_or_unfilter, packer; priors...)

Create a DynamicPPL `Model` that combines an Odin dust likelihood with
user-specified priors. The result can be converted to a `MontyModel` via
`dppl_to_monty_model` for sampling with Odin's native samplers, or
sampled with Turing.jl samplers directly (requires `using Turing`).

## Example

```julia
unfilter = dust_unfilter_create(sir, data; time_start=0.0)
packer = monty_packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))

dppl_model = to_turing_model(unfilter, packer;
    beta = Gamma(2.0, 0.25),
    gamma = Gamma(2.0, 0.05),
)

# Option 1: convert to MontyModel for Odin samplers
posterior = dppl_to_monty_model(dppl_model)
samples = monty_sample(posterior, monty_sampler_nuts(), 5000)

# Option 2: sample with Odin samplers directly
samples = turing_sample(dppl_model, monty_sampler_adaptive(vcv), 5000)
```
"""
function to_turing_model(filter_or_unfilter, packer::MontyPacker; priors...)
    prior_dict = Dict{Symbol, Any}(k => v for (k, v) in priors)
    packer_names = Set(Symbol.(parameter_names(packer)))
    for k in keys(prior_dict)
        k in packer_names || error("Prior parameter :$k not in packer: $(packer_names)")
    end
    return _build_dppl_model(filter_or_unfilter, packer, prior_dict)
end

function _build_dppl_model(likelihood_obj, packer::MontyPacker, priors::Dict{Symbol, Any})
    param_names = Symbol.(parameter_names(packer))
    n_pars = length(param_names)
    prior_dists = [priors[name] for name in param_names]

    DynamicPPL.@model function _odin_turing_model(likelihood_obj, packer, param_names, prior_dists)
        n = length(param_names)
        θ = Vector{Real}(undef, n)
        for idx in 1:n
            θ[idx] ~ prior_dists[idx]
        end
        x = collect(Float64, θ)
        pars = unpack(packer, x)
        ll = if likelihood_obj isa DustFilter
            dust_likelihood_run!(likelihood_obj, pars)
        elseif likelihood_obj isa DustUnfilter
            dust_unfilter_run!(likelihood_obj, pars)
        elseif likelihood_obj isa Likelihood
            loglik(likelihood_obj, pars)
        else
            error("Unsupported likelihood type: $(typeof(likelihood_obj))")
        end
        DynamicPPL.@addlogprob! ll
    end

    return _odin_turing_model(likelihood_obj, packer, param_names, prior_dists)
end

# ═══════════════════════════════════════════════════════════════
# 4. dppl_prior — DynamicPPL @model → MontyModel (prior only)
# ═══════════════════════════════════════════════════════════════

"""
    dppl_prior(dppl_model::DynamicPPL.Model) -> MontyModel

Convert a DynamicPPL `@model` (typically defining priors) into a `MontyModel`
that can be combined with Odin likelihoods and sampled with Odin's native
samplers. Supports hierarchical, vector, and correlated priors — anything
DynamicPPL supports.

## Example

```julia
using DynamicPPL, Distributions

@model function my_priors()
    beta ~ Gamma(2.0, 0.25)
    gamma ~ Gamma(2.0, 0.05)
end

prior = dppl_prior(my_priors())
posterior = likelihood + prior
samples = monty_sample(posterior, monty_sampler_adaptive(diagm([0.01, 0.01])), 5000)
```

## Hierarchical priors

```julia
@model function hierarchical()
    mu_beta ~ Normal(0.5, 0.2)
    sigma_beta ~ Exponential(0.1)
    beta ~ Normal(mu_beta, sigma_beta)
    gamma ~ Gamma(2.0, 0.05)
end
prior = dppl_prior(hierarchical())
```
"""
function dppl_prior(dppl_model::DynamicPPL.Model)
    return _dppl_to_monty(dppl_model; allow_direct_sample=true)
end

# ═══════════════════════════════════════════════════════════════
# 5. dppl_to_monty_model — full DynamicPPL model → MontyModel
# ═══════════════════════════════════════════════════════════════

"""
    dppl_to_monty_model(dppl_model::DynamicPPL.Model) -> MontyModel

Convert any DynamicPPL model (including those with `@addlogprob!` for
dust likelihoods) into a `MontyModel` for sampling with Odin's native samplers.

## Example

```julia
dppl_model = to_turing_model(unfilter, packer;
    beta = Gamma(2.0, 0.25), gamma = Gamma(2.0, 0.05))

monty = dppl_to_monty_model(dppl_model)
samples = monty_sample(monty, monty_sampler_nuts(metric=:dense), 5000)
```
"""
function dppl_to_monty_model(dppl_model::DynamicPPL.Model)
    return _dppl_to_monty(dppl_model; allow_direct_sample=false)
end

# Shared implementation for dppl_prior and dppl_to_monty_model
function _dppl_to_monty(dppl_model::DynamicPPL.Model; allow_direct_sample::Bool=false)
    # Sample from prior to initialise VarInfo and discover parameter structure
    vi = DynamicPPL.VarInfo(dppl_model)
    vn_keys = keys(vi.values)
    param_names_str = String.(Symbol.(vn_keys))
    n_pars = length(vn_keys)

    # Determine how to build NamedTuples for logjoint evaluation.
    # Simple models: VarNames are plain symbols (e.g., :beta, :gamma)
    # Indexed models: VarNames are indexed (e.g., θ[1], θ[2])
    # For indexed models, group by base variable name.
    nt_builder = _make_nt_builder(vn_keys, n_pars)

    density = let model = dppl_model, builder = nt_builder, np = n_pars
        function(x::AbstractVector)
            length(x) == np || return eltype(x)(-Inf)
            nt = builder(x)
            return DynamicPPL.logjoint(model, nt)
        end
    end

    gradient = let d = density
        function(x::AbstractVector)
            ForwardDiff.gradient(d, x)
        end
    end

    # Direct sampling from prior
    direct_sample_fn = nothing
    if allow_direct_sample
        direct_sample_fn = let model = dppl_model, np = n_pars
            function(rng::Random.AbstractRNG)
                vi_new = DynamicPPL.VarInfo(rng, model)
                x = Vector{Float64}(undef, np)
                for (idx, vn) in enumerate(keys(vi_new.values))
                    x[idx] = Float64(vi_new[vn])
                end
                return x
            end
        end
    end

    domain = _infer_domain(dppl_model, vi, n_pars, nt_builder)

    return monty_model(
        density;
        parameters=param_names_str,
        gradient=gradient,
        direct_sample=direct_sample_fn,
        domain=domain,
        properties=MontyModelProperties(
            has_gradient=true,
            has_direct_sample=(direct_sample_fn !== nothing),
        ),
    )
end

"""Build a function that converts a flat vector to a NamedTuple for logjoint."""
function _make_nt_builder(vn_keys, n_pars)
    # Check if all VarNames are simple (no indexing)
    all_simple = all(vn -> !occursin('[', Base.string(vn)), vn_keys)

    if all_simple
        syms = Symbol.(vn_keys)
        return x -> NamedTuple{Tuple(syms)}(Tuple(x))
    else
        # Group indexed variables by base name
        # e.g. θ[1], θ[2] → θ => [1, 2]
        groups = _group_varnames(vn_keys)
        return function(x)
            pairs = Pair{Symbol, Any}[]
            offset = 0
            for (base_sym, count) in groups
                if count == 1
                    push!(pairs, base_sym => x[offset + 1])
                else
                    push!(pairs, base_sym => collect(x[offset+1:offset+count]))
                end
                offset += count
            end
            return NamedTuple(pairs)
        end
    end
end

"""Group VarNames by base variable name, preserving order."""
function _group_varnames(vn_keys)
    groups = Pair{Symbol, Int}[]
    seen = Dict{Symbol, Int}()
    for vn in vn_keys
        s = Base.string(vn)
        base = Symbol(Base.split(s, '[')[1])
        if haskey(seen, base)
            idx = seen[base]
            groups[idx] = base => groups[idx][2] + 1
        else
            push!(groups, base => 1)
            seen[base] = length(groups)
        end
    end
    return groups
end

"""Infer parameter domain by probing with boundary values."""
function _infer_domain(model::DynamicPPL.Model, vi, n_pars::Int, nt_builder)
    domain = fill(-Inf, n_pars, 2)
    domain[:, 2] .= Inf

    vn_keys = keys(vi.values)
    base_vals = Float64[vi[vn] for vn in vn_keys]

    for i in 1:n_pars
        # Test small negative value
        x_test = copy(base_vals)
        x_test[i] = -1e-10
        lj = try
            DynamicPPL.logjoint(model, nt_builder(x_test))
        catch
            -Inf
        end
        if isinf(lj) && lj < 0
            domain[i, 1] = 0.0
        end

        # Test value > 1 to detect [0,1] bounded
        x_pos = copy(base_vals)
        x_pos[i] = 1.0 + 1e-10
        lj2 = try
            DynamicPPL.logjoint(model, nt_builder(x_pos))
        catch
            -Inf
        end
        if isinf(lj2) && lj2 < 0 && domain[i, 1] >= 0.0
            domain[i, 2] = 1.0
        end
    end

    return domain
end

# ═══════════════════════════════════════════════════════════════
# 6. turing_sample — sample DynamicPPL model with Odin samplers
# ═══════════════════════════════════════════════════════════════

"""
    turing_sample(dppl_model::DynamicPPL.Model, sampler::AbstractMontySampler,
                  n_steps; kwargs...) -> MontySamples

Sample a DynamicPPL model using Odin's native samplers.
Internally converts via `dppl_to_monty_model`.

## Example

```julia
dppl_model = to_turing_model(unfilter, packer;
    beta = Gamma(2.0, 0.25), gamma = Gamma(2.0, 0.05))

samples = turing_sample(dppl_model, monty_sampler_nuts(metric=:dense), 5000;
                        n_chains=4, n_burnin=1000)
```
"""
function turing_sample(
    dppl_model::DynamicPPL.Model,
    sampler::AbstractMontySampler,
    n_steps::Int;
    kwargs...,
)
    monty = dppl_to_monty_model(dppl_model)
    return monty_sample(monty, sampler, n_steps; kwargs...)
end
