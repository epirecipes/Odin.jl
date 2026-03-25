# MontyModel: density function wrapper for MCMC samplers.

"""
    MontyModelProperties

Properties of a monty model.
"""
struct MontyModelProperties
    has_gradient::Bool
    has_direct_sample::Bool
    is_stochastic::Bool
    allow_multiple_parameters::Bool
end

MontyModelProperties(;
    has_gradient=false,
    has_direct_sample=false,
    is_stochastic=false,
    allow_multiple_parameters=false,
) = MontyModelProperties(has_gradient, has_direct_sample, is_stochastic, allow_multiple_parameters)

"""
    MontyModel

A model for MCMC sampling: wraps a log-density function and optional gradient.
"""
struct MontyModel{D<:Function, G, S, Dom}
    parameters::Vector{String}
    density::D
    gradient::G                 # nothing or Function
    direct_sample::S            # nothing or Function
    domain::Dom                 # nothing or Matrix{Float64} (n_pars × 2, each row [lo, hi])
    properties::MontyModelProperties
end

"""
    monty_model(density; parameters, gradient, direct_sample, domain, properties)

Create a MontyModel from a density function.
"""
function monty_model(
    density::Function;
    parameters::Vector{String},
    gradient::Union{Nothing, Function}=nothing,
    direct_sample::Union{Nothing, Function}=nothing,
    domain::Union{Nothing, Matrix{Float64}}=nothing,
    properties::MontyModelProperties=MontyModelProperties(
        has_gradient=gradient !== nothing,
        has_direct_sample=direct_sample !== nothing,
    ),
)
    return MontyModel(parameters, density, gradient, direct_sample, domain, properties)
end

"""
    (model::MontyModel)(x::AbstractVector) -> Float64

Evaluate the log-density at `x`.
"""
function (model::MontyModel)(x::AbstractVector)
    # Domain check
    if model.domain !== nothing
        for i in eachindex(x)
            if x[i] < model.domain[i, 1] || x[i] > model.domain[i, 2]
                return -Inf
            end
        end
    end
    return model.density(x)
end

"""
    monty_model_combine(a::MontyModel, b::MontyModel) -> MontyModel

Combine two models by summing their log-densities (e.g., likelihood + prior).
"""
function monty_model_combine(a::MontyModel, b::MontyModel)
    # Combined parameters: union
    all_params = unique(vcat(a.parameters, b.parameters))

    combined_density = function(x)
        return a.density(x) + b.density(x)
    end

    combined_gradient = nothing
    if a.gradient !== nothing && b.gradient !== nothing
        combined_gradient = function(x)
            return a.gradient(x) .+ b.gradient(x)
        end
    elseif a.gradient !== nothing || b.gradient !== nothing
        # One model has gradient — use ForwardDiff on the combined density
        combined_gradient = function(x)
            ForwardDiff.gradient(combined_density, x)
        end
    end

    # Domain: intersection (tightest bounds)
    combined_domain = nothing
    if a.domain !== nothing && b.domain !== nothing
        n = length(all_params)
        combined_domain = zeros(Float64, n, 2)
        combined_domain[:, 1] .= -Inf
        combined_domain[:, 2] .= Inf
        # Use tightest bounds from each
        na = length(a.parameters)
        nb = length(b.parameters)
        if na == n && a.domain !== nothing
            combined_domain[:, 1] .= max.(combined_domain[:, 1], a.domain[:, 1])
            combined_domain[:, 2] .= min.(combined_domain[:, 2], a.domain[:, 2])
        end
        if nb == n && b.domain !== nothing
            combined_domain[:, 1] .= max.(combined_domain[:, 1], b.domain[:, 1])
            combined_domain[:, 2] .= min.(combined_domain[:, 2], b.domain[:, 2])
        end
    elseif a.domain !== nothing
        combined_domain = a.domain
    elseif b.domain !== nothing
        combined_domain = b.domain
    end

    combined_props = MontyModelProperties(
        has_gradient=combined_gradient !== nothing,
        has_direct_sample=false,
        is_stochastic=a.properties.is_stochastic || b.properties.is_stochastic,
        allow_multiple_parameters=false,
    )

    return MontyModel(all_params, combined_density, combined_gradient, nothing, combined_domain, combined_props)
end

# Operator overload: model1 + model2
Base.:+(a::MontyModel, b::MontyModel) = monty_model_combine(a, b)

"""
    dust_likelihood_monty(filter_or_unfilter, packer) -> MontyModel

Convert a dust filter/unfilter + packer into a MontyModel for MCMC.
"""
function dust_likelihood_monty(filter::DustFilter, packer::MontyPacker)
    param_names = parameter_names(packer)

    density = function(x)
        pars = unpack(packer, x)
        return dust_likelihood_run!(filter, pars)
    end

    return monty_model(
        density;
        parameters=param_names,
        properties=MontyModelProperties(is_stochastic=true),
    )
end

function dust_likelihood_monty(unfilter::DustUnfilter, packer::MontyPacker)
    param_names = parameter_names(packer)

    density = function(x)
        pars = unpack(packer, x)
        return dust_unfilter_run!(unfilter, pars)
    end

    # ForwardDiff gradient through the ODE solver for deterministic likelihoods.
    # The generated _odin_rhs! and _odin_compare_data accept Dual numbers,
    # and DifferentialEquations.jl propagates them through the solver.
    gradient = function(x)
        return ForwardDiff.gradient(density, x)
    end

    return monty_model(
        density;
        parameters=param_names,
        gradient=gradient,
        properties=MontyModelProperties(is_stochastic=false, has_gradient=true),
    )
end
