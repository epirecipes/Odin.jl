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
    has_shared_state::Bool
end

MontyModelProperties(;
    has_gradient=false,
    has_direct_sample=false,
    is_stochastic=false,
    allow_multiple_parameters=false,
    has_shared_state=false,
) = MontyModelProperties(
    has_gradient,
    has_direct_sample,
    is_stochastic,
    allow_multiple_parameters,
    has_shared_state,
)

"""
    MontyModel

A model for MCMC sampling: wraps a log-density function and optional gradient.
"""
struct MontyModel{D<:Function, G, S, Dom, C}
    parameters::Vector{String}
    density::D
    gradient::G                 # nothing or Function
    direct_sample::S            # nothing or Function
    domain::Dom                 # nothing or Matrix{Float64} (n_pars × 2, each row [lo, hi])
    properties::MontyModelProperties
    clone::C                    # nothing or Function returning an isolated equivalent model
end

MontyModel(parameters, density, gradient, direct_sample, domain, properties) =
    MontyModel(parameters, density, gradient, direct_sample, domain, properties, nothing)

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
    clone::Union{Nothing, Function}=nothing,
    properties::MontyModelProperties=MontyModelProperties(
        has_gradient=gradient !== nothing,
        has_direct_sample=direct_sample !== nothing,
    ),
)
    return MontyModel(parameters, density, gradient, direct_sample, domain, properties, clone)
end

_has_shared_state(model::MontyModel) = model.properties.has_shared_state
_can_isolate_model(model::MontyModel) = !_has_shared_state(model) || model.clone !== nothing
_clone_model(model::MontyModel) = model.clone === nothing ? model : model.clone()

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
    idx_a = [findfirst(==(p), all_params) for p in a.parameters]
    idx_b = [findfirst(==(p), all_params) for p in b.parameters]
    n = length(all_params)

    _project = (x, idx) -> x[idx]

    combined_density = function(x)
        return a.density(_project(x, idx_a)) + b.density(_project(x, idx_b))
    end

    combined_gradient = nothing
    if a.gradient !== nothing && b.gradient !== nothing
        combined_gradient = function(x)
            grad = zeros(eltype(x), n)
            grad_a = a.gradient(_project(x, idx_a))
            grad_b = b.gradient(_project(x, idx_b))
            for (i, idx) in enumerate(idx_a)
                grad[idx] += grad_a[i]
            end
            for (i, idx) in enumerate(idx_b)
                grad[idx] += grad_b[i]
            end
            return grad
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
        combined_domain = zeros(Float64, n, 2)
        combined_domain[:, 1] .= -Inf
        combined_domain[:, 2] .= Inf
        for (i, idx) in enumerate(idx_a)
            combined_domain[idx, 1] = max(combined_domain[idx, 1], a.domain[i, 1])
            combined_domain[idx, 2] = min(combined_domain[idx, 2], a.domain[i, 2])
        end
        for (i, idx) in enumerate(idx_b)
            combined_domain[idx, 1] = max(combined_domain[idx, 1], b.domain[i, 1])
            combined_domain[idx, 2] = min(combined_domain[idx, 2], b.domain[i, 2])
        end
    elseif a.domain !== nothing
        combined_domain = zeros(Float64, n, 2)
        combined_domain[:, 1] .= -Inf
        combined_domain[:, 2] .= Inf
        for (i, idx) in enumerate(idx_a)
            combined_domain[idx, :] .= a.domain[i, :]
        end
    elseif b.domain !== nothing
        combined_domain = zeros(Float64, n, 2)
        combined_domain[:, 1] .= -Inf
        combined_domain[:, 2] .= Inf
        for (i, idx) in enumerate(idx_b)
            combined_domain[idx, :] .= b.domain[i, :]
        end
    end

    combined_props = MontyModelProperties(
        has_gradient=combined_gradient !== nothing,
        has_direct_sample=false,
        is_stochastic=a.properties.is_stochastic || b.properties.is_stochastic,
        allow_multiple_parameters=false,
        has_shared_state=a.properties.has_shared_state || b.properties.has_shared_state,
    )

    combined_clone = nothing
    if (a.clone !== nothing || b.clone !== nothing || combined_props.has_shared_state) &&
       _can_isolate_model(a) && _can_isolate_model(b)
        combined_clone = () -> monty_model_combine(_clone_model(a), _clone_model(b))
    end

    return MontyModel(
        all_params,
        combined_density,
        combined_gradient,
        nothing,
        combined_domain,
        combined_props,
        combined_clone,
    )
end

# Operator overload: model1 + model2
Base.:+(a::MontyModel, b::MontyModel) = monty_model_combine(a, b)

function _clone_likelihood(filter::DustFilter)
    if filter.n_groups > 1 && filter.group_data !== nothing
        return dust_filter_create(
            filter.generator,
            filter.group_data;
            time_start=Float64(filter.time_start),
            n_particles=filter.n_particles,
            dt=Float64(filter.dt),
            seed=filter.seed,
            save_trajectories=filter.save_trajectories,
        )
    end
    return dust_filter_create(
        filter.generator,
        filter.data;
        time_start=Float64(filter.time_start),
        n_particles=filter.n_particles,
        dt=Float64(filter.dt),
        seed=filter.seed,
        save_trajectories=filter.save_trajectories,
    )
end

function _clone_likelihood(unfilter::DustUnfilter)
    if unfilter.n_groups > 1 && unfilter.group_data !== nothing
        return dust_unfilter_create(
            unfilter.generator,
            unfilter.group_data;
            time_start=Float64(unfilter.time_start),
            ode_control=unfilter.ode_control,
        )
    end
    return dust_unfilter_create(
        unfilter.generator,
        unfilter.data;
        time_start=Float64(unfilter.time_start),
        ode_control=unfilter.ode_control,
    )
end

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
        clone=() -> dust_likelihood_monty(_clone_likelihood(filter), packer),
        properties=MontyModelProperties(is_stochastic=true, has_shared_state=true),
    )
end

function dust_likelihood_monty(filter::DustFilter, packer::MontyPackerGrouped)
    param_names = parameter_names(packer)

    density = function(x)
        pars = unpack(packer, x)
        pars_vec = NamedTuple[pars[g] for g in packer.groups]
        return dust_likelihood_run!(filter, pars_vec)
    end

    return monty_model(
        density;
        parameters=param_names,
        clone=() -> dust_likelihood_monty(_clone_likelihood(filter), packer),
        properties=MontyModelProperties(is_stochastic=true, has_shared_state=true),
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
        clone=() -> dust_likelihood_monty(_clone_likelihood(unfilter), packer),
        properties=MontyModelProperties(
            is_stochastic=false,
            has_gradient=true,
            has_shared_state=true,
        ),
    )
end

function dust_likelihood_monty(unfilter::DustUnfilter, packer::MontyPackerGrouped)
    param_names = parameter_names(packer)

    density = function(x)
        pars = unpack(packer, x)
        pars_vec = NamedTuple[pars[g] for g in packer.groups]
        return dust_unfilter_run!(unfilter, pars_vec)
    end

    gradient = function(x)
        return ForwardDiff.gradient(density, x)
    end

    return monty_model(
        density;
        parameters=param_names,
        gradient=gradient,
        clone=() -> dust_likelihood_monty(_clone_likelihood(unfilter), packer),
        properties=MontyModelProperties(
            is_stochastic=false,
            has_gradient=true,
            has_shared_state=true,
        ),
    )
end
