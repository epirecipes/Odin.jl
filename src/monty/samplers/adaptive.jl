# Adaptive Metropolis-Hastings sampler (Spencer 2021 "accelerated shaping").

"""
    MontyAdaptiveSampler

Adaptive Metropolis-Hastings sampler with online VCV learning.
"""
struct MontyAdaptiveSampler <: AbstractMontySampler
    initial_vcv::Matrix{Float64}
    initial_vcv_weight::Float64
    acceptance_target::Float64
    forget_rate::Float64
    forget_end::Int
    adapt_end::Int
    pre_diminish::Int
    boundaries::Symbol
    log_scaling_update::Bool
end

mutable struct AdaptiveState <: AbstractSamplerState
    iteration::Int
    scaling::Float64
    scaling_weight::Float64
    mean::Vector{Float64}
    autocorrelation::Matrix{Float64}
    weight::Float64
    vcv::Matrix{Float64}
    chol_vcv::LowerTriangular{Float64}
    proposal::Vector{Float64}
    history_pars::Vector{Vector{Float64}}
    n_accepted::Int
    n_total::Int
end

"""
    monty_sampler_adaptive(initial_vcv; kwargs...)

Create an adaptive Metropolis-Hastings sampler.
"""
function monty_sampler_adaptive(
    initial_vcv::AbstractMatrix{Float64};
    initial_vcv_weight::Float64=1000.0,
    acceptance_target::Float64=0.234,
    forget_rate::Float64=0.2,
    forget_end::Int=typemax(Int),
    adapt_end::Int=typemax(Int),
    pre_diminish::Int=0,
    boundaries::Symbol=:reflect,
    log_scaling_update::Bool=true,
)
    return MontyAdaptiveSampler(
        Matrix(initial_vcv), initial_vcv_weight, acceptance_target,
        forget_rate, forget_end, adapt_end, pre_diminish, boundaries,
        log_scaling_update,
    )
end

function initialise(sampler::MontyAdaptiveSampler, chain::ChainState, model::MontyModel, rng::AbstractRNG)
    n = length(chain.pars)
    C = cholesky(Symmetric(sampler.initial_vcv))
    scaling = 1.0

    return AdaptiveState(
        0,                                  # iteration
        scaling,                            # scaling
        1.0,                                # scaling_weight
        copy(chain.pars),                   # mean
        zeros(Float64, n, n),               # autocorrelation
        0.0,                                # weight
        copy(sampler.initial_vcv),          # vcv
        LowerTriangular(C.L),              # chol_vcv
        similar(chain.pars),               # proposal
        Vector{Float64}[],                 # history
        0, 0,                              # accepted, total
    )
end

function step!(sampler::MontyAdaptiveSampler, chain::ChainState, state::AdaptiveState, model::MontyModel, rng::AbstractRNG)
    state.iteration += 1
    n = length(chain.pars)

    # Propose using current adapted VCV
    scaled_chol = state.scaling * state.chol_vcv
    mvn_sample!(state.proposal, chain.pars, scaled_chol, rng)

    # Handle boundaries
    if sampler.boundaries == :reflect && model.domain !== nothing
        reflect_proposal!(state.proposal, model.domain)
    end

    # Evaluate
    if sampler.boundaries == :reject && !in_domain(state.proposal, model.domain)
        density_prop = -Inf
    else
        density_prop = model(state.proposal)
    end

    # Accept/reject
    log_alpha = density_prop - chain.density
    accepted = log(rand(rng)) < log_alpha

    if accepted
        chain.pars .= state.proposal
        chain.density = density_prop
    end

    state.n_total += 1
    if accepted
        state.n_accepted += 1
    end

    # Adapt
    if state.iteration <= sampler.adapt_end
        _adapt!(sampler, state, chain.pars)
    end

    return accepted
end

function _adapt!(sampler::MontyAdaptiveSampler, state::AdaptiveState, pars::Vector{Float64})
    n = length(pars)

    # Update empirical statistics
    state.weight += 1.0
    delta = pars .- state.mean
    state.mean .+= delta ./ state.weight
    delta2 = pars .- state.mean
    state.autocorrelation .+= delta * delta2'

    # Store for forgetting
    push!(state.history_pars, copy(pars))

    # Forget old parameters
    if sampler.forget_rate > 0 && state.iteration < sampler.forget_end
        forget_interval = max(1, round(Int, 1.0 / sampler.forget_rate))
        if state.iteration % forget_interval == 0 && !isempty(state.history_pars)
            popfirst!(state.history_pars)
        end
    end

    # Update scaling
    acceptance_rate = state.n_total > 0 ? state.n_accepted / state.n_total : sampler.acceptance_target
    increment = acceptance_rate > sampler.acceptance_target ? 1.0 : -1.0

    if state.iteration > sampler.pre_diminish
        state.scaling_weight += 1.0
    end

    if sampler.log_scaling_update
        state.scaling *= exp(increment / state.scaling_weight)
    else
        state.scaling = max(state.scaling + increment / state.scaling_weight, 1e-8)
    end

    # Update VCV from empirical covariance
    if state.weight > n + 1
        empirical_vcv = state.autocorrelation ./ (state.weight - 1)
        w_init = sampler.initial_vcv_weight
        w_emp = state.weight
        combined = (w_init .* sampler.initial_vcv .+ w_emp .* empirical_vcv) ./ (w_init + w_emp)

        # Optimal proposal scaling for RW-MH
        combined .*= (2.38^2 / n)

        try
            C = cholesky(Symmetric(combined))
            state.vcv .= combined
            state.chol_vcv = LowerTriangular(C.L)
        catch
            # If not positive definite, keep current
        end
    end

    return nothing
end
