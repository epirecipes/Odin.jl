# Random walk Metropolis-Hastings sampler.

"""
    MontyRandomWalkSampler

Random walk Metropolis-Hastings sampler with multivariate normal proposals.
"""
struct MontyRandomWalkSampler <: AbstractMontySampler
    vcv::Matrix{Float64}
    boundaries::Symbol          # :reflect, :reject, :ignore
    rerun_every::Int            # 0 = never rerun
end

mutable struct RandomWalkState <: AbstractSamplerState
    chol_vcv::LowerTriangular{Float64}
    proposal::Vector{Float64}
    rerun_count::Int
end

"""
    monty_sampler_random_walk(vcv; boundaries=:reflect, rerun_every=0)

Create a random walk Metropolis-Hastings sampler.
"""
function monty_sampler_random_walk(
    vcv::AbstractMatrix{Float64};
    boundaries::Symbol=:reflect,
    rerun_every::Int=0,
)
    boundaries in (:reflect, :reject, :ignore) || error("boundaries must be :reflect, :reject, or :ignore")
    return MontyRandomWalkSampler(Matrix(vcv), boundaries, rerun_every)
end

function initialise(sampler::MontyRandomWalkSampler, chain::ChainState, model::MontyModel, rng::AbstractRNG)
    C = cholesky(Symmetric(sampler.vcv))
    chol_vcv = LowerTriangular(C.L)
    proposal = similar(chain.pars)
    return RandomWalkState(chol_vcv, proposal, 0)
end

function step!(sampler::MontyRandomWalkSampler, chain::ChainState, state::RandomWalkState, model::MontyModel, rng::AbstractRNG)
    # Propose
    mvn_sample!(state.proposal, chain.pars, state.chol_vcv, rng)

    # Handle boundaries
    if sampler.boundaries == :reflect && model.domain !== nothing
        reflect_proposal!(state.proposal, model.domain)
    end

    # Evaluate density at proposal
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

    # Rerun mechanism for stochastic models
    state.rerun_count += 1
    if sampler.rerun_every > 0 && state.rerun_count >= sampler.rerun_every
        chain.density = model(chain.pars)
        state.rerun_count = 0
    end

    return accepted
end
