# Abstract sampler interface.

"""
    AbstractMontySampler

Abstract type for all MCMC samplers.
"""
abstract type AbstractMontySampler end

"""
    AbstractSamplerState

Abstract type for sampler internal state.
"""
abstract type AbstractSamplerState end

"""
    ChainState

State of a single MCMC chain.
"""
mutable struct ChainState
    pars::Vector{Float64}
    density::Float64
end

# Interface methods that samplers must implement:
# initialise(sampler, chain_state, model, rng) -> SamplerState
# step!(sampler, chain_state, sampler_state, model, rng) -> (accepted::Bool, chain_state)
