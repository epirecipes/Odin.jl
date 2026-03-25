# Parallel tempering (replica exchange) sampler.

"""
    MontyParallelTemperingSampler

Parallel tempering sampler using replica exchange.
"""
struct MontyParallelTemperingSampler <: AbstractMontySampler
    base_sampler::AbstractMontySampler
    n_rungs::Int
end

mutable struct ParallelTemperingState <: AbstractSamplerState
    rung_states::Vector{<:AbstractSamplerState}
    rung_chains::Vector{ChainState}
    temperatures::Vector{Float64}
    swap_acceptance::Vector{Int}
    swap_proposed::Vector{Int}
end

"""
    monty_sampler_parallel_tempering(base_sampler, n_rungs)

Create a parallel tempering sampler with `n_rungs` temperature rungs.
"""
function monty_sampler_parallel_tempering(base_sampler::AbstractMontySampler, n_rungs::Int)
    n_rungs >= 2 || error("n_rungs must be >= 2")
    return MontyParallelTemperingSampler(base_sampler, n_rungs)
end

function initialise(sampler::MontyParallelTemperingSampler, chain::ChainState, model::MontyModel, rng::AbstractRNG)
    n_rungs = sampler.n_rungs
    # Temperature schedule: linearly spaced from 1 (cold) to 1/n_rungs (hot)
    temperatures = [1.0 / (1.0 + (i - 1) * (n_rungs - 1.0) / (n_rungs - 1)) for i in 1:n_rungs]
    temperatures[1] = 1.0  # cold chain

    rung_chains = [ChainState(copy(chain.pars), chain.density) for _ in 1:n_rungs]
    rung_states = [initialise(sampler.base_sampler, rung_chains[i], model, rng) for i in 1:n_rungs]

    return ParallelTemperingState(
        rung_states,
        rung_chains,
        temperatures,
        zeros(Int, n_rungs - 1),
        zeros(Int, n_rungs - 1),
    )
end

function step!(sampler::MontyParallelTemperingSampler, chain::ChainState, state::ParallelTemperingState, model::MontyModel, rng::AbstractRNG)
    n_rungs = sampler.n_rungs

    # Step each rung with its tempered density
    for r in 1:n_rungs
        temp = state.temperatures[r]
        # Create a tempered model (preserving gradient if available)
        tempered_density = x -> temp * model.density(x)
        tempered_gradient = if model.gradient !== nothing
            x -> temp .* model.gradient(x)
        else
            nothing
        end
        tempered_model = MontyModel(
            model.parameters, tempered_density, tempered_gradient, nothing, model.domain,
            MontyModelProperties(
                is_stochastic=model.properties.is_stochastic,
                has_gradient=model.properties.has_gradient,
            ),
        )

        # Adjust chain density for temperature
        state.rung_chains[r].density = temp * model.density(state.rung_chains[r].pars)

        step!(sampler.base_sampler, state.rung_chains[r], state.rung_states[r], tempered_model, rng)
    end

    # Attempt swap between adjacent rungs
    for r in 1:(n_rungs - 1)
        state.swap_proposed[r] += 1
        β_i = state.temperatures[r]
        β_j = state.temperatures[r + 1]

        ll_i = model.density(state.rung_chains[r].pars)
        ll_j = model.density(state.rung_chains[r + 1].pars)

        log_alpha = (β_j - β_i) * (ll_i - ll_j)

        if log(rand(rng)) < log_alpha
            # Swap
            state.rung_chains[r].pars, state.rung_chains[r + 1].pars =
                state.rung_chains[r + 1].pars, state.rung_chains[r].pars
            state.swap_acceptance[r] += 1
        end
    end

    # Cold chain (rung 1) is the sample
    chain.pars .= state.rung_chains[1].pars
    chain.density = model.density(chain.pars)

    return true  # always "accepted" for the cold chain view
end
