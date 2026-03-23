# Block Gibbs sampler — cycle through parameter blocks with sub-samplers.

"""
    MontyGibbsSampler

Block Gibbs sampler: cycles through parameter blocks, sampling each block
conditionally on the current values of all other blocks.

Each block can use a different sub-sampler (random walk, slice, MALA, etc.).
"""
struct MontyGibbsSampler <: AbstractMontySampler
    blocks::Vector{Vector{Int}}                    # parameter index groups
    sub_samplers::Vector{<:AbstractMontySampler}   # one sampler per block
end

mutable struct GibbsState <: AbstractSamplerState
    block_states::Vector{<:AbstractSamplerState}
    block_chains::Vector{ChainState}
    block_models::Vector{MontyModel}
end

"""
    monty_sampler_gibbs(blocks, sub_samplers)

Create a block Gibbs sampler.

## Arguments
- `blocks`: vector of parameter index groups, e.g. `[[1,2], [3,4]]`
- `sub_samplers`: one sampler per block (must match length of `blocks`)

Each block is sampled conditionally: the sub-sampler's model density is the full
model density evaluated with non-block parameters fixed at their current values.
"""
function monty_sampler_gibbs(
    blocks::Vector{Vector{Int}},
    sub_samplers::Vector{<:AbstractMontySampler},
)
    length(blocks) == length(sub_samplers) ||
        error("Number of blocks ($(length(blocks))) must equal number of sub_samplers ($(length(sub_samplers)))")
    isempty(blocks) && error("Must provide at least one block")
    return MontyGibbsSampler(blocks, sub_samplers)
end

function initialise(sampler::MontyGibbsSampler, chain::ChainState, model::MontyModel, rng::AbstractRNG)
    n_blocks = length(sampler.blocks)

    block_chains = Vector{ChainState}(undef, n_blocks)
    block_models = Vector{MontyModel}(undef, n_blocks)
    block_states = Vector{AbstractSamplerState}(undef, n_blocks)

    for b in 1:n_blocks
        idx = sampler.blocks[b]
        block_pars = chain.pars[idx]
        block_model = _make_conditional_model(model, chain.pars, idx)
        block_chain = ChainState(copy(block_pars), block_model(block_pars))

        block_chains[b] = block_chain
        block_models[b] = block_model
        block_states[b] = initialise(sampler.sub_samplers[b], block_chain, block_model, rng)
    end

    return GibbsState(block_states, block_chains, block_models)
end

function step!(sampler::MontyGibbsSampler, chain::ChainState, state::GibbsState, model::MontyModel, rng::AbstractRNG)
    any_accepted = false

    for b in 1:length(sampler.blocks)
        idx = sampler.blocks[b]

        # Rebuild conditional model with current full parameter vector
        state.block_models[b] = _make_conditional_model(model, chain.pars, idx)

        # Sync block chain with current full-chain values
        state.block_chains[b].pars .= chain.pars[idx]
        state.block_chains[b].density = state.block_models[b](chain.pars[idx])

        # One step of the sub-sampler on this block
        accepted = step!(
            sampler.sub_samplers[b],
            state.block_chains[b],
            state.block_states[b],
            state.block_models[b],
            rng,
        )

        # Write block samples back into full chain
        chain.pars[idx] .= state.block_chains[b].pars
        any_accepted = any_accepted || accepted
    end

    # Update full density
    chain.density = model(chain.pars)

    return any_accepted
end

"""
Build a MontyModel that evaluates the full model density but only exposes the
parameters in `block_idx`. Non-block parameters are fixed at `full_pars` values.
"""
function _make_conditional_model(model::MontyModel, full_pars::Vector{Float64}, block_idx::Vector{Int})
    fixed_pars = copy(full_pars)
    block_names = model.parameters[block_idx]
    n_block = length(block_idx)

    conditional_density = function(x_block)
        full = copy(fixed_pars)
        full[block_idx] .= x_block
        return model.density(full)
    end

    conditional_gradient = nothing
    if model.gradient !== nothing
        conditional_gradient = function(x_block)
            full = copy(fixed_pars)
            full[block_idx] .= x_block
            grad_full = model.gradient(full)
            return grad_full[block_idx]
        end
    end

    # Extract block domain
    block_domain = nothing
    if model.domain !== nothing
        block_domain = model.domain[block_idx, :]
    end

    return MontyModel(
        block_names,
        conditional_density,
        conditional_gradient,
        nothing,
        block_domain,
        MontyModelProperties(
            has_gradient=conditional_gradient !== nothing,
            has_direct_sample=false,
            is_stochastic=model.properties.is_stochastic,
            allow_multiple_parameters=false,
        ),
    )
end
