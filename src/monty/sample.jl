# Main MCMC sampling loop.

"""
    MontySamples

Result of an MCMC sampling run.
"""
struct MontySamples
    pars::Array{Float64, 3}     # n_pars × n_samples × n_chains
    density::Matrix{Float64}    # n_samples × n_chains
    initial::Matrix{Float64}    # n_pars × n_chains
    parameter_names::Vector{String}
    details::Dict{Symbol, Any}
end

"""
    monty_sample(model, sampler, n_steps; kwargs...)

Run MCMC sampling.

## Arguments
- `model::MontyModel`: the target density
- `sampler::AbstractMontySampler`: the sampling algorithm
- `n_steps::Int`: number of MCMC steps
- `n_chains::Int=4`: number of chains
- `initial::Union{Nothing, Matrix{Float64}}=nothing`: initial parameters (n_pars × n_chains)
- `n_burnin::Int=0`: burn-in steps to discard
- `thinning::Int=1`: keep every `thinning`-th sample
- `runner::AbstractMontyRunner=MontySerialRunner()`: execution strategy
- `seed::Union{Nothing, Int}=nothing`: random seed
"""
function monty_sample(
    model::MontyModel,
    sampler::AbstractMontySampler,
    n_steps::Int;
    n_chains::Int=4,
    initial::Union{Nothing, Matrix{Float64}}=nothing,
    n_burnin::Int=0,
    thinning::Int=1,
    runner::AbstractMontyRunner=MontySerialRunner(),
    seed::Union{Nothing, Int}=nothing,
)
    n_pars = length(model.parameters)

    # Generate initial positions
    if initial === nothing
        if model.direct_sample !== nothing
            initial = zeros(Float64, n_pars, n_chains)
            rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
            for c in 1:n_chains
                initial[:, c] .= model.direct_sample(rng)
            end
        else
            error("No initial parameters provided and model has no direct_sample")
        end
    end

    # Set up RNGs
    base_rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    chain_rngs = [Random.Xoshiro(rand(base_rng, UInt64)) for _ in 1:n_chains]

    n_samples = div(n_steps - n_burnin, thinning)
    n_samples = max(n_samples, 0)

    if runner isa MontySerialRunner
        return _run_serial(model, sampler, n_steps, n_chains, n_burnin, thinning, n_samples, n_pars, initial, chain_rngs)
    elseif runner isa MontyThreadedRunner
        return _run_threaded(model, sampler, n_steps, n_chains, n_burnin, thinning, n_samples, n_pars, initial, chain_rngs)
    else
        error("Unknown runner type")
    end
end

function _run_chain(model, sampler, n_steps, n_burnin, thinning, n_samples, n_pars, initial_pars, rng)
    chain = ChainState(copy(initial_pars), model(initial_pars))
    sampler_state = initialise(sampler, chain, model, rng)

    pars_out = zeros(Float64, n_pars, n_samples)
    density_out = zeros(Float64, n_samples)
    n_accepted = 0
    sample_idx = 0

    for step in 1:n_steps
        accepted = step!(sampler, chain, sampler_state, model, rng)
        accepted && (n_accepted += 1)

        if step > n_burnin && (step - n_burnin) % thinning == 0
            sample_idx += 1
            if sample_idx <= n_samples
                pars_out[:, sample_idx] .= chain.pars
                density_out[sample_idx] = chain.density
            end
        end
    end

    acceptance_rate = n_accepted / n_steps
    return pars_out, density_out, acceptance_rate
end

function _run_serial(model, sampler, n_steps, n_chains, n_burnin, thinning, n_samples, n_pars, initial, chain_rngs)
    all_pars = zeros(Float64, n_pars, n_samples, n_chains)
    all_density = zeros(Float64, n_samples, n_chains)
    details = Dict{Symbol, Any}(:acceptance_rate => zeros(Float64, n_chains))

    for c in 1:n_chains
        pars_c, density_c, acc_rate = _run_chain(
            model, sampler, n_steps, n_burnin, thinning, n_samples, n_pars,
            initial[:, c], chain_rngs[c],
        )
        all_pars[:, :, c] .= pars_c
        all_density[:, c] .= density_c
        details[:acceptance_rate][c] = acc_rate
    end

    return MontySamples(all_pars, all_density, initial, model.parameters, details)
end

function _run_threaded(model, sampler, n_steps, n_chains, n_burnin, thinning, n_samples, n_pars, initial, chain_rngs)
    results = Vector{Any}(undef, n_chains)

    Threads.@threads for c in 1:n_chains
        results[c] = _run_chain(
            model, sampler, n_steps, n_burnin, thinning, n_samples, n_pars,
            initial[:, c], chain_rngs[c],
        )
    end

    all_pars = zeros(Float64, n_pars, n_samples, n_chains)
    all_density = zeros(Float64, n_samples, n_chains)
    details = Dict{Symbol, Any}(:acceptance_rate => zeros(Float64, n_chains))

    for c in 1:n_chains
        pars_c, density_c, acc_rate = results[c]
        all_pars[:, :, c] .= pars_c
        all_density[:, c] .= density_c
        details[:acceptance_rate][c] = acc_rate
    end

    return MontySamples(all_pars, all_density, initial, model.parameters, details)
end

"""
    monty_sample_continue(samples, model, sampler, n_steps; kwargs...)

Continue sampling from the final state of a previous run.
"""
function monty_sample_continue(
    prev::MontySamples,
    model::MontyModel,
    sampler::AbstractMontySampler,
    n_steps::Int;
    kwargs...,
)
    n_chains = size(prev.pars, 3)
    # Use last sample from each chain as initial
    initial = prev.pars[:, end, :]
    return monty_sample(model, sampler, n_steps; n_chains=n_chains, initial=initial, kwargs...)
end
