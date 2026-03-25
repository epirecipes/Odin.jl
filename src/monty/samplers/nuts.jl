# NUTS sampler via AdvancedHMC.jl with bijector support for constrained parameters.
# Bijector infrastructure (_build_bijector, _StackedForward, etc.) is in helpers.jl.

using AdvancedHMC
using LogDensityProblems

"""
    MontyNUTSSampler

No-U-Turn Sampler (NUTS) using AdvancedHMC.jl as backend.
Supports constrained parameters via automatic bijections — domain bounds are
mapped to unconstrained (real-line) space for efficient sampling.
"""
struct MontyNUTSSampler <: AbstractMontySampler
    target_acceptance::Float64
    max_depth::Int
    n_adaption::Union{Nothing, Int}
    metric_type::Symbol  # :diag or :dense
end

"""
    monty_sampler_nuts(; target_acceptance=0.8, max_depth=10, n_adaption=nothing, metric=:diag)

Create a NUTS sampler.

The model must provide a gradient (e.g. from `dust_likelihood_monty` with an unfilter,
or a `@monty_prior` model, or any `MontyModel` with `has_gradient=true`).

## Keyword arguments
- `target_acceptance`: target Metropolis acceptance probability (default 0.8)
- `max_depth`: maximum tree depth (default 10)
- `n_adaption`: number of warmup/adaptation steps (default: half of n_steps)
- `metric`: mass matrix type, `:diag` or `:dense` (default `:diag`)
"""
function monty_sampler_nuts(;
    target_acceptance::Float64=0.8,
    max_depth::Int=10,
    n_adaption::Union{Nothing, Int}=nothing,
    metric::Symbol=:diag,
)
    metric in (:diag, :dense) || error("metric must be :diag or :dense, got :$metric")
    return MontyNUTSSampler(target_acceptance, max_depth, n_adaption, metric)
end


# ── LogDensityProblems bridge ───────────────────────────────

"""
    MontyLogDensity

Wraps a MontyModel + optional bijector to expose the LogDensityProblems interface.
When a bijector is present, `logdensity` operates in unconstrained space and adds
the log-Jacobian correction.
"""
struct MontyLogDensity{M<:MontyModel, B, IB}
    model::M
    bijector::B          # nothing, or callable (constrained → unconstrained)
    inv_bijector::IB     # nothing, or callable (unconstrained → constrained)
    dim::Int
end

function LogDensityProblems.capabilities(::Type{<:MontyLogDensity})
    return LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.dimension(ld::MontyLogDensity)
    return ld.dim
end

function LogDensityProblems.logdensity(ld::MontyLogDensity, x_unconstrained::AbstractVector)
    if ld.inv_bijector === nothing
        return ld.model.density(x_unconstrained)
    else
        x_constrained = ld.inv_bijector(x_unconstrained)
        ld_val = ld.model.density(x_constrained)
        lj = _stacked_inv_logabsdetjac(ld.inv_bijector, x_unconstrained)
        return ld_val + lj
    end
end

function LogDensityProblems.logdensity_and_gradient(ld::MontyLogDensity, x_unconstrained::AbstractVector)
    if ld.inv_bijector === nothing
        ll = ld.model.density(x_unconstrained)
        grad = ld.model.gradient(x_unconstrained)
        return ll, grad
    else
        # ForwardDiff through the full pipeline (transform + density + Jacobian)
        function _full_logdensity(x)
            x_c = ld.inv_bijector(x)
            ld_val = ld.model.density(x_c)
            lj = _stacked_inv_logabsdetjac(ld.inv_bijector, x)
            return ld_val + lj
        end
        ll = _full_logdensity(x_unconstrained)
        grad = ForwardDiff.gradient(_full_logdensity, x_unconstrained)
        return ll, grad
    end
end


# ── Sampler interface ───────────────────────────────────────

mutable struct NUTSState <: AbstractSamplerState
    logdensity::MontyLogDensity
    hamiltonian::Any            # AdvancedHMC.Hamiltonian
    kernel::Any                 # AdvancedHMC.HMCKernel
    adaptor::Any                # AdvancedHMC.StanHMCAdaptor
    θ_current::Vector{Float64}  # current unconstrained position
    n_adaption::Int
    step_count::Int
end

function initialise(sampler::MontyNUTSSampler, chain::ChainState, model::MontyModel, rng::AbstractRNG)
    model.gradient !== nothing || error("NUTS requires a model with gradient")

    n = length(chain.pars)

    # Build bijector from domain
    bij, inv_bij = _build_bijector(model.domain, n)
    ld = MontyLogDensity(model, bij, inv_bij, n)

    # Transform initial position to unconstrained space
    θ_init = if bij !== nothing
        Float64.(bij(chain.pars))
    else
        copy(chain.pars)
    end

    # Set up AdvancedHMC components
    metric = if sampler.metric_type == :diag
        AdvancedHMC.DiagEuclideanMetric(n)
    else
        AdvancedHMC.DenseEuclideanMetric(n)
    end

    hamiltonian = AdvancedHMC.Hamiltonian(metric, ld)
    initial_ε = AdvancedHMC.find_good_stepsize(hamiltonian, θ_init)
    integrator = AdvancedHMC.Leapfrog(initial_ε)

    kernel = AdvancedHMC.HMCKernel(
        AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            integrator,
            AdvancedHMC.GeneralisedNoUTurn(max_depth=sampler.max_depth),
        ),
    )

    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric),
        AdvancedHMC.StepSizeAdaptor(sampler.target_acceptance, integrator),
    )

    # Compute initial density in constrained space
    chain.density = LogDensityProblems.logdensity(ld, θ_init)

    n_adapt = sampler.n_adaption === nothing ? 0 : sampler.n_adaption

    return NUTSState(ld, hamiltonian, kernel, adaptor, θ_init, n_adapt, 0)
end

function step!(sampler::MontyNUTSSampler, chain::ChainState, state::NUTSState, model::MontyModel, rng::AbstractRNG)
    state.step_count += 1

    # Create a PhasePoint from current position
    z = AdvancedHMC.phasepoint(
        state.hamiltonian, state.θ_current, randn(rng, length(state.θ_current)),
    )

    # Perform one NUTS transition
    t = AdvancedHMC.transition(rng, state.hamiltonian, state.kernel, z)

    # Update current unconstrained position
    state.θ_current = t.z.θ

    # Adapt during warmup
    n_adapt = state.n_adaption > 0 ? state.n_adaption : sampler.n_adaption
    if n_adapt !== nothing && state.step_count <= n_adapt
        state.hamiltonian, state.kernel, _ = AdvancedHMC.adapt!(
            state.hamiltonian, state.kernel, state.adaptor,
            state.step_count, n_adapt, state.θ_current, t.stat.acceptance_rate,
        )
    end

    # Map back to constrained space for storage
    if state.logdensity.inv_bijector !== nothing
        chain.pars .= state.logdensity.inv_bijector(state.θ_current)
    else
        chain.pars .= state.θ_current
    end
    chain.density = LogDensityProblems.logdensity(state.logdensity, state.θ_current)

    return !t.stat.numerical_error
end
