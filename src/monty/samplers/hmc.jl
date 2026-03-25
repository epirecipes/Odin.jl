# Hamiltonian Monte Carlo sampler with leapfrog integration.
# Supports constrained parameters via bijections (same transforms as NUTS).

"""
    MontyHMCSampler

Hamiltonian Monte Carlo sampler using leapfrog integration.
Supports constrained parameters via automatic bijections from model domain.
"""
struct MontyHMCSampler <: AbstractMontySampler
    epsilon::Float64            # step size
    n_integration_steps::Int    # number of leapfrog steps
    vcv::Union{Nothing, Matrix{Float64}}  # mass matrix (momentum covariance)
end

mutable struct HMCState <: AbstractSamplerState
    mass_matrix::Matrix{Float64}
    mass_chol::LowerTriangular{Float64}
    mass_inv::Matrix{Float64}
    momentum::Vector{Float64}
    position::Vector{Float64}
    # Bijector support
    bijector::Any               # nothing or _StackedForward
    inv_bijector::Any           # nothing or _StackedInverse
end

"""
    monty_sampler_hmc(epsilon, n_integration_steps; vcv=nothing)

Create an HMC sampler. The model must provide a gradient.
Constrained parameters (from model domain) are automatically handled via bijections.
"""
function monty_sampler_hmc(
    epsilon::Float64,
    n_integration_steps::Int;
    vcv::Union{Nothing, Matrix{Float64}}=nothing,
)
    return MontyHMCSampler(epsilon, n_integration_steps, vcv)
end

function initialise(sampler::MontyHMCSampler, chain::ChainState, model::MontyModel, rng::AbstractRNG)
    n = length(chain.pars)
    M = sampler.vcv !== nothing ? sampler.vcv : Matrix{Float64}(I, n, n)
    C = cholesky(Symmetric(M))
    M_inv = inv(M)

    # Build bijector from domain (shared with NUTS)
    bij, inv_bij = _build_bijector(model.domain, n)

    return HMCState(
        M,
        LowerTriangular(C.L),
        M_inv,
        zeros(Float64, n),
        zeros(Float64, n),
        bij,
        inv_bij,
    )
end

# Density + gradient in unconstrained space (with log-Jacobian correction)
function _hmc_density_gradient(model::MontyModel, q_unconstrained::Vector{Float64},
                                inv_bij, bij)
    if inv_bij !== nothing
        # Map to constrained space
        q_constrained = inv_bij(q_unconstrained)
        # Log-density in constrained space
        ld = model.density(q_constrained)
        # Add log-Jacobian of inverse transform
        ladj = _stacked_inv_logabsdetjac(inv_bij, q_unconstrained)
        target = ld + ladj
    else
        q_constrained = q_unconstrained
        target = model.density(q_unconstrained)
    end

    if !isfinite(target)
        return target, zeros(length(q_unconstrained))
    end

    # Gradient in unconstrained space via ForwardDiff
    grad = ForwardDiff.gradient(q_unconstrained) do y
        if inv_bij !== nothing
            x = inv_bij(y)
            model.density(x) + _stacked_inv_logabsdetjac(inv_bij, y)
        else
            model.density(y)
        end
    end

    return target, grad
end

function step!(sampler::MontyHMCSampler, chain::ChainState, state::HMCState, model::MontyModel, rng::AbstractRNG)
    model.gradient !== nothing || error("HMC requires a model with gradient")

    n = length(chain.pars)
    ε = sampler.epsilon
    L = sampler.n_integration_steps
    bij = state.bijector
    inv_bij = state.inv_bijector

    # Transform current position to unconstrained space
    q = if bij !== nothing
        Float64.(bij(chain.pars))
    else
        copy(chain.pars)
    end

    # Sample momentum: p ~ N(0, M)
    z = randn(rng, n)
    mul!(state.momentum, state.mass_chol, z)
    p = copy(state.momentum)

    # Current Hamiltonian (in unconstrained space)
    current_U_val, grad = _hmc_density_gradient(model, q, inv_bij, bij)
    current_U = -current_U_val
    current_K = 0.5 * dot(p, state.mass_inv * p)

    # Half step for momentum
    p .+= (ε / 2) .* grad

    # Leapfrog integration
    for step in 1:L
        # Full step for position
        q .+= ε .* (state.mass_inv * p)

        # Full step for momentum (except at end)
        if step < L
            _, grad = _hmc_density_gradient(model, q, inv_bij, bij)
            p .+= ε .* grad
        end
    end

    # Half step for momentum
    _, grad = _hmc_density_gradient(model, q, inv_bij, bij)
    p .+= (ε / 2) .* grad

    # Negate momentum
    p .*= -1

    # Proposed Hamiltonian (in unconstrained space)
    proposed_U_val, _ = _hmc_density_gradient(model, q, inv_bij, bij)
    proposed_U = -proposed_U_val
    proposed_K = 0.5 * dot(p, state.mass_inv * p)

    # Accept/reject
    log_alpha = (current_U + current_K) - (proposed_U + proposed_K)
    accepted = isfinite(log_alpha) && log(rand(rng)) < log_alpha

    if accepted && isfinite(proposed_U)
        # Store in constrained space
        if inv_bij !== nothing
            chain.pars .= inv_bij(q)
        else
            chain.pars .= q
        end
        chain.density = -proposed_U
    end

    return accepted
end
