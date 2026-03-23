# Metropolis-Adjusted Langevin Algorithm (MALA).
# Gradient-based proposal with Metropolis-Hastings correction.

"""
    MontyMALASampler

Metropolis-Adjusted Langevin Algorithm sampler.

Proposal: θ* = θ + ε²/2 * ∇log π(θ) + ε * N(0, M)

Requires a model with gradient. Simpler than HMC (single gradient evaluation per
step) but still benefits from gradient information.
"""
struct MontyMALASampler <: AbstractMontySampler
    epsilon::Float64            # step size
    vcv::Union{Nothing, Matrix{Float64}}  # mass matrix (preconditioning)
end

mutable struct MALAState <: AbstractSamplerState
    mass_matrix::Matrix{Float64}
    mass_chol::LowerTriangular{Float64}
    mass_inv::Matrix{Float64}
    proposal::Vector{Float64}
    gradient_current::Vector{Float64}
end

"""
    monty_sampler_mala(epsilon; vcv=nothing)

Create a MALA sampler. The model must provide a gradient.

## Arguments
- `epsilon`: step size
- `vcv`: mass/preconditioning matrix (default: identity)
"""
function monty_sampler_mala(epsilon::Float64; vcv::Union{Nothing, Matrix{Float64}}=nothing)
    epsilon > 0 || error("epsilon must be positive")
    return MontyMALASampler(epsilon, vcv)
end

function initialise(sampler::MontyMALASampler, chain::ChainState, model::MontyModel, rng::AbstractRNG)
    model.gradient !== nothing || error("MALA requires a model with gradient")
    n = length(chain.pars)
    M = sampler.vcv !== nothing ? sampler.vcv : Matrix{Float64}(I, n, n)
    C = cholesky(Symmetric(M))
    M_inv = inv(M)
    grad = model.gradient(chain.pars)
    return MALAState(M, LowerTriangular(C.L), M_inv, similar(chain.pars), grad)
end

function step!(sampler::MontyMALASampler, chain::ChainState, state::MALAState, model::MontyModel, rng::AbstractRNG)
    n = length(chain.pars)
    ε = sampler.epsilon

    # Current gradient
    grad_current = model.gradient(chain.pars)
    state.gradient_current .= grad_current

    # Proposal mean: θ + ε²/2 * M⁻¹ * ∇log π(θ)
    drift = (ε^2 / 2) .* (state.mass_inv * grad_current)
    proposal_mean = chain.pars .+ drift

    # Proposal: θ* = proposal_mean + ε * chol(M) * z, z ~ N(0, I)
    z = randn(rng, n)
    state.proposal .= proposal_mean .+ ε .* (state.mass_chol * z)

    # Handle domain
    if model.domain !== nothing && !in_domain(state.proposal, model.domain)
        return false
    end

    # Evaluate density and gradient at proposal
    density_prop = model(state.proposal)
    if !isfinite(density_prop)
        return false
    end
    grad_prop = model.gradient(state.proposal)

    # Reverse proposal mean: θ* + ε²/2 * M⁻¹ * ∇log π(θ*)
    drift_rev = (ε^2 / 2) .* (state.mass_inv * grad_prop)
    reverse_mean = state.proposal .+ drift_rev

    # Log proposal densities (multivariate normal)
    # q(θ* | θ) = N(θ*; proposal_mean, ε² M)
    # q(θ | θ*) = N(θ; reverse_mean, ε² M)
    log_q_forward = _mala_log_proposal(state.proposal, proposal_mean, ε, state.mass_inv, n)
    log_q_reverse = _mala_log_proposal(chain.pars, reverse_mean, ε, state.mass_inv, n)

    # Metropolis-Hastings acceptance
    log_alpha = (density_prop - chain.density) + (log_q_reverse - log_q_forward)
    accepted = isfinite(log_alpha) && log(rand(rng)) < log_alpha

    if accepted
        chain.pars .= state.proposal
        chain.density = density_prop
        state.gradient_current .= grad_prop
    end

    return accepted
end

# Log density of N(x; mean, ε² M) up to normalising constant
# = -1/2 (x - mean)' (ε² M)⁻¹ (x - mean)
# = -1/(2ε²) (x - mean)' M⁻¹ (x - mean)
function _mala_log_proposal(x::Vector{Float64}, mean::Vector{Float64},
                            ε::Float64, M_inv::Matrix{Float64}, n::Int)
    diff = x .- mean
    return -0.5 / (ε^2) * dot(diff, M_inv * diff)
end
