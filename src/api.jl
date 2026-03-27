# ── Julia-friendly API layer ───────────────────────────────────
#
# Provides clean, idiomatic Julia names for the Odin.jl public API.
# The original R-style names (dust_*, monty_*) remain as aliases for
# backward compatibility. New code should prefer the names in this file.
#
# Design principles:
#   - Types as constructors: Likelihood(model, data) not dust_unfilter_create(...)
#   - Drop package prefixes: no dust_/monty_ since everything is in one module
#   - Short generic verbs: simulate, loglik, sample, predict
#   - Julia conventions: PascalCase types, snake_case functions, ! for mutation

# ═══════════════════════════════════════════════════════════════
# Type aliases (only for types that don't clash)
# ═══════════════════════════════════════════════════════════════

"""
    OdinModel

A compiled odin model, created by `@odin begin ... end`.

Alias for `DustSystemGenerator`. Use with `simulate`, `Likelihood`, etc.
"""
const OdinModel = DustSystemGenerator

"""
    Samples

Result of MCMC sampling via `sample(model, sampler, n_steps; ...)`.

Alias for `MontySamples`. Fields:
- `pars`: parameter array (n_pars × n_samples × n_chains)
- `density`: log-density at each sample
- `parameter_names`: names of parameters
"""
const Samples = MontySamples

"""
    ObservedData{D}

Time-series observation data for likelihood evaluation.

Alias for `FilterData`. Create with `ObservedData(data_list)` or pass
a vector of NamedTuples with a `:time` field.
"""
const ObservedData = FilterData

"""
    ODEControl

Configuration for ODE solvers (tolerances, max steps, solver choice).

Alias for `DustODEControl`.
"""
const ODEControl = DustODEControl

# ObservedData as a constructor function (wraps dust_filter_data)
"""
    ObservedData(data; time_field=:time)

Prepare observation data for likelihood evaluation.

`data` is a vector of NamedTuples, each with a `:time` field.

# Example
```julia
obs = ObservedData([(time=1.0, cases=10.0), (time=2.0, cases=25.0)])
```
"""
ObservedData(data::AbstractVector{<:NamedTuple}; kwargs...) =
    dust_filter_data(data; kwargs...)

# ═══════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════

"""
    simulate(model, pars, times; kwargs...)

Simulate an odin model and return state trajectories.

# Arguments
- `model::OdinModel`: compiled model from `@odin`
- `pars::NamedTuple`: parameter values
- `times::AbstractVector`: times at which to record state

# Keyword Arguments
- `n_particles::Int=1`: number of stochastic realisations
- `dt::Float64=1.0`: time step (discrete models)
- `seed::Union{Nothing,Int}=nothing`: RNG seed
- `solver::Symbol=:dp5`: ODE solver (`:dp5`, `:sdirk`)

# Returns
`Array{Float64,3}` of size `(n_state, n_particles, n_times)`.

# Example
```julia
model = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0; initial(I) = I0; initial(R) = 0
    beta = parameter(0.4); gamma = parameter(0.2)
    I0 = parameter(10); N = parameter(1000)
end
result = simulate(model, (beta=0.4, gamma=0.2, I0=10.0, N=1000.0),
                  0.0:1.0:100.0)
```
"""
function simulate(gen::DustSystemGenerator, pars::NamedTuple,
                  times::AbstractVector; kwargs...)
    dust_system_simulate(gen, pars; times=Float64.(collect(times)), kwargs...)
end

# Also accept times as keyword (backward compat with dust_system_simulate)
function simulate(gen::DustSystemGenerator, pars::NamedTuple;
                  times::AbstractVector, kwargs...)
    dust_system_simulate(gen, pars; times=Float64.(collect(times)), kwargs...)
end

"""
    simulate(sys, times; kwargs...)

Simulate from an existing `System` (mutable state).
"""
function simulate(sys::DustSystem, times::AbstractVector; kwargs...)
    dust_system_simulate(sys, Float64.(collect(times)); kwargs...)
end

function simulate(sys::DustSystem; times::AbstractVector, kwargs...)
    dust_system_simulate(sys, Float64.(collect(times)); kwargs...)
end

"""
    simulate(model, pars_vec::Vector{NamedTuple}, times; kwargs...)

Multi-group simulation: returns `(n_state+n_output, n_particles, n_times, n_groups)`.
"""
function simulate(gen::DustSystemGenerator, pars_vec::AbstractVector{<:NamedTuple},
                  times::AbstractVector; kwargs...)
    dust_system_simulate(gen, pars_vec; times=Float64.(collect(times)), kwargs...)
end

# ── System construction helpers ──────────────────────────────

"""
    System(model, pars; kwargs...)

Create a mutable simulation system from a compiled model.

Equivalent to `dust_system_create`. Useful when you need to
inspect or modify state between simulation steps.

# Example
```julia
sys = System(model, pars; n_particles=10)
reset!(sys)
result = simulate(sys, times)
```
"""
System(gen::DustSystemGenerator, pars::NamedTuple; kwargs...) =
    dust_system_create(gen, pars; kwargs...)

"""
    System(model, pars_vec::Vector{NamedTuple}; kwargs...)

Create a multi-group simulation system.
"""
System(gen::DustSystemGenerator, pars_vec::AbstractVector{<:NamedTuple}; kwargs...) =
    dust_system_create(gen, pars_vec; kwargs...)

"""
    reset!(sys)

Reset system state to initial conditions.
"""
function reset!(sys::DustSystem)
    if sys.n_groups > 1
        dust_system_set_state_initial!(sys, Val(:grouped))
    else
        dust_system_set_state_initial!(sys)
    end
end

"""
    state(sys)

Get current state matrix (n_state × n_particles).
"""
state(sys::DustSystem) = dust_system_state(sys)

"""
    run_to!(sys, time)

Advance system to the specified time.
"""
run_to!(sys::DustSystem, time::Real) =
    dust_system_run_to_time!(sys, Float64(time))


# ═══════════════════════════════════════════════════════════════
# Likelihood
# ═══════════════════════════════════════════════════════════════

"""
    Likelihood(model, data; kwargs...)

Create a likelihood evaluator for fitting a model to observed data.

Automatically selects deterministic (ODE) or stochastic (particle filter)
evaluation based on the `n_particles` keyword argument.

# Deterministic (default)
```julia
lik = Likelihood(model, data)               # ODE-based
ll  = loglik(lik, pars)
```

# Stochastic (particle filter)
```julia
lik = Likelihood(model, data; n_particles=200)  # bootstrap PF
ll  = loglik(lik, pars)
```

# Arguments
- `model::OdinModel`: compiled model with `data()` and `~` comparison
- `data`: vector of NamedTuples with `:time` field, or `ObservedData`

# Keyword Arguments (deterministic)
- `time_start::Float64=0.0`
- `ode_control::ODEControl=ODEControl()`

# Keyword Arguments (stochastic)
- `n_particles::Int`: triggers particle filter mode
- `time_start::Float64=0.0`
- `dt::Float64=1.0`
- `seed::Union{Nothing,Int}=nothing`
- `save_trajectories::Bool=false`
"""
struct Likelihood{T}
    inner::T
end

# Deterministic (unfilter) — default path
function Likelihood(gen::DustSystemGenerator, data::FilterData;
                    time_start::Float64=0.0,
                    ode_control::DustODEControl=DustODEControl(),
                    n_particles::Union{Nothing,Int}=nothing,
                    kwargs...)
    if n_particles !== nothing
        return Likelihood(dust_filter_create(gen, data;
            time_start=time_start, n_particles=n_particles, kwargs...))
    else
        return Likelihood(dust_unfilter_create(gen, data;
            time_start=time_start, ode_control=ode_control))
    end
end

# Accept raw data vector — auto-wrap in FilterData
function Likelihood(gen::DustSystemGenerator, data::AbstractVector{<:NamedTuple};
                    kwargs...)
    Likelihood(gen, dust_filter_data(data); kwargs...)
end

# Wrap existing DustFilter/DustUnfilter
Likelihood(f::DustFilter) = Likelihood{DustFilter}(f)
Likelihood(u::DustUnfilter) = Likelihood{DustUnfilter}(u)

"""
    loglik(lik, pars)

Evaluate the log-likelihood at the given parameters.

# Example
```julia
lik = Likelihood(model, data)
ll = loglik(lik, (beta=0.4, gamma=0.2, I0=10.0, N=1000.0))
```
"""
loglik(lik::Likelihood{<:DustUnfilter}, pars::NamedTuple) =
    dust_unfilter_run!(lik.inner, pars)
loglik(lik::Likelihood{<:DustFilter}, pars::NamedTuple) =
    dust_likelihood_run!(lik.inner, pars)

"""
    loglik_pointwise(lik, pars)

Evaluate per-time-point log-likelihoods (for WAIC, LOO, etc.).
"""
loglik_pointwise(lik::Likelihood{<:DustUnfilter}, pars::NamedTuple) =
    dust_unfilter_run_pointwise!(lik.inner, pars)
loglik_pointwise(lik::Likelihood{<:DustFilter}, pars::NamedTuple) =
    dust_filter_run_pointwise!(lik.inner, pars)

"""
    loglik_gradient(lik, pars)

Compute the gradient of the log-likelihood w.r.t. parameters (via AD).
"""
loglik_gradient(lik::Likelihood{<:DustUnfilter}, pars::NamedTuple) =
    dust_unfilter_gradient(lik.inner, pars)

# ── Convenience: also work on bare DustUnfilter/DustFilter ────
loglik(uf::DustUnfilter, pars::NamedTuple) = dust_unfilter_run!(uf, pars)
loglik(f::DustFilter, pars::NamedTuple) = dust_likelihood_run!(f, pars)
loglik_pointwise(uf::DustUnfilter, pars::NamedTuple) = dust_unfilter_run_pointwise!(uf, pars)
loglik_pointwise(f::DustFilter, pars::NamedTuple) = dust_filter_run_pointwise!(f, pars)
loglik_gradient(uf::DustUnfilter, pars::NamedTuple) = dust_unfilter_gradient(uf, pars)

"""
    as_model(lik, packer)

Convert a `Likelihood` (or bare `DustUnfilter`/`DustFilter`) into a
`MontyModel` suitable for MCMC sampling.

# Example
```julia
lik = Likelihood(model, data)
packer = Packer([:beta, :gamma])
m = as_model(lik, packer)
samples = sample(m, nuts(), 2000)
```
"""
as_model(lik::Likelihood, packer::MontyPacker) =
    dust_likelihood_monty(lik.inner, packer)
as_model(uf::DustUnfilter, packer::MontyPacker) =
    dust_likelihood_monty(uf, packer)
as_model(f::DustFilter, packer::MontyPacker) =
    dust_likelihood_monty(f, packer)


# ═══════════════════════════════════════════════════════════════
# Model (density wrapper)
# ═══════════════════════════════════════════════════════════════

"""
    DensityModel(density; parameters, gradient=nothing, kwargs...)

Create a model from a log-density function for MCMC sampling.

Wraps `monty_model`. For most uses, prefer `as_model(lik, packer)`
or `@prior` combined with `+`.

# Example
```julia
m = DensityModel(θ -> -sum(θ.^2); parameters=["x", "y"])
```
"""
DensityModel(density::Function; kwargs...) = monty_model(density; kwargs...)


# ═══════════════════════════════════════════════════════════════
# Parameter Packing (function wrappers, not type aliases)
# ═══════════════════════════════════════════════════════════════

"""
    Packer(names; array=Dict(), fixed=NamedTuple(), process=nothing)

Create a parameter packer that maps between named parameters and flat vectors.

# Example
```julia
pk = Packer([:beta, :gamma])
pk = Packer([:beta]; array=Dict(:gamma => (3,)))
```
"""
Packer(args...; kwargs...) = monty_packer(args...; kwargs...)

"""
    GroupedPacker(groups; kwargs...)

Create a grouped parameter packer for stratified models.
"""
GroupedPacker(args...; kwargs...) = monty_packer_grouped(args...; kwargs...)


# ═══════════════════════════════════════════════════════════════
# Samplers (function constructors to avoid clashes with
# AdvancedHMC.NUTS, AdvancedHMC.HMC, DynamicPPL.Model, etc.)
# ═══════════════════════════════════════════════════════════════

"""
    nuts(; target_acceptance=0.8, max_depth=10, metric=:diag)

No-U-Turn Sampler (requires model with gradient).
"""
nuts(; kwargs...) = monty_sampler_nuts(; kwargs...)

"""
    random_walk(vcv; boundaries=:reflect)

Random-walk Metropolis–Hastings sampler.

# Example
```julia
sampler = random_walk(0.01 * I(2))
```
"""
random_walk(vcv::AbstractMatrix; kwargs...) =
    monty_sampler_random_walk(vcv; kwargs...)

"""
    hmc(epsilon, n_leapfrog; vcv=nothing)

Hamiltonian Monte Carlo sampler (requires gradient).
"""
hmc(epsilon::Real, n_leapfrog::Int; kwargs...) =
    monty_sampler_hmc(Float64(epsilon), n_leapfrog; kwargs...)

"""
    adaptive_mh(initial_vcv; kwargs...)

Adaptive Metropolis–Hastings with online covariance learning.
"""
adaptive_mh(vcv::AbstractMatrix; kwargs...) =
    monty_sampler_adaptive(vcv; kwargs...)

"""
    mala(epsilon; vcv=nothing)

Metropolis-Adjusted Langevin Algorithm (requires gradient).
"""
mala(epsilon::Real; kwargs...) =
    monty_sampler_mala(Float64(epsilon); kwargs...)

"""
    slice(; w=1.0, max_steps=10)

Slice sampler (no gradient required).
"""
slice(; kwargs...) = monty_sampler_slice(; kwargs...)

"""
    parallel_tempering(base_sampler, n_rungs)

Parallel tempering (replica exchange) wrapper around any base sampler.
"""
parallel_tempering(base::AbstractMontySampler, n::Int) =
    monty_sampler_parallel_tempering(base, n)

"""
    gibbs(blocks, sub_samplers)

Block Gibbs sampler that cycles through parameter groups.
"""
gibbs(blocks, samplers) = monty_sampler_gibbs(blocks, samplers)


# ═══════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════

"""
    Serial()

Run MCMC chains sequentially.
"""
Serial() = monty_runner_serial()

"""
    Threaded()

Run MCMC chains in parallel using Julia threads.
"""
Threaded() = monty_runner_threaded()

"""
    Simultaneous()

Run all MCMC chains simultaneously in lock-step.
Enables cross-chain interactions (e.g., parallel tempering swaps).
"""
Simultaneous() = monty_runner_simultaneous()

"""
    DistributedRunner()

Run MCMC chains on distributed workers via `Distributed.jl`.
Requires `addprocs()` and `@everywhere using Odin` before use.
Falls back to threaded execution if no workers are available.
"""
DistributedRunner() = monty_runner_distributed()


# ═══════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════

"""
    sample(model, sampler, n_steps; n_chains=4, runner=Serial(), kwargs...)

Run MCMC sampling. This is the main inference entry point.

# Example
```julia
posterior = as_model(lik, packer) + @monty_prior(beta ~ Exponential(0.5))
samples = sample(posterior, NUTS(), 2000; n_chains=4)
```

# Keyword Arguments
- `n_chains::Int=4`
- `initial::Union{Nothing,Matrix{Float64}}=nothing`
- `n_burnin::Int=0`
- `thinning::Int=1`
- `runner=Serial()`: or `Threaded()`
- `seed::Union{Nothing,Int}=nothing`
"""
StatsBase.sample(model::MontyModel, sampler::AbstractMontySampler, n_steps::Int; kwargs...) =
    monty_sample(model, sampler, n_steps; kwargs...)

"""
    sample_continue(samples, model, sampler, n_more; kwargs...)

Continue an existing MCMC chain for additional steps.
"""
sample_continue(prev::MontySamples, model::MontyModel,
                sampler::AbstractMontySampler, n::Int; kwargs...) =
    monty_sample_continue(prev, model, sampler, n; kwargs...)


# ═══════════════════════════════════════════════════════════════
# Prior DSL
# ═══════════════════════════════════════════════════════════════

"""
    @prior begin
        beta ~ Exponential(mean=0.5)
        gamma ~ Gamma(2, 0.1)
    end

Define a prior as a `Model` with automatic gradient support.

Alias for `@monty_prior`.
"""
macro prior(block)
    esc(:(Odin.@monty_prior $block))
end


# ═══════════════════════════════════════════════════════════════
# Diagnostics & Validation
# ═══════════════════════════════════════════════════════════════

"""
    posterior_predict(samples, model; times, kwargs...)

Generate posterior predictive simulations.

Alias for `posterior_predictive`.
"""
posterior_predict(samples::MontySamples, gen::DustSystemGenerator; kwargs...) =
    posterior_predictive(samples, gen; kwargs...)


# ═══════════════════════════════════════════════════════════════
# Model selection — drop the `compute_` prefix
# ═══════════════════════════════════════════════════════════════

"""AIC: Akaike Information Criterion."""
aic(ll::Float64, k::Int) = compute_aic(ll, k)

"""AICc: corrected AIC for small samples."""
aicc(ll::Float64, k::Int, n::Int) = compute_aicc(ll, k, n)

"""BIC: Bayesian Information Criterion."""
bic(ll::Float64, k::Int, n::Int) = compute_bic(ll, k, n)

"""DIC: Deviance Information Criterion."""
dic(samples::MontySamples, fn::Function) = compute_dic(samples, fn)

"""WAIC: Widely Applicable Information Criterion."""
waic(pw::Matrix{Float64}) = compute_waic(pw)

"""LOO: Leave-One-Out cross-validation."""
loo(pw::Matrix{Float64}) = compute_loo(pw)

"""Compare multiple models by information criteria."""
compare(; n_observations::Int, models...) =
    compare_models(; n_observations=n_observations, models...)


# ═══════════════════════════════════════════════════════════════
# Sensitivity analysis
# ═══════════════════════════════════════════════════════════════

"""
    sensitivity(model, pars; method=:forward, kwargs...)

Unified sensitivity analysis interface.

# Methods
- `:forward` — forward sensitivity equations (∂u/∂p)
- `:adjoint` — adjoint sensitivity (∂L/∂p, requires loss function)
- `:sobol` — variance-based Sobol indices
- `:morris` — Morris screening (μ*, σ)

# Example
```julia
result = sensitivity(model, pars; method=:sobol, times=0:1:100,
                     params_of_interest=[:beta, :gamma], n_samples=1000)
```
"""
function sensitivity(gen::DustSystemGenerator, pars;
                     method::Symbol=:forward, kwargs...)
    if method == :forward
        dust_sensitivity_forward(gen, pars; kwargs...)
    elseif method == :adjoint
        dust_sensitivity_adjoint(gen, pars; kwargs...)
    elseif method == :sobol
        dust_sensitivity_sobol(gen, pars; kwargs...)
    elseif method == :morris
        dust_sensitivity_morris(gen, pars; kwargs...)
    else
        error("Unknown sensitivity method: $method. Use :forward, :adjoint, :sobol, or :morris.")
    end
end

# 3-arg variant for adjoint (with loss function)
function sensitivity(gen::DustSystemGenerator, pars, loss_fn::Function;
                     method::Symbol=:adjoint, kwargs...)
    if method == :adjoint
        dust_sensitivity_adjoint(gen, pars, loss_fn; kwargs...)
    else
        error("3-argument sensitivity only supports :adjoint method")
    end
end


# ═══════════════════════════════════════════════════════════════
# Categorical (category theory) — cleaner names
# ═══════════════════════════════════════════════════════════════

"""
    SIR(; S0=990, I0=10, R0=0, beta=:beta, gamma=:gamma)

Pre-built SIR epidemiological network.
"""
SIR(; kwargs...) = sir_net(; kwargs...)

"""SEIR epidemiological network."""
SEIR(; kwargs...) = seir_net(; kwargs...)

"""SIS epidemiological network."""
SIS(; kwargs...) = sis_net(; kwargs...)

"""SIRS epidemiological network."""
SIRS(; kwargs...) = sirs_net(; kwargs...)

"""SEIRS epidemiological network."""
SEIRS(; kwargs...) = seirs_net(; kwargs...)

"""SIR with vaccination."""
SIRVax(; kwargs...) = sir_vax_net(; kwargs...)

"""Two-strain SIR network."""
TwoStrainSIR(; kwargs...) = two_strain_sir_net(; kwargs...)

"""
    compile(net; mode=:ode, kwargs...)

Compile an EpiNet into an `OdinModel` for simulation and inference.

Alias for `lower`.

# Example
```julia
net = SIR() |> stratify(; by=:age, groups=["young", "old"])
model = compile(net)
result = simulate(model, pars, times)
```
"""
compile(net::EpiNet; kwargs...) = lower(net; kwargs...)


# ═══════════════════════════════════════════════════════════════
# Observer
# ═══════════════════════════════════════════════════════════════

"""
    Observer(observe; finalise=auto, combine=auto, append=auto)

Create an observer for collecting custom outputs during MCMC sampling.

The `observe` function is called after each accepted sample with
`(model, rng)` and should return an observation (e.g., NamedTuple).

# Example
```julia
obs = Observer((model, rng) -> (trajectories = last_trajectories(filter),))
samples = sample(posterior, sampler, 1000; observer=obs)
samples.observations  # combined observations
```
"""
const Observer = MontyObserver


# ═══════════════════════════════════════════════════════════════
# Likelihood — multi-group overloads
# ═══════════════════════════════════════════════════════════════

"""
    Likelihood(model, group_data::Vector{FilterData}; kwargs...)

Create a multi-group likelihood from per-group data.
"""
function Likelihood(gen::DustSystemGenerator, group_data::Vector{<:FilterData};
                    time_start::Float64=0.0,
                    ode_control::DustODEControl=DustODEControl(),
                    n_particles::Union{Nothing,Int}=nothing,
                    kwargs...)
    if n_particles !== nothing
        return Likelihood(dust_filter_create(gen, group_data;
            time_start=time_start, n_particles=n_particles, kwargs...))
    else
        return Likelihood(dust_unfilter_create(gen, group_data;
            time_start=time_start, ode_control=ode_control))
    end
end

# ═══════════════════════════════════════════════════════════════
# Model introspection
# ═══════════════════════════════════════════════════════════════

"""
    validate_model(block::Expr) -> OdinValidationResult

Parse an odin DSL expression and return structured diagnostics without compiling.

# Example
```julia
result = validate_model(quote
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0; initial(I) = I0; initial(R) = 0
    beta = parameter(0.4); gamma = parameter(0.2)
    I0 = parameter(10); N = parameter(1000)
end)
result.success        # true
result.time_type      # :continuous
result.state_variables # [:S, :I, :R]
```
"""
validate_model(block::Expr) = odin_validate(block)

"""
    show_code(block::Expr; what=:all) -> Expr

Generate and return the Julia code for an odin DSL block.

`what` can be `:all` (default) or a specific method name such as `:update`,
`:rhs`, `:initial`, `:compare`, `:output`, or `:diffusion`.

# Example
```julia
code = show_code(quote
    update(S) = S - n_SI
    update(I) = I + n_SI - n_IR
    update(R) = R + n_IR
    initial(S) = N - I0; initial(I) = I0; initial(R) = 0
    n_SI = Binomial(S, beta); n_IR = Binomial(I, gamma)
    beta = parameter(0.4); gamma = parameter(0.2)
    I0 = parameter(10); N = parameter(1000)
end)
```
"""
show_code(block::Expr; what::Symbol=:all) = odin_show(block; what=what)
