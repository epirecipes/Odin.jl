# Model validation and diagnostic tools: posterior predictive checks,
# residual diagnostics, calibration assessment, and simulation-based calibration.
#
# Uses StatsBase (already imported in Odin.jl) for mean, median, std, quantile.

# ── Result types ─────────────────────────────────────────────

"""Get state + output variable names by creating a temporary system."""
function _get_all_variable_names(generator::DustSystemGenerator, pars::NamedTuple; dt::Float64=1.0)
    sys = dust_system_create(generator, pars; n_particles=1, dt=dt)
    return vcat(sys.state_names, sys.output_names)
end

"""
    PosteriorPredictive

Result of posterior (or prior) predictive sampling.

- `draws`: array of shape `(n_vars, n_times, n_draws)` containing simulated trajectories
- `summary`: per-variable, per-time summaries (mean, median, quantiles)
"""
struct PosteriorPredictive
    times::Vector{Float64}
    draws::Array{Float64, 3}       # n_vars × n_times × n_draws
    variable_names::Vector{Symbol}
    summary::NamedTuple            # (mean, median, q025, q975) each n_vars × n_times
end

"""
    PPCResult

Result of a posterior predictive check comparing observations to predictions.
"""
struct PPCResult
    times::Vector{Float64}
    observed::Vector{Float64}
    predicted_mean::Vector{Float64}
    predicted_median::Vector{Float64}
    predicted_q025::Vector{Float64}
    predicted_q975::Vector{Float64}
    predicted_q10::Vector{Float64}
    predicted_q90::Vector{Float64}
    coverage_50::Float64
    coverage_90::Float64
    coverage_95::Float64
    p_values::Vector{Float64}
    chi_squared::Float64
end

"""
    ResidualDiagnostics

Residual diagnostics for model fit assessment.
"""
struct ResidualDiagnostics
    times::Vector{Float64}
    raw_residuals::Vector{Float64}
    standardized_residuals::Vector{Float64}
    pearson_residuals::Vector{Float64}
    autocorrelation::Vector{Float64}
    ljung_box_p::Float64
    rmse::Float64
    mae::Float64
    bias::Float64
end

"""
    CalibrationResult

Assessment of predictive calibration (nominal vs empirical coverage).
"""
struct CalibrationResult
    nominal_levels::Vector{Float64}
    empirical_levels::Vector{Float64}
    calibration_error::Float64
    is_well_calibrated::Bool
end

"""
    SBCResult

Result of simulation-based calibration.
"""
struct SBCResult
    rank_statistics::Matrix{Int}       # n_pars × n_reps
    uniformity_p_values::Vector{Float64}
    is_calibrated::Vector{Bool}
end

# ── Posterior predictive sampling ────────────────────────────

"""
    posterior_predictive(samples, generator; kwargs...) → PosteriorPredictive

Generate posterior predictive simulations by drawing parameter vectors from
the MCMC output and running the model forward.

## Arguments
- `samples::MontySamples`: MCMC output
- `generator::DustSystemGenerator`: compiled model generator

## Keyword arguments
- `times::Vector{Float64}`: observation times
- `n_draws::Int=200`: number of posterior draws to use
- `n_particles::Int=1`: particles per draw (1 for ODE, >1 for stochastic)
- `dt::Float64=1.0`: time step
- `output_vars::Vector{Symbol}=Symbol[]`: variables to track (empty = all)
- `seed::Union{Nothing,Int}=nothing`: random seed
- `packer::MontyPacker`: packer to convert flat vectors → NamedTuple
"""
function posterior_predictive(
    samples::MontySamples,
    generator::DustSystemGenerator;
    times::Vector{Float64},
    n_draws::Int=200,
    n_particles::Int=1,
    dt::Float64=1.0,
    output_vars::Vector{Symbol}=Symbol[],
    seed::Union{Nothing, Int}=nothing,
    packer::MontyPacker,
)
    n_pars, n_samples, n_chains = size(samples.pars)
    total_samples = n_samples * n_chains
    n_draws = min(n_draws, total_samples)

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    # Flatten chains and sample draw indices
    flat_pars = reshape(samples.pars, n_pars, total_samples)
    draw_indices = sort(Random.randperm(rng, total_samples)[1:n_draws])

    # First draw to determine variable names and dimensions
    first_par_vec = flat_pars[:, draw_indices[1]]
    first_pars_nt = unpack(packer, first_par_vec)
    first_result = dust_system_simulate(generator, first_pars_nt;
                                        times=times, dt=dt, n_particles=n_particles,
                                        seed=seed === nothing ? nothing : seed + 1)

    # Get variable names from a temporary system
    all_names = _get_all_variable_names(generator, first_pars_nt; dt=dt)
    if isempty(output_vars)
        var_indices = 1:length(all_names)
        var_names = all_names
    else
        var_indices = [findfirst(==(v), all_names) for v in output_vars]
        any(isnothing, var_indices) && error("Unknown variable in output_vars")
        var_names = output_vars
    end

    n_vars = length(var_indices)
    n_times = length(times)
    draws = zeros(Float64, n_vars, n_times, n_draws)

    # Store first draw (average over particles)
    for (vi, idx) in enumerate(var_indices)
        for ti in 1:n_times
            draws[vi, ti, 1] = mean(first_result[idx, :, ti])
        end
    end

    # Remaining draws
    cached_sys = nothing
    for d in 2:n_draws
        par_vec = flat_pars[:, draw_indices[d]]
        pars_nt = unpack(packer, par_vec)
        draw_seed = seed === nothing ? nothing : seed + d
        result = dust_system_simulate(generator, pars_nt;
                                      times=times, dt=dt, n_particles=n_particles,
                                      seed=draw_seed)
        for (vi, idx) in enumerate(var_indices)
            for ti in 1:n_times
                draws[vi, ti, d] = mean(result[idx, :, ti])
            end
        end
    end

    summary = _compute_pp_summary(draws)
    return PosteriorPredictive(times, draws, var_names, summary)
end

"""Compute per-variable, per-time summary statistics from draws."""
function _compute_pp_summary(draws::Array{Float64, 3})
    n_vars, n_times, n_draws = size(draws)
    mn = zeros(n_vars, n_times)
    md = zeros(n_vars, n_times)
    q025 = zeros(n_vars, n_times)
    q975 = zeros(n_vars, n_times)

    for vi in 1:n_vars
        for ti in 1:n_times
            vals = @view draws[vi, ti, :]
            mn[vi, ti] = mean(vals)
            md[vi, ti] = median(vals)
            q025[vi, ti] = quantile(vals, 0.025)
            q975[vi, ti] = quantile(vals, 0.975)
        end
    end

    return (mean=mn, median=md, q025=q025, q975=q975)
end

# ── Posterior predictive checks ──────────────────────────────

"""
    ppc_check(pp, data; time_field=:time, data_var=:cases, pred_var=:cases) → PPCResult

Compare observed data with posterior predictive simulations.

## Arguments
- `pp::PosteriorPredictive`: posterior predictive output
- `data::AbstractVector{<:NamedTuple}`: observed data with time and variable fields

## Keyword arguments
- `time_field::Symbol=:time`: name of the time field in data
- `data_var::Symbol=:cases`: name of the observed variable in data
- `pred_var::Symbol=:cases`: name of the predicted variable in `pp`
"""
function ppc_check(
    pp::PosteriorPredictive,
    data::AbstractVector{<:NamedTuple};
    time_field::Symbol=:time,
    data_var::Symbol=:cases,
    pred_var::Symbol=:cases,
)
    var_idx = findfirst(==(pred_var), pp.variable_names)
    var_idx === nothing && error("Variable $pred_var not found in posterior predictive")

    obs_times = Float64[getfield(d, time_field) for d in data]
    obs_vals = Float64[Float64(getfield(d, data_var)) for d in data]
    n_obs = length(obs_times)
    n_draws = size(pp.draws, 3)

    # Match observation times to prediction times
    time_indices = _match_times(obs_times, pp.times)

    pred_mean = zeros(n_obs)
    pred_median = zeros(n_obs)
    pred_q025 = zeros(n_obs)
    pred_q975 = zeros(n_obs)
    pred_q10 = zeros(n_obs)
    pred_q90 = zeros(n_obs)
    pred_q25 = zeros(n_obs)
    pred_q75 = zeros(n_obs)
    p_values = zeros(n_obs)

    for i in 1:n_obs
        ti = time_indices[i]
        vals = @view pp.draws[var_idx, ti, :]
        pred_mean[i] = mean(vals)
        pred_median[i] = median(vals)
        pred_q025[i] = quantile(vals, 0.025)
        pred_q975[i] = quantile(vals, 0.975)
        pred_q10[i] = quantile(vals, 0.10)
        pred_q90[i] = quantile(vals, 0.90)
        pred_q25[i] = quantile(vals, 0.25)
        pred_q75[i] = quantile(vals, 0.75)
        # Bayesian p-value: fraction of draws >= observed
        p_values[i] = count(v -> v >= obs_vals[i], vals) / n_draws
    end

    # Coverage
    coverage_50 = count(i -> pred_q25[i] <= obs_vals[i] <= pred_q75[i], 1:n_obs) / n_obs
    coverage_90 = count(i -> pred_q10[i] <= obs_vals[i] <= pred_q90[i], 1:n_obs) / n_obs
    coverage_95 = count(i -> pred_q025[i] <= obs_vals[i] <= pred_q975[i], 1:n_obs) / n_obs

    # Posterior predictive chi-squared
    chi_sq = 0.0
    for i in 1:n_obs
        ti = time_indices[i]
        vals = @view pp.draws[var_idx, ti, :]
        sd = std(vals)
        if sd > 0
            chi_sq += ((obs_vals[i] - pred_mean[i]) / sd)^2
        end
    end

    return PPCResult(obs_times, obs_vals, pred_mean, pred_median,
                     pred_q025, pred_q975, pred_q10, pred_q90,
                     coverage_50, coverage_90, coverage_95, p_values, chi_sq)
end

"""Find closest time index in `ref_times` for each `query_time`."""
function _match_times(query_times::Vector{Float64}, ref_times::Vector{Float64})
    indices = Int[]
    for qt in query_times
        _, idx = findmin(abs.(ref_times .- qt))
        push!(indices, idx)
    end
    return indices
end

# ── Residual diagnostics ─────────────────────────────────────

"""
    residual_diagnostics(pp, data; data_var=:cases, pred_var=:cases, time_field=:time) → ResidualDiagnostics

Compute residual diagnostics comparing posterior predictive mean to observations.

Returns raw, standardized, and Pearson residuals along with autocorrelation
and a Ljung-Box test for residual independence.
"""
function residual_diagnostics(
    pp::PosteriorPredictive,
    data::AbstractVector{<:NamedTuple};
    data_var::Symbol=:cases,
    pred_var::Symbol=:cases,
    time_field::Symbol=:time,
)
    var_idx = findfirst(==(pred_var), pp.variable_names)
    var_idx === nothing && error("Variable $pred_var not found in posterior predictive")

    obs_times = Float64[getfield(d, time_field) for d in data]
    obs_vals = Float64[Float64(getfield(d, data_var)) for d in data]
    n_obs = length(obs_times)
    time_indices = _match_times(obs_times, pp.times)

    raw = zeros(n_obs)
    standardized = zeros(n_obs)
    pearson = zeros(n_obs)

    for i in 1:n_obs
        ti = time_indices[i]
        vals = @view pp.draws[var_idx, ti, :]
        m = mean(vals)
        s = std(vals)
        raw[i] = obs_vals[i] - m
        standardized[i] = s > 0 ? raw[i] / s : 0.0
        # Pearson residuals: use sqrt(|mean|) for count data
        pearson[i] = abs(m) > 0 ? raw[i] / sqrt(abs(m)) : 0.0
    end

    # Autocorrelation of standardized residuals (lags 1–10)
    max_lag = min(10, n_obs - 1)
    acf = _autocorrelation(standardized, max_lag)

    # Ljung-Box test
    lb_p = _ljung_box_p(acf, n_obs)

    rmse_val = sqrt(mean(raw .^ 2))
    mae_val = mean(abs.(raw))
    bias_val = mean(raw)

    return ResidualDiagnostics(obs_times, raw, standardized, pearson, acf,
                               lb_p, rmse_val, mae_val, bias_val)
end

"""Compute autocorrelation at lags 1 to `max_lag`."""
function _autocorrelation(x::Vector{Float64}, max_lag::Int)
    n = length(x)
    n <= 1 && return Float64[]
    xm = x .- mean(x)
    var_x = sum(xm .^ 2) / n
    var_x ≈ 0 && return zeros(max_lag)

    acf = zeros(max_lag)
    for lag in 1:max_lag
        s = 0.0
        for i in 1:(n - lag)
            s += xm[i] * xm[i + lag]
        end
        acf[lag] = s / (n * var_x)
    end
    return acf
end

"""Ljung-Box test p-value (chi-squared approximation)."""
function _ljung_box_p(acf::Vector{Float64}, n::Int)
    isempty(acf) && return 1.0
    m = length(acf)
    Q = 0.0
    for k in 1:m
        Q += acf[k]^2 / (n - k)
    end
    Q *= n * (n + 2)
    # Chi-squared CDF approximation using regularized incomplete gamma
    return _chi2_survival(Q, m)
end

"""Survival function (1 - CDF) of chi-squared distribution via normal approximation."""
function _chi2_survival(x::Float64, df::Int)
    df <= 0 && return 1.0
    x <= 0 && return 1.0
    # Wilson-Hilferty approximation: transform chi-squared to normal
    z = ((x / df)^(1/3) - (1 - 2 / (9 * df))) / sqrt(2 / (9 * df))
    # Standard normal survival function (Abramowitz & Stegun approximation)
    return _standard_normal_survival(z)
end

"""Approximate standard normal survival P(Z > z) using rational approximation."""
function _standard_normal_survival(z::Float64)
    if z < -8.0
        return 1.0
    elseif z > 8.0
        return 0.0
    end
    # Use the Horner form of the Abramowitz & Stegun approximation (7.1.26)
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
           t * (-1.821255978 + t * 1.330274429))))
    pdf = exp(-0.5 * z * z) / sqrt(2.0 * π)
    cdf_pos = 1.0 - pdf * poly
    if z >= 0
        return 1.0 - cdf_pos
    else
        return cdf_pos
    end
end

# ── Calibration assessment ───────────────────────────────────

"""
    calibration_check(pp, data; kwargs...) → CalibrationResult

Assess predictive calibration by comparing nominal prediction interval
coverage to empirical coverage at a range of quantile levels.

A well-calibrated model has `empirical_levels ≈ nominal_levels`.
"""
function calibration_check(
    pp::PosteriorPredictive,
    data::AbstractVector{<:NamedTuple};
    data_var::Symbol=:cases,
    pred_var::Symbol=:cases,
    time_field::Symbol=:time,
    quantile_levels::Vector{Float64}=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
)
    var_idx = findfirst(==(pred_var), pp.variable_names)
    var_idx === nothing && error("Variable $pred_var not found in posterior predictive")

    obs_times = Float64[getfield(d, time_field) for d in data]
    obs_vals = Float64[Float64(getfield(d, data_var)) for d in data]
    n_obs = length(obs_times)
    time_indices = _match_times(obs_times, pp.times)

    empirical = zeros(length(quantile_levels))

    for (qi, level) in enumerate(quantile_levels)
        # For nominal level p, count fraction of observations below the p-th quantile
        n_below = 0
        for i in 1:n_obs
            ti = time_indices[i]
            vals = @view pp.draws[var_idx, ti, :]
            threshold = quantile(vals, level)
            if obs_vals[i] <= threshold
                n_below += 1
            end
        end
        empirical[qi] = n_below / n_obs
    end

    cal_error = mean(abs.(quantile_levels .- empirical))
    is_good = cal_error < 0.1

    return CalibrationResult(quantile_levels, empirical, cal_error, is_good)
end

# ── Prior predictive sampling ────────────────────────────────

"""
    prior_predictive(prior_model, generator; kwargs...) → PosteriorPredictive

Generate prior predictive simulations by sampling from the prior and running
the model forward. Returns the same `PosteriorPredictive` struct.

## Arguments
- `prior_model::MontyModel`: prior model (must have `direct_sample`)
- `generator::DustSystemGenerator`: compiled model generator

## Keyword arguments
- `times::Vector{Float64}`: observation times
- `packer::MontyPacker`: packer to convert flat vectors → NamedTuple
- `n_draws::Int=200`: number of prior draws
- `n_particles::Int=1`: particles per draw
- `dt::Float64=1.0`: time step
- `output_vars::Vector{Symbol}=Symbol[]`: variables to track
- `seed::Union{Nothing,Int}=nothing`: random seed
"""
function prior_predictive(
    prior_model::MontyModel,
    generator::DustSystemGenerator;
    times::Vector{Float64},
    packer::MontyPacker,
    n_draws::Int=200,
    n_particles::Int=1,
    dt::Float64=1.0,
    output_vars::Vector{Symbol}=Symbol[],
    seed::Union{Nothing, Int}=nothing,
)
    prior_model.direct_sample === nothing &&
        error("prior_model must have a direct_sample function")

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    # First draw to determine dimensions
    first_par_vec = prior_model.direct_sample(rng)
    first_pars_nt = unpack(packer, first_par_vec)

    first_result = dust_system_simulate(generator, first_pars_nt;
                                        times=times, dt=dt, n_particles=n_particles,
                                        seed=seed === nothing ? nothing : seed + 1)

    all_names = _get_all_variable_names(generator, first_pars_nt; dt=dt)
    if isempty(output_vars)
        var_names = all_names
    else
        var_names = output_vars
    end

    if isempty(output_vars)
        var_indices = 1:length(all_names)
    else
        var_indices = [findfirst(==(v), all_names) for v in output_vars]
        any(isnothing, var_indices) && error("Unknown variable in output_vars")
    end

    n_vars = length(var_indices)
    n_times = length(times)
    draws = zeros(Float64, n_vars, n_times, n_draws)

    for (vi, idx) in enumerate(var_indices)
        for ti in 1:n_times
            draws[vi, ti, 1] = mean(first_result[idx, :, ti])
        end
    end

    for d in 2:n_draws
        par_vec = prior_model.direct_sample(rng)
        pars_nt = unpack(packer, par_vec)
        draw_seed = seed === nothing ? nothing : seed + d
        result = dust_system_simulate(generator, pars_nt;
                                      times=times, dt=dt, n_particles=n_particles,
                                      seed=draw_seed)
        for (vi, idx) in enumerate(var_indices)
            for ti in 1:n_times
                draws[vi, ti, d] = mean(result[idx, :, ti])
            end
        end
    end

    summary = _compute_pp_summary(draws)
    return PosteriorPredictive(times, draws, var_names, summary)
end

# ── Simulation-based calibration ─────────────────────────────

"""
    sbc_check(prior_model, likelihood_fn, sampler, packer; kwargs...) → SBCResult

Perform simulation-based calibration (SBC) to verify that the inference
pipeline is self-consistent.

For each replicate:
1. Draw true parameters from the prior
2. Compute the posterior (prior + likelihood) using MCMC
3. Record the rank of the true parameter value within the posterior samples

If inference is correct, the rank statistics should be uniformly distributed.

## Arguments
- `prior_model::MontyModel`: prior (must have `direct_sample`)
- `likelihood_fn::Function`: `pars_vec → log-likelihood`
- `sampler::AbstractMontySampler`: MCMC sampler
- `packer::MontyPacker`: parameter packer

## Keyword arguments
- `n_sbc_reps::Int=100`: number of SBC replicates
- `n_mcmc_steps::Int=500`: MCMC steps per replicate
- `n_chains::Int=1`: chains per replicate
- `seed::Union{Nothing,Int}=nothing`: random seed
"""
function sbc_check(
    prior_model::MontyModel,
    likelihood_fn::Function,
    sampler::AbstractMontySampler,
    packer::MontyPacker;
    n_sbc_reps::Int=100,
    n_mcmc_steps::Int=500,
    n_chains::Int=1,
    seed::Union{Nothing, Int}=nothing,
)
    prior_model.direct_sample === nothing &&
        error("prior_model must have a direct_sample function")

    n_pars = packer.len
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    rank_stats = zeros(Int, n_pars, n_sbc_reps)

    for rep in 1:n_sbc_reps
        # Draw true parameters from the prior
        true_pars = prior_model.direct_sample(rng)

        # Build posterior = prior + likelihood at these true pars
        ll_model = monty_model(
            likelihood_fn;
            parameters=parameter_names(packer),
        )
        posterior = prior_model + ll_model

        # Run MCMC
        rep_seed = seed === nothing ? nothing : seed + rep * 1000
        initial = reshape(true_pars .+ randn(rng, n_pars) .* 0.01, n_pars, n_chains)
        samples = monty_sample(
            posterior, sampler, n_mcmc_steps;
            n_chains=n_chains, initial=initial, seed=rep_seed,
        )

        # Compute ranks: fraction of posterior samples < true value
        flat_pars = reshape(samples.pars, n_pars, :)
        n_post = size(flat_pars, 2)
        for p in 1:n_pars
            rank_stats[p, rep] = count(flat_pars[p, :] .< true_pars[p])
        end
    end

    # Kolmogorov-Smirnov test for uniformity
    n_post_total = n_mcmc_steps * n_chains
    p_values = zeros(n_pars)
    is_cal = fill(false, n_pars)
    for p in 1:n_pars
        p_values[p] = _ks_uniform_p(rank_stats[p, :], n_post_total)
        is_cal[p] = p_values[p] > 0.05
    end

    return SBCResult(rank_stats, p_values, is_cal)
end

"""Kolmogorov-Smirnov test p-value for uniformity on [0, n]."""
function _ks_uniform_p(ranks::Vector{Int}, n_max::Int)
    n = length(ranks)
    n == 0 && return 1.0
    sorted = sort(ranks) ./ n_max
    d_max = 0.0
    for i in 1:n
        ecdf_val = i / n
        d_plus = abs(ecdf_val - sorted[i])
        d_minus = abs(sorted[i] - (i - 1) / n)
        d_max = max(d_max, d_plus, d_minus)
    end
    # Approximate KS p-value
    lambda = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * d_max
    if lambda < 1e-10
        return 1.0
    end
    # Kolmogorov distribution approximation
    p = 2.0 * sum((-1)^(k-1) * exp(-2 * k^2 * lambda^2) for k in 1:100)
    return clamp(p, 0.0, 1.0)
end
