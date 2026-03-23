# Model selection and comparison tools: AIC, BIC, DIC, WAIC, LOO-CV.

# ── Numerically stable log-mean-exp ─────────────────────────

"""
    _log_mean_exp(x) -> Float64

Compute log(mean(exp.(x))) in a numerically stable way.
"""
function _log_mean_exp(x::AbstractVector{<:Real})
    mx = maximum(x)
    isinf(mx) && return -Inf
    s = 0.0
    @inbounds for i in eachindex(x)
        s += exp(x[i] - mx)
    end
    return mx + log(s / length(x))
end

"""Sample mean (avoids importing Statistics)."""
function _ms_mean(x::AbstractVector{<:Real})
    s = 0.0
    @inbounds for v in x
        s += v
    end
    return s / length(x)
end

"""Sample variance (Bessel-corrected, avoids importing Statistics)."""
function _ms_var(x::AbstractVector{<:Real})
    n = length(x)
    n < 2 && return 0.0
    m = _ms_mean(x)
    s = 0.0
    @inbounds for v in x
        d = v - m
        s += d * d
    end
    return s / (n - 1)
end

# ── Information Criteria ─────────────────────────────────────

"""
    compute_aic(log_likelihood, n_parameters) -> Float64

Akaike Information Criterion: AIC = -2·LL + 2·k.
"""
function compute_aic(log_likelihood::Float64, n_parameters::Int)
    return -2.0 * log_likelihood + 2.0 * n_parameters
end

"""
    compute_aicc(log_likelihood, n_parameters, n_observations) -> Float64

Corrected AIC for small samples:
AICc = AIC + 2k(k+1)/(n-k-1).
"""
function compute_aicc(log_likelihood::Float64, n_parameters::Int, n_observations::Int)
    k = n_parameters
    n = n_observations
    if n <= k + 1
        error("n_observations must be > n_parameters + 1 for AICc")
    end
    aic = compute_aic(log_likelihood, k)
    correction = 2.0 * k * (k + 1) / (n - k - 1)
    return aic + correction
end

"""
    compute_bic(log_likelihood, n_parameters, n_observations) -> Float64

Bayesian Information Criterion: BIC = -2·LL + k·log(n).
"""
function compute_bic(log_likelihood::Float64, n_parameters::Int, n_observations::Int)
    return -2.0 * log_likelihood + n_parameters * log(n_observations)
end

"""
    compute_dic(samples, likelihood_fn) -> NamedTuple

Deviance Information Criterion from MCMC posterior samples.

Returns `(dic=..., p_d=..., d_bar=..., d_theta_bar=...)` where:
- `d_bar`: mean deviance over posterior samples
- `d_theta_bar`: deviance at posterior mean
- `p_d`: effective number of parameters = d_bar - d_theta_bar
- `dic`: DIC = d_bar + p_d = 2·d_bar - d_theta_bar

## Arguments
- `samples::MontySamples`: posterior samples
- `likelihood_fn::Function`: `θ::Vector{Float64} -> Float64` returning log-likelihood
"""
function compute_dic(samples::MontySamples, likelihood_fn::Function)
    n_pars, n_samples, n_chains = size(samples.pars)
    total = n_samples * n_chains

    # Compute mean deviance D̄ and posterior mean θ̄
    deviance_sum = 0.0
    theta_bar = zeros(Float64, n_pars)

    for c in 1:n_chains
        for s in 1:n_samples
            theta = @view samples.pars[:, s, c]
            ll = likelihood_fn(collect(theta))
            deviance_sum += -2.0 * ll
            theta_bar .+= theta
        end
    end

    d_bar = deviance_sum / total
    theta_bar ./= total

    # Deviance at posterior mean
    d_theta_bar = -2.0 * likelihood_fn(theta_bar)

    p_d = d_bar - d_theta_bar
    dic = d_bar + p_d  # = 2 * d_bar - d_theta_bar

    return (dic=dic, p_d=p_d, d_bar=d_bar, d_theta_bar=d_theta_bar)
end

"""
    compute_waic(pointwise_log_liks) -> NamedTuple

Watanabe-Akaike Information Criterion (widely applicable IC).

Returns `(waic=..., p_waic=..., lppd=..., pointwise=...)`.

## Arguments
- `pointwise_log_liks::Matrix{Float64}`: n_observations × n_posterior_samples matrix
  of pointwise log-likelihoods.
"""
function compute_waic(pointwise_log_liks::Matrix{Float64})
    n_obs, n_samples = size(pointwise_log_liks)

    lppd = 0.0
    p_waic = 0.0
    pointwise_waic = zeros(Float64, n_obs)

    for i in 1:n_obs
        row = @view pointwise_log_liks[i, :]

        # lppd_i = log(mean(exp(ll_{i,s})))
        lppd_i = _log_mean_exp(row)
        lppd += lppd_i

        # p_waic2_i = var_s(ll_{i,s})
        p_waic_i = _ms_var(row)
        p_waic += p_waic_i

        pointwise_waic[i] = -2.0 * (lppd_i - p_waic_i)
    end

    waic = -2.0 * (lppd - p_waic)
    return (waic=waic, p_waic=p_waic, lppd=lppd, pointwise=pointwise_waic)
end

# ── Pointwise Log-Likelihood Extraction ──────────────────────

"""
    dust_unfilter_run_pointwise!(unfilter, pars) -> Vector{Float64}

Run the deterministic unfilter and return per-observation log-likelihoods.
Returns `[ll_1, ll_2, ..., ll_T]` for each data point.
"""
function dust_unfilter_run_pointwise!(unfilter::DustUnfilter, pars::NamedTuple)
    model = unfilter.generator.model
    n_state = model.n_state
    n_data = length(unfilter.data.times)

    full_pars = _merge_pars(model, pars, 1.0)
    if model.has_interpolation
        full_pars = _odin_setup_pars(model, full_pars)
    end

    state = unfilter._state_cache
    fill!(state, zero(Float64))
    rng = unfilter._rng_cache
    _odin_initial!(model, state, full_pars, rng)

    ctrl = unfilter.ode_control
    data_times = unfilter.data.times
    t_start = unfilter.time_start
    t_end = data_times[end]

    rhs_fn! = (du, u, p, t) -> _odin_rhs!(model, du, u, p, t)
    saveat = unfilter._saveat_cache
    pointwise_ll = zeros(Float64, n_data)

    if ctrl.solver === :sdirk
        if unfilter._sdirk_workspace === nothing
            unfilter._sdirk_workspace = SDIRKWorkspace(n_state, Float64)
        end
        ws = unfilter._sdirk_workspace::SDIRKWorkspace{Float64}
        _sdirk_solve_core!(rhs_fn!, state, (t_start, t_end), full_pars, saveat,
                           ws, nothing, ctrl.atol, ctrl.rtol, ctrl.max_steps)
        for t_idx in 1:n_data
            state_t = @view ws.result_matrix[:, t_idx]
            data_t = unfilter.data.data[t_idx]
            pointwise_ll[t_idx] = _odin_compare_data(model, state_t, full_pars, data_t, data_times[t_idx])
        end
    else
        if unfilter._dp5_workspace === nothing
            unfilter._dp5_workspace = DP5Workspace(n_state, Float64)
        end
        ws = unfilter._dp5_workspace::DP5Workspace{Float64}
        _dp5_solve_core!(rhs_fn!, state, (t_start, t_end), full_pars, saveat,
                         ws, nothing, ctrl.atol, ctrl.rtol, ctrl.max_steps)
        for t_idx in 1:n_data
            state_t = @view ws.result_matrix[:, t_idx]
            data_t = unfilter.data.data[t_idx]
            pointwise_ll[t_idx] = _odin_compare_data(model, state_t, full_pars, data_t, data_times[t_idx])
        end
    end

    return pointwise_ll
end

"""
    dust_filter_run_pointwise!(filter, pars) -> Vector{Float64}

Run the particle filter and return per-observation log-likelihoods.
Returns `[ll_1, ll_2, ..., ll_T]` where each entry is
`log_sum_exp(weights_t) - log(n_particles)`.
"""
function dust_filter_run_pointwise!(filter::DustFilter, pars::NamedTuple)
    sys = _get_or_create_sys!(filter, pars)
    dust_system_set_state_initial!(sys)
    np = filter.n_particles
    return _filter_inner_pointwise!(sys, sys.pars, filter.data, np)
end

"""Type-stable inner loop returning per-observation log-likelihoods."""
function _filter_inner_pointwise!(sys::DustSystem{M,T}, pars::P, data::FilterData{D},
                                  n_particles::Int) where {M,T,P,D}
    n_data = length(data.times)
    n_state = sys.n_state
    model = sys.generator.model
    dt = sys.dt

    state = sys.state
    rngs = sys.rng
    log_weights = sys._work_weights
    indices = sys._work_indices
    state_tmp = sys._work_state_tmp
    state_next = sys._work_state_next
    zero_every = sys.zero_every

    pointwise_ll = zeros(T, n_data)

    for t_idx in 1:n_data
        target_time = data.times[t_idx]
        data_t = data.data[t_idx]

        while sys.time < target_time - dt / 2
            if !isempty(zero_every)
                @inbounds for entry in zero_every
                    period = entry.period
                    if period > 0 && abs(sys.time - round(sys.time / period) * period) < dt / 4
                        for p in 1:n_particles
                            for idx in entry.range
                                state[idx, p] = 0.0
                            end
                        end
                    end
                end
            end

            for p in 1:n_particles
                state_view = @view state[:, p]
                @inbounds for j in 1:n_state
                    state_next[j] = state_view[j]
                end
                _odin_update!(model, state_next, state_view, pars, sys.time, dt, rngs[p])
                @inbounds for j in 1:n_state
                    state_view[j] = state_next[j]
                end
            end
            sys.time += dt
        end

        @inbounds for p in 1:n_particles
            state_col_view = @view state[:, p]
            log_weights[p] = _odin_compare_data(model, state_col_view, pars, data_t, target_time)
        end

        ll_t = log_sum_exp(log_weights) - log(n_particles)
        pointwise_ll[t_idx] = ll_t

        if t_idx < n_data
            _systematic_resample_inplace!(indices, log_weights, rngs[1])
            state_tmp .= state
            @inbounds for p in 1:n_particles
                src = indices[p]
                for j in 1:n_state
                    state[j, p] = state_tmp[j, src]
                end
            end
        end
    end

    return pointwise_ll
end

# ── Akaike / BIC Weights ────────────────────────────────────

"""
    akaike_weights(ic_values) -> Vector{Float64}

Compute Akaike weights from a vector of IC values (AIC or BIC).
Δ_i = IC_i - min(IC), w_i = exp(-0.5·Δ_i) / Σ exp(-0.5·Δ_j).
"""
function akaike_weights(ic_values::Vector{Float64})
    min_ic = minimum(ic_values)
    deltas = ic_values .- min_ic
    raw = exp.(-0.5 .* deltas)
    return raw ./ sum(raw)
end

# ── LOO-CV (PSIS-LOO) ───────────────────────────────────────

"""
    compute_loo(pointwise_log_liks) -> NamedTuple

Pareto-smoothed importance sampling LOO-CV (PSIS-LOO) based on
Vehtari, Gelman, Gabry (2017).

Returns `(loo=..., p_loo=..., pointwise=..., k_diagnostics=...)`.

## Arguments
- `pointwise_log_liks::Matrix{Float64}`: n_observations × n_posterior_samples
"""
function compute_loo(pointwise_log_liks::Matrix{Float64})
    n_obs, n_samples = size(pointwise_log_liks)

    pointwise_loo = zeros(Float64, n_obs)
    k_diagnostics = zeros(Float64, n_obs)
    lppd_loo = 0.0
    p_loo = 0.0

    for i in 1:n_obs
        lls = pointwise_log_liks[i, :]

        # Raw importance ratios: log(1/p(y_i|θ_s)) = -ll_{i,s}
        log_ratios = -lls
        log_ratios .-= maximum(log_ratios)  # stabilise

        # Fit generalised Pareto to the tail and get the shape parameter k
        ratios = exp.(log_ratios)
        k_hat, smoothed_ratios = _psis_smooth(ratios)
        k_diagnostics[i] = k_hat

        # Normalised weights
        log_w = log.(smoothed_ratios)
        log_w .-= _log_sum_exp_vec(log_w)

        # LOO log predictive density for observation i
        loo_lpd_i = _log_sum_exp_vec(log_w .+ lls)
        pointwise_loo[i] = loo_lpd_i
        lppd_loo += loo_lpd_i

        # Full posterior lppd for p_loo
        lppd_full_i = _log_mean_exp(lls)
        p_loo += lppd_full_i - loo_lpd_i
    end

    loo = -2.0 * lppd_loo
    return (loo=loo, p_loo=p_loo, pointwise=pointwise_loo, k_diagnostics=k_diagnostics)
end

"""Vectorised log-sum-exp for LOO computation."""
function _log_sum_exp_vec(x::AbstractVector{<:Real})
    mx = maximum(x)
    isinf(mx) && return -Inf
    return mx + log(sum(exp.(x .- mx)))
end

"""
Pareto-smoothed importance sampling: fit a generalised Pareto distribution
to the largest importance ratios and replace their values.
Returns `(k_hat, smoothed_ratios)`.
"""
function _psis_smooth(ratios::Vector{Float64})
    n = length(ratios)
    sorted_idx = sortperm(ratios)
    sorted = ratios[sorted_idx]

    # Number of tail samples: min(n/5, 3*sqrt(n))
    m = min(floor(Int, n / 5), ceil(Int, 3 * sqrt(n)))
    m = max(m, 5)  # need at least a few samples
    m = min(m, n - 1)

    # Tail: the largest m values
    tail = sorted[(n - m + 1):n]
    threshold = sorted[n - m]

    # Fit generalised Pareto to (tail - threshold) via probability-weighted moments
    k_hat = _fit_gpd_k(tail .- threshold)

    # Smooth the tail: replace with expected order statistics from fitted GPD
    if k_hat < 0.7
        smoothed = copy(ratios)
        for j in 1:m
            # Use quantiles from the fitted GPD
            p_j = (j - 0.5) / m
            if abs(k_hat) < 1e-8
                q = -log(1.0 - p_j) * _ms_mean(tail .- threshold)
            else
                sigma = _ms_mean(tail .- threshold) * (1.0 - k_hat)
                sigma = max(sigma, 1e-10)
                q = sigma / k_hat * ((1.0 - p_j)^(-k_hat) - 1.0)
            end
            idx = sorted_idx[n - m + j]
            smoothed[idx] = threshold + q
        end
        # Truncate at max raw ratio to avoid extreme values
        max_raw = maximum(ratios)
        smoothed .= min.(smoothed, max_raw)
        return k_hat, smoothed
    else
        # k >= 0.7: tail is too heavy, return raw ratios with warning shape
        return k_hat, copy(ratios)
    end
end

"""Fit the shape parameter k of a generalised Pareto distribution via method of moments."""
function _fit_gpd_k(exceedances::Vector{Float64})
    n = length(exceedances)
    n < 2 && return 0.0
    m1 = _ms_mean(exceedances)
    m1 < 1e-20 && return 0.0
    m2 = _ms_mean(exceedances .^ 2)
    # Method of moments: k = 0.5 * (1 - m1^2 / m2)
    # but use the more stable Zhang & Stephens (2009) estimator
    k = 0.5 * (1.0 - m1^2 / (m2 - m1^2 + 1e-20))
    # Clamp to reasonable range
    k = clamp(k, -0.5, 1.5)
    return k
end

# ── Model Comparison Table ───────────────────────────────────

"""
    ModelComparison

A structured comparison of multiple models by information criteria.
"""
struct ModelComparison
    models::Vector{String}
    log_likelihoods::Vector{Float64}
    n_parameters::Vector{Int}
    n_observations::Int
    aic::Vector{Float64}
    aicc::Vector{Float64}
    bic::Vector{Float64}
    dic::Vector{Union{Float64, Nothing}}
    waic::Vector{Union{Float64, Nothing}}
    weights_aic::Vector{Float64}
    weights_bic::Vector{Float64}
end

"""
    compare_models(; n_observations, models...) -> ModelComparison

Compare models using information criteria.

Each model is specified as a named keyword argument with a NamedTuple containing:
- `ll::Float64`: log-likelihood
- `k::Int`: number of parameters
- `dic::Union{Float64, Nothing}` (optional)
- `waic::Union{Float64, Nothing}` (optional)

## Example
```julia
mc = compare_models(;
    n_observations=50,
    sir=(ll=-120.0, k=3),
    seir=(ll=-115.0, k=4, dic=240.0),
)
```
"""
function compare_models(; n_observations::Int, models...)
    n_models = length(models)
    names = String[]
    lls = Float64[]
    ks = Int[]
    dics = Union{Float64, Nothing}[]
    waics = Union{Float64, Nothing}[]

    for (name, spec) in models
        push!(names, string(name))
        push!(lls, spec.ll)
        push!(ks, spec.k)
        push!(dics, haskey(spec, :dic) ? spec.dic : nothing)
        push!(waics, haskey(spec, :waic) ? spec.waic : nothing)
    end

    aic_vals = [compute_aic(ll, k) for (ll, k) in zip(lls, ks)]
    aicc_vals = [compute_aicc(ll, k, n_observations) for (ll, k) in zip(lls, ks)]
    bic_vals = [compute_bic(ll, k, n_observations) for (ll, k) in zip(lls, ks)]

    w_aic = akaike_weights(aic_vals)
    w_bic = akaike_weights(bic_vals)

    # Sort by AIC
    order = sortperm(aic_vals)
    return ModelComparison(
        names[order], lls[order], ks[order], n_observations,
        aic_vals[order], aicc_vals[order], bic_vals[order],
        dics[order], waics[order],
        w_aic[order], w_bic[order],
    )
end

function _fmt_float(x::Float64, decimals::Int=1)
    if decimals == 1
        s = string(round(x; digits=1))
        # Ensure at least one decimal place
        if !occursin('.', s)
            s *= ".0"
        end
        return s
    elseif decimals == 3
        s = string(round(x; digits=3))
        if !occursin('.', s)
            s *= ".000"
        end
        return s
    else
        return string(round(x; digits=decimals))
    end
end

function Base.show(io::IO, mc::ModelComparison)
    n = length(mc.models)
    println(io, "Model Comparison (n_obs = $(mc.n_observations))")
    println(io, "─" ^ 90)

    # Header
    println(io, rpad("Model", 12), " ",
            lpad("LL", 8), " ",
            lpad("k", 4), " ",
            lpad("AIC", 8), " ",
            lpad("AICc", 8), " ",
            lpad("BIC", 8), " ",
            lpad("DIC", 8), " ",
            lpad("WAIC", 8), " ",
            lpad("w_AIC", 7), " ",
            lpad("w_BIC", 7))
    println(io, "─" ^ 90)

    for i in 1:n
        dic_str = mc.dic[i] === nothing ? "—" : _fmt_float(mc.dic[i])
        waic_str = mc.waic[i] === nothing ? "—" : _fmt_float(mc.waic[i])
        println(io, rpad(mc.models[i], 12), " ",
                lpad(_fmt_float(mc.log_likelihoods[i]), 8), " ",
                lpad(string(mc.n_parameters[i]), 4), " ",
                lpad(_fmt_float(mc.aic[i]), 8), " ",
                lpad(_fmt_float(mc.aicc[i]), 8), " ",
                lpad(_fmt_float(mc.bic[i]), 8), " ",
                lpad(dic_str, 8), " ",
                lpad(waic_str, 8), " ",
                lpad(_fmt_float(mc.weights_aic[i], 3), 7), " ",
                lpad(_fmt_float(mc.weights_bic[i], 3), 7))
    end
    println(io, "─" ^ 90)
end
