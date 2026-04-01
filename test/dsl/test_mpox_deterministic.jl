using Test
using Odin
using DelimitedFiles
using Random
using Distributions
using LinearAlgebra

function _read_numeric_csv(path)
    data, _ = readdlm(path, ',', Float64, '\n', header=true)
    return Matrix{Float64}(data)
end

function _fit_logspace(lik, packer, fixed_pars, initial_free; step=0.2, max_iter=40, min_step=1e-6)
    theta = log.(Odin.pack(packer, initial_free))
    current_free = Odin.unpack(packer, exp.(theta))
    current_pars = merge(fixed_pars, current_free)
    current_ll = loglik(lik, current_pars)
    current_grad = loglik_gradient(lik, current_pars, packer; method=:adjoint).gradient
    history = Float64[current_ll]

    for _ in 1:max_iter
        grad_theta = current_grad .* exp.(theta)
        grad_norm = norm(grad_theta)
        grad_norm == 0 && break

        accepted = false
        trial_step = step
        while trial_step >= min_step
            theta_trial = theta .+ trial_step .* grad_theta ./ max(grad_norm, 1.0)
            free_trial = Odin.unpack(packer, exp.(theta_trial))
            pars_trial = merge(fixed_pars, free_trial)
            ll_trial = loglik(lik, pars_trial)
            if isfinite(ll_trial) && ll_trial > current_ll + 1e-6
                theta = theta_trial
                current_free = free_trial
                current_pars = pars_trial
                current_ll = ll_trial
                current_grad = loglik_gradient(lik, current_pars, packer; method=:adjoint).gradient
                push!(history, current_ll)
                accepted = true
                break
            end
            trial_step *= 0.5
        end

        accepted || break
    end

    return (free=current_free, pars=current_pars, log_likelihood=current_ll, history=history)
end

@testset "Deterministic mpox benchmark model" begin
    mpox_det = @odin begin
        total_I = I_mild + I_hosp + I_ICU
        beta = ifelse(time < intervention_day, beta_1, beta_2)
        contact = ifelse(time < intervention_day, contact_1, contact_2)

        deriv(S) = -beta * contact * S * total_I / N
        deriv(E) = beta * contact * S * total_I / N - gamma * E
        deriv(I_mild) = p_mild * gamma * E - sigma * I_mild - mu * I_mild
        deriv(I_hosp) = p_hosp * gamma * E - sigma * I_hosp - mu * I_hosp
        deriv(I_ICU) = p_ICU * gamma * E - sigma * I_ICU - mu * I_ICU
        deriv(R) = sigma * total_I
        deriv(D) = mu * total_I

        N = S + E + I_mild + I_hosp + I_ICU + R + D

        output(total_I_out) = total_I

        cases = data()
        cases ~ Poisson(max(total_I, 1e-6))

        initial(S) = S0
        initial(E) = E0
        initial(I_mild) = I_mild0
        initial(I_hosp) = I_hosp0
        initial(I_ICU) = I_ICU0
        initial(R) = R0
        initial(D) = D0

        S0 = parameter(9_969.0)
        E0 = parameter(20.0)
        I_mild0 = parameter(8.0)
        I_hosp0 = parameter(2.0)
        I_ICU0 = parameter(1.0)
        R0 = parameter(0.0)
        D0 = parameter(0.0)

        gamma = parameter(1 / 10)
        sigma = parameter(1 / 7)
        mu = parameter(0.01)
        p_mild = parameter(0.85)
        p_hosp = parameter(0.10)
        p_ICU = parameter(0.05)

        beta_1 = parameter(0.9, differentiate = true)
        beta_2 = parameter(0.45, differentiate = true)
        contact_1 = parameter(1.0, differentiate = true)
        contact_2 = parameter(0.65, differentiate = true)
        intervention_day = parameter(40.0)
    end

    pars = (
        S0=9_969.0, E0=20.0, I_mild0=8.0, I_hosp0=2.0, I_ICU0=1.0, R0=0.0, D0=0.0,
        gamma=1 / 10, sigma=1 / 7, mu=0.01, p_mild=0.85, p_hosp=0.10, p_ICU=0.05,
        beta_1=0.9, beta_2=0.45, contact_1=1.0, contact_2=0.65, intervention_day=40.0,
    )

    sim_times = collect(0.0:1.0:90.0)
    sim = simulate(mpox_det, pars; times=sim_times, seed=123)
    @test size(sim, 3) == length(sim_times)

    state_labels = string.(Odin._odin_state_names(mpox_det.model, pars))
    idx_mild = findfirst(==("I_mild"), state_labels)
    idx_hosp = findfirst(==("I_hosp"), state_labels)
    idx_icu = findfirst(==("I_ICU"), state_labels)
    @test idx_mild !== nothing
    @test idx_hosp !== nothing
    @test idx_icu !== nothing

    obs_times = collect(10.0:5.0:80.0)
    obs_idx = [findfirst(==(t), sim_times) for t in obs_times]
    rng = Xoshiro(99)
    data = [
        (
            time=t,
            cases=rand(rng, Poisson(max(
                sim[idx_mild, 1, i] + sim[idx_hosp, 1, i] + sim[idx_icu, 1, i],
                1e-6,
            ))),
        ) for (t, i) in zip(obs_times, obs_idx)
    ]

    lik = Likelihood(mpox_det, ObservedData(data); time_start=0.0)
    ll = loglik(lik, pars)
    @test isfinite(ll)

    packer = Packer([:beta_1, :beta_2, :contact_1, :contact_2])
    fwd = loglik_gradient(lik, pars, packer; method=:forward)
    adj = loglik_gradient(lik, pars, packer; method=:adjoint)

    @test isfinite(fwd.log_likelihood)
    @test isfinite(adj.log_likelihood)
    @test all(isfinite, fwd.gradient)
    @test all(isfinite, adj.gradient)
    @test fwd.log_likelihood ≈ adj.log_likelihood atol=1e-3
    @test fwd.gradient ≈ adj.gradient rtol=0.25 atol=1e-2
end

@testset "Deterministic mpox DZA slice" begin
    mpox_det_dza = @odin begin
        contact_multiplier = interpolate(contact_time, contact_value, :constant)
        total_I = I_mild + I_hosp + I_ICU
        force = beta * contact_multiplier * S * total_I / N
        cases_rate = case_scale * gamma * E
        deaths_rate = death_scale * mu * total_I

        deriv(S) = -force
        deriv(E) = force - gamma * E
        deriv(I_mild) = p_mild * gamma * E - sigma * I_mild - mu * I_mild
        deriv(I_hosp) = p_hosp * gamma * E - sigma * I_hosp - mu * I_hosp
        deriv(I_ICU) = p_ICU * gamma * E - sigma * I_ICU - mu * I_ICU
        deriv(R) = sigma * total_I
        deriv(D) = deaths_rate

        N = S + E + I_mild + I_hosp + I_ICU + R + D

        output(cases_rate_out) = cases_rate
        output(deaths_rate_out) = deaths_rate

        cases = data()
        deaths = data()
        cases ~ Poisson(max(cases_rate, 1e-6))
        deaths ~ Poisson(max(deaths_rate, 1e-6))

        initial(S) = S0
        initial(E) = E0
        initial(I_mild) = I_mild0
        initial(I_hosp) = I_hosp0
        initial(I_ICU) = I_ICU0
        initial(R) = R0
        initial(D) = D0

        contact_time = parameter(rank=1)
        contact_value = parameter(rank=1)
        S0 = parameter(500_000.0)
        E0 = parameter(40.0)
        I_mild0 = parameter(10.0)
        I_hosp0 = parameter(3.0)
        I_ICU0 = parameter(1.0)
        R0 = parameter(0.0)
        D0 = parameter(0.0)
        gamma = parameter(1 / 7)
        sigma = parameter(1 / 10)
        mu = parameter(0.004, differentiate = true)
        p_mild = parameter(0.85)
        p_hosp = parameter(0.10)
        p_ICU = parameter(0.05)
        beta = parameter(0.38, differentiate = true)
        case_scale = parameter(8.0, differentiate = true)
        death_scale = parameter(2.5, differentiate = true)
    end

    data_dir = joinpath(
        dirname(dirname(@__DIR__)),
        "vignettes", "38_mpox_deterministic_benchmark", "data",
    )
    dza_daily = _read_numeric_csv(joinpath(data_dir, "dza_daily.csv"))
    dza_intervention = _read_numeric_csv(joinpath(data_dir, "dza_intervention.csv"))

    obs = ObservedData([
        (
            time=row[1],
            cases=round(Int, row[2]),
            deaths=round(Int, row[3]),
        ) for row in eachrow(dza_daily)
    ])

    pars = (
        contact_time=dza_intervention[:, 1],
        contact_value=dza_intervention[:, 2],
        S0=500_000.0, E0=40.0, I_mild0=10.0, I_hosp0=3.0, I_ICU0=1.0, R0=0.0, D0=0.0,
        gamma=1 / 7, sigma=1 / 10, mu=0.004, p_mild=0.85, p_hosp=0.10, p_ICU=0.05,
        beta=0.38, case_scale=8.0, death_scale=2.5,
    )

    max_day = Int(maximum(dza_daily[:, 1]))
    sim = simulate(mpox_det_dza, pars; times=collect(0.0:1.0:max_day), seed=123)
    @test size(sim, 1) == 9
    @test size(sim, 3) == max_day + 1

    lik = Likelihood(mpox_det_dza, obs; time_start=-1.0)
    expected_lik = Likelihood(mpox_det_dza, obs; time_start=-1.0)
    expected_ll = loglik(expected_lik, pars)
    @test isfinite(expected_ll)

    packer = Packer([:beta, :mu, :case_scale, :death_scale])
    fwd = loglik_gradient(lik, pars, packer; method=:forward)
    adj = loglik_gradient(lik, pars, packer; method=:adjoint)

    @test isfinite(fwd.log_likelihood)
    @test isfinite(adj.log_likelihood)
    @test all(isfinite, fwd.gradient)
    @test all(isfinite, adj.gradient)
    @test fwd.log_likelihood ≈ expected_ll rtol=5e-4 atol=2.0
    @test adj.log_likelihood ≈ expected_ll rtol=1e-6 atol=1e-6
    @test fwd.log_likelihood ≈ adj.log_likelihood rtol=5e-4 atol=2.0
    @test fwd.gradient ≈ adj.gradient rtol=0.35 atol=1e-1

    fit_fixed = (
        contact_time=dza_intervention[:, 1],
        contact_value=dza_intervention[:, 2],
        S0=500_000.0, E0=40.0, I_mild0=10.0, I_hosp0=3.0, I_ICU0=1.0, R0=0.0, D0=0.0,
        gamma=1 / 7, sigma=1 / 10, mu=0.004, p_mild=0.85, p_hosp=0.10, p_ICU=0.05,
    )
    fit_initial = (beta=0.24, case_scale=5.5, death_scale=1.6)
    fit_packer = Packer([:beta, :case_scale, :death_scale])
    fit_start_ll = loglik(lik, merge(fit_fixed, fit_initial))
    fit = _fit_logspace(lik, fit_packer, fit_fixed, fit_initial; step=0.2, max_iter=40)

    @test fit.log_likelihood > fit_start_ll + 1000
    @test length(fit.history) > 10
    @test all(x -> x > 0, Tuple(fit.free))
end

@testset "Deterministic age-structured mpox slice" begin
    # Continuous-time 3-age-group SEIRD with age-specific susceptibility,
    # split latent stages, split infectious outcomes (recovery/death), and
    # per-age Poisson case/death observations.  All age-specific constants
    # are passed as parameter arrays to stay compatible with the symbolic
    # Jacobian builder.
    mpox_age = @odin begin
        n_age = parameter(3)

        # Age-specific susceptibility (parameter array)
        dim(susc) = n_age
        susc = parameter(rank = 1)

        # Age-specific case fatality ratio (parameter array)
        dim(cfr) = n_age
        cfr = parameter(rank = 1)

        dim(I_total) = n_age
        I_total[i] = Ir[i] + Id[i]
        total_I = sum(I_total)

        dim(lambda) = n_age
        lambda[i] = beta * susc[i] * total_I / N

        dim(S) = n_age; dim(Ea) = n_age; dim(Eb) = n_age
        dim(Ir) = n_age; dim(Id) = n_age
        dim(R) = n_age; dim(D) = n_age

        deriv(S[i])  = -lambda[i] * S[i]
        deriv(Ea[i]) =  lambda[i] * S[i] - 2 * gamma_E * Ea[i]
        deriv(Eb[i]) =  2 * gamma_E * Ea[i] - 2 * gamma_E * Eb[i]
        deriv(Ir[i]) = (1 - cfr[i]) * 2 * gamma_E * Eb[i] - sigma * Ir[i]
        deriv(Id[i]) =  cfr[i]      * 2 * gamma_E * Eb[i] - mu * Id[i]
        deriv(R[i])  =  sigma * Ir[i]
        deriv(D[i])  =  mu * Id[i]

        N = sum(S) + sum(Ea) + sum(Eb) + sum(Ir) + sum(Id) + sum(R) + sum(D)

        dim(cases_rate) = n_age; dim(deaths_rate) = n_age
        cases_rate[i]  = case_scale  * 2 * gamma_E * Eb[i]
        deaths_rate[i] = death_scale * mu * Id[i]

        output(cases_out)  = sum(cases_rate)
        output(deaths_out) = sum(deaths_rate)

        dim(cases) = n_age; dim(deaths) = n_age
        cases[i]  = data()
        deaths[i] = data()
        cases[i]  ~ Poisson(max(cases_rate[i], 1e-6))
        deaths[i] ~ Poisson(max(deaths_rate[i], 1e-6))

        # Initial conditions (parameter arrays)
        dim(S0) = n_age; dim(Ea0) = n_age
        S0 = parameter(rank = 1)
        Ea0 = parameter(rank = 1)
        initial(S[i])  = S0[i]
        initial(Ea[i]) = Ea0[i]
        initial(Eb[i]) = 0
        initial(Ir[i]) = 0
        initial(Id[i]) = 0
        initial(R[i])  = 0
        initial(D[i])  = 0

        # Scalar parameters
        beta       = parameter(0.065, differentiate = true)
        gamma_E    = parameter(1 / 10)
        sigma      = parameter(1 / 14)
        mu         = parameter(1 / 7)
        case_scale  = parameter(6.0, differentiate = true)
        death_scale = parameter(2.0, differentiate = true)
    end

    pars = (
        n_age = 3,
        susc = [1.0, 1.2, 0.8],
        cfr = [0.005, 0.02, 0.08],
        S0 = [4000.0, 3500.0, 2500.0],
        Ea0 = [8.0, 6.0, 4.0],
        beta = 0.065, gamma_E = 1 / 10, sigma = 1 / 14, mu = 1 / 7,
        case_scale = 6.0, death_scale = 2.0,
    )

    # Simulate
    sim_times = collect(0.0:1.0:120.0)
    sim = simulate(mpox_age, pars; times=sim_times, seed=42)
    n_state = size(sim, 1)
    @test n_state >= 21  # 7 compartments × 3 age groups + outputs

    # Build synthetic observations from simulated trajectory
    snames = string.(Odin._odin_state_names(mpox_age.model, pars))
    idx_Eb = [findfirst(==("Eb[$i]"), snames) for i in 1:3]
    idx_Id = [findfirst(==("Id[$i]"), snames) for i in 1:3]
    @test all(!isnothing, idx_Eb)
    @test all(!isnothing, idx_Id)

    obs_times = collect(7.0:7.0:112.0)  # weekly
    rng = Xoshiro(77)
    data_vec = NamedTuple[]
    for t in obs_times
        ti = findfirst(==(t), sim_times)
        c_rate = [pars.case_scale * 2 * pars.gamma_E * sim[idx_Eb[a], 1, ti] for a in 1:3]
        d_rate = [pars.death_scale * pars.mu * sim[idx_Id[a], 1, ti] for a in 1:3]
        push!(data_vec, (
            time = t,
            cases  = Float64[max(1.0, rand(rng, Poisson(max(r, 1e-6)))) for r in c_rate],
            deaths = Float64[max(1.0, rand(rng, Poisson(max(r, 1e-6)))) for r in d_rate],
        ))
    end

    fdata = ObservedData(data_vec)
    lik = Likelihood(mpox_age, fdata; time_start = 0.0)
    ll = loglik(lik, pars)
    @test isfinite(ll)
    @test ll < 0

    # Trajectory shape
    traj = last_trajectories(lik.inner)
    @test traj !== nothing
    @test size(traj, 1) >= 21
    @test size(traj, 2) == length(obs_times)

    # Forward / adjoint gradient agreement
    packer = Packer([:beta, :case_scale, :death_scale])
    fwd = loglik_gradient(lik, pars, packer; method=:forward)
    adj = loglik_gradient(lik, pars, packer; method=:adjoint)

    @test isfinite(fwd.log_likelihood)
    @test isfinite(adj.log_likelihood)
    @test all(isfinite, fwd.gradient)
    @test all(isfinite, adj.gradient)
    @test fwd.log_likelihood ≈ adj.log_likelihood atol=1e-3
    @test fwd.gradient ≈ adj.gradient rtol=0.35 atol=0.1

    # Point fit improves log-likelihood
    fit_fixed = (
        n_age = 3,
        susc = [1.0, 1.2, 0.8],
        cfr = [0.005, 0.02, 0.08],
        S0 = [4000.0, 3500.0, 2500.0],
        Ea0 = [8.0, 6.0, 4.0],
        gamma_E = 1 / 10, sigma = 1 / 14, mu = 1 / 7,
    )
    fit_initial = (beta = 0.04, case_scale = 4.0, death_scale = 1.0)
    fit_start_ll = loglik(lik, merge(fit_fixed, fit_initial))
    fit = _fit_logspace(lik, packer, fit_fixed, fit_initial; step=0.15, max_iter=30)

    @test fit.log_likelihood > fit_start_ll + 50
    @test length(fit.history) > 5
    @test all(x -> x > 0, Tuple(fit.free))
end
