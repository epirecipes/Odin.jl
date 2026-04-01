using Test
using Odin
using Random
using Distributions

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
