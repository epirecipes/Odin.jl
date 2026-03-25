using Test
using Odin
using Statistics

@testset "Yellow Fever SEIR with 2-Track Vaccination" begin

    yf_vtrack = @odin begin
        # === Configuration ===
        N_age = parameter(5)

        # === Dimensions: 2D arrays (age x vaccination track) ===
        dim(S) = c(N_age, 2)
        dim(E) = c(N_age, 2)
        dim(I) = c(N_age, 2)
        dim(R) = c(N_age, 2)
        dim(C) = N_age

        dim(S_0) = c(N_age, 2)
        dim(E_0) = c(N_age, 2)
        dim(I_0) = c(N_age, 2)
        dim(R_0) = c(N_age, 2)

        dim(E_new) = c(N_age, 2)
        dim(I_new) = c(N_age, 2)
        dim(R_new) = c(N_age, 2)
        dim(S_new_V) = N_age
        dim(R_new_V) = N_age

        dim(P_nV) = N_age
        dim(inv_P_nV) = N_age
        dim(P) = N_age
        dim(inv_P) = N_age
        dim(dP1) = N_age
        dim(dP2) = N_age

        # === Epidemiological parameters ===
        t_latent = parameter(5.0)
        t_infectious = parameter(5.0)
        vaccine_efficacy = parameter(0.95)
        rate1 = 1.0 / t_latent
        rate2 = 1.0 / t_infectious

        # === Time-varying R0 and spillover ===
        R0_t = interpolate(R0_time, R0_value, :linear)
        FOI_sp = interpolate(sp_time, sp_value, :linear)
        beta = R0_t / t_infectious

        # === Population totals (S + R only) ===
        P_nV[i] = max(S[i, 1] + R[i, 1], 1e-99)
        inv_P_nV[i] = 1.0 / P_nV[i]
        P[i] = max(S[i, 1] + R[i, 1] + S[i, 2] + R[i, 2], 1e-99)
        inv_P[i] = 1.0 / P[i]
        P_tot = sum(P)
        I_tot = sum(I)

        # === Force of infection ===
        FOI_raw = beta * I_tot / max(P_tot, 1.0) + FOI_sp
        FOI_sum = min(1.0, FOI_raw)

        # === Transition probabilities ===
        p_inf = 1 - exp(-FOI_sum * dt)
        p_inf_vacc = 1 - exp(-FOI_sum * (1.0 - vaccine_efficacy) * dt)
        p_lat = 1 - exp(-rate1 * dt)
        p_rec = 1 - exp(-rate2 * dt)

        # === Stochastic transitions ===
        E_new[i, j] = if (j == 1) Binomial(S[i, j], p_inf) else Binomial(S[i, j], p_inf_vacc) end
        I_new[i, j] = Binomial(E[i, j], p_lat)
        R_new[i, j] = Binomial(I[i, j], p_rec)

        # === Vaccination flows (track 1 -> track 2) ===
        S_new_V[i] = vacc_rate[i] * S[i, 1] * dt
        R_new_V[i] = vacc_rate[i] * R[i, 1] * dt

        # === Demographic flows ===
        dP1_rate = interpolate(dP1_time, dP1_value, :constant)
        dP2_rate = interpolate(dP2_time, dP2_value, :constant)
        dP1[i] = dP1_rate * 0.01
        dP2[i] = dP2_rate * 0.01

        # === S track 1 (unvaccinated) ===
        update(S[1, 1]) = max(0.0, S[1, 1] - E_new[1, 1] - S_new_V[1]
                              + dP1[1] - dP2[1] * S[1, 1] * inv_P[1])
        update(S[2:N_age, 1]) = max(0.0, S[i, 1] - E_new[i, 1] - S_new_V[i]
                                    + dP1[i] * S[i - 1, 1] * inv_P[i - 1]
                                    - dP2[i] * S[i, 1] * inv_P[i])

        # === S track 2 (vaccinated) ===
        update(S[1, 2]) = max(0.0, S[1, 2] - E_new[1, 2] + S_new_V[1]
                              - dP2[1] * S[1, 2] * inv_P[1])
        update(S[2:N_age, 2]) = max(0.0, S[i, 2] - E_new[i, 2] + S_new_V[i]
                                    + dP1[i] * S[i - 1, 2] * inv_P[i - 1]
                                    - dP2[i] * S[i, 2] * inv_P[i])

        # === E and I: same rule for both tracks ===
        update(E[1:N_age, 1:2]) = max(0.0, E[i, j] + E_new[i, j] - I_new[i, j])
        update(I[1:N_age, 1:2]) = max(0.0, I[i, j] + I_new[i, j] - R_new[i, j])

        # === R track 1 (unvaccinated) ===
        update(R[1, 1]) = max(0.0, R[1, 1] + R_new[1, 1] - R_new_V[1]
                              - dP2[1] * R[1, 1] * inv_P[1])
        update(R[2:N_age, 1]) = max(0.0, R[i, 1] + R_new[i, 1] - R_new_V[i]
                                    + dP1[i] * R[i - 1, 1] * inv_P[i - 1]
                                    - dP2[i] * R[i, 1] * inv_P[i])

        # === R track 2 (vaccinated) ===
        update(R[1, 2]) = max(0.0, R[1, 2] + R_new[1, 2] + R_new_V[1]
                              - dP2[1] * R[1, 2] * inv_P[1])
        update(R[2:N_age, 2]) = max(0.0, R[i, 2] + R_new[i, 2] + R_new_V[i]
                                    + dP1[i] * R[i - 1, 2] * inv_P[i - 1]
                                    - dP2[i] * R[i, 2] * inv_P[i])

        # === New cases per step (both tracks) ===
        initial(C[i], zero_every = 1) = 0
        update(C[i]) = I_new[i, 1] + I_new[i, 2]

        # === Outputs ===
        output(FOI_total) = FOI_sum
        output(total_I) = I_tot
        output(total_pop) = P_tot

        # === Data comparison for particle filter ===
        cases = data()
        C_total = sum(C)
        cases ~ Poisson(max(C_total, 1e-6))

        # === Initial conditions ===
        initial(S[i, j]) = S_0[i, j]
        initial(E[i, j]) = E_0[i, j]
        initial(I[i, j]) = I_0[i, j]
        initial(R[i, j]) = R_0[i, j]

        # === Parameters ===
        S_0 = parameter()
        E_0 = parameter()
        I_0 = parameter()
        R_0 = parameter()
        vacc_rate = parameter(rank = 1)

        R0_time = parameter(rank = 1)
        R0_value = parameter(rank = 1)
        sp_time = parameter(rank = 1)
        sp_value = parameter(rank = 1)
        dP1_time = parameter(rank = 1)
        dP1_value = parameter(rank = 1)
        dP2_time = parameter(rank = 1)
        dP2_value = parameter(rank = 1)
    end

    # --- Shared test parameters ---
    N_age = 5
    pop = [5000.0, 8000.0, 6000.0, 5000.0, 3000.0]
    N_total = sum(pop)

    S_0 = zeros(N_age, 2)
    E_0 = zeros(N_age, 2)
    I_0 = zeros(N_age, 2)
    R_0 = zeros(N_age, 2)

    immun_frac = [0.05, 0.10, 0.15, 0.20, 0.30]
    for i in 1:N_age
        R_0[i, 1] = round(pop[i] * immun_frac[i])
    end

    vacc_frac = [0.10, 0.15, 0.10, 0.05, 0.02]
    for i in 1:N_age
        S_0[i, 2] = round(pop[i] * vacc_frac[i])
    end

    for i in 1:N_age
        S_0[i, 1] = pop[i] - R_0[i, 1] - S_0[i, 2]
    end

    I_0[3, 1] = 10.0
    S_0[3, 1] -= 10.0

    t_end = 365.0
    R0_time = collect(0.0:30.0:(t_end + 30.0))
    R0_value = [3.0 + 1.0 * sin(2π * t / 365) for t in R0_time]
    sp_time = collect(0.0:30.0:(t_end + 30.0))
    sp_value = [1e-6 + 2e-5 * max(0, sin(2π * t / 365 - π / 3))^3 for t in sp_time]
    dP1_time = [0.0, t_end + 1.0]
    dP1_value = [1.0, 1.0]
    dP2_time = [0.0, t_end + 1.0]
    dP2_value = [1.0, 1.0]
    vacc_rate = [0.001, 0.0005, 0.0003, 0.0002, 0.0001]

    pars = (
        N_age = Float64(N_age),
        t_latent = 5.0,
        t_infectious = 5.0,
        vaccine_efficacy = 0.95,
        S_0 = S_0,
        E_0 = E_0,
        I_0 = I_0,
        R_0 = R_0,
        vacc_rate = vacc_rate,
        R0_time = R0_time,
        R0_value = R0_value,
        sp_time = sp_time,
        sp_value = sp_value,
        dP1_time = dP1_time,
        dP1_value = dP1_value,
        dP2_time = dP2_time,
        dP2_value = dP2_value,
    )

    times = collect(0.0:1.0:t_end)

    # State layout (column-major 2D arrays):
    # C[1:5], S[6:15] (5 unvacc + 5 vacc), E[16:25], I[26:35], R[36:45]
    # Outputs: FOI_total, total_I, total_pop (indices 46, 47, 48)
    idx_C = 1:N_age
    idx_S = (N_age + 1):(3 * N_age)
    idx_E = (3 * N_age + 1):(5 * N_age)
    idx_I = (5 * N_age + 1):(7 * N_age)
    idx_R = (7 * N_age + 1):(9 * N_age)
    out_offset = 9 * N_age

    # Track-specific indices
    idx_S1 = (N_age + 1):(2 * N_age)       # S unvaccinated
    idx_S2 = (2 * N_age + 1):(3 * N_age)   # S vaccinated
    idx_I1 = (5 * N_age + 1):(6 * N_age)   # I unvaccinated
    idx_I2 = (6 * N_age + 1):(7 * N_age)   # I vaccinated
    idx_R1 = (7 * N_age + 1):(8 * N_age)   # R unvaccinated
    idx_R2 = (8 * N_age + 1):(9 * N_age)   # R vaccinated

    @testset "compiles and runs" begin
        result = dust_system_simulate(yf_vtrack, pars;
            times=times, dt=1.0, seed=42, n_particles=5)
        # 45 state vars (4 compartments × 5 ages × 2 tracks + 5 C) + 3 outputs = 48
        @test size(result, 1) == 48
        @test size(result, 2) == 5
        @test size(result, 3) == length(times)
        @test all(isfinite, result)
    end

    @testset "2D array state dimensions" begin
        sys = Odin.dust_system_create(yf_vtrack, pars; dt=1.0, seed=1)
        Odin.dust_system_set_state_initial!(sys)
        s = Odin.dust_system_state(sys)

        # S should have 2*N_age elements (N_age per track, column-major)
        @test length(s[idx_S, 1]) == 2 * N_age
        @test length(s[idx_E, 1]) == 2 * N_age
        @test length(s[idx_I, 1]) == 2 * N_age
        @test length(s[idx_R, 1]) == 2 * N_age
        @test length(s[idx_C, 1]) == N_age
    end

    @testset "initial conditions correct" begin
        sys = Odin.dust_system_create(yf_vtrack, pars; dt=1.0, seed=1)
        Odin.dust_system_set_state_initial!(sys)
        s = Odin.dust_system_state(sys)

        # S track 1 (unvaccinated)
        @test s[idx_S1, 1] ≈ S_0[:, 1]
        # S track 2 (vaccinated)
        @test s[idx_S2, 1] ≈ S_0[:, 2]
        # E should be all zeros
        @test all(s[idx_E, 1] .== 0.0)
        # I track 1: only age group 3 seeded
        @test s[idx_I1[3], 1] ≈ 10.0
        @test s[idx_I2, 1] ≈ zeros(N_age)
        # R track 1 (pre-existing immunity)
        @test s[idx_R1, 1] ≈ R_0[:, 1]
        @test s[idx_R2, 1] ≈ zeros(N_age)
    end

    @testset "vaccination moves between tracks" begin
        short_times = collect(0.0:1.0:50.0)

        # With vaccination
        result_vacc = dust_system_simulate(yf_vtrack, pars;
            times=short_times, dt=1.0, seed=42, n_particles=20)

        # Without vaccination
        pars_novacc = merge(pars, (vacc_rate = zeros(N_age),))
        result_novacc = dust_system_simulate(yf_vtrack, pars_novacc;
            times=short_times, dt=1.0, seed=42, n_particles=20)

        # With vaccination, S track 2 (vaccinated) should increase
        S2_start_vacc = mean(sum(result_vacc[idx_S2, :, 1], dims=1))
        S2_end_vacc = mean(sum(result_vacc[idx_S2, :, end], dims=1))

        # S track 1 should decrease faster with vaccination
        S1_end_vacc = mean(sum(result_vacc[idx_S1, :, end], dims=1))
        S1_end_novacc = mean(sum(result_novacc[idx_S1, :, end], dims=1))

        @test S2_end_vacc > S2_start_vacc
        @test S1_end_vacc < S1_end_novacc
    end

    @testset "FOI uses both tracks" begin
        result = dust_system_simulate(yf_vtrack, pars;
            times=times, dt=1.0, seed=42, n_particles=5)
        foi = result[out_offset + 1, :, :]

        # FOI should be non-negative and bounded by 1
        @test all(foi .>= 0.0)
        @test all(foi .<= 1.0)

        # total_I output should sum both tracks
        total_I_out = result[out_offset + 2, :, :]
        for p in 1:5, t in 1:length(times)
            I_both = sum(result[idx_I, p, t])
            @test total_I_out[p, t] ≈ I_both atol=0.1
        end
    end

    @testset "no negative populations" begin
        result = dust_system_simulate(yf_vtrack, pars;
            times=times, dt=1.0, seed=42, n_particles=10)
        for idx in [idx_S, idx_E, idx_I, idx_R]
            @test all(result[idx, :, :] .>= 0.0)
        end
    end

    @testset "population approximately conserved" begin
        result = dust_system_simulate(yf_vtrack, pars;
            times=times, dt=1.0, seed=42, n_particles=5)
        for p in 1:5, t in 1:length(times)
            pop_t = sum(result[idx_S, p, t]) + sum(result[idx_E, p, t]) +
                    sum(result[idx_I, p, t]) + sum(result[idx_R, p, t])
            @test abs(pop_t - N_total) / N_total < 0.15
        end
    end

    @testset "stochastic particles diverge" begin
        result = dust_system_simulate(yf_vtrack, pars;
            times=times, dt=1.0, seed=42, n_particles=10)
        mid = div(length(times), 2)
        C_mid = [sum(result[idx_C, p, mid]) for p in 1:10]
        early = min(31, length(times))
        S_early = [sum(result[idx_S, p, early]) for p in 1:10]
        @test length(unique(C_mid)) > 1 || length(unique(S_early)) > 1
    end

    @testset "particle filter convergence" begin
        # Generate synthetic data
        syn_times = collect(7.0:7.0:100.0)
        syn_result = dust_system_simulate(yf_vtrack, pars;
            times=syn_times, dt=1.0, seed=123, n_particles=1)
        syn_cases = [max(1.0, sum(syn_result[idx_C, 1, t])) for t in 1:length(syn_times)]
        data_vec = [(time=syn_times[i], cases=syn_cases[i]) for i in 1:length(syn_times)]
        fdata = Odin.dust_filter_data(data_vec)

        # Run particle filter
        filter = dust_filter_create(yf_vtrack, fdata;
            n_particles=50, dt=1.0, seed=42)
        ll = dust_likelihood_run!(filter, pars)

        @test isfinite(ll)
        @test ll < 0  # log-likelihood should be negative

        # Worse parameters should give worse log-likelihood
        pars_bad = merge(pars, (t_infectious = 50.0,))
        ll_bad = dust_likelihood_run!(filter, pars_bad)
        @test isfinite(ll_bad)
        @test ll > ll_bad
    end
end
