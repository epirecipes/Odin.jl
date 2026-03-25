using Test
using Odin
using Statistics

@testset "Yellow Fever SEIRV model" begin

    yf_seirv = @odin begin
        # === Configuration ===
        N_age = parameter(5)

        # === Dimensions ===
        dim(S) = N_age
        dim(E) = N_age
        dim(I) = N_age
        dim(R) = N_age
        dim(V) = N_age
        dim(C) = N_age

        dim(S_0) = N_age
        dim(E_0) = N_age
        dim(I_0) = N_age
        dim(R_0) = N_age
        dim(V_0) = N_age

        dim(dP1) = N_age
        dim(dP2) = N_age
        dim(E_new) = N_age
        dim(I_new) = N_age
        dim(R_new) = N_age
        dim(P_nV) = N_age
        dim(inv_P_nV) = N_age
        dim(P) = N_age
        dim(inv_P) = N_age
        dim(vacc_eff) = N_age

        # === Epidemiological rates ===
        t_latent = parameter(5.0)
        t_infectious = parameter(5.0)
        rate1 = 1.0 / t_latent
        rate2 = 1.0 / t_infectious

        # === Time-varying R0 and spillover ===
        R0_t = interpolate(R0_time, R0_value, :linear)
        FOI_sp = interpolate(sp_time, sp_value, :linear)
        beta = R0_t / t_infectious

        # === Force of infection ===
        P_total = sum(P)
        I_total = sum(I)
        FOI_raw = beta * I_total / max(P_total, 1.0) + FOI_sp
        FOI_max = 1.0
        FOI_sum = min(FOI_max, FOI_raw)

        # === Population totals per age ===
        P_nV[i] = max(S[i] + R[i], 1e-99)
        inv_P_nV[i] = 1.0 / P_nV[i]
        P[i] = max(P_nV[i] + V[i], 1e-99)
        inv_P[i] = 1.0 / P[i]

        # === Transitions ===
        p_inf = 1 - exp(-FOI_sum * dt)
        p_lat = 1 - exp(-rate1 * dt)
        p_rec = 1 - exp(-rate2 * dt)

        E_new[i] = Binomial(S[i], p_inf)
        I_new[i] = Binomial(E[i], p_lat)
        R_new[i] = Binomial(I[i], p_rec)

        # === Vaccination ===
        vaccine_efficacy = parameter(0.95)
        vacc_eff[i] = vacc_rate[i] * vaccine_efficacy * dt

        # === Demographic flows ===
        dP1_rate = interpolate(dP1_time, dP1_value, :constant)
        dP2_rate = interpolate(dP2_time, dP2_value, :constant)
        dP1[i] = dP1_rate * 0.01
        dP2[i] = dP2_rate * 0.01

        # === State updates: age group 1 (youngest) ===
        update(S[1]) = max(0.0, S[1] - E_new[1]
                           - vacc_eff[1] * S[1] * inv_P_nV[1]
                           + dP1[1]
                           - dP2[1] * S[1] * inv_P[1])
        update(E[1]) = max(0.0, E[1] + E_new[1] - I_new[1])
        update(I[1]) = max(0.0, I[1] + I_new[1] - R_new[1])
        update(R[1]) = max(0.0, R[1] + R_new[1]
                           - vacc_eff[1] * R[1] * inv_P_nV[1]
                           - dP2[1] * R[1] * inv_P[1])
        update(V[1]) = max(0.0, V[1] + vacc_eff[1]
                           - dP2[1] * V[1] * inv_P[1])

        # === State updates: age groups 2..N_age (aging from i-1) ===
        update(S[2:N_age]) = max(0.0, S[i] - E_new[i]
                                 - vacc_eff[i] * S[i] * inv_P_nV[i]
                                 + dP1[i] * S[i - 1] * inv_P[i - 1]
                                 - dP2[i] * S[i] * inv_P[i])
        update(E[2:N_age]) = max(0.0, E[i] + E_new[i] - I_new[i])
        update(I[2:N_age]) = max(0.0, I[i] + I_new[i] - R_new[i])
        update(R[2:N_age]) = max(0.0, R[i] + R_new[i]
                                 - vacc_eff[i] * R[i] * inv_P_nV[i]
                                 + dP1[i] * R[i - 1] * inv_P[i - 1]
                                 - dP2[i] * R[i] * inv_P[i])
        update(V[2:N_age]) = max(0.0, V[i] + vacc_eff[i]
                                 + dP1[i] * V[i - 1] * inv_P[i - 1]
                                 - dP2[i] * V[i] * inv_P[i])

        # === Cumulative new cases per step ===
        initial(C[i], zero_every = 1) = 0
        update(C[i]) = C[i] + I_new[i]

        # === Outputs ===
        output(FOI_total) = FOI_sum
        output(total_I) = I_total
        output(total_pop) = P_total

        # === Initial conditions ===
        initial(S[i]) = S_0[i]
        initial(E[i]) = E_0[i]
        initial(I[i]) = I_0[i]
        initial(R[i]) = R_0[i]
        initial(V[i]) = V_0[i]

        # === Parameters ===
        S_0 = parameter()
        E_0 = parameter()
        I_0 = parameter()
        R_0 = parameter()
        V_0 = parameter()
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

    S_0 = copy(pop)
    E_0 = zeros(N_age)
    I_0 = zeros(N_age)
    R_0 = zeros(N_age)
    V_0 = zeros(N_age)

    immun_frac = [0.05, 0.10, 0.15, 0.20, 0.30]
    for i in 1:N_age
        R_0[i] = round(pop[i] * immun_frac[i])
        S_0[i] -= R_0[i]
    end

    vacc_frac = [0.10, 0.15, 0.10, 0.05, 0.02]
    for i in 1:N_age
        V_0[i] = round(pop[i] * vacc_frac[i])
        S_0[i] -= V_0[i]
    end

    I_0[3] = 10.0
    S_0[3] -= 10.0

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
        V_0 = V_0,
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
    # State layout follows initial() declaration order:
    # C[1:5] (zero_every declared first), S[6:10], E[11:15], I[16:20], R[21:25], V[26:30]
    idx_C = 1:N_age
    idx_S = (N_age + 1):(2 * N_age)
    idx_E = (2 * N_age + 1):(3 * N_age)
    idx_I = (3 * N_age + 1):(4 * N_age)
    idx_R = (4 * N_age + 1):(5 * N_age)
    idx_V = (5 * N_age + 1):(6 * N_age)
    out_offset = 6 * N_age

    @testset "compiles and runs" begin
        result = dust_system_simulate(yf_seirv, pars;
            times=times, dt=1.0, seed=42, n_particles=5)
        # 30 state vars (6 compartments × 5 ages) + 3 outputs = 33
        @test size(result, 1) == 33
        @test size(result, 2) == 5
        @test size(result, 3) == length(times)
        @test all(isfinite, result)
    end

    @testset "initial conditions correct" begin
        sys = Odin.dust_system_create(yf_seirv, pars; dt=1.0, seed=1)
        Odin.dust_system_set_state_initial!(sys)
        s = Odin.dust_system_state(sys)

        @test s[idx_S, 1] ≈ S_0
        @test s[idx_E, 1] ≈ E_0
        @test s[idx_I, 1] ≈ I_0
        @test s[idx_R, 1] ≈ R_0
        @test s[idx_V, 1] ≈ V_0
    end

    @testset "no negative populations" begin
        result = dust_system_simulate(yf_seirv, pars;
            times=times, dt=1.0, seed=42, n_particles=10)
        # S, E, I, R, V should all be >= 0
        for idx in [idx_S, idx_E, idx_I, idx_R, idx_V]
            @test all(result[idx, :, :] .>= 0.0)
        end
    end

    @testset "population approximately conserved" begin
        result = dust_system_simulate(yf_seirv, pars;
            times=times, dt=1.0, seed=42, n_particles=5)
        for p in 1:5, t in 1:length(times)
            pop_t = sum(result[idx_S, p, t]) + sum(result[idx_E, p, t]) +
                    sum(result[idx_I, p, t]) + sum(result[idx_R, p, t]) +
                    sum(result[idx_V, p, t])
            # Allow up to 15% drift from demographic flows + max floors
            @test abs(pop_t - N_total) / N_total < 0.15
        end
    end

    @testset "FOI_total bounded" begin
        result = dust_system_simulate(yf_seirv, pars;
            times=times, dt=1.0, seed=42, n_particles=5)
        foi = result[out_offset + 1, :, :]
        @test all(foi .>= 0.0)
        @test all(foi .<= 1.0)
    end

    @testset "vaccination reduces susceptibles" begin
        # Run short simulation (50 days) so epidemic doesn't exhaust all S
        short_times = collect(0.0:1.0:50.0)

        # With vaccination
        result_vacc = dust_system_simulate(yf_seirv, pars;
            times=short_times, dt=1.0, seed=42, n_particles=20)

        # Without vaccination
        pars_novacc = merge(pars, (vacc_rate = zeros(N_age),
                                    V_0 = zeros(N_age),
                                    S_0 = pop .- R_0 .- I_0))
        result_novacc = dust_system_simulate(yf_seirv, pars_novacc;
            times=short_times, dt=1.0, seed=42, n_particles=20)

        # Vaccinated compartment should grow over time
        V_start = mean(sum(result_vacc[idx_V, :, 1], dims=1))
        V_end = mean(sum(result_vacc[idx_V, :, end], dims=1))
        @test V_end > V_start

        # Without vaccination, V stays near zero
        V_novacc_end = mean(sum(result_novacc[idx_V, :, end], dims=1))
        @test V_end > V_novacc_end
    end

    @testset "stochastic particles diverge" begin
        result = dust_system_simulate(yf_seirv, pars;
            times=times, dt=1.0, seed=42, n_particles=10)
        # Cumulative cases C at midpoint should vary across particles
        mid = div(length(times), 2)
        C_mid = [sum(result[idx_C, p, mid]) for p in 1:10]
        # Also check S at an early time (day ~30) when epidemic is active
        early = min(31, length(times))
        S_early = [sum(result[idx_S, p, early]) for p in 1:10]
        # At least one of these should show variation
        @test length(unique(C_mid)) > 1 || length(unique(S_early)) > 1
    end

    @testset "cases counter resets each step" begin
        result = dust_system_simulate(yf_seirv, pars;
            times=times, dt=1.0, seed=42, n_particles=1)
        # C values should be non-negative (new cases per step)
        @test all(result[idx_C, :, :] .>= 0.0)
        # C values should not accumulate (bounded by total E)
        max_C = maximum(result[idx_C, :, :])
        @test max_C < N_total
    end
end
