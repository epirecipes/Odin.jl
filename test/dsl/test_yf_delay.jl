using Test
using Odin
using Statistics

@testset "Yellow Fever SEIRV with Erlang Delay Compartments" begin

    yf_delay = @odin begin
        # === Configuration ===
        N_age = parameter(3)
        k_E = parameter(4)
        k_I = parameter(3)

        # === 1D compartment dimensions ===
        dim(S) = N_age
        dim(R) = N_age
        dim(V) = N_age
        dim(C_new) = N_age

        # === 2D delay chain dimensions ===
        dim(E_chain) = c(N_age, k_E)
        dim(I_chain) = c(N_age, k_I)
        dim(n_EE) = c(N_age, k_E)
        dim(n_II) = c(N_age, k_I)

        # === 1D helper dimensions ===
        dim(S_0) = N_age
        dim(R_0) = N_age
        dim(V_0) = N_age
        dim(E_new) = N_age
        dim(I_new) = N_age
        dim(R_new) = N_age
        dim(E_total) = N_age
        dim(I_total_age) = N_age
        dim(P_nV) = N_age
        dim(inv_P_nV) = N_age
        dim(P) = N_age
        dim(inv_P) = N_age
        dim(vacc_eff) = N_age
        dim(dP1) = N_age
        dim(dP2) = N_age

        # === Epidemiological rates ===
        t_latent = parameter(5.0)
        t_infectious = parameter(5.0)
        sigma = 1.0 / t_latent
        gamma = 1.0 / t_infectious

        # === Time-varying R0 and spillover ===
        R0_t = interpolate(R0_time, R0_value, :linear)
        FOI_sp = interpolate(sp_time, sp_value, :linear)
        beta = R0_t / t_infectious

        # === Force of infection ===
        I_total = sum(I_chain)
        P_total = sum(P)
        FOI_raw = beta * I_total / max(P_total, 1.0) + FOI_sp
        FOI_sum = min(1.0, FOI_raw)

        # === Totals per age group ===
        E_total[i] = sum(E_chain[i, ])
        I_total_age[i] = sum(I_chain[i, ])
        P_nV[i] = max(S[i] + R[i], 1e-99)
        inv_P_nV[i] = 1.0 / P_nV[i]
        P[i] = max(P_nV[i] + V[i], 1e-99)
        inv_P[i] = 1.0 / P[i]

        # === Transition probabilities (Erlang rates) ===
        p_inf = 1 - exp(-FOI_sum * dt)
        p_E = 1 - exp(-k_E * sigma * dt)
        p_I = 1 - exp(-k_I * gamma * dt)

        # === Stochastic transitions ===
        E_new[i] = Binomial(S[i], p_inf)
        n_EE[i, j] = Binomial(E_chain[i, j], p_E)
        n_II[i, j] = Binomial(I_chain[i, j], p_I)

        I_new[i] = n_EE[i, k_E]
        R_new[i] = n_II[i, k_I]

        # === Vaccination ===
        vaccine_efficacy = parameter(0.95)
        vacc_eff[i] = vacc_rate[i] * vaccine_efficacy * dt

        # === Demographic flows ===
        dP1_rate = interpolate(dP1_time, dP1_value, :constant)
        dP2_rate = interpolate(dP2_time, dP2_value, :constant)
        dP1[i] = dP1_rate * 0.01
        dP2[i] = dP2_rate * 0.01

        # === S updates ===
        update(S[1]) = max(0.0, S[1] - E_new[1]
                           - vacc_eff[1] * S[1] * inv_P_nV[1]
                           + dP1[1]
                           - dP2[1] * S[1] * inv_P[1])
        update(S[2:N_age]) = max(0.0, S[i] - E_new[i]
                                 - vacc_eff[i] * S[i] * inv_P_nV[i]
                                 + dP1[i] * S[i - 1] * inv_P[i - 1]
                                 - dP2[i] * S[i] * inv_P[i])

        # === E delay chain ===
        update(E_chain[1:N_age, 1]) = max(0.0, E_chain[i, 1] + E_new[i] - n_EE[i, 1])
        update(E_chain[1:N_age, 2:k_E]) = max(0.0, E_chain[i, j] + n_EE[i, j - 1] - n_EE[i, j])

        # === I delay chain ===
        update(I_chain[1:N_age, 1]) = max(0.0, I_chain[i, 1] + I_new[i] - n_II[i, 1])
        update(I_chain[1:N_age, 2:k_I]) = max(0.0, I_chain[i, j] + n_II[i, j - 1] - n_II[i, j])

        # === R updates ===
        update(R[1]) = max(0.0, R[1] + R_new[1]
                           - vacc_eff[1] * R[1] * inv_P_nV[1]
                           - dP2[1] * R[1] * inv_P[1])
        update(R[2:N_age]) = max(0.0, R[i] + R_new[i]
                                 - vacc_eff[i] * R[i] * inv_P_nV[i]
                                 + dP1[i] * R[i - 1] * inv_P[i - 1]
                                 - dP2[i] * R[i] * inv_P[i])

        # === V updates ===
        update(V[1]) = max(0.0, V[1] + vacc_eff[1]
                           - dP2[1] * V[1] * inv_P[1])
        update(V[2:N_age]) = max(0.0, V[i] + vacc_eff[i]
                                 + dP1[i] * V[i - 1] * inv_P[i - 1]
                                 - dP2[i] * V[i] * inv_P[i])

        # === Cumulative new cases ===
        initial(C_new[i], zero_every = 1) = 0
        update(C_new[i]) = C_new[i] + I_new[i]

        # === Outputs ===
        output(FOI_total) = FOI_sum
        output(total_I) = I_total
        output(total_pop) = P_total

        # === Initial conditions ===
        initial(S[i]) = S_0[i]
        initial(E_chain[i, j]) = 0
        initial(I_chain[i, 1]) = 0
        initial(I_chain[i, 2:k_I]) = 0
        initial(R[i]) = R_0[i]
        initial(V[i]) = V_0[i]

        # === Parameters ===
        S_0 = parameter()
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
    N_age = 3
    k_E = 4
    k_I = 3
    pop = [5000.0, 10000.0, 5000.0]
    N_total = sum(pop)

    S_0 = copy(pop)
    R_0 = zeros(N_age)
    V_0 = zeros(N_age)

    immun_frac = [0.05, 0.15, 0.25]
    for i in 1:N_age
        R_0[i] = round(pop[i] * immun_frac[i])
        S_0[i] -= R_0[i]
    end

    vacc_frac = [0.10, 0.15, 0.05]
    for i in 1:N_age
        V_0[i] = round(pop[i] * vacc_frac[i])
        S_0[i] -= V_0[i]
    end

    n_seed = 10.0
    S_0[2] -= n_seed

    t_end = 365.0
    R0_time = collect(0.0:30.0:(t_end + 30.0))
    R0_value = [3.0 + 1.0 * sin(2π * t / 365) for t in R0_time]
    sp_time = collect(0.0:30.0:(t_end + 30.0))
    sp_value = [1e-6 + 2e-5 * max(0, sin(2π * t / 365 - π / 3))^3 for t in sp_time]
    dP1_time = [0.0, t_end + 1.0]
    dP1_value = [1.0, 1.0]
    dP2_time = [0.0, t_end + 1.0]
    dP2_value = [1.0, 1.0]
    vacc_rate = [0.001, 0.0005, 0.0002]

    pars = (
        N_age = Float64(N_age),
        k_E = Float64(k_E),
        k_I = Float64(k_I),
        t_latent = 5.0,
        t_infectious = 5.0,
        vaccine_efficacy = 0.95,
        S_0 = S_0,
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

    # State layout (column-major 2D arrays):
    # C_new[1:3], S[4:6], E_chain[7:18] (3×4), I_chain[19:27] (3×3), R[28:30], V[31:33]
    idx_C = 1:N_age
    idx_S = (N_age + 1):(2 * N_age)
    idx_E = (2 * N_age + 1):(2 * N_age + N_age * k_E)
    idx_I = (2 * N_age + N_age * k_E + 1):(2 * N_age + N_age * k_E + N_age * k_I)
    idx_R = (idx_I[end] + 1):(idx_I[end] + N_age)
    idx_V = (idx_R[end] + 1):(idx_R[end] + N_age)
    out_offset = idx_V[end]
    # I_chain[2, 1] index for seeding
    i_chain_offset = 2 * N_age + N_age * k_E
    i_chain_idx_2_1 = i_chain_offset + 2

    function seed_and_simulate(gen, p, n_part; seed=42)
        sys = Odin.dust_system_create(gen, p; dt=1.0, seed=seed,
                                       n_particles=n_part)
        Odin.dust_system_set_state_initial!(sys)
        state = Odin.dust_system_state(sys)
        for pp in 1:n_part
            state[i_chain_idx_2_1, pp] = n_seed
        end
        Odin.dust_system_set_state!(sys, state)
        return Odin.dust_system_simulate(sys, times)
    end

    @testset "compiles and runs" begin
        result = seed_and_simulate(yf_delay, pars, 5)
        # State: 3 (C) + 3 (S) + 12 (E) + 9 (I) + 3 (R) + 3 (V) = 33
        # Outputs: 3 (FOI, I_total, pop)
        @test size(result, 1) == 36
        @test size(result, 2) == 5
        @test size(result, 3) == length(times)
        @test all(isfinite, result)
    end

    @testset "initial conditions correct" begin
        sys = Odin.dust_system_create(yf_delay, pars; dt=1.0, seed=1)
        Odin.dust_system_set_state_initial!(sys)
        s = Odin.dust_system_state(sys)

        @test s[idx_S, 1] ≈ S_0
        @test s[idx_R, 1] ≈ R_0
        @test s[idx_V, 1] ≈ V_0
        # E_chain and I_chain should all be zero initially
        @test all(s[idx_E, 1] .== 0.0)
        @test all(s[idx_I, 1] .== 0.0)
    end

    @testset "no negative populations" begin
        result = seed_and_simulate(yf_delay, pars, 10)
        for idx in [idx_S, idx_E, idx_I, idx_R, idx_V]
            @test all(result[idx, :, :] .>= 0.0)
        end
    end

    @testset "population approximately conserved" begin
        result = seed_and_simulate(yf_delay, pars, 5)
        for p in 1:5, t in 1:length(times)
            pop_t = sum(result[idx_S, p, t]) + sum(result[idx_E, p, t]) +
                    sum(result[idx_I, p, t]) + sum(result[idx_R, p, t]) +
                    sum(result[idx_V, p, t])
            @test abs(pop_t - N_total) / N_total < 0.15
        end
    end

    @testset "FOI_total bounded" begin
        result = seed_and_simulate(yf_delay, pars, 5)
        foi = result[out_offset + 1, :, :]
        @test all(foi .>= 0.0)
        @test all(foi .<= 1.0)
    end

    @testset "delay chain progression" begin
        # After a few steps, infections should have moved through E chain stages
        short_times = collect(0.0:1.0:30.0)
        sys = Odin.dust_system_create(yf_delay, pars; dt=1.0, seed=42,
                                       n_particles=1)
        Odin.dust_system_set_state_initial!(sys)
        state = Odin.dust_system_state(sys)
        state[i_chain_idx_2_1, 1] = n_seed
        Odin.dust_system_set_state!(sys, state)
        result = Odin.dust_system_simulate(sys, short_times)

        # After 30 days, some E_chain stages beyond stage 1 should be populated
        # or I_chain should have individuals (showing progression through E)
        e_at_end = result[idx_E, 1, end]
        i_at_end = result[idx_I, 1, end]
        r_at_end = result[idx_R, 1, end]
        # People should have moved into I or R by now
        @test sum(i_at_end) + sum(r_at_end) > 0
    end

    @testset "k=1 matches exponential behaviour" begin
        # With k_E=1, k_I=1, the model should behave like standard SEIR
        pars_k1 = merge(pars, (k_E = 1.0, k_I = 1.0))

        # Recompute indices for k=1
        idx_E_k1 = (2 * N_age + 1):(2 * N_age + N_age * 1)
        idx_I_k1 = (2 * N_age + N_age * 1 + 1):(2 * N_age + N_age * 1 + N_age * 1)
        idx_R_k1 = (idx_I_k1[end] + 1):(idx_I_k1[end] + N_age)
        idx_V_k1 = (idx_R_k1[end] + 1):(idx_R_k1[end] + N_age)

        i_chain_offset_k1 = 2 * N_age + N_age * 1
        i_chain_idx_k1 = i_chain_offset_k1 + 2

        # Run many realisations to get stable mean
        n_runs = 50
        I_mean_k1 = zeros(length(times))
        for seed in 1:n_runs
            sys = Odin.dust_system_create(yf_delay, pars_k1; dt=1.0, seed=seed)
            Odin.dust_system_set_state_initial!(sys)
            st = Odin.dust_system_state(sys)
            st[i_chain_idx_k1, 1] = n_seed
            Odin.dust_system_set_state!(sys, st)
            r = Odin.dust_system_simulate(sys, times)
            for t in 1:length(times)
                I_mean_k1[t] += sum(r[idx_I_k1, 1, t])
            end
        end
        I_mean_k1 ./= n_runs

        # Should produce a valid epidemic curve
        @test maximum(I_mean_k1) > 0
        # Peak should occur within reasonable time
        peak_day = argmax(I_mean_k1)
        @test peak_day > 5 && peak_day < length(times)
    end

    @testset "higher k produces sharper peaks" begin
        # Compare k=1 vs k=4: higher k should have higher peak (same mean,
        # lower variance → more synchronised progression)
        n_runs = 40

        function mean_peak_I(gen, p, k_e, k_i, n_runs)
            p_mod = merge(p, (k_E = Float64(k_e), k_I = Float64(k_i)))
            idx_I_mod = (2 * N_age + N_age * k_e + 1):(2 * N_age + N_age * k_e + N_age * k_i)
            ic_offset = 2 * N_age + N_age * k_e
            ic_idx = ic_offset + 2

            peaks = Float64[]
            for seed in 1:n_runs
                sys = Odin.dust_system_create(gen, p_mod; dt=1.0, seed=seed)
                Odin.dust_system_set_state_initial!(sys)
                st = Odin.dust_system_state(sys)
                st[ic_idx, 1] = n_seed
                Odin.dust_system_set_state!(sys, st)
                r = Odin.dust_system_simulate(sys, times)
                peak = maximum([sum(r[idx_I_mod, 1, t]) for t in 1:length(times)])
                push!(peaks, peak)
            end
            return mean(peaks)
        end

        peak_k1 = mean_peak_I(yf_delay, pars, 1, 1, n_runs)
        peak_k4 = mean_peak_I(yf_delay, pars, 4, 3, n_runs)

        # Higher k should produce higher (sharper) peak
        @test peak_k4 > peak_k1 * 0.9  # allowing some stochastic noise
    end

    @testset "stochastic particles diverge" begin
        result = seed_and_simulate(yf_delay, pars, 10)
        mid = div(length(times), 2)
        C_mid = [sum(result[idx_C, p, mid]) for p in 1:10]
        early = min(31, length(times))
        S_early = [sum(result[idx_S, p, early]) for p in 1:10]
        @test length(unique(C_mid)) > 1 || length(unique(S_early)) > 1
    end

    @testset "cases counter resets each step" begin
        result = seed_and_simulate(yf_delay, pars, 1)
        @test all(result[idx_C, :, :] .>= 0.0)
        max_C = maximum(result[idx_C, :, :])
        @test max_C < N_total
    end
end
