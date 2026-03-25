using Test
using Odin

@testset "OROV vector-borne model with relapse" begin

    orov = @odin begin
        # Erlang chain parameters
        k_delay = parameter(12)
        delay_rate = k_delay / delay_tau
        delay_tau = parameter(12.0)

        # === Erlang chain: delayed total human infectiousness ===
        dim(D_Ih) = k_delay
        I_h_total = I_h_asym + I_h_sym0 + I_h_sym1 + I_h_sym2
        deriv(D_Ih[1]) = (I_h_total - D_Ih[1]) * delay_rate
        deriv(D_Ih[2:k_delay]) = (D_Ih[i - 1] - D_Ih[i]) * delay_rate
        initial(D_Ih[1:k_delay]) = I_init_h

        # === Erlang chain: delayed susceptible vectors ===
        dim(D_Sv) = k_delay
        deriv(D_Sv[1]) = (S_v - D_Sv[1]) * delay_rate
        deriv(D_Sv[2:k_delay]) = (D_Sv[i - 1] - D_Sv[i]) * delay_rate
        initial(D_Sv[1:k_delay]) = 1.0 - I_init_v

        # === Forces of infection ===
        lambda_h = m * a * b_h * I_v
        lambda_v = a * b_v * D_Ih[k_delay]

        # === Human dynamics (proportions) ===
        deriv(S_h) = -lambda_h * S_h
        deriv(E_h) = lambda_h * S_h - gamma * E_h
        deriv(I_h_asym) = (1.0 - theta) * gamma * E_h - sigma * I_h_asym
        deriv(I_h_sym0) = theta * gamma * E_h - sigma * I_h_sym0
        deriv(I_h_sym1) = pi_relapse * sigma * I_h_sym0 - psi * I_h_sym1
        deriv(R_h_temp) = psi * I_h_sym1 - omega * R_h_temp
        deriv(I_h_sym2) = omega * R_h_temp - epsilon * I_h_sym2
        deriv(R_h) = sigma * I_h_asym + (1.0 - pi_relapse) * sigma * I_h_sym0 + epsilon * I_h_sym2

        # === Vector dynamics (proportions) ===
        deriv(S_v) = mu * (1.0 - S_v) - lambda_v * S_v
        deriv(E_v) = lambda_v * D_Sv[k_delay] - (mu + kappa) * E_v
        deriv(I_v) = kappa * E_v - mu * I_v

        # === Outputs ===
        output(prevalence_h) = I_h_total
        output(incidence_h) = lambda_h * S_h
        output(prevalence_v) = I_v
        output(human_pop) = S_h + E_h + I_h_asym + I_h_sym0 + I_h_sym1 + R_h_temp + I_h_sym2 + R_h

        # === Initial conditions ===
        initial(S_h) = 1.0 - I_init_h
        initial(E_h) = 0.0
        initial(I_h_asym) = (1.0 - theta) * I_init_h
        initial(I_h_sym0) = theta * I_init_h
        initial(I_h_sym1) = 0.0
        initial(R_h_temp) = 0.0
        initial(I_h_sym2) = 0.0
        initial(R_h) = 0.0
        initial(S_v) = 1.0 - I_init_v
        initial(E_v) = 0.0
        initial(I_v) = I_init_v

        # === Parameters ===
        gamma = parameter(0.167)
        theta = parameter(0.85)
        pi_relapse = parameter(0.5)
        sigma = parameter(0.2)
        psi = parameter(1.0)
        epsilon = parameter(1.0)
        kappa = parameter(0.125)
        omega = parameter(0.2)
        mu = parameter(0.03)
        m = parameter(20.0)
        a = parameter(0.3)
        b_h = parameter(0.2)
        b_v = parameter(0.05)
        I_init_h = parameter(0.001)
        I_init_v = parameter(0.0001)
    end

    pars = (
        k_delay = 12.0,
        gamma = 0.167,
        theta = 0.85,
        pi_relapse = 0.5,
        sigma = 0.2,
        psi = 1.0,
        epsilon = 1.0,
        kappa = 0.125,
        omega = 0.2,
        mu = 0.03,
        m = 20.0,
        a = 0.3,
        b_h = 0.2,
        b_v = 0.05,
        I_init_h = 0.001,
        I_init_v = 0.0001,
        delay_tau = 12.0,
    )

    times = collect(0.0:1.0:365.0)

    # State layout: D_Ih[1:12], D_Sv[1:12], S_h, E_h, I_h_asym, I_h_sym0,
    #   I_h_sym1, R_h_temp, I_h_sym2, R_h, S_v, E_v, I_v, outputs(4)
    idx_Sh = 25
    idx_Eh = 26
    idx_Iha = 27
    idx_Ihs0 = 28
    idx_Ihs1 = 29
    idx_Rht = 30
    idx_Ihs2 = 31
    idx_Rh = 32
    idx_Sv = 33
    idx_Ev = 34
    idx_Iv = 35
    idx_prev_h = 36
    idx_inc_h = 37
    idx_prev_v = 38
    idx_pop_h = 39

    @testset "compiles and runs" begin
        sys = dust_system_create(orov, pars; n_particles = 1)
        dust_system_set_state_initial!(sys)
        result = dust_system_simulate(sys, times)

        # 12 + 12 delay chain + 8 human + 3 vector + 4 outputs = 39
        @test size(result, 1) == 39
        @test size(result, 2) == 1
        @test size(result, 3) == length(times)
        @test all(isfinite, result)
    end

    @testset "human population conserved" begin
        sys = dust_system_create(orov, pars; n_particles = 1)
        dust_system_set_state_initial!(sys)
        result = dust_system_simulate(sys, times)

        for t in 1:length(times)
            pop = result[idx_Sh, 1, t] + result[idx_Eh, 1, t] +
                  result[idx_Iha, 1, t] + result[idx_Ihs0, 1, t] +
                  result[idx_Ihs1, 1, t] + result[idx_Rht, 1, t] +
                  result[idx_Ihs2, 1, t] + result[idx_Rh, 1, t]
            @test pop ≈ 1.0 atol = 1e-6
        end

        # Also check the output variable
        @test all(abs.(result[idx_pop_h, 1, :] .- 1.0) .< 1e-6)
    end

    @testset "omega=0 means no relapse" begin
        pars_nr = merge(pars, (omega = 0.0,))
        sys = dust_system_create(orov, pars_nr; n_particles = 1)
        dust_system_set_state_initial!(sys)
        result = dust_system_simulate(sys, times)

        # I_h_sym2 should stay at zero (initial is 0, and omega=0 blocks inflow)
        @test all(abs.(result[idx_Ihs2, 1, :]) .< 1e-10)
    end

    @testset "epidemic peak occurs" begin
        sys = dust_system_create(orov, pars; n_particles = 1)
        dust_system_set_state_initial!(sys)
        result = dust_system_simulate(sys, times)

        prevalence = result[idx_prev_h, 1, :]
        peak_val = maximum(prevalence)
        peak_idx = argmax(prevalence)

        # Peak should be substantially above initial prevalence
        @test peak_val > pars.I_init_h * 5
        # Peak should not be at the very start or very end
        @test peak_idx > 5
        @test peak_idx < length(times) - 5
    end

    @testset "non-negative compartments" begin
        sys = dust_system_create(orov, pars; n_particles = 1)
        dust_system_set_state_initial!(sys)
        result = dust_system_simulate(sys, times)

        for idx in [idx_Sh, idx_Eh, idx_Iha, idx_Ihs0, idx_Ihs1,
                    idx_Rht, idx_Ihs2, idx_Rh, idx_Sv, idx_Ev, idx_Iv]
            @test all(result[idx, 1, :] .>= -1e-10)
        end
    end

    @testset "higher relapse rate increases I_h_sym2 peak" begin
        pars_low = merge(pars, (omega = 0.1,))
        sys_low = dust_system_create(orov, pars_low; n_particles = 1)
        dust_system_set_state_initial!(sys_low)
        res_low = dust_system_simulate(sys_low, times)

        pars_high = merge(pars, (omega = 0.5,))
        sys_high = dust_system_create(orov, pars_high; n_particles = 1)
        dust_system_set_state_initial!(sys_high)
        res_high = dust_system_simulate(sys_high, times)

        peak_low = maximum(res_low[idx_Ihs2, 1, :])
        peak_high = maximum(res_high[idx_Ihs2, 1, :])
        @test peak_high > peak_low
    end
end
