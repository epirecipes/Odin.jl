using Test
using Odin

@testset "Additional models" begin

    @testset "Logistic growth ODE" begin
        gen = @odin begin
            deriv(N) = r * N * (1 - N / K)
            initial(N) = N0
            r = parameter(0.5)
            K = parameter(100.0)
            N0 = parameter(10.0)
        end

        pars = (r=0.5, K=100.0, N0=10.0)
        sys = dust_system_create(gen, pars; n_particles=1)
        dust_system_set_state_initial!(sys)

        @test dust_system_state(sys)[1, 1] ≈ 10.0

        times = collect(0.0:0.5:100.0)
        output = dust_system_simulate(sys, times)

        @test size(output) == (1, 1, length(times))

        N_vals = output[1, 1, :]
        # N should approach K
        @test N_vals[end] ≈ 100.0 atol=0.5
        # N should be monotonically increasing (since N0 < K)
        # Use relaxed tolerance near equilibrium where ODE solver has tiny fluctuations
        for i in 2:length(N_vals)
            @test N_vals[i] >= N_vals[i-1] - 1e-3
        end
    end

    @testset "Lotka-Volterra ODE" begin
        gen = @odin begin
            deriv(prey) = alpha * prey - beta * prey * predator
            deriv(predator) = delta * prey * predator - gamma_p * predator
            initial(prey) = prey0
            initial(predator) = pred0
            alpha = parameter(1.1)
            beta = parameter(0.4)
            delta = parameter(0.1)
            gamma_p = parameter(0.4)
            prey0 = parameter(10.0)
            pred0 = parameter(10.0)
        end

        pars = (alpha=1.1, beta=0.4, delta=0.1, gamma_p=0.4, prey0=10.0, pred0=10.0)
        sys = dust_system_create(gen, pars; n_particles=1)
        dust_system_set_state_initial!(sys)

        state = dust_system_state(sys)
        @test state[1, 1] ≈ 10.0  # prey
        @test state[2, 1] ≈ 10.0  # predator

        times = collect(0.0:0.1:50.0)
        output = dust_system_simulate(sys, times)

        @test size(output) == (2, 1, length(times))

        prey_vals = output[1, 1, :]
        pred_vals = output[2, 1, :]

        # Both populations should remain positive
        @test all(prey_vals .> 0)
        @test all(pred_vals .> 0)

        # Verify oscillations: prey should have at least one local max and min
        prey_peak = false
        prey_trough = false
        for i in 2:(length(prey_vals)-1)
            if prey_vals[i] > prey_vals[i-1] && prey_vals[i] > prey_vals[i+1]
                prey_peak = true
            end
            if prey_vals[i] < prey_vals[i-1] && prey_vals[i] < prey_vals[i+1]
                prey_trough = true
            end
        end
        @test prey_peak
        @test prey_trough
    end

    @testset "SEIR with vaccination (discrete stochastic)" begin
        gen = @odin begin
            p_SE = 1 - exp(-beta * I / N * dt)
            p_EI = 1 - exp(-sigma * dt)
            p_IR = 1 - exp(-gamma_r * dt)
            p_vax = 1 - exp(-vax_rate * dt)

            n_SE = Binomial(S, p_SE)
            n_EI = Binomial(E, p_EI)
            n_IR = Binomial(I, p_IR)
            n_vax = Binomial(S - n_SE, p_vax)

            update(S) = S - n_SE - n_vax
            update(E) = E + n_SE - n_EI
            update(I) = I + n_EI - n_IR
            update(R) = R + n_IR + n_vax

            initial(S) = N - I0
            initial(E) = 0
            initial(I) = I0
            initial(R) = 0

            N = parameter(10000)
            I0 = parameter(10)
            beta = parameter(0.5)
            sigma = parameter(0.2)
            gamma_r = parameter(0.1)
            vax_rate = parameter(0.01)
        end

        pars = (N=10000.0, I0=10.0, beta=0.5, sigma=0.2, gamma_r=0.1, vax_rate=0.01)
        sys = dust_system_create(gen, pars; n_particles=100, dt=1.0, seed=42)
        dust_system_set_state_initial!(sys)

        times = collect(0.0:1.0:100.0)
        output = dust_system_simulate(sys, times)

        @test size(output) == (4, 100, length(times))

        # Population conservation: S + E + I + R = N for all particles, all times
        for p in 1:100, t in 1:length(times)
            total = output[1, p, t] + output[2, p, t] + output[3, p, t] + output[4, p, t]
            @test total ≈ 10000.0 atol=0.1
        end

        # Initial conditions correct for all particles
        for p in 1:100
            @test output[1, p, 1] ≈ 9990.0 atol=0.1  # S = N - I0
            @test output[2, p, 1] ≈ 0.0 atol=0.1      # E = 0
            @test output[3, p, 1] ≈ 10.0 atol=0.1      # I = I0
            @test output[4, p, 1] ≈ 0.0 atol=0.1       # R = 0
        end

        # Stochasticity: particles should diverge over time
        final_I = output[3, :, end]
        final_R = output[4, :, end]
        @test length(unique(round.(final_R))) > 1
    end

    @testset "SIS model ODE" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N + gamma_r * I
            deriv(I) = beta * S * I / N - gamma_r * I
            initial(S) = N - I0
            initial(I) = I0
            beta = parameter(0.5)
            gamma_r = parameter(0.1)
            N = parameter(1000.0)
            I0 = parameter(10.0)
        end

        pars = (beta=0.5, gamma_r=0.1, N=1000.0, I0=10.0)
        sys = dust_system_create(gen, pars; n_particles=1)
        dust_system_set_state_initial!(sys)

        times = collect(0.0:1.0:500.0)
        output = dust_system_simulate(sys, times)

        @test size(output) == (2, 1, length(times))

        S_vals = output[1, 1, :]
        I_vals = output[2, 1, :]

        # Population conservation: S + I = N at all times
        for t in 1:length(times)
            @test S_vals[t] + I_vals[t] ≈ 1000.0 atol=1.0
        end

        # Endemic equilibrium: I* = N * (1 - gamma/beta)
        I_star = 1000.0 * (1 - 0.1 / 0.5)
        @test I_vals[end] ≈ I_star atol=5.0
    end

    @testset "Two-strain model ODE" begin
        gen = @odin begin
            deriv(S) = -beta1 * S * I1 / N - beta2 * S * I2 / N
            deriv(I1) = beta1 * S * I1 / N - gamma_r * I1
            deriv(I2) = beta2 * S * I2 / N - gamma_r * I2
            deriv(R1) = gamma_r * I1
            deriv(R2) = gamma_r * I2
            initial(S) = N - I10 - I20
            initial(I1) = I10
            initial(I2) = I20
            initial(R1) = 0.0
            initial(R2) = 0.0
            beta1 = parameter(0.5)
            beta2 = parameter(0.3)
            gamma_r = parameter(0.1)
            N = parameter(1000.0)
            I10 = parameter(5.0)
            I20 = parameter(5.0)
        end

        pars = (beta1=0.5, beta2=0.3, gamma_r=0.1, N=1000.0, I10=5.0, I20=5.0)
        sys = dust_system_create(gen, pars; n_particles=1)
        dust_system_set_state_initial!(sys)

        times = collect(0.0:0.5:200.0)
        output = dust_system_simulate(sys, times)

        @test size(output) == (5, 1, length(times))

        # Population conservation: S + I1 + I2 + R1 + R2 = N
        for t in 1:length(times)
            total = sum(output[i, 1, t] for i in 1:5)
            @test total ≈ 1000.0 atol=1.0
        end

        # Strain 1 (higher beta) should peak higher than strain 2
        I1_vals = output[2, 1, :]
        I2_vals = output[3, 1, :]
        @test maximum(I1_vals) > maximum(I2_vals)

        # Both strains should eventually decline
        @test I1_vals[end] < maximum(I1_vals)
        @test I2_vals[end] < maximum(I2_vals)
    end
end
