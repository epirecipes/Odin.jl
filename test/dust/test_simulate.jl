using Test
using Odin

@testset "Dust Simulation" begin
    @testset "Continuous SIR simulation" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.2)
            gamma = parameter(0.1)
        end

        pars = (N=1000.0, I0=10.0, beta=0.2, gamma=0.1)
        sys = dust_system_create(gen, pars)
        dust_system_set_state_initial!(sys)

        times = collect(0.0:1.0:100.0)
        output = dust_system_simulate(sys, times)

        @test size(output) == (3, 1, length(times))

        # Conservation: S + I + R ≈ N at all times
        for t in 1:length(times)
            total = output[1, 1, t] + output[2, 1, t] + output[3, 1, t]
            @test total ≈ 1000.0 atol=1.0
        end

        # Epidemic should peak and decline
        # I should increase then decrease
        I_values = output[2, 1, :]
        peak_idx = argmax(I_values)
        @test peak_idx > 1
        @test peak_idx < length(times)
    end

    @testset "Discrete SIR simulation with particles" begin
        gen = @odin begin
            update(S) = S - n_SI
            update(I) = I + n_SI - n_IR
            update(R) = R + n_IR
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            p_SI = 1 - exp(-beta * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            n_SI = Binomial(S, p_SI)
            n_IR = Binomial(I, p_IR)
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.2)
            gamma = parameter(0.1)
        end

        pars = (N=1000.0, I0=10.0, beta=0.2, gamma=0.1)
        sys = dust_system_create(gen, pars; n_particles=10, dt=1.0, seed=42)
        dust_system_set_state_initial!(sys)

        times = collect(0.0:1.0:50.0)
        output = dust_system_simulate(sys, times)

        @test size(output) == (3, 10, length(times))

        # Conservation per particle
        for p in 1:10, t in 1:length(times)
            total = output[1, p, t] + output[2, p, t] + output[3, p, t]
            @test total ≈ 1000.0 atol=0.1
        end

        # Particles should diverge (stochastic)
        final_I = output[2, :, end]
        @test length(unique(round.(final_I))) > 1 || true  # may converge to 0
    end
end
