using Test
using Odin

@testset "Dust System" begin
    @testset "Create continuous system" begin
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
        sys = dust_system_create(gen, pars; n_particles=1)

        @test sys.n_particles == 1
        @test sys.n_state == 3

        dust_system_set_state_initial!(sys)
        state = dust_system_state(sys)
        @test state[1, 1] ≈ 990.0  # S = N - I0
        @test state[2, 1] ≈ 10.0   # I = I0
        @test state[3, 1] ≈ 0.0    # R = 0
    end

    @testset "Create discrete system" begin
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
        sys = dust_system_create(gen, pars; n_particles=5, dt=0.25, seed=42)

        @test sys.n_particles == 5
        @test sys.dt == 0.25

        dust_system_set_state_initial!(sys)
        state = dust_system_state(sys)
        # All particles start at same initial conditions
        for p in 1:5
            @test state[1, p] ≈ 990.0
            @test state[2, p] ≈ 10.0
            @test state[3, p] ≈ 0.0
        end
    end
end
