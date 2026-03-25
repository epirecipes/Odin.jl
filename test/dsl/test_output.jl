@testset "output() support" begin
    @testset "ODE model with scalar output expression" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            output(total) = S + I + R
            output(prevalence) = I / N
            beta = parameter(0.3)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        @test gen.model.has_output == true
        @test gen.model.is_continuous == true

        pars = (beta=0.3, gamma=0.1, I0=10.0, N=1000.0)
        sys = dust_system_create(gen, pars; dt=1.0)

        # n_output should be 2 (total + prevalence)
        @test sys.n_output == 2
        @test sys.output_names == [:total, :prevalence]

        # State names should be just S, I, R (3 state vars)
        @test sys.n_state == 3

        dust_system_set_state_initial!(sys)
        times = collect(0.0:1.0:100.0)
        result = dust_system_simulate(sys, times)

        # Result should have 5 rows: 3 state + 2 output
        @test size(result, 1) == 5

        # Check total conservation (S + I + R ≈ N)
        for ti in 1:length(times)
            @test result[4, 1, ti] ≈ 1000.0 rtol=1e-6  # total
            @test result[5, 1, ti] ≈ result[2, 1, ti] / 1000.0 rtol=1e-6  # prevalence
        end
    end

    @testset "ODE model with output(x) = true flag form" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            foi = beta * I / N
            output(foi) = true
            beta = parameter(0.3)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (beta=0.3, gamma=0.1, I0=10.0, N=1000.0)
        sys = dust_system_create(gen, pars; dt=1.0)

        @test sys.n_output == 1
        @test sys.output_names == [:foi]

        dust_system_set_state_initial!(sys)
        times = collect(0.0:1.0:50.0)
        result = dust_system_simulate(sys, times)

        # 3 state + 1 output = 4 rows
        @test size(result, 1) == 4

        # At t=0: foi = beta * I0 / N = 0.3 * 10 / 1000 = 0.003
        @test result[4, 1, 1] ≈ 0.003 rtol=1e-3
    end

    @testset "No output model has same rows as state" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.3)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (beta=0.3, gamma=0.1, I0=10.0, N=1000.0)
        sys = dust_system_create(gen, pars; dt=1.0)

        @test sys.n_output == 0
        @test sys.output_names == Symbol[]

        dust_system_set_state_initial!(sys)
        times = collect(0.0:1.0:10.0)
        result = dust_system_simulate(sys, times)

        # Only 3 state rows, no output
        @test size(result, 1) == 3
    end

    @testset "Discrete model with output (should work for discrete too)" begin
        # Although R's odin2 only allows output() for continuous models,
        # our implementation supports it for discrete models too
        gen = @odin begin
            update(S) = S - new_I
            update(I) = I + new_I - new_R
            update(R) = R + new_R
            new_I = Binomial(S, 1 - exp(-beta * I / N * dt))
            new_R = Binomial(I, 1 - exp(-gamma * dt))
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            output(total) = S + I + R
            beta = parameter(0.3)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (beta=0.3, gamma=0.1, I0=10.0, N=1000.0)
        sys = dust_system_create(gen, pars; dt=1.0, seed=42)

        @test sys.n_output == 1
        @test sys.output_names == [:total]

        dust_system_set_state_initial!(sys)
        times = collect(0.0:1.0:50.0)
        result = dust_system_simulate(sys, times)

        # 3 state + 1 output = 4 rows
        @test size(result, 1) == 4

        # Total should be conserved (S + I + R = N)
        for ti in 1:length(times)
            @test result[4, 1, ti] ≈ 1000.0 atol=0.1
        end
    end

    @testset "Output with array state variables" begin
        gen = @odin begin
            n_age = parameter(3)
            dim(S) = n_age
            dim(I) = n_age
            dim(R) = n_age

            deriv(S[i]) = -beta * S[i] * sum(I) / N
            deriv(I[i]) = beta * S[i] * sum(I) / N - gamma * I[i]
            deriv(R[i]) = gamma * I[i]
            initial(S[i]) = S0[i]
            initial(I[i]) = I0[i]
            initial(R[i]) = 0

            output(total_I) = sum(I)
            output(total_pop) = sum(S) + sum(I) + sum(R)

            dim(S0) = n_age
            dim(I0) = n_age
            S0[i] = parameter()
            I0[i] = parameter()
            beta = parameter(0.3)
            gamma = parameter(0.1)
            N = parameter(1000)
        end

        pars = (n_age=3, S0=[330.0, 330.0, 330.0],
                I0=[3.0, 3.0, 4.0], beta=0.3, gamma=0.1, N=1000.0)
        sys = dust_system_create(gen, pars; dt=1.0)

        # 3*3 = 9 state vars, 2 scalar outputs
        @test sys.n_state == 9
        @test sys.n_output == 2

        dust_system_set_state_initial!(sys)
        times = collect(0.0:1.0:50.0)
        result = dust_system_simulate(sys, times)

        @test size(result, 1) == 11  # 9 + 2

        # At t=0: total_I = 3+3+4 = 10, total_pop = 990+10+0 = 1000
        @test result[10, 1, 1] ≈ 10.0 rtol=1e-6
        @test result[11, 1, 1] ≈ 1000.0 rtol=1e-6

        # total_pop should be conserved
        for ti in 1:length(times)
            @test result[11, 1, ti] ≈ 1000.0 rtol=1e-4
        end
    end
end
