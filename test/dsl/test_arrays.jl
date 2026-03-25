@testset "Array models" begin

    @testset "1D array ODE - age-structured SIR" begin
        sir_age = @odin begin
            n_age = parameter(3)
            dim(S) = n_age
            dim(I) = n_age
            dim(R) = n_age
            dim(beta) = n_age
            dim(S0) = n_age
            dim(I0) = n_age

            deriv(S[i]) = -beta[i] * S[i] * total_I / N
            deriv(I[i]) = beta[i] * S[i] * total_I / N - gamma * I[i]
            deriv(R[i]) = gamma * I[i]

            total_I = sum(I)

            initial(S[i]) = S0[i]
            initial(I[i]) = I0[i]
            initial(R[i]) = 0

            S0 = parameter()
            I0 = parameter()
            beta = parameter()
            gamma = parameter(0.1)
            N = parameter(1000)
        end

        pars = (n_age=3.0, S0=[490.0, 300.0, 200.0], I0=[5.0, 3.0, 2.0],
                beta=[0.3, 0.2, 0.15], gamma=0.1, N=1000.0)

        # Dynamic n_state
        sys = Odin.dust_system_create(sir_age, pars; seed=42)
        @test sys.n_state == 9
        @test length(sys.state_names) == 9
        @test sys.state_names[1] == Symbol("S[1]")
        @test sys.state_names[4] == Symbol("I[1]")

        # Initial conditions
        Odin.dust_system_set_state_initial!(sys)
        s0 = Odin.dust_system_state(sys)
        @test s0[1:3, 1] ≈ [490.0, 300.0, 200.0]
        @test s0[4:6, 1] ≈ [5.0, 3.0, 2.0]
        @test s0[7:9, 1] ≈ [0.0, 0.0, 0.0]

        # Simulate and check conservation
        times = collect(0.0:10.0:100.0)
        result = Odin.dust_system_simulate(sir_age, pars; times=times, seed=42)
        for t in 1:length(times)
            @test sum(result[:, 1, t]) ≈ 1000.0 atol=1e-6
        end

        # S should decrease, R should increase
        @test all(result[1:3, 1, end] .< result[1:3, 1, 1])
        @test all(result[7:9, 1, end] .> result[7:9, 1, 1])
    end

    @testset "1D array discrete stochastic" begin
        sir_stoch = @odin begin
            n_age = parameter(2)
            dim(S) = n_age
            dim(I) = n_age
            dim(R) = n_age
            dim(beta) = n_age
            dim(S0) = n_age
            dim(I0) = n_age

            update(S[i]) = S[i] - n_SI[i]
            update(I[i]) = I[i] + n_SI[i] - n_IR[i]
            update(R[i]) = R[i] + n_IR[i]

            total_I = sum(I)
            dim(p_SI) = n_age
            dim(n_SI) = n_age
            dim(p_IR) = n_age
            dim(n_IR) = n_age

            p_SI[i] = 1 - exp(-beta[i] * total_I / N * dt)
            p_IR[i] = 1 - exp(-gamma * dt)
            n_SI[i] = Binomial(S[i], p_SI[i])
            n_IR[i] = Binomial(I[i], p_IR[i])

            initial(S[i]) = S0[i]
            initial(I[i]) = I0[i]
            initial(R[i]) = 0

            S0 = parameter()
            I0 = parameter()
            beta = parameter()
            gamma = parameter(0.1)
            N = parameter(1000)
        end

        pars = (n_age=2.0, S0=[600.0, 390.0], I0=[5.0, 5.0],
                beta=[0.3, 0.2], gamma=0.1, N=1000.0)
        times = collect(0.0:5.0:50.0)
        result = Odin.dust_system_simulate(sir_stoch, pars; times=times, dt=1.0,
                                           seed=42, n_particles=5)
        @test size(result) == (6, 5, length(times))

        # Conservation for each particle
        for p in 1:5, t in 1:length(times)
            @test sum(result[:, p, t]) ≈ 1000.0
        end

        # Stochastic → integer valued
        @test all(result .== round.(result))
    end

    @testset "Array intermediate with reduction" begin
        # Model with array intermediates and sum reduction
        model = @odin begin
            n = parameter(4)
            dim(x) = n
            dim(dx) = n

            deriv(x[i]) = dx[i]
            dx[i] = -rate * x[i]
            initial(x[i]) = x0[i]

            total = sum(x)

            dim(x0) = n
            x0 = parameter()
            rate = parameter(0.1)
        end

        pars = (n=4.0, x0=[10.0, 20.0, 30.0, 40.0], rate=0.1)
        result = Odin.dust_system_simulate(model, pars; times=[0.0, 10.0], seed=1)
        # Each x[i] should decay exponentially: x0[i] * exp(-k*t)
        for i in 1:4
            expected = pars.x0[i] * exp(-0.1 * 10.0)
            @test result[i, 1, 2] ≈ expected rtol=1e-4
        end
    end

    @testset "Literal dimension (no parameter)" begin
        # Dim can be a literal number
        model = @odin begin
            dim(x) = 3
            deriv(x[i]) = -x[i]
            initial(x[i]) = i * 10.0
        end

        pars = NamedTuple()
        sys = Odin.dust_system_create(model, pars; seed=1)
        @test sys.n_state == 3
        Odin.dust_system_set_state_initial!(sys)
        s = Odin.dust_system_state(sys)
        @test s[:, 1] ≈ [10.0, 20.0, 30.0]
    end

    @testset "Mixed scalar and array state" begin
        model = @odin begin
            n = parameter(2)
            dim(S) = n
            dim(S0) = n

            deriv(S[i]) = -gamma * S[i]
            deriv(total) = -gamma * total
            initial(S[i]) = S0[i]
            initial(total) = sum(S0)

            S0 = parameter()
            gamma = parameter(0.1)
        end

        pars = (n=2.0, S0=[100.0, 200.0], gamma=0.1)
        result = Odin.dust_system_simulate(model, pars; times=[0.0, 10.0], seed=1)
        # S[1] + S[2] should equal total at all times
        @test result[1, 1, 2] + result[2, 1, 2] ≈ result[3, 1, 2] rtol=1e-4
    end

    @testset "_odin_n_state and _odin_state_names" begin
        model = @odin begin
            n = parameter(5)
            dim(x) = n
            deriv(x[i]) = -x[i]
            initial(x[i]) = 1.0
        end

        # Check that n_state is computed from pars
        pars3 = (n=3.0, dt=1.0)
        pars5 = (n=5.0, dt=1.0)
        @test Odin._odin_n_state(model.model, pars3) == 3
        @test Odin._odin_n_state(model.model, pars5) == 5
        @test length(Odin._odin_state_names(model.model, pars3)) == 3
        @test length(Odin._odin_state_names(model.model, pars5)) == 5
    end
end
