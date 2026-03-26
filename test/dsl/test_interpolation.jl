@testset "interpolation support" begin
    @testset "constant interpolation — ODE" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = interpolate(beta_time, beta_value, :constant)
            beta_time = parameter(rank=1)
            beta_value = parameter(rank=1)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        # High beta early, low beta late
        pars = (beta_time=[0.0, 50.0], beta_value=[0.5, 0.05],
                gamma=0.1, I0=10.0, N=1000.0)
        sys = System(gen, pars; dt=1.0)
        reset!(sys)
        times = collect(0.0:1.0:100.0)
        result = simulate(sys, times)

        # Conservation
        for ti in 1:length(times)
            @test sum(result[:, 1, ti]) ≈ 1000.0 rtol=1e-4
        end

        # With high beta, epidemic should be well underway by t=50
        @test result[2, 1, 51] > 0.0  # I at t=50
        # Total recovered should be substantial
        @test result[3, 1, end] > 500.0
    end

    @testset "linear interpolation — ODE" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = interpolate(beta_time, beta_value, :linear)
            beta_time = parameter(rank=1)
            beta_value = parameter(rank=1)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (beta_time=[0.0, 50.0, 100.0], beta_value=[0.5, 0.1, 0.3],
                gamma=0.1, I0=10.0, N=1000.0)
        sys = System(gen, pars; dt=1.0)
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:100.0))

        # Conservation
        @test sum(result[:, 1, end]) ≈ 1000.0 rtol=1e-4
        # Epidemic should progress
        @test result[3, 1, end] > 100.0
    end

    @testset "spline interpolation — ODE" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = interpolate(beta_time, beta_value, :spline)
            beta_time = parameter(rank=1)
            beta_value = parameter(rank=1)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (beta_time=collect(0.0:25.0:100.0),
                beta_value=[0.5, 0.3, 0.1, 0.2, 0.4],
                gamma=0.1, I0=10.0, N=1000.0)
        sys = System(gen, pars; dt=1.0)
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:100.0))

        # Conservation
        @test sum(result[:, 1, end]) ≈ 1000.0 rtol=1e-4
    end

    @testset "constant interpolation — discrete stochastic" begin
        gen = @odin begin
            update(S) = S - n_SI
            update(I) = I + n_SI - n_IR
            update(R) = R + n_IR
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = interpolate(beta_time, beta_value, :constant)
            p_SI = 1 - exp(-beta * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            n_SI = Binomial(S, p_SI)
            n_IR = Binomial(I, p_IR)
            beta_time = parameter(rank=1)
            beta_value = parameter(rank=1)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (beta_time=[0.0, 50.0], beta_value=[0.5, 0.05],
                gamma=0.1, I0=10.0, N=1000.0)
        sys = System(gen, pars; dt=1.0, seed=42)
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:100.0))

        # Conservation (stochastic SIR conserves N)
        for ti in 1:size(result, 3)
            @test sum(result[:, 1, ti]) ≈ 1000.0 atol=0.1
        end
    end

    @testset "multiple interpolated variables" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N + delta * R
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I - delta * R
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = interpolate(beta_t, beta_v, :constant)
            gamma = interpolate(gamma_t, gamma_v, :linear)
            beta_t = parameter(rank=1)
            beta_v = parameter(rank=1)
            gamma_t = parameter(rank=1)
            gamma_v = parameter(rank=1)
            delta = parameter(0.01)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (
            beta_t=[0.0, 50.0], beta_v=[0.4, 0.2],
            gamma_t=[0.0, 100.0], gamma_v=[0.05, 0.15],
            delta=0.01, I0=10.0, N=1000.0,
        )
        sys = System(gen, pars; dt=1.0)
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:200.0))

        # Conservation (SIRS conserves N)
        @test sum(result[:, 1, 1]) ≈ 1000.0 rtol=1e-6
        @test sum(result[:, 1, end]) ≈ 1000.0 rtol=1e-4
    end

    @testset "interpolation with particle filter" begin
        gen = @odin begin
            update(S) = S - n_SI
            update(I) = I + n_SI - n_IR
            update(R) = R + n_IR
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            initial(incidence, zero_every=1) = 0
            update(incidence) = incidence + n_SI
            beta = interpolate(beta_time, beta_value, :constant)
            p_SI = 1 - exp(-beta * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            n_SI = Binomial(S, p_SI)
            n_IR = Binomial(I, p_IR)
            cases ~ Poisson(incidence + 1e-6)
            beta_time = parameter(rank=1)
            beta_value = parameter(rank=1)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (beta_time=[0.0, 30.0], beta_value=[0.5, 0.1],
                gamma=0.1, I0=10.0, N=1000.0)

        # Create some fake data
        data = ObservedData(
            [(time=Float64(t), cases=max(1, round(Int, 50 * exp(-0.05 * t)))) for t in 1:50]
        )

        filter = Likelihood(gen, data; n_particles=10, dt=1.0, seed=42)
        ll = loglik(filter, pars)
        @test isfinite(ll)
        @test ll < 0  # negative log-likelihood
    end
end
