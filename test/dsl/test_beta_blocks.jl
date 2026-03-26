using Test
using Odin

@testset "beta blocks model" begin
    @testset "model compiles and runs" begin
        gen = @odin begin
            update(S) = S - new_inf
            update(I) = I + new_inf - new_rec
            initial(S) = N - I0
            initial(I) = I0
            initial(incidence, zero_every = 1) = 0
            update(incidence) = new_inf

            p_SI = 1 - exp(-beta_t * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            new_inf = S * p_SI
            new_rec = I * p_IR

            beta_times = parameter(rank = 1)
            beta_values = parameter(rank = 1)
            beta_t = interpolate(beta_times, beta_values, :constant)

            N = parameter()
            I0 = parameter()
            gamma = parameter()
        end

        pars = (
            beta_times = [0.0, 90.0, 180.0, 270.0],
            beta_values = [0.4, 0.2, 0.3, 0.5],
            N = 10000.0, I0 = 10.0, gamma = 0.1,
        )
        times = collect(0.0:1.0:365.0)
        result = simulate(gen, pars; times = times, dt = 1.0,
            seed = 1, n_particles = 1)

        @test size(result) == (3, 1, length(times))
        @test result[1, 1, 1] ≈ 9990.0   # S(0) = N - I0
        @test result[2, 1, 1] ≈ 10.0      # I(0) = I0
    end

    @testset "population conservation" begin
        gen = @odin begin
            update(S) = S - new_inf
            update(I) = I + new_inf - new_rec
            initial(S) = N - I0
            initial(I) = I0
            initial(incidence, zero_every = 1) = 0
            update(incidence) = new_inf

            p_SI = 1 - exp(-beta_t * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            new_inf = S * p_SI
            new_rec = I * p_IR

            beta_times = parameter(rank = 1)
            beta_values = parameter(rank = 1)
            beta_t = interpolate(beta_times, beta_values, :constant)

            N = parameter()
            I0 = parameter()
            gamma = parameter()
        end

        pars = (
            beta_times = [0.0, 90.0, 180.0, 270.0],
            beta_values = [0.4, 0.2, 0.3, 0.5],
            N = 10000.0, I0 = 10.0, gamma = 0.1,
        )
        times = collect(0.0:1.0:365.0)
        result = simulate(gen, pars; times = times, dt = 1.0,
            seed = 1, n_particles = 1)

        # S + I ≤ N always (recovered = N - S - I ≥ 0)
        # Note: incidence is a separate counter (zero_every resets it),
        # so conservation is S + I + (N - S₀ - I₀ accumulated recovery) = N
        # In this model, we only track S and I; total recovered = N - S - I
        for ti in 1:length(times)
            S = result[1, 1, ti]
            I = result[2, 1, ti]
            @test S >= -0.1          # S stays non-negative
            @test I >= -0.1          # I stays non-negative
            @test S + I <= 10000.0 + 0.1  # S + I ≤ N
        end
    end

    @testset "higher beta leads to more infections" begin
        gen = @odin begin
            update(S) = S - new_inf
            update(I) = I + new_inf - new_rec
            initial(S) = N - I0
            initial(I) = I0
            initial(incidence, zero_every = 1) = 0
            update(incidence) = new_inf

            p_SI = 1 - exp(-beta_t * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            new_inf = S * p_SI
            new_rec = I * p_IR

            beta_times = parameter(rank = 1)
            beta_values = parameter(rank = 1)
            beta_t = interpolate(beta_times, beta_values, :constant)

            N = parameter()
            I0 = parameter()
            gamma = parameter()
        end

        pars_low = (
            beta_times = [0.0], beta_values = [0.2],
            N = 10000.0, I0 = 10.0, gamma = 0.1,
        )
        pars_high = (
            beta_times = [0.0], beta_values = [0.8],
            N = 10000.0, I0 = 10.0, gamma = 0.1,
        )

        times = collect(0.0:1.0:100.0)
        res_low = simulate(gen, pars_low;
            times = times, dt = 1.0, seed = 1, n_particles = 1)
        res_high = simulate(gen, pars_high;
            times = times, dt = 1.0, seed = 1, n_particles = 1)

        # Cumulative incidence (sum daily incidence) should be higher for high beta
        cum_low = sum(res_low[3, 1, 2:end])
        cum_high = sum(res_high[3, 1, 2:end])
        @test cum_high > cum_low

        # S should deplete faster with high beta
        @test res_high[1, 1, end] < res_low[1, 1, end]
    end

    @testset "piecewise-constant interpolation is correct" begin
        gen = @odin begin
            update(S) = S - new_inf
            update(I) = I + new_inf - new_rec
            initial(S) = N - I0
            initial(I) = I0
            initial(incidence, zero_every = 1) = 0
            update(incidence) = new_inf

            p_SI = 1 - exp(-beta_t * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            new_inf = S * p_SI
            new_rec = I * p_IR

            beta_times = parameter(rank = 1)
            beta_values = parameter(rank = 1)
            beta_t = interpolate(beta_times, beta_values, :constant)

            N = parameter()
            I0 = parameter()
            gamma = parameter()
        end

        # Use two blocks: beta=0.5 for [0, 50), beta=0.0 for [50, ∞)
        # With beta=0, no new infections after t=50
        pars = (
            beta_times = [0.0, 50.0],
            beta_values = [0.5, 0.0],
            N = 10000.0, I0 = 10.0, gamma = 0.1,
        )
        times = collect(0.0:1.0:100.0)
        result = simulate(gen, pars; times = times, dt = 1.0,
            seed = 1, n_particles = 1)

        # After beta drops to 0 at t=50, incidence should be ~0
        # Allow a few steps for the transition
        for ti in 53:length(times)
            @test result[3, 1, ti] < 0.01  # near-zero incidence
        end

        # Before t=50, there should be positive incidence (epidemic underway)
        @test any(result[3, 1, 10:50] .> 0.1)
    end

    @testset "deterministic model is reproducible" begin
        gen = @odin begin
            update(S) = S - new_inf
            update(I) = I + new_inf - new_rec
            initial(S) = N - I0
            initial(I) = I0
            initial(incidence, zero_every = 1) = 0
            update(incidence) = new_inf

            p_SI = 1 - exp(-beta_t * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            new_inf = S * p_SI
            new_rec = I * p_IR

            beta_times = parameter(rank = 1)
            beta_values = parameter(rank = 1)
            beta_t = interpolate(beta_times, beta_values, :constant)

            N = parameter()
            I0 = parameter()
            gamma = parameter()
        end

        pars = (
            beta_times = [0.0, 90.0, 180.0, 270.0],
            beta_values = [0.4, 0.2, 0.3, 0.5],
            N = 10000.0, I0 = 10.0, gamma = 0.1,
        )
        times = collect(0.0:1.0:365.0)

        # Different seeds should give identical results (no stochasticity)
        r1 = simulate(gen, pars; times = times, dt = 1.0,
            seed = 1, n_particles = 1)
        r2 = simulate(gen, pars; times = times, dt = 1.0,
            seed = 99, n_particles = 1)

        @test r1 ≈ r2 atol = 1e-10
    end

    @testset "data comparison with particle filter" begin
        gen = @odin begin
            update(S) = S - new_inf
            update(I) = I + new_inf - new_rec
            initial(S) = N - I0
            initial(I) = I0
            initial(incidence, zero_every = 1) = 0
            update(incidence) = new_inf

            p_SI = 1 - exp(-beta_t * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            new_inf = S * p_SI
            new_rec = I * p_IR

            beta_times = parameter(rank = 1)
            beta_values = parameter(rank = 1)
            beta_t = interpolate(beta_times, beta_values, :constant)

            N = parameter()
            I0 = parameter()
            gamma = parameter()

            cases = data()
            cases ~ Poisson(incidence + 1e-6)
        end

        pars = (
            beta_times = [0.0, 50.0],
            beta_values = [0.5, 0.2],
            N = 10000.0, I0 = 10.0, gamma = 0.1,
        )

        data = ObservedData(
            [(time = Float64(t), cases = max(1.0, round(50.0 * exp(-0.03 * t))))
             for t in 1:100]
        )

        filter = Likelihood(gen, data; n_particles = 1, dt = 1.0, seed = 1)
        ll = loglik(filter, pars)

        @test isfinite(ll)
        @test ll < 0  # log-likelihood should be negative
    end
end
