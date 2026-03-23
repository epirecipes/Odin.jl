using Test
using Odin
using Statistics
using Random

@testset "SDE Models" begin
    @testset "SDE SIR model compiles and runs" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            diffusion(S) = sigma_S * sqrt(abs(beta * S * I / N))
            diffusion(I) = sigma_I * sqrt(abs(beta * S * I / N + gamma * I))
            diffusion(R) = sigma_R * sqrt(abs(gamma * I))
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0.0
            beta = parameter(0.5)
            gamma = parameter(0.1)
            sigma_S = parameter(0.1)
            sigma_I = parameter(0.1)
            sigma_R = parameter(0.1)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        @test gen isa Odin.DustSystemGenerator
        @test gen.model.is_continuous == true
        @test gen.model.is_sde == true

        pars = (beta=0.5, gamma=0.1, sigma_S=0.1, sigma_I=0.1, sigma_R=0.1,
                I0=10.0, N=1000.0)
        sys = dust_system_create(gen, pars; dt=0.01, seed=42)
        dust_system_set_state_initial!(sys)

        times = collect(0.0:1.0:50.0)
        output = dust_system_simulate(sys, times)

        @test size(output) == (3, 1, length(times))
        # Initial conditions
        @test output[1, 1, 1] ≈ 990.0 atol=0.1
        @test output[2, 1, 1] ≈ 10.0 atol=0.1
        @test output[3, 1, 1] ≈ 0.0 atol=0.1
        # Values should be finite
        @test all(isfinite, output)
    end

    @testset "Euler-Maruyama produces stochastic trajectories" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            diffusion(S) = sigma * sqrt(abs(beta * S * I / N))
            diffusion(I) = sigma * sqrt(abs(beta * S * I / N + gamma * I))
            diffusion(R) = sigma * sqrt(abs(gamma * I))
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0.0
            beta = parameter(0.5)
            gamma = parameter(0.1)
            sigma = parameter(0.5)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        pars = (beta=0.5, gamma=0.1, sigma=0.5, I0=10.0, N=1000.0)
        n_particles = 10
        sys = dust_system_create(gen, pars; n_particles=n_particles, dt=0.01, seed=123)
        dust_system_set_state_initial!(sys)

        times = collect(0.0:1.0:30.0)
        output = dust_system_simulate(sys, times; solver=:euler_maruyama)

        @test size(output) == (3, n_particles, length(times))

        # Different particles should have different trajectories
        # Compare I values at time 15 across particles
        I_mid = output[2, :, 16]  # time index 16 = time 15
        @test length(unique(I_mid)) > 1  # not all identical
        @test std(I_mid) > 0.0  # there is variation
    end

    @testset "Zero noise reduces to ODE" begin
        # SDE model with σ=0
        sde_gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            diffusion(S) = sigma * sqrt(abs(beta * S * I / N))
            diffusion(I) = sigma * sqrt(abs(beta * S * I / N + gamma * I))
            diffusion(R) = sigma * sqrt(abs(gamma * I))
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0.0
            beta = parameter(0.5)
            gamma = parameter(0.1)
            sigma = parameter(0.0)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        # ODE model (no diffusion)
        ode_gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0.0
            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        pars_sde = (beta=0.5, gamma=0.1, sigma=0.0, I0=10.0, N=1000.0)
        pars_ode = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)

        # SDE with σ=0, very small dt
        sys_sde = dust_system_create(sde_gen, pars_sde; dt=0.001, seed=42)
        dust_system_set_state_initial!(sys_sde)
        times = collect(0.0:1.0:20.0)
        out_sde = dust_system_simulate(sys_sde, times; solver=:euler_maruyama)

        # ODE with DP5
        sys_ode = dust_system_create(ode_gen, pars_ode)
        dust_system_set_state_initial!(sys_ode)
        out_ode = dust_system_simulate(sys_ode, times; solver=:dp5)

        # Should be close (SDE with fixed dt = Euler, ODE with adaptive = DP5)
        # Use moderate tolerance because Euler fixed-step vs adaptive DP5
        for ti in 1:length(times)
            for j in 1:3
                @test out_sde[j, 1, ti] ≈ out_ode[j, 1, ti] rtol=0.05
            end
        end
    end

    @testset "Milstein solver runs correctly" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            diffusion(S) = sigma * sqrt(abs(beta * S * I / N))
            diffusion(I) = sigma * sqrt(abs(beta * S * I / N + gamma * I))
            diffusion(R) = sigma * sqrt(abs(gamma * I))
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0.0
            beta = parameter(0.5)
            gamma = parameter(0.1)
            sigma = parameter(0.1)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        pars = (beta=0.5, gamma=0.1, sigma=0.1, I0=10.0, N=1000.0)
        sys = dust_system_create(gen, pars; dt=0.01, seed=42)
        dust_system_set_state_initial!(sys)

        times = collect(0.0:1.0:30.0)
        output = dust_system_simulate(sys, times; solver=:milstein)

        @test size(output) == (3, 1, length(times))
        @test all(isfinite, output)
        # Initial conditions correct
        @test output[1, 1, 1] ≈ 990.0 atol=0.1
        @test output[2, 1, 1] ≈ 10.0 atol=0.1
    end

    @testset "Weak convergence: mean SDE ≈ ODE" begin
        sde_gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            diffusion(S) = sigma * sqrt(abs(beta * S * I / N))
            diffusion(I) = sigma * sqrt(abs(beta * S * I / N + gamma * I))
            diffusion(R) = sigma * sqrt(abs(gamma * I))
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0.0
            beta = parameter(0.5)
            gamma = parameter(0.1)
            sigma = parameter(0.05)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        ode_gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0.0
            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        pars_sde = (beta=0.5, gamma=0.1, sigma=0.05, I0=10.0, N=1000.0)
        pars_ode = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)

        # Many particles
        n_particles = 200
        sys_sde = dust_system_create(sde_gen, pars_sde; n_particles=n_particles, dt=0.01, seed=1234)
        dust_system_set_state_initial!(sys_sde)
        times = collect(0.0:2.0:20.0)
        out_sde = dust_system_simulate(sys_sde, times; solver=:euler_maruyama)

        sys_ode = dust_system_create(ode_gen, pars_ode)
        dust_system_set_state_initial!(sys_ode)
        out_ode = dust_system_simulate(sys_ode, times; solver=:dp5)

        # Mean of SDE particles should approximate ODE
        for ti in 1:length(times)
            for j in 1:3
                sde_mean = mean(out_sde[j, :, ti])
                ode_val = out_ode[j, 1, ti]
                # Tolerance is relative to scale
                scale = max(abs(ode_val), 10.0)
                @test abs(sde_mean - ode_val) / scale < 0.15
            end
        end
    end

    @testset "Variance scales with noise magnitude" begin
        gen = @odin begin
            deriv(X) = -alpha * X
            diffusion(X) = sigma * X
            initial(X) = 100.0
            alpha = parameter(0.1)
            sigma = parameter(0.1)
        end

        times = collect(0.0:0.5:10.0)
        n_particles = 500

        # Low noise
        pars_low = (alpha=0.1, sigma=0.05)
        sys_low = dust_system_create(gen, pars_low; n_particles=n_particles, dt=0.01, seed=42)
        dust_system_set_state_initial!(sys_low)
        out_low = dust_system_simulate(sys_low, times; solver=:euler_maruyama)
        var_low = var(out_low[1, :, end])

        # High noise
        pars_high = (alpha=0.1, sigma=0.2)
        sys_high = dust_system_create(gen, pars_high; n_particles=n_particles, dt=0.01, seed=42)
        dust_system_set_state_initial!(sys_high)
        out_high = dust_system_simulate(sys_high, times; solver=:euler_maruyama)
        var_high = var(out_high[1, :, end])

        # Higher noise should produce higher variance
        @test var_high > var_low
        # Variance should roughly scale with σ² (4x noise → ~16x variance)
        ratio = var_high / max(var_low, 1e-10)
        @test ratio > 2.0  # conservative lower bound
    end

    @testset "Multi-particle SDE simulation" begin
        gen = @odin begin
            deriv(X) = mu * X
            diffusion(X) = sigma * X
            initial(X) = X0
            mu = parameter(0.05)
            sigma = parameter(0.2)
            X0 = parameter(1.0)
        end

        pars = (mu=0.05, sigma=0.2, X0=1.0)
        n_particles = 50
        sys = dust_system_create(gen, pars; n_particles=n_particles, dt=0.01, seed=999)
        dust_system_set_state_initial!(sys)

        times = collect(0.0:0.5:5.0)
        output = dust_system_simulate(sys, times; solver=:euler_maruyama)

        @test size(output) == (1, n_particles, length(times))
        # All particles start at same value
        @test all(output[1, :, 1] .≈ 1.0)
        # Particles diverge over time
        @test std(output[1, :, end]) > 0.0
        # All values should be finite
        @test all(isfinite, output)
    end

    @testset "Milstein with zero noise matches Euler-Maruyama" begin
        gen = @odin begin
            deriv(X) = -alpha * X
            diffusion(X) = sigma * X
            initial(X) = 100.0
            alpha = parameter(0.1)
            sigma = parameter(0.0)
        end

        pars = (alpha=0.1, sigma=0.0)
        times = collect(0.0:1.0:10.0)

        sys_em = dust_system_create(gen, pars; dt=0.01, seed=42)
        dust_system_set_state_initial!(sys_em)
        out_em = dust_system_simulate(sys_em, times; solver=:euler_maruyama)

        sys_mil = dust_system_create(gen, pars; dt=0.01, seed=42)
        dust_system_set_state_initial!(sys_mil)
        out_mil = dust_system_simulate(sys_mil, times; solver=:milstein)

        # With zero noise, both methods produce identical results
        for ti in 1:length(times)
            @test out_em[1, 1, ti] ≈ out_mil[1, 1, ti] atol=1e-10
        end
    end

    @testset "SDE workspace reuse" begin
        # Test that SDEWorkspace can be reused without allocation issues
        gen = @odin begin
            deriv(X) = -X
            diffusion(X) = 0.1
            initial(X) = 1.0
        end

        pars = NamedTuple()
        sys = dust_system_create(gen, pars; dt=0.01, seed=42)
        dust_system_set_state_initial!(sys)

        times = collect(0.0:1.0:5.0)
        out1 = dust_system_simulate(sys, times; solver=:euler_maruyama)

        # Simulate again (workspace should be reused)
        dust_system_set_state_initial!(sys)
        sys.time = 0.0
        out2 = dust_system_simulate(sys, times; solver=:euler_maruyama)

        @test size(out1) == size(out2)
        @test all(isfinite, out1)
        @test all(isfinite, out2)
    end

    @testset "SDE validation: diffusion without deriv errors" begin
        # diffusion without deriv should error at parse/classify time
        block = quote
            diffusion(X) = 0.1
            initial(X) = 1.0
        end
        exprs = Odin.parse_odin_block(block)
        @test_throws ErrorException Odin.classify_variables(exprs)
    end

    @testset "Standalone SDE solver API" begin
        n = 2
        ws = Odin.SDEWorkspace(n)
        @test ws.n == n

        # Simple linear SDE: dX = -X dt + σ dW
        rhs!(du, u, p, t) = begin du[1] = -u[1]; du[2] = -u[2]; nothing end
        diff!(σ, u, p, t) = begin σ[1] = 0.1; σ[2] = 0.2; nothing end
        u0 = [1.0, 2.0]
        times = collect(0.0:0.1:1.0)
        rng = Random.Xoshiro(42)

        result = sde_solve!(rhs!, diff!, u0, (0.0, 1.0), nothing, 0.01, times;
                            ws=ws, rng=rng, method=:euler_maruyama)
        @test result isa Odin.SDEResult
        @test size(result.u, 1) == n
        @test size(result.u, 2) == length(times)
        @test all(isfinite, result.u)
    end
end
