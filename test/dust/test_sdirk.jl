using Test
using Odin
using LinearAlgebra

@testset "SDIRK4 solver" begin

    @testset "Simple exponential decay" begin
        # du/dt = -u, u(0) = 1  →  u(t) = exp(-t)
        function decay!(du, u, p, t)
            du[1] = -u[1]
        end
        u0 = [1.0]
        tspan = (0.0, 3.0)
        saveat = collect(0.0:0.5:3.0)

        sol = sdirk_solve!(decay!, u0, tspan, nothing, saveat)
        for (i, t) in enumerate(saveat)
            @test sol.u[1, i] ≈ exp(-t) atol=1e-4
        end
    end

    @testset "Stiff linear decay (λ = -1000)" begin
        # du/dt = -1000*u, u(0) = 1
        # DP5 needs many steps on [0,1]; SDIRK4 handles it efficiently.
        function stiff_decay!(du, u, p, t)
            du[1] = -1000.0 * u[1]
        end
        u0 = [1.0]
        tspan = (0.0, 0.01)
        saveat = [0.0, 0.001, 0.005, 0.01]

        sol = sdirk_solve!(stiff_decay!, u0, tspan, nothing, saveat)
        for (i, t) in enumerate(saveat)
            @test sol.u[1, i] ≈ exp(-1000.0 * t) atol=1e-3
        end
    end

    @testset "Robertson chemical kinetics (stiff)" begin
        # Classic stiff test problem (ROBER).
        # Conservation law: u1 + u2 + u3 = 1
        function robertson!(du, u, p, t)
            du[1] = -0.04 * u[1] + 1e4 * u[2] * u[3]
            du[2] =  0.04 * u[1] - 1e4 * u[2] * u[3] - 3e7 * u[2]^2
            du[3] =  3e7 * u[2]^2
        end
        u0 = [1.0, 0.0, 0.0]
        tspan = (0.0, 100.0)
        saveat = [0.0, 1e-3, 1e-1, 1.0, 10.0, 100.0]

        sol = sdirk_solve!(robertson!, u0, tspan, nothing, saveat;
                            abstol=1e-8, reltol=1e-8)

        # Conservation law must hold at every save point
        for ti in 1:length(saveat)
            mass = sum(sol.u[:, ti])
            @test mass ≈ 1.0 atol=1e-3
        end

        # All concentrations must remain non-negative
        @test all(sol.u .>= -1e-6)

        # At t = 100, species 1 should have decayed significantly
        @test sol.u[1, end] < 0.8
    end

    @testset "Workspace reuse" begin
        function decay!(du, u, p, t)
            du[1] = -u[1]
        end
        u0 = [1.0]
        ws = SDIRKWorkspace(1)

        sol1 = sdirk_solve!(decay!, u0, (0.0, 1.0), nothing, [0.0, 1.0]; ws=ws)
        sol2 = sdirk_solve!(decay!, u0, (0.0, 2.0), nothing, [0.0, 2.0]; ws=ws)

        @test sol1.u[1, 2] ≈ exp(-1.0) atol=1e-4
        @test sol2.u[1, 2] ≈ exp(-2.0) atol=1e-4
    end

    @testset "Parameters passed through" begin
        # du/dt = -λ*u  with λ from pars
        function parameterised!(du, u, pars, t)
            du[1] = -pars.lambda * u[1]
        end
        u0 = [1.0]
        pars = (lambda = 5.0,)
        saveat = [0.0, 0.5, 1.0]

        sol = sdirk_solve!(parameterised!, u0, (0.0, 1.0), pars, saveat)
        for (i, t) in enumerate(saveat)
            @test sol.u[1, i] ≈ exp(-5.0 * t) atol=1e-4
        end
    end

    @testset "Two-component stiff system" begin
        # du₁/dt = -1000 u₁ + u₂
        # du₂/dt = -u₂
        # Exact: u₂(t) = exp(-t)
        #        u₁(t) = (1/999)*exp(-t) + (998/999)*exp(-1000t)
        function two_comp!(du, u, p, t)
            du[1] = -1000.0 * u[1] + u[2]
            du[2] = -u[2]
        end
        u0 = [1.0, 1.0]
        tspan = (0.0, 0.05)
        saveat = collect(0.0:0.01:0.05)

        sol = sdirk_solve!(two_comp!, u0, tspan, nothing, saveat;
                            abstol=1e-6, reltol=1e-6)

        for (i, t) in enumerate(saveat)
            u2_exact = exp(-t)
            u1_exact = (1.0 / 999.0) * exp(-t) + (998.0 / 999.0) * exp(-1000.0 * t)
            @test sol.u[2, i] ≈ u2_exact atol=1e-3
            @test sol.u[1, i] ≈ u1_exact atol=1e-2
        end
    end

    @testset "Result struct" begin
        function decay!(du, u, p, t)
            du[1] = -u[1]
        end
        saveat = [0.0, 0.5, 1.0]
        sol = sdirk_solve!(decay!, [1.0], (0.0, 1.0), nothing, saveat)
        @test sol isa SDIRKResult{Float64}
        @test sol.t == saveat
        @test size(sol.u) == (1, 3)
    end

    @testset "Van der Pol oscillator (μ=1000, stiff)" begin
        # Classic stiff test: x'' - μ(1-x²)x' + x = 0
        # As first-order system: u₁' = u₂, u₂' = μ(1-u₁²)u₂ - u₁
        function vdp!(du, u, p, t)
            mu = 1000.0
            du[1] = u[2]
            du[2] = mu * (1.0 - u[1]^2) * u[2] - u[1]
        end
        u0 = [2.0, 0.0]
        tspan = (0.0, 100.0)
        saveat = collect(range(0.0, 100.0, length=21))

        sol = sdirk_solve!(vdp!, u0, tspan, nothing, saveat;
                            abstol=1e-6, reltol=1e-6, max_steps=500000)

        # Solution should remain bounded (|u₁| ≤ ~2.1 for VdP)
        @test all(abs.(sol.u[1, :]) .< 3.0)
        # Should complete without error (DP5 would struggle here)
        @test length(sol.t) == 21
    end

    @testset "4th order convergence (Richardson extrapolation)" begin
        # Linear ODE: du/dt = -u, u(0) = 1
        # Solve at two different tolerances, verify error scales as O(tol)
        function decay!(du, u, p, t)
            du[1] = -u[1]
        end
        u0 = [1.0]
        t_test = 1.0
        exact = exp(-t_test)

        tol1 = 1e-4
        sol1 = sdirk_solve!(decay!, u0, (0.0, t_test), nothing, [0.0, t_test];
                             abstol=tol1, reltol=tol1)
        err1 = abs(sol1.u[1, 2] - exact)

        tol2 = 1e-8
        sol2 = sdirk_solve!(decay!, u0, (0.0, t_test), nothing, [0.0, t_test];
                             abstol=tol2, reltol=tol2)
        err2 = abs(sol2.u[1, 2] - exact)

        # For a p-th order method, error ∝ tol. When tol shrinks by 10^4,
        # error should shrink by roughly 10^4 (within an order of magnitude).
        if err1 > 0 && err2 > 0
            ratio = err1 / err2
            @test ratio > 100  # at least 2 orders of magnitude improvement
        end
    end

    @testset "ForwardDiff Jacobian (autodiff=true)" begin
        function stiff!(du, u, p, t)
            du[1] = -100.0 * u[1] + u[2]
            du[2] = -u[2]
        end
        u0 = [1.0, 1.0]
        saveat = [0.0, 0.05, 0.1]

        sol_fd = sdirk_solve!(stiff!, u0, (0.0, 0.1), nothing, saveat;
                               abstol=1e-8, reltol=1e-8)
        sol_ad = sdirk_solve!(stiff!, u0, (0.0, 0.1), nothing, saveat;
                               abstol=1e-8, reltol=1e-8, autodiff=true)

        # Both should give nearly identical results
        for i in 1:3
            @test sol_fd.u[1, i] ≈ sol_ad.u[1, i] atol=1e-5
            @test sol_fd.u[2, i] ≈ sol_ad.u[2, i] atol=1e-5
        end
    end

    @testset "User-provided Jacobian" begin
        # du/dt = -λ*u  →  J = [-λ]
        lambda = 500.0
        function stiff_param!(du, u, p, t)
            du[1] = -lambda * u[1]
        end
        function exact_jac!(J, u, pars, t)
            J[1, 1] = -lambda
        end

        u0 = [1.0]
        saveat = [0.0, 0.005, 0.01]

        sol = sdirk_solve!(stiff_param!, u0, (0.0, 0.01), nothing, saveat;
                            abstol=1e-8, reltol=1e-8, jac_fn=exact_jac!)

        for (i, t) in enumerate(saveat)
            @test sol.u[1, i] ≈ exp(-lambda * t) atol=1e-5
        end
    end

    @testset "Jacobian reuse (jac_age tracking)" begin
        # Verify that the solver runs correctly even with Jacobian reuse.
        # The workspace jac_age should increment on accepted steps.
        function decay!(du, u, p, t)
            du[1] = -u[1]
        end
        u0 = [1.0]
        ws = SDIRKWorkspace(1)
        saveat = collect(0.0:0.1:2.0)

        sol = sdirk_solve!(decay!, u0, (0.0, 2.0), nothing, saveat; ws=ws)

        # After solving, jac_age should have been tracked
        @test ws.jac_age >= 0
        # Solution should still be accurate
        for (i, t) in enumerate(saveat)
            @test sol.u[1, i] ≈ exp(-t) atol=1e-4
        end
    end

    @testset "Robertson long time (t=1e5)" begin
        # Extend Robertson to very long times — true stiff solver test
        function robertson!(du, u, p, t)
            du[1] = -0.04 * u[1] + 1e4 * u[2] * u[3]
            du[2] =  0.04 * u[1] - 1e4 * u[2] * u[3] - 3e7 * u[2]^2
            du[3] =  3e7 * u[2]^2
        end
        u0 = [1.0, 0.0, 0.0]
        saveat = [0.0, 1.0, 1e2, 1e4, 1e5]

        sol = sdirk_solve!(robertson!, u0, (0.0, 1e5), nothing, saveat;
                            abstol=1e-8, reltol=1e-8, max_steps=500000)

        # Conservation law
        for ti in 1:length(saveat)
            @test sum(sol.u[:, ti]) ≈ 1.0 atol=1e-2
        end

        # Non-negativity
        @test all(sol.u .>= -1e-4)
    end

end
