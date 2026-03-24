using Test
using Odin
using LinearAlgebra
using ForwardDiff

@testset "Symbolic Differentiation" begin

    # ── Test 1: Basic SIR model with symbolic Jacobian ──
    @testset "SIR symbolic VJP generation" begin
        sir_gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.5, differentiate = true)
            gamma = parameter(0.1, differentiate = true)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        model = sir_gen.model
        @test Odin._odin_has_symbolic_jacobian(model)
        @test Odin._odin_diff_param_names(model) == [:beta, :gamma]
        @test Odin._odin_n_diff_params(model) == 2
    end

    # ── Test 2: Symbolic VJP correctness vs analytic ──
    @testset "VJP correctness vs analytic derivatives" begin
        sir_gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.5, differentiate = true)
            gamma = parameter(0.1, differentiate = true)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        model = sir_gen.model
        pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        state = [990.0, 10.0, 0.0]
        t = 0.0

        # Analytic Jacobian for SIR:
        # J = [-β*I/N   -β*S/N   0
        #       β*I/N    β*S/N-γ  0
        #       0        γ        0]
        bIN = pars.beta * state[2] / pars.N  # β*I/N = 0.005
        bSN = pars.beta * state[1] / pars.N  # β*S/N = 0.495
        J_analytic = [-bIN  -bSN     0.0;
                       bIN   bSN-pars.gamma  0.0;
                       0.0   pars.gamma      0.0]

        # Test VJP state for each canonical basis vector
        for j in 1:3
            v = zeros(3); v[j] = 1.0
            result = zeros(3)
            Odin._odin_vjp_state!(model, result, state, v, pars, t)
            expected = J_analytic' * v
            @test isapprox(result, expected, atol=1e-12)
        end

        # Test VJP state for random vector
        v_rand = [0.3, -0.7, 1.2]
        result = zeros(3)
        Odin._odin_vjp_state!(model, result, state, v_rand, pars, t)
        @test isapprox(result, J_analytic' * v_rand, atol=1e-12)

        # Analytic param Jacobian:
        # ∂f/∂β = [-S*I/N; S*I/N; 0]
        # ∂f/∂γ = [0; -I; I]
        SIN = state[1] * state[2] / pars.N  # 9.9
        J_params_analytic = [-SIN  0.0;
                              SIN  -state[2];
                              0.0   state[2]]

        for j in 1:3
            v = zeros(3); v[j] = 1.0
            result = zeros(2)
            Odin._odin_vjp_params!(model, result, state, v, pars, t)
            expected = J_params_analytic' * v
            @test isapprox(result, expected, atol=1e-12)
        end
    end

    # ── Test 3: Symbolic vs ForwardDiff VJP ──
    @testset "Symbolic VJP matches ForwardDiff" begin
        sir_gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.5, differentiate = true)
            gamma = parameter(0.1, differentiate = true)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        model = sir_gen.model
        pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        state = [990.0, 10.0, 0.0]
        t = 0.0

        for _ in 1:5
            v = randn(3)

            # Symbolic VJP
            sym_result = zeros(3)
            Odin._odin_vjp_state!(model, sym_result, state, v, pars, t)

            # ForwardDiff VJP
            fd_result = zeros(3)
            Odin._vjp_state_forwarddiff!(model, fd_result, state, v, pars, t)

            @test isapprox(sym_result, fd_result, atol=1e-10)
        end
    end

    # ── Test 4: Dispatch hierarchy ──
    @testset "compute_vjp dispatches to symbolic" begin
        sir_gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.5, differentiate = true)
            gamma = parameter(0.1, differentiate = true)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        model = sir_gen.model
        pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        state = [990.0, 10.0, 0.0]
        v = [1.0, 0.0, 0.0]

        result = zeros(3)
        method = Odin.compute_vjp_state!(model, result, state, v, pars, 0.0)
        @test method == :symbolic

        result_p = zeros(2)
        method_p = Odin.compute_vjp_params!(model, result_p, state, v, pars, 0.0,
                                             [:beta, :gamma])
        @test method_p == :symbolic
    end

    # ── Test 5: Fallback to ReverseDiff when no differentiate=true ──
    @testset "ReverseDiff fallback when no symbolic" begin
        sir_gen_no_diff = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        model = sir_gen_no_diff.model
        @test !Odin._odin_has_symbolic_jacobian(model)

        pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        state = [990.0, 10.0, 0.0]
        v = [1.0, 0.0, 0.0]

        result = zeros(3)
        method = Odin.compute_vjp_state!(model, result, state, v, pars, 0.0)
        @test method == :reversediff

        # Verify value is correct (matches ForwardDiff)
        fd_result = zeros(3)
        Odin._vjp_state_forwarddiff!(model, fd_result, state, v, pars, 0.0)
        @test isapprox(result, fd_result, atol=1e-10)
    end

    # ── Test 6: Nonlinear model (SIS with log/exp) ──
    @testset "Symbolic VJP for model with exp/log" begin
        sis_gen = @odin begin
            deriv(S) = -beta * S * I / N + gamma * I
            deriv(I) = beta * S * I / N - gamma * I
            initial(S) = N - I0
            initial(I) = I0
            beta = parameter(0.3, differentiate = true)
            gamma = parameter(0.1, differentiate = true)
            I0 = parameter(5.0)
            N = parameter(500.0)
        end

        model = sis_gen.model
        @test Odin._odin_has_symbolic_jacobian(model)

        pars = (beta=0.3, gamma=0.1, I0=5.0, N=500.0)
        state = [495.0, 5.0]
        t = 0.0

        for _ in 1:5
            v = randn(2)
            sym_r = zeros(2)
            Odin._odin_vjp_state!(model, sym_r, state, v, pars, t)

            fd_r = zeros(2)
            Odin._vjp_state_forwarddiff!(model, fd_r, state, v, pars, t)

            @test isapprox(sym_r, fd_r, atol=1e-10)
        end
    end

    # ── Test 7: Model with power and sqrt ──
    @testset "Symbolic VJP with power operations" begin
        power_gen = @odin begin
            deriv(x) = -alpha * x^2 + sqrt(x) * beta
            initial(x) = 1.0
            alpha = parameter(0.1, differentiate = true)
            beta = parameter(0.5, differentiate = true)
        end

        model = power_gen.model
        @test Odin._odin_has_symbolic_jacobian(model)

        pars = (alpha=0.1, beta=0.5)
        state = [2.0]
        t = 0.0

        for _ in 1:5
            v = randn(1)
            sym_r = zeros(1)
            Odin._odin_vjp_state!(model, sym_r, state, v, pars, t)

            fd_r = zeros(1)
            Odin._vjp_state_forwarddiff!(model, fd_r, state, v, pars, t)

            @test isapprox(sym_r, fd_r, atol=1e-10)
        end

        # Param VJP
        for _ in 1:5
            v = randn(1)
            sym_r = zeros(2)
            Odin._odin_vjp_params!(model, sym_r, state, v, pars, t)

            # FD check: ∂f/∂α = -x^2, ∂f/∂β = sqrt(x)
            x = state[1]
            expected = [-x^2 * v[1], sqrt(x) * v[1]]
            @test isapprox(sym_r, expected, atol=1e-10)
        end
    end

    # ── Test 8: End-to-end adjoint sensitivity with symbolic VJP ──
    @testset "Adjoint sensitivity uses symbolic VJP" begin
        sir_gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.5, differentiate = true)
            gamma = parameter(0.1, differentiate = true)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        times = collect(1.0:5.0:30.0)

        # Loss = sum of squared state values at all times
        function loss_fn(traj, times)
            return sum(traj .^ 2)
        end

        adj_result = Odin.dust_sensitivity_adjoint(sir_gen, pars, loss_fn;
            times=times, params_of_interest=[:beta, :gamma])

        @test adj_result.param_names == [:beta, :gamma]
        @test length(adj_result.gradient) == 2

        # Compare with finite differences on total trajectory norm
        function total_loss(p)
            sys = Odin.dust_system_create(sir_gen, p)
            Odin.dust_system_set_state_initial!(sys)
            traj = Odin.dust_system_simulate(sys, times)
            return sum(traj .^ 2)
        end

        eps_fd = 1e-5
        for (jp, pname) in enumerate([:beta, :gamma])
            pv = pars[pname]
            p_plus = merge(pars, NamedTuple{(pname,)}((pv + eps_fd,)))
            p_minus = merge(pars, NamedTuple{(pname,)}((pv - eps_fd,)))
            fd_grad = (total_loss(p_plus) - total_loss(p_minus)) / (2 * eps_fd)
            @test isapprox(adj_result.gradient[jp], fd_grad, rtol=0.01)
        end
    end

    # ── Test 9: Symbolic fallback for array models ──
    @testset "Array models fall back to ReverseDiff" begin
        array_gen = @odin begin
            deriv(S[]) = -beta * S[i] * I_total / N
            deriv(I[]) = beta * S[i] * I_total / N - gamma * I[i]
            I_total = sum(I)
            initial(S[]) = N / n_age - I0
            initial(I[]) = I0
            dim(S) = n_age
            dim(I) = n_age
            beta = parameter(0.3, differentiate = true)
            gamma = parameter(0.1, differentiate = true)
            I0 = parameter(5.0)
            N = parameter(1000.0)
            n_age = parameter(3)
        end

        model = array_gen.model
        # Array models can't be symbolically differentiated (yet)
        @test !Odin._odin_has_symbolic_jacobian(model)
    end

    # ── Test 10: Multiple evaluation points ──
    @testset "VJP correct at multiple state points" begin
        sir_gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.5, differentiate = true)
            gamma = parameter(0.1, differentiate = true)
            I0 = parameter(10.0)
            N = parameter(1000.0)
        end

        model = sir_gen.model
        pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)

        # Test at different points in the epidemic
        states = [
            [990.0, 10.0, 0.0],    # early
            [500.0, 300.0, 200.0],  # peak
            [100.0, 50.0, 850.0],   # late
        ]

        for state in states
            v = randn(3)

            # State VJP: symbolic vs ForwardDiff
            sym_r = zeros(3)
            fd_r = zeros(3)
            Odin._odin_vjp_state!(model, sym_r, state, v, pars, 0.0)
            Odin._vjp_state_forwarddiff!(model, fd_r, state, v, pars, 0.0)
            @test isapprox(sym_r, fd_r, atol=1e-10)

            # Params VJP: symbolic vs compute_vjp (which falls back correctly)
            sym_p = zeros(2)
            Odin._odin_vjp_params!(model, sym_p, state, v, pars, 0.0)

            fallback_p = zeros(2)
            Odin._vjp_params_reversediff!(model, fallback_p, state, v, pars, 0.0,
                                           [:beta, :gamma])
            @test isapprox(sym_p, fallback_p, atol=1e-8)
        end
    end

end
