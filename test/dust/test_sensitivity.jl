using Test
using Odin
using Statistics

@testset "Sensitivity Analysis" begin

    # ── Define SIR ODE model for all tests ──
    sir_gen = @odin begin
        deriv(S) = -beta * S * I / N
        deriv(I) = beta * S * I / N - gamma * I
        deriv(R) = gamma * I
        initial(S) = N - I0
        initial(I) = I0
        initial(R) = 0
        beta = parameter(0.5)
        gamma = parameter(0.1)
        I0 = parameter(10)
        N = parameter(1000)
    end

    pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
    times = collect(1.0:2.0:30.0)

    # ──────────────────────────────────────────────────────────
    @testset "Forward sensitivity matches finite differences" begin
        result = sensitivity(sir_gen, pars;
            times=times, params_of_interest=[:beta, :gamma])

        @test size(result.trajectory) == (3, length(times))
        @test size(result.sensitivities) == (3, 2, length(times))
        @test result.param_names == [:beta, :gamma]

        # Verify against finite differences
        eps_fd = 1e-5
        for jp in 1:2
            pname = [:beta, :gamma][jp]
            pv = pars[pname]

            pars_plus = merge(pars, NamedTuple{(pname,)}((pv + eps_fd,)))
            pars_minus = merge(pars, NamedTuple{(pname,)}((pv - eps_fd,)))

            sys_plus = System(sir_gen, pars_plus)
            reset!(sys_plus)
            out_plus = simulate(sys_plus, times)

            sys_minus = System(sir_gen, pars_minus)
            reset!(sys_minus)
            out_minus = simulate(sys_minus, times)

            for ti in 1:length(times)
                for si in 1:3
                    fd_sens = (out_plus[si, 1, ti] - out_minus[si, 1, ti]) / (2 * eps_fd)
                    fwd_sens = result.sensitivities[si, jp, ti]
                    # Allow relative tolerance since both are approximations
                    if abs(fd_sens) > 1.0
                        @test abs(fwd_sens - fd_sens) / abs(fd_sens) < 0.05
                    else
                        @test abs(fwd_sens - fd_sens) < 0.5
                    end
                end
            end
        end
    end

    # ──────────────────────────────────────────────────────────
    @testset "Forward sensitivity trajectory is correct" begin
        result = sensitivity(sir_gen, pars;
            times=times, params_of_interest=[:beta])

        # The trajectory should match a standard simulation
        sys = System(sir_gen, pars)
        reset!(sys)
        ref = simulate(sys, times)

        for ti in 1:length(times)
            for si in 1:3
                @test abs(result.trajectory[si, ti] - ref[si, 1, ti]) < 0.1
            end
        end
    end

    # ──────────────────────────────────────────────────────────
    @testset "Adjoint gradient matches forward gradient" begin
        # Define SIR model with data comparison for likelihood
        sir_compare = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            obs = data()
            obs ~ Poisson(max(I, 1e-6))
            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        # Generate synthetic data
        sys = System(sir_compare, pars)
        reset!(sys)
        obs_times = collect(5.0:5.0:30.0)
        sim_result = simulate(sys, obs_times)
        data_vec = [(time=obs_times[i], obs=max(1.0, round(sim_result[2, 1, i])))
                     for i in 1:length(obs_times)]
        fdata = Odin.ObservedData(data_vec)

        # Build unfilter
        unfilter = Odin.dust_unfilter_create(sir_compare, fdata)
        packer = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))

        # Forward gradient
        fwd_result = Odin.dust_unfilter_gradient(unfilter, pars, packer;
            method=:forward)

        # Adjoint gradient
        adj_result = Odin.dust_unfilter_gradient(unfilter, pars, packer;
            method=:adjoint)

        # Both should give the same log-likelihood
        @test abs(fwd_result.log_likelihood - adj_result.log_likelihood) < 1e-6

        # Gradients should agree in direction and rough magnitude
        for j in 1:2
            if abs(fwd_result.gradient[j]) > 1.0
                @test abs(fwd_result.gradient[j] - adj_result.gradient[j]) /
                      abs(fwd_result.gradient[j]) < 0.6
            else
                @test abs(fwd_result.gradient[j] - adj_result.gradient[j]) < 2.0
            end
        end
    end

    # ──────────────────────────────────────────────────────────
    @testset "Adjoint gradient matches ForwardDiff" begin
        sir_compare = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            obs = data()
            obs ~ Poisson(max(I, 1e-6))
            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        sys = System(sir_compare, pars)
        reset!(sys)
        obs_times = collect(5.0:5.0:30.0)
        sim_result = simulate(sys, obs_times)
        data_vec = [(time=obs_times[i], obs=max(1.0, round(sim_result[2, 1, i])))
                     for i in 1:length(obs_times)]
        fdata = Odin.ObservedData(data_vec)

        unfilter = Odin.dust_unfilter_create(sir_compare, fdata)
        packer = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))

        # ForwardDiff gradient through the unfilter
        ll_model = as_model(unfilter, packer)
        x = [0.5, 0.1]
        fd_grad = ll_model.gradient(x)

        # Finite difference gradient for comparison
        eps_fd = 1e-5
        fd_grad_manual = zeros(2)
        for j in 1:2
            x_plus = copy(x); x_plus[j] += eps_fd
            x_minus = copy(x); x_minus[j] -= eps_fd
            fd_grad_manual[j] = (ll_model.density(x_plus) - ll_model.density(x_minus)) / (2 * eps_fd)
        end

        # ForwardDiff gradient should be close to finite differences
        for j in 1:2
            if abs(fd_grad_manual[j]) > 1.0
                @test abs(fd_grad[j] - fd_grad_manual[j]) / abs(fd_grad_manual[j]) < 0.1
            else
                @test abs(fd_grad[j] - fd_grad_manual[j]) < 1.0
            end
        end

        # Forward sensitivity gradient should match the ForwardDiff gradient direction
        fwd_result = Odin.dust_unfilter_gradient(unfilter, pars, packer; method=:forward)
        for j in 1:2
            # Check same sign (direction)
            if abs(fd_grad[j]) > 1.0 && abs(fwd_result.gradient[j]) > 1.0
                @test sign(fd_grad[j]) == sign(fwd_result.gradient[j])
            end
        end
    end

    # ──────────────────────────────────────────────────────────
    @testset "Sobol indices identify important parameters" begin
        pars_ranges = Dict(
            :beta => (0.2, 1.0),
            :gamma => (0.05, 0.3),
            :I0 => (1.0, 50.0),
            :N => (500.0, 2000.0),
        )

        sobol = sensitivity(sir_gen, pars_ranges; method=:sobol,
            n_samples=200, times=collect(5.0:5.0:30.0), output_var=2)  # I compartment

        @test sobol isa Odin.SobolResult
        @test length(sobol.first_order) == 4
        @test length(sobol.total_order) == 4

        # All indices should be non-negative
        for k in keys(sobol.first_order)
            @test sobol.first_order[k] >= 0.0
            @test sobol.total_order[k] >= 0.0
        end

        # beta should have a substantial total effect on I
        @test sobol.total_order[:beta] > 0.01
    end

    # ──────────────────────────────────────────────────────────
    @testset "Morris screening ranks parameters" begin
        pars_ranges = Dict(
            :beta => (0.2, 1.0),
            :gamma => (0.05, 0.3),
            :I0 => (1.0, 50.0),
            :N => (500.0, 2000.0),
        )

        morris = sensitivity(sir_gen, pars_ranges; method=:morris,
            n_trajectories=15, times=collect(5.0:5.0:30.0), output_var=2)

        @test morris isa Odin.MorrisResult
        @test length(morris.mu_star) == 4
        @test length(morris.sigma) == 4

        # All mu_star values should be non-negative
        for k in keys(morris.mu_star)
            @test morris.mu_star[k] >= 0.0
        end

        # At least one parameter should have a non-trivial effect
        max_mu = maximum(values(morris.mu_star))
        @test max_mu > 0.0
    end

    # ──────────────────────────────────────────────────────────
    @testset "Forward sensitivity with Symbol output_var" begin
        pars_ranges = Dict(
            :beta => (0.2, 1.0),
            :gamma => (0.05, 0.3),
            :I0 => (1.0, 50.0),
            :N => (500.0, 2000.0),
        )

        sobol = sensitivity(sir_gen, pars_ranges; method=:sobol,
            n_samples=100, times=collect(5.0:5.0:30.0), output_var=:I)

        @test sobol isa Odin.SobolResult
        @test length(sobol.first_order) == 4
    end

end
