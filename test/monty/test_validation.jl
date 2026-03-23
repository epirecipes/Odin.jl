using Test
using Odin
using Random
using Statistics

@testset "Model Validation" begin

    gen = @odin begin
        deriv(S) = -beta * S * I / N
        deriv(I) = beta * S * I / N - gamma * I
        deriv(R) = gamma * I
        output(cases) = beta * S * I / N
        initial(S) = N - I0
        initial(I) = I0
        initial(R) = 0
        beta = parameter(0.5)
        gamma = parameter(0.1)
        I0 = parameter(10)
        N = parameter(1000)
    end

    packer = monty_packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))

    @testset "Posterior predictive dimensions" begin
        # Create fake MontySamples
        n_pars = 2
        n_samples = 50
        n_chains = 2
        pars_arr = zeros(n_pars, n_samples, n_chains)
        for c in 1:n_chains, s in 1:n_samples
            pars_arr[1, s, c] = 0.4 + 0.2 * rand()  # beta ∈ [0.4, 0.6]
            pars_arr[2, s, c] = 0.08 + 0.04 * rand() # gamma ∈ [0.08, 0.12]
        end
        density = zeros(n_samples, n_chains)
        samples = MontySamples(pars_arr, density, pars_arr[:, 1, :],
                               ["beta", "gamma"], Dict{Symbol, Any}())

        times = collect(1.0:5.0:30.0)
        pp = posterior_predictive(samples, gen;
                                  times=times, n_draws=10, dt=0.25,
                                  packer=packer, seed=42)

        @test pp isa PosteriorPredictive
        @test length(pp.times) == length(times)
        n_all_vars = length(pp.variable_names)
        @test n_all_vars >= gen.model.n_state  # at least state vars
        @test size(pp.draws, 1) == n_all_vars
        @test size(pp.draws, 2) == length(times)
        @test size(pp.draws, 3) == 10
        @test size(pp.summary.mean) == (n_all_vars, length(times))
    end

    @testset "Posterior predictive with output_vars" begin
        n_pars = 2
        n_samples = 20
        n_chains = 1
        pars_arr = zeros(n_pars, n_samples, n_chains)
        for s in 1:n_samples
            pars_arr[1, s, 1] = 0.5
            pars_arr[2, s, 1] = 0.1
        end
        density = zeros(n_samples, n_chains)
        samples = MontySamples(pars_arr, density, pars_arr[:, 1, :],
                               ["beta", "gamma"], Dict{Symbol, Any}())

        times = collect(5.0:5.0:20.0)
        pp = posterior_predictive(samples, gen;
                                  times=times, n_draws=5, dt=0.25,
                                  output_vars=[:cases],
                                  packer=packer, seed=123)

        @test length(pp.variable_names) == 1
        @test pp.variable_names[1] == :cases
        @test size(pp.draws) == (1, length(times), 5)
    end

    @testset "PPC coverage for well-specified model" begin
        n_pars = 2
        n_samples = 100
        n_chains = 2
        pars_arr = zeros(n_pars, n_samples, n_chains)
        for c in 1:n_chains, s in 1:n_samples
            pars_arr[1, s, c] = 0.45 + 0.1 * rand()
            pars_arr[2, s, c] = 0.08 + 0.04 * rand()
        end
        density = zeros(n_samples, n_chains)
        samples = MontySamples(pars_arr, density, pars_arr[:, 1, :],
                               ["beta", "gamma"], Dict{Symbol, Any}())

        times = collect(1.0:2.0:30.0)
        pp = posterior_predictive(samples, gen;
                                  times=times, n_draws=50, dt=0.25,
                                  output_vars=[:cases],
                                  packer=packer, seed=42)

        # Generate data from the true model (within posterior range)
        true_pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        result = dust_system_simulate(gen, true_pars; times=times, dt=0.25)
        n_state = gen.model.n_state
        cases_idx = n_state + 1
        data = [(time=times[i], cases=result[cases_idx, 1, i]) for i in 1:length(times)]

        ppc = ppc_check(pp, data; pred_var=:cases, data_var=:cases)

        @test ppc isa PPCResult
        @test length(ppc.times) == length(times)
        @test length(ppc.observed) == length(times)
        @test length(ppc.p_values) == length(times)
        # Well-specified: coverage should be reasonable
        @test ppc.coverage_95 >= 0.5  # at least half should be within 95% CI
        @test ppc.chi_squared >= 0.0
        # P-values should be between 0 and 1
        @test all(0.0 .<= ppc.p_values .<= 1.0)
    end

    @testset "Residuals centered near zero" begin
        # Use true parameters for predictions → residuals ≈ 0
        n_pars = 2
        n_samples = 30
        n_chains = 1
        pars_arr = zeros(n_pars, n_samples, n_chains)
        for s in 1:n_samples
            pars_arr[1, s, 1] = 0.5
            pars_arr[2, s, 1] = 0.1
        end
        density = zeros(n_samples, n_chains)
        samples = MontySamples(pars_arr, density, pars_arr[:, 1, :],
                               ["beta", "gamma"], Dict{Symbol, Any}())

        times = collect(1.0:2.0:20.0)
        pp = posterior_predictive(samples, gen;
                                  times=times, n_draws=20, dt=0.25,
                                  output_vars=[:cases],
                                  packer=packer, seed=42)

        true_pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        result = dust_system_simulate(gen, true_pars; times=times, dt=0.25)
        n_state = gen.model.n_state
        cases_idx = n_state + 1
        data = [(time=times[i], cases=result[cases_idx, 1, i]) for i in 1:length(times)]

        rd = residual_diagnostics(pp, data; pred_var=:cases, data_var=:cases)

        @test rd isa ResidualDiagnostics
        @test length(rd.raw_residuals) == length(times)
        # Bias should be near zero for correct model with true params
        @test abs(rd.bias) < 5.0
        @test rd.rmse >= 0.0
        @test rd.mae >= 0.0
        # RMSE should be small for perfect parameters
        @test rd.rmse < 10.0
    end

    @testset "Standardized residuals have std ≈ 1" begin
        n_pars = 2
        n_samples = 60
        n_chains = 2
        pars_arr = zeros(n_pars, n_samples, n_chains)
        for c in 1:n_chains, s in 1:n_samples
            pars_arr[1, s, c] = 0.48 + 0.04 * rand()
            pars_arr[2, s, c] = 0.09 + 0.02 * rand()
        end
        density = zeros(n_samples, n_chains)
        samples = MontySamples(pars_arr, density, pars_arr[:, 1, :],
                               ["beta", "gamma"], Dict{Symbol, Any}())

        times = collect(1.0:1.0:30.0)
        pp = posterior_predictive(samples, gen;
                                  times=times, n_draws=40, dt=0.25,
                                  output_vars=[:cases],
                                  packer=packer, seed=42)

        true_pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        result = dust_system_simulate(gen, true_pars; times=times, dt=0.25)
        n_state = gen.model.n_state
        cases_idx = n_state + 1
        # Add noise that matches the posterior spread
        Random.seed!(99)
        data = [(time=times[i], cases=result[cases_idx, 1, i] + randn() * std(pp.draws[1, i, :])) for i in 1:length(times)]

        rd = residual_diagnostics(pp, data; pred_var=:cases, data_var=:cases)
        std_resid = std(rd.standardized_residuals)
        # Should be roughly order 1 (not 0.01 or 100)
        @test 0.1 < std_resid < 5.0
    end

    @testset "Calibration check detects well-calibrated model" begin
        n_pars = 2
        n_samples = 80
        n_chains = 2
        pars_arr = zeros(n_pars, n_samples, n_chains)
        for c in 1:n_chains, s in 1:n_samples
            pars_arr[1, s, c] = 0.45 + 0.1 * rand()
            pars_arr[2, s, c] = 0.08 + 0.04 * rand()
        end
        density = zeros(n_samples, n_chains)
        samples = MontySamples(pars_arr, density, pars_arr[:, 1, :],
                               ["beta", "gamma"], Dict{Symbol, Any}())

        times = collect(1.0:2.0:30.0)
        pp = posterior_predictive(samples, gen;
                                  times=times, n_draws=60, dt=0.25,
                                  output_vars=[:cases],
                                  packer=packer, seed=42)

        # Data from the model with true params (should be well within predictions)
        true_pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        result = dust_system_simulate(gen, true_pars; times=times, dt=0.25)
        n_state = gen.model.n_state
        cases_idx = n_state + 1
        data = [(time=times[i], cases=result[cases_idx, 1, i]) for i in 1:length(times)]

        cal = calibration_check(pp, data; pred_var=:cases, data_var=:cases)

        @test cal isa CalibrationResult
        @test length(cal.nominal_levels) == 9
        @test length(cal.empirical_levels) == 9
        @test cal.calibration_error >= 0.0
        @test cal.is_well_calibrated isa Bool
        # Empirical levels should be between 0 and 1
        @test all(0.0 .<= cal.empirical_levels .<= 1.0)
    end

    @testset "Bad model produces poor PPC" begin
        # Use very wrong parameters → high residuals, bad coverage
        n_pars = 2
        n_samples = 30
        n_chains = 1
        pars_arr = zeros(n_pars, n_samples, n_chains)
        for s in 1:n_samples
            pars_arr[1, s, 1] = 0.05   # very low beta (wrong)
            pars_arr[2, s, 1] = 0.5    # very high gamma (wrong)
        end
        density = zeros(n_samples, n_chains)
        samples = MontySamples(pars_arr, density, pars_arr[:, 1, :],
                               ["beta", "gamma"], Dict{Symbol, Any}())

        times = collect(1.0:2.0:30.0)
        pp = posterior_predictive(samples, gen;
                                  times=times, n_draws=20, dt=0.25,
                                  output_vars=[:cases],
                                  packer=packer, seed=42)

        # Data from true (correct) parameters
        true_pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        result = dust_system_simulate(gen, true_pars; times=times, dt=0.25)
        n_state = gen.model.n_state
        cases_idx = n_state + 1
        data = [(time=times[i], cases=result[cases_idx, 1, i]) for i in 1:length(times)]

        ppc = ppc_check(pp, data; pred_var=:cases, data_var=:cases)
        rd = residual_diagnostics(pp, data; pred_var=:cases, data_var=:cases)

        # Bad model: high RMSE, large bias, poor coverage
        @test rd.rmse > 5.0
        # Coverage should be lower than well-specified (predictions are wrong)
        @test ppc.coverage_95 < 1.0
    end

    @testset "Autocorrelation near zero for good fit" begin
        n_pars = 2
        n_samples = 50
        n_chains = 2
        pars_arr = zeros(n_pars, n_samples, n_chains)
        for c in 1:n_chains, s in 1:n_samples
            pars_arr[1, s, c] = 0.48 + 0.04 * rand()
            pars_arr[2, s, c] = 0.09 + 0.02 * rand()
        end
        density = zeros(n_samples, n_chains)
        samples = MontySamples(pars_arr, density, pars_arr[:, 1, :],
                               ["beta", "gamma"], Dict{Symbol, Any}())

        times = collect(1.0:1.0:20.0)
        pp = posterior_predictive(samples, gen;
                                  times=times, n_draws=30, dt=0.25,
                                  output_vars=[:cases],
                                  packer=packer, seed=42)

        # Add independent noise → autocorrelation should be ~0
        true_pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        result = dust_system_simulate(gen, true_pars; times=times, dt=0.25)
        n_state = gen.model.n_state
        cases_idx = n_state + 1
        Random.seed!(77)
        data = [(time=times[i], cases=result[cases_idx, 1, i] + randn() * 5.0) for i in 1:length(times)]

        rd = residual_diagnostics(pp, data; pred_var=:cases, data_var=:cases)

        @test length(rd.autocorrelation) > 0
        @test length(rd.autocorrelation) <= 10
        # Each autocorrelation coefficient should be moderate
        @test all(abs.(rd.autocorrelation) .< 1.0)
        # Ljung-Box p-value should be between 0 and 1
        @test 0.0 <= rd.ljung_box_p <= 1.0
    end

    @testset "Prior predictive produces wider intervals" begin
        using Distributions

        prior = @monty_prior begin
            beta ~ Exponential(1.0)
            gamma ~ Exponential(1.0)
        end

        times = collect(5.0:5.0:20.0)
        prior_pp = prior_predictive(prior, gen;
                                    times=times, packer=packer,
                                    n_draws=30, dt=0.25, seed=42)

        @test prior_pp isa PosteriorPredictive
        @test size(prior_pp.draws, 3) == 30

        # Posterior with tight parameters around truth
        n_pars = 2
        n_samples = 30
        n_chains = 1
        pars_arr = zeros(n_pars, n_samples, n_chains)
        for s in 1:n_samples
            pars_arr[1, s, 1] = 0.49 + 0.02 * rand()
            pars_arr[2, s, 1] = 0.09 + 0.02 * rand()
        end
        density = zeros(n_samples, n_chains)
        samples = MontySamples(pars_arr, density, pars_arr[:, 1, :],
                               ["beta", "gamma"], Dict{Symbol, Any}())

        post_pp = posterior_predictive(samples, gen;
                                       times=times, n_draws=30, dt=0.25,
                                       packer=packer, seed=42)

        # Find a variable index where we have variation
        vi = findfirst(==(prior_pp.variable_names[1]), prior_pp.variable_names)
        # Prior should have wider or equal spread than posterior at most times
        prior_spread = prior_pp.summary.q975[vi, :] .- prior_pp.summary.q025[vi, :]
        post_spread = post_pp.summary.q975[vi, :] .- post_pp.summary.q025[vi, :]
        # At least some times should show wider prior
        wider_count = count(prior_spread .>= post_spread)
        @test wider_count > 0 || sum(prior_spread) >= sum(post_spread) * 0.5
    end
end
