using Test
using Odin
using Statistics
using LinearAlgebra

@testset "Fitting Workflow" begin

    # ── SIR ODE model (continuous-time, for unfilter) ─────────────────────

    sir_ode = @odin begin
        deriv(S) = -beta * S * I / N
        deriv(I) = beta * S * I / N - gamma * I
        deriv(R) = gamma * I
        initial(S) = N - I0
        initial(I) = I0
        initial(R) = 0

        cases = data()
        cases ~ Poisson(max(I, 1e-6))

        N = parameter(1000)
        I0 = parameter(10)
        beta = parameter(0.4)
        gamma = parameter(0.2)
    end

    sir_pars = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)

    # Generate synthetic data from the ODE model
    sir_times_sim = collect(0.0:1.0:20.0)
    sir_sim = simulate(sir_ode, sir_pars; times=sir_times_sim)
    sir_obs = [(time=sir_times_sim[i], cases=max(1.0, sir_sim[2,1,i]))
               for i in 2:length(sir_times_sim)]
    sir_data = ObservedData(sir_obs)

    @testset "SIR ODE model creation and simulation" begin
        result = simulate(sir_ode, sir_pars; times=sir_times_sim)

        @test size(result, 1) == 3  # S, I, R
        @test all(isfinite, result)

        # S + I + R = N conservation
        for t in 1:length(sir_times_sim)
            @test result[1, 1, t] + result[2, 1, t] + result[3, 1, t] ≈ 1000.0 atol=1.0
        end
    end

    @testset "Unfilter likelihood computation" begin
        unfilter = Likelihood(sir_ode, sir_data)

        ll = loglik(unfilter, sir_pars)
        @test isfinite(ll)
        @test ll < 0

        # Deterministic: repeated calls give same result
        ll2 = loglik(unfilter, sir_pars)
        @test ll ≈ ll2 atol=1e-10

        # Bad parameters should give worse likelihood
        ll_bad = loglik(unfilter, (beta=0.01, gamma=0.5, I0=10.0, N=1000.0))
        @test ll_bad < ll
    end

    @testset "Unfilter MCMC runs" begin
        unfilter = Likelihood(sir_ode, sir_data)
        packer = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))
        likelihood = as_model(unfilter, packer)

        prior = @prior begin
            beta ~ Exponential(0.5)
            gamma ~ Exponential(0.3)
        end

        posterior = likelihood + prior

        vcv = [0.01 0.005; 0.005 0.005]
        sampler = random_walk(vcv)
        initial = repeat([0.3, 0.15], 1, 2)

        samples = sample(posterior, sampler, 200;
            n_chains=2, initial=initial, n_burnin=50, seed=42)

        @test size(samples.pars, 1) == 2   # 2 parameters
        @test size(samples.pars, 3) == 2   # 2 chains
        @test size(samples.pars, 2) == 150 # 200 - 50 burnin

        # Parameters should be positive (exponential prior)
        @test all(samples.pars .> 0)

        # Density should be finite
        @test all(isfinite, samples.density)
    end

    # ── SIS with school closure ──────────────────────────────────────────

    sis = @odin begin
        update(S) = S - n_SI + n_IS
        update(I) = I + n_SI - n_IS
        update(incidence) = incidence + n_SI

        initial(S) = N - I0
        initial(I) = I0
        initial(incidence, zero_every = 1) = 0

        schools = interpolate(schools_time, schools_open, :constant)
        schools_time = parameter(rank = 1)
        schools_open = parameter(rank = 1)

        beta = ((1 - schools) * (1 - schools_modifier) + schools) * beta0

        p_SI = 1 - exp(-beta * I / N * dt)
        p_IS = 1 - exp(-gamma * dt)
        n_SI = Binomial(S, p_SI)
        n_IS = Binomial(I, p_IS)

        N = parameter(1000)
        I0 = parameter(10)
        beta0 = parameter(0.2)
        gamma = parameter(0.1)
        schools_modifier = parameter(0.6)

        cases = data()
        cases ~ Poisson(incidence + 1e-6)
    end

    schools_time = [0.0, 50.0, 60.0, 120.0, 130.0, 170.0, 180.0]
    schools_open = [1.0,  0.0,  1.0,   0.0,   1.0,   0.0,   1.0]

    sis_pars = (
        beta0=0.3, gamma=0.1, schools_modifier=0.5,
        schools_time=schools_time, schools_open=schools_open,
        N=1000.0, I0=10.0,
    )

    schools_cases = Float64[
        2,3,2,2,4,7,2,2,0,3,1,5,4,5,4,5,14,6,12,6,
        6,9,4,7,11,19,18,25,15,16,27,15,19,27,35,23,20,32,23,32,
        30,21,58,31,40,46,38,32,42,46,7,10,19,18,18,20,10,7,11,13,
        36,25,33,24,28,30,38,31,48,40,61,32,30,44,52,39,45,47,40,44,
        43,42,44,34,52,45,40,58,55,41,52,40,62,49,36,40,48,58,41,42,
        37,41,59,42,50,52,35,52,44,38,53,65,48,47,57,53,43,52,32,49,
        19,18,17,17,15,18,12,18,12,8,53,57,42,47,42,41,49,51,45,44,
        49,47,53,33,36,37,44,40,70,57,
    ]

    sis_data = ObservedData(
        [(time=Float64(t), cases=c) for (t, c) in enumerate(schools_cases)]
    )

    @testset "SIS model with interpolation" begin
        times = collect(0.0:1.0:200.0)
        result = simulate(sis, sis_pars;
            times=times, dt=1.0, n_particles=5, seed=42)

        @test size(result) == (3, 5, length(times))
        @test all(isfinite, result)

        # S + I = N conservation (SIS)
        for p in 1:5, t in 1:length(times)
            @test result[1, p, t] + result[2, p, t] ≈ 1000.0 atol=0.1
        end

        # School closure should reduce incidence during closure periods
        # Compare incidence during first closure (50-60) vs just before (40-49)
        open_idx = findall(t -> 40 <= t <= 49, times)
        closed_idx = findall(t -> 51 <= t <= 59, times)
        mean_open = mean(result[3, :, open_idx])
        mean_closed = mean(result[3, :, closed_idx])
        @test mean_closed < mean_open + 5.0  # closure should reduce or not increase much
    end

    @testset "Particle filter likelihood" begin
        filter = Likelihood(sis, sis_data;
            n_particles=50, dt=1.0, seed=42)

        ll = loglik(filter, sis_pars)
        @test isfinite(ll)
        @test ll < 0

        # Stochastic: repeated calls may differ
        ll2 = loglik(filter, sis_pars)
        @test isfinite(ll2)

        # Very bad parameters should give much worse likelihood
        bad_pars = merge(sis_pars, (beta0=5.0, gamma=0.001))
        ll_bad = loglik(filter, bad_pars)
        @test ll_bad < ll
    end

    @testset "Particle filter MCMC runs" begin
        filter = Likelihood(sis, sis_data;
            n_particles=50, dt=1.0, seed=42)

        packer = Packer(
            [:beta0, :gamma, :schools_modifier];
            fixed=(
                schools_time=schools_time,
                schools_open=schools_open,
                N=1000.0, I0=10.0,
            ))

        likelihood = as_model(filter, packer)

        prior = @prior begin
            beta0 ~ Exponential(0.3)
            gamma ~ Exponential(0.1)
            schools_modifier ~ Uniform(0.0, 1.0)
        end

        posterior = likelihood + prior

        vcv = diagm([2e-4, 2e-4, 4e-4])
        sampler = random_walk(vcv)
        initial = repeat([0.3, 0.1, 0.5], 1, 2)

        samples = sample(posterior, sampler, 200;
            n_chains=2, initial=initial, n_burnin=50, seed=42)

        @test size(samples.pars, 1) == 3   # 3 parameters
        @test size(samples.pars, 3) == 2   # 2 chains
        @test size(samples.pars, 2) == 150 # 200 - 50 burnin

        # Parameters should be positive
        @test all(samples.pars .> 0)

        # schools_modifier should be in [0, 1]
        @test all(0.0 .< samples.pars[3, :, :] .< 1.0)

        # Density should be finite
        @test all(isfinite, samples.density)

        # Posterior means should be in reasonable range
        beta0_mean = mean(samples.pars[1, :, :])
        gamma_mean = mean(samples.pars[2, :, :])
        @test 0.05 < beta0_mean < 2.0
        @test 0.01 < gamma_mean < 1.0
    end

    @testset "Counterfactual projection" begin
        # Simulate with original and modified school schedules
        times = collect(0.0:1.0:200.0)

        result_base = simulate(sis, sis_pars;
            times=times, dt=1.0, n_particles=5, seed=42)

        # No closures
        pars_no_closure = merge(sis_pars, (
            schools_time=[0.0],
            schools_open=[1.0],
        ))
        result_no_closure = simulate(sis, pars_no_closure;
            times=times, dt=1.0, n_particles=5, seed=42)

        @test size(result_base) == size(result_no_closure)

        # Before first closure (t < 50), trajectories should be identical
        early_idx = findall(t -> 0 <= t <= 49, times)
        @test result_base[1, :, early_idx] == result_no_closure[1, :, early_idx]

        # After closure, trajectories should diverge
        late_idx = findall(t -> 60 <= t <= 100, times)
        @test result_base[2, :, late_idx] != result_no_closure[2, :, late_idx]
    end
end
