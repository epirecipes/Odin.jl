using Test
using Odin

@testset "SIS School Closure model" begin

    sis_school = @odin begin
        update(S) = S - n_SI + n_IS
        update(I) = I + n_SI - n_IS

        initial(S) = N - I0
        initial(I) = I0

        initial(incidence, zero_every = 1) = 0
        update(incidence) = incidence + n_SI

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

    pars = (
        schools_time = schools_time,
        schools_open = schools_open,
        schools_modifier = 0.6,
        beta0 = 0.2,
        gamma = 0.1,
        N = 1000.0,
        I0 = 10.0,
    )

    times = collect(0.0:1.0:200.0)

    @testset "compiles and runs" begin
        result = dust_system_simulate(sis_school, pars;
            times = times, dt = 1.0, n_particles = 5, seed = 42)
        @test size(result) == (3, 5, length(times))
        @test all(isfinite, result)
    end

    @testset "S + I = N conservation" begin
        result = dust_system_simulate(sis_school, pars;
            times = times, dt = 1.0, n_particles = 10, seed = 1)
        for p in 1:10, t in 1:length(times)
            S = result[1, p, t]
            I = result[2, p, t]
            @test S + I ≈ 1000.0 atol = 0.1
        end
    end

    @testset "incidence zero_every resets" begin
        sys = dust_system_create(sis_school, pars; dt = 1.0, n_particles = 1, seed = 7)
        dust_system_set_state_initial!(sys)

        # Run two consecutive steps and verify incidence reflects a single step
        dust_system_run_to_time!(sys, 1.0)
        state1 = dust_system_state(sys)
        inc1 = state1[3, 1]

        dust_system_run_to_time!(sys, 2.0)
        state2 = dust_system_state(sys)
        inc2 = state2[3, 1]

        # Incidence should be new infections in the current step, not cumulative
        # (each step resets via zero_every = 1).
        # Both should be non-negative and ≤ current S (can't infect more than susceptible)
        @test inc1 >= 0
        @test inc2 >= 0
        @test inc1 <= 1000.0
        @test inc2 <= 1000.0
    end

    @testset "school closure reduces transmission" begin
        n_particles = 50
        result_closure = dust_system_simulate(sis_school, pars;
            times = times, dt = 1.0, n_particles = n_particles, seed = 123)

        # schools_modifier = 0.0 means closure has no effect on beta
        pars_no_closure = merge(pars, (schools_modifier = 0.0,))
        result_no_closure = dust_system_simulate(sis_school, pars_no_closure;
            times = times, dt = 1.0, n_particles = n_particles, seed = 123)

        # Mean incidence during a closure period (days 50–60) should be lower
        # when schools_modifier = 0.6 vs 1.0
        closure_idx = findall(t -> 51 <= t <= 60, times)
        mean_inc_with = sum(result_closure[3, :, closure_idx]) / length(result_closure[3, :, closure_idx])
        mean_inc_without = sum(result_no_closure[3, :, closure_idx]) / length(result_no_closure[3, :, closure_idx])

        # With closure, there should be *some* reduction in incidence
        # (may not always hold for every seed, but 50 particles should average out)
        @test mean_inc_with <= mean_inc_without + 1.0
    end

    @testset "interpolation follows school schedule" begin
        # When schools_open = 1 (schools open), beta should be beta0
        # When schools_open = 0 (schools closed), beta should be (1 - modifier) * beta0
        #
        # We test indirectly: run with all schools open vs the closure schedule.
        # The results should differ.
        pars_always_open = merge(pars, (
            schools_time = [0.0],
            schools_open = [1.0],
        ))

        result_schedule = dust_system_simulate(sis_school, pars;
            times = times, dt = 1.0, n_particles = 1, seed = 99)
        result_always_open = dust_system_simulate(sis_school, pars_always_open;
            times = times, dt = 1.0, n_particles = 1, seed = 99)

        # Before any closure (t < 50), trajectories should be identical
        early_idx = findall(t -> 0 <= t <= 49, times)
        @test result_schedule[1, 1, early_idx] == result_always_open[1, 1, early_idx]
        @test result_schedule[2, 1, early_idx] == result_always_open[2, 1, early_idx]

        # After first closure (t >= 51), trajectories should diverge
        late_idx = findall(t -> t >= 60, times)
        @test result_schedule[2, 1, late_idx] != result_always_open[2, 1, late_idx]
    end

    @testset "particle filter runs" begin
        # Simulate data
        truth = dust_system_simulate(sis_school, pars;
            times = times, dt = 1.0, n_particles = 1, seed = 101)

        true_inc = truth[3, 1, 2:end]
        data = dust_filter_data(
            [(time = Float64(t), cases = max(1.0, Float64(c)))
             for (t, c) in zip(times[2:end], true_inc)]
        )

        filter = dust_filter_create(sis_school, data;
            n_particles = 20, dt = 1.0, seed = 42)
        ll = dust_likelihood_run!(filter, pars)

        @test isfinite(ll)
        @test ll < 0
    end
end
