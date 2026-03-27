using Test
using Odin
using Random

@testset "New features" begin

    # ─── Helper: stochastic SIR model ──────────────────────────
    sir_stoch = @odin begin
        update(S) = S - n_SI
        update(I) = I + n_SI - n_IR
        update(R) = R + n_IR
        initial(S) = N - I0
        initial(I) = I0
        initial(R) = 0
        p_SI = 1 - exp(-beta * I / N * dt)
        p_IR = 1 - exp(-gamma * dt)
        n_SI = Binomial(S, p_SI)
        n_IR = Binomial(I, p_IR)
        cases_reported = data()
        cases_reported ~ Poisson(max(n_SI, 1e-6))
        beta = parameter(0.5)
        gamma = parameter(0.1)
        I0 = parameter(10)
        N = parameter(1000)
    end

    # ─── Helper: ODE SIR model ──────────────────────────────
    sir_ode = @odin begin
        deriv(S) = -beta * S * I / N
        deriv(I) = beta * S * I / N - gamma * I
        deriv(R) = gamma * I
        initial(S) = N - I0
        initial(I) = I0
        initial(R) = 0
        cases = data()
        cases ~ Poisson(max(I, 1e-6))
        beta = parameter(0.4)
        gamma = parameter(0.2)
        I0 = parameter(10)
        N = parameter(1000)
    end

    # ═══════════════════════════════════════════════════════════
    # Feature 1: Multi-group / n_groups support
    # ═══════════════════════════════════════════════════════════

    @testset "Multi-group system creation" begin
        pars1 = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)
        pars2 = (beta=0.6, gamma=0.15, I0=5.0, N=1000.0)

        sys = System(sir_ode, [pars1, pars2])
        @test sys.n_groups == 2
        @test sys.group_pars !== nothing
        @test length(sys.group_pars) == 2
        @test sys.group_state !== nothing
        @test length(sys.group_state) == 2
    end

    @testset "Multi-group initial conditions" begin
        pars1 = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)
        pars2 = (beta=0.6, gamma=0.15, I0=5.0, N=1000.0)

        sys = System(sir_ode, [pars1, pars2])
        reset!(sys)

        # Group 1: I0=10 → S = 990
        @test sys.group_state[1][1, 1] ≈ 990.0
        @test sys.group_state[1][2, 1] ≈ 10.0

        # Group 2: I0=5 → S = 995
        @test sys.group_state[2][1, 1] ≈ 995.0
        @test sys.group_state[2][2, 1] ≈ 5.0
    end

    @testset "Multi-group simulation" begin
        pars1 = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)
        pars2 = (beta=0.6, gamma=0.15, I0=5.0, N=1000.0)
        times = 0.0:1.0:10.0

        result = simulate(sir_ode, [pars1, pars2], times)
        @test ndims(result) == 4
        @test size(result, 4) == 2  # 2 groups
        @test size(result, 3) == 11  # 11 time points
        @test size(result, 2) == 1  # 1 particle

        # Different pars → different trajectories
        @test result[2, 1, end, 1] != result[2, 1, end, 2]
    end

    @testset "Single-group vector pars (backward compat)" begin
        pars = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)
        sys = System(sir_ode, [pars])
        @test sys.n_groups == 1
    end

    @testset "Grouped data" begin
        data = [
            (time=1.0, cases=10, group=1),
            (time=1.0, cases=20, group=2),
            (time=2.0, cases=15, group=1),
            (time=2.0, cases=25, group=2),
        ]
        gd = Odin.dust_filter_data_grouped(data)
        @test length(gd) == 2
        @test length(gd[1].times) == 2  # 2 time points for group 1
        @test length(gd[2].times) == 2  # 2 time points for group 2
        @test gd[1].data[1].cases == 10
        @test gd[2].data[1].cases == 20
    end

    @testset "Multi-group unfilter" begin
        pars1 = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)
        pars2 = (beta=0.6, gamma=0.15, I0=5.0, N=1000.0)

        # Simulate to get data for each group
        r1 = simulate(sir_ode, pars1, 1.0:1.0:5.0)
        r2 = simulate(sir_ode, pars2, 1.0:1.0:5.0)

        data1 = [(time=Float64(t), cases=max(1.0, r1[2,1,t])) for t in 1:5]
        data2 = [(time=Float64(t), cases=max(1.0, r2[2,1,t])) for t in 1:5]

        fd1 = ObservedData(data1)
        fd2 = ObservedData(data2)

        lik = Likelihood(sir_ode, [fd1, fd2])
        ll = loglik(lik, pars1)
        @test isfinite(ll)
    end

    # ═══════════════════════════════════════════════════════════
    # Feature 2: Snapshot saving
    # ═══════════════════════════════════════════════════════════

    @testset "Snapshot saving in filter" begin
        pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)

        sys = System(sir_stoch, pars; n_particles=5, seed=42)
        reset!(sys)
        times = collect(1.0:1.0:10.0)
        result = simulate(sys, times)
        data_vec = [(time=Float64(t), cases_reported=max(1.0, round(result[2,1,t])))
                     for t in 1:10]
        fdata = ObservedData(data_vec)

        filter = Odin.dust_filter_create(sir_stoch, fdata;
                                         n_particles=20, seed=42)
        ll = Odin.dust_likelihood_run!(filter, pars;
                                       save_snapshots=Float64[3.0, 6.0, 9.0])
        @test isfinite(ll)
        snaps = last_snapshots(filter)
        @test snaps !== nothing
        @test size(snaps, 3) == 3  # 3 snapshot times
        @test size(snaps, 2) == 20  # 20 particles
        @test size(snaps, 1) == 3   # 3 state vars (S, I, R)
    end

    @testset "Trajectory saving in filter" begin
        pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)

        sys = System(sir_stoch, pars; n_particles=5, seed=42)
        reset!(sys)
        times = collect(1.0:1.0:5.0)
        result = simulate(sys, times)
        data_vec = [(time=Float64(t), cases_reported=max(1.0, round(result[2,1,t])))
                     for t in 1:5]
        fdata = ObservedData(data_vec)

        filter = Odin.dust_filter_create(sir_stoch, fdata;
                                         n_particles=10, seed=42,
                                         save_trajectories=true)
        ll = Odin.dust_likelihood_run!(filter, pars)
        @test isfinite(ll)
        trajs = last_trajectories(filter)
        @test trajs !== nothing
        @test size(trajs) == (3, 10, 5)  # n_state × n_particles × n_data
    end

    @testset "Unfilter snapshot/trajectory" begin
        pars = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)
        result = simulate(sir_ode, pars, 1.0:1.0:10.0)
        data_vec = [(time=Float64(t), cases=max(1.0, result[2,1,t])) for t in 1:10]
        fdata = ObservedData(data_vec)

        uf = Odin.dust_unfilter_create(sir_ode, fdata)
        ll = Odin.dust_unfilter_run!(uf, pars; save_snapshots=Float64[1.0, 5.0])
        @test isfinite(ll)
        trajs = last_trajectories(uf)
        @test trajs !== nothing
        @test size(trajs) == (3, 10)  # n_state × n_data
    end

    # ═══════════════════════════════════════════════════════════
    # Feature 3: Observer pattern
    # ═══════════════════════════════════════════════════════════

    @testset "Observer type construction" begin
        obs = Observer((model, rng) -> (val=rand(rng),))
        @test obs isa MontyObserver
        @test obs.observe isa Function
        @test obs.finalise isa Function
        @test obs.combine isa Function
        @test obs.append isa Function
    end

    @testset "Observer auto-finalise" begin
        observations = [
            (x=1.0, y=[1, 2]),
            (x=2.0, y=[3, 4]),
            (x=3.0, y=[5, 6]),
        ]
        result = Odin._auto_finalise(observations)
        @test result isa NamedTuple
        @test result.x == [1.0, 2.0, 3.0]
        @test result.y == [1 3 5; 2 4 6]
    end

    @testset "Observer integration with sample" begin
        m = DensityModel(x -> -sum(x .^ 2); parameters=["a", "b"])

        call_count = Ref(0)
        obs = Observer((model, rng) -> begin
            call_count[] += 1
            (step=call_count[],)
        end)

        s = sample(m, random_walk(0.1 * [1.0 0; 0 1.0]), 20;
                   n_chains=1, initial=reshape([0.0, 0.0], 2, 1),
                   observer=obs)
        @test s isa Samples
        @test s.observations !== nothing
        @test s.observations isa NamedTuple
        @test length(s.observations.step) == 20  # one per sample
    end

    @testset "Observer without observer" begin
        m = DensityModel(x -> -sum(x .^ 2); parameters=["a", "b"])
        s = sample(m, random_walk(0.1 * [1.0 0; 0 1.0]), 10;
                   n_chains=1, initial=reshape([0.0, 0.0], 2, 1))
        @test s.observations === nothing
    end

    # ═══════════════════════════════════════════════════════════
    # Feature 4: Simultaneous runner
    # ═══════════════════════════════════════════════════════════

    @testset "SimultaneousRunner type" begin
        r = Simultaneous()
        @test r isa Odin.MontySimultaneousRunner
    end

    @testset "Simultaneous runner produces samples" begin
        m = DensityModel(x -> -sum(x .^ 2); parameters=["a", "b"])
        s = sample(m, random_walk(0.1 * [1.0 0; 0 1.0]), 30;
                   n_chains=2, runner=Simultaneous(),
                   initial=reshape([0.0, 0.0, 0.1, 0.1], 2, 2),
                   seed=42)
        @test s isa Samples
        @test size(s.pars, 2) == 30  # 30 steps
        @test size(s.pars, 3) == 2   # 2 chains
        @test all(isfinite, s.density)
    end

    @testset "Simultaneous runner with observer" begin
        m = DensityModel(x -> -sum(x .^ 2); parameters=["a", "b"])

        obs = Observer((model, rng) -> (val=rand(rng),))

        s = sample(m, random_walk(0.1 * [1.0 0; 0 1.0]), 15;
                   n_chains=2, runner=Simultaneous(),
                   initial=reshape([0.0, 0.0, 0.1, 0.1], 2, 2),
                   observer=obs, seed=42)
        @test s.observations !== nothing
    end

    @testset "Serial vs Simultaneous parity" begin
        m = DensityModel(x -> -sum(x .^ 2); parameters=["a", "b"])
        init = reshape([0.1, 0.2], 2, 1)

        s1 = sample(m, random_walk(0.01 * [1.0 0; 0 1.0]), 20;
                    n_chains=1, runner=Serial(), initial=init, seed=123)
        s2 = sample(m, random_walk(0.01 * [1.0 0; 0 1.0]), 20;
                    n_chains=1, runner=Simultaneous(), initial=init, seed=123)

        @test size(s1.pars) == size(s2.pars)
        # Densities should be identical since same seed + same algorithm
        @test s1.density ≈ s2.density
        @test s1.pars ≈ s2.pars
    end

end
