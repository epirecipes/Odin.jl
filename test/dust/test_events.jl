using Test
using Odin

# Define SIR generator at top level (struct definitions can't be inside @testset)
const _events_sir_gen = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    N = parameter(1000)
    I0 = parameter(10)
    beta = parameter(0.5)
    gamma = parameter(0.1)
end

@testset "Event Handling" begin

    # ────────────────────────────────────────────────────────────
    @testset "Timed events — vaccination pulse" begin
        gen = _events_sir_gen
        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        times = collect(0.0:1.0:50.0)

        # Without events
        out_no_event = dust_system_simulate(gen, pars; times=times)

        # With vaccination at t=5 (while S is still ~917): move 30% of S to R
        vax_event = TimedEvent([5.0], (u, pars, t) -> begin
            s = u[1]
            u[1] = s * 0.7        # 70% remain susceptible
            u[3] = u[3] + s * 0.3 # 30% vaccinated → R
        end)
        events = EventSet(timed=[vax_event])

        sys = dust_system_create(gen, pars)
        dust_system_set_state_initial!(sys)
        out_event = dust_system_simulate(sys, times; events=events)

        # Conservation: S + I + R ≈ N
        for ti in 1:length(times)
            total = out_event[1, 1, ti] + out_event[2, 1, ti] + out_event[3, 1, ti]
            @test total ≈ 1000.0 atol=1.0
        end

        # After vaccination, S should be lower than without events at t=10
        idx_10 = findfirst(==(10.0), times)
        @test out_event[1, 1, idx_10] < out_no_event[1, 1, idx_10]

        # More R compartment after vaccination
        @test out_event[3, 1, idx_10] > out_no_event[3, 1, idx_10]
    end

    # ────────────────────────────────────────────────────────────
    @testset "Continuous event — threshold crossing" begin
        gen = _events_sir_gen
        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        times = collect(0.0:0.5:200.0)

        # Event: when I crosses 200 upward, double recovery rate by moving I→R
        threshold = 200.0
        cont_event = ContinuousEvent(
            (u, pars, t) -> u[2] - threshold,   # zero when I = threshold
            (u, pars, t) -> begin
                # Transfer 20% of I to R (treatment)
                transfer = u[2] * 0.2
                u[2] -= transfer
                u[3] += transfer
            end;
            direction=:up,
            rootfind=true,
        )
        events = EventSet(continuous=[cont_event])

        sys = dust_system_create(gen, pars)
        dust_system_set_state_initial!(sys)
        out = dust_system_simulate(sys, times; events=events)

        # Conservation
        for ti in 1:length(times)
            total = out[1, 1, ti] + out[2, 1, ti] + out[3, 1, ti]
            @test total ≈ 1000.0 atol=1.0
        end

        # Peak I should be reduced compared to no events
        out_no = dust_system_simulate(gen, pars; times=times)
        I_peak_event = maximum(out[2, 1, :])
        I_peak_no = maximum(out_no[2, 1, :])
        @test I_peak_event < I_peak_no
    end

    # ────────────────────────────────────────────────────────────
    @testset "Discrete event — step-based check" begin
        gen = _events_sir_gen
        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        times = collect(0.0:1.0:200.0)

        # Discrete event: if I > 300 at any step, reduce by 10%
        triggered_count = Ref(0)
        disc_event = DiscreteEvent(
            (u, pars, t) -> u[2] > 300.0,
            (u, pars, t) -> begin
                transfer = u[2] * 0.1
                u[2] -= transfer
                u[3] += transfer
                triggered_count[] += 1
            end,
        )
        events = EventSet(discrete=[disc_event])

        sys = dust_system_create(gen, pars)
        dust_system_set_state_initial!(sys)
        out = dust_system_simulate(sys, times; events=events)

        # Event should have triggered at least once
        @test triggered_count[] > 0

        # Conservation
        for ti in 1:length(times)
            total = out[1, 1, ti] + out[2, 1, ti] + out[3, 1, ti]
            @test total ≈ 1000.0 atol=1.0
        end
    end

    # ────────────────────────────────────────────────────────────
    @testset "Root finding accuracy" begin
        # Simple ODE: du/dt = 1 (u starts at 0, crosses threshold at t=threshold)
        # Use the raw dp5 solver with events to test root finding precision
        threshold = 17.3  # exact crossing at t=17.3

        f! = (du, u, pars, t) -> begin
            du[1] = 1.0
        end

        saveat = collect(0.0:1.0:30.0)
        u0 = [0.0]

        cont_event = ContinuousEvent(
            (u, pars, t) -> u[1] - threshold,
            (u, pars, t) -> nothing;  # no-op affect
            direction=:up, rootfind=true,
        )
        events = EventSet(continuous=[cont_event])

        result, log = dp5_solve_events!(f!, u0, (0.0, 30.0), nothing, saveat;
                                        events=events)

        # The event should fire at t ≈ 17.3
        @test length(log) >= 1
        event_time = log[1].time
        @test abs(event_time - threshold) < 1e-10

        # Solution should be correct (u = t for all saved times)
        for (i, t) in enumerate(result.t)
            @test result.u[1, i] ≈ t atol=1e-4
        end
    end

    # ────────────────────────────────────────────────────────────
    @testset "Multiple events in same simulation" begin
        gen = _events_sir_gen
        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        times = collect(0.0:1.0:50.0)

        # Timed vaccination at t=3 and t=6 (early, when S is still high)
        vax1 = TimedEvent([3.0], (u, pars, t) -> begin
            s = u[1]; u[1] = s * 0.9; u[3] += s * 0.1
        end)
        vax2 = TimedEvent([6.0], (u, pars, t) -> begin
            s = u[1]; u[1] = s * 0.8; u[3] += s * 0.2
        end)

        # Continuous event: treatment when I > 150
        treatment = ContinuousEvent(
            (u, pars, t) -> u[2] - 150.0,
            (u, pars, t) -> begin
                tr = u[2] * 0.15; u[2] -= tr; u[3] += tr
            end;
            direction=:up,
        )

        events = EventSet(timed=[vax1, vax2], continuous=[treatment])

        sys = dust_system_create(gen, pars)
        dust_system_set_state_initial!(sys)
        out = dust_system_simulate(sys, times; events=events)

        # Conservation
        for ti in 1:length(times)
            total = out[1, 1, ti] + out[2, 1, ti] + out[3, 1, ti]
            @test total ≈ 1000.0 atol=1.0
        end

        # After both vaccinations, S should be reduced
        out_no = dust_system_simulate(gen, pars; times=times)
        idx_10 = findfirst(==(10.0), times)
        @test out[1, 1, idx_10] < out_no[1, 1, idx_10]
    end

    # ────────────────────────────────────────────────────────────
    @testset "No events = same as baseline" begin
        gen = _events_sir_gen
        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        times = collect(0.0:1.0:100.0)

        # Simulate without events
        out1 = dust_system_simulate(gen, pars; times=times)

        # Simulate with empty EventSet
        sys = dust_system_create(gen, pars)
        dust_system_set_state_initial!(sys)
        out2 = dust_system_simulate(sys, times; events=EventSet())

        # Results should be essentially identical
        for ti in 1:length(times)
            for j in 1:3
                @test out1[j, 1, ti] ≈ out2[j, 1, ti] atol=1e-6
            end
        end
    end

    # ────────────────────────────────────────────────────────────
    @testset "Timed event at exact saveat time" begin
        gen = _events_sir_gen
        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        times = collect(0.0:10.0:200.0)

        # Vaccination at t=100, which is a saveat point
        vax = TimedEvent([100.0], (u, pars, t) -> begin
            u[1] *= 0.5; u[3] += u[1]  # move half of remaining S to R
        end)
        events = EventSet(timed=[vax])

        sys = dust_system_create(gen, pars)
        dust_system_set_state_initial!(sys)
        out = dust_system_simulate(sys, times; events=events)

        # Conservation
        for ti in 1:length(times)
            total = out[1, 1, ti] + out[2, 1, ti] + out[3, 1, ti]
            @test total ≈ 1000.0 atol=1.0
        end

        # S should drop at t=100
        idx_90 = findfirst(==(90.0), times)
        idx_110 = findfirst(==(110.0), times)
        # S at t=110 should be significantly lower than at t=90
        # (accounting for natural epidemic dynamics + vaccination)
        @test out[1, 1, idx_110] < out[1, 1, idx_90]
    end

    # ────────────────────────────────────────────────────────────
    @testset "Brent's method direct test" begin
        # Test the root finder directly
        # f(x) = x^2 - 2 has root at √2 ≈ 1.41421356...
        g = x -> x^2 - 2.0
        root = Odin._brent_root(g, 1.0, 2.0, g(1.0), g(2.0); atol=1e-14)
        @test abs(root - sqrt(2.0)) < 1e-12

        # f(x) = sin(x) has root at π
        g2 = x -> sin(x)
        root2 = Odin._brent_root(g2, 3.0, 3.3, g2(3.0), g2(3.3); atol=1e-14)
        @test abs(root2 - π) < 1e-12

        # f(x) = x - 5.5, root at 5.5
        g3 = x -> x - 5.5
        root3 = Odin._brent_root(g3, 0.0, 10.0, g3(0.0), g3(10.0); atol=1e-14)
        @test abs(root3 - 5.5) < 1e-12
    end

    # ────────────────────────────────────────────────────────────
    @testset "Continuous event direction filtering" begin
        # du/dt = cos(t), u = sin(t)
        # u crosses 0.5 in both directions. Test direction filtering.
        f! = (du, u, pars, t) -> begin
            du[1] = cos(t)
        end

        saveat = collect(0.0:0.1:10.0)
        u0 = [0.0]

        # Only trigger on upcrossing of 0.5
        up_count = Ref(0)
        up_event = ContinuousEvent(
            (u, pars, t) -> u[1] - 0.5,
            (u, pars, t) -> (up_count[] += 1; nothing);
            direction=:up,
        )

        # Only trigger on downcrossing of 0.5
        down_count = Ref(0)
        down_event = ContinuousEvent(
            (u, pars, t) -> u[1] - 0.5,
            (u, pars, t) -> (down_count[] += 1; nothing);
            direction=:down,
        )

        events_up = EventSet(continuous=[up_event])
        events_down = EventSet(continuous=[down_event])

        dp5_solve_events!(f!, u0, (0.0, 10.0), nothing, saveat; events=events_up)
        dp5_solve_events!(f!, u0, (0.0, 10.0), nothing, saveat; events=events_down)

        # sin(t) crosses 0.5 upward ~2 times in [0, 10] (around t≈π/6, t≈2π+π/6)
        @test up_count[] >= 1
        @test down_count[] >= 1
    end

    # ────────────────────────────────────────────────────────────
    @testset "EventSet constructors" begin
        # Default empty
        es = EventSet()
        @test isempty(es.continuous)
        @test isempty(es.discrete)
        @test isempty(es.timed)
        @test !Odin._has_events(es)

        # With events
        te = TimedEvent([1.0, 2.0], (u, p, t) -> nothing)
        es2 = EventSet(timed=[te])
        @test length(es2.timed) == 1
        @test Odin._has_events(es2)

        # Nothing
        @test !Odin._has_events(nothing)
    end

    # ────────────────────────────────────────────────────────────
    @testset "Event record logging" begin
        f! = (du, u, pars, t) -> begin
            du[1] = 1.0
        end

        saveat = collect(0.0:1.0:20.0)
        u0 = [0.0]

        te = TimedEvent([5.0, 10.0, 15.0], (u, pars, t) -> nothing)
        events = EventSet(timed=[te])

        result, log = dp5_solve_events!(f!, u0, (0.0, 20.0), nothing, saveat;
                                        events=events)

        @test length(log) == 3
        @test all(r -> r.kind == :timed, log)
        @test log[1].time ≈ 5.0
        @test log[2].time ≈ 10.0
        @test log[3].time ≈ 15.0
    end
end
