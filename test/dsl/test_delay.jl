@testset "DDE — delay() support" begin
    @testset "Parsing" begin
        exprs = Odin.parse_odin_block(quote
            deriv(x) = -a * x_lag
            initial(x) = 1.0
            x_lag = delay(x, tau)
            a = parameter(0.1)
            tau = parameter(1.0)
        end)
        delay_exprs = filter(e -> e.type == Odin.EXPR_DELAY, exprs)
        @test length(delay_exprs) == 1
        @test delay_exprs[1].name == :x_lag
        dinfo = delay_exprs[1].rhs
        @test dinfo isa Odin.DelayInfo
        @test dinfo.expr == :x
        @test dinfo.tau == :tau
    end

    @testset "Classification" begin
        exprs = Odin.parse_odin_block(quote
            deriv(x) = -a * x_lag
            initial(x) = 1.0
            x_lag = delay(x, tau)
            a = parameter(0.1)
            tau = parameter(1.0)
        end)
        cl = Odin.classify_variables(exprs)
        @test haskey(cl.delayed, :x_lag)
        @test cl.delayed[:x_lag].expr == :x
        @test :x_lag in cl.intermediates
    end

    @testset "Delay rejected in discrete models" begin
        exprs = Odin.parse_odin_block(quote
            update(x) = x - a * x_lag
            initial(x) = 1.0
            x_lag = delay(x, tau)
            a = parameter(0.1)
            tau = parameter(1.0)
        end)
        @test_throws ErrorException Odin.classify_variables(exprs)
    end

    @testset "Model compilation" begin
        m = @odin begin
            deriv(x) = -a * x_lag
            initial(x) = 1.0
            x_lag = delay(x, tau)
            a = parameter(0.1)
            tau = parameter(1.0)
        end
        @test m.model.has_delay == true
    end

    @testset "Simulation — constant decay on [0, tau]" begin
        # For t in [0, tau], delay(x, tau) = x(t-tau) = initial = 1.0
        # So deriv(x) = -a * 1.0 = -0.1, giving x(t) = 1.0 - 0.1*t
        m = @odin begin
            deriv(x) = -a * x_lag
            initial(x) = 1.0
            x_lag = delay(x, tau)
            a = parameter(0.1)
            tau = parameter(1.0)
        end
        pars = (a=0.1, tau=1.0)
        times = collect(0.0:0.01:1.0)
        result = simulate(m, pars, times)
        @test result[1,1,1] ≈ 1.0
        # At t=0.5: x = 1 - 0.1*0.5 = 0.95
        @test result[1,1,51] ≈ 0.95 atol=1e-4
        # At t=1.0: x = 1 - 0.1*1.0 = 0.9
        @test result[1,1,end] ≈ 0.9 atol=1e-4
    end

    @testset "Simulation — multi-state DDE" begin
        # Two coupled delayed equations
        m = @odin begin
            deriv(x) = -a * y_lag
            deriv(y) = a * x_lag
            initial(x) = 1.0
            initial(y) = 0.0
            x_lag = delay(x, tau)
            y_lag = delay(y, tau)
            a = parameter(0.5)
            tau = parameter(0.5)
        end
        pars = (a=0.5, tau=0.5)
        times = collect(0.0:0.1:3.0)
        result = simulate(m, pars, times)
        # Check initial values
        @test result[1,1,1] ≈ 1.0
        @test result[2,1,1] ≈ 0.0
        # For t in [0, 0.5]: x_lag = 1.0, y_lag = 0.0
        # So deriv(x) = -0.5*0 = 0, deriv(y) = 0.5*1 = 0.5
        # x stays ~1.0, y grows linearly
        @test result[1,1,6] ≈ 1.0 atol=1e-3  # x(0.5)
        @test result[2,1,6] ≈ 0.25 atol=1e-3  # y(0.5) = 0.5*0.5 = 0.25
        # Simulation shouldn't blow up
        @test isfinite(result[1,1,end])
        @test isfinite(result[2,1,end])
    end

    @testset "DDEHistory — buffer operations" begin
        hist = Odin.DDEHistory(2)
        Odin.dde_history_init!(hist, [1.0, 2.0], 0.0)
        @test hist.initial_state == [1.0, 2.0]
        @test hist.t0 == 0.0
        @test hist.count == 0

        # Before any steps: query returns initial state
        @test Odin.dde_history_eval(hist, -1.0, 1) ≈ 1.0
        @test Odin.dde_history_eval(hist, 0.0, 2) ≈ 2.0

        # Push a step
        Odin.dde_history_push!(hist, 0.0, 0.1, 0.1,
            [1.0, 2.0], [1.1, 1.9],
            [1.0, -1.0], [1.0, -1.0])
        @test hist.count == 1

        # Exact boundary values
        @test Odin.dde_history_eval(hist, 0.0, 1) ≈ 1.0
        @test Odin.dde_history_eval(hist, 0.1, 1) ≈ 1.1
        # Midpoint interpolation
        val = Odin.dde_history_eval(hist, 0.05, 1)
        @test val > 1.0 && val < 1.1
    end

    @testset "Non-delay model unaffected" begin
        # Normal ODE should still work
        m = @odin begin
            deriv(x) = -0.1 * x
            initial(x) = 1.0
        end
        @test m.model.has_delay == false
        result = simulate(m, NamedTuple(), collect(0.0:0.1:5.0))
        @test result[1,1,1] ≈ 1.0
        @test result[1,1,end] ≈ exp(-0.5) atol=1e-4
    end
end
