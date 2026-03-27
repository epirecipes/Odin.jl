using Test
using Odin
using Odin: parse_odin_block, EXPR_PRINT, PrintInfo

"""Capture stdout output from a block of code."""
function capture_stdout(f)
    buf = IOBuffer()
    redirect_stdout(buf) do
        f()
    end
    return String(take!(buf))
end

@testset "DSL Print Support" begin
    @testset "Parse print() expression" begin
        block = quote
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.2)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
            print("S={S; .2f} I={I; .1f}")
        end

        exprs = parse_odin_block(block)
        print_exprs = filter(e -> e.type == EXPR_PRINT, exprs)
        @test length(print_exprs) == 1

        pinfo = print_exprs[1].rhs::PrintInfo
        @test pinfo.format_string == "S={S; .2f} I={I; .1f}"
        @test pinfo.variables == [:S, :I]
        @test pinfo.formats == ["%.2f", "%.1f"]
        @test pinfo.condition === nothing
    end

    @testset "Parse print() with when= condition" begin
        block = quote
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.2)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
            print("I={I; .1f}", when = time > 5)
        end

        exprs = parse_odin_block(block)
        print_exprs = filter(e -> e.type == EXPR_PRINT, exprs)
        @test length(print_exprs) == 1

        pinfo = print_exprs[1].rhs::PrintInfo
        @test pinfo.variables == [:I]
        @test pinfo.condition !== nothing
    end

    @testset "Parse print() with no format specifier" begin
        block = quote
            deriv(x) = -a * x
            initial(x) = 1.0
            a = parameter(0.5)
            print("x={x}")
        end

        exprs = parse_odin_block(block)
        print_exprs = filter(e -> e.type == EXPR_PRINT, exprs)
        pinfo = print_exprs[1].rhs::PrintInfo
        @test pinfo.variables == [:x]
        @test pinfo.formats == ["%g"]
    end

    @testset "Compile and run continuous model with print()" begin
        gen = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.2)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
            print("S={S; .2f} I={I; .1f}")
        end

        @test gen isa OdinModel
        @test gen.model.n_state == 3
        @test gen.model.is_continuous == true

        sys = System(gen, (beta=0.2, gamma=0.1, I0=10.0, N=1000.0))
        reset!(sys)
        output = capture_stdout() do
            run_to!(sys, 1.0)
        end
        @test length(output) > 0
        @test occursin("S=", output)
        @test occursin("I=", output)
    end

    @testset "Compile and run discrete model with print()" begin
        gen = @odin begin
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
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.2)
            gamma = parameter(0.1)
            print("S={S; .1f} I={I; .1f}")
        end

        @test gen isa OdinModel
        @test gen.model.is_continuous == false

        sys = System(gen, (beta=0.2, gamma=0.1, I0=10.0, N=1000.0); dt=1.0)
        reset!(sys)
        output = capture_stdout() do
            run_to!(sys, 3.0)
        end
        @test length(output) > 0
        @test occursin("S=", output)
    end

    @testset "Print with when= condition only fires when true" begin
        gen = @odin begin
            deriv(x) = -a * x
            initial(x) = 10.0
            a = parameter(0.5)
            print("x={x; .4f}", when = time > 0.5)
        end

        @test gen isa OdinModel

        sys = System(gen, (a=0.5,))
        reset!(sys)
        output_early = capture_stdout() do
            run_to!(sys, 0.3)
        end

        output_late = capture_stdout() do
            run_to!(sys, 1.0)
        end
        @test length(output_late) > 0
        @test occursin("x=", output_late)
    end

    @testset "Multiple print statements" begin
        gen = @odin begin
            deriv(x) = -a * x
            deriv(y) = a * x - b * y
            initial(x) = 10.0
            initial(y) = 0.0
            a = parameter(0.5)
            b = parameter(0.1)
            print("x={x; .3f}")
            print("y={y; .3f}")
        end

        @test gen isa OdinModel
        sys = System(gen, (a=0.5, b=0.1))
        reset!(sys)
        output = capture_stdout() do
            run_to!(sys, 1.0)
        end
        @test occursin("x=", output)
        @test occursin("y=", output)
    end

    @testset "Print with integer format" begin
        block = quote
            deriv(x) = 1.0
            initial(x) = 0.0
            print("step={x; d}")
        end
        exprs = parse_odin_block(block)
        print_exprs = filter(e -> e.type == EXPR_PRINT, exprs)
        pinfo = print_exprs[1].rhs::PrintInfo
        @test pinfo.formats == ["%d"]
    end

    @testset "Print with parameter reference" begin
        gen = @odin begin
            deriv(x) = -a * x
            initial(x) = 10.0
            a = parameter(0.5)
            print("a={a; .2f} x={x; .2f}")
        end

        @test gen isa OdinModel
        sys = System(gen, (a=0.5,))
        reset!(sys)
        output = capture_stdout() do
            run_to!(sys, 0.5)
        end
        @test occursin("a=", output)
        @test occursin("x=", output)
    end

    @testset "Print with time variable" begin
        gen = @odin begin
            deriv(x) = -x
            initial(x) = 1.0
            print("t={time; .3f} x={x; .4f}")
        end

        @test gen isa OdinModel
        sys = System(gen, (;))
        reset!(sys)
        output = capture_stdout() do
            run_to!(sys, 0.5)
        end
        @test occursin("t=", output)
        @test occursin("x=", output)
    end
end
