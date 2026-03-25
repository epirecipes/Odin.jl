using Test
using Odin: parse_odin_block, parse_odin_expr, OdinExpr, EXPR_DERIV, EXPR_UPDATE,
    EXPR_INITIAL, EXPR_PARAMETER, EXPR_DATA, EXPR_COMPARE, EXPR_ASSIGNMENT,
    EXPR_DIM, EXPR_INTERPOLATE

@testset "DSL Parsing" begin
    @testset "Simple ODE expressions" begin
        block = quote
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.2)
            gamma = parameter(0.1)
        end

        exprs = parse_odin_block(block)

        # Check counts
        deriv_count = count(e -> e.type == EXPR_DERIV, exprs)
        initial_count = count(e -> e.type == EXPR_INITIAL, exprs)
        param_count = count(e -> e.type == EXPR_PARAMETER, exprs)

        @test deriv_count == 3
        @test initial_count == 3
        @test param_count == 4
    end

    @testset "Discrete update expressions" begin
        block = quote
            update(S) = S - n_SI
            update(I) = I + n_SI - n_IR
            update(R) = R + n_IR
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            n_SI = Binomial(S, p_SI)
            n_IR = Binomial(I, p_IR)
            p_SI = 1 - exp(-beta * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.2)
            gamma = parameter(0.1)
        end

        exprs = parse_odin_block(block)
        update_count = count(e -> e.type == EXPR_UPDATE, exprs)
        @test update_count == 3
    end

    @testset "Comparison expressions" begin
        block = quote
            cases = data()
            cases ~ Poisson(incidence)
        end

        exprs = parse_odin_block(block)
        data_count = count(e -> e.type == EXPR_DATA, exprs)
        compare_count = count(e -> e.type == EXPR_COMPARE, exprs)
        @test data_count == 1
        @test compare_count == 1
    end

    @testset "Parameter options" begin
        block = quote
            x = parameter(5.0, type=:real, constant=true)
        end

        exprs = parse_odin_block(block)
        @test length(exprs) == 1
        pinfo = exprs[1].rhs
        @test pinfo.default == 5.0
        @test pinfo.constant == true
        @test pinfo.type == :real
    end
end
