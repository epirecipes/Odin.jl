using Test
using Odin: parse_odin_block, classify_variables, TIME_CONTINUOUS, TIME_DISCRETE,
    VAR_STATE, VAR_PARAMETER, VAR_INTERMEDIATE

@testset "Variable Classification" begin
    @testset "Continuous SIR" begin
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
        cls = classify_variables(exprs)

        @test cls.time_type == TIME_CONTINUOUS
        @test Set(cls.state_vars) == Set([:S, :I, :R])
        @test Set(keys(cls.parameters)) == Set([:N, :I0, :beta, :gamma])
        @test isempty(cls.data_vars)
        @test isempty(cls.intermediates)
    end

    @testset "Discrete SIR with intermediates" begin
        block = quote
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
        end

        exprs = parse_odin_block(block)
        cls = classify_variables(exprs)

        @test cls.time_type == TIME_DISCRETE
        @test Set(cls.state_vars) == Set([:S, :I, :R])
        @test :p_SI in cls.intermediates
        @test :p_IR in cls.intermediates
        @test :n_SI in cls.intermediates
        @test :n_IR in cls.intermediates
    end

    @testset "Cannot mix deriv and update" begin
        block = quote
            deriv(S) = -beta * S
            update(I) = I + 1
            initial(S) = 100
            initial(I) = 10
            beta = parameter(0.1)
        end

        exprs = parse_odin_block(block)
        @test_throws ErrorException classify_variables(exprs)
    end
end
