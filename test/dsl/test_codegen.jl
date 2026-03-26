using Test
using Odin

@testset "Code Generation — @odin macro" begin
    @testset "Continuous SIR compiles and creates generator" begin
        gen = @odin begin
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

        @test gen isa OdinModel
        @test gen.model.n_state == 3
        @test gen.model.is_continuous == true
        @test Set(gen.model.state_names) == Set([:S, :I, :R])
    end

    @testset "Discrete SIR compiles" begin
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
        end

        @test gen isa OdinModel
        @test gen.model.n_state == 3
        @test gen.model.is_continuous == false
    end
end
