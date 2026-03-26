using Test
using Odin

@testset "@odin_model macro" begin

    @testset "basic ODE system with priors and fixed params" begin
        model = @odin_model begin
            deriv(S) = -beta * S * Inf2 / N
            deriv(Inf2) = beta * S * Inf2 / N - gamma * Inf2
            deriv(R) = gamma * Inf2
            initial(S) = N - I0
            initial(Inf2) = I0
            initial(R) = 0

            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10.0)
            N = parameter(1000.0)

            @prior begin
                beta ~ Gamma(2.0, 0.25)
                gamma ~ Gamma(2.0, 0.05)
            end

            @fixed I0 = 10.0 N = 1000.0
        end

        # Returns a named tuple
        @test haskey(model, :system)
        @test haskey(model, :prior)
        @test haskey(model, :packer)

        # System is a OdinModel
        @test model.system isa Odin.OdinModel

        # Prior is a MontyModel with correct parameters
        @test model.prior isa Odin.MontyModel
        @test model.prior.parameters == ["beta", "gamma"]
        @test model.prior.properties.has_gradient
        @test model.prior.properties.has_direct_sample

        # Packer has free params and fixed values
        @test model.packer isa Odin.MontyPacker
        @test model.packer.scalar_names == [:beta, :gamma]
        @test model.packer.fixed.I0 == 10.0
        @test model.packer.fixed.N == 1000.0
        @test model.packer.len == 2
    end

    @testset "prior density is correct" begin
        model = @odin_model begin
            deriv(S) = -beta * S * Inf2 / N
            deriv(Inf2) = beta * S * Inf2 / N - gamma * Inf2
            deriv(R) = gamma * Inf2
            initial(S) = N - I0
            initial(Inf2) = I0
            initial(R) = 0

            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10.0)
            N = parameter(1000.0)

            @prior begin
                beta ~ Gamma(2.0, 0.25)
                gamma ~ Gamma(2.0, 0.05)
            end

            @fixed I0 = 10.0 N = 1000.0
        end

        x = [0.5, 0.1]
        expected = Odin.Distributions.logpdf(Odin.Distributions.Gamma(2.0, 0.25), 0.5) +
                   Odin.Distributions.logpdf(Odin.Distributions.Gamma(2.0, 0.05), 0.1)
        @test model.prior(x) ≈ expected

        # Gradient is finite
        g = model.prior.gradient(x)
        @test length(g) == 2
        @test all(isfinite, g)
    end

    @testset "packer unpack produces correct NamedTuple" begin
        model = @odin_model begin
            deriv(S) = -beta * S * Inf2 / N
            deriv(Inf2) = beta * S * Inf2 / N - gamma * Inf2
            deriv(R) = gamma * Inf2
            initial(S) = N - I0
            initial(Inf2) = I0
            initial(R) = 0

            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10.0)
            N = parameter(1000.0)

            @prior begin
                beta ~ Gamma(2.0, 0.25)
                gamma ~ Gamma(2.0, 0.05)
            end

            @fixed I0 = 10.0 N = 1000.0
        end

        pars = Odin.unpack(model.packer, [0.3, 0.15])
        @test pars.beta ≈ 0.3
        @test pars.gamma ≈ 0.15
        @test pars.I0 == 10.0
        @test pars.N == 1000.0
    end

    @testset "system can simulate with packed parameters" begin
        model = @odin_model begin
            deriv(S) = -beta * S * Inf2 / N
            deriv(Inf2) = beta * S * Inf2 / N - gamma * Inf2
            deriv(R) = gamma * Inf2
            initial(S) = N - I0
            initial(Inf2) = I0
            initial(R) = 0

            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10.0)
            N = parameter(1000.0)

            @prior begin
                beta ~ Gamma(2.0, 0.25)
                gamma ~ Gamma(2.0, 0.05)
            end

            @fixed I0 = 10.0 N = 1000.0
        end

        pars = Odin.unpack(model.packer, [0.5, 0.1])
        sys = System(model.system, pars; n_particles=1)
        reset!(sys)
        st = state(sys)
        @test st[1, 1] ≈ 990.0  # S = N - I0
        @test st[2, 1] ≈ 10.0   # Inf2 = I0
        @test st[3, 1] ≈ 0.0    # R = 0
    end

    @testset "no @fixed section is allowed" begin
        model = @odin_model begin
            deriv(x) = -alpha * x
            initial(x) = x0

            alpha = parameter(0.5)
            x0 = parameter(1.0)

            @prior begin
                alpha ~ Exponential(1.0)
                x0 ~ Normal(1.0, 0.1)
            end
        end

        @test model.packer.len == 2
        @test isempty(pairs(model.packer.fixed))
        @test model.prior.parameters == ["alpha", "x0"]
    end
end
