using Test
using Odin
using Distributions
using LinearAlgebra

@testset "Validate & Show Code" begin
    @testset "validate_model — continuous model" begin
        result = validate_model(quote
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.4)
            gamma = parameter(0.2)
            I0 = parameter(10)
            N = parameter(1000)
        end)

        @test result isa OdinValidationResult
        @test result.success == true
        @test result.error === nothing
        @test result.time_type == :continuous
        @test Set(result.state_variables) == Set([:S, :I, :R])
        @test :beta in result.parameters
        @test :gamma in result.parameters
        @test result.has_compare == false
        @test result.has_output == false
        @test result.has_diffusion == false
    end

    @testset "validate_model — discrete model" begin
        result = validate_model(quote
            update(S) = S - n_SI
            update(I) = I + n_SI - n_IR
            update(R) = R + n_IR
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            n_SI = Binomial(S, beta)
            n_IR = Binomial(I, gamma)
            beta = parameter(0.4)
            gamma = parameter(0.2)
            I0 = parameter(10)
            N = parameter(1000)
        end)

        @test result.success == true
        @test result.time_type == :discrete
        @test Set(result.state_variables) == Set([:S, :I, :R])
    end

    @testset "validate_model — model with compare" begin
        result = validate_model(quote
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
        end)

        @test result.success == true
        @test result.has_compare == true
        @test :cases in result.data_variables
    end

    @testset "validate_model — invalid model returns error" begin
        result = validate_model(quote
            deriv(S) = -beta * S
            # Missing initial() for S
            beta = parameter(0.4)
        end)

        @test result.success == false
        @test result.error !== nothing
        @test result.error isa String
        @test length(result.error) > 0
    end

    @testset "show_code — returns Expr for all" begin
        code = show_code(quote
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.4)
            gamma = parameter(0.2)
            I0 = parameter(10)
            N = parameter(1000)
        end)

        @test code isa Expr
    end

    @testset "show_code — filter to :initial" begin
        code = show_code(quote
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            beta = parameter(0.4)
            gamma = parameter(0.2)
            I0 = parameter(10)
            N = parameter(1000)
        end; what=:initial)

        # Should return an Expr or nothing; if found, it should be a function
        @test code isa Expr
    end

    @testset "show_code — filter to :update for discrete model" begin
        code = show_code(quote
            update(S) = S - n_SI
            update(I) = I + n_SI - n_IR
            update(R) = R + n_IR
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            n_SI = Binomial(S, beta)
            n_IR = Binomial(I, gamma)
            beta = parameter(0.4)
            gamma = parameter(0.2)
            I0 = parameter(10)
            N = parameter(1000)
        end; what=:update)

        @test code isa Expr
    end
end

@testset "Adaptive sampler — rerun_every" begin
    target_mean = [3.0, -1.0]
    target_cov = [1.0 0.5; 0.5 2.0]
    target_dist = MvNormal(target_mean, target_cov)
    density = x -> logpdf(target_dist, x)
    model = DensityModel(density; parameters=["x", "y"])

    @testset "constructor accepts rerun_every and rerun_random" begin
        vcv = Matrix{Float64}(I, 2, 2)
        s = adaptive_mh(vcv; rerun_every=10)
        @test s isa Odin.MontyAdaptiveSampler
        @test s.rerun_every == 10
        @test s.rerun_random == true

        s2 = adaptive_mh(vcv; rerun_every=5, rerun_random=false)
        @test s2.rerun_every == 5
        @test s2.rerun_random == false
    end

    @testset "sampling works with deterministic rerun" begin
        vcv = Matrix{Float64}(I, 2, 2)
        sampler = adaptive_mh(vcv; rerun_every=10, rerun_random=false)
        initial = zeros(Float64, 2, 2)

        samples = sample(model, sampler, 500;
            n_chains=2, initial=initial, n_burnin=100, seed=42)

        @test size(samples.pars, 1) == 2
        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=2.0
        @test mean_est[2] ≈ target_mean[2] atol=2.0
    end

    @testset "sampling works with random rerun" begin
        vcv = Matrix{Float64}(I, 2, 2)
        sampler = adaptive_mh(vcv; rerun_every=10, rerun_random=true)
        initial = zeros(Float64, 2, 2)

        samples = sample(model, sampler, 500;
            n_chains=2, initial=initial, n_burnin=100, seed=42)

        @test size(samples.pars, 1) == 2
    end

    @testset "default rerun_every=0 means no rerun" begin
        vcv = Matrix{Float64}(I, 2, 2)
        sampler = adaptive_mh(vcv)
        @test sampler.rerun_every == 0
    end
end
