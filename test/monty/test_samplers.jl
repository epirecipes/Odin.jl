using Test
using Odin
using Distributions
using LinearAlgebra

@testset "Monty Samplers" begin
    # Simple 2D Gaussian target for testing samplers
    target_mean = [3.0, -1.0]
    target_cov = [1.0 0.5; 0.5 2.0]
    target_dist = MvNormal(target_mean, target_cov)

    density = x -> logpdf(target_dist, x)
    gradient = x -> -inv(target_cov) * (x .- target_mean)

    model_no_grad = DensityModel(density; parameters=["x", "y"])
    model_with_grad = DensityModel(density; parameters=["x", "y"], gradient=gradient)

    @testset "Random Walk sampler" begin
        vcv = Matrix{Float64}(0.5I, 2, 2)
        sampler = random_walk(vcv)
        initial = zeros(Float64, 2, 2)  # 2 chains

        samples = sample(model_no_grad, sampler, 2000;
            n_chains=2, initial=initial, n_burnin=500, seed=42)

        @test size(samples.pars) == (2, 1500, 2)

        # Check mean is roughly correct (loose tolerance for short chain)
        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=1.0
        @test mean_est[2] ≈ target_mean[2] atol=1.0
    end

    @testset "HMC sampler" begin
        sampler = hmc(0.1, 10)
        initial = zeros(Float64, 2, 2)

        samples = sample(model_with_grad, sampler, 1000;
            n_chains=2, initial=initial, n_burnin=200, seed=42)

        @test size(samples.pars, 1) == 2

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=0.5
        @test mean_est[2] ≈ target_mean[2] atol=0.5
    end

    @testset "Adaptive sampler" begin
        vcv = Matrix{Float64}(I, 2, 2)
        sampler = adaptive_mh(vcv)
        initial = zeros(Float64, 2, 2)

        samples = sample(model_no_grad, sampler, 3000;
            n_chains=2, initial=initial, n_burnin=1000, seed=42)

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=1.0
        @test mean_est[2] ≈ target_mean[2] atol=1.0
    end

    @testset "Parallel Tempering sampler (RW base)" begin
        vcv = Matrix{Float64}(0.5I, 2, 2)
        base = random_walk(vcv)
        sampler = parallel_tempering(base, 3)
        initial = zeros(Float64, 2, 2)

        samples = sample(model_no_grad, sampler, 2000;
            n_chains=2, initial=initial, n_burnin=500, seed=42)

        @test size(samples.pars) == (2, 1500, 2)

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=1.5
        @test mean_est[2] ≈ target_mean[2] atol=1.5
    end

    @testset "Parallel Tempering sampler (HMC base)" begin
        hmc_base = hmc(0.1, 10)
        sampler = parallel_tempering(hmc_base, 3)
        initial = zeros(Float64, 2, 2)

        samples = sample(model_with_grad, sampler, 1000;
            n_chains=2, initial=initial, n_burnin=200, seed=42)

        @test size(samples.pars, 1) == 2

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=1.0
        @test mean_est[2] ≈ target_mean[2] atol=1.0
    end

    @testset "NUTS sampler (unconstrained)" begin
        sampler = nuts(target_acceptance=0.8, n_adaption=250)
        initial = zeros(Float64, 2, 2)

        samples = sample(model_with_grad, sampler, 1000;
            n_chains=2, initial=initial, n_burnin=300, seed=42)

        @test size(samples.pars, 1) == 2

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=0.5
        @test mean_est[2] ≈ target_mean[2] atol=0.5
    end

    @testset "NUTS sampler (constrained, bijectors)" begin
        # Gamma(3,1)-like target: support (0,∞), mean=3
        constrained_model = DensityModel(
            x -> (2.0 * log(x[1]) - x[1]) + (2.0 * log(x[2]) - x[2]);
            parameters=["a", "b"],
            gradient=x -> [2.0/x[1] - 1.0, 2.0/x[2] - 1.0],
            domain=[0.0 Inf; 0.0 Inf],
        )

        sampler = nuts(target_acceptance=0.8, n_adaption=500)
        initial = [2.0 2.5; 2.0 2.5]

        samples = sample(constrained_model, sampler, 2000;
            n_chains=2, initial=initial, n_burnin=600, seed=42)

        # All samples must be positive (bijector working)
        @test all(samples.pars .> 0)

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ 3.0 atol=0.5
        @test mean_est[2] ≈ 3.0 atol=0.5
    end

    @testset "NUTS sampler (bounded [0,1], logit bijector)" begin
        # Beta(2,5)-like target: support (0,1), mean=2/7
        bounded_model = DensityModel(
            x -> (1.0 * log(x[1]) + 4.0 * log(1 - x[1]));
            parameters=["p"],
            gradient=x -> [1.0/x[1] - 4.0/(1 - x[1])],
            domain=reshape([0.0, 1.0], 1, 2),
        )

        sampler = nuts(target_acceptance=0.8, n_adaption=500)
        initial = reshape([0.3, 0.4], 1, 2)

        samples = sample(bounded_model, sampler, 2000;
            n_chains=2, initial=initial, n_burnin=600, seed=123)

        # All samples in (0,1)
        @test all(0 .< samples.pars .< 1)

        mean_est = mean(samples.pars[1, :, :])
        @test mean_est ≈ 2/7 atol=0.1
    end

    @testset "HMC with bijectors — positive constrained" begin
        # Gamma(3,1): mean=3, var=3, support (0,∞)
        domain = [0.0 Inf]
        m = DensityModel(
            x -> logpdf(Gamma(3, 1), x[1]);
            parameters=["x"], domain=domain,
            gradient=x -> ForwardDiff.gradient(y -> logpdf(Gamma(3,1), y[1]), x)
        )
        sampler = hmc(0.1, 20)
        initial = Float64[3.0 3.0]
        samples = sample(m, sampler, 1000; n_chains=2, initial=initial, n_burnin=200, seed=42)
        post = samples.pars[1, 201:end, :]
        @test all(post .> 0)
        @test mean(post) ≈ 3.0 atol=0.5
    end

    @testset "HMC with bijectors — bounded" begin
        # Beta(2,5): mean=2/7≈0.286, support (0,1)
        domain = [0.0 1.0]
        m = DensityModel(
            x -> logpdf(Beta(2, 5), x[1]);
            parameters=["x"], domain=domain,
            gradient=x -> ForwardDiff.gradient(y -> logpdf(Beta(2,5), y[1]), x)
        )
        sampler = hmc(0.02, 20)
        initial = Float64[0.3 0.3]
        samples = sample(m, sampler, 1000; n_chains=2, initial=initial, n_burnin=200, seed=42)
        post = samples.pars[1, 201:end, :]
        @test all(0 .< post .< 1)
        @test mean(post) ≈ 2/7 atol=0.15
    end
end
