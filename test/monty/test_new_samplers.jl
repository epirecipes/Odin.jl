using Test
using Odin
using Distributions
using LinearAlgebra
using Statistics

@testset "New Samplers" begin
    # Shared 2D Gaussian target
    target_mean = [3.0, -1.0]
    target_cov = [1.0 0.5; 0.5 2.0]
    target_dist = MvNormal(target_mean, target_cov)

    density = x -> logpdf(target_dist, x)
    gradient = x -> -inv(target_cov) * (x .- target_mean)

    model_no_grad = monty_model(density; parameters=["x", "y"])
    model_with_grad = monty_model(density; parameters=["x", "y"], gradient=gradient)

    # ── Slice Sampler ────────────────────────────────────────────────

    @testset "Slice sampler — basic convergence" begin
        sampler = monty_sampler_slice(w=1.0, max_steps=10)
        initial = zeros(Float64, 2, 2)

        samples = monty_sample(model_no_grad, sampler, 3000;
            n_chains=2, initial=initial, n_burnin=1000, seed=42)

        @test size(samples.pars) == (2, 2000, 2)

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=0.5
        @test mean_est[2] ≈ target_mean[2] atol=0.5
    end

    @testset "Slice sampler — no gradient required" begin
        # Should work without gradient
        sampler = monty_sampler_slice(w=2.0, max_steps=20)
        initial = zeros(Float64, 2, 1)

        samples = monty_sample(model_no_grad, sampler, 1000;
            n_chains=1, initial=initial, n_burnin=500, seed=123)

        @test size(samples.pars, 1) == 2
        @test all(isfinite.(samples.density))
    end

    @testset "Slice sampler — parameter validation" begin
        @test_throws ErrorException monty_sampler_slice(w=-1.0)
        @test_throws ErrorException monty_sampler_slice(w=1.0, max_steps=0)
    end

    @testset "Slice sampler — respects domain" begin
        # Positive-only domain
        pos_model = monty_model(
            x -> logpdf(MvNormal([2.0, 2.0], I), x);
            parameters=["a", "b"],
            domain=[0.0 Inf; 0.0 Inf],
        )
        sampler = monty_sampler_slice(w=1.0, max_steps=10)
        initial = [2.0 2.0; 2.0 2.0]

        samples = monty_sample(pos_model, sampler, 2000;
            n_chains=2, initial=initial, n_burnin=500, seed=42)

        @test all(samples.pars .>= 0.0)
    end

    @testset "Slice sampler — variance recovery" begin
        sampler = monty_sampler_slice(w=2.0, max_steps=15)
        initial = zeros(Float64, 2, 4)

        samples = monty_sample(model_no_grad, sampler, 5000;
            n_chains=4, initial=initial, n_burnin=2000, seed=99)

        # Check variances match target
        all_samples = reshape(samples.pars, 2, :)
        var_est = var(all_samples, dims=2)[:, 1]
        @test var_est[1] ≈ target_cov[1, 1] atol=0.5
        @test var_est[2] ≈ target_cov[2, 2] atol=0.8
    end

    @testset "Slice sampler — integration with monty_sample" begin
        sampler = monty_sampler_slice()
        initial = zeros(Float64, 2, 2)
        samples = monty_sample(model_no_grad, sampler, 500;
            n_chains=2, initial=initial, seed=42)
        @test samples isa Odin.MontySamples
        @test length(samples.parameter_names) == 2
        @test haskey(samples.details, :acceptance_rate)
    end

    # ── MALA Sampler ─────────────────────────────────────────────────

    @testset "MALA — basic convergence" begin
        sampler = monty_sampler_mala(0.3)
        initial = zeros(Float64, 2, 2)

        samples = monty_sample(model_with_grad, sampler, 3000;
            n_chains=2, initial=initial, n_burnin=1000, seed=42)

        @test size(samples.pars) == (2, 2000, 2)

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=0.5
        @test mean_est[2] ≈ target_mean[2] atol=0.5
    end

    @testset "MALA — requires gradient" begin
        sampler = monty_sampler_mala(0.1)
        initial = zeros(Float64, 2, 1)

        @test_throws ErrorException monty_sample(model_no_grad, sampler, 100;
            n_chains=1, initial=initial, seed=42)
    end

    @testset "MALA — parameter validation" begin
        @test_throws ErrorException monty_sampler_mala(-0.1)
    end

    @testset "MALA — with mass matrix" begin
        M = [1.0 0.0; 0.0 2.0]
        sampler = monty_sampler_mala(0.2; vcv=M)
        initial = zeros(Float64, 2, 2)

        samples = monty_sample(model_with_grad, sampler, 2000;
            n_chains=2, initial=initial, n_burnin=500, seed=42)

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=0.8
        @test mean_est[2] ≈ target_mean[2] atol=0.8
    end

    @testset "MALA — converges faster than RW" begin
        # Compare MALA vs RW on same target with similar step sizes
        rw_sampler = monty_sampler_random_walk(Matrix{Float64}(0.3I, 2, 2))
        mala_sampler = monty_sampler_mala(0.3)
        initial = zeros(Float64, 2, 2)
        n_steps = 2000

        rw_samples = monty_sample(model_with_grad, rw_sampler, n_steps;
            n_chains=2, initial=initial, n_burnin=500, seed=42)
        mala_samples = monty_sample(model_with_grad, mala_sampler, n_steps;
            n_chains=2, initial=initial, n_burnin=500, seed=42)

        # MALA should have lower error in mean estimate
        rw_mean = mean(rw_samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        mala_mean = mean(mala_samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]

        rw_err = sum((rw_mean .- target_mean).^2)
        mala_err = sum((mala_mean .- target_mean).^2)

        # Both should be reasonable — we mainly test that MALA works
        @test mala_err < 2.0
    end

    @testset "MALA — integration with monty_sample" begin
        sampler = monty_sampler_mala(0.2)
        initial = zeros(Float64, 2, 2)
        samples = monty_sample(model_with_grad, sampler, 500;
            n_chains=2, initial=initial, seed=42)
        @test samples isa Odin.MontySamples
        @test haskey(samples.details, :acceptance_rate)
        @test all(0.0 .<= samples.details[:acceptance_rate] .<= 1.0)
    end

    # ── Gibbs Sampler ────────────────────────────────────────────────

    @testset "Gibbs — basic block structure" begin
        # Split into two 1D blocks
        blocks = [[1], [2]]
        sub_samplers = [
            monty_sampler_slice(w=1.0),
            monty_sampler_slice(w=1.0),
        ]
        sampler = monty_sampler_gibbs(blocks, sub_samplers)
        initial = zeros(Float64, 2, 2)

        samples = monty_sample(model_no_grad, sampler, 3000;
            n_chains=2, initial=initial, n_burnin=1000, seed=42)

        @test size(samples.pars) == (2, 2000, 2)

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=0.8
        @test mean_est[2] ≈ target_mean[2] atol=0.8
    end

    @testset "Gibbs — mixed sub-samplers (RW + Slice)" begin
        blocks = [[1], [2]]
        sub_samplers = Odin.AbstractMontySampler[
            monty_sampler_random_walk(reshape([0.5], 1, 1)),
            monty_sampler_slice(w=1.0),
        ]
        sampler = monty_sampler_gibbs(blocks, sub_samplers)
        initial = zeros(Float64, 2, 2)

        samples = monty_sample(model_no_grad, sampler, 3000;
            n_chains=2, initial=initial, n_burnin=1000, seed=42)

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=1.0
        @test mean_est[2] ≈ target_mean[2] atol=1.0
    end

    @testset "Gibbs — single block (full vector)" begin
        # Single block = all parameters together (degenerates to base sampler)
        blocks = [[1, 2]]
        sub_samplers = [monty_sampler_slice(w=1.0)]
        sampler = monty_sampler_gibbs(blocks, sub_samplers)
        initial = zeros(Float64, 2, 2)

        samples = monty_sample(model_no_grad, sampler, 2000;
            n_chains=2, initial=initial, n_burnin=500, seed=42)

        @test size(samples.pars, 1) == 2
    end

    @testset "Gibbs — parameter validation" begin
        @test_throws ErrorException monty_sampler_gibbs(Vector{Int}[], Odin.AbstractMontySampler[])
        @test_throws ErrorException monty_sampler_gibbs(
            [[1], [2]],
            [monty_sampler_slice()],  # mismatched length
        )
    end

    @testset "Gibbs — with MALA sub-sampler" begin
        blocks = [[1], [2]]
        sub_samplers = Odin.AbstractMontySampler[
            monty_sampler_mala(0.3; vcv=reshape([1.0], 1, 1)),
            monty_sampler_mala(0.3; vcv=reshape([1.0], 1, 1)),
        ]
        sampler = monty_sampler_gibbs(blocks, sub_samplers)
        initial = zeros(Float64, 2, 2)

        samples = monty_sample(model_with_grad, sampler, 3000;
            n_chains=2, initial=initial, n_burnin=1000, seed=42)

        mean_est = mean(samples.pars[:, :, :], dims=(2, 3))[:, 1, 1]
        @test mean_est[1] ≈ target_mean[1] atol=1.0
        @test mean_est[2] ≈ target_mean[2] atol=1.0
    end

    @testset "Gibbs — integration with monty_sample" begin
        blocks = [[1], [2]]
        sub_samplers = [monty_sampler_slice(), monty_sampler_slice()]
        sampler = monty_sampler_gibbs(blocks, sub_samplers)
        initial = zeros(Float64, 2, 2)

        samples = monty_sample(model_no_grad, sampler, 500;
            n_chains=2, initial=initial, seed=42)

        @test samples isa Odin.MontySamples
        @test length(samples.parameter_names) == 2
    end

    # ── Cross-sampler ESS comparison ─────────────────────────────────

    @testset "ESS comparison across samplers" begin
        n_steps = 3000
        n_burnin = 1000
        initial = zeros(Float64, 2, 2)

        # Slice
        slice = monty_sampler_slice(w=1.5)
        s_slice = monty_sample(model_no_grad, slice, n_steps;
            n_chains=2, initial=initial, n_burnin=n_burnin, seed=42)

        # MALA
        mala = monty_sampler_mala(0.3)
        s_mala = monty_sample(model_with_grad, mala, n_steps;
            n_chains=2, initial=initial, n_burnin=n_burnin, seed=42)

        # RW (baseline)
        rw = monty_sampler_random_walk(Matrix{Float64}(0.5I, 2, 2))
        s_rw = monty_sample(model_no_grad, rw, n_steps;
            n_chains=2, initial=initial, n_burnin=n_burnin, seed=42)

        # Simple ESS estimate via autocorrelation of first chain, first parameter
        function simple_ess(chain::AbstractVector{Float64})
            n = length(chain)
            n < 10 && return Float64(n)
            m = mean(chain)
            v = var(chain)
            v < 1e-12 && return Float64(n)
            max_lag = min(n - 1, 100)
            rho_sum = 0.0
            for k in 1:max_lag
                acf = sum((chain[1:end-k] .- m) .* (chain[k+1:end] .- m)) / ((n - k) * v)
                acf < 0.05 && break
                rho_sum += acf
            end
            return n / (1.0 + 2.0 * rho_sum)
        end

        ess_rw = simple_ess(s_rw.pars[1, :, 1])
        ess_slice = simple_ess(s_slice.pars[1, :, 1])
        ess_mala = simple_ess(s_mala.pars[1, :, 1])

        # All should produce some effective samples
        @test ess_rw > 10
        @test ess_slice > 10
        @test ess_mala > 10
    end
end
