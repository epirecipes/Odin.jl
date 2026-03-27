using Test
using Odin
using Distributions
using Random

@testset "Zero-inflated & Truncated distributions" begin

    @testset "Fast logpdf — ZIPoisson" begin
        lambda = 5.0
        p0 = 0.3

        expected_k0 = log(p0 + (1 - p0) * exp(-lambda))
        @test Odin._logpdf_zipoisson(lambda, p0, 0) ≈ expected_k0 atol=1e-12

        expected_k3 = log(1 - p0) + logpdf(Poisson(lambda), 3)
        @test Odin._logpdf_zipoisson(lambda, p0, 3) ≈ expected_k3 atol=1e-12

        @test Odin._logpdf_zipoisson(lambda, p0, -1) == -Inf

        @test Odin._logpdf_zipoisson(lambda, 0.0, 5) ≈ logpdf(Poisson(lambda), 5) atol=1e-12
        @test Odin._logpdf_zipoisson(lambda, 0.0, 0) ≈ logpdf(Poisson(lambda), 0) atol=1e-12

        @test Odin._logpdf_zipoisson(lambda, 1.0, 0) ≈ 0.0 atol=1e-12
        @test Odin._logpdf_zipoisson(lambda, 1.0, 3) == -Inf
    end

    @testset "Fast logpdf — ZINegativeBinomial" begin
        r, p, p0 = 3.0, 0.4, 0.25

        expected_k0 = log(p0 + (1 - p0) * p^r)
        @test Odin._logpdf_zinegbinomial(r, p, p0, 0) ≈ expected_k0 atol=1e-12

        expected_k5 = log(1 - p0) + logpdf(NegativeBinomial(r, p), 5)
        @test Odin._logpdf_zinegbinomial(r, p, p0, 5) ≈ expected_k5 atol=1e-12

        @test Odin._logpdf_zinegbinomial(r, p, p0, -1) == -Inf

        @test Odin._logpdf_zinegbinomial(r, p, 0.0, 5) ≈ logpdf(NegativeBinomial(r, p), 5) atol=1e-12
    end

    @testset "Fast logpdf — TruncatedNormal" begin
        mu, sigma = 5.0, 2.0
        lo, hi = 0.0, 10.0
        td = truncated(Normal(mu, sigma), lo, hi)

        for x in [0.5, 2.0, 5.0, 8.0, 9.5]
            expected = logpdf(td, x)
            got = Odin._logpdf_truncnormal(mu, sigma, lo, hi, x)
            @test got ≈ expected atol=1e-10
        end

        @test Odin._logpdf_truncnormal(mu, sigma, lo, hi, -1.0) == -Inf
        @test Odin._logpdf_truncnormal(mu, sigma, lo, hi, 11.0) == -Inf

        for x in [3.0, 5.0, 7.0]
            @test Odin._logpdf_truncnormal(mu, sigma, -1e6, 1e6, x) ≈ logpdf(Normal(mu, sigma), x) atol=1e-6
        end
    end

    @testset "log_sum_exp" begin
        @test Odin._log_sum_exp(log(0.3), log(0.7)) ≈ 0.0 atol=1e-12
        @test Odin._log_sum_exp(-100.0, -200.0) ≈ -100.0 atol=1e-10
        @test Odin._log_sum_exp(-Inf, 0.0) ≈ 0.0 atol=1e-12
    end

    @testset "DSL — ZIPoisson compare" begin
        gen = @odin begin
            update(cases) = cases
            initial(cases) = 0
            lambda = parameter(5.0)
            p0 = parameter(0.2)
            cases ~ ZIPoisson(lambda, p0)
        end
        model = gen.model
        pars = (lambda=5.0, p0=0.2, dt=1.0)
        s = Float64[0.0]

        @test Odin._odin_compare_data(model, s, pars, (cases=0,), 1.0) ≈
            Odin._logpdf_zipoisson(5.0, 0.2, 0) atol=1e-12
        @test Odin._odin_compare_data(model, s, pars, (cases=3,), 1.0) ≈
            Odin._logpdf_zipoisson(5.0, 0.2, 3) atol=1e-12
    end

    @testset "DSL — ZINegBinomial compare" begin
        gen = @odin begin
            update(cases) = cases
            initial(cases) = 0
            r_nb = parameter(3.0)
            p_nb = parameter(0.4)
            p0 = parameter(0.25)
            cases ~ ZINegBinomial(r_nb, p_nb, p0)
        end
        model = gen.model
        pars = (r_nb=3.0, p_nb=0.4, p0=0.25, dt=1.0)
        s = Float64[0.0]

        @test Odin._odin_compare_data(model, s, pars, (cases=0,), 1.0) ≈
            Odin._logpdf_zinegbinomial(3.0, 0.4, 0.25, 0) atol=1e-12
        @test Odin._odin_compare_data(model, s, pars, (cases=5,), 1.0) ≈
            Odin._logpdf_zinegbinomial(3.0, 0.4, 0.25, 5) atol=1e-12
    end

    @testset "DSL — ZINegativeBinomial alias" begin
        gen = @odin begin
            update(cases) = cases
            initial(cases) = 0
            r_nb = parameter(3.0)
            p_nb = parameter(0.4)
            p0 = parameter(0.25)
            cases ~ ZINegativeBinomial(r_nb, p_nb, p0)
        end
        model = gen.model
        pars = (r_nb=3.0, p_nb=0.4, p0=0.25, dt=1.0)
        s = Float64[0.0]

        @test Odin._odin_compare_data(model, s, pars, (cases=2,), 1.0) ≈
            Odin._logpdf_zinegbinomial(3.0, 0.4, 0.25, 2) atol=1e-12
    end

    @testset "DSL — TruncatedNormal compare" begin
        gen = @odin begin
            deriv(x) = -0.1 * x
            initial(x) = 5.0
            sigma = parameter(1.0)
            x ~ TruncatedNormal(x, sigma, 0.0, 10.0)
        end
        model = gen.model
        pars = (sigma=1.0, dt=0.1)
        s = Float64[5.0]

        @test Odin._odin_compare_data(model, s, pars, (x=4.5,), 0.0) ≈
            Odin._logpdf_truncnormal(5.0, 1.0, 0.0, 10.0, 4.5) atol=1e-12
    end

    @testset "Likelihood with ZIPoisson" begin
        gen = @odin begin
            update(cases) = Poisson(lambda)
            initial(cases) = 0
            lambda = parameter(10.0)
            p0 = parameter(0.3)
            cases ~ ZIPoisson(lambda, p0)
        end

        rng = Random.MersenneTwister(42)
        obs = [(time=Float64(t),
                cases=Float64(rand(rng) < 0.3 ? 0 : rand(rng, Poisson(10.0))))
               for t in 1:20]

        lik = Likelihood(gen, obs; n_particles=100, time_start=0.0)
        ll = loglik(lik, (lambda=10.0, p0=0.3))
        @test isfinite(ll)
        @test ll < 0

        ll_lo = loglik(lik, (lambda=10.0, p0=0.1))
        @test isfinite(ll_lo)
    end

    @testset "Unfilter Likelihood with TruncatedNormal" begin
        gen = @odin begin
            deriv(x) = -alpha * x
            initial(x) = x0
            x0 = parameter(5.0)
            alpha = parameter(0.1)
            sigma = parameter(0.5)
            x ~ TruncatedNormal(x, sigma, 0.0, 100.0)
        end

        rng = Random.MersenneTwister(123)
        true_traj = 5.0 .* exp.(-0.1 .* (1.0:10.0))
        obs = [(time=Float64(t), x=clamp(true_traj[t] + 0.5 * randn(rng), 0.0, 100.0))
               for t in 1:10]

        lik = Likelihood(gen, obs; time_start=0.0)
        ll = loglik(lik, (x0=5.0, alpha=0.1, sigma=0.5))
        @test isfinite(ll)
        @test ll < 0

        ll_bad = loglik(lik, (x0=5.0, alpha=0.1, sigma=0.01))
        @test ll > ll_bad
    end
end
