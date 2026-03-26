using Test
using Odin
using Distributions
using DynamicPPL
using LinearAlgebra
using Random
using MCMCChains

import LogDensityProblems

@testset "Turing Bridge" begin

    # ─── 1. LogDensityProblems interface ────────────────────────

    @testset "as_logdensity" begin
        @testset "MontyLogDensityWrapper (no gradient)" begin
            density = x -> logpdf(Normal(0, 1), x[1]) + logpdf(Normal(0, 1), x[2])
            model = DensityModel(density; parameters=["a", "b"])

            w = as_logdensity(model)
            @test w isa Odin.MontyLogDensityWrapper
            @test LogDensityProblems.logdensity(w, [0.0, 0.0]) ≈ -log(2π)
            @test LogDensityProblems.dimension(w) == 2
            @test LogDensityProblems.capabilities(typeof(w)) ==
                  LogDensityProblems.LogDensityOrder{0}()
        end

        @testset "MontyLogDensityGradWrapper (with gradient)" begin
            density = x -> logpdf(Normal(0, 1), x[1]) + logpdf(Normal(0, 1), x[2])
            gradient = x -> -x
            model = DensityModel(density; parameters=["a", "b"], gradient=gradient)

            w = as_logdensity(model)
            @test w isa Odin.MontyLogDensityGradWrapper
            @test LogDensityProblems.logdensity(w, [0.0, 0.0]) ≈ -log(2π)
            @test LogDensityProblems.dimension(w) == 2
            @test LogDensityProblems.capabilities(typeof(w)) ==
                  LogDensityProblems.LogDensityOrder{1}()

            ll, grad = LogDensityProblems.logdensity_and_gradient(w, [1.0, -1.0])
            @test isfinite(ll)
            @test grad ≈ [-1.0, 1.0]
        end
    end

    # ─── 2. MCMCChains conversion ──────────────────────────────

    @testset "to_chains / from_chains" begin
        @testset "Single chain roundtrip" begin
            n_pars, n_samples, n_chains = 2, 50, 1
            pars = randn(n_pars, n_samples, n_chains)
            density = zeros(n_samples, n_chains)
            initial = pars[:, 1:1, :][:, :, 1]  # n_pars × n_chains
            pnames = ["alpha", "beta"]
            ms = Samples(pars, density, initial, pnames, Dict{Symbol, Any}())

            ch = to_chains(ms)
            @test ch isa Chains
            @test size(ch, 2) == n_pars
            @test Base.names(ch, :parameters) == [Symbol("alpha"), Symbol("beta")]

            ms2 = from_chains(ch)
            @test ms2 isa Samples
            @test ms2.pars ≈ pars
            @test ms2.parameter_names == pnames
        end

        @testset "Multiple chains roundtrip" begin
            n_pars, n_samples, n_chains = 3, 100, 4
            pars = randn(n_pars, n_samples, n_chains)
            density = zeros(n_samples, n_chains)
            initial = pars[:, 1, :]
            pnames = ["x", "y", "z"]
            ms = Samples(pars, density, initial, pnames, Dict{Symbol, Any}())

            ch = to_chains(ms)
            @test size(ch, 3) == n_chains
            @test Base.names(ch, :parameters) == [:x, :y, :z]

            ms2 = from_chains(ch)
            @test size(ms2.pars) == (n_pars, n_samples, n_chains)
            @test ms2.pars ≈ pars
        end
    end

    # ─── 3. dppl_prior basic ───────────────────────────────────

    @testset "dppl_prior basic" begin
        DynamicPPL.@model function simple_prior()
            a ~ Gamma(2.0, 0.5)
            b ~ Normal(0.0, 1.0)
        end

        prior = dppl_prior(simple_prior())
        @test prior isa MontyModel
        @test Set(prior.parameters) == Set(["a", "b"])
        @test prior.properties.has_gradient

        # Density at a specific point should match manual logpdf
        x_a, x_b = 0.8, 0.3
        idx_a = findfirst(==("a"), prior.parameters)
        idx_b = findfirst(==("b"), prior.parameters)
        x = zeros(2)
        x[idx_a] = x_a
        x[idx_b] = x_b
        expected = logpdf(Gamma(2.0, 0.5), x_a) + logpdf(Normal(0.0, 1.0), x_b)
        @test prior(x) ≈ expected

        # Gradient is finite
        g = prior.gradient(x)
        @test length(g) == 2
        @test all(isfinite, g)
    end

    # ─── 4. dppl_prior hierarchical ───────────────────────────

    @testset "dppl_prior hierarchical" begin
        DynamicPPL.@model function hierarchical_prior()
            mu ~ Normal(0.0, 1.0)
            x ~ Normal(mu, 1.0)
        end

        prior = dppl_prior(hierarchical_prior())
        @test prior isa MontyModel
        @test length(prior.parameters) == 2

        # Verify density: joint = p(mu) * p(x | mu)
        idx_mu = findfirst(==("mu"), prior.parameters)
        idx_x = findfirst(==("x"), prior.parameters)
        theta = zeros(2)
        theta[idx_mu] = 0.5
        theta[idx_x] = 0.3
        expected = logpdf(Normal(0.0, 1.0), 0.5) + logpdf(Normal(0.5, 1.0), 0.3)
        @test prior(theta) ≈ expected
    end

    # ─── 5–7. SIR model integration tests ─────────────────────

    # Build a shared SIR model + data + unfilter + packer for tests 5–7
    sir = @odin begin
        deriv(S) = -beta * S * I / N
        deriv(I) = beta * S * I / N - gamma * I
        deriv(R) = gamma * I
        initial(S) = N - I0
        initial(I) = I0
        initial(R) = 0
        cases_lambda = beta * S * I / N + 1e-6
        cases ~ Poisson(cases_lambda)
        beta = parameter(0.5)
        gamma = parameter(0.1)
        I0 = parameter(10.0)
        N = parameter(1000.0)
    end

    # Simulate reference data
    sys = System(sir, (beta=0.5, gamma=0.1, I0=10.0, N=1000.0))
    reset!(sys)
    sim_times = collect(5.0:5.0:50.0)
    sim_result = simulate(sys, sim_times)
    # Build observation data from the I compartment (index 2)
    data_vec = [(time=sim_times[i], cases=max(1.0, sim_result[2, 1, i])) for i in eachindex(sim_times)]
    fdata = Odin.ObservedData(data_vec)

    unfilter = Likelihood(sir, fdata)
    packer = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))

    @testset "to_turing_model" begin
        dm = to_turing_model(unfilter, packer;
            beta=Gamma(2.0, 0.25),
            gamma=Gamma(2.0, 0.05),
        )
        @test dm isa DynamicPPL.Model

        # The model uses indexed θ[1], θ[2] — pass as a vector under :θ
        lj = DynamicPPL.logjoint(dm, (θ=[0.5, 0.1],))
        @test isfinite(lj)
    end

    @testset "dppl_to_monty_model" begin
        dm = to_turing_model(unfilter, packer;
            beta=Gamma(2.0, 0.25),
            gamma=Gamma(2.0, 0.05),
        )
        monty_m = dppl_to_monty_model(dm)

        @test monty_m isa MontyModel
        @test length(monty_m.parameters) == 2

        # Density should match logjoint
        expected_lj = DynamicPPL.logjoint(dm, (θ=[0.5, 0.1],))
        @test monty_m([0.5, 0.1]) ≈ expected_lj
    end

    @testset "turing_sample" begin
        dm = to_turing_model(unfilter, packer;
            beta=Gamma(2.0, 0.25),
            gamma=Gamma(2.0, 0.05),
        )

        vcv = diagm([0.001, 0.0001])
        sampler = random_walk(vcv)
        initial = Float64[0.5 0.5; 0.1 0.1]  # 2 pars × 2 chains

        samples = turing_sample(dm, sampler, 100;
            n_chains=2, initial=initial, seed=42)

        @test samples isa Samples
        @test size(samples.pars, 1) == 2   # n_pars
        @test size(samples.pars, 2) == 100 # n_steps
        @test size(samples.pars, 3) == 2   # n_chains
    end
end
