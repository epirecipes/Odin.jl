@testset "Monty DSL" begin

    @testset "@monty_prior basic" begin
        prior = @monty_prior begin
            beta ~ Exponential(1.0)
            gamma ~ Gamma(2.0, 0.5)
        end

        @test prior isa Odin.MontyModel
        @test prior.parameters == ["beta", "gamma"]
        @test prior.properties.has_gradient
        @test prior.properties.has_direct_sample

        # Density matches manual calculation
        x = [0.5, 0.3]
        expected = Odin.Distributions.logpdf(Odin.Distributions.Exponential(1.0), 0.5) +
                   Odin.Distributions.logpdf(Odin.Distributions.Gamma(2.0, 0.5), 0.3)
        @test prior(x) ≈ expected

        # Out of domain returns -Inf
        @test prior([-0.1, 0.3]) == -Inf

        # Gradient is finite
        g = prior.gradient(x)
        @test length(g) == 2
        @test all(isfinite, g)
    end

    @testset "@monty_prior domain extraction" begin
        prior = @monty_prior begin
            x ~ Normal(0.0, 1.0)
            y ~ Uniform(0.0, 1.0)
        end

        @test prior.domain[1, 1] == -Inf  # Normal: (-Inf, Inf)
        @test prior.domain[1, 2] == Inf
        @test prior.domain[2, 1] ≈ 0.0    # Uniform: [0, 1]
        @test prior.domain[2, 2] ≈ 1.0
    end

    @testset "@monty_prior direct sample" begin
        prior = @monty_prior begin
            a ~ Normal(5.0, 0.01)
            b ~ Exponential(0.01)
        end

        rng = Odin.Random.Xoshiro(42)
        s = prior.direct_sample(rng)
        @test length(s) == 2
        @test s[1] ≈ 5.0 atol=0.1  # Normal(5, 0.01) should be very close to 5
        @test s[2] > 0.0            # Exponential is positive
    end

    @testset "@monty_prior + likelihood combination" begin
        prior = @monty_prior begin
            mu ~ Normal(0.0, 10.0)
        end

        likelihood = Odin.monty_model(
            x -> -0.5 * (x[1] - 3.0)^2;
            parameters=["mu"],
        )

        posterior = likelihood + prior
        @test posterior isa Odin.MontyModel
        @test posterior.parameters == ["mu"]

        # Posterior density = likelihood + prior
        x = [3.0]
        @test posterior(x) ≈ likelihood(x) + prior(x)
    end

    @testset "@monty_prior with MCMC" begin
        # Simple inference: estimate mean of Normal
        prior = @monty_prior begin
            mu ~ Normal(0.0, 100.0)
        end

        # Likelihood: data from Normal(5.0, 1.0)
        data_vals = [4.5, 5.2, 4.8, 5.5, 5.0]
        likelihood = Odin.monty_model(
            x -> begin
                s = 0.0
                for d in data_vals
                    s += Odin.Distributions.logpdf(Odin.Distributions.Normal(x[1], 1.0), d)
                end
                s
            end;
            parameters=["mu"],
        )

        posterior = likelihood + prior
        sampler = Odin.monty_sampler_random_walk(fill(1.0, 1, 1))
        samples = Odin.monty_sample(posterior, sampler, 5000;
                                    initial=fill(4.0, 1, 1), n_chains=1)
        # pars shape is (n_pars, n_steps, n_chains)
        mean_mu = mean(samples.pars[1, 1000:end, 1])
        @test mean_mu ≈ 5.0 atol=1.0
    end
end
