using Test
using Odin
using Logging

@testset "Monty Sample" begin
    @testset "Sample continuation" begin
        using Distributions
        density = x -> logpdf(Normal(0, 1), x[1])
        model = DensityModel(density; parameters=["x"])

        vcv = reshape([0.5], 1, 1)
        sampler = random_walk(vcv)
        initial = zeros(Float64, 1, 2)

        s1 = sample(model, sampler, 100; n_chains=2, initial=initial, seed=42)
        @test size(s1.pars) == (1, 100, 2)

        s2 = sample_continue(s1, model, sampler, 50)
        @test size(s2.pars) == (1, 50, 2)
    end

    @testset "Input validation" begin
        density = x -> -0.5 * x[1]^2
        model = DensityModel(density; parameters=["x"], direct_sample=rng -> [0.0])
        sampler = random_walk(reshape([1.0], 1, 1))

        @test_throws ArgumentError sample(model, sampler, 0; n_chains=1)
        @test_throws ArgumentError sample(model, sampler, 10; n_chains=0)
        @test_throws ArgumentError sample(model, sampler, 10; n_chains=1, thinning=0)
        @test_throws ArgumentError sample(model, sampler, 10; n_chains=1, n_burnin=11)
        @test_throws ArgumentError sample(model, sampler, 10; n_chains=2, initial=zeros(1, 1))
    end

    @testset "Threaded runner isolates unfilter likelihood state" begin
        sir = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            obs = data()
            obs ~ Poisson(max(I, 1e-6))
            beta = parameter(0.4)
            gamma = parameter(0.2)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)
        sim = simulate(sir, pars, 0.0:1.0:10.0)
        data = ObservedData([(time=Float64(t), obs=max(1.0, sim[2, 1, t + 1])) for t in 1:10])
        likelihood = as_model(Likelihood(sir, data), Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0)))
        prior = @prior begin
            beta ~ Exponential(0.5)
            gamma ~ Exponential(0.2)
        end
        posterior = likelihood + prior
        sampler = random_walk(0.001 * [1.0 0.0; 0.0 1.0])
        initial = repeat([0.35, 0.18], 1, 2)

        serial = sample(posterior, sampler, 40;
            n_chains=2, initial=initial, n_burnin=10, seed=42, runner=Serial())
        threaded = @test_logs min_level=Logging.Warn sample(posterior, sampler, 40;
            n_chains=2, initial=initial, n_burnin=10, seed=42, runner=Threaded())

        @test threaded.pars == serial.pars
        @test threaded.density == serial.density
    end

    @testset "Threaded runner isolates filter likelihood state" begin
        sis = @odin begin
            update(S) = S - n_SI + n_IS
            update(I) = I + n_SI - n_IS
            initial(S) = N - I0
            initial(I) = I0
            p_SI = 1 - exp(-beta * I / N * dt)
            p_IS = 1 - exp(-gamma * dt)
            n_SI = Binomial(S, p_SI)
            n_IS = Binomial(I, p_IS)
            obs = data()
            obs ~ Poisson(max(I, 1e-6))
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.3)
            gamma = parameter(0.1)
        end

        pars = (beta=0.3, gamma=0.1, I0=10.0, N=1000.0)
        sim = simulate(sis, pars; times=0.0:1.0:12.0, dt=1.0, n_particles=1, seed=7)
        data = ObservedData([(time=Float64(t), obs=max(1.0, sim[2, 1, t + 1])) for t in 1:12])
        likelihood = as_model(
            Likelihood(sis, data; n_particles=32, dt=1.0, seed=99),
            Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0)),
        )
        prior = @prior begin
            beta ~ Exponential(0.5)
            gamma ~ Exponential(0.2)
        end
        posterior = likelihood + prior
        sampler = random_walk(0.0005 * [1.0 0.0; 0.0 1.0])
        initial = repeat([0.28, 0.12], 1, 2)

        serial = sample(posterior, sampler, 30;
            n_chains=2, initial=initial, n_burnin=10, seed=123, runner=Serial())
        threaded = @test_logs min_level=Logging.Warn sample(posterior, sampler, 30;
            n_chains=2, initial=initial, n_burnin=10, seed=123, runner=Threaded())

        @test threaded.pars == serial.pars
        @test threaded.density == serial.density
    end
end
