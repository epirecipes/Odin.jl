using Test
using Odin

@testset "New Julia API" begin

    @testset "Type aliases" begin
        @test OdinModel === OdinModel
        @test Samples === Samples
        @test ObservedData === ObservedData
        @test ODEControl === ODEControl
    end

    @testset "Model compilation" begin
        model = @odin begin
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
        end
        @test model isa OdinModel
    end

    @testset "simulate(model, pars, times)" begin
        model = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0; initial(I) = I0; initial(R) = 0
            beta = parameter(0.4); gamma = parameter(0.2)
            I0 = parameter(10); N = parameter(1000)
        end
        pars = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)

        result = simulate(model, pars, 0.0:1.0:10.0)
        @test size(result) == (3, 1, 11)
        @test result[1, 1, 1] ≈ 990.0  # S(0) = N - I0
    end

    @testset "System / reset! / state" begin
        model = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0; initial(I) = I0; initial(R) = 0
            beta = parameter(0.4); gamma = parameter(0.2)
            I0 = parameter(10); N = parameter(1000)
        end
        pars = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)

        sys = System(model, pars)
        @test sys isa DustSystem
        reset!(sys)
        s = state(sys)
        @test size(s) == (3, 1)
        @test s[1, 1] ≈ 990.0
    end

    @testset "Likelihood / loglik" begin
        model = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0; initial(I) = I0; initial(R) = 0
            cases = data()
            cases ~ Poisson(max(I, 1e-6))
            beta = parameter(0.4); gamma = parameter(0.2)
            I0 = parameter(10); N = parameter(1000)
        end
        pars = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)

        data_list = [(time=Float64(t), cases=10.0) for t in 1:10]
        lik = Likelihood(model, data_list)
        ll = loglik(lik, pars)
        @test isfinite(ll)
        @test ll < 0
    end

    @testset "Packer / GroupedPacker" begin
        pk = Packer([:beta, :gamma])
        @test pk isa MontyPacker
        @test pk.names == [:beta, :gamma]

        pk2 = Packer([:beta]; fixed=(N=1000.0, I0=10.0))
        @test pk2 isa MontyPacker
        @test :N in keys(pk2.fixed)
    end

    @testset "Samplers" begin
        @test nuts() isa Odin.MontyNUTSSampler
        @test random_walk([1.0 0; 0 1.0]) isa Odin.MontyRandomWalkSampler
        @test hmc(0.01, 10) isa Odin.MontyHMCSampler
        @test adaptive_mh([1.0 0; 0 1.0]) isa Odin.MontyAdaptiveSampler
        @test mala(0.01) isa Odin.MontyMALASampler
        @test slice() isa Odin.MontySliceSampler
    end

    @testset "Runners" begin
        @test Serial() isa Odin.MontySerialRunner
        @test Threaded() isa Odin.MontyThreadedRunner
    end

    @testset "sample (end-to-end)" begin
        model = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0; initial(I) = I0; initial(R) = 0
            cases = data()
            cases ~ Poisson(max(I, 1e-6))
            beta = parameter(0.4); gamma = parameter(0.2)
            I0 = parameter(10); N = parameter(1000)
        end
        pars = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)
        result = simulate(model, pars, 0.0:1.0:20.0)
        data_list = [(time=Float64(t), cases=max(round(Int, result[2,1,t]), 1)*1.0)
                     for t in 1:20]

        lik = Likelihood(model, data_list)
        pk = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))
        m = as_model(lik, pk)

        s = sample(m, random_walk(0.01*[1.0 0; 0 1.0]), 50;
                   n_chains=1, initial=reshape([0.4, 0.2], 2, 1))
        @test s isa Samples
        @test size(s.pars, 1) == 2   # 2 parameters
        @test size(s.pars, 2) == 50  # 50 steps
        @test size(s.pars, 3) == 1   # 1 chain
    end

    @testset "@prior" begin
        pr = @prior begin
            beta ~ Exponential(0.5)
            gamma ~ Exponential(0.2)
        end
        @test pr isa MontyModel
    end

    @testset "Model selection" begin
        @test aic(-100.0, 2) ≈ 204.0
        @test bic(-100.0, 2, 50) > 204.0  # BIC penalises more
    end

    @testset "Categorical" begin
        net = SIR()
        @test net isa EpiNet
        net2 = SEIR()
        @test net2 isa EpiNet
        compiled = compile(net)
        @test compiled isa OdinModel
    end
end
