using Test
using Odin

@testset "Dust Particle Filter" begin
    @testset "ObservedData creation" begin
        data = [(time=1.0, cases=5), (time=3.0, cases=10), (time=2.0, cases=7)]
        fd = Odin.ObservedData(data)

        @test fd.times == [1.0, 2.0, 3.0]  # sorted
        @test length(fd.data) == 3
        @test fd.data[1].cases == 5
        @test fd.data[2].cases == 7
        @test fd.data[3].cases == 10
    end

    @testset "ObservedData is parametric" begin
        data = [(time=1.0, cases=5.0), (time=2.0, cases=7.0)]
        fd = Odin.ObservedData(data)
        @test typeof(fd) <: Odin.ObservedData{<:NamedTuple}
        @test eltype(fd.data) <: NamedTuple{(:cases,)}
    end

    @testset "Unfilter with gradient" begin
        # ODE SIR model with data comparison
        sir_compare = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            obs = data()
            obs ~ Poisson(max(I, 1e-6))
            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        # Generate data from ODE SIR
        sys = System(sir_compare, (beta=0.5, gamma=0.1, I0=10.0, N=1000.0))
        reset!(sys)
        times = collect(5.0:5.0:50.0)
        result = simulate(sys, times)
        # result is n_state × n_particles × n_times for ODE; use I (index 2)
        data_vec = [(time=times[i], obs=max(1.0, result[2,1,i])) for i in 1:length(times)]
        fdata = Odin.ObservedData(data_vec)

        unfilter = Likelihood(sir_compare, fdata)
        packer = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))

        # Test that dust_likelihood_monty provides gradient
        ll_model = as_model(unfilter, packer)
        @test ll_model.gradient !== nothing
        @test ll_model.properties.has_gradient == true

        # Test gradient is finite
        x = [0.5, 0.1]
        ll = ll_model.density(x)
        grad = ll_model.gradient(x)
        @test isfinite(ll)
        @test length(grad) == 2
        @test all(isfinite, grad)
    end
end
