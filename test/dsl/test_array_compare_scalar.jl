using Test
using Odin

@testset "Scalar comparison (regression)" begin
    sir_scalar = @odin begin
        deriv(S) = -beta * S * I / N
        deriv(I) = beta * S * I / N - gamma * I
        deriv(R) = gamma * I
        initial(S) = N - I0
        initial(I) = I0
        initial(R) = 0

        cases = data()
        cases ~ Poisson(max(I, 1e-6))

        N = parameter(1000)
        I0 = parameter(10)
        beta = parameter(0.4)
        gamma = parameter(0.2)
    end

    pars = (beta=0.4, gamma=0.2, I0=10.0, N=1000.0)
    sys = System(sir_scalar, pars; n_particles=1)
    reset!(sys)
    times = collect(1.0:1.0:10.0)
    result = simulate(sys, times)

    data_list = NamedTuple[]
    for ti in 1:length(times)
        I_val = result[2, 1, ti]
        push!(data_list, (time=times[ti], cases=max(round(Int, I_val), 1) * 1.0))
    end

    fdata = ObservedData(data_list)
    uf = Likelihood(sir_scalar, fdata)
    ll = loglik(uf, pars)
    @test isfinite(ll)
    @test ll < 0
end
