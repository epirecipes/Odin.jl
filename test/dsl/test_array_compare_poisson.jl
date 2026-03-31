using Test
using Odin

@testset "Per-age-group Poisson comparison" begin
    sir_age = @odin begin
        dim(S) = n_age
        dim(I) = n_age
        dim(R) = n_age
        dim(cases) = n_age

        deriv(S[i]) = -beta * S[i] * sum(I) / N
        deriv(I[i]) = beta * S[i] * sum(I) / N - gamma * I[i]
        deriv(R[i]) = gamma * I[i]

        initial(S[i]) = S0[i]
        initial(I[i]) = I0[i]
        initial(R[i]) = 0

        cases[i] = data()
        cases[i] ~ Poisson(max(I[i], 1e-6))

        beta = parameter(0.4)
        gamma = parameter(0.2)
        N = parameter(3000)
        n_age = parameter(3)
        dim(S0) = n_age
        dim(I0) = n_age
        S0[i] = parameter(990)
        I0[i] = parameter(10)
    end

    @test sir_age isa Odin.OdinModel

    pars = (
        beta = 0.4,
        gamma = 0.2,
        N = 3000.0,
        n_age = 3,
        S0 = [990.0, 990.0, 990.0],
        I0 = [10.0, 10.0, 10.0],
    )

    sys = System(sir_age, pars; n_particles=1)
    reset!(sys)
    times = collect(1.0:1.0:20.0)
    result = simulate(sys, times)

    @test size(result, 1) >= 9
    @test size(result, 3) == 20

    data_list = NamedTuple[]
    for ti in 1:length(times)
        I_vals = result[4:6, 1, ti]
        obs = max.(round.(Int, I_vals), 1) .* 1.0
        push!(data_list, (time=times[ti], cases=obs))
    end

    fdata = ObservedData(data_list)
    @test length(fdata.times) == 20
    @test fdata.data[1].cases isa AbstractVector

    uf = Likelihood(sir_age, fdata)
    ll = loglik(uf, pars)
    @test isfinite(ll)
    @test ll < 0
    @test ll > -1e10

    pars2 = merge(pars, (beta=0.8,))
    ll2 = loglik(uf, pars2)
    @test isfinite(ll2)
    @test ll2 != ll
end
