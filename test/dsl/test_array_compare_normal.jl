using Test
using Odin

@testset "Per-age-group Normal comparison" begin
    sir_normal = @odin begin
        dim(S) = n_age
        dim(I) = n_age
        dim(R) = n_age
        dim(obs) = n_age

        deriv(S[i]) = -beta * S[i] * sum(I) / N
        deriv(I[i]) = beta * S[i] * sum(I) / N - gamma * I[i]
        deriv(R[i]) = gamma * I[i]

        initial(S[i]) = S0[i]
        initial(I[i]) = I0[i]
        initial(R[i]) = 0

        obs[i] = data()
        obs[i] ~ Normal(I[i], sigma)

        beta = parameter(0.4)
        gamma = parameter(0.2)
        N = parameter(3000)
        sigma = parameter(5.0)
        n_age = parameter(3)
        dim(S0) = n_age
        dim(I0) = n_age
        S0[i] = parameter(990)
        I0[i] = parameter(10)
    end

    @test sir_normal isa Odin.OdinModel

    pars = (
        beta = 0.4,
        gamma = 0.2,
        N = 3000.0,
        sigma = 5.0,
        n_age = 3,
        S0 = [990.0, 990.0, 990.0],
        I0 = [10.0, 10.0, 10.0],
    )

    sys = System(sir_normal, pars; n_particles=1)
    reset!(sys)
    times = collect(1.0:1.0:10.0)
    result = simulate(sys, times)

    data_list = NamedTuple[]
    for ti in 1:length(times)
        I_vals = result[4:6, 1, ti]
        obs_vals = I_vals .+ randn(3) .* 5.0
        push!(data_list, (time=times[ti], obs=obs_vals))
    end

    fdata = ObservedData(data_list)
    uf = Likelihood(sir_normal, fdata)
    ll = loglik(uf, pars)
    @test isfinite(ll)
    @test ll < 0
end
