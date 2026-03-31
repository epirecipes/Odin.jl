using Test
using Odin

@testset "Mixed scalar and array comparison" begin
    mixed = @odin begin
        dim(S) = n_age
        dim(I) = n_age
        dim(R) = n_age
        dim(age_cases) = n_age

        deriv(S[i]) = -beta * S[i] * sum(I) / N
        deriv(I[i]) = beta * S[i] * sum(I) / N - gamma * I[i]
        deriv(R[i]) = gamma * I[i]

        initial(S[i]) = S0[i]
        initial(I[i]) = I0[i]
        initial(R[i]) = 0

        total_I = sum(I)
        total_cases = data()
        total_cases ~ Poisson(max(total_I, 1e-6))

        age_cases[i] = data()
        age_cases[i] ~ Poisson(max(I[i], 1e-6))

        beta = parameter(0.4)
        gamma = parameter(0.2)
        N = parameter(3000)
        n_age = parameter(3)
        dim(S0) = n_age
        dim(I0) = n_age
        S0[i] = parameter(990)
        I0[i] = parameter(10)
    end

    @test mixed isa Odin.OdinModel

    pars = (
        beta = 0.4,
        gamma = 0.2,
        N = 3000.0,
        n_age = 3,
        S0 = [990.0, 990.0, 990.0],
        I0 = [10.0, 10.0, 10.0],
    )

    sys = System(mixed, pars; n_particles=1)
    reset!(sys)
    times = collect(1.0:1.0:10.0)
    result = simulate(sys, times)

    data_list = NamedTuple[]
    for ti in 1:length(times)
        I_vals = result[4:6, 1, ti]
        total_I = sum(I_vals)
        push!(data_list, (
            time = times[ti],
            total_cases = max(round(Int, total_I), 1) * 1.0,
            age_cases = max.(round.(Int, I_vals), 1) .* 1.0,
        ))
    end

    fdata = ObservedData(data_list)
    uf = Likelihood(mixed, fdata)
    ll = loglik(uf, pars)
    @test isfinite(ll)
    @test ll < 0
    @test ll < -5
end
