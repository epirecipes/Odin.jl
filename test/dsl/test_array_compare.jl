using Test
using Odin
using Statistics

@testset "Array Comparisons" begin

    # ── Age-structured SIR with per-age-group Poisson observations ────────

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

        # Simulate to generate data
        sys = System(sir_age, pars; n_particles=1)
        reset!(sys)
        times = collect(1.0:1.0:20.0)
        result = simulate(sys, times)

        @test size(result, 1) >= 9  # 3 S + 3 I + 3 R states
        @test size(result, 3) == 20  # 20 time points

        # Build per-age-group observation data from simulated I values
        data_list = NamedTuple[]
        for ti in 1:length(times)
            I_vals = result[4:6, 1, ti]  # I[1:3] for particle 1
            obs = max.(round.(Int, I_vals), 1) .* 1.0
            push!(data_list, (time=times[ti], cases=obs))
        end

        # Create unfilter with array data
        fdata = ObservedData(data_list)
        @test length(fdata.times) == 20
        @test fdata.data[1].cases isa AbstractVector

        uf = Likelihood(sir_age, fdata)
        ll = loglik(uf, pars)
        @test isfinite(ll)
        @test ll < 0  # log-likelihood should be negative
        @test ll > -1e10  # but not absurdly negative

        # Verify that changing parameters changes the likelihood
        pars2 = merge(pars, (beta=0.8,))
        ll2 = loglik(uf, pars2)
        @test isfinite(ll2)
        @test ll2 != ll  # different parameters should give different likelihood
    end

    # ── Per-age-group Normal comparison ───────────────────────────────────

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

        # Simulate and generate noisy observations
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

    # ── Scalar comparison still works ─────────────────────────────────────

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

    # ── Mixture: scalar + array comparison ────────────────────────────────

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

            # Scalar comparison on total cases
            total_I = sum(I)
            total_cases = data()
            total_cases ~ Poisson(max(total_I, 1e-6))

            # Per-age-group comparison
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

        # The total + per-age likelihood should be more negative than either alone
        @test ll < -5  # reasonable lower bound
    end
end
