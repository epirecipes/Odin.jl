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

    @testset "ObservedData validation" begin
        @test_throws ArgumentError Odin.ObservedData(NamedTuple[])
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

    @testset "Filter stores trajectories and is reproducible with seed" begin
        sir_discrete = @odin begin
            update(S) = S - n_SI
            update(I) = I + n_SI - n_IR
            update(R) = R + n_IR
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            p_SI = 1 - exp(-beta * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            n_SI = Binomial(S, p_SI)
            n_IR = Binomial(I, p_IR)
            obs = data()
            obs ~ Poisson(max(I, 1e-6))
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.3)
            gamma = parameter(0.1)
        end

        pars = (N=1000.0, I0=10.0, beta=0.3, gamma=0.1)
        sys = System(sir_discrete, pars; n_particles=1, dt=1.0, seed=11)
        reset!(sys)
        times = collect(1.0:1.0:5.0)
        sim = simulate(sys, times)
        data_vec = [(time=times[i], obs=max(1.0, sim[2, 1, i])) for i in eachindex(times)]
        fdata = Odin.ObservedData(data_vec)

        pf = Likelihood(sir_discrete, fdata; n_particles=32, dt=1.0, seed=99, save_trajectories=true)
        ll1 = loglik(pf, pars)
        traj1 = last_trajectories(pf.inner)
        ll2 = loglik(pf, pars)
        traj2 = last_trajectories(pf.inner)

        @test isfinite(ll1)
        @test ll1 == ll2
        @test traj1 !== nothing
        @test size(traj1) == (3, 32, 5)
        @test traj1 == traj2
    end

    @testset "Filter respects non-zero time_start" begin
        sir_ode = @odin begin
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
        sys = System(sir_ode, pars; time=5.0)
        reset!(sys)
        sim = simulate(sys, 6.0:1.0:10.0)
        data_vec = [(time=Float64(t), obs=max(1.0, sim[2, 1, i])) for (i, t) in enumerate(6.0:1.0:10.0)]
        fdata = ObservedData(data_vec)

        ll_time5 = loglik(Likelihood(sir_ode, fdata; time_start=5.0), pars)
        ll_time0 = loglik(Likelihood(sir_ode, fdata; time_start=0.0), pars)
        @test ll_time5 > ll_time0
    end

    @testset "Unfilter handles parameter-dependent array dimensions" begin
        # Regression: models with dim(X)=n where n=parameter() have
        # model.n_state==0 at construction; the unfilter must resize its
        # state cache at runtime once actual parameters are supplied.
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
            dim(S0) = n_age
            dim(I0) = n_age
            S0[i] = parameter(330)
            I0[i] = parameter(5)
            n_age = parameter(3)
            beta = parameter(0.5)
            gamma = parameter(0.1)
            N = parameter(1000)
        end

        @test sir_age.model.n_state == 0  # static size unknown

        pars = (n_age=3, beta=0.5, gamma=0.1, N=1000.0,
                S0=[330.0, 330.0, 330.0], I0=[5.0, 3.0, 2.0])

        # Simulate and build synthetic data
        sim_times = collect(1.0:1.0:20.0)
        sim = simulate(sir_age, pars; times=sim_times, seed=42)
        @test size(sim, 1) == 9  # 3 S + 3 I + 3 R
        obs_times = collect(5.0:5.0:20.0)
        data_vec = NamedTuple[]
        for t in obs_times
            ti = findfirst(==(t), sim_times)
            I_vals = sim[4:6, 1, ti]
            obs = max.(round.(I_vals), 1.0)
            push!(data_vec, (time=t, cases=obs))
        end
        fdata = ObservedData(data_vec)

        uf = Likelihood(sir_age, fdata; time_start=0.0)
        ll = loglik(uf, pars)
        @test isfinite(ll)

        # Verify trajectory shape is correct (n_state × n_obs)
        traj = last_trajectories(uf.inner)
        @test traj !== nothing
        @test size(traj, 1) == 9
        @test size(traj, 2) == length(obs_times)

        # Verify trajectory matches independent simulation at obs times
        sim_full = simulate(sir_age, pars; times=obs_times, seed=42)
        for (j, _) in enumerate(obs_times)
            @test traj[:, j] ≈ sim_full[:, 1, j] atol=1e-6
        end
    end

    @testset "Grouped packer works with grouped likelihood" begin
        model = @odin begin
            deriv(x) = -rate * x
            initial(x) = x0
            obs = data()
            obs ~ Normal(x, 1.0)
            rate = parameter(0.1)
            x0 = parameter(1.0)
        end

        pars1 = (rate=0.1, x0=1.0)
        pars2 = (rate=0.2, x0=1.0)
        times = [1.0, 2.0, 3.0]
        sim1 = simulate(model, pars1, times)
        sim2 = simulate(model, pars2, times)
        grouped_data = [
            (time=times[i], group=:g1, obs=sim1[1, 1, i]) for i in eachindex(times)
        ]
        append!(grouped_data, [
            (time=times[i], group=:g2, obs=sim2[1, 1, i]) for i in eachindex(times)
        ])

        lik = Likelihood(model, ObservedData(grouped_data; group_field=:group))
        packer = GroupedPacker([:g1, :g2]; shared=[:x0], varied=[:rate])
        ll_model = as_model(lik, packer)

        good = ll_model([1.0, 0.1, 0.2])
        bad = ll_model([1.0, 0.2, 0.1])
        @test isfinite(good)
        @test good > bad
    end

    @testset "Single-particle filter (n_particles=1)" begin
        sir_discrete = @odin begin
            update(S) = S - n_SI
            update(I) = I + n_SI - n_IR
            update(R) = R + n_IR
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0
            p_SI = 1 - exp(-beta * I / N * dt)
            p_IR = 1 - exp(-gamma * dt)
            n_SI = Binomial(S, p_SI)
            n_IR = Binomial(I, p_IR)
            obs = data()
            obs ~ Poisson(max(I, 1e-6))
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.3)
            gamma = parameter(0.1)
        end

        pars = (N=1000.0, I0=10.0, beta=0.3, gamma=0.1)
        sys = System(sir_discrete, pars; n_particles=1, dt=1.0, seed=11)
        reset!(sys)
        times = collect(1.0:1.0:5.0)
        sim = simulate(sys, times)
        data_vec = [(time=times[i], obs=max(1.0, sim[2, 1, i])) for i in eachindex(times)]
        fdata = Odin.ObservedData(data_vec)

        # n_particles = 1 should run without divide-by-zero or shape errors.
        pf = Likelihood(sir_discrete, fdata; n_particles=1, dt=1.0, seed=42)
        ll = loglik(pf, pars)
        @test isfinite(ll)
    end

    @testset "Filter validates positive n_particles and dt" begin
        sir_discrete = @odin begin
            update(S) = S
            update(I) = I
            initial(S) = 100
            initial(I) = 1
            obs = data()
            obs ~ Poisson(max(I, 1e-6))
        end
        fdata = Odin.ObservedData([(time=1.0, obs=1.0)])

        @test_throws ArgumentError Likelihood(sir_discrete, fdata; n_particles=0, dt=1.0)
        @test_throws ArgumentError Likelihood(sir_discrete, fdata; n_particles=-1, dt=1.0)
        @test_throws ArgumentError Likelihood(sir_discrete, fdata; n_particles=8, dt=0.0)
        @test_throws ArgumentError Likelihood(sir_discrete, fdata; n_particles=8, dt=-0.1)
    end

    @testset "Filter handles all -Inf log-weights without NaN" begin
        # Force log-weight = -Inf by using a Poisson(0) observation model with
        # positive observations: logpdf(Poisson(0), k>0) == -Inf for every particle.
        sir_discrete = @odin begin
            update(S) = S
            update(I) = I
            initial(S) = 100
            initial(I) = 0
            obs = data()
            obs ~ Poisson(I)
        end
        fdata = Odin.ObservedData([(time=t, obs=5.0) for t in 1.0:1.0:3.0])
        pf = Likelihood(sir_discrete, fdata; n_particles=16, dt=1.0, seed=7)
        ll = loglik(pf, NamedTuple())
        @test ll == -Inf            # correct log-likelihood
        @test !isnan(ll)             # explicitly: not NaN (regression test)
    end
end
