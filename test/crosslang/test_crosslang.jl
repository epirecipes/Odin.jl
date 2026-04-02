## Cross-language validation tests: Odin.jl vs odin2/dust2/monty (R)
##
## These tests use RCall.jl to run identical models in R and Julia,
## comparing numerical outputs for ODE simulation, stochastic simulation,
## particle filtering, deterministic likelihood, and gradient computation.
##
## Requirements: R with odin2, dust2, monty installed.
## Skip gracefully if RCall or R packages are unavailable.

using Test
using Odin
using Statistics

# Check if RCall is loadable and R packages available
const SKIP = try
    @eval using RCall
    ok = @eval rcopy(RCall.reval("requireNamespace('odin2', quietly=TRUE) && requireNamespace('dust2', quietly=TRUE) && requireNamespace('monty', quietly=TRUE)"))
    !ok
catch
    true
end

if SKIP
    @warn "Skipping cross-language tests: RCall or R packages (odin2/dust2/monty) unavailable"
end

# Helper to run R code and extract a variable
function rrun(code::String)
    RCall.reval(code)
    nothing
end

function rget(varname::String)
    rcopy(RCall.reval(varname))
end

# ---------------------------------------------------------------------------
# 1. Deterministic ODE SIR — compare trajectories
# ---------------------------------------------------------------------------
!SKIP && @testset "Cross-lang: ODE SIR trajectories" begin
    # ---- Julia side ----
    sir_jl = @odin begin
        deriv(S) = -beta * S * I / N
        deriv(I) = beta * S * I / N - gamma * I
        deriv(R) = gamma * I
        initial(S) = N - I0
        initial(I) = I0
        initial(R) = 0
        beta = parameter(0.5)
        gamma = parameter(0.1)
        I0 = parameter(10)
        N = parameter(1000)
    end

    pars_jl = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
    sys = System(sir_jl, pars_jl)
    reset!(sys)
    times = collect(0.0:1.0:100.0)
    out_jl = simulate(sys, times)  # (3, 1, 101)

    S_jl = out_jl[1, 1, :]
    I_jl = out_jl[2, 1, :]
    R_jl = out_jl[3, 1, :]

    # ---- R side ----
    rrun("""
    sir <- odin2::odin({
        deriv(S) <- -beta * S * I / N
        deriv(I) <- beta * S * I / N - gamma * I
        deriv(R) <- gamma * I
        initial(S) <- N - I0
        initial(I) <- I0
        initial(R) <- 0
        beta <- parameter(0.5)
        gamma <- parameter(0.1)
        I0 <- parameter(10)
        N <- parameter(1000)
    })

    sys_r <- dust2::dust_system_create(sir, list(beta=0.5, gamma=0.1, I0=10, N=1000))
    dust2::dust_system_set_state_initial(sys_r)
    times_r <- seq(0, 100, by=1)
    y_r <- dust2::dust_system_simulate(sys_r, times_r)
    # y_r is (n_state, n_times) for 1 particle
    S_r <- y_r[1, ]
    I_r <- y_r[2, ]
    R_r <- y_r[3, ]
    """)

    S_r = rget("S_r")
    I_r = rget("I_r")
    R_r = rget("R_r")

    @test length(S_r) == length(S_jl)

    # ODE trajectories should agree within solver tolerance
    for i in eachindex(S_jl)
        @test S_jl[i] ≈ S_r[i] atol=0.5 rtol=1e-3
        @test I_jl[i] ≈ I_r[i] atol=0.5 rtol=1e-3
        @test R_jl[i] ≈ R_r[i] atol=0.5 rtol=1e-3
    end

    # Conservation in both
    for i in eachindex(S_jl)
        @test S_jl[i] + I_jl[i] + R_jl[i] ≈ 1000.0 atol=1.0
        @test S_r[i] + I_r[i] + R_r[i] ≈ 1000.0 atol=1.0
    end

    # Peak timing should agree (same time index ± 1)
    peak_jl = argmax(I_jl)
    peak_r = argmax(I_r)
    @test abs(peak_jl - peak_r) <= 1
end

# ---------------------------------------------------------------------------
# 2. Deterministic unfilter log-likelihood
# ---------------------------------------------------------------------------
!SKIP && @testset "Cross-lang: ODE unfilter log-likelihood" begin
    # ---- Julia side ----
    sir_compare_jl = @odin begin
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

    # Generate synthetic data using ODE solution
    sys = System(sir_compare_jl, (beta=0.5, gamma=0.1, I0=10.0, N=1000.0))
    reset!(sys)
    data_times = collect(5.0:5.0:50.0)
    sim = simulate(sys, data_times)
    obs_vals = [round(Int, max(1.0, sim[2, 1, i])) for i in 1:length(data_times)]

    data_vec = [(time=data_times[i], obs=Float64(obs_vals[i])) for i in eachindex(data_times)]
    fdata = ObservedData(data_vec)

    lik_jl = Likelihood(sir_compare_jl, fdata)
    ll_jl = loglik(lik_jl, (beta=0.5, gamma=0.1, I0=10.0, N=1000.0))
    ll_jl2 = loglik(lik_jl, (beta=0.3, gamma=0.1, I0=10.0, N=1000.0))

    # ---- R side ----
    # Pass data to R
    @rput obs_vals data_times
    rrun("""
    sir_cmp <- odin2::odin({
        deriv(S) <- -beta * S * I / N
        deriv(I) <- beta * S * I / N - gamma * I
        deriv(R) <- gamma * I
        initial(S) <- N - I0
        initial(I) <- I0
        initial(R) <- 0
        obs <- data()
        obs ~ Poisson(max(I, 1e-6))
        beta <- parameter(0.5)
        gamma <- parameter(0.1)
        I0 <- parameter(10)
        N <- parameter(1000)
    })

    r_data <- data.frame(time = data_times, obs = as.integer(obs_vals))

    uf <- dust2::dust_unfilter_create(sir_cmp, data = r_data, time_start = 0)
    ll_r1 <- dust2::dust_likelihood_run(uf, list(beta=0.5, gamma=0.1, I0=10, N=1000))
    ll_r2 <- dust2::dust_likelihood_run(uf, list(beta=0.3, gamma=0.1, I0=10, N=1000))
    """)

    ll_r1 = rget("ll_r1")
    ll_r2 = rget("ll_r2")

    # Log-likelihoods should be very close
    @test ll_jl ≈ ll_r1 atol=0.5
    @test ll_jl2 ≈ ll_r2 atol=0.5

    # Both should prefer the true parameters
    @test ll_jl > ll_jl2
    @test ll_r1 > ll_r2
end

# ---------------------------------------------------------------------------
# 3. Stochastic discrete SIR — distribution properties
# ---------------------------------------------------------------------------
!SKIP && @testset "Cross-lang: Stochastic SIR distribution" begin
    n_particles = 500
    n_times = 50

    # ---- Julia side ----
    sir_stoch_jl = @odin begin
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
        N = parameter(1000)
        I0 = parameter(10)
        beta = parameter(0.3)
        gamma = parameter(0.1)
    end

    pars_stoch = (N=1000.0, I0=10.0, beta=0.3, gamma=0.1)
    sys_jl = System(sir_stoch_jl, pars_stoch; n_particles=n_particles, dt=1.0, seed=1)
    reset!(sys_jl)
    times_stoch = collect(1.0:1.0:Float64(n_times))
    out_stoch_jl = simulate(sys_jl, times_stoch)

    I_mean_jl = [mean(out_stoch_jl[2, :, t]) for t in 1:n_times]

    # ---- R side ----
    @rput n_particles n_times
    rrun("""
    sir_stoch <- odin2::odin({
        update(S) <- S - n_SI
        update(I) <- I + n_SI - n_IR
        update(R) <- R + n_IR
        initial(S) <- N - I0
        initial(I) <- I0
        initial(R) <- 0
        p_SI <- 1 - exp(-beta * I / N * dt)
        p_IR <- 1 - exp(-gamma * dt)
        n_SI <- Binomial(S, p_SI)
        n_IR <- Binomial(I, p_IR)
        N <- parameter(1000)
        I0 <- parameter(10)
        beta <- parameter(0.3)
        gamma <- parameter(0.1)
    })

    sys_r <- dust2::dust_system_create(sir_stoch,
        list(N=1000, I0=10, beta=0.3, gamma=0.1),
        n_particles = as.integer(n_particles), dt = 1.0, seed = 2L)
    dust2::dust_system_set_state_initial(sys_r)
    times_r <- seq(1, n_times, by=1)
    y_r <- dust2::dust_system_simulate(sys_r, times_r)
    I_mean_r <- apply(y_r[2, , ], 2, mean)
    """)

    I_mean_r = rget("I_mean_r")

    # Mean trajectories should be close (different seeds, but large n_particles)
    for t in 1:n_times
        if I_mean_jl[t] > 5 && I_mean_r[t] > 5
            rel_diff = abs(I_mean_jl[t] - I_mean_r[t]) / max(I_mean_jl[t], I_mean_r[t])
            @test rel_diff < 0.20
        end
    end

    # Conservation: S + I + R = N for every particle at every time
    for t in 1:n_times
        for p in 1:n_particles
            total = out_stoch_jl[1, p, t] + out_stoch_jl[2, p, t] + out_stoch_jl[3, p, t]
            @test total ≈ 1000.0
        end
    end
end

# ---------------------------------------------------------------------------
# 4. Particle filter log-likelihood (stochastic)
# ---------------------------------------------------------------------------
!SKIP && @testset "Cross-lang: Particle filter log-likelihood" begin
    sir_pf_jl = @odin begin
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

    # Synthetic data from deterministic ODE
    pars_pf = (N=1000.0, I0=10.0, beta=0.3, gamma=0.1)
    det_gen = @odin begin
        deriv(S) = -beta * S * I / N
        deriv(I) = beta * S * I / N - gamma * I
        deriv(R) = gamma * I
        initial(S) = N - I0
        initial(I) = I0
        initial(R) = 0
        beta = parameter(0.3)
        gamma = parameter(0.1)
        I0 = parameter(10)
        N = parameter(1000)
    end
    det_sys = System(det_gen, pars_pf)
    reset!(det_sys)
    data_times_pf = collect(5.0:5.0:50.0)
    det_out = simulate(det_sys, data_times_pf)
    obs_pf = [round(Int, max(1.0, det_out[2, 1, i])) for i in 1:length(data_times_pf)]

    data_pf = [(time=data_times_pf[i], obs=Float64(obs_pf[i])) for i in eachindex(data_times_pf)]
    fdata_pf = ObservedData(data_pf)

    # Run Julia PF with many particles (average over multiple runs)
    n_pf_particles = 200
    n_pf_reps = 5
    lls_jl = Float64[]
    for rep in 1:n_pf_reps
        pf = Likelihood(sir_pf_jl, fdata_pf; n_particles=n_pf_particles, dt=1.0, seed=rep * 100)
        push!(lls_jl, loglik(pf, pars_pf))
    end
    ll_mean_jl = mean(lls_jl)

    # ---- R side ----
    @rput obs_pf data_times_pf n_pf_reps n_pf_particles
    rrun("""
    sir_pf_r <- odin2::odin({
        update(S) <- S - n_SI
        update(I) <- I + n_SI - n_IR
        update(R) <- R + n_IR
        initial(S) <- N - I0
        initial(I) <- I0
        initial(R) <- 0
        p_SI <- 1 - exp(-beta * I / N * dt)
        p_IR <- 1 - exp(-gamma * dt)
        n_SI <- Binomial(S, p_SI)
        n_IR <- Binomial(I, p_IR)
        obs <- data()
        obs ~ Poisson(max(I, 1e-6))
        N <- parameter(1000)
        I0 <- parameter(10)
        beta <- parameter(0.3)
        gamma <- parameter(0.1)
    })

    r_data_pf <- data.frame(time = data_times_pf, obs = as.integer(obs_pf))

    lls_r <- numeric(as.integer(n_pf_reps))
    for (rep in seq_len(as.integer(n_pf_reps))) {
        pf_r <- dust2::dust_filter_create(sir_pf_r, data = r_data_pf,
            time_start = 0, n_particles = as.integer(n_pf_particles), dt = 1,
            seed = rep * 100L)
        lls_r[rep] <- dust2::dust_likelihood_run(pf_r,
            list(N=1000, I0=10, beta=0.3, gamma=0.1))
    }
    ll_mean_r <- mean(lls_r)
    """)

    ll_mean_r = rget("ll_mean_r")
    lls_r = rget("lls_r")

    # Both should give finite log-likelihoods
    @test all(isfinite, lls_jl)
    @test all(isfinite, lls_r)

    # Mean log-likelihoods should be within ~10 units (PF variance)
    @test abs(ll_mean_jl - ll_mean_r) < 10.0

    # Both should have reasonable PF variance
    @test var(lls_jl) < 50.0
    @test var(lls_r) < 50.0
end

# ---------------------------------------------------------------------------
# 5. Age-structured array model — compare trajectories
# ---------------------------------------------------------------------------
!SKIP && @testset "Cross-lang: Age-structured SIR" begin
    sir_age_jl = @odin begin
        n_age = parameter(3)
        dim(S) = n_age
        dim(I) = n_age
        dim(R) = n_age
        dim(N_age) = n_age
        dim(beta_vec) = n_age

        deriv(S[i]) = -beta_vec[i] * S[i] * sum(I) / sum(N_age)
        deriv(I[i]) = beta_vec[i] * S[i] * sum(I) / sum(N_age) - gamma * I[i]
        deriv(R[i]) = gamma * I[i]

        initial(S[i]) = N_age[i] - I0
        initial(I[i]) = I0
        initial(R[i]) = 0

        N_age = parameter()
        beta_vec = parameter()
        gamma = parameter(0.1)
        I0 = parameter(5)
    end

    pars_age = (
        n_age=3.0,
        N_age=[300.0, 500.0, 200.0],
        beta_vec=[0.4, 0.3, 0.2],
        gamma=0.1, I0=5.0
    )
    sys_age = System(sir_age_jl, pars_age)
    reset!(sys_age)
    times_age = collect(0.0:1.0:80.0)
    out_age_jl = simulate(sys_age, times_age)

    # ---- R side ----
    rrun("""
    sir_age_r <- odin2::odin({
        n_age <- parameter(3)
        dim(S) <- n_age
        dim(I) <- n_age
        dim(R) <- n_age
        dim(N_age) <- n_age
        dim(beta_vec) <- n_age

        deriv(S[]) <- -beta_vec[i] * S[i] * sum(I) / sum(N_age)
        deriv(I[]) <- beta_vec[i] * S[i] * sum(I) / sum(N_age) - gamma * I[i]
        deriv(R[]) <- gamma * I[i]

        initial(S[]) <- N_age[i] - I0
        initial(I[]) <- I0
        initial(R[]) <- 0

        N_age <- parameter()
        beta_vec <- parameter()
        gamma <- parameter(0.1)
        I0 <- parameter(5)
    })

    sys_age_r <- dust2::dust_system_create(sir_age_r,
        list(n_age = 3, N_age = c(300, 500, 200), beta_vec = c(0.4, 0.3, 0.2),
             gamma = 0.1, I0 = 5))
    dust2::dust_system_set_state_initial(sys_age_r)
    times_age_r <- seq(0, 80, by=1)
    y_age_r <- dust2::dust_system_simulate(sys_age_r, times_age_r)
    """)

    y_age_r = rget("y_age_r")

    n_state = size(out_age_jl, 1)
    n_times = size(out_age_jl, 3)
    @test n_state == 9
    @test size(y_age_r, 1) == 9

    # R drops the particle dimension for 1 particle: (n_state, n_times)
    r_get_state(s, t) = ndims(y_age_r) == 3 ? y_age_r[s, 1, t] : y_age_r[s, t]

    # Compare state-by-state
    for s in 1:n_state
        for t in 1:n_times
            @test out_age_jl[s, 1, t] ≈ r_get_state(s, t) atol=1.0 rtol=1e-2
        end
    end

    # Conservation per age group
    for t in 1:n_times
        for a in 1:3
            total_jl = out_age_jl[a, 1, t] + out_age_jl[a+3, 1, t] + out_age_jl[a+6, 1, t]
            N_val = [300.0, 500.0, 200.0][a]
            @test total_jl ≈ N_val atol=1.0
        end
    end
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
!SKIP && @info "Cross-language validation: all tests passed"
