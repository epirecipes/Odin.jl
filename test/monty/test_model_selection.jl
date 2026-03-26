using Test
using Odin
using Random

@testset "Model Selection" begin

    # ── 1. AIC computation ───────────────────────────────────
    @testset "AIC correctness" begin
        # AIC = -2*LL + 2*k
        @test Odin.aic(-100.0, 3) ≈ 206.0
        @test Odin.aic(-50.0, 1) ≈ 102.0
        @test Odin.aic(0.0, 5) ≈ 10.0
        # More parameters → higher AIC (worse) for same LL
        @test Odin.aic(-100.0, 5) > Odin.aic(-100.0, 3)
    end

    # ── 2. BIC computation ───────────────────────────────────
    @testset "BIC correctness" begin
        # BIC = -2*LL + k*log(n)
        @test Odin.bic(-100.0, 3, 50) ≈ -2*(-100.0) + 3*log(50)
        @test Odin.bic(-100.0, 3, 100) ≈ -2*(-100.0) + 3*log(100)
        # BIC penalises more than AIC for large n
        n_large = 1000
        bic_val = Odin.bic(-100.0, 5, n_large)
        aic_val = Odin.aic(-100.0, 5)
        @test bic_val > aic_val  # BIC penalty > AIC penalty when n > e^2 ≈ 7.4
    end

    # ── 3. AICc converges to AIC as n → ∞ ───────────────────
    @testset "AICc → AIC as n → ∞" begin
        ll = -100.0
        k = 3
        aic = Odin.aic(ll, k)
        for n in [100, 1000, 10000, 100000]
            aicc = Odin.aicc(ll, k, n)
            @test aicc > aic  # correction is always positive
            @test abs(aicc - aic) < 2.0 * k * (k + 1) / (n - k - 1) + 1e-10
        end
        # Very large n → difference vanishes
        aicc_large = Odin.aicc(ll, k, 1_000_000)
        @test abs(aicc_large - aic) < 0.001
    end

    # ── 4. AICc error for small samples ──────────────────────
    @testset "AICc error for small n" begin
        @test_throws ErrorException Odin.aicc(-100.0, 5, 5)   # n = k
        @test_throws ErrorException Odin.aicc(-100.0, 5, 6)   # n = k+1
    end

    # ── 5. DIC from known posterior (Gaussian) ───────────────
    @testset "DIC from Gaussian posterior" begin
        # Create synthetic Samples from a known Normal(3, 1) posterior
        # with a simple Gaussian likelihood: ll(θ) = -0.5*(θ - 3)^2
        n_pars = 1
        n_samples = 5000
        n_chains = 2
        rng = Random.MersenneTwister(42)

        pars = zeros(Float64, n_pars, n_samples, n_chains)
        density = zeros(Float64, n_samples, n_chains)
        for c in 1:n_chains
            for s in 1:n_samples
                theta = 3.0 + randn(rng)
                pars[1, s, c] = theta
                density[s, c] = -0.5 * (theta - 3.0)^2
            end
        end

        samples = Odin.Samples(
            pars, density,
            zeros(Float64, n_pars, n_chains),
            ["theta"],
            Dict{Symbol, Any}(),
        )

        likelihood_fn = θ -> -0.5 * (θ[1] - 3.0)^2

        result = Odin.dic(samples, likelihood_fn)
        @test haskey(result, :dic)
        @test haskey(result, :p_d)
        @test haskey(result, :d_bar)
        @test haskey(result, :d_theta_bar)

        # For Gaussian: D(θ̄) ≈ 0 (at true mean), D̄ ≈ 1 (mean of χ²₁)
        # So pD ≈ 1, DIC ≈ 2
        @test result.d_theta_bar ≈ 0.0 atol=0.5  # deviance at mean ≈ 0
        @test result.p_d ≈ 1.0 atol=0.5  # effective parameters ≈ 1
        @test result.dic ≈ 2.0 atol=1.0  # DIC ≈ 2
    end

    # ── 6. WAIC from known pointwise log-likelihoods ─────────
    @testset "WAIC correctness" begin
        n_obs = 20
        n_samples = 1000

        # Generate pointwise log-likelihoods: each observation has
        # ll ~ Normal(-1, 0.1) across posterior samples
        rng = Random.MersenneTwister(123)
        pointwise_ll = zeros(Float64, n_obs, n_samples)
        for i in 1:n_obs
            for s in 1:n_samples
                pointwise_ll[i, s] = -1.0 + 0.1 * randn(rng)
            end
        end

        result = Odin.waic(pointwise_ll)
        @test haskey(result, :waic)
        @test haskey(result, :p_waic)
        @test haskey(result, :lppd)
        @test haskey(result, :pointwise)

        # lppd should be close to n_obs * (-1) ≈ -20 (log-mean-exp of Normal(-1, 0.1))
        @test result.lppd ≈ -20.0 atol=1.0
        # p_waic = sum of variances ≈ n_obs * 0.01
        @test result.p_waic ≈ n_obs * 0.01 atol=0.5
        # WAIC = -2*(lppd - p_waic) > 0
        @test result.waic > 0
        @test length(result.pointwise) == n_obs
    end

    # ── 7. Akaike weights sum to 1, prefer better model ─────
    @testset "Akaike weights" begin
        aic_vals = [200.0, 210.0, 220.0]
        w = Odin.akaike_weights(aic_vals)
        @test length(w) == 3
        @test sum(w) ≈ 1.0
        # Best model (lowest AIC) gets highest weight
        @test w[1] > w[2] > w[3]
        # All weights non-negative
        @test all(w .>= 0.0)

        # Equal AIC → equal weights
        w_eq = Odin.akaike_weights([100.0, 100.0, 100.0])
        @test all(w_eq .≈ 1.0/3.0)
    end

    # ── 8. Model comparison table ────────────────────────────
    @testset "Model comparison table" begin
        mc = Odin.compare(;
            n_observations=50,
            model_a=(ll=-120.0, k=3),
            model_b=(ll=-115.0, k=4, dic=240.0),
            model_c=(ll=-110.0, k=6, dic=232.0, waic=235.0),
        )

        @test mc isa Odin.ModelComparison
        @test length(mc.models) == 3
        @test mc.n_observations == 50

        # Sorted by AIC — lower is better
        for i in 1:(length(mc.aic) - 1)
            @test mc.aic[i] <= mc.aic[i + 1]
        end

        # Weights sum to 1
        @test sum(mc.weights_aic) ≈ 1.0
        @test sum(mc.weights_bic) ≈ 1.0

        # DIC/WAIC preserved where provided
        dic_vals = filter(x -> x !== nothing, mc.dic)
        @test length(dic_vals) == 2

        # Test show method works (captures output)
        buf = IOBuffer()
        show(buf, mc)
        output = String(take!(buf))
        @test occursin("Model Comparison", output)
        @test occursin("AIC", output)
    end

    # ── 9. Pointwise log-likelihoods sum to total ────────────
    @testset "Pointwise LL matches total (unfilter)" begin
        sir = @odin begin
            deriv(S) = -beta * S * I / N
            deriv(I) = beta * S * I / N - gamma * I
            deriv(R) = gamma * I
            initial(S) = N - I0
            initial(I) = I0
            initial(R) = 0

            cases = data()
            cases ~ Poisson(I + 1e-6)

            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        times = collect(1.0:1.0:20.0)

        # Simulate to get "observed" data
        sim = simulate(sir, pars; times=[0.0; times], dt=0.25)
        cases = [max(1.0, round(sim[2, 1, i+1])) for i in 1:length(times)]

        data_tuples = [(time=times[i], cases=cases[i]) for i in 1:length(times)]
        fdata = ObservedData(data_tuples)

        uf = Likelihood(sir, fdata; time_start=0.0)

        total_ll = loglik(uf, pars)
        pointwise_ll = loglik_pointwise(uf, pars)

        @test length(pointwise_ll) == length(times)
        @test sum(pointwise_ll) ≈ total_ll atol=1e-8
    end

    # ── 10. LOO-CV basic correctness ─────────────────────────
    @testset "LOO-CV" begin
        n_obs = 30
        n_samples = 500

        rng = Random.MersenneTwister(456)
        pointwise_ll = zeros(Float64, n_obs, n_samples)
        for i in 1:n_obs
            for s in 1:n_samples
                pointwise_ll[i, s] = -0.5 + 0.05 * randn(rng)
            end
        end

        result = Odin.loo(pointwise_ll)
        @test haskey(result, :loo)
        @test haskey(result, :p_loo)
        @test haskey(result, :pointwise)
        @test haskey(result, :k_diagnostics)
        @test length(result.pointwise) == n_obs
        @test length(result.k_diagnostics) == n_obs
        @test result.loo > 0  # -2 * lppd_loo
        # k diagnostics should be small for well-behaved posteriors
        @test all(result.k_diagnostics .< 1.0)
    end

    # ── 11. Pointwise filter LL matches total ────────────────
    @testset "Pointwise LL matches total (filter)" begin
        sir_stoch = @odin begin
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

            cases = data()
            cases ~ Poisson(I + 1e-6)

            beta = parameter(0.5)
            gamma = parameter(0.1)
            I0 = parameter(10)
            N = parameter(1000)
        end

        pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
        times = collect(1.0:1.0:15.0)

        sim = simulate(sir_stoch, pars; times=[0.0; times], dt=1.0, seed=1)
        cases = [max(1.0, round(sim[2, 1, i+1])) for i in 1:length(times)]
        data_tuples = [(time=times[i], cases=cases[i]) for i in 1:length(times)]
        fdata = ObservedData(data_tuples)

        filt = Likelihood(sir_stoch, fdata;
                                  time_start=0.0, n_particles=500, dt=1.0, seed=42)

        total_ll = loglik(filt, pars)
        # Need to recreate/reset since filter mutates state
        filt2 = Likelihood(sir_stoch, fdata;
                                   time_start=0.0, n_particles=500, dt=1.0, seed=42)
        pointwise_ll = loglik_pointwise(filt2, pars)

        @test length(pointwise_ll) == length(times)
        @test sum(pointwise_ll) ≈ total_ll atol=1e-6
    end

    # ── 12. Compare two models (SIR vs SEIR) by IC ──────────
    @testset "SIR vs SEIR model comparison" begin
        # Simulate SIR-like log-likelihoods
        ll_sir = -85.0   # fewer parameters, good fit
        ll_seir = -82.0  # more parameters, slightly better fit

        k_sir = 3
        k_seir = 5
        n_obs = 40

        mc = Odin.compare(;
            n_observations=n_obs,
            sir=(ll=ll_sir, k=k_sir),
            seir=(ll=ll_seir, k=k_seir),
        )

        # Both AIC and BIC should be computable
        @test length(mc.aic) == 2
        @test length(mc.bic) == 2
        @test length(mc.weights_aic) == 2
        @test length(mc.weights_bic) == 2

        # BIC penalises more → may prefer simpler model for moderate n
        aic_sir = Odin.aic(ll_sir, k_sir)
        aic_seir = Odin.aic(ll_seir, k_seir)
        bic_sir = Odin.bic(ll_sir, k_sir, n_obs)
        bic_seir = Odin.bic(ll_seir, k_seir, n_obs)

        @test aic_sir ≈ 176.0  # -2*(-85) + 2*3 = 176
        @test aic_seir ≈ 174.0 # -2*(-82) + 2*5 = 174
        @test bic_sir ≈ -2*(-85.0) + 3*log(40)
        @test bic_seir ≈ -2*(-82.0) + 5*log(40)
    end
end
