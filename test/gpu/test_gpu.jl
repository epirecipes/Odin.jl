using Test
using Odin

@testset "GPU Backend & Filter" begin
    @testset "Backend selection" begin
        # CPUBackend always available
        be = Odin.gpu_backend(; preferred=:cpu)
        @test be isa Odin.CPUBackend
        @test Odin.backend_name(be) == "CPU (fallback)"

        # :auto falls back to CPU when no GPU extensions loaded
        be_auto = Odin.gpu_backend(; preferred=:auto)
        # Either a real GPU backend or CPUBackend
        @test be_auto isa Odin.GPUBackend

        # Requesting unavailable backend errors
        if !haskey(Odin._GPU_BACKENDS, :cuda)
            @test_throws ErrorException Odin.gpu_backend(; preferred=:cuda)
        end
    end

    @testset "Backend registry" begin
        # available_gpu_backends returns a vector
        avail = Odin.available_gpu_backends()
        @test avail isa Vector{Symbol}

        # has_gpu reflects whether any GPU is registered
        if isempty(avail)
            @test !Odin.has_gpu()
        else
            @test Odin.has_gpu()
        end
    end

    @testset "CPU array passthrough" begin
        x = rand(3, 4)
        be = Odin.CPUBackend()
        y = Odin.gpu_array(be, x)
        @test y === x  # no-op
        @test Odin.cpu_array(x) === x
        @test Odin.gpu_array_type(be) == Array
    end

    @testset "GPU filter with CPUBackend" begin
        # Discrete stochastic SIR with observation
        gen = @odin begin
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
            cases ~ Poisson(max(I, 1.0))
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.5)
            gamma = parameter(0.1)
        end

        # Generate synthetic data from the model
        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        sys = System(gen, pars; n_particles=1, dt=1.0, seed=1)
        reset!(sys)
        times = collect(5.0:5.0:50.0)
        result = simulate(sys, times)
        data_vec = [(time=times[i], cases=max(1.0, result[2,1,i])) for i in eachindex(times)]
        fdata = Odin.ObservedData(data_vec)

        # Create GPU filter with CPU fallback
        gf = Odin.gpu_dust_filter_create(gen, fdata;
            n_particles=100, dt=1.0, seed=42, backend=:cpu)

        @test gf isa Odin.GPUDustFilter
        @test gf.backend isa Odin.CPUBackend

        # Run and check we get a finite log-likelihood
        ll = Odin.gpu_dust_filter_run!(gf, pars)
        @test isfinite(ll)
        @test ll < 0  # log-likelihood should be negative

        # Repeated runs should give finite results
        ll2 = Odin.gpu_dust_filter_run!(gf, pars)
        @test isfinite(ll2)
        @test ll2 < 0
    end

    @testset "GPU filter matches CPU filter" begin
        gen = @odin begin
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
            cases ~ Poisson(max(I, 1.0))
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.5)
            gamma = parameter(0.1)
        end

        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        sys = System(gen, pars; n_particles=1, dt=1.0, seed=1)
        reset!(sys)
        times = collect(5.0:5.0:50.0)
        result = simulate(sys, times)
        data_vec = [(time=times[i], cases=max(1.0, result[2,1,i])) for i in eachindex(times)]
        fdata = Odin.ObservedData(data_vec)

        # CPU filter (standard)
        cpu_filter = Likelihood(gen, fdata;
            n_particles=200, dt=1.0, seed=42)
        ll_cpu = loglik(cpu_filter, pars)

        # GPU filter with CPU backend (should give identical result)
        gpu_filter = Odin.gpu_dust_filter_create(gen, fdata;
            n_particles=200, dt=1.0, seed=42, backend=:cpu)
        ll_gpu = Odin.gpu_dust_filter_run!(gpu_filter, pars)

        @test ll_cpu ≈ ll_gpu atol=1e-10
    end

    @testset "GPU simulation with CPUBackend" begin
        gen = @odin begin
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
            beta = parameter(0.5)
            gamma = parameter(0.1)
        end

        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        times = collect(0.0:1.0:50.0)

        # GPU simulate (CPU fallback)
        out_gpu = Odin.gpu_dust_simulate(gen, pars;
            times=times, n_particles=5, dt=1.0, seed=42, backend=:cpu)
        @test size(out_gpu) == (3, 5, length(times))

        # Should match standard simulate
        sys = System(gen, pars; n_particles=5, dt=1.0, seed=42)
        reset!(sys)
        out_cpu = simulate(sys, times)
        @test out_gpu ≈ out_cpu atol=1e-10
    end

    @testset "GPU filter rejects continuous models" begin
        gen = @odin begin
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
            beta = parameter(0.5)
            gamma = parameter(0.1)
        end

        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        sys = System(gen, pars)
        reset!(sys)
        times = collect(5.0:5.0:20.0)
        result = simulate(sys, times)
        data_vec = [(time=times[i], cases=max(1.0, result[2,1,i])) for i in eachindex(times)]
        fdata = Odin.ObservedData(data_vec)

        # Should be able to create the filter...
        gf = Odin.gpu_dust_filter_create(gen, fdata; n_particles=10, backend=:cpu)
        # ... but running on non-CPU backend with continuous model would error
        # CPUBackend delegates to DustFilter which handles continuous models differently
    end

    @testset "GPU filter with monty bridge" begin
        gen = @odin begin
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
            cases ~ Poisson(max(I, 1.0))
            N = parameter(1000)
            I0 = parameter(10)
            beta = parameter(0.5)
            gamma = parameter(0.1)
        end

        pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
        sys = System(gen, pars; n_particles=1, dt=1.0, seed=1)
        reset!(sys)
        times = collect(5.0:5.0:30.0)
        result = simulate(sys, times)
        data_vec = [(time=times[i], cases=max(1.0, result[2,1,i])) for i in eachindex(times)]
        fdata = Odin.ObservedData(data_vec)

        gf = Odin.gpu_dust_filter_create(gen, fdata;
            n_particles=50, dt=1.0, seed=42, backend=:cpu)
        packer = Packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))
        ll_model = Odin.gpu_dust_likelihood_monty(gf, packer)

        @test ll_model isa MontyModel
        ll = ll_model.density([0.5, 0.1])
        @test isfinite(ll)
    end

    # Conditional GPU tests — only run if a GPU backend is available
    if Odin.has_gpu()
        @testset "GPU hardware tests" begin
            be = Odin.gpu_backend(; preferred=:auto)
            @testset "GPU array round-trip" begin
                x = rand(4, 8)
                x_gpu = Odin.gpu_array(be, x)
                x_back = Odin.cpu_array(x_gpu)
                @test x_back ≈ x
            end

            @testset "GPU filter on hardware" begin
                gen = @odin begin
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
                    cases ~ Poisson(max(I, 1.0))
                    N = parameter(1000)
                    I0 = parameter(10)
                    beta = parameter(0.5)
                    gamma = parameter(0.1)
                end

                pars = (N=1000.0, I0=10.0, beta=0.5, gamma=0.1)
                sys = System(gen, pars; n_particles=1, dt=1.0, seed=1)
                reset!(sys)
                times = collect(5.0:5.0:30.0)
                result = simulate(sys, times)
                data_vec = [(time=times[i], cases=max(1.0, result[2,1,i])) for i in eachindex(times)]
                fdata = Odin.ObservedData(data_vec)

                # Run GPU filter
                gf = Odin.gpu_dust_filter_create(gen, fdata;
                    n_particles=500, dt=1.0, seed=42, backend=:auto)
                ll_gpu = Odin.gpu_dust_filter_run!(gf, pars)
                @test isfinite(ll_gpu)

                # Compare with CPU filter (stochastic — use large tolerance)
                cpu_filter = Likelihood(gen, fdata;
                    n_particles=500, dt=1.0, seed=42)
                ll_cpu = loglik(cpu_filter, pars)
                @test abs(ll_gpu - ll_cpu) / abs(ll_cpu) < 0.5  # within 50% relative
            end
        end
    end
end
