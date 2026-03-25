#!/usr/bin/env julia
# Benchmark the single-solve unfilter optimization
# Shows allocation reduction and speedup vs R's C++ solver

using Odin, BenchmarkTools, Random
import Logging; Logging.disable_logging(Logging.Warn)

println("=" ^ 60)
println("Unfilter Performance Benchmark (single-solve optimization)")
println("=" ^ 60)

sir = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0; initial(I) = I0; initial(R) = 0.0
    N = parameter(1000.0); I0 = parameter(10.0)
    beta = parameter(0.5); gamma = parameter(0.1)
    cases_lambda = I > 0 ? rho * I : 1e-10
    cases ~ Poisson(cases_lambda)
    rho = parameter(0.3)
end

sys = dust_system_create(sir, (beta=0.5, gamma=0.1, rho=0.3, I0=10.0, N=1000.0); n_particles=1, seed=42)
dust_system_set_state_initial!(sys)
sol = dust_system_simulate(sys, collect(1.0:1.0:50.0))
data_rows = [(time=Float64(i), cases=max(1.0, round(sol[2,1,i] * 0.3))) for i in 1:50]
fdata = dust_filter_data(data_rows; time_field=:time)
uf = dust_unfilter_create(sir, fdata)
pars = (beta=0.5, gamma=0.1, rho=0.3, I0=10.0, N=1000.0)

# Warmup
dust_unfilter_run!(uf, pars)

# Benchmark
b = @benchmark dust_unfilter_run!($uf, $pars) samples=500
println("\nUnfilter (DP5, single solve, 50 data points):")
println("  Median time: $(round(median(b).time/1e3, digits=1)) μs")
println("  Allocations: $(round(Int, median(b).allocs))")
println("  Memory:      $(round(median(b).memory/1024, digits=1)) KiB")

# Also benchmark gradient
using ForwardDiff
packer = monty_packer([:beta, :gamma, :rho]; fixed=(N=1000.0, I0=10.0))
ll_model = dust_likelihood_monty(uf, packer)
x = [0.5, 0.1, 0.3]
ll_model.gradient(x)  # warmup
bg = @benchmark ($ll_model).gradient($x) samples=200
println("\nGradient (ForwardDiff, 3 params):")
println("  Median time: $(round(median(bg).time/1e3, digits=1)) μs")
println("  Allocations: $(round(Int, median(bg).allocs))")

println("\nLL at true params: $(round(dust_unfilter_run!(uf, pars), digits=2))")
println("Gradient at true params: $(round.(ll_model.gradient(x), digits=3))")
