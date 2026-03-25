#!/usr/bin/env julia
# Compare Julia and R inference results: ECDF correlation + benchmarks
#
# Run AFTER:
#   Rscript benchmark/posterior_samples_r.R
#   julia --project=. benchmark/posterior_samples_julia.jl
#   Rscript benchmark/benchmark_inference_r.R
#   julia --project=. benchmark/benchmark_inference_julia.jl
#
# Then: julia --project=. benchmark/compare_inference.jl

using CSV, DataFrames
using Statistics
using Printf

println("=" ^ 72)
println("Inference Comparison: Odin.jl vs odin2/dust2/monty (R)")
println("=" ^ 72)

# ═══════════════════════════════════════════════════════════════════
# PART 1: Posterior ECDF Correlation
# ═══════════════════════════════════════════════════════════════════

println("\n" * "─" ^ 72)
println("PART 1: Posterior Distribution Comparison (ECDF Correlation)")
println("─" ^ 72)

"""
Compute the empirical CDF of `x` evaluated at points `grid`.
"""
function ecdf(x::AbstractVector, grid::AbstractVector)
    n = length(x)
    sorted = sort(x)
    result = similar(grid, Float64)
    for (i, g) in enumerate(grid)
        result[i] = searchsortedlast(sorted, g) / n
    end
    return result
end

"""
Pearson correlation between ECDFs of two samples, evaluated on a shared grid.
"""
function ecdf_correlation(x::AbstractVector, y::AbstractVector; n_grid::Int=500)
    lo = min(minimum(x), minimum(y))
    hi = max(maximum(x), maximum(y))
    grid = range(lo, hi, length=n_grid)
    ecdf_x = ecdf(x, collect(grid))
    ecdf_y = ecdf(y, collect(grid))
    return cor(ecdf_x, ecdf_y)
end

"""
Two-sample Kolmogorov-Smirnov statistic.
"""
function ks_statistic(x::AbstractVector, y::AbstractVector)
    all_vals = sort(unique(vcat(x, y)))
    ecdf_x = ecdf(x, all_vals)
    ecdf_y = ecdf(y, all_vals)
    return maximum(abs.(ecdf_x .- ecdf_y))
end

# Load posterior samples
post_jl = CSV.read("benchmark/posterior_julia.csv", DataFrame)
post_r  = CSV.read("benchmark/posterior_r.csv", DataFrame)

println("\n  Sample sizes:  Julia=$(nrow(post_jl))  R=$(nrow(post_r))")

for param in ["beta", "gamma"]
    jl = post_jl[!, param]
    r  = post_r[!, param]

    # Summary statistics
    println("\n  Parameter: $param")
    @printf("    Julia:  mean=%.4f  sd=%.4f  median=%.4f  [%.4f, %.4f]\n",
            mean(jl), std(jl), median(jl),
            quantile(jl, 0.025), quantile(jl, 0.975))
    @printf("    R:      mean=%.4f  sd=%.4f  median=%.4f  [%.4f, %.4f]\n",
            mean(r), std(r), median(r),
            quantile(r, 0.025), quantile(r, 0.975))

    # ECDF correlation
    ρ = ecdf_correlation(jl, r)
    @printf("    ECDF correlation:  ρ = %.6f\n", ρ)

    # KS statistic
    D = ks_statistic(jl, r)
    @printf("    KS statistic:      D = %.4f\n", D)

    # Relative difference in moments
    Δ_mean = abs(mean(jl) - mean(r)) / mean(r) * 100
    Δ_sd   = abs(std(jl) - std(r)) / std(r) * 100
    @printf("    Relative Δ mean:   %.2f%%\n", Δ_mean)
    @printf("    Relative Δ sd:     %.2f%%\n", Δ_sd)
end

# ═══════════════════════════════════════════════════════════════════
# PART 2: Log-Likelihood Distribution Comparison
# ═══════════════════════════════════════════════════════════════════

println("\n" * "─" ^ 72)
println("PART 2: Log-Likelihood Distribution (same parameters, 500 particles)")
println("─" ^ 72)

ll_jl = CSV.read("benchmark/ll_dist_julia.csv", DataFrame).ll
ll_r  = CSV.read("benchmark/ll_dist_r.csv", DataFrame).ll

@printf("\n  Julia:  mean=%.2f  sd=%.2f  [%.2f, %.2f]\n",
        mean(ll_jl), std(ll_jl), minimum(ll_jl), maximum(ll_jl))
@printf("  R:      mean=%.2f  sd=%.2f  [%.2f, %.2f]\n",
        mean(ll_r), std(ll_r), minimum(ll_r), maximum(ll_r))

ρ_ll = ecdf_correlation(ll_jl, ll_r)
D_ll = ks_statistic(ll_jl, ll_r)
@printf("  ECDF correlation:  ρ = %.6f\n", ρ_ll)
@printf("  KS statistic:      D = %.4f\n", D_ll)

# ═══════════════════════════════════════════════════════════════════
# PART 3: Performance Comparison
# ═══════════════════════════════════════════════════════════════════

println("\n" * "─" ^ 72)
println("PART 3: Performance Comparison (median time in ms)")
println("─" ^ 72)

bench_jl = CSV.read("benchmark/results_inference_julia.csv", DataFrame)
bench_r  = CSV.read("benchmark/results_inference_r.csv", DataFrame)

# Merge on config
comparison = innerjoin(bench_jl, bench_r, on=[:type, :config],
                       makeunique=true, renamecols="_jl" => "_r")

println()
@printf("  %-28s  %10s  %10s  %8s\n",
        "Benchmark", "Julia (ms)", "R (ms)", "Ratio")
@printf("  %-28s  %10s  %10s  %8s\n",
        "─" ^ 28, "─" ^ 10, "─" ^ 10, "─" ^ 8)

for row in eachrow(comparison)
    label = "$(row.type):$(row.config)"
    jl = row.median_ms_jl
    r  = row.median_ms_r
    ratio = jl / r
    color = ratio < 1.0 ? "✓" : (ratio < 1.5 ? "≈" : "✗")
    @printf("  %-28s  %10.2f  %10.2f  %7.2f× %s\n",
            label, jl, r, ratio, color)
end

# ═══════════════════════════════════════════════════════════════════
# PART 4: Scaling Analysis
# ═══════════════════════════════════════════════════════════════════

println("\n" * "─" ^ 72)
println("PART 4: Scaling Analysis")
println("─" ^ 72)

# PF scaling with n_particles
pf_jl = filter(r -> r.type == "pf", bench_jl)
pf_r  = filter(r -> r.type == "pf", bench_r)
if nrow(pf_jl) >= 2
    np_jl = parse.(Int, replace.(pf_jl.config, "np_" => ""))
    ms_jl = pf_jl.median_ms
    np_r  = parse.(Int, replace.(pf_r.config, "np_" => ""))
    ms_r  = pf_r.median_ms

    # Cost per particle (linear fit: ms ≈ a + b * n_particles)
    b_jl = (ms_jl[end] - ms_jl[1]) / (np_jl[end] - np_jl[1])
    b_r  = (ms_r[end] - ms_r[1]) / (np_r[end] - np_r[1])
    println("\n  Particle Filter — cost per particle:")
    @printf("    Julia: %.4f ms/particle\n", b_jl)
    @printf("    R:     %.4f ms/particle\n", b_r)
    @printf("    Ratio: %.2f×\n", b_jl / b_r)
end

# MCMC scaling with n_steps (1 chain)
mcmc_steps_jl = filter(r -> r.type == "mcmc" && occursin("_1chain", r.config) &&
                       startswith(r.config, "steps_"), bench_jl)
mcmc_steps_r  = filter(r -> r.type == "mcmc" && occursin("_1chain", r.config) &&
                       startswith(r.config, "steps_"), bench_r)
if nrow(mcmc_steps_jl) >= 2
    ns_jl = parse.(Int, replace.(replace.(mcmc_steps_jl.config, "steps_" => ""),
                                  "_1chain" => ""))
    ms_jl = mcmc_steps_jl.median_ms
    ns_r  = parse.(Int, replace.(replace.(mcmc_steps_r.config, "steps_" => ""),
                                  "_1chain" => ""))
    ms_r  = mcmc_steps_r.median_ms

    b_jl = (ms_jl[end] - ms_jl[1]) / (ns_jl[end] - ns_jl[1])
    b_r  = (ms_r[end] - ms_r[1]) / (ns_r[end] - ns_r[1])
    println("\n  MCMC — cost per step (200 particles, 1 chain):")
    @printf("    Julia: %.3f ms/step\n", b_jl)
    @printf("    R:     %.3f ms/step\n", b_r)
    @printf("    Ratio: %.2f×\n", b_jl / b_r)
end

# ═══════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 72)
println("VERDICT")
println("=" ^ 72)

ρ_beta  = ecdf_correlation(post_jl.beta, post_r.beta)
ρ_gamma = ecdf_correlation(post_jl.gamma, post_r.gamma)
println("\n  Correctness:")
@printf("    ECDF correlation (β):  %.6f  %s\n", ρ_beta,
        ρ_beta > 0.99 ? "✓ EXCELLENT" : ρ_beta > 0.95 ? "≈ GOOD" : "✗ CHECK")
@printf("    ECDF correlation (γ):  %.6f  %s\n", ρ_gamma,
        ρ_gamma > 0.99 ? "✓ EXCELLENT" : ρ_gamma > 0.95 ? "≈ GOOD" : "✗ CHECK")

# Average PF ratio
pf_comp = filter(r -> r.type == "pf", comparison)
avg_pf = mean(pf_comp.median_ms_jl ./ pf_comp.median_ms_r)
mcmc_comp = filter(r -> r.type == "mcmc", comparison)
avg_mcmc = mean(mcmc_comp.median_ms_jl ./ mcmc_comp.median_ms_r)

println("\n  Performance (Julia/R ratio, lower is better):")
@printf("    Particle filter (avg):  %.2f×\n", avg_pf)
@printf("    MCMC (avg):             %.2f×\n", avg_mcmc)
println()
