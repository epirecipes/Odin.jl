#!/usr/bin/env julia
# Lightweight vignette parity report.
#
# For every numbered vignette under vignettes/, verifies that:
#   - a Julia .qmd source exists
#   - an R/ subdirectory with .qmd source exists
#   - rendered html/md outputs exist for both (under vignettes/_output/)
#
# Prints a summary table and exits with a non-zero status if any vignette
# is incomplete.

using Printf

const ROOT = abspath(joinpath(@__DIR__, "..", "vignettes"))
const OUT  = joinpath(ROOT, "_output")

mutable struct VignetteStatus
    name::String
    has_jl_qmd::Bool
    has_r_qmd::Bool
    has_jl_html::Bool
    has_r_html::Bool
    has_jl_md::Bool
    has_r_md::Bool
    jl_md_size::Int
    r_md_size::Int
end

results = VignetteStatus[]

for d in sort(readdir(ROOT))
    path = joinpath(ROOT, d)
    isdir(path) || continue
    occursin(r"^\d+_", d) || continue

    name = d
    jl_qmd  = joinpath(path, "$name.qmd")
    r_qmd   = joinpath(path, "R", "$name.qmd")
    out_dir = joinpath(OUT, name)
    jl_html = joinpath(out_dir, "$name.html")
    r_html  = joinpath(out_dir, "R", "$name.html")
    jl_md   = joinpath(out_dir, "$name.md")
    r_md    = joinpath(out_dir, "R", "$name.md")

    push!(results, VignetteStatus(
        name,
        isfile(jl_qmd), isfile(r_qmd),
        isfile(jl_html), isfile(r_html),
        isfile(jl_md), isfile(r_md),
        isfile(jl_md) ? filesize(jl_md) : 0,
        isfile(r_md)  ? filesize(r_md)  : 0,
    ))
end

println(rpad("Vignette", 42),
        " | Jl src | R src | Jl html | R html | Jl md (KB) | R md (KB) | Ratio")
println("-"^110)

incomplete = Ref(0)
for r in results
    complete = r.has_jl_qmd && r.has_r_qmd && r.has_jl_html && r.has_r_html &&
               r.has_jl_md && r.has_r_md
    incomplete[] += !complete
    ratio = if r.r_md_size > 0
        Printf.format(Printf.Format("%.2f"), r.jl_md_size / r.r_md_size)
    else
        "n/a"
    end
    @printf "%-42s |   %s    |   %s   |    %s    |   %s    | %10.1f | %9.1f | %s\n" r.name (r.has_jl_qmd ? "Y" : "N") (r.has_r_qmd ? "Y" : "N") (r.has_jl_html ? "Y" : "N") (r.has_r_html ? "Y" : "N") r.jl_md_size/1024 r.r_md_size/1024 ratio
end
println()
println("Total vignettes: $(length(results))")
println("Incomplete:      $(incomplete[])")
exit(incomplete[] == 0 ? 0 : 1)
