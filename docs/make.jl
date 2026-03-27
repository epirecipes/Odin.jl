using Documenter
using Odin

makedocs(;
    modules = [Odin],
    sitename = "Odin.jl",
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Guide" => [
            "DSL Reference" => "dsl.md",
            "Simulation" => "simulation.md",
            "Filtering & Likelihood" => "filtering.md",
            "Inference" => "inference.md",
            "ODE Solvers" => "solvers.md",
            "GPU Acceleration" => "gpu.md",
            "Categorical Models" => "categorical.md",
        ],
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

deploydocs(;
    repo = "github.com/epirecipes/Odin.jl.git",
    devbranch = "master",
    push_preview = true,
)
