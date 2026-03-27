using Documenter
using Odin

makedocs(;
    modules = [Odin],
    sitename = "Odin.jl",
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = String[],
        repolink = "https://github.com/epirecipes/Odin.jl",
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
        "Vignettes" => [
            "Basics" => [
                "Basic ODE Model: SIR" => "vignettes/01_basic_ode.md",
                "Stochastic Discrete-Time SIR" => "vignettes/02_stochastic.md",
                "Incidence Tracking with zero_every" => "vignettes/03_observations.md",
                "Age-Structured SIR with Arrays" => "vignettes/04_arrays.md",
                "Particle Filter and Likelihood" => "vignettes/05_particle_filter.md",
                "Bayesian Inference with MCMC" => "vignettes/06_inference.md",
            ],
            "Intermediate" => [
                "Compositional Model Building" => "vignettes/07_categorical.md",
                "Time-Varying Parameters" => "vignettes/08_time_varying.md",
                "SEIR with Vaccination and Waning" => "vignettes/09_advanced.md",
                "Counterfactual Projections" => "vignettes/10_projections.md",
                "Delay Compartments" => "vignettes/11_delay_model.md",
                "Reactive Vaccination Policy" => "vignettes/12_reactive_policy.md",
                "DynamicPPL Integration" => "vignettes/13_dynamicppl.md",
                "SEIR with Delay and Vaccination" => "vignettes/14_delay_vaccination.md",
                "Vector-Borne Disease Dynamics" => "vignettes/15_vector_borne.md",
            ],
            "Advanced Models" => [
                "Multi-Stream Outbreak" => "vignettes/16_multi_stream.md",
                "Mpox SEIR: Age-Structured" => "vignettes/17_mpox_seir.md",
                "Malaria: Ross-Macdonald" => "vignettes/18_malaria_simple.md",
                "SARS-CoV-2 Multi-Region" => "vignettes/19_sarscov2_multiregion.md",
                "Yellow Fever SEIRV" => "vignettes/20_yellowfever.md",
                "SIS with School Closure" => "vignettes/21_school_closure.md",
                "Beta Blocks" => "vignettes/22_beta_blocks.md",
                "OROV Vector-Borne" => "vignettes/23_orov.md",
                "Yellow Fever with Erlang Delay" => "vignettes/24_yf_delay.md",
                "Yellow Fever 2-Track Vaccination" => "vignettes/25_yf_vtrack.md",
                "Complete Fitting Workflow" => "vignettes/26_fitting_workflow.md",
                "Spatial Composition" => "vignettes/27_spatial_composition.md",
                "Age Stratification" => "vignettes/28_stratification.md",
                "Multi-Pathogen Composition" => "vignettes/29_multi_pathogen.md",
            ],
            "Techniques" => [
                "Stiff ODEs: SDIRK4 Solver" => "vignettes/30_stiff_ode.md",
                "GPU-Accelerated Filtering" => "vignettes/31_gpu_filter.md",
                "Advanced MCMC Samplers" => "vignettes/32_advanced_samplers.md",
                "Sensitivity Analysis" => "vignettes/33_sensitivity.md",
                "Event Handling" => "vignettes/34_events.md",
                "Stochastic Differential Equations" => "vignettes/35_sde.md",
                "Model Selection" => "vignettes/36_model_selection.md",
                "Model Validation" => "vignettes/37_model_validation.md",
            ],
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
