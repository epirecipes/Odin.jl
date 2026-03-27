module Odin

using Random
using LinearAlgebra
using Distributions
using Interpolations
using ForwardDiff
using ReverseDiff
using Symbolics
using StatsBase
import StatsBase: sample
using PoissonRandom
import Catlab
using Catlab: @present, @acset_type, FreeSchema,
              add_part!, add_parts!, nparts, incident

# ── DSL (odin2) ─────────────────────────────────────────────
include("dsl/constants.jl")
include("dsl/parse.jl")
include("dsl/classify.jl")
include("dsl/dependencies.jl")
include("dsl/stages.jl")
include("dsl/arrays.jl")
include("dsl/interpolation.jl")
include("dsl/codegen.jl")
include("dsl/symbolic.jl")
include("dsl/macro.jl")
include("dsl/odin_model.jl")
include("dsl/validate.jl")

# ── Dust runtime (dust2) ────────────────────────────────────
include("dust/fast_random.jl")
include("dust/fast_logpdf.jl")
include("dust/delay.jl")
include("dust/dp5.jl")
include("dust/sdirk.jl")
include("dust/sde.jl")
include("dust/events.jl")
include("dust/system.jl")
include("dust/simulate.jl")
include("dust/resampling.jl")
include("dust/data.jl")
include("dust/ode_control.jl")
include("dust/filter.jl")
include("dust/unfilter.jl")

# ── Monty inference (monty) ─────────────────────────────────
include("monty/packer.jl")
include("monty/model.jl")
include("monty/distributions.jl")
include("monty/samplers/sampler.jl")
include("monty/samplers/helpers.jl")
include("monty/samplers/random_walk.jl")
include("monty/samplers/hmc.jl")
include("monty/samplers/adaptive.jl")
include("monty/samplers/parallel_tempering.jl")
include("monty/samplers/nuts.jl")
include("monty/samplers/slice.jl")
include("monty/samplers/mala.jl")
include("monty/samplers/gibbs.jl")
include("monty/runners.jl")
include("monty/observer.jl")
include("monty/sample.jl")
include("monty/dsl.jl")
include("monty/model_selection.jl")
include("monty/turing_bridge.jl")
include("monty/validation.jl")

# ── Sensitivity analysis ──────────────────────────────────────
include("dust/autodiff.jl")
include("dust/sensitivity.jl")

# ── GPU acceleration ─────────────────────────────────────────
include("gpu/gpu_backend.jl")
include("gpu/gpu_filter.jl")
include("gpu/gpu_simulate.jl")

# ── Categorical (category theory extension) ──────────────────
include("categorical/schemas.jl")
include("categorical/types.jl")
include("categorical/composition.jl")
include("categorical/stratification.jl")
include("categorical/lowering.jl")

# ── Julia-friendly API layer ─────────────────────────────────
include("api.jl")

# ── Public API ──────────────────────────────────────────────
# DSL
export @odin, @odin_model, @prior

# Introspection
export validate_model, show_code, OdinValidationResult

# Type aliases
export OdinModel, Samples, ObservedData, ODEControl

# Core types (still needed for dispatch/type annotations)
export DustSystem, DustFilter, DustUnfilter, MontyModel

# Simulation
export simulate, System, reset!, state, run_to!

# Likelihood
export Likelihood, loglik, loglik_pointwise, loglik_gradient, as_model

# Parameter packing
export Packer, GroupedPacker, MontyPacker, MontyPackerGrouped

# Model construction
export DensityModel

# Samplers
export nuts, random_walk, hmc, adaptive_mh, mala, slice
export parallel_tempering, gibbs

# Runners
export Serial, Threaded, Simultaneous

# Sampling
export sample, sample_continue

# Observer
export Observer, MontyObserver, last_snapshots, last_trajectories

# Diagnostics & validation
export posterior_predict
export PosteriorPredictive, PPCResult, ResidualDiagnostics, CalibrationResult, SBCResult
export ppc_check, residual_diagnostics, calibration_check
export prior_predictive, sbc_check

# Model selection
export aic, aicc, bic, dic, waic, loo, compare
export akaike_weights, ModelComparison

# Sensitivity
export sensitivity
export ForwardSensitivityResult, AdjointSensitivityResult, SobolResult, MorrisResult

# Events
export ContinuousEvent, DiscreteEvent, TimedEvent, EventSet, EventRecord

# SDE/ODE solvers (low-level)
export sdirk_solve!, SDIRKWorkspace, SDIRKResult
export sde_solve!, SDEWorkspace, SDEResult

# Turing/DynamicPPL bridge
export as_logdensity, to_turing_model, turing_sample
export dppl_prior, dppl_to_monty_model
export to_chains, from_chains

# GPU
export GPUBackend, CPUBackend, MetalBackend, CUDABackend, AMDGPUBackend
export gpu_backend, has_gpu, available_gpu_backends, backend_name
export gpu_array, cpu_array, gpu_array_type
export GPUDustFilter

# Categorical
export EpiNet, add_species!, add_transition!
export species_names, species_concentrations, nspecies
export transition_names, transition_rates, ntransitions
export input_species, output_species, stoichiometry_matrix, input_matrix
export compose, compose_with_interface
export stratify, compile, lower_expr
export SIR, SEIR, SIS, SIRS, SEIRS, SIRVax, TwoStrainSIR

end # module
