module Odin

using Random
using LinearAlgebra
using Distributions
using Interpolations
using ForwardDiff
using ReverseDiff
using Symbolics
using StatsBase
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

# ── Dust runtime (dust2) ────────────────────────────────────
include("dust/fast_random.jl")
include("dust/fast_logpdf.jl")
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

# ── Public API ──────────────────────────────────────────────
# DSL
export @odin, @odin_model

# Dust
export DustSystem, DustSystemGenerator
export dust_system_create, dust_system_simulate, dust_system_run_to_time!
export dust_system_state, dust_system_set_state!, dust_system_set_state_initial!
export dust_system_compare_data
export DustFilter, dust_filter_create, dust_likelihood_run!
export dust_filter_data, FilterData
export DustUnfilter, dust_unfilter_create, dust_unfilter_run!
export dust_likelihood_monty, dust_likelihood_run!
export DustODEControl
export sdirk_solve!, SDIRKWorkspace, SDIRKResult
export sde_solve!, SDEWorkspace, SDEResult
export ContinuousEvent, DiscreteEvent, TimedEvent, EventSet, EventRecord
export dp5_solve_events!
export dust_sensitivity_forward, dust_sensitivity_adjoint
export dust_sensitivity_sobol, dust_sensitivity_morris
export dust_unfilter_gradient
export ForwardSensitivityResult, AdjointSensitivityResult, SobolResult, MorrisResult

# Monty
export MontyModel, monty_model, monty_model_combine
export MontyPacker, monty_packer
export MontyPackerGrouped, monty_packer_grouped
export MontyRandomWalkSampler, monty_sampler_random_walk
export MontyHMCSampler, monty_sampler_hmc
export MontyAdaptiveSampler, monty_sampler_adaptive
export MontyParallelTemperingSampler, monty_sampler_parallel_tempering
export MontyNUTSSampler, monty_sampler_nuts
export MontySliceSampler, monty_sampler_slice
export MontyMALASampler, monty_sampler_mala
export MontyGibbsSampler, monty_sampler_gibbs
export MontySerialRunner, MontyThreadedRunner
export monty_runner_serial, monty_runner_threaded
export MontySamples, monty_sample, monty_sample_continue
export @monty_prior

# Validation
export PosteriorPredictive, PPCResult, ResidualDiagnostics, CalibrationResult, SBCResult
export posterior_predictive, ppc_check, residual_diagnostics, calibration_check
export prior_predictive, sbc_check

# Turing/DynamicPPL bridge
export as_logdensity, to_turing_model, turing_sample
export dppl_prior, dppl_to_monty_model
export to_chains, from_chains

# Model selection
export compute_aic, compute_aicc, compute_bic, compute_dic, compute_waic
export compute_loo
export akaike_weights
export dust_unfilter_run_pointwise!, dust_filter_run_pointwise!
export ModelComparison, compare_models

# GPU
export GPUBackend, CPUBackend, MetalBackend, CUDABackend, AMDGPUBackend
export gpu_backend, has_gpu, available_gpu_backends, backend_name
export gpu_array, cpu_array, gpu_array_type
export GPUDustFilter, gpu_dust_filter_create, gpu_dust_filter_run!
export gpu_dust_likelihood_monty
export gpu_dust_simulate

# Categorical
export EpiNet, add_species!, add_transition!
export species_names, species_concentrations, nspecies
export transition_names, transition_rates, ntransitions
export input_species, output_species, stoichiometry_matrix, input_matrix
export compose, compose_with_interface
export stratify
export lower, lower_expr
export sir_net, seir_net, sis_net, sirs_net, seirs_net
export sir_vax_net, two_strain_sir_net

end # module
