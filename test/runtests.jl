using Test

@testset "Odin.jl" begin
    function _run_testfile(relpath::AbstractString)
        testfile = joinpath(@__DIR__, relpath)
        println("Running ", relpath)
        code = join((
            "using Test",
            "using Odin",
            string("include(\"", replace(testfile, "\\" => "\\\\"), "\")"),
        ), '\n')
        cmd = `$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) -e $(code)`
        @test success(cmd)
    end

    # Julia 1.12 can segfault in compiler/GC code when this whole suite runs in
    # one process because the DSL generates many large methods across files.
    # Running each file in a fresh subprocess preserves coverage while avoiding
    # process-wide compiler state accumulation.
    testfiles = [
        "dsl/test_parse.jl",
        "dsl/test_classify.jl",
        "dsl/test_codegen.jl",
        "dsl/test_print.jl",
        "dsl/test_arrays.jl",
        "dsl/test_output.jl",
        "dsl/test_interpolation.jl",
        "dsl/test_delay.jl",
        "dsl/test_zi_truncated.jl",
        "dsl/test_array_compare.jl",
        "dsl/test_odin_model.jl",
        "dsl/test_additional_models.jl",
        "dsl/test_school_closure.jl",
        "dsl/test_yellowfever.jl",
        "dsl/test_yf_delay.jl",
        "dsl/test_yf_vtrack.jl",
        "dsl/test_beta_blocks.jl",
        "dsl/test_orov.jl",
        "dsl/test_fitting_workflow.jl",
        "dsl/test_mpox_deterministic.jl",
        "dsl/test_symbolic.jl",
        "dust/test_system.jl",
        "dust/test_simulate.jl",
        "dust/test_sde.jl",
        "dust/test_sdirk.jl",
        "dust/test_events.jl",
        "dust/test_filter.jl",
        "dust/test_sensitivity.jl",
        "monty/test_packer.jl",
        "monty/test_model.jl",
        "monty/test_samplers.jl",
        "monty/test_new_samplers.jl",
        "monty/test_sample.jl",
        "monty/test_dsl.jl",
        "monty/test_model_selection.jl",
        "monty/test_turing_bridge.jl",
        "monty/test_validation.jl",
        "categorical/test_categorical.jl",
        "categorical/test_categorical_advanced.jl",
        "gpu/test_gpu.jl",
        "test_api.jl",
        "test-validate.jl",
        "test_new_features.jl",
        "crosslang/test_crosslang.jl",
    ]

    for testfile in testfiles
        _run_testfile(testfile)
    end
end
