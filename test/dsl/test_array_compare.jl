using Test

@testset "Array Comparisons" begin
    function _run_subtest(relpath::AbstractString)
        testfile = joinpath(@__DIR__, relpath)
        println("Running ", relpath)
        code = join((
            "using Test",
            "using Odin",
            string("include(\"", replace(testfile, "\\" => "\\\\"), "\")"),
        ), '\n')
        cmd = `$(Base.julia_cmd()) --project=$(dirname(dirname(@__DIR__))) -e $(code)`
        @test success(cmd)
    end

    for testfile in (
        "test_array_compare_poisson.jl",
        "test_array_compare_normal.jl",
        "test_array_compare_scalar.jl",
        "test_array_compare_mixed.jl",
    )
        _run_subtest(testfile)
    end
end
