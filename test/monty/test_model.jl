using Test
using Odin
using Distributions

@testset "Monty Model" begin
    @testset "Basic density evaluation" begin
        density = x -> logpdf(Normal(0, 1), x[1]) + logpdf(Normal(0, 1), x[2])
        model = DensityModel(density; parameters=["a", "b"])

        @test model([0.0, 0.0]) ≈ -log(2π)
        @test model([1.0, 0.0]) < model([0.0, 0.0])
    end

    @testset "Domain checking" begin
        density = x -> -sum(x .^ 2)
        domain = [0.0 10.0; 0.0 10.0]
        model = DensityModel(density; parameters=["a", "b"], domain=domain)

        @test model([5.0, 5.0]) ≈ -50.0
        @test model([-1.0, 5.0]) == -Inf
    end

    @testset "Model combination" begin
        likelihood = DensityModel(x -> -sum(x .^ 2); parameters=["a", "b"])
        prior = DensityModel(x -> logpdf(Normal(0, 10), x[1]) + logpdf(Normal(0, 10), x[2]); parameters=["a", "b"])

        posterior = likelihood + prior

        val_sep = likelihood([1.0, 2.0]) + prior([1.0, 2.0])
        val_combined = posterior([1.0, 2.0])
        @test val_sep ≈ val_combined
    end
end
