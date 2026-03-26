using Test
using Odin

@testset "Monty Sample" begin
    @testset "Sample continuation" begin
        using Distributions
        density = x -> logpdf(Normal(0, 1), x[1])
        model = DensityModel(density; parameters=["x"])

        vcv = reshape([0.5], 1, 1)
        sampler = random_walk(vcv)
        initial = zeros(Float64, 1, 2)

        s1 = sample(model, sampler, 100; n_chains=2, initial=initial, seed=42)
        @test size(s1.pars) == (1, 100, 2)

        s2 = sample_continue(s1, model, sampler, 50)
        @test size(s2.pars) == (1, 50, 2)
    end
end
