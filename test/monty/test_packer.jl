using Test
using Odin

@testset "Monty Packer" begin
    @testset "Scalar packing/unpacking" begin
        p = monty_packer([:beta, :gamma])

        @test p.len == 2

        nt = p([0.2, 0.1])
        @test nt.beta ≈ 0.2
        @test nt.gamma ≈ 0.1

        v = Odin.pack(p, (beta=0.2, gamma=0.1))
        @test v ≈ [0.2, 0.1]
    end

    @testset "Array packing" begin
        p = monty_packer([:a]; array=Dict(:b => 3))

        @test p.len == 4

        nt = p([1.0, 2.0, 3.0, 4.0])
        @test nt.a ≈ 1.0
        @test nt.b ≈ [2.0, 3.0, 4.0]
    end

    @testset "Fixed values" begin
        p = monty_packer([:beta]; fixed=(N=1000, gamma=0.1))

        @test p.len == 1

        nt = p([0.5])
        @test nt.beta ≈ 0.5
        @test nt.N == 1000
        @test nt.gamma ≈ 0.1
    end

    @testset "Process function" begin
        p = monty_packer([:a, :b]; process=nt -> (c=nt.a + nt.b,))

        nt = p([3.0, 4.0])
        @test nt.a ≈ 3.0
        @test nt.b ≈ 4.0
        @test nt.c ≈ 7.0
    end
end
