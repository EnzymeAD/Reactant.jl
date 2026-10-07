using LinearAlgebra, Random, Reactant, StableRNGs, Test

@testset "Direct Backsolve" begin
    A = rand(StableRNG(0), Float32, 4, 4)
    b = rand(StableRNG(1), Float32, 4)
    B = rand(StableRNG(2), Float32, 4, 5)

    B_ra = Reactant.to_rarray(B)
    b_ra = Reactant.to_rarray(b)

    # TODO: test QR once lowering exists

    @testset "lu" begin
        A_ra = Reactant.to_rarray(A)

        @test A \ b ≈ @jit(A_ra \ b_ra) atol = 1e-5 rtol = 1e-3
        @test A \ B ≈ @jit(A_ra \ B_ra) atol = 1e-5 rtol = 1e-3
    end

    @testset "diagonal" begin
        A_diag = collect(Float32, Diagonal(A))
        A_ra = Reactant.to_rarray(A_diag)

        @test A_diag \ b ≈ @jit(A_ra \ b_ra) atol = 1e-5 rtol = 1e-3
        @test A_diag \ B ≈ @jit(A_ra \ B_ra) atol = 1e-5 rtol = 1e-3
    end

    @testset "UpperTriangular" begin
        A_up = collect(Float32, UpperTriangular(A))
        A_ra = Reactant.to_rarray(A_up)

        @test A_up \ b ≈ @jit(A_ra \ b_ra) atol = 1e-5 rtol = 1e-3
        @test A_up \ B ≈ @jit(A_ra \ B_ra) atol = 1e-5 rtol = 1e-3
    end

    @testset "LowerTriangular" begin
        A_low = collect(Float32, LowerTriangular(A))
        A_ra = Reactant.to_rarray(A_low)

        @test A_low \ b ≈ @jit(A_ra \ b_ra) atol = 1e-5 rtol = 1e-3
        @test A_low \ B ≈ @jit(A_ra \ B_ra) atol = 1e-5 rtol = 1e-3
    end
end
