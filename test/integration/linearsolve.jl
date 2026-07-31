using LinearAlgebra, Random, Reactant, StableRNGs, Test

@testset "Direct Backsolve" begin
    A = rand(StableRNG(0), Float32, 4, 4)
    b = rand(StableRNG(1), Float32, 4)
    B = rand(StableRNG(2), Float32, 4, 5)

    B_ra = Reactant.to_rarray(B)
    b_ra = Reactant.to_rarray(b)

    @testset "qr (non-square)" begin
        # over-determined: unique least squares solution, compare against Julia directly
        A_tall = rand(StableRNG(3), Float32, 6, 4)
        b6 = rand(StableRNG(4), Float32, 6)
        B6 = rand(StableRNG(5), Float32, 6, 5)

        A_tall_ra = Reactant.to_rarray(A_tall)
        b6_ra = Reactant.to_rarray(b6)
        B6_ra = Reactant.to_rarray(B6)

        @test A_tall \ b6 ≈ @jit(A_tall_ra \ b6_ra) atol = 1e-5 rtol = 1e-3
        @test A_tall \ B6 ≈ @jit(A_tall_ra \ B6_ra) atol = 1e-5 rtol = 1e-3

        # under-determined: unpivoted QR yields a basic solution that need not match the
        # one Julia's pivoted `\` picks, so check the residual
        A_wide = rand(StableRNG(6), Float32, 4, 6)
        A_wide_ra = Reactant.to_rarray(A_wide)

        x = Array(@jit(A_wide_ra \ b_ra))
        @test length(x) == 6
        @test maximum(abs, A_wide * x .- b) < 1e-3

        X = Array(@jit(A_wide_ra \ B_ra))
        @test size(X) == (6, 5)
        @test maximum(abs, A_wide * X .- B) < 1e-3
    end

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
