using LinearAlgebra, Reactant, Test
using LinearAlgebra: BLAS

@testset "asum" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 32)
    x_ra = Reactant.to_rarray(x)

    @test @jit(BLAS.asum(length(x), x_ra, 1)) ≈ BLAS.asum(length(x), x, 1)
    @test @jit(BLAS.asum(3, x_ra, 5)) ≈ BLAS.asum(3, x, 5)

    y = Reactant.TestUtils.construct_test_array(Complex{Float32}, 32)
    y_ra = Reactant.to_rarray(y)

    @test @jit(BLAS.asum(length(y), y_ra, 1)) ≈ BLAS.asum(length(y), y, 1)
    @test @jit(BLAS.asum(3, y_ra, 5)) ≈ BLAS.asum(3, y, 5)
end

@testset "dot" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 32)
    y = Reactant.TestUtils.construct_test_array(Float32, 32)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test @jit(BLAS.dot(x_ra, y_ra)) ≈ BLAS.dot(x, y)
    @test @jit(BLAS.dot(3, x_ra, 5, y_ra, 4)) ≈ BLAS.dot(3, x, 5, y, 4)

    z1 = Reactant.TestUtils.construct_test_array(Complex{Float32}, 32)
    z2 = Reactant.TestUtils.construct_test_array(Complex{Float32}, 32) .+ 3.0f0
    z1_ra = Reactant.to_rarray(z1)
    z2_ra = Reactant.to_rarray(z2)

    @test @jit(BLAS.dotu(z1_ra, z2_ra)) ≈ BLAS.dotu(z1, z2)
    @test @jit(BLAS.dotc(z1_ra, z2_ra)) ≈ BLAS.dotc(z1, z2)

    @test @jit(BLAS.dotu(2, z1_ra, 3, z2_ra, 5)) ≈ BLAS.dotu(2, z1, 3, z2, 5)
    @test @jit(BLAS.dotc(2, z1_ra, 3, z2_ra, 5)) ≈ BLAS.dotc(2, z1, 3, z2, 5)
end

@testset "scal!" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 32)
    x_ra = Reactant.to_rarray(x)

    @test @jit(BLAS.scal(2.0f0, x_ra)) ≈ BLAS.scal(length(x), 2.0f0, x, 1)
    @test @jit(BLAS.scal(3, 2.0f0, x_ra, 5)) ≈ BLAS.scal(3, 2.0f0, x, 5)

    @jit BLAS.scal!(2.0f0, x_ra)
    @test x_ra ≈ BLAS.scal(length(x), 2.0f0, x, 1)
end
