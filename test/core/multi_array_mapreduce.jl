using Reactant
using Enzyme
using LinearAlgebra
using Test

foo_1554(A, B) = mapreduce(+, +, A, B; init=zero(eltype(A)), dims=1)
dot_1554(A, B) = mapreduce(*, +, A, B)
fma_reduce_1554(A, B, C) = mapreduce(muladd, +, A, B, C; dims=(1, 3), init=1.5f0)
eq_count_1554(A, B) = mapreduce(==, +, A, B)
eq_count_dims_1554(A, B) = mapreduce(==, +, A, B; dims=2)
const_mapreduce_1554(A, B) = mapreduce(Returns(3), +, A, B; dims=1)

function grad_multi_mapreduce_1554(A, B)
    dA = zero(A)
    dB = zero(B)
    Enzyme.autodiff(Reverse, dot_1554, Active, Duplicated(A, dA), Duplicated(B, dB))
    return dA, dB
end

@testset "Multi-array mapreduce (#1554)" begin
    Reactant.allowscalar(false)

    # 1. Exact #1554 reproducer (`@jit` and eager)
    A1 = Float32[1.0, -2.5, 3.25, 4.0, -0.5]
    B1 = Float32[0.5, 1.5, -1.25, 2.0, 3.5]
    rA1, rB1 = Reactant.to_rarray((A1, B1))

    @test Array(@jit(foo_1554(rA1, rB1))) ≈ foo_1554(A1, B1)
    @test Array(mapreduce(+, +, rA1, rB1; init=0.0f0, dims=1)) ≈ foo_1554(A1, B1)
    @test @jit(dot_1554(rA1, rB1)) ≈ dot_1554(A1, B1)
    @test Reactant.to_number(mapreduce(*, +, rA1, rB1)) ≈ dot_1554(A1, B1)

    # 2. 3-array 3D reduction with `dims=(1, 3)` and explicit `init`
    A3 = reshape(collect(Float32, 1:24), 2, 3, 4) ./ 10.0f0
    B3 = reshape(collect(Float32, 24:-1:1), 2, 3, 4) ./ 7.0f0
    C3 = fill(0.25f0, 2, 3, 4)
    rA3, rB3, rC3 = Reactant.to_rarray((A3, B3, C3))

    @test Array(@jit(fma_reduce_1554(rA3, rB3, rC3))) ≈ fma_reduce_1554(A3, B3, C3)
    @test Array(mapreduce(muladd, +, rA3, rB3, rC3; dims=(1, 3), init=1.5f0)) ≈
        fma_reduce_1554(A3, B3, C3)

    # 3. Mixed ConcreteRArray, host Array, SubArray, and Transpose
    M1 = Float64[1.0 2.0 3.0; 4.0 5.0 6.0]
    M2 = Float64[10.0 20.0; 30.0 40.0; 50.0 60.0]
    rM1 = Reactant.to_rarray(M1)
    rM2 = Reactant.to_rarray(M2)

    @test Array(mapreduce(*, +, view(rM1, :, 1:2), view(M1, :, 2:3); dims=1)) ≈
        mapreduce(*, +, view(M1, :, 1:2), view(M1, :, 2:3); dims=1)
    @test Array(mapreduce(+, max, M1, transpose(rM2); dims=2)) ≈
        mapreduce(+, max, M1, transpose(M2); dims=2)

    # 4. Narrowing predicate (`Bool` widened to `Int` by `+`) and constant-returning `f`
    I1 = Int32[1 2 3; 4 5 6]
    I2 = Int32[1 0 3; 9 5 6]
    rI1, rI2 = Reactant.to_rarray((I1, I2))

    @test @jit(eq_count_1554(rI1, rI2)) == eq_count_1554(I1, I2)
    @test Array(@jit(eq_count_dims_1554(rI1, rI2))) == eq_count_dims_1554(I1, I2)
    @test Array(@jit(const_mapreduce_1554(rI1, rI2))) == const_mapreduce_1554(I1, I2)

    # 5. Reverse-mode AD through multi-array mapreduce
    dA, dB = @jit grad_multi_mapreduce_1554(rA1, rB1)
    @test Array(dA) ≈ B1
    @test Array(dB) ≈ A1

    # 6. DimensionMismatch on mismatched shapes
    rShort = Reactant.to_rarray(Float32[1.0, 2.0])
    @test_throws DimensionMismatch mapreduce(+, +, rA1, rShort)
    @test_throws DimensionMismatch @jit(dot_1554(rA1, rShort))
end
