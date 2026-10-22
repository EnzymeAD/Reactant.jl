using Reactant, StaticArrays, LinearAlgebra, Test

# Per-point functions that use small static vectors of traced scalars, broadcast over arrays.
struct DotConst end
(::DotConst)(a, b, c) = dot(SVector(1.0f0, 2.0f0, 3.0f0), SVector(a, b, c))
struct DotTraced end
(::DotTraced)(a, b, c) = dot(SVector(a, b, c), SVector(c, b, a))
struct NormOf{P} end
(::NormOf{P})(a, b, c) where {P} = norm(SVector(a, b, c), P)
struct NormDefault end
(::NormDefault)(a, b, c) = norm(SVector(a, b, c))
struct MuladdSum end
(::MuladdSum)(a, b) = sum(muladd(SVector(1.0, 2.0), 0.5, SVector(a, b)))

@testset "SVector of traced numbers" begin
    xf = Float32[1, -2, 3, 0.5]
    yf = Float32[-1, 4, 2, 0.25]
    zf = Float32[2, 1, -3, 5]
    x, y, z = Reactant.to_rarray.((xf, yf, zf))

    @testset "dot" begin
        @test Array(@jit(broadcast(DotConst(), x, y, z))) ≈ DotConst().(xf, yf, zf)
        @test Array(@jit(broadcast(DotTraced(), x, y, z))) ≈ DotTraced().(xf, yf, zf)
    end

    @testset "norm" begin
        @test Array(@jit(broadcast(NormDefault(), x, y, z))) ≈ NormDefault().(xf, yf, zf)
        for P in (1, 2, 3, Inf)
            @test Array(@jit(broadcast(NormOf{P}(), x, y, z))) ≈ NormOf{P}().(xf, yf, zf)
        end
    end

    @testset "map (muladd with a plain SVector)" begin
        ad = rand(4)
        bd = rand(4)
        a, b = Reactant.to_rarray.((ad, bd))
        @test Array(@jit(broadcast(MuladdSum(), a, b))) ≈ MuladdSum().(ad, bd)
    end
end
