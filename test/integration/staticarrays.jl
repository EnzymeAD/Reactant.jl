using StaticArrays, LinearAlgebra, Reactant, Test

@testset "broadcast with a traced number" begin
    f(p) = SVector(1.0, 2.0) .+ p
    g(p) = SVector(p, 2p) .* p
    h(p) = SMatrix{2,2}(1.0, 2.0, 3.0, 4.0) .- p

    p = ConcreteRNumber(2.0)
    @test @jit(f(p)) ≈ f(2.0)
    @test @jit(g(p)) ≈ g(2.0)
    @test @jit(h(p)) ≈ h(2.0)
end

struct SVectorHolder{V} end
sv_holder_fn(x, ::SVectorHolder) = x .+ 1
sv_holder_specific_fn(x, ::SVectorHolder{SVector(1.0f0, 2.0f0, 3.0f0)}) = x .+ 2
sv_kw_fn(x; h) = sv_holder_fn(x, h)

struct SVectorCallable{V} end
(::SVectorCallable)(x) = x .* 2

# https://github.com/EnzymeAD/Reactant.jl/issues/3553
@testset "StaticArray value as a type parameter" begin
    x = Reactant.to_rarray(ones(Float32, 4))

    h = SVectorHolder{SVector(1.0f0, 2.0f0, 3.0f0)}()
    @test @jit(sv_holder_fn(x, h)) ≈ ones(Float32, 4) .+ 1
    @test @jit(sv_holder_specific_fn(x, h)) ≈ ones(Float32, 4) .+ 2
    @test @jit(sv_kw_fn(x; h)) ≈ ones(Float32, 4) .+ 1

    hm = SVectorHolder{SMatrix{2,2}(1.0f0, 2.0f0, 3.0f0, 4.0f0)}()
    @test @jit(sv_holder_fn(x, hm)) ≈ ones(Float32, 4) .+ 1

    w = SVectorCallable{SVector(1.0f0, 2.0f0)}()
    @test @jit(broadcast(w, x)) ≈ ones(Float32, 4) .* 2
    @test @jit(w(x)) ≈ ones(Float32, 4) .* 2

    hlo = String(@code_hlo optimize = :before_jit debug = true sv_holder_fn(x, h))
    @test occursin("SVectorHolder{Float32[1.0, 2.0, 3.0]}", hlo)
end

struct DotN3538 end
(::DotN3538)(a, b, c) = dot(SVector(1.0f0, 2.0f0, 3.0f0), SVector(a, b, c))

struct Dot3Arg3538 end
function (::Dot3Arg3538)(a, b)
    A = SMatrix{2,2}(2.0f0, 1.0f0, 1.0f0, 3.0f0)
    return dot(SVector(a, b), A, SVector(1.0f0, -1.0f0))
end

struct NormN3538 end
(::NormN3538)(a, b, c) = norm(SVector(a, b, c))

struct NormP3538{P} end
(::NormP3538{P})(a, b, c) where {P} = norm(SVector(a, b, c), P)

struct MapN3538 end
(::MapN3538)(a, b) = sum(muladd(SVector(1.0, 2.0), 0.5, SVector(a, b)))

struct Vec2D3393{T} <: FieldVector{2,T}
    x::T
    y::T
end

add_svec_3393(t) = SVector(t, t) + SVector(1.0, 0.0)
add_svec_rev_3393(t) = SVector(1.0, 0.0) + SVector(t, t)
add_fvec_3393(t) = Vec2D3393(t, t) + Vec2D3393(1.0, 0.0)
dot_complex_svec_3538(z) = dot(SVector(1.0 + 2.0im, -1.0 + 0.5im), SVector(z, 2z))

@testset "StaticArray dot, norm, map, and mixed + (#3538, #3393)" begin
    x_cpu = Float32[1, 2, 3, 4]
    x = Reactant.to_rarray(x_cpu)

    @test Array(@jit broadcast(DotN3538(), x, x, x)) ≈
        broadcast(DotN3538(), x_cpu, x_cpu, x_cpu)
    @test Array(@jit broadcast(Dot3Arg3538(), x, x)) ≈
        broadcast(Dot3Arg3538(), x_cpu, x_cpu)
    @test Array(@jit broadcast(NormN3538(), x, x, x)) ≈
        broadcast(NormN3538(), x_cpu, x_cpu, x_cpu)
    @test Array(@jit broadcast(NormP3538{1}(), x, x, x)) ≈
        broadcast(NormP3538{1}(), x_cpu, x_cpu, x_cpu)
    @test Array(@jit broadcast(NormP3538{Inf}(), x, x, x)) ≈
        broadcast(NormP3538{Inf}(), x_cpu, x_cpu, x_cpu)
    @test Array(@jit broadcast(NormP3538{3}(), x, x, x)) ≈
        broadcast(NormP3538{3}(), x_cpu, x_cpu, x_cpu)

    xd_cpu = [0.25, -0.5, 1.25, 2.0]
    xd = Reactant.to_rarray(xd_cpu)
    @test Array(@jit broadcast(MapN3538(), xd, xd)) ≈ broadcast(MapN3538(), xd_cpu, xd_cpu)

    t = ConcreteRNumber(2.0)
    @test @jit(add_svec_3393(t)) ≈ add_svec_3393(2.0)
    @test @jit(add_svec_rev_3393(t)) ≈ add_svec_rev_3393(2.0)
    @test @jit(add_fvec_3393(t)) ≈ add_fvec_3393(2.0)

    z = ConcreteRNumber(1.5 - 0.5im)
    @test Reactant.to_number(@jit dot_complex_svec_3538(z)) ≈
        dot_complex_svec_3538(1.5 - 0.5im)
end
