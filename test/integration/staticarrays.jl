using StaticArrays, Reactant, Test

const RunningOnMetal = contains(string(Reactant.devices()[1]), "MTL")

@testset "broadcast with a traced number" begin
    f(p) = SVector(1.0, 2.0) .+ p
    g(p) = SVector(p, 2p) .* p
    h(p) = SMatrix{2,2}(1.0, 2.0, 3.0, 4.0) .- p

    p = ConcreteRNumber(2.0)
    @test @jit(f(p)) ≈ f(2.0) skip = RunningOnMetal
    @test @jit(g(p)) ≈ g(2.0) skip = RunningOnMetal
    @test @jit(h(p)) ≈ h(2.0) skip = RunningOnMetal
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
