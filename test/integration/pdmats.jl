using PDMats, Reactant, Test

const RunningOnMetal = contains(string(Reactant.devices()[1]), "MTL")

@testset "wsumsq" begin
    w = rand(5) .+ 0.5
    a = randn(5)
    rw = Reactant.to_rarray(w)
    ra = Reactant.to_rarray(a)
    @test PDMats.wsumsq(w, a) ≈ @jit(PDMats.wsumsq(rw, ra)) rtol = 1e-12 skip =
        RunningOnMetal
    @test PDMats.wsumsq(w, a) ≈ @jit(PDMats.wsumsq(rw, a)) rtol = 1e-12 skip =
        RunningOnMetal
    @test PDMats.wsumsq(w, a) ≈ @jit(PDMats.wsumsq(w, ra)) rtol = 1e-12 skip =
        RunningOnMetal
end

@testset "invwsumsq" begin
    w = rand(5) .+ 0.5
    a = randn(5)
    rw = Reactant.to_rarray(w)
    ra = Reactant.to_rarray(a)
    @test PDMats.invwsumsq(w, a) ≈ @jit(PDMats.invwsumsq(rw, ra)) rtol = 1e-12 skip =
        RunningOnMetal
    @test PDMats.invwsumsq(w, a) ≈ @jit(PDMats.invwsumsq(rw, a)) rtol = 1e-12 skip =
        RunningOnMetal
    @test PDMats.invwsumsq(w, a) ≈ @jit(PDMats.invwsumsq(w, ra)) rtol = 1e-12 skip =
        RunningOnMetal
end
