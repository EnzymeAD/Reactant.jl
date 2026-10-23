using StaticArrays, Reactant, Test

@testset "broadcast with a traced number" begin
    f(p) = SVector(1.0, 2.0) .+ p
    g(p) = SVector(p, 2p) .* p
    h(p) = SMatrix{2,2}(1.0, 2.0, 3.0, 4.0) .- p

    p = ConcreteRNumber(2.0)
    @test @jit(f(p)) ≈ f(2.0)
    @test @jit(g(p)) ≈ g(2.0)
    @test @jit(h(p)) ≈ h(2.0)
end
