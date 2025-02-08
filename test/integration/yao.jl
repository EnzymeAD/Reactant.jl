using YaoBlocks, Reactant

@testset "YaoBlocks" begin
    f(θ) = mat(Rx(θ))

    x = ConcreteRNumber(0.0)
    @test @jit(f(x)) ≈ f(0.0)

    x = ConcreteRNumber(1.0)
    @test @jit(f(x)) ≈ f(1.0)
end
