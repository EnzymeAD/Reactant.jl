using Test
using Reactant

muler(x) = y -> x * y

@testset "closure" begin
    x = Reactant.ConcreteRArray(ones(2, 2))
    y = Reactant.ConcreteRArray(ones(2, 2))

    f = muler(x)
    @test @jit(f(y)) ≈ x * y
end
