@testitem "closure" begin
    muler(x) = y -> x * y

    x = Reactant.ConcreteRArray(ones(2, 2))
    y = Reactant.ConcreteRArray(ones(2, 2))

    f = muler(x)
    g = Reactant.compile(f, (y,))

    @test g(y) ≈ x * y
end
