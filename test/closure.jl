using Test
using Reactant

muler(x) = y -> x * y

@testset "closure" begin
    x = ones(2, 2)
    y = ones(2, 2)
    x_ra = Reactant.ConcretePJRTArray(x)
    y_ra = Reactant.ConcretePJRTArray(y)

    f = muler(x_ra)
    @test @jit(f(y_ra)) ≈ x * y
end
