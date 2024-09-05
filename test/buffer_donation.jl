using Test
using Reactant

# TODO try again with `2` to check automatic conversion from int to float
function donate_fill_x_with_2(x, y)
    x .= 2.0
    return nothing
end

function donate_inplace_mul(x, y)
    x .*= y
    return nothing
end

@testset "buffer_donation" begin
    a = Reactant.ConcreteRArray(ones(2, 2))
    b = Reactant.ConcreteRArray(3 * ones(2, 2))
    f = Reactant.compile(donate_fill_x_with_2, (a, b))
    f(a, b)
    @test convert(Array, a) == 2 * ones(2, 2)

    a = Reactant.ConcreteRArray(2 * ones(2, 2))
    b = Reactant.ConcreteRArray(3 * ones(2, 2))
    f = Reactant.compile(donate_inplace_mul, (a, b))
    f(a, b)
    @test convert(Array, a) == 6 * ones(2, 2)
end
