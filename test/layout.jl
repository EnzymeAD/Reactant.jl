using Reactant
using Test

@testset "Layout" begin
    x = reshape([1.0, 2.0, 3.0, 4.0], (2, 2))

    y = Reactant.to_rarray(x)

    y2 = convert(Array{Float64,2}, y)

    @test x == y2

    @allowscalar begin
        @test y[1, 1] == x[1, 1]
        @test y[1, 2] == x[1, 2]
        @test y[2, 1] == x[2, 1]
        @test y[2, 2] == x[2, 2]
    end
end
