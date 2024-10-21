using Test
using Reactant

@testset "conj" begin
    @testset "$(typeof(x))" for x in (1.0, 1.0 + 2.0im)
        x_concrete = Reactant.to_rarray(x)
        f = @compile conj(x_concrete)
        @test only(f(x_concrete)) == conj(x)
    end

    @testset "$(typeof(x))" for x in (
        fill(1.0 + 2.0im),
        fill(1.0),
        [1.0 + 2.0im; 3.0 + 4.0im],
        [1.0; 3.0],
        [1.0 + 2.0im 3.0 + 4.0im],
        [1.0 2.0],
        [1.0+2.0im 3.0+4.0im; 5.0+6.0im 7.0+8.0im],
        [1.0 3.0; 5.0 7.0],
    )
        x_concrete = Reactant.to_rarray(x)
        f = @compile conj(x_concrete)
        @test f(x_concrete) == conj(x)
    end
end

@testset "conj!" begin
    @testset "$(typeof(x))" for x in (
        fill(1.0 + 2.0im),
        fill(1.0),
        [1.0 + 2.0im; 3.0 + 4.0im],
        [1.0; 3.0],
        [1.0 + 2.0im 3.0 + 4.0im],
        [1.0 2.0],
        [1.0+2.0im 3.0+4.0im; 5.0+6.0im 7.0+8.0im],
        [1.0 3.0; 5.0 7.0],
    )
        x_concrete = Reactant.to_rarray(x)
        f = @compile conj!(x_concrete)
        @test f(x_concrete) == conj(x)
        @test x_concrete == conj(x)
    end
end
