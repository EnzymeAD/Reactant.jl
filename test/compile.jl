using Reactant
using Test

Base.sum(x::NamedTuple{(:a,),Tuple{T}}) where {T<:Reactant.TracedRArray} = (; a=sum(x.a))

@testset "compile" begin
    @testset "create_result" begin
        @testset "NamedTuple" begin
            x = (; a=rand(4, 3))
            x2 = (; a=Reactant.ConcreteRArray(x.a))

            f = Reactant.compile(sum, (x2,))

            @test f(x2) isa @NamedTuple{a::Reactant.ConcreteRArray{Float64,0}}
            @test isapprox(f(x2).a, sum(x.a))
        end
    end

    @testset "world-age" begin
        a = Reactant.ConcreteRArray(ones(2, 10))
        b = Reactant.ConcreteRArray(ones(10, 2))

        function fworld(x, y)
            g = Reactant.compile(*, (a, b))
            return g(x, y)
        end

        @test fworld(a, b) ≈ ones(2, 2) * 10
    end
end
