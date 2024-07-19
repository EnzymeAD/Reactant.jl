using Reactant
using Test

Base.sum(x::NamedTuple{(:a,),Tuple{T}}) where {T<:Reactant.TracedRArray} = (; a=sum(x.a))

@testset "compile" begin
    @testset "create_result" begin
        @testset "NamedTuple" begin
            x = (; a=rand(4, 3))
            x2 = (; a=Reactant.ConcreteRArray(x.a))

            f = Reactant.compile(sum, (x2,))

            @test f(x2) isa @NamedTuple{a::Reactant.ConcreteRArray{T,0}} where {T}
            @test isapprox(f(x2).a, sum(x.a))
        end
    end
end
