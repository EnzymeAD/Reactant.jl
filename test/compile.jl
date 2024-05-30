using Reactant
using Test

@testset "compile" begin
    @testset "create_result" begin
        @testset "NamedTuple" begin
            const MockNamedTuple{T} = @NamedTuple{a::T}
            MockNamedTuple(x::T) where {T} = MockNamedTuple{T}((x,))

            x = MockNamedTuple(rand(4, 3))
            x2 = MockNamedTuple(Reactant.ConcreteRArray(x.a))

            Base.sum(x::MockNamedTuple) = MockNamedTuple((sum(x.a),))
            f = Reactant.compile(sum, (x2,))

            @test f(x2) isa MockNamedTuple{Reactant.ConcreteRArray}
            @test isapprox(f(x2).a, sum(x).a)
        end
    end
end
