using Test
using Reactant: use_overlayed_version, TracedRNumber, TracedRArray

struct MockStruct{A}
    data::A
end

@testset "use_overlayed_version" begin
    # primitive types
    @test use_overlayed_version(Int(1)) == false
    @test use_overlayed_version(Float32(1.0)) == false
    @test use_overlayed_version(Float64(1.0)) == false
    @test use_overlayed_version(1.0 + 1.0im) == false

    # concrete types
    @test use_overlayed_version(ConcreteRArray([1])) == false
    @test use_overlayed_version(ConcreteRArray([1.0])) == false
    @test use_overlayed_version(ConcreteRNumber(1)) == false
    @test use_overlayed_version(ConcreteRNumber(1.0)) == false

    # traced types
    @test use_overlayed_version(TracedRNumber{Int}((), nothing)) == true
    @test use_overlayed_version(TracedRArray{Int,1}((), nothing, (4,))) == true

    # array types
    @test use_overlayed_version([1]) == false
    @test use_overlayed_version([ConcreteRArray([1])]) == false
    @test use_overlayed_version([TracedRNumber{Int}((), nothing)]) == true
    @test use_overlayed_version([TracedRArray{Int,1}((), nothing, (1,))]) == true
    @test use_overlayed_version(Vector{Union{}}(undef, 1)) == false

    # composed types
    @test use_overlayed_version(MockStruct(1)) == false
    @test use_overlayed_version(MockStruct([1])) == false
    @test use_overlayed_version(MockStruct(ConcreteRArray([1]))) == false
    @test use_overlayed_version(MockStruct(ConcreteRNumber(1))) == false
    @test use_overlayed_version(MockStruct(TracedRArray{Int,1}((), nothing, (4,)))) == true
    @test use_overlayed_version(MockStruct(TracedRNumber{Int}((), nothing))) == true
end
