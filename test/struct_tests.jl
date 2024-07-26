@testitem "struct" begin
    # from bsc-quantic/Tenet.jl
    struct MockTensor{T,N,A<:AbstractArray{T,N}}
        data::A
        inds::Vector{Symbol}
    end

    MockTensor(data::A, inds) where {T,N,A<:AbstractArray{T,N}} = MockTensor{T,N,A}(data, inds)
    Base.parent(t::MockTensor) = t.data

    Base.cos(x::MockTensor) = MockTensor(cos(parent(x)), x.inds)

    mutable struct MutableMockTensor{T,N,A<:AbstractArray{T,N}}
        data::A
        inds::Vector{Symbol}
    end

    function MutableMockTensor(data::A, inds) where {T,N,A<:AbstractArray{T,N}}
        return MutableMockTensor{T,N,A}(data, inds)
    end
    Base.parent(t::MutableMockTensor) = t.data

    Base.cos(x::MutableMockTensor) = MutableMockTensor(cos(parent(x)), x.inds)

    # modified from JuliaCollections/DataStructures.jl
    # NOTE original uses abstract type instead of union, which is not supported
    mutable struct MockLinkedList{T}
        head::T
        tail::Union{MockLinkedList{T},Nothing}
    end

    function list(x::T...) where {T}
        l = nothing
        for i in Iterators.reverse(eachindex(x))
            l = MockLinkedList{T}(x[i], l)
        end
        return l
    end

    function Base.sum(x::MockLinkedList{T}) where {T}
        if isnothing(x.tail)
            return sum(x.head)
        else
            return sum(x.head) + sum(x.tail)
        end
    end

    @testset "Struct" begin
        @testset "MockTensor" begin
            @testset "immutable" begin
                x = MockTensor(rand(4, 4), [:i, :j])
                x2 = MockTensor(Reactant.ConcreteRArray(parent(x)), x.inds)

                f = Reactant.compile(cos, (x2,))
                y = f(x2)

                @test y isa MockTensor{Float64,2,Reactant.ConcreteRArray{Float64,(4, 4),2}}
                @test isapprox(parent(y), cos.(parent(x)))
                @test x.inds == [:i, :j]
            end

            @testset "mutable" begin
                x = MutableMockTensor(rand(4, 4), [:i, :j])
                x2 = MutableMockTensor(Reactant.ConcreteRArray(parent(x)), x.inds)

                f = Reactant.compile(cos, (x2,))
                y = f(x2)

                @test y isa
                    MutableMockTensor{Float64,2,Reactant.ConcreteRArray{Float64,(4, 4),2}}
                @test isapprox(parent(y), cos.(parent(x)))
                @test x.inds == [:i, :j]
            end
        end

        @testset "MockLinkedList" begin
            x = [rand(2, 2) for _ in 1:2]
            x2 = list(x...)
            x3 = Reactant.make_tracer(IdDict(), x2, (), Reactant.ArrayToConcrete)

            # TODO this should be able to run without problems, but crashes
            @test_broken begin
                f = Reactant.compile(identity, (x3,))
                isapprox(f(x3), x3)
            end

            f = Reactant.compile(sum, (x3,))

            y = sum(x2)
            y3 = f(x3)

            @test isapprox(y, only(y3))
        end
    end
end