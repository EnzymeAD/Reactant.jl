using Reactant, Test, Adapt

# from bsc-quantic/Tenet.jl
struct MockTensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    inds::Vector{Symbol}
end

MockTensor(data::A, inds) where {T,N,A<:AbstractArray{T,N}} = MockTensor{T,N,A}(data, inds)
Base.parent(t::MockTensor) = t.data
Base.size(t::MockTensor) = size(parent(t))

Adapt.parent_type(::Type{MockTensor{T,N,A}}) where {T,N,A} = A

Base.cos(x::MockTensor) = MockTensor(cos.(parent(x)), x.inds)
bcast_cos(x::MockTensor) = cos(x)

mutable struct MutableMockTensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    inds::Vector{Symbol}
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(A::Type{<:MockTensor}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    T2 = Reactant.traced_type_inner(
        A.parameters[3], seen, mode, track_numbers, sharding, runtime
    )
    MT = MockTensor{eltype(T2),ndims(A),T2}
    return MT
end

function MutableMockTensor(data::A, inds) where {T,N,A<:AbstractArray{T,N}}
    return MutableMockTensor{T,N,A}(data, inds)
end

Base.parent(t::MutableMockTensor) = t.data
Base.size(t::MutableMockTensor) = size(parent(t))

Adapt.parent_type(::Type{MutableMockTensor{T,N,A}}) where {T,N,A} = A

Base.cos(x::MutableMockTensor) = MutableMockTensor(cos.(parent(x)), x.inds)

bcast_cos(x) = cos.(x)
bcast_cos(x::MutableMockTensor) = cos(x)

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
            x = MockTensor(Reactant.TestUtils.construct_test_array(Float64, 4, 4), [:i, :j])
            x2 = MockTensor(Reactant.to_rarray(parent(x)), x.inds)

            y = @jit(bcast_cos(x2))

            @test y isa MockTensor{Float64,2,<:ConcreteRArray{Float64,2}}
            @test size(y) == (4, 4)
            @test isapprox(parent(y), bcast_cos(parent(x)))
            @test x.inds == [:i, :j]
        end

        @testset "mutable" begin
            x = MutableMockTensor(
                Reactant.TestUtils.construct_test_array(Float64, 4, 4), [:i, :j]
            )
            x2 = MutableMockTensor(Reactant.to_rarray(parent(x)), x.inds)

            y = @jit(bcast_cos(x2))

            @test y isa MutableMockTensor{Float64,2,<:ConcreteRArray{Float64,2}}
            @test size(y) == (4, 4)
            @test isapprox(parent(y), bcast_cos(parent(x)))
            @test x.inds == [:i, :j]
        end
    end

    @testset "MockLinkedList" begin
        x = [Reactant.TestUtils.construct_test_array(Float64, 2, 2) for _ in 1:2]
        x2 = list(x...)
        x3 = Reactant.to_rarray(x2)

        # TODO this should be able to run without problems, but crashes
        @test_broken isapprox(@jit(identity(x3)), x3)

        @test isapprox(@allowscalar(sum(x3)), only(@jit(sum(x3))))
    end
end
