using Reactant
using Test

# from bsc-quantic/Tenet.jl
struct MockTensor{T,N,A<:AbstractArray{T,N}}
    data::A
    inds::Vector{Symbol}
end

MockTensor(data::A, inds) where {T,N,A<:AbstractArray{T,N}} = MockTensor{T,N,A}(data, inds)
Base.parent(t::MockTensor) = t.data

Base.cos(x::MockTensor) = MockTensor(cos(parent(x)), x.inds)

@testset "Struct" begin
    x = MockTensor(rand(4, 4), [:i, :j])
    x2 = MockTensor(Reactant.ConcreteRArray(parent(x)), x.inds)

    f = Reactant.compile(cos, (x2,))
    y = f(x2)

    @test y isa MockTensor{Float64,2,Reactant.ConcreteRArray{Float64,(4, 4),2}}
    @test isapprox(parent(y), cos.(parent(x)))
end

struct MutableMockTensor{T,N,A<:AbstractArray{T,N}}
    data::A
    inds::Vector{Symbol}
end

MutableMockTensor(data::A, inds) where {T,N,A<:AbstractArray{T,N}} = MutableMockTensor{T,N,A}(data, inds)
Base.parent(t::MutableMockTensor) = t.data

Base.cos(x::MutableMockTensor) = MutableMockTensor(cos(parent(x)), x.inds)

@testset "Mutable Struct" begin
    x = MutableMockTensor(rand(4, 4), [:i, :j])
    x2 = MutableMockTensor(Reactant.ConcreteRArray(parent(x)), x.inds)

    f = Reactant.compile(cos, (x2,))
    y = f(x2)

    @test y isa MutableMockTensor{Float64,2,Reactant.ConcreteRArray{Float64,(4, 4),2}}
    @test isapprox(parent(y), cos.(parent(x)))
end
