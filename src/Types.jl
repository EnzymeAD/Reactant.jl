abstract type RNumber{T<:ReactantPrimitive} <: Number end

abstract type RArray{T,N} <: AbstractArray{T,N} end

# Traced Types

## MissingTracedValue -- defined in ReactantCore
@leaf MissingTracedValue

## TracedRNumber
mutable struct TracedRNumber{T} <: RNumber{T}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}

    function TracedRNumber{T}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}
    ) where {T}
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == ()
        end
        return new{T}(paths, mlir_data)
    end
end

@leaf TracedRNumber

## TracedRArray
mutable struct TracedRArray{T,N} <: RArray{TracedRNumber{T},N}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}
    shape::NTuple{N,Int}

    function TracedRArray{T,N}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}, shape
    ) where {T,N}
        shape = Tuple(shape)
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == shape "Expected: $(shape), got: $(size(MLIR.IR.type(mlir_data)))"
        end
        return new{T,N}(paths, mlir_data, shape)
    end
end

@leaf TracedRArray
Adapt.parent_type(::Type{TracedRArray{T,N}}) where {T,N} = TracedRArray{T,N}

const WrappedTracedRArray{T,N} = WrappedArray{
    TracedRNumber{T},N,TracedRArray,TracedRArray{T,N}
}
const AnyTracedRArray{T,N} = Union{TracedRArray{T,N},WrappedTracedRArray{T,N}}
const AnyTracedRVector{T} = AnyTracedRArray{T,1}
const AnyTracedRMatrix{T} = Union{
    AnyTracedRArray{T,2},
    LinearAlgebra.Diagonal{TracedRNumber{T},TracedRArray{T,1}},
    LinearAlgebra.Tridiagonal{TracedRNumber{T},TracedRArray{T,1}},
}
const AnyTracedRVecOrMat{T} = Union{AnyTracedRVector{T},AnyTracedRMatrix{T}}

## TracedRNG
mutable struct TracedRNG <: Random.AbstractRNG
    seed::TracedRArray{UInt64,1}
    const algorithm::String
end

## XLAArray
struct XLAArray{T,N} <: RArray{T,N} end
Adapt.parent_type(::Type{XLAArray{T,N}}) where {T,N} = XLAArray{T,N}

# Concrete Types
## ConcreteRNumber
mutable struct ConcreteRNumber{T} <: RNumber{T}
    data::XLA.AsyncBuffer
end

@leaf ConcreteRNumber

## ConcreteRArray
mutable struct ConcreteRArray{T,N} <: RArray{T,N}
    data::XLA.AsyncBuffer
    shape::NTuple{N,Int}
end

@leaf ConcreteRArray
Adapt.parent_type(::Type{ConcreteRArray{T,N}}) where {T,N} = ConcreteRArray{T,N}

const WrappedConcreteRArray{T,N} = WrappedArray{T,N,ConcreteRArray,ConcreteRArray{T,N}}
const AnyConcreteRArray{T,N} = Union{ConcreteRArray{T,N},WrappedConcreteRArray{T,N}}

## ConcreteRNG
mutable struct ConcreteRNG <: Random.AbstractRNG
    seed::ConcreteRArray{UInt64,1}
    const algorithm::String
end

## ShardedConcreteRArray
# TODO: fuse this into ConcreteRArray
mutable struct ShardedConcreteRArray{T,N,M,S<:Sharding.AbstractSharding} <: RArray{T,N}
    data::Array{<:XLA.AsyncBuffer,M}
    shape::NTuple{N,Int}
    mesh::Sharding.Mesh{M}
    sharding::S
end

Base.size(x::ShardedConcreteRArray) = x.shape
