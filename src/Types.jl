abstract type RNumber{T<:ReactantPrimitive} <: Number end

abstract type AbstractConcreteNumber{T} <: RNumber{T} end

abstract type RArray{T,N} <: AbstractArray{T,N} end

abstract type AbstractConcreteArray{T,N} <: RArray{T,N} end

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

# Concrete Types
## ConcretePJRTNumber
mutable struct ConcretePJRTNumber{T,D,S<:Sharding.ShardInfo} <: AbstractConcreteNumber{T}
    data::NTuple{D,XLA.PJRT.AsyncBuffer}
    sharding::S
end

ConcretePJRTNumber{T,1,Sharding.NoShardInfo}(x::Number) where {T} = ConcretePJRTNumber{T}(x)

function ConcretePJRTNumber{T}(data::Tuple{XLA.PJRT.AsyncBuffer}) where {T}
    return ConcretePJRTNumber{T,1,Sharding.NoShardInfo}(data, Sharding.NoShardInfo())
end

@leaf ConcretePJRTNumber

function ConcretePJRTNumber{T}(data::T2; kwargs...) where {T<:Number,T2<:Number}
    carray = ConcretePJRTArray(fill(convert(T, data)); kwargs...)
    if !Sharding.is_sharded(carray.sharding)
        return ConcretePJRTNumber{T,1,typeof(carray.sharding)}(
            (carray.data[1],), carray.sharding
        )
    end
    @assert all(isnothing, carray.sharding.partition_spec) "ConcretePJRTNumber cannot be \
                                                            sharded"
    return ConcretePJRTNumber{T,length(carray.data),typeof(carray.sharding)}(
        carray.data, carray.sharding
    )
end
function ConcretePJRTNumber(data::T; kwargs...) where {T<:Number}
    return ConcretePJRTNumber{T}(data; kwargs...)
end

## ConcretePJRTArray
mutable struct ConcretePJRTArray{T,N,D,S<:Sharding.ShardInfo} <: AbstractConcreteArray{T,N}
    data::NTuple{D,XLA.PJRT.AsyncBuffer}
    shape::NTuple{N,Int}
    sharding::S
end

@leaf ConcretePJRTArray
Adapt.parent_type(::Type{<:ConcretePJRTArray{T,N}}) where {T,N} = ConcretePJRTArray{T,N}
function Adapt.parent_type(::Type{ConcretePJRTArray{T,N,D,S}}) where {T,N,D,S}
    return ConcretePJRTArray{T,N,D,S}
end

Base.@deprecate ConcretePJRTArray(data::Number; kwargs...) ConcretePJRTNumber(
    data; kwargs...
)

function ConcretePJRTArray{T,N}(
    data::Tuple{XLA.PJRT.AsyncBuffer}, shape::NTuple{N,Int}
) where {T,N}
    return ConcretePJRTArray{T,N,1,Sharding.NoShardInfo}(
        data, shape, Sharding.NoShardInfo()
    )
end

function ConcretePJRTArray(
    data::Array{T,N};
    client::XLA.AbstractClient=XLA.default_backend(),
    idx::Union{Int,Nothing}=nothing,
    device::Union{Nothing,XLA.AbstractDevice}=nothing,
    sharding::Sharding.AbstractSharding=Sharding.NoSharding(),
) where {T,N}
    if !Sharding.is_sharded(sharding)
        if device === nothing
            if idx === nothing
                device = XLA.default_device(client)
            else
                device = XLA.get_device(client, idx)
            end
        else
            if idx !== nothing
                device_from_idx = XLA.get_device(client, idx)
                @assert device_from_idx == device "If both `idx` and `device` are \
                                                   specified, `idx` must match `device`"
            end
        end
        sdata, sharding = sharding(client, device, data)
        return ConcretePJRTArray{T,N,1,typeof(sharding)}(sdata, size(data), sharding)
    end
    @assert device === nothing && idx === nothing "If `sharding` is not `NoSharding`, `device` and `idx` cannot be specified!"
    sharded_data, sharding = sharding(client, nothing, data)
    return ConcretePJRTArray{T,N,length(sharded_data),typeof(sharding)}(
        sharded_data, size(data), sharding
    )
end

XLA.await(x::Union{ConcretePJRTArray,ConcretePJRTNumber}) = foreach(XLA.await, x.data)
XLA.client(x::Union{ConcretePJRTArray,ConcretePJRTNumber}) = XLA.client(x.data)
function XLA.device(x::Union{ConcretePJRTArray,ConcretePJRTNumber})
    x.sharding isa Sharding.NoShardInfo && return XLA.device(only(x.data))
    return nothing # This is intentional to make constructing ConcretePJRTArrays easier
end

const ConcretePJRTScalar{T} = Union{ConcretePJRTArray{T,0},ConcretePJRTNumber{T}}
const WrappedConcretePJRTArray{T,N,D,S} = WrappedArray{
    T,N,ConcretePJRTArray,ConcretePJRTArray{T,N,D,S}
}
const AnyConcretePJRTArray{T,N,D,S} = Union{
    ConcretePJRTArray{T,N,D,S},WrappedConcretePJRTArray{T,N,D,S}
}

const AnyConcreteRArray = AnyConcretePJRTArray

ConcretePJRTArray(x::AnyConcretePJRTArray) = ConcretePJRTArray{eltype(x),ndims(x)}(x)
ConcretePJRTArray{T}(x::AnyConcretePJRTArray) where {T} = ConcretePJRTArray{T,ndims(x)}(x)
ConcretePJRTArray{T,N}(x::ConcretePJRTArray{T,N}) where {T,N} = x
function ConcretePJRTArray{T,N}(x::AnyConcretePJRTArray) where {T,N}
    ancestor_x = ancestor(x)
    return ConcretePJRTArray(
        convert(Array{T,N}, x);
        client=XLA.client(ancestor_x),
        device=XLA.device(ancestor_x),
        sharding=ancestor_x.sharding,
    )
end

## ConcreteRNG
mutable struct ConcreteRNG{S<:ConcretePJRTArray} <: Random.AbstractRNG
    seed::S
    const algorithm::String
end

## Aliases to prevent breaking changes
const ConcreteRArray = ConcretePJRTArray
const ConcreteRNumber = ConcretePJRTNumber
