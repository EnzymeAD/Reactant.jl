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

# Concrete Types
## ConcreteRNumber
mutable struct ConcreteRNumber{T} <: RNumber{T}
    data::XLA.AsyncBuffer
end

@leaf ConcreteRNumber

XLA.await(x::ConcreteRNumber) = XLA.await(x.data)
XLA.client(x::ConcreteRNumber) = XLA.client(x.data)
XLA.device(x::ConcreteRNumber) = XLA.device(x.data)

function ConcreteRNumber{T}(data::T2; kwargs...) where {T<:Number,T2<:Number}
    return ConcreteRNumber{T}(ConcreteRArray(fill(convert(T, data)); kwargs...).data[1])
end
ConcreteRNumber(data::T; kwargs...) where {T<:Number} = ConcreteRNumber{T}(data; kwargs...)

## ConcreteRArray
# XXX: make data into a tuple of arrays
mutable struct ConcreteRArray{T,N,D,S<:Sharding.AbstractFinalizedSharding} <: RArray{T,N}
    data::Array{XLA.AsyncBuffer,D}
    shape::NTuple{N,Int}
    sharding::S
end

@leaf ConcreteRArray
Adapt.parent_type(::Type{<:ConcreteRArray{T,N}}) where {T,N} = ConcreteRArray{T,N}
Adapt.parent_type(::Type{ConcreteRArray{T,N,D,S}}) where {T,N,D,S} = ConcreteRArray{T,N,D,S}

Base.@deprecate ConcreteRArray(data::Number; kwargs...) ConcreteRNumber(data; kwargs...)

function ConcreteRArray{T,N}(data::XLA.AsyncBuffer, shape::NTuple{N,Int}) where {T,N}
    return ConcreteRArray{T,N,1,Sharding.FinalizedNoSharding}(
        [data], shape, Sharding.FinalizedNoSharding(Sharding.NoSharding())
    )
end

function ConcreteRArray(
    data::Array{T,N};
    client::XLA.Client=XLA.default_backend[],
    idx::Union{Int,Nothing}=nothing,
    device::Union{Nothing,XLA.Device}=nothing,
    sharding::Sharding.AbstractSharding=Sharding.NoSharding(),
) where {T,N}
    if sharding isa Sharding.NoSharding
        if device === nothing
            idx = idx === nothing ? XLA.default_device_idx[] : idx
            device = XLA.ClientGetAddressableDevice(client, idx)
        else
            if idx !== nothing
                @assert XLA.ClientGetAddressableDevice(client, idx) == device "If both `idx` and `device` are specified, `idx` must match `device`"
            end
        end
        sdata, sharding = sharding(client, device, data)
        return ConcreteRArray{T,N,1,typeof(sharding)}(sdata, size(data), sharding)
    end
    @assert device === nothing && idx === nothing "If `sharding` is not `NoSharding`, `device` and `idx` cannot be specified!"
    sharded_data, sharding = sharding(client, data)
    return ConcreteRArray{T,N,ndims(sharded_data),typeof(sharding)}(
        sharded_data, size(data), sharding
    )
end

XLA.await(x::ConcreteRArray) = foreach(XLA.await, x.data)
XLA.client(x::ConcreteRArray) = XLA.client(x.data)
function XLA.device(x::ConcreteRArray)
    x.sharding isa Sharding.FinalizedNoSharding && return XLA.device(only(x.data))
    return nothing # This is intentional to make constructing ConcreteRArrays easier
end

const ConcreteRScalar{T} = Union{
    ConcreteRArray{T,0,1,Sharding.FinalizedNoSharding},ConcreteRNumber{T}
}
const WrappedConcreteRArray{T,N,D,S} = WrappedArray{
    T,N,ConcreteRArray,ConcreteRArray{T,N,D,S}
}
const AnyConcreteRArray{T,N,D,S} = Union{
    ConcreteRArray{T,N,D,S},WrappedConcreteRArray{T,N,D,S}
}

const UnshardedConcreteRArray{T,N} = ConcreteRArray{T,N,1,Sharding.FinalizedNoSharding}
const UnshardedWrappedConcreteRArray{T,N} = WrappedConcreteRArray{
    T,N,1,Sharding.FinalizedNoSharding
}
const AnyUnshardedConcreteRArray{T,N} = AnyConcreteRArray{
    T,N,1,Sharding.FinalizedNoSharding
}

ConcreteRArray(x::AnyConcreteRArray) = ConcreteRArray{eltype(x),ndims(x)}(x)
ConcreteRArray{T}(x::AnyConcreteRArray) where {T} = ConcreteRArray{T,ndims(x)}(x)
ConcreteRArray{T,N}(x::ConcreteRArray{T,N}) where {T,N} = x
function ConcreteRArray{T,N}(x::AnyConcreteRArray) where {T,N}
    ancestor_x = ancestor(x)
    return ConcreteRArray(
        convert(Array{T,N}, x);
        client=XLA.client(ancestor_x),
        device=XLA.device(ancestor_x),
        sharding=ancestor_x.sharding,
    )
end

## ConcreteRNG
mutable struct ConcreteRNG{D,S} <: Random.AbstractRNG
    seed::ConcreteRArray{UInt64,1,D,S}
    const algorithm::String
end
