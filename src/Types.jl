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

const AnyTracedRArray{T,N} = AbstractArray{<:TracedRNumber{T},N}
const AnyTracedRVector{T} = AnyTracedRArray{T,1}
const AnyTracedRMatrix{T} = AnyTracedRArray{T,2}
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

function ConcretePJRTNumber(data::ConcretePJRTNumber; kwargs...)
    return ConcretePJRTNumber(
        to_number(data);
        client=XLA.client(data),
        device=XLA.device(data),
        data.sharding,
        kwargs...,
    )
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

# XXX (Deprecated): remove in v0.3
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
function ConcretePJRTArray{T,N}(
    data::NTuple{D,XLA.PJRT.AsyncBuffer}, shape::NTuple{N,Int}, sharding
) where {T,N,D}
    return ConcretePJRTArray{T,N,D,typeof(sharding)}(data, shape, sharding)
end

function ConcretePJRTArray(
    data::Array{T,N};
    client::XLA.PJRT.Client=XLA.default_backend(),
    idx::Union{Int,Nothing}=nothing,
    device::Union{Nothing,XLA.PJRT.Device}=nothing,
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
    if device !== nothing || idx !== nothing
        @warn "`device` and `idx` specified for non-`NoSharding` sharding. These arguments \
               will be ignored."
    end
    sharded_data, sharding = sharding(client, nothing, data)
    return ConcretePJRTArray{T,N,length(sharded_data),typeof(sharding)}(
        sharded_data, size(data), sharding
    )
end

Base.wait(x::Union{ConcretePJRTArray,ConcretePJRTNumber}) = foreach(wait, x.data)
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

function ConcretePJRTArray(x::AnyConcretePJRTArray; kwargs...)
    return ConcretePJRTArray{eltype(x),ndims(x)}(x; kwargs...)
end
function ConcretePJRTArray{T}(x::AnyConcretePJRTArray; kwargs...) where {T}
    return ConcretePJRTArray{T,ndims(x)}(x; kwargs...)
end
function ConcretePJRTArray{T,N}(x::AnyConcretePJRTArray; kwargs...) where {T,N}
    ancestor_x = ancestor(x)
    return ConcretePJRTArray(
        convert(Array{T,N}, x);
        client=XLA.client(ancestor_x),
        device=XLA.device(ancestor_x),
        sharding=ancestor_x.sharding,
        kwargs...,
    )
end

# While sharding is part of IFRT.Array, we still need to carry it around for compiling the
# MLIR module.
## ConcreteIFRTNumber
mutable struct ConcreteIFRTNumber{T,S<:Sharding.ShardInfo} <: AbstractConcreteNumber{T}
    data::XLA.IFRT.AsyncArray
    sharding::S
end

ConcreteIFRTNumber{T,Sharding.NoShardInfo}(x::Number) where {T} = ConcreteIFRTNumber{T}(x)

function ConcreteIFRTNumber{T}(data::XLA.IFRT.AsyncArray) where {T}
    return ConcreteIFRTNumber{T,Sharding.NoShardInfo}(data, Sharding.NoShardInfo())
end

@leaf ConcreteIFRTNumber

function ConcreteIFRTNumber{T}(data::T2; kwargs...) where {T<:Number,T2<:Number}
    carray = ConcreteIFRTArray(fill(convert(T, data)); kwargs...)
    return ConcreteIFRTNumber{T,typeof(carray.sharding)}(carray.data, carray.sharding)
end
function ConcreteIFRTNumber(data::T; kwargs...) where {T<:Number}
    return ConcreteIFRTNumber{T}(data; kwargs...)
end

function ConcreteIFRTNumber(data::ConcreteIFRTNumber; kwargs...)
    return ConcreteIFRTNumber(
        to_number(data);
        client=XLA.client(data),
        device=XLA.device(data),
        data.sharding,
        kwargs...,
    )
end

## ConcreteIFRTArray
mutable struct ConcreteIFRTArray{T,N,S<:Sharding.ShardInfo} <: AbstractConcreteArray{T,N}
    data::XLA.IFRT.AsyncArray
    shape::NTuple{N,Int}
    sharding::S
end

@leaf ConcreteIFRTArray

Adapt.parent_type(::Type{<:ConcreteIFRTArray{T,N}}) where {T,N} = ConcreteIFRTArray{T,N}
function Adapt.parent_type(::Type{<:ConcreteIFRTArray{T,N,S}}) where {T,N,S}
    return ConcreteIFRTArray{T,N,S}
end

function ConcreteIFRTArray{T,N}(data::XLA.IFRT.AsyncArray, shape::NTuple{N,Int}) where {T,N}
    return ConcreteIFRTArray{T,N,Sharding.NoShardInfo}(data, shape, Sharding.NoShardInfo())
end
function ConcreteIFRTArray{T,N}(
    data::XLA.IFRT.AsyncArray, shape::NTuple{N,Int}, sharding
) where {T,N}
    return ConcreteIFRTArray{T,N,typeof(sharding)}(data, shape, sharding)
end

function ConcreteIFRTArray(
    data::Array{T,N};
    client::XLA.IFRT.Client=XLA.default_backend(),
    idx::Union{Int,Nothing}=nothing,
    device::Union{Nothing,XLA.IFRT.Device}=nothing,
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
    else
        if device !== nothing || idx !== nothing
            @warn "`device` and `idx` specified for non-`NoSharding` sharding. These \
                   arguments will be ignored."
        end
    end
    sharded_data, sharding = sharding(client, nothing, data)
    return ConcreteIFRTArray{T,N}(sharded_data, size(data), sharding)
end

# Assemble data from multiple arrays. Needed in distributed setting where each process wont
# have enough host memory to hold all the arrays. We assume that the data is only provided
# for all of the addressable devices.
function ConcreteIFRTArray(
    data::Vector{Array{T,N}},
    array_size::Dims{N},
    data_to_addressable_shard::Vector{Vector{Int64}}=[[i] for i in 1:length(data)];
    client::XLA.IFRT.Client=XLA.default_backend(),
    sharding::Sharding.AbstractSharding,
) where {T,N}
    @assert Sharding.is_sharded(sharding)
    @assert length(data) == length(data_to_addressable_shard)

    hlo_sharding = convert(Sharding.HloSharding, sharding).hlo_sharding
    all_devices = XLA.get_device.((client,), sharding.mesh.device_ids)
    ifrt_sharding = XLA.IFRT.Sharding(all_devices, hlo_sharding)

    # Validate that all the slices are as we expected them to be
    slices, _ = XLA.sharding_to_concrete_array_indices(
        hlo_sharding, array_size, 0:(length(all_devices) - 1)
    )
    addressable_slices = [
        slice for (slice, device) in zip(slices, all_devices) if XLA.is_addressable(device)
    ]
    for (i, slice) in enumerate(addressable_slices)
        idx = findfirst(Base.Fix1(in, i), data_to_addressable_shard)
        @assert idx !== nothing
        @assert size(data[idx]) == length.(slice) "Expected data[$idx] to be at \
                                                   $(slice), but got size \
                                                   $(size(data[idx]))"
    end

    # Make the mapping 0-indexed
    @inbounds for shard_idxs in data_to_addressable_shard
        shard_idxs .-= 1
    end
    ifrt_array = XLA.IFRT.AsyncArray(
        XLA.IFRT.Array(client, data, data_to_addressable_shard, array_size, ifrt_sharding),
        nothing,
    )
    return ConcreteIFRTArray{T,N}(
        ifrt_array, array_size, Sharding.ShardInfo(sharding, slices)
    )
end

Base.wait(x::Union{ConcreteIFRTArray,ConcreteIFRTNumber}) = wait(x.data)
XLA.client(x::Union{ConcreteIFRTArray,ConcreteIFRTNumber}) = XLA.client(x.data)
function XLA.device(x::Union{ConcreteIFRTArray,ConcreteIFRTNumber})
    return XLA.device(x.data)
end

const ConcreteIFRTScalar{T} = Union{ConcreteIFRTArray{T,0},ConcreteIFRTNumber{T}}
const WrappedConcreteIFRTArray{T,N,S} = WrappedArray{
    T,N,ConcreteIFRTArray,ConcreteIFRTArray{T,N,S}
}
const AnyConcreteIFRTArray{T,N,S} = Union{
    ConcreteIFRTArray{T,N,S},WrappedConcreteIFRTArray{T,N,S}
}

function ConcreteIFRTArray(x::AnyConcreteIFRTArray; kwargs...)
    return ConcreteIFRTArray{eltype(x),ndims(x)}(x; kwargs...)
end
function ConcreteIFRTArray{T}(x::AnyConcreteIFRTArray; kwargs...) where {T}
    return ConcreteIFRTArray{T,ndims(x)}(x; kwargs...)
end
function ConcreteIFRTArray{T,N}(x::AnyConcreteIFRTArray; kwargs...) where {T,N}
    ancestor_x = ancestor(x)
    return ConcreteIFRTArray(
        convert(Array{T,N}, x);
        client=XLA.client(ancestor_x),
        device=XLA.device(ancestor_x),
        sharding=ancestor_x.sharding,
        kwargs...,
    )
end

## ConcreteRNG
mutable struct ConcreteRNG{S<:AbstractConcreteArray} <: Random.AbstractRNG
    seed::S
    const algorithm::String
end

## Aliases based on the set preferences
if XLA.REACTANT_XLA_RUNTIME == "PJRT"
    const ConcreteRArray = ConcretePJRTArray
    const ConcreteRNumber = ConcretePJRTNumber
    const AnyConcreteRArray = AnyConcretePJRTArray
elseif XLA.REACTANT_XLA_RUNTIME == "IFRT"
    const ConcreteRArray = ConcreteIFRTArray
    const ConcreteRNumber = ConcreteIFRTNumber
    const AnyConcreteRArray = AnyConcreteIFRTArray
end
