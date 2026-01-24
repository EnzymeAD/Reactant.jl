# Abstract type hierarchy for Reactant numbers
# RInteger and RFloat are subtypes of Integer and AbstractFloat respectively
# This allows Reactant to trace through code that uses methods specializing on Real
abstract type RInteger{T<:ReactantPrimitive} <: Integer end
abstract type RFloat{T<:ReactantPrimitive} <: AbstractFloat end
abstract type RComplex{T<:ReactantPrimitive} <: Number end

const RReal{T} = Union{RInteger{T},RFloat{T}}
const RNumber{T} = Union{RReal{T},RComplex{T}}

abstract type AbstractConcreteInteger{T} <: RInteger{T} end
abstract type AbstractConcreteFloat{T} <: RFloat{T} end
abstract type AbstractConcreteComplex{T} <: RComplex{T} end

const AbstractConcreteReal{T} = Union{AbstractConcreteInteger{T},AbstractConcreteFloat{T}}
const AbstractConcreteNumber{T} = Union{AbstractConcreteReal{T},AbstractConcreteComplex{T}}

abstract type RArray{T,N} <: DenseArray{T,N} end

abstract type AbstractConcreteArray{T,N} <: RArray{T,N} end

function Base.getproperty(x::Union{AbstractConcreteArray,AbstractConcreteNumber}, f::Symbol)
    f === :data && x.donated && error("$(typeof(x)) has already been donated!")
    return getfield(x, f)
end

function Base.setproperty!(
    x::Union{AbstractConcreteArray,AbstractConcreteNumber}, f::Symbol, v
)
    f === :data && (x.donated = false)
    return setfield!(x, f, v)
end

function mark_donated!(x::Union{AbstractConcreteArray,AbstractConcreteNumber})
    x.donated && error("Can't donate an already-donated object")
    setfield!(x, :donated, true)
    return nothing
end

has_padding(_) = false
function get_padding(x)
    if hasfield(typeof(x), :padding)
        x.padding !== nothing && return x.padding
    end
    return ntuple(Returns(0), ndims(x))
end

# Traced Types

## MissingTracedValue -- defined in ReactantCore
@leaf MissingTracedValue

## TracedRInteger, TracedRFloat, TracedRComplex
for (TracedType, BaseType) in (
    (:TracedRInteger, :RInteger),
    (:TracedRFloat, :RFloat),
    (:TracedRComplex, :RComplex),
)
    @eval begin
        mutable struct $TracedType{T} <: $BaseType{T}
            paths::Tuple
            mlir_data::Union{Nothing,MLIR.IR.Value}

            function $TracedType{T}(
                paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}
            ) where {T}
                if !isnothing(mlir_data)
                    @assert size(MLIR.IR.type(mlir_data)) == ()
                end
                return new{T}(paths, mlir_data)
            end
        end

        @leaf $TracedType
    end
end

const TracedRReal{T} = Union{TracedRInteger{T},TracedRFloat{T}}
const TracedRNumber{T} = Union{TracedRReal{T},TracedRComplex{T}}

# Type-returning helper functions for dispatch on element type
@inline traced_number_type(::Type{T}) where {T<:Complex} = TracedRComplex{T}
@inline traced_number_type(::Type{T}) where {T<:Integer} = TracedRInteger{T}
@inline traced_number_type(::Type{Bool}) = TracedRInteger{Bool}
@inline traced_number_type(::Type{T}) where {T<:AbstractFloat} = TracedRFloat{T}

@inline pjrt_number_type(::Type{T}, ::Val{D}) where {T<:Complex,D} = ConcretePJRTComplex{T,D}
@inline pjrt_number_type(::Type{T}, ::Val{D}) where {T<:Integer,D} = ConcretePJRTInteger{T,D}
@inline pjrt_number_type(::Type{Bool}, ::Val{D}) where {D} = ConcretePJRTInteger{Bool,D}
@inline pjrt_number_type(::Type{T}, ::Val{D}) where {T<:AbstractFloat,D} = ConcretePJRTFloat{T,D}

@inline ifrt_number_type(::Type{T}) where {T<:Complex} = ConcreteIFRTComplex{T}
@inline ifrt_number_type(::Type{T}) where {T<:Integer} = ConcreteIFRTInteger{T}
@inline ifrt_number_type(::Type{Bool}) = ConcreteIFRTInteger{Bool}
@inline ifrt_number_type(::Type{T}) where {T<:AbstractFloat} = ConcreteIFRTFloat{T}

# Type without dimension parameter (for convenience constructors)
@inline pjrt_number_type_nod(::Type{T}) where {T<:Complex} = ConcretePJRTComplex{T}
@inline pjrt_number_type_nod(::Type{T}) where {T<:Integer} = ConcretePJRTInteger{T}
@inline pjrt_number_type_nod(::Type{Bool}) = ConcretePJRTInteger{Bool}
@inline pjrt_number_type_nod(::Type{T}) where {T<:AbstractFloat} = ConcretePJRTFloat{T}

# Unparameterized type constructors for use with TypeVar in UnionAll handling
@inline pjrt_number_type_unparameterized(::TypeVar, ::Val{D}) where {D} = ConcretePJRTFloat
@inline ifrt_number_type_unparameterized(::TypeVar) = ConcreteIFRTFloat

# Helper methods to create appropriate traced type based on element type
@inline TracedRNumber{T}(paths::Tuple, mlir_data) where {T<:Complex} =
    TracedRComplex{T}(paths, mlir_data)
@inline TracedRNumber{T}(paths::Tuple, mlir_data) where {T<:Integer} =
    TracedRInteger{T}(paths, mlir_data)
@inline TracedRNumber{Bool}(paths::Tuple, mlir_data) = TracedRInteger{Bool}(paths, mlir_data)
@inline TracedRNumber{T}(paths::Tuple, mlir_data) where {T<:AbstractFloat} =
    TracedRFloat{T}(paths, mlir_data)

Base.elsize(::Type{<:TracedRNumber{T}}) where {T} = sizeof(T)
Base.elsize(::Type{<:RNumber{T}}) where {T} = sizeof(T)
Base.elsize(::Type{<:AbstractConcreteNumber{T}}) where {T} = sizeof(T)
Base.elsize(::Type{<:AbstractConcreteArray{T}}) where {T} = sizeof(T)

function repath(x::TracedRInteger{T}, paths) where {T}
    return TracedRInteger{T}(paths, x.mlir_data)
end

function repath(x::TracedRFloat{T}, paths) where {T}
    return TracedRFloat{T}(paths, x.mlir_data)
end

function repath(x::TracedRComplex{T}, paths) where {T}
    return TracedRComplex{T}(paths, x.mlir_data)
end

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

function repath(x::TracedRArray{T,N}, paths) where {T,N}
    return TracedRArray{T,N}(paths, x.mlir_data, x.shape)
end

@leaf TracedRArray
Adapt.parent_type(::Type{TracedRArray{T,N}}) where {T,N} = TracedRArray{T,N}

## TracedStepRangeLen
struct TracedStepRangeLen{T,R,S,L} <: AbstractRange{T}
    ref::R
    step::S
    len::L
    offset::L
end

@leaf TracedStepRangeLen
function Adapt.parent_type(::Type{TracedStepRangeLen{T,R,S,L}}) where {T,R,S,L}
    return TracedStepRangeLen{T,R,S,L}
end

## TracedUnitRange
struct TracedUnitRange{T} <: AbstractUnitRange{T}
    start::T
    stop::T
    function TracedUnitRange{T}(start::T, stop::T) where {T}
        return new(start, unitrange_last(start, stop))
    end
end

function unitrange_last(start::Integer, stop::Integer)
    return ifelse(stop >= start, stop, convert(typeof(stop), start - oneunit(start - stop)))
end
function unitrange_last(start, stop)
    return ifelse(
        stop >= start,
        convert(typeof(stop), start + floor(stop - start)),
        convert(typeof(stop), start - oneunit(start - stop)),
    )
end

@leaf TracedUnitRange
Adapt.parent_type(::Type{TracedUnitRange{T}}) where {T} = TracedUnitRange{T}

const AnyTracedRArray{T,N} = AbstractArray{<:TracedRNumber{T},N}
const AnyTracedRVector{T} = AnyTracedRArray{T,1}
const AnyTracedRMatrix{T} = AnyTracedRArray{T,2}
const AnyTracedRVecOrMat{T} = Union{AnyTracedRVector{T},AnyTracedRMatrix{T}}

const TracedRVector{T} = TracedRArray{T,1}
const TracedRMatrix{T} = TracedRArray{T,2}
const TracedRVecOrMat{T} = Union{TracedRVector{T},TracedRMatrix{T}}

const WrappedTracedRArray{T,N} = WrappedArray{T,N,TracedRArray,TracedRArray{T,N}}

# Concrete Types
## ConcretePJRTInteger, ConcretePJRTFloat, ConcretePJRTComplex
for (ConcreteType, BaseType) in (
    (:ConcretePJRTInteger, :AbstractConcreteInteger),
    (:ConcretePJRTFloat, :AbstractConcreteFloat),
    (:ConcretePJRTComplex, :AbstractConcreteComplex),
)
    @eval begin
        mutable struct $ConcreteType{T,D} <: $BaseType{T}
            data::NTuple{D,XLA.PJRT.AsyncBuffer}
            sharding::Sharding.ShardInfo
            donated::Bool

            function $ConcreteType{T,D}(
                data::NTuple{D,XLA.PJRT.AsyncBuffer}, sharding::Sharding.ShardInfo
            ) where {T,D}
                return new{T,D}(data, sharding, false)
            end
        end

        $ConcreteType{T,1}(x::Number) where {T} = $ConcreteType{T}(x)

        function $ConcreteType{T}(data::Tuple{XLA.PJRT.AsyncBuffer}) where {T}
            return $ConcreteType{T,1}(data, Sharding.NoShardInfo())
        end

        function $ConcreteType{T}(
            data::NTuple{D,XLA.PJRT.AsyncBuffer}, sharding
        ) where {T,D}
            return $ConcreteType{T,D}(data, sharding)
        end

        @leaf $ConcreteType
    end
end

const ConcretePJRTReal{T,D} = Union{ConcretePJRTInteger{T,D},ConcretePJRTFloat{T,D}}
const ConcretePJRTNumber{T,D} = Union{ConcretePJRTReal{T,D},ConcretePJRTComplex{T,D}}

# Helper functions to create appropriate ConcretePJRTNumber based on element type
@inline function _ConcretePJRTNumber(
    ::Type{T}, ::Val{D}, data::NTuple{D,XLA.PJRT.AsyncBuffer}, sharding::Sharding.ShardInfo
) where {T<:Complex,D}
    return ConcretePJRTComplex{T,D}(data, sharding)
end

@inline function _ConcretePJRTNumber(
    ::Type{T}, ::Val{D}, data::NTuple{D,XLA.PJRT.AsyncBuffer}, sharding::Sharding.ShardInfo
) where {T<:Integer,D}
    return ConcretePJRTInteger{T,D}(data, sharding)
end

@inline function _ConcretePJRTNumber(
    ::Type{Bool}, ::Val{D}, data::NTuple{D,XLA.PJRT.AsyncBuffer}, sharding::Sharding.ShardInfo
) where {D}
    return ConcretePJRTInteger{Bool,D}(data, sharding)
end

@inline function _ConcretePJRTNumber(
    ::Type{T}, ::Val{D}, data::NTuple{D,XLA.PJRT.AsyncBuffer}, sharding::Sharding.ShardInfo
) where {T<:AbstractFloat,D}
    return ConcretePJRTFloat{T,D}(data, sharding)
end

function ConcretePJRTNumber{T}(data::T2; kwargs...) where {T<:Number,T2<:Number}
    carray = ConcretePJRTArray(fill(convert(T, data)); kwargs...)
    if !Sharding.is_sharded(carray.sharding)
        return _ConcretePJRTNumber(T, Val(1), (carray.data[1],), carray.sharding)
    end
    @assert all(isnothing, carray.sharding.partition_spec) "ConcretePJRTNumber cannot be \
                                                            sharded"
    return _ConcretePJRTNumber(T, Val(length(carray.data)), carray.data, carray.sharding)
end
function ConcretePJRTNumber{T}(
    data::ConcretePJRTNumber{T2}; kwargs...
) where {T<:Number,T2<:Number}
    return ConcretePJRTNumber{T}(
        convert(T, data);
        client=XLA.client(data),
        device=XLA.device(data),
        data.sharding,
        kwargs...,
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
mutable struct ConcretePJRTArray{T,N,D} <: AbstractConcreteArray{T,N}
    data::NTuple{D,XLA.PJRT.AsyncBuffer}
    shape::NTuple{N,Int}
    sharding::Sharding.ShardInfo
    donated::Bool

    function ConcretePJRTArray{T,N,D}(
        data::NTuple{D,XLA.PJRT.AsyncBuffer},
        shape::NTuple{N,Int},
        sharding::Sharding.ShardInfo,
    ) where {T,N,D}
        return new{T,N,D}(data, shape, sharding, false)
    end
end

@leaf ConcretePJRTArray
Adapt.parent_type(::Type{<:ConcretePJRTArray{T,N}}) where {T,N} = ConcretePJRTArray{T,N}
function Adapt.parent_type(::Type{ConcretePJRTArray{T,N,D}}) where {T,N,D}
    return ConcretePJRTArray{T,N,D}
end

# XXX (Deprecated): remove in v0.3
Base.@deprecate ConcretePJRTArray(data::Number; kwargs...) ConcretePJRTNumber(
    data; kwargs...
)

function ConcretePJRTArray{T,N}(
    data::Tuple{XLA.PJRT.AsyncBuffer}, shape::NTuple{N,Int}
) where {T,N}
    return ConcretePJRTArray{T,N,1}(data, shape, Sharding.NoShardInfo())
end
function ConcretePJRTArray{T,N}(
    data::NTuple{D,XLA.PJRT.AsyncBuffer}, shape::NTuple{N,Int}, sharding
) where {T,N,D}
    return ConcretePJRTArray{T,N,D}(data, shape, sharding)
end

function ConcretePJRTArray(
    data::Array{T,N};
    client::Union{Nothing,XLA.PJRT.Client}=nothing,
    idx::Union{Int,Nothing}=nothing,
    device::Union{Nothing,XLA.PJRT.Device}=nothing,
    sharding::Sharding.AbstractSharding=Sharding.NoSharding(),
) where {T,N}
    theclient, thedevice = _select_client_and_device(client, idx, device, sharding)
    sharded_data, shardinfo = sharding(theclient, thedevice, data)
    shape = size(data)
    nsharded = length(sharded_data)
    return ConcretePJRTArray{T,N,nsharded}(sharded_data, shape, shardinfo)
end

Base.wait(x::Union{ConcretePJRTArray,ConcretePJRTNumber}) = foreach(wait, x.data)
XLA.client(x::Union{ConcretePJRTArray,ConcretePJRTNumber}) = XLA.client(x.data)
function XLA.device(x::Union{ConcretePJRTArray,ConcretePJRTNumber})
    x.sharding isa Sharding.NoShardInfo && return XLA.device(only(x.data))
    return nothing # This is intentional to make constructing ConcretePJRTArrays easier
end

const ConcretePJRTScalar{T} = Union{ConcretePJRTArray{T,0},ConcretePJRTNumber{T}}
const WrappedConcretePJRTArray{T,N,D} = WrappedArray{
    T,N,ConcretePJRTArray,ConcretePJRTArray{T,N,D}
}
const AnyConcretePJRTArray{T,N,D} = Union{
    ConcretePJRTArray{T,N,D},WrappedConcretePJRTArray{T,N,D}
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
## ConcreteIFRTInteger, ConcreteIFRTFloat, ConcreteIFRTComplex
for (ConcreteType, BaseType) in (
    (:ConcreteIFRTInteger, :AbstractConcreteInteger),
    (:ConcreteIFRTFloat, :AbstractConcreteFloat),
    (:ConcreteIFRTComplex, :AbstractConcreteComplex),
)
    @eval begin
        mutable struct $ConcreteType{T} <: $BaseType{T}
            data::XLA.IFRT.AsyncArray
            sharding::Sharding.ShardInfo
            donated::Bool

            function $ConcreteType{T}(
                data::XLA.IFRT.AsyncArray, sharding::Sharding.ShardInfo
            ) where {T}
                return new{T}(data, sharding, false)
            end
        end

        function $ConcreteType{T}(data::XLA.IFRT.AsyncArray) where {T}
            return $ConcreteType{T}(data, Sharding.NoShardInfo())
        end

        @leaf $ConcreteType
    end
end

const ConcreteIFRTReal{T} = Union{ConcreteIFRTInteger{T},ConcreteIFRTFloat{T}}
const ConcreteIFRTNumber{T} = Union{ConcreteIFRTReal{T},ConcreteIFRTComplex{T}}

# Helper functions to create appropriate ConcreteIFRTNumber based on element type
@inline function _ConcreteIFRTNumber(
    ::Type{T}, data::XLA.IFRT.AsyncArray, sharding::Sharding.ShardInfo
) where {T<:Complex}
    return ConcreteIFRTComplex{T}(data, sharding)
end

@inline function _ConcreteIFRTNumber(
    ::Type{T}, data::XLA.IFRT.AsyncArray, sharding::Sharding.ShardInfo
) where {T<:Integer}
    return ConcreteIFRTInteger{T}(data, sharding)
end

@inline function _ConcreteIFRTNumber(
    ::Type{Bool}, data::XLA.IFRT.AsyncArray, sharding::Sharding.ShardInfo
)
    return ConcreteIFRTInteger{Bool}(data, sharding)
end

@inline function _ConcreteIFRTNumber(
    ::Type{T}, data::XLA.IFRT.AsyncArray, sharding::Sharding.ShardInfo
) where {T<:AbstractFloat}
    return ConcreteIFRTFloat{T}(data, sharding)
end

function ConcreteIFRTNumber{T}(data::T2; kwargs...) where {T<:Number,T2<:Number}
    carray = ConcreteIFRTArray(fill(convert(T, data)); kwargs...)
    return _ConcreteIFRTNumber(T, carray.data, carray.sharding)
end
function ConcreteIFRTNumber{T}(
    data::ConcreteIFRTNumber{T2}; kwargs...
) where {T<:Number,T2<:Number}
    return ConcreteIFRTNumber{T}(
        convert(T, data);
        client=XLA.client(data),
        device=XLA.device(data),
        data.sharding,
        kwargs...,
    )
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
mutable struct ConcreteIFRTArray{T,N,P<:Union{Nothing,NTuple{N,Int}}} <:
               AbstractConcreteArray{T,N}
    data::XLA.IFRT.AsyncArray
    shape::NTuple{N,Int}
    sharding::Sharding.ShardInfo
    donated::Bool
    padding::P

    function ConcreteIFRTArray{T,N}(
        data::XLA.IFRT.AsyncArray,
        shape::NTuple{N,Int},
        sharding::Sharding.ShardInfo,
        padding::Union{Nothing,NTuple{N,Int}}=nothing,
    ) where {T,N}
        return new{T,N,typeof(padding)}(data, shape, sharding, false, padding)
    end
end

has_padding(::ConcreteIFRTArray{T,N,Nothing}) where {T,N} = false
has_padding(x::ConcreteIFRTArray{T,N,P}) where {T,N,P} = !all(iszero, x.padding)

@leaf ConcreteIFRTArray

Adapt.parent_type(::Type{<:ConcreteIFRTArray{T,N}}) where {T,N} = ConcreteIFRTArray{T,N}

function ConcreteIFRTArray{T,N}(data::XLA.IFRT.AsyncArray, shape::NTuple{N,Int}) where {T,N}
    return ConcreteIFRTArray{T,N}(data, shape, Sharding.NoShardInfo())
end
function ConcreteIFRTArray{T,N}(
    data::XLA.IFRT.AsyncArray, shape::NTuple{N,Int}, sharding
) where {T,N}
    return ConcreteIFRTArray{T,N}(data, shape, sharding)
end

function ConcreteIFRTArray(
    data::Array{T,N};
    client::Union{Nothing,XLA.IFRT.Client}=nothing,
    idx::Union{Int,Nothing}=nothing,
    device::Union{Nothing,XLA.IFRT.Device}=nothing,
    sharding::Sharding.AbstractSharding=Sharding.NoSharding(),
) where {T,N}
    theclient, thedevice = _select_client_and_device(client, idx, device, sharding)
    shape = size(data)
    # ToDo: How to use specified device (non-sharded case)?
    sharded_data, shardinfo, padding = sharding(theclient, nothing, data)
    return ConcreteIFRTArray{T,N}(sharded_data, shape, shardinfo, padding)
end

# Assemble data from multiple arrays. Needed in distributed setting where each process wont
# have enough host memory to hold all the arrays. We assume that the data is only provided
# for all of the addressable devices.
# TODO: Implement Padding for this version. A bit more finicky that the above case
function ConcreteIFRTArray(
    data::Vector{Array{T,N}},
    array_size::Dims{N},
    data_to_addressable_shard::Vector{Vector{Int64}}=[[i] for i in 1:length(data)];
    client::Union{Nothing,XLA.IFRT.Client}=nothing,
    sharding::Sharding.AbstractSharding,
) where {T,N}
    @assert Sharding.is_sharded(sharding)
    @assert length(data) == length(data_to_addressable_shard)

    client = client === nothing ? XLA.default_backend() : client

    (; hlo_sharding) = Sharding.HloSharding(sharding, array_size)
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

# RNGs
struct ReactantRNG{S<:Union{<:AbstractConcreteArray{UInt64,1},TracedRArray{UInt64,1}}} <:
       Random.AbstractRNG
    seed::S
    algorithm::String
end

Base.@deprecate_binding ConcreteRNG ReactantRNG
Base.@deprecate_binding TracedRNG ReactantRNG

"""
    ConcreteRArray{T}(
        undef, shape::Dims;
        client::Union{Nothing,XLA.AbstractClient} = nothing,
        device::Union{Nothing,XLA.AbstractDevice} = nothing,
        sharding::Sharding.AbstractSharding = Sharding.NoSharding(),
    )

    ConcretePJRTArray{T}(undef, shape::Integer...; kwargs...)

    ConcretePJRTArray(data::Array; kwargs...)

Allocate an uninitialized `ConcreteRArray` of element type `T` and size
`shape` or convert an `Array` to a `ConcreteRArray`.

# Implementation

Depending on the Reactant `xla_runtime` preference setting, `ConcreteRArray`
is an alias for `ConcretePJRTArray` or `ConcreteIFRTArray`. User code should
use `ConcreteRArray`.
"""
const ConcreteRArray = @static if XLA.REACTANT_XLA_RUNTIME == "PJRT"
    ConcretePJRTArray
elseif XLA.REACTANT_XLA_RUNTIME == "IFRT"
    ConcreteIFRTArray
end

"""
    ConcreteRNumber(
        x::Number;
        client::Union{Nothing,XLA.AbstractClient} = nothing,
        device::Union{Nothing,XLA.AbstractDevice} = nothing,
        sharding::Sharding.AbstractSharding = Sharding.NoSharding(),
    )

    ConcreteRNumber{T<:Number}(x; kwargs...)

Wrap a `Number` in a `ConcreteRNumber`.

# Implementation

Depending on the Reactant `xla_runtime` preference setting, `ConcreteRArray`
is an alias for `ConcretePJRTNumber` or `ConcreteIFRTNumber`. User code should
use `ConcreteRNumber`.
"""
const ConcreteRNumber = @static if XLA.REACTANT_XLA_RUNTIME == "PJRT"
    ConcretePJRTNumber
elseif XLA.REACTANT_XLA_RUNTIME == "IFRT"
    ConcreteIFRTNumber
end

## Other Aliases based on the set preferences
@static if XLA.REACTANT_XLA_RUNTIME == "PJRT"
    const AnyConcreteRArray = AnyConcretePJRTArray
elseif XLA.REACTANT_XLA_RUNTIME == "IFRT"
    const AnyConcreteRArray = AnyConcreteIFRTArray
end

const UnionAnyConcreteRArray{T,N,S} = Union{
    AnyConcreteIFRTArray{T,N,S},AnyConcretePJRTArray{T,N,S}
}

for aType in (:ConcretePJRTArray, :ConcreteIFRTArray)
    @eval function $(aType){T}(::UndefInitializer, shape::Integer...; kwargs...) where {T}
        return $(aType){T}(undef, Dims(shape); kwargs...)
    end
end

function ConcretePJRTArray{T}(
    ::UndefInitializer,
    shape::Dims;
    client::Union{Nothing,XLA.AbstractClient}=nothing,
    idx::Union{Int,Nothing}=nothing,
    device::Union{Nothing,XLA.AbstractDevice}=nothing,
    sharding::Sharding.AbstractSharding=Sharding.NoSharding(),
) where {T}
    theclient, thedevice = _select_client_and_device(client, idx, device, sharding)
    sharded_data, shardinfo = sharding(theclient, thedevice, T, shape)
    N = length(shape)
    nsharded = length(sharded_data)
    return ConcretePJRTArray{T,N,nsharded}(sharded_data, shape, shardinfo)
end

function ConcreteIFRTArray{T}(
    ::UndefInitializer,
    shape::Dims;
    client::Union{Nothing,XLA.AbstractClient}=nothing,
    idx::Union{Int,Nothing}=nothing,
    device::Union{Nothing,XLA.AbstractDevice}=nothing,
    sharding::Sharding.AbstractSharding=Sharding.NoSharding(),
) where {T}
    theclient, thedevice = _select_client_and_device(client, idx, device, sharding)
    N = length(shape)
    # ToDo: How to avoid allocating dummy array on host?
    dummy_array = Array{T}(undef, shape)
    # ToDo: How to use specified device (non-sharded case)?
    sharded_data, shardinfo, padding = sharding(theclient, nothing, dummy_array)
    return ConcreteIFRTArray{T,N}(sharded_data, shape, shardinfo, padding)
end

function _select_client_and_device(
    client::Union{Nothing,XLA.AbstractClient},
    idx::Union{Int,Nothing},
    device::Union{Nothing,XLA.AbstractDevice},
    sharding::Sharding.AbstractSharding,
)
    if Sharding.is_sharded(sharding)
        # ToDo: Throw ArgumentError instead of just warning?
        idx isa Nothing ||
            @warn "device index should not be specified for sharded XLA arrays, ignoring it."
        device isa Nothing ||
            @warn "device should not be specified for sharded XLA arrays, ignoring it."
        theclient = client isa Nothing ? XLA.default_backend() : client
        thedevice = nothing
    else
        if device isa Nothing
            theclient = client isa Nothing ? XLA.default_backend() : client
            if idx isa Nothing
                thedevice = XLA.default_device(theclient)
            else
                thedevice = XLA.get_device(theclient, idx)
            end
        else
            thedevice = device
            if client isa Nothing
                theclient = XLA.client(thedevice)
            else
                theclient = client
                XLA.client(thedevice) == theclient ||
                    throw(ArgumentError("XLA device does not match XLA client"))
            end
            if !(idx isa Nothing)
                XLA.get_device(theclient, idx) === thedevice || throw(
                    ArgumentError("XLA device does not match XLA client and device index"),
                )
            end
        end
    end

    return theclient, thedevice
end
