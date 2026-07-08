# Reactant numbers are split by numeric kind so that traced and concrete values
# subtype `Integer` and `AbstractFloat` where possible. Julia has no abstract
# complex type, so complex Reactant numbers can only subtype `Number`.
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
for (TracedT, AbstractT) in
    ((:TracedRInteger, :RInteger), (:TracedRFloat, :RFloat), (:TracedRComplex, :RComplex))
    @eval begin
        mutable struct $TracedT{T} <: $AbstractT{T}
            paths::Tuple
            mlir_data::Union{Nothing,MLIR.IR.Value}

            function $TracedT{T}(
                paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}
            ) where {T}
                if !isnothing(mlir_data)
                    @assert size(MLIR.IR.type(mlir_data)) == ()
                end
                return new{T}(paths, mlir_data)
            end
        end

        @leaf $TracedT
    end
end

const TracedRReal{T} = Union{TracedRInteger{T},TracedRFloat{T}}
const TracedRNumber{T} = Union{TracedRReal{T},TracedRComplex{T}}

"""
    traced_number_type(::Type{T}) where {T<:ReactantPrimitive}

The concrete traced number type for element type `T`, i.e. the member of
`TracedRNumber{T}` that matches the numeric kind of `T`.
"""
@inline traced_number_type(::Type{T}) where {T<:Integer} = TracedRInteger{T}
@inline traced_number_type(::Type{T}) where {T<:AbstractFloat} = TracedRFloat{T}
@inline traced_number_type(::Type{T}) where {T<:Complex} = TracedRComplex{T}

@inline function TracedRNumber{T}(
    paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}
) where {T}
    return traced_number_type(T)(paths, mlir_data)
end

Base.elsize(::Type{<:RNumber{T}}) where {T} = sizeof(T)
Base.elsize(::Type{<:AbstractConcreteArray{T}}) where {T} = sizeof(T)

repath(x::TracedRNumber, paths) = Core.Typeof(x)(paths, x.mlir_data)

## TracedRArray
## The element type `RT` is redundant (it is always `traced_number_type(T)`) but
## exposes the numeric kind of the elements in the `RArray` supertype, so that
## e.g. a traced float array is an `AbstractArray{<:AbstractFloat}`.
mutable struct TracedRArray{T,N,RT<:TracedRNumber{T}} <: RArray{RT,N}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}
    shape::NTuple{N,Int}

    function TracedRArray{T,N,RT}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}, shape
    ) where {T,N,RT<:TracedRNumber{T}}
        RT === traced_number_type(T) ||
            throw(ArgumentError("Element type of TracedRArray{$T} must be \
                                 $(traced_number_type(T)), got $RT"))
        shape = Tuple(shape)
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == shape "Expected: $(shape), got: $(size(MLIR.IR.type(mlir_data)))"
        end
        return new{T,N,RT}(paths, mlir_data, shape)
    end
end

function TracedRArray{T,N}(
    paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}, shape
) where {T,N}
    return TracedRArray{T,N,traced_number_type(T)}(paths, mlir_data, shape)
end

function TracedRArray{T,N}(::UndefInitializer, shape::Integer...) where {T,N}
    return similar(TracedRArray{T,N}, shape...)
end

function TracedRArray{T,N}(::UndefInitializer, shape::NTuple{N,Int}) where {T,N}
    return similar(TracedRArray{T,N}, shape)
end

function TracedRArray{T,N,RT}(
    ::UndefInitializer, shape::Integer...
) where {T,N,RT<:TracedRNumber{T}}
    return similar(TracedRArray{T,N}, shape...)
end

function TracedRArray{T,N,RT}(
    ::UndefInitializer, shape::NTuple{N,Int}
) where {T,N,RT<:TracedRNumber{T}}
    return similar(TracedRArray{T,N}, shape)
end

function repath(x::TracedRArray{T,N}, paths) where {T,N}
    return Core.Typeof(x)(paths, x.mlir_data, x.shape)
end

@leaf TracedRArray
function Adapt.parent_type(::Type{<:TracedRArray{T,N}}) where {T,N}
    return TracedRArray{T,N,traced_number_type(T)}
end

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
## NOTE(#461): we are currently storing the size as a temporary fix to avoid
## dynamic shapes. Once we support that, we can remove the size field
struct TracedUnitRange{T} <: AbstractUnitRange{T}
    start::T
    stop::T
    length::Int  # If < 0, then we need to compute it
    function TracedUnitRange{T}(start::T, stop::T, length::Int=-1) where {T}
        return new(start, unitrange_last(start, stop), length)
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

## TracedRational
struct TracedRational{T<:Integer} <: Real
    num::T
    den::T
end

@leaf TracedRational
Adapt.parent_type(::Type{TracedRational{T}}) where {T} = TracedRational{T}

const AnyTracedRArray{T,N} = AbstractArray{TracedRNumber{T},N}
const AnyTracedRVector{T} = AnyTracedRArray{T,1}
const AnyTracedRMatrix{T} = AnyTracedRArray{T,2}
const AnyTracedRVecOrMat{T} = Union{AnyTracedRVector{T},AnyTracedRMatrix{T}}

# Concrete Types
## ConcretePJRTInteger, ConcretePJRTFloat, ConcretePJRTComplex
for (ConcreteT, AbstractT) in (
    (:ConcretePJRTInteger, :AbstractConcreteInteger),
    (:ConcretePJRTFloat, :AbstractConcreteFloat),
    (:ConcretePJRTComplex, :AbstractConcreteComplex),
)
    @eval begin
        mutable struct $ConcreteT{T,D} <: $AbstractT{T}
            data::NTuple{D,XLA.PJRT.AsyncBuffer}
            sharding::Sharding.ShardInfo
            donated::Bool

            function $ConcreteT{T,D}(
                data::NTuple{D,XLA.PJRT.AsyncBuffer}, sharding::Sharding.ShardInfo
            ) where {T,D}
                return new{T,D}(data, sharding, false)
            end
        end

        @leaf $ConcreteT
    end
end

const ConcretePJRTReal{T,D} = Union{ConcretePJRTInteger{T,D},ConcretePJRTFloat{T,D}}
const ConcretePJRTNumber{T,D} = Union{ConcretePJRTReal{T,D},ConcretePJRTComplex{T,D}}

"""
    pjrt_number_type(::Type{T}) where {T<:ReactantPrimitive}

The concrete PJRT number type for element type `T` (with the device-count
parameter `D` left free), i.e. the member of `ConcretePJRTNumber{T}` that
matches the numeric kind of `T`.
"""
@inline pjrt_number_type(::Type{T}) where {T<:Integer} = ConcretePJRTInteger{T}
@inline pjrt_number_type(::Type{T}) where {T<:AbstractFloat} = ConcretePJRTFloat{T}
@inline pjrt_number_type(::Type{T}) where {T<:Complex} = ConcretePJRTComplex{T}

ConcretePJRTNumber{T,1}(x::Number) where {T} = ConcretePJRTNumber{T}(x)

function ConcretePJRTNumber{T,D}(
    data::NTuple{D,XLA.PJRT.AsyncBuffer}, sharding::Sharding.ShardInfo
) where {T,D}
    return pjrt_number_type(T){D}(data, sharding)
end

function ConcretePJRTNumber{T}(data::Tuple{XLA.PJRT.AsyncBuffer}) where {T}
    return ConcretePJRTNumber{T,1}(data, Sharding.NoShardInfo())
end

function ConcretePJRTNumber{T}(data::NTuple{D,XLA.PJRT.AsyncBuffer}, sharding) where {T,D}
    return ConcretePJRTNumber{T,D}(data, sharding)
end

function ConcretePJRTNumber{T}(data::T2; kwargs...) where {T<:Number,T2<:Number}
    carray = ConcretePJRTArray(fill(convert(T, data)); kwargs...)
    if !Sharding.is_sharded(carray.sharding)
        return ConcretePJRTNumber{T,1}((carray.data[1],), carray.sharding)
    end
    @assert all(isnothing, carray.sharding.partition_spec) "ConcretePJRTNumber cannot be \
                                                            sharded"
    return ConcretePJRTNumber{T,length(carray.data)}(carray.data, carray.sharding)
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

# Value constructors for the kind-specific types, e.g. via
# `convert(ConcretePJRTFloat{Float64,1}, x)`. The `Rational`/`Complex` methods
# disambiguate against Base's constructors on `Integer`/`AbstractFloat`/`Real`.
for CT in (:ConcretePJRTInteger, :ConcretePJRTFloat, :ConcretePJRTComplex),
    XT in (:Number, :Rational, :Complex, :BigFloat)

    @eval function (::Type{CT})(x::$XT; kwargs...) where {CT<:$CT}
        return ConcretePJRTNumber{unwrapped_eltype(CT)}(x; kwargs...)
    end
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

# TODO(#2229): (Deprecated): remove in v0.3
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
for (ConcreteT, AbstractT) in (
    (:ConcreteIFRTInteger, :AbstractConcreteInteger),
    (:ConcreteIFRTFloat, :AbstractConcreteFloat),
    (:ConcreteIFRTComplex, :AbstractConcreteComplex),
)
    @eval begin
        mutable struct $ConcreteT{T} <: $AbstractT{T}
            data::XLA.IFRT.AsyncArray
            sharding::Sharding.ShardInfo
            donated::Bool

            function $ConcreteT{T}(
                data::XLA.IFRT.AsyncArray, sharding::Sharding.ShardInfo
            ) where {T}
                return new{T}(data, sharding, false)
            end
        end

        @leaf $ConcreteT
    end
end

const ConcreteIFRTReal{T} = Union{ConcreteIFRTInteger{T},ConcreteIFRTFloat{T}}
const ConcreteIFRTNumber{T} = Union{ConcreteIFRTReal{T},ConcreteIFRTComplex{T}}

"""
    ifrt_number_type(::Type{T}) where {T<:ReactantPrimitive}

The concrete IFRT number type for element type `T`, i.e. the member of
`ConcreteIFRTNumber{T}` that matches the numeric kind of `T`.
"""
@inline ifrt_number_type(::Type{T}) where {T<:Integer} = ConcreteIFRTInteger{T}
@inline ifrt_number_type(::Type{T}) where {T<:AbstractFloat} = ConcreteIFRTFloat{T}
@inline ifrt_number_type(::Type{T}) where {T<:Complex} = ConcreteIFRTComplex{T}

function ConcreteIFRTNumber{T}(
    data::XLA.IFRT.AsyncArray, sharding::Sharding.ShardInfo
) where {T}
    return ifrt_number_type(T)(data, sharding)
end

function ConcreteIFRTNumber{T}(data::XLA.IFRT.AsyncArray) where {T}
    return ConcreteIFRTNumber{T}(data, Sharding.NoShardInfo())
end

function ConcreteIFRTNumber{T}(data::T2; kwargs...) where {T<:Number,T2<:Number}
    carray = ConcreteIFRTArray(fill(convert(T, data)); kwargs...)
    return ConcreteIFRTNumber{T}(carray.data, carray.sharding)
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

for CT in (:ConcreteIFRTInteger, :ConcreteIFRTFloat, :ConcreteIFRTComplex),
    XT in (:Number, :Rational, :Complex, :BigFloat)

    @eval function (::Type{CT})(x::$XT; kwargs...) where {CT<:$CT}
        return ConcreteIFRTNumber{unwrapped_eltype(CT)}(x; kwargs...)
    end
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
# TODO(#2257): Implement Padding for this version. A bit more finicky that the above case
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

@enumx InterpolationType begin
    Nearest
    Linear
end

function InterpolateArray(
    local_cpu_array::AbstractArray{T,N},
    final_grid_size::Dims{N},
    sharding::Sharding.AbstractSharding,
    interpolation::InterpolationType.T,
    halo::Dims{N}=ntuple(_ -> 0, N);
    client=nothing,
) where {T,N}
    @assert Sharding.is_sharded(sharding)
    client = client === nothing ? XLA.default_backend() : client
    @assert client isa XLA.IFRT.Client "InterpolateArray currently only supports IFRT client"
    (; hlo_sharding) = Sharding.HloSharding(sharding, final_grid_size)
    all_devices = XLA.get_device.((client,), sharding.mesh.device_ids)

    addressable_device_indices = [
        i - 1 for (i, device) in enumerate(all_devices) if XLA.is_addressable(device)
    ]

    addressable_slices, _ = XLA.sharding_to_concrete_array_indices(
        hlo_sharding, final_grid_size, addressable_device_indices
    )
    src_size = size(local_cpu_array)
    ordered_buffers = Vector{Array{T,N}}(undef, length(addressable_slices))
    for (buf_idx, slice) in enumerate(addressable_slices)
        shard_shape = length.(slice)
        if interpolation == InterpolationType.Nearest
            src_idx_ranges = ntuple(N) do dim
                I_range = slice[dim]
                N_dim, M_dim = final_grid_size[dim], src_size[dim]
                H = halo[dim]

                return [
                    begin
                        if I <= H
                            clamp(I, 1, M_dim)
                        elseif I >= N_dim - H + 1
                            clamp(M_dim - N_dim + I, 1, M_dim)
                        else
                            I_shifted = I - H
                            N_dim_shifted = N_dim - 2 * H
                            M_dim_shifted = M_dim - 2 * H

                            if N_dim_shifted <= 0 || M_dim_shifted <= 0
                                clamp(I, 1, M_dim)
                            else
                                idx_shifted =
                                    (I_shifted * M_dim_shifted + N_dim_shifted - 1) ÷
                                    N_dim_shifted
                                clamp(idx_shifted + H, 1, M_dim)
                            end
                        end
                    end for I in I_range
                ]
            end
            buf = Array{T,N}(undef, shard_shape)
            for I in CartesianIndices(shard_shape)
                idx = ntuple(dim -> src_idx_ranges[dim][I.I[dim]], N)
                buf[I] = local_cpu_array[CartesianIndex(idx)]
            end
        elseif interpolation == InterpolationType.Linear
            # We use cell-centered interpolation to map target grid indices to source grid indices.
            # A point at target index `I` maps to `(I - 0.5) * (M_dim / N_dim) + 0.5` in the source grid.
            # This ensures that we match Oceananigans' interpolation behavior.
            # We compute this mapping using pure integers to avoid floating-point inaccuracies.
            # If halo > 0, the mapping is applied to the region between halos.
            lows = ntuple(N) do dim
                I_range = slice[dim]
                N_dim, M_dim = final_grid_size[dim], src_size[dim]
                H = halo[dim]

                return [
                    begin
                        if I <= H
                            clamp(I, 1, M_dim)
                        elseif I >= N_dim - H + 1
                            clamp(M_dim - N_dim + I, 1, M_dim)
                        else
                            I_shifted = I - H
                            N_dim_shifted = N_dim - 2 * H
                            M_dim_shifted = M_dim - 2 * H

                            a = (2 * I_shifted - 1) * M_dim_shifted + N_dim_shifted
                            b = 2 * N_dim_shifted
                            low_shifted = a ÷ b
                            clamp(low_shifted + H, 1, M_dim)
                        end
                    end for I in I_range
                ]
            end

            highs = ntuple(N) do dim
                I_range = slice[dim]
                N_dim, M_dim = final_grid_size[dim], src_size[dim]
                H = halo[dim]

                return [
                    begin
                        if I <= H
                            clamp(I, 1, M_dim)
                        elseif I >= N_dim - H + 1
                            clamp(M_dim - N_dim + I, 1, M_dim)
                        else
                            I_shifted = I - H
                            N_dim_shifted = N_dim - 2 * H
                            M_dim_shifted = M_dim - 2 * H

                            a = (2 * I_shifted - 1) * M_dim_shifted + N_dim_shifted
                            b = 2 * N_dim_shifted
                            low_shifted = a ÷ b
                            clamp(low_shifted + 1 + H, 1, M_dim)
                        end
                    end for I in I_range
                ]
            end

            dens = ntuple(N) do dim
                H = halo[dim]
                return 2 * max(1, final_grid_size[dim] - 2 * H)
            end
            total_den = prod(dens)

            rems = ntuple(N) do dim
                I_range = slice[dim]
                N_dim, M_dim = final_grid_size[dim], src_size[dim]
                H = halo[dim]

                return [
                    begin
                        if I <= H || I >= N_dim - H + 1
                            0
                        else
                            I_shifted = I - H
                            N_dim_shifted = N_dim - 2 * H
                            M_dim_shifted = M_dim - 2 * H

                            a = (2 * I_shifted - 1) * M_dim_shifted + N_dim_shifted
                            b = 2 * N_dim_shifted
                            a % b
                        end
                    end for I in I_range
                ]
            end
            buf = Array{T,N}(undef, shard_shape)
            corner_space = CartesianIndices(ntuple(_ -> 2, N))
            for I in CartesianIndices(shard_shape)
                sum_val = zero(T)
                for c in corner_space
                    idx = ntuple(
                        dim -> c[dim] == 1 ? lows[dim][I.I[dim]] : highs[dim][I.I[dim]], N
                    )

                    w_int = prod(ntuple(dim -> if c[dim] == 1
                        (dens[dim] - rems[dim][I.I[dim]])
                    else
                        rems[dim][I.I[dim]]
                    end, N))

                    sum_val += w_int * local_cpu_array[CartesianIndex(idx)]
                end
                buf[I] = sum_val / total_den
            end
        else
            error("Unsupported interpolation type")
        end
        ordered_buffers[buf_idx] = buf
    end
    return ConcreteIFRTArray(ordered_buffers, final_grid_size; client, sharding)
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
