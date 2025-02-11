module Sharding

using ..Reactant: Reactant, XLA

# NOTE: PjRt doesn't provide a native sharding mechanism, so this file implements sharding
#       at the julia level. With our migration to IFRt, we should be able to rewrite this
#       logic to directly use the sharded arrays from IFRt. This would also simplify our
#       logic of storing multiple arrays in ConcreteRArray struct

struct Mesh{D,ND}
    device_ids::NTuple{ND,Int}
    shape::Dims{D}
    axis_names::NTuple{D,Symbol}

    function Mesh(devices::AbstractArray{XLA.Device}, axis_names)
        return Mesh(XLA.DeviceGetLocalDeviceId.(devices), axis_names)
    end

    function Mesh(devices::NTuple{D,XLA.Device}, shape::Dims{D}, axis_names) where {D}
        return Mesh(XLA.DeviceGetLocalDeviceId.(devices), shape, axis_names)
    end

    function Mesh(
        device_ids::AbstractArray{<:Integer,D}, axis_names::NTuple{D,Union{String,Symbol}}
    ) where {D}
        return Mesh(Tuple(vec(device_ids)), size(device_ids), axis_names)
    end

    function Mesh(
        device_ids::NTuple{D1,Int},
        shape::Dims{D},
        axis_names::NTuple{D,Union{String,Symbol}},
    ) where {D,D1}
        @assert allunique(device_ids)
        return new{D,D1}(device_ids, shape, Symbol.(axis_names))
    end
end

Base.length(::Mesh{D,ND}) where {D,ND} = ND
Base.ndims(::Mesh{D}) where {D} = D

Base.size(mesh::Mesh) = mesh.shape
Base.size(mesh::Mesh, axis::Int) = mesh.shape[axis]
function Base.size(mesh::Mesh, axis::Union{String,Symbol})
    return size(mesh, findfirst(==(Symbol(axis)), mesh.axis_names))
end
Base.size(mesh::Mesh, ::Nothing) = 1

Base.in(axis::Union{String,Symbol}, mesh::Mesh) = Symbol(axis) ∈ mesh.axis_names

abstract type AbstractSharding end

function (T::AbstractSharding)(::XLA.Client, device, ::Union{AbstractArray,Number})
    return error("(::$(T))(::XLA.Client, ::AbstractArray) is not implemented")
end

struct NoSharding <: AbstractSharding end

# This allows us to mark entire branches as NoSharding
Base.getproperty(::NoSharding, x) = NoSharding()

function (::NoSharding)(client::XLA.Client, device, x::Union{AbstractArray,Number})
    buffer = XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, x, device), nothing)
    return (buffer,), ShardInfo(NoSharding(), nothing)
end

# XXX: multiple axes partitioning -- supported by shardy (not in Jax I think)
struct NamedSharding{D1,D2,P<:Tuple,D3} <: AbstractSharding
    mesh::Mesh{D1,D2}
    partition_spec::P
    is_closed::NTuple{D3,Bool}
    priority::NTuple{D3,Int}

    function NamedSharding(
        mesh::Mesh{D1,D2},
        partition_spec::P;
        is_closed::NTuple{D3,Bool}=ntuple(Returns(true), length(partition_spec)),
        # negative priority means that priority is not considered by shardy
        priority::NTuple{D3,Int}=ntuple(i -> -1, length(partition_spec)),
    ) where {D1,D2,P<:Tuple,D3}
        # TODO: we need to check how open sharding works in XLA, i.e. how to specify inputs
        @assert all(is_closed) "All partitions must be closed for now."
        @assert all(p -> p === nothing || p isa String || p isa Symbol, partition_spec)
        partition_spec = map(x -> x isa String ? Symbol(x) : x, partition_spec)
        non_replicated_axes = filter(x -> x !== nothing, partition_spec)
        @assert length(unique(non_replicated_axes)) == length(non_replicated_axes) "Duplicate axis names!"
        return new{D1,D2,typeof(partition_spec),D3}(
            mesh, partition_spec, is_closed, priority
        )
    end
end

function (sharding::NamedSharding)(client::XLA.Client, device, x::Number)
    (; mesh, partition_spec) = sharding
    @assert length(partition_spec) == 0

    data = map(mesh.device_ids) do device_id
        return XLA.AsyncBuffer(
            XLA.ArrayFromHostBuffer(client, fill(x), XLA.device_ordinal(client, device_id)),
            nothing,
        )
    end
    return data, ShardInfo(sharding, ntuple(Returns(()), length(mesh)))
end

function (sharding::NamedSharding)(client::XLA.Client, ::Nothing, x::AbstractArray)
    (; mesh, partition_spec) = sharding
    @assert length(partition_spec) == ndims(x)

    # Fast Path for replicating the input across all devices
    if all(Base.Fix2(===, nothing), partition_spec)
        data = map(mesh.device_ids) do device_id
            return XLA.AsyncBuffer(
                XLA.ArrayFromHostBuffer(
                    client,
                    x,
                    XLA.ClientGetAddressableDevice(
                        client, XLA.device_ordinal(client, device_id)
                    ),
                ),
                nothing,
            )
        end
        device_to_array_slices = ntuple(
            Returns(ntuple(i -> 1:size(x, i), ndims(x))), length(mesh)
        )
        return data, ShardInfo(sharding, device_to_array_slices)
    end

    ndevices = map(Base.Fix1(size, mesh), partition_spec)
    for (sz, ndevice) in zip(size(x), ndevices)
        @assert sz % ndevice == 0 "$(size(x)) must be divisible by $(ndevices)"
    end
    strides = size(x) .÷ ndevices

    slices = Array{NTuple{ndims(x),UnitRange{Int64}},ndims(x)}(undef, ndevices)
    for idx in CartesianIndices(slices)
        idx_tup = Tuple(idx)
        slices[idx] = Tuple(
            (i1 + 1):i2 for (i1, i2) in zip((idx_tup .- 1) .* strides, idx_tup .* strides)
        )
    end

    device_to_array_slices = Array{eltype(slices),ndims(mesh)}(undef, size(mesh))
    for idx in CartesianIndices(device_to_array_slices)
        idx_tup = Tuple(idx)
        slice_idx = ones(Int, ndims(slices))
        for (axis_name, idxᵢ) in zip(mesh.axis_names, idx_tup)
            dim = findfirst(==(axis_name), sharding.partition_spec)
            dim !== nothing && (slice_idx[dim] = idxᵢ)
        end
        device_to_array_slices[idx] = slices[CartesianIndex(slice_idx...)]
    end

    data = ntuple(length(mesh)) do i
        XLA.AsyncBuffer(
            XLA.ArrayFromHostBuffer(
                client,
                x[device_to_array_slices[i]...],
                XLA.ClientGetAddressableDevice(
                    client, XLA.device_ordinal(client, mesh.device_ids[i])
                ),
            ),
            nothing,
        )
    end

    return data, ShardInfo(sharding, Tuple(vec(device_to_array_slices)))
end

# Given Sharding + Array --> ShardInfo
struct ShardInfo{S,D} <: AbstractSharding
    sharding::S
    device_to_array_slices::D
end

function Base.getproperty(sharding::ShardInfo, name::Symbol)
    name ∈ (:sharding, :device_to_array_slices) && return getfield(sharding, name)
    return getfield(sharding.sharding, name)
end

function (sharding::ShardInfo)(client::XLA.Client, device, x::Union{AbstractArray,Number})
    return (sharding.sharding)(client, device, x)
end

const NoShardInfo = ShardInfo{NoSharding,Nothing}

ShardInfo{NoSharding,Nothing}() = ShardInfo(NoSharding(), nothing)

"""
    is_sharded(sharding)
    is_sharded(x::AbstractArray)

Checks whether the given sharding refers to no sharding.
"""
is_sharded(::NoSharding) = false
is_sharded(::NamedSharding) = true
is_sharded(s::ShardInfo) = is_sharded(s.sharding)

function is_sharded(x::AbstractArray)
    ancestor_x = Reactant.ancestor(x)
    hasfield(typeof(ancestor_x), :sharding) && return is_sharded(ancestor_x.sharding)
    return false
end
function is_sharded(x::Number)
    hasfield(typeof(x), :sharding) && return is_sharded(x.sharding)
    return false
end

end
