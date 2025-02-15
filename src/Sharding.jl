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
Base.getproperty(::NoSharding, x::Symbol) = NoSharding()

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

function named_sharding_to_opsharding(sharding::NamedSharding, shape::Dims)
    (; mesh, partition_spec) = sharding
    @assert length(partition_spec) == length(shape)

    # Fast Path for replicating the input across all devices
    if all(Base.Fix2(===, nothing), partition_spec)
        return XLA.OpSharding(
            XLA.OpShardingType.Replicated,
            Int64[],
            Int64[],
            false,
            XLA.OpShardingType.T[],
            Int64[],
            Int64[],
            Int64[],
            Int32[],
            false,
            -1,
            XLA.ShardGroupType.As,
        )
    end

    tile_dims = map(Base.Fix1(size, mesh), partition_spec)
    num_tiles_before_replication = prod(tile_dims)
    total_devices = length(mesh.device_ids)
    replication_factor = cld(total_devices, num_tiles_before_replication)
    replicate_on_last_tile_dim = replication_factor > 1
    replicate_on_last_tile_dim && (tile_dims = (replication_factor, tile_dims...))

    # Create tile assignment array
    tile_assignment = Array{Int}(undef, tile_dims...)
    devices = reshape(collect(mesh.device_ids), size(mesh))

    # Find axes not used in partition_spec for replication
    unused_axes = filter(axis -> axis ∉ partition_spec, mesh.axis_names)
    unused_dims = map(axis -> size(mesh, axis), unused_axes)
    replication_indices = CartesianIndices(Tuple(unused_dims))

    # Fill tile assignment array
    for indices in CartesianIndices(tile_assignment)
        index_tuple = Tuple(indices)
        actual_indices = replicate_on_last_tile_dim ? index_tuple[2:end] : index_tuple
        repl_idx = replicate_on_last_tile_dim ? index_tuple[1] : 1

        # Initialize device index array
        device_index = ones(Int, ndims(mesh))

        # Map partition dimensions to device indices
        for (tile_idx, (pspec, dim_size)) in enumerate(zip(partition_spec, shape))
            if pspec !== nothing
                mesh_axis = findfirst(==(Symbol(pspec)), mesh.axis_names)
                if mesh_axis !== nothing
                    device_index[mesh_axis] = actual_indices[tile_idx]
                end
            end
        end

        # Handle replication for unused axes
        for (i, axis) in enumerate(unused_axes)
            axis_idx = findfirst(==(axis), mesh.axis_names)
            if axis_idx !== nothing
                device_index[axis_idx] = replication_indices[repl_idx][i]
            end
        end

        # Assign device to tile
        tile_assignment[indices] = devices[device_index...]
    end

    return XLA.OpSharding(
        XLA.OpShardingType.Other,
        Int64[],
        Int64[],
        replicate_on_last_tile_dim,
        XLA.OpShardingType.T[],
        collect(Int64, size(tile_assignment)),
        vec(tile_assignment),
        Int64[],
        Int32[],
        false,
        -1,
        XLA.ShardGroupType.As,
    )
end

function (sharding::NamedSharding)(
    client::XLA.Client, ::Nothing, x::Union{AbstractArray,Number}
)
    (; mesh, partition_spec) = sharding
    @assert length(partition_spec) == ndims(x)

    opsharding = named_sharding_to_opsharding(sharding, size(x))
    device_to_array_slices, _ = XLA.compute_array_indices_and_partition_spec(
        opsharding, size(x), mesh
    )

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

    return data, ShardInfo(sharding, device_to_array_slices)
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
