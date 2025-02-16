module Sharding

using ..Reactant: Reactant, XLA

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

Base.vec(mesh::Mesh) = vec(device_ids(mesh))

function device_ids(mesh::Mesh)
    # XXX: Do we need to permute the device ids?
    return permutedims(
        reshape(collect(Int64, mesh.device_ids), size(mesh)...), reverse(1:ndims(mesh))
    )
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

# TODO: At the core we should have an HloSharding Type that doesn't need to store the
#       partition spec and other details

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

function (sharding::NamedSharding)(
    client::XLA.Client, ::Nothing, x::Union{AbstractArray,Number}
)
    (; mesh, partition_spec) = sharding
    @assert length(partition_spec) == ndims(x)

    device_to_array_slices, _ = XLA.compute_array_indices_and_partition_spec(
        XLA.CondensedOpSharding(ShardingWithShape(sharding, size(x))), size(x), mesh
    )
    devices_list = vec(mesh)

    data = ntuple(length(mesh)) do i
        XLA.AsyncBuffer(
            XLA.ArrayFromHostBuffer(
                client,
                x[device_to_array_slices[i]...],
                XLA.ClientGetAddressableDevice(
                    client, XLA.device_ordinal(client, devices_list[i])
                ),
            ),
            nothing,
        )
    end

    return data, ShardInfo(sharding, device_to_array_slices)
end

struct ShardingWithShape{S,D} <: AbstractSharding
    sharding::S
    shape::D
end

# XXX: we need to make the performance of this function better
function XLA.CondensedOpSharding(sharding_and_shape::ShardingWithShape{<:NamedSharding})
    (; sharding, shape) = sharding_and_shape
    (; mesh, partition_spec) = sharding
    @assert length(partition_spec) == length(shape)

    partition_spec = reverse(partition_spec)
    shape = reverse(shape)

    array_mapping = __get_array_mapping(partition_spec)
    mesh_axis_position = Dict(name => i for (i, name) in enumerate(mesh.axis_names))

    replicated_mesh_axes = Tuple{Int64,Int64}[]
    for (i, axis_name) in enumerate(mesh.axis_names)
        if !haskey(array_mapping, axis_name)
            push!(replicated_mesh_axes, (i, size(mesh, axis_name)))
        end
    end

    tile_assignment = device_ids(mesh)

    # Fast Path for replicating the input across all devices
    if length(replicated_mesh_axes) == ndims(mesh)
        return XLA.CondensedOpSharding{ndims(tile_assignment)}(
            XLA.OpShardingType.Replicated, false, tile_assignment
        )
    end

    # Calculate new mesh shape and permutation
    mesh_permutation = Int[]
    new_mesh_shape = ones(Int, length(shape))

    # Sort array mapping by position to ensure consistent order
    for (name, pos) in sort(collect(array_mapping); by=x -> x[2])
        new_mesh_shape[pos] *= size(mesh, name)
        push!(mesh_permutation, mesh_axis_position[name])
    end

    # Handle replicated dimensions at the end
    replicate_on_last_tile_dim = false
    if !isempty(replicated_mesh_axes)
        replicated_size = prod(last(axis) for axis in replicated_mesh_axes)
        push!(new_mesh_shape, replicated_size)
        append!(mesh_permutation, first.(replicated_mesh_axes))

        tile_assignment = reshape(tile_assignment, new_mesh_shape...)
        push!(mesh_permutation, length(mesh_permutation) + 1)
        replicate_on_last_tile_dim = true
    end

    permuted = permutedims(tile_assignment, mesh_permutation)
    final_assignment = reshape(permuted, new_mesh_shape...)

    return XLA.CondensedOpSharding{ndims(final_assignment)}(
        XLA.OpShardingType.Other, replicate_on_last_tile_dim, final_assignment
    )
end

# Given Sharding + Array --> ShardInfo
struct ShardInfo{S,D} <: AbstractSharding
    sharding::S
    device_to_array_slices::D
end

function XLA.CondensedOpSharding(sharding_and_shape::ShardingWithShape{<:ShardInfo})
    return XLA.CondensedOpSharding(
        ShardingWithShape(sharding_and_shape.sharding.sharding, sharding_and_shape.shape)
    )
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

function __get_array_mapping(partition_spec)
    mapping = Dict{Symbol,Int64}()
    for (i, axis) in enumerate(partition_spec)
        axis === nothing && continue
        axis isa Symbol && (axis = (axis,))
        for axis_name in axis
            mapping[axis_name] = i
        end
    end
    return mapping
end

end
