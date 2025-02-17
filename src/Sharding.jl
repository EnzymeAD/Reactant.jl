module Sharding

using ..Reactant: Reactant, XLA, MLIR

struct Mesh{D,ND}
    device_ids::NTuple{ND,Int}
    shape::Dims{D}
    axis_names::NTuple{D,Symbol}

    function Mesh(devices::AbstractArray{XLA.AbstractDevice}, axis_names)
        return Mesh(XLA.get_local_device_id.(devices), axis_names)
    end

    function Mesh(
        devices::NTuple{D,XLA.AbstractDevice}, shape::Dims{D}, axis_names
    ) where {D}
        return Mesh(XLA.get_local_device_id.(devices), shape, axis_names)
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

device_ids(mesh::Mesh) = reshape(collect(Int64, mesh.device_ids), size(mesh)...)

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

function (T::AbstractSharding)(::XLA.AbstractClient, device, ::Union{AbstractArray,Number})
    return error("(::$(T))(::XLA.AbstractClient, ::AbstractArray) is not implemented")
end

struct NoSharding <: AbstractSharding end

# This allows us to mark entire branches as NoSharding
Base.getproperty(::NoSharding, x) = NoSharding()
Base.getproperty(::NoSharding, x::Symbol) = NoSharding()

function (::NoSharding)(client::XLA.PJRT.Client, device, x::Union{AbstractArray,Number})
    buffer = XLA.PJRT.AsyncBuffer(client, x, device)
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
    client::XLA.PJRT.Client, ::Nothing, x::Union{AbstractArray,Number}
)
    (; mesh, partition_spec) = sharding
    @assert length(partition_spec) == ndims(x)

    device_to_array_slices, _ = XLA.compute_array_indices_and_partition_spec(
        convert(XLA.CondensedOpSharding, ShardingWithShape(sharding, size(x))),
        size(x),
        mesh,
    )
    devices_list = vec(mesh)

    data = ntuple(length(mesh)) do i
        XLA.PJRT.AsyncBuffer(
            client,
            x[device_to_array_slices[i]...],
            XLA.get_addressable_device(client, XLA.device_ordinal(client, devices_list[i])),
        )
    end

    return data, ShardInfo(sharding, device_to_array_slices)
end

function get_shardy_tensor_sharding_attribute(
    ctx, N::Int, sharding::NamedSharding, mesh_name; do_transpose=true
)
    dimension_sharding_attrs = Vector{MLIR.API.MlirAttribute}(undef, N)
    for (j, name) in enumerate(sharding.partition_spec)
        if name === nothing
            axes = MLIR.IR.Attribute[]
        else
            @assert name isa Symbol
            axes = [
                MLIR.API.sdyAxisRefAttrGet(
                    ctx, String(name), MLIR.API.MlirAttribute(C_NULL)
                ),
            ]
        end
        dimension_sharding_attrs[j] = MLIR.API.sdyDimensionShardingAttrGet(
            ctx, length(axes), axes, sharding.is_closed[j], sharding.priority[j]
        )
    end

    return MLIR.IR.Attribute(
        MLIR.API.sdyTensorShardingAttrGet(
            ctx,
            mesh_name,
            length(dimension_sharding_attrs),
            do_transpose ? reverse(dimension_sharding_attrs) : dimension_sharding_attrs,
            0,
            MLIR.API.MlirAttribute[],
        ),
    )
end

# An internal abstraction to allow defining `convert` to XLA sharding
struct ShardingWithShape{S,D} <: AbstractSharding
    sharding::S
    shape::D
end

internal_simple_op(x) = Reactant.Ops.negate(x)

# XXX: We do a fake compile here to get the mhlo sharding. Ideally we should be able to use
#      some API to convert shardy annotations to mhlo annotations.
# XXX: We should cache the CondensedOpSharding else we will end up calling this function
#      multiple times.
function Base.convert(
    ::Type{XLA.CondensedOpSharding}, sharding_and_shape::ShardingWithShape{<:NamedSharding}
)
    return convert(XLA.CondensedOpSharding, convert(XLA.OpSharding, sharding_and_shape))
end

function Base.convert(
    ::Type{XLA.OpSharding}, sharding_and_shape::ShardingWithShape{<:NamedSharding}
)
    tmp = Reactant.ConcreteRArray(
        ones(sharding_and_shape.shape); sharding=LazySharding(sharding_and_shape.sharding)
    )
    _, exec, _, _, _ = Reactant.Compiler.compile_xla(internal_simple_op, (tmp,))
    return only(XLA.get_parameter_shardings(exec))
end

# Lazy Sharding. ConcreteArrays with this annotation is not really sharded but we can use it
# to compile the executable.
struct LazySharding{S} <: AbstractSharding
    sharding::S
end

function get_shardy_tensor_sharding_attribute(
    ctx, N::Int, sharding::LazySharding, mesh_name; do_transpose=true
)
    return get_shardy_tensor_sharding_attribute(
        ctx, N, sharding.sharding, mesh_name; do_transpose
    )
end

function (sharding::LazySharding)(
    client::XLA.PJRT.Client, ::Nothing, x::Union{AbstractArray,Number}
)
    data = XLA.PJRT.AsyncBuffer(
        client,
        x,
        XLA.get_addressable_device(
            client, XLA.device_ordinal(client, vec(sharding.sharding.mesh)[1])
        ),
    )

    return (data,), ShardInfo(sharding, (ntuple(i -> 1:size(x, i), ndims(x)),))
end

function Base.getproperty(sharding::LazySharding, name::Symbol)
    name ∈ (:sharding, :device_to_array_slices) && return getfield(sharding, name)
    return getproperty(sharding.sharding, name)
end

# Given Sharding + Array --> ShardInfo
struct ShardInfo{S,D} <: AbstractSharding
    sharding::S
    device_to_array_slices::D
end

function Base.convert(
    ::Type{XLA.CondensedOpSharding}, sharding_and_shape::ShardingWithShape{<:ShardInfo}
)
    return convert(
        XLA.CondensedOpSharding,
        ShardingWithShape(sharding_and_shape.sharding.sharding, sharding_and_shape.shape),
    )
end

function Base.getproperty(sharding::ShardInfo, name::Symbol)
    name ∈ (:sharding, :device_to_array_slices) && return getfield(sharding, name)
    return getproperty(sharding.sharding, name)
end

function get_shardy_tensor_sharding_attribute(
    ctx, sharding::ShardInfo, mesh_name; do_transpose=true
)
    return get_shardy_tensor_sharding_attribute(
        ctx,
        length(first(sharding.device_to_array_slices)),
        sharding.sharding,
        mesh_name;
        do_transpose,
    )
end

function (sharding::ShardInfo)(
    client::XLA.PJRT.Client, device, x::Union{AbstractArray,Number}
)
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
is_sharded(s::LazySharding) = is_sharded(s.sharding)
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
