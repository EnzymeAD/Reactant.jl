module Sharding

using ..Reactant: Reactant, XLA, MLIR

"""
    Mesh(devices::AbstractArray{XLA.AbstractDevice}, axis_names)

Construct a `Mesh` from an array of devices and a tuple of axis names. The size of the i-th
axis is given by `size(devices, i)`. All the axis names must be unique, and cannot be
nothing.

## Examples

Assuming that we have a total of 8 devices, we can construct a mesh with the following:

```julia-repl
julia> devices = Reactant.devices();

julia> mesh = Mesh(reshape(devices, 2, 2, 2), (:x, :y, :z));

julia> mesh = Mesh(reshape(devices, 4, 2), (:x, :y));
```
"""
struct Mesh{D,ND}
    device_ids::NTuple{ND,Int}
    sorted_device_ids::NTuple{ND,Int}
    logical_device_ids::NTuple{ND,Int}
    shape::Dims{D}
    axis_names::NTuple{D,Symbol}

    function Mesh(devices::AbstractArray{<:XLA.AbstractDevice}, axis_names)
        return Mesh(XLA.device_ordinal.(devices), axis_names)
    end

    function Mesh(
        devices::NTuple{D,<:XLA.AbstractDevice}, shape::Dims{D}, axis_names
    ) where {D}
        return Mesh(XLA.device_ordinal.(devices), shape, axis_names)
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
        return new{D,D1}(
            device_ids,
            Tuple(sort([device_ids...])),
            ntuple(Base.Fix2(-, 1), D1),
            shape,
            Symbol.(axis_names),
        )
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

function (T::AbstractSharding)(::XLA.AbstractClient, device, ::Union{AbstractArray,Number})
    return error(
        "(::$(T))(::XLA.AbstractClient, device, ::Union{AbstractArray,Number}) is \
         not implemented"
    )
end

function get_shardy_tensor_sharding_attribute end

"""
    NoSharding()

Sharding annotation that indicates that the array is not sharded.

See also: [`Sharding.NamedSharding`](@ref)
"""
struct NoSharding <: AbstractSharding end

# This allows us to mark entire branches as NoSharding
Base.getproperty(::NoSharding, x) = NoSharding()
Base.getproperty(::NoSharding, x::Symbol) = NoSharding()

function (::NoSharding)(client::XLA.PJRT.Client, device, x::Union{AbstractArray,Number})
    buffer = XLA.PJRT.AsyncBuffer(client, x, device)
    return (buffer,), ShardInfo(NoSharding(), nothing)
end

"""
    NamedSharding(
        mesh::Mesh, partition_spec::Tuple;
        is_closed::NTuple{N,Bool}=ntuple(Returns(true), length(partition_spec)),
        priority::NTuple{N,Int}=ntuple(i -> -1, length(partition_spec)),
    )

Sharding annotation that indicates that the array is sharded along the given `partition_spec`. For details on the sharding representation see the
[Shardy documentation](https://openxla.org/shardy/sharding_representation).

## Arguments

  - `mesh`: [`Sharding.Mesh`](@ref) that describes the mesh of the devices.
  - `partition_spec`: Must be equal to the ndims of the array being sharded. Each element
    can be:
      1. `nothing`: indicating the corresponding dimension is replicated along the axis.
      2. A tuple of axis names indicating the axis names that the corresponding dimension
         is sharded along.
      3. A single axis name indicating the axis name that the corresponding dimension is
         sharded along.

## Keyword Arguments

  - `is_closed`: A tuple of booleans indicating whether the corresponding dimension is
    closed along the axis. Defaults to `true` for all dimensions.
  - `priority`: A tuple of integers indicating the priority of the corresponding dimension.
    Defaults to `-1` for all dimensions. A negative priority means that the priority is
    not considered by shardy.

## Examples

```julia-repl
julia> devices = Reactant.devices();

julia> mesh = Mesh(reshape(devices, 2, 2, 2), (:x, :y, :z));

julia> sharding = NamedSharding(mesh, (:x, :y, nothing)); # 3D Array sharded along x and y on dim 1 and 2 respectively, while dim 3 is replicated

julia> sharding = NamedSharding(mesh, ((:x, :y), nothing, nothing)); # 3D Array sharded along x and y on dim 1, 2 and 3 are replicated

julia> sharding = NamedSharding(mesh, (nothing, nothing)); # fully replicated Matrix
```

See also: [`Sharding.NoSharding`](@ref)
"""
struct NamedSharding{D1,D2,P<:Tuple,D3} <: AbstractSharding
    mesh::Mesh{D1,D2}
    partition_spec::P
    is_closed::NTuple{D3,Bool}
    priority::NTuple{D3,Int}

    function NamedSharding(
        mesh::Mesh{D1,D2},
        partition_spec::P;
        is_closed::NTuple{D3,Bool}=ntuple(Returns(true), length(partition_spec)),
        priority::NTuple{D3,Int}=ntuple(i -> -1, length(partition_spec)),
    ) where {D1,D2,P<:Tuple,D3}
        axis_names = Symbol[]
        pspec = ()
        for p in partition_spec
            if p === nothing
                pspec = (pspec..., nothing)
            elseif p isa Tuple
                @assert all(x -> x isa Symbol || x isa String, p)
                sym_names = Symbol.(p)
                append!(axis_names, sym_names)
                pspec = (pspec..., sym_names)
            elseif p isa Symbol || p isa String
                push!(axis_names, Symbol(p))
                pspec = (pspec..., Symbol(p))
            else
                error("Unexpected partition spec $(partition_spec) [$(p)]")
            end
        end
        @assert allunique(axis_names) "Duplicate axis names!"

        return new{D1,D2,typeof(pspec),D3}(mesh, pspec, is_closed, priority)
    end
end

function (sharding::NamedSharding)(
    client::XLA.PJRT.Client, device::Nothing, x::Union{AbstractArray,Number}
)
    @assert length(sharding.partition_spec) == ndims(x)
    return (convert(HloSharding, sharding))(client, device, x)
end

function get_shardy_tensor_sharding_attribute(
    sharding::NamedSharding, ctx, mesh_name, mesh_attr; do_transpose=true
)
    dimension_sharding_attrs = Vector{MLIR.API.MlirAttribute}(
        undef, length(sharding.partition_spec)
    )
    for (j, name) in enumerate(sharding.partition_spec)
        if name === nothing
            axes = MLIR.IR.Attribute[]
        else
            names = name isa Symbol ? (name,) : name
            axes = [
                MLIR.API.sdyAxisRefAttrGet(
                    ctx, String(name), MLIR.API.MlirAttribute(C_NULL)
                ) for name in names
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

# HloSharding
# This stores the sharding information in the form of XLA.HloSharding, and provides a
# central type for the final storage. It also potentially saves us the pain of not having
# to regenerate the partition spec from the HloSharding.
struct HloSharding{M,D,D2} <: AbstractSharding
    hlo_sharding::XLA.HloSharding
    mesh::Mesh{M,D}
    is_closed::NTuple{D2,Bool}
    priority::NTuple{D2,Int}

    function HloSharding(
        hlo_sharding::XLA.HloSharding, mesh::Mesh{M,D}, is_closed, priority
    ) where {M,D}
        @assert length(is_closed) == length(priority)
        return new{M,D,length(is_closed)}(hlo_sharding, mesh, is_closed, priority)
    end
end

function Base.convert(::Type{HloSharding}, sharding::NamedSharding)
    if MLIR.IR._has_context()
        ctx = MLIR.IR.context()
    else
        ctx = MLIR.IR.Context(Reactant.registry[], false)
        @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
    end

    MLIR.IR.context!(ctx) do
        mesh_op = Reactant.Ops.mesh(
            sharding.mesh; mod=MLIR.IR.Module(MLIR.IR.Location(; context=ctx))
        )

        tensor_sharding_attr = get_shardy_tensor_sharding_attribute(
            sharding, ctx, mesh_op.sym_name, mesh_op.mesh_attr; do_transpose=true
        )

        return HloSharding(
            XLA.HloSharding(
                @ccall MLIR.API.mlir_c.hloShardingFromTensorShardingAttr(
                    tensor_sharding_attr.attribute::MLIR.API.MlirAttribute,
                    mesh_op.mesh_attr.attribute::MLIR.API.MlirAttribute,
                )::Ptr{Cvoid}
            ),
            sharding.mesh,
            sharding.is_closed,
            sharding.priority,
        )
    end
end

function (sharding::HloSharding)(
    client::XLA.PJRT.Client, ::Nothing, x::Union{AbstractArray,Number}
)
    condensed_op_sharding = convert(XLA.CondensedOpSharding, sharding.hlo_sharding)

    device_to_array_slices = XLA.sharding_to_concrete_array_indices(
        condensed_op_sharding, size(x), sharding.mesh.logical_device_ids
    )

    data = ntuple(length(sharding.mesh)) do i
        XLA.PJRT.AsyncBuffer(
            client,
            x[device_to_array_slices[i]...],
            XLA.get_device(client, sharding.mesh.device_ids[i]),
        )
    end

    return data, ShardInfo(sharding, device_to_array_slices)
end

function get_shardy_tensor_sharding_attribute(
    sharding::HloSharding, ctx, mesh_name, mesh_attr; kwargs...
)
    string_mesh_name = MLIR.IR.Attribute(MLIR.IR.flatsymbol(mesh_name); context=ctx)
    GC.@preserve sharding begin
        return MLIR.IR.Attribute(
            @ccall MLIR.API.mlir_c.hloShardingToTensorShardingAttr(
                ctx::MLIR.API.MlirContext,
                sharding.hlo_sharding.ptr::Ptr{Cvoid},
                string_mesh_name.attribute::MLIR.API.MlirAttribute,
                mesh_attr.attribute::MLIR.API.MlirAttribute,
                Int64(length(sharding.is_closed))::Int64,
                Bool[sharding.is_closed...]::Ptr{Bool},
                Int64[sharding.priority...]::Ptr{Int64},
            )::MLIR.API.MlirAttribute
        )
    end
end

# Given Sharding + Array --> ShardInfo
# This is the structure that is stored in the `sharding` field of `ConcreteRArray`
struct ShardInfo{S,D} <: AbstractSharding
    sharding::S
    device_to_array_slices::D
end

function Base.getproperty(sharding::ShardInfo, name::Symbol)
    name ∈ (:sharding, :device_to_array_slices) && return getfield(sharding, name)
    return getproperty(sharding.sharding, name)
end

function get_shardy_tensor_sharding_attribute(sharding::ShardInfo, args...; kwargs...)
    return get_shardy_tensor_sharding_attribute(sharding.sharding, args...; kwargs...)
end

function (sharding::ShardInfo)(
    client::XLA.AbstractClient, device, x::Union{AbstractArray,Number}
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
is_sharded(::HloSharding) = true
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
