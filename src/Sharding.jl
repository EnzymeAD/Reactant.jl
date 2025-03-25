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
struct Mesh{D}
    device_ids::Array{Int64,D}
    logical_device_ids::UnitRange{Int}
    axis_names::NTuple{D,Symbol}

    function Mesh(devices::AbstractArray{<:XLA.AbstractDevice}, axis_names)
        return Mesh(XLA.device_ordinal.(devices), axis_names)
    end

    function Mesh(
        device_ids::AbstractArray{<:Integer,D}, axis_names::NTuple{D,Union{String,Symbol}}
    ) where {D}
        return new{D}(device_ids, 0:(length(device_ids) - 1), Symbol.(axis_names))
    end

    # XXX (Deprecated): remove in v0.3
    function Mesh(
        devices::NTuple{D,<:XLA.AbstractDevice}, shape::Dims{D}, axis_names
    ) where {D}
        Base.depwarn(
            "Mesh(devices::NTuple{D,<:XLA.AbstractDevice}, shape::Dims{D}, axis_names) is \
             deprecated, use Mesh(reshape(collect(XLA.device_ordinal.(devices)), shape), \
             axis_names) instead",
            :Mesh,
        )
        global_ids = reshape(collect(XLA.device_ordinal.(devices)), shape)
        return Mesh(global_ids, axis_names)
    end

    # XXX (Deprecated): remove in v0.3
    function Mesh(
        device_ids::Dims{D1}, shape::Dims{D}, axis_names::NTuple{D,Union{String,Symbol}}
    ) where {D,D1}
        Base.depwarn(
            "Mesh(device_ids::Dims{D1}, shape::Dims{D}, \
             axis_names::NTuple{D,Union{String,Symbol}}) is deprecated, use \
             Mesh(reshape(collect(Int64, device_ids), shape), axis_names) instead",
            :Mesh,
        )
        return Mesh(reshape(collect(Int64, device_ids), shape), axis_names)
    end
end

Base.length(m::Mesh) = length(m.device_ids)
Base.ndims(::Mesh{D}) where {D} = D

Base.size(mesh::Mesh) = size(mesh.device_ids)
Base.size(mesh::Mesh, axis::Int) = size(mesh.device_ids, axis)
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

# By default we use same sharding for all leaf nodes
Base.getproperty(sharding::AbstractSharding, name) = sharding
function Base.getproperty(sharding::AbstractSharding, name::Symbol)
    name ∈ fieldnames(typeof(sharding)) && return getfield(sharding, name)
    return sharding
end

function get_tensor_sharding_attribute end

"""
    sharding_to_array_slices(
        sharding, size_x; client=nothing, return_updated_sharding=Val(false)
    )

Given a sharding and an array size, returns the device to array slices mapping. If
`return_updated_sharding` is `Val(true)`, the updated sharding is returned as well (for
inputs requiring padding).
"""
function sharding_to_array_slices end

"""
    NoSharding()

Sharding annotation that indicates that the array is not sharded.

See also: [`Sharding.NamedSharding`](@ref)
"""
struct NoSharding <: AbstractSharding end

@inline ndevices(::NoSharding) = 1

@inline shard_type(::Type{NoSharding}, _) = ShardInfo{NoSharding,Nothing}

function (::NoSharding)(client::XLA.PJRT.Client, device, x::Union{AbstractArray,Number})
    device === nothing && (device = XLA.default_device(client))
    buffer = XLA.PJRT.AsyncBuffer(client, x, device)
    return (buffer,), ShardInfo(NoSharding(), nothing)
end

function (::NoSharding)(client::XLA.IFRT.Client, device, x::Union{AbstractArray,Number})
    device === nothing && (device = XLA.default_device(client))
    return XLA.IFRT.AsyncArray(client, x, device), ShardInfo(NoSharding(), nothing)
end

function sharding_to_array_slices(
    sharding::NoSharding, size_x; client=nothing, return_updated_sharding=Val(false)
)
    slices = Base.OneTo.(size_x)
    return_updated_sharding isa Val{true} && return (slices, sharding)
    return slices
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
struct NamedSharding{D1,D2} <: AbstractSharding
    mesh::Mesh{D1}
    partition_spec::Vector{Vector{Union{Nothing,Symbol}}}
    is_closed::NTuple{D2,Bool}
    priority::NTuple{D2,Int}
    subaxes::Vector{Vector{Union{Nothing,Dims{2}}}}

    function NamedSharding(
        mesh::Mesh{D1},
        partition_spec;
        subaxes=nothing,
        is_closed::NTuple{D2,Bool}=ntuple(Returns(true), length(partition_spec)),
        priority::NTuple{D2,Int}=ntuple(i -> -1, length(partition_spec)),
    ) where {D1,D2}
        axis_names = Symbol[]

        new_partition_spec = Vector{Vector{Union{Nothing,Symbol}}}(
            undef, length(partition_spec)
        )
        @inbounds for (i, p) in enumerate(partition_spec)
            if p === nothing
                new_partition_spec[i] = [nothing]
            elseif p isa Tuple
                new_partition_spec[i] = Vector{Union{Nothing,Symbol}}(undef, length(p))
                for (j, pⱼ) in enumerate(p)
                    @assert pⱼ isa Symbol || pⱼ isa String
                    new_partition_spec[i][j] = Symbol(pⱼ)
                    push!(axis_names, Symbol(pⱼ))
                end
            elseif p isa Symbol || p isa String
                push!(axis_names, Symbol(p))
                new_partition_spec[i] = [Symbol(p)]
            elseif p isa Vector
                new_partition_spec[i] = copy(p)
            else
                error("Unexpected partition spec $(partition_spec) [$(p)]")
            end
        end
        @assert allunique(axis_names) "Duplicate axis names!"

        if subaxes === nothing
            subaxes = Vector{Vector{Union{Nothing,Dims{2}}}}(undef, length(partition_spec))
            @inbounds for (i, pspec) in enumerate(new_partition_spec)
                subaxes[i] = Vector{Union{Nothing,Dims{2}}}(undef, length(pspec))
                for j in 1:length(pspec)
                    subaxes[i][j] = nothing
                end
            end
        else
            @assert length(subaxes) == length(new_partition_spec)
            for (i, pspec) in enumerate(new_partition_spec)
                @assert length(pspec) == length(subaxes[i])
            end
        end

        return new{D1,D2}(mesh, new_partition_spec, is_closed, priority, subaxes)
    end
end

function named_sharding_from_tensor_sharding_attr(mesh::Mesh, tensor_sharding_attr)
    @assert MLIR.API.sdyAttributeIsATensorShardingAttr(tensor_sharding_attr)

    ndims = MLIR.API.sdyTensorShardingAttrGetDimShardingsSize(tensor_sharding_attr)

    partition_spec = Vector{Vector{Union{Nothing,Symbol}}}(undef, ndims)
    is_closed = Vector{Bool}(undef, ndims)
    priority = Vector{Int}(undef, ndims)
    subaxes = Vector{Vector{Union{Nothing,Dims{2}}}}(undef, ndims)
    for i in 1:ndims
        dim_sharding_attr = MLIR.IR.Attribute(
            MLIR.API.sdyTensorShardingAttrGetDimShardingsElem(tensor_sharding_attr, i - 1)
        )

        naxes = MLIR.API.sdyDimensionShardingAttrGetAxesSize(dim_sharding_attr)
        axes = Vector{Symbol}(undef, naxes)

        if naxes == 0
            subaxes[i] = [nothing]
        else
            subaxes[i] = Vector{Union{Nothing,Dims{2}}}(undef, naxes)
        end

        for j in 1:naxes
            axis_elem = MLIR.IR.Attribute(
                MLIR.API.sdyDimensionShardingAttrGetAxesElem(dim_sharding_attr, j - 1)
            )

            subaxisinfo = MLIR.IR.Attribute(
                MLIR.API.sdyAxisRefAttrGetSubAxisInfo(axis_elem)
            )
            if subaxisinfo.attribute.ptr == C_NULL
                subaxes[i][j] = nothing
            else
                pre_size = MLIR.API.sdySubAxisInfoAttrGetPreSize(subaxisinfo)
                actual_size = MLIR.API.sdySubAxisInfoAttrGetSize(subaxisinfo)
                subaxes[i][j] = (Int64(pre_size), Int64(actual_size))
            end

            axis_name = Symbol(String(MLIR.API.sdyAxisRefAttrGetName(axis_elem)))
            axes[j] = axis_name
        end

        if naxes == 0
            partition_spec[i] = [nothing]
        else
            partition_spec[i] = axes
        end

        is_closed[i] = MLIR.API.sdyDimensionShardingAttrGetIsClosed(dim_sharding_attr)
        priority[i] = MLIR.API.sdyDimensionShardingAttrGetPriority(dim_sharding_attr)
    end
    reverse!(subaxes)
    reverse!(partition_spec)

    # Assuming `do_transpose` is true here
    return NamedSharding(
        mesh, partition_spec; subaxes, is_closed=Tuple(is_closed), priority=Tuple(priority)
    )
end

@inline ndevices(sharding::NamedSharding) = length(sharding.mesh.device_ids)

@inline function shard_type(::Type{NamedSharding{D1,D2}}, N) where {D1,D2}
    return ShardInfo{NamedSharding{D1,D2},Vector{NTuple{N,UnitRange{Int64}}}}
end

function (sharding::NamedSharding)(
    client::XLA.PJRT.Client, _, x::Union{AbstractArray,Number}
)
    device_to_array_slices, sharding = sharding_to_array_slices(
        sharding, size(x); client, return_updated_sharding=Val(true)
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

function (sharding::NamedSharding)(
    client::XLA.IFRT.Client, _, x::Union{AbstractArray,Number}
)
    device_to_array_slices, sharding = sharding_to_array_slices(
        sharding, size(x); client, return_updated_sharding=Val(true)
    )

    ifrt_sharding = XLA.IFRT.Sharding(
        vec(Reactant.XLA.get_device.((client,), sharding.mesh.device_ids)),
        convert(HloSharding, sharding).hlo_sharding,
    )
    data = XLA.IFRT.AsyncArray(client, x, ifrt_sharding)
    return data, ShardInfo(sharding, device_to_array_slices)
end

function get_tensor_sharding_attribute(
    sharding::NamedSharding,
    ctx,
    mesh_name,
    mesh_attr,
    size_arr;
    dialect=:auto,
    do_transpose=true,
)
    dialect == :auto && (dialect = :sdy)
    @assert dialect ∈ (:sdy, :mhlo)

    if dialect == :mhlo
        return get_tensor_sharding_attribute(
            convert(HloSharding, sharding),
            ctx,
            mesh_name,
            mesh_attr,
            size_arr;
            dialect,
            do_transpose,
        )
    end

    dimension_sharding_attrs = Vector{MLIR.API.MlirAttribute}(
        undef, length(sharding.partition_spec)
    )
    for (j, names) in enumerate(sharding.partition_spec)
        if length(names) == 1 && names[1] === nothing
            axes = MLIR.IR.Attribute[]
        else
            axes = Vector{MLIR.API.MlirAttribute}(undef, length(names))
        end

        for (i, (name, subaxisinfo)) in enumerate(zip(names, sharding.subaxes[j]))
            name === nothing && continue
            subaxisinfo = if subaxisinfo === nothing
                MLIR.API.MlirAttribute(C_NULL)
            else
                MLIR.API.sdySubAxisInfoAttrGet(ctx, subaxisinfo[1], subaxisinfo[2])
            end
            axes[i] = MLIR.API.sdyAxisRefAttrGet(ctx, String(name), subaxisinfo)
        end

        dimension_sharding_attrs[j] = MLIR.API.sdyDimensionShardingAttrGet(
            ctx, length(axes), axes, sharding.is_closed[j], sharding.priority[j]
        )
    end

    tensor_sharding_attr = MLIR.IR.Attribute(
        MLIR.API.sdyTensorShardingAttrGet(
            ctx,
            mesh_name,
            length(dimension_sharding_attrs),
            do_transpose ? reverse(dimension_sharding_attrs) : dimension_sharding_attrs,
            0,
            MLIR.API.MlirAttribute[],
        ),
    )
    return tensor_sharding_attr, :sdy
end

function sharding_to_array_slices(
    sharding::NamedSharding, size_x; return_updated_sharding=Val(false), client=nothing
)
    hlo_sharding = convert(HloSharding, sharding)

    # Check if the input needs to be padded. If so this sharding is not valid and we
    # need to request the tensor sharding from XLA
    condensed_op_sharding = convert(XLA.CondensedOpSharding, hlo_sharding.hlo_sharding)
    device_to_array_slices, needs_padding = XLA.sharding_to_concrete_array_indices(
        condensed_op_sharding, size_x, sharding.mesh.logical_device_ids
    )

    if needs_padding
        kws = client === nothing ? (;) : (; client)
        tmp = if length(size_x) == 0
            Reactant.ConcreteRNumber(zero(Float32); kws...)
        else
            Reactant.ConcreteRArray(ones(Float32, size_x...); kws...)
        end
        _, exec, mlir_fn_res, _, _ = Reactant.Compiler.compile_xla(
            Reactant.Ops.negate,
            (tmp,);
            input_shardings=IdDict(tmp => sharding),
            shardy_passes=:no_stablehlo_export,
        )

        get_from_hlo_sharding = true
        result_attrs = MLIR.IR.attr(mlir_fn_res.f, "res_attrs")
        if result_attrs !== nothing && length(result_attrs) == 1
            result_attr = result_attrs[0]
            if MLIR.IR.isdict(result_attr)
                mlir_attr = MLIR.API.mlirDictionaryAttrGetElementByName(
                    result_attr, "sdy.sharding"
                )
                if mlir_attr.ptr != C_NULL
                    sharding = Reactant.Sharding.named_sharding_from_tensor_sharding_attr(
                        mlir_fn_res.sharding_mesh, MLIR.IR.Attribute(mlir_attr)
                    )
                    get_from_hlo_sharding = false
                    condensed_op_sharding = convert(
                        XLA.CondensedOpSharding,
                        convert(Reactant.Sharding.HloSharding, sharding).hlo_sharding,
                    )
                end
            end
        end

        if get_from_hlo_sharding
            condensed_op_sharding = convert(
                XLA.CondensedOpSharding,
                convert(
                    Reactant.XLA.HloSharding,
                    only(Reactant.XLA.get_parameter_shardings(exec)),
                ),
            )
        end

        device_to_array_slices, needs_padding = XLA.sharding_to_concrete_array_indices(
            condensed_op_sharding, size_x, sharding.mesh.logical_device_ids
        )

        @assert !needs_padding "This shouldn't happen. Open an issue on Reactant.jl.\nInput shape: $(size_x).\nOriginal Sharding: $(string(hlo_sharding.hlo_sharding)).\nNew sharding: $(string(convert(Reactant.XLA.HloSharding, only(Reactant.XLA.get_parameter_shardings(exec))))).\nArray Slices: $(device_to_array_slices)."
    end

    return_updated_sharding isa Val{true} && return (device_to_array_slices, sharding)
    return device_to_array_slices
end

# TODO: Something like NamedDims.jl will allow us to support NamedDimsSharding similar to
#       `levanter`

"""
    DimsSharding(
        mesh::Mesh{M},
        dims::NTuple{D,Int},
        partition_spec;
        is_closed::NTuple{D,Bool}=ntuple(Returns(true), D),
        priority::NTuple{D,Int}=ntuple(i -> -1, D),
    )

Similar to [`NamedSharding`](@ref) but works for a arbitrary dimensional array. Dimensions
not specified in `dims` are replicated. If any dimension in `dims` is greater than the total
number of dimensions in the array, the corresponding `partition_spec`, `is_closed` and
`priority` are ignored. Additionally for any negative dimensions in `dims`, the true
dims are calculated as `ndims(x) - dim + 1`. A dims value of `0` will throw an error.
"""
struct DimsSharding{M,D,P} <: AbstractSharding
    mesh::Mesh{M}
    dims::NTuple{D,Int}
    partition_spec::P
    is_closed::NTuple{D,Bool}
    priority::NTuple{D,Int}

    function DimsSharding(
        mesh::Mesh{M},
        dims::NTuple{D,Int},
        partition_spec;
        is_closed::NTuple{D,Bool}=ntuple(Returns(true), length(partition_spec)),
        priority::NTuple{D,Int}=ntuple(i -> -1, length(partition_spec)),
    ) where {M,D}
        @assert length(partition_spec) == length(dims)
        # Validity checks on the inputs are deferred to NamedSharding
        return new{M,D,typeof(partition_spec)}(
            mesh, dims, partition_spec, is_closed, priority
        )
    end
end

@inline ndevices(sharding::DimsSharding) = length(sharding.mesh.device_ids)

@inline function shard_type(::Type{DimsSharding{M,D,P}}, N) where {M,D,P}
    return shard_type(NamedSharding{M,D}, N)
end

function standardize_sharding(sharding::DimsSharding, size_x)
    N = length(size_x)
    final_dims = map(sharding.dims) do d
        @assert !iszero(d) "dims cannot contain 0"
        return ifelse(d < 0, N + d + 1, d)
    end

    dim_indices = ntuple(i -> findfirst(==(i), final_dims), N)
    partition_spec = ntuple(N) do i
        dim_index = dim_indices[i]
        dim_index === nothing && return nothing # replicated dimension
        return sharding.partition_spec[dim_index]
    end
    is_closed = ntuple(N) do i
        dim_index = dim_indices[i]
        dim_index === nothing && return true # replicated dimension
        return sharding.is_closed[dim_index]
    end
    priority = ntuple(N) do i
        dim_index = dim_indices[i]
        dim_index === nothing && return -1 # replicated dimension
        return sharding.priority[dim_index]
    end

    return NamedSharding(sharding.mesh, partition_spec; is_closed, priority)
end

function (sharding::DimsSharding)(
    client::XLA.AbstractClient, device, x::Union{AbstractArray,Number}
)
    return (standardize_sharding(sharding, size(x)))(client, device, x)
end

function sharding_to_array_slices(sharding::DimsSharding, size_x; kwargs...)
    return sharding_to_array_slices(
        standardize_sharding(sharding, size_x), size_x; kwargs...
    )
end

# HloSharding
# This stores the sharding information in the form of XLA.HloSharding, and provides a
# central type for the final storage. It also potentially saves us the pain of not having
# to regenerate the partition spec from the HloSharding.
struct HloSharding{D1,D2,PS} <: AbstractSharding
    hlo_sharding::XLA.HloSharding
    parent_sharding::PS
    mesh::Mesh{D1}
    is_closed::NTuple{D2,Bool}
    priority::NTuple{D2,Int}

    function HloSharding(
        hlo_sharding::XLA.HloSharding,
        mesh::Mesh{D1},
        is_closed,
        priority,
        parent_sharding::Union{Nothing,AbstractSharding}=nothing,
    ) where {D1}
        @assert length(is_closed) == length(priority)
        return new{D1,length(is_closed),typeof(parent_sharding)}(
            hlo_sharding, parent_sharding, mesh, is_closed, priority
        )
    end
end

function Base.convert(::Type{HloSharding}, sharding::NamedSharding)
    MLIR.IR.with_context(; allow_use_existing=true) do ctx
        mesh_op = Reactant.Ops.mesh(
            sharding.mesh; mod=MLIR.IR.Module(MLIR.IR.Location(; context=ctx))
        )

        tensor_sharding_attr, _ = get_tensor_sharding_attribute(
            sharding, ctx, mesh_op.sym_name, mesh_op.mesh_attr, nothing; dialect=:sdy
        )

        hlo_sharding = XLA.HloSharding(
            @ccall MLIR.API.mlir_c.hloShardingFromTensorShardingAttr(
                tensor_sharding_attr::MLIR.API.MlirAttribute,
                mesh_op.mesh_attr.attribute::MLIR.API.MlirAttribute,
            )::Ptr{Cvoid}
        )

        return HloSharding(
            hlo_sharding, sharding.mesh, sharding.is_closed, sharding.priority
        )
    end
end

@inline ndevices(sharding::HloSharding) = length(sharding.mesh.device_ids)

@inline function shard_type(::Type{HloSharding{D1,D2,PS}}, N) where {D1,D2,PS}
    return ShardInfo{HloSharding{D1,D2,PS},Vector{NTuple{N,UnitRange{Int64}}}}
end

function sharding_to_array_slices(
    sharding::HloSharding, size_x; client=nothing, return_updated_sharding=Val(false)
)
    # Check if the input needs to be padded. If so this sharding is not valid and we
    # need to request the tensor sharding from XLA
    condensed_op_sharding = convert(XLA.CondensedOpSharding, sharding.hlo_sharding)
    device_to_array_slices, needs_padding = XLA.sharding_to_concrete_array_indices(
        condensed_op_sharding, size_x, sharding.mesh.logical_device_ids
    )

    if needs_padding
        kws = client === nothing ? (;) : (; client)
        tmp = if length(size_x) == 0
            Reactant.ConcreteRNumber(zero(Float32); kws...)
        else
            Reactant.ConcreteRArray(ones(Float32, size_x...); kws...)
        end
        _, exec, _, _, _ = Reactant.Compiler.compile_xla(
            Reactant.Ops.negate, (tmp,); input_shardings=IdDict(tmp => sharding)
        )

        xla_hlo_sharding = convert(
            Reactant.XLA.HloSharding, only(Reactant.XLA.get_parameter_shardings(exec))
        )
        sharding = HloSharding(
            xla_hlo_sharding, sharding.mesh, sharding.is_closed, sharding.priority
        )

        condensed_op_sharding = convert(XLA.CondensedOpSharding, sharding.hlo_sharding)
        device_to_array_slices, needs_padding = XLA.sharding_to_concrete_array_indices(
            condensed_op_sharding, size_x, sharding.mesh.logical_device_ids
        )

        @assert !needs_padding "This shouldn't happen. Open an issue on Reactant.jl"
    end

    return_updated_sharding isa Val{true} && return (device_to_array_slices, sharding)
    return device_to_array_slices
end

function HloSharding(sharding::NamedSharding, client::XLA.PJRT.Client, _, x)
    device_to_array_slices, hlo_sharding = sharding_to_array_slices(
        convert(HloSharding, sharding), size(x); client, return_updated_sharding=Val(true)
    )

    data = ntuple(length(hlo_sharding.mesh)) do i
        XLA.PJRT.AsyncBuffer(
            client,
            x[device_to_array_slices[i]...],
            XLA.get_device(client, hlo_sharding.mesh.device_ids[i]),
        )
    end

    return data, ShardInfo(hlo_sharding, device_to_array_slices)
end

function HloSharding(sharding::NamedSharding, client::XLA.IFRT.Client, _, x)
    device_to_array_slices, hlo_sharding = sharding_to_array_slices(
        convert(HloSharding, sharding), size(x); client, return_updated_sharding=Val(true)
    )

    ifrt_sharding = XLA.IFRT.Sharding(
        vec(Reactant.XLA.get_device.((client,), hlo_sharding.mesh.device_ids)),
        hlo_sharding.hlo_sharding,
    )
    data = XLA.IFRT.AsyncArray(client, x, ifrt_sharding)
    return data, ShardInfo(hlo_sharding, device_to_array_slices)
end

function (sharding::HloSharding)(
    client::XLA.PJRT.Client, ::Nothing, x::Union{AbstractArray,Number}
)
    device_to_array_slices = sharding_to_array_slices(sharding, size(x); client)

    data = ntuple(length(sharding.mesh)) do i
        XLA.PJRT.AsyncBuffer(
            client,
            x[device_to_array_slices[i]...],
            XLA.get_device(client, sharding.mesh.device_ids[i]),
        )
    end

    return data, ShardInfo(sharding, device_to_array_slices)
end

function (sharding::HloSharding)(
    client::XLA.IFRT.Client, ::Nothing, x::Union{AbstractArray,Number}
)
    device_to_array_slices = sharding_to_array_slices(sharding, size(x); client)

    ifrt_sharding = XLA.IFRT.Sharding(
        vec(Reactant.XLA.get_device.((client,), sharding.mesh.device_ids)),
        sharding.hlo_sharding,
    )
    data = XLA.IFRT.AsyncArray(client, x, ifrt_sharding)

    return data, ShardInfo(sharding, device_to_array_slices)
end

function get_tensor_sharding_attribute(
    sharding::HloSharding, ctx, mesh_name, mesh_attr, size_arr; dialect=:auto, kwargs...
)
    if sharding.parent_sharding !== nothing
        # easier path with existing parent_sharding
        return get_tensor_sharding_attribute(
            sharding.parent_sharding,
            ctx,
            mesh_name,
            mesh_attr,
            size_arr;
            dialect,
            kwargs...,
        )
    end

    dialect == :auto && (dialect = :sdy)

    if dialect == :sdy # XXX: Not recommended path
        string_mesh_name = MLIR.IR.Attribute(MLIR.IR.flatsymbol(mesh_name); context=ctx)
        GC.@preserve sharding begin
            attr = MLIR.IR.Attribute(
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
        return attr, :sdy
    elseif dialect == :mhlo
        sharding_attr = parse(
            MLIR.IR.Attribute, "\"" * string(sharding.hlo_sharding) * "\""
        )
        return sharding_attr, :mhlo
    else
        error("Unknown dialect: $(dialect). Only :sdy and :mhlo are supported.")
    end
end

# Given Sharding + Array --> ShardInfo
# This is the structure that is stored in the `sharding` field of `ConcreteRArray`
struct ShardInfo{S,D} <: AbstractSharding
    sharding::S
    device_to_array_slices::D
end

@inline ndevices(sharding::ShardInfo) = length(sharding.mesh)

@inline shard_type(::Type{ShardInfo{S,D}}, N) where {S,D} = shard_type(S, N)

function Base.getproperty(sharding::ShardInfo, name::Symbol)
    name ∈ (:sharding, :device_to_array_slices) && return getfield(sharding, name)
    return getproperty(unwrap_shardinfo(sharding), name)
end

function get_tensor_sharding_attribute(sharding::ShardInfo, args...; kwargs...)
    return get_tensor_sharding_attribute(sharding.sharding, args...; kwargs...)
end

function (sharding::ShardInfo)(
    client::XLA.AbstractClient, device, x::Union{AbstractArray,Number}
)
    return (sharding.sharding)(client, device, x)
end

function sharding_to_array_slices(sharding::ShardInfo, size_x; kwargs...)
    return sharding_to_array_slices(sharding.sharding, size_x; kwargs...)
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
is_sharded(::DimsSharding) = true
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

"""
    unwrap_shardinfo(x)

Unwraps a sharding info object, returning the sharding object itself.
"""
unwrap_shardinfo(x::AbstractSharding) = x
unwrap_shardinfo(x::ShardInfo) = unwrap_shardinfo(x.sharding)

end
