module Sharding

# XXX: Import ShardyPropagationOptions here to avoid breaking old code
using ..Reactant: Reactant, XLA, MLIR, ShardyPropagationOptions
using ReactantCore: ReactantCore

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
struct Mesh{D,ID<:AbstractVector{Int}}
    device_ids::Vector{Int64}
    logical_device_ids::ID
    axis_names::NTuple{D,Symbol}
    axis_sizes::Dims{D}

    function Mesh(devices::AbstractArray{<:XLA.AbstractDevice}, axis_names)
        return Mesh(XLA.device_ordinal.(devices), axis_names)
    end

    function Mesh(
        device_ids::AbstractArray{<:Integer,D}, axis_names::NTuple{D,Union{String,Symbol}}
    ) where {D}
        return Mesh(device_ids, sortperm(vec(device_ids)) .- 1, axis_names)
    end

    function Mesh(
        device_ids::AbstractArray{<:Integer,D},
        logical_device_ids::AbstractVector{Int64},
        axis_names::NTuple{D,Union{String,Symbol}},
    ) where {D}
        return Mesh(
            sort!(vec(device_ids)), logical_device_ids, axis_names, size(device_ids)
        )
    end

    function Mesh(
        sorted_device_ids::AbstractVector{<:Integer},
        logical_device_ids::AbstractVector{Int64},
        axis_names::NTuple{D,Union{String,Symbol}},
        axis_sizes::Dims{D},
    ) where {D}
        @assert length(logical_device_ids) ≥ 1
        if length(logical_device_ids) == 1
            @warn "Constructing a single device mesh is not well supported and is \
                   equivalent to not specifying any sharding. If you want to mock \
                   multi-device setup on a single cpu host, set the environment variable \
                   XLA_FLAGS=\"--xla_force_host_platform_device_count=12\" before loading \
                   Reactant.jl and force reactant to use `cpu` devices using \
                   `Reactant.set_default_backend(\"cpu\")`." maxlog = 1
        end
        return new{D,typeof(logical_device_ids)}(
            sorted_device_ids, logical_device_ids, Symbol.(axis_names), axis_sizes
        )
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

    # XXX (Deprecated): remove in v0.3
    Mesh(::Tuple{}, ::Tuple{}, ::Tuple{}) = throw(MethodError(Mesh, ((), (), ())))
end

function sdy_mesh_to_reactant_mesh(mesh_attr::MLIR.IR.Attribute, global_device_ids)
    @assert MLIR.API.sdyAttributeIsAMeshAttr(mesh_attr.attribute)

    ndevice_ids = MLIR.API.sdyMeshAttrGetDeviceIdsSize(mesh_attr)
    logical_device_ids = Vector{Int64}(undef, ndevice_ids)
    for i in 1:ndevice_ids
        logical_device_ids[i] = MLIR.API.sdyMeshAttrGetDeviceIdsElem(mesh_attr, i - 1)
    end

    naxes = MLIR.API.sdyMeshAttrGetAxesSize(mesh_attr)
    mesh_axes = Vector{Pair{Symbol,Int64}}(undef, naxes)
    for i in 1:naxes
        mesh_axis_attr = MLIR.IR.Attribute(
            MLIR.API.sdyMeshAttrGetAxesElem(mesh_attr, i - 1)
        )
        @assert MLIR.API.sdyAttributeIsAMeshAxisAttr(mesh_axis_attr)
        mesh_axis_name = String(MLIR.API.sdyMeshAxisAttrGetName(mesh_axis_attr))
        mesh_axis_size = MLIR.API.sdyMeshAxisAttrGetSize(mesh_axis_attr)
        mesh_axes[i] = Symbol(mesh_axis_name) => mesh_axis_size
    end

    if ndevice_ids == 0
        logical_device_ids = 0:(prod(last, mesh_axes) - 1)
    end

    @assert length(logical_device_ids) == length(global_device_ids)

    mesh = Mesh(
        global_device_ids,
        logical_device_ids,
        ntuple(i -> first(mesh_axes[i]), length(mesh_axes)),
        Tuple(last.(mesh_axes)),
    )

    cache = Reactant.Compiler.sdycache(; throw_error=ReactantCore.within_compile())
    key = (mesh.logical_device_ids, mesh.axis_names, size(mesh))
    cache === nothing && return mesh
    haskey(cache, key) && return cache[key].mesh
    return mesh
end

Base.length(m::Mesh) = length(m.logical_device_ids)
Base.ndims(::Mesh{D}) where {D} = D

Base.size(mesh::Mesh) = mesh.axis_sizes
Base.size(mesh::Mesh, axis::Int) = mesh.axis_sizes[axis]
function Base.size(mesh::Mesh, axis::Union{String,Symbol})
    return size(mesh, findfirst(==(Symbol(axis)), mesh.axis_names))
end
Base.size(mesh::Mesh, ::Nothing) = 1

Base.in(axis::Union{String,Symbol}, mesh::Mesh) = Symbol(axis) ∈ mesh.axis_names

abstract type AbstractSharding end

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

function (::NoSharding)(client::XLA.PJRT.Client, device, S::Type, dims::Dims)
    device === nothing && (device = XLA.default_device(client))
    buffer = similar(XLA.PJRT.AsyncBuffer, S, dims; client, device)
    return (buffer,), ShardInfo(NoSharding(), nothing)
end

function (::NoSharding)(client::XLA.IFRT.Client, device, x::Union{AbstractArray,Number})
    device === nothing && (device = XLA.default_device(client))
    return (
        XLA.IFRT.AsyncArray(client, x, device), ShardInfo(NoSharding(), nothing), nothing
    )
end

function sharding_to_array_slices(
    sharding::NoSharding, size_x; client=nothing, return_updated_sharding=Val(false)
)
    slices = (Base.OneTo.(size_x),)
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
struct NamedSharding{D,M<:Mesh} <: AbstractSharding
    mesh::M
    partition_spec::Vector{Vector{Union{Nothing,Symbol}}}
    is_closed::NTuple{D,Bool}
    priority::NTuple{D,Int}
    subaxes::Vector{Vector{Union{Nothing,Dims{2}}}}
end

function codegen_with_new_mesh(named_sharding::NamedSharding, mesh_sym)
    return :($(NamedSharding)(
        $(mesh_sym),
        $(named_sharding.partition_spec),
        $(named_sharding.is_closed),
        $(named_sharding.priority),
        $(named_sharding.subaxes),
    ))
end

function NamedSharding(
    mesh::Mesh,
    partition_spec;
    subaxes=nothing,
    is_closed::NTuple{D,Bool}=ntuple(Returns(true), length(partition_spec)),
    priority::NTuple{D,Int}=ntuple(i -> -1, length(partition_spec)),
) where {D}
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

    return NamedSharding{D,typeof(mesh)}(
        mesh, new_partition_spec, is_closed, priority, subaxes
    )
end

function sdy_tensor_sharding_to_named_sharding(mesh::Mesh, tensor_sharding_attr)
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

@inline ndevices(sharding::NamedSharding) = length(sharding.mesh)

@inline function shard_type(::Type{NamedSharding{D,M}}, N) where {D,M}
    @assert D == N
    return ShardInfo{NamedSharding{D,M},Vector{NTuple{N,UnitRange{Int64}}}}
end

function (sharding::NamedSharding)(
    client::XLA.PJRT.Client, _, x::Union{AbstractArray,Number}
)
    if !issorted(sharding.mesh.logical_device_ids)
        error("PJRT doesn't support non-iota meshes. Use IFRT instead.")
    end

    device_to_array_slices, sharding = sharding_to_array_slices(
        sharding, size(x); client, return_updated_sharding=Val(true)
    )

    data = ntuple(length(sharding.mesh)) do i
        XLA.PJRT.AsyncBuffer(
            client,
            if length(device_to_array_slices[i]) == 0
                x
            else
                x[device_to_array_slices[i]...]
            end,
            XLA.get_device(client, sharding.mesh.device_ids[i]),
        )
    end

    return data, ShardInfo(sharding, device_to_array_slices)
end

function (sharding::NamedSharding)(client::XLA.PJRT.Client, _, S::Type, dims::Dims)
    if !issorted(sharding.mesh.logical_device_ids)
        error("PJRT doesn't support non-iota meshes. Use IFRT instead.")
    end

    device_to_array_slices, sharding = sharding_to_array_slices(
        sharding, dims; client, return_updated_sharding=Val(true)
    )

    data = ntuple(length(sharding.mesh)) do i
        Base.@_inline_meta
        Base.similar(
            XLA.PJRT.AsyncBuffer,
            S,
            Dims(length.(device_to_array_slices[i]));
            client,
            device=XLA.get_device(client, sharding.mesh.device_ids[i]),
        )
    end

    return data, ShardInfo(sharding, device_to_array_slices)
end

function (sharding::NamedSharding)(
    client::XLA.IFRT.Client, _, x::Union{AbstractArray,Number}
)
    if x isa Number
        # Probably doesn't need so much complication
        device_to_array_slices = sharding_to_array_slices(
            sharding, size(x); client, return_updated_sharding=Val(false)
        )
        ifrt_sharding = XLA.IFRT.Sharding(
            vec(Reactant.XLA.get_device.((client,), sharding.mesh.device_ids)),
            convert(HloSharding, sharding).hlo_sharding,
        )
        data = XLA.IFRT.AsyncArray(client, x, ifrt_sharding)
        return data, ShardInfo(sharding, device_to_array_slices), nothing
    end

    partition_sizes = ones(Int64, length(sharding.partition_spec))
    for (i, pspec) in enumerate(sharding.partition_spec)
        for p in pspec
            partition_sizes[i] *= p === nothing ? 1 : size(sharding.mesh, p)
        end
    end
    remainders = size(x) .% partition_sizes

    if all(iszero, remainders) # fast path
        device_to_array_slices = sharding_to_array_slices(sharding, size(x); client)
        ifrt_sharding = XLA.IFRT.Sharding(
            vec(Reactant.XLA.get_device.((client,), sharding.mesh.device_ids)),
            convert(HloSharding, sharding).hlo_sharding,
        )
        return (
            XLA.IFRT.AsyncArray(client, x, ifrt_sharding),
            ShardInfo(sharding, device_to_array_slices),
            nothing,
        )
    end

    padding = Tuple((partition_sizes .- remainders) .% partition_sizes)
    device_to_array_slices = sharding_to_array_slices(sharding, size(x) .+ padding; client)

    ifrt_sharding = XLA.IFRT.Sharding(
        vec(Reactant.XLA.get_device.((client,), sharding.mesh.device_ids)),
        convert(HloSharding, sharding).hlo_sharding,
    )
    return (
        XLA.IFRT.AsyncArray(client, construct_padded_array(x, padding), ifrt_sharding),
        ShardInfo(sharding, device_to_array_slices),
        padding,
    )
end

function construct_padded_array(x::AbstractArray, padding)
    y = similar(x, size(x) .+ padding)
    view(y, [1:size(x, i) for i in 1:ndims(x)]...) .= x
    return y
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
        # MLIR for identity operation, avoid tracing here
        ctx = MLIR.IR.Context(Reactant.registry[], false)
        Reactant.Compiler.context_gc_vector[ctx] = Vector{
            Union{Reactant.TracedRArray,Reactant.TracedRNumber}
        }(
            undef, 0
        )
        @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
        MLIR.IR.activate!(ctx)

        sdycache = Reactant.Compiler.default_sdycache()
        Reactant.Compiler.activate_sdycache!(sdycache)

        try
            data_mlir_type = [
                MLIR.IR.TensorType(collect(Int64, reverse(size_x)), MLIR.IR.Type(Float32))
            ]
            mod = MLIR.IR.Module(MLIR.IR.Location(; context=ctx))

            (; sym_name, mesh_attr) = Reactant.Ops.mesh(sharding.mesh; mod)

            func = MLIR.Dialects.func.func_(;
                sym_name="main",
                function_type=MLIR.IR.FunctionType(data_mlir_type, data_mlir_type),
                no_inline=true,
                body=MLIR.IR.Region(),
            )
            fnbody = MLIR.IR.Block(data_mlir_type, [MLIR.IR.Location()])
            push!(MLIR.IR.region(func, 1), fnbody)
            MLIR.IR.activate!(fnbody)
            try
                MLIR.Dialects.func.return_([MLIR.IR.argument(fnbody, 1)])
            finally
                MLIR.IR.deactivate!(fnbody)
            end
            push!(MLIR.IR.body(mod), func)

            input_tensor_sharding_attr, _ = get_tensor_sharding_attribute(
                sharding, ctx, sym_name, mesh_attr, size_x; dialect=:sdy
            )

            MLIR.API.mlirFuncSetArgAttr(func, 0, "sdy.sharding", input_tensor_sharding_attr)

            Reactant.Compiler.run_pass_pipeline!(
                mod, join(["sdy-propagation-pipeline", "sdy-close-shardings"], ",")
            )

            mlir_attr = MLIR.API.mlirDictionaryAttrGetElementByName(
                MLIR.IR.attr(func, "res_attrs")[0], "sdy.sharding"
            )
            @assert mlir_attr.ptr != C_NULL
            sharding = sdy_tensor_sharding_to_named_sharding(
                sharding.mesh, MLIR.IR.Attribute(mlir_attr)
            )

            new_hlo_sharding = convert(HloSharding, sharding).hlo_sharding
            condensed_op_sharding = convert(XLA.CondensedOpSharding, new_hlo_sharding)

            device_to_array_slices, needs_padding = XLA.sharding_to_concrete_array_indices(
                condensed_op_sharding, size_x, sharding.mesh.logical_device_ids
            )

            @assert !needs_padding "This shouldn't happen. Open an issue on Reactant.jl.\nInput shape: $(size_x).\nOriginal Sharding: $(string(hlo_sharding.hlo_sharding)).\nNew sharding: $(string(new_hlo_sharding)).\nArray Slices: $(device_to_array_slices)."
        finally
            Reactant.Compiler.deactivate_sdycache!(sdycache)
            MLIR.IR.deactivate!(ctx)
        end
    end

    return_updated_sharding isa Val{true} && return (device_to_array_slices, sharding)
    return device_to_array_slices
end

# TODO: Something like NamedDims.jl will allow us to support NamedDimsSharding similar to
#       `levanter`

"""
    DimsSharding(
        mesh::Mesh,
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
struct DimsSharding{D,P,M<:Mesh} <: AbstractSharding
    mesh::M
    dims::NTuple{D,Int}
    partition_spec::P
    is_closed::NTuple{D,Bool}
    priority::NTuple{D,Int}

    function DimsSharding(
        mesh::M,
        dims::NTuple{D,Int},
        partition_spec;
        is_closed::NTuple{D,Bool}=ntuple(Returns(true), length(partition_spec)),
        priority::NTuple{D,Int}=ntuple(i -> -1, length(partition_spec)),
    ) where {M<:Mesh,D}
        @assert length(partition_spec) == length(dims)
        # Validity checks on the inputs are deferred to NamedSharding
        return new{D,typeof(partition_spec),M}(
            mesh, dims, partition_spec, is_closed, priority
        )
    end
end

@inline ndevices(sharding::DimsSharding) = length(sharding.mesh)

@inline function shard_type(::Type{DimsSharding{D,P,M}}, N) where {M,D,P}
    return shard_type(NamedSharding{D,M}, N)
end

function NamedSharding(sharding::DimsSharding, N)
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
    return (NamedSharding(sharding, ndims(x)))(client, device, x)
end

function (sharding::DimsSharding)(client::XLA.PJRT.Client, dev, S::Type, dims::Dims)
    return (NamedSharding(sharding, length(dims)))(client, dev, S, dims)
end

function sharding_to_array_slices(sharding::DimsSharding, size_x; kwargs...)
    return sharding_to_array_slices(
        NamedSharding(sharding, length(size_x)), size_x; kwargs...
    )
end

function get_tensor_sharding_attribute(
    sharding::DimsSharding, ctx, mesh_name, mesh_attr, size_arr; kwargs...
)
    return get_tensor_sharding_attribute(
        NamedSharding(sharding, length(size_arr)),
        ctx,
        mesh_name,
        mesh_attr,
        size_arr;
        kwargs...,
    )
end

"""
    Replicated(mesh::Mesh)

Sharding annotation that indicates that the array is fully replicated along all dimensions.
"""
struct Replicated{M<:Mesh} <: AbstractSharding
    mesh::M
end

codegen_with_new_mesh(::Replicated, mesh_sym) = :($(Replicated)($mesh_sym))

@inline ndevices(sharding::Replicated) = length(sharding.mesh)

@inline shard_type(::Type{Replicated{M}}, N) where {M} = shard_type(NamedSharding{N,M}, N)

function NamedSharding(sharding::Replicated, ndims::Int)
    return NamedSharding(sharding.mesh, ntuple(Returns(nothing), ndims))
end

function (sharding::Replicated)(
    client::XLA.AbstractClient, device, x::Union{AbstractArray,Number}
)
    return (NamedSharding(sharding, ndims(x)))(client, device, x)
end

function (sharding::Replicated)(client::XLA.PJRT.Client, dev, S::Type, dims::Dims)
    return (NamedSharding(sharding, length(dims)))(client, dev, S, dims)
end

function sharding_to_array_slices(sharding::Replicated, size_x; kwargs...)
    return sharding_to_array_slices(
        NamedSharding(sharding, length(size_x)), size_x; kwargs...
    )
end

function get_tensor_sharding_attribute(
    sharding::Replicated, ctx, mesh_name, mesh_attr, size_arr; kwargs...
)
    return get_tensor_sharding_attribute(
        NamedSharding(sharding, length(size_arr)),
        ctx,
        mesh_name,
        mesh_attr,
        size_arr;
        kwargs...,
    )
end

# HloSharding
# This stores the sharding information in the form of XLA.HloSharding, and provides a
# central type for the final storage. It also potentially saves us the pain of not having
# to regenerate the partition spec from the HloSharding.
struct HloSharding{D,PS,M<:Mesh} <: AbstractSharding
    hlo_sharding::XLA.HloSharding
    parent_sharding::PS
    mesh::M
    is_closed::NTuple{D,Bool}
    priority::NTuple{D,Int}

    function HloSharding(
        hlo_sharding::XLA.HloSharding,
        mesh::M,
        is_closed,
        priority,
        parent_sharding::Union{Nothing,AbstractSharding}=nothing,
    ) where {M<:Mesh}
        @assert length(is_closed) == length(priority)
        return new{length(is_closed),typeof(parent_sharding),M}(
            hlo_sharding, parent_sharding, mesh, is_closed, priority
        )
    end
end

HloSharding(sharding::HloSharding, size_x) = sharding

HloSharding(sharding::NamedSharding, size_x) = convert(HloSharding, sharding)

function HloSharding(sharding::Replicated, size_x)
    return convert(HloSharding, NamedSharding(sharding, length(size_x)))
end

function HloSharding(sharding::DimsSharding, size_x)
    return convert(HloSharding, NamedSharding(sharding, length(size_x)))
end

function Base.convert(::Type{HloSharding}, sharding::NamedSharding)
    MLIR.IR.with_context(; allow_use_existing=true) do ctx
        mesh_op = Reactant.Ops.mesh(
            sharding.mesh; mod=MLIR.IR.Module(MLIR.IR.Location(; context=ctx))
        )

        tensor_sharding_attr, _ = get_tensor_sharding_attribute(
            sharding, ctx, mesh_op.sym_name, mesh_op.mesh_attr, nothing; dialect=:sdy
        )

        return HloSharding(
            hlo_sharding_from_sdy_tensor_sharding_attr(
                tensor_sharding_attr, mesh_op.mesh_attr
            ),
            sharding.mesh,
            sharding.is_closed,
            sharding.priority,
        )
    end
end

function hlo_sharding_from_sdy_tensor_sharding_attr(attr, mesh_attr)
    @assert MLIR.API.sdyAttributeIsATensorShardingAttr(attr.attribute)
    @assert MLIR.API.sdyAttributeIsAMeshAttr(mesh_attr.attribute)
    GC.@preserve attr begin
        return XLA.HloSharding(
            @ccall MLIR.API.mlir_c.hloShardingFromTensorShardingAttr(
                attr.attribute::MLIR.API.MlirAttribute,
                mesh_attr.attribute::MLIR.API.MlirAttribute,
            )::Ptr{Cvoid}
        )
    end
end

@inline ndevices(sharding::HloSharding) = length(sharding.mesh)

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
        _, exec, _, _, _, _ = Reactant.Compiler.compile_xla(
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
    if !issorted(sharding.mesh.logical_device_ids)
        error("PJRT doesn't support non-iota meshes. Use IFRT instead.")
    end

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

    # XXX: Can we auto-pad this case too? Will think about it later, for now use
    #      NamedSharding
    return data, ShardInfo(hlo_sharding, device_to_array_slices), nothing
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

function (sharding::HloSharding)(client::XLA.PJRT.Client, ::Nothing, S::Type, dims::Dims)
    device_to_array_slices = sharding_to_array_slices(sharding, dims; client)

    data = ntuple(length(sharding.mesh)) do i
        Base.similar(
            XLA.PJRT.AsyncBuffer,
            S,
            Dims(length.(device_to_array_slices[i]));
            client,
            device=XLA.get_device(client, sharding.mesh.device_ids[i]),
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

    # XXX: Can we auto-pad this case too? Will think about it later, for now use
    #      NamedSharding
    return data, ShardInfo(sharding, device_to_array_slices), nothing
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

    if dialect == :sdy
        if Reactant.XLA.is_replicated(sharding.hlo_sharding)
            named_sharding = NamedSharding(
                sharding.mesh,
                ntuple(Returns(nothing), length(size_arr));
                sharding.is_closed,
                sharding.priority,
            )
            return get_tensor_sharding_attribute(
                named_sharding, ctx, mesh_name, mesh_attr, size_arr; dialect, kwargs...
            )
        end

        # XXX: Not recommended path
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

HloSharding(sharding::ShardInfo, size_x) = HloSharding(unwrap_shardinfo(sharding), size_x)

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
is_sharded(::AbstractSharding) = true
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

# sdy attributes to high-level sharding information
function sdy_sharding_to_reactant_sharding(attr, global_device_ids, mod)
    if !MLIR.IR.isdict(attr)
        return Replicated(
            Mesh(
                global_device_ids,
                0:(length(global_device_ids) - 1),
                (:all_axes,),
                (length(global_device_ids),),
            ),
        )
    end

    mlir_attr = MLIR.API.mlirDictionaryAttrGetElementByName(attr, "sdy.sharding")
    if mlir_attr.ptr == C_NULL
        return Replicated(
            Mesh(
                global_device_ids,
                0:(length(global_device_ids) - 1),
                (:all_axes,),
                (length(global_device_ids),),
            ),
        )
    end

    mesh_op = MLIR.IR.Operation(
        MLIR.API.mlirSymbolTableLookup(
            MLIR.IR.SymbolTable(MLIR.IR.Operation(mod)),
            MLIR.IR.leafref(
                MLIR.IR.Attribute(MLIR.API.sdyTensorShardingAttrGetMeshOrRef(mlir_attr))
            ),
        ),
        false,
    )
    return sdy_tensor_sharding_to_named_sharding(
        sdy_mesh_to_reactant_mesh(MLIR.IR.attr(mesh_op, "mesh"), global_device_ids),
        MLIR.IR.Attribute(mlir_attr),
    )
end

end
