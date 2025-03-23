@enumx OpShardingType begin
    Replicated
    Maximal
    Tuple
    Other
    Manual
    Unknown
end

function Base.convert(::Type{OpShardingType.T}, i::Integer)
    i == 0 && return OpShardingType.Replicated
    i == 1 && return OpShardingType.Maximal
    i == 2 && return OpShardingType.Tuple
    i == 3 && return OpShardingType.Other
    i == 4 && return OpShardingType.Manual
    i == 5 && return OpShardingType.Unknown
    return error("Invalid OpShardingType $i")
end

# xla::OpSharding
mutable struct OpSharding
    ptr::Ptr{Cvoid}

    function OpSharding(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return finalizer(free_op_sharding, new(ptr))
    end
end

function free_op_sharding(op_sharding::OpSharding)
    @ccall MLIR.API.mlir_c.free_op_sharding(op_sharding.ptr::Ptr{Cvoid})::Cvoid
end

function replicate_on_last_tile_dim(op_sharding::OpSharding)
    GC.@preserve op_sharding begin
        return @ccall MLIR.API.mlir_c.op_sharding_replicate_on_last_tile_dim(
            op_sharding.ptr::Ptr{Cvoid}
        )::Bool
    end
end

function op_sharding_type(op_sharding::OpSharding)
    type = GC.@preserve op_sharding begin
        @ccall MLIR.API.mlir_c.op_sharding_to_op_sharding_type(
            op_sharding.ptr::Ptr{Cvoid}
        )::Int32
    end
    return convert(OpShardingType.T, type)
end

function has_iota_reshape_dims(op_sharding::OpSharding)
    GC.@preserve op_sharding begin
        return @ccall MLIR.API.mlir_c.op_sharding_has_iota_reshape_dims(
            op_sharding.ptr::Ptr{Cvoid}
        )::Bool
    end
end

function iota_reshape_dims(op_sharding::OpSharding)
    GC.@preserve op_sharding begin
        ndims = @ccall MLIR.API.mlir_c.op_sharding_iota_reshape_dims_size(
            op_sharding.ptr::Ptr{Cvoid}
        )::Int32
    end
    dimensions = Vector{Int32}(undef, ndims)
    GC.@preserve op_sharding dimensions begin
        @ccall MLIR.API.mlir_c.op_sharding_iota_reshape_dims(
            op_sharding.ptr::Ptr{Cvoid}, dimensions::Ptr{Int32}
        )::Cvoid
    end
    return dimensions
end

function has_iota_transpose_perm(op_sharding::OpSharding)
    GC.@preserve op_sharding begin
        return @ccall MLIR.API.mlir_c.op_sharding_has_iota_transpose_perm(
            op_sharding.ptr::Ptr{Cvoid}
        )::Bool
    end
end

function iota_transpose_perm(op_sharding::OpSharding)
    GC.@preserve op_sharding begin
        ndims = @ccall MLIR.API.mlir_c.op_sharding_iota_transpose_perm_size(
            op_sharding.ptr::Ptr{Cvoid}
        )::Int32
    end
    dimensions = Vector{Int32}(undef, ndims)
    GC.@preserve op_sharding dimensions begin
        @ccall MLIR.API.mlir_c.op_sharding_iota_transpose_perm(
            op_sharding.ptr::Ptr{Cvoid}, dimensions::Ptr{Int32}
        )::Cvoid
    end
    dimensions .+= 1
    return dimensions
end

function tile_assignment_dimensions(op_sharding::OpSharding)
    GC.@preserve op_sharding begin
        ndims = @ccall MLIR.API.mlir_c.op_sharding_tile_assignment_dimensions_size(
            op_sharding.ptr::Ptr{Cvoid}
        )::Int32
    end
    dimensions = Vector{Int32}(undef, ndims)
    GC.@preserve op_sharding dimensions begin
        @ccall MLIR.API.mlir_c.op_sharding_tile_assignment_dimensions(
            op_sharding.ptr::Ptr{Cvoid}, dimensions::Ptr{Int32}
        )::Cvoid
    end
    return dimensions
end

function tile_assignment_devices(op_sharding::OpSharding)
    GC.@preserve op_sharding begin
        ndims = @ccall MLIR.API.mlir_c.op_sharding_tile_assignment_devices_size(
            op_sharding.ptr::Ptr{Cvoid}
        )::Int32
    end
    devices = Vector{Int32}(undef, ndims)
    GC.@preserve op_sharding devices begin
        @ccall MLIR.API.mlir_c.op_sharding_tile_assignment_devices(
            op_sharding.ptr::Ptr{Cvoid}, devices::Ptr{Int32}
        )::Cvoid
    end
    return devices
end

function has_last_tile_dims(op_sharding::OpSharding)
    GC.@preserve op_sharding begin
        return @ccall MLIR.API.mlir_c.op_sharding_has_last_tile_dims(
            op_sharding.ptr::Ptr{Cvoid}
        )::Bool
    end
end

# This separation is mostly for testing purposes
function generate_device_list_from_tile_assignment_devices(sharding::OpSharding)
    return tile_assignment_devices(sharding)
end

function generate_device_list_from_iota_tile(sharding::OpSharding)
    return generate_device_list_from_iota_tile(
        tile_assignment_dimensions(sharding),
        iota_reshape_dims(sharding),
        iota_transpose_perm(sharding),
    )
end

function generate_device_list_from_iota_tile(
    tile_assignment_dimensions, iota_reshape_dims, iota_transpose_perm
)
    # Generate device IDs using iota
    num_devices = prod(tile_assignment_dimensions)
    ird = Int64.(iota_reshape_dims)

    # Permute the iota array if iota_transpose_perm is provided
    # We need to ensure that we account for the col-major ordering in julia. See the
    # unit tests for examples.
    if !isempty(iota_transpose_perm)
        # XXX: Simplify the permutedims
        iota_devices = collect(Int64, reshape(0:(num_devices - 1), reverse(ird)...))

        iota_devices = permutedims(iota_devices, reverse(1:ndims(iota_devices)))
        iota_devices = permutedims(iota_devices, iota_transpose_perm)
        iota_devices = permutedims(iota_devices, reverse(1:ndims(iota_devices)))

        return vec(iota_devices)
    else
        @assert num_devices == prod(ird)
        return collect(0:(num_devices - 1))
    end
end

function generate_device_list(sharding::OpSharding)
    has_iota_reshape_dims(sharding) && return generate_device_list_from_iota_tile(sharding)
    return generate_device_list_from_tile_assignment_devices(sharding)
end

function get_number_of_ways_dim_sharded(op_sharding::OpSharding)
    op_sharding_type(op_sharding) == OpShardingType.Replicated && return Int64[], 1
    td = tile_assignment_dimensions(op_sharding)
    replicate_on_last_tile_dim(op_sharding) && return td[1:(end - 1)], td[end]
    return td, 1
end

function sharding_to_concrete_array_indices(sharding::OpSharding, shape, device_ids)
    return sharding_to_concrete_array_indices(
        convert(CondensedOpSharding, sharding), shape, device_ids
    )
end

function compute_array_indices_and_hlo_sharding(
    sharding::OpSharding, array_size, device_ids
)
    return compute_array_indices_and_hlo_sharding(
        convert(CondensedOpSharding, sharding), array_size, device_ids
    )
end

# This only stores the data that we currently support, and is useful for checking equality
# We would want to extend support to more of the fields at a later time
struct CondensedOpSharding{N}
    opsharding::OpSharding
    type::OpShardingType.T
    replicate_on_last_tile_dim::Bool
    tile_assignment::Array{Int64,N}
end

function Base.:(==)(a::CondensedOpSharding, b::CondensedOpSharding)
    return a.type == b.type &&
           a.replicate_on_last_tile_dim == b.replicate_on_last_tile_dim &&
           a.tile_assignment == b.tile_assignment
end

function Base.convert(::Type{CondensedOpSharding}, sharding::OpSharding)
    @assert !has_last_tile_dims(sharding) "Last Tile dimensions are not supported \
                                           yet!"

    type = op_sharding_type(sharding)

    if type == OpShardingType.Replicated || type == OpShardingType.Maximal
        tile_assignment = generate_device_list(sharding)
    elseif type == OpShardingType.Other
        td = tile_assignment_dimensions(sharding)
        tile_assignment = permutedims(
            reshape(generate_device_list(sharding), Int64.(reverse(td))...),
            reverse(1:length(td)),
        )
    else
        error("Invalid sharding type: $(type)")
    end

    return CondensedOpSharding(
        sharding, type, replicate_on_last_tile_dim(sharding), Int64.(tile_assignment)
    )
end

function get_number_of_ways_dim_sharded(op_sharding::CondensedOpSharding{N}) where {N}
    op_sharding.type == OpShardingType.Replicated && return Int64[], 1

    if op_sharding.replicate_on_last_tile_dim
        return (
            size(op_sharding.tile_assignment)[1:(N - 1)],
            size(op_sharding.tile_assignment, N),
        )
    end
    return size(op_sharding.tile_assignment), 1
end

function sharding_to_concrete_array_indices(
    sharding::CondensedOpSharding, shape::Dims{N}, device_ids
) where {N}
    if sharding.type == OpShardingType.Replicated || sharding.type == OpShardingType.Maximal
        return map(Returns(UnitRange.(1, shape)), device_ids), false
    elseif sharding.type == OpShardingType.Other
        partitions, num_replicas = get_number_of_ways_dim_sharded(sharding)
        @assert length(partitions) == length(shape)
        shape = reverse(shape)

        # XLA will automatically pad the inputs that don't match the final shape
        partitionable_shape = map(zip(shape, partitions)) do (dim, n_shards)
            dim % n_shards == 0 && return dim
            res = dim + n_shards ÷ 2
            return res - res % n_shards
        end
        partitionable_shape = Tuple(partitionable_shape)

        needs_padding = any(partitionable_shape .!= shape)

        # Calculate indices for each dimension
        axis_indices =
            map(zip(partitionable_shape, shape, partitions)) do (dim_padded, dim, n_shards)
                @assert dim > 0 "Invalid dimension: $dim"
                @assert n_shards > 0 "Invalid number of shards: $n_shards"
                n_shards == 1 && return [1:dim]
                shard_size = dim_padded ÷ n_shards

                return [
                    (i * shard_size + 1):min((i + 1) * shard_size, dim) for
                    i in 0:(n_shards - 1)
                ]
            end

        indices = Dict{Int,NTuple{N,UnitRange{Int}}}()
        device_idx = 1
        for _ in 1:num_replicas
            for idx_tuple in Iterators.product(axis_indices...)
                indices[sharding.tile_assignment[device_idx]] = reverse(idx_tuple)
                device_idx += 1
            end
        end

        return map(Base.Fix1(getindex, indices), device_ids), needs_padding
    else
        error("Unsupported sharding type: $(sharding.type)")
    end
end

function compute_array_indices_and_hlo_sharding(
    sharding::CondensedOpSharding, array_size, device_ids
)
    return (
        first(sharding_to_concrete_array_indices(sharding, array_size, device_ids)),
        convert(HloSharding, sharding),
    )
end

# xla::HloSharding
mutable struct HloSharding
    ptr::Ptr{Cvoid}

    function HloSharding(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return finalizer(free_hlo_sharding, new(ptr))
    end
end

function Base.:(==)(hsharding1::HloSharding, hsharding2::HloSharding)
    GC.@preserve hsharding1 hsharding2 begin
        return @ccall MLIR.API.mlir_c.hlo_sharding_check_eq(
            hsharding1.ptr::Ptr{Cvoid}, hsharding2.ptr::Ptr{Cvoid}
        )::Bool
    end
end

function free_hlo_sharding(hlo_sharding::HloSharding)
    @ccall MLIR.API.mlir_c.free_hlo_sharding(hlo_sharding.ptr::Ptr{Cvoid})::Cvoid
end

function Base.convert(::Type{CondensedOpSharding}, hlo_sharding::HloSharding)
    return convert(CondensedOpSharding, convert(OpSharding, hlo_sharding))
end

function Base.convert(::Type{OpSharding}, hlo_sharding::HloSharding)
    GC.@preserve hlo_sharding begin
        return OpSharding(
            @ccall MLIR.API.mlir_c.hlo_sharding_to_op_sharding(
                hlo_sharding.ptr::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function Base.convert(::Type{HloSharding}, op_sharding::OpSharding)
    GC.@preserve op_sharding begin
        return HloSharding(
            @ccall MLIR.API.mlir_c.hlo_sharding_from_op_sharding(
                op_sharding.ptr::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function Base.convert(::Type{HloSharding}, op_sharding::CondensedOpSharding)
    return convert(HloSharding, op_sharding.opsharding)
end

function Base.string(hlo_sharding::HloSharding)
    GC.@preserve hlo_sharding begin
        str = @ccall MLIR.API.mlir_c.hlo_sharding_to_string(
            hlo_sharding.ptr::Ptr{Cvoid}
        )::Cstring
    end
    return unsafe_string_and_free(str)
end

function Base.show(io::IO, hlo_sharding::HloSharding)
    print(io, "XLA.HloSharding(\"", string(hlo_sharding), "\")")
    return nothing
end

function sharding_to_concrete_array_indices(sharding::HloSharding, shape, device_ids)
    return sharding_to_concrete_array_indices(
        convert(CondensedOpSharding, sharding), shape, device_ids
    )
end

function compute_array_indices_and_hlo_sharding(
    sharding::HloSharding, array_size, device_ids
)
    return (
        compute_array_indices_and_hlo_sharding(
            convert(CondensedOpSharding, sharding), array_size, device_ids
        ),
        sharding,
    )
end

function tile_assignment_dimensions(hlo_sharding::HloSharding)
    GC.@preserve hlo_sharding begin
        ndims = @ccall MLIR.API.mlir_c.hlo_sharding_tile_assignment_dimensions_size(
            hlo_sharding.ptr::Ptr{Cvoid}
        )::Int32
    end
    dimensions = Vector{Int64}(undef, ndims)
    GC.@preserve hlo_sharding dimensions begin
        @ccall MLIR.API.mlir_c.hlo_sharding_tile_assignment_dimensions(
            hlo_sharding.ptr::Ptr{Cvoid}, dimensions::Ptr{Int64}, ndims::Int32
        )::Cvoid
    end
    return dimensions
end

function tile_assignment_devices(hlo_sharding::HloSharding)
    GC.@preserve hlo_sharding begin
        ndims = @ccall MLIR.API.mlir_c.hlo_sharding_tile_assignment_devices_size(
            hlo_sharding.ptr::Ptr{Cvoid}
        )::Int32
    end
    devices = Vector{Int64}(undef, ndims)
    GC.@preserve hlo_sharding devices begin
        @ccall MLIR.API.mlir_c.hlo_sharding_tile_assignment_devices(
            hlo_sharding.ptr::Ptr{Cvoid}, devices::Ptr{Int64}, ndims::Int32
        )::Cvoid
    end
    return devices
end

for check in (:is_tiled, :is_maximal, :is_tuple, :is_replicated, :is_manual, :is_unknown)
    cfn = Symbol(:hlo_sharding_, check)
    @eval function $(check)(hlo_sharding::HloSharding)
        GC.@preserve hlo_sharding begin
            return @ccall MLIR.API.mlir_c.$(cfn)(hlo_sharding.ptr::Ptr{Cvoid})::Bool
        end
    end
end

function replicate_on_last_tile_dim(hlo_sharding::HloSharding)
    GC.@preserve hlo_sharding begin
        return @ccall MLIR.API.mlir_c.hlo_sharding_replicate_on_last_tile_dim(
            hlo_sharding.ptr::Ptr{Cvoid}
        )::Bool
    end
end

# Taken from https://github.com/jax-ml/jax/blob/4ca97ad7165453816bc543e941b357407283a59e/jax/_src/sharding_impls.py#L773
# Converts an XLA.HloSharding to a PartitionSpec
function __unflatten_superdims(assignment)
    flat_assignment = collect(Int64, assignment)
    @assert first(flat_assignment) == 0 "First dimension must be 0"
    dims = Dims{2}[]
    while length(flat_assignment) > 1
        stride = flat_assignment[2]
        i = 1
        while i ≤ length(flat_assignment) && flat_assignment[i] == (i - 1) * stride
            i += 1
        end
        size = i - 1
        push!(dims, (size, stride))
        flat_assignment = @view flat_assignment[1:size:end]
    end
    return dims
end

function __explode_superdims(sizes, dims)
    strides_to_sizes = Dict(
        stride => size for (size, stride) in zip(sizes, Base.size_to_strides(1, sizes...))
    )
    final_dims = Dims{2}[]
    for (size, stride) in Iterators.reverse(dims)
        target_size = strides_to_sizes[stride]
        new_dims = Dims{2}[]
        while size > target_size
            @assert target_size > 1  # Ensure progress
            @assert size % target_size == 0
            push!(new_dims, (target_size, stride))
            size ÷= target_size
            stride *= target_size
            target_size = strides_to_sizes[stride]
        end
        @assert size == target_size "Expected size $size to be equal to target size \
                                     $target_size"
        push!(new_dims, (size, stride))
        append!(final_dims, reverse(new_dims))
    end
    return final_dims
end

function __unflatten_array(named_sizes, assignment)
    named_sizes = filter(p -> p.second != 1, named_sizes)
    sizes = collect(values(named_sizes))
    strides = Base.size_to_strides(1, sizes...)
    dims = __explode_superdims(sizes, __unflatten_superdims(assignment))
    dim_to_name = Dict(
        (size, stride) => name for
        (size, stride, name) in zip(sizes, strides, keys(named_sizes))
    )
    return [dim_to_name[d] for d in dims]
end

function generate_partition_spec(hlo_sharding, args...)
    return generate_partition_spec(convert(HloSharding, hlo_sharding), args...)
end

function generate_partition_spec(hlo_sharding::HloSharding, mesh, size_arr)
    @assert mesh isa Reactant.Sharding.Mesh

    is_replicated(hlo_sharding) && return ntuple(Returns(nothing), length(size_arr))

    if is_maximal(hlo_sharding) && length(mesh) == 1
        return ntuple(Returns(nothing), length(size_arr))
    end

    if is_tiled(hlo_sharding)
        mesh_shape = Dict(zip(mesh.axis_names, size(mesh)))
        mesh_axis_order = __unflatten_array(
            mesh_shape, tile_assignment_devices(hlo_sharding)
        )
        axis_idx = 1
        partitions = Union{NTuple{<:Any,Symbol},Symbol,Nothing}[]
        for dim_size in tile_assignment_dimensions(hlo_sharding)
            dim_partitions = Symbol[]
            while dim_size > 1
                axis = mesh_axis_order[axis_idx]
                axis_idx += 1
                axis_size = mesh_shape[axis]
                @assert dim_size % axis_size == 0 "Expected dim_size $dim_size to be \
                                                   divisible by axis_size $axis_size"
                dim_size ÷= axis_size
                push!(dim_partitions, axis)
            end

            if length(dim_partitions) == 0
                push!(partitions, nothing)
            elseif length(dim_partitions) == 1
                push!(partitions, only(dim_partitions))
            else
                push!(partitions, Tuple(dim_partitions))
            end
        end

        if replicate_on_last_tile_dim(hlo_sharding)
            pop!(partitions)
        end

        return reverse(Tuple(partitions))
    end

    return error("Unhandled HLO sharding type. Please open a bug report!")
end
