@enumx OpShardingType begin
    Replicated
    Maximal
    Tuple
    Other
    Manual
    Unknown
end

@enumx ShardGroupType begin
    As
    Like
end

# TODO: tuple sharding / op metadata
struct JLOpSharding
    type::Int32
    n_tile_dimensions::Int32
    tile_dimensions::Ptr{Int64}
    n_layout_minor_to_major::Int32
    layout_minor_to_major::Ptr{Int64}
    replicate_on_last_tile_dim::Bool
    n_last_tile_dims::Int32
    last_tile_dims::Ptr{Int32}
    n_tile_assignment_dimensions::Int32
    tile_assignment_dimensions::Ptr{Int64}
    n_tile_assignment_devices::Int32
    tile_assignment_devices::Ptr{Int64}
    n_iota_reshape_dims::Int32
    iota_reshape_dims::Ptr{Int64}
    n_iota_transpose_perm::Int32
    iota_transpose_perm::Ptr{Int32}
    is_shard_group::Bool
    shard_group_id::Int64
    shard_group_type::Int32
end

struct OpSharding
    type::OpShardingType.T
    tile_dimensions::Vector{Int64}
    layout_minor_to_major::Vector{Int64}
    replicate_on_last_tile_dim::Bool
    last_tile_dims::Vector{OpShardingType.T}
    tile_assignment_dimensions::Vector{Int64}
    tile_assignment_devices::Vector{Int64}
    iota_reshape_dims::Vector{Int64}
    iota_transpose_perm::Vector{Int32}
    is_shard_group::Bool
    shard_group_id::Int64
    shard_group_type::ShardGroupType.T
end

function OpSharding(sharding::JLOpSharding)
    @assert sharding.type != 2 "Tuple sharding is not supported yet!"

    last_tile_dims = unsafe_wrap(Array, sharding.last_tile_dims, sharding.n_last_tile_dims)
    tile_assignment_dimensions = unsafe_wrap(
        Array, sharding.tile_assignment_dimensions, sharding.n_tile_assignment_dimensions
    )
    tile_assignment_devices = unsafe_wrap(
        Array, sharding.tile_assignment_devices, sharding.n_tile_assignment_devices
    )
    iota_reshape_dims = unsafe_wrap(
        Array, sharding.iota_reshape_dims, sharding.n_iota_reshape_dims
    )
    iota_transpose_perm = unsafe_wrap(
        Array, sharding.iota_transpose_perm, sharding.n_iota_transpose_perm
    )
    iota_transpose_perm .+= 1

    tile_dimensions = unsafe_wrap(
        Array, sharding.tile_dimensions, sharding.n_tile_dimensions
    )
    layout_minor_to_major = unsafe_wrap(
        Array, sharding.layout_minor_to_major, sharding.n_layout_minor_to_major
    )

    return OpSharding(
        int_to_op_sharding_type(sharding.type),
        tile_dimensions,
        layout_minor_to_major,
        sharding.replicate_on_last_tile_dim,
        last_tile_dims,
        tile_assignment_dimensions,
        tile_assignment_devices,
        iota_reshape_dims,
        iota_transpose_perm,
        sharding.is_shard_group,
        sharding.shard_group_id,
        int_to_shard_group_type(sharding.shard_group_type),
    )
end

function int_to_op_sharding_type(i::Int32)
    i == 0 && return OpShardingType.Replicated
    i == 1 && return OpShardingType.Maximal
    i == 2 && return OpShardingType.Tuple
    i == 3 && return OpShardingType.Other
    i == 4 && return OpShardingType.Manual
    i == 5 && return OpShardingType.Unknown
    return error("Invalid OpShardingType $i")
end

function int_to_shard_group_type(i::Int32)
    i == 0 && return ShardGroupType.As
    i == 1 && return ShardGroupType.Like
    return error("Invalid ShardGroupType $i")
end

function generate_device_list(sharding::OpSharding)
    if !isempty(sharding.iota_reshape_dims)
        # Generate device IDs using iota
        num_devices = prod(sharding.iota_reshape_dims)
        iota_devices = collect(
            Int64, reshape(0:(num_devices - 1), sharding.iota_reshape_dims...)
        )

        # Permute the iota array if iota_transpose_perm is provided
        if !isempty(sharding.iota_transpose_perm)
            iota_devices = permutedims(iota_devices, Tuple(sharding.iota_transpose_perm))
        end

        # Flatten the permuted iota array to get tile_assignment_devices
        return vec(iota_devices)
    end
    return sharding.tile_assignment_devices
end

function get_number_of_ways_dim_sharded(op_sharding::OpSharding)
    op_sharding.type == OpShardingType.Replicated && return Int64[], 1

    if op_sharding.replicate_on_last_tile_dim
        return (
            op_sharding.tile_assignment_dimensions[1:(end - 1)],
            op_sharding.tile_assignment_dimensions[end],
        )
    end
    return op_sharding.tile_assignment_dimensions, 1
end

function sharding_to_concrete_array_indices(
    sharding::OpSharding, shape::Dims{N}, mesh
) where {N}
    return sharding_to_concrete_array_indices(CondensedOpSharding(sharding), shape, mesh)
end

function compute_array_indices_and_partition_spec(
    sharding::OpSharding, array_size::Dims{N}, mesh
) where {N}
    return compute_array_indices_and_partition_spec(
        CondensedOpSharding(sharding), array_size, mesh
    )
end

# This only stores the data that we currently support, and is useful for checking equality
# We would want to extend support to more of the fields at a later time
struct CondensedOpSharding{N}
    type::OpShardingType.T
    replicate_on_last_tile_dim::Bool
    tile_assignment::Array{Int64,N}
end

function Base.:(==)(a::CondensedOpSharding, b::CondensedOpSharding)
    return a.type == b.type &&
           a.replicate_on_last_tile_dim == b.replicate_on_last_tile_dim &&
           a.tile_assignment == b.tile_assignment
end

function CondensedOpSharding(sharding::OpSharding)
    @assert isempty(sharding.last_tile_dims) "Last Tile dimensions are not supported \
                                              yet!"
    @assert isempty(sharding.tile_dimensions) "Tile dimensions are not supported yet! \
                                               Open an issue with an MWE for this case."
    @assert isempty(sharding.layout_minor_to_major) "Layout transformation is not \
                                                     supported yet!"

    if sharding.type == OpShardingType.Replicated || sharding.type == OpShardingType.Maximal
        tile_assignment = generate_device_list(sharding)
    elseif sharding.type == OpShardingType.Other
        tile_assignment = reshape(
            generate_device_list(sharding), sharding.tile_assignment_dimensions...
        )
    else
        error("Invalid sharding type: $(sharding.type)")
    end

    return CondensedOpSharding(
        sharding.type, sharding.replicate_on_last_tile_dim, tile_assignment
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
    sharding::CondensedOpSharding, shape::Dims{N}, mesh
) where {N}
    if sharding.type == OpShardingType.Replicated || sharding.type == OpShardingType.Maximal
        return ntuple(Returns(ntuple(i -> 1:shape[i], N)), length(mesh))
    elseif sharding.type == OpShardingType.Other
        partitions, num_replicas = get_number_of_ways_dim_sharded(sharding)
        @assert length(partitions) == length(shape)
        shape = reverse(shape)

        # Calculate indices for each dimension
        axis_indices = map(zip(shape, partitions)) do (dim, n_shards)
            if n_shards == 1
                [Colon()]
            elseif n_shards > 1
                shard_size, remainder = divrem(dim, n_shards)
                @assert remainder == 0 "Dimension $dim not evenly divisible by $n_shards \
                                        shards"
                [(i * shard_size + 1):((i + 1) * shard_size) for i in 0:(n_shards - 1)]
            else
                error("Invalid number of shards: $n_shards")
            end
        end

        # XXX: Fix performance of this
        indices = Vector{NTuple{length(shape),Any}}(undef, length(mesh))
        tile_assignment = sharding.tile_assignment
        device_iter = Iterators.Stateful(tile_assignment)

        for idx_tuple in Iterators.product(axis_indices...)
            for _ in 1:num_replicas
                device = popfirst!(device_iter)
                # XXX: incorrect if devices are not contiguous
                indices[device + 1] = reverse(idx_tuple)
            end
        end

        return Tuple(indices)
    else
        error("Unsupported sharding type: $(sharding.type)")
    end
end

# Function to compute array indices for each device
function compute_array_indices_and_partition_spec(
    sharding::CondensedOpSharding, array_size::Dims{N}, mesh
) where {N}
    if sharding.type == OpShardingType.Replicated # All devices have the entire array
        return (
            ntuple(Returns(ntuple(i -> 1:array_size[i], N)), length(mesh)),
            ntuple(Returns(nothing), N),
        )
    elseif sharding.type == OpShardingType.Maximal # Only one device has the entire array
        @assert length(mesh) == 1
        return (
            ntuple(Returns(ntuple(i -> 1:array_size[i], N)), length(mesh)),
            ntuple(Returns(nothing), N),
        )
    elseif sharding.type == OpShardingType.Other # Tiled sharding
        tile_dims, _ = get_number_of_ways_dim_sharded(sharding)
        mesh_devices = Reactant.Sharding.device_ids(mesh)

        # Match dimensions to mesh axes
        used_axes = Set{Int}()
        partition_spec = ntuple(N) do dim
            if dim <= length(tile_dims) && tile_dims[dim] > 1
                tile_seq = __get_device_sequence(
                    sharding.tile_assignment, dim + sharding.replicate_on_last_tile_dim
                )

                for (axis_idx, axis_name) in enumerate(mesh.axis_names)
                    if axis_idx ∉ used_axes && size(mesh_devices, axis_idx) == length(tile_seq)
                        mesh_seq = __get_device_sequence(mesh_devices, axis_idx)
                        if tile_seq == mesh_seq || tile_seq == reverse(mesh_seq)
                            push!(used_axes, axis_idx)
                            return axis_name
                        end
                    end
                end
            end
            return nothing
        end

        device_to_array_indices = sharding_to_concrete_array_indices(
            sharding, array_size, mesh
        )

        return device_to_array_indices, partition_spec
    else
        error("Unsupported sharding type: $(sharding.type)")
    end
end

# Helper function to get device sequence along a dimension
function __get_device_sequence(arr, dim)
    idx = ones(Int, ndims(arr))
    sequence = Int[]
    for i in 1:size(arr, dim)
        idx[dim] = i
        push!(sequence, arr[idx...])
    end
    return sequence
end
