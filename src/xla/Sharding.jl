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

# Function to compute array indices for each device
function compute_array_indices_and_partition_spec(
    sharding::OpSharding, array_size::Dims{N}, mesh
) where {N}
    if sharding.type == OpShardingType.Replicated
        # Replicated: All devices have the entire array
        return (
            ntuple(Returns(ntuple(i -> 1:array_size[i], N)), length(mesh)),
            ntuple(Returns(nothing), N),
        )
    elseif sharding.type == OpShardingType.Maximal
        # Maximal: Only one device has the entire array
        @assert length(mesh) == 1
        return (
            ntuple(Returns(ntuple(i -> 1:array_size[i], N)), length(mesh)),
            ntuple(Returns(nothing), N),
        )
    elseif sharding.type == OpShardingType.Other
        # Other: Tiled sharding
        device_list = generate_device_list(sharding)
        sorted_mesh_devices = sort(collect(Int64, mesh.device_ids))
        @assert sort(device_list) == sorted_mesh_devices "Mismatched devices list: \
                                                          $(device_list) vs \
                                                          $(mesh.device_ids)"
        @assert isempty(sharding.tile_dimensions) "Tile dimensions are not supported yet! \
                                                   Open an issue with an MWE for this case."
        # Handle layout transformation
        dims_order = if !isempty(sharding.layout_minor_to_major)
            sharding.layout_minor_to_major
        else
            collect(1:length(sharding.tile_assignment_dimensions))
        end

        # Reshape considering column-major order and layout
        tile_assignment = reshape(
            device_list, reverse(sharding.tile_assignment_dimensions)...
        )

        # Apply layout transformation
        if !isempty(dims_order)
            tile_assignment = permutedims(tile_assignment, dims_order)
        end

        # Handle replication dimension
        tile_dims = size(tile_assignment)[(1 + sharding.replicate_on_last_tile_dim):end]

        # Calculate tile sizes
        tile_sizes = div.(array_size, tile_dims)
        mesh_devices = reshape([mesh.device_ids...], mesh.shape)

        # Match dimensions to mesh axes
        used_axes = Set{Int}()
        partition_spec = ntuple(N) do dim
            if dim <= length(tile_dims) && tile_dims[dim] > 1
                tile_seq = __get_device_sequence(
                    tile_assignment, dim + sharding.replicate_on_last_tile_dim
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

        device_to_array_indices = map(mesh.device_ids) do device_id
            tile_index = findfirst(==(device_id), tile_assignment)
            @assert tile_index !== nothing "Device ID $device_id not found in tile \
                                            assignment $tile_assignment"
            index_tup = if !sharding.replicate_on_last_tile_dim
                Tuple(tile_index.I)
            else
                Tuple(tile_index.I[2:end])
            end
            tile_start = (index_tup .- 1) .* tile_sizes .+ 1
            tile_end = index_tup .* tile_sizes
            return ntuple(i -> tile_start[i]:tile_end[i], N)
        end

        return device_to_array_indices, partition_spec
    else
        error("Unsupported sharding type: $(sharding.type)")
    end
end

# Helper function to get device sequence along a dimension
function __get_device_sequence(arr, dim)
    # Take first index for all other dimensions
    idx = ones(Int, ndims(arr))
    # Get sequence along target dimension
    sequence = Int[]
    for i in 1:size(arr, dim)
        idx[dim] = i
        push!(sequence, arr[idx...])
    end
    return sequence
end
