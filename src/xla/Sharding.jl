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

# https://github.com/openxla/xla/blob/8bac4a2c3c32144e39b1602450fe318bfab8e15d/xla/xla_data.proto#L897
# TODO: tuple sharding / op metadata
struct JLOpSharding
    type::Int32
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

function get_output_shardings(exec::LoadedExecutable)
    exec.is_sharded || return OpSharding[]

    jl_op_shardings = [Ref{JLOpSharding}() for _ in exec.num_results]
    GC.@preserve jl_op_shardings begin
        @ccall MLIR.API.mlir_c.PjRtLoadedExecutableGetOuputShardings(
            exec.exec::Ptr{Cvoid},
            Base.unsafe_convert(Ptr{Cvoid}, jl_op_shardings)::Ptr{Cvoid},
            exec.num_results::Int32,
        )::Cvoid
    end

    op_shardings = Vector{OpSharding}(undef, exec.num_results)
    for (i, jl_op_sharding) in enumerate(jl_op_shardings)
        last_tile_dims = Vector{Int32}(undef, jl_op_sharding.n_last_tile_dims)
        for j in 1:(jl_op_sharding.n_last_tile_dims)
            last_tile_dims[j] = unsafe_load(jl_op_sharding.last_tile_dims, j)
        end

        tile_assignment_dimensions = Vector{Int64}(
            undef, jl_op_sharding.n_tile_assignment_dimensions
        )
        for j in 1:(jl_op_sharding.n_tile_assignment_dimensions)
            tile_assignment_dimensions[j] = unsafe_load(
                jl_op_sharding.tile_assignment_dimensions, j
            )
        end

        tile_assignment_devices = Vector{Int64}(
            undef, jl_op_sharding.n_tile_assignment_devices
        )
        for j in 1:(jl_op_sharding.n_tile_assignment_devices)
            tile_assignment_devices[j] = unsafe_load(
                jl_op_sharding.tile_assignment_devices, j
            )
        end

        iota_reshape_dims = Vector{Int64}(undef, jl_op_sharding.n_iota_reshape_dims)
        for j in 1:(jl_op_sharding.n_iota_reshape_dims)
            iota_reshape_dims[j] = unsafe_load(jl_op_sharding.iota_reshape_dims, j)
        end

        iota_transpose_perm = Vector{Int32}(undef, jl_op_sharding.n_iota_transpose_perm)
        for j in 1:(jl_op_sharding.n_iota_transpose_perm)
            iota_transpose_perm[j] = unsafe_load(jl_op_sharding.iota_transpose_perm, j)
        end

        op_shardings[i] = OpSharding(
            int_to_op_sharding_type(jl_op_sharding.type),
            jl_op_sharding.replicate_on_last_tile_dim,
            last_tile_dims,
            tile_assignment_dimensions,
            tile_assignment_devices,
            iota_reshape_dims,
            iota_transpose_perm,
            jl_op_sharding.is_shard_group,
            jl_op_sharding.shard_group_id,
            int_to_shard_group_type(jl_op_sharding.shard_group_type),
        )
    end

    return op_shardings
end
