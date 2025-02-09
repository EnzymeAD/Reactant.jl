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
# struct JLOpSharding
#     type
# end

struct OpSharding{T}
    type::OpShardingType.T
    replicate_on_last_tile_dim::Bool
    last_tile_dims::Vector{OpShardingType.T}
    tile_assignment_dimensions::Vector{Int64}
    tile_assignment_devices::Vector{Int64}
    iota_reshape_dims::Vector{Int64}
    iota_transpose_perm::Vector{Int32}
    tuple_shardings::T
    is_shard_group::Bool
    shard_group_id::Int64
    shard_group_type::ShardGroupType.T
end

function get_output_shardings(exec::LoadedExecutable)
    exec.is_sharded || return OpSharding{Nothing}[]
end
