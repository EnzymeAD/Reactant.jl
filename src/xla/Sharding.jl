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
# TODO: tuple sharding
# TODO: op metadata
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

function get_output_shardings(exec::LoadedExecutable)
    exec.is_sharded || return OpSharding[]

    jl_op_shardings = [Ref{JLOpSharding}() for _ in exec.num_results]
    GC.@preserve jl_op_shardings begin
        @ccall MLIR.API.mlir_c.PjRtLoadedExecutableGetOuputShardings(
            exec.exec::Ptr{Cvoid},
            Base.unsafe_convert(Ptr{Cvoid}, jl_op_shardings)::Ptr{Cvoid},
            exec.num_results::Int32
        )::Cvoid
    end

    # TODO: copy the results into `OpSharding`
    return jl_op_shardings
end
