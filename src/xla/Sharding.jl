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

    tile_dimensions = unsafe_wrap(
        Array, sharding.tile_dimensions, sharding.n_tile_dimensions
    )
    layout_minor_to_major = unsafe_wrap(
        Array, sharding.layout_minor_to_major, sharding.n_layout_minor_to_major
    )

    return OpSharding(
        int_to_op_sharding_type(sharding.type),
        reverse(tile_dimensions),
        layout_minor_to_major,
        sharding.replicate_on_last_tile_dim,
        reverse(last_tile_dims),
        reverse(tile_assignment_dimensions),
        reverse(tile_assignment_devices),
        reverse(iota_reshape_dims),
        reverse(iota_transpose_perm),
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

function get_output_shardings(exec::LoadedExecutable)
    exec.is_sharded || return OpSharding[]

    jl_op_shardings = [Ref{JLOpSharding}() for _ in 1:(exec.num_results)]
    jl_op_shardings_ptr = [
        Base.unsafe_convert(Ptr{JLOpSharding}, sharding) for sharding in jl_op_shardings
    ]

    GC.@preserve jl_op_shardings begin
        @ccall MLIR.API.mlir_c.PjRtLoadedExecutableGetOuputShardings(
            exec.exec::Ptr{Cvoid},
            jl_op_shardings_ptr::Ptr{Ptr{JLOpSharding}},
            exec.num_results::Int32,
        )::Cvoid
    end

    return map(OpSharding ∘ getindex, jl_op_shardings)
end
