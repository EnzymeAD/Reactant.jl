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
    type::Cint
    n_tile_dimensions::Cint
    tile_dimensions::Ptr{Clong}
    n_layout_minor_to_major::Cint
    layout_minor_to_major::Ptr{Clong}
    replicate_on_last_tile_dim::Bool
    n_last_tile_dims::Cint
    last_tile_dims::Ptr{Cint}
    n_tile_assignment_dimensions::Cint
    tile_assignment_dimensions::Ptr{Clong}
    n_tile_assignment_devices::Cint
    tile_assignment_devices::Ptr{Clong}
    n_iota_reshape_dims::Cint
    iota_reshape_dims::Ptr{Clong}
    n_iota_transpose_perm::Cint
    iota_transpose_perm::Ptr{Cint}
    is_shard_group::Bool
    shard_group_id::Clong
    shard_group_type::Cint
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
