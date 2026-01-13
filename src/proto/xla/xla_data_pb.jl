import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export RandomDistribution, Statistic, var"TriangularSolveOptions.Transpose"
export var"WhileLoopBackendConfig.KnownInitStep", var"ResultAccuracy.Mode"
export GatherDimensionNumbers, var"DeviceAssignmentProto.ComputationDevice"
export SplitConfigProto, var"PrecisionConfig.Algorithm", RandomAlgorithm, DimLevelType
export var"WhileLoopBackendConfig.KnownTripCount", CollectiveOpGroupMode, PrimitiveType
export ParameterReplication, CompilationEvent, var"ChannelHandle.ChannelType", SortOptions
export ReplicaGroup, var"ResultAccuracy.Tolerance", TileProto, ScatterDimensionNumbers
export SourceTarget, ExecutionHandle, GlobalDataHandle, FftType, ProfileSource
export DotDimensionNumbers, DeviceHandle, var"OpSharding.Type", WindowDimension
export ConvolutionDimensionNumbers, IotaReplicaGroupListProto, OriginalArrayProto
export ComputationStats, FrontendAttributes, ProfileType, AsyncStreamKind
export var"WhileLoopBackendConfig.KnownInductionVariable"
export var"PaddingConfig.PaddingConfigDimension", GemmPerfTableEntry, OutputOperandAliasing
export var"PrecisionConfig.Precision", ExecutionProfile, var"AxisRefProto.SubAxis"
export ProfileGenerationStrategy, PaddingType, var"MeshProto.MeshAxis"
export var"OpSharding.ShardGroupType", CholeskyOptions, StatisticsViz
export TriangularSolveOptions, DeviceAssignmentProto, ChannelHandle, ResultAccuracy
export RaggedDotDimensionNumbers, Window, CollectiveDeviceListProto
export OriginalValueElementProto, WhileLoopBackendConfig, PaddingConfig
export GemmPerfTableEntryValues, PrecisionConfig, AxisRefProto, var"OpMetadata.ProfileInfo"
export MeshProto, OriginalValueProto, GemmPerfTable
export var"NamedShardingProto.DimensionSharding", OpMetadata, MeshAxesReplicaGroupListProto
export NamedShardingProto, LayoutProto, LiteralProto, OpSharding, ProgramShapeProto
export ShapeProto
abstract type var"##Abstract#LiteralProto" end
abstract type var"##Abstract#ShapeProto" end
abstract type var"##Abstract#OpSharding" end
abstract type var"##Abstract#LayoutProto" end
abstract type var"##Abstract#ProgramShapeProto" end


@enumx RandomDistribution RNG_INVALID=0 RNG_UNIFORM=1 RNG_NORMAL=2

struct Statistic
    stat_name::String
    stat_val::Float64
end
Statistic(;stat_name = "", stat_val = zero(Float64)) = Statistic(stat_name, stat_val)
PB.default_values(::Type{Statistic}) = (;stat_name = "", stat_val = zero(Float64))
PB.field_numbers(::Type{Statistic}) = (;stat_name = 1, stat_val = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Statistic})
    stat_name = ""
    stat_val = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            stat_name = PB.decode(d, String)
        elseif field_number == 2
            stat_val = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return Statistic(stat_name, stat_val)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Statistic)
    initpos = position(e.io)
    !isempty(x.stat_name) && PB.encode(e, 1, x.stat_name)
    x.stat_val !== zero(Float64) && PB.encode(e, 2, x.stat_val)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Statistic)
    encoded_size = 0
    !isempty(x.stat_name) && (encoded_size += PB._encoded_size(x.stat_name, 1))
    x.stat_val !== zero(Float64) && (encoded_size += PB._encoded_size(x.stat_val, 2))
    return encoded_size
end

@enumx var"TriangularSolveOptions.Transpose" TRANSPOSE_INVALID=0 NO_TRANSPOSE=1 TRANSPOSE=2 ADJOINT=3

struct var"WhileLoopBackendConfig.KnownInitStep"
    init::Int64
    step::Int64
end
var"WhileLoopBackendConfig.KnownInitStep"(;init = zero(Int64), step = zero(Int64)) = var"WhileLoopBackendConfig.KnownInitStep"(init, step)
PB.default_values(::Type{var"WhileLoopBackendConfig.KnownInitStep"}) = (;init = zero(Int64), step = zero(Int64))
PB.field_numbers(::Type{var"WhileLoopBackendConfig.KnownInitStep"}) = (;init = 1, step = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"WhileLoopBackendConfig.KnownInitStep"})
    init = zero(Int64)
    step = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            init = PB.decode(d, Int64)
        elseif field_number == 2
            step = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"WhileLoopBackendConfig.KnownInitStep"(init, step)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"WhileLoopBackendConfig.KnownInitStep")
    initpos = position(e.io)
    x.init != zero(Int64) && PB.encode(e, 1, x.init)
    x.step != zero(Int64) && PB.encode(e, 2, x.step)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"WhileLoopBackendConfig.KnownInitStep")
    encoded_size = 0
    x.init != zero(Int64) && (encoded_size += PB._encoded_size(x.init, 1))
    x.step != zero(Int64) && (encoded_size += PB._encoded_size(x.step, 2))
    return encoded_size
end

@enumx var"ResultAccuracy.Mode" DEFAULT=0 HIGHEST=1

struct GatherDimensionNumbers
    offset_dims::Vector{Int64}
    collapsed_slice_dims::Vector{Int64}
    start_index_map::Vector{Int64}
    index_vector_dim::Int64
    operand_batching_dims::Vector{Int64}
    start_indices_batching_dims::Vector{Int64}
end
GatherDimensionNumbers(;offset_dims = Vector{Int64}(), collapsed_slice_dims = Vector{Int64}(), start_index_map = Vector{Int64}(), index_vector_dim = zero(Int64), operand_batching_dims = Vector{Int64}(), start_indices_batching_dims = Vector{Int64}()) = GatherDimensionNumbers(offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim, operand_batching_dims, start_indices_batching_dims)
PB.default_values(::Type{GatherDimensionNumbers}) = (;offset_dims = Vector{Int64}(), collapsed_slice_dims = Vector{Int64}(), start_index_map = Vector{Int64}(), index_vector_dim = zero(Int64), operand_batching_dims = Vector{Int64}(), start_indices_batching_dims = Vector{Int64}())
PB.field_numbers(::Type{GatherDimensionNumbers}) = (;offset_dims = 1, collapsed_slice_dims = 2, start_index_map = 3, index_vector_dim = 4, operand_batching_dims = 5, start_indices_batching_dims = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GatherDimensionNumbers})
    offset_dims = PB.BufferedVector{Int64}()
    collapsed_slice_dims = PB.BufferedVector{Int64}()
    start_index_map = PB.BufferedVector{Int64}()
    index_vector_dim = zero(Int64)
    operand_batching_dims = PB.BufferedVector{Int64}()
    start_indices_batching_dims = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, offset_dims)
        elseif field_number == 2
            PB.decode!(d, wire_type, collapsed_slice_dims)
        elseif field_number == 3
            PB.decode!(d, wire_type, start_index_map)
        elseif field_number == 4
            index_vector_dim = PB.decode(d, Int64)
        elseif field_number == 5
            PB.decode!(d, wire_type, operand_batching_dims)
        elseif field_number == 6
            PB.decode!(d, wire_type, start_indices_batching_dims)
        else
            Base.skip(d, wire_type)
        end
    end
    return GatherDimensionNumbers(offset_dims[], collapsed_slice_dims[], start_index_map[], index_vector_dim, operand_batching_dims[], start_indices_batching_dims[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GatherDimensionNumbers)
    initpos = position(e.io)
    !isempty(x.offset_dims) && PB.encode(e, 1, x.offset_dims)
    !isempty(x.collapsed_slice_dims) && PB.encode(e, 2, x.collapsed_slice_dims)
    !isempty(x.start_index_map) && PB.encode(e, 3, x.start_index_map)
    x.index_vector_dim != zero(Int64) && PB.encode(e, 4, x.index_vector_dim)
    !isempty(x.operand_batching_dims) && PB.encode(e, 5, x.operand_batching_dims)
    !isempty(x.start_indices_batching_dims) && PB.encode(e, 6, x.start_indices_batching_dims)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GatherDimensionNumbers)
    encoded_size = 0
    !isempty(x.offset_dims) && (encoded_size += PB._encoded_size(x.offset_dims, 1))
    !isempty(x.collapsed_slice_dims) && (encoded_size += PB._encoded_size(x.collapsed_slice_dims, 2))
    !isempty(x.start_index_map) && (encoded_size += PB._encoded_size(x.start_index_map, 3))
    x.index_vector_dim != zero(Int64) && (encoded_size += PB._encoded_size(x.index_vector_dim, 4))
    !isempty(x.operand_batching_dims) && (encoded_size += PB._encoded_size(x.operand_batching_dims, 5))
    !isempty(x.start_indices_batching_dims) && (encoded_size += PB._encoded_size(x.start_indices_batching_dims, 6))
    return encoded_size
end

struct var"DeviceAssignmentProto.ComputationDevice"
    replica_device_ids::Vector{Int64}
end
var"DeviceAssignmentProto.ComputationDevice"(;replica_device_ids = Vector{Int64}()) = var"DeviceAssignmentProto.ComputationDevice"(replica_device_ids)
PB.default_values(::Type{var"DeviceAssignmentProto.ComputationDevice"}) = (;replica_device_ids = Vector{Int64}())
PB.field_numbers(::Type{var"DeviceAssignmentProto.ComputationDevice"}) = (;replica_device_ids = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"DeviceAssignmentProto.ComputationDevice"})
    replica_device_ids = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, replica_device_ids)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"DeviceAssignmentProto.ComputationDevice"(replica_device_ids[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"DeviceAssignmentProto.ComputationDevice")
    initpos = position(e.io)
    !isempty(x.replica_device_ids) && PB.encode(e, 1, x.replica_device_ids)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"DeviceAssignmentProto.ComputationDevice")
    encoded_size = 0
    !isempty(x.replica_device_ids) && (encoded_size += PB._encoded_size(x.replica_device_ids, 1))
    return encoded_size
end

struct SplitConfigProto
    dimension::Int64
    split_indices::Vector{Int64}
end
SplitConfigProto(;dimension = zero(Int64), split_indices = Vector{Int64}()) = SplitConfigProto(dimension, split_indices)
PB.default_values(::Type{SplitConfigProto}) = (;dimension = zero(Int64), split_indices = Vector{Int64}())
PB.field_numbers(::Type{SplitConfigProto}) = (;dimension = 1, split_indices = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SplitConfigProto})
    dimension = zero(Int64)
    split_indices = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            dimension = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, wire_type, split_indices)
        else
            Base.skip(d, wire_type)
        end
    end
    return SplitConfigProto(dimension, split_indices[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SplitConfigProto)
    initpos = position(e.io)
    x.dimension != zero(Int64) && PB.encode(e, 1, x.dimension)
    !isempty(x.split_indices) && PB.encode(e, 2, x.split_indices)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SplitConfigProto)
    encoded_size = 0
    x.dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.dimension, 1))
    !isempty(x.split_indices) && (encoded_size += PB._encoded_size(x.split_indices, 2))
    return encoded_size
end

@enumx var"PrecisionConfig.Algorithm" ALG_UNSET=0 ALG_DOT_ANY_F8_ANY_F8_F32=1 ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM=2 ALG_DOT_F16_F16_F16=3 ALG_DOT_F16_F16_F32=4 ALG_DOT_BF16_BF16_BF16=5 ALG_DOT_BF16_BF16_F32=6 ALG_DOT_BF16_BF16_F32_X3=7 ALG_DOT_BF16_BF16_F32_X6=8 ALG_DOT_TF32_TF32_F32=9 ALG_DOT_TF32_TF32_F32_X3=10 ALG_DOT_F32_F32_F32=11 ALG_DOT_F64_F64_F64=12 ALG_DOT_BF16_BF16_F32_X9=13

@enumx RandomAlgorithm RNG_DEFAULT=0 RNG_THREE_FRY=1 RNG_PHILOX=2

@enumx DimLevelType DIM_DENSE=0 DIM_COMPRESSED=1 DIM_SINGLETON=2 DIM_LOOSE_COMPRESSED=3

struct var"WhileLoopBackendConfig.KnownTripCount"
    n::Int64
end
var"WhileLoopBackendConfig.KnownTripCount"(;n = zero(Int64)) = var"WhileLoopBackendConfig.KnownTripCount"(n)
PB.default_values(::Type{var"WhileLoopBackendConfig.KnownTripCount"}) = (;n = zero(Int64))
PB.field_numbers(::Type{var"WhileLoopBackendConfig.KnownTripCount"}) = (;n = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"WhileLoopBackendConfig.KnownTripCount"})
    n = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            n = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"WhileLoopBackendConfig.KnownTripCount"(n)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"WhileLoopBackendConfig.KnownTripCount")
    initpos = position(e.io)
    x.n != zero(Int64) && PB.encode(e, 1, x.n)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"WhileLoopBackendConfig.KnownTripCount")
    encoded_size = 0
    x.n != zero(Int64) && (encoded_size += PB._encoded_size(x.n, 1))
    return encoded_size
end

@enumx CollectiveOpGroupMode COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA=0 COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION=1 COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION=2 COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID=3

@enumx PrimitiveType PRIMITIVE_TYPE_INVALID=0 PRED=1 S1=30 S2=26 S4=21 S8=2 S16=3 S32=4 S64=5 U1=31 U2=27 U4=22 U8=6 U16=7 U32=8 U64=9 F16=10 F32=11 BF16=16 F64=12 F8E5M2=19 F8E4M3=28 F8E4M3FN=20 F8E4M3B11FNUZ=23 F8E3M4=29 F8E5M2FNUZ=24 F8E4M3FNUZ=25 F4E2M1FN=32 F8E8M0FNU=33 C64=15 C128=18 TUPLE=13 OPAQUE_TYPE=14 TOKEN=17 BUFFER=34

struct ParameterReplication
    replicated_at_leaf_buffers::Vector{Bool}
end
ParameterReplication(;replicated_at_leaf_buffers = Vector{Bool}()) = ParameterReplication(replicated_at_leaf_buffers)
PB.default_values(::Type{ParameterReplication}) = (;replicated_at_leaf_buffers = Vector{Bool}())
PB.field_numbers(::Type{ParameterReplication}) = (;replicated_at_leaf_buffers = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ParameterReplication})
    replicated_at_leaf_buffers = PB.BufferedVector{Bool}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, replicated_at_leaf_buffers)
        else
            Base.skip(d, wire_type)
        end
    end
    return ParameterReplication(replicated_at_leaf_buffers[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ParameterReplication)
    initpos = position(e.io)
    !isempty(x.replicated_at_leaf_buffers) && PB.encode(e, 1, x.replicated_at_leaf_buffers)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ParameterReplication)
    encoded_size = 0
    !isempty(x.replicated_at_leaf_buffers) && (encoded_size += PB._encoded_size(x.replicated_at_leaf_buffers, 1))
    return encoded_size
end

@enumx CompilationEvent COMPILATION_EVENT_UNKNOWN_EVENT=0 COMPILATION_EVENT_FIRST_COMPILATION=1 COMPILATION_EVENT_RECOMPILATION=2

@enumx var"ChannelHandle.ChannelType" CHANNEL_TYPE_INVALID=0 DEVICE_TO_DEVICE=1 DEVICE_TO_HOST=2 HOST_TO_DEVICE=3

struct SortOptions
    descending::Bool
end
SortOptions(;descending = false) = SortOptions(descending)
PB.default_values(::Type{SortOptions}) = (;descending = false)
PB.field_numbers(::Type{SortOptions}) = (;descending = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SortOptions})
    descending = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            descending = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return SortOptions(descending)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SortOptions)
    initpos = position(e.io)
    x.descending != false && PB.encode(e, 1, x.descending)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SortOptions)
    encoded_size = 0
    x.descending != false && (encoded_size += PB._encoded_size(x.descending, 1))
    return encoded_size
end

struct ReplicaGroup
    replica_ids::Vector{Int64}
end
ReplicaGroup(;replica_ids = Vector{Int64}()) = ReplicaGroup(replica_ids)
PB.default_values(::Type{ReplicaGroup}) = (;replica_ids = Vector{Int64}())
PB.field_numbers(::Type{ReplicaGroup}) = (;replica_ids = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ReplicaGroup})
    replica_ids = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, replica_ids)
        else
            Base.skip(d, wire_type)
        end
    end
    return ReplicaGroup(replica_ids[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ReplicaGroup)
    initpos = position(e.io)
    !isempty(x.replica_ids) && PB.encode(e, 1, x.replica_ids)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ReplicaGroup)
    encoded_size = 0
    !isempty(x.replica_ids) && (encoded_size += PB._encoded_size(x.replica_ids, 1))
    return encoded_size
end

struct var"ResultAccuracy.Tolerance"
    atol::Float64
    rtol::Float64
    ulps::Int64
end
var"ResultAccuracy.Tolerance"(;atol = zero(Float64), rtol = zero(Float64), ulps = zero(Int64)) = var"ResultAccuracy.Tolerance"(atol, rtol, ulps)
PB.default_values(::Type{var"ResultAccuracy.Tolerance"}) = (;atol = zero(Float64), rtol = zero(Float64), ulps = zero(Int64))
PB.field_numbers(::Type{var"ResultAccuracy.Tolerance"}) = (;atol = 1, rtol = 2, ulps = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"ResultAccuracy.Tolerance"})
    atol = zero(Float64)
    rtol = zero(Float64)
    ulps = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            atol = PB.decode(d, Float64)
        elseif field_number == 2
            rtol = PB.decode(d, Float64)
        elseif field_number == 3
            ulps = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"ResultAccuracy.Tolerance"(atol, rtol, ulps)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"ResultAccuracy.Tolerance")
    initpos = position(e.io)
    x.atol !== zero(Float64) && PB.encode(e, 1, x.atol)
    x.rtol !== zero(Float64) && PB.encode(e, 2, x.rtol)
    x.ulps != zero(Int64) && PB.encode(e, 3, x.ulps)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"ResultAccuracy.Tolerance")
    encoded_size = 0
    x.atol !== zero(Float64) && (encoded_size += PB._encoded_size(x.atol, 1))
    x.rtol !== zero(Float64) && (encoded_size += PB._encoded_size(x.rtol, 2))
    x.ulps != zero(Int64) && (encoded_size += PB._encoded_size(x.ulps, 3))
    return encoded_size
end

struct TileProto
    dimensions::Vector{Int64}
end
TileProto(;dimensions = Vector{Int64}()) = TileProto(dimensions)
PB.default_values(::Type{TileProto}) = (;dimensions = Vector{Int64}())
PB.field_numbers(::Type{TileProto}) = (;dimensions = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TileProto})
    dimensions = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, dimensions)
        else
            Base.skip(d, wire_type)
        end
    end
    return TileProto(dimensions[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TileProto)
    initpos = position(e.io)
    !isempty(x.dimensions) && PB.encode(e, 1, x.dimensions)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TileProto)
    encoded_size = 0
    !isempty(x.dimensions) && (encoded_size += PB._encoded_size(x.dimensions, 1))
    return encoded_size
end

struct ScatterDimensionNumbers
    update_window_dims::Vector{Int64}
    inserted_window_dims::Vector{Int64}
    scatter_dims_to_operand_dims::Vector{Int64}
    index_vector_dim::Int64
    input_batching_dims::Vector{Int64}
    scatter_indices_batching_dims::Vector{Int64}
end
ScatterDimensionNumbers(;update_window_dims = Vector{Int64}(), inserted_window_dims = Vector{Int64}(), scatter_dims_to_operand_dims = Vector{Int64}(), index_vector_dim = zero(Int64), input_batching_dims = Vector{Int64}(), scatter_indices_batching_dims = Vector{Int64}()) = ScatterDimensionNumbers(update_window_dims, inserted_window_dims, scatter_dims_to_operand_dims, index_vector_dim, input_batching_dims, scatter_indices_batching_dims)
PB.default_values(::Type{ScatterDimensionNumbers}) = (;update_window_dims = Vector{Int64}(), inserted_window_dims = Vector{Int64}(), scatter_dims_to_operand_dims = Vector{Int64}(), index_vector_dim = zero(Int64), input_batching_dims = Vector{Int64}(), scatter_indices_batching_dims = Vector{Int64}())
PB.field_numbers(::Type{ScatterDimensionNumbers}) = (;update_window_dims = 1, inserted_window_dims = 2, scatter_dims_to_operand_dims = 3, index_vector_dim = 4, input_batching_dims = 5, scatter_indices_batching_dims = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ScatterDimensionNumbers})
    update_window_dims = PB.BufferedVector{Int64}()
    inserted_window_dims = PB.BufferedVector{Int64}()
    scatter_dims_to_operand_dims = PB.BufferedVector{Int64}()
    index_vector_dim = zero(Int64)
    input_batching_dims = PB.BufferedVector{Int64}()
    scatter_indices_batching_dims = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, update_window_dims)
        elseif field_number == 2
            PB.decode!(d, wire_type, inserted_window_dims)
        elseif field_number == 3
            PB.decode!(d, wire_type, scatter_dims_to_operand_dims)
        elseif field_number == 4
            index_vector_dim = PB.decode(d, Int64)
        elseif field_number == 5
            PB.decode!(d, wire_type, input_batching_dims)
        elseif field_number == 6
            PB.decode!(d, wire_type, scatter_indices_batching_dims)
        else
            Base.skip(d, wire_type)
        end
    end
    return ScatterDimensionNumbers(update_window_dims[], inserted_window_dims[], scatter_dims_to_operand_dims[], index_vector_dim, input_batching_dims[], scatter_indices_batching_dims[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ScatterDimensionNumbers)
    initpos = position(e.io)
    !isempty(x.update_window_dims) && PB.encode(e, 1, x.update_window_dims)
    !isempty(x.inserted_window_dims) && PB.encode(e, 2, x.inserted_window_dims)
    !isempty(x.scatter_dims_to_operand_dims) && PB.encode(e, 3, x.scatter_dims_to_operand_dims)
    x.index_vector_dim != zero(Int64) && PB.encode(e, 4, x.index_vector_dim)
    !isempty(x.input_batching_dims) && PB.encode(e, 5, x.input_batching_dims)
    !isempty(x.scatter_indices_batching_dims) && PB.encode(e, 6, x.scatter_indices_batching_dims)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ScatterDimensionNumbers)
    encoded_size = 0
    !isempty(x.update_window_dims) && (encoded_size += PB._encoded_size(x.update_window_dims, 1))
    !isempty(x.inserted_window_dims) && (encoded_size += PB._encoded_size(x.inserted_window_dims, 2))
    !isempty(x.scatter_dims_to_operand_dims) && (encoded_size += PB._encoded_size(x.scatter_dims_to_operand_dims, 3))
    x.index_vector_dim != zero(Int64) && (encoded_size += PB._encoded_size(x.index_vector_dim, 4))
    !isempty(x.input_batching_dims) && (encoded_size += PB._encoded_size(x.input_batching_dims, 5))
    !isempty(x.scatter_indices_batching_dims) && (encoded_size += PB._encoded_size(x.scatter_indices_batching_dims, 6))
    return encoded_size
end

struct SourceTarget
    source::Int64
    target::Int64
end
SourceTarget(;source = zero(Int64), target = zero(Int64)) = SourceTarget(source, target)
PB.default_values(::Type{SourceTarget}) = (;source = zero(Int64), target = zero(Int64))
PB.field_numbers(::Type{SourceTarget}) = (;source = 1, target = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SourceTarget})
    source = zero(Int64)
    target = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            source = PB.decode(d, Int64)
        elseif field_number == 2
            target = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return SourceTarget(source, target)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SourceTarget)
    initpos = position(e.io)
    x.source != zero(Int64) && PB.encode(e, 1, x.source)
    x.target != zero(Int64) && PB.encode(e, 2, x.target)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SourceTarget)
    encoded_size = 0
    x.source != zero(Int64) && (encoded_size += PB._encoded_size(x.source, 1))
    x.target != zero(Int64) && (encoded_size += PB._encoded_size(x.target, 2))
    return encoded_size
end

struct ExecutionHandle
    handle::Int64
end
ExecutionHandle(;handle = zero(Int64)) = ExecutionHandle(handle)
PB.default_values(::Type{ExecutionHandle}) = (;handle = zero(Int64))
PB.field_numbers(::Type{ExecutionHandle}) = (;handle = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ExecutionHandle})
    handle = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            handle = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return ExecutionHandle(handle)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ExecutionHandle)
    initpos = position(e.io)
    x.handle != zero(Int64) && PB.encode(e, 1, x.handle)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ExecutionHandle)
    encoded_size = 0
    x.handle != zero(Int64) && (encoded_size += PB._encoded_size(x.handle, 1))
    return encoded_size
end

struct GlobalDataHandle
    handle::Int64
end
GlobalDataHandle(;handle = zero(Int64)) = GlobalDataHandle(handle)
PB.default_values(::Type{GlobalDataHandle}) = (;handle = zero(Int64))
PB.field_numbers(::Type{GlobalDataHandle}) = (;handle = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GlobalDataHandle})
    handle = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            handle = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return GlobalDataHandle(handle)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GlobalDataHandle)
    initpos = position(e.io)
    x.handle != zero(Int64) && PB.encode(e, 1, x.handle)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GlobalDataHandle)
    encoded_size = 0
    x.handle != zero(Int64) && (encoded_size += PB._encoded_size(x.handle, 1))
    return encoded_size
end

@enumx FftType FFT=0 IFFT=1 RFFT=2 IRFFT=3

@enumx ProfileSource PROFILE_SOURCE_UNKNOWN_SOURCE=0 PROFILE_SOURCE_EMBEDDED=1 PROFILE_SOURCE_REMOTE=2

struct DotDimensionNumbers
    lhs_contracting_dimensions::Vector{Int64}
    rhs_contracting_dimensions::Vector{Int64}
    lhs_batch_dimensions::Vector{Int64}
    rhs_batch_dimensions::Vector{Int64}
end
DotDimensionNumbers(;lhs_contracting_dimensions = Vector{Int64}(), rhs_contracting_dimensions = Vector{Int64}(), lhs_batch_dimensions = Vector{Int64}(), rhs_batch_dimensions = Vector{Int64}()) = DotDimensionNumbers(lhs_contracting_dimensions, rhs_contracting_dimensions, lhs_batch_dimensions, rhs_batch_dimensions)
PB.default_values(::Type{DotDimensionNumbers}) = (;lhs_contracting_dimensions = Vector{Int64}(), rhs_contracting_dimensions = Vector{Int64}(), lhs_batch_dimensions = Vector{Int64}(), rhs_batch_dimensions = Vector{Int64}())
PB.field_numbers(::Type{DotDimensionNumbers}) = (;lhs_contracting_dimensions = 1, rhs_contracting_dimensions = 2, lhs_batch_dimensions = 3, rhs_batch_dimensions = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:DotDimensionNumbers})
    lhs_contracting_dimensions = PB.BufferedVector{Int64}()
    rhs_contracting_dimensions = PB.BufferedVector{Int64}()
    lhs_batch_dimensions = PB.BufferedVector{Int64}()
    rhs_batch_dimensions = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, lhs_contracting_dimensions)
        elseif field_number == 2
            PB.decode!(d, wire_type, rhs_contracting_dimensions)
        elseif field_number == 3
            PB.decode!(d, wire_type, lhs_batch_dimensions)
        elseif field_number == 4
            PB.decode!(d, wire_type, rhs_batch_dimensions)
        else
            Base.skip(d, wire_type)
        end
    end
    return DotDimensionNumbers(lhs_contracting_dimensions[], rhs_contracting_dimensions[], lhs_batch_dimensions[], rhs_batch_dimensions[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::DotDimensionNumbers)
    initpos = position(e.io)
    !isempty(x.lhs_contracting_dimensions) && PB.encode(e, 1, x.lhs_contracting_dimensions)
    !isempty(x.rhs_contracting_dimensions) && PB.encode(e, 2, x.rhs_contracting_dimensions)
    !isempty(x.lhs_batch_dimensions) && PB.encode(e, 3, x.lhs_batch_dimensions)
    !isempty(x.rhs_batch_dimensions) && PB.encode(e, 4, x.rhs_batch_dimensions)
    return position(e.io) - initpos
end
function PB._encoded_size(x::DotDimensionNumbers)
    encoded_size = 0
    !isempty(x.lhs_contracting_dimensions) && (encoded_size += PB._encoded_size(x.lhs_contracting_dimensions, 1))
    !isempty(x.rhs_contracting_dimensions) && (encoded_size += PB._encoded_size(x.rhs_contracting_dimensions, 2))
    !isempty(x.lhs_batch_dimensions) && (encoded_size += PB._encoded_size(x.lhs_batch_dimensions, 3))
    !isempty(x.rhs_batch_dimensions) && (encoded_size += PB._encoded_size(x.rhs_batch_dimensions, 4))
    return encoded_size
end

struct DeviceHandle
    handle::Int64
    device_count::Int64
end
DeviceHandle(;handle = zero(Int64), device_count = zero(Int64)) = DeviceHandle(handle, device_count)
PB.default_values(::Type{DeviceHandle}) = (;handle = zero(Int64), device_count = zero(Int64))
PB.field_numbers(::Type{DeviceHandle}) = (;handle = 1, device_count = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:DeviceHandle})
    handle = zero(Int64)
    device_count = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            handle = PB.decode(d, Int64)
        elseif field_number == 2
            device_count = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return DeviceHandle(handle, device_count)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::DeviceHandle)
    initpos = position(e.io)
    x.handle != zero(Int64) && PB.encode(e, 1, x.handle)
    x.device_count != zero(Int64) && PB.encode(e, 2, x.device_count)
    return position(e.io) - initpos
end
function PB._encoded_size(x::DeviceHandle)
    encoded_size = 0
    x.handle != zero(Int64) && (encoded_size += PB._encoded_size(x.handle, 1))
    x.device_count != zero(Int64) && (encoded_size += PB._encoded_size(x.device_count, 2))
    return encoded_size
end

@enumx var"OpSharding.Type" REPLICATED=0 MAXIMAL=1 TUPLE=2 OTHER=3 MANUAL=4 UNKNOWN=5 UNREDUCED=6

struct WindowDimension
    size::Int64
    stride::Int64
    padding_low::Int64
    padding_high::Int64
    window_dilation::Int64
    base_dilation::Int64
    window_reversal::Bool
end
WindowDimension(;size = zero(Int64), stride = zero(Int64), padding_low = zero(Int64), padding_high = zero(Int64), window_dilation = zero(Int64), base_dilation = zero(Int64), window_reversal = false) = WindowDimension(size, stride, padding_low, padding_high, window_dilation, base_dilation, window_reversal)
PB.default_values(::Type{WindowDimension}) = (;size = zero(Int64), stride = zero(Int64), padding_low = zero(Int64), padding_high = zero(Int64), window_dilation = zero(Int64), base_dilation = zero(Int64), window_reversal = false)
PB.field_numbers(::Type{WindowDimension}) = (;size = 1, stride = 2, padding_low = 3, padding_high = 4, window_dilation = 5, base_dilation = 6, window_reversal = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:WindowDimension})
    size = zero(Int64)
    stride = zero(Int64)
    padding_low = zero(Int64)
    padding_high = zero(Int64)
    window_dilation = zero(Int64)
    base_dilation = zero(Int64)
    window_reversal = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            size = PB.decode(d, Int64)
        elseif field_number == 2
            stride = PB.decode(d, Int64)
        elseif field_number == 3
            padding_low = PB.decode(d, Int64)
        elseif field_number == 4
            padding_high = PB.decode(d, Int64)
        elseif field_number == 5
            window_dilation = PB.decode(d, Int64)
        elseif field_number == 6
            base_dilation = PB.decode(d, Int64)
        elseif field_number == 7
            window_reversal = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return WindowDimension(size, stride, padding_low, padding_high, window_dilation, base_dilation, window_reversal)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::WindowDimension)
    initpos = position(e.io)
    x.size != zero(Int64) && PB.encode(e, 1, x.size)
    x.stride != zero(Int64) && PB.encode(e, 2, x.stride)
    x.padding_low != zero(Int64) && PB.encode(e, 3, x.padding_low)
    x.padding_high != zero(Int64) && PB.encode(e, 4, x.padding_high)
    x.window_dilation != zero(Int64) && PB.encode(e, 5, x.window_dilation)
    x.base_dilation != zero(Int64) && PB.encode(e, 6, x.base_dilation)
    x.window_reversal != false && PB.encode(e, 7, x.window_reversal)
    return position(e.io) - initpos
end
function PB._encoded_size(x::WindowDimension)
    encoded_size = 0
    x.size != zero(Int64) && (encoded_size += PB._encoded_size(x.size, 1))
    x.stride != zero(Int64) && (encoded_size += PB._encoded_size(x.stride, 2))
    x.padding_low != zero(Int64) && (encoded_size += PB._encoded_size(x.padding_low, 3))
    x.padding_high != zero(Int64) && (encoded_size += PB._encoded_size(x.padding_high, 4))
    x.window_dilation != zero(Int64) && (encoded_size += PB._encoded_size(x.window_dilation, 5))
    x.base_dilation != zero(Int64) && (encoded_size += PB._encoded_size(x.base_dilation, 6))
    x.window_reversal != false && (encoded_size += PB._encoded_size(x.window_reversal, 7))
    return encoded_size
end

struct ConvolutionDimensionNumbers
    input_batch_dimension::Int64
    input_feature_dimension::Int64
    input_spatial_dimensions::Vector{Int64}
    kernel_input_feature_dimension::Int64
    kernel_output_feature_dimension::Int64
    kernel_spatial_dimensions::Vector{Int64}
    output_batch_dimension::Int64
    output_feature_dimension::Int64
    output_spatial_dimensions::Vector{Int64}
end
ConvolutionDimensionNumbers(;input_batch_dimension = zero(Int64), input_feature_dimension = zero(Int64), input_spatial_dimensions = Vector{Int64}(), kernel_input_feature_dimension = zero(Int64), kernel_output_feature_dimension = zero(Int64), kernel_spatial_dimensions = Vector{Int64}(), output_batch_dimension = zero(Int64), output_feature_dimension = zero(Int64), output_spatial_dimensions = Vector{Int64}()) = ConvolutionDimensionNumbers(input_batch_dimension, input_feature_dimension, input_spatial_dimensions, kernel_input_feature_dimension, kernel_output_feature_dimension, kernel_spatial_dimensions, output_batch_dimension, output_feature_dimension, output_spatial_dimensions)
PB.default_values(::Type{ConvolutionDimensionNumbers}) = (;input_batch_dimension = zero(Int64), input_feature_dimension = zero(Int64), input_spatial_dimensions = Vector{Int64}(), kernel_input_feature_dimension = zero(Int64), kernel_output_feature_dimension = zero(Int64), kernel_spatial_dimensions = Vector{Int64}(), output_batch_dimension = zero(Int64), output_feature_dimension = zero(Int64), output_spatial_dimensions = Vector{Int64}())
PB.field_numbers(::Type{ConvolutionDimensionNumbers}) = (;input_batch_dimension = 7, input_feature_dimension = 8, input_spatial_dimensions = 11, kernel_input_feature_dimension = 3, kernel_output_feature_dimension = 4, kernel_spatial_dimensions = 6, output_batch_dimension = 9, output_feature_dimension = 10, output_spatial_dimensions = 12)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ConvolutionDimensionNumbers})
    input_batch_dimension = zero(Int64)
    input_feature_dimension = zero(Int64)
    input_spatial_dimensions = PB.BufferedVector{Int64}()
    kernel_input_feature_dimension = zero(Int64)
    kernel_output_feature_dimension = zero(Int64)
    kernel_spatial_dimensions = PB.BufferedVector{Int64}()
    output_batch_dimension = zero(Int64)
    output_feature_dimension = zero(Int64)
    output_spatial_dimensions = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 7
            input_batch_dimension = PB.decode(d, Int64)
        elseif field_number == 8
            input_feature_dimension = PB.decode(d, Int64)
        elseif field_number == 11
            PB.decode!(d, wire_type, input_spatial_dimensions)
        elseif field_number == 3
            kernel_input_feature_dimension = PB.decode(d, Int64)
        elseif field_number == 4
            kernel_output_feature_dimension = PB.decode(d, Int64)
        elseif field_number == 6
            PB.decode!(d, wire_type, kernel_spatial_dimensions)
        elseif field_number == 9
            output_batch_dimension = PB.decode(d, Int64)
        elseif field_number == 10
            output_feature_dimension = PB.decode(d, Int64)
        elseif field_number == 12
            PB.decode!(d, wire_type, output_spatial_dimensions)
        else
            Base.skip(d, wire_type)
        end
    end
    return ConvolutionDimensionNumbers(input_batch_dimension, input_feature_dimension, input_spatial_dimensions[], kernel_input_feature_dimension, kernel_output_feature_dimension, kernel_spatial_dimensions[], output_batch_dimension, output_feature_dimension, output_spatial_dimensions[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ConvolutionDimensionNumbers)
    initpos = position(e.io)
    x.input_batch_dimension != zero(Int64) && PB.encode(e, 7, x.input_batch_dimension)
    x.input_feature_dimension != zero(Int64) && PB.encode(e, 8, x.input_feature_dimension)
    !isempty(x.input_spatial_dimensions) && PB.encode(e, 11, x.input_spatial_dimensions)
    x.kernel_input_feature_dimension != zero(Int64) && PB.encode(e, 3, x.kernel_input_feature_dimension)
    x.kernel_output_feature_dimension != zero(Int64) && PB.encode(e, 4, x.kernel_output_feature_dimension)
    !isempty(x.kernel_spatial_dimensions) && PB.encode(e, 6, x.kernel_spatial_dimensions)
    x.output_batch_dimension != zero(Int64) && PB.encode(e, 9, x.output_batch_dimension)
    x.output_feature_dimension != zero(Int64) && PB.encode(e, 10, x.output_feature_dimension)
    !isempty(x.output_spatial_dimensions) && PB.encode(e, 12, x.output_spatial_dimensions)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ConvolutionDimensionNumbers)
    encoded_size = 0
    x.input_batch_dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.input_batch_dimension, 7))
    x.input_feature_dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.input_feature_dimension, 8))
    !isempty(x.input_spatial_dimensions) && (encoded_size += PB._encoded_size(x.input_spatial_dimensions, 11))
    x.kernel_input_feature_dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.kernel_input_feature_dimension, 3))
    x.kernel_output_feature_dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.kernel_output_feature_dimension, 4))
    !isempty(x.kernel_spatial_dimensions) && (encoded_size += PB._encoded_size(x.kernel_spatial_dimensions, 6))
    x.output_batch_dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.output_batch_dimension, 9))
    x.output_feature_dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.output_feature_dimension, 10))
    !isempty(x.output_spatial_dimensions) && (encoded_size += PB._encoded_size(x.output_spatial_dimensions, 12))
    return encoded_size
end

struct IotaReplicaGroupListProto
    num_replica_groups::Int64
    num_devices_per_group::Int64
    iota_reshape_dims::Vector{Int64}
    iota_transpose_perm::Vector{Int32}
end
IotaReplicaGroupListProto(;num_replica_groups = zero(Int64), num_devices_per_group = zero(Int64), iota_reshape_dims = Vector{Int64}(), iota_transpose_perm = Vector{Int32}()) = IotaReplicaGroupListProto(num_replica_groups, num_devices_per_group, iota_reshape_dims, iota_transpose_perm)
PB.default_values(::Type{IotaReplicaGroupListProto}) = (;num_replica_groups = zero(Int64), num_devices_per_group = zero(Int64), iota_reshape_dims = Vector{Int64}(), iota_transpose_perm = Vector{Int32}())
PB.field_numbers(::Type{IotaReplicaGroupListProto}) = (;num_replica_groups = 1, num_devices_per_group = 2, iota_reshape_dims = 3, iota_transpose_perm = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:IotaReplicaGroupListProto})
    num_replica_groups = zero(Int64)
    num_devices_per_group = zero(Int64)
    iota_reshape_dims = PB.BufferedVector{Int64}()
    iota_transpose_perm = PB.BufferedVector{Int32}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            num_replica_groups = PB.decode(d, Int64)
        elseif field_number == 2
            num_devices_per_group = PB.decode(d, Int64)
        elseif field_number == 3
            PB.decode!(d, wire_type, iota_reshape_dims)
        elseif field_number == 4
            PB.decode!(d, wire_type, iota_transpose_perm)
        else
            Base.skip(d, wire_type)
        end
    end
    return IotaReplicaGroupListProto(num_replica_groups, num_devices_per_group, iota_reshape_dims[], iota_transpose_perm[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::IotaReplicaGroupListProto)
    initpos = position(e.io)
    x.num_replica_groups != zero(Int64) && PB.encode(e, 1, x.num_replica_groups)
    x.num_devices_per_group != zero(Int64) && PB.encode(e, 2, x.num_devices_per_group)
    !isempty(x.iota_reshape_dims) && PB.encode(e, 3, x.iota_reshape_dims)
    !isempty(x.iota_transpose_perm) && PB.encode(e, 4, x.iota_transpose_perm)
    return position(e.io) - initpos
end
function PB._encoded_size(x::IotaReplicaGroupListProto)
    encoded_size = 0
    x.num_replica_groups != zero(Int64) && (encoded_size += PB._encoded_size(x.num_replica_groups, 1))
    x.num_devices_per_group != zero(Int64) && (encoded_size += PB._encoded_size(x.num_devices_per_group, 2))
    !isempty(x.iota_reshape_dims) && (encoded_size += PB._encoded_size(x.iota_reshape_dims, 3))
    !isempty(x.iota_transpose_perm) && (encoded_size += PB._encoded_size(x.iota_transpose_perm, 4))
    return encoded_size
end

struct OriginalArrayProto
    instruction_name::String
    shape_index::Vector{Int64}
end
OriginalArrayProto(;instruction_name = "", shape_index = Vector{Int64}()) = OriginalArrayProto(instruction_name, shape_index)
PB.default_values(::Type{OriginalArrayProto}) = (;instruction_name = "", shape_index = Vector{Int64}())
PB.field_numbers(::Type{OriginalArrayProto}) = (;instruction_name = 1, shape_index = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OriginalArrayProto})
    instruction_name = ""
    shape_index = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            instruction_name = PB.decode(d, String)
        elseif field_number == 2
            PB.decode!(d, wire_type, shape_index)
        else
            Base.skip(d, wire_type)
        end
    end
    return OriginalArrayProto(instruction_name, shape_index[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OriginalArrayProto)
    initpos = position(e.io)
    !isempty(x.instruction_name) && PB.encode(e, 1, x.instruction_name)
    !isempty(x.shape_index) && PB.encode(e, 2, x.shape_index)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OriginalArrayProto)
    encoded_size = 0
    !isempty(x.instruction_name) && (encoded_size += PB._encoded_size(x.instruction_name, 1))
    !isempty(x.shape_index) && (encoded_size += PB._encoded_size(x.shape_index, 2))
    return encoded_size
end

struct ComputationStats
    flop_count::Float64
    transcendental_count::Float64
end
ComputationStats(;flop_count = zero(Float64), transcendental_count = zero(Float64)) = ComputationStats(flop_count, transcendental_count)
PB.default_values(::Type{ComputationStats}) = (;flop_count = zero(Float64), transcendental_count = zero(Float64))
PB.field_numbers(::Type{ComputationStats}) = (;flop_count = 1, transcendental_count = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ComputationStats})
    flop_count = zero(Float64)
    transcendental_count = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            flop_count = PB.decode(d, Float64)
        elseif field_number == 2
            transcendental_count = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return ComputationStats(flop_count, transcendental_count)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ComputationStats)
    initpos = position(e.io)
    x.flop_count !== zero(Float64) && PB.encode(e, 1, x.flop_count)
    x.transcendental_count !== zero(Float64) && PB.encode(e, 2, x.transcendental_count)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ComputationStats)
    encoded_size = 0
    x.flop_count !== zero(Float64) && (encoded_size += PB._encoded_size(x.flop_count, 1))
    x.transcendental_count !== zero(Float64) && (encoded_size += PB._encoded_size(x.transcendental_count, 2))
    return encoded_size
end

struct FrontendAttributes
    map::Dict{String,String}
end
FrontendAttributes(;map = Dict{String,String}()) = FrontendAttributes(map)
PB.default_values(::Type{FrontendAttributes}) = (;map = Dict{String,String}())
PB.field_numbers(::Type{FrontendAttributes}) = (;map = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:FrontendAttributes})
    map = Dict{String,String}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, map)
        else
            Base.skip(d, wire_type)
        end
    end
    return FrontendAttributes(map)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::FrontendAttributes)
    initpos = position(e.io)
    !isempty(x.map) && PB.encode(e, 1, x.map)
    return position(e.io) - initpos
end
function PB._encoded_size(x::FrontendAttributes)
    encoded_size = 0
    !isempty(x.map) && (encoded_size += PB._encoded_size(x.map, 1))
    return encoded_size
end

@enumx ProfileType INVALID=0 WINDOW=1 FLAG=2 INTEGER=3

@enumx AsyncStreamKind ASYNC_STREAM_KIND_COLLECTIVE=0 ASYNC_STREAM_KIND_P2P0=1 ASYNC_STREAM_KIND_P2P1=2 ASYNC_STREAM_KIND_MEMCPYP2P=3

struct var"WhileLoopBackendConfig.KnownInductionVariable"
    tuple_index::Int64
end
var"WhileLoopBackendConfig.KnownInductionVariable"(;tuple_index = zero(Int64)) = var"WhileLoopBackendConfig.KnownInductionVariable"(tuple_index)
PB.default_values(::Type{var"WhileLoopBackendConfig.KnownInductionVariable"}) = (;tuple_index = zero(Int64))
PB.field_numbers(::Type{var"WhileLoopBackendConfig.KnownInductionVariable"}) = (;tuple_index = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"WhileLoopBackendConfig.KnownInductionVariable"})
    tuple_index = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            tuple_index = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"WhileLoopBackendConfig.KnownInductionVariable"(tuple_index)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"WhileLoopBackendConfig.KnownInductionVariable")
    initpos = position(e.io)
    x.tuple_index != zero(Int64) && PB.encode(e, 1, x.tuple_index)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"WhileLoopBackendConfig.KnownInductionVariable")
    encoded_size = 0
    x.tuple_index != zero(Int64) && (encoded_size += PB._encoded_size(x.tuple_index, 1))
    return encoded_size
end

struct var"PaddingConfig.PaddingConfigDimension"
    edge_padding_low::Int64
    edge_padding_high::Int64
    interior_padding::Int64
end
var"PaddingConfig.PaddingConfigDimension"(;edge_padding_low = zero(Int64), edge_padding_high = zero(Int64), interior_padding = zero(Int64)) = var"PaddingConfig.PaddingConfigDimension"(edge_padding_low, edge_padding_high, interior_padding)
PB.default_values(::Type{var"PaddingConfig.PaddingConfigDimension"}) = (;edge_padding_low = zero(Int64), edge_padding_high = zero(Int64), interior_padding = zero(Int64))
PB.field_numbers(::Type{var"PaddingConfig.PaddingConfigDimension"}) = (;edge_padding_low = 1, edge_padding_high = 2, interior_padding = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"PaddingConfig.PaddingConfigDimension"})
    edge_padding_low = zero(Int64)
    edge_padding_high = zero(Int64)
    interior_padding = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            edge_padding_low = PB.decode(d, Int64)
        elseif field_number == 2
            edge_padding_high = PB.decode(d, Int64)
        elseif field_number == 3
            interior_padding = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"PaddingConfig.PaddingConfigDimension"(edge_padding_low, edge_padding_high, interior_padding)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"PaddingConfig.PaddingConfigDimension")
    initpos = position(e.io)
    x.edge_padding_low != zero(Int64) && PB.encode(e, 1, x.edge_padding_low)
    x.edge_padding_high != zero(Int64) && PB.encode(e, 2, x.edge_padding_high)
    x.interior_padding != zero(Int64) && PB.encode(e, 3, x.interior_padding)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"PaddingConfig.PaddingConfigDimension")
    encoded_size = 0
    x.edge_padding_low != zero(Int64) && (encoded_size += PB._encoded_size(x.edge_padding_low, 1))
    x.edge_padding_high != zero(Int64) && (encoded_size += PB._encoded_size(x.edge_padding_high, 2))
    x.interior_padding != zero(Int64) && (encoded_size += PB._encoded_size(x.interior_padding, 3))
    return encoded_size
end

struct GemmPerfTableEntry
    b::Int64
    m::Int64
    n::Int64
    k::Int64
    flops::Dict{String,Int64}
end
GemmPerfTableEntry(;b = zero(Int64), m = zero(Int64), n = zero(Int64), k = zero(Int64), flops = Dict{String,Int64}()) = GemmPerfTableEntry(b, m, n, k, flops)
PB.default_values(::Type{GemmPerfTableEntry}) = (;b = zero(Int64), m = zero(Int64), n = zero(Int64), k = zero(Int64), flops = Dict{String,Int64}())
PB.field_numbers(::Type{GemmPerfTableEntry}) = (;b = 1, m = 2, n = 3, k = 4, flops = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GemmPerfTableEntry})
    b = zero(Int64)
    m = zero(Int64)
    n = zero(Int64)
    k = zero(Int64)
    flops = Dict{String,Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            b = PB.decode(d, Int64)
        elseif field_number == 2
            m = PB.decode(d, Int64)
        elseif field_number == 3
            n = PB.decode(d, Int64)
        elseif field_number == 4
            k = PB.decode(d, Int64)
        elseif field_number == 5
            PB.decode!(d, flops)
        else
            Base.skip(d, wire_type)
        end
    end
    return GemmPerfTableEntry(b, m, n, k, flops)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GemmPerfTableEntry)
    initpos = position(e.io)
    x.b != zero(Int64) && PB.encode(e, 1, x.b)
    x.m != zero(Int64) && PB.encode(e, 2, x.m)
    x.n != zero(Int64) && PB.encode(e, 3, x.n)
    x.k != zero(Int64) && PB.encode(e, 4, x.k)
    !isempty(x.flops) && PB.encode(e, 5, x.flops)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GemmPerfTableEntry)
    encoded_size = 0
    x.b != zero(Int64) && (encoded_size += PB._encoded_size(x.b, 1))
    x.m != zero(Int64) && (encoded_size += PB._encoded_size(x.m, 2))
    x.n != zero(Int64) && (encoded_size += PB._encoded_size(x.n, 3))
    x.k != zero(Int64) && (encoded_size += PB._encoded_size(x.k, 4))
    !isempty(x.flops) && (encoded_size += PB._encoded_size(x.flops, 5))
    return encoded_size
end

struct OutputOperandAliasing
    output_shape_index::Vector{Int64}
    operand_index::Int64
    operand_shape_index::Vector{Int64}
end
OutputOperandAliasing(;output_shape_index = Vector{Int64}(), operand_index = zero(Int64), operand_shape_index = Vector{Int64}()) = OutputOperandAliasing(output_shape_index, operand_index, operand_shape_index)
PB.default_values(::Type{OutputOperandAliasing}) = (;output_shape_index = Vector{Int64}(), operand_index = zero(Int64), operand_shape_index = Vector{Int64}())
PB.field_numbers(::Type{OutputOperandAliasing}) = (;output_shape_index = 1, operand_index = 2, operand_shape_index = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OutputOperandAliasing})
    output_shape_index = PB.BufferedVector{Int64}()
    operand_index = zero(Int64)
    operand_shape_index = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, output_shape_index)
        elseif field_number == 2
            operand_index = PB.decode(d, Int64)
        elseif field_number == 3
            PB.decode!(d, wire_type, operand_shape_index)
        else
            Base.skip(d, wire_type)
        end
    end
    return OutputOperandAliasing(output_shape_index[], operand_index, operand_shape_index[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OutputOperandAliasing)
    initpos = position(e.io)
    !isempty(x.output_shape_index) && PB.encode(e, 1, x.output_shape_index)
    x.operand_index != zero(Int64) && PB.encode(e, 2, x.operand_index)
    !isempty(x.operand_shape_index) && PB.encode(e, 3, x.operand_shape_index)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OutputOperandAliasing)
    encoded_size = 0
    !isempty(x.output_shape_index) && (encoded_size += PB._encoded_size(x.output_shape_index, 1))
    x.operand_index != zero(Int64) && (encoded_size += PB._encoded_size(x.operand_index, 2))
    !isempty(x.operand_shape_index) && (encoded_size += PB._encoded_size(x.operand_shape_index, 3))
    return encoded_size
end

@enumx var"PrecisionConfig.Precision" DEFAULT=0 HIGH=1 HIGHEST=2

struct ExecutionProfile
    compilation_cache_hit::Bool
    compile_time_ms::Int64
    compute_cycle_count::Int64
    compute_time_ns::Int64
    compute_and_transfer_time_ns::Int64
    executable_size_in_bytes::Int64
    profile_cache_hit::Bool
    warmup_run_executed::Bool
end
ExecutionProfile(;compilation_cache_hit = false, compile_time_ms = zero(Int64), compute_cycle_count = zero(Int64), compute_time_ns = zero(Int64), compute_and_transfer_time_ns = zero(Int64), executable_size_in_bytes = zero(Int64), profile_cache_hit = false, warmup_run_executed = false) = ExecutionProfile(compilation_cache_hit, compile_time_ms, compute_cycle_count, compute_time_ns, compute_and_transfer_time_ns, executable_size_in_bytes, profile_cache_hit, warmup_run_executed)
PB.default_values(::Type{ExecutionProfile}) = (;compilation_cache_hit = false, compile_time_ms = zero(Int64), compute_cycle_count = zero(Int64), compute_time_ns = zero(Int64), compute_and_transfer_time_ns = zero(Int64), executable_size_in_bytes = zero(Int64), profile_cache_hit = false, warmup_run_executed = false)
PB.field_numbers(::Type{ExecutionProfile}) = (;compilation_cache_hit = 1, compile_time_ms = 2, compute_cycle_count = 3, compute_time_ns = 4, compute_and_transfer_time_ns = 5, executable_size_in_bytes = 6, profile_cache_hit = 7, warmup_run_executed = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ExecutionProfile})
    compilation_cache_hit = false
    compile_time_ms = zero(Int64)
    compute_cycle_count = zero(Int64)
    compute_time_ns = zero(Int64)
    compute_and_transfer_time_ns = zero(Int64)
    executable_size_in_bytes = zero(Int64)
    profile_cache_hit = false
    warmup_run_executed = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            compilation_cache_hit = PB.decode(d, Bool)
        elseif field_number == 2
            compile_time_ms = PB.decode(d, Int64)
        elseif field_number == 3
            compute_cycle_count = PB.decode(d, Int64)
        elseif field_number == 4
            compute_time_ns = PB.decode(d, Int64)
        elseif field_number == 5
            compute_and_transfer_time_ns = PB.decode(d, Int64)
        elseif field_number == 6
            executable_size_in_bytes = PB.decode(d, Int64)
        elseif field_number == 7
            profile_cache_hit = PB.decode(d, Bool)
        elseif field_number == 8
            warmup_run_executed = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return ExecutionProfile(compilation_cache_hit, compile_time_ms, compute_cycle_count, compute_time_ns, compute_and_transfer_time_ns, executable_size_in_bytes, profile_cache_hit, warmup_run_executed)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ExecutionProfile)
    initpos = position(e.io)
    x.compilation_cache_hit != false && PB.encode(e, 1, x.compilation_cache_hit)
    x.compile_time_ms != zero(Int64) && PB.encode(e, 2, x.compile_time_ms)
    x.compute_cycle_count != zero(Int64) && PB.encode(e, 3, x.compute_cycle_count)
    x.compute_time_ns != zero(Int64) && PB.encode(e, 4, x.compute_time_ns)
    x.compute_and_transfer_time_ns != zero(Int64) && PB.encode(e, 5, x.compute_and_transfer_time_ns)
    x.executable_size_in_bytes != zero(Int64) && PB.encode(e, 6, x.executable_size_in_bytes)
    x.profile_cache_hit != false && PB.encode(e, 7, x.profile_cache_hit)
    x.warmup_run_executed != false && PB.encode(e, 8, x.warmup_run_executed)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ExecutionProfile)
    encoded_size = 0
    x.compilation_cache_hit != false && (encoded_size += PB._encoded_size(x.compilation_cache_hit, 1))
    x.compile_time_ms != zero(Int64) && (encoded_size += PB._encoded_size(x.compile_time_ms, 2))
    x.compute_cycle_count != zero(Int64) && (encoded_size += PB._encoded_size(x.compute_cycle_count, 3))
    x.compute_time_ns != zero(Int64) && (encoded_size += PB._encoded_size(x.compute_time_ns, 4))
    x.compute_and_transfer_time_ns != zero(Int64) && (encoded_size += PB._encoded_size(x.compute_and_transfer_time_ns, 5))
    x.executable_size_in_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.executable_size_in_bytes, 6))
    x.profile_cache_hit != false && (encoded_size += PB._encoded_size(x.profile_cache_hit, 7))
    x.warmup_run_executed != false && (encoded_size += PB._encoded_size(x.warmup_run_executed, 8))
    return encoded_size
end

struct var"AxisRefProto.SubAxis"
    pre_size::Int64
    size::Int64
end
var"AxisRefProto.SubAxis"(;pre_size = zero(Int64), size = zero(Int64)) = var"AxisRefProto.SubAxis"(pre_size, size)
PB.default_values(::Type{var"AxisRefProto.SubAxis"}) = (;pre_size = zero(Int64), size = zero(Int64))
PB.field_numbers(::Type{var"AxisRefProto.SubAxis"}) = (;pre_size = 1, size = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"AxisRefProto.SubAxis"})
    pre_size = zero(Int64)
    size = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            pre_size = PB.decode(d, Int64)
        elseif field_number == 2
            size = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"AxisRefProto.SubAxis"(pre_size, size)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"AxisRefProto.SubAxis")
    initpos = position(e.io)
    x.pre_size != zero(Int64) && PB.encode(e, 1, x.pre_size)
    x.size != zero(Int64) && PB.encode(e, 2, x.size)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"AxisRefProto.SubAxis")
    encoded_size = 0
    x.pre_size != zero(Int64) && (encoded_size += PB._encoded_size(x.pre_size, 1))
    x.size != zero(Int64) && (encoded_size += PB._encoded_size(x.size, 2))
    return encoded_size
end

@enumx ProfileGenerationStrategy PROFILE_GENERATION_STRATEGY_UNKNOWN=0 PROFILE_GENERATION_STRATEGY_GA=1 PROFILE_GENERATION_STRATEGY_FANTA=2 PROFILE_GENERATION_STRATEGY_CFO=3 PROFILE_GENERATION_STRATEGY_EXHAUSTIVE=4 PROFILE_GENERATION_STRATEGY_LCM_GNN=5 PROFILE_GENERATION_STRATEGY_LCM_MOE=6

@enumx PaddingType PADDING_INVALID=0 PADDING_VALID=1 PADDING_SAME=2

struct var"MeshProto.MeshAxis"
    name::String
    size::Int64
end
var"MeshProto.MeshAxis"(;name = "", size = zero(Int64)) = var"MeshProto.MeshAxis"(name, size)
PB.default_values(::Type{var"MeshProto.MeshAxis"}) = (;name = "", size = zero(Int64))
PB.field_numbers(::Type{var"MeshProto.MeshAxis"}) = (;name = 1, size = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"MeshProto.MeshAxis"})
    name = ""
    size = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            size = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"MeshProto.MeshAxis"(name, size)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"MeshProto.MeshAxis")
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    x.size != zero(Int64) && PB.encode(e, 2, x.size)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"MeshProto.MeshAxis")
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    x.size != zero(Int64) && (encoded_size += PB._encoded_size(x.size, 2))
    return encoded_size
end

@enumx var"OpSharding.ShardGroupType" AS=0 LIKE=1

struct CholeskyOptions
    lower::Bool
end
CholeskyOptions(;lower = false) = CholeskyOptions(lower)
PB.default_values(::Type{CholeskyOptions}) = (;lower = false)
PB.field_numbers(::Type{CholeskyOptions}) = (;lower = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CholeskyOptions})
    lower = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            lower = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return CholeskyOptions(lower)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CholeskyOptions)
    initpos = position(e.io)
    x.lower != false && PB.encode(e, 1, x.lower)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CholeskyOptions)
    encoded_size = 0
    x.lower != false && (encoded_size += PB._encoded_size(x.lower, 1))
    return encoded_size
end

struct StatisticsViz
    stat_index_to_visualize::Int64
    statistics::Vector{Statistic}
end
StatisticsViz(;stat_index_to_visualize = zero(Int64), statistics = Vector{Statistic}()) = StatisticsViz(stat_index_to_visualize, statistics)
PB.default_values(::Type{StatisticsViz}) = (;stat_index_to_visualize = zero(Int64), statistics = Vector{Statistic}())
PB.field_numbers(::Type{StatisticsViz}) = (;stat_index_to_visualize = 1, statistics = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:StatisticsViz})
    stat_index_to_visualize = zero(Int64)
    statistics = PB.BufferedVector{Statistic}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            stat_index_to_visualize = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, statistics)
        else
            Base.skip(d, wire_type)
        end
    end
    return StatisticsViz(stat_index_to_visualize, statistics[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::StatisticsViz)
    initpos = position(e.io)
    x.stat_index_to_visualize != zero(Int64) && PB.encode(e, 1, x.stat_index_to_visualize)
    !isempty(x.statistics) && PB.encode(e, 2, x.statistics)
    return position(e.io) - initpos
end
function PB._encoded_size(x::StatisticsViz)
    encoded_size = 0
    x.stat_index_to_visualize != zero(Int64) && (encoded_size += PB._encoded_size(x.stat_index_to_visualize, 1))
    !isempty(x.statistics) && (encoded_size += PB._encoded_size(x.statistics, 2))
    return encoded_size
end

struct TriangularSolveOptions
    left_side::Bool
    lower::Bool
    unit_diagonal::Bool
    transpose_a::var"TriangularSolveOptions.Transpose".T
end
TriangularSolveOptions(;left_side = false, lower = false, unit_diagonal = false, transpose_a = var"TriangularSolveOptions.Transpose".TRANSPOSE_INVALID) = TriangularSolveOptions(left_side, lower, unit_diagonal, transpose_a)
PB.default_values(::Type{TriangularSolveOptions}) = (;left_side = false, lower = false, unit_diagonal = false, transpose_a = var"TriangularSolveOptions.Transpose".TRANSPOSE_INVALID)
PB.field_numbers(::Type{TriangularSolveOptions}) = (;left_side = 1, lower = 2, unit_diagonal = 3, transpose_a = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TriangularSolveOptions})
    left_side = false
    lower = false
    unit_diagonal = false
    transpose_a = var"TriangularSolveOptions.Transpose".TRANSPOSE_INVALID
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            left_side = PB.decode(d, Bool)
        elseif field_number == 2
            lower = PB.decode(d, Bool)
        elseif field_number == 3
            unit_diagonal = PB.decode(d, Bool)
        elseif field_number == 4
            transpose_a = PB.decode(d, var"TriangularSolveOptions.Transpose".T)
        else
            Base.skip(d, wire_type)
        end
    end
    return TriangularSolveOptions(left_side, lower, unit_diagonal, transpose_a)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TriangularSolveOptions)
    initpos = position(e.io)
    x.left_side != false && PB.encode(e, 1, x.left_side)
    x.lower != false && PB.encode(e, 2, x.lower)
    x.unit_diagonal != false && PB.encode(e, 3, x.unit_diagonal)
    x.transpose_a != var"TriangularSolveOptions.Transpose".TRANSPOSE_INVALID && PB.encode(e, 4, x.transpose_a)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TriangularSolveOptions)
    encoded_size = 0
    x.left_side != false && (encoded_size += PB._encoded_size(x.left_side, 1))
    x.lower != false && (encoded_size += PB._encoded_size(x.lower, 2))
    x.unit_diagonal != false && (encoded_size += PB._encoded_size(x.unit_diagonal, 3))
    x.transpose_a != var"TriangularSolveOptions.Transpose".TRANSPOSE_INVALID && (encoded_size += PB._encoded_size(x.transpose_a, 4))
    return encoded_size
end

struct DeviceAssignmentProto
    replica_count::Int32
    computation_count::Int32
    computation_devices::Vector{var"DeviceAssignmentProto.ComputationDevice"}
end
DeviceAssignmentProto(;replica_count = zero(Int32), computation_count = zero(Int32), computation_devices = Vector{var"DeviceAssignmentProto.ComputationDevice"}()) = DeviceAssignmentProto(replica_count, computation_count, computation_devices)
PB.default_values(::Type{DeviceAssignmentProto}) = (;replica_count = zero(Int32), computation_count = zero(Int32), computation_devices = Vector{var"DeviceAssignmentProto.ComputationDevice"}())
PB.field_numbers(::Type{DeviceAssignmentProto}) = (;replica_count = 1, computation_count = 2, computation_devices = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:DeviceAssignmentProto})
    replica_count = zero(Int32)
    computation_count = zero(Int32)
    computation_devices = PB.BufferedVector{var"DeviceAssignmentProto.ComputationDevice"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            replica_count = PB.decode(d, Int32)
        elseif field_number == 2
            computation_count = PB.decode(d, Int32)
        elseif field_number == 3
            PB.decode!(d, computation_devices)
        else
            Base.skip(d, wire_type)
        end
    end
    return DeviceAssignmentProto(replica_count, computation_count, computation_devices[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::DeviceAssignmentProto)
    initpos = position(e.io)
    x.replica_count != zero(Int32) && PB.encode(e, 1, x.replica_count)
    x.computation_count != zero(Int32) && PB.encode(e, 2, x.computation_count)
    !isempty(x.computation_devices) && PB.encode(e, 3, x.computation_devices)
    return position(e.io) - initpos
end
function PB._encoded_size(x::DeviceAssignmentProto)
    encoded_size = 0
    x.replica_count != zero(Int32) && (encoded_size += PB._encoded_size(x.replica_count, 1))
    x.computation_count != zero(Int32) && (encoded_size += PB._encoded_size(x.computation_count, 2))
    !isempty(x.computation_devices) && (encoded_size += PB._encoded_size(x.computation_devices, 3))
    return encoded_size
end

struct ChannelHandle
    handle::Int64
    var"#type"::var"ChannelHandle.ChannelType".T
end
ChannelHandle(;handle = zero(Int64), var"#type" = var"ChannelHandle.ChannelType".CHANNEL_TYPE_INVALID) = ChannelHandle(handle, var"#type")
PB.default_values(::Type{ChannelHandle}) = (;handle = zero(Int64), var"#type" = var"ChannelHandle.ChannelType".CHANNEL_TYPE_INVALID)
PB.field_numbers(::Type{ChannelHandle}) = (;handle = 1, var"#type" = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ChannelHandle})
    handle = zero(Int64)
    var"#type" = var"ChannelHandle.ChannelType".CHANNEL_TYPE_INVALID
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            handle = PB.decode(d, Int64)
        elseif field_number == 2
            var"#type" = PB.decode(d, var"ChannelHandle.ChannelType".T)
        else
            Base.skip(d, wire_type)
        end
    end
    return ChannelHandle(handle, var"#type")
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ChannelHandle)
    initpos = position(e.io)
    x.handle != zero(Int64) && PB.encode(e, 1, x.handle)
    x.var"#type" != var"ChannelHandle.ChannelType".CHANNEL_TYPE_INVALID && PB.encode(e, 2, x.var"#type")
    return position(e.io) - initpos
end
function PB._encoded_size(x::ChannelHandle)
    encoded_size = 0
    x.handle != zero(Int64) && (encoded_size += PB._encoded_size(x.handle, 1))
    x.var"#type" != var"ChannelHandle.ChannelType".CHANNEL_TYPE_INVALID && (encoded_size += PB._encoded_size(x.var"#type", 2))
    return encoded_size
end

struct ResultAccuracy
    specs::Union{Nothing,OneOf{<:Union{var"ResultAccuracy.Mode".T,var"ResultAccuracy.Tolerance"}}}
end
ResultAccuracy(;specs = nothing) = ResultAccuracy(specs)
PB.oneof_field_types(::Type{ResultAccuracy}) = (;
    specs = (;mode=var"ResultAccuracy.Mode".T, tolerance=var"ResultAccuracy.Tolerance"),
)
PB.default_values(::Type{ResultAccuracy}) = (;mode = var"ResultAccuracy.Mode".DEFAULT, tolerance = nothing)
PB.field_numbers(::Type{ResultAccuracy}) = (;mode = 1, tolerance = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ResultAccuracy})
    specs = nothing
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            specs = OneOf(:mode, PB.decode(d, var"ResultAccuracy.Mode".T))
        elseif field_number == 2
            specs = OneOf(:tolerance, PB.decode(d, Ref{var"ResultAccuracy.Tolerance"}))
        else
            Base.skip(d, wire_type)
        end
    end
    return ResultAccuracy(specs)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ResultAccuracy)
    initpos = position(e.io)
    if isnothing(x.specs);
    elseif x.specs.name === :mode
        PB.encode(e, 1, x.specs[]::var"ResultAccuracy.Mode".T)
    elseif x.specs.name === :tolerance
        PB.encode(e, 2, x.specs[]::var"ResultAccuracy.Tolerance")
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::ResultAccuracy)
    encoded_size = 0
    if isnothing(x.specs);
    elseif x.specs.name === :mode
        encoded_size += PB._encoded_size(x.specs[]::var"ResultAccuracy.Mode".T, 1)
    elseif x.specs.name === :tolerance
        encoded_size += PB._encoded_size(x.specs[]::var"ResultAccuracy.Tolerance", 2)
    end
    return encoded_size
end

struct RaggedDotDimensionNumbers
    dot_dimension_numbers::Union{Nothing,DotDimensionNumbers}
    lhs_ragged_dimensions::Vector{Int64}
    rhs_group_dimensions::Vector{Int64}
end
RaggedDotDimensionNumbers(;dot_dimension_numbers = nothing, lhs_ragged_dimensions = Vector{Int64}(), rhs_group_dimensions = Vector{Int64}()) = RaggedDotDimensionNumbers(dot_dimension_numbers, lhs_ragged_dimensions, rhs_group_dimensions)
PB.default_values(::Type{RaggedDotDimensionNumbers}) = (;dot_dimension_numbers = nothing, lhs_ragged_dimensions = Vector{Int64}(), rhs_group_dimensions = Vector{Int64}())
PB.field_numbers(::Type{RaggedDotDimensionNumbers}) = (;dot_dimension_numbers = 1, lhs_ragged_dimensions = 2, rhs_group_dimensions = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:RaggedDotDimensionNumbers})
    dot_dimension_numbers = Ref{Union{Nothing,DotDimensionNumbers}}(nothing)
    lhs_ragged_dimensions = PB.BufferedVector{Int64}()
    rhs_group_dimensions = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, dot_dimension_numbers)
        elseif field_number == 2
            PB.decode!(d, wire_type, lhs_ragged_dimensions)
        elseif field_number == 3
            PB.decode!(d, wire_type, rhs_group_dimensions)
        else
            Base.skip(d, wire_type)
        end
    end
    return RaggedDotDimensionNumbers(dot_dimension_numbers[], lhs_ragged_dimensions[], rhs_group_dimensions[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::RaggedDotDimensionNumbers)
    initpos = position(e.io)
    !isnothing(x.dot_dimension_numbers) && PB.encode(e, 1, x.dot_dimension_numbers)
    !isempty(x.lhs_ragged_dimensions) && PB.encode(e, 2, x.lhs_ragged_dimensions)
    !isempty(x.rhs_group_dimensions) && PB.encode(e, 3, x.rhs_group_dimensions)
    return position(e.io) - initpos
end
function PB._encoded_size(x::RaggedDotDimensionNumbers)
    encoded_size = 0
    !isnothing(x.dot_dimension_numbers) && (encoded_size += PB._encoded_size(x.dot_dimension_numbers, 1))
    !isempty(x.lhs_ragged_dimensions) && (encoded_size += PB._encoded_size(x.lhs_ragged_dimensions, 2))
    !isempty(x.rhs_group_dimensions) && (encoded_size += PB._encoded_size(x.rhs_group_dimensions, 3))
    return encoded_size
end

struct Window
    dimensions::Vector{WindowDimension}
end
Window(;dimensions = Vector{WindowDimension}()) = Window(dimensions)
PB.default_values(::Type{Window}) = (;dimensions = Vector{WindowDimension}())
PB.field_numbers(::Type{Window}) = (;dimensions = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Window})
    dimensions = PB.BufferedVector{WindowDimension}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, dimensions)
        else
            Base.skip(d, wire_type)
        end
    end
    return Window(dimensions[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Window)
    initpos = position(e.io)
    !isempty(x.dimensions) && PB.encode(e, 1, x.dimensions)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Window)
    encoded_size = 0
    !isempty(x.dimensions) && (encoded_size += PB._encoded_size(x.dimensions, 1))
    return encoded_size
end

struct CollectiveDeviceListProto
    replica_groups::Vector{ReplicaGroup}
    iota_replica_group_list::Union{Nothing,IotaReplicaGroupListProto}
end
CollectiveDeviceListProto(;replica_groups = Vector{ReplicaGroup}(), iota_replica_group_list = nothing) = CollectiveDeviceListProto(replica_groups, iota_replica_group_list)
PB.default_values(::Type{CollectiveDeviceListProto}) = (;replica_groups = Vector{ReplicaGroup}(), iota_replica_group_list = nothing)
PB.field_numbers(::Type{CollectiveDeviceListProto}) = (;replica_groups = 1, iota_replica_group_list = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CollectiveDeviceListProto})
    replica_groups = PB.BufferedVector{ReplicaGroup}()
    iota_replica_group_list = Ref{Union{Nothing,IotaReplicaGroupListProto}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, replica_groups)
        elseif field_number == 2
            PB.decode!(d, iota_replica_group_list)
        else
            Base.skip(d, wire_type)
        end
    end
    return CollectiveDeviceListProto(replica_groups[], iota_replica_group_list[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CollectiveDeviceListProto)
    initpos = position(e.io)
    !isempty(x.replica_groups) && PB.encode(e, 1, x.replica_groups)
    !isnothing(x.iota_replica_group_list) && PB.encode(e, 2, x.iota_replica_group_list)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CollectiveDeviceListProto)
    encoded_size = 0
    !isempty(x.replica_groups) && (encoded_size += PB._encoded_size(x.replica_groups, 1))
    !isnothing(x.iota_replica_group_list) && (encoded_size += PB._encoded_size(x.iota_replica_group_list, 2))
    return encoded_size
end

struct OriginalValueElementProto
    shape_index::Vector{Int64}
    original_array::Union{Nothing,OriginalArrayProto}
end
OriginalValueElementProto(;shape_index = Vector{Int64}(), original_array = nothing) = OriginalValueElementProto(shape_index, original_array)
PB.default_values(::Type{OriginalValueElementProto}) = (;shape_index = Vector{Int64}(), original_array = nothing)
PB.field_numbers(::Type{OriginalValueElementProto}) = (;shape_index = 1, original_array = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OriginalValueElementProto})
    shape_index = PB.BufferedVector{Int64}()
    original_array = Ref{Union{Nothing,OriginalArrayProto}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, shape_index)
        elseif field_number == 2
            PB.decode!(d, original_array)
        else
            Base.skip(d, wire_type)
        end
    end
    return OriginalValueElementProto(shape_index[], original_array[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OriginalValueElementProto)
    initpos = position(e.io)
    !isempty(x.shape_index) && PB.encode(e, 1, x.shape_index)
    !isnothing(x.original_array) && PB.encode(e, 2, x.original_array)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OriginalValueElementProto)
    encoded_size = 0
    !isempty(x.shape_index) && (encoded_size += PB._encoded_size(x.shape_index, 1))
    !isnothing(x.original_array) && (encoded_size += PB._encoded_size(x.original_array, 2))
    return encoded_size
end

struct WhileLoopBackendConfig
    known_trip_count::Union{Nothing,var"WhileLoopBackendConfig.KnownTripCount"}
    known_init_step::Union{Nothing,var"WhileLoopBackendConfig.KnownInitStep"}
    known_induction_variable::Union{Nothing,var"WhileLoopBackendConfig.KnownInductionVariable"}
end
WhileLoopBackendConfig(;known_trip_count = nothing, known_init_step = nothing, known_induction_variable = nothing) = WhileLoopBackendConfig(known_trip_count, known_init_step, known_induction_variable)
PB.default_values(::Type{WhileLoopBackendConfig}) = (;known_trip_count = nothing, known_init_step = nothing, known_induction_variable = nothing)
PB.field_numbers(::Type{WhileLoopBackendConfig}) = (;known_trip_count = 1, known_init_step = 2, known_induction_variable = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:WhileLoopBackendConfig})
    known_trip_count = Ref{Union{Nothing,var"WhileLoopBackendConfig.KnownTripCount"}}(nothing)
    known_init_step = Ref{Union{Nothing,var"WhileLoopBackendConfig.KnownInitStep"}}(nothing)
    known_induction_variable = Ref{Union{Nothing,var"WhileLoopBackendConfig.KnownInductionVariable"}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, known_trip_count)
        elseif field_number == 2
            PB.decode!(d, known_init_step)
        elseif field_number == 3
            PB.decode!(d, known_induction_variable)
        else
            Base.skip(d, wire_type)
        end
    end
    return WhileLoopBackendConfig(known_trip_count[], known_init_step[], known_induction_variable[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::WhileLoopBackendConfig)
    initpos = position(e.io)
    !isnothing(x.known_trip_count) && PB.encode(e, 1, x.known_trip_count)
    !isnothing(x.known_init_step) && PB.encode(e, 2, x.known_init_step)
    !isnothing(x.known_induction_variable) && PB.encode(e, 3, x.known_induction_variable)
    return position(e.io) - initpos
end
function PB._encoded_size(x::WhileLoopBackendConfig)
    encoded_size = 0
    !isnothing(x.known_trip_count) && (encoded_size += PB._encoded_size(x.known_trip_count, 1))
    !isnothing(x.known_init_step) && (encoded_size += PB._encoded_size(x.known_init_step, 2))
    !isnothing(x.known_induction_variable) && (encoded_size += PB._encoded_size(x.known_induction_variable, 3))
    return encoded_size
end

struct PaddingConfig
    dimensions::Vector{var"PaddingConfig.PaddingConfigDimension"}
end
PaddingConfig(;dimensions = Vector{var"PaddingConfig.PaddingConfigDimension"}()) = PaddingConfig(dimensions)
PB.default_values(::Type{PaddingConfig}) = (;dimensions = Vector{var"PaddingConfig.PaddingConfigDimension"}())
PB.field_numbers(::Type{PaddingConfig}) = (;dimensions = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PaddingConfig})
    dimensions = PB.BufferedVector{var"PaddingConfig.PaddingConfigDimension"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, dimensions)
        else
            Base.skip(d, wire_type)
        end
    end
    return PaddingConfig(dimensions[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PaddingConfig)
    initpos = position(e.io)
    !isempty(x.dimensions) && PB.encode(e, 1, x.dimensions)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PaddingConfig)
    encoded_size = 0
    !isempty(x.dimensions) && (encoded_size += PB._encoded_size(x.dimensions, 1))
    return encoded_size
end

struct GemmPerfTableEntryValues
    entries::Vector{GemmPerfTableEntry}
end
GemmPerfTableEntryValues(;entries = Vector{GemmPerfTableEntry}()) = GemmPerfTableEntryValues(entries)
PB.default_values(::Type{GemmPerfTableEntryValues}) = (;entries = Vector{GemmPerfTableEntry}())
PB.field_numbers(::Type{GemmPerfTableEntryValues}) = (;entries = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GemmPerfTableEntryValues})
    entries = PB.BufferedVector{GemmPerfTableEntry}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, entries)
        else
            Base.skip(d, wire_type)
        end
    end
    return GemmPerfTableEntryValues(entries[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GemmPerfTableEntryValues)
    initpos = position(e.io)
    !isempty(x.entries) && PB.encode(e, 1, x.entries)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GemmPerfTableEntryValues)
    encoded_size = 0
    !isempty(x.entries) && (encoded_size += PB._encoded_size(x.entries, 1))
    return encoded_size
end

struct PrecisionConfig
    operand_precision::Vector{var"PrecisionConfig.Precision".T}
    algorithm::var"PrecisionConfig.Algorithm".T
end
PrecisionConfig(;operand_precision = Vector{var"PrecisionConfig.Precision".T}(), algorithm = var"PrecisionConfig.Algorithm".ALG_UNSET) = PrecisionConfig(operand_precision, algorithm)
PB.default_values(::Type{PrecisionConfig}) = (;operand_precision = Vector{var"PrecisionConfig.Precision".T}(), algorithm = var"PrecisionConfig.Algorithm".ALG_UNSET)
PB.field_numbers(::Type{PrecisionConfig}) = (;operand_precision = 1, algorithm = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PrecisionConfig})
    operand_precision = PB.BufferedVector{var"PrecisionConfig.Precision".T}()
    algorithm = var"PrecisionConfig.Algorithm".ALG_UNSET
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, operand_precision)
        elseif field_number == 2
            algorithm = PB.decode(d, var"PrecisionConfig.Algorithm".T)
        else
            Base.skip(d, wire_type)
        end
    end
    return PrecisionConfig(operand_precision[], algorithm)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PrecisionConfig)
    initpos = position(e.io)
    !isempty(x.operand_precision) && PB.encode(e, 1, x.operand_precision)
    x.algorithm != var"PrecisionConfig.Algorithm".ALG_UNSET && PB.encode(e, 2, x.algorithm)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PrecisionConfig)
    encoded_size = 0
    !isempty(x.operand_precision) && (encoded_size += PB._encoded_size(x.operand_precision, 1))
    x.algorithm != var"PrecisionConfig.Algorithm".ALG_UNSET && (encoded_size += PB._encoded_size(x.algorithm, 2))
    return encoded_size
end

struct AxisRefProto
    mesh_axis_index::Int64
    sub_axis_info::Union{Nothing,var"AxisRefProto.SubAxis"}
end
AxisRefProto(;mesh_axis_index = zero(Int64), sub_axis_info = nothing) = AxisRefProto(mesh_axis_index, sub_axis_info)
PB.default_values(::Type{AxisRefProto}) = (;mesh_axis_index = zero(Int64), sub_axis_info = nothing)
PB.field_numbers(::Type{AxisRefProto}) = (;mesh_axis_index = 1, sub_axis_info = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AxisRefProto})
    mesh_axis_index = zero(Int64)
    sub_axis_info = Ref{Union{Nothing,var"AxisRefProto.SubAxis"}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            mesh_axis_index = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, sub_axis_info)
        else
            Base.skip(d, wire_type)
        end
    end
    return AxisRefProto(mesh_axis_index, sub_axis_info[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AxisRefProto)
    initpos = position(e.io)
    x.mesh_axis_index != zero(Int64) && PB.encode(e, 1, x.mesh_axis_index)
    !isnothing(x.sub_axis_info) && PB.encode(e, 2, x.sub_axis_info)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AxisRefProto)
    encoded_size = 0
    x.mesh_axis_index != zero(Int64) && (encoded_size += PB._encoded_size(x.mesh_axis_index, 1))
    !isnothing(x.sub_axis_info) && (encoded_size += PB._encoded_size(x.sub_axis_info, 2))
    return encoded_size
end

struct var"OpMetadata.ProfileInfo"
    profile_type::Vector{ProfileType.T}
    relative_speedup::Float64
    profile_source::ProfileSource.T
    compilation_event::CompilationEvent.T
    profile_generation_strategy::ProfileGenerationStrategy.T
end
var"OpMetadata.ProfileInfo"(;profile_type = Vector{ProfileType.T}(), relative_speedup = zero(Float64), profile_source = ProfileSource.PROFILE_SOURCE_UNKNOWN_SOURCE, compilation_event = CompilationEvent.COMPILATION_EVENT_UNKNOWN_EVENT, profile_generation_strategy = ProfileGenerationStrategy.PROFILE_GENERATION_STRATEGY_UNKNOWN) = var"OpMetadata.ProfileInfo"(profile_type, relative_speedup, profile_source, compilation_event, profile_generation_strategy)
PB.default_values(::Type{var"OpMetadata.ProfileInfo"}) = (;profile_type = Vector{ProfileType.T}(), relative_speedup = zero(Float64), profile_source = ProfileSource.PROFILE_SOURCE_UNKNOWN_SOURCE, compilation_event = CompilationEvent.COMPILATION_EVENT_UNKNOWN_EVENT, profile_generation_strategy = ProfileGenerationStrategy.PROFILE_GENERATION_STRATEGY_UNKNOWN)
PB.field_numbers(::Type{var"OpMetadata.ProfileInfo"}) = (;profile_type = 1, relative_speedup = 2, profile_source = 3, compilation_event = 4, profile_generation_strategy = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"OpMetadata.ProfileInfo"})
    profile_type = PB.BufferedVector{ProfileType.T}()
    relative_speedup = zero(Float64)
    profile_source = ProfileSource.PROFILE_SOURCE_UNKNOWN_SOURCE
    compilation_event = CompilationEvent.COMPILATION_EVENT_UNKNOWN_EVENT
    profile_generation_strategy = ProfileGenerationStrategy.PROFILE_GENERATION_STRATEGY_UNKNOWN
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, profile_type)
        elseif field_number == 2
            relative_speedup = PB.decode(d, Float64)
        elseif field_number == 3
            profile_source = PB.decode(d, ProfileSource.T)
        elseif field_number == 4
            compilation_event = PB.decode(d, CompilationEvent.T)
        elseif field_number == 5
            profile_generation_strategy = PB.decode(d, ProfileGenerationStrategy.T)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"OpMetadata.ProfileInfo"(profile_type[], relative_speedup, profile_source, compilation_event, profile_generation_strategy)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"OpMetadata.ProfileInfo")
    initpos = position(e.io)
    !isempty(x.profile_type) && PB.encode(e, 1, x.profile_type)
    x.relative_speedup !== zero(Float64) && PB.encode(e, 2, x.relative_speedup)
    x.profile_source != ProfileSource.PROFILE_SOURCE_UNKNOWN_SOURCE && PB.encode(e, 3, x.profile_source)
    x.compilation_event != CompilationEvent.COMPILATION_EVENT_UNKNOWN_EVENT && PB.encode(e, 4, x.compilation_event)
    x.profile_generation_strategy != ProfileGenerationStrategy.PROFILE_GENERATION_STRATEGY_UNKNOWN && PB.encode(e, 5, x.profile_generation_strategy)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"OpMetadata.ProfileInfo")
    encoded_size = 0
    !isempty(x.profile_type) && (encoded_size += PB._encoded_size(x.profile_type, 1))
    x.relative_speedup !== zero(Float64) && (encoded_size += PB._encoded_size(x.relative_speedup, 2))
    x.profile_source != ProfileSource.PROFILE_SOURCE_UNKNOWN_SOURCE && (encoded_size += PB._encoded_size(x.profile_source, 3))
    x.compilation_event != CompilationEvent.COMPILATION_EVENT_UNKNOWN_EVENT && (encoded_size += PB._encoded_size(x.compilation_event, 4))
    x.profile_generation_strategy != ProfileGenerationStrategy.PROFILE_GENERATION_STRATEGY_UNKNOWN && (encoded_size += PB._encoded_size(x.profile_generation_strategy, 5))
    return encoded_size
end

struct MeshProto
    axes::Vector{var"MeshProto.MeshAxis"}
    device_ids::Vector{Int64}
end
MeshProto(;axes = Vector{var"MeshProto.MeshAxis"}(), device_ids = Vector{Int64}()) = MeshProto(axes, device_ids)
PB.default_values(::Type{MeshProto}) = (;axes = Vector{var"MeshProto.MeshAxis"}(), device_ids = Vector{Int64}())
PB.field_numbers(::Type{MeshProto}) = (;axes = 1, device_ids = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:MeshProto})
    axes = PB.BufferedVector{var"MeshProto.MeshAxis"}()
    device_ids = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, axes)
        elseif field_number == 2
            PB.decode!(d, wire_type, device_ids)
        else
            Base.skip(d, wire_type)
        end
    end
    return MeshProto(axes[], device_ids[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::MeshProto)
    initpos = position(e.io)
    !isempty(x.axes) && PB.encode(e, 1, x.axes)
    !isempty(x.device_ids) && PB.encode(e, 2, x.device_ids)
    return position(e.io) - initpos
end
function PB._encoded_size(x::MeshProto)
    encoded_size = 0
    !isempty(x.axes) && (encoded_size += PB._encoded_size(x.axes, 1))
    !isempty(x.device_ids) && (encoded_size += PB._encoded_size(x.device_ids, 2))
    return encoded_size
end

struct OriginalValueProto
    elements::Vector{OriginalValueElementProto}
    is_synthetic_call::Bool
end
OriginalValueProto(;elements = Vector{OriginalValueElementProto}(), is_synthetic_call = false) = OriginalValueProto(elements, is_synthetic_call)
PB.default_values(::Type{OriginalValueProto}) = (;elements = Vector{OriginalValueElementProto}(), is_synthetic_call = false)
PB.field_numbers(::Type{OriginalValueProto}) = (;elements = 1, is_synthetic_call = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OriginalValueProto})
    elements = PB.BufferedVector{OriginalValueElementProto}()
    is_synthetic_call = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, elements)
        elseif field_number == 2
            is_synthetic_call = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return OriginalValueProto(elements[], is_synthetic_call)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OriginalValueProto)
    initpos = position(e.io)
    !isempty(x.elements) && PB.encode(e, 1, x.elements)
    x.is_synthetic_call != false && PB.encode(e, 2, x.is_synthetic_call)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OriginalValueProto)
    encoded_size = 0
    !isempty(x.elements) && (encoded_size += PB._encoded_size(x.elements, 1))
    x.is_synthetic_call != false && (encoded_size += PB._encoded_size(x.is_synthetic_call, 2))
    return encoded_size
end

struct GemmPerfTable
    entries::Dict{String,GemmPerfTableEntryValues}
end
GemmPerfTable(;entries = Dict{String,GemmPerfTableEntryValues}()) = GemmPerfTable(entries)
PB.default_values(::Type{GemmPerfTable}) = (;entries = Dict{String,GemmPerfTableEntryValues}())
PB.field_numbers(::Type{GemmPerfTable}) = (;entries = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GemmPerfTable})
    entries = Dict{String,GemmPerfTableEntryValues}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, entries)
        else
            Base.skip(d, wire_type)
        end
    end
    return GemmPerfTable(entries)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GemmPerfTable)
    initpos = position(e.io)
    !isempty(x.entries) && PB.encode(e, 1, x.entries)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GemmPerfTable)
    encoded_size = 0
    !isempty(x.entries) && (encoded_size += PB._encoded_size(x.entries, 1))
    return encoded_size
end

struct var"NamedShardingProto.DimensionSharding"
    axes::Vector{AxisRefProto}
    is_closed::Bool
end
var"NamedShardingProto.DimensionSharding"(;axes = Vector{AxisRefProto}(), is_closed = false) = var"NamedShardingProto.DimensionSharding"(axes, is_closed)
PB.default_values(::Type{var"NamedShardingProto.DimensionSharding"}) = (;axes = Vector{AxisRefProto}(), is_closed = false)
PB.field_numbers(::Type{var"NamedShardingProto.DimensionSharding"}) = (;axes = 1, is_closed = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"NamedShardingProto.DimensionSharding"})
    axes = PB.BufferedVector{AxisRefProto}()
    is_closed = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, axes)
        elseif field_number == 2
            is_closed = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"NamedShardingProto.DimensionSharding"(axes[], is_closed)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"NamedShardingProto.DimensionSharding")
    initpos = position(e.io)
    !isempty(x.axes) && PB.encode(e, 1, x.axes)
    x.is_closed != false && PB.encode(e, 2, x.is_closed)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"NamedShardingProto.DimensionSharding")
    encoded_size = 0
    !isempty(x.axes) && (encoded_size += PB._encoded_size(x.axes, 1))
    x.is_closed != false && (encoded_size += PB._encoded_size(x.is_closed, 2))
    return encoded_size
end

struct OpMetadata
    op_type::String
    op_name::String
    source_file::String
    source_line::Int32
    source_end_line::Int32
    source_column::Int32
    source_end_column::Int32
    profile_type::Vector{ProfileType.T}
    size_of_generated_code_in_bytes::Int64
    size_of_memory_working_set_in_bytes::Int64
    profile_info::Union{Nothing,var"OpMetadata.ProfileInfo"}
    deduplicated_name::String
    stack_frame_id::Int32
    scheduling_name::String
end
OpMetadata(;op_type = "", op_name = "", source_file = "", source_line = zero(Int32), source_end_line = zero(Int32), source_column = zero(Int32), source_end_column = zero(Int32), profile_type = Vector{ProfileType.T}(), size_of_generated_code_in_bytes = zero(Int64), size_of_memory_working_set_in_bytes = zero(Int64), profile_info = nothing, deduplicated_name = "", stack_frame_id = zero(Int32), scheduling_name = "") = OpMetadata(op_type, op_name, source_file, source_line, source_end_line, source_column, source_end_column, profile_type, size_of_generated_code_in_bytes, size_of_memory_working_set_in_bytes, profile_info, deduplicated_name, stack_frame_id, scheduling_name)
PB.reserved_fields(::Type{OpMetadata}) = (names = ["creation_pass_id", "logical_creation_pass_id"], numbers = Union{Int,UnitRange{Int}}[6, 7, 11, 13, 14])
PB.default_values(::Type{OpMetadata}) = (;op_type = "", op_name = "", source_file = "", source_line = zero(Int32), source_end_line = zero(Int32), source_column = zero(Int32), source_end_column = zero(Int32), profile_type = Vector{ProfileType.T}(), size_of_generated_code_in_bytes = zero(Int64), size_of_memory_working_set_in_bytes = zero(Int64), profile_info = nothing, deduplicated_name = "", stack_frame_id = zero(Int32), scheduling_name = "")
PB.field_numbers(::Type{OpMetadata}) = (;op_type = 1, op_name = 2, source_file = 3, source_line = 4, source_end_line = 17, source_column = 18, source_end_column = 19, profile_type = 5, size_of_generated_code_in_bytes = 8, size_of_memory_working_set_in_bytes = 9, profile_info = 10, deduplicated_name = 12, stack_frame_id = 15, scheduling_name = 16)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OpMetadata})
    op_type = ""
    op_name = ""
    source_file = ""
    source_line = zero(Int32)
    source_end_line = zero(Int32)
    source_column = zero(Int32)
    source_end_column = zero(Int32)
    profile_type = PB.BufferedVector{ProfileType.T}()
    size_of_generated_code_in_bytes = zero(Int64)
    size_of_memory_working_set_in_bytes = zero(Int64)
    profile_info = Ref{Union{Nothing,var"OpMetadata.ProfileInfo"}}(nothing)
    deduplicated_name = ""
    stack_frame_id = zero(Int32)
    scheduling_name = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            op_type = PB.decode(d, String)
        elseif field_number == 2
            op_name = PB.decode(d, String)
        elseif field_number == 3
            source_file = PB.decode(d, String)
        elseif field_number == 4
            source_line = PB.decode(d, Int32)
        elseif field_number == 17
            source_end_line = PB.decode(d, Int32)
        elseif field_number == 18
            source_column = PB.decode(d, Int32)
        elseif field_number == 19
            source_end_column = PB.decode(d, Int32)
        elseif field_number == 5
            PB.decode!(d, wire_type, profile_type)
        elseif field_number == 8
            size_of_generated_code_in_bytes = PB.decode(d, Int64)
        elseif field_number == 9
            size_of_memory_working_set_in_bytes = PB.decode(d, Int64)
        elseif field_number == 10
            PB.decode!(d, profile_info)
        elseif field_number == 12
            deduplicated_name = PB.decode(d, String)
        elseif field_number == 15
            stack_frame_id = PB.decode(d, Int32)
        elseif field_number == 16
            scheduling_name = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return OpMetadata(op_type, op_name, source_file, source_line, source_end_line, source_column, source_end_column, profile_type[], size_of_generated_code_in_bytes, size_of_memory_working_set_in_bytes, profile_info[], deduplicated_name, stack_frame_id, scheduling_name)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OpMetadata)
    initpos = position(e.io)
    !isempty(x.op_type) && PB.encode(e, 1, x.op_type)
    !isempty(x.op_name) && PB.encode(e, 2, x.op_name)
    !isempty(x.source_file) && PB.encode(e, 3, x.source_file)
    x.source_line != zero(Int32) && PB.encode(e, 4, x.source_line)
    x.source_end_line != zero(Int32) && PB.encode(e, 17, x.source_end_line)
    x.source_column != zero(Int32) && PB.encode(e, 18, x.source_column)
    x.source_end_column != zero(Int32) && PB.encode(e, 19, x.source_end_column)
    !isempty(x.profile_type) && PB.encode(e, 5, x.profile_type)
    x.size_of_generated_code_in_bytes != zero(Int64) && PB.encode(e, 8, x.size_of_generated_code_in_bytes)
    x.size_of_memory_working_set_in_bytes != zero(Int64) && PB.encode(e, 9, x.size_of_memory_working_set_in_bytes)
    !isnothing(x.profile_info) && PB.encode(e, 10, x.profile_info)
    !isempty(x.deduplicated_name) && PB.encode(e, 12, x.deduplicated_name)
    x.stack_frame_id != zero(Int32) && PB.encode(e, 15, x.stack_frame_id)
    !isempty(x.scheduling_name) && PB.encode(e, 16, x.scheduling_name)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OpMetadata)
    encoded_size = 0
    !isempty(x.op_type) && (encoded_size += PB._encoded_size(x.op_type, 1))
    !isempty(x.op_name) && (encoded_size += PB._encoded_size(x.op_name, 2))
    !isempty(x.source_file) && (encoded_size += PB._encoded_size(x.source_file, 3))
    x.source_line != zero(Int32) && (encoded_size += PB._encoded_size(x.source_line, 4))
    x.source_end_line != zero(Int32) && (encoded_size += PB._encoded_size(x.source_end_line, 17))
    x.source_column != zero(Int32) && (encoded_size += PB._encoded_size(x.source_column, 18))
    x.source_end_column != zero(Int32) && (encoded_size += PB._encoded_size(x.source_end_column, 19))
    !isempty(x.profile_type) && (encoded_size += PB._encoded_size(x.profile_type, 5))
    x.size_of_generated_code_in_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.size_of_generated_code_in_bytes, 8))
    x.size_of_memory_working_set_in_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.size_of_memory_working_set_in_bytes, 9))
    !isnothing(x.profile_info) && (encoded_size += PB._encoded_size(x.profile_info, 10))
    !isempty(x.deduplicated_name) && (encoded_size += PB._encoded_size(x.deduplicated_name, 12))
    x.stack_frame_id != zero(Int32) && (encoded_size += PB._encoded_size(x.stack_frame_id, 15))
    !isempty(x.scheduling_name) && (encoded_size += PB._encoded_size(x.scheduling_name, 16))
    return encoded_size
end

struct MeshAxesReplicaGroupListProto
    mesh::Union{Nothing,MeshProto}
    axes::Vector{AxisRefProto}
end
MeshAxesReplicaGroupListProto(;mesh = nothing, axes = Vector{AxisRefProto}()) = MeshAxesReplicaGroupListProto(mesh, axes)
PB.default_values(::Type{MeshAxesReplicaGroupListProto}) = (;mesh = nothing, axes = Vector{AxisRefProto}())
PB.field_numbers(::Type{MeshAxesReplicaGroupListProto}) = (;mesh = 1, axes = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:MeshAxesReplicaGroupListProto})
    mesh = Ref{Union{Nothing,MeshProto}}(nothing)
    axes = PB.BufferedVector{AxisRefProto}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, mesh)
        elseif field_number == 2
            PB.decode!(d, axes)
        else
            Base.skip(d, wire_type)
        end
    end
    return MeshAxesReplicaGroupListProto(mesh[], axes[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::MeshAxesReplicaGroupListProto)
    initpos = position(e.io)
    !isnothing(x.mesh) && PB.encode(e, 1, x.mesh)
    !isempty(x.axes) && PB.encode(e, 2, x.axes)
    return position(e.io) - initpos
end
function PB._encoded_size(x::MeshAxesReplicaGroupListProto)
    encoded_size = 0
    !isnothing(x.mesh) && (encoded_size += PB._encoded_size(x.mesh, 1))
    !isempty(x.axes) && (encoded_size += PB._encoded_size(x.axes, 2))
    return encoded_size
end

struct NamedShardingProto
    mesh::Union{Nothing,MeshProto}
    dim_shardings::Vector{var"NamedShardingProto.DimensionSharding"}
    replicated_axes::Vector{AxisRefProto}
    unreduced_axes::Vector{AxisRefProto}
    metadata::Vector{OpMetadata}
end
NamedShardingProto(;mesh = nothing, dim_shardings = Vector{var"NamedShardingProto.DimensionSharding"}(), replicated_axes = Vector{AxisRefProto}(), unreduced_axes = Vector{AxisRefProto}(), metadata = Vector{OpMetadata}()) = NamedShardingProto(mesh, dim_shardings, replicated_axes, unreduced_axes, metadata)
PB.reserved_fields(::Type{NamedShardingProto}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[1])
PB.default_values(::Type{NamedShardingProto}) = (;mesh = nothing, dim_shardings = Vector{var"NamedShardingProto.DimensionSharding"}(), replicated_axes = Vector{AxisRefProto}(), unreduced_axes = Vector{AxisRefProto}(), metadata = Vector{OpMetadata}())
PB.field_numbers(::Type{NamedShardingProto}) = (;mesh = 2, dim_shardings = 3, replicated_axes = 4, unreduced_axes = 5, metadata = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:NamedShardingProto})
    mesh = Ref{Union{Nothing,MeshProto}}(nothing)
    dim_shardings = PB.BufferedVector{var"NamedShardingProto.DimensionSharding"}()
    replicated_axes = PB.BufferedVector{AxisRefProto}()
    unreduced_axes = PB.BufferedVector{AxisRefProto}()
    metadata = PB.BufferedVector{OpMetadata}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 2
            PB.decode!(d, mesh)
        elseif field_number == 3
            PB.decode!(d, dim_shardings)
        elseif field_number == 4
            PB.decode!(d, replicated_axes)
        elseif field_number == 5
            PB.decode!(d, unreduced_axes)
        elseif field_number == 6
            PB.decode!(d, metadata)
        else
            Base.skip(d, wire_type)
        end
    end
    return NamedShardingProto(mesh[], dim_shardings[], replicated_axes[], unreduced_axes[], metadata[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::NamedShardingProto)
    initpos = position(e.io)
    !isnothing(x.mesh) && PB.encode(e, 2, x.mesh)
    !isempty(x.dim_shardings) && PB.encode(e, 3, x.dim_shardings)
    !isempty(x.replicated_axes) && PB.encode(e, 4, x.replicated_axes)
    !isempty(x.unreduced_axes) && PB.encode(e, 5, x.unreduced_axes)
    !isempty(x.metadata) && PB.encode(e, 6, x.metadata)
    return position(e.io) - initpos
end
function PB._encoded_size(x::NamedShardingProto)
    encoded_size = 0
    !isnothing(x.mesh) && (encoded_size += PB._encoded_size(x.mesh, 2))
    !isempty(x.dim_shardings) && (encoded_size += PB._encoded_size(x.dim_shardings, 3))
    !isempty(x.replicated_axes) && (encoded_size += PB._encoded_size(x.replicated_axes, 4))
    !isempty(x.unreduced_axes) && (encoded_size += PB._encoded_size(x.unreduced_axes, 5))
    !isempty(x.metadata) && (encoded_size += PB._encoded_size(x.metadata, 6))
    return encoded_size
end

# Stub definitions for cyclic types
struct var"##Stub#LayoutProto"{T1<:var"##Abstract#ShapeProto"} <: var"##Abstract#LayoutProto"
    minor_to_major::Vector{Int64}
    dim_level_types::Vector{DimLevelType.T}
    dim_unique::Vector{Bool}
    dim_ordered::Vector{Bool}
    tiles::Vector{TileProto}
    tail_padding_alignment_in_elements::Int64
    element_size_in_bits::Int64
    memory_space::Int64
    index_primitive_type::PrimitiveType.T
    pointer_primitive_type::PrimitiveType.T
    physical_shape::Union{Nothing,T1}
    dynamic_shape_metadata_prefix_bytes::Int64
    split_configs::Vector{SplitConfigProto}
end

struct var"##Stub#LiteralProto"{T1<:var"##Abstract#ShapeProto"} <: var"##Abstract#LiteralProto"
    shape::Union{Nothing,T1}
    preds::Vector{Bool}
    s1s::Vector{UInt8}
    s2s::Vector{UInt8}
    s4s::Vector{UInt8}
    s8s::Vector{UInt8}
    u1s::Vector{UInt8}
    u2s::Vector{UInt8}
    u4s::Vector{UInt8}
    u8s::Vector{UInt8}
    s32s::Vector{Int32}
    s64s::Vector{Int64}
    u32s::Vector{UInt32}
    u64s::Vector{UInt64}
    f32s::Vector{Float32}
    f64s::Vector{Float64}
    c64s::Vector{Float32}
    c128s::Vector{Float64}
    tuple_literals::Vector{var"##Stub#LiteralProto"{T1}}
    f16s::Vector{UInt8}
    bf16s::Vector{UInt8}
    u16s::Vector{UInt8}
    s16s::Vector{UInt8}
    f4e2m1fns::Vector{UInt8}
    f8e3m4s::Vector{UInt8}
    f8e4m3b11fnuzs::Vector{UInt8}
    f8e4m3fns::Vector{UInt8}
    f8e4m3fnuzs::Vector{UInt8}
    f8e4m3s::Vector{UInt8}
    f8e5m2fnuzs::Vector{UInt8}
    f8e5m2s::Vector{UInt8}
    f8e8m0fnus::Vector{UInt8}
    sparse_indices::Vector{Int64}
end

struct var"##Stub#OpSharding"{T1<:var"##Abstract#ShapeProto"} <: var"##Abstract#OpSharding"
    var"#type"::var"OpSharding.Type".T
    tile_shape::Union{Nothing,T1}
    tile_assignment_dimensions::Vector{Int64}
    tile_assignment_devices::Vector{Int64}
    tuple_shardings::Vector{var"##Stub#OpSharding"{T1}}
    replicate_on_last_tile_dim::Bool
    metadata::Vector{OpMetadata}
    last_tile_dims::Vector{var"OpSharding.Type".T}
    iota_reshape_dims::Vector{Int64}
    iota_transpose_perm::Vector{Int32}
    is_shard_group::Bool
    shard_group_id::Int64
    shard_group_type::var"OpSharding.ShardGroupType".T
    named_sharding::Union{Nothing,NamedShardingProto}
end

struct var"##Stub#ProgramShapeProto"{T1<:var"##Abstract#ShapeProto"} <: var"##Abstract#ProgramShapeProto"
    parameters::Vector{T1}
    result::Union{Nothing,T1}
    parameter_names::Vector{String}
end

struct var"##Stub#ShapeProto" <: var"##Abstract#ShapeProto"
    element_type::PrimitiveType.T
    dimensions::Vector{Int64}
    is_dynamic_dimension::Vector{Bool}
    tuple_shapes::Vector{var"##Stub#ShapeProto"}
    layout::Union{Nothing,var"##Stub#LayoutProto"{var"##Stub#ShapeProto"}}
end

const LayoutProto = var"##Stub#LayoutProto"{var"##Stub#ShapeProto"}
LayoutProto(;minor_to_major = Vector{Int64}(), dim_level_types = Vector{DimLevelType.T}(), dim_unique = Vector{Bool}(), dim_ordered = Vector{Bool}(), tiles = Vector{TileProto}(), tail_padding_alignment_in_elements = zero(Int64), element_size_in_bits = zero(Int64), memory_space = zero(Int64), index_primitive_type = PrimitiveType.PRIMITIVE_TYPE_INVALID, pointer_primitive_type = PrimitiveType.PRIMITIVE_TYPE_INVALID, physical_shape = nothing, dynamic_shape_metadata_prefix_bytes = zero(Int64), split_configs = Vector{SplitConfigProto}()) = LayoutProto(minor_to_major, dim_level_types, dim_unique, dim_ordered, tiles, tail_padding_alignment_in_elements, element_size_in_bits, memory_space, index_primitive_type, pointer_primitive_type, physical_shape, dynamic_shape_metadata_prefix_bytes, split_configs)
PB.reserved_fields(::Type{LayoutProto}) = (names = ["padded_dimensions", "padding_value", "format", "max_sparse_elements"], numbers = Union{Int,UnitRange{Int}}[2, 3, 4, 5])
PB.default_values(::Type{LayoutProto}) = (;minor_to_major = Vector{Int64}(), dim_level_types = Vector{DimLevelType.T}(), dim_unique = Vector{Bool}(), dim_ordered = Vector{Bool}(), tiles = Vector{TileProto}(), tail_padding_alignment_in_elements = zero(Int64), element_size_in_bits = zero(Int64), memory_space = zero(Int64), index_primitive_type = PrimitiveType.PRIMITIVE_TYPE_INVALID, pointer_primitive_type = PrimitiveType.PRIMITIVE_TYPE_INVALID, physical_shape = nothing, dynamic_shape_metadata_prefix_bytes = zero(Int64), split_configs = Vector{SplitConfigProto}())
PB.field_numbers(::Type{LayoutProto}) = (;minor_to_major = 1, dim_level_types = 9, dim_unique = 13, dim_ordered = 14, tiles = 6, tail_padding_alignment_in_elements = 16, element_size_in_bits = 7, memory_space = 8, index_primitive_type = 11, pointer_primitive_type = 12, physical_shape = 10, dynamic_shape_metadata_prefix_bytes = 15, split_configs = 17)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:LayoutProto})
    minor_to_major = PB.BufferedVector{Int64}()
    dim_level_types = PB.BufferedVector{DimLevelType.T}()
    dim_unique = PB.BufferedVector{Bool}()
    dim_ordered = PB.BufferedVector{Bool}()
    tiles = PB.BufferedVector{TileProto}()
    tail_padding_alignment_in_elements = zero(Int64)
    element_size_in_bits = zero(Int64)
    memory_space = zero(Int64)
    index_primitive_type = PrimitiveType.PRIMITIVE_TYPE_INVALID
    pointer_primitive_type = PrimitiveType.PRIMITIVE_TYPE_INVALID
    physical_shape = Ref{Union{Nothing,ShapeProto}}(nothing)
    dynamic_shape_metadata_prefix_bytes = zero(Int64)
    split_configs = PB.BufferedVector{SplitConfigProto}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, minor_to_major)
        elseif field_number == 9
            PB.decode!(d, wire_type, dim_level_types)
        elseif field_number == 13
            PB.decode!(d, wire_type, dim_unique)
        elseif field_number == 14
            PB.decode!(d, wire_type, dim_ordered)
        elseif field_number == 6
            PB.decode!(d, tiles)
        elseif field_number == 16
            tail_padding_alignment_in_elements = PB.decode(d, Int64)
        elseif field_number == 7
            element_size_in_bits = PB.decode(d, Int64)
        elseif field_number == 8
            memory_space = PB.decode(d, Int64)
        elseif field_number == 11
            index_primitive_type = PB.decode(d, PrimitiveType.T)
        elseif field_number == 12
            pointer_primitive_type = PB.decode(d, PrimitiveType.T)
        elseif field_number == 10
            PB.decode!(d, physical_shape)
        elseif field_number == 15
            dynamic_shape_metadata_prefix_bytes = PB.decode(d, Int64)
        elseif field_number == 17
            PB.decode!(d, split_configs)
        else
            Base.skip(d, wire_type)
        end
    end
    return LayoutProto(minor_to_major[], dim_level_types[], dim_unique[], dim_ordered[], tiles[], tail_padding_alignment_in_elements, element_size_in_bits, memory_space, index_primitive_type, pointer_primitive_type, physical_shape[], dynamic_shape_metadata_prefix_bytes, split_configs[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::LayoutProto)
    initpos = position(e.io)
    !isempty(x.minor_to_major) && PB.encode(e, 1, x.minor_to_major)
    !isempty(x.dim_level_types) && PB.encode(e, 9, x.dim_level_types)
    !isempty(x.dim_unique) && PB.encode(e, 13, x.dim_unique)
    !isempty(x.dim_ordered) && PB.encode(e, 14, x.dim_ordered)
    !isempty(x.tiles) && PB.encode(e, 6, x.tiles)
    x.tail_padding_alignment_in_elements != zero(Int64) && PB.encode(e, 16, x.tail_padding_alignment_in_elements)
    x.element_size_in_bits != zero(Int64) && PB.encode(e, 7, x.element_size_in_bits)
    x.memory_space != zero(Int64) && PB.encode(e, 8, x.memory_space)
    x.index_primitive_type != PrimitiveType.PRIMITIVE_TYPE_INVALID && PB.encode(e, 11, x.index_primitive_type)
    x.pointer_primitive_type != PrimitiveType.PRIMITIVE_TYPE_INVALID && PB.encode(e, 12, x.pointer_primitive_type)
    !isnothing(x.physical_shape) && PB.encode(e, 10, x.physical_shape)
    x.dynamic_shape_metadata_prefix_bytes != zero(Int64) && PB.encode(e, 15, x.dynamic_shape_metadata_prefix_bytes)
    !isempty(x.split_configs) && PB.encode(e, 17, x.split_configs)
    return position(e.io) - initpos
end
function PB._encoded_size(x::LayoutProto)
    encoded_size = 0
    !isempty(x.minor_to_major) && (encoded_size += PB._encoded_size(x.minor_to_major, 1))
    !isempty(x.dim_level_types) && (encoded_size += PB._encoded_size(x.dim_level_types, 9))
    !isempty(x.dim_unique) && (encoded_size += PB._encoded_size(x.dim_unique, 13))
    !isempty(x.dim_ordered) && (encoded_size += PB._encoded_size(x.dim_ordered, 14))
    !isempty(x.tiles) && (encoded_size += PB._encoded_size(x.tiles, 6))
    x.tail_padding_alignment_in_elements != zero(Int64) && (encoded_size += PB._encoded_size(x.tail_padding_alignment_in_elements, 16))
    x.element_size_in_bits != zero(Int64) && (encoded_size += PB._encoded_size(x.element_size_in_bits, 7))
    x.memory_space != zero(Int64) && (encoded_size += PB._encoded_size(x.memory_space, 8))
    x.index_primitive_type != PrimitiveType.PRIMITIVE_TYPE_INVALID && (encoded_size += PB._encoded_size(x.index_primitive_type, 11))
    x.pointer_primitive_type != PrimitiveType.PRIMITIVE_TYPE_INVALID && (encoded_size += PB._encoded_size(x.pointer_primitive_type, 12))
    !isnothing(x.physical_shape) && (encoded_size += PB._encoded_size(x.physical_shape, 10))
    x.dynamic_shape_metadata_prefix_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.dynamic_shape_metadata_prefix_bytes, 15))
    !isempty(x.split_configs) && (encoded_size += PB._encoded_size(x.split_configs, 17))
    return encoded_size
end

const LiteralProto = var"##Stub#LiteralProto"{var"##Stub#ShapeProto"}
LiteralProto(;shape = nothing, preds = Vector{Bool}(), s1s = UInt8[], s2s = UInt8[], s4s = UInt8[], s8s = UInt8[], u1s = UInt8[], u2s = UInt8[], u4s = UInt8[], u8s = UInt8[], s32s = Vector{Int32}(), s64s = Vector{Int64}(), u32s = Vector{UInt32}(), u64s = Vector{UInt64}(), f32s = Vector{Float32}(), f64s = Vector{Float64}(), c64s = Vector{Float32}(), c128s = Vector{Float64}(), tuple_literals = Vector{LiteralProto}(), f16s = UInt8[], bf16s = UInt8[], u16s = UInt8[], s16s = UInt8[], f4e2m1fns = UInt8[], f8e3m4s = UInt8[], f8e4m3b11fnuzs = UInt8[], f8e4m3fns = UInt8[], f8e4m3fnuzs = UInt8[], f8e4m3s = UInt8[], f8e5m2fnuzs = UInt8[], f8e5m2s = UInt8[], f8e8m0fnus = UInt8[], sparse_indices = Vector{Int64}()) = LiteralProto(shape, preds, s1s, s2s, s4s, s8s, u1s, u2s, u4s, u8s, s32s, s64s, u32s, u64s, f32s, f64s, c64s, c128s, tuple_literals, f16s, bf16s, u16s, s16s, f4e2m1fns, f8e3m4s, f8e4m3b11fnuzs, f8e4m3fns, f8e4m3fnuzs, f8e4m3s, f8e5m2fnuzs, f8e5m2s, f8e8m0fnus, sparse_indices)
PB.default_values(::Type{LiteralProto}) = (;shape = nothing, preds = Vector{Bool}(), s1s = UInt8[], s2s = UInt8[], s4s = UInt8[], s8s = UInt8[], u1s = UInt8[], u2s = UInt8[], u4s = UInt8[], u8s = UInt8[], s32s = Vector{Int32}(), s64s = Vector{Int64}(), u32s = Vector{UInt32}(), u64s = Vector{UInt64}(), f32s = Vector{Float32}(), f64s = Vector{Float64}(), c64s = Vector{Float32}(), c128s = Vector{Float64}(), tuple_literals = Vector{LiteralProto}(), f16s = UInt8[], bf16s = UInt8[], u16s = UInt8[], s16s = UInt8[], f4e2m1fns = UInt8[], f8e3m4s = UInt8[], f8e4m3b11fnuzs = UInt8[], f8e4m3fns = UInt8[], f8e4m3fnuzs = UInt8[], f8e4m3s = UInt8[], f8e5m2fnuzs = UInt8[], f8e5m2s = UInt8[], f8e8m0fnus = UInt8[], sparse_indices = Vector{Int64}())
PB.field_numbers(::Type{LiteralProto}) = (;shape = 1, preds = 2, s1s = 30, s2s = 26, s4s = 21, s8s = 15, u1s = 31, u2s = 27, u4s = 22, u8s = 3, s32s = 4, s64s = 5, u32s = 6, u64s = 7, f32s = 8, f64s = 9, c64s = 12, c128s = 18, tuple_literals = 10, f16s = 11, bf16s = 13, u16s = 16, s16s = 17, f4e2m1fns = 32, f8e3m4s = 29, f8e4m3b11fnuzs = 23, f8e4m3fns = 20, f8e4m3fnuzs = 25, f8e4m3s = 28, f8e5m2fnuzs = 24, f8e5m2s = 19, f8e8m0fnus = 33, sparse_indices = 14)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:LiteralProto})
    shape = Ref{Union{Nothing,ShapeProto}}(nothing)
    preds = PB.BufferedVector{Bool}()
    s1s = UInt8[]
    s2s = UInt8[]
    s4s = UInt8[]
    s8s = UInt8[]
    u1s = UInt8[]
    u2s = UInt8[]
    u4s = UInt8[]
    u8s = UInt8[]
    s32s = PB.BufferedVector{Int32}()
    s64s = PB.BufferedVector{Int64}()
    u32s = PB.BufferedVector{UInt32}()
    u64s = PB.BufferedVector{UInt64}()
    f32s = PB.BufferedVector{Float32}()
    f64s = PB.BufferedVector{Float64}()
    c64s = PB.BufferedVector{Float32}()
    c128s = PB.BufferedVector{Float64}()
    tuple_literals = PB.BufferedVector{LiteralProto}()
    f16s = UInt8[]
    bf16s = UInt8[]
    u16s = UInt8[]
    s16s = UInt8[]
    f4e2m1fns = UInt8[]
    f8e3m4s = UInt8[]
    f8e4m3b11fnuzs = UInt8[]
    f8e4m3fns = UInt8[]
    f8e4m3fnuzs = UInt8[]
    f8e4m3s = UInt8[]
    f8e5m2fnuzs = UInt8[]
    f8e5m2s = UInt8[]
    f8e8m0fnus = UInt8[]
    sparse_indices = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, shape)
        elseif field_number == 2
            PB.decode!(d, wire_type, preds)
        elseif field_number == 30
            s1s = PB.decode(d, Vector{UInt8})
        elseif field_number == 26
            s2s = PB.decode(d, Vector{UInt8})
        elseif field_number == 21
            s4s = PB.decode(d, Vector{UInt8})
        elseif field_number == 15
            s8s = PB.decode(d, Vector{UInt8})
        elseif field_number == 31
            u1s = PB.decode(d, Vector{UInt8})
        elseif field_number == 27
            u2s = PB.decode(d, Vector{UInt8})
        elseif field_number == 22
            u4s = PB.decode(d, Vector{UInt8})
        elseif field_number == 3
            u8s = PB.decode(d, Vector{UInt8})
        elseif field_number == 4
            PB.decode!(d, wire_type, s32s)
        elseif field_number == 5
            PB.decode!(d, wire_type, s64s)
        elseif field_number == 6
            PB.decode!(d, wire_type, u32s)
        elseif field_number == 7
            PB.decode!(d, wire_type, u64s)
        elseif field_number == 8
            PB.decode!(d, wire_type, f32s)
        elseif field_number == 9
            PB.decode!(d, wire_type, f64s)
        elseif field_number == 12
            PB.decode!(d, wire_type, c64s)
        elseif field_number == 18
            PB.decode!(d, wire_type, c128s)
        elseif field_number == 10
            PB.decode!(d, tuple_literals)
        elseif field_number == 11
            f16s = PB.decode(d, Vector{UInt8})
        elseif field_number == 13
            bf16s = PB.decode(d, Vector{UInt8})
        elseif field_number == 16
            u16s = PB.decode(d, Vector{UInt8})
        elseif field_number == 17
            s16s = PB.decode(d, Vector{UInt8})
        elseif field_number == 32
            f4e2m1fns = PB.decode(d, Vector{UInt8})
        elseif field_number == 29
            f8e3m4s = PB.decode(d, Vector{UInt8})
        elseif field_number == 23
            f8e4m3b11fnuzs = PB.decode(d, Vector{UInt8})
        elseif field_number == 20
            f8e4m3fns = PB.decode(d, Vector{UInt8})
        elseif field_number == 25
            f8e4m3fnuzs = PB.decode(d, Vector{UInt8})
        elseif field_number == 28
            f8e4m3s = PB.decode(d, Vector{UInt8})
        elseif field_number == 24
            f8e5m2fnuzs = PB.decode(d, Vector{UInt8})
        elseif field_number == 19
            f8e5m2s = PB.decode(d, Vector{UInt8})
        elseif field_number == 33
            f8e8m0fnus = PB.decode(d, Vector{UInt8})
        elseif field_number == 14
            PB.decode!(d, wire_type, sparse_indices)
        else
            Base.skip(d, wire_type)
        end
    end
    return LiteralProto(shape[], preds[], s1s, s2s, s4s, s8s, u1s, u2s, u4s, u8s, s32s[], s64s[], u32s[], u64s[], f32s[], f64s[], c64s[], c128s[], tuple_literals[], f16s, bf16s, u16s, s16s, f4e2m1fns, f8e3m4s, f8e4m3b11fnuzs, f8e4m3fns, f8e4m3fnuzs, f8e4m3s, f8e5m2fnuzs, f8e5m2s, f8e8m0fnus, sparse_indices[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::LiteralProto)
    initpos = position(e.io)
    !isnothing(x.shape) && PB.encode(e, 1, x.shape)
    !isempty(x.preds) && PB.encode(e, 2, x.preds)
    !isempty(x.s1s) && PB.encode(e, 30, x.s1s)
    !isempty(x.s2s) && PB.encode(e, 26, x.s2s)
    !isempty(x.s4s) && PB.encode(e, 21, x.s4s)
    !isempty(x.s8s) && PB.encode(e, 15, x.s8s)
    !isempty(x.u1s) && PB.encode(e, 31, x.u1s)
    !isempty(x.u2s) && PB.encode(e, 27, x.u2s)
    !isempty(x.u4s) && PB.encode(e, 22, x.u4s)
    !isempty(x.u8s) && PB.encode(e, 3, x.u8s)
    !isempty(x.s32s) && PB.encode(e, 4, x.s32s)
    !isempty(x.s64s) && PB.encode(e, 5, x.s64s)
    !isempty(x.u32s) && PB.encode(e, 6, x.u32s)
    !isempty(x.u64s) && PB.encode(e, 7, x.u64s)
    !isempty(x.f32s) && PB.encode(e, 8, x.f32s)
    !isempty(x.f64s) && PB.encode(e, 9, x.f64s)
    !isempty(x.c64s) && PB.encode(e, 12, x.c64s)
    !isempty(x.c128s) && PB.encode(e, 18, x.c128s)
    !isempty(x.tuple_literals) && PB.encode(e, 10, x.tuple_literals)
    !isempty(x.f16s) && PB.encode(e, 11, x.f16s)
    !isempty(x.bf16s) && PB.encode(e, 13, x.bf16s)
    !isempty(x.u16s) && PB.encode(e, 16, x.u16s)
    !isempty(x.s16s) && PB.encode(e, 17, x.s16s)
    !isempty(x.f4e2m1fns) && PB.encode(e, 32, x.f4e2m1fns)
    !isempty(x.f8e3m4s) && PB.encode(e, 29, x.f8e3m4s)
    !isempty(x.f8e4m3b11fnuzs) && PB.encode(e, 23, x.f8e4m3b11fnuzs)
    !isempty(x.f8e4m3fns) && PB.encode(e, 20, x.f8e4m3fns)
    !isempty(x.f8e4m3fnuzs) && PB.encode(e, 25, x.f8e4m3fnuzs)
    !isempty(x.f8e4m3s) && PB.encode(e, 28, x.f8e4m3s)
    !isempty(x.f8e5m2fnuzs) && PB.encode(e, 24, x.f8e5m2fnuzs)
    !isempty(x.f8e5m2s) && PB.encode(e, 19, x.f8e5m2s)
    !isempty(x.f8e8m0fnus) && PB.encode(e, 33, x.f8e8m0fnus)
    !isempty(x.sparse_indices) && PB.encode(e, 14, x.sparse_indices)
    return position(e.io) - initpos
end
function PB._encoded_size(x::LiteralProto)
    encoded_size = 0
    !isnothing(x.shape) && (encoded_size += PB._encoded_size(x.shape, 1))
    !isempty(x.preds) && (encoded_size += PB._encoded_size(x.preds, 2))
    !isempty(x.s1s) && (encoded_size += PB._encoded_size(x.s1s, 30))
    !isempty(x.s2s) && (encoded_size += PB._encoded_size(x.s2s, 26))
    !isempty(x.s4s) && (encoded_size += PB._encoded_size(x.s4s, 21))
    !isempty(x.s8s) && (encoded_size += PB._encoded_size(x.s8s, 15))
    !isempty(x.u1s) && (encoded_size += PB._encoded_size(x.u1s, 31))
    !isempty(x.u2s) && (encoded_size += PB._encoded_size(x.u2s, 27))
    !isempty(x.u4s) && (encoded_size += PB._encoded_size(x.u4s, 22))
    !isempty(x.u8s) && (encoded_size += PB._encoded_size(x.u8s, 3))
    !isempty(x.s32s) && (encoded_size += PB._encoded_size(x.s32s, 4))
    !isempty(x.s64s) && (encoded_size += PB._encoded_size(x.s64s, 5))
    !isempty(x.u32s) && (encoded_size += PB._encoded_size(x.u32s, 6))
    !isempty(x.u64s) && (encoded_size += PB._encoded_size(x.u64s, 7))
    !isempty(x.f32s) && (encoded_size += PB._encoded_size(x.f32s, 8))
    !isempty(x.f64s) && (encoded_size += PB._encoded_size(x.f64s, 9))
    !isempty(x.c64s) && (encoded_size += PB._encoded_size(x.c64s, 12))
    !isempty(x.c128s) && (encoded_size += PB._encoded_size(x.c128s, 18))
    !isempty(x.tuple_literals) && (encoded_size += PB._encoded_size(x.tuple_literals, 10))
    !isempty(x.f16s) && (encoded_size += PB._encoded_size(x.f16s, 11))
    !isempty(x.bf16s) && (encoded_size += PB._encoded_size(x.bf16s, 13))
    !isempty(x.u16s) && (encoded_size += PB._encoded_size(x.u16s, 16))
    !isempty(x.s16s) && (encoded_size += PB._encoded_size(x.s16s, 17))
    !isempty(x.f4e2m1fns) && (encoded_size += PB._encoded_size(x.f4e2m1fns, 32))
    !isempty(x.f8e3m4s) && (encoded_size += PB._encoded_size(x.f8e3m4s, 29))
    !isempty(x.f8e4m3b11fnuzs) && (encoded_size += PB._encoded_size(x.f8e4m3b11fnuzs, 23))
    !isempty(x.f8e4m3fns) && (encoded_size += PB._encoded_size(x.f8e4m3fns, 20))
    !isempty(x.f8e4m3fnuzs) && (encoded_size += PB._encoded_size(x.f8e4m3fnuzs, 25))
    !isempty(x.f8e4m3s) && (encoded_size += PB._encoded_size(x.f8e4m3s, 28))
    !isempty(x.f8e5m2fnuzs) && (encoded_size += PB._encoded_size(x.f8e5m2fnuzs, 24))
    !isempty(x.f8e5m2s) && (encoded_size += PB._encoded_size(x.f8e5m2s, 19))
    !isempty(x.f8e8m0fnus) && (encoded_size += PB._encoded_size(x.f8e8m0fnus, 33))
    !isempty(x.sparse_indices) && (encoded_size += PB._encoded_size(x.sparse_indices, 14))
    return encoded_size
end

const OpSharding = var"##Stub#OpSharding"{var"##Stub#ShapeProto"}
OpSharding(;var"#type" = var"OpSharding.Type".REPLICATED, tile_shape = nothing, tile_assignment_dimensions = Vector{Int64}(), tile_assignment_devices = Vector{Int64}(), tuple_shardings = Vector{OpSharding}(), replicate_on_last_tile_dim = false, metadata = Vector{OpMetadata}(), last_tile_dims = Vector{var"OpSharding.Type".T}(), iota_reshape_dims = Vector{Int64}(), iota_transpose_perm = Vector{Int32}(), is_shard_group = false, shard_group_id = zero(Int64), shard_group_type = var"OpSharding.ShardGroupType".AS, named_sharding = nothing) = OpSharding(var"#type", tile_shape, tile_assignment_dimensions, tile_assignment_devices, tuple_shardings, replicate_on_last_tile_dim, metadata, last_tile_dims, iota_reshape_dims, iota_transpose_perm, is_shard_group, shard_group_id, shard_group_type, named_sharding)
PB.default_values(::Type{OpSharding}) = (;var"#type" = var"OpSharding.Type".REPLICATED, tile_shape = nothing, tile_assignment_dimensions = Vector{Int64}(), tile_assignment_devices = Vector{Int64}(), tuple_shardings = Vector{OpSharding}(), replicate_on_last_tile_dim = false, metadata = Vector{OpMetadata}(), last_tile_dims = Vector{var"OpSharding.Type".T}(), iota_reshape_dims = Vector{Int64}(), iota_transpose_perm = Vector{Int32}(), is_shard_group = false, shard_group_id = zero(Int64), shard_group_type = var"OpSharding.ShardGroupType".AS, named_sharding = nothing)
PB.field_numbers(::Type{OpSharding}) = (;var"#type" = 1, tile_shape = 2, tile_assignment_dimensions = 3, tile_assignment_devices = 4, tuple_shardings = 5, replicate_on_last_tile_dim = 6, metadata = 7, last_tile_dims = 8, iota_reshape_dims = 9, iota_transpose_perm = 10, is_shard_group = 11, shard_group_id = 12, shard_group_type = 13, named_sharding = 14)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OpSharding})
    var"#type" = var"OpSharding.Type".REPLICATED
    tile_shape = Ref{Union{Nothing,ShapeProto}}(nothing)
    tile_assignment_dimensions = PB.BufferedVector{Int64}()
    tile_assignment_devices = PB.BufferedVector{Int64}()
    tuple_shardings = PB.BufferedVector{OpSharding}()
    replicate_on_last_tile_dim = false
    metadata = PB.BufferedVector{OpMetadata}()
    last_tile_dims = PB.BufferedVector{var"OpSharding.Type".T}()
    iota_reshape_dims = PB.BufferedVector{Int64}()
    iota_transpose_perm = PB.BufferedVector{Int32}()
    is_shard_group = false
    shard_group_id = zero(Int64)
    shard_group_type = var"OpSharding.ShardGroupType".AS
    named_sharding = Ref{Union{Nothing,NamedShardingProto}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            var"#type" = PB.decode(d, var"OpSharding.Type".T)
        elseif field_number == 2
            PB.decode!(d, tile_shape)
        elseif field_number == 3
            PB.decode!(d, wire_type, tile_assignment_dimensions)
        elseif field_number == 4
            PB.decode!(d, wire_type, tile_assignment_devices)
        elseif field_number == 5
            PB.decode!(d, tuple_shardings)
        elseif field_number == 6
            replicate_on_last_tile_dim = PB.decode(d, Bool)
        elseif field_number == 7
            PB.decode!(d, metadata)
        elseif field_number == 8
            PB.decode!(d, wire_type, last_tile_dims)
        elseif field_number == 9
            PB.decode!(d, wire_type, iota_reshape_dims)
        elseif field_number == 10
            PB.decode!(d, wire_type, iota_transpose_perm)
        elseif field_number == 11
            is_shard_group = PB.decode(d, Bool)
        elseif field_number == 12
            shard_group_id = PB.decode(d, Int64)
        elseif field_number == 13
            shard_group_type = PB.decode(d, var"OpSharding.ShardGroupType".T)
        elseif field_number == 14
            PB.decode!(d, named_sharding)
        else
            Base.skip(d, wire_type)
        end
    end
    return OpSharding(var"#type", tile_shape[], tile_assignment_dimensions[], tile_assignment_devices[], tuple_shardings[], replicate_on_last_tile_dim, metadata[], last_tile_dims[], iota_reshape_dims[], iota_transpose_perm[], is_shard_group, shard_group_id, shard_group_type, named_sharding[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OpSharding)
    initpos = position(e.io)
    x.var"#type" != var"OpSharding.Type".REPLICATED && PB.encode(e, 1, x.var"#type")
    !isnothing(x.tile_shape) && PB.encode(e, 2, x.tile_shape)
    !isempty(x.tile_assignment_dimensions) && PB.encode(e, 3, x.tile_assignment_dimensions)
    !isempty(x.tile_assignment_devices) && PB.encode(e, 4, x.tile_assignment_devices)
    !isempty(x.tuple_shardings) && PB.encode(e, 5, x.tuple_shardings)
    x.replicate_on_last_tile_dim != false && PB.encode(e, 6, x.replicate_on_last_tile_dim)
    !isempty(x.metadata) && PB.encode(e, 7, x.metadata)
    !isempty(x.last_tile_dims) && PB.encode(e, 8, x.last_tile_dims)
    !isempty(x.iota_reshape_dims) && PB.encode(e, 9, x.iota_reshape_dims)
    !isempty(x.iota_transpose_perm) && PB.encode(e, 10, x.iota_transpose_perm)
    x.is_shard_group != false && PB.encode(e, 11, x.is_shard_group)
    x.shard_group_id != zero(Int64) && PB.encode(e, 12, x.shard_group_id)
    x.shard_group_type != var"OpSharding.ShardGroupType".AS && PB.encode(e, 13, x.shard_group_type)
    !isnothing(x.named_sharding) && PB.encode(e, 14, x.named_sharding)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OpSharding)
    encoded_size = 0
    x.var"#type" != var"OpSharding.Type".REPLICATED && (encoded_size += PB._encoded_size(x.var"#type", 1))
    !isnothing(x.tile_shape) && (encoded_size += PB._encoded_size(x.tile_shape, 2))
    !isempty(x.tile_assignment_dimensions) && (encoded_size += PB._encoded_size(x.tile_assignment_dimensions, 3))
    !isempty(x.tile_assignment_devices) && (encoded_size += PB._encoded_size(x.tile_assignment_devices, 4))
    !isempty(x.tuple_shardings) && (encoded_size += PB._encoded_size(x.tuple_shardings, 5))
    x.replicate_on_last_tile_dim != false && (encoded_size += PB._encoded_size(x.replicate_on_last_tile_dim, 6))
    !isempty(x.metadata) && (encoded_size += PB._encoded_size(x.metadata, 7))
    !isempty(x.last_tile_dims) && (encoded_size += PB._encoded_size(x.last_tile_dims, 8))
    !isempty(x.iota_reshape_dims) && (encoded_size += PB._encoded_size(x.iota_reshape_dims, 9))
    !isempty(x.iota_transpose_perm) && (encoded_size += PB._encoded_size(x.iota_transpose_perm, 10))
    x.is_shard_group != false && (encoded_size += PB._encoded_size(x.is_shard_group, 11))
    x.shard_group_id != zero(Int64) && (encoded_size += PB._encoded_size(x.shard_group_id, 12))
    x.shard_group_type != var"OpSharding.ShardGroupType".AS && (encoded_size += PB._encoded_size(x.shard_group_type, 13))
    !isnothing(x.named_sharding) && (encoded_size += PB._encoded_size(x.named_sharding, 14))
    return encoded_size
end

const ProgramShapeProto = var"##Stub#ProgramShapeProto"{var"##Stub#ShapeProto"}
ProgramShapeProto(;parameters = Vector{ShapeProto}(), result = nothing, parameter_names = Vector{String}()) = ProgramShapeProto(parameters, result, parameter_names)
PB.default_values(::Type{ProgramShapeProto}) = (;parameters = Vector{ShapeProto}(), result = nothing, parameter_names = Vector{String}())
PB.field_numbers(::Type{ProgramShapeProto}) = (;parameters = 1, result = 2, parameter_names = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ProgramShapeProto})
    parameters = PB.BufferedVector{ShapeProto}()
    result = Ref{Union{Nothing,ShapeProto}}(nothing)
    parameter_names = PB.BufferedVector{String}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, parameters)
        elseif field_number == 2
            PB.decode!(d, result)
        elseif field_number == 3
            PB.decode!(d, parameter_names)
        else
            Base.skip(d, wire_type)
        end
    end
    return ProgramShapeProto(parameters[], result[], parameter_names[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ProgramShapeProto)
    initpos = position(e.io)
    !isempty(x.parameters) && PB.encode(e, 1, x.parameters)
    !isnothing(x.result) && PB.encode(e, 2, x.result)
    !isempty(x.parameter_names) && PB.encode(e, 3, x.parameter_names)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ProgramShapeProto)
    encoded_size = 0
    !isempty(x.parameters) && (encoded_size += PB._encoded_size(x.parameters, 1))
    !isnothing(x.result) && (encoded_size += PB._encoded_size(x.result, 2))
    !isempty(x.parameter_names) && (encoded_size += PB._encoded_size(x.parameter_names, 3))
    return encoded_size
end

const ShapeProto = var"##Stub#ShapeProto"
ShapeProto(;element_type = PrimitiveType.PRIMITIVE_TYPE_INVALID, dimensions = Vector{Int64}(), is_dynamic_dimension = Vector{Bool}(), tuple_shapes = Vector{ShapeProto}(), layout = nothing) = ShapeProto(element_type, dimensions, is_dynamic_dimension, tuple_shapes, layout)
PB.reserved_fields(::Type{ShapeProto}) = (names = ["rank"], numbers = Union{Int,UnitRange{Int}}[1])
PB.default_values(::Type{ShapeProto}) = (;element_type = PrimitiveType.PRIMITIVE_TYPE_INVALID, dimensions = Vector{Int64}(), is_dynamic_dimension = Vector{Bool}(), tuple_shapes = Vector{ShapeProto}(), layout = nothing)
PB.field_numbers(::Type{ShapeProto}) = (;element_type = 2, dimensions = 3, is_dynamic_dimension = 6, tuple_shapes = 4, layout = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ShapeProto})
    element_type = PrimitiveType.PRIMITIVE_TYPE_INVALID
    dimensions = PB.BufferedVector{Int64}()
    is_dynamic_dimension = PB.BufferedVector{Bool}()
    tuple_shapes = PB.BufferedVector{ShapeProto}()
    layout = Ref{Union{Nothing,LayoutProto}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 2
            element_type = PB.decode(d, PrimitiveType.T)
        elseif field_number == 3
            PB.decode!(d, wire_type, dimensions)
        elseif field_number == 6
            PB.decode!(d, wire_type, is_dynamic_dimension)
        elseif field_number == 4
            PB.decode!(d, tuple_shapes)
        elseif field_number == 5
            PB.decode!(d, layout)
        else
            Base.skip(d, wire_type)
        end
    end
    return ShapeProto(element_type, dimensions[], is_dynamic_dimension[], tuple_shapes[], layout[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ShapeProto)
    initpos = position(e.io)
    x.element_type != PrimitiveType.PRIMITIVE_TYPE_INVALID && PB.encode(e, 2, x.element_type)
    !isempty(x.dimensions) && PB.encode(e, 3, x.dimensions)
    !isempty(x.is_dynamic_dimension) && PB.encode(e, 6, x.is_dynamic_dimension)
    !isempty(x.tuple_shapes) && PB.encode(e, 4, x.tuple_shapes)
    !isnothing(x.layout) && PB.encode(e, 5, x.layout)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ShapeProto)
    encoded_size = 0
    x.element_type != PrimitiveType.PRIMITIVE_TYPE_INVALID && (encoded_size += PB._encoded_size(x.element_type, 2))
    !isempty(x.dimensions) && (encoded_size += PB._encoded_size(x.dimensions, 3))
    !isempty(x.is_dynamic_dimension) && (encoded_size += PB._encoded_size(x.is_dynamic_dimension, 6))
    !isempty(x.tuple_shapes) && (encoded_size += PB._encoded_size(x.tuple_shapes, 4))
    !isnothing(x.layout) && (encoded_size += PB._encoded_size(x.layout, 5))
    return encoded_size
end
