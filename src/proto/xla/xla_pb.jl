import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"ScheduleProto.Instruction", var"ExecutionOptions.EffortLevel"
export var"DebugOptions.LibNvJitLinkMode", var"DebugOptions.LibraryFusionType"
export var"DebugOptions.PipelineParallelismOptLevel", ShardableValueUpdatePairProto
export var"DebugOptions.AutotuneCacheMode", var"DebugOptions.CommandBufferSchedulingMode"
export var"DebugOptions.PartitioningAlgorithm", var"HloModuleConfigProto.Int64List"
export var"ScheduleProto.SchedulerStatisticsProto", var"HloModuleConfigProto.BoolList"
export var"DebugOptions.StepMarkerLocation", var"DebugOptions.CollectiveOpType"
export var"DebugOptions.ShapeChecks", var"DebugOptions.XnnGraphFusionMode"
export GpuCompilationEnvironment, var"ScheduleConfigProto.Instruction"
export var"DebugOptions.WhileLoopUnrolling", NodeShardingConfigProto, IntRangeInclusive
export CompilationEnvironmentsProto, var"DebugOptions.PGLEStrictnessLevel"
export var"HloModuleConfigProto.FusionConfigCollection", var"DebugOptions.DetectionMode"
export var"DebugOptions.CommandBufferCmdType", var"HloModuleConfigProto.Int64ListList"
export var"ScheduleProto.ComputationScheduleProto", ScheduleConfigProto
export ShardingConfigProto, ThunkBufferDebugFilter, ScheduleProto, DebugOptions
export HloModuleConfigProto, ExecutionOptions, HloModuleProtoWithConfig


struct var"ScheduleProto.Instruction"
    id::Int64
    start_timestamp_cycles::Float64
    end_timestamp_cycles::Float64
    memory_usage_after::Int64
    peak_memory_after::Int64
end
var"ScheduleProto.Instruction"(;id = zero(Int64), start_timestamp_cycles = zero(Float64), end_timestamp_cycles = zero(Float64), memory_usage_after = zero(Int64), peak_memory_after = zero(Int64)) = var"ScheduleProto.Instruction"(id, start_timestamp_cycles, end_timestamp_cycles, memory_usage_after, peak_memory_after)
PB.default_values(::Type{var"ScheduleProto.Instruction"}) = (;id = zero(Int64), start_timestamp_cycles = zero(Float64), end_timestamp_cycles = zero(Float64), memory_usage_after = zero(Int64), peak_memory_after = zero(Int64))
PB.field_numbers(::Type{var"ScheduleProto.Instruction"}) = (;id = 1, start_timestamp_cycles = 2, end_timestamp_cycles = 3, memory_usage_after = 5, peak_memory_after = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"ScheduleProto.Instruction"})
    id = zero(Int64)
    start_timestamp_cycles = zero(Float64)
    end_timestamp_cycles = zero(Float64)
    memory_usage_after = zero(Int64)
    peak_memory_after = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            id = PB.decode(d, Int64)
        elseif field_number == 2
            start_timestamp_cycles = PB.decode(d, Float64)
        elseif field_number == 3
            end_timestamp_cycles = PB.decode(d, Float64)
        elseif field_number == 5
            memory_usage_after = PB.decode(d, Int64)
        elseif field_number == 6
            peak_memory_after = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"ScheduleProto.Instruction"(id, start_timestamp_cycles, end_timestamp_cycles, memory_usage_after, peak_memory_after)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"ScheduleProto.Instruction")
    initpos = position(e.io)
    x.id != zero(Int64) && PB.encode(e, 1, x.id)
    x.start_timestamp_cycles !== zero(Float64) && PB.encode(e, 2, x.start_timestamp_cycles)
    x.end_timestamp_cycles !== zero(Float64) && PB.encode(e, 3, x.end_timestamp_cycles)
    x.memory_usage_after != zero(Int64) && PB.encode(e, 5, x.memory_usage_after)
    x.peak_memory_after != zero(Int64) && PB.encode(e, 6, x.peak_memory_after)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"ScheduleProto.Instruction")
    encoded_size = 0
    x.id != zero(Int64) && (encoded_size += PB._encoded_size(x.id, 1))
    x.start_timestamp_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.start_timestamp_cycles, 2))
    x.end_timestamp_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.end_timestamp_cycles, 3))
    x.memory_usage_after != zero(Int64) && (encoded_size += PB._encoded_size(x.memory_usage_after, 5))
    x.peak_memory_after != zero(Int64) && (encoded_size += PB._encoded_size(x.peak_memory_after, 6))
    return encoded_size
end

@enumx var"ExecutionOptions.EffortLevel" EFFORT_UNKNOWN=0 EFFORT_O0=9 EFFORT_O1=19 EFFORT_O2=29 EFFORT_O3=39

@enumx var"DebugOptions.LibNvJitLinkMode" LIB_NV_JIT_LINK_MODE_AUTO=0 LIB_NV_JIT_LINK_MODE_DISABLED=1 LIB_NV_JIT_LINK_MODE_ENABLED=2

@enumx var"DebugOptions.LibraryFusionType" LIBRARY_FUSION_TYPE_INVALID=0 LIBRARY_FUSION_TYPE_DOT=1 LIBRARY_FUSION_TYPE_ELTWISE=2 LIBRARY_FUSION_TYPE_REDUCE=3 LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT=4 LIBRARY_FUSION_TYPE_INDIVIDUAL_CONVOLUTION=5

@enumx var"DebugOptions.PipelineParallelismOptLevel" PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE=0 PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE=1
PB.reserved_fields(::Type{var"DebugOptions.PipelineParallelismOptLevel".T}) = (names = ["PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE_CYCLE_DECOMPOSER"], numbers = Union{Int,UnitRange{Int}}[2])

struct ShardableValueUpdatePairProto
    input_parameter_number::Int64
    parameter_shape_index::Vector{Int64}
    output_shape_index::Vector{Int64}
end
ShardableValueUpdatePairProto(;input_parameter_number = zero(Int64), parameter_shape_index = Vector{Int64}(), output_shape_index = Vector{Int64}()) = ShardableValueUpdatePairProto(input_parameter_number, parameter_shape_index, output_shape_index)
PB.default_values(::Type{ShardableValueUpdatePairProto}) = (;input_parameter_number = zero(Int64), parameter_shape_index = Vector{Int64}(), output_shape_index = Vector{Int64}())
PB.field_numbers(::Type{ShardableValueUpdatePairProto}) = (;input_parameter_number = 1, parameter_shape_index = 2, output_shape_index = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ShardableValueUpdatePairProto})
    input_parameter_number = zero(Int64)
    parameter_shape_index = PB.BufferedVector{Int64}()
    output_shape_index = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            input_parameter_number = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, wire_type, parameter_shape_index)
        elseif field_number == 3
            PB.decode!(d, wire_type, output_shape_index)
        else
            Base.skip(d, wire_type)
        end
    end
    return ShardableValueUpdatePairProto(input_parameter_number, parameter_shape_index[], output_shape_index[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ShardableValueUpdatePairProto)
    initpos = position(e.io)
    x.input_parameter_number != zero(Int64) && PB.encode(e, 1, x.input_parameter_number)
    !isempty(x.parameter_shape_index) && PB.encode(e, 2, x.parameter_shape_index)
    !isempty(x.output_shape_index) && PB.encode(e, 3, x.output_shape_index)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ShardableValueUpdatePairProto)
    encoded_size = 0
    x.input_parameter_number != zero(Int64) && (encoded_size += PB._encoded_size(x.input_parameter_number, 1))
    !isempty(x.parameter_shape_index) && (encoded_size += PB._encoded_size(x.parameter_shape_index, 2))
    !isempty(x.output_shape_index) && (encoded_size += PB._encoded_size(x.output_shape_index, 3))
    return encoded_size
end

@enumx var"DebugOptions.AutotuneCacheMode" AUTOTUNE_CACHE_MODE_UNSPECIFIED=0 AUTOTUNE_CACHE_MODE_UPDATE=1 AUTOTUNE_CACHE_MODE_READ=2

@enumx var"DebugOptions.CommandBufferSchedulingMode" SERIALIZE=0 CONCURRENT=1 LHS=2

@enumx var"DebugOptions.PartitioningAlgorithm" PARTITIONING_ALGORITHM_NOOP=0 PARTITIONING_ALGORITHM_EXP0=1 PARTITIONING_ALGORITHM_EXP1=2 PARTITIONING_ALGORITHM_EXP2=3

struct var"HloModuleConfigProto.Int64List"
    vals::Vector{Int64}
end
var"HloModuleConfigProto.Int64List"(;vals = Vector{Int64}()) = var"HloModuleConfigProto.Int64List"(vals)
PB.default_values(::Type{var"HloModuleConfigProto.Int64List"}) = (;vals = Vector{Int64}())
PB.field_numbers(::Type{var"HloModuleConfigProto.Int64List"}) = (;vals = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"HloModuleConfigProto.Int64List"})
    vals = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, vals)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"HloModuleConfigProto.Int64List"(vals[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"HloModuleConfigProto.Int64List")
    initpos = position(e.io)
    !isempty(x.vals) && PB.encode(e, 1, x.vals)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"HloModuleConfigProto.Int64List")
    encoded_size = 0
    !isempty(x.vals) && (encoded_size += PB._encoded_size(x.vals, 1))
    return encoded_size
end

struct var"ScheduleProto.SchedulerStatisticsProto"
    all_gather_wasted_cycles::Float64
    all_reduce_wasted_cycles::Float64
    collective_broadcast_wasted_cycles::Float64
    collective_permute_wasted_cycles::Float64
    all_to_all_wasted_cycles::Float64
    ragged_all_to_all_wasted_cycles::Float64
    reduce_scatter_wasted_cycles::Float64
    send_wasted_cycles::Float64
    recv_wasted_cycles::Float64
    call_wasted_cycles::Float64
    total_wasted_cycles::Float64
    total_cycles::Float64
    memory_pressure_peak::Int64
end
var"ScheduleProto.SchedulerStatisticsProto"(;all_gather_wasted_cycles = zero(Float64), all_reduce_wasted_cycles = zero(Float64), collective_broadcast_wasted_cycles = zero(Float64), collective_permute_wasted_cycles = zero(Float64), all_to_all_wasted_cycles = zero(Float64), ragged_all_to_all_wasted_cycles = zero(Float64), reduce_scatter_wasted_cycles = zero(Float64), send_wasted_cycles = zero(Float64), recv_wasted_cycles = zero(Float64), call_wasted_cycles = zero(Float64), total_wasted_cycles = zero(Float64), total_cycles = zero(Float64), memory_pressure_peak = zero(Int64)) = var"ScheduleProto.SchedulerStatisticsProto"(all_gather_wasted_cycles, all_reduce_wasted_cycles, collective_broadcast_wasted_cycles, collective_permute_wasted_cycles, all_to_all_wasted_cycles, ragged_all_to_all_wasted_cycles, reduce_scatter_wasted_cycles, send_wasted_cycles, recv_wasted_cycles, call_wasted_cycles, total_wasted_cycles, total_cycles, memory_pressure_peak)
PB.default_values(::Type{var"ScheduleProto.SchedulerStatisticsProto"}) = (;all_gather_wasted_cycles = zero(Float64), all_reduce_wasted_cycles = zero(Float64), collective_broadcast_wasted_cycles = zero(Float64), collective_permute_wasted_cycles = zero(Float64), all_to_all_wasted_cycles = zero(Float64), ragged_all_to_all_wasted_cycles = zero(Float64), reduce_scatter_wasted_cycles = zero(Float64), send_wasted_cycles = zero(Float64), recv_wasted_cycles = zero(Float64), call_wasted_cycles = zero(Float64), total_wasted_cycles = zero(Float64), total_cycles = zero(Float64), memory_pressure_peak = zero(Int64))
PB.field_numbers(::Type{var"ScheduleProto.SchedulerStatisticsProto"}) = (;all_gather_wasted_cycles = 1, all_reduce_wasted_cycles = 2, collective_broadcast_wasted_cycles = 3, collective_permute_wasted_cycles = 4, all_to_all_wasted_cycles = 5, ragged_all_to_all_wasted_cycles = 6, reduce_scatter_wasted_cycles = 7, send_wasted_cycles = 8, recv_wasted_cycles = 9, call_wasted_cycles = 10, total_wasted_cycles = 11, total_cycles = 12, memory_pressure_peak = 13)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"ScheduleProto.SchedulerStatisticsProto"})
    all_gather_wasted_cycles = zero(Float64)
    all_reduce_wasted_cycles = zero(Float64)
    collective_broadcast_wasted_cycles = zero(Float64)
    collective_permute_wasted_cycles = zero(Float64)
    all_to_all_wasted_cycles = zero(Float64)
    ragged_all_to_all_wasted_cycles = zero(Float64)
    reduce_scatter_wasted_cycles = zero(Float64)
    send_wasted_cycles = zero(Float64)
    recv_wasted_cycles = zero(Float64)
    call_wasted_cycles = zero(Float64)
    total_wasted_cycles = zero(Float64)
    total_cycles = zero(Float64)
    memory_pressure_peak = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            all_gather_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 2
            all_reduce_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 3
            collective_broadcast_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 4
            collective_permute_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 5
            all_to_all_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 6
            ragged_all_to_all_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 7
            reduce_scatter_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 8
            send_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 9
            recv_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 10
            call_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 11
            total_wasted_cycles = PB.decode(d, Float64)
        elseif field_number == 12
            total_cycles = PB.decode(d, Float64)
        elseif field_number == 13
            memory_pressure_peak = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"ScheduleProto.SchedulerStatisticsProto"(all_gather_wasted_cycles, all_reduce_wasted_cycles, collective_broadcast_wasted_cycles, collective_permute_wasted_cycles, all_to_all_wasted_cycles, ragged_all_to_all_wasted_cycles, reduce_scatter_wasted_cycles, send_wasted_cycles, recv_wasted_cycles, call_wasted_cycles, total_wasted_cycles, total_cycles, memory_pressure_peak)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"ScheduleProto.SchedulerStatisticsProto")
    initpos = position(e.io)
    x.all_gather_wasted_cycles !== zero(Float64) && PB.encode(e, 1, x.all_gather_wasted_cycles)
    x.all_reduce_wasted_cycles !== zero(Float64) && PB.encode(e, 2, x.all_reduce_wasted_cycles)
    x.collective_broadcast_wasted_cycles !== zero(Float64) && PB.encode(e, 3, x.collective_broadcast_wasted_cycles)
    x.collective_permute_wasted_cycles !== zero(Float64) && PB.encode(e, 4, x.collective_permute_wasted_cycles)
    x.all_to_all_wasted_cycles !== zero(Float64) && PB.encode(e, 5, x.all_to_all_wasted_cycles)
    x.ragged_all_to_all_wasted_cycles !== zero(Float64) && PB.encode(e, 6, x.ragged_all_to_all_wasted_cycles)
    x.reduce_scatter_wasted_cycles !== zero(Float64) && PB.encode(e, 7, x.reduce_scatter_wasted_cycles)
    x.send_wasted_cycles !== zero(Float64) && PB.encode(e, 8, x.send_wasted_cycles)
    x.recv_wasted_cycles !== zero(Float64) && PB.encode(e, 9, x.recv_wasted_cycles)
    x.call_wasted_cycles !== zero(Float64) && PB.encode(e, 10, x.call_wasted_cycles)
    x.total_wasted_cycles !== zero(Float64) && PB.encode(e, 11, x.total_wasted_cycles)
    x.total_cycles !== zero(Float64) && PB.encode(e, 12, x.total_cycles)
    x.memory_pressure_peak != zero(Int64) && PB.encode(e, 13, x.memory_pressure_peak)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"ScheduleProto.SchedulerStatisticsProto")
    encoded_size = 0
    x.all_gather_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.all_gather_wasted_cycles, 1))
    x.all_reduce_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.all_reduce_wasted_cycles, 2))
    x.collective_broadcast_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.collective_broadcast_wasted_cycles, 3))
    x.collective_permute_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.collective_permute_wasted_cycles, 4))
    x.all_to_all_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.all_to_all_wasted_cycles, 5))
    x.ragged_all_to_all_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.ragged_all_to_all_wasted_cycles, 6))
    x.reduce_scatter_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.reduce_scatter_wasted_cycles, 7))
    x.send_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.send_wasted_cycles, 8))
    x.recv_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.recv_wasted_cycles, 9))
    x.call_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.call_wasted_cycles, 10))
    x.total_wasted_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_wasted_cycles, 11))
    x.total_cycles !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_cycles, 12))
    x.memory_pressure_peak != zero(Int64) && (encoded_size += PB._encoded_size(x.memory_pressure_peak, 13))
    return encoded_size
end

struct var"HloModuleConfigProto.BoolList"
    vals::Vector{Bool}
end
var"HloModuleConfigProto.BoolList"(;vals = Vector{Bool}()) = var"HloModuleConfigProto.BoolList"(vals)
PB.default_values(::Type{var"HloModuleConfigProto.BoolList"}) = (;vals = Vector{Bool}())
PB.field_numbers(::Type{var"HloModuleConfigProto.BoolList"}) = (;vals = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"HloModuleConfigProto.BoolList"})
    vals = PB.BufferedVector{Bool}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, vals)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"HloModuleConfigProto.BoolList"(vals[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"HloModuleConfigProto.BoolList")
    initpos = position(e.io)
    !isempty(x.vals) && PB.encode(e, 1, x.vals)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"HloModuleConfigProto.BoolList")
    encoded_size = 0
    !isempty(x.vals) && (encoded_size += PB._encoded_size(x.vals, 1))
    return encoded_size
end

@enumx var"DebugOptions.StepMarkerLocation" STEP_MARK_AT_ENTRY=0 STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP=1 STEP_MARK_AT_SECOND_LEVEL_WHILE_LOOP=3 STEP_MARK_NONE=2

@enumx var"DebugOptions.CollectiveOpType" NOOP=0 ALLREDUCE=1 ALLGATHER=2 REDUCESCATTER=3 COLLECTIVEBROADCAST=4 ALLTOALL=5 COLLECTIVEPERMUTE=6 RAGGEDALLTOALL=7 ALLCOLLECTIVES=8

@enumx var"DebugOptions.ShapeChecks" IGNORE=0 RUNTIME=1 COMPILE_TIME=2

@enumx var"DebugOptions.XnnGraphFusionMode" XNN_GRAPH_FUSION_MODE_DISABLED=0 XNN_GRAPH_FUSION_MODE_GREEDY=1 XNN_GRAPH_FUSION_MODE_GREEDY_SLINKY=2 XNN_GRAPH_FUSION_MODE_BYPASS_COST_MODEL=3

struct GpuCompilationEnvironment
    dummy_flag::Int64
end
GpuCompilationEnvironment(;dummy_flag = zero(Int64)) = GpuCompilationEnvironment(dummy_flag)
PB.default_values(::Type{GpuCompilationEnvironment}) = (;dummy_flag = zero(Int64))
PB.field_numbers(::Type{GpuCompilationEnvironment}) = (;dummy_flag = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GpuCompilationEnvironment})
    dummy_flag = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            dummy_flag = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return GpuCompilationEnvironment(dummy_flag)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GpuCompilationEnvironment)
    initpos = position(e.io)
    x.dummy_flag != zero(Int64) && PB.encode(e, 1, x.dummy_flag)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GpuCompilationEnvironment)
    encoded_size = 0
    x.dummy_flag != zero(Int64) && (encoded_size += PB._encoded_size(x.dummy_flag, 1))
    return encoded_size
end

struct var"ScheduleConfigProto.Instruction"
    name::String
end
var"ScheduleConfigProto.Instruction"(;name = "") = var"ScheduleConfigProto.Instruction"(name)
PB.default_values(::Type{var"ScheduleConfigProto.Instruction"}) = (;name = "")
PB.field_numbers(::Type{var"ScheduleConfigProto.Instruction"}) = (;name = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"ScheduleConfigProto.Instruction"})
    name = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"ScheduleConfigProto.Instruction"(name)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"ScheduleConfigProto.Instruction")
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"ScheduleConfigProto.Instruction")
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    return encoded_size
end

@enumx var"DebugOptions.WhileLoopUnrolling" WHILE_LOOP_UNROLLING_NO_UNROLL=0 WHILE_LOOP_UNROLLING_DOUBLE_BUFFER=1 WHILE_LOOP_UNROLLING_FULL_UNROLL=2 WHILE_LOOP_UNROLLING_AUTO_UNROLL=3

struct NodeShardingConfigProto
    sharding::Union{Nothing,OpSharding}
    nodes::Vector{NodeShardingConfigProto}
end
NodeShardingConfigProto(;sharding = nothing, nodes = Vector{NodeShardingConfigProto}()) = NodeShardingConfigProto(sharding, nodes)
PB.default_values(::Type{NodeShardingConfigProto}) = (;sharding = nothing, nodes = Vector{NodeShardingConfigProto}())
PB.field_numbers(::Type{NodeShardingConfigProto}) = (;sharding = 1, nodes = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:NodeShardingConfigProto})
    sharding = Ref{Union{Nothing,OpSharding}}(nothing)
    nodes = PB.BufferedVector{NodeShardingConfigProto}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, sharding)
        elseif field_number == 2
            PB.decode!(d, nodes)
        else
            Base.skip(d, wire_type)
        end
    end
    return NodeShardingConfigProto(sharding[], nodes[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::NodeShardingConfigProto)
    initpos = position(e.io)
    !isnothing(x.sharding) && PB.encode(e, 1, x.sharding)
    !isempty(x.nodes) && PB.encode(e, 2, x.nodes)
    return position(e.io) - initpos
end
function PB._encoded_size(x::NodeShardingConfigProto)
    encoded_size = 0
    !isnothing(x.sharding) && (encoded_size += PB._encoded_size(x.sharding, 1))
    !isempty(x.nodes) && (encoded_size += PB._encoded_size(x.nodes, 2))
    return encoded_size
end

struct IntRangeInclusive
    first::Int64
    last::Int64
end
IntRangeInclusive(;first = zero(Int64), last = zero(Int64)) = IntRangeInclusive(first, last)
PB.default_values(::Type{IntRangeInclusive}) = (;first = zero(Int64), last = zero(Int64))
PB.field_numbers(::Type{IntRangeInclusive}) = (;first = 1, last = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:IntRangeInclusive})
    first = zero(Int64)
    last = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            first = PB.decode(d, Int64)
        elseif field_number == 2
            last = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return IntRangeInclusive(first, last)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::IntRangeInclusive)
    initpos = position(e.io)
    x.first != zero(Int64) && PB.encode(e, 1, x.first)
    x.last != zero(Int64) && PB.encode(e, 2, x.last)
    return position(e.io) - initpos
end
function PB._encoded_size(x::IntRangeInclusive)
    encoded_size = 0
    x.first != zero(Int64) && (encoded_size += PB._encoded_size(x.first, 1))
    x.last != zero(Int64) && (encoded_size += PB._encoded_size(x.last, 2))
    return encoded_size
end

struct CompilationEnvironmentsProto
    environments::Vector{google.protobuf.var"#Any"}
end
CompilationEnvironmentsProto(;environments = Vector{google.protobuf.var"#Any"}()) = CompilationEnvironmentsProto(environments)
PB.default_values(::Type{CompilationEnvironmentsProto}) = (;environments = Vector{google.protobuf.var"#Any"}())
PB.field_numbers(::Type{CompilationEnvironmentsProto}) = (;environments = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CompilationEnvironmentsProto})
    environments = PB.BufferedVector{google.protobuf.var"#Any"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, environments)
        else
            Base.skip(d, wire_type)
        end
    end
    return CompilationEnvironmentsProto(environments[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CompilationEnvironmentsProto)
    initpos = position(e.io)
    !isempty(x.environments) && PB.encode(e, 1, x.environments)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CompilationEnvironmentsProto)
    encoded_size = 0
    !isempty(x.environments) && (encoded_size += PB._encoded_size(x.environments, 1))
    return encoded_size
end

@enumx var"DebugOptions.PGLEStrictnessLevel" PGLE_STRICTNESS_LEVEL_OFF=0 PGLE_STRICTNESS_LEVEL_WARN=1 PGLE_STRICTNESS_LEVEL_ERROR=2

@enumx var"HloModuleConfigProto.FusionConfigCollection" OFF=0 PER_EDGE=1 PER_NODE=2

@enumx var"DebugOptions.DetectionMode" DETECTION_MODE_NONE=0 DETECTION_MODE_WARNING=1 DETECTION_MODE_FAIL=2

@enumx var"DebugOptions.CommandBufferCmdType" INVALID=0 FUSION=1 CUBLAS=2 CUDNN=3 COLLECTIVES=4 CONDITIONAL=5 WHILE=6 CUSTOM_CALL=7 CUBLASLT=8 DYNAMIC_SLICE_FUSION=9 DYNAMIC_SLICE_COPY_FUSION=10

struct var"HloModuleConfigProto.Int64ListList"
    lists::Vector{var"HloModuleConfigProto.Int64List"}
end
var"HloModuleConfigProto.Int64ListList"(;lists = Vector{var"HloModuleConfigProto.Int64List"}()) = var"HloModuleConfigProto.Int64ListList"(lists)
PB.default_values(::Type{var"HloModuleConfigProto.Int64ListList"}) = (;lists = Vector{var"HloModuleConfigProto.Int64List"}())
PB.field_numbers(::Type{var"HloModuleConfigProto.Int64ListList"}) = (;lists = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"HloModuleConfigProto.Int64ListList"})
    lists = PB.BufferedVector{var"HloModuleConfigProto.Int64List"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, lists)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"HloModuleConfigProto.Int64ListList"(lists[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"HloModuleConfigProto.Int64ListList")
    initpos = position(e.io)
    !isempty(x.lists) && PB.encode(e, 1, x.lists)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"HloModuleConfigProto.Int64ListList")
    encoded_size = 0
    !isempty(x.lists) && (encoded_size += PB._encoded_size(x.lists, 1))
    return encoded_size
end

struct var"ScheduleProto.ComputationScheduleProto"
    computation_id::Int64
    instructions::Vector{var"ScheduleProto.Instruction"}
    scheduler_statistics::Union{Nothing,var"ScheduleProto.SchedulerStatisticsProto"}
    cycles_per_microsecond::Int64
end
var"ScheduleProto.ComputationScheduleProto"(;computation_id = zero(Int64), instructions = Vector{var"ScheduleProto.Instruction"}(), scheduler_statistics = nothing, cycles_per_microsecond = zero(Int64)) = var"ScheduleProto.ComputationScheduleProto"(computation_id, instructions, scheduler_statistics, cycles_per_microsecond)
PB.default_values(::Type{var"ScheduleProto.ComputationScheduleProto"}) = (;computation_id = zero(Int64), instructions = Vector{var"ScheduleProto.Instruction"}(), scheduler_statistics = nothing, cycles_per_microsecond = zero(Int64))
PB.field_numbers(::Type{var"ScheduleProto.ComputationScheduleProto"}) = (;computation_id = 1, instructions = 2, scheduler_statistics = 3, cycles_per_microsecond = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"ScheduleProto.ComputationScheduleProto"})
    computation_id = zero(Int64)
    instructions = PB.BufferedVector{var"ScheduleProto.Instruction"}()
    scheduler_statistics = Ref{Union{Nothing,var"ScheduleProto.SchedulerStatisticsProto"}}(nothing)
    cycles_per_microsecond = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            computation_id = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, instructions)
        elseif field_number == 3
            PB.decode!(d, scheduler_statistics)
        elseif field_number == 4
            cycles_per_microsecond = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"ScheduleProto.ComputationScheduleProto"(computation_id, instructions[], scheduler_statistics[], cycles_per_microsecond)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"ScheduleProto.ComputationScheduleProto")
    initpos = position(e.io)
    x.computation_id != zero(Int64) && PB.encode(e, 1, x.computation_id)
    !isempty(x.instructions) && PB.encode(e, 2, x.instructions)
    !isnothing(x.scheduler_statistics) && PB.encode(e, 3, x.scheduler_statistics)
    x.cycles_per_microsecond != zero(Int64) && PB.encode(e, 4, x.cycles_per_microsecond)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"ScheduleProto.ComputationScheduleProto")
    encoded_size = 0
    x.computation_id != zero(Int64) && (encoded_size += PB._encoded_size(x.computation_id, 1))
    !isempty(x.instructions) && (encoded_size += PB._encoded_size(x.instructions, 2))
    !isnothing(x.scheduler_statistics) && (encoded_size += PB._encoded_size(x.scheduler_statistics, 3))
    x.cycles_per_microsecond != zero(Int64) && (encoded_size += PB._encoded_size(x.cycles_per_microsecond, 4))
    return encoded_size
end

struct ScheduleConfigProto
    sequence::Vector{var"ScheduleConfigProto.Instruction"}
end
ScheduleConfigProto(;sequence = Vector{var"ScheduleConfigProto.Instruction"}()) = ScheduleConfigProto(sequence)
PB.default_values(::Type{ScheduleConfigProto}) = (;sequence = Vector{var"ScheduleConfigProto.Instruction"}())
PB.field_numbers(::Type{ScheduleConfigProto}) = (;sequence = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ScheduleConfigProto})
    sequence = PB.BufferedVector{var"ScheduleConfigProto.Instruction"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, sequence)
        else
            Base.skip(d, wire_type)
        end
    end
    return ScheduleConfigProto(sequence[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ScheduleConfigProto)
    initpos = position(e.io)
    !isempty(x.sequence) && PB.encode(e, 1, x.sequence)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ScheduleConfigProto)
    encoded_size = 0
    !isempty(x.sequence) && (encoded_size += PB._encoded_size(x.sequence, 1))
    return encoded_size
end

struct ShardingConfigProto
    nodes::Vector{NodeShardingConfigProto}
end
ShardingConfigProto(;nodes = Vector{NodeShardingConfigProto}()) = ShardingConfigProto(nodes)
PB.default_values(::Type{ShardingConfigProto}) = (;nodes = Vector{NodeShardingConfigProto}())
PB.field_numbers(::Type{ShardingConfigProto}) = (;nodes = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ShardingConfigProto})
    nodes = PB.BufferedVector{NodeShardingConfigProto}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, nodes)
        else
            Base.skip(d, wire_type)
        end
    end
    return ShardingConfigProto(nodes[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ShardingConfigProto)
    initpos = position(e.io)
    !isempty(x.nodes) && PB.encode(e, 1, x.nodes)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ShardingConfigProto)
    encoded_size = 0
    !isempty(x.nodes) && (encoded_size += PB._encoded_size(x.nodes, 1))
    return encoded_size
end

struct ThunkBufferDebugFilter
    thunk_id_ranges::Vector{IntRangeInclusive}
    profile_annotation_regexes::Vector{String}
end
ThunkBufferDebugFilter(;thunk_id_ranges = Vector{IntRangeInclusive}(), profile_annotation_regexes = Vector{String}()) = ThunkBufferDebugFilter(thunk_id_ranges, profile_annotation_regexes)
PB.default_values(::Type{ThunkBufferDebugFilter}) = (;thunk_id_ranges = Vector{IntRangeInclusive}(), profile_annotation_regexes = Vector{String}())
PB.field_numbers(::Type{ThunkBufferDebugFilter}) = (;thunk_id_ranges = 1, profile_annotation_regexes = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ThunkBufferDebugFilter})
    thunk_id_ranges = PB.BufferedVector{IntRangeInclusive}()
    profile_annotation_regexes = PB.BufferedVector{String}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, thunk_id_ranges)
        elseif field_number == 2
            PB.decode!(d, profile_annotation_regexes)
        else
            Base.skip(d, wire_type)
        end
    end
    return ThunkBufferDebugFilter(thunk_id_ranges[], profile_annotation_regexes[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ThunkBufferDebugFilter)
    initpos = position(e.io)
    !isempty(x.thunk_id_ranges) && PB.encode(e, 1, x.thunk_id_ranges)
    !isempty(x.profile_annotation_regexes) && PB.encode(e, 2, x.profile_annotation_regexes)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ThunkBufferDebugFilter)
    encoded_size = 0
    !isempty(x.thunk_id_ranges) && (encoded_size += PB._encoded_size(x.thunk_id_ranges, 1))
    !isempty(x.profile_annotation_regexes) && (encoded_size += PB._encoded_size(x.profile_annotation_regexes, 2))
    return encoded_size
end

struct ScheduleProto
    hlo_module::Union{Nothing,HloModuleProto}
    computation_schedules::Vector{var"ScheduleProto.ComputationScheduleProto"}
end
ScheduleProto(;hlo_module = nothing, computation_schedules = Vector{var"ScheduleProto.ComputationScheduleProto"}()) = ScheduleProto(hlo_module, computation_schedules)
PB.default_values(::Type{ScheduleProto}) = (;hlo_module = nothing, computation_schedules = Vector{var"ScheduleProto.ComputationScheduleProto"}())
PB.field_numbers(::Type{ScheduleProto}) = (;hlo_module = 1, computation_schedules = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ScheduleProto})
    hlo_module = Ref{Union{Nothing,HloModuleProto}}(nothing)
    computation_schedules = PB.BufferedVector{var"ScheduleProto.ComputationScheduleProto"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, hlo_module)
        elseif field_number == 2
            PB.decode!(d, computation_schedules)
        else
            Base.skip(d, wire_type)
        end
    end
    return ScheduleProto(hlo_module[], computation_schedules[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ScheduleProto)
    initpos = position(e.io)
    !isnothing(x.hlo_module) && PB.encode(e, 1, x.hlo_module)
    !isempty(x.computation_schedules) && PB.encode(e, 2, x.computation_schedules)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ScheduleProto)
    encoded_size = 0
    !isnothing(x.hlo_module) && (encoded_size += PB._encoded_size(x.hlo_module, 1))
    !isempty(x.computation_schedules) && (encoded_size += PB._encoded_size(x.computation_schedules, 2))
    return encoded_size
end

struct DebugOptions
    xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled::Bool
    xla_disable_automatic_host_compute_offload::Bool
    xla_enable_scoped_logging_timers::Bool
    xla_hlo_pass_fix_detect_cycles::Bool
    xla_keep_shardings_after_spmd::Bool
    xla_unsupported_crash_on_hlo_pass_fix_max_iterations::Bool
    xla_unsupported_crash_on_hlo_pass_noop_change::Bool
    xla_unsupported_crash_on_hlo_pass_silent_hlo_change::Bool
    xla_cpu_collective_call_terminate_timeout_seconds::Int32
    xla_cpu_collective_call_warn_stuck_seconds::Int32
    xla_cpu_collective_timeout_seconds::Int32
    xla_cpu_copy_insertion_use_region_analysis::Bool
    xla_cpu_emitter_verification_level::Int32
    xla_cpu_enable_concurrency_optimized_scheduler::Bool
    xla_cpu_enable_fast_math::Bool
    xla_cpu_enable_fast_min_max::Bool
    xla_cpu_enable_platform_dependent_math::Bool
    xla_cpu_experimental_onednn_custom_call::Bool
    xla_cpu_experimental_onednn_fusion_type::Vector{var"DebugOptions.LibraryFusionType".T}
    xla_cpu_experimental_xnn_fusion_type::Vector{var"DebugOptions.LibraryFusionType".T}
    xla_cpu_experimental_xnn_graph_fusion_mode::var"DebugOptions.XnnGraphFusionMode".T
    xla_cpu_experimental_ynn_fusion_type::Vector{var"DebugOptions.LibraryFusionType".T}
    xla_cpu_fast_math_honor_division::Bool
    xla_cpu_fast_math_honor_functions::Bool
    xla_cpu_fast_math_honor_infs::Bool
    xla_cpu_fast_math_honor_nans::Bool
    xla_cpu_generate_unique_c_style_kernel_entry_points::Bool
    xla_cpu_max_isa::String
    xla_cpu_parallel_codegen_split_count::Int32
    xla_cpu_prefer_vector_width::Int32
    xla_cpu_use_fusion_emitters::Bool
    xla_cpu_use_xnnpack::Bool
    xla_enable_fast_math::Bool
    xla_gpu_experimental_thunk_buffer_debug_filter::Union{Nothing,ThunkBufferDebugFilter}
    xla_dump_hlo_unoptimized_snapshots::Bool
    xla_enable_enzyme_comms_opt::Bool
    xla_gpu_algorithm_denylist_path::String
    xla_gpu_all_gather_combine_threshold_bytes::Int64
    xla_gpu_all_reduce_blueconnect_num_devices_per_host::Int32
    xla_gpu_all_reduce_combine_threshold_bytes::Int64
    xla_gpu_analytical_latency_estimator_options::Dict{String,String}
    xla_gpu_async_dot::Bool
    xla_gpu_auto_spmd_partitioning_memory_budget_gb::Int32
    xla_gpu_auto_spmd_partitioning_memory_budget_ratio::Float32
    xla_gpu_autotune_gemm_rtol::Float32
    xla_gpu_autotune_level::Int32
    xla_gpu_autotune_max_solutions::Int64
    xla_gpu_collect_cost_model_stats::Bool
    xla_gpu_collective_inflation_factor::Int32
    xla_gpu_collective_permute_combine_threshold_bytes::Int64
    xla_gpu_collective_permute_decomposer_threshold::Int64
    xla_gpu_collectives_use_persistent_cliques::Bool
    xla_gpu_command_buffer_scheduling_mode::var"DebugOptions.CommandBufferSchedulingMode".T
    xla_gpu_command_buffer_unroll_loops::Bool
    xla_gpu_copy_insertion_use_region_analysis::Bool
    xla_gpu_crash_on_verification_failures::Bool
    xla_gpu_cublas_fallback::Bool
    xla_gpu_cuda_data_dir::String
    xla_gpu_cudnn_gemm_fusion_level::Int32
    xla_gpu_cudnn_gemm_max_plans::Int32
    xla_gpu_deterministic_ops::Bool
    xla_gpu_disable_async_collectives::Vector{var"DebugOptions.CollectiveOpType".T}
    xla_gpu_disable_gpuasm_optimizations::Bool
    xla_gpu_dot_merger_threshold_mb::Int32
    xla_gpu_dump_autotune_logs_to::String
    xla_gpu_dump_autotune_results_to::String
    xla_gpu_dump_autotuned_gemm_fusions::Bool
    xla_gpu_dump_llvmir::Bool
    xla_gpu_enable_all_gather_combine_by_dim::Bool
    xla_gpu_enable_analytical_latency_estimator::Bool
    xla_gpu_enable_analytical_sol_latency_estimator::Bool
    xla_gpu_enable_approx_costly_collectives::Bool
    xla_gpu_enable_command_buffer::Vector{var"DebugOptions.CommandBufferCmdType".T}
    xla_gpu_enable_cub_radix_sort::Bool
    xla_gpu_enable_cublaslt::Bool
    xla_gpu_enable_cudnn_int8x32_convolution_reordering::Bool
    xla_gpu_enable_cudnn_layer_norm::Bool
    xla_gpu_enable_dynamic_slice_fusion::Bool
    xla_gpu_enable_fast_min_max::Bool
    xla_gpu_enable_highest_priority_async_stream::Bool
    xla_gpu_enable_host_memory_offloading::Bool
    xla_gpu_enable_latency_hiding_scheduler::Bool
    xla_gpu_enable_libnvptxcompiler::Bool
    xla_gpu_enable_llvm_module_compilation_parallelism::Bool
    xla_gpu_enable_nccl_clique_optimization::Bool
    xla_gpu_enable_nccl_comm_splitting::Bool
    xla_gpu_enable_nccl_user_buffers::Bool
    xla_gpu_enable_pipelined_all_gather::Bool
    xla_gpu_enable_pipelined_all_reduce::Bool
    xla_gpu_enable_pipelined_collectives::Bool
    xla_gpu_enable_pipelined_p2p::Bool
    xla_gpu_enable_pipelined_reduce_scatter::Bool
    xla_gpu_enable_reassociation_for_converted_ar::Bool
    xla_gpu_enable_reduce_scatter_combine_by_dim::Bool
    xla_gpu_enable_reduction_epilogue_fusion::Bool
    xla_gpu_enable_scatter_determinism_expander::Bool
    xla_gpu_enable_shared_constants::Bool
    xla_gpu_enable_split_k_autotuning::Bool
    xla_gpu_enable_triton_gemm::Bool
    xla_gpu_enable_while_loop_double_buffering::Bool
    xla_gpu_enable_while_loop_reduce_scatter_code_motion::Bool
    xla_gpu_enable_while_loop_unrolling::var"DebugOptions.WhileLoopUnrolling".T
    xla_gpu_exclude_nondeterministic_ops::Bool
    xla_gpu_executable_embed_debug_info::Bool
    xla_gpu_executable_terminate_timeout_seconds::Int32
    xla_gpu_executable_warn_stuck_timeout_seconds::Int32
    xla_gpu_exhaustive_tiling_search::Bool
    xla_gpu_experimental_allow_unroll_factor_eight::Bool
    xla_gpu_experimental_aot_compiled_thunks::Bool
    xla_gpu_experimental_autotune_cache_mode::var"DebugOptions.AutotuneCacheMode".T
    xla_gpu_experimental_autotuner_cache_dir::String
    xla_gpu_experimental_collective_cse_distance_threshold::Int64
    xla_gpu_experimental_collective_perf_table_path::String
    xla_gpu_experimental_disable_binary_libraries::Bool
    xla_gpu_experimental_dump_fdo_profiles::Bool
    xla_gpu_experimental_dump_gpu_executable::Bool
    xla_gpu_experimental_enable_alltoall_windowed_einsum::Bool
    xla_gpu_experimental_enable_buffer_saver_on_thunks::Bool
    xla_gpu_experimental_enable_checksum_tracing_on_thunks::Bool
    xla_gpu_experimental_enable_fusion_autotuner::Bool
    xla_gpu_experimental_enable_fusion_block_level_rewriter::Bool
    xla_gpu_experimental_enable_heuristic_collective_combining::Bool
    xla_gpu_experimental_enable_nccl_symmetric_buffers::Bool
    xla_gpu_experimental_enable_nvshmem::Bool
    xla_gpu_experimental_enable_split_k_rewrite::Bool
    xla_gpu_experimental_enable_subchannel_dequantisation_fusion::Bool
    xla_gpu_experimental_enable_triton_heroless_priority_fusion::Bool
    xla_gpu_experimental_enable_triton_warp_specialization::Bool
    xla_gpu_experimental_pack_dot_operands_along_k_dimension::Bool
    xla_gpu_experimental_parallel_collective_overlap_limit::Int32
    xla_gpu_experimental_pipeline_parallelism_opt_level::var"DebugOptions.PipelineParallelismOptLevel".T
    xla_gpu_experimental_stream_annotation::Bool
    xla_gpu_experimental_use_autotuner_pass::Bool
    xla_gpu_experimental_use_ragged_dot_fusion::Bool
    xla_gpu_fail_ptx_compilation_on_register_spilling::Bool
    xla_gpu_filter_kernels_spilling_registers_on_autotuning::Bool
    xla_gpu_first_collective_call_terminate_timeout_seconds::Int32
    xla_gpu_first_collective_call_warn_stuck_timeout_seconds::Int32
    xla_gpu_force_compilation_parallelism::Int32
    xla_gpu_force_conv_nchw::Bool
    xla_gpu_force_conv_nhwc::Bool
    xla_gpu_ftz::Bool
    xla_gpu_fused_attention_use_cudnn_rng::Bool
    xla_gpu_gemm_autotuner_override_file::String
    xla_gpu_gemm_rewrite_size_threshold::Int64
    xla_gpu_generate_debug_info::Bool
    xla_gpu_generate_line_info::Bool
    xla_gpu_graph_enable_concurrent_region::Bool
    xla_gpu_graph_min_graph_size::Int32
    xla_gpu_kernel_cache_file::String
    xla_gpu_libnvjitlink_mode::var"DebugOptions.LibNvJitLinkMode".T
    xla_gpu_llvm_ir_file::Vector{String}
    xla_gpu_llvm_verification_level::Int32
    xla_gpu_load_autotune_results_from::String
    xla_gpu_memory_limit_slop_factor::Int32
    xla_gpu_mock_custom_calls::Bool
    xla_gpu_multi_streamed_windowed_einsum::Bool
    xla_gpu_nccl_async_execution::Bool
    xla_gpu_nccl_blocking_communicators::Bool
    xla_gpu_nccl_collective_max_nchannels::Int64
    xla_gpu_nccl_init_max_rank_per_root_ratio::Int64
    xla_gpu_nccl_p2p_max_nchannels::Int64
    xla_gpu_nccl_terminate_on_error::Bool
    xla_gpu_nccl_termination_timeout_seconds::Int64
    xla_gpu_operand_bytes_threshold_for_windowed_einsum::Int64
    xla_gpu_override_gemm_autotuner::String
    xla_gpu_per_fusion_autotune_cache_dir::String
    xla_gpu_pgle_accuracy_checker::var"DebugOptions.PGLEStrictnessLevel".T
    xla_gpu_pgle_profile_file_or_directory_path::String
    xla_gpu_ptx_file::Vector{String}
    xla_gpu_reduce_scatter_combine_threshold_bytes::Int64
    xla_gpu_redzone_padding_bytes::Int64
    xla_gpu_require_complete_aot_autotune_results::Bool
    xla_gpu_require_exclusive_lock::Bool
    xla_gpu_shape_checks::var"DebugOptions.ShapeChecks".T
    xla_gpu_shard_autotuning::Bool
    xla_gpu_strict_conv_algorithm_picker::Bool
    xla_gpu_target_config_filename::String
    xla_gpu_temp_buffer_use_separate_color::Bool
    xla_gpu_threshold_for_windowed_einsum_mib::Int64
    xla_gpu_triton_gemm_any::Bool
    xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found::Bool
    xla_gpu_unsupported_enable_all_reduce_decomposer::Bool
    xla_gpu_unsupported_enable_ragged_all_to_all_decomposer::Bool
    xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer::Bool
    xla_gpu_unsupported_enable_triton_gemm::Bool
    xla_gpu_unsupported_enable_triton_multi_output_fusion::Bool
    xla_gpu_unsupported_override_fast_interconnect_slice_size::Int64
    xla_gpu_unsupported_use_all_reduce_one_shot_kernel::Bool
    xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel::Bool
    xla_gpu_use_embeded_device_lib::Bool
    xla_gpu_use_inprocess_lld::Bool
    xla_gpu_use_memcpy_local_p2p::Bool
    xla_gpu_use_runtime_fusion::Bool
    xla_gpu_verify_triton_fusion_numerics::Bool
    xla_hlo_graph_addresses::Bool
    xla_hlo_profile::Bool
    xla_disable_hlo_passes::Vector{String}
    xla_enable_hlo_passes_only::Vector{String}
    xla_disable_all_hlo_passes::Bool
    xla_backend_optimization_level::Int32
    xla_embed_ir_in_executable::Bool
    xla_eliminate_hlo_implicit_broadcast::Bool
    xla_cpu_multi_thread_eigen::Bool
    xla_llvm_enable_alias_scope_metadata::Bool
    xla_llvm_enable_noalias_metadata::Bool
    xla_llvm_enable_invariant_load_metadata::Bool
    xla_llvm_disable_expensive_passes::Bool
    xla_test_all_output_layouts::Bool
    xla_test_all_input_layouts::Bool
    xla_hlo_graph_sharding_color::Bool
    xla_cpu_use_onednn::Bool
    xla_allow_excess_precision::Bool
    xla_force_host_platform_device_count::Int32
    xla_hlo_evaluator_use_fast_path::Bool
    xla_allow_scalar_index_dynamic_ops::Bool
    xla_step_marker_location::var"DebugOptions.StepMarkerLocation".T
    xla_dump_to::String
    xla_flags_reset::Bool
    xla_dump_hlo_module_re::String
    xla_dump_hlo_pass_re::String
    xla_dump_emitter_re::String
    xla_dump_hlo_as_text::Bool
    xla_dump_hlo_as_proto::Bool
    xla_dump_hlo_as_dot::Bool
    xla_dump_hlo_as_url::Bool
    xla_dump_hlo_as_html::Bool
    xla_dump_fusion_visualization::Bool
    xla_dump_hlo_snapshots::Bool
    xla_dump_include_timestamp::Bool
    xla_dump_max_hlo_modules::Int32
    xla_dump_module_metadata::Bool
    xla_dump_compress_protos::Bool
    xla_dump_hlo_as_long_text::Bool
    xla_dump_enable_mlir_pretty_form::Bool
    xla_dump_full_hlo_config::Bool
    xla_tpu_detect_nan::Bool
    xla_tpu_detect_inf::Bool
    xla_cpu_enable_xprof_traceme::Bool
    xla_multiheap_size_constraint_per_heap::Int32
    xla_detailed_logging::Bool
    xla_enable_dumping::Bool
    xla_llvm_force_inline_before_split::Bool
    xla_dump_disable_metadata::Bool
    xla_dump_hlo_pipeline_re::String
    xla_cpu_use_acl::Bool
    xla_cpu_strict_dot_conv_math::Bool
    xla_dump_latency_hiding_schedule::Bool
    xla_partitioning_algorithm::var"DebugOptions.PartitioningAlgorithm".T
    xla_debug_buffer_assignment_show_max::Int64
    xla_detect_unstable_reductions::var"DebugOptions.DetectionMode".T
    xla_detect_unstable_reductions_post_optimizations::var"DebugOptions.DetectionMode".T
    xla_gpu_detect_nan::var"DebugOptions.DetectionMode".T
    xla_gpu_detect_inf::var"DebugOptions.DetectionMode".T
    xla_dump_large_constants::Bool
    xla_reduce_window_rewrite_base_length::Int64
    xla_cmd_buffer_trace_cache_size::Int64
    xla_syntax_sugar_async_ops::Bool
    xla_enable_command_buffers_during_profiling::Bool
    xla_ignore_channel_id::Bool
    xla_pjrt_allow_auto_layout_in_hlo::Bool
    xla_test_add_command_buffer_mode::Bool
    xla_gpu_experimental_matmul_perf_table_path::String
    xla_early_exit_with_layouts::Bool
    xla_gpu_experimental_scaled_dot_with_triton::Bool
    xla_gpu_experimental_use_raft_select_k::Bool
    xla_backend_extra_options::Dict{String,String}
end
DebugOptions(;xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled = false, xla_disable_automatic_host_compute_offload = false, xla_enable_scoped_logging_timers = false, xla_hlo_pass_fix_detect_cycles = false, xla_keep_shardings_after_spmd = false, xla_unsupported_crash_on_hlo_pass_fix_max_iterations = false, xla_unsupported_crash_on_hlo_pass_noop_change = false, xla_unsupported_crash_on_hlo_pass_silent_hlo_change = false, xla_cpu_collective_call_terminate_timeout_seconds = zero(Int32), xla_cpu_collective_call_warn_stuck_seconds = zero(Int32), xla_cpu_collective_timeout_seconds = zero(Int32), xla_cpu_copy_insertion_use_region_analysis = false, xla_cpu_emitter_verification_level = zero(Int32), xla_cpu_enable_concurrency_optimized_scheduler = false, xla_cpu_enable_fast_math = false, xla_cpu_enable_fast_min_max = false, xla_cpu_enable_platform_dependent_math = false, xla_cpu_experimental_onednn_custom_call = false, xla_cpu_experimental_onednn_fusion_type = Vector{var"DebugOptions.LibraryFusionType".T}(), xla_cpu_experimental_xnn_fusion_type = Vector{var"DebugOptions.LibraryFusionType".T}(), xla_cpu_experimental_xnn_graph_fusion_mode = var"DebugOptions.XnnGraphFusionMode".XNN_GRAPH_FUSION_MODE_DISABLED, xla_cpu_experimental_ynn_fusion_type = Vector{var"DebugOptions.LibraryFusionType".T}(), xla_cpu_fast_math_honor_division = false, xla_cpu_fast_math_honor_functions = false, xla_cpu_fast_math_honor_infs = false, xla_cpu_fast_math_honor_nans = false, xla_cpu_generate_unique_c_style_kernel_entry_points = false, xla_cpu_max_isa = "", xla_cpu_parallel_codegen_split_count = zero(Int32), xla_cpu_prefer_vector_width = zero(Int32), xla_cpu_use_fusion_emitters = false, xla_cpu_use_xnnpack = false, xla_enable_fast_math = false, xla_gpu_experimental_thunk_buffer_debug_filter = nothing, xla_dump_hlo_unoptimized_snapshots = false, xla_enable_enzyme_comms_opt = false, xla_gpu_algorithm_denylist_path = "", xla_gpu_all_gather_combine_threshold_bytes = zero(Int64), xla_gpu_all_reduce_blueconnect_num_devices_per_host = zero(Int32), xla_gpu_all_reduce_combine_threshold_bytes = zero(Int64), xla_gpu_analytical_latency_estimator_options = Dict{String,String}(), xla_gpu_async_dot = false, xla_gpu_auto_spmd_partitioning_memory_budget_gb = zero(Int32), xla_gpu_auto_spmd_partitioning_memory_budget_ratio = zero(Float32), xla_gpu_autotune_gemm_rtol = zero(Float32), xla_gpu_autotune_level = zero(Int32), xla_gpu_autotune_max_solutions = zero(Int64), xla_gpu_collect_cost_model_stats = false, xla_gpu_collective_inflation_factor = zero(Int32), xla_gpu_collective_permute_combine_threshold_bytes = zero(Int64), xla_gpu_collective_permute_decomposer_threshold = zero(Int64), xla_gpu_collectives_use_persistent_cliques = false, xla_gpu_command_buffer_scheduling_mode = var"DebugOptions.CommandBufferSchedulingMode".SERIALIZE, xla_gpu_command_buffer_unroll_loops = false, xla_gpu_copy_insertion_use_region_analysis = false, xla_gpu_crash_on_verification_failures = false, xla_gpu_cublas_fallback = false, xla_gpu_cuda_data_dir = "", xla_gpu_cudnn_gemm_fusion_level = zero(Int32), xla_gpu_cudnn_gemm_max_plans = zero(Int32), xla_gpu_deterministic_ops = false, xla_gpu_disable_async_collectives = Vector{var"DebugOptions.CollectiveOpType".T}(), xla_gpu_disable_gpuasm_optimizations = false, xla_gpu_dot_merger_threshold_mb = zero(Int32), xla_gpu_dump_autotune_logs_to = "", xla_gpu_dump_autotune_results_to = "", xla_gpu_dump_autotuned_gemm_fusions = false, xla_gpu_dump_llvmir = false, xla_gpu_enable_all_gather_combine_by_dim = false, xla_gpu_enable_analytical_latency_estimator = false, xla_gpu_enable_analytical_sol_latency_estimator = false, xla_gpu_enable_approx_costly_collectives = false, xla_gpu_enable_command_buffer = Vector{var"DebugOptions.CommandBufferCmdType".T}(), xla_gpu_enable_cub_radix_sort = false, xla_gpu_enable_cublaslt = false, xla_gpu_enable_cudnn_int8x32_convolution_reordering = false, xla_gpu_enable_cudnn_layer_norm = false, xla_gpu_enable_dynamic_slice_fusion = false, xla_gpu_enable_fast_min_max = false, xla_gpu_enable_highest_priority_async_stream = false, xla_gpu_enable_host_memory_offloading = false, xla_gpu_enable_latency_hiding_scheduler = false, xla_gpu_enable_libnvptxcompiler = false, xla_gpu_enable_llvm_module_compilation_parallelism = false, xla_gpu_enable_nccl_clique_optimization = false, xla_gpu_enable_nccl_comm_splitting = false, xla_gpu_enable_nccl_user_buffers = false, xla_gpu_enable_pipelined_all_gather = false, xla_gpu_enable_pipelined_all_reduce = false, xla_gpu_enable_pipelined_collectives = false, xla_gpu_enable_pipelined_p2p = false, xla_gpu_enable_pipelined_reduce_scatter = false, xla_gpu_enable_reassociation_for_converted_ar = false, xla_gpu_enable_reduce_scatter_combine_by_dim = false, xla_gpu_enable_reduction_epilogue_fusion = false, xla_gpu_enable_scatter_determinism_expander = false, xla_gpu_enable_shared_constants = false, xla_gpu_enable_split_k_autotuning = false, xla_gpu_enable_triton_gemm = false, xla_gpu_enable_while_loop_double_buffering = false, xla_gpu_enable_while_loop_reduce_scatter_code_motion = false, xla_gpu_enable_while_loop_unrolling = var"DebugOptions.WhileLoopUnrolling".WHILE_LOOP_UNROLLING_NO_UNROLL, xla_gpu_exclude_nondeterministic_ops = false, xla_gpu_executable_embed_debug_info = false, xla_gpu_executable_terminate_timeout_seconds = zero(Int32), xla_gpu_executable_warn_stuck_timeout_seconds = zero(Int32), xla_gpu_exhaustive_tiling_search = false, xla_gpu_experimental_allow_unroll_factor_eight = false, xla_gpu_experimental_aot_compiled_thunks = false, xla_gpu_experimental_autotune_cache_mode = var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_UNSPECIFIED, xla_gpu_experimental_autotuner_cache_dir = "", xla_gpu_experimental_collective_cse_distance_threshold = zero(Int64), xla_gpu_experimental_collective_perf_table_path = "", xla_gpu_experimental_disable_binary_libraries = false, xla_gpu_experimental_dump_fdo_profiles = false, xla_gpu_experimental_dump_gpu_executable = false, xla_gpu_experimental_enable_alltoall_windowed_einsum = false, xla_gpu_experimental_enable_buffer_saver_on_thunks = false, xla_gpu_experimental_enable_checksum_tracing_on_thunks = false, xla_gpu_experimental_enable_fusion_autotuner = false, xla_gpu_experimental_enable_fusion_block_level_rewriter = false, xla_gpu_experimental_enable_heuristic_collective_combining = false, xla_gpu_experimental_enable_nccl_symmetric_buffers = false, xla_gpu_experimental_enable_nvshmem = false, xla_gpu_experimental_enable_split_k_rewrite = false, xla_gpu_experimental_enable_subchannel_dequantisation_fusion = false, xla_gpu_experimental_enable_triton_heroless_priority_fusion = false, xla_gpu_experimental_enable_triton_warp_specialization = false, xla_gpu_experimental_pack_dot_operands_along_k_dimension = false, xla_gpu_experimental_parallel_collective_overlap_limit = zero(Int32), xla_gpu_experimental_pipeline_parallelism_opt_level = var"DebugOptions.PipelineParallelismOptLevel".PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE, xla_gpu_experimental_stream_annotation = false, xla_gpu_experimental_use_autotuner_pass = false, xla_gpu_experimental_use_ragged_dot_fusion = false, xla_gpu_fail_ptx_compilation_on_register_spilling = false, xla_gpu_filter_kernels_spilling_registers_on_autotuning = false, xla_gpu_first_collective_call_terminate_timeout_seconds = zero(Int32), xla_gpu_first_collective_call_warn_stuck_timeout_seconds = zero(Int32), xla_gpu_force_compilation_parallelism = zero(Int32), xla_gpu_force_conv_nchw = false, xla_gpu_force_conv_nhwc = false, xla_gpu_ftz = false, xla_gpu_fused_attention_use_cudnn_rng = false, xla_gpu_gemm_autotuner_override_file = "", xla_gpu_gemm_rewrite_size_threshold = zero(Int64), xla_gpu_generate_debug_info = false, xla_gpu_generate_line_info = false, xla_gpu_graph_enable_concurrent_region = false, xla_gpu_graph_min_graph_size = zero(Int32), xla_gpu_kernel_cache_file = "", xla_gpu_libnvjitlink_mode = var"DebugOptions.LibNvJitLinkMode".LIB_NV_JIT_LINK_MODE_AUTO, xla_gpu_llvm_ir_file = Vector{String}(), xla_gpu_llvm_verification_level = zero(Int32), xla_gpu_load_autotune_results_from = "", xla_gpu_memory_limit_slop_factor = zero(Int32), xla_gpu_mock_custom_calls = false, xla_gpu_multi_streamed_windowed_einsum = false, xla_gpu_nccl_async_execution = false, xla_gpu_nccl_blocking_communicators = false, xla_gpu_nccl_collective_max_nchannels = zero(Int64), xla_gpu_nccl_init_max_rank_per_root_ratio = zero(Int64), xla_gpu_nccl_p2p_max_nchannels = zero(Int64), xla_gpu_nccl_terminate_on_error = false, xla_gpu_nccl_termination_timeout_seconds = zero(Int64), xla_gpu_operand_bytes_threshold_for_windowed_einsum = zero(Int64), xla_gpu_override_gemm_autotuner = "", xla_gpu_per_fusion_autotune_cache_dir = "", xla_gpu_pgle_accuracy_checker = var"DebugOptions.PGLEStrictnessLevel".PGLE_STRICTNESS_LEVEL_OFF, xla_gpu_pgle_profile_file_or_directory_path = "", xla_gpu_ptx_file = Vector{String}(), xla_gpu_reduce_scatter_combine_threshold_bytes = zero(Int64), xla_gpu_redzone_padding_bytes = zero(Int64), xla_gpu_require_complete_aot_autotune_results = false, xla_gpu_require_exclusive_lock = false, xla_gpu_shape_checks = var"DebugOptions.ShapeChecks".IGNORE, xla_gpu_shard_autotuning = false, xla_gpu_strict_conv_algorithm_picker = false, xla_gpu_target_config_filename = "", xla_gpu_temp_buffer_use_separate_color = false, xla_gpu_threshold_for_windowed_einsum_mib = zero(Int64), xla_gpu_triton_gemm_any = false, xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found = false, xla_gpu_unsupported_enable_all_reduce_decomposer = false, xla_gpu_unsupported_enable_ragged_all_to_all_decomposer = false, xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer = false, xla_gpu_unsupported_enable_triton_gemm = false, xla_gpu_unsupported_enable_triton_multi_output_fusion = false, xla_gpu_unsupported_override_fast_interconnect_slice_size = zero(Int64), xla_gpu_unsupported_use_all_reduce_one_shot_kernel = false, xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel = false, xla_gpu_use_embeded_device_lib = false, xla_gpu_use_inprocess_lld = false, xla_gpu_use_memcpy_local_p2p = false, xla_gpu_use_runtime_fusion = false, xla_gpu_verify_triton_fusion_numerics = false, xla_hlo_graph_addresses = false, xla_hlo_profile = false, xla_disable_hlo_passes = Vector{String}(), xla_enable_hlo_passes_only = Vector{String}(), xla_disable_all_hlo_passes = false, xla_backend_optimization_level = zero(Int32), xla_embed_ir_in_executable = false, xla_eliminate_hlo_implicit_broadcast = false, xla_cpu_multi_thread_eigen = false, xla_llvm_enable_alias_scope_metadata = false, xla_llvm_enable_noalias_metadata = false, xla_llvm_enable_invariant_load_metadata = false, xla_llvm_disable_expensive_passes = false, xla_test_all_output_layouts = false, xla_test_all_input_layouts = false, xla_hlo_graph_sharding_color = false, xla_cpu_use_onednn = false, xla_allow_excess_precision = false, xla_force_host_platform_device_count = zero(Int32), xla_hlo_evaluator_use_fast_path = false, xla_allow_scalar_index_dynamic_ops = false, xla_step_marker_location = var"DebugOptions.StepMarkerLocation".STEP_MARK_AT_ENTRY, xla_dump_to = "", xla_flags_reset = false, xla_dump_hlo_module_re = "", xla_dump_hlo_pass_re = "", xla_dump_emitter_re = "", xla_dump_hlo_as_text = false, xla_dump_hlo_as_proto = false, xla_dump_hlo_as_dot = false, xla_dump_hlo_as_url = false, xla_dump_hlo_as_html = false, xla_dump_fusion_visualization = false, xla_dump_hlo_snapshots = false, xla_dump_include_timestamp = false, xla_dump_max_hlo_modules = zero(Int32), xla_dump_module_metadata = false, xla_dump_compress_protos = false, xla_dump_hlo_as_long_text = false, xla_dump_enable_mlir_pretty_form = false, xla_dump_full_hlo_config = false, xla_tpu_detect_nan = false, xla_tpu_detect_inf = false, xla_cpu_enable_xprof_traceme = false, xla_multiheap_size_constraint_per_heap = zero(Int32), xla_detailed_logging = false, xla_enable_dumping = false, xla_llvm_force_inline_before_split = false, xla_dump_disable_metadata = false, xla_dump_hlo_pipeline_re = "", xla_cpu_use_acl = false, xla_cpu_strict_dot_conv_math = false, xla_dump_latency_hiding_schedule = false, xla_partitioning_algorithm = var"DebugOptions.PartitioningAlgorithm".PARTITIONING_ALGORITHM_NOOP, xla_debug_buffer_assignment_show_max = zero(Int64), xla_detect_unstable_reductions = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE, xla_detect_unstable_reductions_post_optimizations = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE, xla_gpu_detect_nan = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE, xla_gpu_detect_inf = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE, xla_dump_large_constants = false, xla_reduce_window_rewrite_base_length = zero(Int64), xla_cmd_buffer_trace_cache_size = zero(Int64), xla_syntax_sugar_async_ops = false, xla_enable_command_buffers_during_profiling = false, xla_ignore_channel_id = false, xla_pjrt_allow_auto_layout_in_hlo = false, xla_test_add_command_buffer_mode = false, xla_gpu_experimental_matmul_perf_table_path = "", xla_early_exit_with_layouts = false, xla_gpu_experimental_scaled_dot_with_triton = false, xla_gpu_experimental_use_raft_select_k = false, xla_backend_extra_options = Dict{String,String}()) = DebugOptions(xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled, xla_disable_automatic_host_compute_offload, xla_enable_scoped_logging_timers, xla_hlo_pass_fix_detect_cycles, xla_keep_shardings_after_spmd, xla_unsupported_crash_on_hlo_pass_fix_max_iterations, xla_unsupported_crash_on_hlo_pass_noop_change, xla_unsupported_crash_on_hlo_pass_silent_hlo_change, xla_cpu_collective_call_terminate_timeout_seconds, xla_cpu_collective_call_warn_stuck_seconds, xla_cpu_collective_timeout_seconds, xla_cpu_copy_insertion_use_region_analysis, xla_cpu_emitter_verification_level, xla_cpu_enable_concurrency_optimized_scheduler, xla_cpu_enable_fast_math, xla_cpu_enable_fast_min_max, xla_cpu_enable_platform_dependent_math, xla_cpu_experimental_onednn_custom_call, xla_cpu_experimental_onednn_fusion_type, xla_cpu_experimental_xnn_fusion_type, xla_cpu_experimental_xnn_graph_fusion_mode, xla_cpu_experimental_ynn_fusion_type, xla_cpu_fast_math_honor_division, xla_cpu_fast_math_honor_functions, xla_cpu_fast_math_honor_infs, xla_cpu_fast_math_honor_nans, xla_cpu_generate_unique_c_style_kernel_entry_points, xla_cpu_max_isa, xla_cpu_parallel_codegen_split_count, xla_cpu_prefer_vector_width, xla_cpu_use_fusion_emitters, xla_cpu_use_xnnpack, xla_enable_fast_math, xla_gpu_experimental_thunk_buffer_debug_filter, xla_dump_hlo_unoptimized_snapshots, xla_enable_enzyme_comms_opt, xla_gpu_algorithm_denylist_path, xla_gpu_all_gather_combine_threshold_bytes, xla_gpu_all_reduce_blueconnect_num_devices_per_host, xla_gpu_all_reduce_combine_threshold_bytes, xla_gpu_analytical_latency_estimator_options, xla_gpu_async_dot, xla_gpu_auto_spmd_partitioning_memory_budget_gb, xla_gpu_auto_spmd_partitioning_memory_budget_ratio, xla_gpu_autotune_gemm_rtol, xla_gpu_autotune_level, xla_gpu_autotune_max_solutions, xla_gpu_collect_cost_model_stats, xla_gpu_collective_inflation_factor, xla_gpu_collective_permute_combine_threshold_bytes, xla_gpu_collective_permute_decomposer_threshold, xla_gpu_collectives_use_persistent_cliques, xla_gpu_command_buffer_scheduling_mode, xla_gpu_command_buffer_unroll_loops, xla_gpu_copy_insertion_use_region_analysis, xla_gpu_crash_on_verification_failures, xla_gpu_cublas_fallback, xla_gpu_cuda_data_dir, xla_gpu_cudnn_gemm_fusion_level, xla_gpu_cudnn_gemm_max_plans, xla_gpu_deterministic_ops, xla_gpu_disable_async_collectives, xla_gpu_disable_gpuasm_optimizations, xla_gpu_dot_merger_threshold_mb, xla_gpu_dump_autotune_logs_to, xla_gpu_dump_autotune_results_to, xla_gpu_dump_autotuned_gemm_fusions, xla_gpu_dump_llvmir, xla_gpu_enable_all_gather_combine_by_dim, xla_gpu_enable_analytical_latency_estimator, xla_gpu_enable_analytical_sol_latency_estimator, xla_gpu_enable_approx_costly_collectives, xla_gpu_enable_command_buffer, xla_gpu_enable_cub_radix_sort, xla_gpu_enable_cublaslt, xla_gpu_enable_cudnn_int8x32_convolution_reordering, xla_gpu_enable_cudnn_layer_norm, xla_gpu_enable_dynamic_slice_fusion, xla_gpu_enable_fast_min_max, xla_gpu_enable_highest_priority_async_stream, xla_gpu_enable_host_memory_offloading, xla_gpu_enable_latency_hiding_scheduler, xla_gpu_enable_libnvptxcompiler, xla_gpu_enable_llvm_module_compilation_parallelism, xla_gpu_enable_nccl_clique_optimization, xla_gpu_enable_nccl_comm_splitting, xla_gpu_enable_nccl_user_buffers, xla_gpu_enable_pipelined_all_gather, xla_gpu_enable_pipelined_all_reduce, xla_gpu_enable_pipelined_collectives, xla_gpu_enable_pipelined_p2p, xla_gpu_enable_pipelined_reduce_scatter, xla_gpu_enable_reassociation_for_converted_ar, xla_gpu_enable_reduce_scatter_combine_by_dim, xla_gpu_enable_reduction_epilogue_fusion, xla_gpu_enable_scatter_determinism_expander, xla_gpu_enable_shared_constants, xla_gpu_enable_split_k_autotuning, xla_gpu_enable_triton_gemm, xla_gpu_enable_while_loop_double_buffering, xla_gpu_enable_while_loop_reduce_scatter_code_motion, xla_gpu_enable_while_loop_unrolling, xla_gpu_exclude_nondeterministic_ops, xla_gpu_executable_embed_debug_info, xla_gpu_executable_terminate_timeout_seconds, xla_gpu_executable_warn_stuck_timeout_seconds, xla_gpu_exhaustive_tiling_search, xla_gpu_experimental_allow_unroll_factor_eight, xla_gpu_experimental_aot_compiled_thunks, xla_gpu_experimental_autotune_cache_mode, xla_gpu_experimental_autotuner_cache_dir, xla_gpu_experimental_collective_cse_distance_threshold, xla_gpu_experimental_collective_perf_table_path, xla_gpu_experimental_disable_binary_libraries, xla_gpu_experimental_dump_fdo_profiles, xla_gpu_experimental_dump_gpu_executable, xla_gpu_experimental_enable_alltoall_windowed_einsum, xla_gpu_experimental_enable_buffer_saver_on_thunks, xla_gpu_experimental_enable_checksum_tracing_on_thunks, xla_gpu_experimental_enable_fusion_autotuner, xla_gpu_experimental_enable_fusion_block_level_rewriter, xla_gpu_experimental_enable_heuristic_collective_combining, xla_gpu_experimental_enable_nccl_symmetric_buffers, xla_gpu_experimental_enable_nvshmem, xla_gpu_experimental_enable_split_k_rewrite, xla_gpu_experimental_enable_subchannel_dequantisation_fusion, xla_gpu_experimental_enable_triton_heroless_priority_fusion, xla_gpu_experimental_enable_triton_warp_specialization, xla_gpu_experimental_pack_dot_operands_along_k_dimension, xla_gpu_experimental_parallel_collective_overlap_limit, xla_gpu_experimental_pipeline_parallelism_opt_level, xla_gpu_experimental_stream_annotation, xla_gpu_experimental_use_autotuner_pass, xla_gpu_experimental_use_ragged_dot_fusion, xla_gpu_fail_ptx_compilation_on_register_spilling, xla_gpu_filter_kernels_spilling_registers_on_autotuning, xla_gpu_first_collective_call_terminate_timeout_seconds, xla_gpu_first_collective_call_warn_stuck_timeout_seconds, xla_gpu_force_compilation_parallelism, xla_gpu_force_conv_nchw, xla_gpu_force_conv_nhwc, xla_gpu_ftz, xla_gpu_fused_attention_use_cudnn_rng, xla_gpu_gemm_autotuner_override_file, xla_gpu_gemm_rewrite_size_threshold, xla_gpu_generate_debug_info, xla_gpu_generate_line_info, xla_gpu_graph_enable_concurrent_region, xla_gpu_graph_min_graph_size, xla_gpu_kernel_cache_file, xla_gpu_libnvjitlink_mode, xla_gpu_llvm_ir_file, xla_gpu_llvm_verification_level, xla_gpu_load_autotune_results_from, xla_gpu_memory_limit_slop_factor, xla_gpu_mock_custom_calls, xla_gpu_multi_streamed_windowed_einsum, xla_gpu_nccl_async_execution, xla_gpu_nccl_blocking_communicators, xla_gpu_nccl_collective_max_nchannels, xla_gpu_nccl_init_max_rank_per_root_ratio, xla_gpu_nccl_p2p_max_nchannels, xla_gpu_nccl_terminate_on_error, xla_gpu_nccl_termination_timeout_seconds, xla_gpu_operand_bytes_threshold_for_windowed_einsum, xla_gpu_override_gemm_autotuner, xla_gpu_per_fusion_autotune_cache_dir, xla_gpu_pgle_accuracy_checker, xla_gpu_pgle_profile_file_or_directory_path, xla_gpu_ptx_file, xla_gpu_reduce_scatter_combine_threshold_bytes, xla_gpu_redzone_padding_bytes, xla_gpu_require_complete_aot_autotune_results, xla_gpu_require_exclusive_lock, xla_gpu_shape_checks, xla_gpu_shard_autotuning, xla_gpu_strict_conv_algorithm_picker, xla_gpu_target_config_filename, xla_gpu_temp_buffer_use_separate_color, xla_gpu_threshold_for_windowed_einsum_mib, xla_gpu_triton_gemm_any, xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found, xla_gpu_unsupported_enable_all_reduce_decomposer, xla_gpu_unsupported_enable_ragged_all_to_all_decomposer, xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer, xla_gpu_unsupported_enable_triton_gemm, xla_gpu_unsupported_enable_triton_multi_output_fusion, xla_gpu_unsupported_override_fast_interconnect_slice_size, xla_gpu_unsupported_use_all_reduce_one_shot_kernel, xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel, xla_gpu_use_embeded_device_lib, xla_gpu_use_inprocess_lld, xla_gpu_use_memcpy_local_p2p, xla_gpu_use_runtime_fusion, xla_gpu_verify_triton_fusion_numerics, xla_hlo_graph_addresses, xla_hlo_profile, xla_disable_hlo_passes, xla_enable_hlo_passes_only, xla_disable_all_hlo_passes, xla_backend_optimization_level, xla_embed_ir_in_executable, xla_eliminate_hlo_implicit_broadcast, xla_cpu_multi_thread_eigen, xla_llvm_enable_alias_scope_metadata, xla_llvm_enable_noalias_metadata, xla_llvm_enable_invariant_load_metadata, xla_llvm_disable_expensive_passes, xla_test_all_output_layouts, xla_test_all_input_layouts, xla_hlo_graph_sharding_color, xla_cpu_use_onednn, xla_allow_excess_precision, xla_force_host_platform_device_count, xla_hlo_evaluator_use_fast_path, xla_allow_scalar_index_dynamic_ops, xla_step_marker_location, xla_dump_to, xla_flags_reset, xla_dump_hlo_module_re, xla_dump_hlo_pass_re, xla_dump_emitter_re, xla_dump_hlo_as_text, xla_dump_hlo_as_proto, xla_dump_hlo_as_dot, xla_dump_hlo_as_url, xla_dump_hlo_as_html, xla_dump_fusion_visualization, xla_dump_hlo_snapshots, xla_dump_include_timestamp, xla_dump_max_hlo_modules, xla_dump_module_metadata, xla_dump_compress_protos, xla_dump_hlo_as_long_text, xla_dump_enable_mlir_pretty_form, xla_dump_full_hlo_config, xla_tpu_detect_nan, xla_tpu_detect_inf, xla_cpu_enable_xprof_traceme, xla_multiheap_size_constraint_per_heap, xla_detailed_logging, xla_enable_dumping, xla_llvm_force_inline_before_split, xla_dump_disable_metadata, xla_dump_hlo_pipeline_re, xla_cpu_use_acl, xla_cpu_strict_dot_conv_math, xla_dump_latency_hiding_schedule, xla_partitioning_algorithm, xla_debug_buffer_assignment_show_max, xla_detect_unstable_reductions, xla_detect_unstable_reductions_post_optimizations, xla_gpu_detect_nan, xla_gpu_detect_inf, xla_dump_large_constants, xla_reduce_window_rewrite_base_length, xla_cmd_buffer_trace_cache_size, xla_syntax_sugar_async_ops, xla_enable_command_buffers_during_profiling, xla_ignore_channel_id, xla_pjrt_allow_auto_layout_in_hlo, xla_test_add_command_buffer_mode, xla_gpu_experimental_matmul_perf_table_path, xla_early_exit_with_layouts, xla_gpu_experimental_scaled_dot_with_triton, xla_gpu_experimental_use_raft_select_k, xla_backend_extra_options)
PB.reserved_fields(::Type{DebugOptions}) = (names = ["hlo_reduce_precision_options", "legacy_command_buffer_custom_call_targets", "xla_allow_get_default_platform", "xla_cpu_dump_unoptimized_hlo_snapshots", "xla_cpu_enable_custom_matmul_tiling", "xla_cpu_enable_experimental_deallocation", "xla_cpu_enable_mlir_fusion_outlining", "xla_cpu_enable_mlir_lowering", "xla_cpu_enable_mlir_tiling_and_fusion", "xla_cpu_matmul_tiling_k_dim", "xla_cpu_matmul_tiling_m_dim", "xla_cpu_matmul_tiling_n_dim", "xla_cpu_sparse_cuda_threads", "xla_cpu_use_thunk_runtime", "xla_cpu_use_xla_runtime", "xla_detailed_logging_and_dumping", "xla_dump_ir", "xla_experimental_exec_time_optimization_effort", "xla_gpu_all_reduce_contiguous", "xla_gpu_allow_all_reduce_kernel", "xla_gpu_asm_extra_flags", "xla_gpu_bef_executable", "xla_gpu_bef_thunk", "xla_gpu_deterministic_reductions", "xla_gpu_disable_multi_streaming", "xla_gpu_dump_hlo_unoptimized_snapshots", "xla_gpu_enable_all_reduce_splitter", "xla_gpu_enable_async_all_gather", "xla_gpu_enable_async_all_reduce", "xla_gpu_enable_async_all_to_all", "xla_gpu_enable_async_collective_broadcast", "xla_gpu_enable_async_collective_permute", "xla_gpu_enable_async_collectives", "xla_gpu_enable_async_reduce_scatter", "xla_gpu_enable_bf16_3way_gemm", "xla_gpu_enable_bf16_6way_gemm", "xla_gpu_enable_cuda_graphs", "xla_gpu_enable_cudnn_fmha", "xla_gpu_enable_cudnn_frontend", "xla_gpu_enable_custom_fusions_re", "xla_gpu_enable_custom_fusions", "xla_gpu_enable_dot_strength_reduction", "xla_gpu_enable_experimental_block_size", "xla_gpu_enable_gpu2_hal", "xla_gpu_enable_gpu2_runtime", "xla_gpu_enable_heuristic_pass_configuration", "xla_gpu_enable_libnvjitlink", "xla_gpu_enable_mlir_emitters", "xla_gpu_enable_mlir_lowering", "xla_gpu_enable_nccl_per_stream_comms", "xla_gpu_enable_persistent_temp_buffers", "xla_gpu_enable_pgle_accuracy_checker", "xla_gpu_enable_priority_fusion", "xla_gpu_enable_softmax_fusion", "xla_gpu_enable_triton_gemm_int4", "xla_gpu_enable_triton_hopper", "xla_gpu_enable_triton_softmax_fusion", "xla_gpu_enable_triton_softmax_priority_fusion", "xla_gpu_enable_xla_runtime_executable", "xla_gpu_ensure_minor_dot_contraction_dims", "xla_gpu_experimental_enable_dynamic_dot_search_space", "xla_gpu_experimental_enable_nan_counter_on_thunks", "xla_gpu_experimental_enable_triton_i4_rewrites", "xla_gpu_experimental_enable_triton_softmax_priority_fusion", "xla_gpu_graph_eviction_timeout_seconds", "xla_gpu_graph_level", "xla_gpu_graph_num_runs_to_instantiate", "xla_gpu_lhs_enable_gpu_async_tracker", "xla_gpu_max_kernel_unroll_factor", "xla_gpu_max_mlir_kernels", "xla_gpu_mlir_emitter_level", "xla_gpu_normalize_layouts", "xla_gpu_redzone_scratch_max_megabytes", "xla_gpu_run_post_layout_collective_pipeliner", "xla_gpu_simplify_all_fp_conversions", "xla_gpu_simplify_gathers", "xla_gpu_simplify_scatters", "xla_gpu_single_wave_autotuning", "xla_gpu_skip_mlir_kernels", "xla_gpu_triton_fusion_level", "xla_gpu_triton_gemm_disable_reduced_precision_reduction", "xla_gpu_unsafe_fallback_to_driver_on_ptxas_error", "xla_gpu_unsafe_pipelined_loop_annotator", "xla_gpu_unsupported_enable_generic_triton_emitter_for_gemms", "xla_gpu_unsupported_force_triton_gemm", "xla_gpu_unsupported_generic_triton_emitter_features", "xla_gpu_use_cudnn_batchnorm", "xla_gpu_use_horizontal_fusion", "xla_gpu_use_random_streams", "xla_hlo_dump_as_graphdef", "xla_hlo_tfgraph_device_scopes", "xla_use_shardy", "xla_gpu_unsupported_annotate_with_emitter_loc", "xla_gpu_experimental_enable_command_buffer_on_thunks", "xla_gpu_experimental_enable_triton_tma"], numbers = Union{Int,UnitRange{Int}}[5, 63, 80, 93, 94, 98, 117, 130, 133, 134, 139, 141, 143, 152, 158, 160, 161, 162, 167, 168, 169, 171, 172, 173, 176, 177, 178, 179, 180, 183, 184, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 204, 206, 207, 211, 214, 218, 220, 221, 226, 229, 230, 233, 234, 238, 242, 249, 263, 264, 266, 270, 271, 275, 276, 278, 279, 281, 282, 286, 298, 299, 302, 303, 309, 313, 314, 319, 320, 325, 326, 332, 346, 352, 355, 358, 361, 367, 369, 371, 385, 394, 398, 402, 423])
PB.default_values(::Type{DebugOptions}) = (;xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled = false, xla_disable_automatic_host_compute_offload = false, xla_enable_scoped_logging_timers = false, xla_hlo_pass_fix_detect_cycles = false, xla_keep_shardings_after_spmd = false, xla_unsupported_crash_on_hlo_pass_fix_max_iterations = false, xla_unsupported_crash_on_hlo_pass_noop_change = false, xla_unsupported_crash_on_hlo_pass_silent_hlo_change = false, xla_cpu_collective_call_terminate_timeout_seconds = zero(Int32), xla_cpu_collective_call_warn_stuck_seconds = zero(Int32), xla_cpu_collective_timeout_seconds = zero(Int32), xla_cpu_copy_insertion_use_region_analysis = false, xla_cpu_emitter_verification_level = zero(Int32), xla_cpu_enable_concurrency_optimized_scheduler = false, xla_cpu_enable_fast_math = false, xla_cpu_enable_fast_min_max = false, xla_cpu_enable_platform_dependent_math = false, xla_cpu_experimental_onednn_custom_call = false, xla_cpu_experimental_onednn_fusion_type = Vector{var"DebugOptions.LibraryFusionType".T}(), xla_cpu_experimental_xnn_fusion_type = Vector{var"DebugOptions.LibraryFusionType".T}(), xla_cpu_experimental_xnn_graph_fusion_mode = var"DebugOptions.XnnGraphFusionMode".XNN_GRAPH_FUSION_MODE_DISABLED, xla_cpu_experimental_ynn_fusion_type = Vector{var"DebugOptions.LibraryFusionType".T}(), xla_cpu_fast_math_honor_division = false, xla_cpu_fast_math_honor_functions = false, xla_cpu_fast_math_honor_infs = false, xla_cpu_fast_math_honor_nans = false, xla_cpu_generate_unique_c_style_kernel_entry_points = false, xla_cpu_max_isa = "", xla_cpu_parallel_codegen_split_count = zero(Int32), xla_cpu_prefer_vector_width = zero(Int32), xla_cpu_use_fusion_emitters = false, xla_cpu_use_xnnpack = false, xla_enable_fast_math = false, xla_gpu_experimental_thunk_buffer_debug_filter = nothing, xla_dump_hlo_unoptimized_snapshots = false, xla_enable_enzyme_comms_opt = false, xla_gpu_algorithm_denylist_path = "", xla_gpu_all_gather_combine_threshold_bytes = zero(Int64), xla_gpu_all_reduce_blueconnect_num_devices_per_host = zero(Int32), xla_gpu_all_reduce_combine_threshold_bytes = zero(Int64), xla_gpu_analytical_latency_estimator_options = Dict{String,String}(), xla_gpu_async_dot = false, xla_gpu_auto_spmd_partitioning_memory_budget_gb = zero(Int32), xla_gpu_auto_spmd_partitioning_memory_budget_ratio = zero(Float32), xla_gpu_autotune_gemm_rtol = zero(Float32), xla_gpu_autotune_level = zero(Int32), xla_gpu_autotune_max_solutions = zero(Int64), xla_gpu_collect_cost_model_stats = false, xla_gpu_collective_inflation_factor = zero(Int32), xla_gpu_collective_permute_combine_threshold_bytes = zero(Int64), xla_gpu_collective_permute_decomposer_threshold = zero(Int64), xla_gpu_collectives_use_persistent_cliques = false, xla_gpu_command_buffer_scheduling_mode = var"DebugOptions.CommandBufferSchedulingMode".SERIALIZE, xla_gpu_command_buffer_unroll_loops = false, xla_gpu_copy_insertion_use_region_analysis = false, xla_gpu_crash_on_verification_failures = false, xla_gpu_cublas_fallback = false, xla_gpu_cuda_data_dir = "", xla_gpu_cudnn_gemm_fusion_level = zero(Int32), xla_gpu_cudnn_gemm_max_plans = zero(Int32), xla_gpu_deterministic_ops = false, xla_gpu_disable_async_collectives = Vector{var"DebugOptions.CollectiveOpType".T}(), xla_gpu_disable_gpuasm_optimizations = false, xla_gpu_dot_merger_threshold_mb = zero(Int32), xla_gpu_dump_autotune_logs_to = "", xla_gpu_dump_autotune_results_to = "", xla_gpu_dump_autotuned_gemm_fusions = false, xla_gpu_dump_llvmir = false, xla_gpu_enable_all_gather_combine_by_dim = false, xla_gpu_enable_analytical_latency_estimator = false, xla_gpu_enable_analytical_sol_latency_estimator = false, xla_gpu_enable_approx_costly_collectives = false, xla_gpu_enable_command_buffer = Vector{var"DebugOptions.CommandBufferCmdType".T}(), xla_gpu_enable_cub_radix_sort = false, xla_gpu_enable_cublaslt = false, xla_gpu_enable_cudnn_int8x32_convolution_reordering = false, xla_gpu_enable_cudnn_layer_norm = false, xla_gpu_enable_dynamic_slice_fusion = false, xla_gpu_enable_fast_min_max = false, xla_gpu_enable_highest_priority_async_stream = false, xla_gpu_enable_host_memory_offloading = false, xla_gpu_enable_latency_hiding_scheduler = false, xla_gpu_enable_libnvptxcompiler = false, xla_gpu_enable_llvm_module_compilation_parallelism = false, xla_gpu_enable_nccl_clique_optimization = false, xla_gpu_enable_nccl_comm_splitting = false, xla_gpu_enable_nccl_user_buffers = false, xla_gpu_enable_pipelined_all_gather = false, xla_gpu_enable_pipelined_all_reduce = false, xla_gpu_enable_pipelined_collectives = false, xla_gpu_enable_pipelined_p2p = false, xla_gpu_enable_pipelined_reduce_scatter = false, xla_gpu_enable_reassociation_for_converted_ar = false, xla_gpu_enable_reduce_scatter_combine_by_dim = false, xla_gpu_enable_reduction_epilogue_fusion = false, xla_gpu_enable_scatter_determinism_expander = false, xla_gpu_enable_shared_constants = false, xla_gpu_enable_split_k_autotuning = false, xla_gpu_enable_triton_gemm = false, xla_gpu_enable_while_loop_double_buffering = false, xla_gpu_enable_while_loop_reduce_scatter_code_motion = false, xla_gpu_enable_while_loop_unrolling = var"DebugOptions.WhileLoopUnrolling".WHILE_LOOP_UNROLLING_NO_UNROLL, xla_gpu_exclude_nondeterministic_ops = false, xla_gpu_executable_embed_debug_info = false, xla_gpu_executable_terminate_timeout_seconds = zero(Int32), xla_gpu_executable_warn_stuck_timeout_seconds = zero(Int32), xla_gpu_exhaustive_tiling_search = false, xla_gpu_experimental_allow_unroll_factor_eight = false, xla_gpu_experimental_aot_compiled_thunks = false, xla_gpu_experimental_autotune_cache_mode = var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_UNSPECIFIED, xla_gpu_experimental_autotuner_cache_dir = "", xla_gpu_experimental_collective_cse_distance_threshold = zero(Int64), xla_gpu_experimental_collective_perf_table_path = "", xla_gpu_experimental_disable_binary_libraries = false, xla_gpu_experimental_dump_fdo_profiles = false, xla_gpu_experimental_dump_gpu_executable = false, xla_gpu_experimental_enable_alltoall_windowed_einsum = false, xla_gpu_experimental_enable_buffer_saver_on_thunks = false, xla_gpu_experimental_enable_checksum_tracing_on_thunks = false, xla_gpu_experimental_enable_fusion_autotuner = false, xla_gpu_experimental_enable_fusion_block_level_rewriter = false, xla_gpu_experimental_enable_heuristic_collective_combining = false, xla_gpu_experimental_enable_nccl_symmetric_buffers = false, xla_gpu_experimental_enable_nvshmem = false, xla_gpu_experimental_enable_split_k_rewrite = false, xla_gpu_experimental_enable_subchannel_dequantisation_fusion = false, xla_gpu_experimental_enable_triton_heroless_priority_fusion = false, xla_gpu_experimental_enable_triton_warp_specialization = false, xla_gpu_experimental_pack_dot_operands_along_k_dimension = false, xla_gpu_experimental_parallel_collective_overlap_limit = zero(Int32), xla_gpu_experimental_pipeline_parallelism_opt_level = var"DebugOptions.PipelineParallelismOptLevel".PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE, xla_gpu_experimental_stream_annotation = false, xla_gpu_experimental_use_autotuner_pass = false, xla_gpu_experimental_use_ragged_dot_fusion = false, xla_gpu_fail_ptx_compilation_on_register_spilling = false, xla_gpu_filter_kernels_spilling_registers_on_autotuning = false, xla_gpu_first_collective_call_terminate_timeout_seconds = zero(Int32), xla_gpu_first_collective_call_warn_stuck_timeout_seconds = zero(Int32), xla_gpu_force_compilation_parallelism = zero(Int32), xla_gpu_force_conv_nchw = false, xla_gpu_force_conv_nhwc = false, xla_gpu_ftz = false, xla_gpu_fused_attention_use_cudnn_rng = false, xla_gpu_gemm_autotuner_override_file = "", xla_gpu_gemm_rewrite_size_threshold = zero(Int64), xla_gpu_generate_debug_info = false, xla_gpu_generate_line_info = false, xla_gpu_graph_enable_concurrent_region = false, xla_gpu_graph_min_graph_size = zero(Int32), xla_gpu_kernel_cache_file = "", xla_gpu_libnvjitlink_mode = var"DebugOptions.LibNvJitLinkMode".LIB_NV_JIT_LINK_MODE_AUTO, xla_gpu_llvm_ir_file = Vector{String}(), xla_gpu_llvm_verification_level = zero(Int32), xla_gpu_load_autotune_results_from = "", xla_gpu_memory_limit_slop_factor = zero(Int32), xla_gpu_mock_custom_calls = false, xla_gpu_multi_streamed_windowed_einsum = false, xla_gpu_nccl_async_execution = false, xla_gpu_nccl_blocking_communicators = false, xla_gpu_nccl_collective_max_nchannels = zero(Int64), xla_gpu_nccl_init_max_rank_per_root_ratio = zero(Int64), xla_gpu_nccl_p2p_max_nchannels = zero(Int64), xla_gpu_nccl_terminate_on_error = false, xla_gpu_nccl_termination_timeout_seconds = zero(Int64), xla_gpu_operand_bytes_threshold_for_windowed_einsum = zero(Int64), xla_gpu_override_gemm_autotuner = "", xla_gpu_per_fusion_autotune_cache_dir = "", xla_gpu_pgle_accuracy_checker = var"DebugOptions.PGLEStrictnessLevel".PGLE_STRICTNESS_LEVEL_OFF, xla_gpu_pgle_profile_file_or_directory_path = "", xla_gpu_ptx_file = Vector{String}(), xla_gpu_reduce_scatter_combine_threshold_bytes = zero(Int64), xla_gpu_redzone_padding_bytes = zero(Int64), xla_gpu_require_complete_aot_autotune_results = false, xla_gpu_require_exclusive_lock = false, xla_gpu_shape_checks = var"DebugOptions.ShapeChecks".IGNORE, xla_gpu_shard_autotuning = false, xla_gpu_strict_conv_algorithm_picker = false, xla_gpu_target_config_filename = "", xla_gpu_temp_buffer_use_separate_color = false, xla_gpu_threshold_for_windowed_einsum_mib = zero(Int64), xla_gpu_triton_gemm_any = false, xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found = false, xla_gpu_unsupported_enable_all_reduce_decomposer = false, xla_gpu_unsupported_enable_ragged_all_to_all_decomposer = false, xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer = false, xla_gpu_unsupported_enable_triton_gemm = false, xla_gpu_unsupported_enable_triton_multi_output_fusion = false, xla_gpu_unsupported_override_fast_interconnect_slice_size = zero(Int64), xla_gpu_unsupported_use_all_reduce_one_shot_kernel = false, xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel = false, xla_gpu_use_embeded_device_lib = false, xla_gpu_use_inprocess_lld = false, xla_gpu_use_memcpy_local_p2p = false, xla_gpu_use_runtime_fusion = false, xla_gpu_verify_triton_fusion_numerics = false, xla_hlo_graph_addresses = false, xla_hlo_profile = false, xla_disable_hlo_passes = Vector{String}(), xla_enable_hlo_passes_only = Vector{String}(), xla_disable_all_hlo_passes = false, xla_backend_optimization_level = zero(Int32), xla_embed_ir_in_executable = false, xla_eliminate_hlo_implicit_broadcast = false, xla_cpu_multi_thread_eigen = false, xla_llvm_enable_alias_scope_metadata = false, xla_llvm_enable_noalias_metadata = false, xla_llvm_enable_invariant_load_metadata = false, xla_llvm_disable_expensive_passes = false, xla_test_all_output_layouts = false, xla_test_all_input_layouts = false, xla_hlo_graph_sharding_color = false, xla_cpu_use_onednn = false, xla_allow_excess_precision = false, xla_force_host_platform_device_count = zero(Int32), xla_hlo_evaluator_use_fast_path = false, xla_allow_scalar_index_dynamic_ops = false, xla_step_marker_location = var"DebugOptions.StepMarkerLocation".STEP_MARK_AT_ENTRY, xla_dump_to = "", xla_flags_reset = false, xla_dump_hlo_module_re = "", xla_dump_hlo_pass_re = "", xla_dump_emitter_re = "", xla_dump_hlo_as_text = false, xla_dump_hlo_as_proto = false, xla_dump_hlo_as_dot = false, xla_dump_hlo_as_url = false, xla_dump_hlo_as_html = false, xla_dump_fusion_visualization = false, xla_dump_hlo_snapshots = false, xla_dump_include_timestamp = false, xla_dump_max_hlo_modules = zero(Int32), xla_dump_module_metadata = false, xla_dump_compress_protos = false, xla_dump_hlo_as_long_text = false, xla_dump_enable_mlir_pretty_form = false, xla_dump_full_hlo_config = false, xla_tpu_detect_nan = false, xla_tpu_detect_inf = false, xla_cpu_enable_xprof_traceme = false, xla_multiheap_size_constraint_per_heap = zero(Int32), xla_detailed_logging = false, xla_enable_dumping = false, xla_llvm_force_inline_before_split = false, xla_dump_disable_metadata = false, xla_dump_hlo_pipeline_re = "", xla_cpu_use_acl = false, xla_cpu_strict_dot_conv_math = false, xla_dump_latency_hiding_schedule = false, xla_partitioning_algorithm = var"DebugOptions.PartitioningAlgorithm".PARTITIONING_ALGORITHM_NOOP, xla_debug_buffer_assignment_show_max = zero(Int64), xla_detect_unstable_reductions = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE, xla_detect_unstable_reductions_post_optimizations = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE, xla_gpu_detect_nan = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE, xla_gpu_detect_inf = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE, xla_dump_large_constants = false, xla_reduce_window_rewrite_base_length = zero(Int64), xla_cmd_buffer_trace_cache_size = zero(Int64), xla_syntax_sugar_async_ops = false, xla_enable_command_buffers_during_profiling = false, xla_ignore_channel_id = false, xla_pjrt_allow_auto_layout_in_hlo = false, xla_test_add_command_buffer_mode = false, xla_gpu_experimental_matmul_perf_table_path = "", xla_early_exit_with_layouts = false, xla_gpu_experimental_scaled_dot_with_triton = false, xla_gpu_experimental_use_raft_select_k = false, xla_backend_extra_options = Dict{String,String}())
PB.field_numbers(::Type{DebugOptions}) = (;xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled = 439, xla_disable_automatic_host_compute_offload = 408, xla_enable_scoped_logging_timers = 436, xla_hlo_pass_fix_detect_cycles = 370, xla_keep_shardings_after_spmd = 419, xla_unsupported_crash_on_hlo_pass_fix_max_iterations = 363, xla_unsupported_crash_on_hlo_pass_noop_change = 379, xla_unsupported_crash_on_hlo_pass_silent_hlo_change = 380, xla_cpu_collective_call_terminate_timeout_seconds = 417, xla_cpu_collective_call_warn_stuck_seconds = 418, xla_cpu_collective_timeout_seconds = 438, xla_cpu_copy_insertion_use_region_analysis = 337, xla_cpu_emitter_verification_level = 395, xla_cpu_enable_concurrency_optimized_scheduler = 307, xla_cpu_enable_fast_math = 99, xla_cpu_enable_fast_min_max = 140, xla_cpu_enable_platform_dependent_math = 425, xla_cpu_experimental_onednn_custom_call = 412, xla_cpu_experimental_onednn_fusion_type = 399, xla_cpu_experimental_xnn_fusion_type = 400, xla_cpu_experimental_xnn_graph_fusion_mode = 365, xla_cpu_experimental_ynn_fusion_type = 422, xla_cpu_fast_math_honor_division = 126, xla_cpu_fast_math_honor_functions = 129, xla_cpu_fast_math_honor_infs = 121, xla_cpu_fast_math_honor_nans = 120, xla_cpu_generate_unique_c_style_kernel_entry_points = 372, xla_cpu_max_isa = 333, xla_cpu_parallel_codegen_split_count = 323, xla_cpu_prefer_vector_width = 308, xla_cpu_use_fusion_emitters = 376, xla_cpu_use_xnnpack = 359, xla_enable_fast_math = 335, xla_gpu_experimental_thunk_buffer_debug_filter = 424, xla_dump_hlo_unoptimized_snapshots = 405, xla_enable_enzyme_comms_opt = 429, xla_gpu_algorithm_denylist_path = 128, xla_gpu_all_gather_combine_threshold_bytes = 212, xla_gpu_all_reduce_blueconnect_num_devices_per_host = 159, xla_gpu_all_reduce_combine_threshold_bytes = 157, xla_gpu_analytical_latency_estimator_options = 357, xla_gpu_async_dot = 321, xla_gpu_auto_spmd_partitioning_memory_budget_gb = 224, xla_gpu_auto_spmd_partitioning_memory_budget_ratio = 225, xla_gpu_autotune_gemm_rtol = 316, xla_gpu_autotune_level = 123, xla_gpu_autotune_max_solutions = 288, xla_gpu_collect_cost_model_stats = 240, xla_gpu_collective_inflation_factor = 205, xla_gpu_collective_permute_combine_threshold_bytes = 378, xla_gpu_collective_permute_decomposer_threshold = 237, xla_gpu_collectives_use_persistent_cliques = 354, xla_gpu_command_buffer_scheduling_mode = 404, xla_gpu_command_buffer_unroll_loops = 411, xla_gpu_copy_insertion_use_region_analysis = 236, xla_gpu_crash_on_verification_failures = 101, xla_gpu_cublas_fallback = 247, xla_gpu_cuda_data_dir = 61, xla_gpu_cudnn_gemm_fusion_level = 285, xla_gpu_cudnn_gemm_max_plans = 318, xla_gpu_deterministic_ops = 148, xla_gpu_disable_async_collectives = 289, xla_gpu_disable_gpuasm_optimizations = 103, xla_gpu_dot_merger_threshold_mb = 331, xla_gpu_dump_autotune_logs_to = 292, xla_gpu_dump_autotune_results_to = 222, xla_gpu_dump_autotuned_gemm_fusions = 232, xla_gpu_dump_llvmir = 155, xla_gpu_enable_all_gather_combine_by_dim = 254, xla_gpu_enable_analytical_latency_estimator = 255, xla_gpu_enable_analytical_sol_latency_estimator = 356, xla_gpu_enable_approx_costly_collectives = 305, xla_gpu_enable_command_buffer = 258, xla_gpu_enable_cub_radix_sort = 259, xla_gpu_enable_cublaslt = 166, xla_gpu_enable_cudnn_int8x32_convolution_reordering = 189, xla_gpu_enable_cudnn_layer_norm = 262, xla_gpu_enable_dynamic_slice_fusion = 105, xla_gpu_enable_fast_min_max = 100, xla_gpu_enable_highest_priority_async_stream = 216, xla_gpu_enable_host_memory_offloading = 296, xla_gpu_enable_latency_hiding_scheduler = 186, xla_gpu_enable_libnvptxcompiler = 269, xla_gpu_enable_llvm_module_compilation_parallelism = 268, xla_gpu_enable_nccl_clique_optimization = 244, xla_gpu_enable_nccl_comm_splitting = 272, xla_gpu_enable_nccl_user_buffers = 267, xla_gpu_enable_pipelined_all_gather = 227, xla_gpu_enable_pipelined_all_reduce = 217, xla_gpu_enable_pipelined_collectives = 239, xla_gpu_enable_pipelined_p2p = 246, xla_gpu_enable_pipelined_reduce_scatter = 231, xla_gpu_enable_reassociation_for_converted_ar = 209, xla_gpu_enable_reduce_scatter_combine_by_dim = 257, xla_gpu_enable_reduction_epilogue_fusion = 243, xla_gpu_enable_scatter_determinism_expander = 345, xla_gpu_enable_shared_constants = 165, xla_gpu_enable_split_k_autotuning = 241, xla_gpu_enable_triton_gemm = 188, xla_gpu_enable_while_loop_double_buffering = 248, xla_gpu_enable_while_loop_reduce_scatter_code_motion = 203, xla_gpu_enable_while_loop_unrolling = 294, xla_gpu_exclude_nondeterministic_ops = 297, xla_gpu_executable_embed_debug_info = 437, xla_gpu_executable_terminate_timeout_seconds = 328, xla_gpu_executable_warn_stuck_timeout_seconds = 327, xla_gpu_exhaustive_tiling_search = 219, xla_gpu_experimental_allow_unroll_factor_eight = 430, xla_gpu_experimental_aot_compiled_thunks = 435, xla_gpu_experimental_autotune_cache_mode = 324, xla_gpu_experimental_autotuner_cache_dir = 407, xla_gpu_experimental_collective_cse_distance_threshold = 374, xla_gpu_experimental_collective_perf_table_path = 377, xla_gpu_experimental_disable_binary_libraries = 329, xla_gpu_experimental_dump_fdo_profiles = 338, xla_gpu_experimental_dump_gpu_executable = 427, xla_gpu_experimental_enable_alltoall_windowed_einsum = 360, xla_gpu_experimental_enable_buffer_saver_on_thunks = 431, xla_gpu_experimental_enable_checksum_tracing_on_thunks = 414, xla_gpu_experimental_enable_fusion_autotuner = 409, xla_gpu_experimental_enable_fusion_block_level_rewriter = 334, xla_gpu_experimental_enable_heuristic_collective_combining = 366, xla_gpu_experimental_enable_nccl_symmetric_buffers = 406, xla_gpu_experimental_enable_nvshmem = 388, xla_gpu_experimental_enable_split_k_rewrite = 386, xla_gpu_experimental_enable_subchannel_dequantisation_fusion = 368, xla_gpu_experimental_enable_triton_heroless_priority_fusion = 340, xla_gpu_experimental_enable_triton_warp_specialization = 421, xla_gpu_experimental_pack_dot_operands_along_k_dimension = 362, xla_gpu_experimental_parallel_collective_overlap_limit = 336, xla_gpu_experimental_pipeline_parallelism_opt_level = 351, xla_gpu_experimental_stream_annotation = 342, xla_gpu_experimental_use_autotuner_pass = 396, xla_gpu_experimental_use_ragged_dot_fusion = 401, xla_gpu_fail_ptx_compilation_on_register_spilling = 353, xla_gpu_filter_kernels_spilling_registers_on_autotuning = 250, xla_gpu_first_collective_call_terminate_timeout_seconds = 392, xla_gpu_first_collective_call_warn_stuck_timeout_seconds = 391, xla_gpu_force_compilation_parallelism = 147, xla_gpu_force_conv_nchw = 125, xla_gpu_force_conv_nhwc = 146, xla_gpu_ftz = 62, xla_gpu_fused_attention_use_cudnn_rng = 235, xla_gpu_gemm_autotuner_override_file = 434, xla_gpu_gemm_rewrite_size_threshold = 283, xla_gpu_generate_debug_info = 348, xla_gpu_generate_line_info = 349, xla_gpu_graph_enable_concurrent_region = 215, xla_gpu_graph_min_graph_size = 208, xla_gpu_kernel_cache_file = 306, xla_gpu_libnvjitlink_mode = 343, xla_gpu_llvm_ir_file = 150, xla_gpu_llvm_verification_level = 256, xla_gpu_load_autotune_results_from = 223, xla_gpu_memory_limit_slop_factor = 260, xla_gpu_mock_custom_calls = 245, xla_gpu_multi_streamed_windowed_einsum = 280, xla_gpu_nccl_async_execution = 393, xla_gpu_nccl_blocking_communicators = 390, xla_gpu_nccl_collective_max_nchannels = 273, xla_gpu_nccl_init_max_rank_per_root_ratio = 277, xla_gpu_nccl_p2p_max_nchannels = 274, xla_gpu_nccl_terminate_on_error = 301, xla_gpu_nccl_termination_timeout_seconds = 163, xla_gpu_operand_bytes_threshold_for_windowed_einsum = 339, xla_gpu_override_gemm_autotuner = 295, xla_gpu_per_fusion_autotune_cache_dir = 310, xla_gpu_pgle_accuracy_checker = 341, xla_gpu_pgle_profile_file_or_directory_path = 210, xla_gpu_ptx_file = 127, xla_gpu_reduce_scatter_combine_threshold_bytes = 213, xla_gpu_redzone_padding_bytes = 228, xla_gpu_require_complete_aot_autotune_results = 284, xla_gpu_require_exclusive_lock = 347, xla_gpu_shape_checks = 170, xla_gpu_shard_autotuning = 304, xla_gpu_strict_conv_algorithm_picker = 156, xla_gpu_target_config_filename = 261, xla_gpu_temp_buffer_use_separate_color = 312, xla_gpu_threshold_for_windowed_einsum_mib = 265, xla_gpu_triton_gemm_any = 190, xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found = 138, xla_gpu_unsupported_enable_all_reduce_decomposer = 384, xla_gpu_unsupported_enable_ragged_all_to_all_decomposer = 350, xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer = 415, xla_gpu_unsupported_enable_triton_gemm = 322, xla_gpu_unsupported_enable_triton_multi_output_fusion = 382, xla_gpu_unsupported_override_fast_interconnect_slice_size = 416, xla_gpu_unsupported_use_all_reduce_one_shot_kernel = 387, xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel = 375, xla_gpu_use_embeded_device_lib = 420, xla_gpu_use_inprocess_lld = 389, xla_gpu_use_memcpy_local_p2p = 287, xla_gpu_use_runtime_fusion = 181, xla_gpu_verify_triton_fusion_numerics = 291, xla_hlo_graph_addresses = 2, xla_hlo_profile = 9, xla_disable_hlo_passes = 30, xla_enable_hlo_passes_only = 124, xla_disable_all_hlo_passes = 104, xla_backend_optimization_level = 31, xla_embed_ir_in_executable = 33, xla_eliminate_hlo_implicit_broadcast = 35, xla_cpu_multi_thread_eigen = 60, xla_llvm_enable_alias_scope_metadata = 70, xla_llvm_enable_noalias_metadata = 71, xla_llvm_enable_invariant_load_metadata = 72, xla_llvm_disable_expensive_passes = 73, xla_test_all_output_layouts = 90, xla_test_all_input_layouts = 91, xla_hlo_graph_sharding_color = 92, xla_cpu_use_onednn = 97, xla_allow_excess_precision = 122, xla_force_host_platform_device_count = 102, xla_hlo_evaluator_use_fast_path = 106, xla_allow_scalar_index_dynamic_ops = 107, xla_step_marker_location = 108, xla_dump_to = 109, xla_flags_reset = 364, xla_dump_hlo_module_re = 110, xla_dump_hlo_pass_re = 111, xla_dump_emitter_re = 433, xla_dump_hlo_as_text = 112, xla_dump_hlo_as_proto = 113, xla_dump_hlo_as_dot = 114, xla_dump_hlo_as_url = 115, xla_dump_hlo_as_html = 116, xla_dump_fusion_visualization = 149, xla_dump_hlo_snapshots = 118, xla_dump_include_timestamp = 131, xla_dump_max_hlo_modules = 132, xla_dump_module_metadata = 144, xla_dump_compress_protos = 151, xla_dump_hlo_as_long_text = 164, xla_dump_enable_mlir_pretty_form = 185, xla_dump_full_hlo_config = 381, xla_tpu_detect_nan = 135, xla_tpu_detect_inf = 136, xla_cpu_enable_xprof_traceme = 137, xla_multiheap_size_constraint_per_heap = 142, xla_detailed_logging = 252, xla_enable_dumping = 253, xla_llvm_force_inline_before_split = 300, xla_dump_disable_metadata = 153, xla_dump_hlo_pipeline_re = 154, xla_cpu_use_acl = 174, xla_cpu_strict_dot_conv_math = 175, xla_dump_latency_hiding_schedule = 182, xla_partitioning_algorithm = 187, xla_debug_buffer_assignment_show_max = 251, xla_detect_unstable_reductions = 403, xla_detect_unstable_reductions_post_optimizations = 432, xla_gpu_detect_nan = 426, xla_gpu_detect_inf = 428, xla_dump_large_constants = 290, xla_reduce_window_rewrite_base_length = 293, xla_cmd_buffer_trace_cache_size = 311, xla_syntax_sugar_async_ops = 315, xla_enable_command_buffers_during_profiling = 317, xla_ignore_channel_id = 330, xla_pjrt_allow_auto_layout_in_hlo = 344, xla_test_add_command_buffer_mode = 373, xla_gpu_experimental_matmul_perf_table_path = 383, xla_early_exit_with_layouts = 397, xla_gpu_experimental_scaled_dot_with_triton = 410, xla_gpu_experimental_use_raft_select_k = 413, xla_backend_extra_options = 500)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:DebugOptions})
    xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled = false
    xla_disable_automatic_host_compute_offload = false
    xla_enable_scoped_logging_timers = false
    xla_hlo_pass_fix_detect_cycles = false
    xla_keep_shardings_after_spmd = false
    xla_unsupported_crash_on_hlo_pass_fix_max_iterations = false
    xla_unsupported_crash_on_hlo_pass_noop_change = false
    xla_unsupported_crash_on_hlo_pass_silent_hlo_change = false
    xla_cpu_collective_call_terminate_timeout_seconds = zero(Int32)
    xla_cpu_collective_call_warn_stuck_seconds = zero(Int32)
    xla_cpu_collective_timeout_seconds = zero(Int32)
    xla_cpu_copy_insertion_use_region_analysis = false
    xla_cpu_emitter_verification_level = zero(Int32)
    xla_cpu_enable_concurrency_optimized_scheduler = false
    xla_cpu_enable_fast_math = false
    xla_cpu_enable_fast_min_max = false
    xla_cpu_enable_platform_dependent_math = false
    xla_cpu_experimental_onednn_custom_call = false
    xla_cpu_experimental_onednn_fusion_type = PB.BufferedVector{var"DebugOptions.LibraryFusionType".T}()
    xla_cpu_experimental_xnn_fusion_type = PB.BufferedVector{var"DebugOptions.LibraryFusionType".T}()
    xla_cpu_experimental_xnn_graph_fusion_mode = var"DebugOptions.XnnGraphFusionMode".XNN_GRAPH_FUSION_MODE_DISABLED
    xla_cpu_experimental_ynn_fusion_type = PB.BufferedVector{var"DebugOptions.LibraryFusionType".T}()
    xla_cpu_fast_math_honor_division = false
    xla_cpu_fast_math_honor_functions = false
    xla_cpu_fast_math_honor_infs = false
    xla_cpu_fast_math_honor_nans = false
    xla_cpu_generate_unique_c_style_kernel_entry_points = false
    xla_cpu_max_isa = ""
    xla_cpu_parallel_codegen_split_count = zero(Int32)
    xla_cpu_prefer_vector_width = zero(Int32)
    xla_cpu_use_fusion_emitters = false
    xla_cpu_use_xnnpack = false
    xla_enable_fast_math = false
    xla_gpu_experimental_thunk_buffer_debug_filter = Ref{Union{Nothing,ThunkBufferDebugFilter}}(nothing)
    xla_dump_hlo_unoptimized_snapshots = false
    xla_enable_enzyme_comms_opt = false
    xla_gpu_algorithm_denylist_path = ""
    xla_gpu_all_gather_combine_threshold_bytes = zero(Int64)
    xla_gpu_all_reduce_blueconnect_num_devices_per_host = zero(Int32)
    xla_gpu_all_reduce_combine_threshold_bytes = zero(Int64)
    xla_gpu_analytical_latency_estimator_options = Dict{String,String}()
    xla_gpu_async_dot = false
    xla_gpu_auto_spmd_partitioning_memory_budget_gb = zero(Int32)
    xla_gpu_auto_spmd_partitioning_memory_budget_ratio = zero(Float32)
    xla_gpu_autotune_gemm_rtol = zero(Float32)
    xla_gpu_autotune_level = zero(Int32)
    xla_gpu_autotune_max_solutions = zero(Int64)
    xla_gpu_collect_cost_model_stats = false
    xla_gpu_collective_inflation_factor = zero(Int32)
    xla_gpu_collective_permute_combine_threshold_bytes = zero(Int64)
    xla_gpu_collective_permute_decomposer_threshold = zero(Int64)
    xla_gpu_collectives_use_persistent_cliques = false
    xla_gpu_command_buffer_scheduling_mode = var"DebugOptions.CommandBufferSchedulingMode".SERIALIZE
    xla_gpu_command_buffer_unroll_loops = false
    xla_gpu_copy_insertion_use_region_analysis = false
    xla_gpu_crash_on_verification_failures = false
    xla_gpu_cublas_fallback = false
    xla_gpu_cuda_data_dir = ""
    xla_gpu_cudnn_gemm_fusion_level = zero(Int32)
    xla_gpu_cudnn_gemm_max_plans = zero(Int32)
    xla_gpu_deterministic_ops = false
    xla_gpu_disable_async_collectives = PB.BufferedVector{var"DebugOptions.CollectiveOpType".T}()
    xla_gpu_disable_gpuasm_optimizations = false
    xla_gpu_dot_merger_threshold_mb = zero(Int32)
    xla_gpu_dump_autotune_logs_to = ""
    xla_gpu_dump_autotune_results_to = ""
    xla_gpu_dump_autotuned_gemm_fusions = false
    xla_gpu_dump_llvmir = false
    xla_gpu_enable_all_gather_combine_by_dim = false
    xla_gpu_enable_analytical_latency_estimator = false
    xla_gpu_enable_analytical_sol_latency_estimator = false
    xla_gpu_enable_approx_costly_collectives = false
    xla_gpu_enable_command_buffer = PB.BufferedVector{var"DebugOptions.CommandBufferCmdType".T}()
    xla_gpu_enable_cub_radix_sort = false
    xla_gpu_enable_cublaslt = false
    xla_gpu_enable_cudnn_int8x32_convolution_reordering = false
    xla_gpu_enable_cudnn_layer_norm = false
    xla_gpu_enable_dynamic_slice_fusion = false
    xla_gpu_enable_fast_min_max = false
    xla_gpu_enable_highest_priority_async_stream = false
    xla_gpu_enable_host_memory_offloading = false
    xla_gpu_enable_latency_hiding_scheduler = false
    xla_gpu_enable_libnvptxcompiler = false
    xla_gpu_enable_llvm_module_compilation_parallelism = false
    xla_gpu_enable_nccl_clique_optimization = false
    xla_gpu_enable_nccl_comm_splitting = false
    xla_gpu_enable_nccl_user_buffers = false
    xla_gpu_enable_pipelined_all_gather = false
    xla_gpu_enable_pipelined_all_reduce = false
    xla_gpu_enable_pipelined_collectives = false
    xla_gpu_enable_pipelined_p2p = false
    xla_gpu_enable_pipelined_reduce_scatter = false
    xla_gpu_enable_reassociation_for_converted_ar = false
    xla_gpu_enable_reduce_scatter_combine_by_dim = false
    xla_gpu_enable_reduction_epilogue_fusion = false
    xla_gpu_enable_scatter_determinism_expander = false
    xla_gpu_enable_shared_constants = false
    xla_gpu_enable_split_k_autotuning = false
    xla_gpu_enable_triton_gemm = false
    xla_gpu_enable_while_loop_double_buffering = false
    xla_gpu_enable_while_loop_reduce_scatter_code_motion = false
    xla_gpu_enable_while_loop_unrolling = var"DebugOptions.WhileLoopUnrolling".WHILE_LOOP_UNROLLING_NO_UNROLL
    xla_gpu_exclude_nondeterministic_ops = false
    xla_gpu_executable_embed_debug_info = false
    xla_gpu_executable_terminate_timeout_seconds = zero(Int32)
    xla_gpu_executable_warn_stuck_timeout_seconds = zero(Int32)
    xla_gpu_exhaustive_tiling_search = false
    xla_gpu_experimental_allow_unroll_factor_eight = false
    xla_gpu_experimental_aot_compiled_thunks = false
    xla_gpu_experimental_autotune_cache_mode = var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_UNSPECIFIED
    xla_gpu_experimental_autotuner_cache_dir = ""
    xla_gpu_experimental_collective_cse_distance_threshold = zero(Int64)
    xla_gpu_experimental_collective_perf_table_path = ""
    xla_gpu_experimental_disable_binary_libraries = false
    xla_gpu_experimental_dump_fdo_profiles = false
    xla_gpu_experimental_dump_gpu_executable = false
    xla_gpu_experimental_enable_alltoall_windowed_einsum = false
    xla_gpu_experimental_enable_buffer_saver_on_thunks = false
    xla_gpu_experimental_enable_checksum_tracing_on_thunks = false
    xla_gpu_experimental_enable_fusion_autotuner = false
    xla_gpu_experimental_enable_fusion_block_level_rewriter = false
    xla_gpu_experimental_enable_heuristic_collective_combining = false
    xla_gpu_experimental_enable_nccl_symmetric_buffers = false
    xla_gpu_experimental_enable_nvshmem = false
    xla_gpu_experimental_enable_split_k_rewrite = false
    xla_gpu_experimental_enable_subchannel_dequantisation_fusion = false
    xla_gpu_experimental_enable_triton_heroless_priority_fusion = false
    xla_gpu_experimental_enable_triton_warp_specialization = false
    xla_gpu_experimental_pack_dot_operands_along_k_dimension = false
    xla_gpu_experimental_parallel_collective_overlap_limit = zero(Int32)
    xla_gpu_experimental_pipeline_parallelism_opt_level = var"DebugOptions.PipelineParallelismOptLevel".PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE
    xla_gpu_experimental_stream_annotation = false
    xla_gpu_experimental_use_autotuner_pass = false
    xla_gpu_experimental_use_ragged_dot_fusion = false
    xla_gpu_fail_ptx_compilation_on_register_spilling = false
    xla_gpu_filter_kernels_spilling_registers_on_autotuning = false
    xla_gpu_first_collective_call_terminate_timeout_seconds = zero(Int32)
    xla_gpu_first_collective_call_warn_stuck_timeout_seconds = zero(Int32)
    xla_gpu_force_compilation_parallelism = zero(Int32)
    xla_gpu_force_conv_nchw = false
    xla_gpu_force_conv_nhwc = false
    xla_gpu_ftz = false
    xla_gpu_fused_attention_use_cudnn_rng = false
    xla_gpu_gemm_autotuner_override_file = ""
    xla_gpu_gemm_rewrite_size_threshold = zero(Int64)
    xla_gpu_generate_debug_info = false
    xla_gpu_generate_line_info = false
    xla_gpu_graph_enable_concurrent_region = false
    xla_gpu_graph_min_graph_size = zero(Int32)
    xla_gpu_kernel_cache_file = ""
    xla_gpu_libnvjitlink_mode = var"DebugOptions.LibNvJitLinkMode".LIB_NV_JIT_LINK_MODE_AUTO
    xla_gpu_llvm_ir_file = PB.BufferedVector{String}()
    xla_gpu_llvm_verification_level = zero(Int32)
    xla_gpu_load_autotune_results_from = ""
    xla_gpu_memory_limit_slop_factor = zero(Int32)
    xla_gpu_mock_custom_calls = false
    xla_gpu_multi_streamed_windowed_einsum = false
    xla_gpu_nccl_async_execution = false
    xla_gpu_nccl_blocking_communicators = false
    xla_gpu_nccl_collective_max_nchannels = zero(Int64)
    xla_gpu_nccl_init_max_rank_per_root_ratio = zero(Int64)
    xla_gpu_nccl_p2p_max_nchannels = zero(Int64)
    xla_gpu_nccl_terminate_on_error = false
    xla_gpu_nccl_termination_timeout_seconds = zero(Int64)
    xla_gpu_operand_bytes_threshold_for_windowed_einsum = zero(Int64)
    xla_gpu_override_gemm_autotuner = ""
    xla_gpu_per_fusion_autotune_cache_dir = ""
    xla_gpu_pgle_accuracy_checker = var"DebugOptions.PGLEStrictnessLevel".PGLE_STRICTNESS_LEVEL_OFF
    xla_gpu_pgle_profile_file_or_directory_path = ""
    xla_gpu_ptx_file = PB.BufferedVector{String}()
    xla_gpu_reduce_scatter_combine_threshold_bytes = zero(Int64)
    xla_gpu_redzone_padding_bytes = zero(Int64)
    xla_gpu_require_complete_aot_autotune_results = false
    xla_gpu_require_exclusive_lock = false
    xla_gpu_shape_checks = var"DebugOptions.ShapeChecks".IGNORE
    xla_gpu_shard_autotuning = false
    xla_gpu_strict_conv_algorithm_picker = false
    xla_gpu_target_config_filename = ""
    xla_gpu_temp_buffer_use_separate_color = false
    xla_gpu_threshold_for_windowed_einsum_mib = zero(Int64)
    xla_gpu_triton_gemm_any = false
    xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found = false
    xla_gpu_unsupported_enable_all_reduce_decomposer = false
    xla_gpu_unsupported_enable_ragged_all_to_all_decomposer = false
    xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer = false
    xla_gpu_unsupported_enable_triton_gemm = false
    xla_gpu_unsupported_enable_triton_multi_output_fusion = false
    xla_gpu_unsupported_override_fast_interconnect_slice_size = zero(Int64)
    xla_gpu_unsupported_use_all_reduce_one_shot_kernel = false
    xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel = false
    xla_gpu_use_embeded_device_lib = false
    xla_gpu_use_inprocess_lld = false
    xla_gpu_use_memcpy_local_p2p = false
    xla_gpu_use_runtime_fusion = false
    xla_gpu_verify_triton_fusion_numerics = false
    xla_hlo_graph_addresses = false
    xla_hlo_profile = false
    xla_disable_hlo_passes = PB.BufferedVector{String}()
    xla_enable_hlo_passes_only = PB.BufferedVector{String}()
    xla_disable_all_hlo_passes = false
    xla_backend_optimization_level = zero(Int32)
    xla_embed_ir_in_executable = false
    xla_eliminate_hlo_implicit_broadcast = false
    xla_cpu_multi_thread_eigen = false
    xla_llvm_enable_alias_scope_metadata = false
    xla_llvm_enable_noalias_metadata = false
    xla_llvm_enable_invariant_load_metadata = false
    xla_llvm_disable_expensive_passes = false
    xla_test_all_output_layouts = false
    xla_test_all_input_layouts = false
    xla_hlo_graph_sharding_color = false
    xla_cpu_use_onednn = false
    xla_allow_excess_precision = false
    xla_force_host_platform_device_count = zero(Int32)
    xla_hlo_evaluator_use_fast_path = false
    xla_allow_scalar_index_dynamic_ops = false
    xla_step_marker_location = var"DebugOptions.StepMarkerLocation".STEP_MARK_AT_ENTRY
    xla_dump_to = ""
    xla_flags_reset = false
    xla_dump_hlo_module_re = ""
    xla_dump_hlo_pass_re = ""
    xla_dump_emitter_re = ""
    xla_dump_hlo_as_text = false
    xla_dump_hlo_as_proto = false
    xla_dump_hlo_as_dot = false
    xla_dump_hlo_as_url = false
    xla_dump_hlo_as_html = false
    xla_dump_fusion_visualization = false
    xla_dump_hlo_snapshots = false
    xla_dump_include_timestamp = false
    xla_dump_max_hlo_modules = zero(Int32)
    xla_dump_module_metadata = false
    xla_dump_compress_protos = false
    xla_dump_hlo_as_long_text = false
    xla_dump_enable_mlir_pretty_form = false
    xla_dump_full_hlo_config = false
    xla_tpu_detect_nan = false
    xla_tpu_detect_inf = false
    xla_cpu_enable_xprof_traceme = false
    xla_multiheap_size_constraint_per_heap = zero(Int32)
    xla_detailed_logging = false
    xla_enable_dumping = false
    xla_llvm_force_inline_before_split = false
    xla_dump_disable_metadata = false
    xla_dump_hlo_pipeline_re = ""
    xla_cpu_use_acl = false
    xla_cpu_strict_dot_conv_math = false
    xla_dump_latency_hiding_schedule = false
    xla_partitioning_algorithm = var"DebugOptions.PartitioningAlgorithm".PARTITIONING_ALGORITHM_NOOP
    xla_debug_buffer_assignment_show_max = zero(Int64)
    xla_detect_unstable_reductions = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE
    xla_detect_unstable_reductions_post_optimizations = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE
    xla_gpu_detect_nan = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE
    xla_gpu_detect_inf = var"DebugOptions.DetectionMode".DETECTION_MODE_NONE
    xla_dump_large_constants = false
    xla_reduce_window_rewrite_base_length = zero(Int64)
    xla_cmd_buffer_trace_cache_size = zero(Int64)
    xla_syntax_sugar_async_ops = false
    xla_enable_command_buffers_during_profiling = false
    xla_ignore_channel_id = false
    xla_pjrt_allow_auto_layout_in_hlo = false
    xla_test_add_command_buffer_mode = false
    xla_gpu_experimental_matmul_perf_table_path = ""
    xla_early_exit_with_layouts = false
    xla_gpu_experimental_scaled_dot_with_triton = false
    xla_gpu_experimental_use_raft_select_k = false
    xla_backend_extra_options = Dict{String,String}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 439
            xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled = PB.decode(d, Bool)
        elseif field_number == 408
            xla_disable_automatic_host_compute_offload = PB.decode(d, Bool)
        elseif field_number == 436
            xla_enable_scoped_logging_timers = PB.decode(d, Bool)
        elseif field_number == 370
            xla_hlo_pass_fix_detect_cycles = PB.decode(d, Bool)
        elseif field_number == 419
            xla_keep_shardings_after_spmd = PB.decode(d, Bool)
        elseif field_number == 363
            xla_unsupported_crash_on_hlo_pass_fix_max_iterations = PB.decode(d, Bool)
        elseif field_number == 379
            xla_unsupported_crash_on_hlo_pass_noop_change = PB.decode(d, Bool)
        elseif field_number == 380
            xla_unsupported_crash_on_hlo_pass_silent_hlo_change = PB.decode(d, Bool)
        elseif field_number == 417
            xla_cpu_collective_call_terminate_timeout_seconds = PB.decode(d, Int32)
        elseif field_number == 418
            xla_cpu_collective_call_warn_stuck_seconds = PB.decode(d, Int32)
        elseif field_number == 438
            xla_cpu_collective_timeout_seconds = PB.decode(d, Int32)
        elseif field_number == 337
            xla_cpu_copy_insertion_use_region_analysis = PB.decode(d, Bool)
        elseif field_number == 395
            xla_cpu_emitter_verification_level = PB.decode(d, Int32)
        elseif field_number == 307
            xla_cpu_enable_concurrency_optimized_scheduler = PB.decode(d, Bool)
        elseif field_number == 99
            xla_cpu_enable_fast_math = PB.decode(d, Bool)
        elseif field_number == 140
            xla_cpu_enable_fast_min_max = PB.decode(d, Bool)
        elseif field_number == 425
            xla_cpu_enable_platform_dependent_math = PB.decode(d, Bool)
        elseif field_number == 412
            xla_cpu_experimental_onednn_custom_call = PB.decode(d, Bool)
        elseif field_number == 399
            PB.decode!(d, wire_type, xla_cpu_experimental_onednn_fusion_type)
        elseif field_number == 400
            PB.decode!(d, wire_type, xla_cpu_experimental_xnn_fusion_type)
        elseif field_number == 365
            xla_cpu_experimental_xnn_graph_fusion_mode = PB.decode(d, var"DebugOptions.XnnGraphFusionMode".T)
        elseif field_number == 422
            PB.decode!(d, wire_type, xla_cpu_experimental_ynn_fusion_type)
        elseif field_number == 126
            xla_cpu_fast_math_honor_division = PB.decode(d, Bool)
        elseif field_number == 129
            xla_cpu_fast_math_honor_functions = PB.decode(d, Bool)
        elseif field_number == 121
            xla_cpu_fast_math_honor_infs = PB.decode(d, Bool)
        elseif field_number == 120
            xla_cpu_fast_math_honor_nans = PB.decode(d, Bool)
        elseif field_number == 372
            xla_cpu_generate_unique_c_style_kernel_entry_points = PB.decode(d, Bool)
        elseif field_number == 333
            xla_cpu_max_isa = PB.decode(d, String)
        elseif field_number == 323
            xla_cpu_parallel_codegen_split_count = PB.decode(d, Int32)
        elseif field_number == 308
            xla_cpu_prefer_vector_width = PB.decode(d, Int32)
        elseif field_number == 376
            xla_cpu_use_fusion_emitters = PB.decode(d, Bool)
        elseif field_number == 359
            xla_cpu_use_xnnpack = PB.decode(d, Bool)
        elseif field_number == 335
            xla_enable_fast_math = PB.decode(d, Bool)
        elseif field_number == 424
            PB.decode!(d, xla_gpu_experimental_thunk_buffer_debug_filter)
        elseif field_number == 405
            xla_dump_hlo_unoptimized_snapshots = PB.decode(d, Bool)
        elseif field_number == 429
            xla_enable_enzyme_comms_opt = PB.decode(d, Bool)
        elseif field_number == 128
            xla_gpu_algorithm_denylist_path = PB.decode(d, String)
        elseif field_number == 212
            xla_gpu_all_gather_combine_threshold_bytes = PB.decode(d, Int64)
        elseif field_number == 159
            xla_gpu_all_reduce_blueconnect_num_devices_per_host = PB.decode(d, Int32)
        elseif field_number == 157
            xla_gpu_all_reduce_combine_threshold_bytes = PB.decode(d, Int64)
        elseif field_number == 357
            PB.decode!(d, xla_gpu_analytical_latency_estimator_options)
        elseif field_number == 321
            xla_gpu_async_dot = PB.decode(d, Bool)
        elseif field_number == 224
            xla_gpu_auto_spmd_partitioning_memory_budget_gb = PB.decode(d, Int32)
        elseif field_number == 225
            xla_gpu_auto_spmd_partitioning_memory_budget_ratio = PB.decode(d, Float32)
        elseif field_number == 316
            xla_gpu_autotune_gemm_rtol = PB.decode(d, Float32)
        elseif field_number == 123
            xla_gpu_autotune_level = PB.decode(d, Int32)
        elseif field_number == 288
            xla_gpu_autotune_max_solutions = PB.decode(d, Int64)
        elseif field_number == 240
            xla_gpu_collect_cost_model_stats = PB.decode(d, Bool)
        elseif field_number == 205
            xla_gpu_collective_inflation_factor = PB.decode(d, Int32)
        elseif field_number == 378
            xla_gpu_collective_permute_combine_threshold_bytes = PB.decode(d, Int64)
        elseif field_number == 237
            xla_gpu_collective_permute_decomposer_threshold = PB.decode(d, Int64)
        elseif field_number == 354
            xla_gpu_collectives_use_persistent_cliques = PB.decode(d, Bool)
        elseif field_number == 404
            xla_gpu_command_buffer_scheduling_mode = PB.decode(d, var"DebugOptions.CommandBufferSchedulingMode".T)
        elseif field_number == 411
            xla_gpu_command_buffer_unroll_loops = PB.decode(d, Bool)
        elseif field_number == 236
            xla_gpu_copy_insertion_use_region_analysis = PB.decode(d, Bool)
        elseif field_number == 101
            xla_gpu_crash_on_verification_failures = PB.decode(d, Bool)
        elseif field_number == 247
            xla_gpu_cublas_fallback = PB.decode(d, Bool)
        elseif field_number == 61
            xla_gpu_cuda_data_dir = PB.decode(d, String)
        elseif field_number == 285
            xla_gpu_cudnn_gemm_fusion_level = PB.decode(d, Int32)
        elseif field_number == 318
            xla_gpu_cudnn_gemm_max_plans = PB.decode(d, Int32)
        elseif field_number == 148
            xla_gpu_deterministic_ops = PB.decode(d, Bool)
        elseif field_number == 289
            PB.decode!(d, wire_type, xla_gpu_disable_async_collectives)
        elseif field_number == 103
            xla_gpu_disable_gpuasm_optimizations = PB.decode(d, Bool)
        elseif field_number == 331
            xla_gpu_dot_merger_threshold_mb = PB.decode(d, Int32)
        elseif field_number == 292
            xla_gpu_dump_autotune_logs_to = PB.decode(d, String)
        elseif field_number == 222
            xla_gpu_dump_autotune_results_to = PB.decode(d, String)
        elseif field_number == 232
            xla_gpu_dump_autotuned_gemm_fusions = PB.decode(d, Bool)
        elseif field_number == 155
            xla_gpu_dump_llvmir = PB.decode(d, Bool)
        elseif field_number == 254
            xla_gpu_enable_all_gather_combine_by_dim = PB.decode(d, Bool)
        elseif field_number == 255
            xla_gpu_enable_analytical_latency_estimator = PB.decode(d, Bool)
        elseif field_number == 356
            xla_gpu_enable_analytical_sol_latency_estimator = PB.decode(d, Bool)
        elseif field_number == 305
            xla_gpu_enable_approx_costly_collectives = PB.decode(d, Bool)
        elseif field_number == 258
            PB.decode!(d, wire_type, xla_gpu_enable_command_buffer)
        elseif field_number == 259
            xla_gpu_enable_cub_radix_sort = PB.decode(d, Bool)
        elseif field_number == 166
            xla_gpu_enable_cublaslt = PB.decode(d, Bool)
        elseif field_number == 189
            xla_gpu_enable_cudnn_int8x32_convolution_reordering = PB.decode(d, Bool)
        elseif field_number == 262
            xla_gpu_enable_cudnn_layer_norm = PB.decode(d, Bool)
        elseif field_number == 105
            xla_gpu_enable_dynamic_slice_fusion = PB.decode(d, Bool)
        elseif field_number == 100
            xla_gpu_enable_fast_min_max = PB.decode(d, Bool)
        elseif field_number == 216
            xla_gpu_enable_highest_priority_async_stream = PB.decode(d, Bool)
        elseif field_number == 296
            xla_gpu_enable_host_memory_offloading = PB.decode(d, Bool)
        elseif field_number == 186
            xla_gpu_enable_latency_hiding_scheduler = PB.decode(d, Bool)
        elseif field_number == 269
            xla_gpu_enable_libnvptxcompiler = PB.decode(d, Bool)
        elseif field_number == 268
            xla_gpu_enable_llvm_module_compilation_parallelism = PB.decode(d, Bool)
        elseif field_number == 244
            xla_gpu_enable_nccl_clique_optimization = PB.decode(d, Bool)
        elseif field_number == 272
            xla_gpu_enable_nccl_comm_splitting = PB.decode(d, Bool)
        elseif field_number == 267
            xla_gpu_enable_nccl_user_buffers = PB.decode(d, Bool)
        elseif field_number == 227
            xla_gpu_enable_pipelined_all_gather = PB.decode(d, Bool)
        elseif field_number == 217
            xla_gpu_enable_pipelined_all_reduce = PB.decode(d, Bool)
        elseif field_number == 239
            xla_gpu_enable_pipelined_collectives = PB.decode(d, Bool)
        elseif field_number == 246
            xla_gpu_enable_pipelined_p2p = PB.decode(d, Bool)
        elseif field_number == 231
            xla_gpu_enable_pipelined_reduce_scatter = PB.decode(d, Bool)
        elseif field_number == 209
            xla_gpu_enable_reassociation_for_converted_ar = PB.decode(d, Bool)
        elseif field_number == 257
            xla_gpu_enable_reduce_scatter_combine_by_dim = PB.decode(d, Bool)
        elseif field_number == 243
            xla_gpu_enable_reduction_epilogue_fusion = PB.decode(d, Bool)
        elseif field_number == 345
            xla_gpu_enable_scatter_determinism_expander = PB.decode(d, Bool)
        elseif field_number == 165
            xla_gpu_enable_shared_constants = PB.decode(d, Bool)
        elseif field_number == 241
            xla_gpu_enable_split_k_autotuning = PB.decode(d, Bool)
        elseif field_number == 188
            xla_gpu_enable_triton_gemm = PB.decode(d, Bool)
        elseif field_number == 248
            xla_gpu_enable_while_loop_double_buffering = PB.decode(d, Bool)
        elseif field_number == 203
            xla_gpu_enable_while_loop_reduce_scatter_code_motion = PB.decode(d, Bool)
        elseif field_number == 294
            xla_gpu_enable_while_loop_unrolling = PB.decode(d, var"DebugOptions.WhileLoopUnrolling".T)
        elseif field_number == 297
            xla_gpu_exclude_nondeterministic_ops = PB.decode(d, Bool)
        elseif field_number == 437
            xla_gpu_executable_embed_debug_info = PB.decode(d, Bool)
        elseif field_number == 328
            xla_gpu_executable_terminate_timeout_seconds = PB.decode(d, Int32)
        elseif field_number == 327
            xla_gpu_executable_warn_stuck_timeout_seconds = PB.decode(d, Int32)
        elseif field_number == 219
            xla_gpu_exhaustive_tiling_search = PB.decode(d, Bool)
        elseif field_number == 430
            xla_gpu_experimental_allow_unroll_factor_eight = PB.decode(d, Bool)
        elseif field_number == 435
            xla_gpu_experimental_aot_compiled_thunks = PB.decode(d, Bool)
        elseif field_number == 324
            xla_gpu_experimental_autotune_cache_mode = PB.decode(d, var"DebugOptions.AutotuneCacheMode".T)
        elseif field_number == 407
            xla_gpu_experimental_autotuner_cache_dir = PB.decode(d, String)
        elseif field_number == 374
            xla_gpu_experimental_collective_cse_distance_threshold = PB.decode(d, Int64)
        elseif field_number == 377
            xla_gpu_experimental_collective_perf_table_path = PB.decode(d, String)
        elseif field_number == 329
            xla_gpu_experimental_disable_binary_libraries = PB.decode(d, Bool)
        elseif field_number == 338
            xla_gpu_experimental_dump_fdo_profiles = PB.decode(d, Bool)
        elseif field_number == 427
            xla_gpu_experimental_dump_gpu_executable = PB.decode(d, Bool)
        elseif field_number == 360
            xla_gpu_experimental_enable_alltoall_windowed_einsum = PB.decode(d, Bool)
        elseif field_number == 431
            xla_gpu_experimental_enable_buffer_saver_on_thunks = PB.decode(d, Bool)
        elseif field_number == 414
            xla_gpu_experimental_enable_checksum_tracing_on_thunks = PB.decode(d, Bool)
        elseif field_number == 409
            xla_gpu_experimental_enable_fusion_autotuner = PB.decode(d, Bool)
        elseif field_number == 334
            xla_gpu_experimental_enable_fusion_block_level_rewriter = PB.decode(d, Bool)
        elseif field_number == 366
            xla_gpu_experimental_enable_heuristic_collective_combining = PB.decode(d, Bool)
        elseif field_number == 406
            xla_gpu_experimental_enable_nccl_symmetric_buffers = PB.decode(d, Bool)
        elseif field_number == 388
            xla_gpu_experimental_enable_nvshmem = PB.decode(d, Bool)
        elseif field_number == 386
            xla_gpu_experimental_enable_split_k_rewrite = PB.decode(d, Bool)
        elseif field_number == 368
            xla_gpu_experimental_enable_subchannel_dequantisation_fusion = PB.decode(d, Bool)
        elseif field_number == 340
            xla_gpu_experimental_enable_triton_heroless_priority_fusion = PB.decode(d, Bool)
        elseif field_number == 421
            xla_gpu_experimental_enable_triton_warp_specialization = PB.decode(d, Bool)
        elseif field_number == 362
            xla_gpu_experimental_pack_dot_operands_along_k_dimension = PB.decode(d, Bool)
        elseif field_number == 336
            xla_gpu_experimental_parallel_collective_overlap_limit = PB.decode(d, Int32)
        elseif field_number == 351
            xla_gpu_experimental_pipeline_parallelism_opt_level = PB.decode(d, var"DebugOptions.PipelineParallelismOptLevel".T)
        elseif field_number == 342
            xla_gpu_experimental_stream_annotation = PB.decode(d, Bool)
        elseif field_number == 396
            xla_gpu_experimental_use_autotuner_pass = PB.decode(d, Bool)
        elseif field_number == 401
            xla_gpu_experimental_use_ragged_dot_fusion = PB.decode(d, Bool)
        elseif field_number == 353
            xla_gpu_fail_ptx_compilation_on_register_spilling = PB.decode(d, Bool)
        elseif field_number == 250
            xla_gpu_filter_kernels_spilling_registers_on_autotuning = PB.decode(d, Bool)
        elseif field_number == 392
            xla_gpu_first_collective_call_terminate_timeout_seconds = PB.decode(d, Int32)
        elseif field_number == 391
            xla_gpu_first_collective_call_warn_stuck_timeout_seconds = PB.decode(d, Int32)
        elseif field_number == 147
            xla_gpu_force_compilation_parallelism = PB.decode(d, Int32)
        elseif field_number == 125
            xla_gpu_force_conv_nchw = PB.decode(d, Bool)
        elseif field_number == 146
            xla_gpu_force_conv_nhwc = PB.decode(d, Bool)
        elseif field_number == 62
            xla_gpu_ftz = PB.decode(d, Bool)
        elseif field_number == 235
            xla_gpu_fused_attention_use_cudnn_rng = PB.decode(d, Bool)
        elseif field_number == 434
            xla_gpu_gemm_autotuner_override_file = PB.decode(d, String)
        elseif field_number == 283
            xla_gpu_gemm_rewrite_size_threshold = PB.decode(d, Int64)
        elseif field_number == 348
            xla_gpu_generate_debug_info = PB.decode(d, Bool)
        elseif field_number == 349
            xla_gpu_generate_line_info = PB.decode(d, Bool)
        elseif field_number == 215
            xla_gpu_graph_enable_concurrent_region = PB.decode(d, Bool)
        elseif field_number == 208
            xla_gpu_graph_min_graph_size = PB.decode(d, Int32)
        elseif field_number == 306
            xla_gpu_kernel_cache_file = PB.decode(d, String)
        elseif field_number == 343
            xla_gpu_libnvjitlink_mode = PB.decode(d, var"DebugOptions.LibNvJitLinkMode".T)
        elseif field_number == 150
            PB.decode!(d, xla_gpu_llvm_ir_file)
        elseif field_number == 256
            xla_gpu_llvm_verification_level = PB.decode(d, Int32)
        elseif field_number == 223
            xla_gpu_load_autotune_results_from = PB.decode(d, String)
        elseif field_number == 260
            xla_gpu_memory_limit_slop_factor = PB.decode(d, Int32)
        elseif field_number == 245
            xla_gpu_mock_custom_calls = PB.decode(d, Bool)
        elseif field_number == 280
            xla_gpu_multi_streamed_windowed_einsum = PB.decode(d, Bool)
        elseif field_number == 393
            xla_gpu_nccl_async_execution = PB.decode(d, Bool)
        elseif field_number == 390
            xla_gpu_nccl_blocking_communicators = PB.decode(d, Bool)
        elseif field_number == 273
            xla_gpu_nccl_collective_max_nchannels = PB.decode(d, Int64)
        elseif field_number == 277
            xla_gpu_nccl_init_max_rank_per_root_ratio = PB.decode(d, Int64)
        elseif field_number == 274
            xla_gpu_nccl_p2p_max_nchannels = PB.decode(d, Int64)
        elseif field_number == 301
            xla_gpu_nccl_terminate_on_error = PB.decode(d, Bool)
        elseif field_number == 163
            xla_gpu_nccl_termination_timeout_seconds = PB.decode(d, Int64)
        elseif field_number == 339
            xla_gpu_operand_bytes_threshold_for_windowed_einsum = PB.decode(d, Int64)
        elseif field_number == 295
            xla_gpu_override_gemm_autotuner = PB.decode(d, String)
        elseif field_number == 310
            xla_gpu_per_fusion_autotune_cache_dir = PB.decode(d, String)
        elseif field_number == 341
            xla_gpu_pgle_accuracy_checker = PB.decode(d, var"DebugOptions.PGLEStrictnessLevel".T)
        elseif field_number == 210
            xla_gpu_pgle_profile_file_or_directory_path = PB.decode(d, String)
        elseif field_number == 127
            PB.decode!(d, xla_gpu_ptx_file)
        elseif field_number == 213
            xla_gpu_reduce_scatter_combine_threshold_bytes = PB.decode(d, Int64)
        elseif field_number == 228
            xla_gpu_redzone_padding_bytes = PB.decode(d, Int64)
        elseif field_number == 284
            xla_gpu_require_complete_aot_autotune_results = PB.decode(d, Bool)
        elseif field_number == 347
            xla_gpu_require_exclusive_lock = PB.decode(d, Bool)
        elseif field_number == 170
            xla_gpu_shape_checks = PB.decode(d, var"DebugOptions.ShapeChecks".T)
        elseif field_number == 304
            xla_gpu_shard_autotuning = PB.decode(d, Bool)
        elseif field_number == 156
            xla_gpu_strict_conv_algorithm_picker = PB.decode(d, Bool)
        elseif field_number == 261
            xla_gpu_target_config_filename = PB.decode(d, String)
        elseif field_number == 312
            xla_gpu_temp_buffer_use_separate_color = PB.decode(d, Bool)
        elseif field_number == 265
            xla_gpu_threshold_for_windowed_einsum_mib = PB.decode(d, Int64)
        elseif field_number == 190
            xla_gpu_triton_gemm_any = PB.decode(d, Bool)
        elseif field_number == 138
            xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found = PB.decode(d, Bool)
        elseif field_number == 384
            xla_gpu_unsupported_enable_all_reduce_decomposer = PB.decode(d, Bool)
        elseif field_number == 350
            xla_gpu_unsupported_enable_ragged_all_to_all_decomposer = PB.decode(d, Bool)
        elseif field_number == 415
            xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer = PB.decode(d, Bool)
        elseif field_number == 322
            xla_gpu_unsupported_enable_triton_gemm = PB.decode(d, Bool)
        elseif field_number == 382
            xla_gpu_unsupported_enable_triton_multi_output_fusion = PB.decode(d, Bool)
        elseif field_number == 416
            xla_gpu_unsupported_override_fast_interconnect_slice_size = PB.decode(d, Int64)
        elseif field_number == 387
            xla_gpu_unsupported_use_all_reduce_one_shot_kernel = PB.decode(d, Bool)
        elseif field_number == 375
            xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel = PB.decode(d, Bool)
        elseif field_number == 420
            xla_gpu_use_embeded_device_lib = PB.decode(d, Bool)
        elseif field_number == 389
            xla_gpu_use_inprocess_lld = PB.decode(d, Bool)
        elseif field_number == 287
            xla_gpu_use_memcpy_local_p2p = PB.decode(d, Bool)
        elseif field_number == 181
            xla_gpu_use_runtime_fusion = PB.decode(d, Bool)
        elseif field_number == 291
            xla_gpu_verify_triton_fusion_numerics = PB.decode(d, Bool)
        elseif field_number == 2
            xla_hlo_graph_addresses = PB.decode(d, Bool)
        elseif field_number == 9
            xla_hlo_profile = PB.decode(d, Bool)
        elseif field_number == 30
            PB.decode!(d, xla_disable_hlo_passes)
        elseif field_number == 124
            PB.decode!(d, xla_enable_hlo_passes_only)
        elseif field_number == 104
            xla_disable_all_hlo_passes = PB.decode(d, Bool)
        elseif field_number == 31
            xla_backend_optimization_level = PB.decode(d, Int32)
        elseif field_number == 33
            xla_embed_ir_in_executable = PB.decode(d, Bool)
        elseif field_number == 35
            xla_eliminate_hlo_implicit_broadcast = PB.decode(d, Bool)
        elseif field_number == 60
            xla_cpu_multi_thread_eigen = PB.decode(d, Bool)
        elseif field_number == 70
            xla_llvm_enable_alias_scope_metadata = PB.decode(d, Bool)
        elseif field_number == 71
            xla_llvm_enable_noalias_metadata = PB.decode(d, Bool)
        elseif field_number == 72
            xla_llvm_enable_invariant_load_metadata = PB.decode(d, Bool)
        elseif field_number == 73
            xla_llvm_disable_expensive_passes = PB.decode(d, Bool)
        elseif field_number == 90
            xla_test_all_output_layouts = PB.decode(d, Bool)
        elseif field_number == 91
            xla_test_all_input_layouts = PB.decode(d, Bool)
        elseif field_number == 92
            xla_hlo_graph_sharding_color = PB.decode(d, Bool)
        elseif field_number == 97
            xla_cpu_use_onednn = PB.decode(d, Bool)
        elseif field_number == 122
            xla_allow_excess_precision = PB.decode(d, Bool)
        elseif field_number == 102
            xla_force_host_platform_device_count = PB.decode(d, Int32)
        elseif field_number == 106
            xla_hlo_evaluator_use_fast_path = PB.decode(d, Bool)
        elseif field_number == 107
            xla_allow_scalar_index_dynamic_ops = PB.decode(d, Bool)
        elseif field_number == 108
            xla_step_marker_location = PB.decode(d, var"DebugOptions.StepMarkerLocation".T)
        elseif field_number == 109
            xla_dump_to = PB.decode(d, String)
        elseif field_number == 364
            xla_flags_reset = PB.decode(d, Bool)
        elseif field_number == 110
            xla_dump_hlo_module_re = PB.decode(d, String)
        elseif field_number == 111
            xla_dump_hlo_pass_re = PB.decode(d, String)
        elseif field_number == 433
            xla_dump_emitter_re = PB.decode(d, String)
        elseif field_number == 112
            xla_dump_hlo_as_text = PB.decode(d, Bool)
        elseif field_number == 113
            xla_dump_hlo_as_proto = PB.decode(d, Bool)
        elseif field_number == 114
            xla_dump_hlo_as_dot = PB.decode(d, Bool)
        elseif field_number == 115
            xla_dump_hlo_as_url = PB.decode(d, Bool)
        elseif field_number == 116
            xla_dump_hlo_as_html = PB.decode(d, Bool)
        elseif field_number == 149
            xla_dump_fusion_visualization = PB.decode(d, Bool)
        elseif field_number == 118
            xla_dump_hlo_snapshots = PB.decode(d, Bool)
        elseif field_number == 131
            xla_dump_include_timestamp = PB.decode(d, Bool)
        elseif field_number == 132
            xla_dump_max_hlo_modules = PB.decode(d, Int32)
        elseif field_number == 144
            xla_dump_module_metadata = PB.decode(d, Bool)
        elseif field_number == 151
            xla_dump_compress_protos = PB.decode(d, Bool)
        elseif field_number == 164
            xla_dump_hlo_as_long_text = PB.decode(d, Bool)
        elseif field_number == 185
            xla_dump_enable_mlir_pretty_form = PB.decode(d, Bool)
        elseif field_number == 381
            xla_dump_full_hlo_config = PB.decode(d, Bool)
        elseif field_number == 135
            xla_tpu_detect_nan = PB.decode(d, Bool)
        elseif field_number == 136
            xla_tpu_detect_inf = PB.decode(d, Bool)
        elseif field_number == 137
            xla_cpu_enable_xprof_traceme = PB.decode(d, Bool)
        elseif field_number == 142
            xla_multiheap_size_constraint_per_heap = PB.decode(d, Int32)
        elseif field_number == 252
            xla_detailed_logging = PB.decode(d, Bool)
        elseif field_number == 253
            xla_enable_dumping = PB.decode(d, Bool)
        elseif field_number == 300
            xla_llvm_force_inline_before_split = PB.decode(d, Bool)
        elseif field_number == 153
            xla_dump_disable_metadata = PB.decode(d, Bool)
        elseif field_number == 154
            xla_dump_hlo_pipeline_re = PB.decode(d, String)
        elseif field_number == 174
            xla_cpu_use_acl = PB.decode(d, Bool)
        elseif field_number == 175
            xla_cpu_strict_dot_conv_math = PB.decode(d, Bool)
        elseif field_number == 182
            xla_dump_latency_hiding_schedule = PB.decode(d, Bool)
        elseif field_number == 187
            xla_partitioning_algorithm = PB.decode(d, var"DebugOptions.PartitioningAlgorithm".T)
        elseif field_number == 251
            xla_debug_buffer_assignment_show_max = PB.decode(d, Int64)
        elseif field_number == 403
            xla_detect_unstable_reductions = PB.decode(d, var"DebugOptions.DetectionMode".T)
        elseif field_number == 432
            xla_detect_unstable_reductions_post_optimizations = PB.decode(d, var"DebugOptions.DetectionMode".T)
        elseif field_number == 426
            xla_gpu_detect_nan = PB.decode(d, var"DebugOptions.DetectionMode".T)
        elseif field_number == 428
            xla_gpu_detect_inf = PB.decode(d, var"DebugOptions.DetectionMode".T)
        elseif field_number == 290
            xla_dump_large_constants = PB.decode(d, Bool)
        elseif field_number == 293
            xla_reduce_window_rewrite_base_length = PB.decode(d, Int64)
        elseif field_number == 311
            xla_cmd_buffer_trace_cache_size = PB.decode(d, Int64)
        elseif field_number == 315
            xla_syntax_sugar_async_ops = PB.decode(d, Bool)
        elseif field_number == 317
            xla_enable_command_buffers_during_profiling = PB.decode(d, Bool)
        elseif field_number == 330
            xla_ignore_channel_id = PB.decode(d, Bool)
        elseif field_number == 344
            xla_pjrt_allow_auto_layout_in_hlo = PB.decode(d, Bool)
        elseif field_number == 373
            xla_test_add_command_buffer_mode = PB.decode(d, Bool)
        elseif field_number == 383
            xla_gpu_experimental_matmul_perf_table_path = PB.decode(d, String)
        elseif field_number == 397
            xla_early_exit_with_layouts = PB.decode(d, Bool)
        elseif field_number == 410
            xla_gpu_experimental_scaled_dot_with_triton = PB.decode(d, Bool)
        elseif field_number == 413
            xla_gpu_experimental_use_raft_select_k = PB.decode(d, Bool)
        elseif field_number == 500
            PB.decode!(d, xla_backend_extra_options)
        else
            Base.skip(d, wire_type)
        end
    end
    return DebugOptions(xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled, xla_disable_automatic_host_compute_offload, xla_enable_scoped_logging_timers, xla_hlo_pass_fix_detect_cycles, xla_keep_shardings_after_spmd, xla_unsupported_crash_on_hlo_pass_fix_max_iterations, xla_unsupported_crash_on_hlo_pass_noop_change, xla_unsupported_crash_on_hlo_pass_silent_hlo_change, xla_cpu_collective_call_terminate_timeout_seconds, xla_cpu_collective_call_warn_stuck_seconds, xla_cpu_collective_timeout_seconds, xla_cpu_copy_insertion_use_region_analysis, xla_cpu_emitter_verification_level, xla_cpu_enable_concurrency_optimized_scheduler, xla_cpu_enable_fast_math, xla_cpu_enable_fast_min_max, xla_cpu_enable_platform_dependent_math, xla_cpu_experimental_onednn_custom_call, xla_cpu_experimental_onednn_fusion_type[], xla_cpu_experimental_xnn_fusion_type[], xla_cpu_experimental_xnn_graph_fusion_mode, xla_cpu_experimental_ynn_fusion_type[], xla_cpu_fast_math_honor_division, xla_cpu_fast_math_honor_functions, xla_cpu_fast_math_honor_infs, xla_cpu_fast_math_honor_nans, xla_cpu_generate_unique_c_style_kernel_entry_points, xla_cpu_max_isa, xla_cpu_parallel_codegen_split_count, xla_cpu_prefer_vector_width, xla_cpu_use_fusion_emitters, xla_cpu_use_xnnpack, xla_enable_fast_math, xla_gpu_experimental_thunk_buffer_debug_filter[], xla_dump_hlo_unoptimized_snapshots, xla_enable_enzyme_comms_opt, xla_gpu_algorithm_denylist_path, xla_gpu_all_gather_combine_threshold_bytes, xla_gpu_all_reduce_blueconnect_num_devices_per_host, xla_gpu_all_reduce_combine_threshold_bytes, xla_gpu_analytical_latency_estimator_options, xla_gpu_async_dot, xla_gpu_auto_spmd_partitioning_memory_budget_gb, xla_gpu_auto_spmd_partitioning_memory_budget_ratio, xla_gpu_autotune_gemm_rtol, xla_gpu_autotune_level, xla_gpu_autotune_max_solutions, xla_gpu_collect_cost_model_stats, xla_gpu_collective_inflation_factor, xla_gpu_collective_permute_combine_threshold_bytes, xla_gpu_collective_permute_decomposer_threshold, xla_gpu_collectives_use_persistent_cliques, xla_gpu_command_buffer_scheduling_mode, xla_gpu_command_buffer_unroll_loops, xla_gpu_copy_insertion_use_region_analysis, xla_gpu_crash_on_verification_failures, xla_gpu_cublas_fallback, xla_gpu_cuda_data_dir, xla_gpu_cudnn_gemm_fusion_level, xla_gpu_cudnn_gemm_max_plans, xla_gpu_deterministic_ops, xla_gpu_disable_async_collectives[], xla_gpu_disable_gpuasm_optimizations, xla_gpu_dot_merger_threshold_mb, xla_gpu_dump_autotune_logs_to, xla_gpu_dump_autotune_results_to, xla_gpu_dump_autotuned_gemm_fusions, xla_gpu_dump_llvmir, xla_gpu_enable_all_gather_combine_by_dim, xla_gpu_enable_analytical_latency_estimator, xla_gpu_enable_analytical_sol_latency_estimator, xla_gpu_enable_approx_costly_collectives, xla_gpu_enable_command_buffer[], xla_gpu_enable_cub_radix_sort, xla_gpu_enable_cublaslt, xla_gpu_enable_cudnn_int8x32_convolution_reordering, xla_gpu_enable_cudnn_layer_norm, xla_gpu_enable_dynamic_slice_fusion, xla_gpu_enable_fast_min_max, xla_gpu_enable_highest_priority_async_stream, xla_gpu_enable_host_memory_offloading, xla_gpu_enable_latency_hiding_scheduler, xla_gpu_enable_libnvptxcompiler, xla_gpu_enable_llvm_module_compilation_parallelism, xla_gpu_enable_nccl_clique_optimization, xla_gpu_enable_nccl_comm_splitting, xla_gpu_enable_nccl_user_buffers, xla_gpu_enable_pipelined_all_gather, xla_gpu_enable_pipelined_all_reduce, xla_gpu_enable_pipelined_collectives, xla_gpu_enable_pipelined_p2p, xla_gpu_enable_pipelined_reduce_scatter, xla_gpu_enable_reassociation_for_converted_ar, xla_gpu_enable_reduce_scatter_combine_by_dim, xla_gpu_enable_reduction_epilogue_fusion, xla_gpu_enable_scatter_determinism_expander, xla_gpu_enable_shared_constants, xla_gpu_enable_split_k_autotuning, xla_gpu_enable_triton_gemm, xla_gpu_enable_while_loop_double_buffering, xla_gpu_enable_while_loop_reduce_scatter_code_motion, xla_gpu_enable_while_loop_unrolling, xla_gpu_exclude_nondeterministic_ops, xla_gpu_executable_embed_debug_info, xla_gpu_executable_terminate_timeout_seconds, xla_gpu_executable_warn_stuck_timeout_seconds, xla_gpu_exhaustive_tiling_search, xla_gpu_experimental_allow_unroll_factor_eight, xla_gpu_experimental_aot_compiled_thunks, xla_gpu_experimental_autotune_cache_mode, xla_gpu_experimental_autotuner_cache_dir, xla_gpu_experimental_collective_cse_distance_threshold, xla_gpu_experimental_collective_perf_table_path, xla_gpu_experimental_disable_binary_libraries, xla_gpu_experimental_dump_fdo_profiles, xla_gpu_experimental_dump_gpu_executable, xla_gpu_experimental_enable_alltoall_windowed_einsum, xla_gpu_experimental_enable_buffer_saver_on_thunks, xla_gpu_experimental_enable_checksum_tracing_on_thunks, xla_gpu_experimental_enable_fusion_autotuner, xla_gpu_experimental_enable_fusion_block_level_rewriter, xla_gpu_experimental_enable_heuristic_collective_combining, xla_gpu_experimental_enable_nccl_symmetric_buffers, xla_gpu_experimental_enable_nvshmem, xla_gpu_experimental_enable_split_k_rewrite, xla_gpu_experimental_enable_subchannel_dequantisation_fusion, xla_gpu_experimental_enable_triton_heroless_priority_fusion, xla_gpu_experimental_enable_triton_warp_specialization, xla_gpu_experimental_pack_dot_operands_along_k_dimension, xla_gpu_experimental_parallel_collective_overlap_limit, xla_gpu_experimental_pipeline_parallelism_opt_level, xla_gpu_experimental_stream_annotation, xla_gpu_experimental_use_autotuner_pass, xla_gpu_experimental_use_ragged_dot_fusion, xla_gpu_fail_ptx_compilation_on_register_spilling, xla_gpu_filter_kernels_spilling_registers_on_autotuning, xla_gpu_first_collective_call_terminate_timeout_seconds, xla_gpu_first_collective_call_warn_stuck_timeout_seconds, xla_gpu_force_compilation_parallelism, xla_gpu_force_conv_nchw, xla_gpu_force_conv_nhwc, xla_gpu_ftz, xla_gpu_fused_attention_use_cudnn_rng, xla_gpu_gemm_autotuner_override_file, xla_gpu_gemm_rewrite_size_threshold, xla_gpu_generate_debug_info, xla_gpu_generate_line_info, xla_gpu_graph_enable_concurrent_region, xla_gpu_graph_min_graph_size, xla_gpu_kernel_cache_file, xla_gpu_libnvjitlink_mode, xla_gpu_llvm_ir_file[], xla_gpu_llvm_verification_level, xla_gpu_load_autotune_results_from, xla_gpu_memory_limit_slop_factor, xla_gpu_mock_custom_calls, xla_gpu_multi_streamed_windowed_einsum, xla_gpu_nccl_async_execution, xla_gpu_nccl_blocking_communicators, xla_gpu_nccl_collective_max_nchannels, xla_gpu_nccl_init_max_rank_per_root_ratio, xla_gpu_nccl_p2p_max_nchannels, xla_gpu_nccl_terminate_on_error, xla_gpu_nccl_termination_timeout_seconds, xla_gpu_operand_bytes_threshold_for_windowed_einsum, xla_gpu_override_gemm_autotuner, xla_gpu_per_fusion_autotune_cache_dir, xla_gpu_pgle_accuracy_checker, xla_gpu_pgle_profile_file_or_directory_path, xla_gpu_ptx_file[], xla_gpu_reduce_scatter_combine_threshold_bytes, xla_gpu_redzone_padding_bytes, xla_gpu_require_complete_aot_autotune_results, xla_gpu_require_exclusive_lock, xla_gpu_shape_checks, xla_gpu_shard_autotuning, xla_gpu_strict_conv_algorithm_picker, xla_gpu_target_config_filename, xla_gpu_temp_buffer_use_separate_color, xla_gpu_threshold_for_windowed_einsum_mib, xla_gpu_triton_gemm_any, xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found, xla_gpu_unsupported_enable_all_reduce_decomposer, xla_gpu_unsupported_enable_ragged_all_to_all_decomposer, xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer, xla_gpu_unsupported_enable_triton_gemm, xla_gpu_unsupported_enable_triton_multi_output_fusion, xla_gpu_unsupported_override_fast_interconnect_slice_size, xla_gpu_unsupported_use_all_reduce_one_shot_kernel, xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel, xla_gpu_use_embeded_device_lib, xla_gpu_use_inprocess_lld, xla_gpu_use_memcpy_local_p2p, xla_gpu_use_runtime_fusion, xla_gpu_verify_triton_fusion_numerics, xla_hlo_graph_addresses, xla_hlo_profile, xla_disable_hlo_passes[], xla_enable_hlo_passes_only[], xla_disable_all_hlo_passes, xla_backend_optimization_level, xla_embed_ir_in_executable, xla_eliminate_hlo_implicit_broadcast, xla_cpu_multi_thread_eigen, xla_llvm_enable_alias_scope_metadata, xla_llvm_enable_noalias_metadata, xla_llvm_enable_invariant_load_metadata, xla_llvm_disable_expensive_passes, xla_test_all_output_layouts, xla_test_all_input_layouts, xla_hlo_graph_sharding_color, xla_cpu_use_onednn, xla_allow_excess_precision, xla_force_host_platform_device_count, xla_hlo_evaluator_use_fast_path, xla_allow_scalar_index_dynamic_ops, xla_step_marker_location, xla_dump_to, xla_flags_reset, xla_dump_hlo_module_re, xla_dump_hlo_pass_re, xla_dump_emitter_re, xla_dump_hlo_as_text, xla_dump_hlo_as_proto, xla_dump_hlo_as_dot, xla_dump_hlo_as_url, xla_dump_hlo_as_html, xla_dump_fusion_visualization, xla_dump_hlo_snapshots, xla_dump_include_timestamp, xla_dump_max_hlo_modules, xla_dump_module_metadata, xla_dump_compress_protos, xla_dump_hlo_as_long_text, xla_dump_enable_mlir_pretty_form, xla_dump_full_hlo_config, xla_tpu_detect_nan, xla_tpu_detect_inf, xla_cpu_enable_xprof_traceme, xla_multiheap_size_constraint_per_heap, xla_detailed_logging, xla_enable_dumping, xla_llvm_force_inline_before_split, xla_dump_disable_metadata, xla_dump_hlo_pipeline_re, xla_cpu_use_acl, xla_cpu_strict_dot_conv_math, xla_dump_latency_hiding_schedule, xla_partitioning_algorithm, xla_debug_buffer_assignment_show_max, xla_detect_unstable_reductions, xla_detect_unstable_reductions_post_optimizations, xla_gpu_detect_nan, xla_gpu_detect_inf, xla_dump_large_constants, xla_reduce_window_rewrite_base_length, xla_cmd_buffer_trace_cache_size, xla_syntax_sugar_async_ops, xla_enable_command_buffers_during_profiling, xla_ignore_channel_id, xla_pjrt_allow_auto_layout_in_hlo, xla_test_add_command_buffer_mode, xla_gpu_experimental_matmul_perf_table_path, xla_early_exit_with_layouts, xla_gpu_experimental_scaled_dot_with_triton, xla_gpu_experimental_use_raft_select_k, xla_backend_extra_options)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::DebugOptions)
    initpos = position(e.io)
    x.xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled != false && PB.encode(e, 439, x.xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled)
    x.xla_disable_automatic_host_compute_offload != false && PB.encode(e, 408, x.xla_disable_automatic_host_compute_offload)
    x.xla_enable_scoped_logging_timers != false && PB.encode(e, 436, x.xla_enable_scoped_logging_timers)
    x.xla_hlo_pass_fix_detect_cycles != false && PB.encode(e, 370, x.xla_hlo_pass_fix_detect_cycles)
    x.xla_keep_shardings_after_spmd != false && PB.encode(e, 419, x.xla_keep_shardings_after_spmd)
    x.xla_unsupported_crash_on_hlo_pass_fix_max_iterations != false && PB.encode(e, 363, x.xla_unsupported_crash_on_hlo_pass_fix_max_iterations)
    x.xla_unsupported_crash_on_hlo_pass_noop_change != false && PB.encode(e, 379, x.xla_unsupported_crash_on_hlo_pass_noop_change)
    x.xla_unsupported_crash_on_hlo_pass_silent_hlo_change != false && PB.encode(e, 380, x.xla_unsupported_crash_on_hlo_pass_silent_hlo_change)
    x.xla_cpu_collective_call_terminate_timeout_seconds != zero(Int32) && PB.encode(e, 417, x.xla_cpu_collective_call_terminate_timeout_seconds)
    x.xla_cpu_collective_call_warn_stuck_seconds != zero(Int32) && PB.encode(e, 418, x.xla_cpu_collective_call_warn_stuck_seconds)
    x.xla_cpu_collective_timeout_seconds != zero(Int32) && PB.encode(e, 438, x.xla_cpu_collective_timeout_seconds)
    x.xla_cpu_copy_insertion_use_region_analysis != false && PB.encode(e, 337, x.xla_cpu_copy_insertion_use_region_analysis)
    x.xla_cpu_emitter_verification_level != zero(Int32) && PB.encode(e, 395, x.xla_cpu_emitter_verification_level)
    x.xla_cpu_enable_concurrency_optimized_scheduler != false && PB.encode(e, 307, x.xla_cpu_enable_concurrency_optimized_scheduler)
    x.xla_cpu_enable_fast_math != false && PB.encode(e, 99, x.xla_cpu_enable_fast_math)
    x.xla_cpu_enable_fast_min_max != false && PB.encode(e, 140, x.xla_cpu_enable_fast_min_max)
    x.xla_cpu_enable_platform_dependent_math != false && PB.encode(e, 425, x.xla_cpu_enable_platform_dependent_math)
    x.xla_cpu_experimental_onednn_custom_call != false && PB.encode(e, 412, x.xla_cpu_experimental_onednn_custom_call)
    !isempty(x.xla_cpu_experimental_onednn_fusion_type) && PB.encode(e, 399, x.xla_cpu_experimental_onednn_fusion_type)
    !isempty(x.xla_cpu_experimental_xnn_fusion_type) && PB.encode(e, 400, x.xla_cpu_experimental_xnn_fusion_type)
    x.xla_cpu_experimental_xnn_graph_fusion_mode != var"DebugOptions.XnnGraphFusionMode".XNN_GRAPH_FUSION_MODE_DISABLED && PB.encode(e, 365, x.xla_cpu_experimental_xnn_graph_fusion_mode)
    !isempty(x.xla_cpu_experimental_ynn_fusion_type) && PB.encode(e, 422, x.xla_cpu_experimental_ynn_fusion_type)
    x.xla_cpu_fast_math_honor_division != false && PB.encode(e, 126, x.xla_cpu_fast_math_honor_division)
    x.xla_cpu_fast_math_honor_functions != false && PB.encode(e, 129, x.xla_cpu_fast_math_honor_functions)
    x.xla_cpu_fast_math_honor_infs != false && PB.encode(e, 121, x.xla_cpu_fast_math_honor_infs)
    x.xla_cpu_fast_math_honor_nans != false && PB.encode(e, 120, x.xla_cpu_fast_math_honor_nans)
    x.xla_cpu_generate_unique_c_style_kernel_entry_points != false && PB.encode(e, 372, x.xla_cpu_generate_unique_c_style_kernel_entry_points)
    !isempty(x.xla_cpu_max_isa) && PB.encode(e, 333, x.xla_cpu_max_isa)
    x.xla_cpu_parallel_codegen_split_count != zero(Int32) && PB.encode(e, 323, x.xla_cpu_parallel_codegen_split_count)
    x.xla_cpu_prefer_vector_width != zero(Int32) && PB.encode(e, 308, x.xla_cpu_prefer_vector_width)
    x.xla_cpu_use_fusion_emitters != false && PB.encode(e, 376, x.xla_cpu_use_fusion_emitters)
    x.xla_cpu_use_xnnpack != false && PB.encode(e, 359, x.xla_cpu_use_xnnpack)
    x.xla_enable_fast_math != false && PB.encode(e, 335, x.xla_enable_fast_math)
    !isnothing(x.xla_gpu_experimental_thunk_buffer_debug_filter) && PB.encode(e, 424, x.xla_gpu_experimental_thunk_buffer_debug_filter)
    x.xla_dump_hlo_unoptimized_snapshots != false && PB.encode(e, 405, x.xla_dump_hlo_unoptimized_snapshots)
    x.xla_enable_enzyme_comms_opt != false && PB.encode(e, 429, x.xla_enable_enzyme_comms_opt)
    !isempty(x.xla_gpu_algorithm_denylist_path) && PB.encode(e, 128, x.xla_gpu_algorithm_denylist_path)
    x.xla_gpu_all_gather_combine_threshold_bytes != zero(Int64) && PB.encode(e, 212, x.xla_gpu_all_gather_combine_threshold_bytes)
    x.xla_gpu_all_reduce_blueconnect_num_devices_per_host != zero(Int32) && PB.encode(e, 159, x.xla_gpu_all_reduce_blueconnect_num_devices_per_host)
    x.xla_gpu_all_reduce_combine_threshold_bytes != zero(Int64) && PB.encode(e, 157, x.xla_gpu_all_reduce_combine_threshold_bytes)
    !isempty(x.xla_gpu_analytical_latency_estimator_options) && PB.encode(e, 357, x.xla_gpu_analytical_latency_estimator_options)
    x.xla_gpu_async_dot != false && PB.encode(e, 321, x.xla_gpu_async_dot)
    x.xla_gpu_auto_spmd_partitioning_memory_budget_gb != zero(Int32) && PB.encode(e, 224, x.xla_gpu_auto_spmd_partitioning_memory_budget_gb)
    x.xla_gpu_auto_spmd_partitioning_memory_budget_ratio !== zero(Float32) && PB.encode(e, 225, x.xla_gpu_auto_spmd_partitioning_memory_budget_ratio)
    x.xla_gpu_autotune_gemm_rtol !== zero(Float32) && PB.encode(e, 316, x.xla_gpu_autotune_gemm_rtol)
    x.xla_gpu_autotune_level != zero(Int32) && PB.encode(e, 123, x.xla_gpu_autotune_level)
    x.xla_gpu_autotune_max_solutions != zero(Int64) && PB.encode(e, 288, x.xla_gpu_autotune_max_solutions)
    x.xla_gpu_collect_cost_model_stats != false && PB.encode(e, 240, x.xla_gpu_collect_cost_model_stats)
    x.xla_gpu_collective_inflation_factor != zero(Int32) && PB.encode(e, 205, x.xla_gpu_collective_inflation_factor)
    x.xla_gpu_collective_permute_combine_threshold_bytes != zero(Int64) && PB.encode(e, 378, x.xla_gpu_collective_permute_combine_threshold_bytes)
    x.xla_gpu_collective_permute_decomposer_threshold != zero(Int64) && PB.encode(e, 237, x.xla_gpu_collective_permute_decomposer_threshold)
    x.xla_gpu_collectives_use_persistent_cliques != false && PB.encode(e, 354, x.xla_gpu_collectives_use_persistent_cliques)
    x.xla_gpu_command_buffer_scheduling_mode != var"DebugOptions.CommandBufferSchedulingMode".SERIALIZE && PB.encode(e, 404, x.xla_gpu_command_buffer_scheduling_mode)
    x.xla_gpu_command_buffer_unroll_loops != false && PB.encode(e, 411, x.xla_gpu_command_buffer_unroll_loops)
    x.xla_gpu_copy_insertion_use_region_analysis != false && PB.encode(e, 236, x.xla_gpu_copy_insertion_use_region_analysis)
    x.xla_gpu_crash_on_verification_failures != false && PB.encode(e, 101, x.xla_gpu_crash_on_verification_failures)
    x.xla_gpu_cublas_fallback != false && PB.encode(e, 247, x.xla_gpu_cublas_fallback)
    !isempty(x.xla_gpu_cuda_data_dir) && PB.encode(e, 61, x.xla_gpu_cuda_data_dir)
    x.xla_gpu_cudnn_gemm_fusion_level != zero(Int32) && PB.encode(e, 285, x.xla_gpu_cudnn_gemm_fusion_level)
    x.xla_gpu_cudnn_gemm_max_plans != zero(Int32) && PB.encode(e, 318, x.xla_gpu_cudnn_gemm_max_plans)
    x.xla_gpu_deterministic_ops != false && PB.encode(e, 148, x.xla_gpu_deterministic_ops)
    !isempty(x.xla_gpu_disable_async_collectives) && PB.encode(e, 289, x.xla_gpu_disable_async_collectives)
    x.xla_gpu_disable_gpuasm_optimizations != false && PB.encode(e, 103, x.xla_gpu_disable_gpuasm_optimizations)
    x.xla_gpu_dot_merger_threshold_mb != zero(Int32) && PB.encode(e, 331, x.xla_gpu_dot_merger_threshold_mb)
    !isempty(x.xla_gpu_dump_autotune_logs_to) && PB.encode(e, 292, x.xla_gpu_dump_autotune_logs_to)
    !isempty(x.xla_gpu_dump_autotune_results_to) && PB.encode(e, 222, x.xla_gpu_dump_autotune_results_to)
    x.xla_gpu_dump_autotuned_gemm_fusions != false && PB.encode(e, 232, x.xla_gpu_dump_autotuned_gemm_fusions)
    x.xla_gpu_dump_llvmir != false && PB.encode(e, 155, x.xla_gpu_dump_llvmir)
    x.xla_gpu_enable_all_gather_combine_by_dim != false && PB.encode(e, 254, x.xla_gpu_enable_all_gather_combine_by_dim)
    x.xla_gpu_enable_analytical_latency_estimator != false && PB.encode(e, 255, x.xla_gpu_enable_analytical_latency_estimator)
    x.xla_gpu_enable_analytical_sol_latency_estimator != false && PB.encode(e, 356, x.xla_gpu_enable_analytical_sol_latency_estimator)
    x.xla_gpu_enable_approx_costly_collectives != false && PB.encode(e, 305, x.xla_gpu_enable_approx_costly_collectives)
    !isempty(x.xla_gpu_enable_command_buffer) && PB.encode(e, 258, x.xla_gpu_enable_command_buffer)
    x.xla_gpu_enable_cub_radix_sort != false && PB.encode(e, 259, x.xla_gpu_enable_cub_radix_sort)
    x.xla_gpu_enable_cublaslt != false && PB.encode(e, 166, x.xla_gpu_enable_cublaslt)
    x.xla_gpu_enable_cudnn_int8x32_convolution_reordering != false && PB.encode(e, 189, x.xla_gpu_enable_cudnn_int8x32_convolution_reordering)
    x.xla_gpu_enable_cudnn_layer_norm != false && PB.encode(e, 262, x.xla_gpu_enable_cudnn_layer_norm)
    x.xla_gpu_enable_dynamic_slice_fusion != false && PB.encode(e, 105, x.xla_gpu_enable_dynamic_slice_fusion)
    x.xla_gpu_enable_fast_min_max != false && PB.encode(e, 100, x.xla_gpu_enable_fast_min_max)
    x.xla_gpu_enable_highest_priority_async_stream != false && PB.encode(e, 216, x.xla_gpu_enable_highest_priority_async_stream)
    x.xla_gpu_enable_host_memory_offloading != false && PB.encode(e, 296, x.xla_gpu_enable_host_memory_offloading)
    x.xla_gpu_enable_latency_hiding_scheduler != false && PB.encode(e, 186, x.xla_gpu_enable_latency_hiding_scheduler)
    x.xla_gpu_enable_libnvptxcompiler != false && PB.encode(e, 269, x.xla_gpu_enable_libnvptxcompiler)
    x.xla_gpu_enable_llvm_module_compilation_parallelism != false && PB.encode(e, 268, x.xla_gpu_enable_llvm_module_compilation_parallelism)
    x.xla_gpu_enable_nccl_clique_optimization != false && PB.encode(e, 244, x.xla_gpu_enable_nccl_clique_optimization)
    x.xla_gpu_enable_nccl_comm_splitting != false && PB.encode(e, 272, x.xla_gpu_enable_nccl_comm_splitting)
    x.xla_gpu_enable_nccl_user_buffers != false && PB.encode(e, 267, x.xla_gpu_enable_nccl_user_buffers)
    x.xla_gpu_enable_pipelined_all_gather != false && PB.encode(e, 227, x.xla_gpu_enable_pipelined_all_gather)
    x.xla_gpu_enable_pipelined_all_reduce != false && PB.encode(e, 217, x.xla_gpu_enable_pipelined_all_reduce)
    x.xla_gpu_enable_pipelined_collectives != false && PB.encode(e, 239, x.xla_gpu_enable_pipelined_collectives)
    x.xla_gpu_enable_pipelined_p2p != false && PB.encode(e, 246, x.xla_gpu_enable_pipelined_p2p)
    x.xla_gpu_enable_pipelined_reduce_scatter != false && PB.encode(e, 231, x.xla_gpu_enable_pipelined_reduce_scatter)
    x.xla_gpu_enable_reassociation_for_converted_ar != false && PB.encode(e, 209, x.xla_gpu_enable_reassociation_for_converted_ar)
    x.xla_gpu_enable_reduce_scatter_combine_by_dim != false && PB.encode(e, 257, x.xla_gpu_enable_reduce_scatter_combine_by_dim)
    x.xla_gpu_enable_reduction_epilogue_fusion != false && PB.encode(e, 243, x.xla_gpu_enable_reduction_epilogue_fusion)
    x.xla_gpu_enable_scatter_determinism_expander != false && PB.encode(e, 345, x.xla_gpu_enable_scatter_determinism_expander)
    x.xla_gpu_enable_shared_constants != false && PB.encode(e, 165, x.xla_gpu_enable_shared_constants)
    x.xla_gpu_enable_split_k_autotuning != false && PB.encode(e, 241, x.xla_gpu_enable_split_k_autotuning)
    x.xla_gpu_enable_triton_gemm != false && PB.encode(e, 188, x.xla_gpu_enable_triton_gemm)
    x.xla_gpu_enable_while_loop_double_buffering != false && PB.encode(e, 248, x.xla_gpu_enable_while_loop_double_buffering)
    x.xla_gpu_enable_while_loop_reduce_scatter_code_motion != false && PB.encode(e, 203, x.xla_gpu_enable_while_loop_reduce_scatter_code_motion)
    x.xla_gpu_enable_while_loop_unrolling != var"DebugOptions.WhileLoopUnrolling".WHILE_LOOP_UNROLLING_NO_UNROLL && PB.encode(e, 294, x.xla_gpu_enable_while_loop_unrolling)
    x.xla_gpu_exclude_nondeterministic_ops != false && PB.encode(e, 297, x.xla_gpu_exclude_nondeterministic_ops)
    x.xla_gpu_executable_embed_debug_info != false && PB.encode(e, 437, x.xla_gpu_executable_embed_debug_info)
    x.xla_gpu_executable_terminate_timeout_seconds != zero(Int32) && PB.encode(e, 328, x.xla_gpu_executable_terminate_timeout_seconds)
    x.xla_gpu_executable_warn_stuck_timeout_seconds != zero(Int32) && PB.encode(e, 327, x.xla_gpu_executable_warn_stuck_timeout_seconds)
    x.xla_gpu_exhaustive_tiling_search != false && PB.encode(e, 219, x.xla_gpu_exhaustive_tiling_search)
    x.xla_gpu_experimental_allow_unroll_factor_eight != false && PB.encode(e, 430, x.xla_gpu_experimental_allow_unroll_factor_eight)
    x.xla_gpu_experimental_aot_compiled_thunks != false && PB.encode(e, 435, x.xla_gpu_experimental_aot_compiled_thunks)
    x.xla_gpu_experimental_autotune_cache_mode != var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_UNSPECIFIED && PB.encode(e, 324, x.xla_gpu_experimental_autotune_cache_mode)
    !isempty(x.xla_gpu_experimental_autotuner_cache_dir) && PB.encode(e, 407, x.xla_gpu_experimental_autotuner_cache_dir)
    x.xla_gpu_experimental_collective_cse_distance_threshold != zero(Int64) && PB.encode(e, 374, x.xla_gpu_experimental_collective_cse_distance_threshold)
    !isempty(x.xla_gpu_experimental_collective_perf_table_path) && PB.encode(e, 377, x.xla_gpu_experimental_collective_perf_table_path)
    x.xla_gpu_experimental_disable_binary_libraries != false && PB.encode(e, 329, x.xla_gpu_experimental_disable_binary_libraries)
    x.xla_gpu_experimental_dump_fdo_profiles != false && PB.encode(e, 338, x.xla_gpu_experimental_dump_fdo_profiles)
    x.xla_gpu_experimental_dump_gpu_executable != false && PB.encode(e, 427, x.xla_gpu_experimental_dump_gpu_executable)
    x.xla_gpu_experimental_enable_alltoall_windowed_einsum != false && PB.encode(e, 360, x.xla_gpu_experimental_enable_alltoall_windowed_einsum)
    x.xla_gpu_experimental_enable_buffer_saver_on_thunks != false && PB.encode(e, 431, x.xla_gpu_experimental_enable_buffer_saver_on_thunks)
    x.xla_gpu_experimental_enable_checksum_tracing_on_thunks != false && PB.encode(e, 414, x.xla_gpu_experimental_enable_checksum_tracing_on_thunks)
    x.xla_gpu_experimental_enable_fusion_autotuner != false && PB.encode(e, 409, x.xla_gpu_experimental_enable_fusion_autotuner)
    x.xla_gpu_experimental_enable_fusion_block_level_rewriter != false && PB.encode(e, 334, x.xla_gpu_experimental_enable_fusion_block_level_rewriter)
    x.xla_gpu_experimental_enable_heuristic_collective_combining != false && PB.encode(e, 366, x.xla_gpu_experimental_enable_heuristic_collective_combining)
    x.xla_gpu_experimental_enable_nccl_symmetric_buffers != false && PB.encode(e, 406, x.xla_gpu_experimental_enable_nccl_symmetric_buffers)
    x.xla_gpu_experimental_enable_nvshmem != false && PB.encode(e, 388, x.xla_gpu_experimental_enable_nvshmem)
    x.xla_gpu_experimental_enable_split_k_rewrite != false && PB.encode(e, 386, x.xla_gpu_experimental_enable_split_k_rewrite)
    x.xla_gpu_experimental_enable_subchannel_dequantisation_fusion != false && PB.encode(e, 368, x.xla_gpu_experimental_enable_subchannel_dequantisation_fusion)
    x.xla_gpu_experimental_enable_triton_heroless_priority_fusion != false && PB.encode(e, 340, x.xla_gpu_experimental_enable_triton_heroless_priority_fusion)
    x.xla_gpu_experimental_enable_triton_warp_specialization != false && PB.encode(e, 421, x.xla_gpu_experimental_enable_triton_warp_specialization)
    x.xla_gpu_experimental_pack_dot_operands_along_k_dimension != false && PB.encode(e, 362, x.xla_gpu_experimental_pack_dot_operands_along_k_dimension)
    x.xla_gpu_experimental_parallel_collective_overlap_limit != zero(Int32) && PB.encode(e, 336, x.xla_gpu_experimental_parallel_collective_overlap_limit)
    x.xla_gpu_experimental_pipeline_parallelism_opt_level != var"DebugOptions.PipelineParallelismOptLevel".PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE && PB.encode(e, 351, x.xla_gpu_experimental_pipeline_parallelism_opt_level)
    x.xla_gpu_experimental_stream_annotation != false && PB.encode(e, 342, x.xla_gpu_experimental_stream_annotation)
    x.xla_gpu_experimental_use_autotuner_pass != false && PB.encode(e, 396, x.xla_gpu_experimental_use_autotuner_pass)
    x.xla_gpu_experimental_use_ragged_dot_fusion != false && PB.encode(e, 401, x.xla_gpu_experimental_use_ragged_dot_fusion)
    x.xla_gpu_fail_ptx_compilation_on_register_spilling != false && PB.encode(e, 353, x.xla_gpu_fail_ptx_compilation_on_register_spilling)
    x.xla_gpu_filter_kernels_spilling_registers_on_autotuning != false && PB.encode(e, 250, x.xla_gpu_filter_kernels_spilling_registers_on_autotuning)
    x.xla_gpu_first_collective_call_terminate_timeout_seconds != zero(Int32) && PB.encode(e, 392, x.xla_gpu_first_collective_call_terminate_timeout_seconds)
    x.xla_gpu_first_collective_call_warn_stuck_timeout_seconds != zero(Int32) && PB.encode(e, 391, x.xla_gpu_first_collective_call_warn_stuck_timeout_seconds)
    x.xla_gpu_force_compilation_parallelism != zero(Int32) && PB.encode(e, 147, x.xla_gpu_force_compilation_parallelism)
    x.xla_gpu_force_conv_nchw != false && PB.encode(e, 125, x.xla_gpu_force_conv_nchw)
    x.xla_gpu_force_conv_nhwc != false && PB.encode(e, 146, x.xla_gpu_force_conv_nhwc)
    x.xla_gpu_ftz != false && PB.encode(e, 62, x.xla_gpu_ftz)
    x.xla_gpu_fused_attention_use_cudnn_rng != false && PB.encode(e, 235, x.xla_gpu_fused_attention_use_cudnn_rng)
    !isempty(x.xla_gpu_gemm_autotuner_override_file) && PB.encode(e, 434, x.xla_gpu_gemm_autotuner_override_file)
    x.xla_gpu_gemm_rewrite_size_threshold != zero(Int64) && PB.encode(e, 283, x.xla_gpu_gemm_rewrite_size_threshold)
    x.xla_gpu_generate_debug_info != false && PB.encode(e, 348, x.xla_gpu_generate_debug_info)
    x.xla_gpu_generate_line_info != false && PB.encode(e, 349, x.xla_gpu_generate_line_info)
    x.xla_gpu_graph_enable_concurrent_region != false && PB.encode(e, 215, x.xla_gpu_graph_enable_concurrent_region)
    x.xla_gpu_graph_min_graph_size != zero(Int32) && PB.encode(e, 208, x.xla_gpu_graph_min_graph_size)
    !isempty(x.xla_gpu_kernel_cache_file) && PB.encode(e, 306, x.xla_gpu_kernel_cache_file)
    x.xla_gpu_libnvjitlink_mode != var"DebugOptions.LibNvJitLinkMode".LIB_NV_JIT_LINK_MODE_AUTO && PB.encode(e, 343, x.xla_gpu_libnvjitlink_mode)
    !isempty(x.xla_gpu_llvm_ir_file) && PB.encode(e, 150, x.xla_gpu_llvm_ir_file)
    x.xla_gpu_llvm_verification_level != zero(Int32) && PB.encode(e, 256, x.xla_gpu_llvm_verification_level)
    !isempty(x.xla_gpu_load_autotune_results_from) && PB.encode(e, 223, x.xla_gpu_load_autotune_results_from)
    x.xla_gpu_memory_limit_slop_factor != zero(Int32) && PB.encode(e, 260, x.xla_gpu_memory_limit_slop_factor)
    x.xla_gpu_mock_custom_calls != false && PB.encode(e, 245, x.xla_gpu_mock_custom_calls)
    x.xla_gpu_multi_streamed_windowed_einsum != false && PB.encode(e, 280, x.xla_gpu_multi_streamed_windowed_einsum)
    x.xla_gpu_nccl_async_execution != false && PB.encode(e, 393, x.xla_gpu_nccl_async_execution)
    x.xla_gpu_nccl_blocking_communicators != false && PB.encode(e, 390, x.xla_gpu_nccl_blocking_communicators)
    x.xla_gpu_nccl_collective_max_nchannels != zero(Int64) && PB.encode(e, 273, x.xla_gpu_nccl_collective_max_nchannels)
    x.xla_gpu_nccl_init_max_rank_per_root_ratio != zero(Int64) && PB.encode(e, 277, x.xla_gpu_nccl_init_max_rank_per_root_ratio)
    x.xla_gpu_nccl_p2p_max_nchannels != zero(Int64) && PB.encode(e, 274, x.xla_gpu_nccl_p2p_max_nchannels)
    x.xla_gpu_nccl_terminate_on_error != false && PB.encode(e, 301, x.xla_gpu_nccl_terminate_on_error)
    x.xla_gpu_nccl_termination_timeout_seconds != zero(Int64) && PB.encode(e, 163, x.xla_gpu_nccl_termination_timeout_seconds)
    x.xla_gpu_operand_bytes_threshold_for_windowed_einsum != zero(Int64) && PB.encode(e, 339, x.xla_gpu_operand_bytes_threshold_for_windowed_einsum)
    !isempty(x.xla_gpu_override_gemm_autotuner) && PB.encode(e, 295, x.xla_gpu_override_gemm_autotuner)
    !isempty(x.xla_gpu_per_fusion_autotune_cache_dir) && PB.encode(e, 310, x.xla_gpu_per_fusion_autotune_cache_dir)
    x.xla_gpu_pgle_accuracy_checker != var"DebugOptions.PGLEStrictnessLevel".PGLE_STRICTNESS_LEVEL_OFF && PB.encode(e, 341, x.xla_gpu_pgle_accuracy_checker)
    !isempty(x.xla_gpu_pgle_profile_file_or_directory_path) && PB.encode(e, 210, x.xla_gpu_pgle_profile_file_or_directory_path)
    !isempty(x.xla_gpu_ptx_file) && PB.encode(e, 127, x.xla_gpu_ptx_file)
    x.xla_gpu_reduce_scatter_combine_threshold_bytes != zero(Int64) && PB.encode(e, 213, x.xla_gpu_reduce_scatter_combine_threshold_bytes)
    x.xla_gpu_redzone_padding_bytes != zero(Int64) && PB.encode(e, 228, x.xla_gpu_redzone_padding_bytes)
    x.xla_gpu_require_complete_aot_autotune_results != false && PB.encode(e, 284, x.xla_gpu_require_complete_aot_autotune_results)
    x.xla_gpu_require_exclusive_lock != false && PB.encode(e, 347, x.xla_gpu_require_exclusive_lock)
    x.xla_gpu_shape_checks != var"DebugOptions.ShapeChecks".IGNORE && PB.encode(e, 170, x.xla_gpu_shape_checks)
    x.xla_gpu_shard_autotuning != false && PB.encode(e, 304, x.xla_gpu_shard_autotuning)
    x.xla_gpu_strict_conv_algorithm_picker != false && PB.encode(e, 156, x.xla_gpu_strict_conv_algorithm_picker)
    !isempty(x.xla_gpu_target_config_filename) && PB.encode(e, 261, x.xla_gpu_target_config_filename)
    x.xla_gpu_temp_buffer_use_separate_color != false && PB.encode(e, 312, x.xla_gpu_temp_buffer_use_separate_color)
    x.xla_gpu_threshold_for_windowed_einsum_mib != zero(Int64) && PB.encode(e, 265, x.xla_gpu_threshold_for_windowed_einsum_mib)
    x.xla_gpu_triton_gemm_any != false && PB.encode(e, 190, x.xla_gpu_triton_gemm_any)
    x.xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found != false && PB.encode(e, 138, x.xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found)
    x.xla_gpu_unsupported_enable_all_reduce_decomposer != false && PB.encode(e, 384, x.xla_gpu_unsupported_enable_all_reduce_decomposer)
    x.xla_gpu_unsupported_enable_ragged_all_to_all_decomposer != false && PB.encode(e, 350, x.xla_gpu_unsupported_enable_ragged_all_to_all_decomposer)
    x.xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer != false && PB.encode(e, 415, x.xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer)
    x.xla_gpu_unsupported_enable_triton_gemm != false && PB.encode(e, 322, x.xla_gpu_unsupported_enable_triton_gemm)
    x.xla_gpu_unsupported_enable_triton_multi_output_fusion != false && PB.encode(e, 382, x.xla_gpu_unsupported_enable_triton_multi_output_fusion)
    x.xla_gpu_unsupported_override_fast_interconnect_slice_size != zero(Int64) && PB.encode(e, 416, x.xla_gpu_unsupported_override_fast_interconnect_slice_size)
    x.xla_gpu_unsupported_use_all_reduce_one_shot_kernel != false && PB.encode(e, 387, x.xla_gpu_unsupported_use_all_reduce_one_shot_kernel)
    x.xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel != false && PB.encode(e, 375, x.xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel)
    x.xla_gpu_use_embeded_device_lib != false && PB.encode(e, 420, x.xla_gpu_use_embeded_device_lib)
    x.xla_gpu_use_inprocess_lld != false && PB.encode(e, 389, x.xla_gpu_use_inprocess_lld)
    x.xla_gpu_use_memcpy_local_p2p != false && PB.encode(e, 287, x.xla_gpu_use_memcpy_local_p2p)
    x.xla_gpu_use_runtime_fusion != false && PB.encode(e, 181, x.xla_gpu_use_runtime_fusion)
    x.xla_gpu_verify_triton_fusion_numerics != false && PB.encode(e, 291, x.xla_gpu_verify_triton_fusion_numerics)
    x.xla_hlo_graph_addresses != false && PB.encode(e, 2, x.xla_hlo_graph_addresses)
    x.xla_hlo_profile != false && PB.encode(e, 9, x.xla_hlo_profile)
    !isempty(x.xla_disable_hlo_passes) && PB.encode(e, 30, x.xla_disable_hlo_passes)
    !isempty(x.xla_enable_hlo_passes_only) && PB.encode(e, 124, x.xla_enable_hlo_passes_only)
    x.xla_disable_all_hlo_passes != false && PB.encode(e, 104, x.xla_disable_all_hlo_passes)
    x.xla_backend_optimization_level != zero(Int32) && PB.encode(e, 31, x.xla_backend_optimization_level)
    x.xla_embed_ir_in_executable != false && PB.encode(e, 33, x.xla_embed_ir_in_executable)
    x.xla_eliminate_hlo_implicit_broadcast != false && PB.encode(e, 35, x.xla_eliminate_hlo_implicit_broadcast)
    x.xla_cpu_multi_thread_eigen != false && PB.encode(e, 60, x.xla_cpu_multi_thread_eigen)
    x.xla_llvm_enable_alias_scope_metadata != false && PB.encode(e, 70, x.xla_llvm_enable_alias_scope_metadata)
    x.xla_llvm_enable_noalias_metadata != false && PB.encode(e, 71, x.xla_llvm_enable_noalias_metadata)
    x.xla_llvm_enable_invariant_load_metadata != false && PB.encode(e, 72, x.xla_llvm_enable_invariant_load_metadata)
    x.xla_llvm_disable_expensive_passes != false && PB.encode(e, 73, x.xla_llvm_disable_expensive_passes)
    x.xla_test_all_output_layouts != false && PB.encode(e, 90, x.xla_test_all_output_layouts)
    x.xla_test_all_input_layouts != false && PB.encode(e, 91, x.xla_test_all_input_layouts)
    x.xla_hlo_graph_sharding_color != false && PB.encode(e, 92, x.xla_hlo_graph_sharding_color)
    x.xla_cpu_use_onednn != false && PB.encode(e, 97, x.xla_cpu_use_onednn)
    x.xla_allow_excess_precision != false && PB.encode(e, 122, x.xla_allow_excess_precision)
    x.xla_force_host_platform_device_count != zero(Int32) && PB.encode(e, 102, x.xla_force_host_platform_device_count)
    x.xla_hlo_evaluator_use_fast_path != false && PB.encode(e, 106, x.xla_hlo_evaluator_use_fast_path)
    x.xla_allow_scalar_index_dynamic_ops != false && PB.encode(e, 107, x.xla_allow_scalar_index_dynamic_ops)
    x.xla_step_marker_location != var"DebugOptions.StepMarkerLocation".STEP_MARK_AT_ENTRY && PB.encode(e, 108, x.xla_step_marker_location)
    !isempty(x.xla_dump_to) && PB.encode(e, 109, x.xla_dump_to)
    x.xla_flags_reset != false && PB.encode(e, 364, x.xla_flags_reset)
    !isempty(x.xla_dump_hlo_module_re) && PB.encode(e, 110, x.xla_dump_hlo_module_re)
    !isempty(x.xla_dump_hlo_pass_re) && PB.encode(e, 111, x.xla_dump_hlo_pass_re)
    !isempty(x.xla_dump_emitter_re) && PB.encode(e, 433, x.xla_dump_emitter_re)
    x.xla_dump_hlo_as_text != false && PB.encode(e, 112, x.xla_dump_hlo_as_text)
    x.xla_dump_hlo_as_proto != false && PB.encode(e, 113, x.xla_dump_hlo_as_proto)
    x.xla_dump_hlo_as_dot != false && PB.encode(e, 114, x.xla_dump_hlo_as_dot)
    x.xla_dump_hlo_as_url != false && PB.encode(e, 115, x.xla_dump_hlo_as_url)
    x.xla_dump_hlo_as_html != false && PB.encode(e, 116, x.xla_dump_hlo_as_html)
    x.xla_dump_fusion_visualization != false && PB.encode(e, 149, x.xla_dump_fusion_visualization)
    x.xla_dump_hlo_snapshots != false && PB.encode(e, 118, x.xla_dump_hlo_snapshots)
    x.xla_dump_include_timestamp != false && PB.encode(e, 131, x.xla_dump_include_timestamp)
    x.xla_dump_max_hlo_modules != zero(Int32) && PB.encode(e, 132, x.xla_dump_max_hlo_modules)
    x.xla_dump_module_metadata != false && PB.encode(e, 144, x.xla_dump_module_metadata)
    x.xla_dump_compress_protos != false && PB.encode(e, 151, x.xla_dump_compress_protos)
    x.xla_dump_hlo_as_long_text != false && PB.encode(e, 164, x.xla_dump_hlo_as_long_text)
    x.xla_dump_enable_mlir_pretty_form != false && PB.encode(e, 185, x.xla_dump_enable_mlir_pretty_form)
    x.xla_dump_full_hlo_config != false && PB.encode(e, 381, x.xla_dump_full_hlo_config)
    x.xla_tpu_detect_nan != false && PB.encode(e, 135, x.xla_tpu_detect_nan)
    x.xla_tpu_detect_inf != false && PB.encode(e, 136, x.xla_tpu_detect_inf)
    x.xla_cpu_enable_xprof_traceme != false && PB.encode(e, 137, x.xla_cpu_enable_xprof_traceme)
    x.xla_multiheap_size_constraint_per_heap != zero(Int32) && PB.encode(e, 142, x.xla_multiheap_size_constraint_per_heap)
    x.xla_detailed_logging != false && PB.encode(e, 252, x.xla_detailed_logging)
    x.xla_enable_dumping != false && PB.encode(e, 253, x.xla_enable_dumping)
    x.xla_llvm_force_inline_before_split != false && PB.encode(e, 300, x.xla_llvm_force_inline_before_split)
    x.xla_dump_disable_metadata != false && PB.encode(e, 153, x.xla_dump_disable_metadata)
    !isempty(x.xla_dump_hlo_pipeline_re) && PB.encode(e, 154, x.xla_dump_hlo_pipeline_re)
    x.xla_cpu_use_acl != false && PB.encode(e, 174, x.xla_cpu_use_acl)
    x.xla_cpu_strict_dot_conv_math != false && PB.encode(e, 175, x.xla_cpu_strict_dot_conv_math)
    x.xla_dump_latency_hiding_schedule != false && PB.encode(e, 182, x.xla_dump_latency_hiding_schedule)
    x.xla_partitioning_algorithm != var"DebugOptions.PartitioningAlgorithm".PARTITIONING_ALGORITHM_NOOP && PB.encode(e, 187, x.xla_partitioning_algorithm)
    x.xla_debug_buffer_assignment_show_max != zero(Int64) && PB.encode(e, 251, x.xla_debug_buffer_assignment_show_max)
    x.xla_detect_unstable_reductions != var"DebugOptions.DetectionMode".DETECTION_MODE_NONE && PB.encode(e, 403, x.xla_detect_unstable_reductions)
    x.xla_detect_unstable_reductions_post_optimizations != var"DebugOptions.DetectionMode".DETECTION_MODE_NONE && PB.encode(e, 432, x.xla_detect_unstable_reductions_post_optimizations)
    x.xla_gpu_detect_nan != var"DebugOptions.DetectionMode".DETECTION_MODE_NONE && PB.encode(e, 426, x.xla_gpu_detect_nan)
    x.xla_gpu_detect_inf != var"DebugOptions.DetectionMode".DETECTION_MODE_NONE && PB.encode(e, 428, x.xla_gpu_detect_inf)
    x.xla_dump_large_constants != false && PB.encode(e, 290, x.xla_dump_large_constants)
    x.xla_reduce_window_rewrite_base_length != zero(Int64) && PB.encode(e, 293, x.xla_reduce_window_rewrite_base_length)
    x.xla_cmd_buffer_trace_cache_size != zero(Int64) && PB.encode(e, 311, x.xla_cmd_buffer_trace_cache_size)
    x.xla_syntax_sugar_async_ops != false && PB.encode(e, 315, x.xla_syntax_sugar_async_ops)
    x.xla_enable_command_buffers_during_profiling != false && PB.encode(e, 317, x.xla_enable_command_buffers_during_profiling)
    x.xla_ignore_channel_id != false && PB.encode(e, 330, x.xla_ignore_channel_id)
    x.xla_pjrt_allow_auto_layout_in_hlo != false && PB.encode(e, 344, x.xla_pjrt_allow_auto_layout_in_hlo)
    x.xla_test_add_command_buffer_mode != false && PB.encode(e, 373, x.xla_test_add_command_buffer_mode)
    !isempty(x.xla_gpu_experimental_matmul_perf_table_path) && PB.encode(e, 383, x.xla_gpu_experimental_matmul_perf_table_path)
    x.xla_early_exit_with_layouts != false && PB.encode(e, 397, x.xla_early_exit_with_layouts)
    x.xla_gpu_experimental_scaled_dot_with_triton != false && PB.encode(e, 410, x.xla_gpu_experimental_scaled_dot_with_triton)
    x.xla_gpu_experimental_use_raft_select_k != false && PB.encode(e, 413, x.xla_gpu_experimental_use_raft_select_k)
    !isempty(x.xla_backend_extra_options) && PB.encode(e, 500, x.xla_backend_extra_options)
    return position(e.io) - initpos
end
function PB._encoded_size(x::DebugOptions)
    encoded_size = 0
    x.xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled != false && (encoded_size += PB._encoded_size(x.xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled, 439))
    x.xla_disable_automatic_host_compute_offload != false && (encoded_size += PB._encoded_size(x.xla_disable_automatic_host_compute_offload, 408))
    x.xla_enable_scoped_logging_timers != false && (encoded_size += PB._encoded_size(x.xla_enable_scoped_logging_timers, 436))
    x.xla_hlo_pass_fix_detect_cycles != false && (encoded_size += PB._encoded_size(x.xla_hlo_pass_fix_detect_cycles, 370))
    x.xla_keep_shardings_after_spmd != false && (encoded_size += PB._encoded_size(x.xla_keep_shardings_after_spmd, 419))
    x.xla_unsupported_crash_on_hlo_pass_fix_max_iterations != false && (encoded_size += PB._encoded_size(x.xla_unsupported_crash_on_hlo_pass_fix_max_iterations, 363))
    x.xla_unsupported_crash_on_hlo_pass_noop_change != false && (encoded_size += PB._encoded_size(x.xla_unsupported_crash_on_hlo_pass_noop_change, 379))
    x.xla_unsupported_crash_on_hlo_pass_silent_hlo_change != false && (encoded_size += PB._encoded_size(x.xla_unsupported_crash_on_hlo_pass_silent_hlo_change, 380))
    x.xla_cpu_collective_call_terminate_timeout_seconds != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_cpu_collective_call_terminate_timeout_seconds, 417))
    x.xla_cpu_collective_call_warn_stuck_seconds != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_cpu_collective_call_warn_stuck_seconds, 418))
    x.xla_cpu_collective_timeout_seconds != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_cpu_collective_timeout_seconds, 438))
    x.xla_cpu_copy_insertion_use_region_analysis != false && (encoded_size += PB._encoded_size(x.xla_cpu_copy_insertion_use_region_analysis, 337))
    x.xla_cpu_emitter_verification_level != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_cpu_emitter_verification_level, 395))
    x.xla_cpu_enable_concurrency_optimized_scheduler != false && (encoded_size += PB._encoded_size(x.xla_cpu_enable_concurrency_optimized_scheduler, 307))
    x.xla_cpu_enable_fast_math != false && (encoded_size += PB._encoded_size(x.xla_cpu_enable_fast_math, 99))
    x.xla_cpu_enable_fast_min_max != false && (encoded_size += PB._encoded_size(x.xla_cpu_enable_fast_min_max, 140))
    x.xla_cpu_enable_platform_dependent_math != false && (encoded_size += PB._encoded_size(x.xla_cpu_enable_platform_dependent_math, 425))
    x.xla_cpu_experimental_onednn_custom_call != false && (encoded_size += PB._encoded_size(x.xla_cpu_experimental_onednn_custom_call, 412))
    !isempty(x.xla_cpu_experimental_onednn_fusion_type) && (encoded_size += PB._encoded_size(x.xla_cpu_experimental_onednn_fusion_type, 399))
    !isempty(x.xla_cpu_experimental_xnn_fusion_type) && (encoded_size += PB._encoded_size(x.xla_cpu_experimental_xnn_fusion_type, 400))
    x.xla_cpu_experimental_xnn_graph_fusion_mode != var"DebugOptions.XnnGraphFusionMode".XNN_GRAPH_FUSION_MODE_DISABLED && (encoded_size += PB._encoded_size(x.xla_cpu_experimental_xnn_graph_fusion_mode, 365))
    !isempty(x.xla_cpu_experimental_ynn_fusion_type) && (encoded_size += PB._encoded_size(x.xla_cpu_experimental_ynn_fusion_type, 422))
    x.xla_cpu_fast_math_honor_division != false && (encoded_size += PB._encoded_size(x.xla_cpu_fast_math_honor_division, 126))
    x.xla_cpu_fast_math_honor_functions != false && (encoded_size += PB._encoded_size(x.xla_cpu_fast_math_honor_functions, 129))
    x.xla_cpu_fast_math_honor_infs != false && (encoded_size += PB._encoded_size(x.xla_cpu_fast_math_honor_infs, 121))
    x.xla_cpu_fast_math_honor_nans != false && (encoded_size += PB._encoded_size(x.xla_cpu_fast_math_honor_nans, 120))
    x.xla_cpu_generate_unique_c_style_kernel_entry_points != false && (encoded_size += PB._encoded_size(x.xla_cpu_generate_unique_c_style_kernel_entry_points, 372))
    !isempty(x.xla_cpu_max_isa) && (encoded_size += PB._encoded_size(x.xla_cpu_max_isa, 333))
    x.xla_cpu_parallel_codegen_split_count != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_cpu_parallel_codegen_split_count, 323))
    x.xla_cpu_prefer_vector_width != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_cpu_prefer_vector_width, 308))
    x.xla_cpu_use_fusion_emitters != false && (encoded_size += PB._encoded_size(x.xla_cpu_use_fusion_emitters, 376))
    x.xla_cpu_use_xnnpack != false && (encoded_size += PB._encoded_size(x.xla_cpu_use_xnnpack, 359))
    x.xla_enable_fast_math != false && (encoded_size += PB._encoded_size(x.xla_enable_fast_math, 335))
    !isnothing(x.xla_gpu_experimental_thunk_buffer_debug_filter) && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_thunk_buffer_debug_filter, 424))
    x.xla_dump_hlo_unoptimized_snapshots != false && (encoded_size += PB._encoded_size(x.xla_dump_hlo_unoptimized_snapshots, 405))
    x.xla_enable_enzyme_comms_opt != false && (encoded_size += PB._encoded_size(x.xla_enable_enzyme_comms_opt, 429))
    !isempty(x.xla_gpu_algorithm_denylist_path) && (encoded_size += PB._encoded_size(x.xla_gpu_algorithm_denylist_path, 128))
    x.xla_gpu_all_gather_combine_threshold_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_all_gather_combine_threshold_bytes, 212))
    x.xla_gpu_all_reduce_blueconnect_num_devices_per_host != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_all_reduce_blueconnect_num_devices_per_host, 159))
    x.xla_gpu_all_reduce_combine_threshold_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_all_reduce_combine_threshold_bytes, 157))
    !isempty(x.xla_gpu_analytical_latency_estimator_options) && (encoded_size += PB._encoded_size(x.xla_gpu_analytical_latency_estimator_options, 357))
    x.xla_gpu_async_dot != false && (encoded_size += PB._encoded_size(x.xla_gpu_async_dot, 321))
    x.xla_gpu_auto_spmd_partitioning_memory_budget_gb != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_auto_spmd_partitioning_memory_budget_gb, 224))
    x.xla_gpu_auto_spmd_partitioning_memory_budget_ratio !== zero(Float32) && (encoded_size += PB._encoded_size(x.xla_gpu_auto_spmd_partitioning_memory_budget_ratio, 225))
    x.xla_gpu_autotune_gemm_rtol !== zero(Float32) && (encoded_size += PB._encoded_size(x.xla_gpu_autotune_gemm_rtol, 316))
    x.xla_gpu_autotune_level != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_autotune_level, 123))
    x.xla_gpu_autotune_max_solutions != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_autotune_max_solutions, 288))
    x.xla_gpu_collect_cost_model_stats != false && (encoded_size += PB._encoded_size(x.xla_gpu_collect_cost_model_stats, 240))
    x.xla_gpu_collective_inflation_factor != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_collective_inflation_factor, 205))
    x.xla_gpu_collective_permute_combine_threshold_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_collective_permute_combine_threshold_bytes, 378))
    x.xla_gpu_collective_permute_decomposer_threshold != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_collective_permute_decomposer_threshold, 237))
    x.xla_gpu_collectives_use_persistent_cliques != false && (encoded_size += PB._encoded_size(x.xla_gpu_collectives_use_persistent_cliques, 354))
    x.xla_gpu_command_buffer_scheduling_mode != var"DebugOptions.CommandBufferSchedulingMode".SERIALIZE && (encoded_size += PB._encoded_size(x.xla_gpu_command_buffer_scheduling_mode, 404))
    x.xla_gpu_command_buffer_unroll_loops != false && (encoded_size += PB._encoded_size(x.xla_gpu_command_buffer_unroll_loops, 411))
    x.xla_gpu_copy_insertion_use_region_analysis != false && (encoded_size += PB._encoded_size(x.xla_gpu_copy_insertion_use_region_analysis, 236))
    x.xla_gpu_crash_on_verification_failures != false && (encoded_size += PB._encoded_size(x.xla_gpu_crash_on_verification_failures, 101))
    x.xla_gpu_cublas_fallback != false && (encoded_size += PB._encoded_size(x.xla_gpu_cublas_fallback, 247))
    !isempty(x.xla_gpu_cuda_data_dir) && (encoded_size += PB._encoded_size(x.xla_gpu_cuda_data_dir, 61))
    x.xla_gpu_cudnn_gemm_fusion_level != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_cudnn_gemm_fusion_level, 285))
    x.xla_gpu_cudnn_gemm_max_plans != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_cudnn_gemm_max_plans, 318))
    x.xla_gpu_deterministic_ops != false && (encoded_size += PB._encoded_size(x.xla_gpu_deterministic_ops, 148))
    !isempty(x.xla_gpu_disable_async_collectives) && (encoded_size += PB._encoded_size(x.xla_gpu_disable_async_collectives, 289))
    x.xla_gpu_disable_gpuasm_optimizations != false && (encoded_size += PB._encoded_size(x.xla_gpu_disable_gpuasm_optimizations, 103))
    x.xla_gpu_dot_merger_threshold_mb != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_dot_merger_threshold_mb, 331))
    !isempty(x.xla_gpu_dump_autotune_logs_to) && (encoded_size += PB._encoded_size(x.xla_gpu_dump_autotune_logs_to, 292))
    !isempty(x.xla_gpu_dump_autotune_results_to) && (encoded_size += PB._encoded_size(x.xla_gpu_dump_autotune_results_to, 222))
    x.xla_gpu_dump_autotuned_gemm_fusions != false && (encoded_size += PB._encoded_size(x.xla_gpu_dump_autotuned_gemm_fusions, 232))
    x.xla_gpu_dump_llvmir != false && (encoded_size += PB._encoded_size(x.xla_gpu_dump_llvmir, 155))
    x.xla_gpu_enable_all_gather_combine_by_dim != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_all_gather_combine_by_dim, 254))
    x.xla_gpu_enable_analytical_latency_estimator != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_analytical_latency_estimator, 255))
    x.xla_gpu_enable_analytical_sol_latency_estimator != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_analytical_sol_latency_estimator, 356))
    x.xla_gpu_enable_approx_costly_collectives != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_approx_costly_collectives, 305))
    !isempty(x.xla_gpu_enable_command_buffer) && (encoded_size += PB._encoded_size(x.xla_gpu_enable_command_buffer, 258))
    x.xla_gpu_enable_cub_radix_sort != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_cub_radix_sort, 259))
    x.xla_gpu_enable_cublaslt != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_cublaslt, 166))
    x.xla_gpu_enable_cudnn_int8x32_convolution_reordering != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_cudnn_int8x32_convolution_reordering, 189))
    x.xla_gpu_enable_cudnn_layer_norm != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_cudnn_layer_norm, 262))
    x.xla_gpu_enable_dynamic_slice_fusion != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_dynamic_slice_fusion, 105))
    x.xla_gpu_enable_fast_min_max != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_fast_min_max, 100))
    x.xla_gpu_enable_highest_priority_async_stream != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_highest_priority_async_stream, 216))
    x.xla_gpu_enable_host_memory_offloading != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_host_memory_offloading, 296))
    x.xla_gpu_enable_latency_hiding_scheduler != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_latency_hiding_scheduler, 186))
    x.xla_gpu_enable_libnvptxcompiler != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_libnvptxcompiler, 269))
    x.xla_gpu_enable_llvm_module_compilation_parallelism != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_llvm_module_compilation_parallelism, 268))
    x.xla_gpu_enable_nccl_clique_optimization != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_nccl_clique_optimization, 244))
    x.xla_gpu_enable_nccl_comm_splitting != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_nccl_comm_splitting, 272))
    x.xla_gpu_enable_nccl_user_buffers != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_nccl_user_buffers, 267))
    x.xla_gpu_enable_pipelined_all_gather != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_pipelined_all_gather, 227))
    x.xla_gpu_enable_pipelined_all_reduce != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_pipelined_all_reduce, 217))
    x.xla_gpu_enable_pipelined_collectives != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_pipelined_collectives, 239))
    x.xla_gpu_enable_pipelined_p2p != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_pipelined_p2p, 246))
    x.xla_gpu_enable_pipelined_reduce_scatter != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_pipelined_reduce_scatter, 231))
    x.xla_gpu_enable_reassociation_for_converted_ar != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_reassociation_for_converted_ar, 209))
    x.xla_gpu_enable_reduce_scatter_combine_by_dim != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_reduce_scatter_combine_by_dim, 257))
    x.xla_gpu_enable_reduction_epilogue_fusion != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_reduction_epilogue_fusion, 243))
    x.xla_gpu_enable_scatter_determinism_expander != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_scatter_determinism_expander, 345))
    x.xla_gpu_enable_shared_constants != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_shared_constants, 165))
    x.xla_gpu_enable_split_k_autotuning != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_split_k_autotuning, 241))
    x.xla_gpu_enable_triton_gemm != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_triton_gemm, 188))
    x.xla_gpu_enable_while_loop_double_buffering != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_while_loop_double_buffering, 248))
    x.xla_gpu_enable_while_loop_reduce_scatter_code_motion != false && (encoded_size += PB._encoded_size(x.xla_gpu_enable_while_loop_reduce_scatter_code_motion, 203))
    x.xla_gpu_enable_while_loop_unrolling != var"DebugOptions.WhileLoopUnrolling".WHILE_LOOP_UNROLLING_NO_UNROLL && (encoded_size += PB._encoded_size(x.xla_gpu_enable_while_loop_unrolling, 294))
    x.xla_gpu_exclude_nondeterministic_ops != false && (encoded_size += PB._encoded_size(x.xla_gpu_exclude_nondeterministic_ops, 297))
    x.xla_gpu_executable_embed_debug_info != false && (encoded_size += PB._encoded_size(x.xla_gpu_executable_embed_debug_info, 437))
    x.xla_gpu_executable_terminate_timeout_seconds != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_executable_terminate_timeout_seconds, 328))
    x.xla_gpu_executable_warn_stuck_timeout_seconds != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_executable_warn_stuck_timeout_seconds, 327))
    x.xla_gpu_exhaustive_tiling_search != false && (encoded_size += PB._encoded_size(x.xla_gpu_exhaustive_tiling_search, 219))
    x.xla_gpu_experimental_allow_unroll_factor_eight != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_allow_unroll_factor_eight, 430))
    x.xla_gpu_experimental_aot_compiled_thunks != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_aot_compiled_thunks, 435))
    x.xla_gpu_experimental_autotune_cache_mode != var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_UNSPECIFIED && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_autotune_cache_mode, 324))
    !isempty(x.xla_gpu_experimental_autotuner_cache_dir) && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_autotuner_cache_dir, 407))
    x.xla_gpu_experimental_collective_cse_distance_threshold != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_collective_cse_distance_threshold, 374))
    !isempty(x.xla_gpu_experimental_collective_perf_table_path) && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_collective_perf_table_path, 377))
    x.xla_gpu_experimental_disable_binary_libraries != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_disable_binary_libraries, 329))
    x.xla_gpu_experimental_dump_fdo_profiles != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_dump_fdo_profiles, 338))
    x.xla_gpu_experimental_dump_gpu_executable != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_dump_gpu_executable, 427))
    x.xla_gpu_experimental_enable_alltoall_windowed_einsum != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_alltoall_windowed_einsum, 360))
    x.xla_gpu_experimental_enable_buffer_saver_on_thunks != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_buffer_saver_on_thunks, 431))
    x.xla_gpu_experimental_enable_checksum_tracing_on_thunks != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_checksum_tracing_on_thunks, 414))
    x.xla_gpu_experimental_enable_fusion_autotuner != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_fusion_autotuner, 409))
    x.xla_gpu_experimental_enable_fusion_block_level_rewriter != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_fusion_block_level_rewriter, 334))
    x.xla_gpu_experimental_enable_heuristic_collective_combining != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_heuristic_collective_combining, 366))
    x.xla_gpu_experimental_enable_nccl_symmetric_buffers != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_nccl_symmetric_buffers, 406))
    x.xla_gpu_experimental_enable_nvshmem != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_nvshmem, 388))
    x.xla_gpu_experimental_enable_split_k_rewrite != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_split_k_rewrite, 386))
    x.xla_gpu_experimental_enable_subchannel_dequantisation_fusion != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_subchannel_dequantisation_fusion, 368))
    x.xla_gpu_experimental_enable_triton_heroless_priority_fusion != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_triton_heroless_priority_fusion, 340))
    x.xla_gpu_experimental_enable_triton_warp_specialization != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_enable_triton_warp_specialization, 421))
    x.xla_gpu_experimental_pack_dot_operands_along_k_dimension != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_pack_dot_operands_along_k_dimension, 362))
    x.xla_gpu_experimental_parallel_collective_overlap_limit != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_parallel_collective_overlap_limit, 336))
    x.xla_gpu_experimental_pipeline_parallelism_opt_level != var"DebugOptions.PipelineParallelismOptLevel".PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_pipeline_parallelism_opt_level, 351))
    x.xla_gpu_experimental_stream_annotation != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_stream_annotation, 342))
    x.xla_gpu_experimental_use_autotuner_pass != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_use_autotuner_pass, 396))
    x.xla_gpu_experimental_use_ragged_dot_fusion != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_use_ragged_dot_fusion, 401))
    x.xla_gpu_fail_ptx_compilation_on_register_spilling != false && (encoded_size += PB._encoded_size(x.xla_gpu_fail_ptx_compilation_on_register_spilling, 353))
    x.xla_gpu_filter_kernels_spilling_registers_on_autotuning != false && (encoded_size += PB._encoded_size(x.xla_gpu_filter_kernels_spilling_registers_on_autotuning, 250))
    x.xla_gpu_first_collective_call_terminate_timeout_seconds != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_first_collective_call_terminate_timeout_seconds, 392))
    x.xla_gpu_first_collective_call_warn_stuck_timeout_seconds != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_first_collective_call_warn_stuck_timeout_seconds, 391))
    x.xla_gpu_force_compilation_parallelism != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_force_compilation_parallelism, 147))
    x.xla_gpu_force_conv_nchw != false && (encoded_size += PB._encoded_size(x.xla_gpu_force_conv_nchw, 125))
    x.xla_gpu_force_conv_nhwc != false && (encoded_size += PB._encoded_size(x.xla_gpu_force_conv_nhwc, 146))
    x.xla_gpu_ftz != false && (encoded_size += PB._encoded_size(x.xla_gpu_ftz, 62))
    x.xla_gpu_fused_attention_use_cudnn_rng != false && (encoded_size += PB._encoded_size(x.xla_gpu_fused_attention_use_cudnn_rng, 235))
    !isempty(x.xla_gpu_gemm_autotuner_override_file) && (encoded_size += PB._encoded_size(x.xla_gpu_gemm_autotuner_override_file, 434))
    x.xla_gpu_gemm_rewrite_size_threshold != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_gemm_rewrite_size_threshold, 283))
    x.xla_gpu_generate_debug_info != false && (encoded_size += PB._encoded_size(x.xla_gpu_generate_debug_info, 348))
    x.xla_gpu_generate_line_info != false && (encoded_size += PB._encoded_size(x.xla_gpu_generate_line_info, 349))
    x.xla_gpu_graph_enable_concurrent_region != false && (encoded_size += PB._encoded_size(x.xla_gpu_graph_enable_concurrent_region, 215))
    x.xla_gpu_graph_min_graph_size != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_graph_min_graph_size, 208))
    !isempty(x.xla_gpu_kernel_cache_file) && (encoded_size += PB._encoded_size(x.xla_gpu_kernel_cache_file, 306))
    x.xla_gpu_libnvjitlink_mode != var"DebugOptions.LibNvJitLinkMode".LIB_NV_JIT_LINK_MODE_AUTO && (encoded_size += PB._encoded_size(x.xla_gpu_libnvjitlink_mode, 343))
    !isempty(x.xla_gpu_llvm_ir_file) && (encoded_size += PB._encoded_size(x.xla_gpu_llvm_ir_file, 150))
    x.xla_gpu_llvm_verification_level != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_llvm_verification_level, 256))
    !isempty(x.xla_gpu_load_autotune_results_from) && (encoded_size += PB._encoded_size(x.xla_gpu_load_autotune_results_from, 223))
    x.xla_gpu_memory_limit_slop_factor != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_gpu_memory_limit_slop_factor, 260))
    x.xla_gpu_mock_custom_calls != false && (encoded_size += PB._encoded_size(x.xla_gpu_mock_custom_calls, 245))
    x.xla_gpu_multi_streamed_windowed_einsum != false && (encoded_size += PB._encoded_size(x.xla_gpu_multi_streamed_windowed_einsum, 280))
    x.xla_gpu_nccl_async_execution != false && (encoded_size += PB._encoded_size(x.xla_gpu_nccl_async_execution, 393))
    x.xla_gpu_nccl_blocking_communicators != false && (encoded_size += PB._encoded_size(x.xla_gpu_nccl_blocking_communicators, 390))
    x.xla_gpu_nccl_collective_max_nchannels != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_nccl_collective_max_nchannels, 273))
    x.xla_gpu_nccl_init_max_rank_per_root_ratio != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_nccl_init_max_rank_per_root_ratio, 277))
    x.xla_gpu_nccl_p2p_max_nchannels != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_nccl_p2p_max_nchannels, 274))
    x.xla_gpu_nccl_terminate_on_error != false && (encoded_size += PB._encoded_size(x.xla_gpu_nccl_terminate_on_error, 301))
    x.xla_gpu_nccl_termination_timeout_seconds != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_nccl_termination_timeout_seconds, 163))
    x.xla_gpu_operand_bytes_threshold_for_windowed_einsum != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_operand_bytes_threshold_for_windowed_einsum, 339))
    !isempty(x.xla_gpu_override_gemm_autotuner) && (encoded_size += PB._encoded_size(x.xla_gpu_override_gemm_autotuner, 295))
    !isempty(x.xla_gpu_per_fusion_autotune_cache_dir) && (encoded_size += PB._encoded_size(x.xla_gpu_per_fusion_autotune_cache_dir, 310))
    x.xla_gpu_pgle_accuracy_checker != var"DebugOptions.PGLEStrictnessLevel".PGLE_STRICTNESS_LEVEL_OFF && (encoded_size += PB._encoded_size(x.xla_gpu_pgle_accuracy_checker, 341))
    !isempty(x.xla_gpu_pgle_profile_file_or_directory_path) && (encoded_size += PB._encoded_size(x.xla_gpu_pgle_profile_file_or_directory_path, 210))
    !isempty(x.xla_gpu_ptx_file) && (encoded_size += PB._encoded_size(x.xla_gpu_ptx_file, 127))
    x.xla_gpu_reduce_scatter_combine_threshold_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_reduce_scatter_combine_threshold_bytes, 213))
    x.xla_gpu_redzone_padding_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_redzone_padding_bytes, 228))
    x.xla_gpu_require_complete_aot_autotune_results != false && (encoded_size += PB._encoded_size(x.xla_gpu_require_complete_aot_autotune_results, 284))
    x.xla_gpu_require_exclusive_lock != false && (encoded_size += PB._encoded_size(x.xla_gpu_require_exclusive_lock, 347))
    x.xla_gpu_shape_checks != var"DebugOptions.ShapeChecks".IGNORE && (encoded_size += PB._encoded_size(x.xla_gpu_shape_checks, 170))
    x.xla_gpu_shard_autotuning != false && (encoded_size += PB._encoded_size(x.xla_gpu_shard_autotuning, 304))
    x.xla_gpu_strict_conv_algorithm_picker != false && (encoded_size += PB._encoded_size(x.xla_gpu_strict_conv_algorithm_picker, 156))
    !isempty(x.xla_gpu_target_config_filename) && (encoded_size += PB._encoded_size(x.xla_gpu_target_config_filename, 261))
    x.xla_gpu_temp_buffer_use_separate_color != false && (encoded_size += PB._encoded_size(x.xla_gpu_temp_buffer_use_separate_color, 312))
    x.xla_gpu_threshold_for_windowed_einsum_mib != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_threshold_for_windowed_einsum_mib, 265))
    x.xla_gpu_triton_gemm_any != false && (encoded_size += PB._encoded_size(x.xla_gpu_triton_gemm_any, 190))
    x.xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found != false && (encoded_size += PB._encoded_size(x.xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found, 138))
    x.xla_gpu_unsupported_enable_all_reduce_decomposer != false && (encoded_size += PB._encoded_size(x.xla_gpu_unsupported_enable_all_reduce_decomposer, 384))
    x.xla_gpu_unsupported_enable_ragged_all_to_all_decomposer != false && (encoded_size += PB._encoded_size(x.xla_gpu_unsupported_enable_ragged_all_to_all_decomposer, 350))
    x.xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer != false && (encoded_size += PB._encoded_size(x.xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer, 415))
    x.xla_gpu_unsupported_enable_triton_gemm != false && (encoded_size += PB._encoded_size(x.xla_gpu_unsupported_enable_triton_gemm, 322))
    x.xla_gpu_unsupported_enable_triton_multi_output_fusion != false && (encoded_size += PB._encoded_size(x.xla_gpu_unsupported_enable_triton_multi_output_fusion, 382))
    x.xla_gpu_unsupported_override_fast_interconnect_slice_size != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_gpu_unsupported_override_fast_interconnect_slice_size, 416))
    x.xla_gpu_unsupported_use_all_reduce_one_shot_kernel != false && (encoded_size += PB._encoded_size(x.xla_gpu_unsupported_use_all_reduce_one_shot_kernel, 387))
    x.xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel != false && (encoded_size += PB._encoded_size(x.xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel, 375))
    x.xla_gpu_use_embeded_device_lib != false && (encoded_size += PB._encoded_size(x.xla_gpu_use_embeded_device_lib, 420))
    x.xla_gpu_use_inprocess_lld != false && (encoded_size += PB._encoded_size(x.xla_gpu_use_inprocess_lld, 389))
    x.xla_gpu_use_memcpy_local_p2p != false && (encoded_size += PB._encoded_size(x.xla_gpu_use_memcpy_local_p2p, 287))
    x.xla_gpu_use_runtime_fusion != false && (encoded_size += PB._encoded_size(x.xla_gpu_use_runtime_fusion, 181))
    x.xla_gpu_verify_triton_fusion_numerics != false && (encoded_size += PB._encoded_size(x.xla_gpu_verify_triton_fusion_numerics, 291))
    x.xla_hlo_graph_addresses != false && (encoded_size += PB._encoded_size(x.xla_hlo_graph_addresses, 2))
    x.xla_hlo_profile != false && (encoded_size += PB._encoded_size(x.xla_hlo_profile, 9))
    !isempty(x.xla_disable_hlo_passes) && (encoded_size += PB._encoded_size(x.xla_disable_hlo_passes, 30))
    !isempty(x.xla_enable_hlo_passes_only) && (encoded_size += PB._encoded_size(x.xla_enable_hlo_passes_only, 124))
    x.xla_disable_all_hlo_passes != false && (encoded_size += PB._encoded_size(x.xla_disable_all_hlo_passes, 104))
    x.xla_backend_optimization_level != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_backend_optimization_level, 31))
    x.xla_embed_ir_in_executable != false && (encoded_size += PB._encoded_size(x.xla_embed_ir_in_executable, 33))
    x.xla_eliminate_hlo_implicit_broadcast != false && (encoded_size += PB._encoded_size(x.xla_eliminate_hlo_implicit_broadcast, 35))
    x.xla_cpu_multi_thread_eigen != false && (encoded_size += PB._encoded_size(x.xla_cpu_multi_thread_eigen, 60))
    x.xla_llvm_enable_alias_scope_metadata != false && (encoded_size += PB._encoded_size(x.xla_llvm_enable_alias_scope_metadata, 70))
    x.xla_llvm_enable_noalias_metadata != false && (encoded_size += PB._encoded_size(x.xla_llvm_enable_noalias_metadata, 71))
    x.xla_llvm_enable_invariant_load_metadata != false && (encoded_size += PB._encoded_size(x.xla_llvm_enable_invariant_load_metadata, 72))
    x.xla_llvm_disable_expensive_passes != false && (encoded_size += PB._encoded_size(x.xla_llvm_disable_expensive_passes, 73))
    x.xla_test_all_output_layouts != false && (encoded_size += PB._encoded_size(x.xla_test_all_output_layouts, 90))
    x.xla_test_all_input_layouts != false && (encoded_size += PB._encoded_size(x.xla_test_all_input_layouts, 91))
    x.xla_hlo_graph_sharding_color != false && (encoded_size += PB._encoded_size(x.xla_hlo_graph_sharding_color, 92))
    x.xla_cpu_use_onednn != false && (encoded_size += PB._encoded_size(x.xla_cpu_use_onednn, 97))
    x.xla_allow_excess_precision != false && (encoded_size += PB._encoded_size(x.xla_allow_excess_precision, 122))
    x.xla_force_host_platform_device_count != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_force_host_platform_device_count, 102))
    x.xla_hlo_evaluator_use_fast_path != false && (encoded_size += PB._encoded_size(x.xla_hlo_evaluator_use_fast_path, 106))
    x.xla_allow_scalar_index_dynamic_ops != false && (encoded_size += PB._encoded_size(x.xla_allow_scalar_index_dynamic_ops, 107))
    x.xla_step_marker_location != var"DebugOptions.StepMarkerLocation".STEP_MARK_AT_ENTRY && (encoded_size += PB._encoded_size(x.xla_step_marker_location, 108))
    !isempty(x.xla_dump_to) && (encoded_size += PB._encoded_size(x.xla_dump_to, 109))
    x.xla_flags_reset != false && (encoded_size += PB._encoded_size(x.xla_flags_reset, 364))
    !isempty(x.xla_dump_hlo_module_re) && (encoded_size += PB._encoded_size(x.xla_dump_hlo_module_re, 110))
    !isempty(x.xla_dump_hlo_pass_re) && (encoded_size += PB._encoded_size(x.xla_dump_hlo_pass_re, 111))
    !isempty(x.xla_dump_emitter_re) && (encoded_size += PB._encoded_size(x.xla_dump_emitter_re, 433))
    x.xla_dump_hlo_as_text != false && (encoded_size += PB._encoded_size(x.xla_dump_hlo_as_text, 112))
    x.xla_dump_hlo_as_proto != false && (encoded_size += PB._encoded_size(x.xla_dump_hlo_as_proto, 113))
    x.xla_dump_hlo_as_dot != false && (encoded_size += PB._encoded_size(x.xla_dump_hlo_as_dot, 114))
    x.xla_dump_hlo_as_url != false && (encoded_size += PB._encoded_size(x.xla_dump_hlo_as_url, 115))
    x.xla_dump_hlo_as_html != false && (encoded_size += PB._encoded_size(x.xla_dump_hlo_as_html, 116))
    x.xla_dump_fusion_visualization != false && (encoded_size += PB._encoded_size(x.xla_dump_fusion_visualization, 149))
    x.xla_dump_hlo_snapshots != false && (encoded_size += PB._encoded_size(x.xla_dump_hlo_snapshots, 118))
    x.xla_dump_include_timestamp != false && (encoded_size += PB._encoded_size(x.xla_dump_include_timestamp, 131))
    x.xla_dump_max_hlo_modules != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_dump_max_hlo_modules, 132))
    x.xla_dump_module_metadata != false && (encoded_size += PB._encoded_size(x.xla_dump_module_metadata, 144))
    x.xla_dump_compress_protos != false && (encoded_size += PB._encoded_size(x.xla_dump_compress_protos, 151))
    x.xla_dump_hlo_as_long_text != false && (encoded_size += PB._encoded_size(x.xla_dump_hlo_as_long_text, 164))
    x.xla_dump_enable_mlir_pretty_form != false && (encoded_size += PB._encoded_size(x.xla_dump_enable_mlir_pretty_form, 185))
    x.xla_dump_full_hlo_config != false && (encoded_size += PB._encoded_size(x.xla_dump_full_hlo_config, 381))
    x.xla_tpu_detect_nan != false && (encoded_size += PB._encoded_size(x.xla_tpu_detect_nan, 135))
    x.xla_tpu_detect_inf != false && (encoded_size += PB._encoded_size(x.xla_tpu_detect_inf, 136))
    x.xla_cpu_enable_xprof_traceme != false && (encoded_size += PB._encoded_size(x.xla_cpu_enable_xprof_traceme, 137))
    x.xla_multiheap_size_constraint_per_heap != zero(Int32) && (encoded_size += PB._encoded_size(x.xla_multiheap_size_constraint_per_heap, 142))
    x.xla_detailed_logging != false && (encoded_size += PB._encoded_size(x.xla_detailed_logging, 252))
    x.xla_enable_dumping != false && (encoded_size += PB._encoded_size(x.xla_enable_dumping, 253))
    x.xla_llvm_force_inline_before_split != false && (encoded_size += PB._encoded_size(x.xla_llvm_force_inline_before_split, 300))
    x.xla_dump_disable_metadata != false && (encoded_size += PB._encoded_size(x.xla_dump_disable_metadata, 153))
    !isempty(x.xla_dump_hlo_pipeline_re) && (encoded_size += PB._encoded_size(x.xla_dump_hlo_pipeline_re, 154))
    x.xla_cpu_use_acl != false && (encoded_size += PB._encoded_size(x.xla_cpu_use_acl, 174))
    x.xla_cpu_strict_dot_conv_math != false && (encoded_size += PB._encoded_size(x.xla_cpu_strict_dot_conv_math, 175))
    x.xla_dump_latency_hiding_schedule != false && (encoded_size += PB._encoded_size(x.xla_dump_latency_hiding_schedule, 182))
    x.xla_partitioning_algorithm != var"DebugOptions.PartitioningAlgorithm".PARTITIONING_ALGORITHM_NOOP && (encoded_size += PB._encoded_size(x.xla_partitioning_algorithm, 187))
    x.xla_debug_buffer_assignment_show_max != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_debug_buffer_assignment_show_max, 251))
    x.xla_detect_unstable_reductions != var"DebugOptions.DetectionMode".DETECTION_MODE_NONE && (encoded_size += PB._encoded_size(x.xla_detect_unstable_reductions, 403))
    x.xla_detect_unstable_reductions_post_optimizations != var"DebugOptions.DetectionMode".DETECTION_MODE_NONE && (encoded_size += PB._encoded_size(x.xla_detect_unstable_reductions_post_optimizations, 432))
    x.xla_gpu_detect_nan != var"DebugOptions.DetectionMode".DETECTION_MODE_NONE && (encoded_size += PB._encoded_size(x.xla_gpu_detect_nan, 426))
    x.xla_gpu_detect_inf != var"DebugOptions.DetectionMode".DETECTION_MODE_NONE && (encoded_size += PB._encoded_size(x.xla_gpu_detect_inf, 428))
    x.xla_dump_large_constants != false && (encoded_size += PB._encoded_size(x.xla_dump_large_constants, 290))
    x.xla_reduce_window_rewrite_base_length != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_reduce_window_rewrite_base_length, 293))
    x.xla_cmd_buffer_trace_cache_size != zero(Int64) && (encoded_size += PB._encoded_size(x.xla_cmd_buffer_trace_cache_size, 311))
    x.xla_syntax_sugar_async_ops != false && (encoded_size += PB._encoded_size(x.xla_syntax_sugar_async_ops, 315))
    x.xla_enable_command_buffers_during_profiling != false && (encoded_size += PB._encoded_size(x.xla_enable_command_buffers_during_profiling, 317))
    x.xla_ignore_channel_id != false && (encoded_size += PB._encoded_size(x.xla_ignore_channel_id, 330))
    x.xla_pjrt_allow_auto_layout_in_hlo != false && (encoded_size += PB._encoded_size(x.xla_pjrt_allow_auto_layout_in_hlo, 344))
    x.xla_test_add_command_buffer_mode != false && (encoded_size += PB._encoded_size(x.xla_test_add_command_buffer_mode, 373))
    !isempty(x.xla_gpu_experimental_matmul_perf_table_path) && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_matmul_perf_table_path, 383))
    x.xla_early_exit_with_layouts != false && (encoded_size += PB._encoded_size(x.xla_early_exit_with_layouts, 397))
    x.xla_gpu_experimental_scaled_dot_with_triton != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_scaled_dot_with_triton, 410))
    x.xla_gpu_experimental_use_raft_select_k != false && (encoded_size += PB._encoded_size(x.xla_gpu_experimental_use_raft_select_k, 413))
    !isempty(x.xla_backend_extra_options) && (encoded_size += PB._encoded_size(x.xla_backend_extra_options, 500))
    return encoded_size
end

struct HloModuleConfigProto
    entry_computation_layout::Union{Nothing,ProgramShapeProto}
    seed::UInt64
    launch_id::Int32
    replica_count::Int64
    num_partitions::Int64
    param_requires_broadcast_via_collectives::Vector{Bool}
    use_spmd_partitioning::Bool
    use_auto_spmd_partitioning::Bool
    auto_spmd_partitioning_mesh_shape::Vector{Int64}
    auto_spmd_partitioning_mesh_ids::Vector{Int64}
    exec_time_optimization_effort::Float32
    memory_fitting_effort::Float32
    optimization_level::var"ExecutionOptions.EffortLevel".T
    memory_fitting_level::var"ExecutionOptions.EffortLevel".T
    deduplicate_hlo::Bool
    intra_op_parallelism_threads::Int64
    device_type::String
    debug_options::Union{Nothing,DebugOptions}
    static_device_assignment::Union{Nothing,DeviceAssignmentProto}
    pre_simulation_device_assignment::Union{Nothing,DeviceAssignmentProto}
    allow_separate_sharding_programs::Bool
    shardable_value_update_pairs::Vector{ShardableValueUpdatePairProto}
    alias_passthrough_params::Bool
    content_aware_computation_sorting::Bool
    fusion_config_collection::var"HloModuleConfigProto.FusionConfigCollection".T
    fusion_config::Vector{var"HloModuleConfigProto.BoolList"}
    dot_config::Dict{String,var"HloModuleConfigProto.Int64List"}
    layout_config::Vector{var"HloModuleConfigProto.Int64ListList"}
    memory_space_assignment_config::Vector{UInt64}
    phase_ordering_config::Vector{var"HloModuleConfigProto.BoolList"}
    phase_index::Int32
    allow_spmd_sharding_propagation_to_parameters::Vector{Bool}
    allow_spmd_sharding_propagation_to_output::Vector{Bool}
    analysis_allowance_map::Dict{String,Int64}
    matrix_unit_operand_precision::var"PrecisionConfig.Precision".T
    fdo_profile::Vector{UInt8}
    device_memory_size::Int64
    use_shardy_partitioner::Bool
    sharding_config::Union{Nothing,ShardingConfigProto}
    schedule_config::Union{Nothing,ScheduleConfigProto}
    partition_size::Int64
end
HloModuleConfigProto(;entry_computation_layout = nothing, seed = zero(UInt64), launch_id = zero(Int32), replica_count = zero(Int64), num_partitions = zero(Int64), param_requires_broadcast_via_collectives = Vector{Bool}(), use_spmd_partitioning = false, use_auto_spmd_partitioning = false, auto_spmd_partitioning_mesh_shape = Vector{Int64}(), auto_spmd_partitioning_mesh_ids = Vector{Int64}(), exec_time_optimization_effort = zero(Float32), memory_fitting_effort = zero(Float32), optimization_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, memory_fitting_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, deduplicate_hlo = false, intra_op_parallelism_threads = zero(Int64), device_type = "", debug_options = nothing, static_device_assignment = nothing, pre_simulation_device_assignment = nothing, allow_separate_sharding_programs = false, shardable_value_update_pairs = Vector{ShardableValueUpdatePairProto}(), alias_passthrough_params = false, content_aware_computation_sorting = false, fusion_config_collection = var"HloModuleConfigProto.FusionConfigCollection".OFF, fusion_config = Vector{var"HloModuleConfigProto.BoolList"}(), dot_config = Dict{String,var"HloModuleConfigProto.Int64List"}(), layout_config = Vector{var"HloModuleConfigProto.Int64ListList"}(), memory_space_assignment_config = Vector{UInt64}(), phase_ordering_config = Vector{var"HloModuleConfigProto.BoolList"}(), phase_index = zero(Int32), allow_spmd_sharding_propagation_to_parameters = Vector{Bool}(), allow_spmd_sharding_propagation_to_output = Vector{Bool}(), analysis_allowance_map = Dict{String,Int64}(), matrix_unit_operand_precision = var"PrecisionConfig.Precision".DEFAULT, fdo_profile = UInt8[], device_memory_size = zero(Int64), use_shardy_partitioner = false, sharding_config = nothing, schedule_config = nothing, partition_size = zero(Int64)) = HloModuleConfigProto(entry_computation_layout, seed, launch_id, replica_count, num_partitions, param_requires_broadcast_via_collectives, use_spmd_partitioning, use_auto_spmd_partitioning, auto_spmd_partitioning_mesh_shape, auto_spmd_partitioning_mesh_ids, exec_time_optimization_effort, memory_fitting_effort, optimization_level, memory_fitting_level, deduplicate_hlo, intra_op_parallelism_threads, device_type, debug_options, static_device_assignment, pre_simulation_device_assignment, allow_separate_sharding_programs, shardable_value_update_pairs, alias_passthrough_params, content_aware_computation_sorting, fusion_config_collection, fusion_config, dot_config, layout_config, memory_space_assignment_config, phase_ordering_config, phase_index, allow_spmd_sharding_propagation_to_parameters, allow_spmd_sharding_propagation_to_output, analysis_allowance_map, matrix_unit_operand_precision, fdo_profile, device_memory_size, use_shardy_partitioner, sharding_config, schedule_config, partition_size)
PB.reserved_fields(::Type{HloModuleConfigProto}) = (names = ["flag_config"], numbers = Union{Int,UnitRange{Int}}[26])
PB.default_values(::Type{HloModuleConfigProto}) = (;entry_computation_layout = nothing, seed = zero(UInt64), launch_id = zero(Int32), replica_count = zero(Int64), num_partitions = zero(Int64), param_requires_broadcast_via_collectives = Vector{Bool}(), use_spmd_partitioning = false, use_auto_spmd_partitioning = false, auto_spmd_partitioning_mesh_shape = Vector{Int64}(), auto_spmd_partitioning_mesh_ids = Vector{Int64}(), exec_time_optimization_effort = zero(Float32), memory_fitting_effort = zero(Float32), optimization_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, memory_fitting_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, deduplicate_hlo = false, intra_op_parallelism_threads = zero(Int64), device_type = "", debug_options = nothing, static_device_assignment = nothing, pre_simulation_device_assignment = nothing, allow_separate_sharding_programs = false, shardable_value_update_pairs = Vector{ShardableValueUpdatePairProto}(), alias_passthrough_params = false, content_aware_computation_sorting = false, fusion_config_collection = var"HloModuleConfigProto.FusionConfigCollection".OFF, fusion_config = Vector{var"HloModuleConfigProto.BoolList"}(), dot_config = Dict{String,var"HloModuleConfigProto.Int64List"}(), layout_config = Vector{var"HloModuleConfigProto.Int64ListList"}(), memory_space_assignment_config = Vector{UInt64}(), phase_ordering_config = Vector{var"HloModuleConfigProto.BoolList"}(), phase_index = zero(Int32), allow_spmd_sharding_propagation_to_parameters = Vector{Bool}(), allow_spmd_sharding_propagation_to_output = Vector{Bool}(), analysis_allowance_map = Dict{String,Int64}(), matrix_unit_operand_precision = var"PrecisionConfig.Precision".DEFAULT, fdo_profile = UInt8[], device_memory_size = zero(Int64), use_shardy_partitioner = false, sharding_config = nothing, schedule_config = nothing, partition_size = zero(Int64))
PB.field_numbers(::Type{HloModuleConfigProto}) = (;entry_computation_layout = 1, seed = 2, launch_id = 3, replica_count = 4, num_partitions = 5, param_requires_broadcast_via_collectives = 6, use_spmd_partitioning = 7, use_auto_spmd_partitioning = 8, auto_spmd_partitioning_mesh_shape = 9, auto_spmd_partitioning_mesh_ids = 10, exec_time_optimization_effort = 36, memory_fitting_effort = 37, optimization_level = 39, memory_fitting_level = 40, deduplicate_hlo = 11, intra_op_parallelism_threads = 12, device_type = 13, debug_options = 14, static_device_assignment = 15, pre_simulation_device_assignment = 35, allow_separate_sharding_programs = 30, shardable_value_update_pairs = 16, alias_passthrough_params = 17, content_aware_computation_sorting = 18, fusion_config_collection = 19, fusion_config = 20, dot_config = 21, layout_config = 22, memory_space_assignment_config = 23, phase_ordering_config = 24, phase_index = 25, allow_spmd_sharding_propagation_to_parameters = 33, allow_spmd_sharding_propagation_to_output = 27, analysis_allowance_map = 28, matrix_unit_operand_precision = 29, fdo_profile = 31, device_memory_size = 32, use_shardy_partitioner = 34, sharding_config = 38, schedule_config = 41, partition_size = 42)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloModuleConfigProto})
    entry_computation_layout = Ref{Union{Nothing,ProgramShapeProto}}(nothing)
    seed = zero(UInt64)
    launch_id = zero(Int32)
    replica_count = zero(Int64)
    num_partitions = zero(Int64)
    param_requires_broadcast_via_collectives = PB.BufferedVector{Bool}()
    use_spmd_partitioning = false
    use_auto_spmd_partitioning = false
    auto_spmd_partitioning_mesh_shape = PB.BufferedVector{Int64}()
    auto_spmd_partitioning_mesh_ids = PB.BufferedVector{Int64}()
    exec_time_optimization_effort = zero(Float32)
    memory_fitting_effort = zero(Float32)
    optimization_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN
    memory_fitting_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN
    deduplicate_hlo = false
    intra_op_parallelism_threads = zero(Int64)
    device_type = ""
    debug_options = Ref{Union{Nothing,DebugOptions}}(nothing)
    static_device_assignment = Ref{Union{Nothing,DeviceAssignmentProto}}(nothing)
    pre_simulation_device_assignment = Ref{Union{Nothing,DeviceAssignmentProto}}(nothing)
    allow_separate_sharding_programs = false
    shardable_value_update_pairs = PB.BufferedVector{ShardableValueUpdatePairProto}()
    alias_passthrough_params = false
    content_aware_computation_sorting = false
    fusion_config_collection = var"HloModuleConfigProto.FusionConfigCollection".OFF
    fusion_config = PB.BufferedVector{var"HloModuleConfigProto.BoolList"}()
    dot_config = Dict{String,var"HloModuleConfigProto.Int64List"}()
    layout_config = PB.BufferedVector{var"HloModuleConfigProto.Int64ListList"}()
    memory_space_assignment_config = PB.BufferedVector{UInt64}()
    phase_ordering_config = PB.BufferedVector{var"HloModuleConfigProto.BoolList"}()
    phase_index = zero(Int32)
    allow_spmd_sharding_propagation_to_parameters = PB.BufferedVector{Bool}()
    allow_spmd_sharding_propagation_to_output = PB.BufferedVector{Bool}()
    analysis_allowance_map = Dict{String,Int64}()
    matrix_unit_operand_precision = var"PrecisionConfig.Precision".DEFAULT
    fdo_profile = UInt8[]
    device_memory_size = zero(Int64)
    use_shardy_partitioner = false
    sharding_config = Ref{Union{Nothing,ShardingConfigProto}}(nothing)
    schedule_config = Ref{Union{Nothing,ScheduleConfigProto}}(nothing)
    partition_size = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, entry_computation_layout)
        elseif field_number == 2
            seed = PB.decode(d, UInt64)
        elseif field_number == 3
            launch_id = PB.decode(d, Int32)
        elseif field_number == 4
            replica_count = PB.decode(d, Int64)
        elseif field_number == 5
            num_partitions = PB.decode(d, Int64)
        elseif field_number == 6
            PB.decode!(d, wire_type, param_requires_broadcast_via_collectives)
        elseif field_number == 7
            use_spmd_partitioning = PB.decode(d, Bool)
        elseif field_number == 8
            use_auto_spmd_partitioning = PB.decode(d, Bool)
        elseif field_number == 9
            PB.decode!(d, wire_type, auto_spmd_partitioning_mesh_shape)
        elseif field_number == 10
            PB.decode!(d, wire_type, auto_spmd_partitioning_mesh_ids)
        elseif field_number == 36
            exec_time_optimization_effort = PB.decode(d, Float32)
        elseif field_number == 37
            memory_fitting_effort = PB.decode(d, Float32)
        elseif field_number == 39
            optimization_level = PB.decode(d, var"ExecutionOptions.EffortLevel".T)
        elseif field_number == 40
            memory_fitting_level = PB.decode(d, var"ExecutionOptions.EffortLevel".T)
        elseif field_number == 11
            deduplicate_hlo = PB.decode(d, Bool)
        elseif field_number == 12
            intra_op_parallelism_threads = PB.decode(d, Int64)
        elseif field_number == 13
            device_type = PB.decode(d, String)
        elseif field_number == 14
            PB.decode!(d, debug_options)
        elseif field_number == 15
            PB.decode!(d, static_device_assignment)
        elseif field_number == 35
            PB.decode!(d, pre_simulation_device_assignment)
        elseif field_number == 30
            allow_separate_sharding_programs = PB.decode(d, Bool)
        elseif field_number == 16
            PB.decode!(d, shardable_value_update_pairs)
        elseif field_number == 17
            alias_passthrough_params = PB.decode(d, Bool)
        elseif field_number == 18
            content_aware_computation_sorting = PB.decode(d, Bool)
        elseif field_number == 19
            fusion_config_collection = PB.decode(d, var"HloModuleConfigProto.FusionConfigCollection".T)
        elseif field_number == 20
            PB.decode!(d, fusion_config)
        elseif field_number == 21
            PB.decode!(d, dot_config)
        elseif field_number == 22
            PB.decode!(d, layout_config)
        elseif field_number == 23
            PB.decode!(d, wire_type, memory_space_assignment_config)
        elseif field_number == 24
            PB.decode!(d, phase_ordering_config)
        elseif field_number == 25
            phase_index = PB.decode(d, Int32)
        elseif field_number == 33
            PB.decode!(d, wire_type, allow_spmd_sharding_propagation_to_parameters)
        elseif field_number == 27
            PB.decode!(d, wire_type, allow_spmd_sharding_propagation_to_output)
        elseif field_number == 28
            PB.decode!(d, analysis_allowance_map)
        elseif field_number == 29
            matrix_unit_operand_precision = PB.decode(d, var"PrecisionConfig.Precision".T)
        elseif field_number == 31
            fdo_profile = PB.decode(d, Vector{UInt8})
        elseif field_number == 32
            device_memory_size = PB.decode(d, Int64)
        elseif field_number == 34
            use_shardy_partitioner = PB.decode(d, Bool)
        elseif field_number == 38
            PB.decode!(d, sharding_config)
        elseif field_number == 41
            PB.decode!(d, schedule_config)
        elseif field_number == 42
            partition_size = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloModuleConfigProto(entry_computation_layout[], seed, launch_id, replica_count, num_partitions, param_requires_broadcast_via_collectives[], use_spmd_partitioning, use_auto_spmd_partitioning, auto_spmd_partitioning_mesh_shape[], auto_spmd_partitioning_mesh_ids[], exec_time_optimization_effort, memory_fitting_effort, optimization_level, memory_fitting_level, deduplicate_hlo, intra_op_parallelism_threads, device_type, debug_options[], static_device_assignment[], pre_simulation_device_assignment[], allow_separate_sharding_programs, shardable_value_update_pairs[], alias_passthrough_params, content_aware_computation_sorting, fusion_config_collection, fusion_config[], dot_config, layout_config[], memory_space_assignment_config[], phase_ordering_config[], phase_index, allow_spmd_sharding_propagation_to_parameters[], allow_spmd_sharding_propagation_to_output[], analysis_allowance_map, matrix_unit_operand_precision, fdo_profile, device_memory_size, use_shardy_partitioner, sharding_config[], schedule_config[], partition_size)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloModuleConfigProto)
    initpos = position(e.io)
    !isnothing(x.entry_computation_layout) && PB.encode(e, 1, x.entry_computation_layout)
    x.seed != zero(UInt64) && PB.encode(e, 2, x.seed)
    x.launch_id != zero(Int32) && PB.encode(e, 3, x.launch_id)
    x.replica_count != zero(Int64) && PB.encode(e, 4, x.replica_count)
    x.num_partitions != zero(Int64) && PB.encode(e, 5, x.num_partitions)
    !isempty(x.param_requires_broadcast_via_collectives) && PB.encode(e, 6, x.param_requires_broadcast_via_collectives)
    x.use_spmd_partitioning != false && PB.encode(e, 7, x.use_spmd_partitioning)
    x.use_auto_spmd_partitioning != false && PB.encode(e, 8, x.use_auto_spmd_partitioning)
    !isempty(x.auto_spmd_partitioning_mesh_shape) && PB.encode(e, 9, x.auto_spmd_partitioning_mesh_shape)
    !isempty(x.auto_spmd_partitioning_mesh_ids) && PB.encode(e, 10, x.auto_spmd_partitioning_mesh_ids)
    x.exec_time_optimization_effort !== zero(Float32) && PB.encode(e, 36, x.exec_time_optimization_effort)
    x.memory_fitting_effort !== zero(Float32) && PB.encode(e, 37, x.memory_fitting_effort)
    x.optimization_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && PB.encode(e, 39, x.optimization_level)
    x.memory_fitting_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && PB.encode(e, 40, x.memory_fitting_level)
    x.deduplicate_hlo != false && PB.encode(e, 11, x.deduplicate_hlo)
    x.intra_op_parallelism_threads != zero(Int64) && PB.encode(e, 12, x.intra_op_parallelism_threads)
    !isempty(x.device_type) && PB.encode(e, 13, x.device_type)
    !isnothing(x.debug_options) && PB.encode(e, 14, x.debug_options)
    !isnothing(x.static_device_assignment) && PB.encode(e, 15, x.static_device_assignment)
    !isnothing(x.pre_simulation_device_assignment) && PB.encode(e, 35, x.pre_simulation_device_assignment)
    x.allow_separate_sharding_programs != false && PB.encode(e, 30, x.allow_separate_sharding_programs)
    !isempty(x.shardable_value_update_pairs) && PB.encode(e, 16, x.shardable_value_update_pairs)
    x.alias_passthrough_params != false && PB.encode(e, 17, x.alias_passthrough_params)
    x.content_aware_computation_sorting != false && PB.encode(e, 18, x.content_aware_computation_sorting)
    x.fusion_config_collection != var"HloModuleConfigProto.FusionConfigCollection".OFF && PB.encode(e, 19, x.fusion_config_collection)
    !isempty(x.fusion_config) && PB.encode(e, 20, x.fusion_config)
    !isempty(x.dot_config) && PB.encode(e, 21, x.dot_config)
    !isempty(x.layout_config) && PB.encode(e, 22, x.layout_config)
    !isempty(x.memory_space_assignment_config) && PB.encode(e, 23, x.memory_space_assignment_config)
    !isempty(x.phase_ordering_config) && PB.encode(e, 24, x.phase_ordering_config)
    x.phase_index != zero(Int32) && PB.encode(e, 25, x.phase_index)
    !isempty(x.allow_spmd_sharding_propagation_to_parameters) && PB.encode(e, 33, x.allow_spmd_sharding_propagation_to_parameters)
    !isempty(x.allow_spmd_sharding_propagation_to_output) && PB.encode(e, 27, x.allow_spmd_sharding_propagation_to_output)
    !isempty(x.analysis_allowance_map) && PB.encode(e, 28, x.analysis_allowance_map)
    x.matrix_unit_operand_precision != var"PrecisionConfig.Precision".DEFAULT && PB.encode(e, 29, x.matrix_unit_operand_precision)
    !isempty(x.fdo_profile) && PB.encode(e, 31, x.fdo_profile)
    x.device_memory_size != zero(Int64) && PB.encode(e, 32, x.device_memory_size)
    x.use_shardy_partitioner != false && PB.encode(e, 34, x.use_shardy_partitioner)
    !isnothing(x.sharding_config) && PB.encode(e, 38, x.sharding_config)
    !isnothing(x.schedule_config) && PB.encode(e, 41, x.schedule_config)
    x.partition_size != zero(Int64) && PB.encode(e, 42, x.partition_size)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloModuleConfigProto)
    encoded_size = 0
    !isnothing(x.entry_computation_layout) && (encoded_size += PB._encoded_size(x.entry_computation_layout, 1))
    x.seed != zero(UInt64) && (encoded_size += PB._encoded_size(x.seed, 2))
    x.launch_id != zero(Int32) && (encoded_size += PB._encoded_size(x.launch_id, 3))
    x.replica_count != zero(Int64) && (encoded_size += PB._encoded_size(x.replica_count, 4))
    x.num_partitions != zero(Int64) && (encoded_size += PB._encoded_size(x.num_partitions, 5))
    !isempty(x.param_requires_broadcast_via_collectives) && (encoded_size += PB._encoded_size(x.param_requires_broadcast_via_collectives, 6))
    x.use_spmd_partitioning != false && (encoded_size += PB._encoded_size(x.use_spmd_partitioning, 7))
    x.use_auto_spmd_partitioning != false && (encoded_size += PB._encoded_size(x.use_auto_spmd_partitioning, 8))
    !isempty(x.auto_spmd_partitioning_mesh_shape) && (encoded_size += PB._encoded_size(x.auto_spmd_partitioning_mesh_shape, 9))
    !isempty(x.auto_spmd_partitioning_mesh_ids) && (encoded_size += PB._encoded_size(x.auto_spmd_partitioning_mesh_ids, 10))
    x.exec_time_optimization_effort !== zero(Float32) && (encoded_size += PB._encoded_size(x.exec_time_optimization_effort, 36))
    x.memory_fitting_effort !== zero(Float32) && (encoded_size += PB._encoded_size(x.memory_fitting_effort, 37))
    x.optimization_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && (encoded_size += PB._encoded_size(x.optimization_level, 39))
    x.memory_fitting_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && (encoded_size += PB._encoded_size(x.memory_fitting_level, 40))
    x.deduplicate_hlo != false && (encoded_size += PB._encoded_size(x.deduplicate_hlo, 11))
    x.intra_op_parallelism_threads != zero(Int64) && (encoded_size += PB._encoded_size(x.intra_op_parallelism_threads, 12))
    !isempty(x.device_type) && (encoded_size += PB._encoded_size(x.device_type, 13))
    !isnothing(x.debug_options) && (encoded_size += PB._encoded_size(x.debug_options, 14))
    !isnothing(x.static_device_assignment) && (encoded_size += PB._encoded_size(x.static_device_assignment, 15))
    !isnothing(x.pre_simulation_device_assignment) && (encoded_size += PB._encoded_size(x.pre_simulation_device_assignment, 35))
    x.allow_separate_sharding_programs != false && (encoded_size += PB._encoded_size(x.allow_separate_sharding_programs, 30))
    !isempty(x.shardable_value_update_pairs) && (encoded_size += PB._encoded_size(x.shardable_value_update_pairs, 16))
    x.alias_passthrough_params != false && (encoded_size += PB._encoded_size(x.alias_passthrough_params, 17))
    x.content_aware_computation_sorting != false && (encoded_size += PB._encoded_size(x.content_aware_computation_sorting, 18))
    x.fusion_config_collection != var"HloModuleConfigProto.FusionConfigCollection".OFF && (encoded_size += PB._encoded_size(x.fusion_config_collection, 19))
    !isempty(x.fusion_config) && (encoded_size += PB._encoded_size(x.fusion_config, 20))
    !isempty(x.dot_config) && (encoded_size += PB._encoded_size(x.dot_config, 21))
    !isempty(x.layout_config) && (encoded_size += PB._encoded_size(x.layout_config, 22))
    !isempty(x.memory_space_assignment_config) && (encoded_size += PB._encoded_size(x.memory_space_assignment_config, 23))
    !isempty(x.phase_ordering_config) && (encoded_size += PB._encoded_size(x.phase_ordering_config, 24))
    x.phase_index != zero(Int32) && (encoded_size += PB._encoded_size(x.phase_index, 25))
    !isempty(x.allow_spmd_sharding_propagation_to_parameters) && (encoded_size += PB._encoded_size(x.allow_spmd_sharding_propagation_to_parameters, 33))
    !isempty(x.allow_spmd_sharding_propagation_to_output) && (encoded_size += PB._encoded_size(x.allow_spmd_sharding_propagation_to_output, 27))
    !isempty(x.analysis_allowance_map) && (encoded_size += PB._encoded_size(x.analysis_allowance_map, 28))
    x.matrix_unit_operand_precision != var"PrecisionConfig.Precision".DEFAULT && (encoded_size += PB._encoded_size(x.matrix_unit_operand_precision, 29))
    !isempty(x.fdo_profile) && (encoded_size += PB._encoded_size(x.fdo_profile, 31))
    x.device_memory_size != zero(Int64) && (encoded_size += PB._encoded_size(x.device_memory_size, 32))
    x.use_shardy_partitioner != false && (encoded_size += PB._encoded_size(x.use_shardy_partitioner, 34))
    !isnothing(x.sharding_config) && (encoded_size += PB._encoded_size(x.sharding_config, 38))
    !isnothing(x.schedule_config) && (encoded_size += PB._encoded_size(x.schedule_config, 41))
    x.partition_size != zero(Int64) && (encoded_size += PB._encoded_size(x.partition_size, 42))
    return encoded_size
end

struct ExecutionOptions
    shape_with_output_layout::Union{Nothing,ShapeProto}
    seed::UInt64
    debug_options::Union{Nothing,DebugOptions}
    device_handles::Vector{DeviceHandle}
    num_replicas::Int32
    device_assignment::Union{Nothing,DeviceAssignmentProto}
    alias_passthrough_params::Bool
    num_partitions::Int32
    launch_id::Int32
    use_spmd_partitioning::Bool
    use_auto_spmd_partitioning::Bool
    auto_spmd_partitioning_mesh_shape::Vector{Int64}
    auto_spmd_partitioning_mesh_ids::Vector{Int64}
    exec_time_optimization_effort::Float32
    memory_fitting_effort::Float32
    optimization_level::var"ExecutionOptions.EffortLevel".T
    memory_fitting_level::var"ExecutionOptions.EffortLevel".T
    deduplicate_hlo::Bool
    allow_spmd_sharding_propagation_to_parameters::Vector{Bool}
    allow_spmd_sharding_propagation_to_output::Vector{Bool}
    param_requires_broadcast_via_collectives::Vector{Bool}
    allow_separate_sharding_programs::Bool
    shardable_value_update_pairs::Vector{ShardableValueUpdatePairProto}
    fdo_profile::Vector{UInt8}
    device_memory_size::Int64
    use_shardy_partitioner::Bool
end
ExecutionOptions(;shape_with_output_layout = nothing, seed = zero(UInt64), debug_options = nothing, device_handles = Vector{DeviceHandle}(), num_replicas = zero(Int32), device_assignment = nothing, alias_passthrough_params = false, num_partitions = zero(Int32), launch_id = zero(Int32), use_spmd_partitioning = false, use_auto_spmd_partitioning = false, auto_spmd_partitioning_mesh_shape = Vector{Int64}(), auto_spmd_partitioning_mesh_ids = Vector{Int64}(), exec_time_optimization_effort = zero(Float32), memory_fitting_effort = zero(Float32), optimization_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, memory_fitting_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, deduplicate_hlo = false, allow_spmd_sharding_propagation_to_parameters = Vector{Bool}(), allow_spmd_sharding_propagation_to_output = Vector{Bool}(), param_requires_broadcast_via_collectives = Vector{Bool}(), allow_separate_sharding_programs = false, shardable_value_update_pairs = Vector{ShardableValueUpdatePairProto}(), fdo_profile = UInt8[], device_memory_size = zero(Int64), use_shardy_partitioner = false) = ExecutionOptions(shape_with_output_layout, seed, debug_options, device_handles, num_replicas, device_assignment, alias_passthrough_params, num_partitions, launch_id, use_spmd_partitioning, use_auto_spmd_partitioning, auto_spmd_partitioning_mesh_shape, auto_spmd_partitioning_mesh_ids, exec_time_optimization_effort, memory_fitting_effort, optimization_level, memory_fitting_level, deduplicate_hlo, allow_spmd_sharding_propagation_to_parameters, allow_spmd_sharding_propagation_to_output, param_requires_broadcast_via_collectives, allow_separate_sharding_programs, shardable_value_update_pairs, fdo_profile, device_memory_size, use_shardy_partitioner)
PB.reserved_fields(::Type{ExecutionOptions}) = (names = ["broadcast_replicated_parameters_via_collectives"], numbers = Union{Int,UnitRange{Int}}[13])
PB.default_values(::Type{ExecutionOptions}) = (;shape_with_output_layout = nothing, seed = zero(UInt64), debug_options = nothing, device_handles = Vector{DeviceHandle}(), num_replicas = zero(Int32), device_assignment = nothing, alias_passthrough_params = false, num_partitions = zero(Int32), launch_id = zero(Int32), use_spmd_partitioning = false, use_auto_spmd_partitioning = false, auto_spmd_partitioning_mesh_shape = Vector{Int64}(), auto_spmd_partitioning_mesh_ids = Vector{Int64}(), exec_time_optimization_effort = zero(Float32), memory_fitting_effort = zero(Float32), optimization_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, memory_fitting_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, deduplicate_hlo = false, allow_spmd_sharding_propagation_to_parameters = Vector{Bool}(), allow_spmd_sharding_propagation_to_output = Vector{Bool}(), param_requires_broadcast_via_collectives = Vector{Bool}(), allow_separate_sharding_programs = false, shardable_value_update_pairs = Vector{ShardableValueUpdatePairProto}(), fdo_profile = UInt8[], device_memory_size = zero(Int64), use_shardy_partitioner = false)
PB.field_numbers(::Type{ExecutionOptions}) = (;shape_with_output_layout = 2, seed = 3, debug_options = 4, device_handles = 5, num_replicas = 6, device_assignment = 7, alias_passthrough_params = 8, num_partitions = 9, launch_id = 10, use_spmd_partitioning = 11, use_auto_spmd_partitioning = 15, auto_spmd_partitioning_mesh_shape = 16, auto_spmd_partitioning_mesh_ids = 17, exec_time_optimization_effort = 25, memory_fitting_effort = 26, optimization_level = 27, memory_fitting_level = 28, deduplicate_hlo = 12, allow_spmd_sharding_propagation_to_parameters = 23, allow_spmd_sharding_propagation_to_output = 14, param_requires_broadcast_via_collectives = 18, allow_separate_sharding_programs = 19, shardable_value_update_pairs = 20, fdo_profile = 21, device_memory_size = 22, use_shardy_partitioner = 24)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ExecutionOptions})
    shape_with_output_layout = Ref{Union{Nothing,ShapeProto}}(nothing)
    seed = zero(UInt64)
    debug_options = Ref{Union{Nothing,DebugOptions}}(nothing)
    device_handles = PB.BufferedVector{DeviceHandle}()
    num_replicas = zero(Int32)
    device_assignment = Ref{Union{Nothing,DeviceAssignmentProto}}(nothing)
    alias_passthrough_params = false
    num_partitions = zero(Int32)
    launch_id = zero(Int32)
    use_spmd_partitioning = false
    use_auto_spmd_partitioning = false
    auto_spmd_partitioning_mesh_shape = PB.BufferedVector{Int64}()
    auto_spmd_partitioning_mesh_ids = PB.BufferedVector{Int64}()
    exec_time_optimization_effort = zero(Float32)
    memory_fitting_effort = zero(Float32)
    optimization_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN
    memory_fitting_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN
    deduplicate_hlo = false
    allow_spmd_sharding_propagation_to_parameters = PB.BufferedVector{Bool}()
    allow_spmd_sharding_propagation_to_output = PB.BufferedVector{Bool}()
    param_requires_broadcast_via_collectives = PB.BufferedVector{Bool}()
    allow_separate_sharding_programs = false
    shardable_value_update_pairs = PB.BufferedVector{ShardableValueUpdatePairProto}()
    fdo_profile = UInt8[]
    device_memory_size = zero(Int64)
    use_shardy_partitioner = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 2
            PB.decode!(d, shape_with_output_layout)
        elseif field_number == 3
            seed = PB.decode(d, UInt64)
        elseif field_number == 4
            PB.decode!(d, debug_options)
        elseif field_number == 5
            PB.decode!(d, device_handles)
        elseif field_number == 6
            num_replicas = PB.decode(d, Int32)
        elseif field_number == 7
            PB.decode!(d, device_assignment)
        elseif field_number == 8
            alias_passthrough_params = PB.decode(d, Bool)
        elseif field_number == 9
            num_partitions = PB.decode(d, Int32)
        elseif field_number == 10
            launch_id = PB.decode(d, Int32)
        elseif field_number == 11
            use_spmd_partitioning = PB.decode(d, Bool)
        elseif field_number == 15
            use_auto_spmd_partitioning = PB.decode(d, Bool)
        elseif field_number == 16
            PB.decode!(d, wire_type, auto_spmd_partitioning_mesh_shape)
        elseif field_number == 17
            PB.decode!(d, wire_type, auto_spmd_partitioning_mesh_ids)
        elseif field_number == 25
            exec_time_optimization_effort = PB.decode(d, Float32)
        elseif field_number == 26
            memory_fitting_effort = PB.decode(d, Float32)
        elseif field_number == 27
            optimization_level = PB.decode(d, var"ExecutionOptions.EffortLevel".T)
        elseif field_number == 28
            memory_fitting_level = PB.decode(d, var"ExecutionOptions.EffortLevel".T)
        elseif field_number == 12
            deduplicate_hlo = PB.decode(d, Bool)
        elseif field_number == 23
            PB.decode!(d, wire_type, allow_spmd_sharding_propagation_to_parameters)
        elseif field_number == 14
            PB.decode!(d, wire_type, allow_spmd_sharding_propagation_to_output)
        elseif field_number == 18
            PB.decode!(d, wire_type, param_requires_broadcast_via_collectives)
        elseif field_number == 19
            allow_separate_sharding_programs = PB.decode(d, Bool)
        elseif field_number == 20
            PB.decode!(d, shardable_value_update_pairs)
        elseif field_number == 21
            fdo_profile = PB.decode(d, Vector{UInt8})
        elseif field_number == 22
            device_memory_size = PB.decode(d, Int64)
        elseif field_number == 24
            use_shardy_partitioner = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return ExecutionOptions(shape_with_output_layout[], seed, debug_options[], device_handles[], num_replicas, device_assignment[], alias_passthrough_params, num_partitions, launch_id, use_spmd_partitioning, use_auto_spmd_partitioning, auto_spmd_partitioning_mesh_shape[], auto_spmd_partitioning_mesh_ids[], exec_time_optimization_effort, memory_fitting_effort, optimization_level, memory_fitting_level, deduplicate_hlo, allow_spmd_sharding_propagation_to_parameters[], allow_spmd_sharding_propagation_to_output[], param_requires_broadcast_via_collectives[], allow_separate_sharding_programs, shardable_value_update_pairs[], fdo_profile, device_memory_size, use_shardy_partitioner)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ExecutionOptions)
    initpos = position(e.io)
    !isnothing(x.shape_with_output_layout) && PB.encode(e, 2, x.shape_with_output_layout)
    x.seed != zero(UInt64) && PB.encode(e, 3, x.seed)
    !isnothing(x.debug_options) && PB.encode(e, 4, x.debug_options)
    !isempty(x.device_handles) && PB.encode(e, 5, x.device_handles)
    x.num_replicas != zero(Int32) && PB.encode(e, 6, x.num_replicas)
    !isnothing(x.device_assignment) && PB.encode(e, 7, x.device_assignment)
    x.alias_passthrough_params != false && PB.encode(e, 8, x.alias_passthrough_params)
    x.num_partitions != zero(Int32) && PB.encode(e, 9, x.num_partitions)
    x.launch_id != zero(Int32) && PB.encode(e, 10, x.launch_id)
    x.use_spmd_partitioning != false && PB.encode(e, 11, x.use_spmd_partitioning)
    x.use_auto_spmd_partitioning != false && PB.encode(e, 15, x.use_auto_spmd_partitioning)
    !isempty(x.auto_spmd_partitioning_mesh_shape) && PB.encode(e, 16, x.auto_spmd_partitioning_mesh_shape)
    !isempty(x.auto_spmd_partitioning_mesh_ids) && PB.encode(e, 17, x.auto_spmd_partitioning_mesh_ids)
    x.exec_time_optimization_effort !== zero(Float32) && PB.encode(e, 25, x.exec_time_optimization_effort)
    x.memory_fitting_effort !== zero(Float32) && PB.encode(e, 26, x.memory_fitting_effort)
    x.optimization_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && PB.encode(e, 27, x.optimization_level)
    x.memory_fitting_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && PB.encode(e, 28, x.memory_fitting_level)
    x.deduplicate_hlo != false && PB.encode(e, 12, x.deduplicate_hlo)
    !isempty(x.allow_spmd_sharding_propagation_to_parameters) && PB.encode(e, 23, x.allow_spmd_sharding_propagation_to_parameters)
    !isempty(x.allow_spmd_sharding_propagation_to_output) && PB.encode(e, 14, x.allow_spmd_sharding_propagation_to_output)
    !isempty(x.param_requires_broadcast_via_collectives) && PB.encode(e, 18, x.param_requires_broadcast_via_collectives)
    x.allow_separate_sharding_programs != false && PB.encode(e, 19, x.allow_separate_sharding_programs)
    !isempty(x.shardable_value_update_pairs) && PB.encode(e, 20, x.shardable_value_update_pairs)
    !isempty(x.fdo_profile) && PB.encode(e, 21, x.fdo_profile)
    x.device_memory_size != zero(Int64) && PB.encode(e, 22, x.device_memory_size)
    x.use_shardy_partitioner != false && PB.encode(e, 24, x.use_shardy_partitioner)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ExecutionOptions)
    encoded_size = 0
    !isnothing(x.shape_with_output_layout) && (encoded_size += PB._encoded_size(x.shape_with_output_layout, 2))
    x.seed != zero(UInt64) && (encoded_size += PB._encoded_size(x.seed, 3))
    !isnothing(x.debug_options) && (encoded_size += PB._encoded_size(x.debug_options, 4))
    !isempty(x.device_handles) && (encoded_size += PB._encoded_size(x.device_handles, 5))
    x.num_replicas != zero(Int32) && (encoded_size += PB._encoded_size(x.num_replicas, 6))
    !isnothing(x.device_assignment) && (encoded_size += PB._encoded_size(x.device_assignment, 7))
    x.alias_passthrough_params != false && (encoded_size += PB._encoded_size(x.alias_passthrough_params, 8))
    x.num_partitions != zero(Int32) && (encoded_size += PB._encoded_size(x.num_partitions, 9))
    x.launch_id != zero(Int32) && (encoded_size += PB._encoded_size(x.launch_id, 10))
    x.use_spmd_partitioning != false && (encoded_size += PB._encoded_size(x.use_spmd_partitioning, 11))
    x.use_auto_spmd_partitioning != false && (encoded_size += PB._encoded_size(x.use_auto_spmd_partitioning, 15))
    !isempty(x.auto_spmd_partitioning_mesh_shape) && (encoded_size += PB._encoded_size(x.auto_spmd_partitioning_mesh_shape, 16))
    !isempty(x.auto_spmd_partitioning_mesh_ids) && (encoded_size += PB._encoded_size(x.auto_spmd_partitioning_mesh_ids, 17))
    x.exec_time_optimization_effort !== zero(Float32) && (encoded_size += PB._encoded_size(x.exec_time_optimization_effort, 25))
    x.memory_fitting_effort !== zero(Float32) && (encoded_size += PB._encoded_size(x.memory_fitting_effort, 26))
    x.optimization_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && (encoded_size += PB._encoded_size(x.optimization_level, 27))
    x.memory_fitting_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && (encoded_size += PB._encoded_size(x.memory_fitting_level, 28))
    x.deduplicate_hlo != false && (encoded_size += PB._encoded_size(x.deduplicate_hlo, 12))
    !isempty(x.allow_spmd_sharding_propagation_to_parameters) && (encoded_size += PB._encoded_size(x.allow_spmd_sharding_propagation_to_parameters, 23))
    !isempty(x.allow_spmd_sharding_propagation_to_output) && (encoded_size += PB._encoded_size(x.allow_spmd_sharding_propagation_to_output, 14))
    !isempty(x.param_requires_broadcast_via_collectives) && (encoded_size += PB._encoded_size(x.param_requires_broadcast_via_collectives, 18))
    x.allow_separate_sharding_programs != false && (encoded_size += PB._encoded_size(x.allow_separate_sharding_programs, 19))
    !isempty(x.shardable_value_update_pairs) && (encoded_size += PB._encoded_size(x.shardable_value_update_pairs, 20))
    !isempty(x.fdo_profile) && (encoded_size += PB._encoded_size(x.fdo_profile, 21))
    x.device_memory_size != zero(Int64) && (encoded_size += PB._encoded_size(x.device_memory_size, 22))
    x.use_shardy_partitioner != false && (encoded_size += PB._encoded_size(x.use_shardy_partitioner, 24))
    return encoded_size
end

struct HloModuleProtoWithConfig
    hlo_module::Union{Nothing,HloModuleProto}
    config::Union{Nothing,HloModuleConfigProto}
end
HloModuleProtoWithConfig(;hlo_module = nothing, config = nothing) = HloModuleProtoWithConfig(hlo_module, config)
PB.default_values(::Type{HloModuleProtoWithConfig}) = (;hlo_module = nothing, config = nothing)
PB.field_numbers(::Type{HloModuleProtoWithConfig}) = (;hlo_module = 1, config = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloModuleProtoWithConfig})
    hlo_module = Ref{Union{Nothing,HloModuleProto}}(nothing)
    config = Ref{Union{Nothing,HloModuleConfigProto}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, hlo_module)
        elseif field_number == 2
            PB.decode!(d, config)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloModuleProtoWithConfig(hlo_module[], config[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloModuleProtoWithConfig)
    initpos = position(e.io)
    !isnothing(x.hlo_module) && PB.encode(e, 1, x.hlo_module)
    !isnothing(x.config) && PB.encode(e, 2, x.config)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloModuleProtoWithConfig)
    encoded_size = 0
    !isnothing(x.hlo_module) && (encoded_size += PB._encoded_size(x.hlo_module, 1))
    !isnothing(x.config) && (encoded_size += PB._encoded_size(x.config, 2))
    return encoded_size
end
