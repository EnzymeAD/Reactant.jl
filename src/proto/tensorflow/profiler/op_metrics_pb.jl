import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export PrecisionStats, MemBwType, var"PerformanceInfo.MemoryAccessed.MemorySpace"
export LayoutDimensionSemantics, var"OpMetrics.MemoryAccessed.OperationType"
export var"OpMetrics.TpuCoreType", MemorySpace, var"PerformanceInfo.MemoryAccessed"
export var"LayoutAnalysis.Dimension", var"OpMetrics.MemoryAccessed", PerformanceInfo
export LayoutAnalysis, MemoryAccessBreakdown, OpMetrics, OpMetricsDb
abstract type var"##Abstract#OpMetrics" end
abstract type var"##Abstract#OpMetricsDb" end


struct PrecisionStats
    compute_16bit_ps::UInt64
    compute_32bit_ps::UInt64
end
PrecisionStats(;compute_16bit_ps = zero(UInt64), compute_32bit_ps = zero(UInt64)) = PrecisionStats(compute_16bit_ps, compute_32bit_ps)
PB.default_values(::Type{PrecisionStats}) = (;compute_16bit_ps = zero(UInt64), compute_32bit_ps = zero(UInt64))
PB.field_numbers(::Type{PrecisionStats}) = (;compute_16bit_ps = 1, compute_32bit_ps = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PrecisionStats})
    compute_16bit_ps = zero(UInt64)
    compute_32bit_ps = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            compute_16bit_ps = PB.decode(d, UInt64)
        elseif field_number == 2
            compute_32bit_ps = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return PrecisionStats(compute_16bit_ps, compute_32bit_ps)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PrecisionStats)
    initpos = position(e.io)
    x.compute_16bit_ps != zero(UInt64) && PB.encode(e, 1, x.compute_16bit_ps)
    x.compute_32bit_ps != zero(UInt64) && PB.encode(e, 2, x.compute_32bit_ps)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PrecisionStats)
    encoded_size = 0
    x.compute_16bit_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.compute_16bit_ps, 1))
    x.compute_32bit_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.compute_32bit_ps, 2))
    return encoded_size
end

@enumx MemBwType MEM_BW_TYPE_FIRST=0 MEM_BW_TYPE_HBM_RW=0 MEM_BW_TYPE_SRAM_RD=1 MEM_BW_TYPE_SRAM_WR=2 MEM_BW_TYPE_CMEM_RD=3 MEM_BW_TYPE_CMEM_WR=4 MEM_BW_TYPE_VMEM_RD=5 MEM_BW_TYPE_VMEM_WR=6 MEM_BW_TYPE_MAX=2

@enumx var"PerformanceInfo.MemoryAccessed.MemorySpace" UNKNOWN=0 HBM=1 CMEM=2 VMEM=3

@enumx LayoutDimensionSemantics UNKNOWN_SEMANTICS=0 FEATURE=1 BATCH=2 SPATIAL=3

@enumx var"OpMetrics.MemoryAccessed.OperationType" UNKNOWN=0 READ=1 WRITE=2

@enumx var"OpMetrics.TpuCoreType" UNKNOWN=0 TENSOR_CORE=1 SPARSE_CORE=2

@enumx MemorySpace MEMORY_SPACE_UNDEFINED=0 MEMORY_SPACE_HBM=1 MEMORY_SPACE_ON_CHIP=2147483646 MEMORY_SPACE_ALL=2147483647

struct var"PerformanceInfo.MemoryAccessed"
    is_read::Bool
    memory_space::var"PerformanceInfo.MemoryAccessed.MemorySpace".T
    bytes_accessed::Int64
end
var"PerformanceInfo.MemoryAccessed"(;is_read = false, memory_space = var"PerformanceInfo.MemoryAccessed.MemorySpace".UNKNOWN, bytes_accessed = zero(Int64)) = var"PerformanceInfo.MemoryAccessed"(is_read, memory_space, bytes_accessed)
PB.default_values(::Type{var"PerformanceInfo.MemoryAccessed"}) = (;is_read = false, memory_space = var"PerformanceInfo.MemoryAccessed.MemorySpace".UNKNOWN, bytes_accessed = zero(Int64))
PB.field_numbers(::Type{var"PerformanceInfo.MemoryAccessed"}) = (;is_read = 1, memory_space = 2, bytes_accessed = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"PerformanceInfo.MemoryAccessed"})
    is_read = false
    memory_space = var"PerformanceInfo.MemoryAccessed.MemorySpace".UNKNOWN
    bytes_accessed = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            is_read = PB.decode(d, Bool)
        elseif field_number == 2
            memory_space = PB.decode(d, var"PerformanceInfo.MemoryAccessed.MemorySpace".T)
        elseif field_number == 3
            bytes_accessed = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"PerformanceInfo.MemoryAccessed"(is_read, memory_space, bytes_accessed)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"PerformanceInfo.MemoryAccessed")
    initpos = position(e.io)
    x.is_read != false && PB.encode(e, 1, x.is_read)
    x.memory_space != var"PerformanceInfo.MemoryAccessed.MemorySpace".UNKNOWN && PB.encode(e, 2, x.memory_space)
    x.bytes_accessed != zero(Int64) && PB.encode(e, 3, x.bytes_accessed)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"PerformanceInfo.MemoryAccessed")
    encoded_size = 0
    x.is_read != false && (encoded_size += PB._encoded_size(x.is_read, 1))
    x.memory_space != var"PerformanceInfo.MemoryAccessed.MemorySpace".UNKNOWN && (encoded_size += PB._encoded_size(x.memory_space, 2))
    x.bytes_accessed != zero(Int64) && (encoded_size += PB._encoded_size(x.bytes_accessed, 3))
    return encoded_size
end

struct var"LayoutAnalysis.Dimension"
    size::Int32
    alignment::Int32
    semantics::LayoutDimensionSemantics.T
end
var"LayoutAnalysis.Dimension"(;size = zero(Int32), alignment = zero(Int32), semantics = LayoutDimensionSemantics.UNKNOWN_SEMANTICS) = var"LayoutAnalysis.Dimension"(size, alignment, semantics)
PB.default_values(::Type{var"LayoutAnalysis.Dimension"}) = (;size = zero(Int32), alignment = zero(Int32), semantics = LayoutDimensionSemantics.UNKNOWN_SEMANTICS)
PB.field_numbers(::Type{var"LayoutAnalysis.Dimension"}) = (;size = 1, alignment = 2, semantics = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"LayoutAnalysis.Dimension"})
    size = zero(Int32)
    alignment = zero(Int32)
    semantics = LayoutDimensionSemantics.UNKNOWN_SEMANTICS
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            size = PB.decode(d, Int32)
        elseif field_number == 2
            alignment = PB.decode(d, Int32)
        elseif field_number == 3
            semantics = PB.decode(d, LayoutDimensionSemantics.T)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"LayoutAnalysis.Dimension"(size, alignment, semantics)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"LayoutAnalysis.Dimension")
    initpos = position(e.io)
    x.size != zero(Int32) && PB.encode(e, 1, x.size)
    x.alignment != zero(Int32) && PB.encode(e, 2, x.alignment)
    x.semantics != LayoutDimensionSemantics.UNKNOWN_SEMANTICS && PB.encode(e, 3, x.semantics)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"LayoutAnalysis.Dimension")
    encoded_size = 0
    x.size != zero(Int32) && (encoded_size += PB._encoded_size(x.size, 1))
    x.alignment != zero(Int32) && (encoded_size += PB._encoded_size(x.alignment, 2))
    x.semantics != LayoutDimensionSemantics.UNKNOWN_SEMANTICS && (encoded_size += PB._encoded_size(x.semantics, 3))
    return encoded_size
end

struct var"OpMetrics.MemoryAccessed"
    operation_type::var"OpMetrics.MemoryAccessed.OperationType".T
    memory_space::UInt64
    bytes_accessed::UInt64
end
var"OpMetrics.MemoryAccessed"(;operation_type = var"OpMetrics.MemoryAccessed.OperationType".UNKNOWN, memory_space = zero(UInt64), bytes_accessed = zero(UInt64)) = var"OpMetrics.MemoryAccessed"(operation_type, memory_space, bytes_accessed)
PB.default_values(::Type{var"OpMetrics.MemoryAccessed"}) = (;operation_type = var"OpMetrics.MemoryAccessed.OperationType".UNKNOWN, memory_space = zero(UInt64), bytes_accessed = zero(UInt64))
PB.field_numbers(::Type{var"OpMetrics.MemoryAccessed"}) = (;operation_type = 1, memory_space = 2, bytes_accessed = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"OpMetrics.MemoryAccessed"})
    operation_type = var"OpMetrics.MemoryAccessed.OperationType".UNKNOWN
    memory_space = zero(UInt64)
    bytes_accessed = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            operation_type = PB.decode(d, var"OpMetrics.MemoryAccessed.OperationType".T)
        elseif field_number == 2
            memory_space = PB.decode(d, UInt64)
        elseif field_number == 3
            bytes_accessed = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"OpMetrics.MemoryAccessed"(operation_type, memory_space, bytes_accessed)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"OpMetrics.MemoryAccessed")
    initpos = position(e.io)
    x.operation_type != var"OpMetrics.MemoryAccessed.OperationType".UNKNOWN && PB.encode(e, 1, x.operation_type)
    x.memory_space != zero(UInt64) && PB.encode(e, 2, x.memory_space)
    x.bytes_accessed != zero(UInt64) && PB.encode(e, 3, x.bytes_accessed)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"OpMetrics.MemoryAccessed")
    encoded_size = 0
    x.operation_type != var"OpMetrics.MemoryAccessed.OperationType".UNKNOWN && (encoded_size += PB._encoded_size(x.operation_type, 1))
    x.memory_space != zero(UInt64) && (encoded_size += PB._encoded_size(x.memory_space, 2))
    x.bytes_accessed != zero(UInt64) && (encoded_size += PB._encoded_size(x.bytes_accessed, 3))
    return encoded_size
end

struct PerformanceInfo
    flops::Int64
    bytes_accessed::Int64
    memory_accessed_breakdown::Vector{var"PerformanceInfo.MemoryAccessed"}
end
PerformanceInfo(;flops = zero(Int64), bytes_accessed = zero(Int64), memory_accessed_breakdown = Vector{var"PerformanceInfo.MemoryAccessed"}()) = PerformanceInfo(flops, bytes_accessed, memory_accessed_breakdown)
PB.reserved_fields(::Type{PerformanceInfo}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[4])
PB.default_values(::Type{PerformanceInfo}) = (;flops = zero(Int64), bytes_accessed = zero(Int64), memory_accessed_breakdown = Vector{var"PerformanceInfo.MemoryAccessed"}())
PB.field_numbers(::Type{PerformanceInfo}) = (;flops = 1, bytes_accessed = 2, memory_accessed_breakdown = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PerformanceInfo})
    flops = zero(Int64)
    bytes_accessed = zero(Int64)
    memory_accessed_breakdown = PB.BufferedVector{var"PerformanceInfo.MemoryAccessed"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            flops = PB.decode(d, Int64)
        elseif field_number == 2
            bytes_accessed = PB.decode(d, Int64)
        elseif field_number == 3
            PB.decode!(d, memory_accessed_breakdown)
        else
            Base.skip(d, wire_type)
        end
    end
    return PerformanceInfo(flops, bytes_accessed, memory_accessed_breakdown[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PerformanceInfo)
    initpos = position(e.io)
    x.flops != zero(Int64) && PB.encode(e, 1, x.flops)
    x.bytes_accessed != zero(Int64) && PB.encode(e, 2, x.bytes_accessed)
    !isempty(x.memory_accessed_breakdown) && PB.encode(e, 3, x.memory_accessed_breakdown)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PerformanceInfo)
    encoded_size = 0
    x.flops != zero(Int64) && (encoded_size += PB._encoded_size(x.flops, 1))
    x.bytes_accessed != zero(Int64) && (encoded_size += PB._encoded_size(x.bytes_accessed, 2))
    !isempty(x.memory_accessed_breakdown) && (encoded_size += PB._encoded_size(x.memory_accessed_breakdown, 3))
    return encoded_size
end

struct LayoutAnalysis
    dimensions::Vector{var"LayoutAnalysis.Dimension"}
end
LayoutAnalysis(;dimensions = Vector{var"LayoutAnalysis.Dimension"}()) = LayoutAnalysis(dimensions)
PB.default_values(::Type{LayoutAnalysis}) = (;dimensions = Vector{var"LayoutAnalysis.Dimension"}())
PB.field_numbers(::Type{LayoutAnalysis}) = (;dimensions = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:LayoutAnalysis})
    dimensions = PB.BufferedVector{var"LayoutAnalysis.Dimension"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, dimensions)
        else
            Base.skip(d, wire_type)
        end
    end
    return LayoutAnalysis(dimensions[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::LayoutAnalysis)
    initpos = position(e.io)
    !isempty(x.dimensions) && PB.encode(e, 1, x.dimensions)
    return position(e.io) - initpos
end
function PB._encoded_size(x::LayoutAnalysis)
    encoded_size = 0
    !isempty(x.dimensions) && (encoded_size += PB._encoded_size(x.dimensions, 1))
    return encoded_size
end

struct MemoryAccessBreakdown
    memory_accessed::Vector{var"OpMetrics.MemoryAccessed"}
end
MemoryAccessBreakdown(;memory_accessed = Vector{var"OpMetrics.MemoryAccessed"}()) = MemoryAccessBreakdown(memory_accessed)
PB.default_values(::Type{MemoryAccessBreakdown}) = (;memory_accessed = Vector{var"OpMetrics.MemoryAccessed"}())
PB.field_numbers(::Type{MemoryAccessBreakdown}) = (;memory_accessed = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:MemoryAccessBreakdown})
    memory_accessed = PB.BufferedVector{var"OpMetrics.MemoryAccessed"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, memory_accessed)
        else
            Base.skip(d, wire_type)
        end
    end
    return MemoryAccessBreakdown(memory_accessed[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::MemoryAccessBreakdown)
    initpos = position(e.io)
    !isempty(x.memory_accessed) && PB.encode(e, 1, x.memory_accessed)
    return position(e.io) - initpos
end
function PB._encoded_size(x::MemoryAccessBreakdown)
    encoded_size = 0
    !isempty(x.memory_accessed) && (encoded_size += PB._encoded_size(x.memory_accessed, 1))
    return encoded_size
end

# Stub definitions for cyclic types
struct var"##Stub#OpMetrics"{T1<:var"##Abstract#OpMetricsDb"} <: var"##Abstract#OpMetrics"
    hlo_module_id::UInt64
    name::String
    long_name::String
    category::String
    provenance::String
    is_eager::Bool
    occurrences::UInt32
    time_ps::UInt64
    normalized_time_ps::UInt64
    min_time_ps::UInt64
    self_time_ps::UInt64
    flops::UInt64
    model_flops::UInt64
    fingerprint::UInt64
    bytes_accessed::UInt64
    memory_accessed_breakdown::Vector{var"OpMetrics.MemoryAccessed"}
    dma_stall_ps::UInt64
    layout::Union{Nothing,LayoutAnalysis}
    deduplicated_name::String
    children::Union{Nothing,T1}
    num_cores::UInt32
    computation_primitive_size::UInt32
    autotuned::Bool
    source_info::Union{Nothing,SourceInfo}
    core_type::var"OpMetrics.TpuCoreType".T
end

struct var"##Stub#OpMetricsDb" <: var"##Abstract#OpMetricsDb"
    metrics_db::Vector{var"##Stub#OpMetrics"{var"##Stub#OpMetricsDb"}}
    total_host_infeed_enq_duration_ps::UInt64
    total_host_infeed_enq_start_timestamp_ps_diff::UInt64
    total_time_ps::UInt64
    total_op_time_ps::UInt64
    normalized_total_op_time_ps::UInt64
    precision_stats::Union{Nothing,PrecisionStats}
    idle_time_ps::UInt64
    busy_time_ps::UInt64
end

const OpMetrics = var"##Stub#OpMetrics"{var"##Stub#OpMetricsDb"}
OpMetrics(;hlo_module_id = zero(UInt64), name = "", long_name = "", category = "", provenance = "", is_eager = false, occurrences = zero(UInt32), time_ps = zero(UInt64), normalized_time_ps = zero(UInt64), min_time_ps = zero(UInt64), self_time_ps = zero(UInt64), flops = zero(UInt64), model_flops = zero(UInt64), fingerprint = zero(UInt64), bytes_accessed = zero(UInt64), memory_accessed_breakdown = Vector{var"OpMetrics.MemoryAccessed"}(), dma_stall_ps = zero(UInt64), layout = nothing, deduplicated_name = "", children = nothing, num_cores = zero(UInt32), computation_primitive_size = zero(UInt32), autotuned = false, source_info = nothing, core_type = var"OpMetrics.TpuCoreType".UNKNOWN) = OpMetrics(hlo_module_id, name, long_name, category, provenance, is_eager, occurrences, time_ps, normalized_time_ps, min_time_ps, self_time_ps, flops, model_flops, fingerprint, bytes_accessed, memory_accessed_breakdown, dma_stall_ps, layout, deduplicated_name, children, num_cores, computation_primitive_size, autotuned, source_info, core_type)
PB.reserved_fields(::Type{OpMetrics}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[4, 8, 9])
PB.default_values(::Type{OpMetrics}) = (;hlo_module_id = zero(UInt64), name = "", long_name = "", category = "", provenance = "", is_eager = false, occurrences = zero(UInt32), time_ps = zero(UInt64), normalized_time_ps = zero(UInt64), min_time_ps = zero(UInt64), self_time_ps = zero(UInt64), flops = zero(UInt64), model_flops = zero(UInt64), fingerprint = zero(UInt64), bytes_accessed = zero(UInt64), memory_accessed_breakdown = Vector{var"OpMetrics.MemoryAccessed"}(), dma_stall_ps = zero(UInt64), layout = nothing, deduplicated_name = "", children = nothing, num_cores = zero(UInt32), computation_primitive_size = zero(UInt32), autotuned = false, source_info = nothing, core_type = var"OpMetrics.TpuCoreType".UNKNOWN)
PB.field_numbers(::Type{OpMetrics}) = (;hlo_module_id = 13, name = 6, long_name = 20, category = 11, provenance = 12, is_eager = 18, occurrences = 3, time_ps = 7, normalized_time_ps = 27, min_time_ps = 17, self_time_ps = 1, flops = 2, model_flops = 24, fingerprint = 25, bytes_accessed = 5, memory_accessed_breakdown = 19, dma_stall_ps = 10, layout = 14, deduplicated_name = 15, children = 16, num_cores = 21, computation_primitive_size = 22, autotuned = 23, source_info = 26, core_type = 28)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OpMetrics})
    hlo_module_id = zero(UInt64)
    name = ""
    long_name = ""
    category = ""
    provenance = ""
    is_eager = false
    occurrences = zero(UInt32)
    time_ps = zero(UInt64)
    normalized_time_ps = zero(UInt64)
    min_time_ps = zero(UInt64)
    self_time_ps = zero(UInt64)
    flops = zero(UInt64)
    model_flops = zero(UInt64)
    fingerprint = zero(UInt64)
    bytes_accessed = zero(UInt64)
    memory_accessed_breakdown = PB.BufferedVector{var"OpMetrics.MemoryAccessed"}()
    dma_stall_ps = zero(UInt64)
    layout = Ref{Union{Nothing,LayoutAnalysis}}(nothing)
    deduplicated_name = ""
    children = Ref{Union{Nothing,OpMetricsDb}}(nothing)
    num_cores = zero(UInt32)
    computation_primitive_size = zero(UInt32)
    autotuned = false
    source_info = Ref{Union{Nothing,SourceInfo}}(nothing)
    core_type = var"OpMetrics.TpuCoreType".UNKNOWN
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 13
            hlo_module_id = PB.decode(d, UInt64)
        elseif field_number == 6
            name = PB.decode(d, String)
        elseif field_number == 20
            long_name = PB.decode(d, String)
        elseif field_number == 11
            category = PB.decode(d, String)
        elseif field_number == 12
            provenance = PB.decode(d, String)
        elseif field_number == 18
            is_eager = PB.decode(d, Bool)
        elseif field_number == 3
            occurrences = PB.decode(d, UInt32)
        elseif field_number == 7
            time_ps = PB.decode(d, UInt64)
        elseif field_number == 27
            normalized_time_ps = PB.decode(d, UInt64)
        elseif field_number == 17
            min_time_ps = PB.decode(d, UInt64)
        elseif field_number == 1
            self_time_ps = PB.decode(d, UInt64)
        elseif field_number == 2
            flops = PB.decode(d, UInt64)
        elseif field_number == 24
            model_flops = PB.decode(d, UInt64)
        elseif field_number == 25
            fingerprint = PB.decode(d, UInt64)
        elseif field_number == 5
            bytes_accessed = PB.decode(d, UInt64)
        elseif field_number == 19
            PB.decode!(d, memory_accessed_breakdown)
        elseif field_number == 10
            dma_stall_ps = PB.decode(d, UInt64)
        elseif field_number == 14
            PB.decode!(d, layout)
        elseif field_number == 15
            deduplicated_name = PB.decode(d, String)
        elseif field_number == 16
            PB.decode!(d, children)
        elseif field_number == 21
            num_cores = PB.decode(d, UInt32)
        elseif field_number == 22
            computation_primitive_size = PB.decode(d, UInt32)
        elseif field_number == 23
            autotuned = PB.decode(d, Bool)
        elseif field_number == 26
            PB.decode!(d, source_info)
        elseif field_number == 28
            core_type = PB.decode(d, var"OpMetrics.TpuCoreType".T)
        else
            Base.skip(d, wire_type)
        end
    end
    return OpMetrics(hlo_module_id, name, long_name, category, provenance, is_eager, occurrences, time_ps, normalized_time_ps, min_time_ps, self_time_ps, flops, model_flops, fingerprint, bytes_accessed, memory_accessed_breakdown[], dma_stall_ps, layout[], deduplicated_name, children[], num_cores, computation_primitive_size, autotuned, source_info[], core_type)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OpMetrics)
    initpos = position(e.io)
    x.hlo_module_id != zero(UInt64) && PB.encode(e, 13, x.hlo_module_id)
    !isempty(x.name) && PB.encode(e, 6, x.name)
    !isempty(x.long_name) && PB.encode(e, 20, x.long_name)
    !isempty(x.category) && PB.encode(e, 11, x.category)
    !isempty(x.provenance) && PB.encode(e, 12, x.provenance)
    x.is_eager != false && PB.encode(e, 18, x.is_eager)
    x.occurrences != zero(UInt32) && PB.encode(e, 3, x.occurrences)
    x.time_ps != zero(UInt64) && PB.encode(e, 7, x.time_ps)
    x.normalized_time_ps != zero(UInt64) && PB.encode(e, 27, x.normalized_time_ps)
    x.min_time_ps != zero(UInt64) && PB.encode(e, 17, x.min_time_ps)
    x.self_time_ps != zero(UInt64) && PB.encode(e, 1, x.self_time_ps)
    x.flops != zero(UInt64) && PB.encode(e, 2, x.flops)
    x.model_flops != zero(UInt64) && PB.encode(e, 24, x.model_flops)
    x.fingerprint != zero(UInt64) && PB.encode(e, 25, x.fingerprint)
    x.bytes_accessed != zero(UInt64) && PB.encode(e, 5, x.bytes_accessed)
    !isempty(x.memory_accessed_breakdown) && PB.encode(e, 19, x.memory_accessed_breakdown)
    x.dma_stall_ps != zero(UInt64) && PB.encode(e, 10, x.dma_stall_ps)
    !isnothing(x.layout) && PB.encode(e, 14, x.layout)
    !isempty(x.deduplicated_name) && PB.encode(e, 15, x.deduplicated_name)
    !isnothing(x.children) && PB.encode(e, 16, x.children)
    x.num_cores != zero(UInt32) && PB.encode(e, 21, x.num_cores)
    x.computation_primitive_size != zero(UInt32) && PB.encode(e, 22, x.computation_primitive_size)
    x.autotuned != false && PB.encode(e, 23, x.autotuned)
    !isnothing(x.source_info) && PB.encode(e, 26, x.source_info)
    x.core_type != var"OpMetrics.TpuCoreType".UNKNOWN && PB.encode(e, 28, x.core_type)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OpMetrics)
    encoded_size = 0
    x.hlo_module_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.hlo_module_id, 13))
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 6))
    !isempty(x.long_name) && (encoded_size += PB._encoded_size(x.long_name, 20))
    !isempty(x.category) && (encoded_size += PB._encoded_size(x.category, 11))
    !isempty(x.provenance) && (encoded_size += PB._encoded_size(x.provenance, 12))
    x.is_eager != false && (encoded_size += PB._encoded_size(x.is_eager, 18))
    x.occurrences != zero(UInt32) && (encoded_size += PB._encoded_size(x.occurrences, 3))
    x.time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.time_ps, 7))
    x.normalized_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.normalized_time_ps, 27))
    x.min_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.min_time_ps, 17))
    x.self_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.self_time_ps, 1))
    x.flops != zero(UInt64) && (encoded_size += PB._encoded_size(x.flops, 2))
    x.model_flops != zero(UInt64) && (encoded_size += PB._encoded_size(x.model_flops, 24))
    x.fingerprint != zero(UInt64) && (encoded_size += PB._encoded_size(x.fingerprint, 25))
    x.bytes_accessed != zero(UInt64) && (encoded_size += PB._encoded_size(x.bytes_accessed, 5))
    !isempty(x.memory_accessed_breakdown) && (encoded_size += PB._encoded_size(x.memory_accessed_breakdown, 19))
    x.dma_stall_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.dma_stall_ps, 10))
    !isnothing(x.layout) && (encoded_size += PB._encoded_size(x.layout, 14))
    !isempty(x.deduplicated_name) && (encoded_size += PB._encoded_size(x.deduplicated_name, 15))
    !isnothing(x.children) && (encoded_size += PB._encoded_size(x.children, 16))
    x.num_cores != zero(UInt32) && (encoded_size += PB._encoded_size(x.num_cores, 21))
    x.computation_primitive_size != zero(UInt32) && (encoded_size += PB._encoded_size(x.computation_primitive_size, 22))
    x.autotuned != false && (encoded_size += PB._encoded_size(x.autotuned, 23))
    !isnothing(x.source_info) && (encoded_size += PB._encoded_size(x.source_info, 26))
    x.core_type != var"OpMetrics.TpuCoreType".UNKNOWN && (encoded_size += PB._encoded_size(x.core_type, 28))
    return encoded_size
end

const OpMetricsDb = var"##Stub#OpMetricsDb"
OpMetricsDb(;metrics_db = Vector{OpMetrics}(), total_host_infeed_enq_duration_ps = zero(UInt64), total_host_infeed_enq_start_timestamp_ps_diff = zero(UInt64), total_time_ps = zero(UInt64), total_op_time_ps = zero(UInt64), normalized_total_op_time_ps = zero(UInt64), precision_stats = nothing, idle_time_ps = zero(UInt64), busy_time_ps = zero(UInt64)) = OpMetricsDb(metrics_db, total_host_infeed_enq_duration_ps, total_host_infeed_enq_start_timestamp_ps_diff, total_time_ps, total_op_time_ps, normalized_total_op_time_ps, precision_stats, idle_time_ps, busy_time_ps)
PB.reserved_fields(::Type{OpMetricsDb}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[1, 4, 5, 6, 7, 8, 9])
PB.default_values(::Type{OpMetricsDb}) = (;metrics_db = Vector{OpMetrics}(), total_host_infeed_enq_duration_ps = zero(UInt64), total_host_infeed_enq_start_timestamp_ps_diff = zero(UInt64), total_time_ps = zero(UInt64), total_op_time_ps = zero(UInt64), normalized_total_op_time_ps = zero(UInt64), precision_stats = nothing, idle_time_ps = zero(UInt64), busy_time_ps = zero(UInt64))
PB.field_numbers(::Type{OpMetricsDb}) = (;metrics_db = 10, total_host_infeed_enq_duration_ps = 2, total_host_infeed_enq_start_timestamp_ps_diff = 3, total_time_ps = 11, total_op_time_ps = 12, normalized_total_op_time_ps = 16, precision_stats = 13, idle_time_ps = 14, busy_time_ps = 15)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OpMetricsDb})
    metrics_db = PB.BufferedVector{OpMetrics}()
    total_host_infeed_enq_duration_ps = zero(UInt64)
    total_host_infeed_enq_start_timestamp_ps_diff = zero(UInt64)
    total_time_ps = zero(UInt64)
    total_op_time_ps = zero(UInt64)
    normalized_total_op_time_ps = zero(UInt64)
    precision_stats = Ref{Union{Nothing,PrecisionStats}}(nothing)
    idle_time_ps = zero(UInt64)
    busy_time_ps = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 10
            PB.decode!(d, metrics_db)
        elseif field_number == 2
            total_host_infeed_enq_duration_ps = PB.decode(d, UInt64)
        elseif field_number == 3
            total_host_infeed_enq_start_timestamp_ps_diff = PB.decode(d, UInt64)
        elseif field_number == 11
            total_time_ps = PB.decode(d, UInt64)
        elseif field_number == 12
            total_op_time_ps = PB.decode(d, UInt64)
        elseif field_number == 16
            normalized_total_op_time_ps = PB.decode(d, UInt64)
        elseif field_number == 13
            PB.decode!(d, precision_stats)
        elseif field_number == 14
            idle_time_ps = PB.decode(d, UInt64)
        elseif field_number == 15
            busy_time_ps = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return OpMetricsDb(metrics_db[], total_host_infeed_enq_duration_ps, total_host_infeed_enq_start_timestamp_ps_diff, total_time_ps, total_op_time_ps, normalized_total_op_time_ps, precision_stats[], idle_time_ps, busy_time_ps)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OpMetricsDb)
    initpos = position(e.io)
    !isempty(x.metrics_db) && PB.encode(e, 10, x.metrics_db)
    x.total_host_infeed_enq_duration_ps != zero(UInt64) && PB.encode(e, 2, x.total_host_infeed_enq_duration_ps)
    x.total_host_infeed_enq_start_timestamp_ps_diff != zero(UInt64) && PB.encode(e, 3, x.total_host_infeed_enq_start_timestamp_ps_diff)
    x.total_time_ps != zero(UInt64) && PB.encode(e, 11, x.total_time_ps)
    x.total_op_time_ps != zero(UInt64) && PB.encode(e, 12, x.total_op_time_ps)
    x.normalized_total_op_time_ps != zero(UInt64) && PB.encode(e, 16, x.normalized_total_op_time_ps)
    !isnothing(x.precision_stats) && PB.encode(e, 13, x.precision_stats)
    x.idle_time_ps != zero(UInt64) && PB.encode(e, 14, x.idle_time_ps)
    x.busy_time_ps != zero(UInt64) && PB.encode(e, 15, x.busy_time_ps)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OpMetricsDb)
    encoded_size = 0
    !isempty(x.metrics_db) && (encoded_size += PB._encoded_size(x.metrics_db, 10))
    x.total_host_infeed_enq_duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.total_host_infeed_enq_duration_ps, 2))
    x.total_host_infeed_enq_start_timestamp_ps_diff != zero(UInt64) && (encoded_size += PB._encoded_size(x.total_host_infeed_enq_start_timestamp_ps_diff, 3))
    x.total_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.total_time_ps, 11))
    x.total_op_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.total_op_time_ps, 12))
    x.normalized_total_op_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.normalized_total_op_time_ps, 16))
    !isnothing(x.precision_stats) && (encoded_size += PB._encoded_size(x.precision_stats, 13))
    x.idle_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.idle_time_ps, 14))
    x.busy_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.busy_time_ps, 15))
    return encoded_size
end
