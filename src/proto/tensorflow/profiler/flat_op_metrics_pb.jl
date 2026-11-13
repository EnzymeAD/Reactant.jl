import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"FlatOpMetrics.TpuCoreType", var"FlatOpMetrics.MemoryAccessed.OperationType"
export var"FlatOpMetrics.MemoryAccessed", FlatOpMetrics, FlatOpMetricsDb
export FlatOpMetricsDbMap


@enumx var"FlatOpMetrics.TpuCoreType" UNKNOWN=0 TENSOR_CORE=1 SPARSE_CORE=2

@enumx var"FlatOpMetrics.MemoryAccessed.OperationType" UNKNOWN=0 READ=1 WRITE=2

struct var"FlatOpMetrics.MemoryAccessed"
    operation_type::var"FlatOpMetrics.MemoryAccessed.OperationType".T
    memory_space::UInt64
    bytes_accessed::UInt64
end
PB.default_values(::Type{var"FlatOpMetrics.MemoryAccessed"}) = (;operation_type = var"FlatOpMetrics.MemoryAccessed.OperationType".UNKNOWN, memory_space = zero(UInt64), bytes_accessed = zero(UInt64))
PB.field_numbers(::Type{var"FlatOpMetrics.MemoryAccessed"}) = (;operation_type = 1, memory_space = 2, bytes_accessed = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"FlatOpMetrics.MemoryAccessed"}, _endpos::Int=0, _group::Bool=false)
    operation_type = var"FlatOpMetrics.MemoryAccessed.OperationType".UNKNOWN
    memory_space = zero(UInt64)
    bytes_accessed = zero(UInt64)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            operation_type = PB.decode(d, var"FlatOpMetrics.MemoryAccessed.OperationType".T)
        elseif field_number == 2
            memory_space = PB.decode(d, UInt64)
        elseif field_number == 3
            bytes_accessed = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"FlatOpMetrics.MemoryAccessed"(operation_type, memory_space, bytes_accessed)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"FlatOpMetrics.MemoryAccessed")
    initpos = position(e.io)
    x.operation_type != var"FlatOpMetrics.MemoryAccessed.OperationType".UNKNOWN && PB.encode(e, 1, x.operation_type)
    x.memory_space != zero(UInt64) && PB.encode(e, 2, x.memory_space)
    x.bytes_accessed != zero(UInt64) && PB.encode(e, 3, x.bytes_accessed)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"FlatOpMetrics.MemoryAccessed")
    encoded_size = 0
    x.operation_type != var"FlatOpMetrics.MemoryAccessed.OperationType".UNKNOWN && (encoded_size += PB._encoded_size(x.operation_type, 1))
    x.memory_space != zero(UInt64) && (encoded_size += PB._encoded_size(x.memory_space, 2))
    x.bytes_accessed != zero(UInt64) && (encoded_size += PB._encoded_size(x.bytes_accessed, 3))
    return encoded_size
end

struct FlatOpMetrics
    op_id::UInt64
    parent_op_id::UInt64
    symbol_id::UInt64
    hlo_module_id::UInt64
    children_count::UInt64
    has_sc_children::Bool
    hlo_name::String
    long_name::String
    category::String
    provenance::String
    deduplicated_name::String
    self_time_ps::UInt64
    time_ps::UInt64
    total_time_ps::UInt64
    normalized_time_ps::UInt64
    min_time_ps::UInt64
    dma_stall_ps::UInt64
    occurrences::UInt64
    model_flops::UInt64
    model_flops_v2::Float64
    flops_v2::Float64
    flops::UInt64
    bytes_accessed::Float64
    memory_accessed_breakdown::Vector{var"FlatOpMetrics.MemoryAccessed"}
    is_eager::Bool
    autotuned::Bool
    num_cores::UInt32
    core_type::var"FlatOpMetrics.TpuCoreType".T
    is_fusion_child::Bool
    source_info::Union{Nothing,SourceInfo}
    estimated_flops::Float64
    estimated_bytes_accessed::Float64
end
PB.default_values(::Type{FlatOpMetrics}) = (;op_id = zero(UInt64), parent_op_id = zero(UInt64), symbol_id = zero(UInt64), hlo_module_id = zero(UInt64), children_count = zero(UInt64), has_sc_children = false, hlo_name = "", long_name = "", category = "", provenance = "", deduplicated_name = "", self_time_ps = zero(UInt64), time_ps = zero(UInt64), total_time_ps = zero(UInt64), normalized_time_ps = zero(UInt64), min_time_ps = zero(UInt64), dma_stall_ps = zero(UInt64), occurrences = zero(UInt64), model_flops = zero(UInt64), model_flops_v2 = zero(Float64), flops_v2 = zero(Float64), flops = zero(UInt64), bytes_accessed = zero(Float64), memory_accessed_breakdown = Vector{var"FlatOpMetrics.MemoryAccessed"}(), is_eager = false, autotuned = false, num_cores = zero(UInt32), core_type = var"FlatOpMetrics.TpuCoreType".UNKNOWN, is_fusion_child = false, source_info = nothing, estimated_flops = zero(Float64), estimated_bytes_accessed = zero(Float64))
PB.field_numbers(::Type{FlatOpMetrics}) = (;op_id = 1, parent_op_id = 2, symbol_id = 3, hlo_module_id = 4, children_count = 5, has_sc_children = 6, hlo_name = 7, long_name = 8, category = 9, provenance = 10, deduplicated_name = 11, self_time_ps = 12, time_ps = 13, total_time_ps = 14, normalized_time_ps = 15, min_time_ps = 16, dma_stall_ps = 17, occurrences = 18, model_flops = 19, model_flops_v2 = 20, flops_v2 = 21, flops = 22, bytes_accessed = 23, memory_accessed_breakdown = 24, is_eager = 25, autotuned = 26, num_cores = 27, core_type = 28, is_fusion_child = 29, source_info = 30, estimated_flops = 31, estimated_bytes_accessed = 32)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:FlatOpMetrics}, _endpos::Int=0, _group::Bool=false)
    op_id = zero(UInt64)
    parent_op_id = zero(UInt64)
    symbol_id = zero(UInt64)
    hlo_module_id = zero(UInt64)
    children_count = zero(UInt64)
    has_sc_children = false
    hlo_name = ""
    long_name = ""
    category = ""
    provenance = ""
    deduplicated_name = ""
    self_time_ps = zero(UInt64)
    time_ps = zero(UInt64)
    total_time_ps = zero(UInt64)
    normalized_time_ps = zero(UInt64)
    min_time_ps = zero(UInt64)
    dma_stall_ps = zero(UInt64)
    occurrences = zero(UInt64)
    model_flops = zero(UInt64)
    model_flops_v2 = zero(Float64)
    flops_v2 = zero(Float64)
    flops = zero(UInt64)
    bytes_accessed = zero(Float64)
    memory_accessed_breakdown = PB.BufferedVector{var"FlatOpMetrics.MemoryAccessed"}()
    is_eager = false
    autotuned = false
    num_cores = zero(UInt32)
    core_type = var"FlatOpMetrics.TpuCoreType".UNKNOWN
    is_fusion_child = false
    source_info = Ref{Union{Nothing,SourceInfo}}(nothing)
    estimated_flops = zero(Float64)
    estimated_bytes_accessed = zero(Float64)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            op_id = PB.decode(d, UInt64)
        elseif field_number == 2
            parent_op_id = PB.decode(d, UInt64)
        elseif field_number == 3
            symbol_id = PB.decode(d, UInt64)
        elseif field_number == 4
            hlo_module_id = PB.decode(d, UInt64)
        elseif field_number == 5
            children_count = PB.decode(d, UInt64)
        elseif field_number == 6
            has_sc_children = PB.decode(d, Bool)
        elseif field_number == 7
            hlo_name = PB.decode(d, String)
        elseif field_number == 8
            long_name = PB.decode(d, String)
        elseif field_number == 9
            category = PB.decode(d, String)
        elseif field_number == 10
            provenance = PB.decode(d, String)
        elseif field_number == 11
            deduplicated_name = PB.decode(d, String)
        elseif field_number == 12
            self_time_ps = PB.decode(d, UInt64)
        elseif field_number == 13
            time_ps = PB.decode(d, UInt64)
        elseif field_number == 14
            total_time_ps = PB.decode(d, UInt64)
        elseif field_number == 15
            normalized_time_ps = PB.decode(d, UInt64)
        elseif field_number == 16
            min_time_ps = PB.decode(d, UInt64)
        elseif field_number == 17
            dma_stall_ps = PB.decode(d, UInt64)
        elseif field_number == 18
            occurrences = PB.decode(d, UInt64)
        elseif field_number == 19
            model_flops = PB.decode(d, UInt64)
        elseif field_number == 20
            model_flops_v2 = PB.decode(d, Float64)
        elseif field_number == 21
            flops_v2 = PB.decode(d, Float64)
        elseif field_number == 22
            flops = PB.decode(d, UInt64)
        elseif field_number == 23
            bytes_accessed = PB.decode(d, Float64)
        elseif field_number == 24
            PB.decode!(d, memory_accessed_breakdown)
        elseif field_number == 25
            is_eager = PB.decode(d, Bool)
        elseif field_number == 26
            autotuned = PB.decode(d, Bool)
        elseif field_number == 27
            num_cores = PB.decode(d, UInt32)
        elseif field_number == 28
            core_type = PB.decode(d, var"FlatOpMetrics.TpuCoreType".T)
        elseif field_number == 29
            is_fusion_child = PB.decode(d, Bool)
        elseif field_number == 30
            PB.decode!(d, source_info)
        elseif field_number == 31
            estimated_flops = PB.decode(d, Float64)
        elseif field_number == 32
            estimated_bytes_accessed = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return FlatOpMetrics(op_id, parent_op_id, symbol_id, hlo_module_id, children_count, has_sc_children, hlo_name, long_name, category, provenance, deduplicated_name, self_time_ps, time_ps, total_time_ps, normalized_time_ps, min_time_ps, dma_stall_ps, occurrences, model_flops, model_flops_v2, flops_v2, flops, bytes_accessed, memory_accessed_breakdown[], is_eager, autotuned, num_cores, core_type, is_fusion_child, source_info[], estimated_flops, estimated_bytes_accessed)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::FlatOpMetrics)
    initpos = position(e.io)
    x.op_id != zero(UInt64) && PB.encode(e, 1, x.op_id)
    x.parent_op_id != zero(UInt64) && PB.encode(e, 2, x.parent_op_id)
    x.symbol_id != zero(UInt64) && PB.encode(e, 3, x.symbol_id)
    x.hlo_module_id != zero(UInt64) && PB.encode(e, 4, x.hlo_module_id)
    x.children_count != zero(UInt64) && PB.encode(e, 5, x.children_count)
    x.has_sc_children != false && PB.encode(e, 6, x.has_sc_children)
    !isempty(x.hlo_name) && PB.encode(e, 7, x.hlo_name)
    !isempty(x.long_name) && PB.encode(e, 8, x.long_name)
    !isempty(x.category) && PB.encode(e, 9, x.category)
    !isempty(x.provenance) && PB.encode(e, 10, x.provenance)
    !isempty(x.deduplicated_name) && PB.encode(e, 11, x.deduplicated_name)
    x.self_time_ps != zero(UInt64) && PB.encode(e, 12, x.self_time_ps)
    x.time_ps != zero(UInt64) && PB.encode(e, 13, x.time_ps)
    x.total_time_ps != zero(UInt64) && PB.encode(e, 14, x.total_time_ps)
    x.normalized_time_ps != zero(UInt64) && PB.encode(e, 15, x.normalized_time_ps)
    x.min_time_ps != zero(UInt64) && PB.encode(e, 16, x.min_time_ps)
    x.dma_stall_ps != zero(UInt64) && PB.encode(e, 17, x.dma_stall_ps)
    x.occurrences != zero(UInt64) && PB.encode(e, 18, x.occurrences)
    x.model_flops != zero(UInt64) && PB.encode(e, 19, x.model_flops)
    x.model_flops_v2 !== zero(Float64) && PB.encode(e, 20, x.model_flops_v2)
    x.flops_v2 !== zero(Float64) && PB.encode(e, 21, x.flops_v2)
    x.flops != zero(UInt64) && PB.encode(e, 22, x.flops)
    x.bytes_accessed !== zero(Float64) && PB.encode(e, 23, x.bytes_accessed)
    !isempty(x.memory_accessed_breakdown) && PB.encode(e, 24, x.memory_accessed_breakdown)
    x.is_eager != false && PB.encode(e, 25, x.is_eager)
    x.autotuned != false && PB.encode(e, 26, x.autotuned)
    x.num_cores != zero(UInt32) && PB.encode(e, 27, x.num_cores)
    x.core_type != var"FlatOpMetrics.TpuCoreType".UNKNOWN && PB.encode(e, 28, x.core_type)
    x.is_fusion_child != false && PB.encode(e, 29, x.is_fusion_child)
    !isnothing(x.source_info) && PB.encode(e, 30, x.source_info)
    x.estimated_flops !== zero(Float64) && PB.encode(e, 31, x.estimated_flops)
    x.estimated_bytes_accessed !== zero(Float64) && PB.encode(e, 32, x.estimated_bytes_accessed)
    return position(e.io) - initpos
end
function PB._encoded_size(x::FlatOpMetrics)
    encoded_size = 0
    x.op_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.op_id, 1))
    x.parent_op_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.parent_op_id, 2))
    x.symbol_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.symbol_id, 3))
    x.hlo_module_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.hlo_module_id, 4))
    x.children_count != zero(UInt64) && (encoded_size += PB._encoded_size(x.children_count, 5))
    x.has_sc_children != false && (encoded_size += PB._encoded_size(x.has_sc_children, 6))
    !isempty(x.hlo_name) && (encoded_size += PB._encoded_size(x.hlo_name, 7))
    !isempty(x.long_name) && (encoded_size += PB._encoded_size(x.long_name, 8))
    !isempty(x.category) && (encoded_size += PB._encoded_size(x.category, 9))
    !isempty(x.provenance) && (encoded_size += PB._encoded_size(x.provenance, 10))
    !isempty(x.deduplicated_name) && (encoded_size += PB._encoded_size(x.deduplicated_name, 11))
    x.self_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.self_time_ps, 12))
    x.time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.time_ps, 13))
    x.total_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.total_time_ps, 14))
    x.normalized_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.normalized_time_ps, 15))
    x.min_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.min_time_ps, 16))
    x.dma_stall_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.dma_stall_ps, 17))
    x.occurrences != zero(UInt64) && (encoded_size += PB._encoded_size(x.occurrences, 18))
    x.model_flops != zero(UInt64) && (encoded_size += PB._encoded_size(x.model_flops, 19))
    x.model_flops_v2 !== zero(Float64) && (encoded_size += PB._encoded_size(x.model_flops_v2, 20))
    x.flops_v2 !== zero(Float64) && (encoded_size += PB._encoded_size(x.flops_v2, 21))
    x.flops != zero(UInt64) && (encoded_size += PB._encoded_size(x.flops, 22))
    x.bytes_accessed !== zero(Float64) && (encoded_size += PB._encoded_size(x.bytes_accessed, 23))
    !isempty(x.memory_accessed_breakdown) && (encoded_size += PB._encoded_size(x.memory_accessed_breakdown, 24))
    x.is_eager != false && (encoded_size += PB._encoded_size(x.is_eager, 25))
    x.autotuned != false && (encoded_size += PB._encoded_size(x.autotuned, 26))
    x.num_cores != zero(UInt32) && (encoded_size += PB._encoded_size(x.num_cores, 27))
    x.core_type != var"FlatOpMetrics.TpuCoreType".UNKNOWN && (encoded_size += PB._encoded_size(x.core_type, 28))
    x.is_fusion_child != false && (encoded_size += PB._encoded_size(x.is_fusion_child, 29))
    !isnothing(x.source_info) && (encoded_size += PB._encoded_size(x.source_info, 30))
    x.estimated_flops !== zero(Float64) && (encoded_size += PB._encoded_size(x.estimated_flops, 31))
    x.estimated_bytes_accessed !== zero(Float64) && (encoded_size += PB._encoded_size(x.estimated_bytes_accessed, 32))
    return encoded_size
end

struct FlatOpMetricsDb
    op_instances::Vector{FlatOpMetrics}
    total_time_ps::UInt64
    profile_time_ps::UInt64
    total_op_time_ps::UInt64
    idle_time_ps::UInt64
    busy_time_ps::UInt64
    normalized_total_op_time_ps::UInt64
    precision_stats::Union{Nothing,PrecisionStats}
end
PB.default_values(::Type{FlatOpMetricsDb}) = (;op_instances = Vector{FlatOpMetrics}(), total_time_ps = zero(UInt64), profile_time_ps = zero(UInt64), total_op_time_ps = zero(UInt64), idle_time_ps = zero(UInt64), busy_time_ps = zero(UInt64), normalized_total_op_time_ps = zero(UInt64), precision_stats = nothing)
PB.field_numbers(::Type{FlatOpMetricsDb}) = (;op_instances = 1, total_time_ps = 2, profile_time_ps = 3, total_op_time_ps = 4, idle_time_ps = 5, busy_time_ps = 6, normalized_total_op_time_ps = 7, precision_stats = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:FlatOpMetricsDb}, _endpos::Int=0, _group::Bool=false)
    op_instances = PB.BufferedVector{FlatOpMetrics}()
    total_time_ps = zero(UInt64)
    profile_time_ps = zero(UInt64)
    total_op_time_ps = zero(UInt64)
    idle_time_ps = zero(UInt64)
    busy_time_ps = zero(UInt64)
    normalized_total_op_time_ps = zero(UInt64)
    precision_stats = Ref{Union{Nothing,PrecisionStats}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, op_instances)
        elseif field_number == 2
            total_time_ps = PB.decode(d, UInt64)
        elseif field_number == 3
            profile_time_ps = PB.decode(d, UInt64)
        elseif field_number == 4
            total_op_time_ps = PB.decode(d, UInt64)
        elseif field_number == 5
            idle_time_ps = PB.decode(d, UInt64)
        elseif field_number == 6
            busy_time_ps = PB.decode(d, UInt64)
        elseif field_number == 7
            normalized_total_op_time_ps = PB.decode(d, UInt64)
        elseif field_number == 8
            PB.decode!(d, precision_stats)
        else
            Base.skip(d, wire_type)
        end
    end
    return FlatOpMetricsDb(op_instances[], total_time_ps, profile_time_ps, total_op_time_ps, idle_time_ps, busy_time_ps, normalized_total_op_time_ps, precision_stats[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::FlatOpMetricsDb)
    initpos = position(e.io)
    !isempty(x.op_instances) && PB.encode(e, 1, x.op_instances)
    x.total_time_ps != zero(UInt64) && PB.encode(e, 2, x.total_time_ps)
    x.profile_time_ps != zero(UInt64) && PB.encode(e, 3, x.profile_time_ps)
    x.total_op_time_ps != zero(UInt64) && PB.encode(e, 4, x.total_op_time_ps)
    x.idle_time_ps != zero(UInt64) && PB.encode(e, 5, x.idle_time_ps)
    x.busy_time_ps != zero(UInt64) && PB.encode(e, 6, x.busy_time_ps)
    x.normalized_total_op_time_ps != zero(UInt64) && PB.encode(e, 7, x.normalized_total_op_time_ps)
    !isnothing(x.precision_stats) && PB.encode(e, 8, x.precision_stats)
    return position(e.io) - initpos
end
function PB._encoded_size(x::FlatOpMetricsDb)
    encoded_size = 0
    !isempty(x.op_instances) && (encoded_size += PB._encoded_size(x.op_instances, 1))
    x.total_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.total_time_ps, 2))
    x.profile_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.profile_time_ps, 3))
    x.total_op_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.total_op_time_ps, 4))
    x.idle_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.idle_time_ps, 5))
    x.busy_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.busy_time_ps, 6))
    x.normalized_total_op_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.normalized_total_op_time_ps, 7))
    !isnothing(x.precision_stats) && (encoded_size += PB._encoded_size(x.precision_stats, 8))
    return encoded_size
end

struct FlatOpMetricsDbMap
    program_to_db::Dict{UInt64,FlatOpMetricsDb}
end
PB.default_values(::Type{FlatOpMetricsDbMap}) = (;program_to_db = Dict{UInt64,FlatOpMetricsDb}())
PB.field_numbers(::Type{FlatOpMetricsDbMap}) = (;program_to_db = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:FlatOpMetricsDbMap}, _endpos::Int=0, _group::Bool=false)
    program_to_db = Dict{UInt64,FlatOpMetricsDb}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, program_to_db)
        else
            Base.skip(d, wire_type)
        end
    end
    return FlatOpMetricsDbMap(program_to_db)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::FlatOpMetricsDbMap)
    initpos = position(e.io)
    !isempty(x.program_to_db) && PB.encode(e, 1, x.program_to_db)
    return position(e.io) - initpos
end
function PB._encoded_size(x::FlatOpMetricsDbMap)
    encoded_size = 0
    !isempty(x.program_to_db) && (encoded_size += PB._encoded_size(x.program_to_db, 1))
    return encoded_size
end
