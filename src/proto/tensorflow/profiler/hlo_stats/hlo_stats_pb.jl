import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export HloStatsRecord, HloStatsDatabase


struct HloStatsRecord
    rank::UInt64
    program_id::UInt64
    hlo_category::String
    hlo_expression::String
    tf_op_name::String
    occurrences::Int64
    total_time_in_us::Float64
    avg_time_in_us::Float64
    total_self_time_in_us::Float64
    avg_self_time_in_us::Float64
    total_self_time_as_fraction::Float64
    cumulative_total_self_time_as_fraction::Float64
    dma_stall_fraction::Float64
    measured_flop_rate::Float64
    model_flop_rate::Float64
    measured_memory_bw::Float64
    hbm_bw::Float64
    cmem_read_bw::Float64
    cmem_write_bw::Float64
    vmem_read_bw::Float64
    vmem_write_bw::Float64
    operational_intensity::Float64
    hbm_operational_intensity::Float64
    cmem_read_operational_intensity::Float64
    cmem_write_operational_intensity::Float64
    vmem_read_operational_intensity::Float64
    vmem_write_operational_intensity::Float64
    bottleneck_operational_intensity::Float64
    bound_by::String
    rematerialization::Bool
    outside_compilation::Bool
    autotuned::Bool
    flops::UInt64
    bytes_accessed::UInt64
    source_info::Union{Nothing,tensorflow.profiler.SourceInfo}
end
HloStatsRecord(;rank = zero(UInt64), program_id = zero(UInt64), hlo_category = "", hlo_expression = "", tf_op_name = "", occurrences = zero(Int64), total_time_in_us = zero(Float64), avg_time_in_us = zero(Float64), total_self_time_in_us = zero(Float64), avg_self_time_in_us = zero(Float64), total_self_time_as_fraction = zero(Float64), cumulative_total_self_time_as_fraction = zero(Float64), dma_stall_fraction = zero(Float64), measured_flop_rate = zero(Float64), model_flop_rate = zero(Float64), measured_memory_bw = zero(Float64), hbm_bw = zero(Float64), cmem_read_bw = zero(Float64), cmem_write_bw = zero(Float64), vmem_read_bw = zero(Float64), vmem_write_bw = zero(Float64), operational_intensity = zero(Float64), hbm_operational_intensity = zero(Float64), cmem_read_operational_intensity = zero(Float64), cmem_write_operational_intensity = zero(Float64), vmem_read_operational_intensity = zero(Float64), vmem_write_operational_intensity = zero(Float64), bottleneck_operational_intensity = zero(Float64), bound_by = "", rematerialization = false, outside_compilation = false, autotuned = false, flops = zero(UInt64), bytes_accessed = zero(UInt64), source_info = nothing) = HloStatsRecord(rank, program_id, hlo_category, hlo_expression, tf_op_name, occurrences, total_time_in_us, avg_time_in_us, total_self_time_in_us, avg_self_time_in_us, total_self_time_as_fraction, cumulative_total_self_time_as_fraction, dma_stall_fraction, measured_flop_rate, model_flop_rate, measured_memory_bw, hbm_bw, cmem_read_bw, cmem_write_bw, vmem_read_bw, vmem_write_bw, operational_intensity, hbm_operational_intensity, cmem_read_operational_intensity, cmem_write_operational_intensity, vmem_read_operational_intensity, vmem_write_operational_intensity, bottleneck_operational_intensity, bound_by, rematerialization, outside_compilation, autotuned, flops, bytes_accessed, source_info)
PB.reserved_fields(::Type{HloStatsRecord}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[11, 12, 18, 19])
PB.default_values(::Type{HloStatsRecord}) = (;rank = zero(UInt64), program_id = zero(UInt64), hlo_category = "", hlo_expression = "", tf_op_name = "", occurrences = zero(Int64), total_time_in_us = zero(Float64), avg_time_in_us = zero(Float64), total_self_time_in_us = zero(Float64), avg_self_time_in_us = zero(Float64), total_self_time_as_fraction = zero(Float64), cumulative_total_self_time_as_fraction = zero(Float64), dma_stall_fraction = zero(Float64), measured_flop_rate = zero(Float64), model_flop_rate = zero(Float64), measured_memory_bw = zero(Float64), hbm_bw = zero(Float64), cmem_read_bw = zero(Float64), cmem_write_bw = zero(Float64), vmem_read_bw = zero(Float64), vmem_write_bw = zero(Float64), operational_intensity = zero(Float64), hbm_operational_intensity = zero(Float64), cmem_read_operational_intensity = zero(Float64), cmem_write_operational_intensity = zero(Float64), vmem_read_operational_intensity = zero(Float64), vmem_write_operational_intensity = zero(Float64), bottleneck_operational_intensity = zero(Float64), bound_by = "", rematerialization = false, outside_compilation = false, autotuned = false, flops = zero(UInt64), bytes_accessed = zero(UInt64), source_info = nothing)
PB.field_numbers(::Type{HloStatsRecord}) = (;rank = 1, program_id = 30, hlo_category = 17, hlo_expression = 2, tf_op_name = 21, occurrences = 3, total_time_in_us = 4, avg_time_in_us = 5, total_self_time_in_us = 6, avg_self_time_in_us = 7, total_self_time_as_fraction = 8, cumulative_total_self_time_as_fraction = 9, dma_stall_fraction = 10, measured_flop_rate = 13, model_flop_rate = 34, measured_memory_bw = 14, hbm_bw = 22, cmem_read_bw = 23, cmem_write_bw = 24, vmem_read_bw = 35, vmem_write_bw = 36, operational_intensity = 15, hbm_operational_intensity = 26, cmem_read_operational_intensity = 27, cmem_write_operational_intensity = 28, vmem_read_operational_intensity = 37, vmem_write_operational_intensity = 38, bottleneck_operational_intensity = 29, bound_by = 16, rematerialization = 20, outside_compilation = 25, autotuned = 31, flops = 32, bytes_accessed = 33, source_info = 39)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloStatsRecord})
    rank = zero(UInt64)
    program_id = zero(UInt64)
    hlo_category = ""
    hlo_expression = ""
    tf_op_name = ""
    occurrences = zero(Int64)
    total_time_in_us = zero(Float64)
    avg_time_in_us = zero(Float64)
    total_self_time_in_us = zero(Float64)
    avg_self_time_in_us = zero(Float64)
    total_self_time_as_fraction = zero(Float64)
    cumulative_total_self_time_as_fraction = zero(Float64)
    dma_stall_fraction = zero(Float64)
    measured_flop_rate = zero(Float64)
    model_flop_rate = zero(Float64)
    measured_memory_bw = zero(Float64)
    hbm_bw = zero(Float64)
    cmem_read_bw = zero(Float64)
    cmem_write_bw = zero(Float64)
    vmem_read_bw = zero(Float64)
    vmem_write_bw = zero(Float64)
    operational_intensity = zero(Float64)
    hbm_operational_intensity = zero(Float64)
    cmem_read_operational_intensity = zero(Float64)
    cmem_write_operational_intensity = zero(Float64)
    vmem_read_operational_intensity = zero(Float64)
    vmem_write_operational_intensity = zero(Float64)
    bottleneck_operational_intensity = zero(Float64)
    bound_by = ""
    rematerialization = false
    outside_compilation = false
    autotuned = false
    flops = zero(UInt64)
    bytes_accessed = zero(UInt64)
    source_info = Ref{Union{Nothing,tensorflow.profiler.SourceInfo}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            rank = PB.decode(d, UInt64)
        elseif field_number == 30
            program_id = PB.decode(d, UInt64)
        elseif field_number == 17
            hlo_category = PB.decode(d, String)
        elseif field_number == 2
            hlo_expression = PB.decode(d, String)
        elseif field_number == 21
            tf_op_name = PB.decode(d, String)
        elseif field_number == 3
            occurrences = PB.decode(d, Int64)
        elseif field_number == 4
            total_time_in_us = PB.decode(d, Float64)
        elseif field_number == 5
            avg_time_in_us = PB.decode(d, Float64)
        elseif field_number == 6
            total_self_time_in_us = PB.decode(d, Float64)
        elseif field_number == 7
            avg_self_time_in_us = PB.decode(d, Float64)
        elseif field_number == 8
            total_self_time_as_fraction = PB.decode(d, Float64)
        elseif field_number == 9
            cumulative_total_self_time_as_fraction = PB.decode(d, Float64)
        elseif field_number == 10
            dma_stall_fraction = PB.decode(d, Float64)
        elseif field_number == 13
            measured_flop_rate = PB.decode(d, Float64)
        elseif field_number == 34
            model_flop_rate = PB.decode(d, Float64)
        elseif field_number == 14
            measured_memory_bw = PB.decode(d, Float64)
        elseif field_number == 22
            hbm_bw = PB.decode(d, Float64)
        elseif field_number == 23
            cmem_read_bw = PB.decode(d, Float64)
        elseif field_number == 24
            cmem_write_bw = PB.decode(d, Float64)
        elseif field_number == 35
            vmem_read_bw = PB.decode(d, Float64)
        elseif field_number == 36
            vmem_write_bw = PB.decode(d, Float64)
        elseif field_number == 15
            operational_intensity = PB.decode(d, Float64)
        elseif field_number == 26
            hbm_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 27
            cmem_read_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 28
            cmem_write_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 37
            vmem_read_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 38
            vmem_write_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 29
            bottleneck_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 16
            bound_by = PB.decode(d, String)
        elseif field_number == 20
            rematerialization = PB.decode(d, Bool)
        elseif field_number == 25
            outside_compilation = PB.decode(d, Bool)
        elseif field_number == 31
            autotuned = PB.decode(d, Bool)
        elseif field_number == 32
            flops = PB.decode(d, UInt64)
        elseif field_number == 33
            bytes_accessed = PB.decode(d, UInt64)
        elseif field_number == 39
            PB.decode!(d, source_info)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloStatsRecord(rank, program_id, hlo_category, hlo_expression, tf_op_name, occurrences, total_time_in_us, avg_time_in_us, total_self_time_in_us, avg_self_time_in_us, total_self_time_as_fraction, cumulative_total_self_time_as_fraction, dma_stall_fraction, measured_flop_rate, model_flop_rate, measured_memory_bw, hbm_bw, cmem_read_bw, cmem_write_bw, vmem_read_bw, vmem_write_bw, operational_intensity, hbm_operational_intensity, cmem_read_operational_intensity, cmem_write_operational_intensity, vmem_read_operational_intensity, vmem_write_operational_intensity, bottleneck_operational_intensity, bound_by, rematerialization, outside_compilation, autotuned, flops, bytes_accessed, source_info[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloStatsRecord)
    initpos = position(e.io)
    x.rank != zero(UInt64) && PB.encode(e, 1, x.rank)
    x.program_id != zero(UInt64) && PB.encode(e, 30, x.program_id)
    !isempty(x.hlo_category) && PB.encode(e, 17, x.hlo_category)
    !isempty(x.hlo_expression) && PB.encode(e, 2, x.hlo_expression)
    !isempty(x.tf_op_name) && PB.encode(e, 21, x.tf_op_name)
    x.occurrences != zero(Int64) && PB.encode(e, 3, x.occurrences)
    x.total_time_in_us !== zero(Float64) && PB.encode(e, 4, x.total_time_in_us)
    x.avg_time_in_us !== zero(Float64) && PB.encode(e, 5, x.avg_time_in_us)
    x.total_self_time_in_us !== zero(Float64) && PB.encode(e, 6, x.total_self_time_in_us)
    x.avg_self_time_in_us !== zero(Float64) && PB.encode(e, 7, x.avg_self_time_in_us)
    x.total_self_time_as_fraction !== zero(Float64) && PB.encode(e, 8, x.total_self_time_as_fraction)
    x.cumulative_total_self_time_as_fraction !== zero(Float64) && PB.encode(e, 9, x.cumulative_total_self_time_as_fraction)
    x.dma_stall_fraction !== zero(Float64) && PB.encode(e, 10, x.dma_stall_fraction)
    x.measured_flop_rate !== zero(Float64) && PB.encode(e, 13, x.measured_flop_rate)
    x.model_flop_rate !== zero(Float64) && PB.encode(e, 34, x.model_flop_rate)
    x.measured_memory_bw !== zero(Float64) && PB.encode(e, 14, x.measured_memory_bw)
    x.hbm_bw !== zero(Float64) && PB.encode(e, 22, x.hbm_bw)
    x.cmem_read_bw !== zero(Float64) && PB.encode(e, 23, x.cmem_read_bw)
    x.cmem_write_bw !== zero(Float64) && PB.encode(e, 24, x.cmem_write_bw)
    x.vmem_read_bw !== zero(Float64) && PB.encode(e, 35, x.vmem_read_bw)
    x.vmem_write_bw !== zero(Float64) && PB.encode(e, 36, x.vmem_write_bw)
    x.operational_intensity !== zero(Float64) && PB.encode(e, 15, x.operational_intensity)
    x.hbm_operational_intensity !== zero(Float64) && PB.encode(e, 26, x.hbm_operational_intensity)
    x.cmem_read_operational_intensity !== zero(Float64) && PB.encode(e, 27, x.cmem_read_operational_intensity)
    x.cmem_write_operational_intensity !== zero(Float64) && PB.encode(e, 28, x.cmem_write_operational_intensity)
    x.vmem_read_operational_intensity !== zero(Float64) && PB.encode(e, 37, x.vmem_read_operational_intensity)
    x.vmem_write_operational_intensity !== zero(Float64) && PB.encode(e, 38, x.vmem_write_operational_intensity)
    x.bottleneck_operational_intensity !== zero(Float64) && PB.encode(e, 29, x.bottleneck_operational_intensity)
    !isempty(x.bound_by) && PB.encode(e, 16, x.bound_by)
    x.rematerialization != false && PB.encode(e, 20, x.rematerialization)
    x.outside_compilation != false && PB.encode(e, 25, x.outside_compilation)
    x.autotuned != false && PB.encode(e, 31, x.autotuned)
    x.flops != zero(UInt64) && PB.encode(e, 32, x.flops)
    x.bytes_accessed != zero(UInt64) && PB.encode(e, 33, x.bytes_accessed)
    !isnothing(x.source_info) && PB.encode(e, 39, x.source_info)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloStatsRecord)
    encoded_size = 0
    x.rank != zero(UInt64) && (encoded_size += PB._encoded_size(x.rank, 1))
    x.program_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.program_id, 30))
    !isempty(x.hlo_category) && (encoded_size += PB._encoded_size(x.hlo_category, 17))
    !isempty(x.hlo_expression) && (encoded_size += PB._encoded_size(x.hlo_expression, 2))
    !isempty(x.tf_op_name) && (encoded_size += PB._encoded_size(x.tf_op_name, 21))
    x.occurrences != zero(Int64) && (encoded_size += PB._encoded_size(x.occurrences, 3))
    x.total_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_time_in_us, 4))
    x.avg_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.avg_time_in_us, 5))
    x.total_self_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_self_time_in_us, 6))
    x.avg_self_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.avg_self_time_in_us, 7))
    x.total_self_time_as_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_self_time_as_fraction, 8))
    x.cumulative_total_self_time_as_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.cumulative_total_self_time_as_fraction, 9))
    x.dma_stall_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.dma_stall_fraction, 10))
    x.measured_flop_rate !== zero(Float64) && (encoded_size += PB._encoded_size(x.measured_flop_rate, 13))
    x.model_flop_rate !== zero(Float64) && (encoded_size += PB._encoded_size(x.model_flop_rate, 34))
    x.measured_memory_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.measured_memory_bw, 14))
    x.hbm_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.hbm_bw, 22))
    x.cmem_read_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_read_bw, 23))
    x.cmem_write_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_write_bw, 24))
    x.vmem_read_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_read_bw, 35))
    x.vmem_write_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_write_bw, 36))
    x.operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.operational_intensity, 15))
    x.hbm_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.hbm_operational_intensity, 26))
    x.cmem_read_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_read_operational_intensity, 27))
    x.cmem_write_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_write_operational_intensity, 28))
    x.vmem_read_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_read_operational_intensity, 37))
    x.vmem_write_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_write_operational_intensity, 38))
    x.bottleneck_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.bottleneck_operational_intensity, 29))
    !isempty(x.bound_by) && (encoded_size += PB._encoded_size(x.bound_by, 16))
    x.rematerialization != false && (encoded_size += PB._encoded_size(x.rematerialization, 20))
    x.outside_compilation != false && (encoded_size += PB._encoded_size(x.outside_compilation, 25))
    x.autotuned != false && (encoded_size += PB._encoded_size(x.autotuned, 31))
    x.flops != zero(UInt64) && (encoded_size += PB._encoded_size(x.flops, 32))
    x.bytes_accessed != zero(UInt64) && (encoded_size += PB._encoded_size(x.bytes_accessed, 33))
    !isnothing(x.source_info) && (encoded_size += PB._encoded_size(x.source_info, 39))
    return encoded_size
end

struct HloStatsDatabase
    hlo_stats_record::Vector{HloStatsRecord}
end
HloStatsDatabase(;hlo_stats_record = Vector{HloStatsRecord}()) = HloStatsDatabase(hlo_stats_record)
PB.default_values(::Type{HloStatsDatabase}) = (;hlo_stats_record = Vector{HloStatsRecord}())
PB.field_numbers(::Type{HloStatsDatabase}) = (;hlo_stats_record = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloStatsDatabase})
    hlo_stats_record = PB.BufferedVector{HloStatsRecord}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, hlo_stats_record)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloStatsDatabase(hlo_stats_record[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloStatsDatabase)
    initpos = position(e.io)
    !isempty(x.hlo_stats_record) && PB.encode(e, 1, x.hlo_stats_record)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloStatsDatabase)
    encoded_size = 0
    !isempty(x.hlo_stats_record) && (encoded_size += PB._encoded_size(x.hlo_stats_record, 1))
    return encoded_size
end
