import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export TfStatsRecord, TfStatsTable, TfStatsDatabase


struct TfStatsRecord
    rank::UInt64
    host_or_device::String
    op_type::String
    op_name::String
    occurrences::Int64
    total_time_in_us::Float64
    avg_time_in_us::Float64
    total_self_time_in_us::Float64
    avg_self_time_in_us::Float64
    device_total_self_time_as_fraction::Float64
    device_cumulative_total_self_time_as_fraction::Float64
    host_total_self_time_as_fraction::Float64
    host_cumulative_total_self_time_as_fraction::Float64
    measured_flop_rate::Float64
    model_flop_rate::Float64
    measured_memory_bw::Float64
    operational_intensity::Float64
    bound_by::String
    is_eager::Bool
    gpu_tensorcore_utilization::Float64
    hbm_bw::Float64
    cmem_read_bw::Float64
    cmem_write_bw::Float64
    vmem_read_bw::Float64
    vmem_write_bw::Float64
    hbm_operational_intensity::Float64
    cmem_read_operational_intensity::Float64
    cmem_write_operational_intensity::Float64
    vmem_read_operational_intensity::Float64
    vmem_write_operational_intensity::Float64
    bottleneck_operational_intensity::Float64
    flops::UInt64
    flops_v2::Float64
    bytes_accessed::UInt64
end
PB.default_values(::Type{TfStatsRecord}) = (;rank = zero(UInt64), host_or_device = "", op_type = "", op_name = "", occurrences = zero(Int64), total_time_in_us = zero(Float64), avg_time_in_us = zero(Float64), total_self_time_in_us = zero(Float64), avg_self_time_in_us = zero(Float64), device_total_self_time_as_fraction = zero(Float64), device_cumulative_total_self_time_as_fraction = zero(Float64), host_total_self_time_as_fraction = zero(Float64), host_cumulative_total_self_time_as_fraction = zero(Float64), measured_flop_rate = zero(Float64), model_flop_rate = zero(Float64), measured_memory_bw = zero(Float64), operational_intensity = zero(Float64), bound_by = "", is_eager = false, gpu_tensorcore_utilization = zero(Float64), hbm_bw = zero(Float64), cmem_read_bw = zero(Float64), cmem_write_bw = zero(Float64), vmem_read_bw = zero(Float64), vmem_write_bw = zero(Float64), hbm_operational_intensity = zero(Float64), cmem_read_operational_intensity = zero(Float64), cmem_write_operational_intensity = zero(Float64), vmem_read_operational_intensity = zero(Float64), vmem_write_operational_intensity = zero(Float64), bottleneck_operational_intensity = zero(Float64), flops = zero(UInt64), flops_v2 = zero(Float64), bytes_accessed = zero(UInt64))
PB.field_numbers(::Type{TfStatsRecord}) = (;rank = 1, host_or_device = 2, op_type = 3, op_name = 4, occurrences = 5, total_time_in_us = 6, avg_time_in_us = 7, total_self_time_in_us = 8, avg_self_time_in_us = 9, device_total_self_time_as_fraction = 10, device_cumulative_total_self_time_as_fraction = 11, host_total_self_time_as_fraction = 12, host_cumulative_total_self_time_as_fraction = 13, measured_flop_rate = 14, model_flop_rate = 20, measured_memory_bw = 15, operational_intensity = 16, bound_by = 17, is_eager = 18, gpu_tensorcore_utilization = 19, hbm_bw = 21, cmem_read_bw = 22, cmem_write_bw = 23, vmem_read_bw = 24, vmem_write_bw = 25, hbm_operational_intensity = 26, cmem_read_operational_intensity = 27, cmem_write_operational_intensity = 28, vmem_read_operational_intensity = 29, vmem_write_operational_intensity = 30, bottleneck_operational_intensity = 31, flops = 32, flops_v2 = 34, bytes_accessed = 33)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TfStatsRecord}, _endpos::Int=0, _group::Bool=false)
    rank = zero(UInt64)
    host_or_device = ""
    op_type = ""
    op_name = ""
    occurrences = zero(Int64)
    total_time_in_us = zero(Float64)
    avg_time_in_us = zero(Float64)
    total_self_time_in_us = zero(Float64)
    avg_self_time_in_us = zero(Float64)
    device_total_self_time_as_fraction = zero(Float64)
    device_cumulative_total_self_time_as_fraction = zero(Float64)
    host_total_self_time_as_fraction = zero(Float64)
    host_cumulative_total_self_time_as_fraction = zero(Float64)
    measured_flop_rate = zero(Float64)
    model_flop_rate = zero(Float64)
    measured_memory_bw = zero(Float64)
    operational_intensity = zero(Float64)
    bound_by = ""
    is_eager = false
    gpu_tensorcore_utilization = zero(Float64)
    hbm_bw = zero(Float64)
    cmem_read_bw = zero(Float64)
    cmem_write_bw = zero(Float64)
    vmem_read_bw = zero(Float64)
    vmem_write_bw = zero(Float64)
    hbm_operational_intensity = zero(Float64)
    cmem_read_operational_intensity = zero(Float64)
    cmem_write_operational_intensity = zero(Float64)
    vmem_read_operational_intensity = zero(Float64)
    vmem_write_operational_intensity = zero(Float64)
    bottleneck_operational_intensity = zero(Float64)
    flops = zero(UInt64)
    flops_v2 = zero(Float64)
    bytes_accessed = zero(UInt64)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            rank = PB.decode(d, UInt64)
        elseif field_number == 2
            host_or_device = PB.decode(d, String)
        elseif field_number == 3
            op_type = PB.decode(d, String)
        elseif field_number == 4
            op_name = PB.decode(d, String)
        elseif field_number == 5
            occurrences = PB.decode(d, Int64)
        elseif field_number == 6
            total_time_in_us = PB.decode(d, Float64)
        elseif field_number == 7
            avg_time_in_us = PB.decode(d, Float64)
        elseif field_number == 8
            total_self_time_in_us = PB.decode(d, Float64)
        elseif field_number == 9
            avg_self_time_in_us = PB.decode(d, Float64)
        elseif field_number == 10
            device_total_self_time_as_fraction = PB.decode(d, Float64)
        elseif field_number == 11
            device_cumulative_total_self_time_as_fraction = PB.decode(d, Float64)
        elseif field_number == 12
            host_total_self_time_as_fraction = PB.decode(d, Float64)
        elseif field_number == 13
            host_cumulative_total_self_time_as_fraction = PB.decode(d, Float64)
        elseif field_number == 14
            measured_flop_rate = PB.decode(d, Float64)
        elseif field_number == 20
            model_flop_rate = PB.decode(d, Float64)
        elseif field_number == 15
            measured_memory_bw = PB.decode(d, Float64)
        elseif field_number == 16
            operational_intensity = PB.decode(d, Float64)
        elseif field_number == 17
            bound_by = PB.decode(d, String)
        elseif field_number == 18
            is_eager = PB.decode(d, Bool)
        elseif field_number == 19
            gpu_tensorcore_utilization = PB.decode(d, Float64)
        elseif field_number == 21
            hbm_bw = PB.decode(d, Float64)
        elseif field_number == 22
            cmem_read_bw = PB.decode(d, Float64)
        elseif field_number == 23
            cmem_write_bw = PB.decode(d, Float64)
        elseif field_number == 24
            vmem_read_bw = PB.decode(d, Float64)
        elseif field_number == 25
            vmem_write_bw = PB.decode(d, Float64)
        elseif field_number == 26
            hbm_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 27
            cmem_read_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 28
            cmem_write_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 29
            vmem_read_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 30
            vmem_write_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 31
            bottleneck_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 32
            flops = PB.decode(d, UInt64)
        elseif field_number == 34
            flops_v2 = PB.decode(d, Float64)
        elseif field_number == 33
            bytes_accessed = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return TfStatsRecord(rank, host_or_device, op_type, op_name, occurrences, total_time_in_us, avg_time_in_us, total_self_time_in_us, avg_self_time_in_us, device_total_self_time_as_fraction, device_cumulative_total_self_time_as_fraction, host_total_self_time_as_fraction, host_cumulative_total_self_time_as_fraction, measured_flop_rate, model_flop_rate, measured_memory_bw, operational_intensity, bound_by, is_eager, gpu_tensorcore_utilization, hbm_bw, cmem_read_bw, cmem_write_bw, vmem_read_bw, vmem_write_bw, hbm_operational_intensity, cmem_read_operational_intensity, cmem_write_operational_intensity, vmem_read_operational_intensity, vmem_write_operational_intensity, bottleneck_operational_intensity, flops, flops_v2, bytes_accessed)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TfStatsRecord)
    initpos = position(e.io)
    x.rank != zero(UInt64) && PB.encode(e, 1, x.rank)
    !isempty(x.host_or_device) && PB.encode(e, 2, x.host_or_device)
    !isempty(x.op_type) && PB.encode(e, 3, x.op_type)
    !isempty(x.op_name) && PB.encode(e, 4, x.op_name)
    x.occurrences != zero(Int64) && PB.encode(e, 5, x.occurrences)
    x.total_time_in_us !== zero(Float64) && PB.encode(e, 6, x.total_time_in_us)
    x.avg_time_in_us !== zero(Float64) && PB.encode(e, 7, x.avg_time_in_us)
    x.total_self_time_in_us !== zero(Float64) && PB.encode(e, 8, x.total_self_time_in_us)
    x.avg_self_time_in_us !== zero(Float64) && PB.encode(e, 9, x.avg_self_time_in_us)
    x.device_total_self_time_as_fraction !== zero(Float64) && PB.encode(e, 10, x.device_total_self_time_as_fraction)
    x.device_cumulative_total_self_time_as_fraction !== zero(Float64) && PB.encode(e, 11, x.device_cumulative_total_self_time_as_fraction)
    x.host_total_self_time_as_fraction !== zero(Float64) && PB.encode(e, 12, x.host_total_self_time_as_fraction)
    x.host_cumulative_total_self_time_as_fraction !== zero(Float64) && PB.encode(e, 13, x.host_cumulative_total_self_time_as_fraction)
    x.measured_flop_rate !== zero(Float64) && PB.encode(e, 14, x.measured_flop_rate)
    x.model_flop_rate !== zero(Float64) && PB.encode(e, 20, x.model_flop_rate)
    x.measured_memory_bw !== zero(Float64) && PB.encode(e, 15, x.measured_memory_bw)
    x.operational_intensity !== zero(Float64) && PB.encode(e, 16, x.operational_intensity)
    !isempty(x.bound_by) && PB.encode(e, 17, x.bound_by)
    x.is_eager != false && PB.encode(e, 18, x.is_eager)
    x.gpu_tensorcore_utilization !== zero(Float64) && PB.encode(e, 19, x.gpu_tensorcore_utilization)
    x.hbm_bw !== zero(Float64) && PB.encode(e, 21, x.hbm_bw)
    x.cmem_read_bw !== zero(Float64) && PB.encode(e, 22, x.cmem_read_bw)
    x.cmem_write_bw !== zero(Float64) && PB.encode(e, 23, x.cmem_write_bw)
    x.vmem_read_bw !== zero(Float64) && PB.encode(e, 24, x.vmem_read_bw)
    x.vmem_write_bw !== zero(Float64) && PB.encode(e, 25, x.vmem_write_bw)
    x.hbm_operational_intensity !== zero(Float64) && PB.encode(e, 26, x.hbm_operational_intensity)
    x.cmem_read_operational_intensity !== zero(Float64) && PB.encode(e, 27, x.cmem_read_operational_intensity)
    x.cmem_write_operational_intensity !== zero(Float64) && PB.encode(e, 28, x.cmem_write_operational_intensity)
    x.vmem_read_operational_intensity !== zero(Float64) && PB.encode(e, 29, x.vmem_read_operational_intensity)
    x.vmem_write_operational_intensity !== zero(Float64) && PB.encode(e, 30, x.vmem_write_operational_intensity)
    x.bottleneck_operational_intensity !== zero(Float64) && PB.encode(e, 31, x.bottleneck_operational_intensity)
    x.flops != zero(UInt64) && PB.encode(e, 32, x.flops)
    x.flops_v2 !== zero(Float64) && PB.encode(e, 34, x.flops_v2)
    x.bytes_accessed != zero(UInt64) && PB.encode(e, 33, x.bytes_accessed)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TfStatsRecord)
    encoded_size = 0
    x.rank != zero(UInt64) && (encoded_size += PB._encoded_size(x.rank, 1))
    !isempty(x.host_or_device) && (encoded_size += PB._encoded_size(x.host_or_device, 2))
    !isempty(x.op_type) && (encoded_size += PB._encoded_size(x.op_type, 3))
    !isempty(x.op_name) && (encoded_size += PB._encoded_size(x.op_name, 4))
    x.occurrences != zero(Int64) && (encoded_size += PB._encoded_size(x.occurrences, 5))
    x.total_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_time_in_us, 6))
    x.avg_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.avg_time_in_us, 7))
    x.total_self_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_self_time_in_us, 8))
    x.avg_self_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.avg_self_time_in_us, 9))
    x.device_total_self_time_as_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_total_self_time_as_fraction, 10))
    x.device_cumulative_total_self_time_as_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_cumulative_total_self_time_as_fraction, 11))
    x.host_total_self_time_as_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_total_self_time_as_fraction, 12))
    x.host_cumulative_total_self_time_as_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_cumulative_total_self_time_as_fraction, 13))
    x.measured_flop_rate !== zero(Float64) && (encoded_size += PB._encoded_size(x.measured_flop_rate, 14))
    x.model_flop_rate !== zero(Float64) && (encoded_size += PB._encoded_size(x.model_flop_rate, 20))
    x.measured_memory_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.measured_memory_bw, 15))
    x.operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.operational_intensity, 16))
    !isempty(x.bound_by) && (encoded_size += PB._encoded_size(x.bound_by, 17))
    x.is_eager != false && (encoded_size += PB._encoded_size(x.is_eager, 18))
    x.gpu_tensorcore_utilization !== zero(Float64) && (encoded_size += PB._encoded_size(x.gpu_tensorcore_utilization, 19))
    x.hbm_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.hbm_bw, 21))
    x.cmem_read_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_read_bw, 22))
    x.cmem_write_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_write_bw, 23))
    x.vmem_read_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_read_bw, 24))
    x.vmem_write_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_write_bw, 25))
    x.hbm_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.hbm_operational_intensity, 26))
    x.cmem_read_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_read_operational_intensity, 27))
    x.cmem_write_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_write_operational_intensity, 28))
    x.vmem_read_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_read_operational_intensity, 29))
    x.vmem_write_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_write_operational_intensity, 30))
    x.bottleneck_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.bottleneck_operational_intensity, 31))
    x.flops != zero(UInt64) && (encoded_size += PB._encoded_size(x.flops, 32))
    x.flops_v2 !== zero(Float64) && (encoded_size += PB._encoded_size(x.flops_v2, 34))
    x.bytes_accessed != zero(UInt64) && (encoded_size += PB._encoded_size(x.bytes_accessed, 33))
    return encoded_size
end

struct TfStatsTable
    tf_stats_record::Vector{TfStatsRecord}
    host_tf_pprof_key::String
    device_tf_pprof_key::String
end
PB.default_values(::Type{TfStatsTable}) = (;tf_stats_record = Vector{TfStatsRecord}(), host_tf_pprof_key = "", device_tf_pprof_key = "")
PB.field_numbers(::Type{TfStatsTable}) = (;tf_stats_record = 1, host_tf_pprof_key = 2, device_tf_pprof_key = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TfStatsTable}, _endpos::Int=0, _group::Bool=false)
    tf_stats_record = PB.BufferedVector{TfStatsRecord}()
    host_tf_pprof_key = ""
    device_tf_pprof_key = ""
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, tf_stats_record)
        elseif field_number == 2
            host_tf_pprof_key = PB.decode(d, String)
        elseif field_number == 3
            device_tf_pprof_key = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return TfStatsTable(tf_stats_record[], host_tf_pprof_key, device_tf_pprof_key)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TfStatsTable)
    initpos = position(e.io)
    !isempty(x.tf_stats_record) && PB.encode(e, 1, x.tf_stats_record)
    !isempty(x.host_tf_pprof_key) && PB.encode(e, 2, x.host_tf_pprof_key)
    !isempty(x.device_tf_pprof_key) && PB.encode(e, 3, x.device_tf_pprof_key)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TfStatsTable)
    encoded_size = 0
    !isempty(x.tf_stats_record) && (encoded_size += PB._encoded_size(x.tf_stats_record, 1))
    !isempty(x.host_tf_pprof_key) && (encoded_size += PB._encoded_size(x.host_tf_pprof_key, 2))
    !isempty(x.device_tf_pprof_key) && (encoded_size += PB._encoded_size(x.device_tf_pprof_key, 3))
    return encoded_size
end

struct TfStatsDatabase
    with_idle::Union{Nothing,TfStatsTable}
    without_idle::Union{Nothing,TfStatsTable}
    device_type::String
end
PB.reserved_fields(::Type{TfStatsDatabase}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[1, 2, 3])
PB.default_values(::Type{TfStatsDatabase}) = (;with_idle = nothing, without_idle = nothing, device_type = "")
PB.field_numbers(::Type{TfStatsDatabase}) = (;with_idle = 4, without_idle = 5, device_type = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TfStatsDatabase}, _endpos::Int=0, _group::Bool=false)
    with_idle = Ref{Union{Nothing,TfStatsTable}}(nothing)
    without_idle = Ref{Union{Nothing,TfStatsTable}}(nothing)
    device_type = ""
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 4
            PB.decode!(d, with_idle)
        elseif field_number == 5
            PB.decode!(d, without_idle)
        elseif field_number == 6
            device_type = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return TfStatsDatabase(with_idle[], without_idle[], device_type)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TfStatsDatabase)
    initpos = position(e.io)
    !isnothing(x.with_idle) && PB.encode(e, 4, x.with_idle)
    !isnothing(x.without_idle) && PB.encode(e, 5, x.without_idle)
    !isempty(x.device_type) && PB.encode(e, 6, x.device_type)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TfStatsDatabase)
    encoded_size = 0
    !isnothing(x.with_idle) && (encoded_size += PB._encoded_size(x.with_idle, 4))
    !isnothing(x.without_idle) && (encoded_size += PB._encoded_size(x.without_idle, 5))
    !isempty(x.device_type) && (encoded_size += PB._encoded_size(x.device_type, 6))
    return encoded_size
end
