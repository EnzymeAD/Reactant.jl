import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export RecordType, RooflineModelRecord, RooflineModelDatabase


@enumx RecordType INVALID_RECORD_TYPE=0 ALL=1 AVERAGE_STEP=2 PER_STEP=3 ALL_HW=4

struct RooflineModelRecord
    record_type::RecordType.T
    step_num::UInt32
    rank::UInt64
    hlo_module_id::UInt64
    hlo_category::String
    hlo_name::String
    occurrences::Int64
    total_time_in_us::Float64
    total_time_per_core_in_us::Float64
    total_time_in_percentage::Float64
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
    optimal_flop_rate::Float64
    roofline_efficiency::Float64
    flop_rate_relative_to_hw_limit::Float64
    memory_bw_relative_to_hw_limit::Float64
    include_infeed_outfeed::Bool
    apply_time_scale_multiplier::Bool
    flops::UInt64
    bytes_accessed::UInt64
    source_info::Union{Nothing,tensorflow.profiler.SourceInfo}
end
RooflineModelRecord(;record_type = RecordType.INVALID_RECORD_TYPE, step_num = zero(UInt32), rank = zero(UInt64), hlo_module_id = zero(UInt64), hlo_category = "", hlo_name = "", occurrences = zero(Int64), total_time_in_us = zero(Float64), total_time_per_core_in_us = zero(Float64), total_time_in_percentage = zero(Float64), avg_time_in_us = zero(Float64), total_self_time_in_us = zero(Float64), avg_self_time_in_us = zero(Float64), total_self_time_as_fraction = zero(Float64), cumulative_total_self_time_as_fraction = zero(Float64), dma_stall_fraction = zero(Float64), measured_flop_rate = zero(Float64), model_flop_rate = zero(Float64), measured_memory_bw = zero(Float64), hbm_bw = zero(Float64), cmem_read_bw = zero(Float64), cmem_write_bw = zero(Float64), vmem_read_bw = zero(Float64), vmem_write_bw = zero(Float64), operational_intensity = zero(Float64), hbm_operational_intensity = zero(Float64), cmem_read_operational_intensity = zero(Float64), cmem_write_operational_intensity = zero(Float64), vmem_read_operational_intensity = zero(Float64), vmem_write_operational_intensity = zero(Float64), bottleneck_operational_intensity = zero(Float64), bound_by = "", optimal_flop_rate = zero(Float64), roofline_efficiency = zero(Float64), flop_rate_relative_to_hw_limit = zero(Float64), memory_bw_relative_to_hw_limit = zero(Float64), include_infeed_outfeed = false, apply_time_scale_multiplier = false, flops = zero(UInt64), bytes_accessed = zero(UInt64), source_info = nothing) = RooflineModelRecord(record_type, step_num, rank, hlo_module_id, hlo_category, hlo_name, occurrences, total_time_in_us, total_time_per_core_in_us, total_time_in_percentage, avg_time_in_us, total_self_time_in_us, avg_self_time_in_us, total_self_time_as_fraction, cumulative_total_self_time_as_fraction, dma_stall_fraction, measured_flop_rate, model_flop_rate, measured_memory_bw, hbm_bw, cmem_read_bw, cmem_write_bw, vmem_read_bw, vmem_write_bw, operational_intensity, hbm_operational_intensity, cmem_read_operational_intensity, cmem_write_operational_intensity, vmem_read_operational_intensity, vmem_write_operational_intensity, bottleneck_operational_intensity, bound_by, optimal_flop_rate, roofline_efficiency, flop_rate_relative_to_hw_limit, memory_bw_relative_to_hw_limit, include_infeed_outfeed, apply_time_scale_multiplier, flops, bytes_accessed, source_info)
PB.reserved_fields(::Type{RooflineModelRecord}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[11, 12, 23])
PB.default_values(::Type{RooflineModelRecord}) = (;record_type = RecordType.INVALID_RECORD_TYPE, step_num = zero(UInt32), rank = zero(UInt64), hlo_module_id = zero(UInt64), hlo_category = "", hlo_name = "", occurrences = zero(Int64), total_time_in_us = zero(Float64), total_time_per_core_in_us = zero(Float64), total_time_in_percentage = zero(Float64), avg_time_in_us = zero(Float64), total_self_time_in_us = zero(Float64), avg_self_time_in_us = zero(Float64), total_self_time_as_fraction = zero(Float64), cumulative_total_self_time_as_fraction = zero(Float64), dma_stall_fraction = zero(Float64), measured_flop_rate = zero(Float64), model_flop_rate = zero(Float64), measured_memory_bw = zero(Float64), hbm_bw = zero(Float64), cmem_read_bw = zero(Float64), cmem_write_bw = zero(Float64), vmem_read_bw = zero(Float64), vmem_write_bw = zero(Float64), operational_intensity = zero(Float64), hbm_operational_intensity = zero(Float64), cmem_read_operational_intensity = zero(Float64), cmem_write_operational_intensity = zero(Float64), vmem_read_operational_intensity = zero(Float64), vmem_write_operational_intensity = zero(Float64), bottleneck_operational_intensity = zero(Float64), bound_by = "", optimal_flop_rate = zero(Float64), roofline_efficiency = zero(Float64), flop_rate_relative_to_hw_limit = zero(Float64), memory_bw_relative_to_hw_limit = zero(Float64), include_infeed_outfeed = false, apply_time_scale_multiplier = false, flops = zero(UInt64), bytes_accessed = zero(UInt64), source_info = nothing)
PB.field_numbers(::Type{RooflineModelRecord}) = (;record_type = 18, step_num = 19, rank = 1, hlo_module_id = 35, hlo_category = 17, hlo_name = 2, occurrences = 3, total_time_in_us = 4, total_time_per_core_in_us = 20, total_time_in_percentage = 21, avg_time_in_us = 5, total_self_time_in_us = 6, avg_self_time_in_us = 7, total_self_time_as_fraction = 8, cumulative_total_self_time_as_fraction = 9, dma_stall_fraction = 10, measured_flop_rate = 13, model_flop_rate = 38, measured_memory_bw = 14, hbm_bw = 27, cmem_read_bw = 28, cmem_write_bw = 29, vmem_read_bw = 39, vmem_write_bw = 40, operational_intensity = 15, hbm_operational_intensity = 30, cmem_read_operational_intensity = 31, cmem_write_operational_intensity = 32, vmem_read_operational_intensity = 41, vmem_write_operational_intensity = 42, bottleneck_operational_intensity = 33, bound_by = 16, optimal_flop_rate = 22, roofline_efficiency = 34, flop_rate_relative_to_hw_limit = 24, memory_bw_relative_to_hw_limit = 25, include_infeed_outfeed = 26, apply_time_scale_multiplier = 44, flops = 36, bytes_accessed = 37, source_info = 43)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:RooflineModelRecord})
    record_type = RecordType.INVALID_RECORD_TYPE
    step_num = zero(UInt32)
    rank = zero(UInt64)
    hlo_module_id = zero(UInt64)
    hlo_category = ""
    hlo_name = ""
    occurrences = zero(Int64)
    total_time_in_us = zero(Float64)
    total_time_per_core_in_us = zero(Float64)
    total_time_in_percentage = zero(Float64)
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
    optimal_flop_rate = zero(Float64)
    roofline_efficiency = zero(Float64)
    flop_rate_relative_to_hw_limit = zero(Float64)
    memory_bw_relative_to_hw_limit = zero(Float64)
    include_infeed_outfeed = false
    apply_time_scale_multiplier = false
    flops = zero(UInt64)
    bytes_accessed = zero(UInt64)
    source_info = Ref{Union{Nothing,tensorflow.profiler.SourceInfo}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 18
            record_type = PB.decode(d, RecordType.T)
        elseif field_number == 19
            step_num = PB.decode(d, UInt32)
        elseif field_number == 1
            rank = PB.decode(d, UInt64)
        elseif field_number == 35
            hlo_module_id = PB.decode(d, UInt64)
        elseif field_number == 17
            hlo_category = PB.decode(d, String)
        elseif field_number == 2
            hlo_name = PB.decode(d, String)
        elseif field_number == 3
            occurrences = PB.decode(d, Int64)
        elseif field_number == 4
            total_time_in_us = PB.decode(d, Float64)
        elseif field_number == 20
            total_time_per_core_in_us = PB.decode(d, Float64)
        elseif field_number == 21
            total_time_in_percentage = PB.decode(d, Float64)
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
        elseif field_number == 38
            model_flop_rate = PB.decode(d, Float64)
        elseif field_number == 14
            measured_memory_bw = PB.decode(d, Float64)
        elseif field_number == 27
            hbm_bw = PB.decode(d, Float64)
        elseif field_number == 28
            cmem_read_bw = PB.decode(d, Float64)
        elseif field_number == 29
            cmem_write_bw = PB.decode(d, Float64)
        elseif field_number == 39
            vmem_read_bw = PB.decode(d, Float64)
        elseif field_number == 40
            vmem_write_bw = PB.decode(d, Float64)
        elseif field_number == 15
            operational_intensity = PB.decode(d, Float64)
        elseif field_number == 30
            hbm_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 31
            cmem_read_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 32
            cmem_write_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 41
            vmem_read_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 42
            vmem_write_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 33
            bottleneck_operational_intensity = PB.decode(d, Float64)
        elseif field_number == 16
            bound_by = PB.decode(d, String)
        elseif field_number == 22
            optimal_flop_rate = PB.decode(d, Float64)
        elseif field_number == 34
            roofline_efficiency = PB.decode(d, Float64)
        elseif field_number == 24
            flop_rate_relative_to_hw_limit = PB.decode(d, Float64)
        elseif field_number == 25
            memory_bw_relative_to_hw_limit = PB.decode(d, Float64)
        elseif field_number == 26
            include_infeed_outfeed = PB.decode(d, Bool)
        elseif field_number == 44
            apply_time_scale_multiplier = PB.decode(d, Bool)
        elseif field_number == 36
            flops = PB.decode(d, UInt64)
        elseif field_number == 37
            bytes_accessed = PB.decode(d, UInt64)
        elseif field_number == 43
            PB.decode!(d, source_info)
        else
            Base.skip(d, wire_type)
        end
    end
    return RooflineModelRecord(record_type, step_num, rank, hlo_module_id, hlo_category, hlo_name, occurrences, total_time_in_us, total_time_per_core_in_us, total_time_in_percentage, avg_time_in_us, total_self_time_in_us, avg_self_time_in_us, total_self_time_as_fraction, cumulative_total_self_time_as_fraction, dma_stall_fraction, measured_flop_rate, model_flop_rate, measured_memory_bw, hbm_bw, cmem_read_bw, cmem_write_bw, vmem_read_bw, vmem_write_bw, operational_intensity, hbm_operational_intensity, cmem_read_operational_intensity, cmem_write_operational_intensity, vmem_read_operational_intensity, vmem_write_operational_intensity, bottleneck_operational_intensity, bound_by, optimal_flop_rate, roofline_efficiency, flop_rate_relative_to_hw_limit, memory_bw_relative_to_hw_limit, include_infeed_outfeed, apply_time_scale_multiplier, flops, bytes_accessed, source_info[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::RooflineModelRecord)
    initpos = position(e.io)
    x.record_type != RecordType.INVALID_RECORD_TYPE && PB.encode(e, 18, x.record_type)
    x.step_num != zero(UInt32) && PB.encode(e, 19, x.step_num)
    x.rank != zero(UInt64) && PB.encode(e, 1, x.rank)
    x.hlo_module_id != zero(UInt64) && PB.encode(e, 35, x.hlo_module_id)
    !isempty(x.hlo_category) && PB.encode(e, 17, x.hlo_category)
    !isempty(x.hlo_name) && PB.encode(e, 2, x.hlo_name)
    x.occurrences != zero(Int64) && PB.encode(e, 3, x.occurrences)
    x.total_time_in_us !== zero(Float64) && PB.encode(e, 4, x.total_time_in_us)
    x.total_time_per_core_in_us !== zero(Float64) && PB.encode(e, 20, x.total_time_per_core_in_us)
    x.total_time_in_percentage !== zero(Float64) && PB.encode(e, 21, x.total_time_in_percentage)
    x.avg_time_in_us !== zero(Float64) && PB.encode(e, 5, x.avg_time_in_us)
    x.total_self_time_in_us !== zero(Float64) && PB.encode(e, 6, x.total_self_time_in_us)
    x.avg_self_time_in_us !== zero(Float64) && PB.encode(e, 7, x.avg_self_time_in_us)
    x.total_self_time_as_fraction !== zero(Float64) && PB.encode(e, 8, x.total_self_time_as_fraction)
    x.cumulative_total_self_time_as_fraction !== zero(Float64) && PB.encode(e, 9, x.cumulative_total_self_time_as_fraction)
    x.dma_stall_fraction !== zero(Float64) && PB.encode(e, 10, x.dma_stall_fraction)
    x.measured_flop_rate !== zero(Float64) && PB.encode(e, 13, x.measured_flop_rate)
    x.model_flop_rate !== zero(Float64) && PB.encode(e, 38, x.model_flop_rate)
    x.measured_memory_bw !== zero(Float64) && PB.encode(e, 14, x.measured_memory_bw)
    x.hbm_bw !== zero(Float64) && PB.encode(e, 27, x.hbm_bw)
    x.cmem_read_bw !== zero(Float64) && PB.encode(e, 28, x.cmem_read_bw)
    x.cmem_write_bw !== zero(Float64) && PB.encode(e, 29, x.cmem_write_bw)
    x.vmem_read_bw !== zero(Float64) && PB.encode(e, 39, x.vmem_read_bw)
    x.vmem_write_bw !== zero(Float64) && PB.encode(e, 40, x.vmem_write_bw)
    x.operational_intensity !== zero(Float64) && PB.encode(e, 15, x.operational_intensity)
    x.hbm_operational_intensity !== zero(Float64) && PB.encode(e, 30, x.hbm_operational_intensity)
    x.cmem_read_operational_intensity !== zero(Float64) && PB.encode(e, 31, x.cmem_read_operational_intensity)
    x.cmem_write_operational_intensity !== zero(Float64) && PB.encode(e, 32, x.cmem_write_operational_intensity)
    x.vmem_read_operational_intensity !== zero(Float64) && PB.encode(e, 41, x.vmem_read_operational_intensity)
    x.vmem_write_operational_intensity !== zero(Float64) && PB.encode(e, 42, x.vmem_write_operational_intensity)
    x.bottleneck_operational_intensity !== zero(Float64) && PB.encode(e, 33, x.bottleneck_operational_intensity)
    !isempty(x.bound_by) && PB.encode(e, 16, x.bound_by)
    x.optimal_flop_rate !== zero(Float64) && PB.encode(e, 22, x.optimal_flop_rate)
    x.roofline_efficiency !== zero(Float64) && PB.encode(e, 34, x.roofline_efficiency)
    x.flop_rate_relative_to_hw_limit !== zero(Float64) && PB.encode(e, 24, x.flop_rate_relative_to_hw_limit)
    x.memory_bw_relative_to_hw_limit !== zero(Float64) && PB.encode(e, 25, x.memory_bw_relative_to_hw_limit)
    x.include_infeed_outfeed != false && PB.encode(e, 26, x.include_infeed_outfeed)
    x.apply_time_scale_multiplier != false && PB.encode(e, 44, x.apply_time_scale_multiplier)
    x.flops != zero(UInt64) && PB.encode(e, 36, x.flops)
    x.bytes_accessed != zero(UInt64) && PB.encode(e, 37, x.bytes_accessed)
    !isnothing(x.source_info) && PB.encode(e, 43, x.source_info)
    return position(e.io) - initpos
end
function PB._encoded_size(x::RooflineModelRecord)
    encoded_size = 0
    x.record_type != RecordType.INVALID_RECORD_TYPE && (encoded_size += PB._encoded_size(x.record_type, 18))
    x.step_num != zero(UInt32) && (encoded_size += PB._encoded_size(x.step_num, 19))
    x.rank != zero(UInt64) && (encoded_size += PB._encoded_size(x.rank, 1))
    x.hlo_module_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.hlo_module_id, 35))
    !isempty(x.hlo_category) && (encoded_size += PB._encoded_size(x.hlo_category, 17))
    !isempty(x.hlo_name) && (encoded_size += PB._encoded_size(x.hlo_name, 2))
    x.occurrences != zero(Int64) && (encoded_size += PB._encoded_size(x.occurrences, 3))
    x.total_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_time_in_us, 4))
    x.total_time_per_core_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_time_per_core_in_us, 20))
    x.total_time_in_percentage !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_time_in_percentage, 21))
    x.avg_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.avg_time_in_us, 5))
    x.total_self_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_self_time_in_us, 6))
    x.avg_self_time_in_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.avg_self_time_in_us, 7))
    x.total_self_time_as_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_self_time_as_fraction, 8))
    x.cumulative_total_self_time_as_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.cumulative_total_self_time_as_fraction, 9))
    x.dma_stall_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.dma_stall_fraction, 10))
    x.measured_flop_rate !== zero(Float64) && (encoded_size += PB._encoded_size(x.measured_flop_rate, 13))
    x.model_flop_rate !== zero(Float64) && (encoded_size += PB._encoded_size(x.model_flop_rate, 38))
    x.measured_memory_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.measured_memory_bw, 14))
    x.hbm_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.hbm_bw, 27))
    x.cmem_read_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_read_bw, 28))
    x.cmem_write_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_write_bw, 29))
    x.vmem_read_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_read_bw, 39))
    x.vmem_write_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_write_bw, 40))
    x.operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.operational_intensity, 15))
    x.hbm_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.hbm_operational_intensity, 30))
    x.cmem_read_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_read_operational_intensity, 31))
    x.cmem_write_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.cmem_write_operational_intensity, 32))
    x.vmem_read_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_read_operational_intensity, 41))
    x.vmem_write_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.vmem_write_operational_intensity, 42))
    x.bottleneck_operational_intensity !== zero(Float64) && (encoded_size += PB._encoded_size(x.bottleneck_operational_intensity, 33))
    !isempty(x.bound_by) && (encoded_size += PB._encoded_size(x.bound_by, 16))
    x.optimal_flop_rate !== zero(Float64) && (encoded_size += PB._encoded_size(x.optimal_flop_rate, 22))
    x.roofline_efficiency !== zero(Float64) && (encoded_size += PB._encoded_size(x.roofline_efficiency, 34))
    x.flop_rate_relative_to_hw_limit !== zero(Float64) && (encoded_size += PB._encoded_size(x.flop_rate_relative_to_hw_limit, 24))
    x.memory_bw_relative_to_hw_limit !== zero(Float64) && (encoded_size += PB._encoded_size(x.memory_bw_relative_to_hw_limit, 25))
    x.include_infeed_outfeed != false && (encoded_size += PB._encoded_size(x.include_infeed_outfeed, 26))
    x.apply_time_scale_multiplier != false && (encoded_size += PB._encoded_size(x.apply_time_scale_multiplier, 44))
    x.flops != zero(UInt64) && (encoded_size += PB._encoded_size(x.flops, 36))
    x.bytes_accessed != zero(UInt64) && (encoded_size += PB._encoded_size(x.bytes_accessed, 37))
    !isnothing(x.source_info) && (encoded_size += PB._encoded_size(x.source_info, 43))
    return encoded_size
end

struct RooflineModelDatabase
    device_type::String
    megacore::Bool
    has_cmem::Bool
    has_merged_vmem::Bool
    peak_flop_rate::Float64
    peak_hbm_bw::Float64
    peak_cmem_read_bw::Float64
    peak_cmem_write_bw::Float64
    peak_vmem_read_bw::Float64
    peak_vmem_write_bw::Float64
    roofline_model_record::Vector{RooflineModelRecord}
    diagnostics::Union{Nothing,tensorflow.profiler.Diagnostics}
    time_scale_multiplier::Float64
end
RooflineModelDatabase(;device_type = "", megacore = false, has_cmem = false, has_merged_vmem = false, peak_flop_rate = zero(Float64), peak_hbm_bw = zero(Float64), peak_cmem_read_bw = zero(Float64), peak_cmem_write_bw = zero(Float64), peak_vmem_read_bw = zero(Float64), peak_vmem_write_bw = zero(Float64), roofline_model_record = Vector{RooflineModelRecord}(), diagnostics = nothing, time_scale_multiplier = zero(Float64)) = RooflineModelDatabase(device_type, megacore, has_cmem, has_merged_vmem, peak_flop_rate, peak_hbm_bw, peak_cmem_read_bw, peak_cmem_write_bw, peak_vmem_read_bw, peak_vmem_write_bw, roofline_model_record, diagnostics, time_scale_multiplier)
PB.reserved_fields(::Type{RooflineModelDatabase}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[3, 4, 6])
PB.default_values(::Type{RooflineModelDatabase}) = (;device_type = "", megacore = false, has_cmem = false, has_merged_vmem = false, peak_flop_rate = zero(Float64), peak_hbm_bw = zero(Float64), peak_cmem_read_bw = zero(Float64), peak_cmem_write_bw = zero(Float64), peak_vmem_read_bw = zero(Float64), peak_vmem_write_bw = zero(Float64), roofline_model_record = Vector{RooflineModelRecord}(), diagnostics = nothing, time_scale_multiplier = zero(Float64))
PB.field_numbers(::Type{RooflineModelDatabase}) = (;device_type = 1, megacore = 12, has_cmem = 8, has_merged_vmem = 15, peak_flop_rate = 2, peak_hbm_bw = 9, peak_cmem_read_bw = 10, peak_cmem_write_bw = 11, peak_vmem_read_bw = 13, peak_vmem_write_bw = 14, roofline_model_record = 5, diagnostics = 7, time_scale_multiplier = 16)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:RooflineModelDatabase})
    device_type = ""
    megacore = false
    has_cmem = false
    has_merged_vmem = false
    peak_flop_rate = zero(Float64)
    peak_hbm_bw = zero(Float64)
    peak_cmem_read_bw = zero(Float64)
    peak_cmem_write_bw = zero(Float64)
    peak_vmem_read_bw = zero(Float64)
    peak_vmem_write_bw = zero(Float64)
    roofline_model_record = PB.BufferedVector{RooflineModelRecord}()
    diagnostics = Ref{Union{Nothing,tensorflow.profiler.Diagnostics}}(nothing)
    time_scale_multiplier = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            device_type = PB.decode(d, String)
        elseif field_number == 12
            megacore = PB.decode(d, Bool)
        elseif field_number == 8
            has_cmem = PB.decode(d, Bool)
        elseif field_number == 15
            has_merged_vmem = PB.decode(d, Bool)
        elseif field_number == 2
            peak_flop_rate = PB.decode(d, Float64)
        elseif field_number == 9
            peak_hbm_bw = PB.decode(d, Float64)
        elseif field_number == 10
            peak_cmem_read_bw = PB.decode(d, Float64)
        elseif field_number == 11
            peak_cmem_write_bw = PB.decode(d, Float64)
        elseif field_number == 13
            peak_vmem_read_bw = PB.decode(d, Float64)
        elseif field_number == 14
            peak_vmem_write_bw = PB.decode(d, Float64)
        elseif field_number == 5
            PB.decode!(d, roofline_model_record)
        elseif field_number == 7
            PB.decode!(d, diagnostics)
        elseif field_number == 16
            time_scale_multiplier = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return RooflineModelDatabase(device_type, megacore, has_cmem, has_merged_vmem, peak_flop_rate, peak_hbm_bw, peak_cmem_read_bw, peak_cmem_write_bw, peak_vmem_read_bw, peak_vmem_write_bw, roofline_model_record[], diagnostics[], time_scale_multiplier)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::RooflineModelDatabase)
    initpos = position(e.io)
    !isempty(x.device_type) && PB.encode(e, 1, x.device_type)
    x.megacore != false && PB.encode(e, 12, x.megacore)
    x.has_cmem != false && PB.encode(e, 8, x.has_cmem)
    x.has_merged_vmem != false && PB.encode(e, 15, x.has_merged_vmem)
    x.peak_flop_rate !== zero(Float64) && PB.encode(e, 2, x.peak_flop_rate)
    x.peak_hbm_bw !== zero(Float64) && PB.encode(e, 9, x.peak_hbm_bw)
    x.peak_cmem_read_bw !== zero(Float64) && PB.encode(e, 10, x.peak_cmem_read_bw)
    x.peak_cmem_write_bw !== zero(Float64) && PB.encode(e, 11, x.peak_cmem_write_bw)
    x.peak_vmem_read_bw !== zero(Float64) && PB.encode(e, 13, x.peak_vmem_read_bw)
    x.peak_vmem_write_bw !== zero(Float64) && PB.encode(e, 14, x.peak_vmem_write_bw)
    !isempty(x.roofline_model_record) && PB.encode(e, 5, x.roofline_model_record)
    !isnothing(x.diagnostics) && PB.encode(e, 7, x.diagnostics)
    x.time_scale_multiplier !== zero(Float64) && PB.encode(e, 16, x.time_scale_multiplier)
    return position(e.io) - initpos
end
function PB._encoded_size(x::RooflineModelDatabase)
    encoded_size = 0
    !isempty(x.device_type) && (encoded_size += PB._encoded_size(x.device_type, 1))
    x.megacore != false && (encoded_size += PB._encoded_size(x.megacore, 12))
    x.has_cmem != false && (encoded_size += PB._encoded_size(x.has_cmem, 8))
    x.has_merged_vmem != false && (encoded_size += PB._encoded_size(x.has_merged_vmem, 15))
    x.peak_flop_rate !== zero(Float64) && (encoded_size += PB._encoded_size(x.peak_flop_rate, 2))
    x.peak_hbm_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.peak_hbm_bw, 9))
    x.peak_cmem_read_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.peak_cmem_read_bw, 10))
    x.peak_cmem_write_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.peak_cmem_write_bw, 11))
    x.peak_vmem_read_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.peak_vmem_read_bw, 13))
    x.peak_vmem_write_bw !== zero(Float64) && (encoded_size += PB._encoded_size(x.peak_vmem_write_bw, 14))
    !isempty(x.roofline_model_record) && (encoded_size += PB._encoded_size(x.roofline_model_record, 5))
    !isnothing(x.diagnostics) && (encoded_size += PB._encoded_size(x.diagnostics, 7))
    x.time_scale_multiplier !== zero(Float64) && (encoded_size += PB._encoded_size(x.time_scale_multiplier, 16))
    return encoded_size
end
