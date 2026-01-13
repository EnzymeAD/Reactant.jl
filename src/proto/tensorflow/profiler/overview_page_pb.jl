import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export OverviewTfOp, OverviewPageHostDependentJobInfo, OverviewLatencyBreakdown
export GenericRecommendation, OverviewPageTip, OverviewPageHostIndependentJobInfo
export OverviewPageAnalysis, OverviewInferenceLatency, OverviewPageRecommendation
export OverviewPageRunEnvironment, OverviewPage


struct OverviewTfOp
    name::String
    category::String
    self_time_fraction::Float64
    cumulative_time_fraction::Float64
    flop_rate::Float64
    is_op_tensorcore_eligible::Bool
    is_op_using_tensorcore::Bool
end
OverviewTfOp(;name = "", category = "", self_time_fraction = zero(Float64), cumulative_time_fraction = zero(Float64), flop_rate = zero(Float64), is_op_tensorcore_eligible = false, is_op_using_tensorcore = false) = OverviewTfOp(name, category, self_time_fraction, cumulative_time_fraction, flop_rate, is_op_tensorcore_eligible, is_op_using_tensorcore)
PB.default_values(::Type{OverviewTfOp}) = (;name = "", category = "", self_time_fraction = zero(Float64), cumulative_time_fraction = zero(Float64), flop_rate = zero(Float64), is_op_tensorcore_eligible = false, is_op_using_tensorcore = false)
PB.field_numbers(::Type{OverviewTfOp}) = (;name = 1, category = 2, self_time_fraction = 3, cumulative_time_fraction = 4, flop_rate = 5, is_op_tensorcore_eligible = 6, is_op_using_tensorcore = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OverviewTfOp})
    name = ""
    category = ""
    self_time_fraction = zero(Float64)
    cumulative_time_fraction = zero(Float64)
    flop_rate = zero(Float64)
    is_op_tensorcore_eligible = false
    is_op_using_tensorcore = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            category = PB.decode(d, String)
        elseif field_number == 3
            self_time_fraction = PB.decode(d, Float64)
        elseif field_number == 4
            cumulative_time_fraction = PB.decode(d, Float64)
        elseif field_number == 5
            flop_rate = PB.decode(d, Float64)
        elseif field_number == 6
            is_op_tensorcore_eligible = PB.decode(d, Bool)
        elseif field_number == 7
            is_op_using_tensorcore = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return OverviewTfOp(name, category, self_time_fraction, cumulative_time_fraction, flop_rate, is_op_tensorcore_eligible, is_op_using_tensorcore)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OverviewTfOp)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    !isempty(x.category) && PB.encode(e, 2, x.category)
    x.self_time_fraction !== zero(Float64) && PB.encode(e, 3, x.self_time_fraction)
    x.cumulative_time_fraction !== zero(Float64) && PB.encode(e, 4, x.cumulative_time_fraction)
    x.flop_rate !== zero(Float64) && PB.encode(e, 5, x.flop_rate)
    x.is_op_tensorcore_eligible != false && PB.encode(e, 6, x.is_op_tensorcore_eligible)
    x.is_op_using_tensorcore != false && PB.encode(e, 7, x.is_op_using_tensorcore)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OverviewTfOp)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    !isempty(x.category) && (encoded_size += PB._encoded_size(x.category, 2))
    x.self_time_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.self_time_fraction, 3))
    x.cumulative_time_fraction !== zero(Float64) && (encoded_size += PB._encoded_size(x.cumulative_time_fraction, 4))
    x.flop_rate !== zero(Float64) && (encoded_size += PB._encoded_size(x.flop_rate, 5))
    x.is_op_tensorcore_eligible != false && (encoded_size += PB._encoded_size(x.is_op_tensorcore_eligible, 6))
    x.is_op_using_tensorcore != false && (encoded_size += PB._encoded_size(x.is_op_using_tensorcore, 7))
    return encoded_size
end

struct OverviewPageHostDependentJobInfo
    host_id::String
    command_line::String
    start_time::Int64
    bns_address::String
    profile_time_ns::UInt64
end
OverviewPageHostDependentJobInfo(;host_id = "", command_line = "", start_time = zero(Int64), bns_address = "", profile_time_ns = zero(UInt64)) = OverviewPageHostDependentJobInfo(host_id, command_line, start_time, bns_address, profile_time_ns)
PB.default_values(::Type{OverviewPageHostDependentJobInfo}) = (;host_id = "", command_line = "", start_time = zero(Int64), bns_address = "", profile_time_ns = zero(UInt64))
PB.field_numbers(::Type{OverviewPageHostDependentJobInfo}) = (;host_id = 1, command_line = 2, start_time = 3, bns_address = 4, profile_time_ns = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OverviewPageHostDependentJobInfo})
    host_id = ""
    command_line = ""
    start_time = zero(Int64)
    bns_address = ""
    profile_time_ns = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            host_id = PB.decode(d, String)
        elseif field_number == 2
            command_line = PB.decode(d, String)
        elseif field_number == 3
            start_time = PB.decode(d, Int64)
        elseif field_number == 4
            bns_address = PB.decode(d, String)
        elseif field_number == 5
            profile_time_ns = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return OverviewPageHostDependentJobInfo(host_id, command_line, start_time, bns_address, profile_time_ns)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OverviewPageHostDependentJobInfo)
    initpos = position(e.io)
    !isempty(x.host_id) && PB.encode(e, 1, x.host_id)
    !isempty(x.command_line) && PB.encode(e, 2, x.command_line)
    x.start_time != zero(Int64) && PB.encode(e, 3, x.start_time)
    !isempty(x.bns_address) && PB.encode(e, 4, x.bns_address)
    x.profile_time_ns != zero(UInt64) && PB.encode(e, 5, x.profile_time_ns)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OverviewPageHostDependentJobInfo)
    encoded_size = 0
    !isempty(x.host_id) && (encoded_size += PB._encoded_size(x.host_id, 1))
    !isempty(x.command_line) && (encoded_size += PB._encoded_size(x.command_line, 2))
    x.start_time != zero(Int64) && (encoded_size += PB._encoded_size(x.start_time, 3))
    !isempty(x.bns_address) && (encoded_size += PB._encoded_size(x.bns_address, 4))
    x.profile_time_ns != zero(UInt64) && (encoded_size += PB._encoded_size(x.profile_time_ns, 5))
    return encoded_size
end

struct OverviewLatencyBreakdown
    total_latency_us::Float64
    host_latency_us::Float64
    device_latency_us::Float64
    communication_latency_us::Float64
end
OverviewLatencyBreakdown(;total_latency_us = zero(Float64), host_latency_us = zero(Float64), device_latency_us = zero(Float64), communication_latency_us = zero(Float64)) = OverviewLatencyBreakdown(total_latency_us, host_latency_us, device_latency_us, communication_latency_us)
PB.default_values(::Type{OverviewLatencyBreakdown}) = (;total_latency_us = zero(Float64), host_latency_us = zero(Float64), device_latency_us = zero(Float64), communication_latency_us = zero(Float64))
PB.field_numbers(::Type{OverviewLatencyBreakdown}) = (;total_latency_us = 1, host_latency_us = 2, device_latency_us = 3, communication_latency_us = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OverviewLatencyBreakdown})
    total_latency_us = zero(Float64)
    host_latency_us = zero(Float64)
    device_latency_us = zero(Float64)
    communication_latency_us = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            total_latency_us = PB.decode(d, Float64)
        elseif field_number == 2
            host_latency_us = PB.decode(d, Float64)
        elseif field_number == 3
            device_latency_us = PB.decode(d, Float64)
        elseif field_number == 4
            communication_latency_us = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return OverviewLatencyBreakdown(total_latency_us, host_latency_us, device_latency_us, communication_latency_us)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OverviewLatencyBreakdown)
    initpos = position(e.io)
    x.total_latency_us !== zero(Float64) && PB.encode(e, 1, x.total_latency_us)
    x.host_latency_us !== zero(Float64) && PB.encode(e, 2, x.host_latency_us)
    x.device_latency_us !== zero(Float64) && PB.encode(e, 3, x.device_latency_us)
    x.communication_latency_us !== zero(Float64) && PB.encode(e, 4, x.communication_latency_us)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OverviewLatencyBreakdown)
    encoded_size = 0
    x.total_latency_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_latency_us, 1))
    x.host_latency_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_latency_us, 2))
    x.device_latency_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_latency_us, 3))
    x.communication_latency_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.communication_latency_us, 4))
    return encoded_size
end

struct GenericRecommendation
    kernel_launch_bottleneck::String
    kernel_launch_statement::String
    all_other_bottleneck::String
    all_other_statement::String
    precision_statement::String
    device_collectives_bottleneck::String
    device_collectives_statement::String
end
GenericRecommendation(;kernel_launch_bottleneck = "", kernel_launch_statement = "", all_other_bottleneck = "", all_other_statement = "", precision_statement = "", device_collectives_bottleneck = "", device_collectives_statement = "") = GenericRecommendation(kernel_launch_bottleneck, kernel_launch_statement, all_other_bottleneck, all_other_statement, precision_statement, device_collectives_bottleneck, device_collectives_statement)
PB.default_values(::Type{GenericRecommendation}) = (;kernel_launch_bottleneck = "", kernel_launch_statement = "", all_other_bottleneck = "", all_other_statement = "", precision_statement = "", device_collectives_bottleneck = "", device_collectives_statement = "")
PB.field_numbers(::Type{GenericRecommendation}) = (;kernel_launch_bottleneck = 1, kernel_launch_statement = 2, all_other_bottleneck = 3, all_other_statement = 4, precision_statement = 5, device_collectives_bottleneck = 6, device_collectives_statement = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GenericRecommendation})
    kernel_launch_bottleneck = ""
    kernel_launch_statement = ""
    all_other_bottleneck = ""
    all_other_statement = ""
    precision_statement = ""
    device_collectives_bottleneck = ""
    device_collectives_statement = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            kernel_launch_bottleneck = PB.decode(d, String)
        elseif field_number == 2
            kernel_launch_statement = PB.decode(d, String)
        elseif field_number == 3
            all_other_bottleneck = PB.decode(d, String)
        elseif field_number == 4
            all_other_statement = PB.decode(d, String)
        elseif field_number == 5
            precision_statement = PB.decode(d, String)
        elseif field_number == 6
            device_collectives_bottleneck = PB.decode(d, String)
        elseif field_number == 7
            device_collectives_statement = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return GenericRecommendation(kernel_launch_bottleneck, kernel_launch_statement, all_other_bottleneck, all_other_statement, precision_statement, device_collectives_bottleneck, device_collectives_statement)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GenericRecommendation)
    initpos = position(e.io)
    !isempty(x.kernel_launch_bottleneck) && PB.encode(e, 1, x.kernel_launch_bottleneck)
    !isempty(x.kernel_launch_statement) && PB.encode(e, 2, x.kernel_launch_statement)
    !isempty(x.all_other_bottleneck) && PB.encode(e, 3, x.all_other_bottleneck)
    !isempty(x.all_other_statement) && PB.encode(e, 4, x.all_other_statement)
    !isempty(x.precision_statement) && PB.encode(e, 5, x.precision_statement)
    !isempty(x.device_collectives_bottleneck) && PB.encode(e, 6, x.device_collectives_bottleneck)
    !isempty(x.device_collectives_statement) && PB.encode(e, 7, x.device_collectives_statement)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GenericRecommendation)
    encoded_size = 0
    !isempty(x.kernel_launch_bottleneck) && (encoded_size += PB._encoded_size(x.kernel_launch_bottleneck, 1))
    !isempty(x.kernel_launch_statement) && (encoded_size += PB._encoded_size(x.kernel_launch_statement, 2))
    !isempty(x.all_other_bottleneck) && (encoded_size += PB._encoded_size(x.all_other_bottleneck, 3))
    !isempty(x.all_other_statement) && (encoded_size += PB._encoded_size(x.all_other_statement, 4))
    !isempty(x.precision_statement) && (encoded_size += PB._encoded_size(x.precision_statement, 5))
    !isempty(x.device_collectives_bottleneck) && (encoded_size += PB._encoded_size(x.device_collectives_bottleneck, 6))
    !isempty(x.device_collectives_statement) && (encoded_size += PB._encoded_size(x.device_collectives_statement, 7))
    return encoded_size
end

struct OverviewPageTip
    link::String
end
OverviewPageTip(;link = "") = OverviewPageTip(link)
PB.default_values(::Type{OverviewPageTip}) = (;link = "")
PB.field_numbers(::Type{OverviewPageTip}) = (;link = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OverviewPageTip})
    link = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            link = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return OverviewPageTip(link)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OverviewPageTip)
    initpos = position(e.io)
    !isempty(x.link) && PB.encode(e, 1, x.link)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OverviewPageTip)
    encoded_size = 0
    !isempty(x.link) && (encoded_size += PB._encoded_size(x.link, 1))
    return encoded_size
end

struct OverviewPageHostIndependentJobInfo
    change_list::Int64
    workspace_id::String
    snapshot::Int64
    build_time::Int64
    build_target::String
    profile_duration_ms::UInt32
end
OverviewPageHostIndependentJobInfo(;change_list = zero(Int64), workspace_id = "", snapshot = zero(Int64), build_time = zero(Int64), build_target = "", profile_duration_ms = zero(UInt32)) = OverviewPageHostIndependentJobInfo(change_list, workspace_id, snapshot, build_time, build_target, profile_duration_ms)
PB.default_values(::Type{OverviewPageHostIndependentJobInfo}) = (;change_list = zero(Int64), workspace_id = "", snapshot = zero(Int64), build_time = zero(Int64), build_target = "", profile_duration_ms = zero(UInt32))
PB.field_numbers(::Type{OverviewPageHostIndependentJobInfo}) = (;change_list = 1, workspace_id = 5, snapshot = 6, build_time = 2, build_target = 3, profile_duration_ms = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OverviewPageHostIndependentJobInfo})
    change_list = zero(Int64)
    workspace_id = ""
    snapshot = zero(Int64)
    build_time = zero(Int64)
    build_target = ""
    profile_duration_ms = zero(UInt32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            change_list = PB.decode(d, Int64)
        elseif field_number == 5
            workspace_id = PB.decode(d, String)
        elseif field_number == 6
            snapshot = PB.decode(d, Int64)
        elseif field_number == 2
            build_time = PB.decode(d, Int64)
        elseif field_number == 3
            build_target = PB.decode(d, String)
        elseif field_number == 4
            profile_duration_ms = PB.decode(d, UInt32)
        else
            Base.skip(d, wire_type)
        end
    end
    return OverviewPageHostIndependentJobInfo(change_list, workspace_id, snapshot, build_time, build_target, profile_duration_ms)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OverviewPageHostIndependentJobInfo)
    initpos = position(e.io)
    x.change_list != zero(Int64) && PB.encode(e, 1, x.change_list)
    !isempty(x.workspace_id) && PB.encode(e, 5, x.workspace_id)
    x.snapshot != zero(Int64) && PB.encode(e, 6, x.snapshot)
    x.build_time != zero(Int64) && PB.encode(e, 2, x.build_time)
    !isempty(x.build_target) && PB.encode(e, 3, x.build_target)
    x.profile_duration_ms != zero(UInt32) && PB.encode(e, 4, x.profile_duration_ms)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OverviewPageHostIndependentJobInfo)
    encoded_size = 0
    x.change_list != zero(Int64) && (encoded_size += PB._encoded_size(x.change_list, 1))
    !isempty(x.workspace_id) && (encoded_size += PB._encoded_size(x.workspace_id, 5))
    x.snapshot != zero(Int64) && (encoded_size += PB._encoded_size(x.snapshot, 6))
    x.build_time != zero(Int64) && (encoded_size += PB._encoded_size(x.build_time, 2))
    !isempty(x.build_target) && (encoded_size += PB._encoded_size(x.build_target, 3))
    x.profile_duration_ms != zero(UInt32) && (encoded_size += PB._encoded_size(x.profile_duration_ms, 4))
    return encoded_size
end

struct OverviewPageAnalysis
    mxu_utilization_percent::Float64
    device_idle_time_percent::Float64
    host_idle_time_percent::Float64
    top_device_ops::Vector{OverviewTfOp}
    remark_text::String
    remark_color::String
    flop_rate_utilization_relative_to_roofline_percent::Float64
    memory_bw_utilization_relative_to_hw_limit_percent::Float64
    device_compute_16bit_percent::Float64
    device_compute_32bit_percent::Float64
    host_tf_op_percent::Float64
    device_tf_op_percent::Float64
    host_trace_level::UInt32
    host_op_time_eager_percent::Float64
    device_op_time_eager_percent::Float64
    device_op_time_outside_compilation_percent::Float64
    device_duty_cycle_percent::Float64
    hbm_utilization_percent::Float64
    program_goodput_percent::Float64
    sc_step_time_ms_average::Float64
    sc_infeed_time_ms_avg::Float64
    sc_outfeed_time_ms_avg::Float64
    sc_idle_time_ms_avg::Float64
    fw_max_vdd_core_pl1_power_watts::Float64
    fw_max_vdd_core_pl2_power_watts::Float64
    fw_max_vdd_core_pl3_power_watts::Float64
    fw_max_vdd_core_pl4_power_watts::Float64
    fw_max_hbm_pl1_power_watts::Float64
    fw_max_hbm_pl2_power_watts::Float64
    fw_max_hbm_pl3_power_watts::Float64
    fw_max_hbm_pl4_power_watts::Float64
end
OverviewPageAnalysis(;mxu_utilization_percent = zero(Float64), device_idle_time_percent = zero(Float64), host_idle_time_percent = zero(Float64), top_device_ops = Vector{OverviewTfOp}(), remark_text = "", remark_color = "", flop_rate_utilization_relative_to_roofline_percent = zero(Float64), memory_bw_utilization_relative_to_hw_limit_percent = zero(Float64), device_compute_16bit_percent = zero(Float64), device_compute_32bit_percent = zero(Float64), host_tf_op_percent = zero(Float64), device_tf_op_percent = zero(Float64), host_trace_level = zero(UInt32), host_op_time_eager_percent = zero(Float64), device_op_time_eager_percent = zero(Float64), device_op_time_outside_compilation_percent = zero(Float64), device_duty_cycle_percent = zero(Float64), hbm_utilization_percent = zero(Float64), program_goodput_percent = zero(Float64), sc_step_time_ms_average = zero(Float64), sc_infeed_time_ms_avg = zero(Float64), sc_outfeed_time_ms_avg = zero(Float64), sc_idle_time_ms_avg = zero(Float64), fw_max_vdd_core_pl1_power_watts = zero(Float64), fw_max_vdd_core_pl2_power_watts = zero(Float64), fw_max_vdd_core_pl3_power_watts = zero(Float64), fw_max_vdd_core_pl4_power_watts = zero(Float64), fw_max_hbm_pl1_power_watts = zero(Float64), fw_max_hbm_pl2_power_watts = zero(Float64), fw_max_hbm_pl3_power_watts = zero(Float64), fw_max_hbm_pl4_power_watts = zero(Float64)) = OverviewPageAnalysis(mxu_utilization_percent, device_idle_time_percent, host_idle_time_percent, top_device_ops, remark_text, remark_color, flop_rate_utilization_relative_to_roofline_percent, memory_bw_utilization_relative_to_hw_limit_percent, device_compute_16bit_percent, device_compute_32bit_percent, host_tf_op_percent, device_tf_op_percent, host_trace_level, host_op_time_eager_percent, device_op_time_eager_percent, device_op_time_outside_compilation_percent, device_duty_cycle_percent, hbm_utilization_percent, program_goodput_percent, sc_step_time_ms_average, sc_infeed_time_ms_avg, sc_outfeed_time_ms_avg, sc_idle_time_ms_avg, fw_max_vdd_core_pl1_power_watts, fw_max_vdd_core_pl2_power_watts, fw_max_vdd_core_pl3_power_watts, fw_max_vdd_core_pl4_power_watts, fw_max_hbm_pl1_power_watts, fw_max_hbm_pl2_power_watts, fw_max_hbm_pl3_power_watts, fw_max_hbm_pl4_power_watts)
PB.default_values(::Type{OverviewPageAnalysis}) = (;mxu_utilization_percent = zero(Float64), device_idle_time_percent = zero(Float64), host_idle_time_percent = zero(Float64), top_device_ops = Vector{OverviewTfOp}(), remark_text = "", remark_color = "", flop_rate_utilization_relative_to_roofline_percent = zero(Float64), memory_bw_utilization_relative_to_hw_limit_percent = zero(Float64), device_compute_16bit_percent = zero(Float64), device_compute_32bit_percent = zero(Float64), host_tf_op_percent = zero(Float64), device_tf_op_percent = zero(Float64), host_trace_level = zero(UInt32), host_op_time_eager_percent = zero(Float64), device_op_time_eager_percent = zero(Float64), device_op_time_outside_compilation_percent = zero(Float64), device_duty_cycle_percent = zero(Float64), hbm_utilization_percent = zero(Float64), program_goodput_percent = zero(Float64), sc_step_time_ms_average = zero(Float64), sc_infeed_time_ms_avg = zero(Float64), sc_outfeed_time_ms_avg = zero(Float64), sc_idle_time_ms_avg = zero(Float64), fw_max_vdd_core_pl1_power_watts = zero(Float64), fw_max_vdd_core_pl2_power_watts = zero(Float64), fw_max_vdd_core_pl3_power_watts = zero(Float64), fw_max_vdd_core_pl4_power_watts = zero(Float64), fw_max_hbm_pl1_power_watts = zero(Float64), fw_max_hbm_pl2_power_watts = zero(Float64), fw_max_hbm_pl3_power_watts = zero(Float64), fw_max_hbm_pl4_power_watts = zero(Float64))
PB.field_numbers(::Type{OverviewPageAnalysis}) = (;mxu_utilization_percent = 1, device_idle_time_percent = 2, host_idle_time_percent = 3, top_device_ops = 4, remark_text = 5, remark_color = 6, flop_rate_utilization_relative_to_roofline_percent = 7, memory_bw_utilization_relative_to_hw_limit_percent = 8, device_compute_16bit_percent = 9, device_compute_32bit_percent = 10, host_tf_op_percent = 11, device_tf_op_percent = 12, host_trace_level = 13, host_op_time_eager_percent = 14, device_op_time_eager_percent = 15, device_op_time_outside_compilation_percent = 16, device_duty_cycle_percent = 17, hbm_utilization_percent = 31, program_goodput_percent = 18, sc_step_time_ms_average = 19, sc_infeed_time_ms_avg = 20, sc_outfeed_time_ms_avg = 21, sc_idle_time_ms_avg = 22, fw_max_vdd_core_pl1_power_watts = 23, fw_max_vdd_core_pl2_power_watts = 24, fw_max_vdd_core_pl3_power_watts = 25, fw_max_vdd_core_pl4_power_watts = 26, fw_max_hbm_pl1_power_watts = 27, fw_max_hbm_pl2_power_watts = 28, fw_max_hbm_pl3_power_watts = 29, fw_max_hbm_pl4_power_watts = 30)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OverviewPageAnalysis})
    mxu_utilization_percent = zero(Float64)
    device_idle_time_percent = zero(Float64)
    host_idle_time_percent = zero(Float64)
    top_device_ops = PB.BufferedVector{OverviewTfOp}()
    remark_text = ""
    remark_color = ""
    flop_rate_utilization_relative_to_roofline_percent = zero(Float64)
    memory_bw_utilization_relative_to_hw_limit_percent = zero(Float64)
    device_compute_16bit_percent = zero(Float64)
    device_compute_32bit_percent = zero(Float64)
    host_tf_op_percent = zero(Float64)
    device_tf_op_percent = zero(Float64)
    host_trace_level = zero(UInt32)
    host_op_time_eager_percent = zero(Float64)
    device_op_time_eager_percent = zero(Float64)
    device_op_time_outside_compilation_percent = zero(Float64)
    device_duty_cycle_percent = zero(Float64)
    hbm_utilization_percent = zero(Float64)
    program_goodput_percent = zero(Float64)
    sc_step_time_ms_average = zero(Float64)
    sc_infeed_time_ms_avg = zero(Float64)
    sc_outfeed_time_ms_avg = zero(Float64)
    sc_idle_time_ms_avg = zero(Float64)
    fw_max_vdd_core_pl1_power_watts = zero(Float64)
    fw_max_vdd_core_pl2_power_watts = zero(Float64)
    fw_max_vdd_core_pl3_power_watts = zero(Float64)
    fw_max_vdd_core_pl4_power_watts = zero(Float64)
    fw_max_hbm_pl1_power_watts = zero(Float64)
    fw_max_hbm_pl2_power_watts = zero(Float64)
    fw_max_hbm_pl3_power_watts = zero(Float64)
    fw_max_hbm_pl4_power_watts = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            mxu_utilization_percent = PB.decode(d, Float64)
        elseif field_number == 2
            device_idle_time_percent = PB.decode(d, Float64)
        elseif field_number == 3
            host_idle_time_percent = PB.decode(d, Float64)
        elseif field_number == 4
            PB.decode!(d, top_device_ops)
        elseif field_number == 5
            remark_text = PB.decode(d, String)
        elseif field_number == 6
            remark_color = PB.decode(d, String)
        elseif field_number == 7
            flop_rate_utilization_relative_to_roofline_percent = PB.decode(d, Float64)
        elseif field_number == 8
            memory_bw_utilization_relative_to_hw_limit_percent = PB.decode(d, Float64)
        elseif field_number == 9
            device_compute_16bit_percent = PB.decode(d, Float64)
        elseif field_number == 10
            device_compute_32bit_percent = PB.decode(d, Float64)
        elseif field_number == 11
            host_tf_op_percent = PB.decode(d, Float64)
        elseif field_number == 12
            device_tf_op_percent = PB.decode(d, Float64)
        elseif field_number == 13
            host_trace_level = PB.decode(d, UInt32)
        elseif field_number == 14
            host_op_time_eager_percent = PB.decode(d, Float64)
        elseif field_number == 15
            device_op_time_eager_percent = PB.decode(d, Float64)
        elseif field_number == 16
            device_op_time_outside_compilation_percent = PB.decode(d, Float64)
        elseif field_number == 17
            device_duty_cycle_percent = PB.decode(d, Float64)
        elseif field_number == 31
            hbm_utilization_percent = PB.decode(d, Float64)
        elseif field_number == 18
            program_goodput_percent = PB.decode(d, Float64)
        elseif field_number == 19
            sc_step_time_ms_average = PB.decode(d, Float64)
        elseif field_number == 20
            sc_infeed_time_ms_avg = PB.decode(d, Float64)
        elseif field_number == 21
            sc_outfeed_time_ms_avg = PB.decode(d, Float64)
        elseif field_number == 22
            sc_idle_time_ms_avg = PB.decode(d, Float64)
        elseif field_number == 23
            fw_max_vdd_core_pl1_power_watts = PB.decode(d, Float64)
        elseif field_number == 24
            fw_max_vdd_core_pl2_power_watts = PB.decode(d, Float64)
        elseif field_number == 25
            fw_max_vdd_core_pl3_power_watts = PB.decode(d, Float64)
        elseif field_number == 26
            fw_max_vdd_core_pl4_power_watts = PB.decode(d, Float64)
        elseif field_number == 27
            fw_max_hbm_pl1_power_watts = PB.decode(d, Float64)
        elseif field_number == 28
            fw_max_hbm_pl2_power_watts = PB.decode(d, Float64)
        elseif field_number == 29
            fw_max_hbm_pl3_power_watts = PB.decode(d, Float64)
        elseif field_number == 30
            fw_max_hbm_pl4_power_watts = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return OverviewPageAnalysis(mxu_utilization_percent, device_idle_time_percent, host_idle_time_percent, top_device_ops[], remark_text, remark_color, flop_rate_utilization_relative_to_roofline_percent, memory_bw_utilization_relative_to_hw_limit_percent, device_compute_16bit_percent, device_compute_32bit_percent, host_tf_op_percent, device_tf_op_percent, host_trace_level, host_op_time_eager_percent, device_op_time_eager_percent, device_op_time_outside_compilation_percent, device_duty_cycle_percent, hbm_utilization_percent, program_goodput_percent, sc_step_time_ms_average, sc_infeed_time_ms_avg, sc_outfeed_time_ms_avg, sc_idle_time_ms_avg, fw_max_vdd_core_pl1_power_watts, fw_max_vdd_core_pl2_power_watts, fw_max_vdd_core_pl3_power_watts, fw_max_vdd_core_pl4_power_watts, fw_max_hbm_pl1_power_watts, fw_max_hbm_pl2_power_watts, fw_max_hbm_pl3_power_watts, fw_max_hbm_pl4_power_watts)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OverviewPageAnalysis)
    initpos = position(e.io)
    x.mxu_utilization_percent !== zero(Float64) && PB.encode(e, 1, x.mxu_utilization_percent)
    x.device_idle_time_percent !== zero(Float64) && PB.encode(e, 2, x.device_idle_time_percent)
    x.host_idle_time_percent !== zero(Float64) && PB.encode(e, 3, x.host_idle_time_percent)
    !isempty(x.top_device_ops) && PB.encode(e, 4, x.top_device_ops)
    !isempty(x.remark_text) && PB.encode(e, 5, x.remark_text)
    !isempty(x.remark_color) && PB.encode(e, 6, x.remark_color)
    x.flop_rate_utilization_relative_to_roofline_percent !== zero(Float64) && PB.encode(e, 7, x.flop_rate_utilization_relative_to_roofline_percent)
    x.memory_bw_utilization_relative_to_hw_limit_percent !== zero(Float64) && PB.encode(e, 8, x.memory_bw_utilization_relative_to_hw_limit_percent)
    x.device_compute_16bit_percent !== zero(Float64) && PB.encode(e, 9, x.device_compute_16bit_percent)
    x.device_compute_32bit_percent !== zero(Float64) && PB.encode(e, 10, x.device_compute_32bit_percent)
    x.host_tf_op_percent !== zero(Float64) && PB.encode(e, 11, x.host_tf_op_percent)
    x.device_tf_op_percent !== zero(Float64) && PB.encode(e, 12, x.device_tf_op_percent)
    x.host_trace_level != zero(UInt32) && PB.encode(e, 13, x.host_trace_level)
    x.host_op_time_eager_percent !== zero(Float64) && PB.encode(e, 14, x.host_op_time_eager_percent)
    x.device_op_time_eager_percent !== zero(Float64) && PB.encode(e, 15, x.device_op_time_eager_percent)
    x.device_op_time_outside_compilation_percent !== zero(Float64) && PB.encode(e, 16, x.device_op_time_outside_compilation_percent)
    x.device_duty_cycle_percent !== zero(Float64) && PB.encode(e, 17, x.device_duty_cycle_percent)
    x.hbm_utilization_percent !== zero(Float64) && PB.encode(e, 31, x.hbm_utilization_percent)
    x.program_goodput_percent !== zero(Float64) && PB.encode(e, 18, x.program_goodput_percent)
    x.sc_step_time_ms_average !== zero(Float64) && PB.encode(e, 19, x.sc_step_time_ms_average)
    x.sc_infeed_time_ms_avg !== zero(Float64) && PB.encode(e, 20, x.sc_infeed_time_ms_avg)
    x.sc_outfeed_time_ms_avg !== zero(Float64) && PB.encode(e, 21, x.sc_outfeed_time_ms_avg)
    x.sc_idle_time_ms_avg !== zero(Float64) && PB.encode(e, 22, x.sc_idle_time_ms_avg)
    x.fw_max_vdd_core_pl1_power_watts !== zero(Float64) && PB.encode(e, 23, x.fw_max_vdd_core_pl1_power_watts)
    x.fw_max_vdd_core_pl2_power_watts !== zero(Float64) && PB.encode(e, 24, x.fw_max_vdd_core_pl2_power_watts)
    x.fw_max_vdd_core_pl3_power_watts !== zero(Float64) && PB.encode(e, 25, x.fw_max_vdd_core_pl3_power_watts)
    x.fw_max_vdd_core_pl4_power_watts !== zero(Float64) && PB.encode(e, 26, x.fw_max_vdd_core_pl4_power_watts)
    x.fw_max_hbm_pl1_power_watts !== zero(Float64) && PB.encode(e, 27, x.fw_max_hbm_pl1_power_watts)
    x.fw_max_hbm_pl2_power_watts !== zero(Float64) && PB.encode(e, 28, x.fw_max_hbm_pl2_power_watts)
    x.fw_max_hbm_pl3_power_watts !== zero(Float64) && PB.encode(e, 29, x.fw_max_hbm_pl3_power_watts)
    x.fw_max_hbm_pl4_power_watts !== zero(Float64) && PB.encode(e, 30, x.fw_max_hbm_pl4_power_watts)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OverviewPageAnalysis)
    encoded_size = 0
    x.mxu_utilization_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.mxu_utilization_percent, 1))
    x.device_idle_time_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_idle_time_percent, 2))
    x.host_idle_time_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_idle_time_percent, 3))
    !isempty(x.top_device_ops) && (encoded_size += PB._encoded_size(x.top_device_ops, 4))
    !isempty(x.remark_text) && (encoded_size += PB._encoded_size(x.remark_text, 5))
    !isempty(x.remark_color) && (encoded_size += PB._encoded_size(x.remark_color, 6))
    x.flop_rate_utilization_relative_to_roofline_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.flop_rate_utilization_relative_to_roofline_percent, 7))
    x.memory_bw_utilization_relative_to_hw_limit_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.memory_bw_utilization_relative_to_hw_limit_percent, 8))
    x.device_compute_16bit_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_compute_16bit_percent, 9))
    x.device_compute_32bit_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_compute_32bit_percent, 10))
    x.host_tf_op_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_tf_op_percent, 11))
    x.device_tf_op_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_tf_op_percent, 12))
    x.host_trace_level != zero(UInt32) && (encoded_size += PB._encoded_size(x.host_trace_level, 13))
    x.host_op_time_eager_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_op_time_eager_percent, 14))
    x.device_op_time_eager_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_op_time_eager_percent, 15))
    x.device_op_time_outside_compilation_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_op_time_outside_compilation_percent, 16))
    x.device_duty_cycle_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_duty_cycle_percent, 17))
    x.hbm_utilization_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.hbm_utilization_percent, 31))
    x.program_goodput_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.program_goodput_percent, 18))
    x.sc_step_time_ms_average !== zero(Float64) && (encoded_size += PB._encoded_size(x.sc_step_time_ms_average, 19))
    x.sc_infeed_time_ms_avg !== zero(Float64) && (encoded_size += PB._encoded_size(x.sc_infeed_time_ms_avg, 20))
    x.sc_outfeed_time_ms_avg !== zero(Float64) && (encoded_size += PB._encoded_size(x.sc_outfeed_time_ms_avg, 21))
    x.sc_idle_time_ms_avg !== zero(Float64) && (encoded_size += PB._encoded_size(x.sc_idle_time_ms_avg, 22))
    x.fw_max_vdd_core_pl1_power_watts !== zero(Float64) && (encoded_size += PB._encoded_size(x.fw_max_vdd_core_pl1_power_watts, 23))
    x.fw_max_vdd_core_pl2_power_watts !== zero(Float64) && (encoded_size += PB._encoded_size(x.fw_max_vdd_core_pl2_power_watts, 24))
    x.fw_max_vdd_core_pl3_power_watts !== zero(Float64) && (encoded_size += PB._encoded_size(x.fw_max_vdd_core_pl3_power_watts, 25))
    x.fw_max_vdd_core_pl4_power_watts !== zero(Float64) && (encoded_size += PB._encoded_size(x.fw_max_vdd_core_pl4_power_watts, 26))
    x.fw_max_hbm_pl1_power_watts !== zero(Float64) && (encoded_size += PB._encoded_size(x.fw_max_hbm_pl1_power_watts, 27))
    x.fw_max_hbm_pl2_power_watts !== zero(Float64) && (encoded_size += PB._encoded_size(x.fw_max_hbm_pl2_power_watts, 28))
    x.fw_max_hbm_pl3_power_watts !== zero(Float64) && (encoded_size += PB._encoded_size(x.fw_max_hbm_pl3_power_watts, 29))
    x.fw_max_hbm_pl4_power_watts !== zero(Float64) && (encoded_size += PB._encoded_size(x.fw_max_hbm_pl4_power_watts, 30))
    return encoded_size
end

struct OverviewInferenceLatency
    percentile_numbers::Vector{Float64}
    latency_breakdowns::Vector{OverviewLatencyBreakdown}
    max_latency_us::Float64
    min_latency_us::Float64
    sessions_per_second::Float64
end
OverviewInferenceLatency(;percentile_numbers = Vector{Float64}(), latency_breakdowns = Vector{OverviewLatencyBreakdown}(), max_latency_us = zero(Float64), min_latency_us = zero(Float64), sessions_per_second = zero(Float64)) = OverviewInferenceLatency(percentile_numbers, latency_breakdowns, max_latency_us, min_latency_us, sessions_per_second)
PB.default_values(::Type{OverviewInferenceLatency}) = (;percentile_numbers = Vector{Float64}(), latency_breakdowns = Vector{OverviewLatencyBreakdown}(), max_latency_us = zero(Float64), min_latency_us = zero(Float64), sessions_per_second = zero(Float64))
PB.field_numbers(::Type{OverviewInferenceLatency}) = (;percentile_numbers = 1, latency_breakdowns = 2, max_latency_us = 3, min_latency_us = 4, sessions_per_second = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OverviewInferenceLatency})
    percentile_numbers = PB.BufferedVector{Float64}()
    latency_breakdowns = PB.BufferedVector{OverviewLatencyBreakdown}()
    max_latency_us = zero(Float64)
    min_latency_us = zero(Float64)
    sessions_per_second = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, percentile_numbers)
        elseif field_number == 2
            PB.decode!(d, latency_breakdowns)
        elseif field_number == 3
            max_latency_us = PB.decode(d, Float64)
        elseif field_number == 4
            min_latency_us = PB.decode(d, Float64)
        elseif field_number == 5
            sessions_per_second = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return OverviewInferenceLatency(percentile_numbers[], latency_breakdowns[], max_latency_us, min_latency_us, sessions_per_second)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OverviewInferenceLatency)
    initpos = position(e.io)
    !isempty(x.percentile_numbers) && PB.encode(e, 1, x.percentile_numbers)
    !isempty(x.latency_breakdowns) && PB.encode(e, 2, x.latency_breakdowns)
    x.max_latency_us !== zero(Float64) && PB.encode(e, 3, x.max_latency_us)
    x.min_latency_us !== zero(Float64) && PB.encode(e, 4, x.min_latency_us)
    x.sessions_per_second !== zero(Float64) && PB.encode(e, 5, x.sessions_per_second)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OverviewInferenceLatency)
    encoded_size = 0
    !isempty(x.percentile_numbers) && (encoded_size += PB._encoded_size(x.percentile_numbers, 1))
    !isempty(x.latency_breakdowns) && (encoded_size += PB._encoded_size(x.latency_breakdowns, 2))
    x.max_latency_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.max_latency_us, 3))
    x.min_latency_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.min_latency_us, 4))
    x.sessions_per_second !== zero(Float64) && (encoded_size += PB._encoded_size(x.sessions_per_second, 5))
    return encoded_size
end

struct OverviewPageRecommendation
    bottleneck::String
    statement::String
    input_tips::Vector{OverviewPageTip}
    output_statement::String
    eager_statement_html::String
    outside_compilation_statement_html::String
    tf_function_statement_html::String
    host_tips::Vector{OverviewPageTip}
    device_tips::Vector{OverviewPageTip}
    documentation_tips::Vector{OverviewPageTip}
    recommendation::Union{Nothing,google.protobuf.var"#Any"}
    faq_tips::Vector{OverviewPageTip}
    inference_tips::Vector{OverviewPageTip}
end
OverviewPageRecommendation(;bottleneck = "", statement = "", input_tips = Vector{OverviewPageTip}(), output_statement = "", eager_statement_html = "", outside_compilation_statement_html = "", tf_function_statement_html = "", host_tips = Vector{OverviewPageTip}(), device_tips = Vector{OverviewPageTip}(), documentation_tips = Vector{OverviewPageTip}(), recommendation = nothing, faq_tips = Vector{OverviewPageTip}(), inference_tips = Vector{OverviewPageTip}()) = OverviewPageRecommendation(bottleneck, statement, input_tips, output_statement, eager_statement_html, outside_compilation_statement_html, tf_function_statement_html, host_tips, device_tips, documentation_tips, recommendation, faq_tips, inference_tips)
PB.default_values(::Type{OverviewPageRecommendation}) = (;bottleneck = "", statement = "", input_tips = Vector{OverviewPageTip}(), output_statement = "", eager_statement_html = "", outside_compilation_statement_html = "", tf_function_statement_html = "", host_tips = Vector{OverviewPageTip}(), device_tips = Vector{OverviewPageTip}(), documentation_tips = Vector{OverviewPageTip}(), recommendation = nothing, faq_tips = Vector{OverviewPageTip}(), inference_tips = Vector{OverviewPageTip}())
PB.field_numbers(::Type{OverviewPageRecommendation}) = (;bottleneck = 1, statement = 2, input_tips = 11, output_statement = 9, eager_statement_html = 12, outside_compilation_statement_html = 13, tf_function_statement_html = 10, host_tips = 3, device_tips = 4, documentation_tips = 5, recommendation = 6, faq_tips = 7, inference_tips = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OverviewPageRecommendation})
    bottleneck = ""
    statement = ""
    input_tips = PB.BufferedVector{OverviewPageTip}()
    output_statement = ""
    eager_statement_html = ""
    outside_compilation_statement_html = ""
    tf_function_statement_html = ""
    host_tips = PB.BufferedVector{OverviewPageTip}()
    device_tips = PB.BufferedVector{OverviewPageTip}()
    documentation_tips = PB.BufferedVector{OverviewPageTip}()
    recommendation = Ref{Union{Nothing,google.protobuf.var"#Any"}}(nothing)
    faq_tips = PB.BufferedVector{OverviewPageTip}()
    inference_tips = PB.BufferedVector{OverviewPageTip}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            bottleneck = PB.decode(d, String)
        elseif field_number == 2
            statement = PB.decode(d, String)
        elseif field_number == 11
            PB.decode!(d, input_tips)
        elseif field_number == 9
            output_statement = PB.decode(d, String)
        elseif field_number == 12
            eager_statement_html = PB.decode(d, String)
        elseif field_number == 13
            outside_compilation_statement_html = PB.decode(d, String)
        elseif field_number == 10
            tf_function_statement_html = PB.decode(d, String)
        elseif field_number == 3
            PB.decode!(d, host_tips)
        elseif field_number == 4
            PB.decode!(d, device_tips)
        elseif field_number == 5
            PB.decode!(d, documentation_tips)
        elseif field_number == 6
            PB.decode!(d, recommendation)
        elseif field_number == 7
            PB.decode!(d, faq_tips)
        elseif field_number == 8
            PB.decode!(d, inference_tips)
        else
            Base.skip(d, wire_type)
        end
    end
    return OverviewPageRecommendation(bottleneck, statement, input_tips[], output_statement, eager_statement_html, outside_compilation_statement_html, tf_function_statement_html, host_tips[], device_tips[], documentation_tips[], recommendation[], faq_tips[], inference_tips[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OverviewPageRecommendation)
    initpos = position(e.io)
    !isempty(x.bottleneck) && PB.encode(e, 1, x.bottleneck)
    !isempty(x.statement) && PB.encode(e, 2, x.statement)
    !isempty(x.input_tips) && PB.encode(e, 11, x.input_tips)
    !isempty(x.output_statement) && PB.encode(e, 9, x.output_statement)
    !isempty(x.eager_statement_html) && PB.encode(e, 12, x.eager_statement_html)
    !isempty(x.outside_compilation_statement_html) && PB.encode(e, 13, x.outside_compilation_statement_html)
    !isempty(x.tf_function_statement_html) && PB.encode(e, 10, x.tf_function_statement_html)
    !isempty(x.host_tips) && PB.encode(e, 3, x.host_tips)
    !isempty(x.device_tips) && PB.encode(e, 4, x.device_tips)
    !isempty(x.documentation_tips) && PB.encode(e, 5, x.documentation_tips)
    !isnothing(x.recommendation) && PB.encode(e, 6, x.recommendation)
    !isempty(x.faq_tips) && PB.encode(e, 7, x.faq_tips)
    !isempty(x.inference_tips) && PB.encode(e, 8, x.inference_tips)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OverviewPageRecommendation)
    encoded_size = 0
    !isempty(x.bottleneck) && (encoded_size += PB._encoded_size(x.bottleneck, 1))
    !isempty(x.statement) && (encoded_size += PB._encoded_size(x.statement, 2))
    !isempty(x.input_tips) && (encoded_size += PB._encoded_size(x.input_tips, 11))
    !isempty(x.output_statement) && (encoded_size += PB._encoded_size(x.output_statement, 9))
    !isempty(x.eager_statement_html) && (encoded_size += PB._encoded_size(x.eager_statement_html, 12))
    !isempty(x.outside_compilation_statement_html) && (encoded_size += PB._encoded_size(x.outside_compilation_statement_html, 13))
    !isempty(x.tf_function_statement_html) && (encoded_size += PB._encoded_size(x.tf_function_statement_html, 10))
    !isempty(x.host_tips) && (encoded_size += PB._encoded_size(x.host_tips, 3))
    !isempty(x.device_tips) && (encoded_size += PB._encoded_size(x.device_tips, 4))
    !isempty(x.documentation_tips) && (encoded_size += PB._encoded_size(x.documentation_tips, 5))
    !isnothing(x.recommendation) && (encoded_size += PB._encoded_size(x.recommendation, 6))
    !isempty(x.faq_tips) && (encoded_size += PB._encoded_size(x.faq_tips, 7))
    !isempty(x.inference_tips) && (encoded_size += PB._encoded_size(x.inference_tips, 8))
    return encoded_size
end

struct OverviewPageRunEnvironment
    host_count::Int32
    task_count::Int32
    hostnames::Dict{String,Bool}
    device_type::String
    device_core_count::Int32
    host_independent_job_info::Union{Nothing,OverviewPageHostIndependentJobInfo}
    host_dependent_job_info::Vector{OverviewPageHostDependentJobInfo}
    replica_count::Int32
    num_cores_per_replica::Int32
    is_training::Bool
    power_metrics::Union{Nothing,PowerMetrics}
end
OverviewPageRunEnvironment(;host_count = zero(Int32), task_count = zero(Int32), hostnames = Dict{String,Bool}(), device_type = "", device_core_count = zero(Int32), host_independent_job_info = nothing, host_dependent_job_info = Vector{OverviewPageHostDependentJobInfo}(), replica_count = zero(Int32), num_cores_per_replica = zero(Int32), is_training = false, power_metrics = nothing) = OverviewPageRunEnvironment(host_count, task_count, hostnames, device_type, device_core_count, host_independent_job_info, host_dependent_job_info, replica_count, num_cores_per_replica, is_training, power_metrics)
PB.reserved_fields(::Type{OverviewPageRunEnvironment}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[6])
PB.default_values(::Type{OverviewPageRunEnvironment}) = (;host_count = zero(Int32), task_count = zero(Int32), hostnames = Dict{String,Bool}(), device_type = "", device_core_count = zero(Int32), host_independent_job_info = nothing, host_dependent_job_info = Vector{OverviewPageHostDependentJobInfo}(), replica_count = zero(Int32), num_cores_per_replica = zero(Int32), is_training = false, power_metrics = nothing)
PB.field_numbers(::Type{OverviewPageRunEnvironment}) = (;host_count = 1, task_count = 2, hostnames = 3, device_type = 4, device_core_count = 5, host_independent_job_info = 7, host_dependent_job_info = 8, replica_count = 9, num_cores_per_replica = 10, is_training = 11, power_metrics = 12)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OverviewPageRunEnvironment})
    host_count = zero(Int32)
    task_count = zero(Int32)
    hostnames = Dict{String,Bool}()
    device_type = ""
    device_core_count = zero(Int32)
    host_independent_job_info = Ref{Union{Nothing,OverviewPageHostIndependentJobInfo}}(nothing)
    host_dependent_job_info = PB.BufferedVector{OverviewPageHostDependentJobInfo}()
    replica_count = zero(Int32)
    num_cores_per_replica = zero(Int32)
    is_training = false
    power_metrics = Ref{Union{Nothing,PowerMetrics}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            host_count = PB.decode(d, Int32)
        elseif field_number == 2
            task_count = PB.decode(d, Int32)
        elseif field_number == 3
            PB.decode!(d, hostnames)
        elseif field_number == 4
            device_type = PB.decode(d, String)
        elseif field_number == 5
            device_core_count = PB.decode(d, Int32)
        elseif field_number == 7
            PB.decode!(d, host_independent_job_info)
        elseif field_number == 8
            PB.decode!(d, host_dependent_job_info)
        elseif field_number == 9
            replica_count = PB.decode(d, Int32)
        elseif field_number == 10
            num_cores_per_replica = PB.decode(d, Int32)
        elseif field_number == 11
            is_training = PB.decode(d, Bool)
        elseif field_number == 12
            PB.decode!(d, power_metrics)
        else
            Base.skip(d, wire_type)
        end
    end
    return OverviewPageRunEnvironment(host_count, task_count, hostnames, device_type, device_core_count, host_independent_job_info[], host_dependent_job_info[], replica_count, num_cores_per_replica, is_training, power_metrics[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OverviewPageRunEnvironment)
    initpos = position(e.io)
    x.host_count != zero(Int32) && PB.encode(e, 1, x.host_count)
    x.task_count != zero(Int32) && PB.encode(e, 2, x.task_count)
    !isempty(x.hostnames) && PB.encode(e, 3, x.hostnames)
    !isempty(x.device_type) && PB.encode(e, 4, x.device_type)
    x.device_core_count != zero(Int32) && PB.encode(e, 5, x.device_core_count)
    !isnothing(x.host_independent_job_info) && PB.encode(e, 7, x.host_independent_job_info)
    !isempty(x.host_dependent_job_info) && PB.encode(e, 8, x.host_dependent_job_info)
    x.replica_count != zero(Int32) && PB.encode(e, 9, x.replica_count)
    x.num_cores_per_replica != zero(Int32) && PB.encode(e, 10, x.num_cores_per_replica)
    x.is_training != false && PB.encode(e, 11, x.is_training)
    !isnothing(x.power_metrics) && PB.encode(e, 12, x.power_metrics)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OverviewPageRunEnvironment)
    encoded_size = 0
    x.host_count != zero(Int32) && (encoded_size += PB._encoded_size(x.host_count, 1))
    x.task_count != zero(Int32) && (encoded_size += PB._encoded_size(x.task_count, 2))
    !isempty(x.hostnames) && (encoded_size += PB._encoded_size(x.hostnames, 3))
    !isempty(x.device_type) && (encoded_size += PB._encoded_size(x.device_type, 4))
    x.device_core_count != zero(Int32) && (encoded_size += PB._encoded_size(x.device_core_count, 5))
    !isnothing(x.host_independent_job_info) && (encoded_size += PB._encoded_size(x.host_independent_job_info, 7))
    !isempty(x.host_dependent_job_info) && (encoded_size += PB._encoded_size(x.host_dependent_job_info, 8))
    x.replica_count != zero(Int32) && (encoded_size += PB._encoded_size(x.replica_count, 9))
    x.num_cores_per_replica != zero(Int32) && (encoded_size += PB._encoded_size(x.num_cores_per_replica, 10))
    x.is_training != false && (encoded_size += PB._encoded_size(x.is_training, 11))
    !isnothing(x.power_metrics) && (encoded_size += PB._encoded_size(x.power_metrics, 12))
    return encoded_size
end

struct OverviewPage
    run_environment::Union{Nothing,OverviewPageRunEnvironment}
    input_analysis::Union{Nothing,InputPipelineAnalysisResult}
    analysis::Union{Nothing,OverviewPageAnalysis}
    recommendation::Union{Nothing,OverviewPageRecommendation}
    diagnostics::Union{Nothing,Diagnostics}
    inference_latency::Union{Nothing,OverviewInferenceLatency}
end
OverviewPage(;run_environment = nothing, input_analysis = nothing, analysis = nothing, recommendation = nothing, diagnostics = nothing, inference_latency = nothing) = OverviewPage(run_environment, input_analysis, analysis, recommendation, diagnostics, inference_latency)
PB.reserved_fields(::Type{OverviewPage}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[1, 5, 7])
PB.default_values(::Type{OverviewPage}) = (;run_environment = nothing, input_analysis = nothing, analysis = nothing, recommendation = nothing, diagnostics = nothing, inference_latency = nothing)
PB.field_numbers(::Type{OverviewPage}) = (;run_environment = 6, input_analysis = 2, analysis = 3, recommendation = 4, diagnostics = 8, inference_latency = 9)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OverviewPage})
    run_environment = Ref{Union{Nothing,OverviewPageRunEnvironment}}(nothing)
    input_analysis = Ref{Union{Nothing,InputPipelineAnalysisResult}}(nothing)
    analysis = Ref{Union{Nothing,OverviewPageAnalysis}}(nothing)
    recommendation = Ref{Union{Nothing,OverviewPageRecommendation}}(nothing)
    diagnostics = Ref{Union{Nothing,Diagnostics}}(nothing)
    inference_latency = Ref{Union{Nothing,OverviewInferenceLatency}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 6
            PB.decode!(d, run_environment)
        elseif field_number == 2
            PB.decode!(d, input_analysis)
        elseif field_number == 3
            PB.decode!(d, analysis)
        elseif field_number == 4
            PB.decode!(d, recommendation)
        elseif field_number == 8
            PB.decode!(d, diagnostics)
        elseif field_number == 9
            PB.decode!(d, inference_latency)
        else
            Base.skip(d, wire_type)
        end
    end
    return OverviewPage(run_environment[], input_analysis[], analysis[], recommendation[], diagnostics[], inference_latency[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OverviewPage)
    initpos = position(e.io)
    !isnothing(x.run_environment) && PB.encode(e, 6, x.run_environment)
    !isnothing(x.input_analysis) && PB.encode(e, 2, x.input_analysis)
    !isnothing(x.analysis) && PB.encode(e, 3, x.analysis)
    !isnothing(x.recommendation) && PB.encode(e, 4, x.recommendation)
    !isnothing(x.diagnostics) && PB.encode(e, 8, x.diagnostics)
    !isnothing(x.inference_latency) && PB.encode(e, 9, x.inference_latency)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OverviewPage)
    encoded_size = 0
    !isnothing(x.run_environment) && (encoded_size += PB._encoded_size(x.run_environment, 6))
    !isnothing(x.input_analysis) && (encoded_size += PB._encoded_size(x.input_analysis, 2))
    !isnothing(x.analysis) && (encoded_size += PB._encoded_size(x.analysis, 3))
    !isnothing(x.recommendation) && (encoded_size += PB._encoded_size(x.recommendation, 4))
    !isnothing(x.diagnostics) && (encoded_size += PB._encoded_size(x.diagnostics, 8))
    !isnothing(x.inference_latency) && (encoded_size += PB._encoded_size(x.inference_latency, 9))
    return encoded_size
end
