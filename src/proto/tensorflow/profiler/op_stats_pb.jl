import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export HostIndependentJobInfoResult, CoreDetails, HostDependentJobInfoResult
export SystemTopology, PerformanceCounterResult, PerfEnv, RunEnvironment, OpStats


struct HostIndependentJobInfoResult
    change_list::Int64
    workspace_id::String
    snapshot::Int64
    build_time::Int64
    build_target::String
    profile_duration_ms::UInt32
end
HostIndependentJobInfoResult(;change_list = zero(Int64), workspace_id = "", snapshot = zero(Int64), build_time = zero(Int64), build_target = "", profile_duration_ms = zero(UInt32)) = HostIndependentJobInfoResult(change_list, workspace_id, snapshot, build_time, build_target, profile_duration_ms)
PB.default_values(::Type{HostIndependentJobInfoResult}) = (;change_list = zero(Int64), workspace_id = "", snapshot = zero(Int64), build_time = zero(Int64), build_target = "", profile_duration_ms = zero(UInt32))
PB.field_numbers(::Type{HostIndependentJobInfoResult}) = (;change_list = 1, workspace_id = 5, snapshot = 6, build_time = 2, build_target = 3, profile_duration_ms = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HostIndependentJobInfoResult})
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
    return HostIndependentJobInfoResult(change_list, workspace_id, snapshot, build_time, build_target, profile_duration_ms)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HostIndependentJobInfoResult)
    initpos = position(e.io)
    x.change_list != zero(Int64) && PB.encode(e, 1, x.change_list)
    !isempty(x.workspace_id) && PB.encode(e, 5, x.workspace_id)
    x.snapshot != zero(Int64) && PB.encode(e, 6, x.snapshot)
    x.build_time != zero(Int64) && PB.encode(e, 2, x.build_time)
    !isempty(x.build_target) && PB.encode(e, 3, x.build_target)
    x.profile_duration_ms != zero(UInt32) && PB.encode(e, 4, x.profile_duration_ms)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HostIndependentJobInfoResult)
    encoded_size = 0
    x.change_list != zero(Int64) && (encoded_size += PB._encoded_size(x.change_list, 1))
    !isempty(x.workspace_id) && (encoded_size += PB._encoded_size(x.workspace_id, 5))
    x.snapshot != zero(Int64) && (encoded_size += PB._encoded_size(x.snapshot, 6))
    x.build_time != zero(Int64) && (encoded_size += PB._encoded_size(x.build_time, 2))
    !isempty(x.build_target) && (encoded_size += PB._encoded_size(x.build_target, 3))
    x.profile_duration_ms != zero(UInt32) && (encoded_size += PB._encoded_size(x.profile_duration_ms, 4))
    return encoded_size
end

struct CoreDetails
    hostname::String
    device_ordinal::UInt32
    core_num::UInt32
    local_chip_id::UInt32
    global_chip_id::UInt32
    global_core_id::UInt32
    is_sparse_core::Bool
end
CoreDetails(;hostname = "", device_ordinal = zero(UInt32), core_num = zero(UInt32), local_chip_id = zero(UInt32), global_chip_id = zero(UInt32), global_core_id = zero(UInt32), is_sparse_core = false) = CoreDetails(hostname, device_ordinal, core_num, local_chip_id, global_chip_id, global_core_id, is_sparse_core)
PB.default_values(::Type{CoreDetails}) = (;hostname = "", device_ordinal = zero(UInt32), core_num = zero(UInt32), local_chip_id = zero(UInt32), global_chip_id = zero(UInt32), global_core_id = zero(UInt32), is_sparse_core = false)
PB.field_numbers(::Type{CoreDetails}) = (;hostname = 1, device_ordinal = 2, core_num = 3, local_chip_id = 4, global_chip_id = 5, global_core_id = 6, is_sparse_core = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CoreDetails})
    hostname = ""
    device_ordinal = zero(UInt32)
    core_num = zero(UInt32)
    local_chip_id = zero(UInt32)
    global_chip_id = zero(UInt32)
    global_core_id = zero(UInt32)
    is_sparse_core = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            hostname = PB.decode(d, String)
        elseif field_number == 2
            device_ordinal = PB.decode(d, UInt32)
        elseif field_number == 3
            core_num = PB.decode(d, UInt32)
        elseif field_number == 4
            local_chip_id = PB.decode(d, UInt32)
        elseif field_number == 5
            global_chip_id = PB.decode(d, UInt32)
        elseif field_number == 6
            global_core_id = PB.decode(d, UInt32)
        elseif field_number == 7
            is_sparse_core = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return CoreDetails(hostname, device_ordinal, core_num, local_chip_id, global_chip_id, global_core_id, is_sparse_core)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CoreDetails)
    initpos = position(e.io)
    !isempty(x.hostname) && PB.encode(e, 1, x.hostname)
    x.device_ordinal != zero(UInt32) && PB.encode(e, 2, x.device_ordinal)
    x.core_num != zero(UInt32) && PB.encode(e, 3, x.core_num)
    x.local_chip_id != zero(UInt32) && PB.encode(e, 4, x.local_chip_id)
    x.global_chip_id != zero(UInt32) && PB.encode(e, 5, x.global_chip_id)
    x.global_core_id != zero(UInt32) && PB.encode(e, 6, x.global_core_id)
    x.is_sparse_core != false && PB.encode(e, 7, x.is_sparse_core)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CoreDetails)
    encoded_size = 0
    !isempty(x.hostname) && (encoded_size += PB._encoded_size(x.hostname, 1))
    x.device_ordinal != zero(UInt32) && (encoded_size += PB._encoded_size(x.device_ordinal, 2))
    x.core_num != zero(UInt32) && (encoded_size += PB._encoded_size(x.core_num, 3))
    x.local_chip_id != zero(UInt32) && (encoded_size += PB._encoded_size(x.local_chip_id, 4))
    x.global_chip_id != zero(UInt32) && (encoded_size += PB._encoded_size(x.global_chip_id, 5))
    x.global_core_id != zero(UInt32) && (encoded_size += PB._encoded_size(x.global_core_id, 6))
    x.is_sparse_core != false && (encoded_size += PB._encoded_size(x.is_sparse_core, 7))
    return encoded_size
end

struct HostDependentJobInfoResult
    host_id::String
    command_line::String
    start_time::Int64
    bns_address::String
    profile_time_ns::UInt64
end
HostDependentJobInfoResult(;host_id = "", command_line = "", start_time = zero(Int64), bns_address = "", profile_time_ns = zero(UInt64)) = HostDependentJobInfoResult(host_id, command_line, start_time, bns_address, profile_time_ns)
PB.default_values(::Type{HostDependentJobInfoResult}) = (;host_id = "", command_line = "", start_time = zero(Int64), bns_address = "", profile_time_ns = zero(UInt64))
PB.field_numbers(::Type{HostDependentJobInfoResult}) = (;host_id = 1, command_line = 2, start_time = 3, bns_address = 4, profile_time_ns = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HostDependentJobInfoResult})
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
    return HostDependentJobInfoResult(host_id, command_line, start_time, bns_address, profile_time_ns)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HostDependentJobInfoResult)
    initpos = position(e.io)
    !isempty(x.host_id) && PB.encode(e, 1, x.host_id)
    !isempty(x.command_line) && PB.encode(e, 2, x.command_line)
    x.start_time != zero(Int64) && PB.encode(e, 3, x.start_time)
    !isempty(x.bns_address) && PB.encode(e, 4, x.bns_address)
    x.profile_time_ns != zero(UInt64) && PB.encode(e, 5, x.profile_time_ns)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HostDependentJobInfoResult)
    encoded_size = 0
    !isempty(x.host_id) && (encoded_size += PB._encoded_size(x.host_id, 1))
    !isempty(x.command_line) && (encoded_size += PB._encoded_size(x.command_line, 2))
    x.start_time != zero(Int64) && (encoded_size += PB._encoded_size(x.start_time, 3))
    !isempty(x.bns_address) && (encoded_size += PB._encoded_size(x.bns_address, 4))
    x.profile_time_ns != zero(UInt64) && (encoded_size += PB._encoded_size(x.profile_time_ns, 5))
    return encoded_size
end

struct SystemTopology
    x_dimension::Int64
    y_dimension::Int64
    z_dimension::Int64
    num_expected_reduced_chips::Int64
end
SystemTopology(;x_dimension = zero(Int64), y_dimension = zero(Int64), z_dimension = zero(Int64), num_expected_reduced_chips = zero(Int64)) = SystemTopology(x_dimension, y_dimension, z_dimension, num_expected_reduced_chips)
PB.default_values(::Type{SystemTopology}) = (;x_dimension = zero(Int64), y_dimension = zero(Int64), z_dimension = zero(Int64), num_expected_reduced_chips = zero(Int64))
PB.field_numbers(::Type{SystemTopology}) = (;x_dimension = 1, y_dimension = 2, z_dimension = 3, num_expected_reduced_chips = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SystemTopology})
    x_dimension = zero(Int64)
    y_dimension = zero(Int64)
    z_dimension = zero(Int64)
    num_expected_reduced_chips = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            x_dimension = PB.decode(d, Int64)
        elseif field_number == 2
            y_dimension = PB.decode(d, Int64)
        elseif field_number == 3
            z_dimension = PB.decode(d, Int64)
        elseif field_number == 4
            num_expected_reduced_chips = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return SystemTopology(x_dimension, y_dimension, z_dimension, num_expected_reduced_chips)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SystemTopology)
    initpos = position(e.io)
    x.x_dimension != zero(Int64) && PB.encode(e, 1, x.x_dimension)
    x.y_dimension != zero(Int64) && PB.encode(e, 2, x.y_dimension)
    x.z_dimension != zero(Int64) && PB.encode(e, 3, x.z_dimension)
    x.num_expected_reduced_chips != zero(Int64) && PB.encode(e, 4, x.num_expected_reduced_chips)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SystemTopology)
    encoded_size = 0
    x.x_dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.x_dimension, 1))
    x.y_dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.y_dimension, 2))
    x.z_dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.z_dimension, 3))
    x.num_expected_reduced_chips != zero(Int64) && (encoded_size += PB._encoded_size(x.num_expected_reduced_chips, 4))
    return encoded_size
end

struct PerformanceCounterResult
    matrix_unit_utilization_percent::Float64
    hbm_utilization_percent::Float64
end
PerformanceCounterResult(;matrix_unit_utilization_percent = zero(Float64), hbm_utilization_percent = zero(Float64)) = PerformanceCounterResult(matrix_unit_utilization_percent, hbm_utilization_percent)
PB.default_values(::Type{PerformanceCounterResult}) = (;matrix_unit_utilization_percent = zero(Float64), hbm_utilization_percent = zero(Float64))
PB.field_numbers(::Type{PerformanceCounterResult}) = (;matrix_unit_utilization_percent = 1, hbm_utilization_percent = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PerformanceCounterResult})
    matrix_unit_utilization_percent = zero(Float64)
    hbm_utilization_percent = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            matrix_unit_utilization_percent = PB.decode(d, Float64)
        elseif field_number == 2
            hbm_utilization_percent = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return PerformanceCounterResult(matrix_unit_utilization_percent, hbm_utilization_percent)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PerformanceCounterResult)
    initpos = position(e.io)
    x.matrix_unit_utilization_percent !== zero(Float64) && PB.encode(e, 1, x.matrix_unit_utilization_percent)
    x.hbm_utilization_percent !== zero(Float64) && PB.encode(e, 2, x.hbm_utilization_percent)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PerformanceCounterResult)
    encoded_size = 0
    x.matrix_unit_utilization_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.matrix_unit_utilization_percent, 1))
    x.hbm_utilization_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.hbm_utilization_percent, 2))
    return encoded_size
end

struct PerfEnv
    peak_tera_flops_per_second::Float64
    peak_bw_giga_bytes_per_second::Float64
    peak_hbm_bw_giga_bytes_per_second::Float64
    peak_bws_giga_bytes_per_second::Vector{Float64}
    ridge_point::Float64
    has_cmem::Bool
    has_merged_vmem::Bool
    has_megacore::Bool
end
PerfEnv(;peak_tera_flops_per_second = zero(Float64), peak_bw_giga_bytes_per_second = zero(Float64), peak_hbm_bw_giga_bytes_per_second = zero(Float64), peak_bws_giga_bytes_per_second = Vector{Float64}(), ridge_point = zero(Float64), has_cmem = false, has_merged_vmem = false, has_megacore = false) = PerfEnv(peak_tera_flops_per_second, peak_bw_giga_bytes_per_second, peak_hbm_bw_giga_bytes_per_second, peak_bws_giga_bytes_per_second, ridge_point, has_cmem, has_merged_vmem, has_megacore)
PB.default_values(::Type{PerfEnv}) = (;peak_tera_flops_per_second = zero(Float64), peak_bw_giga_bytes_per_second = zero(Float64), peak_hbm_bw_giga_bytes_per_second = zero(Float64), peak_bws_giga_bytes_per_second = Vector{Float64}(), ridge_point = zero(Float64), has_cmem = false, has_merged_vmem = false, has_megacore = false)
PB.field_numbers(::Type{PerfEnv}) = (;peak_tera_flops_per_second = 1, peak_bw_giga_bytes_per_second = 4, peak_hbm_bw_giga_bytes_per_second = 2, peak_bws_giga_bytes_per_second = 5, ridge_point = 3, has_cmem = 6, has_merged_vmem = 7, has_megacore = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PerfEnv})
    peak_tera_flops_per_second = zero(Float64)
    peak_bw_giga_bytes_per_second = zero(Float64)
    peak_hbm_bw_giga_bytes_per_second = zero(Float64)
    peak_bws_giga_bytes_per_second = PB.BufferedVector{Float64}()
    ridge_point = zero(Float64)
    has_cmem = false
    has_merged_vmem = false
    has_megacore = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            peak_tera_flops_per_second = PB.decode(d, Float64)
        elseif field_number == 4
            peak_bw_giga_bytes_per_second = PB.decode(d, Float64)
        elseif field_number == 2
            peak_hbm_bw_giga_bytes_per_second = PB.decode(d, Float64)
        elseif field_number == 5
            PB.decode!(d, wire_type, peak_bws_giga_bytes_per_second)
        elseif field_number == 3
            ridge_point = PB.decode(d, Float64)
        elseif field_number == 6
            has_cmem = PB.decode(d, Bool)
        elseif field_number == 7
            has_merged_vmem = PB.decode(d, Bool)
        elseif field_number == 8
            has_megacore = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return PerfEnv(peak_tera_flops_per_second, peak_bw_giga_bytes_per_second, peak_hbm_bw_giga_bytes_per_second, peak_bws_giga_bytes_per_second[], ridge_point, has_cmem, has_merged_vmem, has_megacore)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PerfEnv)
    initpos = position(e.io)
    x.peak_tera_flops_per_second !== zero(Float64) && PB.encode(e, 1, x.peak_tera_flops_per_second)
    x.peak_bw_giga_bytes_per_second !== zero(Float64) && PB.encode(e, 4, x.peak_bw_giga_bytes_per_second)
    x.peak_hbm_bw_giga_bytes_per_second !== zero(Float64) && PB.encode(e, 2, x.peak_hbm_bw_giga_bytes_per_second)
    !isempty(x.peak_bws_giga_bytes_per_second) && PB.encode(e, 5, x.peak_bws_giga_bytes_per_second)
    x.ridge_point !== zero(Float64) && PB.encode(e, 3, x.ridge_point)
    x.has_cmem != false && PB.encode(e, 6, x.has_cmem)
    x.has_merged_vmem != false && PB.encode(e, 7, x.has_merged_vmem)
    x.has_megacore != false && PB.encode(e, 8, x.has_megacore)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PerfEnv)
    encoded_size = 0
    x.peak_tera_flops_per_second !== zero(Float64) && (encoded_size += PB._encoded_size(x.peak_tera_flops_per_second, 1))
    x.peak_bw_giga_bytes_per_second !== zero(Float64) && (encoded_size += PB._encoded_size(x.peak_bw_giga_bytes_per_second, 4))
    x.peak_hbm_bw_giga_bytes_per_second !== zero(Float64) && (encoded_size += PB._encoded_size(x.peak_hbm_bw_giga_bytes_per_second, 2))
    !isempty(x.peak_bws_giga_bytes_per_second) && (encoded_size += PB._encoded_size(x.peak_bws_giga_bytes_per_second, 5))
    x.ridge_point !== zero(Float64) && (encoded_size += PB._encoded_size(x.ridge_point, 3))
    x.has_cmem != false && (encoded_size += PB._encoded_size(x.has_cmem, 6))
    x.has_merged_vmem != false && (encoded_size += PB._encoded_size(x.has_merged_vmem, 7))
    x.has_megacore != false && (encoded_size += PB._encoded_size(x.has_megacore, 8))
    return encoded_size
end

struct RunEnvironment
    host_count::Int32
    task_count::Int32
    hostnames::Dict{String,Bool}
    device_type::String
    device_core_count::Int32
    host_independent_job_info::Union{Nothing,HostIndependentJobInfoResult}
    host_dependent_job_info::Vector{HostDependentJobInfoResult}
    replica_count::Int32
    num_cores_per_replica::Int32
    host_trace_level::UInt32
    system_topology::Union{Nothing,Topology}
    is_training::Bool
    power_metrics::Union{Nothing,PowerMetrics}
    hardware_type::HardwareType.T
end
RunEnvironment(;host_count = zero(Int32), task_count = zero(Int32), hostnames = Dict{String,Bool}(), device_type = "", device_core_count = zero(Int32), host_independent_job_info = nothing, host_dependent_job_info = Vector{HostDependentJobInfoResult}(), replica_count = zero(Int32), num_cores_per_replica = zero(Int32), host_trace_level = zero(UInt32), system_topology = nothing, is_training = false, power_metrics = nothing, hardware_type = HardwareType.UNKNOWN_HARDWARE) = RunEnvironment(host_count, task_count, hostnames, device_type, device_core_count, host_independent_job_info, host_dependent_job_info, replica_count, num_cores_per_replica, host_trace_level, system_topology, is_training, power_metrics, hardware_type)
PB.reserved_fields(::Type{RunEnvironment}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[6, 11])
PB.default_values(::Type{RunEnvironment}) = (;host_count = zero(Int32), task_count = zero(Int32), hostnames = Dict{String,Bool}(), device_type = "", device_core_count = zero(Int32), host_independent_job_info = nothing, host_dependent_job_info = Vector{HostDependentJobInfoResult}(), replica_count = zero(Int32), num_cores_per_replica = zero(Int32), host_trace_level = zero(UInt32), system_topology = nothing, is_training = false, power_metrics = nothing, hardware_type = HardwareType.UNKNOWN_HARDWARE)
PB.field_numbers(::Type{RunEnvironment}) = (;host_count = 1, task_count = 2, hostnames = 3, device_type = 4, device_core_count = 5, host_independent_job_info = 7, host_dependent_job_info = 8, replica_count = 9, num_cores_per_replica = 10, host_trace_level = 12, system_topology = 13, is_training = 14, power_metrics = 15, hardware_type = 16)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:RunEnvironment})
    host_count = zero(Int32)
    task_count = zero(Int32)
    hostnames = Dict{String,Bool}()
    device_type = ""
    device_core_count = zero(Int32)
    host_independent_job_info = Ref{Union{Nothing,HostIndependentJobInfoResult}}(nothing)
    host_dependent_job_info = PB.BufferedVector{HostDependentJobInfoResult}()
    replica_count = zero(Int32)
    num_cores_per_replica = zero(Int32)
    host_trace_level = zero(UInt32)
    system_topology = Ref{Union{Nothing,Topology}}(nothing)
    is_training = false
    power_metrics = Ref{Union{Nothing,PowerMetrics}}(nothing)
    hardware_type = HardwareType.UNKNOWN_HARDWARE
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
        elseif field_number == 12
            host_trace_level = PB.decode(d, UInt32)
        elseif field_number == 13
            PB.decode!(d, system_topology)
        elseif field_number == 14
            is_training = PB.decode(d, Bool)
        elseif field_number == 15
            PB.decode!(d, power_metrics)
        elseif field_number == 16
            hardware_type = PB.decode(d, HardwareType.T)
        else
            Base.skip(d, wire_type)
        end
    end
    return RunEnvironment(host_count, task_count, hostnames, device_type, device_core_count, host_independent_job_info[], host_dependent_job_info[], replica_count, num_cores_per_replica, host_trace_level, system_topology[], is_training, power_metrics[], hardware_type)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::RunEnvironment)
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
    x.host_trace_level != zero(UInt32) && PB.encode(e, 12, x.host_trace_level)
    !isnothing(x.system_topology) && PB.encode(e, 13, x.system_topology)
    x.is_training != false && PB.encode(e, 14, x.is_training)
    !isnothing(x.power_metrics) && PB.encode(e, 15, x.power_metrics)
    x.hardware_type != HardwareType.UNKNOWN_HARDWARE && PB.encode(e, 16, x.hardware_type)
    return position(e.io) - initpos
end
function PB._encoded_size(x::RunEnvironment)
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
    x.host_trace_level != zero(UInt32) && (encoded_size += PB._encoded_size(x.host_trace_level, 12))
    !isnothing(x.system_topology) && (encoded_size += PB._encoded_size(x.system_topology, 13))
    x.is_training != false && (encoded_size += PB._encoded_size(x.is_training, 14))
    !isnothing(x.power_metrics) && (encoded_size += PB._encoded_size(x.power_metrics, 15))
    x.hardware_type != HardwareType.UNKNOWN_HARDWARE && (encoded_size += PB._encoded_size(x.hardware_type, 16))
    return encoded_size
end

struct OpStats
    host_op_metrics_db::Union{Nothing,OpMetricsDb}
    device_op_metrics_db::Union{Nothing,OpMetricsDb}
    hlo_metrics_db_complete_steps_only::Union{Nothing,OpMetricsDb}
    perf_env::Union{Nothing,PerfEnv}
    step_db::Union{Nothing,StepDatabaseResult}
    run_environment::Union{Nothing,RunEnvironment}
    kernel_stats_db::Union{Nothing,KernelStatsDb}
    tf_function_db::Union{Nothing,TfFunctionDb}
    core_id_to_details::Dict{UInt32,CoreDetails}
    diagnostics::Union{Nothing,Diagnostics}
    program_id_to_name_map::Dict{UInt64,String}
    performance_counter_result::Union{Nothing,PerformanceCounterResult}
    source_stats::Union{Nothing,SourceStats}
end
OpStats(;host_op_metrics_db = nothing, device_op_metrics_db = nothing, hlo_metrics_db_complete_steps_only = nothing, perf_env = nothing, step_db = nothing, run_environment = nothing, kernel_stats_db = nothing, tf_function_db = nothing, core_id_to_details = Dict{UInt32,CoreDetails}(), diagnostics = nothing, program_id_to_name_map = Dict{UInt64,String}(), performance_counter_result = nothing, source_stats = nothing) = OpStats(host_op_metrics_db, device_op_metrics_db, hlo_metrics_db_complete_steps_only, perf_env, step_db, run_environment, kernel_stats_db, tf_function_db, core_id_to_details, diagnostics, program_id_to_name_map, performance_counter_result, source_stats)
PB.reserved_fields(::Type{OpStats}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[7])
PB.default_values(::Type{OpStats}) = (;host_op_metrics_db = nothing, device_op_metrics_db = nothing, hlo_metrics_db_complete_steps_only = nothing, perf_env = nothing, step_db = nothing, run_environment = nothing, kernel_stats_db = nothing, tf_function_db = nothing, core_id_to_details = Dict{UInt32,CoreDetails}(), diagnostics = nothing, program_id_to_name_map = Dict{UInt64,String}(), performance_counter_result = nothing, source_stats = nothing)
PB.field_numbers(::Type{OpStats}) = (;host_op_metrics_db = 1, device_op_metrics_db = 2, hlo_metrics_db_complete_steps_only = 10, perf_env = 3, step_db = 4, run_environment = 5, kernel_stats_db = 6, tf_function_db = 8, core_id_to_details = 11, diagnostics = 9, program_id_to_name_map = 12, performance_counter_result = 13, source_stats = 14)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OpStats})
    host_op_metrics_db = Ref{Union{Nothing,OpMetricsDb}}(nothing)
    device_op_metrics_db = Ref{Union{Nothing,OpMetricsDb}}(nothing)
    hlo_metrics_db_complete_steps_only = Ref{Union{Nothing,OpMetricsDb}}(nothing)
    perf_env = Ref{Union{Nothing,PerfEnv}}(nothing)
    step_db = Ref{Union{Nothing,StepDatabaseResult}}(nothing)
    run_environment = Ref{Union{Nothing,RunEnvironment}}(nothing)
    kernel_stats_db = Ref{Union{Nothing,KernelStatsDb}}(nothing)
    tf_function_db = Ref{Union{Nothing,TfFunctionDb}}(nothing)
    core_id_to_details = Dict{UInt32,CoreDetails}()
    diagnostics = Ref{Union{Nothing,Diagnostics}}(nothing)
    program_id_to_name_map = Dict{UInt64,String}()
    performance_counter_result = Ref{Union{Nothing,PerformanceCounterResult}}(nothing)
    source_stats = Ref{Union{Nothing,SourceStats}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, host_op_metrics_db)
        elseif field_number == 2
            PB.decode!(d, device_op_metrics_db)
        elseif field_number == 10
            PB.decode!(d, hlo_metrics_db_complete_steps_only)
        elseif field_number == 3
            PB.decode!(d, perf_env)
        elseif field_number == 4
            PB.decode!(d, step_db)
        elseif field_number == 5
            PB.decode!(d, run_environment)
        elseif field_number == 6
            PB.decode!(d, kernel_stats_db)
        elseif field_number == 8
            PB.decode!(d, tf_function_db)
        elseif field_number == 11
            PB.decode!(d, core_id_to_details)
        elseif field_number == 9
            PB.decode!(d, diagnostics)
        elseif field_number == 12
            PB.decode!(d, program_id_to_name_map)
        elseif field_number == 13
            PB.decode!(d, performance_counter_result)
        elseif field_number == 14
            PB.decode!(d, source_stats)
        else
            Base.skip(d, wire_type)
        end
    end
    return OpStats(host_op_metrics_db[], device_op_metrics_db[], hlo_metrics_db_complete_steps_only[], perf_env[], step_db[], run_environment[], kernel_stats_db[], tf_function_db[], core_id_to_details, diagnostics[], program_id_to_name_map, performance_counter_result[], source_stats[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OpStats)
    initpos = position(e.io)
    !isnothing(x.host_op_metrics_db) && PB.encode(e, 1, x.host_op_metrics_db)
    !isnothing(x.device_op_metrics_db) && PB.encode(e, 2, x.device_op_metrics_db)
    !isnothing(x.hlo_metrics_db_complete_steps_only) && PB.encode(e, 10, x.hlo_metrics_db_complete_steps_only)
    !isnothing(x.perf_env) && PB.encode(e, 3, x.perf_env)
    !isnothing(x.step_db) && PB.encode(e, 4, x.step_db)
    !isnothing(x.run_environment) && PB.encode(e, 5, x.run_environment)
    !isnothing(x.kernel_stats_db) && PB.encode(e, 6, x.kernel_stats_db)
    !isnothing(x.tf_function_db) && PB.encode(e, 8, x.tf_function_db)
    !isempty(x.core_id_to_details) && PB.encode(e, 11, x.core_id_to_details)
    !isnothing(x.diagnostics) && PB.encode(e, 9, x.diagnostics)
    !isempty(x.program_id_to_name_map) && PB.encode(e, 12, x.program_id_to_name_map)
    !isnothing(x.performance_counter_result) && PB.encode(e, 13, x.performance_counter_result)
    !isnothing(x.source_stats) && PB.encode(e, 14, x.source_stats)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OpStats)
    encoded_size = 0
    !isnothing(x.host_op_metrics_db) && (encoded_size += PB._encoded_size(x.host_op_metrics_db, 1))
    !isnothing(x.device_op_metrics_db) && (encoded_size += PB._encoded_size(x.device_op_metrics_db, 2))
    !isnothing(x.hlo_metrics_db_complete_steps_only) && (encoded_size += PB._encoded_size(x.hlo_metrics_db_complete_steps_only, 10))
    !isnothing(x.perf_env) && (encoded_size += PB._encoded_size(x.perf_env, 3))
    !isnothing(x.step_db) && (encoded_size += PB._encoded_size(x.step_db, 4))
    !isnothing(x.run_environment) && (encoded_size += PB._encoded_size(x.run_environment, 5))
    !isnothing(x.kernel_stats_db) && (encoded_size += PB._encoded_size(x.kernel_stats_db, 6))
    !isnothing(x.tf_function_db) && (encoded_size += PB._encoded_size(x.tf_function_db, 8))
    !isempty(x.core_id_to_details) && (encoded_size += PB._encoded_size(x.core_id_to_details, 11))
    !isnothing(x.diagnostics) && (encoded_size += PB._encoded_size(x.diagnostics, 9))
    !isempty(x.program_id_to_name_map) && (encoded_size += PB._encoded_size(x.program_id_to_name_map, 12))
    !isnothing(x.performance_counter_result) && (encoded_size += PB._encoded_size(x.performance_counter_result, 13))
    !isnothing(x.source_stats) && (encoded_size += PB._encoded_size(x.source_stats, 14))
    return encoded_size
end
