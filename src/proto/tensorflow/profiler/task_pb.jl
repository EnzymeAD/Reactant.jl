import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export Task


struct Task
    changelist::Int64
    workspace_id::String
    snapshot::Int64
    clean_build::Bool
    build_time::Int64
    build_target::String
    command_line::String
    start_time::Int64
    task_address::String
    profile_time_ns::UInt64
    profile_duration_ms::UInt32
    host_trace_level::UInt32
    tensor_core_freq_hz::UInt64
    sparse_core_freq_hz::UInt64
    gtc_freq_hz::UInt64
    peak_memory_usage::UInt64
    cpu_limit::Float64
    cpu_usage::Float64
end
Task(;changelist = zero(Int64), workspace_id = "", snapshot = zero(Int64), clean_build = false, build_time = zero(Int64), build_target = "", command_line = "", start_time = zero(Int64), task_address = "", profile_time_ns = zero(UInt64), profile_duration_ms = zero(UInt32), host_trace_level = zero(UInt32), tensor_core_freq_hz = zero(UInt64), sparse_core_freq_hz = zero(UInt64), gtc_freq_hz = zero(UInt64), peak_memory_usage = zero(UInt64), cpu_limit = zero(Float64), cpu_usage = zero(Float64)) = Task(changelist, workspace_id, snapshot, clean_build, build_time, build_target, command_line, start_time, task_address, profile_time_ns, profile_duration_ms, host_trace_level, tensor_core_freq_hz, sparse_core_freq_hz, gtc_freq_hz, peak_memory_usage, cpu_limit, cpu_usage)
PB.default_values(::Type{Task}) = (;changelist = zero(Int64), workspace_id = "", snapshot = zero(Int64), clean_build = false, build_time = zero(Int64), build_target = "", command_line = "", start_time = zero(Int64), task_address = "", profile_time_ns = zero(UInt64), profile_duration_ms = zero(UInt32), host_trace_level = zero(UInt32), tensor_core_freq_hz = zero(UInt64), sparse_core_freq_hz = zero(UInt64), gtc_freq_hz = zero(UInt64), peak_memory_usage = zero(UInt64), cpu_limit = zero(Float64), cpu_usage = zero(Float64))
PB.field_numbers(::Type{Task}) = (;changelist = 1, workspace_id = 17, snapshot = 18, clean_build = 2, build_time = 3, build_target = 4, command_line = 5, start_time = 6, task_address = 7, profile_time_ns = 8, profile_duration_ms = 9, host_trace_level = 10, tensor_core_freq_hz = 11, sparse_core_freq_hz = 12, gtc_freq_hz = 13, peak_memory_usage = 14, cpu_limit = 15, cpu_usage = 16)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Task})
    changelist = zero(Int64)
    workspace_id = ""
    snapshot = zero(Int64)
    clean_build = false
    build_time = zero(Int64)
    build_target = ""
    command_line = ""
    start_time = zero(Int64)
    task_address = ""
    profile_time_ns = zero(UInt64)
    profile_duration_ms = zero(UInt32)
    host_trace_level = zero(UInt32)
    tensor_core_freq_hz = zero(UInt64)
    sparse_core_freq_hz = zero(UInt64)
    gtc_freq_hz = zero(UInt64)
    peak_memory_usage = zero(UInt64)
    cpu_limit = zero(Float64)
    cpu_usage = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            changelist = PB.decode(d, Int64)
        elseif field_number == 17
            workspace_id = PB.decode(d, String)
        elseif field_number == 18
            snapshot = PB.decode(d, Int64)
        elseif field_number == 2
            clean_build = PB.decode(d, Bool)
        elseif field_number == 3
            build_time = PB.decode(d, Int64)
        elseif field_number == 4
            build_target = PB.decode(d, String)
        elseif field_number == 5
            command_line = PB.decode(d, String)
        elseif field_number == 6
            start_time = PB.decode(d, Int64)
        elseif field_number == 7
            task_address = PB.decode(d, String)
        elseif field_number == 8
            profile_time_ns = PB.decode(d, UInt64)
        elseif field_number == 9
            profile_duration_ms = PB.decode(d, UInt32)
        elseif field_number == 10
            host_trace_level = PB.decode(d, UInt32)
        elseif field_number == 11
            tensor_core_freq_hz = PB.decode(d, UInt64)
        elseif field_number == 12
            sparse_core_freq_hz = PB.decode(d, UInt64)
        elseif field_number == 13
            gtc_freq_hz = PB.decode(d, UInt64)
        elseif field_number == 14
            peak_memory_usage = PB.decode(d, UInt64)
        elseif field_number == 15
            cpu_limit = PB.decode(d, Float64)
        elseif field_number == 16
            cpu_usage = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return Task(changelist, workspace_id, snapshot, clean_build, build_time, build_target, command_line, start_time, task_address, profile_time_ns, profile_duration_ms, host_trace_level, tensor_core_freq_hz, sparse_core_freq_hz, gtc_freq_hz, peak_memory_usage, cpu_limit, cpu_usage)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Task)
    initpos = position(e.io)
    x.changelist != zero(Int64) && PB.encode(e, 1, x.changelist)
    !isempty(x.workspace_id) && PB.encode(e, 17, x.workspace_id)
    x.snapshot != zero(Int64) && PB.encode(e, 18, x.snapshot)
    x.clean_build != false && PB.encode(e, 2, x.clean_build)
    x.build_time != zero(Int64) && PB.encode(e, 3, x.build_time)
    !isempty(x.build_target) && PB.encode(e, 4, x.build_target)
    !isempty(x.command_line) && PB.encode(e, 5, x.command_line)
    x.start_time != zero(Int64) && PB.encode(e, 6, x.start_time)
    !isempty(x.task_address) && PB.encode(e, 7, x.task_address)
    x.profile_time_ns != zero(UInt64) && PB.encode(e, 8, x.profile_time_ns)
    x.profile_duration_ms != zero(UInt32) && PB.encode(e, 9, x.profile_duration_ms)
    x.host_trace_level != zero(UInt32) && PB.encode(e, 10, x.host_trace_level)
    x.tensor_core_freq_hz != zero(UInt64) && PB.encode(e, 11, x.tensor_core_freq_hz)
    x.sparse_core_freq_hz != zero(UInt64) && PB.encode(e, 12, x.sparse_core_freq_hz)
    x.gtc_freq_hz != zero(UInt64) && PB.encode(e, 13, x.gtc_freq_hz)
    x.peak_memory_usage != zero(UInt64) && PB.encode(e, 14, x.peak_memory_usage)
    x.cpu_limit !== zero(Float64) && PB.encode(e, 15, x.cpu_limit)
    x.cpu_usage !== zero(Float64) && PB.encode(e, 16, x.cpu_usage)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Task)
    encoded_size = 0
    x.changelist != zero(Int64) && (encoded_size += PB._encoded_size(x.changelist, 1))
    !isempty(x.workspace_id) && (encoded_size += PB._encoded_size(x.workspace_id, 17))
    x.snapshot != zero(Int64) && (encoded_size += PB._encoded_size(x.snapshot, 18))
    x.clean_build != false && (encoded_size += PB._encoded_size(x.clean_build, 2))
    x.build_time != zero(Int64) && (encoded_size += PB._encoded_size(x.build_time, 3))
    !isempty(x.build_target) && (encoded_size += PB._encoded_size(x.build_target, 4))
    !isempty(x.command_line) && (encoded_size += PB._encoded_size(x.command_line, 5))
    x.start_time != zero(Int64) && (encoded_size += PB._encoded_size(x.start_time, 6))
    !isempty(x.task_address) && (encoded_size += PB._encoded_size(x.task_address, 7))
    x.profile_time_ns != zero(UInt64) && (encoded_size += PB._encoded_size(x.profile_time_ns, 8))
    x.profile_duration_ms != zero(UInt32) && (encoded_size += PB._encoded_size(x.profile_duration_ms, 9))
    x.host_trace_level != zero(UInt32) && (encoded_size += PB._encoded_size(x.host_trace_level, 10))
    x.tensor_core_freq_hz != zero(UInt64) && (encoded_size += PB._encoded_size(x.tensor_core_freq_hz, 11))
    x.sparse_core_freq_hz != zero(UInt64) && (encoded_size += PB._encoded_size(x.sparse_core_freq_hz, 12))
    x.gtc_freq_hz != zero(UInt64) && (encoded_size += PB._encoded_size(x.gtc_freq_hz, 13))
    x.peak_memory_usage != zero(UInt64) && (encoded_size += PB._encoded_size(x.peak_memory_usage, 14))
    x.cpu_limit !== zero(Float64) && (encoded_size += PB._encoded_size(x.cpu_limit, 15))
    x.cpu_usage !== zero(Float64) && (encoded_size += PB._encoded_size(x.cpu_usage, 16))
    return encoded_size
end
