import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export JobInfo, TagMetric, var"CompilationLogEntry.CompilationStage", KeyValueMetric
export PassMetrics, CompilationLogEntry


struct JobInfo
    name::String
    cell::String
    user::String
    uid::Int64
    task_id::Int64
    task_uid::Int64
    process_id::Int64
    thread_id::Int64
end
JobInfo(;name = "", cell = "", user = "", uid = zero(Int64), task_id = zero(Int64), task_uid = zero(Int64), process_id = zero(Int64), thread_id = zero(Int64)) = JobInfo(name, cell, user, uid, task_id, task_uid, process_id, thread_id)
PB.default_values(::Type{JobInfo}) = (;name = "", cell = "", user = "", uid = zero(Int64), task_id = zero(Int64), task_uid = zero(Int64), process_id = zero(Int64), thread_id = zero(Int64))
PB.field_numbers(::Type{JobInfo}) = (;name = 1, cell = 2, user = 3, uid = 4, task_id = 5, task_uid = 6, process_id = 7, thread_id = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:JobInfo})
    name = ""
    cell = ""
    user = ""
    uid = zero(Int64)
    task_id = zero(Int64)
    task_uid = zero(Int64)
    process_id = zero(Int64)
    thread_id = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            cell = PB.decode(d, String)
        elseif field_number == 3
            user = PB.decode(d, String)
        elseif field_number == 4
            uid = PB.decode(d, Int64)
        elseif field_number == 5
            task_id = PB.decode(d, Int64)
        elseif field_number == 6
            task_uid = PB.decode(d, Int64)
        elseif field_number == 7
            process_id = PB.decode(d, Int64)
        elseif field_number == 8
            thread_id = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return JobInfo(name, cell, user, uid, task_id, task_uid, process_id, thread_id)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::JobInfo)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    !isempty(x.cell) && PB.encode(e, 2, x.cell)
    !isempty(x.user) && PB.encode(e, 3, x.user)
    x.uid != zero(Int64) && PB.encode(e, 4, x.uid)
    x.task_id != zero(Int64) && PB.encode(e, 5, x.task_id)
    x.task_uid != zero(Int64) && PB.encode(e, 6, x.task_uid)
    x.process_id != zero(Int64) && PB.encode(e, 7, x.process_id)
    x.thread_id != zero(Int64) && PB.encode(e, 8, x.thread_id)
    return position(e.io) - initpos
end
function PB._encoded_size(x::JobInfo)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    !isempty(x.cell) && (encoded_size += PB._encoded_size(x.cell, 2))
    !isempty(x.user) && (encoded_size += PB._encoded_size(x.user, 3))
    x.uid != zero(Int64) && (encoded_size += PB._encoded_size(x.uid, 4))
    x.task_id != zero(Int64) && (encoded_size += PB._encoded_size(x.task_id, 5))
    x.task_uid != zero(Int64) && (encoded_size += PB._encoded_size(x.task_uid, 6))
    x.process_id != zero(Int64) && (encoded_size += PB._encoded_size(x.process_id, 7))
    x.thread_id != zero(Int64) && (encoded_size += PB._encoded_size(x.thread_id, 8))
    return encoded_size
end

struct TagMetric
    key::String
    value::String
end
TagMetric(;key = "", value = "") = TagMetric(key, value)
PB.default_values(::Type{TagMetric}) = (;key = "", value = "")
PB.field_numbers(::Type{TagMetric}) = (;key = 1, value = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TagMetric})
    key = ""
    value = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            key = PB.decode(d, String)
        elseif field_number == 2
            value = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return TagMetric(key, value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TagMetric)
    initpos = position(e.io)
    !isempty(x.key) && PB.encode(e, 1, x.key)
    !isempty(x.value) && PB.encode(e, 2, x.value)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TagMetric)
    encoded_size = 0
    !isempty(x.key) && (encoded_size += PB._encoded_size(x.key, 1))
    !isempty(x.value) && (encoded_size += PB._encoded_size(x.value, 2))
    return encoded_size
end

@enumx var"CompilationLogEntry.CompilationStage" UNSPECIFIED=0 END_TO_END=1 HLO_PASSES=2 CODE_GENERATION=3 BACKEND_PASSES=4

struct KeyValueMetric
    key::String
    value::Int64
end
KeyValueMetric(;key = "", value = zero(Int64)) = KeyValueMetric(key, value)
PB.default_values(::Type{KeyValueMetric}) = (;key = "", value = zero(Int64))
PB.field_numbers(::Type{KeyValueMetric}) = (;key = 1, value = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:KeyValueMetric})
    key = ""
    value = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            key = PB.decode(d, String)
        elseif field_number == 2
            value = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return KeyValueMetric(key, value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::KeyValueMetric)
    initpos = position(e.io)
    !isempty(x.key) && PB.encode(e, 1, x.key)
    x.value != zero(Int64) && PB.encode(e, 2, x.value)
    return position(e.io) - initpos
end
function PB._encoded_size(x::KeyValueMetric)
    encoded_size = 0
    !isempty(x.key) && (encoded_size += PB._encoded_size(x.key, 1))
    x.value != zero(Int64) && (encoded_size += PB._encoded_size(x.value, 2))
    return encoded_size
end

struct PassMetrics
    module_id::UInt64
    pass_name::String
    pass_duration::Union{Nothing,google.protobuf.Duration}
    custom_metrics::Union{Nothing,google.protobuf.var"#Any"}
    kv_metrics::Vector{KeyValueMetric}
end
PassMetrics(;module_id = zero(UInt64), pass_name = "", pass_duration = nothing, custom_metrics = nothing, kv_metrics = Vector{KeyValueMetric}()) = PassMetrics(module_id, pass_name, pass_duration, custom_metrics, kv_metrics)
PB.default_values(::Type{PassMetrics}) = (;module_id = zero(UInt64), pass_name = "", pass_duration = nothing, custom_metrics = nothing, kv_metrics = Vector{KeyValueMetric}())
PB.field_numbers(::Type{PassMetrics}) = (;module_id = 1, pass_name = 2, pass_duration = 3, custom_metrics = 4, kv_metrics = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PassMetrics})
    module_id = zero(UInt64)
    pass_name = ""
    pass_duration = Ref{Union{Nothing,google.protobuf.Duration}}(nothing)
    custom_metrics = Ref{Union{Nothing,google.protobuf.var"#Any"}}(nothing)
    kv_metrics = PB.BufferedVector{KeyValueMetric}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            module_id = PB.decode(d, UInt64)
        elseif field_number == 2
            pass_name = PB.decode(d, String)
        elseif field_number == 3
            PB.decode!(d, pass_duration)
        elseif field_number == 4
            PB.decode!(d, custom_metrics)
        elseif field_number == 5
            PB.decode!(d, kv_metrics)
        else
            Base.skip(d, wire_type)
        end
    end
    return PassMetrics(module_id, pass_name, pass_duration[], custom_metrics[], kv_metrics[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PassMetrics)
    initpos = position(e.io)
    x.module_id != zero(UInt64) && PB.encode(e, 1, x.module_id)
    !isempty(x.pass_name) && PB.encode(e, 2, x.pass_name)
    !isnothing(x.pass_duration) && PB.encode(e, 3, x.pass_duration)
    !isnothing(x.custom_metrics) && PB.encode(e, 4, x.custom_metrics)
    !isempty(x.kv_metrics) && PB.encode(e, 5, x.kv_metrics)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PassMetrics)
    encoded_size = 0
    x.module_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.module_id, 1))
    !isempty(x.pass_name) && (encoded_size += PB._encoded_size(x.pass_name, 2))
    !isnothing(x.pass_duration) && (encoded_size += PB._encoded_size(x.pass_duration, 3))
    !isnothing(x.custom_metrics) && (encoded_size += PB._encoded_size(x.custom_metrics, 4))
    !isempty(x.kv_metrics) && (encoded_size += PB._encoded_size(x.kv_metrics, 5))
    return encoded_size
end

struct CompilationLogEntry
    timestamp::Union{Nothing,google.protobuf.Timestamp}
    stage::var"CompilationLogEntry.CompilationStage".T
    duration::Union{Nothing,google.protobuf.Duration}
    task_index::Int32
    pass_metrics::Vector{PassMetrics}
    module_ids::Vector{UInt64}
    job_info::Union{Nothing,JobInfo}
    hlo_module_name::String
    tag::Vector{TagMetric}
end
CompilationLogEntry(;timestamp = nothing, stage = var"CompilationLogEntry.CompilationStage".UNSPECIFIED, duration = nothing, task_index = zero(Int32), pass_metrics = Vector{PassMetrics}(), module_ids = Vector{UInt64}(), job_info = nothing, hlo_module_name = "", tag = Vector{TagMetric}()) = CompilationLogEntry(timestamp, stage, duration, task_index, pass_metrics, module_ids, job_info, hlo_module_name, tag)
PB.default_values(::Type{CompilationLogEntry}) = (;timestamp = nothing, stage = var"CompilationLogEntry.CompilationStage".UNSPECIFIED, duration = nothing, task_index = zero(Int32), pass_metrics = Vector{PassMetrics}(), module_ids = Vector{UInt64}(), job_info = nothing, hlo_module_name = "", tag = Vector{TagMetric}())
PB.field_numbers(::Type{CompilationLogEntry}) = (;timestamp = 1, stage = 2, duration = 3, task_index = 4, pass_metrics = 5, module_ids = 6, job_info = 7, hlo_module_name = 8, tag = 9)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CompilationLogEntry})
    timestamp = Ref{Union{Nothing,google.protobuf.Timestamp}}(nothing)
    stage = var"CompilationLogEntry.CompilationStage".UNSPECIFIED
    duration = Ref{Union{Nothing,google.protobuf.Duration}}(nothing)
    task_index = zero(Int32)
    pass_metrics = PB.BufferedVector{PassMetrics}()
    module_ids = PB.BufferedVector{UInt64}()
    job_info = Ref{Union{Nothing,JobInfo}}(nothing)
    hlo_module_name = ""
    tag = PB.BufferedVector{TagMetric}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, timestamp)
        elseif field_number == 2
            stage = PB.decode(d, var"CompilationLogEntry.CompilationStage".T)
        elseif field_number == 3
            PB.decode!(d, duration)
        elseif field_number == 4
            task_index = PB.decode(d, Int32)
        elseif field_number == 5
            PB.decode!(d, pass_metrics)
        elseif field_number == 6
            PB.decode!(d, wire_type, module_ids)
        elseif field_number == 7
            PB.decode!(d, job_info)
        elseif field_number == 8
            hlo_module_name = PB.decode(d, String)
        elseif field_number == 9
            PB.decode!(d, tag)
        else
            Base.skip(d, wire_type)
        end
    end
    return CompilationLogEntry(timestamp[], stage, duration[], task_index, pass_metrics[], module_ids[], job_info[], hlo_module_name, tag[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CompilationLogEntry)
    initpos = position(e.io)
    !isnothing(x.timestamp) && PB.encode(e, 1, x.timestamp)
    x.stage != var"CompilationLogEntry.CompilationStage".UNSPECIFIED && PB.encode(e, 2, x.stage)
    !isnothing(x.duration) && PB.encode(e, 3, x.duration)
    x.task_index != zero(Int32) && PB.encode(e, 4, x.task_index)
    !isempty(x.pass_metrics) && PB.encode(e, 5, x.pass_metrics)
    !isempty(x.module_ids) && PB.encode(e, 6, x.module_ids)
    !isnothing(x.job_info) && PB.encode(e, 7, x.job_info)
    !isempty(x.hlo_module_name) && PB.encode(e, 8, x.hlo_module_name)
    !isempty(x.tag) && PB.encode(e, 9, x.tag)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CompilationLogEntry)
    encoded_size = 0
    !isnothing(x.timestamp) && (encoded_size += PB._encoded_size(x.timestamp, 1))
    x.stage != var"CompilationLogEntry.CompilationStage".UNSPECIFIED && (encoded_size += PB._encoded_size(x.stage, 2))
    !isnothing(x.duration) && (encoded_size += PB._encoded_size(x.duration, 3))
    x.task_index != zero(Int32) && (encoded_size += PB._encoded_size(x.task_index, 4))
    !isempty(x.pass_metrics) && (encoded_size += PB._encoded_size(x.pass_metrics, 5))
    !isempty(x.module_ids) && (encoded_size += PB._encoded_size(x.module_ids, 6))
    !isnothing(x.job_info) && (encoded_size += PB._encoded_size(x.job_info, 7))
    !isempty(x.hlo_module_name) && (encoded_size += PB._encoded_size(x.hlo_module_name, 8))
    !isempty(x.tag) && (encoded_size += PB._encoded_size(x.tag, 9))
    return encoded_size
end
