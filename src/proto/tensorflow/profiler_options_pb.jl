import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"ProfileOptions.TraceOptions", var"ProfileOptions.DeviceType"
export var"ProfileOptions.AdvancedConfigValue", ProfileOptions
export RemoteProfilerSessionManagerOptions


struct var"ProfileOptions.TraceOptions"
    host_traceme_filter_mask::UInt64
end
var"ProfileOptions.TraceOptions"(;host_traceme_filter_mask = zero(UInt64)) = var"ProfileOptions.TraceOptions"(host_traceme_filter_mask)
PB.default_values(::Type{var"ProfileOptions.TraceOptions"}) = (;host_traceme_filter_mask = zero(UInt64))
PB.field_numbers(::Type{var"ProfileOptions.TraceOptions"}) = (;host_traceme_filter_mask = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"ProfileOptions.TraceOptions"})
    host_traceme_filter_mask = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            host_traceme_filter_mask = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"ProfileOptions.TraceOptions"(host_traceme_filter_mask)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"ProfileOptions.TraceOptions")
    initpos = position(e.io)
    x.host_traceme_filter_mask != zero(UInt64) && PB.encode(e, 1, x.host_traceme_filter_mask)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"ProfileOptions.TraceOptions")
    encoded_size = 0
    x.host_traceme_filter_mask != zero(UInt64) && (encoded_size += PB._encoded_size(x.host_traceme_filter_mask, 1))
    return encoded_size
end

@enumx var"ProfileOptions.DeviceType" UNSPECIFIED=0 CPU=1 GPU=2 TPU=3 PLUGGABLE_DEVICE=4

struct var"ProfileOptions.AdvancedConfigValue"
    value::Union{Nothing,OneOf{<:Union{String,Bool,Int64}}}
end
var"ProfileOptions.AdvancedConfigValue"(;value = nothing) = var"ProfileOptions.AdvancedConfigValue"(value)
PB.oneof_field_types(::Type{var"ProfileOptions.AdvancedConfigValue"}) = (;
    value = (;string_value=String, bool_value=Bool, int64_value=Int64),
)
PB.default_values(::Type{var"ProfileOptions.AdvancedConfigValue"}) = (;string_value = "", bool_value = false, int64_value = zero(Int64))
PB.field_numbers(::Type{var"ProfileOptions.AdvancedConfigValue"}) = (;string_value = 1, bool_value = 2, int64_value = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"ProfileOptions.AdvancedConfigValue"})
    value = nothing
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            value = OneOf(:string_value, PB.decode(d, String))
        elseif field_number == 2
            value = OneOf(:bool_value, PB.decode(d, Bool))
        elseif field_number == 3
            value = OneOf(:int64_value, PB.decode(d, Int64))
        else
            Base.skip(d, wire_type)
        end
    end
    return var"ProfileOptions.AdvancedConfigValue"(value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"ProfileOptions.AdvancedConfigValue")
    initpos = position(e.io)
    if isnothing(x.value);
    elseif x.value.name === :string_value
        PB.encode(e, 1, x.value[]::String)
    elseif x.value.name === :bool_value
        PB.encode(e, 2, x.value[]::Bool)
    elseif x.value.name === :int64_value
        PB.encode(e, 3, x.value[]::Int64)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"ProfileOptions.AdvancedConfigValue")
    encoded_size = 0
    if isnothing(x.value);
    elseif x.value.name === :string_value
        encoded_size += PB._encoded_size(x.value[]::String, 1)
    elseif x.value.name === :bool_value
        encoded_size += PB._encoded_size(x.value[]::Bool, 2)
    elseif x.value.name === :int64_value
        encoded_size += PB._encoded_size(x.value[]::Int64, 3)
    end
    return encoded_size
end

struct ProfileOptions
    version::UInt32
    device_type::var"ProfileOptions.DeviceType".T
    include_dataset_ops::Bool
    host_tracer_level::UInt32
    device_tracer_level::UInt32
    python_tracer_level::UInt32
    enable_hlo_proto::Bool
    start_timestamp_ns::UInt64
    duration_ms::UInt64
    repository_path::String
    trace_options::Union{Nothing,var"ProfileOptions.TraceOptions"}
    advanced_configuration::Dict{String,var"ProfileOptions.AdvancedConfigValue"}
    raise_error_on_start_failure::Bool
    session_id::String
end
ProfileOptions(;version = zero(UInt32), device_type = var"ProfileOptions.DeviceType".UNSPECIFIED, include_dataset_ops = false, host_tracer_level = zero(UInt32), device_tracer_level = zero(UInt32), python_tracer_level = zero(UInt32), enable_hlo_proto = false, start_timestamp_ns = zero(UInt64), duration_ms = zero(UInt64), repository_path = "", trace_options = nothing, advanced_configuration = Dict{String,var"ProfileOptions.AdvancedConfigValue"}(), raise_error_on_start_failure = false, session_id = "") = ProfileOptions(version, device_type, include_dataset_ops, host_tracer_level, device_tracer_level, python_tracer_level, enable_hlo_proto, start_timestamp_ns, duration_ms, repository_path, trace_options, advanced_configuration, raise_error_on_start_failure, session_id)
PB.default_values(::Type{ProfileOptions}) = (;version = zero(UInt32), device_type = var"ProfileOptions.DeviceType".UNSPECIFIED, include_dataset_ops = false, host_tracer_level = zero(UInt32), device_tracer_level = zero(UInt32), python_tracer_level = zero(UInt32), enable_hlo_proto = false, start_timestamp_ns = zero(UInt64), duration_ms = zero(UInt64), repository_path = "", trace_options = nothing, advanced_configuration = Dict{String,var"ProfileOptions.AdvancedConfigValue"}(), raise_error_on_start_failure = false, session_id = "")
PB.field_numbers(::Type{ProfileOptions}) = (;version = 5, device_type = 6, include_dataset_ops = 1, host_tracer_level = 2, device_tracer_level = 3, python_tracer_level = 4, enable_hlo_proto = 7, start_timestamp_ns = 8, duration_ms = 9, repository_path = 10, trace_options = 11, advanced_configuration = 12, raise_error_on_start_failure = 13, session_id = 14)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ProfileOptions})
    version = zero(UInt32)
    device_type = var"ProfileOptions.DeviceType".UNSPECIFIED
    include_dataset_ops = false
    host_tracer_level = zero(UInt32)
    device_tracer_level = zero(UInt32)
    python_tracer_level = zero(UInt32)
    enable_hlo_proto = false
    start_timestamp_ns = zero(UInt64)
    duration_ms = zero(UInt64)
    repository_path = ""
    trace_options = Ref{Union{Nothing,var"ProfileOptions.TraceOptions"}}(nothing)
    advanced_configuration = Dict{String,var"ProfileOptions.AdvancedConfigValue"}()
    raise_error_on_start_failure = false
    session_id = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 5
            version = PB.decode(d, UInt32)
        elseif field_number == 6
            device_type = PB.decode(d, var"ProfileOptions.DeviceType".T)
        elseif field_number == 1
            include_dataset_ops = PB.decode(d, Bool)
        elseif field_number == 2
            host_tracer_level = PB.decode(d, UInt32)
        elseif field_number == 3
            device_tracer_level = PB.decode(d, UInt32)
        elseif field_number == 4
            python_tracer_level = PB.decode(d, UInt32)
        elseif field_number == 7
            enable_hlo_proto = PB.decode(d, Bool)
        elseif field_number == 8
            start_timestamp_ns = PB.decode(d, UInt64)
        elseif field_number == 9
            duration_ms = PB.decode(d, UInt64)
        elseif field_number == 10
            repository_path = PB.decode(d, String)
        elseif field_number == 11
            PB.decode!(d, trace_options)
        elseif field_number == 12
            PB.decode!(d, advanced_configuration)
        elseif field_number == 13
            raise_error_on_start_failure = PB.decode(d, Bool)
        elseif field_number == 14
            session_id = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return ProfileOptions(version, device_type, include_dataset_ops, host_tracer_level, device_tracer_level, python_tracer_level, enable_hlo_proto, start_timestamp_ns, duration_ms, repository_path, trace_options[], advanced_configuration, raise_error_on_start_failure, session_id)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ProfileOptions)
    initpos = position(e.io)
    x.version != zero(UInt32) && PB.encode(e, 5, x.version)
    x.device_type != var"ProfileOptions.DeviceType".UNSPECIFIED && PB.encode(e, 6, x.device_type)
    x.include_dataset_ops != false && PB.encode(e, 1, x.include_dataset_ops)
    x.host_tracer_level != zero(UInt32) && PB.encode(e, 2, x.host_tracer_level)
    x.device_tracer_level != zero(UInt32) && PB.encode(e, 3, x.device_tracer_level)
    x.python_tracer_level != zero(UInt32) && PB.encode(e, 4, x.python_tracer_level)
    x.enable_hlo_proto != false && PB.encode(e, 7, x.enable_hlo_proto)
    x.start_timestamp_ns != zero(UInt64) && PB.encode(e, 8, x.start_timestamp_ns)
    x.duration_ms != zero(UInt64) && PB.encode(e, 9, x.duration_ms)
    !isempty(x.repository_path) && PB.encode(e, 10, x.repository_path)
    !isnothing(x.trace_options) && PB.encode(e, 11, x.trace_options)
    !isempty(x.advanced_configuration) && PB.encode(e, 12, x.advanced_configuration)
    x.raise_error_on_start_failure != false && PB.encode(e, 13, x.raise_error_on_start_failure)
    !isempty(x.session_id) && PB.encode(e, 14, x.session_id)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ProfileOptions)
    encoded_size = 0
    x.version != zero(UInt32) && (encoded_size += PB._encoded_size(x.version, 5))
    x.device_type != var"ProfileOptions.DeviceType".UNSPECIFIED && (encoded_size += PB._encoded_size(x.device_type, 6))
    x.include_dataset_ops != false && (encoded_size += PB._encoded_size(x.include_dataset_ops, 1))
    x.host_tracer_level != zero(UInt32) && (encoded_size += PB._encoded_size(x.host_tracer_level, 2))
    x.device_tracer_level != zero(UInt32) && (encoded_size += PB._encoded_size(x.device_tracer_level, 3))
    x.python_tracer_level != zero(UInt32) && (encoded_size += PB._encoded_size(x.python_tracer_level, 4))
    x.enable_hlo_proto != false && (encoded_size += PB._encoded_size(x.enable_hlo_proto, 7))
    x.start_timestamp_ns != zero(UInt64) && (encoded_size += PB._encoded_size(x.start_timestamp_ns, 8))
    x.duration_ms != zero(UInt64) && (encoded_size += PB._encoded_size(x.duration_ms, 9))
    !isempty(x.repository_path) && (encoded_size += PB._encoded_size(x.repository_path, 10))
    !isnothing(x.trace_options) && (encoded_size += PB._encoded_size(x.trace_options, 11))
    !isempty(x.advanced_configuration) && (encoded_size += PB._encoded_size(x.advanced_configuration, 12))
    x.raise_error_on_start_failure != false && (encoded_size += PB._encoded_size(x.raise_error_on_start_failure, 13))
    !isempty(x.session_id) && (encoded_size += PB._encoded_size(x.session_id, 14))
    return encoded_size
end

struct RemoteProfilerSessionManagerOptions
    profiler_options::Union{Nothing,ProfileOptions}
    service_addresses::Vector{String}
    session_creation_timestamp_ns::UInt64
    max_session_duration_ms::UInt64
    delay_ms::UInt64
end
RemoteProfilerSessionManagerOptions(;profiler_options = nothing, service_addresses = Vector{String}(), session_creation_timestamp_ns = zero(UInt64), max_session_duration_ms = zero(UInt64), delay_ms = zero(UInt64)) = RemoteProfilerSessionManagerOptions(profiler_options, service_addresses, session_creation_timestamp_ns, max_session_duration_ms, delay_ms)
PB.default_values(::Type{RemoteProfilerSessionManagerOptions}) = (;profiler_options = nothing, service_addresses = Vector{String}(), session_creation_timestamp_ns = zero(UInt64), max_session_duration_ms = zero(UInt64), delay_ms = zero(UInt64))
PB.field_numbers(::Type{RemoteProfilerSessionManagerOptions}) = (;profiler_options = 1, service_addresses = 2, session_creation_timestamp_ns = 3, max_session_duration_ms = 4, delay_ms = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:RemoteProfilerSessionManagerOptions})
    profiler_options = Ref{Union{Nothing,ProfileOptions}}(nothing)
    service_addresses = PB.BufferedVector{String}()
    session_creation_timestamp_ns = zero(UInt64)
    max_session_duration_ms = zero(UInt64)
    delay_ms = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, profiler_options)
        elseif field_number == 2
            PB.decode!(d, service_addresses)
        elseif field_number == 3
            session_creation_timestamp_ns = PB.decode(d, UInt64)
        elseif field_number == 4
            max_session_duration_ms = PB.decode(d, UInt64)
        elseif field_number == 5
            delay_ms = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return RemoteProfilerSessionManagerOptions(profiler_options[], service_addresses[], session_creation_timestamp_ns, max_session_duration_ms, delay_ms)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::RemoteProfilerSessionManagerOptions)
    initpos = position(e.io)
    !isnothing(x.profiler_options) && PB.encode(e, 1, x.profiler_options)
    !isempty(x.service_addresses) && PB.encode(e, 2, x.service_addresses)
    x.session_creation_timestamp_ns != zero(UInt64) && PB.encode(e, 3, x.session_creation_timestamp_ns)
    x.max_session_duration_ms != zero(UInt64) && PB.encode(e, 4, x.max_session_duration_ms)
    x.delay_ms != zero(UInt64) && PB.encode(e, 5, x.delay_ms)
    return position(e.io) - initpos
end
function PB._encoded_size(x::RemoteProfilerSessionManagerOptions)
    encoded_size = 0
    !isnothing(x.profiler_options) && (encoded_size += PB._encoded_size(x.profiler_options, 1))
    !isempty(x.service_addresses) && (encoded_size += PB._encoded_size(x.service_addresses, 2))
    x.session_creation_timestamp_ns != zero(UInt64) && (encoded_size += PB._encoded_size(x.session_creation_timestamp_ns, 3))
    x.max_session_duration_ms != zero(UInt64) && (encoded_size += PB._encoded_size(x.max_session_duration_ms, 4))
    x.delay_ms != zero(UInt64) && (encoded_size += PB._encoded_size(x.delay_ms, 5))
    return encoded_size
end
