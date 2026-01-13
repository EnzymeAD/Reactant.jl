import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export Resource, TraceEvent, Device, Trace


struct Resource
    name::String
    resource_id::UInt32
    sort_index::UInt32
end
Resource(;name = "", resource_id = zero(UInt32), sort_index = zero(UInt32)) = Resource(name, resource_id, sort_index)
PB.default_values(::Type{Resource}) = (;name = "", resource_id = zero(UInt32), sort_index = zero(UInt32))
PB.field_numbers(::Type{Resource}) = (;name = 1, resource_id = 2, sort_index = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Resource})
    name = ""
    resource_id = zero(UInt32)
    sort_index = zero(UInt32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            resource_id = PB.decode(d, UInt32)
        elseif field_number == 3
            sort_index = PB.decode(d, UInt32)
        else
            Base.skip(d, wire_type)
        end
    end
    return Resource(name, resource_id, sort_index)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Resource)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    x.resource_id != zero(UInt32) && PB.encode(e, 2, x.resource_id)
    x.sort_index != zero(UInt32) && PB.encode(e, 3, x.sort_index)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Resource)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    x.resource_id != zero(UInt32) && (encoded_size += PB._encoded_size(x.resource_id, 2))
    x.sort_index != zero(UInt32) && (encoded_size += PB._encoded_size(x.sort_index, 3))
    return encoded_size
end

struct TraceEvent
    device_id::UInt32
    resource_id::UInt32
    name::String
    timestamp_ps::UInt64
    duration_ps::UInt64
    args::Dict{String,String}
end
TraceEvent(;device_id = zero(UInt32), resource_id = zero(UInt32), name = "", timestamp_ps = zero(UInt64), duration_ps = zero(UInt64), args = Dict{String,String}()) = TraceEvent(device_id, resource_id, name, timestamp_ps, duration_ps, args)
PB.default_values(::Type{TraceEvent}) = (;device_id = zero(UInt32), resource_id = zero(UInt32), name = "", timestamp_ps = zero(UInt64), duration_ps = zero(UInt64), args = Dict{String,String}())
PB.field_numbers(::Type{TraceEvent}) = (;device_id = 1, resource_id = 2, name = 3, timestamp_ps = 9, duration_ps = 10, args = 11)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TraceEvent})
    device_id = zero(UInt32)
    resource_id = zero(UInt32)
    name = ""
    timestamp_ps = zero(UInt64)
    duration_ps = zero(UInt64)
    args = Dict{String,String}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            device_id = PB.decode(d, UInt32)
        elseif field_number == 2
            resource_id = PB.decode(d, UInt32)
        elseif field_number == 3
            name = PB.decode(d, String)
        elseif field_number == 9
            timestamp_ps = PB.decode(d, UInt64)
        elseif field_number == 10
            duration_ps = PB.decode(d, UInt64)
        elseif field_number == 11
            PB.decode!(d, args)
        else
            Base.skip(d, wire_type)
        end
    end
    return TraceEvent(device_id, resource_id, name, timestamp_ps, duration_ps, args)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TraceEvent)
    initpos = position(e.io)
    x.device_id != zero(UInt32) && PB.encode(e, 1, x.device_id)
    x.resource_id != zero(UInt32) && PB.encode(e, 2, x.resource_id)
    !isempty(x.name) && PB.encode(e, 3, x.name)
    x.timestamp_ps != zero(UInt64) && PB.encode(e, 9, x.timestamp_ps)
    x.duration_ps != zero(UInt64) && PB.encode(e, 10, x.duration_ps)
    !isempty(x.args) && PB.encode(e, 11, x.args)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TraceEvent)
    encoded_size = 0
    x.device_id != zero(UInt32) && (encoded_size += PB._encoded_size(x.device_id, 1))
    x.resource_id != zero(UInt32) && (encoded_size += PB._encoded_size(x.resource_id, 2))
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 3))
    x.timestamp_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.timestamp_ps, 9))
    x.duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.duration_ps, 10))
    !isempty(x.args) && (encoded_size += PB._encoded_size(x.args, 11))
    return encoded_size
end

struct Device
    name::String
    device_id::UInt32
    resources::Dict{UInt32,Resource}
end
Device(;name = "", device_id = zero(UInt32), resources = Dict{UInt32,Resource}()) = Device(name, device_id, resources)
PB.default_values(::Type{Device}) = (;name = "", device_id = zero(UInt32), resources = Dict{UInt32,Resource}())
PB.field_numbers(::Type{Device}) = (;name = 1, device_id = 2, resources = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Device})
    name = ""
    device_id = zero(UInt32)
    resources = Dict{UInt32,Resource}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            device_id = PB.decode(d, UInt32)
        elseif field_number == 3
            PB.decode!(d, resources)
        else
            Base.skip(d, wire_type)
        end
    end
    return Device(name, device_id, resources)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Device)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    x.device_id != zero(UInt32) && PB.encode(e, 2, x.device_id)
    !isempty(x.resources) && PB.encode(e, 3, x.resources)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Device)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    x.device_id != zero(UInt32) && (encoded_size += PB._encoded_size(x.device_id, 2))
    !isempty(x.resources) && (encoded_size += PB._encoded_size(x.resources, 3))
    return encoded_size
end

struct Trace
    devices::Dict{UInt32,Device}
    trace_events::Vector{TraceEvent}
end
Trace(;devices = Dict{UInt32,Device}(), trace_events = Vector{TraceEvent}()) = Trace(devices, trace_events)
PB.default_values(::Type{Trace}) = (;devices = Dict{UInt32,Device}(), trace_events = Vector{TraceEvent}())
PB.field_numbers(::Type{Trace}) = (;devices = 1, trace_events = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Trace})
    devices = Dict{UInt32,Device}()
    trace_events = PB.BufferedVector{TraceEvent}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, devices)
        elseif field_number == 4
            PB.decode!(d, trace_events)
        else
            Base.skip(d, wire_type)
        end
    end
    return Trace(devices, trace_events[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Trace)
    initpos = position(e.io)
    !isempty(x.devices) && PB.encode(e, 1, x.devices)
    !isempty(x.trace_events) && PB.encode(e, 4, x.trace_events)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Trace)
    encoded_size = 0
    !isempty(x.devices) && (encoded_size += PB._encoded_size(x.devices, 1))
    !isempty(x.trace_events) && (encoded_size += PB._encoded_size(x.trace_events, 4))
    return encoded_size
end
