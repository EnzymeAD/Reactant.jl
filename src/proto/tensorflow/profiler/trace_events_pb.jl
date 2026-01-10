import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"TraceEvent.FlowEntryType", var"TraceEvent.EventType", Resource, TraceEvent
export Device, Trace


@enumx var"TraceEvent.FlowEntryType" FLOW_NONE=0 FLOW_START=1 FLOW_MID=2 FLOW_END=3

@enumx var"TraceEvent.EventType" EVENT_TYPE_UNSPECIFIED=0 EVENT_TYPE_COMPLETE=1 EVENT_TYPE_ASYNC=3 EVENT_TYPE_COUNTER=4
PB.reserved_fields(::Type{var"TraceEvent.EventType".T}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[2])

struct Resource
    name::String
    resource_id::UInt64
    num_events::UInt32
end
Resource(;name = "", resource_id = zero(UInt64), num_events = zero(UInt32)) = Resource(name, resource_id, num_events)
PB.default_values(::Type{Resource}) = (;name = "", resource_id = zero(UInt64), num_events = zero(UInt32))
PB.field_numbers(::Type{Resource}) = (;name = 1, resource_id = 2, num_events = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Resource})
    name = ""
    resource_id = zero(UInt64)
    num_events = zero(UInt32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            resource_id = PB.decode(d, UInt64)
        elseif field_number == 3
            num_events = PB.decode(d, UInt32)
        else
            Base.skip(d, wire_type)
        end
    end
    return Resource(name, resource_id, num_events)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Resource)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    x.resource_id != zero(UInt64) && PB.encode(e, 2, x.resource_id)
    x.num_events != zero(UInt32) && PB.encode(e, 3, x.num_events)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Resource)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    x.resource_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.resource_id, 2))
    x.num_events != zero(UInt32) && (encoded_size += PB._encoded_size(x.num_events, 3))
    return encoded_size
end

struct TraceEvent
    var"#type"::var"TraceEvent.EventType".T
    device_id::UInt32
    resource_id::UInt64
    name_oneof::Union{Nothing,OneOf{<:Union{String,UInt64}}}
    group_id::Int64
    timestamp_ps::UInt64
    duration_ps::UInt64
    raw_data::Vector{UInt8}
    flow_id::UInt64
    flow_entry_type::var"TraceEvent.FlowEntryType".T
    flow_category::UInt32
    serial::UInt32
end
TraceEvent(;var"#type" = var"TraceEvent.EventType".EVENT_TYPE_UNSPECIFIED, device_id = zero(UInt32), resource_id = zero(UInt64), name_oneof = nothing, group_id = Int64(-1), timestamp_ps = zero(UInt64), duration_ps = zero(UInt64), raw_data = UInt8[], flow_id = zero(UInt64), flow_entry_type = var"TraceEvent.FlowEntryType".FLOW_NONE, flow_category = zero(UInt32), serial = zero(UInt32)) = TraceEvent(var"#type", device_id, resource_id, name_oneof, group_id, timestamp_ps, duration_ps, raw_data, flow_id, flow_entry_type, flow_category, serial)
PB.reserved_fields(::Type{TraceEvent}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[4])
PB.oneof_field_types(::Type{TraceEvent}) = (;
    name_oneof = (;name=String, name_ref=UInt64),
)
PB.default_values(::Type{TraceEvent}) = (;var"#type" = var"TraceEvent.EventType".EVENT_TYPE_UNSPECIFIED, device_id = zero(UInt32), resource_id = zero(UInt64), name = "", name_ref = zero(UInt64), group_id = Int64(-1), timestamp_ps = zero(UInt64), duration_ps = zero(UInt64), raw_data = UInt8[], flow_id = zero(UInt64), flow_entry_type = var"TraceEvent.FlowEntryType".FLOW_NONE, flow_category = zero(UInt32), serial = zero(UInt32))
PB.field_numbers(::Type{TraceEvent}) = (;var"#type" = 14, device_id = 1, resource_id = 2, name = 3, name_ref = 12, group_id = 5, timestamp_ps = 6, duration_ps = 7, raw_data = 8, flow_id = 9, flow_entry_type = 10, flow_category = 11, serial = 13)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TraceEvent})
    var"#type" = var"TraceEvent.EventType".EVENT_TYPE_UNSPECIFIED
    device_id = zero(UInt32)
    resource_id = zero(UInt64)
    name_oneof = nothing
    group_id = Int64(-1)
    timestamp_ps = zero(UInt64)
    duration_ps = zero(UInt64)
    raw_data = UInt8[]
    flow_id = zero(UInt64)
    flow_entry_type = var"TraceEvent.FlowEntryType".FLOW_NONE
    flow_category = zero(UInt32)
    serial = zero(UInt32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 14
            var"#type" = PB.decode(d, var"TraceEvent.EventType".T)
        elseif field_number == 1
            device_id = PB.decode(d, UInt32)
        elseif field_number == 2
            resource_id = PB.decode(d, UInt64)
        elseif field_number == 3
            name_oneof = OneOf(:name, PB.decode(d, String))
        elseif field_number == 12
            name_oneof = OneOf(:name_ref, PB.decode(d, UInt64, Val{:fixed}))
        elseif field_number == 5
            group_id = PB.decode(d, Int64)
        elseif field_number == 6
            timestamp_ps = PB.decode(d, UInt64)
        elseif field_number == 7
            duration_ps = PB.decode(d, UInt64)
        elseif field_number == 8
            raw_data = PB.decode(d, Vector{UInt8})
        elseif field_number == 9
            flow_id = PB.decode(d, UInt64)
        elseif field_number == 10
            flow_entry_type = PB.decode(d, var"TraceEvent.FlowEntryType".T)
        elseif field_number == 11
            flow_category = PB.decode(d, UInt32)
        elseif field_number == 13
            serial = PB.decode(d, UInt32)
        else
            Base.skip(d, wire_type)
        end
    end
    return TraceEvent(var"#type", device_id, resource_id, name_oneof, group_id, timestamp_ps, duration_ps, raw_data, flow_id, flow_entry_type, flow_category, serial)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TraceEvent)
    initpos = position(e.io)
    x.var"#type" != var"TraceEvent.EventType".EVENT_TYPE_UNSPECIFIED && PB.encode(e, 14, x.var"#type")
    x.device_id != zero(UInt32) && PB.encode(e, 1, x.device_id)
    x.resource_id != zero(UInt64) && PB.encode(e, 2, x.resource_id)
    if isnothing(x.name_oneof);
    elseif x.name_oneof.name === :name
        PB.encode(e, 3, x.name_oneof[]::String)
    elseif x.name_oneof.name === :name_ref
        PB.encode(e, 12, x.name_oneof[]::UInt64, Val{:fixed})
    end
    x.group_id != Int64(-1) && PB.encode(e, 5, x.group_id)
    x.timestamp_ps != zero(UInt64) && PB.encode(e, 6, x.timestamp_ps)
    x.duration_ps != zero(UInt64) && PB.encode(e, 7, x.duration_ps)
    !isempty(x.raw_data) && PB.encode(e, 8, x.raw_data)
    x.flow_id != zero(UInt64) && PB.encode(e, 9, x.flow_id)
    x.flow_entry_type != var"TraceEvent.FlowEntryType".FLOW_NONE && PB.encode(e, 10, x.flow_entry_type)
    x.flow_category != zero(UInt32) && PB.encode(e, 11, x.flow_category)
    x.serial != zero(UInt32) && PB.encode(e, 13, x.serial)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TraceEvent)
    encoded_size = 0
    x.var"#type" != var"TraceEvent.EventType".EVENT_TYPE_UNSPECIFIED && (encoded_size += PB._encoded_size(x.var"#type", 14))
    x.device_id != zero(UInt32) && (encoded_size += PB._encoded_size(x.device_id, 1))
    x.resource_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.resource_id, 2))
    if isnothing(x.name_oneof);
    elseif x.name_oneof.name === :name
        encoded_size += PB._encoded_size(x.name_oneof[]::String, 3)
    elseif x.name_oneof.name === :name_ref
        encoded_size += PB._encoded_size(x.name_oneof[]::UInt64, 12, Val{:fixed})
    end
    x.group_id != Int64(-1) && (encoded_size += PB._encoded_size(x.group_id, 5))
    x.timestamp_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.timestamp_ps, 6))
    x.duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.duration_ps, 7))
    !isempty(x.raw_data) && (encoded_size += PB._encoded_size(x.raw_data, 8))
    x.flow_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.flow_id, 9))
    x.flow_entry_type != var"TraceEvent.FlowEntryType".FLOW_NONE && (encoded_size += PB._encoded_size(x.flow_entry_type, 10))
    x.flow_category != zero(UInt32) && (encoded_size += PB._encoded_size(x.flow_category, 11))
    x.serial != zero(UInt32) && (encoded_size += PB._encoded_size(x.serial, 13))
    return encoded_size
end

struct Device
    name::String
    device_id::UInt32
    resources::Dict{UInt64,Resource}
end
Device(;name = "", device_id = zero(UInt32), resources = Dict{UInt64,Resource}()) = Device(name, device_id, resources)
PB.reserved_fields(::Type{Device}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[4])
PB.default_values(::Type{Device}) = (;name = "", device_id = zero(UInt32), resources = Dict{UInt64,Resource}())
PB.field_numbers(::Type{Device}) = (;name = 1, device_id = 2, resources = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Device})
    name = ""
    device_id = zero(UInt32)
    resources = Dict{UInt64,Resource}()
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
    tasks::Dict{UInt32,Task}
    min_timestamp_ps::UInt64
    max_timestamp_ps::UInt64
    num_events::UInt64
    name_table::Dict{UInt64,String}
end
Trace(;devices = Dict{UInt32,Device}(), tasks = Dict{UInt32,Task}(), min_timestamp_ps = zero(UInt64), max_timestamp_ps = zero(UInt64), num_events = zero(UInt64), name_table = Dict{UInt64,String}()) = Trace(devices, tasks, min_timestamp_ps, max_timestamp_ps, num_events, name_table)
PB.reserved_fields(::Type{Trace}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[2, 3])
PB.default_values(::Type{Trace}) = (;devices = Dict{UInt32,Device}(), tasks = Dict{UInt32,Task}(), min_timestamp_ps = zero(UInt64), max_timestamp_ps = zero(UInt64), num_events = zero(UInt64), name_table = Dict{UInt64,String}())
PB.field_numbers(::Type{Trace}) = (;devices = 1, tasks = 6, min_timestamp_ps = 4, max_timestamp_ps = 5, num_events = 7, name_table = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Trace})
    devices = Dict{UInt32,Device}()
    tasks = Dict{UInt32,Task}()
    min_timestamp_ps = zero(UInt64)
    max_timestamp_ps = zero(UInt64)
    num_events = zero(UInt64)
    name_table = Dict{UInt64,String}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, devices)
        elseif field_number == 6
            PB.decode!(d, tasks)
        elseif field_number == 4
            min_timestamp_ps = PB.decode(d, UInt64)
        elseif field_number == 5
            max_timestamp_ps = PB.decode(d, UInt64)
        elseif field_number == 7
            num_events = PB.decode(d, UInt64)
        elseif field_number == 8
            PB.decode!(d, name_table, Val{Tuple{:fixed,Nothing}})
        else
            Base.skip(d, wire_type)
        end
    end
    return Trace(devices, tasks, min_timestamp_ps, max_timestamp_ps, num_events, name_table)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Trace)
    initpos = position(e.io)
    !isempty(x.devices) && PB.encode(e, 1, x.devices)
    !isempty(x.tasks) && PB.encode(e, 6, x.tasks)
    x.min_timestamp_ps != zero(UInt64) && PB.encode(e, 4, x.min_timestamp_ps)
    x.max_timestamp_ps != zero(UInt64) && PB.encode(e, 5, x.max_timestamp_ps)
    x.num_events != zero(UInt64) && PB.encode(e, 7, x.num_events)
    !isempty(x.name_table) && PB.encode(e, 8, x.name_table, Val{Tuple{:fixed,Nothing}})
    return position(e.io) - initpos
end
function PB._encoded_size(x::Trace)
    encoded_size = 0
    !isempty(x.devices) && (encoded_size += PB._encoded_size(x.devices, 1))
    !isempty(x.tasks) && (encoded_size += PB._encoded_size(x.tasks, 6))
    x.min_timestamp_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.min_timestamp_ps, 4))
    x.max_timestamp_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.max_timestamp_ps, 5))
    x.num_events != zero(UInt64) && (encoded_size += PB._encoded_size(x.num_events, 7))
    !isempty(x.name_table) && (encoded_size += PB._encoded_size(x.name_table, 8, Val{Tuple{:fixed,Nothing}}))
    return encoded_size
end
