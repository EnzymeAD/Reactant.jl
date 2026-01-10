import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export XStatMetadata, XStat, XEventMetadata, XEvent, XLine, XPlane, XSpace


struct XStatMetadata
    id::Int64
    name::String
    description::String
end
XStatMetadata(;id = zero(Int64), name = "", description = "") = XStatMetadata(id, name, description)
PB.default_values(::Type{XStatMetadata}) = (;id = zero(Int64), name = "", description = "")
PB.field_numbers(::Type{XStatMetadata}) = (;id = 1, name = 2, description = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XStatMetadata})
    id = zero(Int64)
    name = ""
    description = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            id = PB.decode(d, Int64)
        elseif field_number == 2
            name = PB.decode(d, String)
        elseif field_number == 3
            description = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return XStatMetadata(id, name, description)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XStatMetadata)
    initpos = position(e.io)
    x.id != zero(Int64) && PB.encode(e, 1, x.id)
    !isempty(x.name) && PB.encode(e, 2, x.name)
    !isempty(x.description) && PB.encode(e, 3, x.description)
    return position(e.io) - initpos
end
function PB._encoded_size(x::XStatMetadata)
    encoded_size = 0
    x.id != zero(Int64) && (encoded_size += PB._encoded_size(x.id, 1))
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 2))
    !isempty(x.description) && (encoded_size += PB._encoded_size(x.description, 3))
    return encoded_size
end

struct XStat
    metadata_id::Int64
    value::Union{Nothing,OneOf{<:Union{Float64,UInt64,Int64,String,Vector{UInt8}}}}
end
XStat(;metadata_id = zero(Int64), value = nothing) = XStat(metadata_id, value)
PB.oneof_field_types(::Type{XStat}) = (;
    value = (;double_value=Float64, uint64_value=UInt64, int64_value=Int64, str_value=String, bytes_value=Vector{UInt8}, ref_value=UInt64),
)
PB.default_values(::Type{XStat}) = (;metadata_id = zero(Int64), double_value = zero(Float64), uint64_value = zero(UInt64), int64_value = zero(Int64), str_value = "", bytes_value = UInt8[], ref_value = zero(UInt64))
PB.field_numbers(::Type{XStat}) = (;metadata_id = 1, double_value = 2, uint64_value = 3, int64_value = 4, str_value = 5, bytes_value = 6, ref_value = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XStat})
    metadata_id = zero(Int64)
    value = nothing
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            metadata_id = PB.decode(d, Int64)
        elseif field_number == 2
            value = OneOf(:double_value, PB.decode(d, Float64))
        elseif field_number == 3
            value = OneOf(:uint64_value, PB.decode(d, UInt64))
        elseif field_number == 4
            value = OneOf(:int64_value, PB.decode(d, Int64))
        elseif field_number == 5
            value = OneOf(:str_value, PB.decode(d, String))
        elseif field_number == 6
            value = OneOf(:bytes_value, PB.decode(d, Vector{UInt8}))
        elseif field_number == 7
            value = OneOf(:ref_value, PB.decode(d, UInt64))
        else
            Base.skip(d, wire_type)
        end
    end
    return XStat(metadata_id, value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XStat)
    initpos = position(e.io)
    x.metadata_id != zero(Int64) && PB.encode(e, 1, x.metadata_id)
    if isnothing(x.value);
    elseif x.value.name === :double_value
        PB.encode(e, 2, x.value[]::Float64)
    elseif x.value.name === :uint64_value
        PB.encode(e, 3, x.value[]::UInt64)
    elseif x.value.name === :int64_value
        PB.encode(e, 4, x.value[]::Int64)
    elseif x.value.name === :str_value
        PB.encode(e, 5, x.value[]::String)
    elseif x.value.name === :bytes_value
        PB.encode(e, 6, x.value[]::Vector{UInt8})
    elseif x.value.name === :ref_value
        PB.encode(e, 7, x.value[]::UInt64)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::XStat)
    encoded_size = 0
    x.metadata_id != zero(Int64) && (encoded_size += PB._encoded_size(x.metadata_id, 1))
    if isnothing(x.value);
    elseif x.value.name === :double_value
        encoded_size += PB._encoded_size(x.value[]::Float64, 2)
    elseif x.value.name === :uint64_value
        encoded_size += PB._encoded_size(x.value[]::UInt64, 3)
    elseif x.value.name === :int64_value
        encoded_size += PB._encoded_size(x.value[]::Int64, 4)
    elseif x.value.name === :str_value
        encoded_size += PB._encoded_size(x.value[]::String, 5)
    elseif x.value.name === :bytes_value
        encoded_size += PB._encoded_size(x.value[]::Vector{UInt8}, 6)
    elseif x.value.name === :ref_value
        encoded_size += PB._encoded_size(x.value[]::UInt64, 7)
    end
    return encoded_size
end

struct XEventMetadata
    id::Int64
    name::String
    display_name::String
    metadata::Vector{UInt8}
    stats::Vector{XStat}
    child_id::Vector{Int64}
end
XEventMetadata(;id = zero(Int64), name = "", display_name = "", metadata = UInt8[], stats = Vector{XStat}(), child_id = Vector{Int64}()) = XEventMetadata(id, name, display_name, metadata, stats, child_id)
PB.default_values(::Type{XEventMetadata}) = (;id = zero(Int64), name = "", display_name = "", metadata = UInt8[], stats = Vector{XStat}(), child_id = Vector{Int64}())
PB.field_numbers(::Type{XEventMetadata}) = (;id = 1, name = 2, display_name = 4, metadata = 3, stats = 5, child_id = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XEventMetadata})
    id = zero(Int64)
    name = ""
    display_name = ""
    metadata = UInt8[]
    stats = PB.BufferedVector{XStat}()
    child_id = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            id = PB.decode(d, Int64)
        elseif field_number == 2
            name = PB.decode(d, String)
        elseif field_number == 4
            display_name = PB.decode(d, String)
        elseif field_number == 3
            metadata = PB.decode(d, Vector{UInt8})
        elseif field_number == 5
            PB.decode!(d, stats)
        elseif field_number == 6
            PB.decode!(d, wire_type, child_id)
        else
            Base.skip(d, wire_type)
        end
    end
    return XEventMetadata(id, name, display_name, metadata, stats[], child_id[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XEventMetadata)
    initpos = position(e.io)
    x.id != zero(Int64) && PB.encode(e, 1, x.id)
    !isempty(x.name) && PB.encode(e, 2, x.name)
    !isempty(x.display_name) && PB.encode(e, 4, x.display_name)
    !isempty(x.metadata) && PB.encode(e, 3, x.metadata)
    !isempty(x.stats) && PB.encode(e, 5, x.stats)
    !isempty(x.child_id) && PB.encode(e, 6, x.child_id)
    return position(e.io) - initpos
end
function PB._encoded_size(x::XEventMetadata)
    encoded_size = 0
    x.id != zero(Int64) && (encoded_size += PB._encoded_size(x.id, 1))
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 2))
    !isempty(x.display_name) && (encoded_size += PB._encoded_size(x.display_name, 4))
    !isempty(x.metadata) && (encoded_size += PB._encoded_size(x.metadata, 3))
    !isempty(x.stats) && (encoded_size += PB._encoded_size(x.stats, 5))
    !isempty(x.child_id) && (encoded_size += PB._encoded_size(x.child_id, 6))
    return encoded_size
end

struct XEvent
    metadata_id::Int64
    data::Union{Nothing,OneOf{Int64}}
    duration_ps::Int64
    stats::Vector{XStat}
end
XEvent(;metadata_id = zero(Int64), data = nothing, duration_ps = zero(Int64), stats = Vector{XStat}()) = XEvent(metadata_id, data, duration_ps, stats)
PB.oneof_field_types(::Type{XEvent}) = (;
    data = (;offset_ps=Int64, num_occurrences=Int64),
)
PB.default_values(::Type{XEvent}) = (;metadata_id = zero(Int64), offset_ps = zero(Int64), num_occurrences = zero(Int64), duration_ps = zero(Int64), stats = Vector{XStat}())
PB.field_numbers(::Type{XEvent}) = (;metadata_id = 1, offset_ps = 2, num_occurrences = 5, duration_ps = 3, stats = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XEvent})
    metadata_id = zero(Int64)
    data = nothing
    duration_ps = zero(Int64)
    stats = PB.BufferedVector{XStat}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            metadata_id = PB.decode(d, Int64)
        elseif field_number == 2
            data = OneOf(:offset_ps, PB.decode(d, Int64))
        elseif field_number == 5
            data = OneOf(:num_occurrences, PB.decode(d, Int64))
        elseif field_number == 3
            duration_ps = PB.decode(d, Int64)
        elseif field_number == 4
            PB.decode!(d, stats)
        else
            Base.skip(d, wire_type)
        end
    end
    return XEvent(metadata_id, data, duration_ps, stats[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XEvent)
    initpos = position(e.io)
    x.metadata_id != zero(Int64) && PB.encode(e, 1, x.metadata_id)
    if isnothing(x.data);
    elseif x.data.name === :offset_ps
        PB.encode(e, 2, x.data[]::Int64)
    elseif x.data.name === :num_occurrences
        PB.encode(e, 5, x.data[]::Int64)
    end
    x.duration_ps != zero(Int64) && PB.encode(e, 3, x.duration_ps)
    !isempty(x.stats) && PB.encode(e, 4, x.stats)
    return position(e.io) - initpos
end
function PB._encoded_size(x::XEvent)
    encoded_size = 0
    x.metadata_id != zero(Int64) && (encoded_size += PB._encoded_size(x.metadata_id, 1))
    if isnothing(x.data);
    elseif x.data.name === :offset_ps
        encoded_size += PB._encoded_size(x.data[]::Int64, 2)
    elseif x.data.name === :num_occurrences
        encoded_size += PB._encoded_size(x.data[]::Int64, 5)
    end
    x.duration_ps != zero(Int64) && (encoded_size += PB._encoded_size(x.duration_ps, 3))
    !isempty(x.stats) && (encoded_size += PB._encoded_size(x.stats, 4))
    return encoded_size
end

struct XLine
    id::Int64
    display_id::Int64
    name::String
    display_name::String
    timestamp_ns::Int64
    duration_ps::Int64
    events::Vector{XEvent}
end
XLine(;id = zero(Int64), display_id = zero(Int64), name = "", display_name = "", timestamp_ns = zero(Int64), duration_ps = zero(Int64), events = Vector{XEvent}()) = XLine(id, display_id, name, display_name, timestamp_ns, duration_ps, events)
PB.reserved_fields(::Type{XLine}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[5, 6, 7, 8])
PB.default_values(::Type{XLine}) = (;id = zero(Int64), display_id = zero(Int64), name = "", display_name = "", timestamp_ns = zero(Int64), duration_ps = zero(Int64), events = Vector{XEvent}())
PB.field_numbers(::Type{XLine}) = (;id = 1, display_id = 10, name = 2, display_name = 11, timestamp_ns = 3, duration_ps = 9, events = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XLine})
    id = zero(Int64)
    display_id = zero(Int64)
    name = ""
    display_name = ""
    timestamp_ns = zero(Int64)
    duration_ps = zero(Int64)
    events = PB.BufferedVector{XEvent}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            id = PB.decode(d, Int64)
        elseif field_number == 10
            display_id = PB.decode(d, Int64)
        elseif field_number == 2
            name = PB.decode(d, String)
        elseif field_number == 11
            display_name = PB.decode(d, String)
        elseif field_number == 3
            timestamp_ns = PB.decode(d, Int64)
        elseif field_number == 9
            duration_ps = PB.decode(d, Int64)
        elseif field_number == 4
            PB.decode!(d, events)
        else
            Base.skip(d, wire_type)
        end
    end
    return XLine(id, display_id, name, display_name, timestamp_ns, duration_ps, events[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XLine)
    initpos = position(e.io)
    x.id != zero(Int64) && PB.encode(e, 1, x.id)
    x.display_id != zero(Int64) && PB.encode(e, 10, x.display_id)
    !isempty(x.name) && PB.encode(e, 2, x.name)
    !isempty(x.display_name) && PB.encode(e, 11, x.display_name)
    x.timestamp_ns != zero(Int64) && PB.encode(e, 3, x.timestamp_ns)
    x.duration_ps != zero(Int64) && PB.encode(e, 9, x.duration_ps)
    !isempty(x.events) && PB.encode(e, 4, x.events)
    return position(e.io) - initpos
end
function PB._encoded_size(x::XLine)
    encoded_size = 0
    x.id != zero(Int64) && (encoded_size += PB._encoded_size(x.id, 1))
    x.display_id != zero(Int64) && (encoded_size += PB._encoded_size(x.display_id, 10))
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 2))
    !isempty(x.display_name) && (encoded_size += PB._encoded_size(x.display_name, 11))
    x.timestamp_ns != zero(Int64) && (encoded_size += PB._encoded_size(x.timestamp_ns, 3))
    x.duration_ps != zero(Int64) && (encoded_size += PB._encoded_size(x.duration_ps, 9))
    !isempty(x.events) && (encoded_size += PB._encoded_size(x.events, 4))
    return encoded_size
end

struct XPlane
    id::Int64
    name::String
    lines::Vector{XLine}
    event_metadata::Dict{Int64,XEventMetadata}
    stat_metadata::Dict{Int64,XStatMetadata}
    stats::Vector{XStat}
end
XPlane(;id = zero(Int64), name = "", lines = Vector{XLine}(), event_metadata = Dict{Int64,XEventMetadata}(), stat_metadata = Dict{Int64,XStatMetadata}(), stats = Vector{XStat}()) = XPlane(id, name, lines, event_metadata, stat_metadata, stats)
PB.default_values(::Type{XPlane}) = (;id = zero(Int64), name = "", lines = Vector{XLine}(), event_metadata = Dict{Int64,XEventMetadata}(), stat_metadata = Dict{Int64,XStatMetadata}(), stats = Vector{XStat}())
PB.field_numbers(::Type{XPlane}) = (;id = 1, name = 2, lines = 3, event_metadata = 4, stat_metadata = 5, stats = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XPlane})
    id = zero(Int64)
    name = ""
    lines = PB.BufferedVector{XLine}()
    event_metadata = Dict{Int64,XEventMetadata}()
    stat_metadata = Dict{Int64,XStatMetadata}()
    stats = PB.BufferedVector{XStat}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            id = PB.decode(d, Int64)
        elseif field_number == 2
            name = PB.decode(d, String)
        elseif field_number == 3
            PB.decode!(d, lines)
        elseif field_number == 4
            PB.decode!(d, event_metadata)
        elseif field_number == 5
            PB.decode!(d, stat_metadata)
        elseif field_number == 6
            PB.decode!(d, stats)
        else
            Base.skip(d, wire_type)
        end
    end
    return XPlane(id, name, lines[], event_metadata, stat_metadata, stats[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XPlane)
    initpos = position(e.io)
    x.id != zero(Int64) && PB.encode(e, 1, x.id)
    !isempty(x.name) && PB.encode(e, 2, x.name)
    !isempty(x.lines) && PB.encode(e, 3, x.lines)
    !isempty(x.event_metadata) && PB.encode(e, 4, x.event_metadata)
    !isempty(x.stat_metadata) && PB.encode(e, 5, x.stat_metadata)
    !isempty(x.stats) && PB.encode(e, 6, x.stats)
    return position(e.io) - initpos
end
function PB._encoded_size(x::XPlane)
    encoded_size = 0
    x.id != zero(Int64) && (encoded_size += PB._encoded_size(x.id, 1))
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 2))
    !isempty(x.lines) && (encoded_size += PB._encoded_size(x.lines, 3))
    !isempty(x.event_metadata) && (encoded_size += PB._encoded_size(x.event_metadata, 4))
    !isempty(x.stat_metadata) && (encoded_size += PB._encoded_size(x.stat_metadata, 5))
    !isempty(x.stats) && (encoded_size += PB._encoded_size(x.stats, 6))
    return encoded_size
end

struct XSpace
    planes::Vector{XPlane}
    errors::Vector{String}
    warnings::Vector{String}
    hostnames::Vector{String}
end
XSpace(;planes = Vector{XPlane}(), errors = Vector{String}(), warnings = Vector{String}(), hostnames = Vector{String}()) = XSpace(planes, errors, warnings, hostnames)
PB.default_values(::Type{XSpace}) = (;planes = Vector{XPlane}(), errors = Vector{String}(), warnings = Vector{String}(), hostnames = Vector{String}())
PB.field_numbers(::Type{XSpace}) = (;planes = 1, errors = 2, warnings = 3, hostnames = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XSpace})
    planes = PB.BufferedVector{XPlane}()
    errors = PB.BufferedVector{String}()
    warnings = PB.BufferedVector{String}()
    hostnames = PB.BufferedVector{String}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, planes)
        elseif field_number == 2
            PB.decode!(d, errors)
        elseif field_number == 3
            PB.decode!(d, warnings)
        elseif field_number == 4
            PB.decode!(d, hostnames)
        else
            Base.skip(d, wire_type)
        end
    end
    return XSpace(planes[], errors[], warnings[], hostnames[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XSpace)
    initpos = position(e.io)
    !isempty(x.planes) && PB.encode(e, 1, x.planes)
    !isempty(x.errors) && PB.encode(e, 2, x.errors)
    !isempty(x.warnings) && PB.encode(e, 3, x.warnings)
    !isempty(x.hostnames) && PB.encode(e, 4, x.hostnames)
    return position(e.io) - initpos
end
function PB._encoded_size(x::XSpace)
    encoded_size = 0
    !isempty(x.planes) && (encoded_size += PB._encoded_size(x.planes, 1))
    !isempty(x.errors) && (encoded_size += PB._encoded_size(x.errors, 2))
    !isempty(x.warnings) && (encoded_size += PB._encoded_size(x.warnings, 3))
    !isempty(x.hostnames) && (encoded_size += PB._encoded_size(x.hostnames, 4))
    return encoded_size
end
