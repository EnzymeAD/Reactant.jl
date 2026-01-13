import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export Line, Label, ValueType, Mapping, Function, Location, Sample, Profile


struct Line
    function_id::UInt64
    line::Int64
end
Line(;function_id = zero(UInt64), line = zero(Int64)) = Line(function_id, line)
PB.default_values(::Type{Line}) = (;function_id = zero(UInt64), line = zero(Int64))
PB.field_numbers(::Type{Line}) = (;function_id = 1, line = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Line})
    function_id = zero(UInt64)
    line = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            function_id = PB.decode(d, UInt64)
        elseif field_number == 2
            line = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return Line(function_id, line)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Line)
    initpos = position(e.io)
    x.function_id != zero(UInt64) && PB.encode(e, 1, x.function_id)
    x.line != zero(Int64) && PB.encode(e, 2, x.line)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Line)
    encoded_size = 0
    x.function_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.function_id, 1))
    x.line != zero(Int64) && (encoded_size += PB._encoded_size(x.line, 2))
    return encoded_size
end

struct Label
    key::Int64
    str::Int64
    num::Int64
end
Label(;key = zero(Int64), str = zero(Int64), num = zero(Int64)) = Label(key, str, num)
PB.default_values(::Type{Label}) = (;key = zero(Int64), str = zero(Int64), num = zero(Int64))
PB.field_numbers(::Type{Label}) = (;key = 1, str = 2, num = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Label})
    key = zero(Int64)
    str = zero(Int64)
    num = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            key = PB.decode(d, Int64)
        elseif field_number == 2
            str = PB.decode(d, Int64)
        elseif field_number == 3
            num = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return Label(key, str, num)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Label)
    initpos = position(e.io)
    x.key != zero(Int64) && PB.encode(e, 1, x.key)
    x.str != zero(Int64) && PB.encode(e, 2, x.str)
    x.num != zero(Int64) && PB.encode(e, 3, x.num)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Label)
    encoded_size = 0
    x.key != zero(Int64) && (encoded_size += PB._encoded_size(x.key, 1))
    x.str != zero(Int64) && (encoded_size += PB._encoded_size(x.str, 2))
    x.num != zero(Int64) && (encoded_size += PB._encoded_size(x.num, 3))
    return encoded_size
end

struct ValueType
    var"#type"::Int64
    unit::Int64
end
ValueType(;var"#type" = zero(Int64), unit = zero(Int64)) = ValueType(var"#type", unit)
PB.default_values(::Type{ValueType}) = (;var"#type" = zero(Int64), unit = zero(Int64))
PB.field_numbers(::Type{ValueType}) = (;var"#type" = 1, unit = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ValueType})
    var"#type" = zero(Int64)
    unit = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            var"#type" = PB.decode(d, Int64)
        elseif field_number == 2
            unit = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return ValueType(var"#type", unit)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ValueType)
    initpos = position(e.io)
    x.var"#type" != zero(Int64) && PB.encode(e, 1, x.var"#type")
    x.unit != zero(Int64) && PB.encode(e, 2, x.unit)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ValueType)
    encoded_size = 0
    x.var"#type" != zero(Int64) && (encoded_size += PB._encoded_size(x.var"#type", 1))
    x.unit != zero(Int64) && (encoded_size += PB._encoded_size(x.unit, 2))
    return encoded_size
end

struct Mapping
    id::UInt64
    memory_start::UInt64
    memory_limit::UInt64
    file_offset::UInt64
    filename::Int64
    build_id::Int64
    has_functions::Bool
    has_filenames::Bool
    has_line_numbers::Bool
    has_inline_frames::Bool
end
Mapping(;id = zero(UInt64), memory_start = zero(UInt64), memory_limit = zero(UInt64), file_offset = zero(UInt64), filename = zero(Int64), build_id = zero(Int64), has_functions = false, has_filenames = false, has_line_numbers = false, has_inline_frames = false) = Mapping(id, memory_start, memory_limit, file_offset, filename, build_id, has_functions, has_filenames, has_line_numbers, has_inline_frames)
PB.default_values(::Type{Mapping}) = (;id = zero(UInt64), memory_start = zero(UInt64), memory_limit = zero(UInt64), file_offset = zero(UInt64), filename = zero(Int64), build_id = zero(Int64), has_functions = false, has_filenames = false, has_line_numbers = false, has_inline_frames = false)
PB.field_numbers(::Type{Mapping}) = (;id = 1, memory_start = 2, memory_limit = 3, file_offset = 4, filename = 5, build_id = 6, has_functions = 7, has_filenames = 8, has_line_numbers = 9, has_inline_frames = 10)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Mapping})
    id = zero(UInt64)
    memory_start = zero(UInt64)
    memory_limit = zero(UInt64)
    file_offset = zero(UInt64)
    filename = zero(Int64)
    build_id = zero(Int64)
    has_functions = false
    has_filenames = false
    has_line_numbers = false
    has_inline_frames = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            id = PB.decode(d, UInt64)
        elseif field_number == 2
            memory_start = PB.decode(d, UInt64)
        elseif field_number == 3
            memory_limit = PB.decode(d, UInt64)
        elseif field_number == 4
            file_offset = PB.decode(d, UInt64)
        elseif field_number == 5
            filename = PB.decode(d, Int64)
        elseif field_number == 6
            build_id = PB.decode(d, Int64)
        elseif field_number == 7
            has_functions = PB.decode(d, Bool)
        elseif field_number == 8
            has_filenames = PB.decode(d, Bool)
        elseif field_number == 9
            has_line_numbers = PB.decode(d, Bool)
        elseif field_number == 10
            has_inline_frames = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return Mapping(id, memory_start, memory_limit, file_offset, filename, build_id, has_functions, has_filenames, has_line_numbers, has_inline_frames)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Mapping)
    initpos = position(e.io)
    x.id != zero(UInt64) && PB.encode(e, 1, x.id)
    x.memory_start != zero(UInt64) && PB.encode(e, 2, x.memory_start)
    x.memory_limit != zero(UInt64) && PB.encode(e, 3, x.memory_limit)
    x.file_offset != zero(UInt64) && PB.encode(e, 4, x.file_offset)
    x.filename != zero(Int64) && PB.encode(e, 5, x.filename)
    x.build_id != zero(Int64) && PB.encode(e, 6, x.build_id)
    x.has_functions != false && PB.encode(e, 7, x.has_functions)
    x.has_filenames != false && PB.encode(e, 8, x.has_filenames)
    x.has_line_numbers != false && PB.encode(e, 9, x.has_line_numbers)
    x.has_inline_frames != false && PB.encode(e, 10, x.has_inline_frames)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Mapping)
    encoded_size = 0
    x.id != zero(UInt64) && (encoded_size += PB._encoded_size(x.id, 1))
    x.memory_start != zero(UInt64) && (encoded_size += PB._encoded_size(x.memory_start, 2))
    x.memory_limit != zero(UInt64) && (encoded_size += PB._encoded_size(x.memory_limit, 3))
    x.file_offset != zero(UInt64) && (encoded_size += PB._encoded_size(x.file_offset, 4))
    x.filename != zero(Int64) && (encoded_size += PB._encoded_size(x.filename, 5))
    x.build_id != zero(Int64) && (encoded_size += PB._encoded_size(x.build_id, 6))
    x.has_functions != false && (encoded_size += PB._encoded_size(x.has_functions, 7))
    x.has_filenames != false && (encoded_size += PB._encoded_size(x.has_filenames, 8))
    x.has_line_numbers != false && (encoded_size += PB._encoded_size(x.has_line_numbers, 9))
    x.has_inline_frames != false && (encoded_size += PB._encoded_size(x.has_inline_frames, 10))
    return encoded_size
end

struct Function
    id::UInt64
    name::Int64
    system_name::Int64
    filename::Int64
    start_line::Int64
end
Function(;id = zero(UInt64), name = zero(Int64), system_name = zero(Int64), filename = zero(Int64), start_line = zero(Int64)) = Function(id, name, system_name, filename, start_line)
PB.default_values(::Type{Function}) = (;id = zero(UInt64), name = zero(Int64), system_name = zero(Int64), filename = zero(Int64), start_line = zero(Int64))
PB.field_numbers(::Type{Function}) = (;id = 1, name = 2, system_name = 3, filename = 4, start_line = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Function})
    id = zero(UInt64)
    name = zero(Int64)
    system_name = zero(Int64)
    filename = zero(Int64)
    start_line = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            id = PB.decode(d, UInt64)
        elseif field_number == 2
            name = PB.decode(d, Int64)
        elseif field_number == 3
            system_name = PB.decode(d, Int64)
        elseif field_number == 4
            filename = PB.decode(d, Int64)
        elseif field_number == 5
            start_line = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return Function(id, name, system_name, filename, start_line)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Function)
    initpos = position(e.io)
    x.id != zero(UInt64) && PB.encode(e, 1, x.id)
    x.name != zero(Int64) && PB.encode(e, 2, x.name)
    x.system_name != zero(Int64) && PB.encode(e, 3, x.system_name)
    x.filename != zero(Int64) && PB.encode(e, 4, x.filename)
    x.start_line != zero(Int64) && PB.encode(e, 5, x.start_line)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Function)
    encoded_size = 0
    x.id != zero(UInt64) && (encoded_size += PB._encoded_size(x.id, 1))
    x.name != zero(Int64) && (encoded_size += PB._encoded_size(x.name, 2))
    x.system_name != zero(Int64) && (encoded_size += PB._encoded_size(x.system_name, 3))
    x.filename != zero(Int64) && (encoded_size += PB._encoded_size(x.filename, 4))
    x.start_line != zero(Int64) && (encoded_size += PB._encoded_size(x.start_line, 5))
    return encoded_size
end

struct Location
    id::UInt64
    mapping_id::UInt64
    address::UInt64
    line::Vector{Line}
end
Location(;id = zero(UInt64), mapping_id = zero(UInt64), address = zero(UInt64), line = Vector{Line}()) = Location(id, mapping_id, address, line)
PB.default_values(::Type{Location}) = (;id = zero(UInt64), mapping_id = zero(UInt64), address = zero(UInt64), line = Vector{Line}())
PB.field_numbers(::Type{Location}) = (;id = 1, mapping_id = 2, address = 3, line = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Location})
    id = zero(UInt64)
    mapping_id = zero(UInt64)
    address = zero(UInt64)
    line = PB.BufferedVector{Line}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            id = PB.decode(d, UInt64)
        elseif field_number == 2
            mapping_id = PB.decode(d, UInt64)
        elseif field_number == 3
            address = PB.decode(d, UInt64)
        elseif field_number == 4
            PB.decode!(d, line)
        else
            Base.skip(d, wire_type)
        end
    end
    return Location(id, mapping_id, address, line[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Location)
    initpos = position(e.io)
    x.id != zero(UInt64) && PB.encode(e, 1, x.id)
    x.mapping_id != zero(UInt64) && PB.encode(e, 2, x.mapping_id)
    x.address != zero(UInt64) && PB.encode(e, 3, x.address)
    !isempty(x.line) && PB.encode(e, 4, x.line)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Location)
    encoded_size = 0
    x.id != zero(UInt64) && (encoded_size += PB._encoded_size(x.id, 1))
    x.mapping_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.mapping_id, 2))
    x.address != zero(UInt64) && (encoded_size += PB._encoded_size(x.address, 3))
    !isempty(x.line) && (encoded_size += PB._encoded_size(x.line, 4))
    return encoded_size
end

struct Sample
    location_id::Vector{UInt64}
    value::Vector{Int64}
    label::Vector{Label}
end
Sample(;location_id = Vector{UInt64}(), value = Vector{Int64}(), label = Vector{Label}()) = Sample(location_id, value, label)
PB.default_values(::Type{Sample}) = (;location_id = Vector{UInt64}(), value = Vector{Int64}(), label = Vector{Label}())
PB.field_numbers(::Type{Sample}) = (;location_id = 1, value = 2, label = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Sample})
    location_id = PB.BufferedVector{UInt64}()
    value = PB.BufferedVector{Int64}()
    label = PB.BufferedVector{Label}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, location_id)
        elseif field_number == 2
            PB.decode!(d, wire_type, value)
        elseif field_number == 3
            PB.decode!(d, label)
        else
            Base.skip(d, wire_type)
        end
    end
    return Sample(location_id[], value[], label[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Sample)
    initpos = position(e.io)
    !isempty(x.location_id) && PB.encode(e, 1, x.location_id)
    !isempty(x.value) && PB.encode(e, 2, x.value)
    !isempty(x.label) && PB.encode(e, 3, x.label)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Sample)
    encoded_size = 0
    !isempty(x.location_id) && (encoded_size += PB._encoded_size(x.location_id, 1))
    !isempty(x.value) && (encoded_size += PB._encoded_size(x.value, 2))
    !isempty(x.label) && (encoded_size += PB._encoded_size(x.label, 3))
    return encoded_size
end

struct Profile
    sample_type::Vector{ValueType}
    sample::Vector{Sample}
    mapping::Vector{Mapping}
    location::Vector{Location}
    var"#function"::Vector{Function}
    string_table::Vector{String}
    drop_frames::Int64
    keep_frames::Int64
    time_nanos::Int64
    duration_nanos::Int64
    period_type::Union{Nothing,ValueType}
    period::Int64
    comment::Vector{Int64}
    default_sample_type::Int64
end
Profile(;sample_type = Vector{ValueType}(), sample = Vector{Sample}(), mapping = Vector{Mapping}(), location = Vector{Location}(), var"#function" = Vector{Function}(), string_table = Vector{String}(), drop_frames = zero(Int64), keep_frames = zero(Int64), time_nanos = zero(Int64), duration_nanos = zero(Int64), period_type = nothing, period = zero(Int64), comment = Vector{Int64}(), default_sample_type = zero(Int64)) = Profile(sample_type, sample, mapping, location, var"#function", string_table, drop_frames, keep_frames, time_nanos, duration_nanos, period_type, period, comment, default_sample_type)
PB.default_values(::Type{Profile}) = (;sample_type = Vector{ValueType}(), sample = Vector{Sample}(), mapping = Vector{Mapping}(), location = Vector{Location}(), var"#function" = Vector{Function}(), string_table = Vector{String}(), drop_frames = zero(Int64), keep_frames = zero(Int64), time_nanos = zero(Int64), duration_nanos = zero(Int64), period_type = nothing, period = zero(Int64), comment = Vector{Int64}(), default_sample_type = zero(Int64))
PB.field_numbers(::Type{Profile}) = (;sample_type = 1, sample = 2, mapping = 3, location = 4, var"#function" = 5, string_table = 6, drop_frames = 7, keep_frames = 8, time_nanos = 9, duration_nanos = 10, period_type = 11, period = 12, comment = 13, default_sample_type = 14)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Profile})
    sample_type = PB.BufferedVector{ValueType}()
    sample = PB.BufferedVector{Sample}()
    mapping = PB.BufferedVector{Mapping}()
    location = PB.BufferedVector{Location}()
    var"#function" = PB.BufferedVector{Function}()
    string_table = PB.BufferedVector{String}()
    drop_frames = zero(Int64)
    keep_frames = zero(Int64)
    time_nanos = zero(Int64)
    duration_nanos = zero(Int64)
    period_type = Ref{Union{Nothing,ValueType}}(nothing)
    period = zero(Int64)
    comment = PB.BufferedVector{Int64}()
    default_sample_type = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, sample_type)
        elseif field_number == 2
            PB.decode!(d, sample)
        elseif field_number == 3
            PB.decode!(d, mapping)
        elseif field_number == 4
            PB.decode!(d, location)
        elseif field_number == 5
            PB.decode!(d, var"#function")
        elseif field_number == 6
            PB.decode!(d, string_table)
        elseif field_number == 7
            drop_frames = PB.decode(d, Int64)
        elseif field_number == 8
            keep_frames = PB.decode(d, Int64)
        elseif field_number == 9
            time_nanos = PB.decode(d, Int64)
        elseif field_number == 10
            duration_nanos = PB.decode(d, Int64)
        elseif field_number == 11
            PB.decode!(d, period_type)
        elseif field_number == 12
            period = PB.decode(d, Int64)
        elseif field_number == 13
            PB.decode!(d, wire_type, comment)
        elseif field_number == 14
            default_sample_type = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return Profile(sample_type[], sample[], mapping[], location[], var"#function"[], string_table[], drop_frames, keep_frames, time_nanos, duration_nanos, period_type[], period, comment[], default_sample_type)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Profile)
    initpos = position(e.io)
    !isempty(x.sample_type) && PB.encode(e, 1, x.sample_type)
    !isempty(x.sample) && PB.encode(e, 2, x.sample)
    !isempty(x.mapping) && PB.encode(e, 3, x.mapping)
    !isempty(x.location) && PB.encode(e, 4, x.location)
    !isempty(x.var"#function") && PB.encode(e, 5, x.var"#function")
    !isempty(x.string_table) && PB.encode(e, 6, x.string_table)
    x.drop_frames != zero(Int64) && PB.encode(e, 7, x.drop_frames)
    x.keep_frames != zero(Int64) && PB.encode(e, 8, x.keep_frames)
    x.time_nanos != zero(Int64) && PB.encode(e, 9, x.time_nanos)
    x.duration_nanos != zero(Int64) && PB.encode(e, 10, x.duration_nanos)
    !isnothing(x.period_type) && PB.encode(e, 11, x.period_type)
    x.period != zero(Int64) && PB.encode(e, 12, x.period)
    !isempty(x.comment) && PB.encode(e, 13, x.comment)
    x.default_sample_type != zero(Int64) && PB.encode(e, 14, x.default_sample_type)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Profile)
    encoded_size = 0
    !isempty(x.sample_type) && (encoded_size += PB._encoded_size(x.sample_type, 1))
    !isempty(x.sample) && (encoded_size += PB._encoded_size(x.sample, 2))
    !isempty(x.mapping) && (encoded_size += PB._encoded_size(x.mapping, 3))
    !isempty(x.location) && (encoded_size += PB._encoded_size(x.location, 4))
    !isempty(x.var"#function") && (encoded_size += PB._encoded_size(x.var"#function", 5))
    !isempty(x.string_table) && (encoded_size += PB._encoded_size(x.string_table, 6))
    x.drop_frames != zero(Int64) && (encoded_size += PB._encoded_size(x.drop_frames, 7))
    x.keep_frames != zero(Int64) && (encoded_size += PB._encoded_size(x.keep_frames, 8))
    x.time_nanos != zero(Int64) && (encoded_size += PB._encoded_size(x.time_nanos, 9))
    x.duration_nanos != zero(Int64) && (encoded_size += PB._encoded_size(x.duration_nanos, 10))
    !isnothing(x.period_type) && (encoded_size += PB._encoded_size(x.period_type, 11))
    x.period != zero(Int64) && (encoded_size += PB._encoded_size(x.period, 12))
    !isempty(x.comment) && (encoded_size += PB._encoded_size(x.comment, 13))
    x.default_sample_type != zero(Int64) && (encoded_size += PB._encoded_size(x.default_sample_type, 14))
    return encoded_size
end
