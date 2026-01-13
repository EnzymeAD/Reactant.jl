import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export SourceInfo


struct SourceInfo
    file_name::String
    line_number::Int32
    stack_frame::String
end
SourceInfo(;file_name = "", line_number = zero(Int32), stack_frame = "") = SourceInfo(file_name, line_number, stack_frame)
PB.default_values(::Type{SourceInfo}) = (;file_name = "", line_number = zero(Int32), stack_frame = "")
PB.field_numbers(::Type{SourceInfo}) = (;file_name = 1, line_number = 2, stack_frame = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SourceInfo})
    file_name = ""
    line_number = zero(Int32)
    stack_frame = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            file_name = PB.decode(d, String)
        elseif field_number == 2
            line_number = PB.decode(d, Int32)
        elseif field_number == 3
            stack_frame = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return SourceInfo(file_name, line_number, stack_frame)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SourceInfo)
    initpos = position(e.io)
    !isempty(x.file_name) && PB.encode(e, 1, x.file_name)
    x.line_number != zero(Int32) && PB.encode(e, 2, x.line_number)
    !isempty(x.stack_frame) && PB.encode(e, 3, x.stack_frame)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SourceInfo)
    encoded_size = 0
    !isempty(x.file_name) && (encoded_size += PB._encoded_size(x.file_name, 1))
    x.line_number != zero(Int32) && (encoded_size += PB._encoded_size(x.line_number, 2))
    !isempty(x.stack_frame) && (encoded_size += PB._encoded_size(x.stack_frame, 3))
    return encoded_size
end
