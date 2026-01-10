import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"#Any"


struct var"#Any"
    type_url::String
    value::Vector{UInt8}
end
var"#Any"(;type_url = "", value = UInt8[]) = var"#Any"(type_url, value)
PB.default_values(::Type{var"#Any"}) = (;type_url = "", value = UInt8[])
PB.field_numbers(::Type{var"#Any"}) = (;type_url = 1, value = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"#Any"})
    type_url = ""
    value = UInt8[]
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            type_url = PB.decode(d, String)
        elseif field_number == 2
            value = PB.decode(d, Vector{UInt8})
        else
            Base.skip(d, wire_type)
        end
    end
    return var"#Any"(type_url, value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"#Any")
    initpos = position(e.io)
    !isempty(x.type_url) && PB.encode(e, 1, x.type_url)
    !isempty(x.value) && PB.encode(e, 2, x.value)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"#Any")
    encoded_size = 0
    !isempty(x.type_url) && (encoded_size += PB._encoded_size(x.type_url, 1))
    !isempty(x.value) && (encoded_size += PB._encoded_size(x.value, 2))
    return encoded_size
end
