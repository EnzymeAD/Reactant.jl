import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export Timestamp


struct Timestamp
    seconds::Int64
    nanos::Int32
end
Timestamp(;seconds = zero(Int64), nanos = zero(Int32)) = Timestamp(seconds, nanos)
PB.default_values(::Type{Timestamp}) = (;seconds = zero(Int64), nanos = zero(Int32))
PB.field_numbers(::Type{Timestamp}) = (;seconds = 1, nanos = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Timestamp})
    seconds = zero(Int64)
    nanos = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            seconds = PB.decode(d, Int64)
        elseif field_number == 2
            nanos = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return Timestamp(seconds, nanos)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Timestamp)
    initpos = position(e.io)
    x.seconds != zero(Int64) && PB.encode(e, 1, x.seconds)
    x.nanos != zero(Int32) && PB.encode(e, 2, x.nanos)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Timestamp)
    encoded_size = 0
    x.seconds != zero(Int64) && (encoded_size += PB._encoded_size(x.seconds, 1))
    x.nanos != zero(Int32) && (encoded_size += PB._encoded_size(x.nanos, 2))
    return encoded_size
end
