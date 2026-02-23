import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export OneAPIComputeCapabilityProto


struct OneAPIComputeCapabilityProto
    architecture::String
    variant::String
end
PB.default_values(::Type{OneAPIComputeCapabilityProto}) = (;architecture = "", variant = "")
PB.field_numbers(::Type{OneAPIComputeCapabilityProto}) = (;architecture = 1, variant = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneAPIComputeCapabilityProto})
    architecture = ""
    variant = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            architecture = PB.decode(d, String)
        elseif field_number == 2
            variant = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneAPIComputeCapabilityProto(architecture, variant)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneAPIComputeCapabilityProto)
    initpos = position(e.io)
    !isempty(x.architecture) && PB.encode(e, 1, x.architecture)
    !isempty(x.variant) && PB.encode(e, 2, x.variant)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneAPIComputeCapabilityProto)
    encoded_size = 0
    !isempty(x.architecture) && (encoded_size += PB._encoded_size(x.architecture, 1))
    !isempty(x.variant) && (encoded_size += PB._encoded_size(x.variant, 2))
    return encoded_size
end
