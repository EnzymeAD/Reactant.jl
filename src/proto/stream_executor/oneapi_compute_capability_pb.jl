import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export OneAPIComputeCapabilityProto


struct OneAPIComputeCapabilityProto
    architecture::String
    generation::UInt32
end
PB.reserved_fields(::Type{OneAPIComputeCapabilityProto}) = (names = ["variant"], numbers = Union{Int,UnitRange{Int}}[2])
PB.default_values(::Type{OneAPIComputeCapabilityProto}) = (;architecture = "", generation = zero(UInt32))
PB.field_numbers(::Type{OneAPIComputeCapabilityProto}) = (;architecture = 1, generation = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneAPIComputeCapabilityProto}, _endpos::Int=0, _group::Bool=false)
    architecture = ""
    generation = zero(UInt32)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            architecture = PB.decode(d, String)
        elseif field_number == 3
            generation = PB.decode(d, UInt32)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneAPIComputeCapabilityProto(architecture, generation)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneAPIComputeCapabilityProto)
    initpos = position(e.io)
    !isempty(x.architecture) && PB.encode(e, 1, x.architecture)
    x.generation != zero(UInt32) && PB.encode(e, 3, x.generation)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneAPIComputeCapabilityProto)
    encoded_size = 0
    !isempty(x.architecture) && (encoded_size += PB._encoded_size(x.architecture, 1))
    x.generation != zero(UInt32) && (encoded_size += PB._encoded_size(x.generation, 3))
    return encoded_size
end
