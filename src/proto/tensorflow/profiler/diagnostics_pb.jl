import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export Diagnostics


struct Diagnostics
    info::Vector{String}
    warnings::Vector{String}
    errors::Vector{String}
end
Diagnostics(;info = Vector{String}(), warnings = Vector{String}(), errors = Vector{String}()) = Diagnostics(info, warnings, errors)
PB.default_values(::Type{Diagnostics}) = (;info = Vector{String}(), warnings = Vector{String}(), errors = Vector{String}())
PB.field_numbers(::Type{Diagnostics}) = (;info = 1, warnings = 2, errors = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Diagnostics})
    info = PB.BufferedVector{String}()
    warnings = PB.BufferedVector{String}()
    errors = PB.BufferedVector{String}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, info)
        elseif field_number == 2
            PB.decode!(d, warnings)
        elseif field_number == 3
            PB.decode!(d, errors)
        else
            Base.skip(d, wire_type)
        end
    end
    return Diagnostics(info[], warnings[], errors[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Diagnostics)
    initpos = position(e.io)
    !isempty(x.info) && PB.encode(e, 1, x.info)
    !isempty(x.warnings) && PB.encode(e, 2, x.warnings)
    !isempty(x.errors) && PB.encode(e, 3, x.errors)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Diagnostics)
    encoded_size = 0
    !isempty(x.info) && (encoded_size += PB._encoded_size(x.info, 1))
    !isempty(x.warnings) && (encoded_size += PB._encoded_size(x.warnings, 2))
    !isempty(x.errors) && (encoded_size += PB._encoded_size(x.errors, 3))
    return encoded_size
end
