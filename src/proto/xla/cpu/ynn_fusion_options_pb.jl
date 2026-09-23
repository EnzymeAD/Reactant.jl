import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export YnnFusionOptions


struct YnnFusionOptions
    use_threadpool::Bool
end
PB.default_values(::Type{YnnFusionOptions}) = (;use_threadpool = false)
PB.field_numbers(::Type{YnnFusionOptions}) = (;use_threadpool = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:YnnFusionOptions}, _endpos::Int=0, _group::Bool=false)
    use_threadpool = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            use_threadpool = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return YnnFusionOptions(use_threadpool)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::YnnFusionOptions)
    initpos = position(e.io)
    x.use_threadpool != false && PB.encode(e, 1, x.use_threadpool)
    return position(e.io) - initpos
end
function PB._encoded_size(x::YnnFusionOptions)
    encoded_size = 0
    x.use_threadpool != false && (encoded_size += PB._encoded_size(x.use_threadpool, 1))
    return encoded_size
end
