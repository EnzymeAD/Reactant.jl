import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export ShapedSliceProto, NullableShapedSliceProto


struct ShapedSliceProto
    slice::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
    shape::Union{Nothing,ShapeProto}
end
PB.default_values(::Type{ShapedSliceProto}) = (;slice = nothing, shape = nothing)
PB.field_numbers(::Type{ShapedSliceProto}) = (;slice = 1, shape = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ShapedSliceProto}, _endpos::Int=0, _group::Bool=false)
    slice = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    shape = Ref{Union{Nothing,ShapeProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, slice)
        elseif field_number == 2
            PB.decode!(d, shape)
        else
            Base.skip(d, wire_type)
        end
    end
    return ShapedSliceProto(slice[], shape[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ShapedSliceProto)
    initpos = position(e.io)
    !isnothing(x.slice) && PB.encode(e, 1, x.slice)
    !isnothing(x.shape) && PB.encode(e, 2, x.shape)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ShapedSliceProto)
    encoded_size = 0
    !isnothing(x.slice) && (encoded_size += PB._encoded_size(x.slice, 1))
    !isnothing(x.shape) && (encoded_size += PB._encoded_size(x.shape, 2))
    return encoded_size
end

struct NullableShapedSliceProto
    shaped_slice::Union{Nothing,ShapedSliceProto}
end
PB.default_values(::Type{NullableShapedSliceProto}) = (;shaped_slice = nothing)
PB.field_numbers(::Type{NullableShapedSliceProto}) = (;shaped_slice = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:NullableShapedSliceProto}, _endpos::Int=0, _group::Bool=false)
    shaped_slice = Ref{Union{Nothing,ShapedSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, shaped_slice)
        else
            Base.skip(d, wire_type)
        end
    end
    return NullableShapedSliceProto(shaped_slice[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::NullableShapedSliceProto)
    initpos = position(e.io)
    !isnothing(x.shaped_slice) && PB.encode(e, 1, x.shaped_slice)
    return position(e.io) - initpos
end
function PB._encoded_size(x::NullableShapedSliceProto)
    encoded_size = 0
    !isnothing(x.shaped_slice) && (encoded_size += PB._encoded_size(x.shaped_slice, 1))
    return encoded_size
end
