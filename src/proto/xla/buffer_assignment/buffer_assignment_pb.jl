import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export BufferIsolationConfig
export var"AssignmentAlgorithmForComputationsWithoutOrderingProto.Value"
export AssignmentAlgorithmForComputationsWithoutOrderingProto, BufferAllocationSliceProto
export var"BufferAssignmentAlgorithmProto.Value", BufferAssignmentAlgorithmProto


struct BufferIsolationConfig
    base_offset_bytes::Int64
    isolation_fuel::Int64
    isolation_padding_bytes::Int64
    isolation_order_salt::UInt64
    isolation_colors::Vector{Int32}
end
PB.default_values(::Type{BufferIsolationConfig}) = (;base_offset_bytes = zero(Int64), isolation_fuel = zero(Int64), isolation_padding_bytes = zero(Int64), isolation_order_salt = zero(UInt64), isolation_colors = Vector{Int32}())
PB.field_numbers(::Type{BufferIsolationConfig}) = (;base_offset_bytes = 1, isolation_fuel = 2, isolation_padding_bytes = 3, isolation_order_salt = 4, isolation_colors = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:BufferIsolationConfig}, _endpos::Int=0, _group::Bool=false)
    base_offset_bytes = zero(Int64)
    isolation_fuel = zero(Int64)
    isolation_padding_bytes = zero(Int64)
    isolation_order_salt = zero(UInt64)
    isolation_colors = PB.BufferedVector{Int32}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            base_offset_bytes = PB.decode(d, Int64)
        elseif field_number == 2
            isolation_fuel = PB.decode(d, Int64)
        elseif field_number == 3
            isolation_padding_bytes = PB.decode(d, Int64)
        elseif field_number == 4
            isolation_order_salt = PB.decode(d, UInt64)
        elseif field_number == 5
            PB.decode!(d, wire_type, isolation_colors)
        else
            Base.skip(d, wire_type)
        end
    end
    return BufferIsolationConfig(base_offset_bytes, isolation_fuel, isolation_padding_bytes, isolation_order_salt, isolation_colors[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::BufferIsolationConfig)
    initpos = position(e.io)
    x.base_offset_bytes != zero(Int64) && PB.encode(e, 1, x.base_offset_bytes)
    x.isolation_fuel != zero(Int64) && PB.encode(e, 2, x.isolation_fuel)
    x.isolation_padding_bytes != zero(Int64) && PB.encode(e, 3, x.isolation_padding_bytes)
    x.isolation_order_salt != zero(UInt64) && PB.encode(e, 4, x.isolation_order_salt)
    !isempty(x.isolation_colors) && PB.encode(e, 5, x.isolation_colors)
    return position(e.io) - initpos
end
function PB._encoded_size(x::BufferIsolationConfig)
    encoded_size = 0
    x.base_offset_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.base_offset_bytes, 1))
    x.isolation_fuel != zero(Int64) && (encoded_size += PB._encoded_size(x.isolation_fuel, 2))
    x.isolation_padding_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.isolation_padding_bytes, 3))
    x.isolation_order_salt != zero(UInt64) && (encoded_size += PB._encoded_size(x.isolation_order_salt, 4))
    !isempty(x.isolation_colors) && (encoded_size += PB._encoded_size(x.isolation_colors, 5))
    return encoded_size
end

@enumx var"AssignmentAlgorithmForComputationsWithoutOrderingProto.Value" DEFAULT=0 FAST_MERGE=1

struct AssignmentAlgorithmForComputationsWithoutOrderingProto end

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AssignmentAlgorithmForComputationsWithoutOrderingProto}, _endpos::Int=0, _group::Bool=false)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        Base.skip(d, wire_type)
    end
    return AssignmentAlgorithmForComputationsWithoutOrderingProto()
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AssignmentAlgorithmForComputationsWithoutOrderingProto)
    initpos = position(e.io)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AssignmentAlgorithmForComputationsWithoutOrderingProto)
    encoded_size = 0
    return encoded_size
end

struct BufferAllocationSliceProto
    offset::Int64
    size::Int64
    buffer_allocation_index::Int64
    element_type::xla.PrimitiveType.T
end
PB.default_values(::Type{BufferAllocationSliceProto}) = (;offset = zero(Int64), size = zero(Int64), buffer_allocation_index = zero(Int64), element_type = xla.PrimitiveType.PRIMITIVE_TYPE_INVALID)
PB.field_numbers(::Type{BufferAllocationSliceProto}) = (;offset = 1, size = 2, buffer_allocation_index = 3, element_type = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:BufferAllocationSliceProto}, _endpos::Int=0, _group::Bool=false)
    offset = zero(Int64)
    size = zero(Int64)
    buffer_allocation_index = zero(Int64)
    element_type = xla.PrimitiveType.PRIMITIVE_TYPE_INVALID
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            offset = PB.decode(d, Int64)
        elseif field_number == 2
            size = PB.decode(d, Int64)
        elseif field_number == 3
            buffer_allocation_index = PB.decode(d, Int64)
        elseif field_number == 4
            element_type = PB.decode(d, xla.PrimitiveType.T)
        else
            Base.skip(d, wire_type)
        end
    end
    return BufferAllocationSliceProto(offset, size, buffer_allocation_index, element_type)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::BufferAllocationSliceProto)
    initpos = position(e.io)
    x.offset != zero(Int64) && PB.encode(e, 1, x.offset)
    x.size != zero(Int64) && PB.encode(e, 2, x.size)
    x.buffer_allocation_index != zero(Int64) && PB.encode(e, 3, x.buffer_allocation_index)
    x.element_type != xla.PrimitiveType.PRIMITIVE_TYPE_INVALID && PB.encode(e, 4, x.element_type)
    return position(e.io) - initpos
end
function PB._encoded_size(x::BufferAllocationSliceProto)
    encoded_size = 0
    x.offset != zero(Int64) && (encoded_size += PB._encoded_size(x.offset, 1))
    x.size != zero(Int64) && (encoded_size += PB._encoded_size(x.size, 2))
    x.buffer_allocation_index != zero(Int64) && (encoded_size += PB._encoded_size(x.buffer_allocation_index, 3))
    x.element_type != xla.PrimitiveType.PRIMITIVE_TYPE_INVALID && (encoded_size += PB._encoded_size(x.element_type, 4))
    return encoded_size
end

@enumx var"BufferAssignmentAlgorithmProto.Value" DEFAULT=0 BEST_OF_SPATIAL_TEMPORAL=1 SPATIAL=2 TEMPORAL=3 FAST_MERGE=4 FAST_SPLIT=5 FAST_MERGE_WITH_FALLBACK=6

struct BufferAssignmentAlgorithmProto
    fallback_algorithm::var"BufferAssignmentAlgorithmProto.Value".T
end
PB.default_values(::Type{BufferAssignmentAlgorithmProto}) = (;fallback_algorithm = var"BufferAssignmentAlgorithmProto.Value".DEFAULT)
PB.field_numbers(::Type{BufferAssignmentAlgorithmProto}) = (;fallback_algorithm = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:BufferAssignmentAlgorithmProto}, _endpos::Int=0, _group::Bool=false)
    fallback_algorithm = var"BufferAssignmentAlgorithmProto.Value".DEFAULT
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            fallback_algorithm = PB.decode(d, var"BufferAssignmentAlgorithmProto.Value".T)
        else
            Base.skip(d, wire_type)
        end
    end
    return BufferAssignmentAlgorithmProto(fallback_algorithm)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::BufferAssignmentAlgorithmProto)
    initpos = position(e.io)
    x.fallback_algorithm != var"BufferAssignmentAlgorithmProto.Value".DEFAULT && PB.encode(e, 1, x.fallback_algorithm)
    return position(e.io) - initpos
end
function PB._encoded_size(x::BufferAssignmentAlgorithmProto)
    encoded_size = 0
    x.fallback_algorithm != var"BufferAssignmentAlgorithmProto.Value".DEFAULT && (encoded_size += PB._encoded_size(x.fallback_algorithm, 1))
    return encoded_size
end
