import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export ExecutionModeProto, ExecuteOptionsProto


@enumx ExecutionModeProto EXECUTION_MODE_UNSPECIFIED=0 EXECUTION_MODE_DEFAULT=1 EXECUTION_MODE_SYNCHRONOUS=2 EXECUTION_MODE_ASYNCHRONOUS=3

struct ExecuteOptionsProto
    arguments_are_tupled::Bool
    untuple_result::Bool
    launch_id::Int32
    strict_shape_checking::Bool
    use_major_to_minor_data_layout_for_callbacks::Bool
    execution_mode::ExecutionModeProto.T
    non_donatable_input_indices::Vector{Int32}
end
ExecuteOptionsProto(;arguments_are_tupled = false, untuple_result = false, launch_id = zero(Int32), strict_shape_checking = false, use_major_to_minor_data_layout_for_callbacks = false, execution_mode = ExecutionModeProto.EXECUTION_MODE_UNSPECIFIED, non_donatable_input_indices = Vector{Int32}()) = ExecuteOptionsProto(arguments_are_tupled, untuple_result, launch_id, strict_shape_checking, use_major_to_minor_data_layout_for_callbacks, execution_mode, non_donatable_input_indices)
PB.default_values(::Type{ExecuteOptionsProto}) = (;arguments_are_tupled = false, untuple_result = false, launch_id = zero(Int32), strict_shape_checking = false, use_major_to_minor_data_layout_for_callbacks = false, execution_mode = ExecutionModeProto.EXECUTION_MODE_UNSPECIFIED, non_donatable_input_indices = Vector{Int32}())
PB.field_numbers(::Type{ExecuteOptionsProto}) = (;arguments_are_tupled = 1, untuple_result = 2, launch_id = 3, strict_shape_checking = 4, use_major_to_minor_data_layout_for_callbacks = 8, execution_mode = 6, non_donatable_input_indices = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ExecuteOptionsProto})
    arguments_are_tupled = false
    untuple_result = false
    launch_id = zero(Int32)
    strict_shape_checking = false
    use_major_to_minor_data_layout_for_callbacks = false
    execution_mode = ExecutionModeProto.EXECUTION_MODE_UNSPECIFIED
    non_donatable_input_indices = PB.BufferedVector{Int32}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            arguments_are_tupled = PB.decode(d, Bool)
        elseif field_number == 2
            untuple_result = PB.decode(d, Bool)
        elseif field_number == 3
            launch_id = PB.decode(d, Int32)
        elseif field_number == 4
            strict_shape_checking = PB.decode(d, Bool)
        elseif field_number == 8
            use_major_to_minor_data_layout_for_callbacks = PB.decode(d, Bool)
        elseif field_number == 6
            execution_mode = PB.decode(d, ExecutionModeProto.T)
        elseif field_number == 7
            PB.decode!(d, wire_type, non_donatable_input_indices)
        else
            Base.skip(d, wire_type)
        end
    end
    return ExecuteOptionsProto(arguments_are_tupled, untuple_result, launch_id, strict_shape_checking, use_major_to_minor_data_layout_for_callbacks, execution_mode, non_donatable_input_indices[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ExecuteOptionsProto)
    initpos = position(e.io)
    x.arguments_are_tupled != false && PB.encode(e, 1, x.arguments_are_tupled)
    x.untuple_result != false && PB.encode(e, 2, x.untuple_result)
    x.launch_id != zero(Int32) && PB.encode(e, 3, x.launch_id)
    x.strict_shape_checking != false && PB.encode(e, 4, x.strict_shape_checking)
    x.use_major_to_minor_data_layout_for_callbacks != false && PB.encode(e, 8, x.use_major_to_minor_data_layout_for_callbacks)
    x.execution_mode != ExecutionModeProto.EXECUTION_MODE_UNSPECIFIED && PB.encode(e, 6, x.execution_mode)
    !isempty(x.non_donatable_input_indices) && PB.encode(e, 7, x.non_donatable_input_indices)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ExecuteOptionsProto)
    encoded_size = 0
    x.arguments_are_tupled != false && (encoded_size += PB._encoded_size(x.arguments_are_tupled, 1))
    x.untuple_result != false && (encoded_size += PB._encoded_size(x.untuple_result, 2))
    x.launch_id != zero(Int32) && (encoded_size += PB._encoded_size(x.launch_id, 3))
    x.strict_shape_checking != false && (encoded_size += PB._encoded_size(x.strict_shape_checking, 4))
    x.use_major_to_minor_data_layout_for_callbacks != false && (encoded_size += PB._encoded_size(x.use_major_to_minor_data_layout_for_callbacks, 8))
    x.execution_mode != ExecutionModeProto.EXECUTION_MODE_UNSPECIFIED && (encoded_size += PB._encoded_size(x.execution_mode, 6))
    !isempty(x.non_donatable_input_indices) && (encoded_size += PB._encoded_size(x.non_donatable_input_indices, 7))
    return encoded_size
end
