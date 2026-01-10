import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"HloScheduleProto.InstructionSequence", CustomCallApiVersion
export var"StackFrameIndexProto.StackFrame", var"HloInputs.LiteralDescriptor"
export HloPassMetadata, var"BufferAllocationProto.Assigned", Kind
export var"StackFrameIndexProto.FileLocation", var"LogicalBufferProto.Location"
export CrossProgramPrefetch, var"HloBufferDonorProto.BufferDonorEntryProto"
export var"HloInstructionProto.SliceDimensions", var"HloModuleProto.ProfileType"
export CustomCallSchedule, var"HeapSimulatorTrace.Event.Kind", HloScheduleProto, HloInputs
export HloModuleMetadataProto, BufferAllocationProto
export var"HloInputOutputAliasProto.AliasEntryProto", StackFrameIndexProto
export LogicalBufferProto, var"BufferAssignmentProto.BufferAlias", HloBufferDonorProto
export var"HloModuleProto.ProfileInfo", HloInstructionProto, var"HeapSimulatorTrace.Event"
export HloInputOutputAliasProto, HloComputationProto, HeapSimulatorTrace
export BufferAssignmentProto, HloModuleGroupProto, HloModuleProto, HloProto, HloSnapshot
export HloUnoptimizedSnapshot, OriginalValueRecoveryTableProto
export var"OriginalValueRecoveryTableProto.Entry"
abstract type var"##Abstract#HloProto" end
abstract type var"##Abstract#HloModuleGroupProto" end
abstract type var"##Abstract#HloSnapshot" end
abstract type var"##Abstract#OriginalValueRecoveryTableProto" end
abstract type var"##Abstract#OriginalValueRecoveryTableProto.Entry" end
abstract type var"##Abstract#HloModuleProto" end
abstract type var"##Abstract#HloUnoptimizedSnapshot" end


struct var"HloScheduleProto.InstructionSequence"
    instruction_ids::Vector{Int64}
end
var"HloScheduleProto.InstructionSequence"(;instruction_ids = Vector{Int64}()) = var"HloScheduleProto.InstructionSequence"(instruction_ids)
PB.default_values(::Type{var"HloScheduleProto.InstructionSequence"}) = (;instruction_ids = Vector{Int64}())
PB.field_numbers(::Type{var"HloScheduleProto.InstructionSequence"}) = (;instruction_ids = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"HloScheduleProto.InstructionSequence"})
    instruction_ids = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, instruction_ids)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"HloScheduleProto.InstructionSequence"(instruction_ids[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"HloScheduleProto.InstructionSequence")
    initpos = position(e.io)
    !isempty(x.instruction_ids) && PB.encode(e, 1, x.instruction_ids)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"HloScheduleProto.InstructionSequence")
    encoded_size = 0
    !isempty(x.instruction_ids) && (encoded_size += PB._encoded_size(x.instruction_ids, 1))
    return encoded_size
end

@enumx CustomCallApiVersion API_VERSION_UNSPECIFIED=0 API_VERSION_ORIGINAL=1 API_VERSION_STATUS_RETURNING=2 API_VERSION_STATUS_RETURNING_UNIFIED=3 API_VERSION_TYPED_FFI=4

struct var"StackFrameIndexProto.StackFrame"
    file_location_id::Int32
    parent_frame_id::Int32
end
var"StackFrameIndexProto.StackFrame"(;file_location_id = zero(Int32), parent_frame_id = zero(Int32)) = var"StackFrameIndexProto.StackFrame"(file_location_id, parent_frame_id)
PB.default_values(::Type{var"StackFrameIndexProto.StackFrame"}) = (;file_location_id = zero(Int32), parent_frame_id = zero(Int32))
PB.field_numbers(::Type{var"StackFrameIndexProto.StackFrame"}) = (;file_location_id = 1, parent_frame_id = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"StackFrameIndexProto.StackFrame"})
    file_location_id = zero(Int32)
    parent_frame_id = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            file_location_id = PB.decode(d, Int32)
        elseif field_number == 2
            parent_frame_id = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"StackFrameIndexProto.StackFrame"(file_location_id, parent_frame_id)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"StackFrameIndexProto.StackFrame")
    initpos = position(e.io)
    x.file_location_id != zero(Int32) && PB.encode(e, 1, x.file_location_id)
    x.parent_frame_id != zero(Int32) && PB.encode(e, 2, x.parent_frame_id)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"StackFrameIndexProto.StackFrame")
    encoded_size = 0
    x.file_location_id != zero(Int32) && (encoded_size += PB._encoded_size(x.file_location_id, 1))
    x.parent_frame_id != zero(Int32) && (encoded_size += PB._encoded_size(x.parent_frame_id, 2))
    return encoded_size
end

struct var"HloInputs.LiteralDescriptor"
    version::Int32
    argument_size_bytes::UInt64
end
var"HloInputs.LiteralDescriptor"(;version = zero(Int32), argument_size_bytes = zero(UInt64)) = var"HloInputs.LiteralDescriptor"(version, argument_size_bytes)
PB.default_values(::Type{var"HloInputs.LiteralDescriptor"}) = (;version = zero(Int32), argument_size_bytes = zero(UInt64))
PB.field_numbers(::Type{var"HloInputs.LiteralDescriptor"}) = (;version = 1, argument_size_bytes = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"HloInputs.LiteralDescriptor"})
    version = zero(Int32)
    argument_size_bytes = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            version = PB.decode(d, Int32)
        elseif field_number == 2
            argument_size_bytes = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"HloInputs.LiteralDescriptor"(version, argument_size_bytes)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"HloInputs.LiteralDescriptor")
    initpos = position(e.io)
    x.version != zero(Int32) && PB.encode(e, 1, x.version)
    x.argument_size_bytes != zero(UInt64) && PB.encode(e, 2, x.argument_size_bytes)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"HloInputs.LiteralDescriptor")
    encoded_size = 0
    x.version != zero(Int32) && (encoded_size += PB._encoded_size(x.version, 1))
    x.argument_size_bytes != zero(UInt64) && (encoded_size += PB._encoded_size(x.argument_size_bytes, 2))
    return encoded_size
end

struct HloPassMetadata
    pass_id::Int64
    pass_name::String
    pipeline_name::String
    dump_filenames::Vector{String}
    module_changed::Bool
    module_id::Int64
    module_group_module_ids::Vector{Int64}
    start_timestamp_usec::Int64
    end_timestamp_usec::Int64
    custom_metadata::Union{Nothing,google.protobuf.var"#Any"}
    kv_metrics::Vector{KeyValueMetric}
end
HloPassMetadata(;pass_id = zero(Int64), pass_name = "", pipeline_name = "", dump_filenames = Vector{String}(), module_changed = false, module_id = zero(Int64), module_group_module_ids = Vector{Int64}(), start_timestamp_usec = zero(Int64), end_timestamp_usec = zero(Int64), custom_metadata = nothing, kv_metrics = Vector{KeyValueMetric}()) = HloPassMetadata(pass_id, pass_name, pipeline_name, dump_filenames, module_changed, module_id, module_group_module_ids, start_timestamp_usec, end_timestamp_usec, custom_metadata, kv_metrics)
PB.default_values(::Type{HloPassMetadata}) = (;pass_id = zero(Int64), pass_name = "", pipeline_name = "", dump_filenames = Vector{String}(), module_changed = false, module_id = zero(Int64), module_group_module_ids = Vector{Int64}(), start_timestamp_usec = zero(Int64), end_timestamp_usec = zero(Int64), custom_metadata = nothing, kv_metrics = Vector{KeyValueMetric}())
PB.field_numbers(::Type{HloPassMetadata}) = (;pass_id = 1, pass_name = 2, pipeline_name = 3, dump_filenames = 4, module_changed = 5, module_id = 6, module_group_module_ids = 7, start_timestamp_usec = 8, end_timestamp_usec = 9, custom_metadata = 10, kv_metrics = 11)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloPassMetadata})
    pass_id = zero(Int64)
    pass_name = ""
    pipeline_name = ""
    dump_filenames = PB.BufferedVector{String}()
    module_changed = false
    module_id = zero(Int64)
    module_group_module_ids = PB.BufferedVector{Int64}()
    start_timestamp_usec = zero(Int64)
    end_timestamp_usec = zero(Int64)
    custom_metadata = Ref{Union{Nothing,google.protobuf.var"#Any"}}(nothing)
    kv_metrics = PB.BufferedVector{KeyValueMetric}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            pass_id = PB.decode(d, Int64)
        elseif field_number == 2
            pass_name = PB.decode(d, String)
        elseif field_number == 3
            pipeline_name = PB.decode(d, String)
        elseif field_number == 4
            PB.decode!(d, dump_filenames)
        elseif field_number == 5
            module_changed = PB.decode(d, Bool)
        elseif field_number == 6
            module_id = PB.decode(d, Int64)
        elseif field_number == 7
            PB.decode!(d, wire_type, module_group_module_ids)
        elseif field_number == 8
            start_timestamp_usec = PB.decode(d, Int64)
        elseif field_number == 9
            end_timestamp_usec = PB.decode(d, Int64)
        elseif field_number == 10
            PB.decode!(d, custom_metadata)
        elseif field_number == 11
            PB.decode!(d, kv_metrics)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloPassMetadata(pass_id, pass_name, pipeline_name, dump_filenames[], module_changed, module_id, module_group_module_ids[], start_timestamp_usec, end_timestamp_usec, custom_metadata[], kv_metrics[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloPassMetadata)
    initpos = position(e.io)
    x.pass_id != zero(Int64) && PB.encode(e, 1, x.pass_id)
    !isempty(x.pass_name) && PB.encode(e, 2, x.pass_name)
    !isempty(x.pipeline_name) && PB.encode(e, 3, x.pipeline_name)
    !isempty(x.dump_filenames) && PB.encode(e, 4, x.dump_filenames)
    x.module_changed != false && PB.encode(e, 5, x.module_changed)
    x.module_id != zero(Int64) && PB.encode(e, 6, x.module_id)
    !isempty(x.module_group_module_ids) && PB.encode(e, 7, x.module_group_module_ids)
    x.start_timestamp_usec != zero(Int64) && PB.encode(e, 8, x.start_timestamp_usec)
    x.end_timestamp_usec != zero(Int64) && PB.encode(e, 9, x.end_timestamp_usec)
    !isnothing(x.custom_metadata) && PB.encode(e, 10, x.custom_metadata)
    !isempty(x.kv_metrics) && PB.encode(e, 11, x.kv_metrics)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloPassMetadata)
    encoded_size = 0
    x.pass_id != zero(Int64) && (encoded_size += PB._encoded_size(x.pass_id, 1))
    !isempty(x.pass_name) && (encoded_size += PB._encoded_size(x.pass_name, 2))
    !isempty(x.pipeline_name) && (encoded_size += PB._encoded_size(x.pipeline_name, 3))
    !isempty(x.dump_filenames) && (encoded_size += PB._encoded_size(x.dump_filenames, 4))
    x.module_changed != false && (encoded_size += PB._encoded_size(x.module_changed, 5))
    x.module_id != zero(Int64) && (encoded_size += PB._encoded_size(x.module_id, 6))
    !isempty(x.module_group_module_ids) && (encoded_size += PB._encoded_size(x.module_group_module_ids, 7))
    x.start_timestamp_usec != zero(Int64) && (encoded_size += PB._encoded_size(x.start_timestamp_usec, 8))
    x.end_timestamp_usec != zero(Int64) && (encoded_size += PB._encoded_size(x.end_timestamp_usec, 9))
    !isnothing(x.custom_metadata) && (encoded_size += PB._encoded_size(x.custom_metadata, 10))
    !isempty(x.kv_metrics) && (encoded_size += PB._encoded_size(x.kv_metrics, 11))
    return encoded_size
end

struct var"BufferAllocationProto.Assigned"
    logical_buffer_id::Int64
    offset::Int64
    size::Int64
    element_type::PrimitiveType.T
end
var"BufferAllocationProto.Assigned"(;logical_buffer_id = zero(Int64), offset = zero(Int64), size = zero(Int64), element_type = PrimitiveType.PRIMITIVE_TYPE_INVALID) = var"BufferAllocationProto.Assigned"(logical_buffer_id, offset, size, element_type)
PB.default_values(::Type{var"BufferAllocationProto.Assigned"}) = (;logical_buffer_id = zero(Int64), offset = zero(Int64), size = zero(Int64), element_type = PrimitiveType.PRIMITIVE_TYPE_INVALID)
PB.field_numbers(::Type{var"BufferAllocationProto.Assigned"}) = (;logical_buffer_id = 1, offset = 2, size = 3, element_type = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"BufferAllocationProto.Assigned"})
    logical_buffer_id = zero(Int64)
    offset = zero(Int64)
    size = zero(Int64)
    element_type = PrimitiveType.PRIMITIVE_TYPE_INVALID
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            logical_buffer_id = PB.decode(d, Int64)
        elseif field_number == 2
            offset = PB.decode(d, Int64)
        elseif field_number == 3
            size = PB.decode(d, Int64)
        elseif field_number == 4
            element_type = PB.decode(d, PrimitiveType.T)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"BufferAllocationProto.Assigned"(logical_buffer_id, offset, size, element_type)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"BufferAllocationProto.Assigned")
    initpos = position(e.io)
    x.logical_buffer_id != zero(Int64) && PB.encode(e, 1, x.logical_buffer_id)
    x.offset != zero(Int64) && PB.encode(e, 2, x.offset)
    x.size != zero(Int64) && PB.encode(e, 3, x.size)
    x.element_type != PrimitiveType.PRIMITIVE_TYPE_INVALID && PB.encode(e, 4, x.element_type)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"BufferAllocationProto.Assigned")
    encoded_size = 0
    x.logical_buffer_id != zero(Int64) && (encoded_size += PB._encoded_size(x.logical_buffer_id, 1))
    x.offset != zero(Int64) && (encoded_size += PB._encoded_size(x.offset, 2))
    x.size != zero(Int64) && (encoded_size += PB._encoded_size(x.size, 3))
    x.element_type != PrimitiveType.PRIMITIVE_TYPE_INVALID && (encoded_size += PB._encoded_size(x.element_type, 4))
    return encoded_size
end

@enumx Kind UNDEFINED_ALIAS=0 MAY_ALIAS=1 MUST_ALIAS=2

struct var"StackFrameIndexProto.FileLocation"
    file_name_id::Int32
    function_name_id::Int32
    line::Int32
    end_line::Int32
    column::Int32
    end_column::Int32
end
var"StackFrameIndexProto.FileLocation"(;file_name_id = zero(Int32), function_name_id = zero(Int32), line = zero(Int32), end_line = zero(Int32), column = zero(Int32), end_column = zero(Int32)) = var"StackFrameIndexProto.FileLocation"(file_name_id, function_name_id, line, end_line, column, end_column)
PB.default_values(::Type{var"StackFrameIndexProto.FileLocation"}) = (;file_name_id = zero(Int32), function_name_id = zero(Int32), line = zero(Int32), end_line = zero(Int32), column = zero(Int32), end_column = zero(Int32))
PB.field_numbers(::Type{var"StackFrameIndexProto.FileLocation"}) = (;file_name_id = 1, function_name_id = 2, line = 3, end_line = 5, column = 4, end_column = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"StackFrameIndexProto.FileLocation"})
    file_name_id = zero(Int32)
    function_name_id = zero(Int32)
    line = zero(Int32)
    end_line = zero(Int32)
    column = zero(Int32)
    end_column = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            file_name_id = PB.decode(d, Int32)
        elseif field_number == 2
            function_name_id = PB.decode(d, Int32)
        elseif field_number == 3
            line = PB.decode(d, Int32)
        elseif field_number == 5
            end_line = PB.decode(d, Int32)
        elseif field_number == 4
            column = PB.decode(d, Int32)
        elseif field_number == 6
            end_column = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"StackFrameIndexProto.FileLocation"(file_name_id, function_name_id, line, end_line, column, end_column)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"StackFrameIndexProto.FileLocation")
    initpos = position(e.io)
    x.file_name_id != zero(Int32) && PB.encode(e, 1, x.file_name_id)
    x.function_name_id != zero(Int32) && PB.encode(e, 2, x.function_name_id)
    x.line != zero(Int32) && PB.encode(e, 3, x.line)
    x.end_line != zero(Int32) && PB.encode(e, 5, x.end_line)
    x.column != zero(Int32) && PB.encode(e, 4, x.column)
    x.end_column != zero(Int32) && PB.encode(e, 6, x.end_column)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"StackFrameIndexProto.FileLocation")
    encoded_size = 0
    x.file_name_id != zero(Int32) && (encoded_size += PB._encoded_size(x.file_name_id, 1))
    x.function_name_id != zero(Int32) && (encoded_size += PB._encoded_size(x.function_name_id, 2))
    x.line != zero(Int32) && (encoded_size += PB._encoded_size(x.line, 3))
    x.end_line != zero(Int32) && (encoded_size += PB._encoded_size(x.end_line, 5))
    x.column != zero(Int32) && (encoded_size += PB._encoded_size(x.column, 4))
    x.end_column != zero(Int32) && (encoded_size += PB._encoded_size(x.end_column, 6))
    return encoded_size
end

struct var"LogicalBufferProto.Location"
    instruction_name::String
    instruction_id::Int64
    shape_index::Vector{Int64}
end
var"LogicalBufferProto.Location"(;instruction_name = "", instruction_id = zero(Int64), shape_index = Vector{Int64}()) = var"LogicalBufferProto.Location"(instruction_name, instruction_id, shape_index)
PB.reserved_fields(::Type{var"LogicalBufferProto.Location"}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[1])
PB.default_values(::Type{var"LogicalBufferProto.Location"}) = (;instruction_name = "", instruction_id = zero(Int64), shape_index = Vector{Int64}())
PB.field_numbers(::Type{var"LogicalBufferProto.Location"}) = (;instruction_name = 2, instruction_id = 4, shape_index = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"LogicalBufferProto.Location"})
    instruction_name = ""
    instruction_id = zero(Int64)
    shape_index = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 2
            instruction_name = PB.decode(d, String)
        elseif field_number == 4
            instruction_id = PB.decode(d, Int64)
        elseif field_number == 3
            PB.decode!(d, wire_type, shape_index)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"LogicalBufferProto.Location"(instruction_name, instruction_id, shape_index[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"LogicalBufferProto.Location")
    initpos = position(e.io)
    !isempty(x.instruction_name) && PB.encode(e, 2, x.instruction_name)
    x.instruction_id != zero(Int64) && PB.encode(e, 4, x.instruction_id)
    !isempty(x.shape_index) && PB.encode(e, 3, x.shape_index)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"LogicalBufferProto.Location")
    encoded_size = 0
    !isempty(x.instruction_name) && (encoded_size += PB._encoded_size(x.instruction_name, 2))
    x.instruction_id != zero(Int64) && (encoded_size += PB._encoded_size(x.instruction_id, 4))
    !isempty(x.shape_index) && (encoded_size += PB._encoded_size(x.shape_index, 3))
    return encoded_size
end

struct CrossProgramPrefetch
    parameter::Int64
    index::Vector{Int64}
    offset::Int64
end
CrossProgramPrefetch(;parameter = zero(Int64), index = Vector{Int64}(), offset = zero(Int64)) = CrossProgramPrefetch(parameter, index, offset)
PB.default_values(::Type{CrossProgramPrefetch}) = (;parameter = zero(Int64), index = Vector{Int64}(), offset = zero(Int64))
PB.field_numbers(::Type{CrossProgramPrefetch}) = (;parameter = 1, index = 2, offset = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CrossProgramPrefetch})
    parameter = zero(Int64)
    index = PB.BufferedVector{Int64}()
    offset = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            parameter = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, wire_type, index)
        elseif field_number == 3
            offset = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return CrossProgramPrefetch(parameter, index[], offset)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CrossProgramPrefetch)
    initpos = position(e.io)
    x.parameter != zero(Int64) && PB.encode(e, 1, x.parameter)
    !isempty(x.index) && PB.encode(e, 2, x.index)
    x.offset != zero(Int64) && PB.encode(e, 3, x.offset)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CrossProgramPrefetch)
    encoded_size = 0
    x.parameter != zero(Int64) && (encoded_size += PB._encoded_size(x.parameter, 1))
    !isempty(x.index) && (encoded_size += PB._encoded_size(x.index, 2))
    x.offset != zero(Int64) && (encoded_size += PB._encoded_size(x.offset, 3))
    return encoded_size
end

struct var"HloBufferDonorProto.BufferDonorEntryProto"
    parameter_number::Int64
    parameter_shape_index::Vector{Int64}
end
var"HloBufferDonorProto.BufferDonorEntryProto"(;parameter_number = zero(Int64), parameter_shape_index = Vector{Int64}()) = var"HloBufferDonorProto.BufferDonorEntryProto"(parameter_number, parameter_shape_index)
PB.default_values(::Type{var"HloBufferDonorProto.BufferDonorEntryProto"}) = (;parameter_number = zero(Int64), parameter_shape_index = Vector{Int64}())
PB.field_numbers(::Type{var"HloBufferDonorProto.BufferDonorEntryProto"}) = (;parameter_number = 1, parameter_shape_index = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"HloBufferDonorProto.BufferDonorEntryProto"})
    parameter_number = zero(Int64)
    parameter_shape_index = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            parameter_number = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, wire_type, parameter_shape_index)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"HloBufferDonorProto.BufferDonorEntryProto"(parameter_number, parameter_shape_index[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"HloBufferDonorProto.BufferDonorEntryProto")
    initpos = position(e.io)
    x.parameter_number != zero(Int64) && PB.encode(e, 1, x.parameter_number)
    !isempty(x.parameter_shape_index) && PB.encode(e, 2, x.parameter_shape_index)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"HloBufferDonorProto.BufferDonorEntryProto")
    encoded_size = 0
    x.parameter_number != zero(Int64) && (encoded_size += PB._encoded_size(x.parameter_number, 1))
    !isempty(x.parameter_shape_index) && (encoded_size += PB._encoded_size(x.parameter_shape_index, 2))
    return encoded_size
end

struct var"HloInstructionProto.SliceDimensions"
    start::Int64
    limit::Int64
    stride::Int64
end
var"HloInstructionProto.SliceDimensions"(;start = zero(Int64), limit = zero(Int64), stride = zero(Int64)) = var"HloInstructionProto.SliceDimensions"(start, limit, stride)
PB.default_values(::Type{var"HloInstructionProto.SliceDimensions"}) = (;start = zero(Int64), limit = zero(Int64), stride = zero(Int64))
PB.field_numbers(::Type{var"HloInstructionProto.SliceDimensions"}) = (;start = 1, limit = 2, stride = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"HloInstructionProto.SliceDimensions"})
    start = zero(Int64)
    limit = zero(Int64)
    stride = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            start = PB.decode(d, Int64)
        elseif field_number == 2
            limit = PB.decode(d, Int64)
        elseif field_number == 3
            stride = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"HloInstructionProto.SliceDimensions"(start, limit, stride)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"HloInstructionProto.SliceDimensions")
    initpos = position(e.io)
    x.start != zero(Int64) && PB.encode(e, 1, x.start)
    x.limit != zero(Int64) && PB.encode(e, 2, x.limit)
    x.stride != zero(Int64) && PB.encode(e, 3, x.stride)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"HloInstructionProto.SliceDimensions")
    encoded_size = 0
    x.start != zero(Int64) && (encoded_size += PB._encoded_size(x.start, 1))
    x.limit != zero(Int64) && (encoded_size += PB._encoded_size(x.limit, 2))
    x.stride != zero(Int64) && (encoded_size += PB._encoded_size(x.stride, 3))
    return encoded_size
end

@enumx var"HloModuleProto.ProfileType" INVALID=0 FLAG=1 FUSION=2 LAYOUT=3 DOT=4 FLAGNET=5 SHARDING=6 SCHEDULE=7

@enumx CustomCallSchedule SCHEDULE_NONE=0 SCHEDULE_LATEST=1 SCHEDULE_EARLIEST=2

@enumx var"HeapSimulatorTrace.Event.Kind" ALLOC=0 FREE=1 SHARE_WITH=2

struct HloScheduleProto
    sequences::Dict{Int64,var"HloScheduleProto.InstructionSequence"}
end
HloScheduleProto(;sequences = Dict{Int64,var"HloScheduleProto.InstructionSequence"}()) = HloScheduleProto(sequences)
PB.default_values(::Type{HloScheduleProto}) = (;sequences = Dict{Int64,var"HloScheduleProto.InstructionSequence"}())
PB.field_numbers(::Type{HloScheduleProto}) = (;sequences = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloScheduleProto})
    sequences = Dict{Int64,var"HloScheduleProto.InstructionSequence"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, sequences)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloScheduleProto(sequences)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloScheduleProto)
    initpos = position(e.io)
    !isempty(x.sequences) && PB.encode(e, 1, x.sequences)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloScheduleProto)
    encoded_size = 0
    !isempty(x.sequences) && (encoded_size += PB._encoded_size(x.sequences, 1))
    return encoded_size
end

struct HloInputs
    arguments::Vector{LiteralProto}
    arguments_descriptors::Vector{var"HloInputs.LiteralDescriptor"}
end
HloInputs(;arguments = Vector{LiteralProto}(), arguments_descriptors = Vector{var"HloInputs.LiteralDescriptor"}()) = HloInputs(arguments, arguments_descriptors)
PB.default_values(::Type{HloInputs}) = (;arguments = Vector{LiteralProto}(), arguments_descriptors = Vector{var"HloInputs.LiteralDescriptor"}())
PB.field_numbers(::Type{HloInputs}) = (;arguments = 1, arguments_descriptors = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloInputs})
    arguments = PB.BufferedVector{LiteralProto}()
    arguments_descriptors = PB.BufferedVector{var"HloInputs.LiteralDescriptor"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, arguments)
        elseif field_number == 2
            PB.decode!(d, arguments_descriptors)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloInputs(arguments[], arguments_descriptors[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloInputs)
    initpos = position(e.io)
    !isempty(x.arguments) && PB.encode(e, 1, x.arguments)
    !isempty(x.arguments_descriptors) && PB.encode(e, 2, x.arguments_descriptors)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloInputs)
    encoded_size = 0
    !isempty(x.arguments) && (encoded_size += PB._encoded_size(x.arguments, 1))
    !isempty(x.arguments_descriptors) && (encoded_size += PB._encoded_size(x.arguments_descriptors, 2))
    return encoded_size
end

struct HloModuleMetadataProto
    canonical_module_id::Int64
    module_group_name::String
    original_module_id::Int64
    partitioned_module_ids::Vector{Int64}
    pass_metadata::Vector{HloPassMetadata}
end
HloModuleMetadataProto(;canonical_module_id = zero(Int64), module_group_name = "", original_module_id = zero(Int64), partitioned_module_ids = Vector{Int64}(), pass_metadata = Vector{HloPassMetadata}()) = HloModuleMetadataProto(canonical_module_id, module_group_name, original_module_id, partitioned_module_ids, pass_metadata)
PB.default_values(::Type{HloModuleMetadataProto}) = (;canonical_module_id = zero(Int64), module_group_name = "", original_module_id = zero(Int64), partitioned_module_ids = Vector{Int64}(), pass_metadata = Vector{HloPassMetadata}())
PB.field_numbers(::Type{HloModuleMetadataProto}) = (;canonical_module_id = 1, module_group_name = 2, original_module_id = 3, partitioned_module_ids = 4, pass_metadata = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloModuleMetadataProto})
    canonical_module_id = zero(Int64)
    module_group_name = ""
    original_module_id = zero(Int64)
    partitioned_module_ids = PB.BufferedVector{Int64}()
    pass_metadata = PB.BufferedVector{HloPassMetadata}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            canonical_module_id = PB.decode(d, Int64)
        elseif field_number == 2
            module_group_name = PB.decode(d, String)
        elseif field_number == 3
            original_module_id = PB.decode(d, Int64)
        elseif field_number == 4
            PB.decode!(d, wire_type, partitioned_module_ids)
        elseif field_number == 5
            PB.decode!(d, pass_metadata)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloModuleMetadataProto(canonical_module_id, module_group_name, original_module_id, partitioned_module_ids[], pass_metadata[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloModuleMetadataProto)
    initpos = position(e.io)
    x.canonical_module_id != zero(Int64) && PB.encode(e, 1, x.canonical_module_id)
    !isempty(x.module_group_name) && PB.encode(e, 2, x.module_group_name)
    x.original_module_id != zero(Int64) && PB.encode(e, 3, x.original_module_id)
    !isempty(x.partitioned_module_ids) && PB.encode(e, 4, x.partitioned_module_ids)
    !isempty(x.pass_metadata) && PB.encode(e, 5, x.pass_metadata)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloModuleMetadataProto)
    encoded_size = 0
    x.canonical_module_id != zero(Int64) && (encoded_size += PB._encoded_size(x.canonical_module_id, 1))
    !isempty(x.module_group_name) && (encoded_size += PB._encoded_size(x.module_group_name, 2))
    x.original_module_id != zero(Int64) && (encoded_size += PB._encoded_size(x.original_module_id, 3))
    !isempty(x.partitioned_module_ids) && (encoded_size += PB._encoded_size(x.partitioned_module_ids, 4))
    !isempty(x.pass_metadata) && (encoded_size += PB._encoded_size(x.pass_metadata, 5))
    return encoded_size
end

struct BufferAllocationProto
    index::Int64
    size::Int64
    is_thread_local::Bool
    is_tuple::Bool
    is_entry_computation_parameter::Bool
    is_constant::Bool
    parameter_number::Int64
    parameter_shape_index::Vector{Int64}
    is_parameter_aliased_with_output::Bool
    maybe_live_out::Bool
    color::Int64
    assigned::Vector{var"BufferAllocationProto.Assigned"}
end
BufferAllocationProto(;index = zero(Int64), size = zero(Int64), is_thread_local = false, is_tuple = false, is_entry_computation_parameter = false, is_constant = false, parameter_number = zero(Int64), parameter_shape_index = Vector{Int64}(), is_parameter_aliased_with_output = false, maybe_live_out = false, color = zero(Int64), assigned = Vector{var"BufferAllocationProto.Assigned"}()) = BufferAllocationProto(index, size, is_thread_local, is_tuple, is_entry_computation_parameter, is_constant, parameter_number, parameter_shape_index, is_parameter_aliased_with_output, maybe_live_out, color, assigned)
PB.default_values(::Type{BufferAllocationProto}) = (;index = zero(Int64), size = zero(Int64), is_thread_local = false, is_tuple = false, is_entry_computation_parameter = false, is_constant = false, parameter_number = zero(Int64), parameter_shape_index = Vector{Int64}(), is_parameter_aliased_with_output = false, maybe_live_out = false, color = zero(Int64), assigned = Vector{var"BufferAllocationProto.Assigned"}())
PB.field_numbers(::Type{BufferAllocationProto}) = (;index = 1, size = 2, is_thread_local = 3, is_tuple = 11, is_entry_computation_parameter = 5, is_constant = 12, parameter_number = 6, parameter_shape_index = 10, is_parameter_aliased_with_output = 13, maybe_live_out = 7, color = 8, assigned = 9)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:BufferAllocationProto})
    index = zero(Int64)
    size = zero(Int64)
    is_thread_local = false
    is_tuple = false
    is_entry_computation_parameter = false
    is_constant = false
    parameter_number = zero(Int64)
    parameter_shape_index = PB.BufferedVector{Int64}()
    is_parameter_aliased_with_output = false
    maybe_live_out = false
    color = zero(Int64)
    assigned = PB.BufferedVector{var"BufferAllocationProto.Assigned"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            index = PB.decode(d, Int64)
        elseif field_number == 2
            size = PB.decode(d, Int64)
        elseif field_number == 3
            is_thread_local = PB.decode(d, Bool)
        elseif field_number == 11
            is_tuple = PB.decode(d, Bool)
        elseif field_number == 5
            is_entry_computation_parameter = PB.decode(d, Bool)
        elseif field_number == 12
            is_constant = PB.decode(d, Bool)
        elseif field_number == 6
            parameter_number = PB.decode(d, Int64)
        elseif field_number == 10
            PB.decode!(d, wire_type, parameter_shape_index)
        elseif field_number == 13
            is_parameter_aliased_with_output = PB.decode(d, Bool)
        elseif field_number == 7
            maybe_live_out = PB.decode(d, Bool)
        elseif field_number == 8
            color = PB.decode(d, Int64)
        elseif field_number == 9
            PB.decode!(d, assigned)
        else
            Base.skip(d, wire_type)
        end
    end
    return BufferAllocationProto(index, size, is_thread_local, is_tuple, is_entry_computation_parameter, is_constant, parameter_number, parameter_shape_index[], is_parameter_aliased_with_output, maybe_live_out, color, assigned[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::BufferAllocationProto)
    initpos = position(e.io)
    x.index != zero(Int64) && PB.encode(e, 1, x.index)
    x.size != zero(Int64) && PB.encode(e, 2, x.size)
    x.is_thread_local != false && PB.encode(e, 3, x.is_thread_local)
    x.is_tuple != false && PB.encode(e, 11, x.is_tuple)
    x.is_entry_computation_parameter != false && PB.encode(e, 5, x.is_entry_computation_parameter)
    x.is_constant != false && PB.encode(e, 12, x.is_constant)
    x.parameter_number != zero(Int64) && PB.encode(e, 6, x.parameter_number)
    !isempty(x.parameter_shape_index) && PB.encode(e, 10, x.parameter_shape_index)
    x.is_parameter_aliased_with_output != false && PB.encode(e, 13, x.is_parameter_aliased_with_output)
    x.maybe_live_out != false && PB.encode(e, 7, x.maybe_live_out)
    x.color != zero(Int64) && PB.encode(e, 8, x.color)
    !isempty(x.assigned) && PB.encode(e, 9, x.assigned)
    return position(e.io) - initpos
end
function PB._encoded_size(x::BufferAllocationProto)
    encoded_size = 0
    x.index != zero(Int64) && (encoded_size += PB._encoded_size(x.index, 1))
    x.size != zero(Int64) && (encoded_size += PB._encoded_size(x.size, 2))
    x.is_thread_local != false && (encoded_size += PB._encoded_size(x.is_thread_local, 3))
    x.is_tuple != false && (encoded_size += PB._encoded_size(x.is_tuple, 11))
    x.is_entry_computation_parameter != false && (encoded_size += PB._encoded_size(x.is_entry_computation_parameter, 5))
    x.is_constant != false && (encoded_size += PB._encoded_size(x.is_constant, 12))
    x.parameter_number != zero(Int64) && (encoded_size += PB._encoded_size(x.parameter_number, 6))
    !isempty(x.parameter_shape_index) && (encoded_size += PB._encoded_size(x.parameter_shape_index, 10))
    x.is_parameter_aliased_with_output != false && (encoded_size += PB._encoded_size(x.is_parameter_aliased_with_output, 13))
    x.maybe_live_out != false && (encoded_size += PB._encoded_size(x.maybe_live_out, 7))
    x.color != zero(Int64) && (encoded_size += PB._encoded_size(x.color, 8))
    !isempty(x.assigned) && (encoded_size += PB._encoded_size(x.assigned, 9))
    return encoded_size
end

struct var"HloInputOutputAliasProto.AliasEntryProto"
    output_shape_index::Vector{Int64}
    parameter_number::Int64
    parameter_shape_index::Vector{Int64}
    kind::Kind.T
end
var"HloInputOutputAliasProto.AliasEntryProto"(;output_shape_index = Vector{Int64}(), parameter_number = zero(Int64), parameter_shape_index = Vector{Int64}(), kind = Kind.UNDEFINED_ALIAS) = var"HloInputOutputAliasProto.AliasEntryProto"(output_shape_index, parameter_number, parameter_shape_index, kind)
PB.default_values(::Type{var"HloInputOutputAliasProto.AliasEntryProto"}) = (;output_shape_index = Vector{Int64}(), parameter_number = zero(Int64), parameter_shape_index = Vector{Int64}(), kind = Kind.UNDEFINED_ALIAS)
PB.field_numbers(::Type{var"HloInputOutputAliasProto.AliasEntryProto"}) = (;output_shape_index = 1, parameter_number = 2, parameter_shape_index = 3, kind = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"HloInputOutputAliasProto.AliasEntryProto"})
    output_shape_index = PB.BufferedVector{Int64}()
    parameter_number = zero(Int64)
    parameter_shape_index = PB.BufferedVector{Int64}()
    kind = Kind.UNDEFINED_ALIAS
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, output_shape_index)
        elseif field_number == 2
            parameter_number = PB.decode(d, Int64)
        elseif field_number == 3
            PB.decode!(d, wire_type, parameter_shape_index)
        elseif field_number == 4
            kind = PB.decode(d, Kind.T)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"HloInputOutputAliasProto.AliasEntryProto"(output_shape_index[], parameter_number, parameter_shape_index[], kind)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"HloInputOutputAliasProto.AliasEntryProto")
    initpos = position(e.io)
    !isempty(x.output_shape_index) && PB.encode(e, 1, x.output_shape_index)
    x.parameter_number != zero(Int64) && PB.encode(e, 2, x.parameter_number)
    !isempty(x.parameter_shape_index) && PB.encode(e, 3, x.parameter_shape_index)
    x.kind != Kind.UNDEFINED_ALIAS && PB.encode(e, 4, x.kind)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"HloInputOutputAliasProto.AliasEntryProto")
    encoded_size = 0
    !isempty(x.output_shape_index) && (encoded_size += PB._encoded_size(x.output_shape_index, 1))
    x.parameter_number != zero(Int64) && (encoded_size += PB._encoded_size(x.parameter_number, 2))
    !isempty(x.parameter_shape_index) && (encoded_size += PB._encoded_size(x.parameter_shape_index, 3))
    x.kind != Kind.UNDEFINED_ALIAS && (encoded_size += PB._encoded_size(x.kind, 4))
    return encoded_size
end

struct StackFrameIndexProto
    file_names::Vector{String}
    function_names::Vector{String}
    file_locations::Vector{var"StackFrameIndexProto.FileLocation"}
    stack_frames::Vector{var"StackFrameIndexProto.StackFrame"}
end
StackFrameIndexProto(;file_names = Vector{String}(), function_names = Vector{String}(), file_locations = Vector{var"StackFrameIndexProto.FileLocation"}(), stack_frames = Vector{var"StackFrameIndexProto.StackFrame"}()) = StackFrameIndexProto(file_names, function_names, file_locations, stack_frames)
PB.default_values(::Type{StackFrameIndexProto}) = (;file_names = Vector{String}(), function_names = Vector{String}(), file_locations = Vector{var"StackFrameIndexProto.FileLocation"}(), stack_frames = Vector{var"StackFrameIndexProto.StackFrame"}())
PB.field_numbers(::Type{StackFrameIndexProto}) = (;file_names = 1, function_names = 2, file_locations = 3, stack_frames = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:StackFrameIndexProto})
    file_names = PB.BufferedVector{String}()
    function_names = PB.BufferedVector{String}()
    file_locations = PB.BufferedVector{var"StackFrameIndexProto.FileLocation"}()
    stack_frames = PB.BufferedVector{var"StackFrameIndexProto.StackFrame"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, file_names)
        elseif field_number == 2
            PB.decode!(d, function_names)
        elseif field_number == 3
            PB.decode!(d, file_locations)
        elseif field_number == 4
            PB.decode!(d, stack_frames)
        else
            Base.skip(d, wire_type)
        end
    end
    return StackFrameIndexProto(file_names[], function_names[], file_locations[], stack_frames[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::StackFrameIndexProto)
    initpos = position(e.io)
    !isempty(x.file_names) && PB.encode(e, 1, x.file_names)
    !isempty(x.function_names) && PB.encode(e, 2, x.function_names)
    !isempty(x.file_locations) && PB.encode(e, 3, x.file_locations)
    !isempty(x.stack_frames) && PB.encode(e, 4, x.stack_frames)
    return position(e.io) - initpos
end
function PB._encoded_size(x::StackFrameIndexProto)
    encoded_size = 0
    !isempty(x.file_names) && (encoded_size += PB._encoded_size(x.file_names, 1))
    !isempty(x.function_names) && (encoded_size += PB._encoded_size(x.function_names, 2))
    !isempty(x.file_locations) && (encoded_size += PB._encoded_size(x.file_locations, 3))
    !isempty(x.stack_frames) && (encoded_size += PB._encoded_size(x.stack_frames, 4))
    return encoded_size
end

struct LogicalBufferProto
    id::Int64
    size::Int64
    defined_at::Union{Nothing,var"LogicalBufferProto.Location"}
    color::Int64
end
LogicalBufferProto(;id = zero(Int64), size = zero(Int64), defined_at = nothing, color = zero(Int64)) = LogicalBufferProto(id, size, defined_at, color)
PB.default_values(::Type{LogicalBufferProto}) = (;id = zero(Int64), size = zero(Int64), defined_at = nothing, color = zero(Int64))
PB.field_numbers(::Type{LogicalBufferProto}) = (;id = 1, size = 2, defined_at = 3, color = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:LogicalBufferProto})
    id = zero(Int64)
    size = zero(Int64)
    defined_at = Ref{Union{Nothing,var"LogicalBufferProto.Location"}}(nothing)
    color = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            id = PB.decode(d, Int64)
        elseif field_number == 2
            size = PB.decode(d, Int64)
        elseif field_number == 3
            PB.decode!(d, defined_at)
        elseif field_number == 4
            color = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return LogicalBufferProto(id, size, defined_at[], color)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::LogicalBufferProto)
    initpos = position(e.io)
    x.id != zero(Int64) && PB.encode(e, 1, x.id)
    x.size != zero(Int64) && PB.encode(e, 2, x.size)
    !isnothing(x.defined_at) && PB.encode(e, 3, x.defined_at)
    x.color != zero(Int64) && PB.encode(e, 4, x.color)
    return position(e.io) - initpos
end
function PB._encoded_size(x::LogicalBufferProto)
    encoded_size = 0
    x.id != zero(Int64) && (encoded_size += PB._encoded_size(x.id, 1))
    x.size != zero(Int64) && (encoded_size += PB._encoded_size(x.size, 2))
    !isnothing(x.defined_at) && (encoded_size += PB._encoded_size(x.defined_at, 3))
    x.color != zero(Int64) && (encoded_size += PB._encoded_size(x.color, 4))
    return encoded_size
end

struct var"BufferAssignmentProto.BufferAlias"
    source_buffer_id::Int64
    location::Union{Nothing,var"LogicalBufferProto.Location"}
end
var"BufferAssignmentProto.BufferAlias"(;source_buffer_id = zero(Int64), location = nothing) = var"BufferAssignmentProto.BufferAlias"(source_buffer_id, location)
PB.default_values(::Type{var"BufferAssignmentProto.BufferAlias"}) = (;source_buffer_id = zero(Int64), location = nothing)
PB.field_numbers(::Type{var"BufferAssignmentProto.BufferAlias"}) = (;source_buffer_id = 1, location = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"BufferAssignmentProto.BufferAlias"})
    source_buffer_id = zero(Int64)
    location = Ref{Union{Nothing,var"LogicalBufferProto.Location"}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            source_buffer_id = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, location)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"BufferAssignmentProto.BufferAlias"(source_buffer_id, location[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"BufferAssignmentProto.BufferAlias")
    initpos = position(e.io)
    x.source_buffer_id != zero(Int64) && PB.encode(e, 1, x.source_buffer_id)
    !isnothing(x.location) && PB.encode(e, 2, x.location)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"BufferAssignmentProto.BufferAlias")
    encoded_size = 0
    x.source_buffer_id != zero(Int64) && (encoded_size += PB._encoded_size(x.source_buffer_id, 1))
    !isnothing(x.location) && (encoded_size += PB._encoded_size(x.location, 2))
    return encoded_size
end

struct HloBufferDonorProto
    entries::Vector{var"HloBufferDonorProto.BufferDonorEntryProto"}
end
HloBufferDonorProto(;entries = Vector{var"HloBufferDonorProto.BufferDonorEntryProto"}()) = HloBufferDonorProto(entries)
PB.default_values(::Type{HloBufferDonorProto}) = (;entries = Vector{var"HloBufferDonorProto.BufferDonorEntryProto"}())
PB.field_numbers(::Type{HloBufferDonorProto}) = (;entries = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloBufferDonorProto})
    entries = PB.BufferedVector{var"HloBufferDonorProto.BufferDonorEntryProto"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, entries)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloBufferDonorProto(entries[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloBufferDonorProto)
    initpos = position(e.io)
    !isempty(x.entries) && PB.encode(e, 1, x.entries)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloBufferDonorProto)
    encoded_size = 0
    !isempty(x.entries) && (encoded_size += PB._encoded_size(x.entries, 1))
    return encoded_size
end

struct var"HloModuleProto.ProfileInfo"
    profile_type::var"HloModuleProto.ProfileType".T
    relative_speedup::Float64
    profile_source::ProfileSource.T
    compilation_event::CompilationEvent.T
    fingerprint::String
    profile_generation_strategy::ProfileGenerationStrategy.T
    original_changelist::Int64
    changelist::Int64
end
var"HloModuleProto.ProfileInfo"(;profile_type = var"HloModuleProto.ProfileType".INVALID, relative_speedup = zero(Float64), profile_source = ProfileSource.PROFILE_SOURCE_UNKNOWN_SOURCE, compilation_event = CompilationEvent.COMPILATION_EVENT_UNKNOWN_EVENT, fingerprint = "", profile_generation_strategy = ProfileGenerationStrategy.PROFILE_GENERATION_STRATEGY_UNKNOWN, original_changelist = zero(Int64), changelist = zero(Int64)) = var"HloModuleProto.ProfileInfo"(profile_type, relative_speedup, profile_source, compilation_event, fingerprint, profile_generation_strategy, original_changelist, changelist)
PB.default_values(::Type{var"HloModuleProto.ProfileInfo"}) = (;profile_type = var"HloModuleProto.ProfileType".INVALID, relative_speedup = zero(Float64), profile_source = ProfileSource.PROFILE_SOURCE_UNKNOWN_SOURCE, compilation_event = CompilationEvent.COMPILATION_EVENT_UNKNOWN_EVENT, fingerprint = "", profile_generation_strategy = ProfileGenerationStrategy.PROFILE_GENERATION_STRATEGY_UNKNOWN, original_changelist = zero(Int64), changelist = zero(Int64))
PB.field_numbers(::Type{var"HloModuleProto.ProfileInfo"}) = (;profile_type = 1, relative_speedup = 2, profile_source = 3, compilation_event = 4, fingerprint = 5, profile_generation_strategy = 6, original_changelist = 7, changelist = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"HloModuleProto.ProfileInfo"})
    profile_type = var"HloModuleProto.ProfileType".INVALID
    relative_speedup = zero(Float64)
    profile_source = ProfileSource.PROFILE_SOURCE_UNKNOWN_SOURCE
    compilation_event = CompilationEvent.COMPILATION_EVENT_UNKNOWN_EVENT
    fingerprint = ""
    profile_generation_strategy = ProfileGenerationStrategy.PROFILE_GENERATION_STRATEGY_UNKNOWN
    original_changelist = zero(Int64)
    changelist = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            profile_type = PB.decode(d, var"HloModuleProto.ProfileType".T)
        elseif field_number == 2
            relative_speedup = PB.decode(d, Float64)
        elseif field_number == 3
            profile_source = PB.decode(d, ProfileSource.T)
        elseif field_number == 4
            compilation_event = PB.decode(d, CompilationEvent.T)
        elseif field_number == 5
            fingerprint = PB.decode(d, String)
        elseif field_number == 6
            profile_generation_strategy = PB.decode(d, ProfileGenerationStrategy.T)
        elseif field_number == 7
            original_changelist = PB.decode(d, Int64)
        elseif field_number == 8
            changelist = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"HloModuleProto.ProfileInfo"(profile_type, relative_speedup, profile_source, compilation_event, fingerprint, profile_generation_strategy, original_changelist, changelist)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"HloModuleProto.ProfileInfo")
    initpos = position(e.io)
    x.profile_type != var"HloModuleProto.ProfileType".INVALID && PB.encode(e, 1, x.profile_type)
    x.relative_speedup !== zero(Float64) && PB.encode(e, 2, x.relative_speedup)
    x.profile_source != ProfileSource.PROFILE_SOURCE_UNKNOWN_SOURCE && PB.encode(e, 3, x.profile_source)
    x.compilation_event != CompilationEvent.COMPILATION_EVENT_UNKNOWN_EVENT && PB.encode(e, 4, x.compilation_event)
    !isempty(x.fingerprint) && PB.encode(e, 5, x.fingerprint)
    x.profile_generation_strategy != ProfileGenerationStrategy.PROFILE_GENERATION_STRATEGY_UNKNOWN && PB.encode(e, 6, x.profile_generation_strategy)
    x.original_changelist != zero(Int64) && PB.encode(e, 7, x.original_changelist)
    x.changelist != zero(Int64) && PB.encode(e, 8, x.changelist)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"HloModuleProto.ProfileInfo")
    encoded_size = 0
    x.profile_type != var"HloModuleProto.ProfileType".INVALID && (encoded_size += PB._encoded_size(x.profile_type, 1))
    x.relative_speedup !== zero(Float64) && (encoded_size += PB._encoded_size(x.relative_speedup, 2))
    x.profile_source != ProfileSource.PROFILE_SOURCE_UNKNOWN_SOURCE && (encoded_size += PB._encoded_size(x.profile_source, 3))
    x.compilation_event != CompilationEvent.COMPILATION_EVENT_UNKNOWN_EVENT && (encoded_size += PB._encoded_size(x.compilation_event, 4))
    !isempty(x.fingerprint) && (encoded_size += PB._encoded_size(x.fingerprint, 5))
    x.profile_generation_strategy != ProfileGenerationStrategy.PROFILE_GENERATION_STRATEGY_UNKNOWN && (encoded_size += PB._encoded_size(x.profile_generation_strategy, 6))
    x.original_changelist != zero(Int64) && (encoded_size += PB._encoded_size(x.original_changelist, 7))
    x.changelist != zero(Int64) && (encoded_size += PB._encoded_size(x.changelist, 8))
    return encoded_size
end

struct HloInstructionProto
    name::String
    opcode::String
    shape::Union{Nothing,ShapeProto}
    metadata::Union{Nothing,OpMetadata}
    literal::Union{Nothing,LiteralProto}
    parameter_number::Int64
    fusion_kind::String
    tuple_index::Int64
    dimensions::Vector{Int64}
    window::Union{Nothing,Window}
    convolution_dimension_numbers::Union{Nothing,ConvolutionDimensionNumbers}
    feature_group_count::Int64
    batch_group_count::Int64
    slice_dimensions::Vector{var"HloInstructionProto.SliceDimensions"}
    exponent_bits::Int32
    mantissa_bits::Int32
    dynamic_slice_sizes::Vector{Int64}
    padding_config::Union{Nothing,PaddingConfig}
    outfeed_config::Vector{UInt8}
    distribution::RandomDistribution.T
    epsilon::Float32
    feature_index::Int64
    channel_id::Int64
    infeed_config::Vector{UInt8}
    custom_call_target::String
    outfeed_shape::Union{Nothing,ShapeProto}
    dot_dimension_numbers::Union{Nothing,DotDimensionNumbers}
    ragged_dot_dimension_numbers::Union{Nothing,RaggedDotDimensionNumbers}
    fft_type::FftType.T
    fft_length::Vector{Int64}
    comparison_direction::String
    gather_dimension_numbers::Union{Nothing,GatherDimensionNumbers}
    gather_slice_sizes::Vector{Int64}
    id::Int64
    operand_ids::Vector{Int64}
    control_predecessor_ids::Vector{Int64}
    called_computation_ids::Vector{Int64}
    sharding::Union{Nothing,OpSharding}
    backend_config::Vector{UInt8}
    replica_groups::Vector{ReplicaGroup}
    all_reduce_id::Int64
    use_global_device_ids::Bool
    is_host_transfer::Bool
    is_stable::Bool
    scatter_dimension_numbers::Union{Nothing,ScatterDimensionNumbers}
    precision_config::Union{Nothing,PrecisionConfig}
    source_target_pairs::Vector{SourceTarget}
    domain_entry_sharding::Union{Nothing,OpSharding}
    domain_exit_sharding::Union{Nothing,OpSharding}
    constrain_layout::Bool
    operand_shapes_with_layout::Vector{ShapeProto}
    triangular_solve_options::Union{Nothing,TriangularSolveOptions}
    cholesky_options::Union{Nothing,CholeskyOptions}
    parameter_replication::Union{Nothing,ParameterReplication}
    custom_call_has_side_effect::Bool
    output_operand_aliasing::Vector{OutputOperandAliasing}
    custom_call_schedule::CustomCallSchedule.T
    delta::Int64
    indices_are_sorted::Bool
    frontend_attributes::Union{Nothing,FrontendAttributes}
    unique_indices::Bool
    rng_algorithm::RandomAlgorithm.T
    comparison_type::String
    is_cross_program_prefetch::Bool
    optional_cross_program_prefetch_index::Union{Nothing,OneOf{Int32}}
    padding_type::PaddingType.T
    custom_call_api_version::CustomCallApiVersion.T
    async_execution_thread::String
    k::Int64
    largest::Bool
    statistics_viz::Union{Nothing,StatisticsViz}
    collective_device_list::Union{Nothing,CollectiveDeviceListProto}
    original_value::Union{Nothing,OriginalValueProto}
    is_composite::Bool
    result_accuracy::Union{Nothing,ResultAccuracy}
end
HloInstructionProto(;name = "", opcode = "", shape = nothing, metadata = nothing, literal = nothing, parameter_number = zero(Int64), fusion_kind = "", tuple_index = zero(Int64), dimensions = Vector{Int64}(), window = nothing, convolution_dimension_numbers = nothing, feature_group_count = zero(Int64), batch_group_count = zero(Int64), slice_dimensions = Vector{var"HloInstructionProto.SliceDimensions"}(), exponent_bits = zero(Int32), mantissa_bits = zero(Int32), dynamic_slice_sizes = Vector{Int64}(), padding_config = nothing, outfeed_config = UInt8[], distribution = RandomDistribution.RNG_INVALID, epsilon = zero(Float32), feature_index = zero(Int64), channel_id = zero(Int64), infeed_config = UInt8[], custom_call_target = "", outfeed_shape = nothing, dot_dimension_numbers = nothing, ragged_dot_dimension_numbers = nothing, fft_type = FftType.FFT, fft_length = Vector{Int64}(), comparison_direction = "", gather_dimension_numbers = nothing, gather_slice_sizes = Vector{Int64}(), id = zero(Int64), operand_ids = Vector{Int64}(), control_predecessor_ids = Vector{Int64}(), called_computation_ids = Vector{Int64}(), sharding = nothing, backend_config = UInt8[], replica_groups = Vector{ReplicaGroup}(), all_reduce_id = zero(Int64), use_global_device_ids = false, is_host_transfer = false, is_stable = false, scatter_dimension_numbers = nothing, precision_config = nothing, source_target_pairs = Vector{SourceTarget}(), domain_entry_sharding = nothing, domain_exit_sharding = nothing, constrain_layout = false, operand_shapes_with_layout = Vector{ShapeProto}(), triangular_solve_options = nothing, cholesky_options = nothing, parameter_replication = nothing, custom_call_has_side_effect = false, output_operand_aliasing = Vector{OutputOperandAliasing}(), custom_call_schedule = CustomCallSchedule.SCHEDULE_NONE, delta = zero(Int64), indices_are_sorted = false, frontend_attributes = nothing, unique_indices = false, rng_algorithm = RandomAlgorithm.RNG_DEFAULT, comparison_type = "", is_cross_program_prefetch = false, optional_cross_program_prefetch_index = nothing, padding_type = PaddingType.PADDING_INVALID, custom_call_api_version = CustomCallApiVersion.API_VERSION_UNSPECIFIED, async_execution_thread = "", k = zero(Int64), largest = false, statistics_viz = nothing, collective_device_list = nothing, original_value = nothing, is_composite = false, result_accuracy = nothing) = HloInstructionProto(name, opcode, shape, metadata, literal, parameter_number, fusion_kind, tuple_index, dimensions, window, convolution_dimension_numbers, feature_group_count, batch_group_count, slice_dimensions, exponent_bits, mantissa_bits, dynamic_slice_sizes, padding_config, outfeed_config, distribution, epsilon, feature_index, channel_id, infeed_config, custom_call_target, outfeed_shape, dot_dimension_numbers, ragged_dot_dimension_numbers, fft_type, fft_length, comparison_direction, gather_dimension_numbers, gather_slice_sizes, id, operand_ids, control_predecessor_ids, called_computation_ids, sharding, backend_config, replica_groups, all_reduce_id, use_global_device_ids, is_host_transfer, is_stable, scatter_dimension_numbers, precision_config, source_target_pairs, domain_entry_sharding, domain_exit_sharding, constrain_layout, operand_shapes_with_layout, triangular_solve_options, cholesky_options, parameter_replication, custom_call_has_side_effect, output_operand_aliasing, custom_call_schedule, delta, indices_are_sorted, frontend_attributes, unique_indices, rng_algorithm, comparison_type, is_cross_program_prefetch, optional_cross_program_prefetch_index, padding_type, custom_call_api_version, async_execution_thread, k, largest, statistics_viz, collective_device_list, original_value, is_composite, result_accuracy)
PB.reserved_fields(::Type{HloInstructionProto}) = (names = ["parameter_name", "fused_instructions_computation", "operand_names", "control_predecessor_names", "called_computation_names", "replica_group_ids", "custom_call_opaque", "all_reduce_barrier"], numbers = Union{Int,UnitRange{Int}}[10, 12, 4, 5, 6, 44, 53, 46, 41, 42, 64, 78, 83, 84, 86])
PB.oneof_field_types(::Type{HloInstructionProto}) = (;
    optional_cross_program_prefetch_index = (;cross_program_prefetch_index=Int32),
)
PB.default_values(::Type{HloInstructionProto}) = (;name = "", opcode = "", shape = nothing, metadata = nothing, literal = nothing, parameter_number = zero(Int64), fusion_kind = "", tuple_index = zero(Int64), dimensions = Vector{Int64}(), window = nothing, convolution_dimension_numbers = nothing, feature_group_count = zero(Int64), batch_group_count = zero(Int64), slice_dimensions = Vector{var"HloInstructionProto.SliceDimensions"}(), exponent_bits = zero(Int32), mantissa_bits = zero(Int32), dynamic_slice_sizes = Vector{Int64}(), padding_config = nothing, outfeed_config = UInt8[], distribution = RandomDistribution.RNG_INVALID, epsilon = zero(Float32), feature_index = zero(Int64), channel_id = zero(Int64), infeed_config = UInt8[], custom_call_target = "", outfeed_shape = nothing, dot_dimension_numbers = nothing, ragged_dot_dimension_numbers = nothing, fft_type = FftType.FFT, fft_length = Vector{Int64}(), comparison_direction = "", gather_dimension_numbers = nothing, gather_slice_sizes = Vector{Int64}(), id = zero(Int64), operand_ids = Vector{Int64}(), control_predecessor_ids = Vector{Int64}(), called_computation_ids = Vector{Int64}(), sharding = nothing, backend_config = UInt8[], replica_groups = Vector{ReplicaGroup}(), all_reduce_id = zero(Int64), use_global_device_ids = false, is_host_transfer = false, is_stable = false, scatter_dimension_numbers = nothing, precision_config = nothing, source_target_pairs = Vector{SourceTarget}(), domain_entry_sharding = nothing, domain_exit_sharding = nothing, constrain_layout = false, operand_shapes_with_layout = Vector{ShapeProto}(), triangular_solve_options = nothing, cholesky_options = nothing, parameter_replication = nothing, custom_call_has_side_effect = false, output_operand_aliasing = Vector{OutputOperandAliasing}(), custom_call_schedule = CustomCallSchedule.SCHEDULE_NONE, delta = zero(Int64), indices_are_sorted = false, frontend_attributes = nothing, unique_indices = false, rng_algorithm = RandomAlgorithm.RNG_DEFAULT, comparison_type = "", is_cross_program_prefetch = false, cross_program_prefetch_index = zero(Int32), padding_type = PaddingType.PADDING_INVALID, custom_call_api_version = CustomCallApiVersion.API_VERSION_UNSPECIFIED, async_execution_thread = "", k = zero(Int64), largest = false, statistics_viz = nothing, collective_device_list = nothing, original_value = nothing, is_composite = false, result_accuracy = nothing)
PB.field_numbers(::Type{HloInstructionProto}) = (;name = 1, opcode = 2, shape = 3, metadata = 7, literal = 8, parameter_number = 9, fusion_kind = 11, tuple_index = 13, dimensions = 14, window = 15, convolution_dimension_numbers = 16, feature_group_count = 50, batch_group_count = 58, slice_dimensions = 17, exponent_bits = 18, mantissa_bits = 19, dynamic_slice_sizes = 20, padding_config = 21, outfeed_config = 22, distribution = 23, epsilon = 24, feature_index = 25, channel_id = 26, infeed_config = 27, custom_call_target = 28, outfeed_shape = 29, dot_dimension_numbers = 30, ragged_dot_dimension_numbers = 90, fft_type = 31, fft_length = 32, comparison_direction = 63, gather_dimension_numbers = 33, gather_slice_sizes = 34, id = 35, operand_ids = 36, control_predecessor_ids = 37, called_computation_ids = 38, sharding = 40, backend_config = 43, replica_groups = 49, all_reduce_id = 45, use_global_device_ids = 71, is_host_transfer = 47, is_stable = 60, scatter_dimension_numbers = 48, precision_config = 51, source_target_pairs = 52, domain_entry_sharding = 54, domain_exit_sharding = 55, constrain_layout = 56, operand_shapes_with_layout = 57, triangular_solve_options = 59, cholesky_options = 62, parameter_replication = 61, custom_call_has_side_effect = 65, output_operand_aliasing = 74, custom_call_schedule = 76, delta = 66, indices_are_sorted = 67, frontend_attributes = 68, unique_indices = 69, rng_algorithm = 70, comparison_type = 72, is_cross_program_prefetch = 73, cross_program_prefetch_index = 80, padding_type = 75, custom_call_api_version = 77, async_execution_thread = 79, k = 81, largest = 85, statistics_viz = 82, collective_device_list = 87, original_value = 88, is_composite = 89, result_accuracy = 91)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloInstructionProto})
    name = ""
    opcode = ""
    shape = Ref{Union{Nothing,ShapeProto}}(nothing)
    metadata = Ref{Union{Nothing,OpMetadata}}(nothing)
    literal = Ref{Union{Nothing,LiteralProto}}(nothing)
    parameter_number = zero(Int64)
    fusion_kind = ""
    tuple_index = zero(Int64)
    dimensions = PB.BufferedVector{Int64}()
    window = Ref{Union{Nothing,Window}}(nothing)
    convolution_dimension_numbers = Ref{Union{Nothing,ConvolutionDimensionNumbers}}(nothing)
    feature_group_count = zero(Int64)
    batch_group_count = zero(Int64)
    slice_dimensions = PB.BufferedVector{var"HloInstructionProto.SliceDimensions"}()
    exponent_bits = zero(Int32)
    mantissa_bits = zero(Int32)
    dynamic_slice_sizes = PB.BufferedVector{Int64}()
    padding_config = Ref{Union{Nothing,PaddingConfig}}(nothing)
    outfeed_config = UInt8[]
    distribution = RandomDistribution.RNG_INVALID
    epsilon = zero(Float32)
    feature_index = zero(Int64)
    channel_id = zero(Int64)
    infeed_config = UInt8[]
    custom_call_target = ""
    outfeed_shape = Ref{Union{Nothing,ShapeProto}}(nothing)
    dot_dimension_numbers = Ref{Union{Nothing,DotDimensionNumbers}}(nothing)
    ragged_dot_dimension_numbers = Ref{Union{Nothing,RaggedDotDimensionNumbers}}(nothing)
    fft_type = FftType.FFT
    fft_length = PB.BufferedVector{Int64}()
    comparison_direction = ""
    gather_dimension_numbers = Ref{Union{Nothing,GatherDimensionNumbers}}(nothing)
    gather_slice_sizes = PB.BufferedVector{Int64}()
    id = zero(Int64)
    operand_ids = PB.BufferedVector{Int64}()
    control_predecessor_ids = PB.BufferedVector{Int64}()
    called_computation_ids = PB.BufferedVector{Int64}()
    sharding = Ref{Union{Nothing,OpSharding}}(nothing)
    backend_config = UInt8[]
    replica_groups = PB.BufferedVector{ReplicaGroup}()
    all_reduce_id = zero(Int64)
    use_global_device_ids = false
    is_host_transfer = false
    is_stable = false
    scatter_dimension_numbers = Ref{Union{Nothing,ScatterDimensionNumbers}}(nothing)
    precision_config = Ref{Union{Nothing,PrecisionConfig}}(nothing)
    source_target_pairs = PB.BufferedVector{SourceTarget}()
    domain_entry_sharding = Ref{Union{Nothing,OpSharding}}(nothing)
    domain_exit_sharding = Ref{Union{Nothing,OpSharding}}(nothing)
    constrain_layout = false
    operand_shapes_with_layout = PB.BufferedVector{ShapeProto}()
    triangular_solve_options = Ref{Union{Nothing,TriangularSolveOptions}}(nothing)
    cholesky_options = Ref{Union{Nothing,CholeskyOptions}}(nothing)
    parameter_replication = Ref{Union{Nothing,ParameterReplication}}(nothing)
    custom_call_has_side_effect = false
    output_operand_aliasing = PB.BufferedVector{OutputOperandAliasing}()
    custom_call_schedule = CustomCallSchedule.SCHEDULE_NONE
    delta = zero(Int64)
    indices_are_sorted = false
    frontend_attributes = Ref{Union{Nothing,FrontendAttributes}}(nothing)
    unique_indices = false
    rng_algorithm = RandomAlgorithm.RNG_DEFAULT
    comparison_type = ""
    is_cross_program_prefetch = false
    optional_cross_program_prefetch_index = nothing
    padding_type = PaddingType.PADDING_INVALID
    custom_call_api_version = CustomCallApiVersion.API_VERSION_UNSPECIFIED
    async_execution_thread = ""
    k = zero(Int64)
    largest = false
    statistics_viz = Ref{Union{Nothing,StatisticsViz}}(nothing)
    collective_device_list = Ref{Union{Nothing,CollectiveDeviceListProto}}(nothing)
    original_value = Ref{Union{Nothing,OriginalValueProto}}(nothing)
    is_composite = false
    result_accuracy = Ref{Union{Nothing,ResultAccuracy}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            opcode = PB.decode(d, String)
        elseif field_number == 3
            PB.decode!(d, shape)
        elseif field_number == 7
            PB.decode!(d, metadata)
        elseif field_number == 8
            PB.decode!(d, literal)
        elseif field_number == 9
            parameter_number = PB.decode(d, Int64)
        elseif field_number == 11
            fusion_kind = PB.decode(d, String)
        elseif field_number == 13
            tuple_index = PB.decode(d, Int64)
        elseif field_number == 14
            PB.decode!(d, wire_type, dimensions)
        elseif field_number == 15
            PB.decode!(d, window)
        elseif field_number == 16
            PB.decode!(d, convolution_dimension_numbers)
        elseif field_number == 50
            feature_group_count = PB.decode(d, Int64)
        elseif field_number == 58
            batch_group_count = PB.decode(d, Int64)
        elseif field_number == 17
            PB.decode!(d, slice_dimensions)
        elseif field_number == 18
            exponent_bits = PB.decode(d, Int32)
        elseif field_number == 19
            mantissa_bits = PB.decode(d, Int32)
        elseif field_number == 20
            PB.decode!(d, wire_type, dynamic_slice_sizes)
        elseif field_number == 21
            PB.decode!(d, padding_config)
        elseif field_number == 22
            outfeed_config = PB.decode(d, Vector{UInt8})
        elseif field_number == 23
            distribution = PB.decode(d, RandomDistribution.T)
        elseif field_number == 24
            epsilon = PB.decode(d, Float32)
        elseif field_number == 25
            feature_index = PB.decode(d, Int64)
        elseif field_number == 26
            channel_id = PB.decode(d, Int64)
        elseif field_number == 27
            infeed_config = PB.decode(d, Vector{UInt8})
        elseif field_number == 28
            custom_call_target = PB.decode(d, String)
        elseif field_number == 29
            PB.decode!(d, outfeed_shape)
        elseif field_number == 30
            PB.decode!(d, dot_dimension_numbers)
        elseif field_number == 90
            PB.decode!(d, ragged_dot_dimension_numbers)
        elseif field_number == 31
            fft_type = PB.decode(d, FftType.T)
        elseif field_number == 32
            PB.decode!(d, wire_type, fft_length)
        elseif field_number == 63
            comparison_direction = PB.decode(d, String)
        elseif field_number == 33
            PB.decode!(d, gather_dimension_numbers)
        elseif field_number == 34
            PB.decode!(d, wire_type, gather_slice_sizes)
        elseif field_number == 35
            id = PB.decode(d, Int64)
        elseif field_number == 36
            PB.decode!(d, wire_type, operand_ids)
        elseif field_number == 37
            PB.decode!(d, wire_type, control_predecessor_ids)
        elseif field_number == 38
            PB.decode!(d, wire_type, called_computation_ids)
        elseif field_number == 40
            PB.decode!(d, sharding)
        elseif field_number == 43
            backend_config = PB.decode(d, Vector{UInt8})
        elseif field_number == 49
            PB.decode!(d, replica_groups)
        elseif field_number == 45
            all_reduce_id = PB.decode(d, Int64)
        elseif field_number == 71
            use_global_device_ids = PB.decode(d, Bool)
        elseif field_number == 47
            is_host_transfer = PB.decode(d, Bool)
        elseif field_number == 60
            is_stable = PB.decode(d, Bool)
        elseif field_number == 48
            PB.decode!(d, scatter_dimension_numbers)
        elseif field_number == 51
            PB.decode!(d, precision_config)
        elseif field_number == 52
            PB.decode!(d, source_target_pairs)
        elseif field_number == 54
            PB.decode!(d, domain_entry_sharding)
        elseif field_number == 55
            PB.decode!(d, domain_exit_sharding)
        elseif field_number == 56
            constrain_layout = PB.decode(d, Bool)
        elseif field_number == 57
            PB.decode!(d, operand_shapes_with_layout)
        elseif field_number == 59
            PB.decode!(d, triangular_solve_options)
        elseif field_number == 62
            PB.decode!(d, cholesky_options)
        elseif field_number == 61
            PB.decode!(d, parameter_replication)
        elseif field_number == 65
            custom_call_has_side_effect = PB.decode(d, Bool)
        elseif field_number == 74
            PB.decode!(d, output_operand_aliasing)
        elseif field_number == 76
            custom_call_schedule = PB.decode(d, CustomCallSchedule.T)
        elseif field_number == 66
            delta = PB.decode(d, Int64)
        elseif field_number == 67
            indices_are_sorted = PB.decode(d, Bool)
        elseif field_number == 68
            PB.decode!(d, frontend_attributes)
        elseif field_number == 69
            unique_indices = PB.decode(d, Bool)
        elseif field_number == 70
            rng_algorithm = PB.decode(d, RandomAlgorithm.T)
        elseif field_number == 72
            comparison_type = PB.decode(d, String)
        elseif field_number == 73
            is_cross_program_prefetch = PB.decode(d, Bool)
        elseif field_number == 80
            optional_cross_program_prefetch_index = OneOf(:cross_program_prefetch_index, PB.decode(d, Int32))
        elseif field_number == 75
            padding_type = PB.decode(d, PaddingType.T)
        elseif field_number == 77
            custom_call_api_version = PB.decode(d, CustomCallApiVersion.T)
        elseif field_number == 79
            async_execution_thread = PB.decode(d, String)
        elseif field_number == 81
            k = PB.decode(d, Int64)
        elseif field_number == 85
            largest = PB.decode(d, Bool)
        elseif field_number == 82
            PB.decode!(d, statistics_viz)
        elseif field_number == 87
            PB.decode!(d, collective_device_list)
        elseif field_number == 88
            PB.decode!(d, original_value)
        elseif field_number == 89
            is_composite = PB.decode(d, Bool)
        elseif field_number == 91
            PB.decode!(d, result_accuracy)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloInstructionProto(name, opcode, shape[], metadata[], literal[], parameter_number, fusion_kind, tuple_index, dimensions[], window[], convolution_dimension_numbers[], feature_group_count, batch_group_count, slice_dimensions[], exponent_bits, mantissa_bits, dynamic_slice_sizes[], padding_config[], outfeed_config, distribution, epsilon, feature_index, channel_id, infeed_config, custom_call_target, outfeed_shape[], dot_dimension_numbers[], ragged_dot_dimension_numbers[], fft_type, fft_length[], comparison_direction, gather_dimension_numbers[], gather_slice_sizes[], id, operand_ids[], control_predecessor_ids[], called_computation_ids[], sharding[], backend_config, replica_groups[], all_reduce_id, use_global_device_ids, is_host_transfer, is_stable, scatter_dimension_numbers[], precision_config[], source_target_pairs[], domain_entry_sharding[], domain_exit_sharding[], constrain_layout, operand_shapes_with_layout[], triangular_solve_options[], cholesky_options[], parameter_replication[], custom_call_has_side_effect, output_operand_aliasing[], custom_call_schedule, delta, indices_are_sorted, frontend_attributes[], unique_indices, rng_algorithm, comparison_type, is_cross_program_prefetch, optional_cross_program_prefetch_index, padding_type, custom_call_api_version, async_execution_thread, k, largest, statistics_viz[], collective_device_list[], original_value[], is_composite, result_accuracy[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloInstructionProto)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    !isempty(x.opcode) && PB.encode(e, 2, x.opcode)
    !isnothing(x.shape) && PB.encode(e, 3, x.shape)
    !isnothing(x.metadata) && PB.encode(e, 7, x.metadata)
    !isnothing(x.literal) && PB.encode(e, 8, x.literal)
    x.parameter_number != zero(Int64) && PB.encode(e, 9, x.parameter_number)
    !isempty(x.fusion_kind) && PB.encode(e, 11, x.fusion_kind)
    x.tuple_index != zero(Int64) && PB.encode(e, 13, x.tuple_index)
    !isempty(x.dimensions) && PB.encode(e, 14, x.dimensions)
    !isnothing(x.window) && PB.encode(e, 15, x.window)
    !isnothing(x.convolution_dimension_numbers) && PB.encode(e, 16, x.convolution_dimension_numbers)
    x.feature_group_count != zero(Int64) && PB.encode(e, 50, x.feature_group_count)
    x.batch_group_count != zero(Int64) && PB.encode(e, 58, x.batch_group_count)
    !isempty(x.slice_dimensions) && PB.encode(e, 17, x.slice_dimensions)
    x.exponent_bits != zero(Int32) && PB.encode(e, 18, x.exponent_bits)
    x.mantissa_bits != zero(Int32) && PB.encode(e, 19, x.mantissa_bits)
    !isempty(x.dynamic_slice_sizes) && PB.encode(e, 20, x.dynamic_slice_sizes)
    !isnothing(x.padding_config) && PB.encode(e, 21, x.padding_config)
    !isempty(x.outfeed_config) && PB.encode(e, 22, x.outfeed_config)
    x.distribution != RandomDistribution.RNG_INVALID && PB.encode(e, 23, x.distribution)
    x.epsilon !== zero(Float32) && PB.encode(e, 24, x.epsilon)
    x.feature_index != zero(Int64) && PB.encode(e, 25, x.feature_index)
    x.channel_id != zero(Int64) && PB.encode(e, 26, x.channel_id)
    !isempty(x.infeed_config) && PB.encode(e, 27, x.infeed_config)
    !isempty(x.custom_call_target) && PB.encode(e, 28, x.custom_call_target)
    !isnothing(x.outfeed_shape) && PB.encode(e, 29, x.outfeed_shape)
    !isnothing(x.dot_dimension_numbers) && PB.encode(e, 30, x.dot_dimension_numbers)
    !isnothing(x.ragged_dot_dimension_numbers) && PB.encode(e, 90, x.ragged_dot_dimension_numbers)
    x.fft_type != FftType.FFT && PB.encode(e, 31, x.fft_type)
    !isempty(x.fft_length) && PB.encode(e, 32, x.fft_length)
    !isempty(x.comparison_direction) && PB.encode(e, 63, x.comparison_direction)
    !isnothing(x.gather_dimension_numbers) && PB.encode(e, 33, x.gather_dimension_numbers)
    !isempty(x.gather_slice_sizes) && PB.encode(e, 34, x.gather_slice_sizes)
    x.id != zero(Int64) && PB.encode(e, 35, x.id)
    !isempty(x.operand_ids) && PB.encode(e, 36, x.operand_ids)
    !isempty(x.control_predecessor_ids) && PB.encode(e, 37, x.control_predecessor_ids)
    !isempty(x.called_computation_ids) && PB.encode(e, 38, x.called_computation_ids)
    !isnothing(x.sharding) && PB.encode(e, 40, x.sharding)
    !isempty(x.backend_config) && PB.encode(e, 43, x.backend_config)
    !isempty(x.replica_groups) && PB.encode(e, 49, x.replica_groups)
    x.all_reduce_id != zero(Int64) && PB.encode(e, 45, x.all_reduce_id)
    x.use_global_device_ids != false && PB.encode(e, 71, x.use_global_device_ids)
    x.is_host_transfer != false && PB.encode(e, 47, x.is_host_transfer)
    x.is_stable != false && PB.encode(e, 60, x.is_stable)
    !isnothing(x.scatter_dimension_numbers) && PB.encode(e, 48, x.scatter_dimension_numbers)
    !isnothing(x.precision_config) && PB.encode(e, 51, x.precision_config)
    !isempty(x.source_target_pairs) && PB.encode(e, 52, x.source_target_pairs)
    !isnothing(x.domain_entry_sharding) && PB.encode(e, 54, x.domain_entry_sharding)
    !isnothing(x.domain_exit_sharding) && PB.encode(e, 55, x.domain_exit_sharding)
    x.constrain_layout != false && PB.encode(e, 56, x.constrain_layout)
    !isempty(x.operand_shapes_with_layout) && PB.encode(e, 57, x.operand_shapes_with_layout)
    !isnothing(x.triangular_solve_options) && PB.encode(e, 59, x.triangular_solve_options)
    !isnothing(x.cholesky_options) && PB.encode(e, 62, x.cholesky_options)
    !isnothing(x.parameter_replication) && PB.encode(e, 61, x.parameter_replication)
    x.custom_call_has_side_effect != false && PB.encode(e, 65, x.custom_call_has_side_effect)
    !isempty(x.output_operand_aliasing) && PB.encode(e, 74, x.output_operand_aliasing)
    x.custom_call_schedule != CustomCallSchedule.SCHEDULE_NONE && PB.encode(e, 76, x.custom_call_schedule)
    x.delta != zero(Int64) && PB.encode(e, 66, x.delta)
    x.indices_are_sorted != false && PB.encode(e, 67, x.indices_are_sorted)
    !isnothing(x.frontend_attributes) && PB.encode(e, 68, x.frontend_attributes)
    x.unique_indices != false && PB.encode(e, 69, x.unique_indices)
    x.rng_algorithm != RandomAlgorithm.RNG_DEFAULT && PB.encode(e, 70, x.rng_algorithm)
    !isempty(x.comparison_type) && PB.encode(e, 72, x.comparison_type)
    x.is_cross_program_prefetch != false && PB.encode(e, 73, x.is_cross_program_prefetch)
    if isnothing(x.optional_cross_program_prefetch_index);
    elseif x.optional_cross_program_prefetch_index.name === :cross_program_prefetch_index
        PB.encode(e, 80, x.optional_cross_program_prefetch_index[]::Int32)
    end
    x.padding_type != PaddingType.PADDING_INVALID && PB.encode(e, 75, x.padding_type)
    x.custom_call_api_version != CustomCallApiVersion.API_VERSION_UNSPECIFIED && PB.encode(e, 77, x.custom_call_api_version)
    !isempty(x.async_execution_thread) && PB.encode(e, 79, x.async_execution_thread)
    x.k != zero(Int64) && PB.encode(e, 81, x.k)
    x.largest != false && PB.encode(e, 85, x.largest)
    !isnothing(x.statistics_viz) && PB.encode(e, 82, x.statistics_viz)
    !isnothing(x.collective_device_list) && PB.encode(e, 87, x.collective_device_list)
    !isnothing(x.original_value) && PB.encode(e, 88, x.original_value)
    x.is_composite != false && PB.encode(e, 89, x.is_composite)
    !isnothing(x.result_accuracy) && PB.encode(e, 91, x.result_accuracy)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloInstructionProto)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    !isempty(x.opcode) && (encoded_size += PB._encoded_size(x.opcode, 2))
    !isnothing(x.shape) && (encoded_size += PB._encoded_size(x.shape, 3))
    !isnothing(x.metadata) && (encoded_size += PB._encoded_size(x.metadata, 7))
    !isnothing(x.literal) && (encoded_size += PB._encoded_size(x.literal, 8))
    x.parameter_number != zero(Int64) && (encoded_size += PB._encoded_size(x.parameter_number, 9))
    !isempty(x.fusion_kind) && (encoded_size += PB._encoded_size(x.fusion_kind, 11))
    x.tuple_index != zero(Int64) && (encoded_size += PB._encoded_size(x.tuple_index, 13))
    !isempty(x.dimensions) && (encoded_size += PB._encoded_size(x.dimensions, 14))
    !isnothing(x.window) && (encoded_size += PB._encoded_size(x.window, 15))
    !isnothing(x.convolution_dimension_numbers) && (encoded_size += PB._encoded_size(x.convolution_dimension_numbers, 16))
    x.feature_group_count != zero(Int64) && (encoded_size += PB._encoded_size(x.feature_group_count, 50))
    x.batch_group_count != zero(Int64) && (encoded_size += PB._encoded_size(x.batch_group_count, 58))
    !isempty(x.slice_dimensions) && (encoded_size += PB._encoded_size(x.slice_dimensions, 17))
    x.exponent_bits != zero(Int32) && (encoded_size += PB._encoded_size(x.exponent_bits, 18))
    x.mantissa_bits != zero(Int32) && (encoded_size += PB._encoded_size(x.mantissa_bits, 19))
    !isempty(x.dynamic_slice_sizes) && (encoded_size += PB._encoded_size(x.dynamic_slice_sizes, 20))
    !isnothing(x.padding_config) && (encoded_size += PB._encoded_size(x.padding_config, 21))
    !isempty(x.outfeed_config) && (encoded_size += PB._encoded_size(x.outfeed_config, 22))
    x.distribution != RandomDistribution.RNG_INVALID && (encoded_size += PB._encoded_size(x.distribution, 23))
    x.epsilon !== zero(Float32) && (encoded_size += PB._encoded_size(x.epsilon, 24))
    x.feature_index != zero(Int64) && (encoded_size += PB._encoded_size(x.feature_index, 25))
    x.channel_id != zero(Int64) && (encoded_size += PB._encoded_size(x.channel_id, 26))
    !isempty(x.infeed_config) && (encoded_size += PB._encoded_size(x.infeed_config, 27))
    !isempty(x.custom_call_target) && (encoded_size += PB._encoded_size(x.custom_call_target, 28))
    !isnothing(x.outfeed_shape) && (encoded_size += PB._encoded_size(x.outfeed_shape, 29))
    !isnothing(x.dot_dimension_numbers) && (encoded_size += PB._encoded_size(x.dot_dimension_numbers, 30))
    !isnothing(x.ragged_dot_dimension_numbers) && (encoded_size += PB._encoded_size(x.ragged_dot_dimension_numbers, 90))
    x.fft_type != FftType.FFT && (encoded_size += PB._encoded_size(x.fft_type, 31))
    !isempty(x.fft_length) && (encoded_size += PB._encoded_size(x.fft_length, 32))
    !isempty(x.comparison_direction) && (encoded_size += PB._encoded_size(x.comparison_direction, 63))
    !isnothing(x.gather_dimension_numbers) && (encoded_size += PB._encoded_size(x.gather_dimension_numbers, 33))
    !isempty(x.gather_slice_sizes) && (encoded_size += PB._encoded_size(x.gather_slice_sizes, 34))
    x.id != zero(Int64) && (encoded_size += PB._encoded_size(x.id, 35))
    !isempty(x.operand_ids) && (encoded_size += PB._encoded_size(x.operand_ids, 36))
    !isempty(x.control_predecessor_ids) && (encoded_size += PB._encoded_size(x.control_predecessor_ids, 37))
    !isempty(x.called_computation_ids) && (encoded_size += PB._encoded_size(x.called_computation_ids, 38))
    !isnothing(x.sharding) && (encoded_size += PB._encoded_size(x.sharding, 40))
    !isempty(x.backend_config) && (encoded_size += PB._encoded_size(x.backend_config, 43))
    !isempty(x.replica_groups) && (encoded_size += PB._encoded_size(x.replica_groups, 49))
    x.all_reduce_id != zero(Int64) && (encoded_size += PB._encoded_size(x.all_reduce_id, 45))
    x.use_global_device_ids != false && (encoded_size += PB._encoded_size(x.use_global_device_ids, 71))
    x.is_host_transfer != false && (encoded_size += PB._encoded_size(x.is_host_transfer, 47))
    x.is_stable != false && (encoded_size += PB._encoded_size(x.is_stable, 60))
    !isnothing(x.scatter_dimension_numbers) && (encoded_size += PB._encoded_size(x.scatter_dimension_numbers, 48))
    !isnothing(x.precision_config) && (encoded_size += PB._encoded_size(x.precision_config, 51))
    !isempty(x.source_target_pairs) && (encoded_size += PB._encoded_size(x.source_target_pairs, 52))
    !isnothing(x.domain_entry_sharding) && (encoded_size += PB._encoded_size(x.domain_entry_sharding, 54))
    !isnothing(x.domain_exit_sharding) && (encoded_size += PB._encoded_size(x.domain_exit_sharding, 55))
    x.constrain_layout != false && (encoded_size += PB._encoded_size(x.constrain_layout, 56))
    !isempty(x.operand_shapes_with_layout) && (encoded_size += PB._encoded_size(x.operand_shapes_with_layout, 57))
    !isnothing(x.triangular_solve_options) && (encoded_size += PB._encoded_size(x.triangular_solve_options, 59))
    !isnothing(x.cholesky_options) && (encoded_size += PB._encoded_size(x.cholesky_options, 62))
    !isnothing(x.parameter_replication) && (encoded_size += PB._encoded_size(x.parameter_replication, 61))
    x.custom_call_has_side_effect != false && (encoded_size += PB._encoded_size(x.custom_call_has_side_effect, 65))
    !isempty(x.output_operand_aliasing) && (encoded_size += PB._encoded_size(x.output_operand_aliasing, 74))
    x.custom_call_schedule != CustomCallSchedule.SCHEDULE_NONE && (encoded_size += PB._encoded_size(x.custom_call_schedule, 76))
    x.delta != zero(Int64) && (encoded_size += PB._encoded_size(x.delta, 66))
    x.indices_are_sorted != false && (encoded_size += PB._encoded_size(x.indices_are_sorted, 67))
    !isnothing(x.frontend_attributes) && (encoded_size += PB._encoded_size(x.frontend_attributes, 68))
    x.unique_indices != false && (encoded_size += PB._encoded_size(x.unique_indices, 69))
    x.rng_algorithm != RandomAlgorithm.RNG_DEFAULT && (encoded_size += PB._encoded_size(x.rng_algorithm, 70))
    !isempty(x.comparison_type) && (encoded_size += PB._encoded_size(x.comparison_type, 72))
    x.is_cross_program_prefetch != false && (encoded_size += PB._encoded_size(x.is_cross_program_prefetch, 73))
    if isnothing(x.optional_cross_program_prefetch_index);
    elseif x.optional_cross_program_prefetch_index.name === :cross_program_prefetch_index
        encoded_size += PB._encoded_size(x.optional_cross_program_prefetch_index[]::Int32, 80)
    end
    x.padding_type != PaddingType.PADDING_INVALID && (encoded_size += PB._encoded_size(x.padding_type, 75))
    x.custom_call_api_version != CustomCallApiVersion.API_VERSION_UNSPECIFIED && (encoded_size += PB._encoded_size(x.custom_call_api_version, 77))
    !isempty(x.async_execution_thread) && (encoded_size += PB._encoded_size(x.async_execution_thread, 79))
    x.k != zero(Int64) && (encoded_size += PB._encoded_size(x.k, 81))
    x.largest != false && (encoded_size += PB._encoded_size(x.largest, 85))
    !isnothing(x.statistics_viz) && (encoded_size += PB._encoded_size(x.statistics_viz, 82))
    !isnothing(x.collective_device_list) && (encoded_size += PB._encoded_size(x.collective_device_list, 87))
    !isnothing(x.original_value) && (encoded_size += PB._encoded_size(x.original_value, 88))
    x.is_composite != false && (encoded_size += PB._encoded_size(x.is_composite, 89))
    !isnothing(x.result_accuracy) && (encoded_size += PB._encoded_size(x.result_accuracy, 91))
    return encoded_size
end

struct var"HeapSimulatorTrace.Event"
    kind::var"HeapSimulatorTrace.Event.Kind".T
    buffer_id::Int64
    computation_name::String
    instruction_name::String
    share_with_canonical_id::Int64
end
var"HeapSimulatorTrace.Event"(;kind = var"HeapSimulatorTrace.Event.Kind".ALLOC, buffer_id = zero(Int64), computation_name = "", instruction_name = "", share_with_canonical_id = zero(Int64)) = var"HeapSimulatorTrace.Event"(kind, buffer_id, computation_name, instruction_name, share_with_canonical_id)
PB.default_values(::Type{var"HeapSimulatorTrace.Event"}) = (;kind = var"HeapSimulatorTrace.Event.Kind".ALLOC, buffer_id = zero(Int64), computation_name = "", instruction_name = "", share_with_canonical_id = zero(Int64))
PB.field_numbers(::Type{var"HeapSimulatorTrace.Event"}) = (;kind = 1, buffer_id = 2, computation_name = 3, instruction_name = 4, share_with_canonical_id = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"HeapSimulatorTrace.Event"})
    kind = var"HeapSimulatorTrace.Event.Kind".ALLOC
    buffer_id = zero(Int64)
    computation_name = ""
    instruction_name = ""
    share_with_canonical_id = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            kind = PB.decode(d, var"HeapSimulatorTrace.Event.Kind".T)
        elseif field_number == 2
            buffer_id = PB.decode(d, Int64)
        elseif field_number == 3
            computation_name = PB.decode(d, String)
        elseif field_number == 4
            instruction_name = PB.decode(d, String)
        elseif field_number == 5
            share_with_canonical_id = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"HeapSimulatorTrace.Event"(kind, buffer_id, computation_name, instruction_name, share_with_canonical_id)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"HeapSimulatorTrace.Event")
    initpos = position(e.io)
    x.kind != var"HeapSimulatorTrace.Event.Kind".ALLOC && PB.encode(e, 1, x.kind)
    x.buffer_id != zero(Int64) && PB.encode(e, 2, x.buffer_id)
    !isempty(x.computation_name) && PB.encode(e, 3, x.computation_name)
    !isempty(x.instruction_name) && PB.encode(e, 4, x.instruction_name)
    x.share_with_canonical_id != zero(Int64) && PB.encode(e, 5, x.share_with_canonical_id)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"HeapSimulatorTrace.Event")
    encoded_size = 0
    x.kind != var"HeapSimulatorTrace.Event.Kind".ALLOC && (encoded_size += PB._encoded_size(x.kind, 1))
    x.buffer_id != zero(Int64) && (encoded_size += PB._encoded_size(x.buffer_id, 2))
    !isempty(x.computation_name) && (encoded_size += PB._encoded_size(x.computation_name, 3))
    !isempty(x.instruction_name) && (encoded_size += PB._encoded_size(x.instruction_name, 4))
    x.share_with_canonical_id != zero(Int64) && (encoded_size += PB._encoded_size(x.share_with_canonical_id, 5))
    return encoded_size
end

struct HloInputOutputAliasProto
    entries::Vector{var"HloInputOutputAliasProto.AliasEntryProto"}
end
HloInputOutputAliasProto(;entries = Vector{var"HloInputOutputAliasProto.AliasEntryProto"}()) = HloInputOutputAliasProto(entries)
PB.default_values(::Type{HloInputOutputAliasProto}) = (;entries = Vector{var"HloInputOutputAliasProto.AliasEntryProto"}())
PB.field_numbers(::Type{HloInputOutputAliasProto}) = (;entries = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloInputOutputAliasProto})
    entries = PB.BufferedVector{var"HloInputOutputAliasProto.AliasEntryProto"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, entries)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloInputOutputAliasProto(entries[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloInputOutputAliasProto)
    initpos = position(e.io)
    !isempty(x.entries) && PB.encode(e, 1, x.entries)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloInputOutputAliasProto)
    encoded_size = 0
    !isempty(x.entries) && (encoded_size += PB._encoded_size(x.entries, 1))
    return encoded_size
end

struct HloComputationProto
    name::String
    instructions::Vector{HloInstructionProto}
    program_shape::Union{Nothing,ProgramShapeProto}
    id::Int64
    root_id::Int64
    is_fusion_computation::Bool
    execution_thread::String
end
HloComputationProto(;name = "", instructions = Vector{HloInstructionProto}(), program_shape = nothing, id = zero(Int64), root_id = zero(Int64), is_fusion_computation = false, execution_thread = "") = HloComputationProto(name, instructions, program_shape, id, root_id, is_fusion_computation, execution_thread)
PB.reserved_fields(::Type{HloComputationProto}) = (names = ["root_name"], numbers = Union{Int,UnitRange{Int}}[3])
PB.default_values(::Type{HloComputationProto}) = (;name = "", instructions = Vector{HloInstructionProto}(), program_shape = nothing, id = zero(Int64), root_id = zero(Int64), is_fusion_computation = false, execution_thread = "")
PB.field_numbers(::Type{HloComputationProto}) = (;name = 1, instructions = 2, program_shape = 4, id = 5, root_id = 6, is_fusion_computation = 7, execution_thread = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloComputationProto})
    name = ""
    instructions = PB.BufferedVector{HloInstructionProto}()
    program_shape = Ref{Union{Nothing,ProgramShapeProto}}(nothing)
    id = zero(Int64)
    root_id = zero(Int64)
    is_fusion_computation = false
    execution_thread = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            PB.decode!(d, instructions)
        elseif field_number == 4
            PB.decode!(d, program_shape)
        elseif field_number == 5
            id = PB.decode(d, Int64)
        elseif field_number == 6
            root_id = PB.decode(d, Int64)
        elseif field_number == 7
            is_fusion_computation = PB.decode(d, Bool)
        elseif field_number == 8
            execution_thread = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloComputationProto(name, instructions[], program_shape[], id, root_id, is_fusion_computation, execution_thread)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloComputationProto)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    !isempty(x.instructions) && PB.encode(e, 2, x.instructions)
    !isnothing(x.program_shape) && PB.encode(e, 4, x.program_shape)
    x.id != zero(Int64) && PB.encode(e, 5, x.id)
    x.root_id != zero(Int64) && PB.encode(e, 6, x.root_id)
    x.is_fusion_computation != false && PB.encode(e, 7, x.is_fusion_computation)
    !isempty(x.execution_thread) && PB.encode(e, 8, x.execution_thread)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloComputationProto)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    !isempty(x.instructions) && (encoded_size += PB._encoded_size(x.instructions, 2))
    !isnothing(x.program_shape) && (encoded_size += PB._encoded_size(x.program_shape, 4))
    x.id != zero(Int64) && (encoded_size += PB._encoded_size(x.id, 5))
    x.root_id != zero(Int64) && (encoded_size += PB._encoded_size(x.root_id, 6))
    x.is_fusion_computation != false && (encoded_size += PB._encoded_size(x.is_fusion_computation, 7))
    !isempty(x.execution_thread) && (encoded_size += PB._encoded_size(x.execution_thread, 8))
    return encoded_size
end

struct HeapSimulatorTrace
    events::Vector{var"HeapSimulatorTrace.Event"}
    whole_module_simulation::Bool
    buffer_allocation_index::Int64
end
HeapSimulatorTrace(;events = Vector{var"HeapSimulatorTrace.Event"}(), whole_module_simulation = false, buffer_allocation_index = zero(Int64)) = HeapSimulatorTrace(events, whole_module_simulation, buffer_allocation_index)
PB.default_values(::Type{HeapSimulatorTrace}) = (;events = Vector{var"HeapSimulatorTrace.Event"}(), whole_module_simulation = false, buffer_allocation_index = zero(Int64))
PB.field_numbers(::Type{HeapSimulatorTrace}) = (;events = 1, whole_module_simulation = 2, buffer_allocation_index = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HeapSimulatorTrace})
    events = PB.BufferedVector{var"HeapSimulatorTrace.Event"}()
    whole_module_simulation = false
    buffer_allocation_index = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, events)
        elseif field_number == 2
            whole_module_simulation = PB.decode(d, Bool)
        elseif field_number == 3
            buffer_allocation_index = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return HeapSimulatorTrace(events[], whole_module_simulation, buffer_allocation_index)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HeapSimulatorTrace)
    initpos = position(e.io)
    !isempty(x.events) && PB.encode(e, 1, x.events)
    x.whole_module_simulation != false && PB.encode(e, 2, x.whole_module_simulation)
    x.buffer_allocation_index != zero(Int64) && PB.encode(e, 3, x.buffer_allocation_index)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HeapSimulatorTrace)
    encoded_size = 0
    !isempty(x.events) && (encoded_size += PB._encoded_size(x.events, 1))
    x.whole_module_simulation != false && (encoded_size += PB._encoded_size(x.whole_module_simulation, 2))
    x.buffer_allocation_index != zero(Int64) && (encoded_size += PB._encoded_size(x.buffer_allocation_index, 3))
    return encoded_size
end

struct BufferAssignmentProto
    logical_buffers::Vector{LogicalBufferProto}
    buffer_aliases::Vector{var"BufferAssignmentProto.BufferAlias"}
    buffer_allocations::Vector{BufferAllocationProto}
    heap_simulator_traces::Vector{HeapSimulatorTrace}
end
BufferAssignmentProto(;logical_buffers = Vector{LogicalBufferProto}(), buffer_aliases = Vector{var"BufferAssignmentProto.BufferAlias"}(), buffer_allocations = Vector{BufferAllocationProto}(), heap_simulator_traces = Vector{HeapSimulatorTrace}()) = BufferAssignmentProto(logical_buffers, buffer_aliases, buffer_allocations, heap_simulator_traces)
PB.default_values(::Type{BufferAssignmentProto}) = (;logical_buffers = Vector{LogicalBufferProto}(), buffer_aliases = Vector{var"BufferAssignmentProto.BufferAlias"}(), buffer_allocations = Vector{BufferAllocationProto}(), heap_simulator_traces = Vector{HeapSimulatorTrace}())
PB.field_numbers(::Type{BufferAssignmentProto}) = (;logical_buffers = 1, buffer_aliases = 2, buffer_allocations = 3, heap_simulator_traces = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:BufferAssignmentProto})
    logical_buffers = PB.BufferedVector{LogicalBufferProto}()
    buffer_aliases = PB.BufferedVector{var"BufferAssignmentProto.BufferAlias"}()
    buffer_allocations = PB.BufferedVector{BufferAllocationProto}()
    heap_simulator_traces = PB.BufferedVector{HeapSimulatorTrace}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, logical_buffers)
        elseif field_number == 2
            PB.decode!(d, buffer_aliases)
        elseif field_number == 3
            PB.decode!(d, buffer_allocations)
        elseif field_number == 4
            PB.decode!(d, heap_simulator_traces)
        else
            Base.skip(d, wire_type)
        end
    end
    return BufferAssignmentProto(logical_buffers[], buffer_aliases[], buffer_allocations[], heap_simulator_traces[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::BufferAssignmentProto)
    initpos = position(e.io)
    !isempty(x.logical_buffers) && PB.encode(e, 1, x.logical_buffers)
    !isempty(x.buffer_aliases) && PB.encode(e, 2, x.buffer_aliases)
    !isempty(x.buffer_allocations) && PB.encode(e, 3, x.buffer_allocations)
    !isempty(x.heap_simulator_traces) && PB.encode(e, 4, x.heap_simulator_traces)
    return position(e.io) - initpos
end
function PB._encoded_size(x::BufferAssignmentProto)
    encoded_size = 0
    !isempty(x.logical_buffers) && (encoded_size += PB._encoded_size(x.logical_buffers, 1))
    !isempty(x.buffer_aliases) && (encoded_size += PB._encoded_size(x.buffer_aliases, 2))
    !isempty(x.buffer_allocations) && (encoded_size += PB._encoded_size(x.buffer_allocations, 3))
    !isempty(x.heap_simulator_traces) && (encoded_size += PB._encoded_size(x.heap_simulator_traces, 4))
    return encoded_size
end

# Stub definitions for cyclic types
struct var"##Stub#HloModuleGroupProto"{T1<:var"##Abstract#HloModuleProto"} <: var"##Abstract#HloModuleGroupProto"
    name::String
    hlo_modules::Vector{T1}
end

struct var"##Stub#HloModuleProto"{T1<:var"##Abstract#OriginalValueRecoveryTableProto"} <: var"##Abstract#HloModuleProto"
    name::String
    entry_computation_name::String
    entry_computation_id::Int64
    computations::Vector{HloComputationProto}
    host_program_shape::Union{Nothing,ProgramShapeProto}
    id::Int64
    schedule::Union{Nothing,HloScheduleProto}
    input_output_alias::Union{Nothing,HloInputOutputAliasProto}
    buffer_donor::Union{Nothing,HloBufferDonorProto}
    cross_program_prefetches::Vector{CrossProgramPrefetch}
    is_dynamic::Bool
    spmd_output_sharding::Union{Nothing,OpSharding}
    spmd_parameters_shardings::Vector{OpSharding}
    use_auto_spmd_partitioning::Bool
    profile_info::Vector{var"HloModuleProto.ProfileInfo"}
    device_assignment::Union{Nothing,DeviceAssignmentProto}
    stack_frame_index::Union{Nothing,StackFrameIndexProto}
    frontend_attributes::Union{Nothing,FrontendAttributes}
    original_value_recovery_table::Union{Nothing,T1}
end

struct var"##Stub#HloProto"{T1<:var"##Abstract#OriginalValueRecoveryTableProto"} <: var"##Abstract#HloProto"
    hlo_module::Union{Nothing,var"##Stub#HloModuleProto"{T1}}
    buffer_assignment::Union{Nothing,BufferAssignmentProto}
end

struct var"##Stub#HloSnapshot"{T1<:var"##Abstract#OriginalValueRecoveryTableProto"} <: var"##Abstract#HloSnapshot"
    hlo::Union{Nothing,var"##Stub#HloProto"{T1}}
    arguments::Vector{LiteralProto}
    result::Union{Nothing,LiteralProto}
    execution_platform::String
end

struct var"##Stub#HloUnoptimizedSnapshot"{T1<:var"##Abstract#OriginalValueRecoveryTableProto"} <: var"##Abstract#HloUnoptimizedSnapshot"
    hlo_module::Union{Nothing,var"##Stub#HloModuleProto"{T1}}
    partitions::Vector{HloInputs}
    version::Int32
end

struct var"##Stub#OriginalValueRecoveryTableProto"{T1<:var"##Abstract#OriginalValueRecoveryTableProto.Entry"} <: var"##Abstract#OriginalValueRecoveryTableProto"
    entries::Vector{T1}
end

struct var"##Stub#OriginalValueRecoveryTableProto.Entry" <: var"##Abstract#OriginalValueRecoveryTableProto.Entry"
    old_original_array::Union{Nothing,OriginalArrayProto}
    new_original_array::Union{Nothing,OriginalArrayProto}
    recovery_module::Union{Nothing,var"##Stub#HloModuleProto"{var"##Stub#OriginalValueRecoveryTableProto"{var"##Stub#OriginalValueRecoveryTableProto.Entry"}}}
end

const HloModuleGroupProto = var"##Stub#HloModuleGroupProto"{var"##Stub#HloModuleProto"{var"##Stub#OriginalValueRecoveryTableProto"{var"##Stub#OriginalValueRecoveryTableProto.Entry"}}}
HloModuleGroupProto(;name = "", hlo_modules = Vector{HloModuleProto}()) = HloModuleGroupProto(name, hlo_modules)
PB.default_values(::Type{HloModuleGroupProto}) = (;name = "", hlo_modules = Vector{HloModuleProto}())
PB.field_numbers(::Type{HloModuleGroupProto}) = (;name = 1, hlo_modules = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloModuleGroupProto})
    name = ""
    hlo_modules = PB.BufferedVector{HloModuleProto}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            PB.decode!(d, hlo_modules)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloModuleGroupProto(name, hlo_modules[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloModuleGroupProto)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    !isempty(x.hlo_modules) && PB.encode(e, 2, x.hlo_modules)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloModuleGroupProto)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    !isempty(x.hlo_modules) && (encoded_size += PB._encoded_size(x.hlo_modules, 2))
    return encoded_size
end

const HloModuleProto = var"##Stub#HloModuleProto"{var"##Stub#OriginalValueRecoveryTableProto"{var"##Stub#OriginalValueRecoveryTableProto.Entry"}}
HloModuleProto(;name = "", entry_computation_name = "", entry_computation_id = zero(Int64), computations = Vector{HloComputationProto}(), host_program_shape = nothing, id = zero(Int64), schedule = nothing, input_output_alias = nothing, buffer_donor = nothing, cross_program_prefetches = Vector{CrossProgramPrefetch}(), is_dynamic = false, spmd_output_sharding = nothing, spmd_parameters_shardings = Vector{OpSharding}(), use_auto_spmd_partitioning = false, profile_info = Vector{var"HloModuleProto.ProfileInfo"}(), device_assignment = nothing, stack_frame_index = nothing, frontend_attributes = nothing, original_value_recovery_table = nothing) = HloModuleProto(name, entry_computation_name, entry_computation_id, computations, host_program_shape, id, schedule, input_output_alias, buffer_donor, cross_program_prefetches, is_dynamic, spmd_output_sharding, spmd_parameters_shardings, use_auto_spmd_partitioning, profile_info, device_assignment, stack_frame_index, frontend_attributes, original_value_recovery_table)
PB.reserved_fields(::Type{HloModuleProto}) = (names = ["dynamic_parameter_binding"], numbers = Union{Int,UnitRange{Int}}[9])
PB.default_values(::Type{HloModuleProto}) = (;name = "", entry_computation_name = "", entry_computation_id = zero(Int64), computations = Vector{HloComputationProto}(), host_program_shape = nothing, id = zero(Int64), schedule = nothing, input_output_alias = nothing, buffer_donor = nothing, cross_program_prefetches = Vector{CrossProgramPrefetch}(), is_dynamic = false, spmd_output_sharding = nothing, spmd_parameters_shardings = Vector{OpSharding}(), use_auto_spmd_partitioning = false, profile_info = Vector{var"HloModuleProto.ProfileInfo"}(), device_assignment = nothing, stack_frame_index = nothing, frontend_attributes = nothing, original_value_recovery_table = nothing)
PB.field_numbers(::Type{HloModuleProto}) = (;name = 1, entry_computation_name = 2, entry_computation_id = 6, computations = 3, host_program_shape = 4, id = 5, schedule = 7, input_output_alias = 8, buffer_donor = 18, cross_program_prefetches = 10, is_dynamic = 11, spmd_output_sharding = 12, spmd_parameters_shardings = 14, use_auto_spmd_partitioning = 16, profile_info = 13, device_assignment = 15, stack_frame_index = 17, frontend_attributes = 19, original_value_recovery_table = 20)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloModuleProto})
    name = ""
    entry_computation_name = ""
    entry_computation_id = zero(Int64)
    computations = PB.BufferedVector{HloComputationProto}()
    host_program_shape = Ref{Union{Nothing,ProgramShapeProto}}(nothing)
    id = zero(Int64)
    schedule = Ref{Union{Nothing,HloScheduleProto}}(nothing)
    input_output_alias = Ref{Union{Nothing,HloInputOutputAliasProto}}(nothing)
    buffer_donor = Ref{Union{Nothing,HloBufferDonorProto}}(nothing)
    cross_program_prefetches = PB.BufferedVector{CrossProgramPrefetch}()
    is_dynamic = false
    spmd_output_sharding = Ref{Union{Nothing,OpSharding}}(nothing)
    spmd_parameters_shardings = PB.BufferedVector{OpSharding}()
    use_auto_spmd_partitioning = false
    profile_info = PB.BufferedVector{var"HloModuleProto.ProfileInfo"}()
    device_assignment = Ref{Union{Nothing,DeviceAssignmentProto}}(nothing)
    stack_frame_index = Ref{Union{Nothing,StackFrameIndexProto}}(nothing)
    frontend_attributes = Ref{Union{Nothing,FrontendAttributes}}(nothing)
    original_value_recovery_table = Ref{Union{Nothing,OriginalValueRecoveryTableProto}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            entry_computation_name = PB.decode(d, String)
        elseif field_number == 6
            entry_computation_id = PB.decode(d, Int64)
        elseif field_number == 3
            PB.decode!(d, computations)
        elseif field_number == 4
            PB.decode!(d, host_program_shape)
        elseif field_number == 5
            id = PB.decode(d, Int64)
        elseif field_number == 7
            PB.decode!(d, schedule)
        elseif field_number == 8
            PB.decode!(d, input_output_alias)
        elseif field_number == 18
            PB.decode!(d, buffer_donor)
        elseif field_number == 10
            PB.decode!(d, cross_program_prefetches)
        elseif field_number == 11
            is_dynamic = PB.decode(d, Bool)
        elseif field_number == 12
            PB.decode!(d, spmd_output_sharding)
        elseif field_number == 14
            PB.decode!(d, spmd_parameters_shardings)
        elseif field_number == 16
            use_auto_spmd_partitioning = PB.decode(d, Bool)
        elseif field_number == 13
            PB.decode!(d, profile_info)
        elseif field_number == 15
            PB.decode!(d, device_assignment)
        elseif field_number == 17
            PB.decode!(d, stack_frame_index)
        elseif field_number == 19
            PB.decode!(d, frontend_attributes)
        elseif field_number == 20
            PB.decode!(d, original_value_recovery_table)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloModuleProto(name, entry_computation_name, entry_computation_id, computations[], host_program_shape[], id, schedule[], input_output_alias[], buffer_donor[], cross_program_prefetches[], is_dynamic, spmd_output_sharding[], spmd_parameters_shardings[], use_auto_spmd_partitioning, profile_info[], device_assignment[], stack_frame_index[], frontend_attributes[], original_value_recovery_table[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloModuleProto)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    !isempty(x.entry_computation_name) && PB.encode(e, 2, x.entry_computation_name)
    x.entry_computation_id != zero(Int64) && PB.encode(e, 6, x.entry_computation_id)
    !isempty(x.computations) && PB.encode(e, 3, x.computations)
    !isnothing(x.host_program_shape) && PB.encode(e, 4, x.host_program_shape)
    x.id != zero(Int64) && PB.encode(e, 5, x.id)
    !isnothing(x.schedule) && PB.encode(e, 7, x.schedule)
    !isnothing(x.input_output_alias) && PB.encode(e, 8, x.input_output_alias)
    !isnothing(x.buffer_donor) && PB.encode(e, 18, x.buffer_donor)
    !isempty(x.cross_program_prefetches) && PB.encode(e, 10, x.cross_program_prefetches)
    x.is_dynamic != false && PB.encode(e, 11, x.is_dynamic)
    !isnothing(x.spmd_output_sharding) && PB.encode(e, 12, x.spmd_output_sharding)
    !isempty(x.spmd_parameters_shardings) && PB.encode(e, 14, x.spmd_parameters_shardings)
    x.use_auto_spmd_partitioning != false && PB.encode(e, 16, x.use_auto_spmd_partitioning)
    !isempty(x.profile_info) && PB.encode(e, 13, x.profile_info)
    !isnothing(x.device_assignment) && PB.encode(e, 15, x.device_assignment)
    !isnothing(x.stack_frame_index) && PB.encode(e, 17, x.stack_frame_index)
    !isnothing(x.frontend_attributes) && PB.encode(e, 19, x.frontend_attributes)
    !isnothing(x.original_value_recovery_table) && PB.encode(e, 20, x.original_value_recovery_table)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloModuleProto)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    !isempty(x.entry_computation_name) && (encoded_size += PB._encoded_size(x.entry_computation_name, 2))
    x.entry_computation_id != zero(Int64) && (encoded_size += PB._encoded_size(x.entry_computation_id, 6))
    !isempty(x.computations) && (encoded_size += PB._encoded_size(x.computations, 3))
    !isnothing(x.host_program_shape) && (encoded_size += PB._encoded_size(x.host_program_shape, 4))
    x.id != zero(Int64) && (encoded_size += PB._encoded_size(x.id, 5))
    !isnothing(x.schedule) && (encoded_size += PB._encoded_size(x.schedule, 7))
    !isnothing(x.input_output_alias) && (encoded_size += PB._encoded_size(x.input_output_alias, 8))
    !isnothing(x.buffer_donor) && (encoded_size += PB._encoded_size(x.buffer_donor, 18))
    !isempty(x.cross_program_prefetches) && (encoded_size += PB._encoded_size(x.cross_program_prefetches, 10))
    x.is_dynamic != false && (encoded_size += PB._encoded_size(x.is_dynamic, 11))
    !isnothing(x.spmd_output_sharding) && (encoded_size += PB._encoded_size(x.spmd_output_sharding, 12))
    !isempty(x.spmd_parameters_shardings) && (encoded_size += PB._encoded_size(x.spmd_parameters_shardings, 14))
    x.use_auto_spmd_partitioning != false && (encoded_size += PB._encoded_size(x.use_auto_spmd_partitioning, 16))
    !isempty(x.profile_info) && (encoded_size += PB._encoded_size(x.profile_info, 13))
    !isnothing(x.device_assignment) && (encoded_size += PB._encoded_size(x.device_assignment, 15))
    !isnothing(x.stack_frame_index) && (encoded_size += PB._encoded_size(x.stack_frame_index, 17))
    !isnothing(x.frontend_attributes) && (encoded_size += PB._encoded_size(x.frontend_attributes, 19))
    !isnothing(x.original_value_recovery_table) && (encoded_size += PB._encoded_size(x.original_value_recovery_table, 20))
    return encoded_size
end

const HloProto = var"##Stub#HloProto"{var"##Stub#OriginalValueRecoveryTableProto"{var"##Stub#OriginalValueRecoveryTableProto.Entry"}}
HloProto(;hlo_module = nothing, buffer_assignment = nothing) = HloProto(hlo_module, buffer_assignment)
PB.reserved_fields(::Type{HloProto}) = (names = ["hlo_ordering"], numbers = Union{Int,UnitRange{Int}}[2])
PB.default_values(::Type{HloProto}) = (;hlo_module = nothing, buffer_assignment = nothing)
PB.field_numbers(::Type{HloProto}) = (;hlo_module = 1, buffer_assignment = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloProto})
    hlo_module = Ref{Union{Nothing,HloModuleProto}}(nothing)
    buffer_assignment = Ref{Union{Nothing,BufferAssignmentProto}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, hlo_module)
        elseif field_number == 3
            PB.decode!(d, buffer_assignment)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloProto(hlo_module[], buffer_assignment[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloProto)
    initpos = position(e.io)
    !isnothing(x.hlo_module) && PB.encode(e, 1, x.hlo_module)
    !isnothing(x.buffer_assignment) && PB.encode(e, 3, x.buffer_assignment)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloProto)
    encoded_size = 0
    !isnothing(x.hlo_module) && (encoded_size += PB._encoded_size(x.hlo_module, 1))
    !isnothing(x.buffer_assignment) && (encoded_size += PB._encoded_size(x.buffer_assignment, 3))
    return encoded_size
end

const HloSnapshot = var"##Stub#HloSnapshot"{var"##Stub#OriginalValueRecoveryTableProto"{var"##Stub#OriginalValueRecoveryTableProto.Entry"}}
HloSnapshot(;hlo = nothing, arguments = Vector{LiteralProto}(), result = nothing, execution_platform = "") = HloSnapshot(hlo, arguments, result, execution_platform)
PB.default_values(::Type{HloSnapshot}) = (;hlo = nothing, arguments = Vector{LiteralProto}(), result = nothing, execution_platform = "")
PB.field_numbers(::Type{HloSnapshot}) = (;hlo = 1, arguments = 2, result = 3, execution_platform = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloSnapshot})
    hlo = Ref{Union{Nothing,HloProto}}(nothing)
    arguments = PB.BufferedVector{LiteralProto}()
    result = Ref{Union{Nothing,LiteralProto}}(nothing)
    execution_platform = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, hlo)
        elseif field_number == 2
            PB.decode!(d, arguments)
        elseif field_number == 3
            PB.decode!(d, result)
        elseif field_number == 4
            execution_platform = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloSnapshot(hlo[], arguments[], result[], execution_platform)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloSnapshot)
    initpos = position(e.io)
    !isnothing(x.hlo) && PB.encode(e, 1, x.hlo)
    !isempty(x.arguments) && PB.encode(e, 2, x.arguments)
    !isnothing(x.result) && PB.encode(e, 3, x.result)
    !isempty(x.execution_platform) && PB.encode(e, 4, x.execution_platform)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloSnapshot)
    encoded_size = 0
    !isnothing(x.hlo) && (encoded_size += PB._encoded_size(x.hlo, 1))
    !isempty(x.arguments) && (encoded_size += PB._encoded_size(x.arguments, 2))
    !isnothing(x.result) && (encoded_size += PB._encoded_size(x.result, 3))
    !isempty(x.execution_platform) && (encoded_size += PB._encoded_size(x.execution_platform, 4))
    return encoded_size
end

const HloUnoptimizedSnapshot = var"##Stub#HloUnoptimizedSnapshot"{var"##Stub#OriginalValueRecoveryTableProto"{var"##Stub#OriginalValueRecoveryTableProto.Entry"}}
HloUnoptimizedSnapshot(;hlo_module = nothing, partitions = Vector{HloInputs}(), version = zero(Int32)) = HloUnoptimizedSnapshot(hlo_module, partitions, version)
PB.default_values(::Type{HloUnoptimizedSnapshot}) = (;hlo_module = nothing, partitions = Vector{HloInputs}(), version = zero(Int32))
PB.field_numbers(::Type{HloUnoptimizedSnapshot}) = (;hlo_module = 1, partitions = 2, version = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:HloUnoptimizedSnapshot})
    hlo_module = Ref{Union{Nothing,HloModuleProto}}(nothing)
    partitions = PB.BufferedVector{HloInputs}()
    version = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, hlo_module)
        elseif field_number == 2
            PB.decode!(d, partitions)
        elseif field_number == 3
            version = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return HloUnoptimizedSnapshot(hlo_module[], partitions[], version)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::HloUnoptimizedSnapshot)
    initpos = position(e.io)
    !isnothing(x.hlo_module) && PB.encode(e, 1, x.hlo_module)
    !isempty(x.partitions) && PB.encode(e, 2, x.partitions)
    x.version != zero(Int32) && PB.encode(e, 3, x.version)
    return position(e.io) - initpos
end
function PB._encoded_size(x::HloUnoptimizedSnapshot)
    encoded_size = 0
    !isnothing(x.hlo_module) && (encoded_size += PB._encoded_size(x.hlo_module, 1))
    !isempty(x.partitions) && (encoded_size += PB._encoded_size(x.partitions, 2))
    x.version != zero(Int32) && (encoded_size += PB._encoded_size(x.version, 3))
    return encoded_size
end

const OriginalValueRecoveryTableProto = var"##Stub#OriginalValueRecoveryTableProto"{var"##Stub#OriginalValueRecoveryTableProto.Entry"}
OriginalValueRecoveryTableProto(;entries = Vector{var"OriginalValueRecoveryTableProto.Entry"}()) = OriginalValueRecoveryTableProto(entries)
PB.default_values(::Type{OriginalValueRecoveryTableProto}) = (;entries = Vector{var"OriginalValueRecoveryTableProto.Entry"}())
PB.field_numbers(::Type{OriginalValueRecoveryTableProto}) = (;entries = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OriginalValueRecoveryTableProto})
    entries = PB.BufferedVector{var"OriginalValueRecoveryTableProto.Entry"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, entries)
        else
            Base.skip(d, wire_type)
        end
    end
    return OriginalValueRecoveryTableProto(entries[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OriginalValueRecoveryTableProto)
    initpos = position(e.io)
    !isempty(x.entries) && PB.encode(e, 1, x.entries)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OriginalValueRecoveryTableProto)
    encoded_size = 0
    !isempty(x.entries) && (encoded_size += PB._encoded_size(x.entries, 1))
    return encoded_size
end

const var"OriginalValueRecoveryTableProto.Entry" = var"##Stub#OriginalValueRecoveryTableProto.Entry"
var"OriginalValueRecoveryTableProto.Entry"(;old_original_array = nothing, new_original_array = nothing, recovery_module = nothing) = var"OriginalValueRecoveryTableProto.Entry"(old_original_array, new_original_array, recovery_module)
PB.default_values(::Type{var"OriginalValueRecoveryTableProto.Entry"}) = (;old_original_array = nothing, new_original_array = nothing, recovery_module = nothing)
PB.field_numbers(::Type{var"OriginalValueRecoveryTableProto.Entry"}) = (;old_original_array = 1, new_original_array = 2, recovery_module = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"OriginalValueRecoveryTableProto.Entry"})
    old_original_array = Ref{Union{Nothing,OriginalArrayProto}}(nothing)
    new_original_array = Ref{Union{Nothing,OriginalArrayProto}}(nothing)
    recovery_module = Ref{Union{Nothing,HloModuleProto}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, old_original_array)
        elseif field_number == 2
            PB.decode!(d, new_original_array)
        elseif field_number == 3
            PB.decode!(d, recovery_module)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"OriginalValueRecoveryTableProto.Entry"(old_original_array[], new_original_array[], recovery_module[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"OriginalValueRecoveryTableProto.Entry")
    initpos = position(e.io)
    !isnothing(x.old_original_array) && PB.encode(e, 1, x.old_original_array)
    !isnothing(x.new_original_array) && PB.encode(e, 2, x.new_original_array)
    !isnothing(x.recovery_module) && PB.encode(e, 3, x.recovery_module)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"OriginalValueRecoveryTableProto.Entry")
    encoded_size = 0
    !isnothing(x.old_original_array) && (encoded_size += PB._encoded_size(x.old_original_array, 1))
    !isnothing(x.new_original_array) && (encoded_size += PB._encoded_size(x.new_original_array, 2))
    !isnothing(x.recovery_module) && (encoded_size += PB._encoded_size(x.recovery_module, 3))
    return encoded_size
end
