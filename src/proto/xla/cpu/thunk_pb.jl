import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export AllGatherThunkProto, Int64Optional, AllReduceThunkProto, PartitionIdThunkProto
export RngGetAndUpdateStateThunkProto, TopKThunkProto, ReduceScatterThunkProto, InfoProto
export var"ResourceProto.Kind", SortDirectionProto, BoolOptional, RngSeedThunkProto
export AllToAllThunkProto, ShapeBufferAllocationSliceProto
export var"ConvolutionThunkProto.Options", var"KernelThunkProto.NumWorkGroups"
export ReplicaIdThunkProto, var"CollectivePermuteThunkProto.SourceTargetPairProto"
export ResourceProto, SortDirectionOptional, OpParamsProto, XnnFusionThunkProtoImpl
export CopyThunkProto, FftThunkProto, XnnDotThunkProto, DotThunkProto
export XnnConvolutionThunkProto, OpBuffersProto, YnnFusionThunkProto
export var"CustomCallThunkProto.OpBuffers", ConvolutionThunkProto, KernelThunkProto
export CollectivePermuteThunkProto, ResourceOptional
export var"ThunkSequenceProto.ResourceUsersProto", SortThunkProto, XnnFusionThunkProto
export CustomCallThunkProto, OpResourcesProto, var"OutfeedThunkProto.OutfeedResource"
export var"InfeedThunkProto.InfeedResource", CollectiveThunkProto, OutfeedThunkProto
export InfeedThunkProto, CallThunkProto, ConditionalThunkProto, ThunkSequenceProto
export WhileThunkProto, ThunkProto
abstract type var"##Abstract#CallThunkProto" end
abstract type var"##Abstract#ThunkProto" end
abstract type var"##Abstract#ConditionalThunkProto" end
abstract type var"##Abstract#WhileThunkProto" end
abstract type var"##Abstract#ThunkSequenceProto" end


struct AllGatherThunkProto end

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AllGatherThunkProto}, _endpos::Int=0, _group::Bool=false)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        Base.skip(d, wire_type)
    end
    return AllGatherThunkProto()
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AllGatherThunkProto)
    initpos = position(e.io)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AllGatherThunkProto)
    encoded_size = 0
    return encoded_size
end

struct Int64Optional
    value::Int64
    contains_value::Bool
end
PB.default_values(::Type{Int64Optional}) = (;value = zero(Int64), contains_value = false)
PB.field_numbers(::Type{Int64Optional}) = (;value = 1, contains_value = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Int64Optional}, _endpos::Int=0, _group::Bool=false)
    value = zero(Int64)
    contains_value = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            value = PB.decode(d, Int64)
        elseif field_number == 2
            contains_value = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return Int64Optional(value, contains_value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Int64Optional)
    initpos = position(e.io)
    x.value != zero(Int64) && PB.encode(e, 1, x.value)
    x.contains_value != false && PB.encode(e, 2, x.contains_value)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Int64Optional)
    encoded_size = 0
    x.value != zero(Int64) && (encoded_size += PB._encoded_size(x.value, 1))
    x.contains_value != false && (encoded_size += PB._encoded_size(x.contains_value, 2))
    return encoded_size
end

struct AllReduceThunkProto
    reduction_kind::String
    single_replica::Bool
end
PB.default_values(::Type{AllReduceThunkProto}) = (;reduction_kind = "", single_replica = false)
PB.field_numbers(::Type{AllReduceThunkProto}) = (;reduction_kind = 1, single_replica = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AllReduceThunkProto}, _endpos::Int=0, _group::Bool=false)
    reduction_kind = ""
    single_replica = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            reduction_kind = PB.decode(d, String)
        elseif field_number == 2
            single_replica = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return AllReduceThunkProto(reduction_kind, single_replica)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AllReduceThunkProto)
    initpos = position(e.io)
    !isempty(x.reduction_kind) && PB.encode(e, 1, x.reduction_kind)
    x.single_replica != false && PB.encode(e, 2, x.single_replica)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AllReduceThunkProto)
    encoded_size = 0
    !isempty(x.reduction_kind) && (encoded_size += PB._encoded_size(x.reduction_kind, 1))
    x.single_replica != false && (encoded_size += PB._encoded_size(x.single_replica, 2))
    return encoded_size
end

struct PartitionIdThunkProto
    logical_id_buffer::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
end
PB.default_values(::Type{PartitionIdThunkProto}) = (;logical_id_buffer = nothing)
PB.field_numbers(::Type{PartitionIdThunkProto}) = (;logical_id_buffer = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PartitionIdThunkProto}, _endpos::Int=0, _group::Bool=false)
    logical_id_buffer = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, logical_id_buffer)
        else
            Base.skip(d, wire_type)
        end
    end
    return PartitionIdThunkProto(logical_id_buffer[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PartitionIdThunkProto)
    initpos = position(e.io)
    !isnothing(x.logical_id_buffer) && PB.encode(e, 1, x.logical_id_buffer)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PartitionIdThunkProto)
    encoded_size = 0
    !isnothing(x.logical_id_buffer) && (encoded_size += PB._encoded_size(x.logical_id_buffer, 1))
    return encoded_size
end

struct RngGetAndUpdateStateThunkProto
    delta::Int64
    state_buffer::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
end
PB.default_values(::Type{RngGetAndUpdateStateThunkProto}) = (;delta = zero(Int64), state_buffer = nothing)
PB.field_numbers(::Type{RngGetAndUpdateStateThunkProto}) = (;delta = 1, state_buffer = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:RngGetAndUpdateStateThunkProto}, _endpos::Int=0, _group::Bool=false)
    delta = zero(Int64)
    state_buffer = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            delta = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, state_buffer)
        else
            Base.skip(d, wire_type)
        end
    end
    return RngGetAndUpdateStateThunkProto(delta, state_buffer[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::RngGetAndUpdateStateThunkProto)
    initpos = position(e.io)
    x.delta != zero(Int64) && PB.encode(e, 1, x.delta)
    !isnothing(x.state_buffer) && PB.encode(e, 2, x.state_buffer)
    return position(e.io) - initpos
end
function PB._encoded_size(x::RngGetAndUpdateStateThunkProto)
    encoded_size = 0
    x.delta != zero(Int64) && (encoded_size += PB._encoded_size(x.delta, 1))
    !isnothing(x.state_buffer) && (encoded_size += PB._encoded_size(x.state_buffer, 2))
    return encoded_size
end

struct TopKThunkProto
    batch_size::Int64
    input_size::Int64
    k::Int64
    values_buffer::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
    output_buffer::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
    indices_buffer::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
end
PB.default_values(::Type{TopKThunkProto}) = (;batch_size = zero(Int64), input_size = zero(Int64), k = zero(Int64), values_buffer = nothing, output_buffer = nothing, indices_buffer = nothing)
PB.field_numbers(::Type{TopKThunkProto}) = (;batch_size = 1, input_size = 2, k = 3, values_buffer = 4, output_buffer = 5, indices_buffer = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TopKThunkProto}, _endpos::Int=0, _group::Bool=false)
    batch_size = zero(Int64)
    input_size = zero(Int64)
    k = zero(Int64)
    values_buffer = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    output_buffer = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    indices_buffer = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            batch_size = PB.decode(d, Int64)
        elseif field_number == 2
            input_size = PB.decode(d, Int64)
        elseif field_number == 3
            k = PB.decode(d, Int64)
        elseif field_number == 4
            PB.decode!(d, values_buffer)
        elseif field_number == 5
            PB.decode!(d, output_buffer)
        elseif field_number == 6
            PB.decode!(d, indices_buffer)
        else
            Base.skip(d, wire_type)
        end
    end
    return TopKThunkProto(batch_size, input_size, k, values_buffer[], output_buffer[], indices_buffer[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TopKThunkProto)
    initpos = position(e.io)
    x.batch_size != zero(Int64) && PB.encode(e, 1, x.batch_size)
    x.input_size != zero(Int64) && PB.encode(e, 2, x.input_size)
    x.k != zero(Int64) && PB.encode(e, 3, x.k)
    !isnothing(x.values_buffer) && PB.encode(e, 4, x.values_buffer)
    !isnothing(x.output_buffer) && PB.encode(e, 5, x.output_buffer)
    !isnothing(x.indices_buffer) && PB.encode(e, 6, x.indices_buffer)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TopKThunkProto)
    encoded_size = 0
    x.batch_size != zero(Int64) && (encoded_size += PB._encoded_size(x.batch_size, 1))
    x.input_size != zero(Int64) && (encoded_size += PB._encoded_size(x.input_size, 2))
    x.k != zero(Int64) && (encoded_size += PB._encoded_size(x.k, 3))
    !isnothing(x.values_buffer) && (encoded_size += PB._encoded_size(x.values_buffer, 4))
    !isnothing(x.output_buffer) && (encoded_size += PB._encoded_size(x.output_buffer, 5))
    !isnothing(x.indices_buffer) && (encoded_size += PB._encoded_size(x.indices_buffer, 6))
    return encoded_size
end

struct ReduceScatterThunkProto
    reduction_kind::String
end
PB.default_values(::Type{ReduceScatterThunkProto}) = (;reduction_kind = "")
PB.field_numbers(::Type{ReduceScatterThunkProto}) = (;reduction_kind = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ReduceScatterThunkProto}, _endpos::Int=0, _group::Bool=false)
    reduction_kind = ""
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            reduction_kind = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return ReduceScatterThunkProto(reduction_kind)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ReduceScatterThunkProto)
    initpos = position(e.io)
    !isempty(x.reduction_kind) && PB.encode(e, 1, x.reduction_kind)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ReduceScatterThunkProto)
    encoded_size = 0
    !isempty(x.reduction_kind) && (encoded_size += PB._encoded_size(x.reduction_kind, 1))
    return encoded_size
end

struct InfoProto
    op_name::String
    module_name::String
    module_id::Int64
end
PB.default_values(::Type{InfoProto}) = (;op_name = "", module_name = "", module_id = zero(Int64))
PB.field_numbers(::Type{InfoProto}) = (;op_name = 1, module_name = 2, module_id = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:InfoProto}, _endpos::Int=0, _group::Bool=false)
    op_name = ""
    module_name = ""
    module_id = zero(Int64)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            op_name = PB.decode(d, String)
        elseif field_number == 2
            module_name = PB.decode(d, String)
        elseif field_number == 3
            module_id = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return InfoProto(op_name, module_name, module_id)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::InfoProto)
    initpos = position(e.io)
    !isempty(x.op_name) && PB.encode(e, 1, x.op_name)
    !isempty(x.module_name) && PB.encode(e, 2, x.module_name)
    x.module_id != zero(Int64) && PB.encode(e, 3, x.module_id)
    return position(e.io) - initpos
end
function PB._encoded_size(x::InfoProto)
    encoded_size = 0
    !isempty(x.op_name) && (encoded_size += PB._encoded_size(x.op_name, 1))
    !isempty(x.module_name) && (encoded_size += PB._encoded_size(x.module_name, 2))
    x.module_id != zero(Int64) && (encoded_size += PB._encoded_size(x.module_id, 3))
    return encoded_size
end

@enumx var"ResourceProto.Kind" UNKNOWN=0 TOKEN=1 COLLECTIVE_COMMUNICATOR=2

@enumx SortDirectionProto UNKNOWN=0 ASCENDING=1 DESCENDING=2

struct BoolOptional
    value::Bool
    contains_value::Bool
end
PB.default_values(::Type{BoolOptional}) = (;value = false, contains_value = false)
PB.field_numbers(::Type{BoolOptional}) = (;value = 1, contains_value = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:BoolOptional}, _endpos::Int=0, _group::Bool=false)
    value = false
    contains_value = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            value = PB.decode(d, Bool)
        elseif field_number == 2
            contains_value = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return BoolOptional(value, contains_value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::BoolOptional)
    initpos = position(e.io)
    x.value != false && PB.encode(e, 1, x.value)
    x.contains_value != false && PB.encode(e, 2, x.contains_value)
    return position(e.io) - initpos
end
function PB._encoded_size(x::BoolOptional)
    encoded_size = 0
    x.value != false && (encoded_size += PB._encoded_size(x.value, 1))
    x.contains_value != false && (encoded_size += PB._encoded_size(x.contains_value, 2))
    return encoded_size
end

struct RngSeedThunkProto
    dest_buffer::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
end
PB.default_values(::Type{RngSeedThunkProto}) = (;dest_buffer = nothing)
PB.field_numbers(::Type{RngSeedThunkProto}) = (;dest_buffer = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:RngSeedThunkProto}, _endpos::Int=0, _group::Bool=false)
    dest_buffer = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, dest_buffer)
        else
            Base.skip(d, wire_type)
        end
    end
    return RngSeedThunkProto(dest_buffer[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::RngSeedThunkProto)
    initpos = position(e.io)
    !isnothing(x.dest_buffer) && PB.encode(e, 1, x.dest_buffer)
    return position(e.io) - initpos
end
function PB._encoded_size(x::RngSeedThunkProto)
    encoded_size = 0
    !isnothing(x.dest_buffer) && (encoded_size += PB._encoded_size(x.dest_buffer, 1))
    return encoded_size
end

struct AllToAllThunkProto end

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AllToAllThunkProto}, _endpos::Int=0, _group::Bool=false)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        Base.skip(d, wire_type)
    end
    return AllToAllThunkProto()
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AllToAllThunkProto)
    initpos = position(e.io)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AllToAllThunkProto)
    encoded_size = 0
    return encoded_size
end

struct ShapeBufferAllocationSliceProto
    shape::Union{Nothing,xla.ShapeProto}
    slice::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
end
PB.default_values(::Type{ShapeBufferAllocationSliceProto}) = (;shape = nothing, slice = nothing)
PB.field_numbers(::Type{ShapeBufferAllocationSliceProto}) = (;shape = 1, slice = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ShapeBufferAllocationSliceProto}, _endpos::Int=0, _group::Bool=false)
    shape = Ref{Union{Nothing,xla.ShapeProto}}(nothing)
    slice = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, shape)
        elseif field_number == 2
            PB.decode!(d, slice)
        else
            Base.skip(d, wire_type)
        end
    end
    return ShapeBufferAllocationSliceProto(shape[], slice[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ShapeBufferAllocationSliceProto)
    initpos = position(e.io)
    !isnothing(x.shape) && PB.encode(e, 1, x.shape)
    !isnothing(x.slice) && PB.encode(e, 2, x.slice)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ShapeBufferAllocationSliceProto)
    encoded_size = 0
    !isnothing(x.shape) && (encoded_size += PB._encoded_size(x.shape, 1))
    !isnothing(x.slice) && (encoded_size += PB._encoded_size(x.slice, 2))
    return encoded_size
end

struct var"ConvolutionThunkProto.Options"
    multi_threaded::Bool
end
PB.default_values(::Type{var"ConvolutionThunkProto.Options"}) = (;multi_threaded = false)
PB.field_numbers(::Type{var"ConvolutionThunkProto.Options"}) = (;multi_threaded = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"ConvolutionThunkProto.Options"}, _endpos::Int=0, _group::Bool=false)
    multi_threaded = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            multi_threaded = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"ConvolutionThunkProto.Options"(multi_threaded)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"ConvolutionThunkProto.Options")
    initpos = position(e.io)
    x.multi_threaded != false && PB.encode(e, 1, x.multi_threaded)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"ConvolutionThunkProto.Options")
    encoded_size = 0
    x.multi_threaded != false && (encoded_size += PB._encoded_size(x.multi_threaded, 1))
    return encoded_size
end

struct var"KernelThunkProto.NumWorkGroups"
    x::Int64
    y::Int64
    z::Int64
end
PB.default_values(::Type{var"KernelThunkProto.NumWorkGroups"}) = (;x = zero(Int64), y = zero(Int64), z = zero(Int64))
PB.field_numbers(::Type{var"KernelThunkProto.NumWorkGroups"}) = (;x = 1, y = 2, z = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"KernelThunkProto.NumWorkGroups"}, _endpos::Int=0, _group::Bool=false)
    x = zero(Int64)
    y = zero(Int64)
    z = zero(Int64)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            x = PB.decode(d, Int64)
        elseif field_number == 2
            y = PB.decode(d, Int64)
        elseif field_number == 3
            z = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"KernelThunkProto.NumWorkGroups"(x, y, z)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"KernelThunkProto.NumWorkGroups")
    initpos = position(e.io)
    x.x != zero(Int64) && PB.encode(e, 1, x.x)
    x.y != zero(Int64) && PB.encode(e, 2, x.y)
    x.z != zero(Int64) && PB.encode(e, 3, x.z)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"KernelThunkProto.NumWorkGroups")
    encoded_size = 0
    x.x != zero(Int64) && (encoded_size += PB._encoded_size(x.x, 1))
    x.y != zero(Int64) && (encoded_size += PB._encoded_size(x.y, 2))
    x.z != zero(Int64) && (encoded_size += PB._encoded_size(x.z, 3))
    return encoded_size
end

struct ReplicaIdThunkProto
    logical_id_buffer::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
end
PB.default_values(::Type{ReplicaIdThunkProto}) = (;logical_id_buffer = nothing)
PB.field_numbers(::Type{ReplicaIdThunkProto}) = (;logical_id_buffer = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ReplicaIdThunkProto}, _endpos::Int=0, _group::Bool=false)
    logical_id_buffer = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, logical_id_buffer)
        else
            Base.skip(d, wire_type)
        end
    end
    return ReplicaIdThunkProto(logical_id_buffer[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ReplicaIdThunkProto)
    initpos = position(e.io)
    !isnothing(x.logical_id_buffer) && PB.encode(e, 1, x.logical_id_buffer)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ReplicaIdThunkProto)
    encoded_size = 0
    !isnothing(x.logical_id_buffer) && (encoded_size += PB._encoded_size(x.logical_id_buffer, 1))
    return encoded_size
end

struct var"CollectivePermuteThunkProto.SourceTargetPairProto"
    source::Int64
    target::Int64
end
PB.default_values(::Type{var"CollectivePermuteThunkProto.SourceTargetPairProto"}) = (;source = zero(Int64), target = zero(Int64))
PB.field_numbers(::Type{var"CollectivePermuteThunkProto.SourceTargetPairProto"}) = (;source = 1, target = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"CollectivePermuteThunkProto.SourceTargetPairProto"}, _endpos::Int=0, _group::Bool=false)
    source = zero(Int64)
    target = zero(Int64)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            source = PB.decode(d, Int64)
        elseif field_number == 2
            target = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"CollectivePermuteThunkProto.SourceTargetPairProto"(source, target)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"CollectivePermuteThunkProto.SourceTargetPairProto")
    initpos = position(e.io)
    x.source != zero(Int64) && PB.encode(e, 1, x.source)
    x.target != zero(Int64) && PB.encode(e, 2, x.target)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"CollectivePermuteThunkProto.SourceTargetPairProto")
    encoded_size = 0
    x.source != zero(Int64) && (encoded_size += PB._encoded_size(x.source, 1))
    x.target != zero(Int64) && (encoded_size += PB._encoded_size(x.target, 2))
    return encoded_size
end

struct ResourceProto
    kind::var"ResourceProto.Kind".T
end
PB.default_values(::Type{ResourceProto}) = (;kind = var"ResourceProto.Kind".UNKNOWN)
PB.field_numbers(::Type{ResourceProto}) = (;kind = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ResourceProto}, _endpos::Int=0, _group::Bool=false)
    kind = var"ResourceProto.Kind".UNKNOWN
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            kind = PB.decode(d, var"ResourceProto.Kind".T)
        else
            Base.skip(d, wire_type)
        end
    end
    return ResourceProto(kind)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ResourceProto)
    initpos = position(e.io)
    x.kind != var"ResourceProto.Kind".UNKNOWN && PB.encode(e, 1, x.kind)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ResourceProto)
    encoded_size = 0
    x.kind != var"ResourceProto.Kind".UNKNOWN && (encoded_size += PB._encoded_size(x.kind, 1))
    return encoded_size
end

struct SortDirectionOptional
    value::SortDirectionProto.T
    contains_value::Bool
end
PB.default_values(::Type{SortDirectionOptional}) = (;value = SortDirectionProto.UNKNOWN, contains_value = false)
PB.field_numbers(::Type{SortDirectionOptional}) = (;value = 1, contains_value = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SortDirectionOptional}, _endpos::Int=0, _group::Bool=false)
    value = SortDirectionProto.UNKNOWN
    contains_value = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            value = PB.decode(d, SortDirectionProto.T)
        elseif field_number == 2
            contains_value = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return SortDirectionOptional(value, contains_value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SortDirectionOptional)
    initpos = position(e.io)
    x.value != SortDirectionProto.UNKNOWN && PB.encode(e, 1, x.value)
    x.contains_value != false && PB.encode(e, 2, x.contains_value)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SortDirectionOptional)
    encoded_size = 0
    x.value != SortDirectionProto.UNKNOWN && (encoded_size += PB._encoded_size(x.value, 1))
    x.contains_value != false && (encoded_size += PB._encoded_size(x.contains_value, 2))
    return encoded_size
end

struct OpParamsProto
    op_id::Int64
    has_channel_id::Bool
    use_global_device_ids::Union{Nothing,BoolOptional}
    replica_group::Vector{xla.ReplicaGroup}
end
PB.default_values(::Type{OpParamsProto}) = (;op_id = zero(Int64), has_channel_id = false, use_global_device_ids = nothing, replica_group = Vector{xla.ReplicaGroup}())
PB.field_numbers(::Type{OpParamsProto}) = (;op_id = 1, has_channel_id = 2, use_global_device_ids = 3, replica_group = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OpParamsProto}, _endpos::Int=0, _group::Bool=false)
    op_id = zero(Int64)
    has_channel_id = false
    use_global_device_ids = Ref{Union{Nothing,BoolOptional}}(nothing)
    replica_group = PB.BufferedVector{xla.ReplicaGroup}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            op_id = PB.decode(d, Int64)
        elseif field_number == 2
            has_channel_id = PB.decode(d, Bool)
        elseif field_number == 3
            PB.decode!(d, use_global_device_ids)
        elseif field_number == 4
            PB.decode!(d, replica_group)
        else
            Base.skip(d, wire_type)
        end
    end
    return OpParamsProto(op_id, has_channel_id, use_global_device_ids[], replica_group[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OpParamsProto)
    initpos = position(e.io)
    x.op_id != zero(Int64) && PB.encode(e, 1, x.op_id)
    x.has_channel_id != false && PB.encode(e, 2, x.has_channel_id)
    !isnothing(x.use_global_device_ids) && PB.encode(e, 3, x.use_global_device_ids)
    !isempty(x.replica_group) && PB.encode(e, 4, x.replica_group)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OpParamsProto)
    encoded_size = 0
    x.op_id != zero(Int64) && (encoded_size += PB._encoded_size(x.op_id, 1))
    x.has_channel_id != false && (encoded_size += PB._encoded_size(x.has_channel_id, 2))
    !isnothing(x.use_global_device_ids) && (encoded_size += PB._encoded_size(x.use_global_device_ids, 3))
    !isempty(x.replica_group) && (encoded_size += PB._encoded_size(x.replica_group, 4))
    return encoded_size
end

struct XnnFusionThunkProtoImpl
    arguments_shapes::Vector{ShapeBufferAllocationSliceProto}
    results_shapes::Vector{ShapeBufferAllocationSliceProto}
end
PB.default_values(::Type{XnnFusionThunkProtoImpl}) = (;arguments_shapes = Vector{ShapeBufferAllocationSliceProto}(), results_shapes = Vector{ShapeBufferAllocationSliceProto}())
PB.field_numbers(::Type{XnnFusionThunkProtoImpl}) = (;arguments_shapes = 1, results_shapes = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XnnFusionThunkProtoImpl}, _endpos::Int=0, _group::Bool=false)
    arguments_shapes = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    results_shapes = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, arguments_shapes)
        elseif field_number == 2
            PB.decode!(d, results_shapes)
        else
            Base.skip(d, wire_type)
        end
    end
    return XnnFusionThunkProtoImpl(arguments_shapes[], results_shapes[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XnnFusionThunkProtoImpl)
    initpos = position(e.io)
    !isempty(x.arguments_shapes) && PB.encode(e, 1, x.arguments_shapes)
    !isempty(x.results_shapes) && PB.encode(e, 2, x.results_shapes)
    return position(e.io) - initpos
end
function PB._encoded_size(x::XnnFusionThunkProtoImpl)
    encoded_size = 0
    !isempty(x.arguments_shapes) && (encoded_size += PB._encoded_size(x.arguments_shapes, 1))
    !isempty(x.results_shapes) && (encoded_size += PB._encoded_size(x.results_shapes, 2))
    return encoded_size
end

struct CopyThunkProto
    src_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    dst_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
end
PB.default_values(::Type{CopyThunkProto}) = (;src_buffer_shape = nothing, dst_buffer_shape = nothing)
PB.field_numbers(::Type{CopyThunkProto}) = (;src_buffer_shape = 1, dst_buffer_shape = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CopyThunkProto}, _endpos::Int=0, _group::Bool=false)
    src_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    dst_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, src_buffer_shape)
        elseif field_number == 2
            PB.decode!(d, dst_buffer_shape)
        else
            Base.skip(d, wire_type)
        end
    end
    return CopyThunkProto(src_buffer_shape[], dst_buffer_shape[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CopyThunkProto)
    initpos = position(e.io)
    !isnothing(x.src_buffer_shape) && PB.encode(e, 1, x.src_buffer_shape)
    !isnothing(x.dst_buffer_shape) && PB.encode(e, 2, x.dst_buffer_shape)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CopyThunkProto)
    encoded_size = 0
    !isnothing(x.src_buffer_shape) && (encoded_size += PB._encoded_size(x.src_buffer_shape, 1))
    !isnothing(x.dst_buffer_shape) && (encoded_size += PB._encoded_size(x.dst_buffer_shape, 2))
    return encoded_size
end

struct FftThunkProto
    is_multi_thread_eigen::Bool
    fft_type::Int32
    fft_length::Vector{Int64}
    input_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    output_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
end
PB.default_values(::Type{FftThunkProto}) = (;is_multi_thread_eigen = false, fft_type = zero(Int32), fft_length = Vector{Int64}(), input_buffer_shape = nothing, output_buffer_shape = nothing)
PB.field_numbers(::Type{FftThunkProto}) = (;is_multi_thread_eigen = 1, fft_type = 2, fft_length = 3, input_buffer_shape = 4, output_buffer_shape = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:FftThunkProto}, _endpos::Int=0, _group::Bool=false)
    is_multi_thread_eigen = false
    fft_type = zero(Int32)
    fft_length = PB.BufferedVector{Int64}()
    input_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    output_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            is_multi_thread_eigen = PB.decode(d, Bool)
        elseif field_number == 2
            fft_type = PB.decode(d, Int32)
        elseif field_number == 3
            PB.decode!(d, wire_type, fft_length)
        elseif field_number == 4
            PB.decode!(d, input_buffer_shape)
        elseif field_number == 5
            PB.decode!(d, output_buffer_shape)
        else
            Base.skip(d, wire_type)
        end
    end
    return FftThunkProto(is_multi_thread_eigen, fft_type, fft_length[], input_buffer_shape[], output_buffer_shape[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::FftThunkProto)
    initpos = position(e.io)
    x.is_multi_thread_eigen != false && PB.encode(e, 1, x.is_multi_thread_eigen)
    x.fft_type != zero(Int32) && PB.encode(e, 2, x.fft_type)
    !isempty(x.fft_length) && PB.encode(e, 3, x.fft_length)
    !isnothing(x.input_buffer_shape) && PB.encode(e, 4, x.input_buffer_shape)
    !isnothing(x.output_buffer_shape) && PB.encode(e, 5, x.output_buffer_shape)
    return position(e.io) - initpos
end
function PB._encoded_size(x::FftThunkProto)
    encoded_size = 0
    x.is_multi_thread_eigen != false && (encoded_size += PB._encoded_size(x.is_multi_thread_eigen, 1))
    x.fft_type != zero(Int32) && (encoded_size += PB._encoded_size(x.fft_type, 2))
    !isempty(x.fft_length) && (encoded_size += PB._encoded_size(x.fft_length, 3))
    !isnothing(x.input_buffer_shape) && (encoded_size += PB._encoded_size(x.input_buffer_shape, 4))
    !isnothing(x.output_buffer_shape) && (encoded_size += PB._encoded_size(x.output_buffer_shape, 5))
    return encoded_size
end

struct XnnDotThunkProto
    dot_dimensions::Union{Nothing,xla.DotDimensionNumbers}
    lhs_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    rhs_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    out_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    capture_rhs::Bool
end
PB.default_values(::Type{XnnDotThunkProto}) = (;dot_dimensions = nothing, lhs_buffer_shape = nothing, rhs_buffer_shape = nothing, out_buffer_shape = nothing, capture_rhs = false)
PB.field_numbers(::Type{XnnDotThunkProto}) = (;dot_dimensions = 1, lhs_buffer_shape = 2, rhs_buffer_shape = 3, out_buffer_shape = 4, capture_rhs = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XnnDotThunkProto}, _endpos::Int=0, _group::Bool=false)
    dot_dimensions = Ref{Union{Nothing,xla.DotDimensionNumbers}}(nothing)
    lhs_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    rhs_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    out_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    capture_rhs = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, dot_dimensions)
        elseif field_number == 2
            PB.decode!(d, lhs_buffer_shape)
        elseif field_number == 3
            PB.decode!(d, rhs_buffer_shape)
        elseif field_number == 4
            PB.decode!(d, out_buffer_shape)
        elseif field_number == 5
            capture_rhs = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return XnnDotThunkProto(dot_dimensions[], lhs_buffer_shape[], rhs_buffer_shape[], out_buffer_shape[], capture_rhs)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XnnDotThunkProto)
    initpos = position(e.io)
    !isnothing(x.dot_dimensions) && PB.encode(e, 1, x.dot_dimensions)
    !isnothing(x.lhs_buffer_shape) && PB.encode(e, 2, x.lhs_buffer_shape)
    !isnothing(x.rhs_buffer_shape) && PB.encode(e, 3, x.rhs_buffer_shape)
    !isnothing(x.out_buffer_shape) && PB.encode(e, 4, x.out_buffer_shape)
    x.capture_rhs != false && PB.encode(e, 5, x.capture_rhs)
    return position(e.io) - initpos
end
function PB._encoded_size(x::XnnDotThunkProto)
    encoded_size = 0
    !isnothing(x.dot_dimensions) && (encoded_size += PB._encoded_size(x.dot_dimensions, 1))
    !isnothing(x.lhs_buffer_shape) && (encoded_size += PB._encoded_size(x.lhs_buffer_shape, 2))
    !isnothing(x.rhs_buffer_shape) && (encoded_size += PB._encoded_size(x.rhs_buffer_shape, 3))
    !isnothing(x.out_buffer_shape) && (encoded_size += PB._encoded_size(x.out_buffer_shape, 4))
    x.capture_rhs != false && (encoded_size += PB._encoded_size(x.capture_rhs, 5))
    return encoded_size
end

struct DotThunkProto
    dot_dimensions::Union{Nothing,xla.DotDimensionNumbers}
    lhs_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    rhs_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    out_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
end
PB.default_values(::Type{DotThunkProto}) = (;dot_dimensions = nothing, lhs_buffer_shape = nothing, rhs_buffer_shape = nothing, out_buffer_shape = nothing)
PB.field_numbers(::Type{DotThunkProto}) = (;dot_dimensions = 1, lhs_buffer_shape = 2, rhs_buffer_shape = 3, out_buffer_shape = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:DotThunkProto}, _endpos::Int=0, _group::Bool=false)
    dot_dimensions = Ref{Union{Nothing,xla.DotDimensionNumbers}}(nothing)
    lhs_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    rhs_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    out_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, dot_dimensions)
        elseif field_number == 2
            PB.decode!(d, lhs_buffer_shape)
        elseif field_number == 3
            PB.decode!(d, rhs_buffer_shape)
        elseif field_number == 4
            PB.decode!(d, out_buffer_shape)
        else
            Base.skip(d, wire_type)
        end
    end
    return DotThunkProto(dot_dimensions[], lhs_buffer_shape[], rhs_buffer_shape[], out_buffer_shape[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::DotThunkProto)
    initpos = position(e.io)
    !isnothing(x.dot_dimensions) && PB.encode(e, 1, x.dot_dimensions)
    !isnothing(x.lhs_buffer_shape) && PB.encode(e, 2, x.lhs_buffer_shape)
    !isnothing(x.rhs_buffer_shape) && PB.encode(e, 3, x.rhs_buffer_shape)
    !isnothing(x.out_buffer_shape) && PB.encode(e, 4, x.out_buffer_shape)
    return position(e.io) - initpos
end
function PB._encoded_size(x::DotThunkProto)
    encoded_size = 0
    !isnothing(x.dot_dimensions) && (encoded_size += PB._encoded_size(x.dot_dimensions, 1))
    !isnothing(x.lhs_buffer_shape) && (encoded_size += PB._encoded_size(x.lhs_buffer_shape, 2))
    !isnothing(x.rhs_buffer_shape) && (encoded_size += PB._encoded_size(x.rhs_buffer_shape, 3))
    !isnothing(x.out_buffer_shape) && (encoded_size += PB._encoded_size(x.out_buffer_shape, 4))
    return encoded_size
end

struct XnnConvolutionThunkProto
    dimension_numbers::Union{Nothing,xla.ConvolutionDimensionNumbers}
    window::Union{Nothing,xla.Window}
    feature_group_count::Int64
    input_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    kernel_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    output_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
end
PB.default_values(::Type{XnnConvolutionThunkProto}) = (;dimension_numbers = nothing, window = nothing, feature_group_count = zero(Int64), input_buffer_shape = nothing, kernel_buffer_shape = nothing, output_buffer_shape = nothing)
PB.field_numbers(::Type{XnnConvolutionThunkProto}) = (;dimension_numbers = 1, window = 2, feature_group_count = 3, input_buffer_shape = 4, kernel_buffer_shape = 5, output_buffer_shape = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XnnConvolutionThunkProto}, _endpos::Int=0, _group::Bool=false)
    dimension_numbers = Ref{Union{Nothing,xla.ConvolutionDimensionNumbers}}(nothing)
    window = Ref{Union{Nothing,xla.Window}}(nothing)
    feature_group_count = zero(Int64)
    input_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    kernel_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    output_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, dimension_numbers)
        elseif field_number == 2
            PB.decode!(d, window)
        elseif field_number == 3
            feature_group_count = PB.decode(d, Int64)
        elseif field_number == 4
            PB.decode!(d, input_buffer_shape)
        elseif field_number == 5
            PB.decode!(d, kernel_buffer_shape)
        elseif field_number == 6
            PB.decode!(d, output_buffer_shape)
        else
            Base.skip(d, wire_type)
        end
    end
    return XnnConvolutionThunkProto(dimension_numbers[], window[], feature_group_count, input_buffer_shape[], kernel_buffer_shape[], output_buffer_shape[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XnnConvolutionThunkProto)
    initpos = position(e.io)
    !isnothing(x.dimension_numbers) && PB.encode(e, 1, x.dimension_numbers)
    !isnothing(x.window) && PB.encode(e, 2, x.window)
    x.feature_group_count != zero(Int64) && PB.encode(e, 3, x.feature_group_count)
    !isnothing(x.input_buffer_shape) && PB.encode(e, 4, x.input_buffer_shape)
    !isnothing(x.kernel_buffer_shape) && PB.encode(e, 5, x.kernel_buffer_shape)
    !isnothing(x.output_buffer_shape) && PB.encode(e, 6, x.output_buffer_shape)
    return position(e.io) - initpos
end
function PB._encoded_size(x::XnnConvolutionThunkProto)
    encoded_size = 0
    !isnothing(x.dimension_numbers) && (encoded_size += PB._encoded_size(x.dimension_numbers, 1))
    !isnothing(x.window) && (encoded_size += PB._encoded_size(x.window, 2))
    x.feature_group_count != zero(Int64) && (encoded_size += PB._encoded_size(x.feature_group_count, 3))
    !isnothing(x.input_buffer_shape) && (encoded_size += PB._encoded_size(x.input_buffer_shape, 4))
    !isnothing(x.kernel_buffer_shape) && (encoded_size += PB._encoded_size(x.kernel_buffer_shape, 5))
    !isnothing(x.output_buffer_shape) && (encoded_size += PB._encoded_size(x.output_buffer_shape, 6))
    return encoded_size
end

struct OpBuffersProto
    source_shapes_buffer_slices::Vector{ShapeBufferAllocationSliceProto}
    destination_shapes_buffer_slices::Vector{ShapeBufferAllocationSliceProto}
end
PB.default_values(::Type{OpBuffersProto}) = (;source_shapes_buffer_slices = Vector{ShapeBufferAllocationSliceProto}(), destination_shapes_buffer_slices = Vector{ShapeBufferAllocationSliceProto}())
PB.field_numbers(::Type{OpBuffersProto}) = (;source_shapes_buffer_slices = 1, destination_shapes_buffer_slices = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OpBuffersProto}, _endpos::Int=0, _group::Bool=false)
    source_shapes_buffer_slices = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    destination_shapes_buffer_slices = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, source_shapes_buffer_slices)
        elseif field_number == 2
            PB.decode!(d, destination_shapes_buffer_slices)
        else
            Base.skip(d, wire_type)
        end
    end
    return OpBuffersProto(source_shapes_buffer_slices[], destination_shapes_buffer_slices[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OpBuffersProto)
    initpos = position(e.io)
    !isempty(x.source_shapes_buffer_slices) && PB.encode(e, 1, x.source_shapes_buffer_slices)
    !isempty(x.destination_shapes_buffer_slices) && PB.encode(e, 2, x.destination_shapes_buffer_slices)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OpBuffersProto)
    encoded_size = 0
    !isempty(x.source_shapes_buffer_slices) && (encoded_size += PB._encoded_size(x.source_shapes_buffer_slices, 1))
    !isempty(x.destination_shapes_buffer_slices) && (encoded_size += PB._encoded_size(x.destination_shapes_buffer_slices, 2))
    return encoded_size
end

struct YnnFusionThunkProto
    options::Union{Nothing,YnnFusionOptions}
    instruction_id::Int64
    arguments_shapes::Vector{ShapeBufferAllocationSliceProto}
    results_shapes::Vector{ShapeBufferAllocationSliceProto}
end
PB.default_values(::Type{YnnFusionThunkProto}) = (;options = nothing, instruction_id = zero(Int64), arguments_shapes = Vector{ShapeBufferAllocationSliceProto}(), results_shapes = Vector{ShapeBufferAllocationSliceProto}())
PB.field_numbers(::Type{YnnFusionThunkProto}) = (;options = 1, instruction_id = 2, arguments_shapes = 3, results_shapes = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:YnnFusionThunkProto}, _endpos::Int=0, _group::Bool=false)
    options = Ref{Union{Nothing,YnnFusionOptions}}(nothing)
    instruction_id = zero(Int64)
    arguments_shapes = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    results_shapes = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, options)
        elseif field_number == 2
            instruction_id = PB.decode(d, Int64)
        elseif field_number == 3
            PB.decode!(d, arguments_shapes)
        elseif field_number == 4
            PB.decode!(d, results_shapes)
        else
            Base.skip(d, wire_type)
        end
    end
    return YnnFusionThunkProto(options[], instruction_id, arguments_shapes[], results_shapes[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::YnnFusionThunkProto)
    initpos = position(e.io)
    !isnothing(x.options) && PB.encode(e, 1, x.options)
    x.instruction_id != zero(Int64) && PB.encode(e, 2, x.instruction_id)
    !isempty(x.arguments_shapes) && PB.encode(e, 3, x.arguments_shapes)
    !isempty(x.results_shapes) && PB.encode(e, 4, x.results_shapes)
    return position(e.io) - initpos
end
function PB._encoded_size(x::YnnFusionThunkProto)
    encoded_size = 0
    !isnothing(x.options) && (encoded_size += PB._encoded_size(x.options, 1))
    x.instruction_id != zero(Int64) && (encoded_size += PB._encoded_size(x.instruction_id, 2))
    !isempty(x.arguments_shapes) && (encoded_size += PB._encoded_size(x.arguments_shapes, 3))
    !isempty(x.results_shapes) && (encoded_size += PB._encoded_size(x.results_shapes, 4))
    return encoded_size
end

struct var"CustomCallThunkProto.OpBuffers"
    arguments_shapes::Vector{ShapeBufferAllocationSliceProto}
    results_shapes::Vector{ShapeBufferAllocationSliceProto}
    is_tuple_result::Bool
end
PB.default_values(::Type{var"CustomCallThunkProto.OpBuffers"}) = (;arguments_shapes = Vector{ShapeBufferAllocationSliceProto}(), results_shapes = Vector{ShapeBufferAllocationSliceProto}(), is_tuple_result = false)
PB.field_numbers(::Type{var"CustomCallThunkProto.OpBuffers"}) = (;arguments_shapes = 1, results_shapes = 2, is_tuple_result = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"CustomCallThunkProto.OpBuffers"}, _endpos::Int=0, _group::Bool=false)
    arguments_shapes = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    results_shapes = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    is_tuple_result = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, arguments_shapes)
        elseif field_number == 2
            PB.decode!(d, results_shapes)
        elseif field_number == 3
            is_tuple_result = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"CustomCallThunkProto.OpBuffers"(arguments_shapes[], results_shapes[], is_tuple_result)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"CustomCallThunkProto.OpBuffers")
    initpos = position(e.io)
    !isempty(x.arguments_shapes) && PB.encode(e, 1, x.arguments_shapes)
    !isempty(x.results_shapes) && PB.encode(e, 2, x.results_shapes)
    x.is_tuple_result != false && PB.encode(e, 3, x.is_tuple_result)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"CustomCallThunkProto.OpBuffers")
    encoded_size = 0
    !isempty(x.arguments_shapes) && (encoded_size += PB._encoded_size(x.arguments_shapes, 1))
    !isempty(x.results_shapes) && (encoded_size += PB._encoded_size(x.results_shapes, 2))
    x.is_tuple_result != false && (encoded_size += PB._encoded_size(x.is_tuple_result, 3))
    return encoded_size
end

struct ConvolutionThunkProto
    dimension_numbers::Union{Nothing,xla.ConvolutionDimensionNumbers}
    window::Union{Nothing,xla.Window}
    feature_group_count::Int64
    input_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    kernel_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    output_buffer_shape::Union{Nothing,ShapeBufferAllocationSliceProto}
    options::Union{Nothing,var"ConvolutionThunkProto.Options"}
end
PB.default_values(::Type{ConvolutionThunkProto}) = (;dimension_numbers = nothing, window = nothing, feature_group_count = zero(Int64), input_buffer_shape = nothing, kernel_buffer_shape = nothing, output_buffer_shape = nothing, options = nothing)
PB.field_numbers(::Type{ConvolutionThunkProto}) = (;dimension_numbers = 1, window = 2, feature_group_count = 3, input_buffer_shape = 4, kernel_buffer_shape = 5, output_buffer_shape = 6, options = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ConvolutionThunkProto}, _endpos::Int=0, _group::Bool=false)
    dimension_numbers = Ref{Union{Nothing,xla.ConvolutionDimensionNumbers}}(nothing)
    window = Ref{Union{Nothing,xla.Window}}(nothing)
    feature_group_count = zero(Int64)
    input_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    kernel_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    output_buffer_shape = Ref{Union{Nothing,ShapeBufferAllocationSliceProto}}(nothing)
    options = Ref{Union{Nothing,var"ConvolutionThunkProto.Options"}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, dimension_numbers)
        elseif field_number == 2
            PB.decode!(d, window)
        elseif field_number == 3
            feature_group_count = PB.decode(d, Int64)
        elseif field_number == 4
            PB.decode!(d, input_buffer_shape)
        elseif field_number == 5
            PB.decode!(d, kernel_buffer_shape)
        elseif field_number == 6
            PB.decode!(d, output_buffer_shape)
        elseif field_number == 7
            PB.decode!(d, options)
        else
            Base.skip(d, wire_type)
        end
    end
    return ConvolutionThunkProto(dimension_numbers[], window[], feature_group_count, input_buffer_shape[], kernel_buffer_shape[], output_buffer_shape[], options[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ConvolutionThunkProto)
    initpos = position(e.io)
    !isnothing(x.dimension_numbers) && PB.encode(e, 1, x.dimension_numbers)
    !isnothing(x.window) && PB.encode(e, 2, x.window)
    x.feature_group_count != zero(Int64) && PB.encode(e, 3, x.feature_group_count)
    !isnothing(x.input_buffer_shape) && PB.encode(e, 4, x.input_buffer_shape)
    !isnothing(x.kernel_buffer_shape) && PB.encode(e, 5, x.kernel_buffer_shape)
    !isnothing(x.output_buffer_shape) && PB.encode(e, 6, x.output_buffer_shape)
    !isnothing(x.options) && PB.encode(e, 7, x.options)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ConvolutionThunkProto)
    encoded_size = 0
    !isnothing(x.dimension_numbers) && (encoded_size += PB._encoded_size(x.dimension_numbers, 1))
    !isnothing(x.window) && (encoded_size += PB._encoded_size(x.window, 2))
    x.feature_group_count != zero(Int64) && (encoded_size += PB._encoded_size(x.feature_group_count, 3))
    !isnothing(x.input_buffer_shape) && (encoded_size += PB._encoded_size(x.input_buffer_shape, 4))
    !isnothing(x.kernel_buffer_shape) && (encoded_size += PB._encoded_size(x.kernel_buffer_shape, 5))
    !isnothing(x.output_buffer_shape) && (encoded_size += PB._encoded_size(x.output_buffer_shape, 6))
    !isnothing(x.options) && (encoded_size += PB._encoded_size(x.options, 7))
    return encoded_size
end

struct KernelThunkProto
    kernel_name::String
    num_workgroups::Union{Nothing,var"KernelThunkProto.NumWorkGroups"}
    invariant_arguments::Vector{Int64}
    min_alignment::Union{Nothing,Int64Optional}
    arguments_buffers::Vector{xla.ShapedSliceProto}
    results_buffers::Vector{xla.ShapedSliceProto}
end
PB.default_values(::Type{KernelThunkProto}) = (;kernel_name = "", num_workgroups = nothing, invariant_arguments = Vector{Int64}(), min_alignment = nothing, arguments_buffers = Vector{xla.ShapedSliceProto}(), results_buffers = Vector{xla.ShapedSliceProto}())
PB.field_numbers(::Type{KernelThunkProto}) = (;kernel_name = 1, num_workgroups = 2, invariant_arguments = 3, min_alignment = 4, arguments_buffers = 5, results_buffers = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:KernelThunkProto}, _endpos::Int=0, _group::Bool=false)
    kernel_name = ""
    num_workgroups = Ref{Union{Nothing,var"KernelThunkProto.NumWorkGroups"}}(nothing)
    invariant_arguments = PB.BufferedVector{Int64}()
    min_alignment = Ref{Union{Nothing,Int64Optional}}(nothing)
    arguments_buffers = PB.BufferedVector{xla.ShapedSliceProto}()
    results_buffers = PB.BufferedVector{xla.ShapedSliceProto}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            kernel_name = PB.decode(d, String)
        elseif field_number == 2
            PB.decode!(d, num_workgroups)
        elseif field_number == 3
            PB.decode!(d, wire_type, invariant_arguments)
        elseif field_number == 4
            PB.decode!(d, min_alignment)
        elseif field_number == 5
            PB.decode!(d, arguments_buffers)
        elseif field_number == 6
            PB.decode!(d, results_buffers)
        else
            Base.skip(d, wire_type)
        end
    end
    return KernelThunkProto(kernel_name, num_workgroups[], invariant_arguments[], min_alignment[], arguments_buffers[], results_buffers[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::KernelThunkProto)
    initpos = position(e.io)
    !isempty(x.kernel_name) && PB.encode(e, 1, x.kernel_name)
    !isnothing(x.num_workgroups) && PB.encode(e, 2, x.num_workgroups)
    !isempty(x.invariant_arguments) && PB.encode(e, 3, x.invariant_arguments)
    !isnothing(x.min_alignment) && PB.encode(e, 4, x.min_alignment)
    !isempty(x.arguments_buffers) && PB.encode(e, 5, x.arguments_buffers)
    !isempty(x.results_buffers) && PB.encode(e, 6, x.results_buffers)
    return position(e.io) - initpos
end
function PB._encoded_size(x::KernelThunkProto)
    encoded_size = 0
    !isempty(x.kernel_name) && (encoded_size += PB._encoded_size(x.kernel_name, 1))
    !isnothing(x.num_workgroups) && (encoded_size += PB._encoded_size(x.num_workgroups, 2))
    !isempty(x.invariant_arguments) && (encoded_size += PB._encoded_size(x.invariant_arguments, 3))
    !isnothing(x.min_alignment) && (encoded_size += PB._encoded_size(x.min_alignment, 4))
    !isempty(x.arguments_buffers) && (encoded_size += PB._encoded_size(x.arguments_buffers, 5))
    !isempty(x.results_buffers) && (encoded_size += PB._encoded_size(x.results_buffers, 6))
    return encoded_size
end

struct CollectivePermuteThunkProto
    source_target_pairs::Vector{var"CollectivePermuteThunkProto.SourceTargetPairProto"}
end
PB.default_values(::Type{CollectivePermuteThunkProto}) = (;source_target_pairs = Vector{var"CollectivePermuteThunkProto.SourceTargetPairProto"}())
PB.field_numbers(::Type{CollectivePermuteThunkProto}) = (;source_target_pairs = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CollectivePermuteThunkProto}, _endpos::Int=0, _group::Bool=false)
    source_target_pairs = PB.BufferedVector{var"CollectivePermuteThunkProto.SourceTargetPairProto"}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, source_target_pairs)
        else
            Base.skip(d, wire_type)
        end
    end
    return CollectivePermuteThunkProto(source_target_pairs[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CollectivePermuteThunkProto)
    initpos = position(e.io)
    !isempty(x.source_target_pairs) && PB.encode(e, 1, x.source_target_pairs)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CollectivePermuteThunkProto)
    encoded_size = 0
    !isempty(x.source_target_pairs) && (encoded_size += PB._encoded_size(x.source_target_pairs, 1))
    return encoded_size
end

struct ResourceOptional
    value::Union{Nothing,ResourceProto}
    contains_value::Bool
end
PB.default_values(::Type{ResourceOptional}) = (;value = nothing, contains_value = false)
PB.field_numbers(::Type{ResourceOptional}) = (;value = 1, contains_value = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ResourceOptional}, _endpos::Int=0, _group::Bool=false)
    value = Ref{Union{Nothing,ResourceProto}}(nothing)
    contains_value = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, value)
        elseif field_number == 2
            contains_value = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return ResourceOptional(value[], contains_value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ResourceOptional)
    initpos = position(e.io)
    !isnothing(x.value) && PB.encode(e, 1, x.value)
    x.contains_value != false && PB.encode(e, 2, x.contains_value)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ResourceOptional)
    encoded_size = 0
    !isnothing(x.value) && (encoded_size += PB._encoded_size(x.value, 1))
    x.contains_value != false && (encoded_size += PB._encoded_size(x.contains_value, 2))
    return encoded_size
end

struct var"ThunkSequenceProto.ResourceUsersProto"
    thunk_indices::Vector{Int64}
    resource::Union{Nothing,ResourceProto}
end
PB.default_values(::Type{var"ThunkSequenceProto.ResourceUsersProto"}) = (;thunk_indices = Vector{Int64}(), resource = nothing)
PB.field_numbers(::Type{var"ThunkSequenceProto.ResourceUsersProto"}) = (;thunk_indices = 1, resource = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"ThunkSequenceProto.ResourceUsersProto"}, _endpos::Int=0, _group::Bool=false)
    thunk_indices = PB.BufferedVector{Int64}()
    resource = Ref{Union{Nothing,ResourceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, thunk_indices)
        elseif field_number == 2
            PB.decode!(d, resource)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"ThunkSequenceProto.ResourceUsersProto"(thunk_indices[], resource[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"ThunkSequenceProto.ResourceUsersProto")
    initpos = position(e.io)
    !isempty(x.thunk_indices) && PB.encode(e, 1, x.thunk_indices)
    !isnothing(x.resource) && PB.encode(e, 2, x.resource)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"ThunkSequenceProto.ResourceUsersProto")
    encoded_size = 0
    !isempty(x.thunk_indices) && (encoded_size += PB._encoded_size(x.thunk_indices, 1))
    !isnothing(x.resource) && (encoded_size += PB._encoded_size(x.resource, 2))
    return encoded_size
end

struct SortThunkProto
    dimension::Int64
    is_stable::Bool
    direction::Union{Nothing,SortDirectionOptional}
    comparator_name::String
    inputs_shapes::Vector{ShapeBufferAllocationSliceProto}
end
PB.default_values(::Type{SortThunkProto}) = (;dimension = zero(Int64), is_stable = false, direction = nothing, comparator_name = "", inputs_shapes = Vector{ShapeBufferAllocationSliceProto}())
PB.field_numbers(::Type{SortThunkProto}) = (;dimension = 1, is_stable = 2, direction = 3, comparator_name = 4, inputs_shapes = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SortThunkProto}, _endpos::Int=0, _group::Bool=false)
    dimension = zero(Int64)
    is_stable = false
    direction = Ref{Union{Nothing,SortDirectionOptional}}(nothing)
    comparator_name = ""
    inputs_shapes = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            dimension = PB.decode(d, Int64)
        elseif field_number == 2
            is_stable = PB.decode(d, Bool)
        elseif field_number == 3
            PB.decode!(d, direction)
        elseif field_number == 4
            comparator_name = PB.decode(d, String)
        elseif field_number == 5
            PB.decode!(d, inputs_shapes)
        else
            Base.skip(d, wire_type)
        end
    end
    return SortThunkProto(dimension, is_stable, direction[], comparator_name, inputs_shapes[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SortThunkProto)
    initpos = position(e.io)
    x.dimension != zero(Int64) && PB.encode(e, 1, x.dimension)
    x.is_stable != false && PB.encode(e, 2, x.is_stable)
    !isnothing(x.direction) && PB.encode(e, 3, x.direction)
    !isempty(x.comparator_name) && PB.encode(e, 4, x.comparator_name)
    !isempty(x.inputs_shapes) && PB.encode(e, 5, x.inputs_shapes)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SortThunkProto)
    encoded_size = 0
    x.dimension != zero(Int64) && (encoded_size += PB._encoded_size(x.dimension, 1))
    x.is_stable != false && (encoded_size += PB._encoded_size(x.is_stable, 2))
    !isnothing(x.direction) && (encoded_size += PB._encoded_size(x.direction, 3))
    !isempty(x.comparator_name) && (encoded_size += PB._encoded_size(x.comparator_name, 4))
    !isempty(x.inputs_shapes) && (encoded_size += PB._encoded_size(x.inputs_shapes, 5))
    return encoded_size
end

struct XnnFusionThunkProto
    options::Union{Nothing,XnnFusionOptions}
    impl::Union{Nothing,OneOf{<:Union{XnnDotThunkProto,XnnConvolutionThunkProto,XnnFusionThunkProtoImpl}}}
end
PB.oneof_field_types(::Type{XnnFusionThunkProto}) = (;
    impl = (;xnn_dot_thunk=XnnDotThunkProto, xnn_convolution_thunk=XnnConvolutionThunkProto, xnn_fusion_thunk=XnnFusionThunkProtoImpl),
)
PB.default_values(::Type{XnnFusionThunkProto}) = (;options = nothing, xnn_dot_thunk = nothing, xnn_convolution_thunk = nothing, xnn_fusion_thunk = nothing)
PB.field_numbers(::Type{XnnFusionThunkProto}) = (;options = 1, xnn_dot_thunk = 2, xnn_convolution_thunk = 3, xnn_fusion_thunk = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:XnnFusionThunkProto}, _endpos::Int=0, _group::Bool=false)
    options = Ref{Union{Nothing,XnnFusionOptions}}(nothing)
    impl = nothing
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, options)
        elseif field_number == 2
            impl = OneOf(:xnn_dot_thunk, PB.decode(d, Ref{XnnDotThunkProto}))
        elseif field_number == 3
            impl = OneOf(:xnn_convolution_thunk, PB.decode(d, Ref{XnnConvolutionThunkProto}))
        elseif field_number == 4
            impl = OneOf(:xnn_fusion_thunk, PB.decode(d, Ref{XnnFusionThunkProtoImpl}))
        else
            Base.skip(d, wire_type)
        end
    end
    return XnnFusionThunkProto(options[], impl)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::XnnFusionThunkProto)
    initpos = position(e.io)
    !isnothing(x.options) && PB.encode(e, 1, x.options)
    if isnothing(x.impl);
    elseif x.impl.name === :xnn_dot_thunk
        PB.encode(e, 2, x.impl[]::XnnDotThunkProto)
    elseif x.impl.name === :xnn_convolution_thunk
        PB.encode(e, 3, x.impl[]::XnnConvolutionThunkProto)
    elseif x.impl.name === :xnn_fusion_thunk
        PB.encode(e, 4, x.impl[]::XnnFusionThunkProtoImpl)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::XnnFusionThunkProto)
    encoded_size = 0
    !isnothing(x.options) && (encoded_size += PB._encoded_size(x.options, 1))
    if isnothing(x.impl);
    elseif x.impl.name === :xnn_dot_thunk
        encoded_size += PB._encoded_size(x.impl[]::XnnDotThunkProto, 2)
    elseif x.impl.name === :xnn_convolution_thunk
        encoded_size += PB._encoded_size(x.impl[]::XnnConvolutionThunkProto, 3)
    elseif x.impl.name === :xnn_fusion_thunk
        encoded_size += PB._encoded_size(x.impl[]::XnnFusionThunkProtoImpl, 4)
    end
    return encoded_size
end

struct CustomCallThunkProto
    api_version::xla.CustomCallApiVersion.T
    target_name::String
    backend_config::String
    op_buffers::Union{Nothing,var"CustomCallThunkProto.OpBuffers"}
end
PB.default_values(::Type{CustomCallThunkProto}) = (;api_version = xla.CustomCallApiVersion.API_VERSION_UNSPECIFIED, target_name = "", backend_config = "", op_buffers = nothing)
PB.field_numbers(::Type{CustomCallThunkProto}) = (;api_version = 1, target_name = 2, backend_config = 3, op_buffers = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CustomCallThunkProto}, _endpos::Int=0, _group::Bool=false)
    api_version = xla.CustomCallApiVersion.API_VERSION_UNSPECIFIED
    target_name = ""
    backend_config = ""
    op_buffers = Ref{Union{Nothing,var"CustomCallThunkProto.OpBuffers"}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            api_version = PB.decode(d, xla.CustomCallApiVersion.T)
        elseif field_number == 2
            target_name = PB.decode(d, String)
        elseif field_number == 3
            backend_config = PB.decode(d, String)
        elseif field_number == 4
            PB.decode!(d, op_buffers)
        else
            Base.skip(d, wire_type)
        end
    end
    return CustomCallThunkProto(api_version, target_name, backend_config, op_buffers[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CustomCallThunkProto)
    initpos = position(e.io)
    x.api_version != xla.CustomCallApiVersion.API_VERSION_UNSPECIFIED && PB.encode(e, 1, x.api_version)
    !isempty(x.target_name) && PB.encode(e, 2, x.target_name)
    !isempty(x.backend_config) && PB.encode(e, 3, x.backend_config)
    !isnothing(x.op_buffers) && PB.encode(e, 4, x.op_buffers)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CustomCallThunkProto)
    encoded_size = 0
    x.api_version != xla.CustomCallApiVersion.API_VERSION_UNSPECIFIED && (encoded_size += PB._encoded_size(x.api_version, 1))
    !isempty(x.target_name) && (encoded_size += PB._encoded_size(x.target_name, 2))
    !isempty(x.backend_config) && (encoded_size += PB._encoded_size(x.backend_config, 3))
    !isnothing(x.op_buffers) && (encoded_size += PB._encoded_size(x.op_buffers, 4))
    return encoded_size
end

struct OpResourcesProto
    communicator_resource::Union{Nothing,ResourceOptional}
end
PB.default_values(::Type{OpResourcesProto}) = (;communicator_resource = nothing)
PB.field_numbers(::Type{OpResourcesProto}) = (;communicator_resource = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OpResourcesProto}, _endpos::Int=0, _group::Bool=false)
    communicator_resource = Ref{Union{Nothing,ResourceOptional}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, communicator_resource)
        else
            Base.skip(d, wire_type)
        end
    end
    return OpResourcesProto(communicator_resource[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OpResourcesProto)
    initpos = position(e.io)
    !isnothing(x.communicator_resource) && PB.encode(e, 1, x.communicator_resource)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OpResourcesProto)
    encoded_size = 0
    !isnothing(x.communicator_resource) && (encoded_size += PB._encoded_size(x.communicator_resource, 1))
    return encoded_size
end

struct var"OutfeedThunkProto.OutfeedResource"
    consume_token::Union{Nothing,ResourceOptional}
    produce_token::Union{Nothing,ResourceOptional}
end
PB.default_values(::Type{var"OutfeedThunkProto.OutfeedResource"}) = (;consume_token = nothing, produce_token = nothing)
PB.field_numbers(::Type{var"OutfeedThunkProto.OutfeedResource"}) = (;consume_token = 1, produce_token = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"OutfeedThunkProto.OutfeedResource"}, _endpos::Int=0, _group::Bool=false)
    consume_token = Ref{Union{Nothing,ResourceOptional}}(nothing)
    produce_token = Ref{Union{Nothing,ResourceOptional}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, consume_token)
        elseif field_number == 2
            PB.decode!(d, produce_token)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"OutfeedThunkProto.OutfeedResource"(consume_token[], produce_token[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"OutfeedThunkProto.OutfeedResource")
    initpos = position(e.io)
    !isnothing(x.consume_token) && PB.encode(e, 1, x.consume_token)
    !isnothing(x.produce_token) && PB.encode(e, 2, x.produce_token)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"OutfeedThunkProto.OutfeedResource")
    encoded_size = 0
    !isnothing(x.consume_token) && (encoded_size += PB._encoded_size(x.consume_token, 1))
    !isnothing(x.produce_token) && (encoded_size += PB._encoded_size(x.produce_token, 2))
    return encoded_size
end

struct var"InfeedThunkProto.InfeedResource"
    consume_token::Union{Nothing,ResourceOptional}
    produce_token::Union{Nothing,ResourceOptional}
end
PB.default_values(::Type{var"InfeedThunkProto.InfeedResource"}) = (;consume_token = nothing, produce_token = nothing)
PB.field_numbers(::Type{var"InfeedThunkProto.InfeedResource"}) = (;consume_token = 1, produce_token = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"InfeedThunkProto.InfeedResource"}, _endpos::Int=0, _group::Bool=false)
    consume_token = Ref{Union{Nothing,ResourceOptional}}(nothing)
    produce_token = Ref{Union{Nothing,ResourceOptional}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, consume_token)
        elseif field_number == 2
            PB.decode!(d, produce_token)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"InfeedThunkProto.InfeedResource"(consume_token[], produce_token[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"InfeedThunkProto.InfeedResource")
    initpos = position(e.io)
    !isnothing(x.consume_token) && PB.encode(e, 1, x.consume_token)
    !isnothing(x.produce_token) && PB.encode(e, 2, x.produce_token)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"InfeedThunkProto.InfeedResource")
    encoded_size = 0
    !isnothing(x.consume_token) && (encoded_size += PB._encoded_size(x.consume_token, 1))
    !isnothing(x.produce_token) && (encoded_size += PB._encoded_size(x.produce_token, 2))
    return encoded_size
end

struct CollectiveThunkProto
    op_params::Union{Nothing,OpParamsProto}
    op_buffers::Union{Nothing,OpBuffersProto}
    op_resources::Union{Nothing,OpResourcesProto}
    impl::Union{Nothing,OneOf{<:Union{AllGatherThunkProto,AllReduceThunkProto,AllToAllThunkProto,ReduceScatterThunkProto,CollectivePermuteThunkProto}}}
end
PB.oneof_field_types(::Type{CollectiveThunkProto}) = (;
    impl = (;all_gather_thunk=AllGatherThunkProto, all_reduce_thunk=AllReduceThunkProto, all_to_all_thunk=AllToAllThunkProto, reduce_scatter_thunk=ReduceScatterThunkProto, collective_permute_thunk=CollectivePermuteThunkProto),
)
PB.default_values(::Type{CollectiveThunkProto}) = (;op_params = nothing, op_buffers = nothing, op_resources = nothing, all_gather_thunk = nothing, all_reduce_thunk = nothing, all_to_all_thunk = nothing, reduce_scatter_thunk = nothing, collective_permute_thunk = nothing)
PB.field_numbers(::Type{CollectiveThunkProto}) = (;op_params = 1, op_buffers = 2, op_resources = 3, all_gather_thunk = 4, all_reduce_thunk = 5, all_to_all_thunk = 6, reduce_scatter_thunk = 7, collective_permute_thunk = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CollectiveThunkProto}, _endpos::Int=0, _group::Bool=false)
    op_params = Ref{Union{Nothing,OpParamsProto}}(nothing)
    op_buffers = Ref{Union{Nothing,OpBuffersProto}}(nothing)
    op_resources = Ref{Union{Nothing,OpResourcesProto}}(nothing)
    impl = nothing
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, op_params)
        elseif field_number == 2
            PB.decode!(d, op_buffers)
        elseif field_number == 3
            PB.decode!(d, op_resources)
        elseif field_number == 4
            impl = OneOf(:all_gather_thunk, PB.decode(d, Ref{AllGatherThunkProto}))
        elseif field_number == 5
            impl = OneOf(:all_reduce_thunk, PB.decode(d, Ref{AllReduceThunkProto}))
        elseif field_number == 6
            impl = OneOf(:all_to_all_thunk, PB.decode(d, Ref{AllToAllThunkProto}))
        elseif field_number == 7
            impl = OneOf(:reduce_scatter_thunk, PB.decode(d, Ref{ReduceScatterThunkProto}))
        elseif field_number == 8
            impl = OneOf(:collective_permute_thunk, PB.decode(d, Ref{CollectivePermuteThunkProto}))
        else
            Base.skip(d, wire_type)
        end
    end
    return CollectiveThunkProto(op_params[], op_buffers[], op_resources[], impl)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CollectiveThunkProto)
    initpos = position(e.io)
    !isnothing(x.op_params) && PB.encode(e, 1, x.op_params)
    !isnothing(x.op_buffers) && PB.encode(e, 2, x.op_buffers)
    !isnothing(x.op_resources) && PB.encode(e, 3, x.op_resources)
    if isnothing(x.impl);
    elseif x.impl.name === :all_gather_thunk
        PB.encode(e, 4, x.impl[]::AllGatherThunkProto)
    elseif x.impl.name === :all_reduce_thunk
        PB.encode(e, 5, x.impl[]::AllReduceThunkProto)
    elseif x.impl.name === :all_to_all_thunk
        PB.encode(e, 6, x.impl[]::AllToAllThunkProto)
    elseif x.impl.name === :reduce_scatter_thunk
        PB.encode(e, 7, x.impl[]::ReduceScatterThunkProto)
    elseif x.impl.name === :collective_permute_thunk
        PB.encode(e, 8, x.impl[]::CollectivePermuteThunkProto)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::CollectiveThunkProto)
    encoded_size = 0
    !isnothing(x.op_params) && (encoded_size += PB._encoded_size(x.op_params, 1))
    !isnothing(x.op_buffers) && (encoded_size += PB._encoded_size(x.op_buffers, 2))
    !isnothing(x.op_resources) && (encoded_size += PB._encoded_size(x.op_resources, 3))
    if isnothing(x.impl);
    elseif x.impl.name === :all_gather_thunk
        encoded_size += PB._encoded_size(x.impl[]::AllGatherThunkProto, 4)
    elseif x.impl.name === :all_reduce_thunk
        encoded_size += PB._encoded_size(x.impl[]::AllReduceThunkProto, 5)
    elseif x.impl.name === :all_to_all_thunk
        encoded_size += PB._encoded_size(x.impl[]::AllToAllThunkProto, 6)
    elseif x.impl.name === :reduce_scatter_thunk
        encoded_size += PB._encoded_size(x.impl[]::ReduceScatterThunkProto, 7)
    elseif x.impl.name === :collective_permute_thunk
        encoded_size += PB._encoded_size(x.impl[]::CollectivePermuteThunkProto, 8)
    end
    return encoded_size
end

struct OutfeedThunkProto
    outfeed_resources::Union{Nothing,var"OutfeedThunkProto.OutfeedResource"}
    outfeed_buffers_shapes::Vector{ShapeBufferAllocationSliceProto}
end
PB.default_values(::Type{OutfeedThunkProto}) = (;outfeed_resources = nothing, outfeed_buffers_shapes = Vector{ShapeBufferAllocationSliceProto}())
PB.field_numbers(::Type{OutfeedThunkProto}) = (;outfeed_resources = 1, outfeed_buffers_shapes = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OutfeedThunkProto}, _endpos::Int=0, _group::Bool=false)
    outfeed_resources = Ref{Union{Nothing,var"OutfeedThunkProto.OutfeedResource"}}(nothing)
    outfeed_buffers_shapes = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, outfeed_resources)
        elseif field_number == 2
            PB.decode!(d, outfeed_buffers_shapes)
        else
            Base.skip(d, wire_type)
        end
    end
    return OutfeedThunkProto(outfeed_resources[], outfeed_buffers_shapes[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OutfeedThunkProto)
    initpos = position(e.io)
    !isnothing(x.outfeed_resources) && PB.encode(e, 1, x.outfeed_resources)
    !isempty(x.outfeed_buffers_shapes) && PB.encode(e, 2, x.outfeed_buffers_shapes)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OutfeedThunkProto)
    encoded_size = 0
    !isnothing(x.outfeed_resources) && (encoded_size += PB._encoded_size(x.outfeed_resources, 1))
    !isempty(x.outfeed_buffers_shapes) && (encoded_size += PB._encoded_size(x.outfeed_buffers_shapes, 2))
    return encoded_size
end

struct InfeedThunkProto
    infeed_resources::Union{Nothing,var"InfeedThunkProto.InfeedResource"}
    infeed_buffers_shapes::Vector{ShapeBufferAllocationSliceProto}
end
PB.default_values(::Type{InfeedThunkProto}) = (;infeed_resources = nothing, infeed_buffers_shapes = Vector{ShapeBufferAllocationSliceProto}())
PB.field_numbers(::Type{InfeedThunkProto}) = (;infeed_resources = 1, infeed_buffers_shapes = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:InfeedThunkProto}, _endpos::Int=0, _group::Bool=false)
    infeed_resources = Ref{Union{Nothing,var"InfeedThunkProto.InfeedResource"}}(nothing)
    infeed_buffers_shapes = PB.BufferedVector{ShapeBufferAllocationSliceProto}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, infeed_resources)
        elseif field_number == 2
            PB.decode!(d, infeed_buffers_shapes)
        else
            Base.skip(d, wire_type)
        end
    end
    return InfeedThunkProto(infeed_resources[], infeed_buffers_shapes[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::InfeedThunkProto)
    initpos = position(e.io)
    !isnothing(x.infeed_resources) && PB.encode(e, 1, x.infeed_resources)
    !isempty(x.infeed_buffers_shapes) && PB.encode(e, 2, x.infeed_buffers_shapes)
    return position(e.io) - initpos
end
function PB._encoded_size(x::InfeedThunkProto)
    encoded_size = 0
    !isnothing(x.infeed_resources) && (encoded_size += PB._encoded_size(x.infeed_resources, 1))
    !isempty(x.infeed_buffers_shapes) && (encoded_size += PB._encoded_size(x.infeed_buffers_shapes, 2))
    return encoded_size
end

# Stub definitions for cyclic types
struct var"##Stub#CallThunkProto"{T1<:var"##Abstract#ThunkSequenceProto"} <: var"##Abstract#CallThunkProto"
    called_sequence::Union{Nothing,T1}
end

struct var"##Stub#ConditionalThunkProto"{T1<:var"##Abstract#ThunkSequenceProto"} <: var"##Abstract#ConditionalThunkProto"
    branch_sequences::Vector{T1}
    branch_index_buffer::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
end

struct var"##Stub#ThunkSequenceProto"{T1<:var"##Abstract#ThunkProto"} <: var"##Abstract#ThunkSequenceProto"
    thunks::Vector{T1}
    thunk_resources::Vector{var"ThunkSequenceProto.ResourceUsersProto"}
end

struct var"##Stub#WhileThunkProto"{T1<:var"##Abstract#ThunkProto"} <: var"##Abstract#WhileThunkProto"
    cond_sequence::Union{Nothing,var"##Stub#ThunkSequenceProto"{T1}}
    body_sequence::Union{Nothing,var"##Stub#ThunkSequenceProto"{T1}}
    trip_count::Union{Nothing,Int64Optional}
    cond_buffer::Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}
end

struct var"##Stub#ThunkProto" <: var"##Abstract#ThunkProto"
    kind::String
    info::Union{Nothing,InfoProto}
    impl::Union{Nothing,OneOf{<:Union{var"##Stub#CallThunkProto"{var"##Stub#ThunkSequenceProto"{var"##Stub#ThunkProto"}},var"##Stub#ConditionalThunkProto"{var"##Stub#ThunkSequenceProto"{var"##Stub#ThunkProto"}},SortThunkProto,XnnFusionThunkProto,DotThunkProto,RngGetAndUpdateStateThunkProto,TopKThunkProto,var"##Stub#WhileThunkProto"{var"##Stub#ThunkProto"},KernelThunkProto,CopyThunkProto,FftThunkProto,InfeedThunkProto,OutfeedThunkProto,CustomCallThunkProto,ConvolutionThunkProto,CollectiveThunkProto,PartitionIdThunkProto,ReplicaIdThunkProto,YnnFusionThunkProto,RngSeedThunkProto}}}
end

const CallThunkProto = var"##Stub#CallThunkProto"{var"##Stub#ThunkSequenceProto"{var"##Stub#ThunkProto"}}
PB.default_values(::Type{CallThunkProto}) = (;called_sequence = nothing)
PB.field_numbers(::Type{CallThunkProto}) = (;called_sequence = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CallThunkProto}, _endpos::Int=0, _group::Bool=false)
    called_sequence = Ref{Union{Nothing,ThunkSequenceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, called_sequence)
        else
            Base.skip(d, wire_type)
        end
    end
    return CallThunkProto(called_sequence[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CallThunkProto)
    initpos = position(e.io)
    !isnothing(x.called_sequence) && PB.encode(e, 1, x.called_sequence)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CallThunkProto)
    encoded_size = 0
    !isnothing(x.called_sequence) && (encoded_size += PB._encoded_size(x.called_sequence, 1))
    return encoded_size
end

const ConditionalThunkProto = var"##Stub#ConditionalThunkProto"{var"##Stub#ThunkSequenceProto"{var"##Stub#ThunkProto"}}
PB.default_values(::Type{ConditionalThunkProto}) = (;branch_sequences = Vector{ThunkSequenceProto}(), branch_index_buffer = nothing)
PB.field_numbers(::Type{ConditionalThunkProto}) = (;branch_sequences = 1, branch_index_buffer = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ConditionalThunkProto}, _endpos::Int=0, _group::Bool=false)
    branch_sequences = PB.BufferedVector{ThunkSequenceProto}()
    branch_index_buffer = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, branch_sequences)
        elseif field_number == 2
            PB.decode!(d, branch_index_buffer)
        else
            Base.skip(d, wire_type)
        end
    end
    return ConditionalThunkProto(branch_sequences[], branch_index_buffer[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ConditionalThunkProto)
    initpos = position(e.io)
    !isempty(x.branch_sequences) && PB.encode(e, 1, x.branch_sequences)
    !isnothing(x.branch_index_buffer) && PB.encode(e, 2, x.branch_index_buffer)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ConditionalThunkProto)
    encoded_size = 0
    !isempty(x.branch_sequences) && (encoded_size += PB._encoded_size(x.branch_sequences, 1))
    !isnothing(x.branch_index_buffer) && (encoded_size += PB._encoded_size(x.branch_index_buffer, 2))
    return encoded_size
end

const ThunkSequenceProto = var"##Stub#ThunkSequenceProto"{var"##Stub#ThunkProto"}
PB.default_values(::Type{ThunkSequenceProto}) = (;thunks = Vector{ThunkProto}(), thunk_resources = Vector{var"ThunkSequenceProto.ResourceUsersProto"}())
PB.field_numbers(::Type{ThunkSequenceProto}) = (;thunks = 1, thunk_resources = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ThunkSequenceProto}, _endpos::Int=0, _group::Bool=false)
    thunks = PB.BufferedVector{ThunkProto}()
    thunk_resources = PB.BufferedVector{var"ThunkSequenceProto.ResourceUsersProto"}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, thunks)
        elseif field_number == 2
            PB.decode!(d, thunk_resources)
        else
            Base.skip(d, wire_type)
        end
    end
    return ThunkSequenceProto(thunks[], thunk_resources[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ThunkSequenceProto)
    initpos = position(e.io)
    !isempty(x.thunks) && PB.encode(e, 1, x.thunks)
    !isempty(x.thunk_resources) && PB.encode(e, 2, x.thunk_resources)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ThunkSequenceProto)
    encoded_size = 0
    !isempty(x.thunks) && (encoded_size += PB._encoded_size(x.thunks, 1))
    !isempty(x.thunk_resources) && (encoded_size += PB._encoded_size(x.thunk_resources, 2))
    return encoded_size
end

const WhileThunkProto = var"##Stub#WhileThunkProto"{var"##Stub#ThunkProto"}
PB.default_values(::Type{WhileThunkProto}) = (;cond_sequence = nothing, body_sequence = nothing, trip_count = nothing, cond_buffer = nothing)
PB.field_numbers(::Type{WhileThunkProto}) = (;cond_sequence = 1, body_sequence = 2, trip_count = 3, cond_buffer = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:WhileThunkProto}, _endpos::Int=0, _group::Bool=false)
    cond_sequence = Ref{Union{Nothing,ThunkSequenceProto}}(nothing)
    body_sequence = Ref{Union{Nothing,ThunkSequenceProto}}(nothing)
    trip_count = Ref{Union{Nothing,Int64Optional}}(nothing)
    cond_buffer = Ref{Union{Nothing,xla.buffer_assignment.BufferAllocationSliceProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, cond_sequence)
        elseif field_number == 2
            PB.decode!(d, body_sequence)
        elseif field_number == 3
            PB.decode!(d, trip_count)
        elseif field_number == 4
            PB.decode!(d, cond_buffer)
        else
            Base.skip(d, wire_type)
        end
    end
    return WhileThunkProto(cond_sequence[], body_sequence[], trip_count[], cond_buffer[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::WhileThunkProto)
    initpos = position(e.io)
    !isnothing(x.cond_sequence) && PB.encode(e, 1, x.cond_sequence)
    !isnothing(x.body_sequence) && PB.encode(e, 2, x.body_sequence)
    !isnothing(x.trip_count) && PB.encode(e, 3, x.trip_count)
    !isnothing(x.cond_buffer) && PB.encode(e, 4, x.cond_buffer)
    return position(e.io) - initpos
end
function PB._encoded_size(x::WhileThunkProto)
    encoded_size = 0
    !isnothing(x.cond_sequence) && (encoded_size += PB._encoded_size(x.cond_sequence, 1))
    !isnothing(x.body_sequence) && (encoded_size += PB._encoded_size(x.body_sequence, 2))
    !isnothing(x.trip_count) && (encoded_size += PB._encoded_size(x.trip_count, 3))
    !isnothing(x.cond_buffer) && (encoded_size += PB._encoded_size(x.cond_buffer, 4))
    return encoded_size
end

const ThunkProto = var"##Stub#ThunkProto"
PB.oneof_field_types(::Type{ThunkProto}) = (;
    impl = (;call_thunk=CallThunkProto, conditional_thunk=ConditionalThunkProto, sort_thunk=SortThunkProto, xnn_fusion_thunk=XnnFusionThunkProto, dot_thunk=DotThunkProto, rng_get_and_update_state_thunk=RngGetAndUpdateStateThunkProto, top_k_thunk=TopKThunkProto, while_thunk=WhileThunkProto, kernel_thunk=KernelThunkProto, copy_thunk=CopyThunkProto, fft_thunk=FftThunkProto, infeed_thunk=InfeedThunkProto, outfeed_thunk=OutfeedThunkProto, custom_call_thunk=CustomCallThunkProto, convolution_thunk=ConvolutionThunkProto, collective_thunk=CollectiveThunkProto, partition_id_thunk=PartitionIdThunkProto, replica_id_thunk=ReplicaIdThunkProto, ynn_fusion_thunk=YnnFusionThunkProto, rng_seed_thunk=RngSeedThunkProto),
)
PB.default_values(::Type{ThunkProto}) = (;kind = "", info = nothing, call_thunk = nothing, conditional_thunk = nothing, sort_thunk = nothing, xnn_fusion_thunk = nothing, dot_thunk = nothing, rng_get_and_update_state_thunk = nothing, top_k_thunk = nothing, while_thunk = nothing, kernel_thunk = nothing, copy_thunk = nothing, fft_thunk = nothing, infeed_thunk = nothing, outfeed_thunk = nothing, custom_call_thunk = nothing, convolution_thunk = nothing, collective_thunk = nothing, partition_id_thunk = nothing, replica_id_thunk = nothing, ynn_fusion_thunk = nothing, rng_seed_thunk = nothing)
PB.field_numbers(::Type{ThunkProto}) = (;kind = 1, info = 2, call_thunk = 3, conditional_thunk = 4, sort_thunk = 5, xnn_fusion_thunk = 6, dot_thunk = 7, rng_get_and_update_state_thunk = 8, top_k_thunk = 9, while_thunk = 10, kernel_thunk = 11, copy_thunk = 12, fft_thunk = 13, infeed_thunk = 14, outfeed_thunk = 15, custom_call_thunk = 16, convolution_thunk = 17, collective_thunk = 18, partition_id_thunk = 19, replica_id_thunk = 20, ynn_fusion_thunk = 21, rng_seed_thunk = 22)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ThunkProto}, _endpos::Int=0, _group::Bool=false)
    kind = ""
    info = Ref{Union{Nothing,InfoProto}}(nothing)
    impl = nothing
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            kind = PB.decode(d, String)
        elseif field_number == 2
            PB.decode!(d, info)
        elseif field_number == 3
            impl = OneOf(:call_thunk, PB.decode(d, Ref{CallThunkProto}))
        elseif field_number == 4
            impl = OneOf(:conditional_thunk, PB.decode(d, Ref{ConditionalThunkProto}))
        elseif field_number == 5
            impl = OneOf(:sort_thunk, PB.decode(d, Ref{SortThunkProto}))
        elseif field_number == 6
            impl = OneOf(:xnn_fusion_thunk, PB.decode(d, Ref{XnnFusionThunkProto}))
        elseif field_number == 7
            impl = OneOf(:dot_thunk, PB.decode(d, Ref{DotThunkProto}))
        elseif field_number == 8
            impl = OneOf(:rng_get_and_update_state_thunk, PB.decode(d, Ref{RngGetAndUpdateStateThunkProto}))
        elseif field_number == 9
            impl = OneOf(:top_k_thunk, PB.decode(d, Ref{TopKThunkProto}))
        elseif field_number == 10
            impl = OneOf(:while_thunk, PB.decode(d, Ref{WhileThunkProto}))
        elseif field_number == 11
            impl = OneOf(:kernel_thunk, PB.decode(d, Ref{KernelThunkProto}))
        elseif field_number == 12
            impl = OneOf(:copy_thunk, PB.decode(d, Ref{CopyThunkProto}))
        elseif field_number == 13
            impl = OneOf(:fft_thunk, PB.decode(d, Ref{FftThunkProto}))
        elseif field_number == 14
            impl = OneOf(:infeed_thunk, PB.decode(d, Ref{InfeedThunkProto}))
        elseif field_number == 15
            impl = OneOf(:outfeed_thunk, PB.decode(d, Ref{OutfeedThunkProto}))
        elseif field_number == 16
            impl = OneOf(:custom_call_thunk, PB.decode(d, Ref{CustomCallThunkProto}))
        elseif field_number == 17
            impl = OneOf(:convolution_thunk, PB.decode(d, Ref{ConvolutionThunkProto}))
        elseif field_number == 18
            impl = OneOf(:collective_thunk, PB.decode(d, Ref{CollectiveThunkProto}))
        elseif field_number == 19
            impl = OneOf(:partition_id_thunk, PB.decode(d, Ref{PartitionIdThunkProto}))
        elseif field_number == 20
            impl = OneOf(:replica_id_thunk, PB.decode(d, Ref{ReplicaIdThunkProto}))
        elseif field_number == 21
            impl = OneOf(:ynn_fusion_thunk, PB.decode(d, Ref{YnnFusionThunkProto}))
        elseif field_number == 22
            impl = OneOf(:rng_seed_thunk, PB.decode(d, Ref{RngSeedThunkProto}))
        else
            Base.skip(d, wire_type)
        end
    end
    return ThunkProto(kind, info[], impl)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ThunkProto)
    initpos = position(e.io)
    !isempty(x.kind) && PB.encode(e, 1, x.kind)
    !isnothing(x.info) && PB.encode(e, 2, x.info)
    if isnothing(x.impl);
    elseif x.impl.name === :call_thunk
        PB.encode(e, 3, x.impl[]::CallThunkProto)
    elseif x.impl.name === :conditional_thunk
        PB.encode(e, 4, x.impl[]::ConditionalThunkProto)
    elseif x.impl.name === :sort_thunk
        PB.encode(e, 5, x.impl[]::SortThunkProto)
    elseif x.impl.name === :xnn_fusion_thunk
        PB.encode(e, 6, x.impl[]::XnnFusionThunkProto)
    elseif x.impl.name === :dot_thunk
        PB.encode(e, 7, x.impl[]::DotThunkProto)
    elseif x.impl.name === :rng_get_and_update_state_thunk
        PB.encode(e, 8, x.impl[]::RngGetAndUpdateStateThunkProto)
    elseif x.impl.name === :top_k_thunk
        PB.encode(e, 9, x.impl[]::TopKThunkProto)
    elseif x.impl.name === :while_thunk
        PB.encode(e, 10, x.impl[]::WhileThunkProto)
    elseif x.impl.name === :kernel_thunk
        PB.encode(e, 11, x.impl[]::KernelThunkProto)
    elseif x.impl.name === :copy_thunk
        PB.encode(e, 12, x.impl[]::CopyThunkProto)
    elseif x.impl.name === :fft_thunk
        PB.encode(e, 13, x.impl[]::FftThunkProto)
    elseif x.impl.name === :infeed_thunk
        PB.encode(e, 14, x.impl[]::InfeedThunkProto)
    elseif x.impl.name === :outfeed_thunk
        PB.encode(e, 15, x.impl[]::OutfeedThunkProto)
    elseif x.impl.name === :custom_call_thunk
        PB.encode(e, 16, x.impl[]::CustomCallThunkProto)
    elseif x.impl.name === :convolution_thunk
        PB.encode(e, 17, x.impl[]::ConvolutionThunkProto)
    elseif x.impl.name === :collective_thunk
        PB.encode(e, 18, x.impl[]::CollectiveThunkProto)
    elseif x.impl.name === :partition_id_thunk
        PB.encode(e, 19, x.impl[]::PartitionIdThunkProto)
    elseif x.impl.name === :replica_id_thunk
        PB.encode(e, 20, x.impl[]::ReplicaIdThunkProto)
    elseif x.impl.name === :ynn_fusion_thunk
        PB.encode(e, 21, x.impl[]::YnnFusionThunkProto)
    elseif x.impl.name === :rng_seed_thunk
        PB.encode(e, 22, x.impl[]::RngSeedThunkProto)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::ThunkProto)
    encoded_size = 0
    !isempty(x.kind) && (encoded_size += PB._encoded_size(x.kind, 1))
    !isnothing(x.info) && (encoded_size += PB._encoded_size(x.info, 2))
    if isnothing(x.impl);
    elseif x.impl.name === :call_thunk
        encoded_size += PB._encoded_size(x.impl[]::CallThunkProto, 3)
    elseif x.impl.name === :conditional_thunk
        encoded_size += PB._encoded_size(x.impl[]::ConditionalThunkProto, 4)
    elseif x.impl.name === :sort_thunk
        encoded_size += PB._encoded_size(x.impl[]::SortThunkProto, 5)
    elseif x.impl.name === :xnn_fusion_thunk
        encoded_size += PB._encoded_size(x.impl[]::XnnFusionThunkProto, 6)
    elseif x.impl.name === :dot_thunk
        encoded_size += PB._encoded_size(x.impl[]::DotThunkProto, 7)
    elseif x.impl.name === :rng_get_and_update_state_thunk
        encoded_size += PB._encoded_size(x.impl[]::RngGetAndUpdateStateThunkProto, 8)
    elseif x.impl.name === :top_k_thunk
        encoded_size += PB._encoded_size(x.impl[]::TopKThunkProto, 9)
    elseif x.impl.name === :while_thunk
        encoded_size += PB._encoded_size(x.impl[]::WhileThunkProto, 10)
    elseif x.impl.name === :kernel_thunk
        encoded_size += PB._encoded_size(x.impl[]::KernelThunkProto, 11)
    elseif x.impl.name === :copy_thunk
        encoded_size += PB._encoded_size(x.impl[]::CopyThunkProto, 12)
    elseif x.impl.name === :fft_thunk
        encoded_size += PB._encoded_size(x.impl[]::FftThunkProto, 13)
    elseif x.impl.name === :infeed_thunk
        encoded_size += PB._encoded_size(x.impl[]::InfeedThunkProto, 14)
    elseif x.impl.name === :outfeed_thunk
        encoded_size += PB._encoded_size(x.impl[]::OutfeedThunkProto, 15)
    elseif x.impl.name === :custom_call_thunk
        encoded_size += PB._encoded_size(x.impl[]::CustomCallThunkProto, 16)
    elseif x.impl.name === :convolution_thunk
        encoded_size += PB._encoded_size(x.impl[]::ConvolutionThunkProto, 17)
    elseif x.impl.name === :collective_thunk
        encoded_size += PB._encoded_size(x.impl[]::CollectiveThunkProto, 18)
    elseif x.impl.name === :partition_id_thunk
        encoded_size += PB._encoded_size(x.impl[]::PartitionIdThunkProto, 19)
    elseif x.impl.name === :replica_id_thunk
        encoded_size += PB._encoded_size(x.impl[]::ReplicaIdThunkProto, 20)
    elseif x.impl.name === :ynn_fusion_thunk
        encoded_size += PB._encoded_size(x.impl[]::YnnFusionThunkProto, 21)
    elseif x.impl.name === :rng_seed_thunk
        encoded_size += PB._encoded_size(x.impl[]::RngSeedThunkProto, 22)
    end
    return encoded_size
end
