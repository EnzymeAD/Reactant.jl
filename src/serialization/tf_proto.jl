# Minimal TensorFlow SavedModel & TensorBundle Protocol Buffer encoders.
# Schema references:
#   tensorflow/core/framework/{types,tensor_shape,versions,resource_handle,tensor,attr_value,node_def,op_def,function,graph}.proto
#   tensorflow/core/protobuf/{saver,meta_graph,saved_model,saved_object_graph,struct,trackable_object_graph,tensor_bundle}.proto

module TFProto

using ProtoBuf: OneOf
import ProtoBuf.Codecs: AbstractProtoEncoder, ProtoEncoder, encode, _encoded_size

function proto_to_bytes(msg)::Vector{UInt8}
    io = IOBuffer()
    encode(ProtoEncoder(io), msg)
    return take!(io)
end

@enum TFDataType::Int32 begin
    DT_INVALID = 0
    DT_FLOAT = 1
    DT_DOUBLE = 2
    DT_INT32 = 3
    DT_UINT8 = 4
    DT_INT16 = 5
    DT_INT8 = 6
    DT_STRING = 7
    DT_COMPLEX64 = 8
    DT_INT64 = 9
    DT_BOOL = 10
    DT_BFLOAT16 = 14
    DT_UINT16 = 17
    DT_COMPLEX128 = 18
    DT_HALF = 19
    DT_RESOURCE = 20
    DT_UINT32 = 22
    DT_UINT64 = 23
end

@enum CheckpointFormatVersion::Int32 begin
    SAVER_LEGACY = 0
    SAVER_V1 = 1
    SAVER_V2 = 2
end

@enum BundleEndianness::Int32 begin
    BUNDLE_LITTLE = 0
    BUNDLE_BIG = 1
end

# google.protobuf.BoolValue
struct BoolValue
    value::Bool
end

function encode(e::AbstractProtoEncoder, x::BoolValue)
    initpos = position(e.io)
    x.value && encode(e, 1, x.value)
    return position(e.io) - initpos
end

function _encoded_size(x::BoolValue)
    encoded_size = 0
    x.value && (encoded_size += _encoded_size(x.value, 1))
    return encoded_size
end

# tensorflow.VersionDef
struct VersionDef
    producer::Int32
    min_consumer::Int32
    bad_consumers::Vector{Int32}
end

function VersionDef(;
    producer::Integer=0, min_consumer::Integer=0, bad_consumers::Vector{Int32}=Int32[]
)
    return VersionDef(Int32(producer), Int32(min_consumer), bad_consumers)
end

function encode(e::AbstractProtoEncoder, x::VersionDef)
    initpos = position(e.io)
    x.producer != zero(Int32) && encode(e, 1, x.producer)
    x.min_consumer != zero(Int32) && encode(e, 2, x.min_consumer)
    !isempty(x.bad_consumers) && encode(e, 3, x.bad_consumers)
    return position(e.io) - initpos
end

function _encoded_size(x::VersionDef)
    encoded_size = 0
    x.producer != zero(Int32) && (encoded_size += _encoded_size(x.producer, 1))
    x.min_consumer != zero(Int32) && (encoded_size += _encoded_size(x.min_consumer, 2))
    !isempty(x.bad_consumers) && (encoded_size += _encoded_size(x.bad_consumers, 3))
    return encoded_size
end

# tensorflow.TensorShapeProto
struct TensorShapeDim
    size::Int64
    name::String
end

TensorShapeDim(size::Integer) = TensorShapeDim(Int64(size), "")

function encode(e::AbstractProtoEncoder, x::TensorShapeDim)
    initpos = position(e.io)
    x.size != zero(Int64) && encode(e, 1, x.size)
    !isempty(x.name) && encode(e, 2, x.name)
    return position(e.io) - initpos
end

function _encoded_size(x::TensorShapeDim)
    encoded_size = 0
    x.size != zero(Int64) && (encoded_size += _encoded_size(x.size, 1))
    !isempty(x.name) && (encoded_size += _encoded_size(x.name, 2))
    return encoded_size
end

struct TensorShapeProto
    dim::Vector{TensorShapeDim}
    unknown_rank::Bool
end

function TensorShapeProto(dims::AbstractVector{<:Integer})
    return TensorShapeProto([TensorShapeDim(d) for d in dims], false)
end

function TensorShapeProto(; unknown_rank::Bool=false)
    return TensorShapeProto(TensorShapeDim[], unknown_rank)
end

function encode(e::AbstractProtoEncoder, x::TensorShapeProto)
    initpos = position(e.io)
    !isempty(x.dim) && encode(e, 2, x.dim)
    x.unknown_rank && encode(e, 3, x.unknown_rank)
    return position(e.io) - initpos
end

function _encoded_size(x::TensorShapeProto)
    encoded_size = 0
    !isempty(x.dim) && (encoded_size += _encoded_size(x.dim, 2))
    x.unknown_rank && (encoded_size += _encoded_size(x.unknown_rank, 3))
    return encoded_size
end

# tensorflow.ResourceHandleProto.DtypeAndShape
struct ResourceDtypeAndShape
    dtype::TFDataType
    shape::Union{Nothing,TensorShapeProto}
end

function encode(e::AbstractProtoEncoder, x::ResourceDtypeAndShape)
    initpos = position(e.io)
    x.dtype != DT_INVALID && encode(e, 1, x.dtype)
    !isnothing(x.shape) && encode(e, 2, x.shape)
    return position(e.io) - initpos
end

function _encoded_size(x::ResourceDtypeAndShape)
    encoded_size = 0
    x.dtype != DT_INVALID && (encoded_size += _encoded_size(x.dtype, 1))
    !isnothing(x.shape) && (encoded_size += _encoded_size(x.shape, 2))
    return encoded_size
end

# tensorflow.TensorProto (subset needed for SavedModel constants)
struct TensorProto
    dtype::TFDataType
    tensor_shape::Union{Nothing,TensorShapeProto}
    tensor_content::Vector{UInt8}
    string_val::Vector{Vector{UInt8}}
end

function TensorProto(;
    dtype::TFDataType=DT_INVALID,
    tensor_shape::Union{Nothing,TensorShapeProto}=nothing,
    tensor_content::Vector{UInt8}=UInt8[],
    string_val::Vector{Vector{UInt8}}=Vector{UInt8}[],
)
    return TensorProto(dtype, tensor_shape, tensor_content, string_val)
end

function encode(e::AbstractProtoEncoder, x::TensorProto)
    initpos = position(e.io)
    x.dtype != DT_INVALID && encode(e, 1, x.dtype)
    !isnothing(x.tensor_shape) && encode(e, 2, x.tensor_shape)
    !isempty(x.tensor_content) && encode(e, 4, x.tensor_content)
    !isempty(x.string_val) && encode(e, 8, x.string_val)
    return position(e.io) - initpos
end

function _encoded_size(x::TensorProto)
    encoded_size = 0
    x.dtype != DT_INVALID && (encoded_size += _encoded_size(x.dtype, 1))
    !isnothing(x.tensor_shape) && (encoded_size += _encoded_size(x.tensor_shape, 2))
    !isempty(x.tensor_content) && (encoded_size += _encoded_size(x.tensor_content, 4))
    !isempty(x.string_val) && (encoded_size += _encoded_size(x.string_val, 8))
    return encoded_size
end

# tensorflow.AttrValue, AttrValue.ListValue, NameAttrList
struct AttrValue
    value::Union{Nothing,OneOf}
end

struct NameAttrList
    name::String
    attr::Dict{String,AttrValue}
end

function NameAttrList(name::String)
    return NameAttrList(name, Dict{String,AttrValue}())
end

struct AttrListValue
    s::Vector{Vector{UInt8}}
    i::Vector{Int64}
    f::Vector{Float32}
    b::Vector{Bool}
    type::Vector{TFDataType}
    shape::Vector{TensorShapeProto}
    tensor::Vector{TensorProto}
    func::Vector{NameAttrList}
end

function AttrListValue(;
    s::Vector{Vector{UInt8}}=Vector{UInt8}[],
    i::Vector{Int64}=Int64[],
    f::Vector{Float32}=Float32[],
    b::Vector{Bool}=Bool[],
    type::Vector{TFDataType}=TFDataType[],
    shape::Vector{TensorShapeProto}=TensorShapeProto[],
    tensor::Vector{TensorProto}=TensorProto[],
    func::Vector{NameAttrList}=NameAttrList[],
)
    return AttrListValue(s, i, f, b, type, shape, tensor, func)
end

function encode(e::AbstractProtoEncoder, x::NameAttrList)
    initpos = position(e.io)
    !isempty(x.name) && encode(e, 1, x.name)
    !isempty(x.attr) && encode(e, 2, x.attr)
    return position(e.io) - initpos
end

function _encoded_size(x::NameAttrList)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += _encoded_size(x.name, 1))
    !isempty(x.attr) && (encoded_size += _encoded_size(x.attr, 2))
    return encoded_size
end

function encode(e::AbstractProtoEncoder, x::AttrListValue)
    initpos = position(e.io)
    !isempty(x.s) && encode(e, 2, x.s)
    !isempty(x.i) && encode(e, 3, x.i)
    !isempty(x.f) && encode(e, 4, x.f)
    !isempty(x.b) && encode(e, 5, x.b)
    !isempty(x.type) && encode(e, 6, x.type)
    !isempty(x.shape) && encode(e, 7, x.shape)
    !isempty(x.tensor) && encode(e, 8, x.tensor)
    !isempty(x.func) && encode(e, 9, x.func)
    return position(e.io) - initpos
end

function _encoded_size(x::AttrListValue)
    encoded_size = 0
    !isempty(x.s) && (encoded_size += _encoded_size(x.s, 2))
    !isempty(x.i) && (encoded_size += _encoded_size(x.i, 3))
    !isempty(x.f) && (encoded_size += _encoded_size(x.f, 4))
    !isempty(x.b) && (encoded_size += _encoded_size(x.b, 5))
    !isempty(x.type) && (encoded_size += _encoded_size(x.type, 6))
    !isempty(x.shape) && (encoded_size += _encoded_size(x.shape, 7))
    !isempty(x.tensor) && (encoded_size += _encoded_size(x.tensor, 8))
    !isempty(x.func) && (encoded_size += _encoded_size(x.func, 9))
    return encoded_size
end

function encode(e::AbstractProtoEncoder, x::AttrValue)
    initpos = position(e.io)
    if isnothing(x.value)
        nothing
    elseif x.value.name === :list
        encode(e, 1, x.value[]::AttrListValue)
    elseif x.value.name === :s
        encode(e, 2, x.value[]::Vector{UInt8})
    elseif x.value.name === :i
        encode(e, 3, x.value[]::Int64)
    elseif x.value.name === :f
        encode(e, 4, x.value[]::Float32)
    elseif x.value.name === :b
        encode(e, 5, x.value[]::Bool)
    elseif x.value.name === :type
        encode(e, 6, x.value[]::TFDataType)
    elseif x.value.name === :shape
        encode(e, 7, x.value[]::TensorShapeProto)
    elseif x.value.name === :tensor
        encode(e, 8, x.value[]::TensorProto)
    elseif x.value.name === :placeholder
        encode(e, 9, x.value[]::String)
    elseif x.value.name === :func
        encode(e, 10, x.value[]::NameAttrList)
    else
        error("Unsupported AttrValue oneof field: $(x.value.name)")
    end
    return position(e.io) - initpos
end

function _encoded_size(x::AttrValue)
    encoded_size = 0
    if isnothing(x.value)
        nothing
    elseif x.value.name === :list
        encoded_size += _encoded_size(x.value[]::AttrListValue, 1)
    elseif x.value.name === :s
        encoded_size += _encoded_size(x.value[]::Vector{UInt8}, 2)
    elseif x.value.name === :i
        encoded_size += _encoded_size(x.value[]::Int64, 3)
    elseif x.value.name === :f
        encoded_size += _encoded_size(x.value[]::Float32, 4)
    elseif x.value.name === :b
        encoded_size += _encoded_size(x.value[]::Bool, 5)
    elseif x.value.name === :type
        encoded_size += _encoded_size(x.value[]::TFDataType, 6)
    elseif x.value.name === :shape
        encoded_size += _encoded_size(x.value[]::TensorShapeProto, 7)
    elseif x.value.name === :tensor
        encoded_size += _encoded_size(x.value[]::TensorProto, 8)
    elseif x.value.name === :placeholder
        encoded_size += _encoded_size(x.value[]::String, 9)
    elseif x.value.name === :func
        encoded_size += _encoded_size(x.value[]::NameAttrList, 10)
    else
        error("Unsupported AttrValue oneof field: $(x.value.name)")
    end
    return encoded_size
end

attr_s(s::AbstractString) = AttrValue(OneOf(:s, Vector{UInt8}(codeunits(String(s)))))
attr_bytes(b::AbstractVector{UInt8}) = AttrValue(OneOf(:s, Vector{UInt8}(b)))
attr_i(i::Integer) = AttrValue(OneOf(:i, Int64(i)))
attr_b(b::Bool) = AttrValue(OneOf(:b, b))
attr_type(t::TFDataType) = AttrValue(OneOf(:type, t))
attr_shape(s::TensorShapeProto) = AttrValue(OneOf(:shape, s))
attr_tensor(t::TensorProto) = AttrValue(OneOf(:tensor, t))
attr_func(f::NameAttrList) = AttrValue(OneOf(:func, f))
attr_list(l::AttrListValue) = AttrValue(OneOf(:list, l))

# tensorflow.NodeDef
struct NodeDef
    name::String
    op::String
    input::Vector{String}
    device::String
    attr::Dict{String,AttrValue}
end

function NodeDef(;
    name::String,
    op::String,
    input::Vector{String}=String[],
    device::String="",
    attr::Dict{String,AttrValue}=Dict{String,AttrValue}(),
)
    return NodeDef(name, op, input, device, attr)
end

function encode(e::AbstractProtoEncoder, x::NodeDef)
    initpos = position(e.io)
    !isempty(x.name) && encode(e, 1, x.name)
    !isempty(x.op) && encode(e, 2, x.op)
    !isempty(x.input) && encode(e, 3, x.input)
    !isempty(x.device) && encode(e, 4, x.device)
    !isempty(x.attr) && encode(e, 5, x.attr)
    return position(e.io) - initpos
end

function _encoded_size(x::NodeDef)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += _encoded_size(x.name, 1))
    !isempty(x.op) && (encoded_size += _encoded_size(x.op, 2))
    !isempty(x.input) && (encoded_size += _encoded_size(x.input, 3))
    !isempty(x.device) && (encoded_size += _encoded_size(x.device, 4))
    !isempty(x.attr) && (encoded_size += _encoded_size(x.attr, 5))
    return encoded_size
end

# tensorflow.OpDef & OpDef.ArgDef
struct OpArgDef
    name::String
    type::TFDataType
    handle_data::Vector{ResourceDtypeAndShape}
end

function OpArgDef(name::String, type::TFDataType)
    return OpArgDef(name, type, ResourceDtypeAndShape[])
end

function encode(e::AbstractProtoEncoder, x::OpArgDef)
    initpos = position(e.io)
    !isempty(x.name) && encode(e, 1, x.name)
    x.type != DT_INVALID && encode(e, 3, x.type)
    !isempty(x.handle_data) && encode(e, 7, x.handle_data)
    return position(e.io) - initpos
end

function _encoded_size(x::OpArgDef)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += _encoded_size(x.name, 1))
    x.type != DT_INVALID && (encoded_size += _encoded_size(x.type, 3))
    !isempty(x.handle_data) && (encoded_size += _encoded_size(x.handle_data, 7))
    return encoded_size
end

struct OpDef
    name::String
    input_arg::Vector{OpArgDef}
    output_arg::Vector{OpArgDef}
    is_stateful::Bool
    control_output::Vector{String}
end

function OpDef(;
    name::String,
    input_arg::Vector{OpArgDef}=OpArgDef[],
    output_arg::Vector{OpArgDef}=OpArgDef[],
    is_stateful::Bool=false,
    control_output::Vector{String}=String[],
)
    return OpDef(name, input_arg, output_arg, is_stateful, control_output)
end

function encode(e::AbstractProtoEncoder, x::OpDef)
    initpos = position(e.io)
    !isempty(x.name) && encode(e, 1, x.name)
    !isempty(x.input_arg) && encode(e, 2, x.input_arg)
    !isempty(x.output_arg) && encode(e, 3, x.output_arg)
    x.is_stateful && encode(e, 17, x.is_stateful)
    !isempty(x.control_output) && encode(e, 20, x.control_output)
    return position(e.io) - initpos
end

function _encoded_size(x::OpDef)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += _encoded_size(x.name, 1))
    !isempty(x.input_arg) && (encoded_size += _encoded_size(x.input_arg, 2))
    !isempty(x.output_arg) && (encoded_size += _encoded_size(x.output_arg, 3))
    x.is_stateful && (encoded_size += _encoded_size(x.is_stateful, 17))
    !isempty(x.control_output) && (encoded_size += _encoded_size(x.control_output, 20))
    return encoded_size
end

# tensorflow.FunctionDef & FunctionDefLibrary
struct FunctionArgAttrs
    attr::Dict{String,AttrValue}
end

function encode(e::AbstractProtoEncoder, x::FunctionArgAttrs)
    initpos = position(e.io)
    !isempty(x.attr) && encode(e, 1, x.attr)
    return position(e.io) - initpos
end

function _encoded_size(x::FunctionArgAttrs)
    encoded_size = 0
    !isempty(x.attr) && (encoded_size += _encoded_size(x.attr, 1))
    return encoded_size
end

struct FunctionDef
    signature::Union{Nothing,OpDef}
    node_def::Vector{NodeDef}
    ret::Dict{String,String}
    attr::Dict{String,AttrValue}
    control_ret::Dict{String,String}
    arg_attr::Dict{UInt32,FunctionArgAttrs}
end

function FunctionDef(;
    signature::OpDef,
    node_def::Vector{NodeDef}=NodeDef[],
    ret::Dict{String,String}=Dict{String,String}(),
    attr::Dict{String,AttrValue}=Dict{String,AttrValue}(),
    control_ret::Dict{String,String}=Dict{String,String}(),
    arg_attr::Dict{UInt32,FunctionArgAttrs}=Dict{UInt32,FunctionArgAttrs}(),
)
    return FunctionDef(signature, node_def, ret, attr, control_ret, arg_attr)
end

function encode(e::AbstractProtoEncoder, x::FunctionDef)
    initpos = position(e.io)
    !isnothing(x.signature) && encode(e, 1, x.signature)
    !isempty(x.node_def) && encode(e, 3, x.node_def)
    !isempty(x.ret) && encode(e, 4, x.ret)
    !isempty(x.attr) && encode(e, 5, x.attr)
    !isempty(x.control_ret) && encode(e, 6, x.control_ret)
    !isempty(x.arg_attr) && encode(e, 7, x.arg_attr)
    return position(e.io) - initpos
end

function _encoded_size(x::FunctionDef)
    encoded_size = 0
    !isnothing(x.signature) && (encoded_size += _encoded_size(x.signature, 1))
    !isempty(x.node_def) && (encoded_size += _encoded_size(x.node_def, 3))
    !isempty(x.ret) && (encoded_size += _encoded_size(x.ret, 4))
    !isempty(x.attr) && (encoded_size += _encoded_size(x.attr, 5))
    !isempty(x.control_ret) && (encoded_size += _encoded_size(x.control_ret, 6))
    !isempty(x.arg_attr) && (encoded_size += _encoded_size(x.arg_attr, 7))
    return encoded_size
end

struct FunctionDefLibrary
    function_defs::Vector{FunctionDef}
end

function encode(e::AbstractProtoEncoder, x::FunctionDefLibrary)
    initpos = position(e.io)
    !isempty(x.function_defs) && encode(e, 1, x.function_defs)
    return position(e.io) - initpos
end

function _encoded_size(x::FunctionDefLibrary)
    encoded_size = 0
    !isempty(x.function_defs) && (encoded_size += _encoded_size(x.function_defs, 1))
    return encoded_size
end

# tensorflow.GraphDef
struct GraphDef
    node::Vector{NodeDef}
    library::Union{Nothing,FunctionDefLibrary}
    versions::Union{Nothing,VersionDef}
end

function encode(e::AbstractProtoEncoder, x::GraphDef)
    initpos = position(e.io)
    !isempty(x.node) && encode(e, 1, x.node)
    !isnothing(x.library) && encode(e, 2, x.library)
    !isnothing(x.versions) && encode(e, 4, x.versions)
    return position(e.io) - initpos
end

function _encoded_size(x::GraphDef)
    encoded_size = 0
    !isempty(x.node) && (encoded_size += _encoded_size(x.node, 1))
    !isnothing(x.library) && (encoded_size += _encoded_size(x.library, 2))
    !isnothing(x.versions) && (encoded_size += _encoded_size(x.versions, 4))
    return encoded_size
end

# tensorflow.SaverDef
struct SaverDef
    filename_tensor_name::String
    save_tensor_name::String
    restore_op_name::String
    version::CheckpointFormatVersion
end

function encode(e::AbstractProtoEncoder, x::SaverDef)
    initpos = position(e.io)
    !isempty(x.filename_tensor_name) && encode(e, 1, x.filename_tensor_name)
    !isempty(x.save_tensor_name) && encode(e, 2, x.save_tensor_name)
    !isempty(x.restore_op_name) && encode(e, 3, x.restore_op_name)
    x.version != SAVER_LEGACY && encode(e, 7, x.version)
    return position(e.io) - initpos
end

function _encoded_size(x::SaverDef)
    encoded_size = 0
    !isempty(x.filename_tensor_name) &&
        (encoded_size += _encoded_size(x.filename_tensor_name, 1))
    !isempty(x.save_tensor_name) && (encoded_size += _encoded_size(x.save_tensor_name, 2))
    !isempty(x.restore_op_name) && (encoded_size += _encoded_size(x.restore_op_name, 3))
    x.version != SAVER_LEGACY && (encoded_size += _encoded_size(x.version, 7))
    return encoded_size
end

# tensorflow.CollectionDef, TensorInfo, SignatureDef
struct CollectionNodeList
    value::Vector{String}
end

function encode(e::AbstractProtoEncoder, x::CollectionNodeList)
    initpos = position(e.io)
    !isempty(x.value) && encode(e, 1, x.value)
    return position(e.io) - initpos
end

function _encoded_size(x::CollectionNodeList)
    encoded_size = 0
    !isempty(x.value) && (encoded_size += _encoded_size(x.value, 1))
    return encoded_size
end

struct CollectionDef
    kind::Union{Nothing,OneOf}
end

CollectionDef(node_list::CollectionNodeList) = CollectionDef(OneOf(:node_list, node_list))

function encode(e::AbstractProtoEncoder, x::CollectionDef)
    initpos = position(e.io)
    if isnothing(x.kind)
        nothing
    elseif x.kind.name === :node_list
        encode(e, 1, x.kind[]::CollectionNodeList)
    else
        error("Unsupported CollectionDef oneof field: $(x.kind.name)")
    end
    return position(e.io) - initpos
end

function _encoded_size(x::CollectionDef)
    encoded_size = 0
    if isnothing(x.kind)
        nothing
    elseif x.kind.name === :node_list
        encoded_size += _encoded_size(x.kind[]::CollectionNodeList, 1)
    else
        error("Unsupported CollectionDef oneof field: $(x.kind.name)")
    end
    return encoded_size
end

struct TensorInfo
    encoding::Union{Nothing,OneOf}
    dtype::TFDataType
    tensor_shape::Union{Nothing,TensorShapeProto}
end

function TensorInfo(
    name::String,
    dtype::TFDataType=DT_INVALID,
    tensor_shape::Union{Nothing,TensorShapeProto}=nothing,
)
    return TensorInfo(OneOf(:name, name), dtype, tensor_shape)
end

function encode(e::AbstractProtoEncoder, x::TensorInfo)
    initpos = position(e.io)
    if isnothing(x.encoding)
        nothing
    elseif x.encoding.name === :name
        encode(e, 1, x.encoding[]::String)
    else
        error("Unsupported TensorInfo oneof field: $(x.encoding.name)")
    end
    x.dtype != DT_INVALID && encode(e, 2, x.dtype)
    !isnothing(x.tensor_shape) && encode(e, 3, x.tensor_shape)
    return position(e.io) - initpos
end

function _encoded_size(x::TensorInfo)
    encoded_size = 0
    if isnothing(x.encoding)
        nothing
    elseif x.encoding.name === :name
        encoded_size += _encoded_size(x.encoding[]::String, 1)
    else
        error("Unsupported TensorInfo oneof field: $(x.encoding.name)")
    end
    x.dtype != DT_INVALID && (encoded_size += _encoded_size(x.dtype, 2))
    !isnothing(x.tensor_shape) && (encoded_size += _encoded_size(x.tensor_shape, 3))
    return encoded_size
end

struct SignatureDef
    inputs::Dict{String,TensorInfo}
    outputs::Dict{String,TensorInfo}
    method_name::String
end

function SignatureDef(;
    inputs::Dict{String,TensorInfo}=Dict{String,TensorInfo}(),
    outputs::Dict{String,TensorInfo}=Dict{String,TensorInfo}(),
    method_name::String="",
)
    return SignatureDef(inputs, outputs, method_name)
end

function encode(e::AbstractProtoEncoder, x::SignatureDef)
    initpos = position(e.io)
    !isempty(x.inputs) && encode(e, 1, x.inputs)
    !isempty(x.outputs) && encode(e, 2, x.outputs)
    !isempty(x.method_name) && encode(e, 3, x.method_name)
    return position(e.io) - initpos
end

function _encoded_size(x::SignatureDef)
    encoded_size = 0
    !isempty(x.inputs) && (encoded_size += _encoded_size(x.inputs, 1))
    !isempty(x.outputs) && (encoded_size += _encoded_size(x.outputs, 2))
    !isempty(x.method_name) && (encoded_size += _encoded_size(x.method_name, 3))
    return encoded_size
end

# tensorflow.StructuredValue (struct.proto)
struct StructuredValue
    kind::Union{Nothing,OneOf}
end

struct NoneValue end

encode(::AbstractProtoEncoder, ::NoneValue) = 0
_encoded_size(::NoneValue) = 0

struct ListValue
    values::Vector{StructuredValue}
end

function encode(e::AbstractProtoEncoder, x::ListValue)
    initpos = position(e.io)
    !isempty(x.values) && encode(e, 1, x.values)
    return position(e.io) - initpos
end

function _encoded_size(x::ListValue)
    encoded_size = 0
    !isempty(x.values) && (encoded_size += _encoded_size(x.values, 1))
    return encoded_size
end

struct TupleValue
    values::Vector{StructuredValue}
end

function encode(e::AbstractProtoEncoder, x::TupleValue)
    initpos = position(e.io)
    !isempty(x.values) && encode(e, 1, x.values)
    return position(e.io) - initpos
end

function _encoded_size(x::TupleValue)
    encoded_size = 0
    !isempty(x.values) && (encoded_size += _encoded_size(x.values, 1))
    return encoded_size
end

struct DictValue
    fields::Dict{String,StructuredValue}
end

function encode(e::AbstractProtoEncoder, x::DictValue)
    initpos = position(e.io)
    !isempty(x.fields) && encode(e, 1, x.fields)
    return position(e.io) - initpos
end

function _encoded_size(x::DictValue)
    encoded_size = 0
    !isempty(x.fields) && (encoded_size += _encoded_size(x.fields, 1))
    return encoded_size
end

struct PairValue
    key::String
    value::Union{Nothing,StructuredValue}
end

function encode(e::AbstractProtoEncoder, x::PairValue)
    initpos = position(e.io)
    !isempty(x.key) && encode(e, 1, x.key)
    !isnothing(x.value) && encode(e, 2, x.value)
    return position(e.io) - initpos
end

function _encoded_size(x::PairValue)
    encoded_size = 0
    !isempty(x.key) && (encoded_size += _encoded_size(x.key, 1))
    !isnothing(x.value) && (encoded_size += _encoded_size(x.value, 2))
    return encoded_size
end

struct NamedTupleValue
    name::String
    values::Vector{PairValue}
end

function encode(e::AbstractProtoEncoder, x::NamedTupleValue)
    initpos = position(e.io)
    !isempty(x.name) && encode(e, 1, x.name)
    !isempty(x.values) && encode(e, 2, x.values)
    return position(e.io) - initpos
end

function _encoded_size(x::NamedTupleValue)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += _encoded_size(x.name, 1))
    !isempty(x.values) && (encoded_size += _encoded_size(x.values, 2))
    return encoded_size
end

struct TensorSpecProto
    name::String
    shape::Union{Nothing,TensorShapeProto}
    dtype::TFDataType
end

function encode(e::AbstractProtoEncoder, x::TensorSpecProto)
    initpos = position(e.io)
    !isempty(x.name) && encode(e, 1, x.name)
    !isnothing(x.shape) && encode(e, 2, x.shape)
    x.dtype != DT_INVALID && encode(e, 3, x.dtype)
    return position(e.io) - initpos
end

function _encoded_size(x::TensorSpecProto)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += _encoded_size(x.name, 1))
    !isnothing(x.shape) && (encoded_size += _encoded_size(x.shape, 2))
    x.dtype != DT_INVALID && (encoded_size += _encoded_size(x.dtype, 3))
    return encoded_size
end

function encode(e::AbstractProtoEncoder, x::StructuredValue)
    initpos = position(e.io)
    if isnothing(x.kind)
        nothing
    elseif x.kind.name === :none_value
        encode(e, 1, x.kind[]::NoneValue)
    elseif x.kind.name === :string_value
        encode(e, 13, x.kind[]::String)
    elseif x.kind.name === :bool_value
        encode(e, 14, x.kind[]::Bool)
    elseif x.kind.name === :tensor_shape_value
        encode(e, 31, x.kind[]::TensorShapeProto)
    elseif x.kind.name === :tensor_dtype_value
        encode(e, 32, x.kind[]::TFDataType)
    elseif x.kind.name === :tensor_spec_value
        encode(e, 33, x.kind[]::TensorSpecProto)
    elseif x.kind.name === :list_value
        encode(e, 51, x.kind[]::ListValue)
    elseif x.kind.name === :tuple_value
        encode(e, 52, x.kind[]::TupleValue)
    elseif x.kind.name === :dict_value
        encode(e, 53, x.kind[]::DictValue)
    elseif x.kind.name === :named_tuple_value
        encode(e, 54, x.kind[]::NamedTupleValue)
    else
        error("Unsupported StructuredValue oneof field: $(x.kind.name)")
    end
    return position(e.io) - initpos
end

function _encoded_size(x::StructuredValue)
    encoded_size = 0
    if isnothing(x.kind)
        nothing
    elseif x.kind.name === :none_value
        encoded_size += _encoded_size(x.kind[]::NoneValue, 1)
    elseif x.kind.name === :string_value
        encoded_size += _encoded_size(x.kind[]::String, 13)
    elseif x.kind.name === :bool_value
        encoded_size += _encoded_size(x.kind[]::Bool, 14)
    elseif x.kind.name === :tensor_shape_value
        encoded_size += _encoded_size(x.kind[]::TensorShapeProto, 31)
    elseif x.kind.name === :tensor_dtype_value
        encoded_size += _encoded_size(x.kind[]::TFDataType, 32)
    elseif x.kind.name === :tensor_spec_value
        encoded_size += _encoded_size(x.kind[]::TensorSpecProto, 33)
    elseif x.kind.name === :list_value
        encoded_size += _encoded_size(x.kind[]::ListValue, 51)
    elseif x.kind.name === :tuple_value
        encoded_size += _encoded_size(x.kind[]::TupleValue, 52)
    elseif x.kind.name === :dict_value
        encoded_size += _encoded_size(x.kind[]::DictValue, 53)
    elseif x.kind.name === :named_tuple_value
        encoded_size += _encoded_size(x.kind[]::NamedTupleValue, 54)
    else
        error("Unsupported StructuredValue oneof field: $(x.kind.name)")
    end
    return encoded_size
end

sv_none() = StructuredValue(OneOf(:none_value, NoneValue()))
sv_string(s::String) = StructuredValue(OneOf(:string_value, s))
sv_spec(s::TensorSpecProto) = StructuredValue(OneOf(:tensor_spec_value, s))
function sv_list(vals::Vector{StructuredValue}=StructuredValue[])
    return StructuredValue(OneOf(:list_value, ListValue(vals)))
end
function sv_tuple(vals::Vector{StructuredValue}=StructuredValue[])
    return StructuredValue(OneOf(:tuple_value, TupleValue(vals)))
end
function sv_dict(fields::Dict{String,StructuredValue}=Dict{String,StructuredValue}())
    return StructuredValue(OneOf(:dict_value, DictValue(fields)))
end
function sv_namedtuple(name::String, pairs::Vector{PairValue})
    return StructuredValue(OneOf(:named_tuple_value, NamedTupleValue(name, pairs)))
end

# tensorflow.TrackableObjectGraph (trackable_object_graph.proto)
struct ObjectReference
    node_id::Int32
    local_name::String
end

function encode(e::AbstractProtoEncoder, x::ObjectReference)
    initpos = position(e.io)
    x.node_id != zero(Int32) && encode(e, 1, x.node_id)
    !isempty(x.local_name) && encode(e, 2, x.local_name)
    return position(e.io) - initpos
end

function _encoded_size(x::ObjectReference)
    encoded_size = 0
    x.node_id != zero(Int32) && (encoded_size += _encoded_size(x.node_id, 1))
    !isempty(x.local_name) && (encoded_size += _encoded_size(x.local_name, 2))
    return encoded_size
end

struct SerializedTensor
    name::String
    full_name::String
    checkpoint_key::String
end

function encode(e::AbstractProtoEncoder, x::SerializedTensor)
    initpos = position(e.io)
    !isempty(x.name) && encode(e, 1, x.name)
    !isempty(x.full_name) && encode(e, 2, x.full_name)
    !isempty(x.checkpoint_key) && encode(e, 3, x.checkpoint_key)
    return position(e.io) - initpos
end

function _encoded_size(x::SerializedTensor)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += _encoded_size(x.name, 1))
    !isempty(x.full_name) && (encoded_size += _encoded_size(x.full_name, 2))
    !isempty(x.checkpoint_key) && (encoded_size += _encoded_size(x.checkpoint_key, 3))
    return encoded_size
end

struct TrackableObject
    children::Vector{ObjectReference}
    attributes::Vector{SerializedTensor}
    has_checkpoint_values::Union{Nothing,BoolValue}
end

function TrackableObject(;
    children::Vector{ObjectReference}=ObjectReference[],
    attributes::Vector{SerializedTensor}=SerializedTensor[],
    has_checkpoint_values::Union{Nothing,BoolValue}=nothing,
)
    return TrackableObject(children, attributes, has_checkpoint_values)
end

function encode(e::AbstractProtoEncoder, x::TrackableObject)
    initpos = position(e.io)
    !isempty(x.children) && encode(e, 1, x.children)
    !isempty(x.attributes) && encode(e, 2, x.attributes)
    !isnothing(x.has_checkpoint_values) && encode(e, 5, x.has_checkpoint_values)
    return position(e.io) - initpos
end

function _encoded_size(x::TrackableObject)
    encoded_size = 0
    !isempty(x.children) && (encoded_size += _encoded_size(x.children, 1))
    !isempty(x.attributes) && (encoded_size += _encoded_size(x.attributes, 2))
    !isnothing(x.has_checkpoint_values) &&
        (encoded_size += _encoded_size(x.has_checkpoint_values, 5))
    return encoded_size
end

struct TrackableObjectGraph
    nodes::Vector{TrackableObject}
end

function encode(e::AbstractProtoEncoder, x::TrackableObjectGraph)
    initpos = position(e.io)
    !isempty(x.nodes) && encode(e, 1, x.nodes)
    return position(e.io) - initpos
end

function _encoded_size(x::TrackableObjectGraph)
    encoded_size = 0
    !isempty(x.nodes) && (encoded_size += _encoded_size(x.nodes, 1))
    return encoded_size
end

# tensorflow.SavedObjectGraph (saved_object_graph.proto)
struct SavedUserObject
    identifier::String
    version::Union{Nothing,VersionDef}
end

function encode(e::AbstractProtoEncoder, x::SavedUserObject)
    initpos = position(e.io)
    !isempty(x.identifier) && encode(e, 1, x.identifier)
    !isnothing(x.version) && encode(e, 2, x.version)
    return position(e.io) - initpos
end

function _encoded_size(x::SavedUserObject)
    encoded_size = 0
    !isempty(x.identifier) && (encoded_size += _encoded_size(x.identifier, 1))
    !isnothing(x.version) && (encoded_size += _encoded_size(x.version, 2))
    return encoded_size
end

struct FunctionSpec
    fullargspec::Union{Nothing,StructuredValue}
    is_method::Bool
    input_signature::Union{Nothing,StructuredValue}
end

function encode(e::AbstractProtoEncoder, x::FunctionSpec)
    initpos = position(e.io)
    !isnothing(x.fullargspec) && encode(e, 1, x.fullargspec)
    x.is_method && encode(e, 2, x.is_method)
    !isnothing(x.input_signature) && encode(e, 5, x.input_signature)
    return position(e.io) - initpos
end

function _encoded_size(x::FunctionSpec)
    encoded_size = 0
    !isnothing(x.fullargspec) && (encoded_size += _encoded_size(x.fullargspec, 1))
    x.is_method && (encoded_size += _encoded_size(x.is_method, 2))
    !isnothing(x.input_signature) && (encoded_size += _encoded_size(x.input_signature, 5))
    return encoded_size
end

struct SavedFunction
    concrete_functions::Vector{String}
    function_spec::Union{Nothing,FunctionSpec}
end

function encode(e::AbstractProtoEncoder, x::SavedFunction)
    initpos = position(e.io)
    !isempty(x.concrete_functions) && encode(e, 1, x.concrete_functions)
    !isnothing(x.function_spec) && encode(e, 2, x.function_spec)
    return position(e.io) - initpos
end

function _encoded_size(x::SavedFunction)
    encoded_size = 0
    !isempty(x.concrete_functions) &&
        (encoded_size += _encoded_size(x.concrete_functions, 1))
    !isnothing(x.function_spec) && (encoded_size += _encoded_size(x.function_spec, 2))
    return encoded_size
end

struct SavedConcreteFunction
    bound_inputs::Vector{Int32}
    canonicalized_input_signature::Union{Nothing,StructuredValue}
    output_signature::Union{Nothing,StructuredValue}
end

function encode(e::AbstractProtoEncoder, x::SavedConcreteFunction)
    initpos = position(e.io)
    !isempty(x.bound_inputs) && encode(e, 2, x.bound_inputs)
    !isnothing(x.canonicalized_input_signature) &&
        encode(e, 3, x.canonicalized_input_signature)
    !isnothing(x.output_signature) && encode(e, 4, x.output_signature)
    return position(e.io) - initpos
end

function _encoded_size(x::SavedConcreteFunction)
    encoded_size = 0
    !isempty(x.bound_inputs) && (encoded_size += _encoded_size(x.bound_inputs, 2))
    !isnothing(x.canonicalized_input_signature) &&
        (encoded_size += _encoded_size(x.canonicalized_input_signature, 3))
    !isnothing(x.output_signature) && (encoded_size += _encoded_size(x.output_signature, 4))
    return encoded_size
end

struct SavedBareConcreteFunction
    concrete_function_name::String
    argument_keywords::Vector{String}
    allowed_positional_arguments::Int64
    function_spec::Union{Nothing,FunctionSpec}
end

function encode(e::AbstractProtoEncoder, x::SavedBareConcreteFunction)
    initpos = position(e.io)
    !isempty(x.concrete_function_name) && encode(e, 1, x.concrete_function_name)
    !isempty(x.argument_keywords) && encode(e, 2, x.argument_keywords)
    x.allowed_positional_arguments != zero(Int64) &&
        encode(e, 3, x.allowed_positional_arguments)
    !isnothing(x.function_spec) && encode(e, 4, x.function_spec)
    return position(e.io) - initpos
end

function _encoded_size(x::SavedBareConcreteFunction)
    encoded_size = 0
    !isempty(x.concrete_function_name) &&
        (encoded_size += _encoded_size(x.concrete_function_name, 1))
    !isempty(x.argument_keywords) && (encoded_size += _encoded_size(x.argument_keywords, 2))
    x.allowed_positional_arguments != zero(Int64) &&
        (encoded_size += _encoded_size(x.allowed_positional_arguments, 3))
    !isnothing(x.function_spec) && (encoded_size += _encoded_size(x.function_spec, 4))
    return encoded_size
end

struct SavedVariable
    dtype::TFDataType
    shape::Union{Nothing,TensorShapeProto}
    trainable::Bool
    name::String
    device::String
end

function SavedVariable(;
    dtype::TFDataType,
    shape::TensorShapeProto,
    name::String,
    trainable::Bool=false,
    device::String="",
)
    return SavedVariable(dtype, shape, trainable, name, device)
end

function encode(e::AbstractProtoEncoder, x::SavedVariable)
    initpos = position(e.io)
    x.dtype != DT_INVALID && encode(e, 1, x.dtype)
    !isnothing(x.shape) && encode(e, 2, x.shape)
    x.trainable && encode(e, 3, x.trainable)
    !isempty(x.name) && encode(e, 6, x.name)
    !isempty(x.device) && encode(e, 7, x.device)
    return position(e.io) - initpos
end

function _encoded_size(x::SavedVariable)
    encoded_size = 0
    x.dtype != DT_INVALID && (encoded_size += _encoded_size(x.dtype, 1))
    !isnothing(x.shape) && (encoded_size += _encoded_size(x.shape, 2))
    x.trainable && (encoded_size += _encoded_size(x.trainable, 3))
    !isempty(x.name) && (encoded_size += _encoded_size(x.name, 6))
    !isempty(x.device) && (encoded_size += _encoded_size(x.device, 7))
    return encoded_size
end

struct SavedObject
    children::Vector{ObjectReference}
    kind::Union{Nothing,OneOf}
    dependencies::Vector{ObjectReference}
end

function SavedObject(;
    kind::OneOf,
    children::Vector{ObjectReference}=ObjectReference[],
    dependencies::Vector{ObjectReference}=ObjectReference[],
)
    return SavedObject(children, kind, dependencies)
end

function encode(e::AbstractProtoEncoder, x::SavedObject)
    initpos = position(e.io)
    !isempty(x.children) && encode(e, 1, x.children)
    if isnothing(x.kind)
        nothing
    elseif x.kind.name === :user_object
        encode(e, 4, x.kind[]::SavedUserObject)
    elseif x.kind.name === :function
        encode(e, 6, x.kind[]::SavedFunction)
    elseif x.kind.name === :variable
        encode(e, 7, x.kind[]::SavedVariable)
    elseif x.kind.name === :bare_concrete_function
        encode(e, 8, x.kind[]::SavedBareConcreteFunction)
    else
        error("Unsupported SavedObject oneof field: $(x.kind.name)")
    end
    !isempty(x.dependencies) && encode(e, 15, x.dependencies)
    return position(e.io) - initpos
end

function _encoded_size(x::SavedObject)
    encoded_size = 0
    !isempty(x.children) && (encoded_size += _encoded_size(x.children, 1))
    if isnothing(x.kind)
        nothing
    elseif x.kind.name === :user_object
        encoded_size += _encoded_size(x.kind[]::SavedUserObject, 4)
    elseif x.kind.name === :function
        encoded_size += _encoded_size(x.kind[]::SavedFunction, 6)
    elseif x.kind.name === :variable
        encoded_size += _encoded_size(x.kind[]::SavedVariable, 7)
    elseif x.kind.name === :bare_concrete_function
        encoded_size += _encoded_size(x.kind[]::SavedBareConcreteFunction, 8)
    else
        error("Unsupported SavedObject oneof field: $(x.kind.name)")
    end
    !isempty(x.dependencies) && (encoded_size += _encoded_size(x.dependencies, 15))
    return encoded_size
end

struct SavedObjectGraph
    nodes::Vector{SavedObject}
    concrete_functions::Dict{String,SavedConcreteFunction}
end

function encode(e::AbstractProtoEncoder, x::SavedObjectGraph)
    initpos = position(e.io)
    !isempty(x.nodes) && encode(e, 1, x.nodes)
    !isempty(x.concrete_functions) && encode(e, 2, x.concrete_functions)
    return position(e.io) - initpos
end

function _encoded_size(x::SavedObjectGraph)
    encoded_size = 0
    !isempty(x.nodes) && (encoded_size += _encoded_size(x.nodes, 1))
    !isempty(x.concrete_functions) &&
        (encoded_size += _encoded_size(x.concrete_functions, 2))
    return encoded_size
end

# tensorflow.MetaGraphDef & SavedModel
struct MetaInfoDef
    tags::Vector{String}
    tensorflow_version::String
    stripped_default_attrs::Bool
    function_aliases::Dict{String,String}
end

function encode(e::AbstractProtoEncoder, x::MetaInfoDef)
    initpos = position(e.io)
    !isempty(x.tags) && encode(e, 4, x.tags)
    !isempty(x.tensorflow_version) && encode(e, 5, x.tensorflow_version)
    x.stripped_default_attrs && encode(e, 7, x.stripped_default_attrs)
    !isempty(x.function_aliases) && encode(e, 8, x.function_aliases)
    return position(e.io) - initpos
end

function _encoded_size(x::MetaInfoDef)
    encoded_size = 0
    !isempty(x.tags) && (encoded_size += _encoded_size(x.tags, 4))
    !isempty(x.tensorflow_version) &&
        (encoded_size += _encoded_size(x.tensorflow_version, 5))
    x.stripped_default_attrs && (encoded_size += _encoded_size(x.stripped_default_attrs, 7))
    !isempty(x.function_aliases) && (encoded_size += _encoded_size(x.function_aliases, 8))
    return encoded_size
end

struct MetaGraphDef
    meta_info_def::Union{Nothing,MetaInfoDef}
    graph_def::Union{Nothing,GraphDef}
    saver_def::Union{Nothing,SaverDef}
    collection_def::Dict{String,CollectionDef}
    signature_def::Dict{String,SignatureDef}
    object_graph_def::Union{Nothing,SavedObjectGraph}
end

function encode(e::AbstractProtoEncoder, x::MetaGraphDef)
    initpos = position(e.io)
    !isnothing(x.meta_info_def) && encode(e, 1, x.meta_info_def)
    !isnothing(x.graph_def) && encode(e, 2, x.graph_def)
    !isnothing(x.saver_def) && encode(e, 3, x.saver_def)
    !isempty(x.collection_def) && encode(e, 4, x.collection_def)
    !isempty(x.signature_def) && encode(e, 5, x.signature_def)
    !isnothing(x.object_graph_def) && encode(e, 7, x.object_graph_def)
    return position(e.io) - initpos
end

function _encoded_size(x::MetaGraphDef)
    encoded_size = 0
    !isnothing(x.meta_info_def) && (encoded_size += _encoded_size(x.meta_info_def, 1))
    !isnothing(x.graph_def) && (encoded_size += _encoded_size(x.graph_def, 2))
    !isnothing(x.saver_def) && (encoded_size += _encoded_size(x.saver_def, 3))
    !isempty(x.collection_def) && (encoded_size += _encoded_size(x.collection_def, 4))
    !isempty(x.signature_def) && (encoded_size += _encoded_size(x.signature_def, 5))
    !isnothing(x.object_graph_def) && (encoded_size += _encoded_size(x.object_graph_def, 7))
    return encoded_size
end

struct SavedModel
    saved_model_schema_version::Int64
    meta_graphs::Vector{MetaGraphDef}
end

function encode(e::AbstractProtoEncoder, x::SavedModel)
    initpos = position(e.io)
    x.saved_model_schema_version != zero(Int64) &&
        encode(e, 1, x.saved_model_schema_version)
    !isempty(x.meta_graphs) && encode(e, 2, x.meta_graphs)
    return position(e.io) - initpos
end

function _encoded_size(x::SavedModel)
    encoded_size = 0
    x.saved_model_schema_version != zero(Int64) &&
        (encoded_size += _encoded_size(x.saved_model_schema_version, 1))
    !isempty(x.meta_graphs) && (encoded_size += _encoded_size(x.meta_graphs, 2))
    return encoded_size
end

# tensorflow.BundleHeaderProto & BundleEntryProto (tensor_bundle.proto)
struct BundleHeaderProto
    num_shards::Int32
    endianness::BundleEndianness
    version::Union{Nothing,VersionDef}
end

function encode(e::AbstractProtoEncoder, x::BundleHeaderProto)
    initpos = position(e.io)
    x.num_shards != zero(Int32) && encode(e, 1, x.num_shards)
    x.endianness != BUNDLE_LITTLE && encode(e, 2, x.endianness)
    !isnothing(x.version) && encode(e, 3, x.version)
    return position(e.io) - initpos
end

function _encoded_size(x::BundleHeaderProto)
    encoded_size = 0
    x.num_shards != zero(Int32) && (encoded_size += _encoded_size(x.num_shards, 1))
    x.endianness != BUNDLE_LITTLE && (encoded_size += _encoded_size(x.endianness, 2))
    !isnothing(x.version) && (encoded_size += _encoded_size(x.version, 3))
    return encoded_size
end

struct BundleEntryProto
    dtype::TFDataType
    shape::Union{Nothing,TensorShapeProto}
    shard_id::Int32
    offset::Int64
    size::Int64
    crc32c::UInt32
end

function encode(e::AbstractProtoEncoder, x::BundleEntryProto)
    initpos = position(e.io)
    x.dtype != DT_INVALID && encode(e, 1, x.dtype)
    !isnothing(x.shape) && encode(e, 2, x.shape)
    x.shard_id != zero(Int32) && encode(e, 3, x.shard_id)
    x.offset != zero(Int64) && encode(e, 4, x.offset)
    x.size != zero(Int64) && encode(e, 5, x.size)
    x.crc32c != zero(UInt32) && encode(e, 6, x.crc32c, Val{:fixed})
    return position(e.io) - initpos
end

function _encoded_size(x::BundleEntryProto)
    encoded_size = 0
    x.dtype != DT_INVALID && (encoded_size += _encoded_size(x.dtype, 1))
    !isnothing(x.shape) && (encoded_size += _encoded_size(x.shape, 2))
    x.shard_id != zero(Int32) && (encoded_size += _encoded_size(x.shard_id, 3))
    x.offset != zero(Int64) && (encoded_size += _encoded_size(x.offset, 4))
    x.size != zero(Int64) && (encoded_size += _encoded_size(x.size, 5))
    x.crc32c != zero(UInt32) && (encoded_size += _encoded_size(x.crc32c, 6, Val{:fixed}))
    return encoded_size
end

end # module TFProto
