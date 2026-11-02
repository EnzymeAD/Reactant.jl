using CEnum: CEnum, @cenum

struct XLA_FFI_TypeId
    type_id::Int64
end

@cenum XLA_FFI_InternalExtension_Type::UInt32 begin
    XLA_FFI_Extension_Metadata = 1
end

struct XLA_FFI_InternalExtension
    struct_size::Csize_t
    type::XLA_FFI_InternalExtension_Type
    next::Ptr{XLA_FFI_InternalExtension}
end

struct XLA_FFI_Api_Version
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    major_version::Cint
    minor_version::Cint
end

mutable struct XLA_FFI_InternalApi end

# typedef XLA_FFI_Error * XLA_FFI_Error_Create ( XLA_FFI_Error_Create_Args * args )
const XLA_FFI_Error_Create = Cvoid

# typedef void XLA_FFI_Error_GetDetails ( XLA_FFI_Error_GetDetails_Args * args )
const XLA_FFI_Error_GetDetails = Cvoid

# typedef void XLA_FFI_Error_Destroy ( XLA_FFI_Error_Destroy_Args * args )
const XLA_FFI_Error_Destroy = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_Handler_Register ( XLA_FFI_Handler_Register_Args * args )
const XLA_FFI_Handler_Register = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_Stream_Get ( XLA_FFI_Stream_Get_Args * args )
const XLA_FFI_Stream_Get = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_Type_Register ( XLA_FFI_Type_Register_Args * args )
const XLA_FFI_Type_Register = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_InvokeContext_Get ( XLA_FFI_InvokeContext_Get_Args * args )
const XLA_FFI_InvokeContext_Get = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_State_Set ( XLA_FFI_State_Set_Args * args )
const XLA_FFI_State_Set = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_State_Get ( XLA_FFI_State_Get_Args * args )
const XLA_FFI_State_Get = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_DeviceMemory_Allocate ( XLA_FFI_DeviceMemory_Allocate_Args * args )
const XLA_FFI_DeviceMemory_Allocate = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_DeviceMemory_Free ( XLA_FFI_DeviceMemory_Free_Args * args )
const XLA_FFI_DeviceMemory_Free = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_ThreadPool_Schedule ( XLA_FFI_ThreadPool_Schedule_Args * args )
const XLA_FFI_ThreadPool_Schedule = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_ThreadPool_NumThreads ( XLA_FFI_ThreadPool_NumThreads_Args * args )
const XLA_FFI_ThreadPool_NumThreads = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_Future_Create ( XLA_FFI_Future_Create_Args * args )
const XLA_FFI_Future_Create = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_Future_SetAvailable ( XLA_FFI_Future_SetAvailable_Args * args )
const XLA_FFI_Future_SetAvailable = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_Future_SetError ( XLA_FFI_Future_SetError_Args * args )
const XLA_FFI_Future_SetError = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_RunId_Get ( XLA_FFI_RunId_Get_Args * args )
const XLA_FFI_RunId_Get = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_DeviceOrdinal_Get ( XLA_FFI_DeviceOrdinal_Get_Args * args )
const XLA_FFI_DeviceOrdinal_Get = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_InvokeContext_FindExtension ( XLA_FFI_InvokeContext_FindExtension_Args * args )
const XLA_FFI_InvokeContext_FindExtension = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_CustomOptions_Get ( XLA_FFI_CustomOptions_Get_Args * args )
const XLA_FFI_CustomOptions_Get = Cvoid

struct XLA_FFI_Api
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    api_version::XLA_FFI_Api_Version
    internal_api::Ptr{XLA_FFI_InternalApi}
    XLA_FFI_Error_Create::Ptr{XLA_FFI_Error_Create}
    XLA_FFI_Error_GetDetails::Ptr{XLA_FFI_Error_GetDetails}
    XLA_FFI_Error_Destroy::Ptr{XLA_FFI_Error_Destroy}
    XLA_FFI_Handler_Register::Ptr{XLA_FFI_Handler_Register}
    XLA_FFI_Stream_Get::Ptr{XLA_FFI_Stream_Get}
    XLA_FFI_Type_Register::Ptr{XLA_FFI_Type_Register}
    XLA_FFI_InvokeContext_Get::Ptr{XLA_FFI_InvokeContext_Get}
    XLA_FFI_State_Set::Ptr{XLA_FFI_State_Set}
    XLA_FFI_State_Get::Ptr{XLA_FFI_State_Get}
    XLA_FFI_DeviceMemory_Allocate::Ptr{XLA_FFI_DeviceMemory_Allocate}
    XLA_FFI_DeviceMemory_Free::Ptr{XLA_FFI_DeviceMemory_Free}
    XLA_FFI_ThreadPool_Schedule::Ptr{XLA_FFI_ThreadPool_Schedule}
    XLA_FFI_ThreadPool_NumThreads::Ptr{XLA_FFI_ThreadPool_NumThreads}
    XLA_FFI_Future_Create::Ptr{XLA_FFI_Future_Create}
    XLA_FFI_Future_SetAvailable::Ptr{XLA_FFI_Future_SetAvailable}
    XLA_FFI_Future_SetError::Ptr{XLA_FFI_Future_SetError}
    XLA_FFI_RunId_Get::Ptr{XLA_FFI_RunId_Get}
    XLA_FFI_DeviceOrdinal_Get::Ptr{XLA_FFI_DeviceOrdinal_Get}
    XLA_FFI_InvokeContext_FindExtension::Ptr{XLA_FFI_InvokeContext_FindExtension}
    XLA_FFI_CustomOptions_Get::Ptr{XLA_FFI_CustomOptions_Get}
end

csizeof(::Type{XLA_FFI_InternalExtension}) = Csize_t(24)

struct XLA_FFI_ExtensionId
    extension_type::Int64
    major_version::Int32
    minor_version::Int32
end

csizeof(::Type{XLA_FFI_ExtensionId}) = Csize_t(16)

struct XLA_FFI_Extension
    struct_size::Csize_t
    id::XLA_FFI_ExtensionId
    next::Ptr{XLA_FFI_Extension}
end

csizeof(::Type{XLA_FFI_Extension}) = Csize_t(32)

csizeof(::Type{XLA_FFI_Api_Version}) = Csize_t(24)

mutable struct XLA_FFI_Error end

@cenum XLA_FFI_Error_Code::UInt32 begin
    XLA_FFI_Error_Code_OK = 0
    XLA_FFI_Error_Code_CANCELLED = 1
    XLA_FFI_Error_Code_UNKNOWN = 2
    XLA_FFI_Error_Code_INVALID_ARGUMENT = 3
    XLA_FFI_Error_Code_DEADLINE_EXCEEDED = 4
    XLA_FFI_Error_Code_NOT_FOUND = 5
    XLA_FFI_Error_Code_ALREADY_EXISTS = 6
    XLA_FFI_Error_Code_PERMISSION_DENIED = 7
    XLA_FFI_Error_Code_RESOURCE_EXHAUSTED = 8
    XLA_FFI_Error_Code_FAILED_PRECONDITION = 9
    XLA_FFI_Error_Code_ABORTED = 10
    XLA_FFI_Error_Code_OUT_OF_RANGE = 11
    XLA_FFI_Error_Code_UNIMPLEMENTED = 12
    XLA_FFI_Error_Code_INTERNAL = 13
    XLA_FFI_Error_Code_UNAVAILABLE = 14
    XLA_FFI_Error_Code_DATA_LOSS = 15
    XLA_FFI_Error_Code_UNAUTHENTICATED = 16
end

struct XLA_FFI_Error_Create_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    message::Cstring
    errc::XLA_FFI_Error_Code
    function XLA_FFI_Error_Create_Args(extension_start, message, errc)
        return new(csizeof(XLA_FFI_Error_Create_Args), extension_start, message, errc)
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Error_Create_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :message && return Ptr{Cstring}(x + 16)
    f === :errc && return Ptr{XLA_FFI_Error_Code}(x + 24)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Error_Create_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Error_Create_Args}) = Csize_t(28)

struct XLA_FFI_Error_GetDetails_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    error::Ptr{XLA_FFI_Error}
    message::Cstring
    errc::XLA_FFI_Error_Code
    function XLA_FFI_Error_GetDetails_Args(extension_start, error, message, errc)
        return new(
            csizeof(XLA_FFI_Error_GetDetails_Args), extension_start, error, message, errc
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Error_GetDetails_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :error && return Ptr{Ptr{XLA_FFI_Error}}(x + 16)
    f === :message && return Ptr{Cstring}(x + 24)
    f === :errc && return Ptr{XLA_FFI_Error_Code}(x + 32)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Error_GetDetails_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Error_GetDetails_Args}) = Csize_t(36)

struct XLA_FFI_Error_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    error::Ptr{XLA_FFI_Error}
    function XLA_FFI_Error_Destroy_Args(extension_start, error)
        return new(csizeof(XLA_FFI_Error_Destroy_Args), extension_start, error)
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Error_Destroy_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :error && return Ptr{Ptr{XLA_FFI_Error}}(x + 16)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Error_Destroy_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Error_Destroy_Args}) = Csize_t(24)

@cenum XLA_FFI_DataType::UInt32 begin
    XLA_FFI_DataType_INVALID = 0
    XLA_FFI_DataType_PRED = 1
    XLA_FFI_DataType_S1 = 30
    XLA_FFI_DataType_S2 = 26
    XLA_FFI_DataType_S4 = 21
    XLA_FFI_DataType_S8 = 2
    XLA_FFI_DataType_S16 = 3
    XLA_FFI_DataType_S32 = 4
    XLA_FFI_DataType_S64 = 5
    XLA_FFI_DataType_U1 = 31
    XLA_FFI_DataType_U2 = 27
    XLA_FFI_DataType_U4 = 22
    XLA_FFI_DataType_U8 = 6
    XLA_FFI_DataType_U16 = 7
    XLA_FFI_DataType_U32 = 8
    XLA_FFI_DataType_U64 = 9
    XLA_FFI_DataType_F16 = 10
    XLA_FFI_DataType_F32 = 11
    XLA_FFI_DataType_F64 = 12
    XLA_FFI_DataType_BF16 = 16
    XLA_FFI_DataType_C64 = 15
    XLA_FFI_DataType_C128 = 18
    XLA_FFI_DataType_TOKEN = 17
    XLA_FFI_DataType_F8E5M2 = 19
    XLA_FFI_DataType_F8E3M4 = 29
    XLA_FFI_DataType_F8E4M3 = 28
    XLA_FFI_DataType_F8E4M3FN = 20
    XLA_FFI_DataType_F8E4M3B11FNUZ = 23
    XLA_FFI_DataType_F8E5M2FNUZ = 24
    XLA_FFI_DataType_F8E4M3FNUZ = 25
    XLA_FFI_DataType_F4E2M1FN = 32
    XLA_FFI_DataType_F8E8M0FNU = 33
end

struct XLA_FFI_Buffer
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    dtype::XLA_FFI_DataType
    data::Ptr{Cvoid}
    rank::Int64
    dims::Ptr{Int64}
end

csizeof(::Type{XLA_FFI_Buffer}) = Csize_t(48)

@cenum XLA_FFI_ArgType::UInt32 begin
    XLA_FFI_ArgType_BUFFER = 1
end

@cenum XLA_FFI_RetType::UInt32 begin
    XLA_FFI_RetType_BUFFER = 1
end

@cenum XLA_FFI_AttrType::UInt32 begin
    XLA_FFI_AttrType_ARRAY = 1
    XLA_FFI_AttrType_DICTIONARY = 2
    XLA_FFI_AttrType_SCALAR = 3
    XLA_FFI_AttrType_STRING = 4
end

mutable struct XLA_FFI_InvokeContext end

const XLA_FFI_ExecutionContext = XLA_FFI_InvokeContext

struct XLA_FFI_ByteSpan
    ptr::Cstring
    len::Csize_t
end

struct XLA_FFI_Scalar
    dtype::XLA_FFI_DataType
    value::Ptr{Cvoid}
end

struct XLA_FFI_Array
    dtype::XLA_FFI_DataType
    size::Csize_t
    data::Ptr{Cvoid}
end

struct XLA_FFI_TypeInfo
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    deleter::Ptr{Cvoid}
end

csizeof(::Type{XLA_FFI_TypeInfo}) = Csize_t(24)

mutable struct XLA_FFI_Future end

struct XLA_FFI_Future_Create_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    future::Ptr{XLA_FFI_Future}
    function XLA_FFI_Future_Create_Args(extension_start, future)
        return new(csizeof(XLA_FFI_Future_Create_Args), extension_start, future)
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Future_Create_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :future && return Ptr{Ptr{XLA_FFI_Future}}(x + 16)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Future_Create_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Future_Create_Args}) = Csize_t(16)

struct XLA_FFI_Future_SetAvailable_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    future::Ptr{XLA_FFI_Future}
    function XLA_FFI_Future_SetAvailable_Args(extension_start, future)
        return new(csizeof(XLA_FFI_Future_SetAvailable_Args), extension_start, future)
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Future_SetAvailable_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :future && return Ptr{Ptr{XLA_FFI_Future}}(x + 16)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Future_SetAvailable_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Future_SetAvailable_Args}) = Csize_t(24)

struct XLA_FFI_Future_SetError_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    future::Ptr{XLA_FFI_Future}
    error::Ptr{XLA_FFI_Error}
    function XLA_FFI_Future_SetError_Args(extension_start, future, error)
        return new(csizeof(XLA_FFI_Future_SetError_Args), extension_start, future, error)
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Future_SetError_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :future && return Ptr{Ptr{XLA_FFI_Future}}(x + 16)
    f === :error && return Ptr{Ptr{XLA_FFI_Error}}(x + 24)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Future_SetError_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Future_SetError_Args}) = Csize_t(32)

@cenum XLA_FFI_ExecutionStage::UInt32 begin
    XLA_FFI_ExecutionStage_INSTANTIATE = 0
    XLA_FFI_ExecutionStage_PREPARE = 1
    XLA_FFI_ExecutionStage_INITIALIZE = 2
    XLA_FFI_ExecutionStage_EXECUTE = 3
    XLA_FFI_ExecutionStage_RECORD = 4
end

struct XLA_FFI_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    size::Int64
    types::Ptr{XLA_FFI_ArgType}
    args::Ptr{Ptr{Cvoid}}
    function XLA_FFI_Args(extension_start, size, types, args)
        return new(csizeof(XLA_FFI_Args), extension_start, size, types, args)
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :size && return Ptr{Int64}(x + 16)
    f === :types && return Ptr{Ptr{XLA_FFI_ArgType}}(x + 24)
    f === :args && return Ptr{Ptr{Ptr{Cvoid}}}(x + 32)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Args}) = Csize_t(40)

struct XLA_FFI_Rets
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    size::Int64
    types::Ptr{XLA_FFI_RetType}
    rets::Ptr{Ptr{Cvoid}}
end

csizeof(::Type{XLA_FFI_Rets}) = Csize_t(40)

struct XLA_FFI_Attrs
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    size::Int64
    types::Ptr{XLA_FFI_AttrType}
    names::Ptr{Ptr{XLA_FFI_ByteSpan}}
    attrs::Ptr{Ptr{Cvoid}}
end

csizeof(::Type{XLA_FFI_Attrs}) = Csize_t(48)

struct XLA_FFI_CallFrame
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    api::Ptr{XLA_FFI_Api}
    ctx::Ptr{XLA_FFI_InvokeContext}
    stage::XLA_FFI_ExecutionStage
    args::XLA_FFI_Args
    rets::XLA_FFI_Rets
    attrs::XLA_FFI_Attrs
    future::Ptr{XLA_FFI_Future}
end

csizeof(::Type{XLA_FFI_CallFrame}) = Csize_t(168)

# typedef XLA_FFI_Error * XLA_FFI_Handler ( XLA_FFI_CallFrame * call_frame )
const XLA_FFI_Handler = Cvoid

struct XLA_FFI_Handler_Bundle
    instantiate::Ptr{XLA_FFI_Handler}
    prepare::Ptr{XLA_FFI_Handler}
    initialize::Ptr{XLA_FFI_Handler}
    execute::Ptr{XLA_FFI_Handler}
    record::Ptr{XLA_FFI_Handler}
end

@cenum XLA_FFI_Handler_TraitsBits::UInt32 begin
    XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE = 1
end

const XLA_FFI_Handler_Traits = UInt32

struct XLA_FFI_Handler_Register_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    name::XLA_FFI_ByteSpan
    platform::XLA_FFI_ByteSpan
    bundle::XLA_FFI_Handler_Bundle
    traits::XLA_FFI_Handler_Traits
    function XLA_FFI_Handler_Register_Args(extension_start, name, platform, bundle, traits)
        return new(
            csizeof(XLA_FFI_Handler_Register_Args),
            extension_start,
            name,
            platform,
            bundle,
            traits,
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Handler_Register_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :name && return Ptr{XLA_FFI_ByteSpan}(x + 16)
    f === :platform && return Ptr{XLA_FFI_ByteSpan}(x + 32)
    f === :bundle && return Ptr{XLA_FFI_Handler_Bundle}(x + 48)
    f === :traits && return Ptr{XLA_FFI_Handler_Traits}(x + 88)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Handler_Register_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Handler_Register_Args}) = Csize_t(92)

struct XLA_FFI_Type_Register_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    name::XLA_FFI_ByteSpan
    type_id::Ptr{XLA_FFI_TypeId}
    type_info::Ptr{XLA_FFI_TypeInfo}
    function XLA_FFI_Type_Register_Args(extension_start, name, type_id, type_info)
        return new(
            csizeof(XLA_FFI_Type_Register_Args), extension_start, name, type_id, type_info
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Type_Register_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :name && return Ptr{XLA_FFI_ByteSpan}(x + 16)
    f === :type_id && return Ptr{Ptr{XLA_FFI_TypeId}}(x + 32)
    f === :type_info && return Ptr{Ptr{XLA_FFI_TypeInfo}}(x + 40)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Type_Register_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Type_Register_Args}) = Csize_t(40)

struct XLA_FFI_InvokeContext_FindExtension_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    extension_type::Int64
    extension::Ptr{XLA_FFI_Extension}
    function XLA_FFI_InvokeContext_FindExtension_Args(
        extension_start, ctx, extension_type, extension
    )
        return new(
            csizeof(XLA_FFI_InvokeContext_FindExtension_Args),
            extension_start,
            ctx,
            extension_type,
            extension,
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_InvokeContext_FindExtension_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :extension_type && return Ptr{Int64}(x + 24)
    f === :extension && return Ptr{Ptr{XLA_FFI_Extension}}(x + 32)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_InvokeContext_FindExtension_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_InvokeContext_FindExtension_Args}) = Csize_t(40)

struct XLA_FFI_InvokeContext_Get_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    type_id::Ptr{XLA_FFI_TypeId}
    data::Ptr{Cvoid}
    function XLA_FFI_InvokeContext_Get_Args(extension_start, ctx, type_id, data)
        return new(
            csizeof(XLA_FFI_InvokeContext_Get_Args), extension_start, ctx, type_id, data
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_InvokeContext_Get_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :type_id && return Ptr{Ptr{XLA_FFI_TypeId}}(x + 24)
    f === :data && return Ptr{Ptr{Cvoid}}(x + 32)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_InvokeContext_Get_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_InvokeContext_Get_Args}) = Csize_t(40)

struct XLA_FFI_CustomOptions_Get_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    attrs::Ptr{XLA_FFI_Attrs}
    function XLA_FFI_CustomOptions_Get_Args(extension_start, ctx, attrs)
        return new(csizeof(XLA_FFI_CustomOptions_Get_Args), extension_start, ctx, attrs)
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_CustomOptions_Get_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :attrs && return Ptr{Ptr{XLA_FFI_Attrs}}(x + 24)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_CustomOptions_Get_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_CustomOptions_Get_Args}) = Csize_t(32)

struct XLA_FFI_State_Set_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    stage::XLA_FFI_ExecutionStage
    type_id::Ptr{XLA_FFI_TypeId}
    state::Ptr{Cvoid}
    function XLA_FFI_State_Set_Args(extension_start, ctx, stage, type_id, state)
        return new(
            csizeof(XLA_FFI_State_Set_Args), extension_start, ctx, stage, type_id, state
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_State_Set_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :stage && return Ptr{XLA_FFI_ExecutionStage}(x + 24)
    f === :type_id && return Ptr{Ptr{XLA_FFI_TypeId}}(x + 32)
    f === :state && return Ptr{Ptr{Cvoid}}(x + 40)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_State_Set_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_State_Set_Args}) = Csize_t(48)

struct XLA_FFI_State_Get_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    stage::XLA_FFI_ExecutionStage
    type_id::Ptr{XLA_FFI_TypeId}
    state::Ptr{Cvoid}
    function XLA_FFI_State_Get_Args(extension_start, ctx, stage, type_id, state)
        return new(
            csizeof(XLA_FFI_State_Get_Args), extension_start, ctx, stage, type_id, state
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_State_Get_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :stage && return Ptr{XLA_FFI_ExecutionStage}(x + 24)
    f === :type_id && return Ptr{Ptr{XLA_FFI_TypeId}}(x + 32)
    f === :state && return Ptr{Ptr{Cvoid}}(x + 40)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_State_Get_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_State_Get_Args}) = Csize_t(48)

struct XLA_FFI_Stream_Get_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    stream::Ptr{Cvoid}
    function XLA_FFI_Stream_Get_Args(extension_start, ctx, stream)
        return new(csizeof(XLA_FFI_Stream_Get_Args), extension_start, ctx, stream)
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Stream_Get_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :stream && return Ptr{Ptr{Cvoid}}(x + 24)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Stream_Get_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Stream_Get_Args}) = Csize_t(32)

struct XLA_FFI_DeviceMemory_Allocate_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    size::Csize_t
    alignment::Csize_t
    data::Ptr{Cvoid}
    function XLA_FFI_DeviceMemory_Allocate_Args(extension_start, ctx, size, alignment, data)
        return new(
            csizeof(XLA_FFI_DeviceMemory_Allocate_Args),
            extension_start,
            ctx,
            size,
            alignment,
            data,
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_DeviceMemory_Allocate_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :size && return Ptr{Csize_t}(x + 24)
    f === :alignment && return Ptr{Csize_t}(x + 32)
    f === :data && return Ptr{Ptr{Cvoid}}(x + 40)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_DeviceMemory_Allocate_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_DeviceMemory_Allocate_Args}) = Csize_t(48)

struct XLA_FFI_DeviceMemory_Free_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    size::Csize_t
    data::Ptr{Cvoid}
    function XLA_FFI_DeviceMemory_Free_Args(extension_start, ctx, size, data)
        return new(
            csizeof(XLA_FFI_DeviceMemory_Free_Args), extension_start, ctx, size, data
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_DeviceMemory_Free_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :size && return Ptr{Csize_t}(x + 24)
    f === :data && return Ptr{Ptr{Cvoid}}(x + 32)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_DeviceMemory_Free_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_DeviceMemory_Free_Args}) = Csize_t(40)

# typedef void XLA_FFI_Task ( void * data )
const XLA_FFI_Task = Cvoid

struct XLA_FFI_ThreadPool_Schedule_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    task::Ptr{XLA_FFI_Task}
    data::Ptr{Cvoid}
    function XLA_FFI_ThreadPool_Schedule_Args(extension_start, ctx, task, data)
        return new(
            csizeof(XLA_FFI_ThreadPool_Schedule_Args), extension_start, ctx, task, data
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_ThreadPool_Schedule_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :task && return Ptr{Ptr{XLA_FFI_Task}}(x + 24)
    f === :data && return Ptr{Ptr{Cvoid}}(x + 32)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_ThreadPool_Schedule_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_ThreadPool_Schedule_Args}) = Csize_t(40)

struct XLA_FFI_ThreadPool_NumThreads_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    num_threads::Ptr{Int64}
    function XLA_FFI_ThreadPool_NumThreads_Args(extension_start, ctx, num_threads)
        return new(
            csizeof(XLA_FFI_ThreadPool_NumThreads_Args), extension_start, ctx, num_threads
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_ThreadPool_NumThreads_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :num_threads && return Ptr{Ptr{Int64}}(x + 24)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_ThreadPool_NumThreads_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_ThreadPool_NumThreads_Args}) = Csize_t(32)

struct XLA_FFI_RunId_Get_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    run_id::Int64
    function XLA_FFI_RunId_Get_Args(extension_start, ctx, run_id)
        return new(csizeof(XLA_FFI_RunId_Get_Args), extension_start, ctx, run_id)
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_RunId_Get_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :run_id && return Ptr{Int64}(x + 24)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_RunId_Get_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_RunId_Get_Args}) = Csize_t(32)

struct XLA_FFI_DeviceOrdinal_Get_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    ctx::Ptr{XLA_FFI_InvokeContext}
    device_ordinal::Int32
    function XLA_FFI_DeviceOrdinal_Get_Args(extension_start, ctx, device_ordinal)
        return new(
            csizeof(XLA_FFI_DeviceOrdinal_Get_Args), extension_start, ctx, device_ordinal
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_DeviceOrdinal_Get_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :ctx && return Ptr{Ptr{XLA_FFI_InvokeContext}}(x + 16)
    f === :device_ordinal && return Ptr{Int32}(x + 24)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_DeviceOrdinal_Get_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_DeviceOrdinal_Get_Args}) = Csize_t(28)

struct XLA_FFI_Metadata
    struct_size::Csize_t
    api_version::XLA_FFI_Api_Version
    traits::XLA_FFI_Handler_Traits
    state_type_id::XLA_FFI_TypeId
end

csizeof(::Type{XLA_FFI_Metadata}) = Csize_t(36)

struct XLA_FFI_Metadata_Extension
    extension_base::XLA_FFI_InternalExtension
    metadata::Ptr{XLA_FFI_Metadata}
end

csizeof(::Type{XLA_FFI_Metadata_Extension}) = Csize_t(32)

csizeof(::Type{XLA_FFI_Api}) = Csize_t(208)

# no prototype is found for this function at c_api.h:874:20, please use with caution
function XLA_FFI_GetApi()
    @ccall libxla.XLA_FFI_GetApi()::Ptr{XLA_FFI_Api}
end

@cenum XLA_FFI_CollectiveGroupMode::UInt32 begin
    XLA_FFI_GROUP_CROSS_REPLICA = 0
    XLA_FFI_GROUP_CROSS_PARTITION = 1
    XLA_FFI_GROUP_CROSS_REPLICA_AND_PARTITION = 2
    XLA_FFI_GROUP_FLATTENED_ID = 3
end

struct XLA_FFI_ReplicaGroup
    ids::Ptr{Int64}
    size::Csize_t
end

mutable struct XLA_FFI_CollectivesState end

# typedef XLA_FFI_Error * XLA_FFI_Communicator_Request ( const XLA_FFI_Collectives_Extension * self , XLA_FFI_Communicator_Request_Args * args )
const XLA_FFI_Communicator_Request = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_Communicator_Get ( const XLA_FFI_Collectives_Extension * self , XLA_FFI_Communicator_Get_Args * args )
const XLA_FFI_Communicator_Get = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_Window_Request ( const XLA_FFI_Collectives_Extension * self , XLA_FFI_Window_Request_Args * args )
const XLA_FFI_Window_Request = Cvoid

# typedef XLA_FFI_Error * XLA_FFI_Window_Get ( const XLA_FFI_Collectives_Extension * self , XLA_FFI_Window_Get_Args * args )
const XLA_FFI_Window_Get = Cvoid

struct XLA_FFI_Collectives_Extension
    extension_base::XLA_FFI_Extension
    state::Ptr{XLA_FFI_CollectivesState}
    request_communicator::Ptr{XLA_FFI_Communicator_Request}
    get_communicator::Ptr{XLA_FFI_Communicator_Get}
    request_window::Ptr{XLA_FFI_Window_Request}
    get_window::Ptr{XLA_FFI_Window_Get}
end

mutable struct XLA_FFI_Communicator end

struct XLA_FFI_Communicator_Request_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    group_mode::XLA_FFI_CollectiveGroupMode
    groups::Ptr{XLA_FFI_ReplicaGroup}
    num_groups::Csize_t
    communication_id::Int64
    function XLA_FFI_Communicator_Request_Args(
        extension_start, group_mode, groups, num_groups, communication_id
    )
        return new(
            csizeof(XLA_FFI_Communicator_Request_Args),
            extension_start,
            group_mode,
            groups,
            num_groups,
            communication_id,
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Communicator_Request_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :group_mode && return Ptr{XLA_FFI_CollectiveGroupMode}(x + 16)
    f === :groups && return Ptr{Ptr{XLA_FFI_ReplicaGroup}}(x + 24)
    f === :num_groups && return Ptr{Csize_t}(x + 32)
    f === :communication_id && return Ptr{Int64}(x + 40)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Communicator_Request_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Communicator_Request_Args}) = Csize_t(48)

struct XLA_FFI_Communicator_Get_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    group_mode::XLA_FFI_CollectiveGroupMode
    groups::Ptr{XLA_FFI_ReplicaGroup}
    num_groups::Csize_t
    communication_id::Int64
    communicator::Ptr{XLA_FFI_Communicator}
    function XLA_FFI_Communicator_Get_Args(
        extension_start, group_mode, groups, num_groups, communication_id, communicator
    )
        return new(
            csizeof(XLA_FFI_Communicator_Get_Args),
            extension_start,
            group_mode,
            groups,
            num_groups,
            communication_id,
            communicator,
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Communicator_Get_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :group_mode && return Ptr{XLA_FFI_CollectiveGroupMode}(x + 16)
    f === :groups && return Ptr{Ptr{XLA_FFI_ReplicaGroup}}(x + 24)
    f === :num_groups && return Ptr{Csize_t}(x + 32)
    f === :communication_id && return Ptr{Int64}(x + 40)
    f === :communicator && return Ptr{Ptr{XLA_FFI_Communicator}}(x + 48)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Communicator_Get_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Communicator_Get_Args}) = Csize_t(56)

mutable struct XLA_FFI_Window end

struct XLA_FFI_CollectiveMemoryRegion
    buffer::Ptr{Cvoid}
    byte_size::Csize_t
    flags::UInt64
end

struct XLA_FFI_Window_Request_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    group_mode::XLA_FFI_CollectiveGroupMode
    groups::Ptr{XLA_FFI_ReplicaGroup}
    num_groups::Csize_t
    communication_id::Int64
    regions::Ptr{XLA_FFI_CollectiveMemoryRegion}
    num_regions::Csize_t
    function XLA_FFI_Window_Request_Args(
        extension_start,
        group_mode,
        groups,
        num_groups,
        communication_id,
        regions,
        num_regions,
    )
        return new(
            csizeof(XLA_FFI_Window_Request_Args),
            extension_start,
            group_mode,
            groups,
            num_groups,
            communication_id,
            regions,
            num_regions,
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Window_Request_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :group_mode && return Ptr{XLA_FFI_CollectiveGroupMode}(x + 16)
    f === :groups && return Ptr{Ptr{XLA_FFI_ReplicaGroup}}(x + 24)
    f === :num_groups && return Ptr{Csize_t}(x + 32)
    f === :communication_id && return Ptr{Int64}(x + 40)
    f === :regions && return Ptr{Ptr{XLA_FFI_CollectiveMemoryRegion}}(x + 48)
    f === :num_regions && return Ptr{Csize_t}(x + 56)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Window_Request_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Window_Request_Args}) = Csize_t(64)

struct XLA_FFI_Window_Get_Args
    struct_size::Csize_t
    extension_start::Ptr{XLA_FFI_InternalExtension}
    group_mode::XLA_FFI_CollectiveGroupMode
    groups::Ptr{XLA_FFI_ReplicaGroup}
    num_groups::Csize_t
    communication_id::Int64
    buffer::Ptr{Cvoid}
    window::Ptr{XLA_FFI_Window}
    window_offset::Csize_t
    function XLA_FFI_Window_Get_Args(
        extension_start,
        group_mode,
        groups,
        num_groups,
        communication_id,
        buffer,
        window,
        window_offset,
    )
        return new(
            csizeof(XLA_FFI_Window_Get_Args),
            extension_start,
            group_mode,
            groups,
            num_groups,
            communication_id,
            buffer,
            window,
            window_offset,
        )
    end
end
function Base.getproperty(x::Ptr{XLA_FFI_Window_Get_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{XLA_FFI_InternalExtension}}(x + 8)
    f === :group_mode && return Ptr{XLA_FFI_CollectiveGroupMode}(x + 16)
    f === :groups && return Ptr{Ptr{XLA_FFI_ReplicaGroup}}(x + 24)
    f === :num_groups && return Ptr{Csize_t}(x + 32)
    f === :communication_id && return Ptr{Int64}(x + 40)
    f === :buffer && return Ptr{Ptr{Cvoid}}(x + 48)
    f === :window && return Ptr{Ptr{XLA_FFI_Window}}(x + 56)
    f === :window_offset && return Ptr{Csize_t}(x + 64)
    return getfield(x, f)
end

function Base.setproperty!(x::Ptr{XLA_FFI_Window_Get_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

csizeof(::Type{XLA_FFI_Window_Get_Args}) = Csize_t(72)

csizeof(::Type{XLA_FFI_Collectives_Extension}) = Csize_t(72)

mutable struct XLA_FFI_RecordContext end

mutable struct XLA_FFI_Command end

@cenum XLA_FFI_RecordAction::UInt32 begin
    XLA_FFI_RecordAction_Create = 0
    XLA_FFI_RecordAction_Update = 1
end

@cenum XLA_FFI_SourceFormat::UInt32 begin
    XLA_FFI_SourceFormat_PTX = 0
    XLA_FFI_SourceFormat_CUBIN = 1
    XLA_FFI_SourceFormat_FUNCTION_PTR = 2
end

struct XLA_FFI_Dim3
    x::Int32
    y::Int32
    z::Int32
end

struct XLA_FFI_LaunchDims
    grid::XLA_FFI_Dim3
    block::XLA_FFI_Dim3
    cluster::XLA_FFI_Dim3
end

@cenum XLA_FFI_KernelArgType::UInt32 begin
    XLA_FFI_KernelArgType_DevicePtr = 0
    XLA_FFI_KernelArgType_HostValue = 1
end

struct XLA_FFI_KernelArg
    arg_address::Ptr{Cvoid}
    size::Int64
    type::XLA_FFI_KernelArgType
end

struct XLA_FFI_KernelArgs
    args::Ptr{XLA_FFI_KernelArg}
    num_args::Int64
end

struct XLA_FFI_RecordApi
    create_launch::Ptr{Cvoid}
    update_launch::Ptr{Cvoid}
    create_memcpy_d2d::Ptr{Cvoid}
    update_memcpy_d2d::Ptr{Cvoid}
    request_stream_capture::Ptr{Cvoid}
    create_empty_command::Ptr{Cvoid}
end

struct XLA_FFI_RecordFrame
    record_ctx::Ptr{XLA_FFI_RecordContext}
    api::Ptr{XLA_FFI_RecordApi}
    action::XLA_FFI_RecordAction
    commands::Ptr{Ptr{XLA_FFI_Command}}
    num_commands::Ptr{Int64}
    max_commands::Int64
end

struct XLA_FFI_Record_Extension
    extension_base::XLA_FFI_Extension
    record_frame::Ptr{XLA_FFI_RecordFrame}
end

csizeof(::Type{XLA_FFI_Record_Extension}) = Csize_t(40)

const XLA_FFI_API_MAJOR = 0

const XLA_FFI_API_MINOR = 5

const XLA_FFI_Extension_Collectives = 129

const XLA_FFI_Extension_Collectives_MajorVersion = 0

const XLA_FFI_Extension_Collectives_MinorVersion = 2

const XLA_FFI_Extension_Record = 128

const XLA_FFI_Extension_Record_MajorVersion = 0

const XLA_FFI_Extension_Record_MinorVersion = 2
