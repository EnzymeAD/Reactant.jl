using CEnum: CEnum, @cenum

@cenum PJRT_Extension_Type::UInt32 begin
    PJRT_Extension_Type_Gpu_Custom_Call = 0
    PJRT_Extension_Type_Profiler = 1
    PJRT_Extension_Type_Custom_Partitioner = 2
    PJRT_Extension_Type_Stream = 3
    PJRT_Extension_Type_Layouts = 4
    PJRT_Extension_Type_FFI = 5
    PJRT_Extension_Type_MemoryDescriptions = 6
    PJRT_Extension_Type_Triton = 7
    PJRT_Extension_Type_RawBuffer = 8
    PJRT_Extension_Type_PhaseCompile = 9
    PJRT_Extension_Type_Example = 10
    PJRT_Extension_Type_Unknown = 11
    PJRT_Extension_Type_CrossHostTransfers = 12
    PJRT_Extension_Type_ExecutableMetadata = 13
    PJRT_Extension_Type_Callback = 14
    PJRT_Extension_Type_HostAllocator = 15
    PJRT_Extension_Type_TpuTopology = 16
    PJRT_Extension_Type_TpuExecutable = 17
    PJRT_Extension_Type_Megascale = 18
    PJRT_Extension_Type_Shardings = 19
    PJRT_Extension_Type_AbiVersion = 20
    PJRT_Extension_Type_Collectives = 21
    PJRT_Extension_Type_MultiSlice = 22
    PJRT_Extension_Type_HostMemoryAllocator = 23
    PJRT_Extension_Type_XlaTransform = 24
end

struct PJRT_Extension_Base
    struct_size::Csize_t
    type::PJRT_Extension_Type
    next::Ptr{PJRT_Extension_Base}
end

csizeof(::Type{PJRT_Extension_Base}) = Csize_t(24)

struct PJRT_Api_Version
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    major_version::Cint
    minor_version::Cint
end

csizeof(::Type{PJRT_Api_Version}) = Csize_t(24)

struct PJRT_Error_FunctionTable
    struct_size::Csize_t
    instance_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    destroy::Ptr{Cvoid}
    message::Ptr{Cvoid}
    get_code::Ptr{Cvoid}
    for_each_payload::Ptr{Cvoid}
end

struct PJRT_Error
    vtable::Ptr{PJRT_Error_FunctionTable}
end

struct PJRT_Error_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    error::Ptr{PJRT_Error}
    function PJRT_Error_Destroy_Args(extension_start, error)
        return new(csizeof(PJRT_Error_Destroy_Args), extension_start, error)
    end
end

csizeof(::Type{PJRT_Error_Destroy_Args}) = Csize_t(24)

# typedef void PJRT_Error_Destroy ( PJRT_Error_Destroy_Args * args )
const PJRT_Error_Destroy = Cvoid

struct PJRT_Error_Message_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    error::Ptr{PJRT_Error}
    message::Cstring
    message_size::Csize_t
    function PJRT_Error_Message_Args(extension_start, error, message, message_size)
        return new(
            csizeof(PJRT_Error_Message_Args), extension_start, error, message, message_size
        )
    end
end

csizeof(::Type{PJRT_Error_Message_Args}) = Csize_t(40)

# typedef void PJRT_Error_Message ( PJRT_Error_Message_Args * args )
const PJRT_Error_Message = Cvoid

@cenum PJRT_Error_Code::UInt32 begin
    PJRT_Error_Code_OK = 0
    PJRT_Error_Code_CANCELLED = 1
    PJRT_Error_Code_UNKNOWN = 2
    PJRT_Error_Code_INVALID_ARGUMENT = 3
    PJRT_Error_Code_DEADLINE_EXCEEDED = 4
    PJRT_Error_Code_NOT_FOUND = 5
    PJRT_Error_Code_ALREADY_EXISTS = 6
    PJRT_Error_Code_PERMISSION_DENIED = 7
    PJRT_Error_Code_RESOURCE_EXHAUSTED = 8
    PJRT_Error_Code_FAILED_PRECONDITION = 9
    PJRT_Error_Code_ABORTED = 10
    PJRT_Error_Code_OUT_OF_RANGE = 11
    PJRT_Error_Code_UNIMPLEMENTED = 12
    PJRT_Error_Code_INTERNAL = 13
    PJRT_Error_Code_UNAVAILABLE = 14
    PJRT_Error_Code_DATA_LOSS = 15
    PJRT_Error_Code_UNAUTHENTICATED = 16
end

# typedef void ( * PJRT_Error_PayloadVisitor ) ( const char * key , size_t key_size , const char * value , size_t value_size , void * user_arg )
const PJRT_Error_PayloadVisitor = Ptr{Cvoid}

csizeof(::Type{PJRT_Error_FunctionTable}) = Csize_t(56)

csizeof(::Type{PJRT_Error}) = Csize_t(8)

struct PJRT_Error_GetCode_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    error::Ptr{PJRT_Error}
    code::PJRT_Error_Code
    function PJRT_Error_GetCode_Args(extension_start, error, code)
        return new(csizeof(PJRT_Error_GetCode_Args), extension_start, error, code)
    end
end

csizeof(::Type{PJRT_Error_GetCode_Args}) = Csize_t(28)

# typedef PJRT_Error * PJRT_Error_GetCode ( PJRT_Error_GetCode_Args * args )
const PJRT_Error_GetCode = Cvoid

struct PJRT_Error_ForEachPayload_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    error::Ptr{PJRT_Error}
    visitor::PJRT_Error_PayloadVisitor
    user_arg::Ptr{Cvoid}
    function PJRT_Error_ForEachPayload_Args(extension_start, error, visitor, user_arg)
        return new(
            csizeof(PJRT_Error_ForEachPayload_Args),
            extension_start,
            error,
            visitor,
            user_arg,
        )
    end
end

csizeof(::Type{PJRT_Error_ForEachPayload_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Error_ForEachPayload ( PJRT_Error_ForEachPayload_Args * args )
const PJRT_Error_ForEachPayload = Cvoid

# typedef PJRT_Error * ( * PJRT_CallbackError ) ( PJRT_Error_Code code , const char * message , size_t message_size )
const PJRT_CallbackError = Ptr{Cvoid}

@cenum PJRT_NamedValue_Type::UInt32 begin
    PJRT_NamedValue_kString = 0
    PJRT_NamedValue_kInt64 = 1
    PJRT_NamedValue_kInt64List = 2
    PJRT_NamedValue_kFloat = 3
    PJRT_NamedValue_kBool = 4
end

struct PJRT_NamedValue
    data::NTuple{56,UInt8}
end

function Base.getproperty(x::Ptr{PJRT_NamedValue}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{PJRT_Extension_Base}}(x + 8)
    f === :name && return Ptr{Cstring}(x + 16)
    f === :name_size && return Ptr{Csize_t}(x + 24)
    f === :type && return Ptr{PJRT_NamedValue_Type}(x + 32)
    f === :string_value && return Ptr{Cstring}(x + 40)
    f === :int64_value && return Ptr{Int64}(x + 40)
    f === :int64_array_value && return Ptr{Ptr{Int64}}(x + 40)
    f === :float_value && return Ptr{Cfloat}(x + 40)
    f === :bool_value && return Ptr{Bool}(x + 40)
    f === :value_size && return Ptr{Csize_t}(x + 48)
    return getfield(x, f)
end

function Base.getproperty(x::PJRT_NamedValue, f::Symbol)
    r = Ref{PJRT_NamedValue}(x)
    ptr = Base.unsafe_convert(Ptr{PJRT_NamedValue}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{PJRT_NamedValue}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::PJRT_NamedValue, private::Bool=false)
    return (
        :struct_size,
        :extension_start,
        :name,
        :name_size,
        :type,
        :string_value,
        :int64_value,
        :int64_array_value,
        :float_value,
        :bool_value,
        :value_size,
        if private
            fieldnames(typeof(x))
        else
            ()
        end...,
    )
end

csizeof(::Type{PJRT_NamedValue}) = Csize_t(56)

struct PJRT_Plugin_Initialize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    function PJRT_Plugin_Initialize_Args(extension_start)
        return new(csizeof(PJRT_Plugin_Initialize_Args), extension_start)
    end
end

csizeof(::Type{PJRT_Plugin_Initialize_Args}) = Csize_t(16)

# typedef PJRT_Error * PJRT_Plugin_Initialize ( PJRT_Plugin_Initialize_Args * args )
const PJRT_Plugin_Initialize = Cvoid

struct PJRT_Plugin_Attributes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    attributes::Ptr{PJRT_NamedValue}
    num_attributes::Csize_t
    function PJRT_Plugin_Attributes_Args(extension_start, attributes, num_attributes)
        return new(
            csizeof(PJRT_Plugin_Attributes_Args),
            extension_start,
            attributes,
            num_attributes,
        )
    end
end

csizeof(::Type{PJRT_Plugin_Attributes_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Plugin_Attributes ( PJRT_Plugin_Attributes_Args * args )
const PJRT_Plugin_Attributes = Cvoid

mutable struct PJRT_Event end

struct PJRT_Event_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
    function PJRT_Event_Destroy_Args(extension_start, event)
        return new(csizeof(PJRT_Event_Destroy_Args), extension_start, event)
    end
end

csizeof(::Type{PJRT_Event_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Event_Destroy ( PJRT_Event_Destroy_Args * args )
const PJRT_Event_Destroy = Cvoid

struct PJRT_Event_IsReady_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
    is_ready::Bool
    function PJRT_Event_IsReady_Args(extension_start, event, is_ready)
        return new(csizeof(PJRT_Event_IsReady_Args), extension_start, event, is_ready)
    end
end

csizeof(::Type{PJRT_Event_IsReady_Args}) = Csize_t(25)

# typedef PJRT_Error * PJRT_Event_IsReady ( PJRT_Event_IsReady_Args * args )
const PJRT_Event_IsReady = Cvoid

struct PJRT_Event_Error_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
    function PJRT_Event_Error_Args(extension_start, event)
        return new(csizeof(PJRT_Event_Error_Args), extension_start, event)
    end
end

csizeof(::Type{PJRT_Event_Error_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Event_Error ( PJRT_Event_Error_Args * args )
const PJRT_Event_Error = Cvoid

struct PJRT_Event_Await_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
    function PJRT_Event_Await_Args(extension_start, event)
        return new(csizeof(PJRT_Event_Await_Args), extension_start, event)
    end
end

csizeof(::Type{PJRT_Event_Await_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Event_Await ( PJRT_Event_Await_Args * args )
const PJRT_Event_Await = Cvoid

# typedef void ( * PJRT_Event_OnReadyCallback ) ( PJRT_Error * error , void * user_arg )
const PJRT_Event_OnReadyCallback = Ptr{Cvoid}

struct PJRT_Event_OnReady_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
    callback::PJRT_Event_OnReadyCallback
    user_arg::Ptr{Cvoid}
    function PJRT_Event_OnReady_Args(extension_start, event, callback, user_arg)
        return new(
            csizeof(PJRT_Event_OnReady_Args), extension_start, event, callback, user_arg
        )
    end
end

csizeof(::Type{PJRT_Event_OnReady_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Event_OnReady ( PJRT_Event_OnReady_Args * args )
const PJRT_Event_OnReady = Cvoid

struct PJRT_Event_Create_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
    function PJRT_Event_Create_Args(extension_start, event)
        return new(csizeof(PJRT_Event_Create_Args), extension_start, event)
    end
end

csizeof(::Type{PJRT_Event_Create_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Event_Create ( PJRT_Event_Create_Args * args )
const PJRT_Event_Create = Cvoid

struct PJRT_Event_Set_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
    function PJRT_Event_Set_Args(
        extension_start, event, error_code, error_message, error_message_size
    )
        return new(
            csizeof(PJRT_Event_Set_Args),
            extension_start,
            event,
            error_code,
            error_message,
            error_message_size,
        )
    end
end

csizeof(::Type{PJRT_Event_Set_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Event_Set ( PJRT_Event_Set_Args * args )
const PJRT_Event_Set = Cvoid

mutable struct PJRT_Client end

mutable struct PJRT_Device end

struct PJRT_Memory_FunctionTable
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    instance_struct_size::Csize_t
    get_user_data::Ptr{Cvoid}
    set_user_data::Ptr{Cvoid}
end

struct PJRT_Memory
    vtable::Ptr{PJRT_Memory_FunctionTable}
end

csizeof(::Type{PJRT_Memory_FunctionTable}) = Csize_t(40)

csizeof(::Type{PJRT_Memory}) = Csize_t(8)

@cenum PJRT_Buffer_Type::UInt32 begin
    PJRT_Buffer_Type_INVALID = 0
    PJRT_Buffer_Type_PRED = 1
    PJRT_Buffer_Type_S8 = 2
    PJRT_Buffer_Type_S16 = 3
    PJRT_Buffer_Type_S32 = 4
    PJRT_Buffer_Type_S64 = 5
    PJRT_Buffer_Type_U8 = 6
    PJRT_Buffer_Type_U16 = 7
    PJRT_Buffer_Type_U32 = 8
    PJRT_Buffer_Type_U64 = 9
    PJRT_Buffer_Type_F16 = 10
    PJRT_Buffer_Type_F32 = 11
    PJRT_Buffer_Type_F64 = 12
    PJRT_Buffer_Type_BF16 = 13
    PJRT_Buffer_Type_C64 = 14
    PJRT_Buffer_Type_C128 = 15
    PJRT_Buffer_Type_F8E5M2 = 16
    PJRT_Buffer_Type_F8E4M3FN = 17
    PJRT_Buffer_Type_F8E4M3B11FNUZ = 18
    PJRT_Buffer_Type_F8E5M2FNUZ = 19
    PJRT_Buffer_Type_F8E4M3FNUZ = 20
    PJRT_Buffer_Type_S4 = 21
    PJRT_Buffer_Type_U4 = 22
    PJRT_Buffer_Type_TOKEN = 23
    PJRT_Buffer_Type_S2 = 24
    PJRT_Buffer_Type_U2 = 25
    PJRT_Buffer_Type_F8E4M3 = 26
    PJRT_Buffer_Type_F8E3M4 = 27
    PJRT_Buffer_Type_F8E8M0FNU = 28
    PJRT_Buffer_Type_F4E2M1FN = 29
    PJRT_Buffer_Type_S1 = 30
    PJRT_Buffer_Type_U1 = 31
    PJRT_Buffer_Type_F6E2M3FN = 32
    PJRT_Buffer_Type_F6E3M2FN = 33
end

struct PJRT_ShapeSpec
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    dims::Ptr{Int64}
    num_dims::Csize_t
    element_type::PJRT_Buffer_Type
end

mutable struct PJRT_DeviceDescription end

mutable struct PJRT_TopologyDescription end

mutable struct PJRT_Executable end

mutable struct PJRT_LoadedExecutable end

mutable struct PJRT_Buffer end

mutable struct PJRT_FulfillAliasBufferCallback end

mutable struct PJRT_AsyncHostToDeviceTransferManager end

mutable struct PJRT_PhaseCompiler end

# typedef void ( * PJRT_KeyValueGetCallback_ValueDeleter ) ( char * value )
const PJRT_KeyValueGetCallback_ValueDeleter = Ptr{Cvoid}

struct PJRT_KeyValueGetCallback_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    key::Cstring
    key_size::Csize_t
    timeout_in_ms::Cint
    callback_error::Ptr{PJRT_CallbackError}
    user_arg::Ptr{Cvoid}
    value::Cstring
    value_size::Csize_t
    value_deleter_callback::PJRT_KeyValueGetCallback_ValueDeleter
    function PJRT_KeyValueGetCallback_Args(
        extension_start,
        key,
        key_size,
        timeout_in_ms,
        callback_error,
        user_arg,
        value,
        value_size,
        value_deleter_callback,
    )
        return new(
            csizeof(PJRT_KeyValueGetCallback_Args),
            extension_start,
            key,
            key_size,
            timeout_in_ms,
            callback_error,
            user_arg,
            value,
            value_size,
            value_deleter_callback,
        )
    end
end

csizeof(::Type{PJRT_KeyValueGetCallback_Args}) = Csize_t(80)

# typedef PJRT_Error * ( * PJRT_KeyValueGetCallback ) ( PJRT_KeyValueGetCallback_Args * args )
const PJRT_KeyValueGetCallback = Ptr{Cvoid}

# typedef void ( * PJRT_KeyValueTryGetCallback_ValueDeleter ) ( char * value )
const PJRT_KeyValueTryGetCallback_ValueDeleter = Ptr{Cvoid}

struct PJRT_KeyValueTryGetCallback_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    key::Cstring
    key_size::Csize_t
    callback_error::Ptr{PJRT_CallbackError}
    user_arg::Ptr{Cvoid}
    value::Cstring
    value_size::Csize_t
    value_deleter_callback::PJRT_KeyValueTryGetCallback_ValueDeleter
    function PJRT_KeyValueTryGetCallback_Args(
        extension_start,
        key,
        key_size,
        callback_error,
        user_arg,
        value,
        value_size,
        value_deleter_callback,
    )
        return new(
            csizeof(PJRT_KeyValueTryGetCallback_Args),
            extension_start,
            key,
            key_size,
            callback_error,
            user_arg,
            value,
            value_size,
            value_deleter_callback,
        )
    end
end

csizeof(::Type{PJRT_KeyValueTryGetCallback_Args}) = Csize_t(72)

# typedef PJRT_Error * ( * PJRT_KeyValueTryGetCallback ) ( PJRT_KeyValueTryGetCallback_Args * args )
const PJRT_KeyValueTryGetCallback = Ptr{Cvoid}

struct PJRT_KeyValuePutCallback_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    key::Cstring
    key_size::Csize_t
    value::Cstring
    value_size::Csize_t
    callback_error::Ptr{PJRT_CallbackError}
    user_arg::Ptr{Cvoid}
    function PJRT_KeyValuePutCallback_Args(
        extension_start, key, key_size, value, value_size, callback_error, user_arg
    )
        return new(
            csizeof(PJRT_KeyValuePutCallback_Args),
            extension_start,
            key,
            key_size,
            value,
            value_size,
            callback_error,
            user_arg,
        )
    end
end

csizeof(::Type{PJRT_KeyValuePutCallback_Args}) = Csize_t(64)

# typedef PJRT_Error * ( * PJRT_KeyValuePutCallback ) ( PJRT_KeyValuePutCallback_Args * args )
const PJRT_KeyValuePutCallback = Ptr{Cvoid}

struct PJRT_Client_Create_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    create_options::Ptr{PJRT_NamedValue}
    num_options::Csize_t
    kv_get_callback::PJRT_KeyValueGetCallback
    kv_get_user_arg::Ptr{Cvoid}
    kv_put_callback::PJRT_KeyValuePutCallback
    kv_put_user_arg::Ptr{Cvoid}
    client::Ptr{PJRT_Client}
    kv_try_get_callback::PJRT_KeyValueTryGetCallback
    kv_try_get_user_arg::Ptr{Cvoid}
    function PJRT_Client_Create_Args(
        extension_start,
        create_options,
        num_options,
        kv_get_callback,
        kv_get_user_arg,
        kv_put_callback,
        kv_put_user_arg,
        client,
        kv_try_get_callback,
        kv_try_get_user_arg,
    )
        return new(
            csizeof(PJRT_Client_Create_Args),
            extension_start,
            create_options,
            num_options,
            kv_get_callback,
            kv_get_user_arg,
            kv_put_callback,
            kv_put_user_arg,
            client,
            kv_try_get_callback,
            kv_try_get_user_arg,
        )
    end
end

csizeof(::Type{PJRT_Client_Create_Args}) = Csize_t(88)

# typedef PJRT_Error * PJRT_Client_Create ( PJRT_Client_Create_Args * args )
const PJRT_Client_Create = Cvoid

struct PJRT_Client_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    function PJRT_Client_Destroy_Args(extension_start, client)
        return new(csizeof(PJRT_Client_Destroy_Args), extension_start, client)
    end
end

csizeof(::Type{PJRT_Client_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Client_Destroy ( PJRT_Client_Destroy_Args * args )
const PJRT_Client_Destroy = Cvoid

struct PJRT_Client_PlatformName_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    platform_name::Cstring
    platform_name_size::Csize_t
    function PJRT_Client_PlatformName_Args(
        extension_start, client, platform_name, platform_name_size
    )
        return new(
            csizeof(PJRT_Client_PlatformName_Args),
            extension_start,
            client,
            platform_name,
            platform_name_size,
        )
    end
end

csizeof(::Type{PJRT_Client_PlatformName_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Client_PlatformName ( PJRT_Client_PlatformName_Args * args )
const PJRT_Client_PlatformName = Cvoid

struct PJRT_Client_ProcessIndex_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    process_index::Cint
    function PJRT_Client_ProcessIndex_Args(extension_start, client, process_index)
        return new(
            csizeof(PJRT_Client_ProcessIndex_Args), extension_start, client, process_index
        )
    end
end

csizeof(::Type{PJRT_Client_ProcessIndex_Args}) = Csize_t(28)

# typedef PJRT_Error * PJRT_Client_ProcessIndex ( PJRT_Client_ProcessIndex_Args * args )
const PJRT_Client_ProcessIndex = Cvoid

struct PJRT_Client_PlatformVersion_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    platform_version::Cstring
    platform_version_size::Csize_t
    function PJRT_Client_PlatformVersion_Args(
        extension_start, client, platform_version, platform_version_size
    )
        return new(
            csizeof(PJRT_Client_PlatformVersion_Args),
            extension_start,
            client,
            platform_version,
            platform_version_size,
        )
    end
end

csizeof(::Type{PJRT_Client_PlatformVersion_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Client_PlatformVersion ( PJRT_Client_PlatformVersion_Args * args )
const PJRT_Client_PlatformVersion = Cvoid

struct PJRT_Client_TopologyDescription_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    topology::Ptr{PJRT_TopologyDescription}
    function PJRT_Client_TopologyDescription_Args(extension_start, client, topology)
        return new(
            csizeof(PJRT_Client_TopologyDescription_Args), extension_start, client, topology
        )
    end
end

csizeof(::Type{PJRT_Client_TopologyDescription_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Client_TopologyDescription ( PJRT_Client_TopologyDescription_Args * args )
const PJRT_Client_TopologyDescription = Cvoid

struct PJRT_Client_Devices_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    devices::Ptr{Ptr{PJRT_Device}}
    num_devices::Csize_t
    function PJRT_Client_Devices_Args(extension_start, client, devices, num_devices)
        return new(
            csizeof(PJRT_Client_Devices_Args), extension_start, client, devices, num_devices
        )
    end
end

csizeof(::Type{PJRT_Client_Devices_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Client_Devices ( PJRT_Client_Devices_Args * args )
const PJRT_Client_Devices = Cvoid

struct PJRT_Client_AddressableDevices_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    addressable_devices::Ptr{Ptr{PJRT_Device}}
    num_addressable_devices::Csize_t
    function PJRT_Client_AddressableDevices_Args(
        extension_start, client, addressable_devices, num_addressable_devices
    )
        return new(
            csizeof(PJRT_Client_AddressableDevices_Args),
            extension_start,
            client,
            addressable_devices,
            num_addressable_devices,
        )
    end
end

csizeof(::Type{PJRT_Client_AddressableDevices_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Client_AddressableDevices ( PJRT_Client_AddressableDevices_Args * args )
const PJRT_Client_AddressableDevices = Cvoid

struct PJRT_Client_LookupDevice_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    id::Cint
    device::Ptr{PJRT_Device}
    function PJRT_Client_LookupDevice_Args(extension_start, client, id, device)
        return new(
            csizeof(PJRT_Client_LookupDevice_Args), extension_start, client, id, device
        )
    end
end

csizeof(::Type{PJRT_Client_LookupDevice_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Client_LookupDevice ( PJRT_Client_LookupDevice_Args * args )
const PJRT_Client_LookupDevice = Cvoid

struct PJRT_Client_LookupAddressableDevice_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    local_hardware_id::Cint
    addressable_device::Ptr{PJRT_Device}
    function PJRT_Client_LookupAddressableDevice_Args(
        extension_start, client, local_hardware_id, addressable_device
    )
        return new(
            csizeof(PJRT_Client_LookupAddressableDevice_Args),
            extension_start,
            client,
            local_hardware_id,
            addressable_device,
        )
    end
end

csizeof(::Type{PJRT_Client_LookupAddressableDevice_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Client_LookupAddressableDevice ( PJRT_Client_LookupAddressableDevice_Args * args )
const PJRT_Client_LookupAddressableDevice = Cvoid

@cenum PJRT_ProcessState::UInt32 begin
    PJRT_ProcessState_kUnspecified = 0
    PJRT_ProcessState_kUninitialized = 1
    PJRT_ProcessState_kDisconnected = 2
    PJRT_ProcessState_kConnected = 3
    PJRT_ProcessState_kError = 4
end

struct PJRT_ProcessInfo
    struct_size::Csize_t
    task_id::Cint
    incarnation_id::UInt64
    state::PJRT_ProcessState
    error_code::Cint
    error_message::Cstring
    error_message_size::Csize_t
end

csizeof(::Type{PJRT_ProcessInfo}) = Csize_t(48)

struct PJRT_Client_UpdateGlobalProcessInfo_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    process_infos::Ptr{PJRT_ProcessInfo}
    num_process_infos::Csize_t
    function PJRT_Client_UpdateGlobalProcessInfo_Args(
        extension_start, client, process_infos, num_process_infos
    )
        return new(
            csizeof(PJRT_Client_UpdateGlobalProcessInfo_Args),
            extension_start,
            client,
            process_infos,
            num_process_infos,
        )
    end
end

csizeof(::Type{PJRT_Client_UpdateGlobalProcessInfo_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Client_UpdateGlobalProcessInfo ( PJRT_Client_UpdateGlobalProcessInfo_Args * args )
const PJRT_Client_UpdateGlobalProcessInfo = Cvoid

struct PJRT_Client_AddressableMemories_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    addressable_memories::Ptr{Ptr{PJRT_Memory}}
    num_addressable_memories::Csize_t
    function PJRT_Client_AddressableMemories_Args(
        extension_start, client, addressable_memories, num_addressable_memories
    )
        return new(
            csizeof(PJRT_Client_AddressableMemories_Args),
            extension_start,
            client,
            addressable_memories,
            num_addressable_memories,
        )
    end
end

csizeof(::Type{PJRT_Client_AddressableMemories_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Client_AddressableMemories ( PJRT_Client_AddressableMemories_Args * args )
const PJRT_Client_AddressableMemories = Cvoid

struct PJRT_Program
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    code::Cstring
    code_size::Csize_t
    format::Cstring
    format_size::Csize_t
end

csizeof(::Type{PJRT_Program}) = Csize_t(48)

struct PJRT_Client_Compile_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    program::Ptr{PJRT_Program}
    compile_options::Cstring
    compile_options_size::Csize_t
    executable::Ptr{PJRT_LoadedExecutable}
    function PJRT_Client_Compile_Args(
        extension_start, client, program, compile_options, compile_options_size, executable
    )
        return new(
            csizeof(PJRT_Client_Compile_Args),
            extension_start,
            client,
            program,
            compile_options,
            compile_options_size,
            executable,
        )
    end
end

csizeof(::Type{PJRT_Client_Compile_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_Client_Compile ( PJRT_Client_Compile_Args * args )
const PJRT_Client_Compile = Cvoid

struct PJRT_Client_Load_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    executable::Ptr{PJRT_Executable}
    compile_options::Cstring
    compile_options_size::Csize_t
    loaded_executable::Ptr{PJRT_LoadedExecutable}
    function PJRT_Client_Load_Args(
        extension_start,
        client,
        executable,
        compile_options,
        compile_options_size,
        loaded_executable,
    )
        return new(
            csizeof(PJRT_Client_Load_Args),
            extension_start,
            client,
            executable,
            compile_options,
            compile_options_size,
            loaded_executable,
        )
    end
end

csizeof(::Type{PJRT_Client_Load_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_Client_Load ( PJRT_Client_Load_Args * args )
const PJRT_Client_Load = Cvoid

struct PJRT_Client_DefaultDeviceAssignment_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    num_replicas::Cint
    num_partitions::Cint
    default_assignment_size::Csize_t
    default_assignment::Ptr{Cint}
    function PJRT_Client_DefaultDeviceAssignment_Args(
        extension_start,
        client,
        num_replicas,
        num_partitions,
        default_assignment_size,
        default_assignment,
    )
        return new(
            csizeof(PJRT_Client_DefaultDeviceAssignment_Args),
            extension_start,
            client,
            num_replicas,
            num_partitions,
            default_assignment_size,
            default_assignment,
        )
    end
end

csizeof(::Type{PJRT_Client_DefaultDeviceAssignment_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Client_DefaultDeviceAssignment ( PJRT_Client_DefaultDeviceAssignment_Args * args )
const PJRT_Client_DefaultDeviceAssignment = Cvoid

struct PJRT_Client_DmaMap_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    data::Ptr{Cvoid}
    size::Csize_t
    function PJRT_Client_DmaMap_Args(extension_start, client, data, size)
        return new(csizeof(PJRT_Client_DmaMap_Args), extension_start, client, data, size)
    end
end

csizeof(::Type{PJRT_Client_DmaMap_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Client_DmaMap ( PJRT_Client_DmaMap_Args * args )
const PJRT_Client_DmaMap = Cvoid

struct PJRT_Client_DmaUnmap_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    data::Ptr{Cvoid}
    function PJRT_Client_DmaUnmap_Args(extension_start, client, data)
        return new(csizeof(PJRT_Client_DmaUnmap_Args), extension_start, client, data)
    end
end

csizeof(::Type{PJRT_Client_DmaUnmap_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Client_DmaUnmap ( PJRT_Client_DmaUnmap_Args * args )
const PJRT_Client_DmaUnmap = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    function PJRT_AsyncHostToDeviceTransferManager_Destroy_Args(
        extension_start, transfer_manager
    )
        return new(
            csizeof(PJRT_AsyncHostToDeviceTransferManager_Destroy_Args),
            extension_start,
            transfer_manager,
        )
    end
end

csizeof(::Type{PJRT_AsyncHostToDeviceTransferManager_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_Destroy ( PJRT_AsyncHostToDeviceTransferManager_Destroy_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_Destroy = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_TransferData_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    buffer_index::Cint
    data::Ptr{Cvoid}
    offset::Int64
    transfer_size::Int64
    is_last_transfer::Bool
    done_with_h2d_transfer::Ptr{PJRT_Event}
    function PJRT_AsyncHostToDeviceTransferManager_TransferData_Args(
        extension_start,
        transfer_manager,
        buffer_index,
        data,
        offset,
        transfer_size,
        is_last_transfer,
        done_with_h2d_transfer,
    )
        return new(
            csizeof(PJRT_AsyncHostToDeviceTransferManager_TransferData_Args),
            extension_start,
            transfer_manager,
            buffer_index,
            data,
            offset,
            transfer_size,
            is_last_transfer,
            done_with_h2d_transfer,
        )
    end
end

csizeof(::Type{PJRT_AsyncHostToDeviceTransferManager_TransferData_Args}) = Csize_t(72)

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_TransferData ( PJRT_AsyncHostToDeviceTransferManager_TransferData_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_TransferData = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    buffer_index::Cint
    buffer_out::Ptr{PJRT_Buffer}
    function PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args(
        extension_start, transfer_manager, buffer_index, buffer_out
    )
        return new(
            csizeof(PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args),
            extension_start,
            transfer_manager,
            buffer_index,
            buffer_out,
        )
    end
end

csizeof(::Type{PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer ( PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_Device_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    device_out::Ptr{PJRT_Device}
    function PJRT_AsyncHostToDeviceTransferManager_Device_Args(
        extension_start, transfer_manager, device_out
    )
        return new(
            csizeof(PJRT_AsyncHostToDeviceTransferManager_Device_Args),
            extension_start,
            transfer_manager,
            device_out,
        )
    end
end

csizeof(::Type{PJRT_AsyncHostToDeviceTransferManager_Device_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_Device ( PJRT_AsyncHostToDeviceTransferManager_Device_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_Device = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    buffer_count::Csize_t
    function PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args(
        extension_start, transfer_manager, buffer_count
    )
        return new(
            csizeof(PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args),
            extension_start,
            transfer_manager,
            buffer_count,
        )
    end
end

csizeof(::Type{PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_BufferCount ( PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_BufferCount = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    buffer_index::Cint
    buffer_size::Csize_t
    function PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args(
        extension_start, transfer_manager, buffer_index, buffer_size
    )
        return new(
            csizeof(PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args),
            extension_start,
            transfer_manager,
            buffer_index,
            buffer_size,
        )
    end
end

csizeof(::Type{PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_BufferSize ( PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_BufferSize = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    buffer_index::Cint
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
    function PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args(
        extension_start,
        transfer_manager,
        buffer_index,
        error_code,
        error_message,
        error_message_size,
    )
        return new(
            csizeof(PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args),
            extension_start,
            transfer_manager,
            buffer_index,
            error_code,
            error_message,
            error_message_size,
        )
    end
end

csizeof(::Type{PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_SetBufferError ( PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_SetBufferError = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    transfer_metadata::Ptr{PJRT_NamedValue}
    num_metadata::Csize_t
    function PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args(
        extension_start, transfer_manager, transfer_metadata, num_metadata
    )
        return new(
            csizeof(PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args),
            extension_start,
            transfer_manager,
            transfer_metadata,
            num_metadata,
        )
    end
end

csizeof(::Type{PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_AddMetadata ( PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_AddMetadata = Cvoid

@cenum PJRT_HostBufferSemantics::UInt32 begin
    PJRT_HostBufferSemantics_kImmutableOnlyDuringCall = 0
    PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes = 1
    PJRT_HostBufferSemantics_kImmutableZeroCopy = 2
    PJRT_HostBufferSemantics_kMutableZeroCopy = 3
end

@cenum PJRT_Buffer_MemoryLayout_Type::UInt32 begin
    PJRT_Buffer_MemoryLayout_Type_Tiled = 0
    PJRT_Buffer_MemoryLayout_Type_Strides = 1
end

struct PJRT_Buffer_MemoryLayout_Tiled
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    minor_to_major::Ptr{Int64}
    minor_to_major_size::Csize_t
    tile_dims::Ptr{Int64}
    tile_dim_sizes::Ptr{Csize_t}
    num_tiles::Csize_t
end

csizeof(::Type{PJRT_Buffer_MemoryLayout_Tiled}) = Csize_t(56)

struct PJRT_Buffer_MemoryLayout_Strides
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    byte_strides::Ptr{Int64}
    num_byte_strides::Csize_t
end

csizeof(::Type{PJRT_Buffer_MemoryLayout_Strides}) = Csize_t(32)

struct PJRT_Buffer_MemoryLayout
    data::NTuple{80,UInt8}
end

function Base.getproperty(x::Ptr{PJRT_Buffer_MemoryLayout}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{PJRT_Extension_Base}}(x + 8)
    f === :tiled && return Ptr{PJRT_Buffer_MemoryLayout_Tiled}(x + 16)
    f === :strides && return Ptr{PJRT_Buffer_MemoryLayout_Strides}(x + 16)
    f === :type && return Ptr{PJRT_Buffer_MemoryLayout_Type}(x + 72)
    return getfield(x, f)
end

function Base.getproperty(x::PJRT_Buffer_MemoryLayout, f::Symbol)
    r = Ref{PJRT_Buffer_MemoryLayout}(x)
    ptr = Base.unsafe_convert(Ptr{PJRT_Buffer_MemoryLayout}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{PJRT_Buffer_MemoryLayout}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::PJRT_Buffer_MemoryLayout, private::Bool=false)
    return (
        :struct_size,
        :extension_start,
        :tiled,
        :strides,
        :type,
        if private
            fieldnames(typeof(x))
        else
            ()
        end...,
    )
end

csizeof(::Type{PJRT_Buffer_MemoryLayout}) = Csize_t(76)

struct PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    buffer_index::Cint
    data::Ptr{Cvoid}
    shape_dims::Ptr{Int64}
    shape_num_dims::Csize_t
    shape_element_type::PJRT_Buffer_Type
    shape_layout::Ptr{PJRT_Buffer_MemoryLayout}
    done_with_h2d_transfer::Ptr{PJRT_Event}
    function PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args(
        extension_start,
        transfer_manager,
        buffer_index,
        data,
        shape_dims,
        shape_num_dims,
        shape_element_type,
        shape_layout,
        done_with_h2d_transfer,
    )
        return new(
            csizeof(PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args),
            extension_start,
            transfer_manager,
            buffer_index,
            data,
            shape_dims,
            shape_num_dims,
            shape_element_type,
            shape_layout,
            done_with_h2d_transfer,
        )
    end
end

csizeof(::Type{PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args}) = Csize_t(80)

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_TransferLiteral ( PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_TransferLiteral = Cvoid

struct PJRT_Client_CreateUninitializedBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    shape_dims::Ptr{Int64}
    shape_num_dims::Csize_t
    shape_element_type::PJRT_Buffer_Type
    shape_layout::Ptr{PJRT_Buffer_MemoryLayout}
    device::Ptr{PJRT_Device}
    memory::Ptr{PJRT_Memory}
    buffer::Ptr{PJRT_Buffer}
    function PJRT_Client_CreateUninitializedBuffer_Args(
        extension_start,
        client,
        shape_dims,
        shape_num_dims,
        shape_element_type,
        shape_layout,
        device,
        memory,
        buffer,
    )
        return new(
            csizeof(PJRT_Client_CreateUninitializedBuffer_Args),
            extension_start,
            client,
            shape_dims,
            shape_num_dims,
            shape_element_type,
            shape_layout,
            device,
            memory,
            buffer,
        )
    end
end

csizeof(::Type{PJRT_Client_CreateUninitializedBuffer_Args}) = Csize_t(80)

# typedef PJRT_Error * PJRT_Client_CreateUninitializedBuffer ( PJRT_Client_CreateUninitializedBuffer_Args * args )
const PJRT_Client_CreateUninitializedBuffer = Cvoid

struct PJRT_Client_CreateErrorBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
    shape_dims::Ptr{Int64}
    shape_num_dims::Csize_t
    shape_element_type::PJRT_Buffer_Type
    shape_layout::Ptr{PJRT_Buffer_MemoryLayout}
    memory::Ptr{PJRT_Memory}
    buffer::Ptr{PJRT_Buffer}
    payload::Ptr{PJRT_NamedValue}
    num_payload::Csize_t
    device_layout::Ptr{PJRT_Buffer_MemoryLayout}
    function PJRT_Client_CreateErrorBuffer_Args(
        extension_start,
        client,
        error_code,
        error_message,
        error_message_size,
        shape_dims,
        shape_num_dims,
        shape_element_type,
        shape_layout,
        memory,
        buffer,
        payload,
        num_payload,
        device_layout,
    )
        return new(
            csizeof(PJRT_Client_CreateErrorBuffer_Args),
            extension_start,
            client,
            error_code,
            error_message,
            error_message_size,
            shape_dims,
            shape_num_dims,
            shape_element_type,
            shape_layout,
            memory,
            buffer,
            payload,
            num_payload,
            device_layout,
        )
    end
end

csizeof(::Type{PJRT_Client_CreateErrorBuffer_Args}) = Csize_t(120)

# typedef PJRT_Error * PJRT_Client_CreateErrorBuffer ( PJRT_Client_CreateErrorBuffer_Args * args )
const PJRT_Client_CreateErrorBuffer = Cvoid

struct PJRT_Client_CreateAliasBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    memory::Ptr{PJRT_Memory}
    shape_dims::Ptr{Int64}
    shape_num_dims::Csize_t
    shape_element_type::PJRT_Buffer_Type
    shape_layout::Ptr{PJRT_Buffer_MemoryLayout}
    alias_buffer::Ptr{PJRT_Buffer}
    fulfill_alias_buffer_cb::Ptr{PJRT_FulfillAliasBufferCallback}
    function PJRT_Client_CreateAliasBuffer_Args(
        extension_start,
        client,
        memory,
        shape_dims,
        shape_num_dims,
        shape_element_type,
        shape_layout,
        alias_buffer,
        fulfill_alias_buffer_cb,
    )
        return new(
            csizeof(PJRT_Client_CreateAliasBuffer_Args),
            extension_start,
            client,
            memory,
            shape_dims,
            shape_num_dims,
            shape_element_type,
            shape_layout,
            alias_buffer,
            fulfill_alias_buffer_cb,
        )
    end
end

csizeof(::Type{PJRT_Client_CreateAliasBuffer_Args}) = Csize_t(80)

# typedef PJRT_Error * PJRT_Client_CreateAliasBuffer ( PJRT_Client_CreateAliasBuffer_Args * args )
const PJRT_Client_CreateAliasBuffer = Cvoid

struct PJRT_Client_FulfillAliasBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    buffer::Ptr{PJRT_Buffer}
    status_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
    fulfill_alias_buffer_cb::Ptr{PJRT_FulfillAliasBufferCallback}
    function PJRT_Client_FulfillAliasBuffer_Args(
        extension_start,
        client,
        buffer,
        status_code,
        error_message,
        error_message_size,
        fulfill_alias_buffer_cb,
    )
        return new(
            csizeof(PJRT_Client_FulfillAliasBuffer_Args),
            extension_start,
            client,
            buffer,
            status_code,
            error_message,
            error_message_size,
            fulfill_alias_buffer_cb,
        )
    end
end

csizeof(::Type{PJRT_Client_FulfillAliasBuffer_Args}) = Csize_t(64)

# typedef PJRT_Error * PJRT_Client_FulfillAliasBuffer ( PJRT_Client_FulfillAliasBuffer_Args * args )
const PJRT_Client_FulfillAliasBuffer = Cvoid

struct PJRT_Client_BufferFromHostBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    data::Ptr{Cvoid}
    type::PJRT_Buffer_Type
    dims::Ptr{Int64}
    num_dims::Csize_t
    byte_strides::Ptr{Int64}
    num_byte_strides::Csize_t
    host_buffer_semantics::PJRT_HostBufferSemantics
    device::Ptr{PJRT_Device}
    memory::Ptr{PJRT_Memory}
    device_layout::Ptr{PJRT_Buffer_MemoryLayout}
    done_with_host_buffer::Ptr{PJRT_Event}
    buffer::Ptr{PJRT_Buffer}
    function PJRT_Client_BufferFromHostBuffer_Args(
        extension_start,
        client,
        data,
        type,
        dims,
        num_dims,
        byte_strides,
        num_byte_strides,
        host_buffer_semantics,
        device,
        memory,
        device_layout,
        done_with_host_buffer,
        buffer,
    )
        return new(
            csizeof(PJRT_Client_BufferFromHostBuffer_Args),
            extension_start,
            client,
            data,
            type,
            dims,
            num_dims,
            byte_strides,
            num_byte_strides,
            host_buffer_semantics,
            device,
            memory,
            device_layout,
            done_with_host_buffer,
            buffer,
        )
    end
end

csizeof(::Type{PJRT_Client_BufferFromHostBuffer_Args}) = Csize_t(120)

# typedef PJRT_Error * PJRT_Client_BufferFromHostBuffer ( PJRT_Client_BufferFromHostBuffer_Args * args )
const PJRT_Client_BufferFromHostBuffer = Cvoid

struct PJRT_Client_CreateViewOfDeviceBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    device_buffer_ptr::Ptr{Cvoid}
    dims::Ptr{Int64}
    num_dims::Csize_t
    element_type::PJRT_Buffer_Type
    layout::Ptr{PJRT_Buffer_MemoryLayout}
    device::Ptr{PJRT_Device}
    on_delete_callback::Ptr{Cvoid}
    on_delete_callback_arg::Ptr{Cvoid}
    stream::Cptrdiff_t
    buffer::Ptr{PJRT_Buffer}
    memory::Ptr{PJRT_Memory}
    function PJRT_Client_CreateViewOfDeviceBuffer_Args(
        extension_start,
        client,
        device_buffer_ptr,
        dims,
        num_dims,
        element_type,
        layout,
        device,
        on_delete_callback,
        on_delete_callback_arg,
        stream,
        buffer,
        memory,
    )
        return new(
            csizeof(PJRT_Client_CreateViewOfDeviceBuffer_Args),
            extension_start,
            client,
            device_buffer_ptr,
            dims,
            num_dims,
            element_type,
            layout,
            device,
            on_delete_callback,
            on_delete_callback_arg,
            stream,
            buffer,
            memory,
        )
    end
end

csizeof(::Type{PJRT_Client_CreateViewOfDeviceBuffer_Args}) = Csize_t(112)

# typedef PJRT_Error * PJRT_Client_CreateViewOfDeviceBuffer ( PJRT_Client_CreateViewOfDeviceBuffer_Args * args )
const PJRT_Client_CreateViewOfDeviceBuffer = Cvoid

csizeof(::Type{PJRT_ShapeSpec}) = Csize_t(36)

struct PJRT_Client_CreateBuffersForAsyncHostToDevice_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    shape_specs::Ptr{PJRT_ShapeSpec}
    num_shape_specs::Csize_t
    device_layouts::Ptr{Ptr{PJRT_Buffer_MemoryLayout}}
    num_device_layouts::Csize_t
    memory::Ptr{PJRT_Memory}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    function PJRT_Client_CreateBuffersForAsyncHostToDevice_Args(
        extension_start,
        client,
        shape_specs,
        num_shape_specs,
        device_layouts,
        num_device_layouts,
        memory,
        transfer_manager,
    )
        return new(
            csizeof(PJRT_Client_CreateBuffersForAsyncHostToDevice_Args),
            extension_start,
            client,
            shape_specs,
            num_shape_specs,
            device_layouts,
            num_device_layouts,
            memory,
            transfer_manager,
        )
    end
end

csizeof(::Type{PJRT_Client_CreateBuffersForAsyncHostToDevice_Args}) = Csize_t(72)

# typedef PJRT_Error * PJRT_Client_CreateBuffersForAsyncHostToDevice ( PJRT_Client_CreateBuffersForAsyncHostToDevice_Args * args )
const PJRT_Client_CreateBuffersForAsyncHostToDevice = Cvoid

struct PJRT_DeviceDescription_Id_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    id::Cint
    function PJRT_DeviceDescription_Id_Args(extension_start, device_description, id)
        return new(
            csizeof(PJRT_DeviceDescription_Id_Args), extension_start, device_description, id
        )
    end
end

csizeof(::Type{PJRT_DeviceDescription_Id_Args}) = Csize_t(28)

# typedef PJRT_Error * PJRT_DeviceDescription_Id ( PJRT_DeviceDescription_Id_Args * args )
const PJRT_DeviceDescription_Id = Cvoid

struct PJRT_DeviceDescription_ProcessIndex_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    process_index::Cint
    function PJRT_DeviceDescription_ProcessIndex_Args(
        extension_start, device_description, process_index
    )
        return new(
            csizeof(PJRT_DeviceDescription_ProcessIndex_Args),
            extension_start,
            device_description,
            process_index,
        )
    end
end

csizeof(::Type{PJRT_DeviceDescription_ProcessIndex_Args}) = Csize_t(28)

# typedef PJRT_Error * PJRT_DeviceDescription_ProcessIndex ( PJRT_DeviceDescription_ProcessIndex_Args * args )
const PJRT_DeviceDescription_ProcessIndex = Cvoid

struct PJRT_DeviceDescription_Attributes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    num_attributes::Csize_t
    attributes::Ptr{PJRT_NamedValue}
    function PJRT_DeviceDescription_Attributes_Args(
        extension_start, device_description, num_attributes, attributes
    )
        return new(
            csizeof(PJRT_DeviceDescription_Attributes_Args),
            extension_start,
            device_description,
            num_attributes,
            attributes,
        )
    end
end

csizeof(::Type{PJRT_DeviceDescription_Attributes_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_DeviceDescription_Attributes ( PJRT_DeviceDescription_Attributes_Args * args )
const PJRT_DeviceDescription_Attributes = Cvoid

struct PJRT_DeviceDescription_Kind_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    device_kind::Cstring
    device_kind_size::Csize_t
    function PJRT_DeviceDescription_Kind_Args(
        extension_start, device_description, device_kind, device_kind_size
    )
        return new(
            csizeof(PJRT_DeviceDescription_Kind_Args),
            extension_start,
            device_description,
            device_kind,
            device_kind_size,
        )
    end
end

csizeof(::Type{PJRT_DeviceDescription_Kind_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_DeviceDescription_Kind ( PJRT_DeviceDescription_Kind_Args * args )
const PJRT_DeviceDescription_Kind = Cvoid

struct PJRT_DeviceDescription_DebugString_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    debug_string::Cstring
    debug_string_size::Csize_t
    function PJRT_DeviceDescription_DebugString_Args(
        extension_start, device_description, debug_string, debug_string_size
    )
        return new(
            csizeof(PJRT_DeviceDescription_DebugString_Args),
            extension_start,
            device_description,
            debug_string,
            debug_string_size,
        )
    end
end

csizeof(::Type{PJRT_DeviceDescription_DebugString_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_DeviceDescription_DebugString ( PJRT_DeviceDescription_DebugString_Args * args )
const PJRT_DeviceDescription_DebugString = Cvoid

struct PJRT_DeviceDescription_ToString_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    to_string::Cstring
    to_string_size::Csize_t
    function PJRT_DeviceDescription_ToString_Args(
        extension_start, device_description, to_string, to_string_size
    )
        return new(
            csizeof(PJRT_DeviceDescription_ToString_Args),
            extension_start,
            device_description,
            to_string,
            to_string_size,
        )
    end
end

csizeof(::Type{PJRT_DeviceDescription_ToString_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_DeviceDescription_ToString ( PJRT_DeviceDescription_ToString_Args * args )
const PJRT_DeviceDescription_ToString = Cvoid

struct PJRT_Device_GetDescription_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    device_description::Ptr{PJRT_DeviceDescription}
    function PJRT_Device_GetDescription_Args(extension_start, device, device_description)
        return new(
            csizeof(PJRT_Device_GetDescription_Args),
            extension_start,
            device,
            device_description,
        )
    end
end

csizeof(::Type{PJRT_Device_GetDescription_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Device_GetDescription ( PJRT_Device_GetDescription_Args * args )
const PJRT_Device_GetDescription = Cvoid

struct PJRT_Device_IsAddressable_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    is_addressable::Bool
    function PJRT_Device_IsAddressable_Args(extension_start, device, is_addressable)
        return new(
            csizeof(PJRT_Device_IsAddressable_Args), extension_start, device, is_addressable
        )
    end
end

csizeof(::Type{PJRT_Device_IsAddressable_Args}) = Csize_t(25)

# typedef PJRT_Error * PJRT_Device_IsAddressable ( PJRT_Device_IsAddressable_Args * args )
const PJRT_Device_IsAddressable = Cvoid

struct PJRT_Device_LocalHardwareId_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    local_hardware_id::Cint
    function PJRT_Device_LocalHardwareId_Args(extension_start, device, local_hardware_id)
        return new(
            csizeof(PJRT_Device_LocalHardwareId_Args),
            extension_start,
            device,
            local_hardware_id,
        )
    end
end

csizeof(::Type{PJRT_Device_LocalHardwareId_Args}) = Csize_t(28)

# typedef PJRT_Error * PJRT_Device_LocalHardwareId ( PJRT_Device_LocalHardwareId_Args * args )
const PJRT_Device_LocalHardwareId = Cvoid

struct PJRT_Device_AddressableMemories_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    memories::Ptr{Ptr{PJRT_Memory}}
    num_memories::Csize_t
    function PJRT_Device_AddressableMemories_Args(
        extension_start, device, memories, num_memories
    )
        return new(
            csizeof(PJRT_Device_AddressableMemories_Args),
            extension_start,
            device,
            memories,
            num_memories,
        )
    end
end

csizeof(::Type{PJRT_Device_AddressableMemories_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Device_AddressableMemories ( PJRT_Device_AddressableMemories_Args * args )
const PJRT_Device_AddressableMemories = Cvoid

struct PJRT_Device_DefaultMemory_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    memory::Ptr{PJRT_Memory}
    function PJRT_Device_DefaultMemory_Args(extension_start, device, memory)
        return new(csizeof(PJRT_Device_DefaultMemory_Args), extension_start, device, memory)
    end
end

csizeof(::Type{PJRT_Device_DefaultMemory_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Device_DefaultMemory ( PJRT_Device_DefaultMemory_Args * args )
const PJRT_Device_DefaultMemory = Cvoid

struct PJRT_Device_MemoryStats_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    bytes_in_use::Int64
    peak_bytes_in_use::Int64
    peak_bytes_in_use_is_set::Bool
    num_allocs::Int64
    num_allocs_is_set::Bool
    largest_alloc_size::Int64
    largest_alloc_size_is_set::Bool
    bytes_limit::Int64
    bytes_limit_is_set::Bool
    bytes_reserved::Int64
    bytes_reserved_is_set::Bool
    peak_bytes_reserved::Int64
    peak_bytes_reserved_is_set::Bool
    bytes_reservable_limit::Int64
    bytes_reservable_limit_is_set::Bool
    largest_free_block_bytes::Int64
    largest_free_block_bytes_is_set::Bool
    pool_bytes::Int64
    pool_bytes_is_set::Bool
    peak_pool_bytes::Int64
    peak_pool_bytes_is_set::Bool
    peak_allocated_bytes::Int64
    peak_allocated_bytes_is_set::Bool
    function PJRT_Device_MemoryStats_Args(
        extension_start,
        device,
        bytes_in_use,
        peak_bytes_in_use,
        peak_bytes_in_use_is_set,
        num_allocs,
        num_allocs_is_set,
        largest_alloc_size,
        largest_alloc_size_is_set,
        bytes_limit,
        bytes_limit_is_set,
        bytes_reserved,
        bytes_reserved_is_set,
        peak_bytes_reserved,
        peak_bytes_reserved_is_set,
        bytes_reservable_limit,
        bytes_reservable_limit_is_set,
        largest_free_block_bytes,
        largest_free_block_bytes_is_set,
        pool_bytes,
        pool_bytes_is_set,
        peak_pool_bytes,
        peak_pool_bytes_is_set,
        peak_allocated_bytes,
        peak_allocated_bytes_is_set,
    )
        return new(
            csizeof(PJRT_Device_MemoryStats_Args),
            extension_start,
            device,
            bytes_in_use,
            peak_bytes_in_use,
            peak_bytes_in_use_is_set,
            num_allocs,
            num_allocs_is_set,
            largest_alloc_size,
            largest_alloc_size_is_set,
            bytes_limit,
            bytes_limit_is_set,
            bytes_reserved,
            bytes_reserved_is_set,
            peak_bytes_reserved,
            peak_bytes_reserved_is_set,
            bytes_reservable_limit,
            bytes_reservable_limit_is_set,
            largest_free_block_bytes,
            largest_free_block_bytes_is_set,
            pool_bytes,
            pool_bytes_is_set,
            peak_pool_bytes,
            peak_pool_bytes_is_set,
            peak_allocated_bytes,
            peak_allocated_bytes_is_set,
        )
    end
end

csizeof(::Type{PJRT_Device_MemoryStats_Args}) = Csize_t(201)

# typedef PJRT_Error * PJRT_Device_MemoryStats ( PJRT_Device_MemoryStats_Args * args )
const PJRT_Device_MemoryStats = Cvoid

struct PJRT_Device_ClearMemoryStats_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    function PJRT_Device_ClearMemoryStats_Args(extension_start, device)
        return new(csizeof(PJRT_Device_ClearMemoryStats_Args), extension_start, device)
    end
end

csizeof(::Type{PJRT_Device_ClearMemoryStats_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Device_ClearMemoryStats ( PJRT_Device_ClearMemoryStats_Args * args )
const PJRT_Device_ClearMemoryStats = Cvoid

struct PJRT_Device_PoisonExecution_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    launch_id::Int32
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
    poisoned::Bool
    payload::Ptr{PJRT_NamedValue}
    num_payload::Csize_t
    function PJRT_Device_PoisonExecution_Args(
        extension_start,
        device,
        launch_id,
        error_code,
        error_message,
        error_message_size,
        poisoned,
        payload,
        num_payload,
    )
        return new(
            csizeof(PJRT_Device_PoisonExecution_Args),
            extension_start,
            device,
            launch_id,
            error_code,
            error_message,
            error_message_size,
            poisoned,
            payload,
            num_payload,
        )
    end
end

csizeof(::Type{PJRT_Device_PoisonExecution_Args}) = Csize_t(72)

# typedef PJRT_Error * PJRT_Device_PoisonExecution ( PJRT_Device_PoisonExecution_Args * args )
const PJRT_Device_PoisonExecution = Cvoid

mutable struct PJRT_Device_Attributes end

struct PJRT_Device_GetAttributes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    attributes::Ptr{PJRT_NamedValue}
    num_attributes::Csize_t
    device_attributes::Ptr{PJRT_Device_Attributes}
    attributes_deleter::Ptr{Cvoid}
    function PJRT_Device_GetAttributes_Args(
        extension_start,
        device,
        attributes,
        num_attributes,
        device_attributes,
        attributes_deleter,
    )
        return new(
            csizeof(PJRT_Device_GetAttributes_Args),
            extension_start,
            device,
            attributes,
            num_attributes,
            device_attributes,
            attributes_deleter,
        )
    end
end

csizeof(::Type{PJRT_Device_GetAttributes_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_Device_GetAttributes ( PJRT_Device_GetAttributes_Args * args )
const PJRT_Device_GetAttributes = Cvoid

mutable struct PJRT_AsyncTrackingEvent end

struct PJRT_Device_CreateAsyncTrackingEvent_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    description::Cstring
    description_size::Csize_t
    event::Ptr{PJRT_AsyncTrackingEvent}
    function PJRT_Device_CreateAsyncTrackingEvent_Args(
        extension_start, device, description, description_size, event
    )
        return new(
            csizeof(PJRT_Device_CreateAsyncTrackingEvent_Args),
            extension_start,
            device,
            description,
            description_size,
            event,
        )
    end
end

csizeof(::Type{PJRT_Device_CreateAsyncTrackingEvent_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Device_CreateAsyncTrackingEvent ( PJRT_Device_CreateAsyncTrackingEvent_Args * args )
const PJRT_Device_CreateAsyncTrackingEvent = Cvoid

struct PJRT_AsyncTrackingEvent_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_AsyncTrackingEvent}
    function PJRT_AsyncTrackingEvent_Destroy_Args(extension_start, event)
        return new(csizeof(PJRT_AsyncTrackingEvent_Destroy_Args), extension_start, event)
    end
end

csizeof(::Type{PJRT_AsyncTrackingEvent_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_AsyncTrackingEvent_Destroy ( PJRT_AsyncTrackingEvent_Destroy_Args * args )
const PJRT_AsyncTrackingEvent_Destroy = Cvoid

struct PJRT_Memory_Id_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    id::Cint
    function PJRT_Memory_Id_Args(extension_start, memory, id)
        return new(csizeof(PJRT_Memory_Id_Args), extension_start, memory, id)
    end
end

csizeof(::Type{PJRT_Memory_Id_Args}) = Csize_t(28)

# typedef PJRT_Error * PJRT_Memory_Id ( PJRT_Memory_Id_Args * args )
const PJRT_Memory_Id = Cvoid

struct PJRT_Memory_Kind_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    kind::Cstring
    kind_size::Csize_t
    function PJRT_Memory_Kind_Args(extension_start, memory, kind, kind_size)
        return new(csizeof(PJRT_Memory_Kind_Args), extension_start, memory, kind, kind_size)
    end
end

csizeof(::Type{PJRT_Memory_Kind_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Memory_Kind ( PJRT_Memory_Kind_Args * args )
const PJRT_Memory_Kind = Cvoid

struct PJRT_Memory_Kind_Id_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    kind_id::Cint
    function PJRT_Memory_Kind_Id_Args(extension_start, memory, kind_id)
        return new(csizeof(PJRT_Memory_Kind_Id_Args), extension_start, memory, kind_id)
    end
end

csizeof(::Type{PJRT_Memory_Kind_Id_Args}) = Csize_t(28)

# typedef PJRT_Error * PJRT_Memory_Kind_Id ( PJRT_Memory_Kind_Id_Args * args )
const PJRT_Memory_Kind_Id = Cvoid

struct PJRT_Memory_DebugString_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    debug_string::Cstring
    debug_string_size::Csize_t
    function PJRT_Memory_DebugString_Args(
        extension_start, memory, debug_string, debug_string_size
    )
        return new(
            csizeof(PJRT_Memory_DebugString_Args),
            extension_start,
            memory,
            debug_string,
            debug_string_size,
        )
    end
end

csizeof(::Type{PJRT_Memory_DebugString_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Memory_DebugString ( PJRT_Memory_DebugString_Args * args )
const PJRT_Memory_DebugString = Cvoid

struct PJRT_Memory_ToString_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    to_string::Cstring
    to_string_size::Csize_t
    function PJRT_Memory_ToString_Args(extension_start, memory, to_string, to_string_size)
        return new(
            csizeof(PJRT_Memory_ToString_Args),
            extension_start,
            memory,
            to_string,
            to_string_size,
        )
    end
end

csizeof(::Type{PJRT_Memory_ToString_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Memory_ToString ( PJRT_Memory_ToString_Args * args )
const PJRT_Memory_ToString = Cvoid

struct PJRT_Memory_AddressableByDevices_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    devices::Ptr{Ptr{PJRT_Device}}
    num_devices::Csize_t
    function PJRT_Memory_AddressableByDevices_Args(
        extension_start, memory, devices, num_devices
    )
        return new(
            csizeof(PJRT_Memory_AddressableByDevices_Args),
            extension_start,
            memory,
            devices,
            num_devices,
        )
    end
end

csizeof(::Type{PJRT_Memory_AddressableByDevices_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Memory_AddressableByDevices ( PJRT_Memory_AddressableByDevices_Args * args )
const PJRT_Memory_AddressableByDevices = Cvoid

mutable struct PJRT_ExecuteContext end

struct PJRT_ExecuteContext_Create_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    context::Ptr{PJRT_ExecuteContext}
    function PJRT_ExecuteContext_Create_Args(extension_start, context)
        return new(csizeof(PJRT_ExecuteContext_Create_Args), extension_start, context)
    end
end

csizeof(::Type{PJRT_ExecuteContext_Create_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_ExecuteContext_Create ( PJRT_ExecuteContext_Create_Args * args )
const PJRT_ExecuteContext_Create = Cvoid

struct PJRT_ExecuteContext_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    context::Ptr{PJRT_ExecuteContext}
    function PJRT_ExecuteContext_Destroy_Args(extension_start, context)
        return new(csizeof(PJRT_ExecuteContext_Destroy_Args), extension_start, context)
    end
end

csizeof(::Type{PJRT_ExecuteContext_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_ExecuteContext_Destroy ( PJRT_ExecuteContext_Destroy_Args * args )
const PJRT_ExecuteContext_Destroy = Cvoid

struct PJRT_Executable_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    function PJRT_Executable_Destroy_Args(extension_start, executable)
        return new(csizeof(PJRT_Executable_Destroy_Args), extension_start, executable)
    end
end

csizeof(::Type{PJRT_Executable_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Executable_Destroy ( PJRT_Executable_Destroy_Args * args )
const PJRT_Executable_Destroy = Cvoid

struct PJRT_LoadedExecutable_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    function PJRT_LoadedExecutable_Destroy_Args(extension_start, executable)
        return new(csizeof(PJRT_LoadedExecutable_Destroy_Args), extension_start, executable)
    end
end

csizeof(::Type{PJRT_LoadedExecutable_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_LoadedExecutable_Destroy ( PJRT_LoadedExecutable_Destroy_Args * args )
const PJRT_LoadedExecutable_Destroy = Cvoid

struct PJRT_LoadedExecutable_GetExecutable_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    loaded_executable::Ptr{PJRT_LoadedExecutable}
    executable::Ptr{PJRT_Executable}
    function PJRT_LoadedExecutable_GetExecutable_Args(
        extension_start, loaded_executable, executable
    )
        return new(
            csizeof(PJRT_LoadedExecutable_GetExecutable_Args),
            extension_start,
            loaded_executable,
            executable,
        )
    end
end

csizeof(::Type{PJRT_LoadedExecutable_GetExecutable_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_LoadedExecutable_GetExecutable ( PJRT_LoadedExecutable_GetExecutable_Args * args )
const PJRT_LoadedExecutable_GetExecutable = Cvoid

mutable struct PJRT_DeviceAssignmentSerialized end

struct PJRT_LoadedExecutable_GetDeviceAssignment_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    serialized_bytes::Cstring
    serialized_bytes_size::Csize_t
    serialized_device_assignment::Ptr{PJRT_DeviceAssignmentSerialized}
    serialized_device_assignment_deleter::Ptr{Cvoid}
    function PJRT_LoadedExecutable_GetDeviceAssignment_Args(
        extension_start,
        executable,
        serialized_bytes,
        serialized_bytes_size,
        serialized_device_assignment,
        serialized_device_assignment_deleter,
    )
        return new(
            csizeof(PJRT_LoadedExecutable_GetDeviceAssignment_Args),
            extension_start,
            executable,
            serialized_bytes,
            serialized_bytes_size,
            serialized_device_assignment,
            serialized_device_assignment_deleter,
        )
    end
end

csizeof(::Type{PJRT_LoadedExecutable_GetDeviceAssignment_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_LoadedExecutable_GetDeviceAssignment ( PJRT_LoadedExecutable_GetDeviceAssignment_Args * args )
const PJRT_LoadedExecutable_GetDeviceAssignment = Cvoid

struct PJRT_Executable_Name_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    executable_name::Cstring
    executable_name_size::Csize_t
    function PJRT_Executable_Name_Args(
        extension_start, executable, executable_name, executable_name_size
    )
        return new(
            csizeof(PJRT_Executable_Name_Args),
            extension_start,
            executable,
            executable_name,
            executable_name_size,
        )
    end
end

csizeof(::Type{PJRT_Executable_Name_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Executable_Name ( PJRT_Executable_Name_Args * args )
const PJRT_Executable_Name = Cvoid

struct PJRT_Executable_NumReplicas_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_replicas::Csize_t
    function PJRT_Executable_NumReplicas_Args(extension_start, executable, num_replicas)
        return new(
            csizeof(PJRT_Executable_NumReplicas_Args),
            extension_start,
            executable,
            num_replicas,
        )
    end
end

csizeof(::Type{PJRT_Executable_NumReplicas_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Executable_NumReplicas ( PJRT_Executable_NumReplicas_Args * args )
const PJRT_Executable_NumReplicas = Cvoid

struct PJRT_Executable_NumPartitions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_partitions::Csize_t
    function PJRT_Executable_NumPartitions_Args(extension_start, executable, num_partitions)
        return new(
            csizeof(PJRT_Executable_NumPartitions_Args),
            extension_start,
            executable,
            num_partitions,
        )
    end
end

csizeof(::Type{PJRT_Executable_NumPartitions_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Executable_NumPartitions ( PJRT_Executable_NumPartitions_Args * args )
const PJRT_Executable_NumPartitions = Cvoid

struct PJRT_LogicalDeviceIds
    replica::Cint
    partition::Cint
end

struct PJRT_LoadedExecutable_AddressableDevices_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    addressable_devices::Ptr{Ptr{PJRT_Device}}
    num_addressable_devices::Csize_t
    function PJRT_LoadedExecutable_AddressableDevices_Args(
        extension_start, executable, addressable_devices, num_addressable_devices
    )
        return new(
            csizeof(PJRT_LoadedExecutable_AddressableDevices_Args),
            extension_start,
            executable,
            addressable_devices,
            num_addressable_devices,
        )
    end
end

csizeof(::Type{PJRT_LoadedExecutable_AddressableDevices_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_LoadedExecutable_AddressableDevices ( PJRT_LoadedExecutable_AddressableDevices_Args * args )
const PJRT_LoadedExecutable_AddressableDevices = Cvoid

struct PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    addressable_device_logical_ids::Ptr{PJRT_LogicalDeviceIds}
    num_addressable_device_logical_ids::Csize_t
    function PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args(
        extension_start,
        executable,
        addressable_device_logical_ids,
        num_addressable_device_logical_ids,
    )
        return new(
            csizeof(PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args),
            extension_start,
            executable,
            addressable_device_logical_ids,
            num_addressable_device_logical_ids,
        )
    end
end

csizeof(::Type{PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_LoadedExecutable_AddressableDeviceLogicalIds ( PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args * args )
const PJRT_LoadedExecutable_AddressableDeviceLogicalIds = Cvoid

struct PJRT_Executable_OptimizedProgram_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    program::Ptr{PJRT_Program}
    function PJRT_Executable_OptimizedProgram_Args(extension_start, executable, program)
        return new(
            csizeof(PJRT_Executable_OptimizedProgram_Args),
            extension_start,
            executable,
            program,
        )
    end
end

csizeof(::Type{PJRT_Executable_OptimizedProgram_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Executable_OptimizedProgram ( PJRT_Executable_OptimizedProgram_Args * args )
const PJRT_Executable_OptimizedProgram = Cvoid

struct PJRT_LoadedExecutable_Delete_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    function PJRT_LoadedExecutable_Delete_Args(extension_start, executable)
        return new(csizeof(PJRT_LoadedExecutable_Delete_Args), extension_start, executable)
    end
end

csizeof(::Type{PJRT_LoadedExecutable_Delete_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_LoadedExecutable_Delete ( PJRT_LoadedExecutable_Delete_Args * args )
const PJRT_LoadedExecutable_Delete = Cvoid

struct PJRT_LoadedExecutable_IsDeleted_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    is_deleted::Bool
    function PJRT_LoadedExecutable_IsDeleted_Args(extension_start, executable, is_deleted)
        return new(
            csizeof(PJRT_LoadedExecutable_IsDeleted_Args),
            extension_start,
            executable,
            is_deleted,
        )
    end
end

csizeof(::Type{PJRT_LoadedExecutable_IsDeleted_Args}) = Csize_t(25)

# typedef PJRT_Error * PJRT_LoadedExecutable_IsDeleted ( PJRT_LoadedExecutable_IsDeleted_Args * args )
const PJRT_LoadedExecutable_IsDeleted = Cvoid

struct PJRT_Chunk
    data::Ptr{Cvoid}
    size::Csize_t
    deleter::Ptr{Cvoid}
    deleter_arg::Ptr{Cvoid}
end

mutable struct PJRT_CopyToDeviceStream end

# typedef PJRT_Error * ( * PJRT_SendCallback ) ( PJRT_Chunk * chunk , PJRT_CallbackError * callback_error , size_t total_size_in_bytes , bool done , void * user_arg )
const PJRT_SendCallback = Ptr{Cvoid}

# typedef void ( * PJRT_RecvCallback ) ( PJRT_CopyToDeviceStream * stream , void * user_arg )
const PJRT_RecvCallback = Ptr{Cvoid}

struct PJRT_SendCallbackInfo
    channel_id::Int64
    user_arg::Ptr{Cvoid}
    send_callback::PJRT_SendCallback
end

csizeof(::Type{PJRT_SendCallbackInfo}) = Csize_t(24)

struct PJRT_RecvCallbackInfo
    channel_id::Int64
    user_arg::Ptr{Cvoid}
    recv_callback::PJRT_RecvCallback
end

csizeof(::Type{PJRT_RecvCallbackInfo}) = Csize_t(24)

# typedef void ( * PJRT_HloOutputCallback ) ( int64_t replica_id , int64_t partition_id , const void * data , const int64_t * shape_dims , size_t shape_num_dims , PJRT_Buffer_Type shape_element_type , int64_t operand_index , void * user_arg )
const PJRT_HloOutputCallback = Ptr{Cvoid}

struct PJRT_HloOutputCallbackInfo
    user_arg::Ptr{Cvoid}
    callback::PJRT_HloOutputCallback
    callback_id::Int64
    num_operands::Csize_t
end

csizeof(::Type{PJRT_HloOutputCallbackInfo}) = Csize_t(32)

mutable struct PJRT_MultiSlice_Config end

struct PJRT_ExecuteOptions
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    send_callbacks::Ptr{Ptr{PJRT_SendCallbackInfo}}
    recv_callbacks::Ptr{Ptr{PJRT_RecvCallbackInfo}}
    num_send_ops::Csize_t
    num_recv_ops::Csize_t
    launch_id::Cint
    non_donatable_input_indices::Ptr{Int64}
    num_non_donatable_input_indices::Csize_t
    context::Ptr{PJRT_ExecuteContext}
    call_location::Cstring
    num_tasks::Csize_t
    task_ids::Ptr{Cint}
    incarnation_ids::Ptr{Int64}
    multi_slice_config::Ptr{PJRT_MultiSlice_Config}
    use_major_to_minor_data_layout_for_callbacks::Bool
    hlo_output_callbacks::Ptr{PJRT_HloOutputCallbackInfo}
    num_hlo_output_callbacks::Csize_t
    custom_options::Ptr{PJRT_NamedValue}
    num_custom_options::Csize_t
end

csizeof(::Type{PJRT_ExecuteOptions}) = Csize_t(160)

struct PJRT_LoadedExecutable_Execute_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    options::Ptr{PJRT_ExecuteOptions}
    argument_lists::Ptr{Ptr{Ptr{PJRT_Buffer}}}
    num_devices::Csize_t
    num_args::Csize_t
    output_lists::Ptr{Ptr{Ptr{PJRT_Buffer}}}
    device_complete_events::Ptr{Ptr{PJRT_Event}}
    execute_device::Ptr{PJRT_Device}
    function PJRT_LoadedExecutable_Execute_Args(
        extension_start,
        executable,
        options,
        argument_lists,
        num_devices,
        num_args,
        output_lists,
        device_complete_events,
        execute_device,
    )
        return new(
            csizeof(PJRT_LoadedExecutable_Execute_Args),
            extension_start,
            executable,
            options,
            argument_lists,
            num_devices,
            num_args,
            output_lists,
            device_complete_events,
            execute_device,
        )
    end
end

csizeof(::Type{PJRT_LoadedExecutable_Execute_Args}) = Csize_t(80)

# typedef PJRT_Error * PJRT_LoadedExecutable_Execute ( PJRT_LoadedExecutable_Execute_Args * args )
const PJRT_LoadedExecutable_Execute = Cvoid

struct PJRT_Executable_NumOutputs_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_outputs::Csize_t
    function PJRT_Executable_NumOutputs_Args(extension_start, executable, num_outputs)
        return new(
            csizeof(PJRT_Executable_NumOutputs_Args),
            extension_start,
            executable,
            num_outputs,
        )
    end
end

csizeof(::Type{PJRT_Executable_NumOutputs_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Executable_NumOutputs ( PJRT_Executable_NumOutputs_Args * args )
const PJRT_Executable_NumOutputs = Cvoid

struct PJRT_Executable_SizeOfGeneratedCodeInBytes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    size_in_bytes::Int64
    function PJRT_Executable_SizeOfGeneratedCodeInBytes_Args(
        extension_start, executable, size_in_bytes
    )
        return new(
            csizeof(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args),
            extension_start,
            executable,
            size_in_bytes,
        )
    end
end

csizeof(::Type{PJRT_Executable_SizeOfGeneratedCodeInBytes_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Executable_SizeOfGeneratedCodeInBytes ( PJRT_Executable_SizeOfGeneratedCodeInBytes_Args * args )
const PJRT_Executable_SizeOfGeneratedCodeInBytes = Cvoid

struct PJRT_Executable_Fingerprint_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    executable_fingerprint::Cstring
    executable_fingerprint_size::Csize_t
    function PJRT_Executable_Fingerprint_Args(
        extension_start, executable, executable_fingerprint, executable_fingerprint_size
    )
        return new(
            csizeof(PJRT_Executable_Fingerprint_Args),
            extension_start,
            executable,
            executable_fingerprint,
            executable_fingerprint_size,
        )
    end
end

csizeof(::Type{PJRT_Executable_Fingerprint_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Executable_Fingerprint ( PJRT_Executable_Fingerprint_Args * args )
const PJRT_Executable_Fingerprint = Cvoid

struct PJRT_Executable_GetCostAnalysis_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_properties::Csize_t
    properties::Ptr{PJRT_NamedValue}
    function PJRT_Executable_GetCostAnalysis_Args(
        extension_start, executable, num_properties, properties
    )
        return new(
            csizeof(PJRT_Executable_GetCostAnalysis_Args),
            extension_start,
            executable,
            num_properties,
            properties,
        )
    end
end

csizeof(::Type{PJRT_Executable_GetCostAnalysis_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Executable_GetCostAnalysis ( PJRT_Executable_GetCostAnalysis_Args * args )
const PJRT_Executable_GetCostAnalysis = Cvoid

struct PJRT_Executable_GetCompiledMemoryStats_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    generated_code_size_in_bytes::Int64
    argument_size_in_bytes::Int64
    output_size_in_bytes::Int64
    alias_size_in_bytes::Int64
    temp_size_in_bytes::Int64
    host_generated_code_size_in_bytes::Int64
    host_argument_size_in_bytes::Int64
    host_output_size_in_bytes::Int64
    host_alias_size_in_bytes::Int64
    host_temp_size_in_bytes::Int64
    peak_memory_in_bytes::Int64
    total_size_in_bytes::Int64
    total_allocation_bytes::Int64
    indefinite_allocations::Int64
    peak_unpadded_heap_bytes::Int64
    function PJRT_Executable_GetCompiledMemoryStats_Args(
        extension_start,
        executable,
        generated_code_size_in_bytes,
        argument_size_in_bytes,
        output_size_in_bytes,
        alias_size_in_bytes,
        temp_size_in_bytes,
        host_generated_code_size_in_bytes,
        host_argument_size_in_bytes,
        host_output_size_in_bytes,
        host_alias_size_in_bytes,
        host_temp_size_in_bytes,
        peak_memory_in_bytes,
        total_size_in_bytes,
        total_allocation_bytes,
        indefinite_allocations,
        peak_unpadded_heap_bytes,
    )
        return new(
            csizeof(PJRT_Executable_GetCompiledMemoryStats_Args),
            extension_start,
            executable,
            generated_code_size_in_bytes,
            argument_size_in_bytes,
            output_size_in_bytes,
            alias_size_in_bytes,
            temp_size_in_bytes,
            host_generated_code_size_in_bytes,
            host_argument_size_in_bytes,
            host_output_size_in_bytes,
            host_alias_size_in_bytes,
            host_temp_size_in_bytes,
            peak_memory_in_bytes,
            total_size_in_bytes,
            total_allocation_bytes,
            indefinite_allocations,
            peak_unpadded_heap_bytes,
        )
    end
end

csizeof(::Type{PJRT_Executable_GetCompiledMemoryStats_Args}) = Csize_t(144)

# typedef PJRT_Error * PJRT_Executable_GetCompiledMemoryStats ( PJRT_Executable_GetCompiledMemoryStats_Args * args )
const PJRT_Executable_GetCompiledMemoryStats = Cvoid

struct PJRT_Executable_OutputElementTypes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    output_types::Ptr{PJRT_Buffer_Type}
    num_output_types::Csize_t
    function PJRT_Executable_OutputElementTypes_Args(
        extension_start, executable, output_types, num_output_types
    )
        return new(
            csizeof(PJRT_Executable_OutputElementTypes_Args),
            extension_start,
            executable,
            output_types,
            num_output_types,
        )
    end
end

csizeof(::Type{PJRT_Executable_OutputElementTypes_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Executable_OutputElementTypes ( PJRT_Executable_OutputElementTypes_Args * args )
const PJRT_Executable_OutputElementTypes = Cvoid

struct PJRT_Executable_OutputDimensions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_outputs::Csize_t
    dims::Ptr{Int64}
    dim_sizes::Ptr{Csize_t}
    function PJRT_Executable_OutputDimensions_Args(
        extension_start, executable, num_outputs, dims, dim_sizes
    )
        return new(
            csizeof(PJRT_Executable_OutputDimensions_Args),
            extension_start,
            executable,
            num_outputs,
            dims,
            dim_sizes,
        )
    end
end

csizeof(::Type{PJRT_Executable_OutputDimensions_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Executable_OutputDimensions ( PJRT_Executable_OutputDimensions_Args * args )
const PJRT_Executable_OutputDimensions = Cvoid

struct PJRT_Executable_ParameterMemoryKinds_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_parameters::Csize_t
    memory_kinds::Ptr{Cstring}
    memory_kind_sizes::Ptr{Csize_t}
    function PJRT_Executable_ParameterMemoryKinds_Args(
        extension_start, executable, num_parameters, memory_kinds, memory_kind_sizes
    )
        return new(
            csizeof(PJRT_Executable_ParameterMemoryKinds_Args),
            extension_start,
            executable,
            num_parameters,
            memory_kinds,
            memory_kind_sizes,
        )
    end
end

csizeof(::Type{PJRT_Executable_ParameterMemoryKinds_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Executable_ParameterMemoryKinds ( PJRT_Executable_ParameterMemoryKinds_Args * args )
const PJRT_Executable_ParameterMemoryKinds = Cvoid

struct PJRT_Executable_OutputMemoryKinds_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_outputs::Csize_t
    memory_kinds::Ptr{Cstring}
    memory_kind_sizes::Ptr{Csize_t}
    function PJRT_Executable_OutputMemoryKinds_Args(
        extension_start, executable, num_outputs, memory_kinds, memory_kind_sizes
    )
        return new(
            csizeof(PJRT_Executable_OutputMemoryKinds_Args),
            extension_start,
            executable,
            num_outputs,
            memory_kinds,
            memory_kind_sizes,
        )
    end
end

csizeof(::Type{PJRT_Executable_OutputMemoryKinds_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Executable_OutputMemoryKinds ( PJRT_Executable_OutputMemoryKinds_Args * args )
const PJRT_Executable_OutputMemoryKinds = Cvoid

mutable struct PJRT_SerializedExecutable end

mutable struct PJRT_SerializedCompileOptions end

struct PJRT_Executable_Serialize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    serialized_bytes::Cstring
    serialized_bytes_size::Csize_t
    serialized_executable::Ptr{PJRT_SerializedExecutable}
    serialized_executable_deleter::Ptr{Cvoid}
    function PJRT_Executable_Serialize_Args(
        extension_start,
        executable,
        serialized_bytes,
        serialized_bytes_size,
        serialized_executable,
        serialized_executable_deleter,
    )
        return new(
            csizeof(PJRT_Executable_Serialize_Args),
            extension_start,
            executable,
            serialized_bytes,
            serialized_bytes_size,
            serialized_executable,
            serialized_executable_deleter,
        )
    end
end

csizeof(::Type{PJRT_Executable_Serialize_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_Executable_Serialize ( PJRT_Executable_Serialize_Args * args )
const PJRT_Executable_Serialize = Cvoid

struct PJRT_Executable_GetCompileOptions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    serialized_bytes::Cstring
    serialized_bytes_size::Csize_t
    serialized_compile_options::Ptr{PJRT_SerializedCompileOptions}
    serialized_compile_options_deleter::Ptr{Cvoid}
    function PJRT_Executable_GetCompileOptions_Args(
        extension_start,
        executable,
        serialized_bytes,
        serialized_bytes_size,
        serialized_compile_options,
        serialized_compile_options_deleter,
    )
        return new(
            csizeof(PJRT_Executable_GetCompileOptions_Args),
            extension_start,
            executable,
            serialized_bytes,
            serialized_bytes_size,
            serialized_compile_options,
            serialized_compile_options_deleter,
        )
    end
end

csizeof(::Type{PJRT_Executable_GetCompileOptions_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_Executable_GetCompileOptions ( PJRT_Executable_GetCompileOptions_Args * args )
const PJRT_Executable_GetCompileOptions = Cvoid

struct PJRT_LoadOptions
    struct_size::Csize_t
    computation_origin::Ptr{Int32}
    computation_origin_size::Csize_t
    multi_slice_config::Ptr{PJRT_MultiSlice_Config}
end

csizeof(::Type{PJRT_LoadOptions}) = Csize_t(32)

struct PJRT_Executable_DeserializeAndLoad_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    serialized_executable::Cstring
    serialized_executable_size::Csize_t
    loaded_executable::Ptr{PJRT_LoadedExecutable}
    overridden_serialized_compile_options::Cstring
    overridden_serialized_compile_options_size::Csize_t
    load_options::Ptr{PJRT_LoadOptions}
    function PJRT_Executable_DeserializeAndLoad_Args(
        extension_start,
        client,
        serialized_executable,
        serialized_executable_size,
        loaded_executable,
        overridden_serialized_compile_options,
        overridden_serialized_compile_options_size,
        load_options,
    )
        return new(
            csizeof(PJRT_Executable_DeserializeAndLoad_Args),
            extension_start,
            client,
            serialized_executable,
            serialized_executable_size,
            loaded_executable,
            overridden_serialized_compile_options,
            overridden_serialized_compile_options_size,
            load_options,
        )
    end
end

csizeof(::Type{PJRT_Executable_DeserializeAndLoad_Args}) = Csize_t(72)

# typedef PJRT_Error * PJRT_Executable_DeserializeAndLoad ( PJRT_Executable_DeserializeAndLoad_Args * args )
const PJRT_Executable_DeserializeAndLoad = Cvoid

struct PJRT_LoadedExecutable_Fingerprint_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    executable_fingerprint::Cstring
    executable_fingerprint_size::Csize_t
    function PJRT_LoadedExecutable_Fingerprint_Args(
        extension_start, executable, executable_fingerprint, executable_fingerprint_size
    )
        return new(
            csizeof(PJRT_LoadedExecutable_Fingerprint_Args),
            extension_start,
            executable,
            executable_fingerprint,
            executable_fingerprint_size,
        )
    end
end

csizeof(::Type{PJRT_LoadedExecutable_Fingerprint_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_LoadedExecutable_Fingerprint ( PJRT_LoadedExecutable_Fingerprint_Args * args )
const PJRT_LoadedExecutable_Fingerprint = Cvoid

struct PJRT_Buffer_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    function PJRT_Buffer_Destroy_Args(extension_start, buffer)
        return new(csizeof(PJRT_Buffer_Destroy_Args), extension_start, buffer)
    end
end

csizeof(::Type{PJRT_Buffer_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Buffer_Destroy ( PJRT_Buffer_Destroy_Args * args )
const PJRT_Buffer_Destroy = Cvoid

struct PJRT_Buffer_ElementType_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    type::PJRT_Buffer_Type
    function PJRT_Buffer_ElementType_Args(extension_start, buffer, type)
        return new(csizeof(PJRT_Buffer_ElementType_Args), extension_start, buffer, type)
    end
end

csizeof(::Type{PJRT_Buffer_ElementType_Args}) = Csize_t(28)

# typedef PJRT_Error * PJRT_Buffer_ElementType ( PJRT_Buffer_ElementType_Args * args )
const PJRT_Buffer_ElementType = Cvoid

struct PJRT_Buffer_Dimensions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    dims::Ptr{Int64}
    num_dims::Csize_t
    function PJRT_Buffer_Dimensions_Args(extension_start, buffer, dims, num_dims)
        return new(
            csizeof(PJRT_Buffer_Dimensions_Args), extension_start, buffer, dims, num_dims
        )
    end
end

csizeof(::Type{PJRT_Buffer_Dimensions_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Buffer_Dimensions ( PJRT_Buffer_Dimensions_Args * args )
const PJRT_Buffer_Dimensions = Cvoid

struct PJRT_Buffer_UnpaddedDimensions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    unpadded_dims::Ptr{Int64}
    num_dims::Csize_t
    function PJRT_Buffer_UnpaddedDimensions_Args(
        extension_start, buffer, unpadded_dims, num_dims
    )
        return new(
            csizeof(PJRT_Buffer_UnpaddedDimensions_Args),
            extension_start,
            buffer,
            unpadded_dims,
            num_dims,
        )
    end
end

csizeof(::Type{PJRT_Buffer_UnpaddedDimensions_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Buffer_UnpaddedDimensions ( PJRT_Buffer_UnpaddedDimensions_Args * args )
const PJRT_Buffer_UnpaddedDimensions = Cvoid

struct PJRT_Buffer_DynamicDimensionIndices_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    dynamic_dim_indices::Ptr{Csize_t}
    num_dynamic_dims::Csize_t
    function PJRT_Buffer_DynamicDimensionIndices_Args(
        extension_start, buffer, dynamic_dim_indices, num_dynamic_dims
    )
        return new(
            csizeof(PJRT_Buffer_DynamicDimensionIndices_Args),
            extension_start,
            buffer,
            dynamic_dim_indices,
            num_dynamic_dims,
        )
    end
end

csizeof(::Type{PJRT_Buffer_DynamicDimensionIndices_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Buffer_DynamicDimensionIndices ( PJRT_Buffer_DynamicDimensionIndices_Args * args )
const PJRT_Buffer_DynamicDimensionIndices = Cvoid

struct PJRT_Buffer_GetMemoryLayout_Args
    data::NTuple{104,UInt8}
end

function Base.getproperty(x::Ptr{PJRT_Buffer_GetMemoryLayout_Args}, f::Symbol)
    f === :struct_size && return Ptr{Csize_t}(x + 0)
    f === :extension_start && return Ptr{Ptr{PJRT_Extension_Base}}(x + 8)
    f === :buffer && return Ptr{Ptr{PJRT_Buffer}}(x + 16)
    f === :layout && return Ptr{PJRT_Buffer_MemoryLayout}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::PJRT_Buffer_GetMemoryLayout_Args, f::Symbol)
    r = Ref{PJRT_Buffer_GetMemoryLayout_Args}(x)
    ptr = Base.unsafe_convert(Ptr{PJRT_Buffer_GetMemoryLayout_Args}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{PJRT_Buffer_GetMemoryLayout_Args}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::PJRT_Buffer_GetMemoryLayout_Args, private::Bool=false)
    return (
        :struct_size, :extension_start, :buffer, :layout, if private
            fieldnames(typeof(x))
        else
            ()
        end...
    )
end

csizeof(::Type{PJRT_Buffer_GetMemoryLayout_Args}) = Csize_t(104)

# typedef PJRT_Error * PJRT_Buffer_GetMemoryLayout ( PJRT_Buffer_GetMemoryLayout_Args * args )
const PJRT_Buffer_GetMemoryLayout = Cvoid

struct PJRT_Buffer_ToHostBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    src::Ptr{PJRT_Buffer}
    host_layout::Ptr{PJRT_Buffer_MemoryLayout}
    dst::Ptr{Cvoid}
    dst_size::Csize_t
    event::Ptr{PJRT_Event}
    function PJRT_Buffer_ToHostBuffer_Args(
        extension_start, src, host_layout, dst, dst_size, event
    )
        return new(
            csizeof(PJRT_Buffer_ToHostBuffer_Args),
            extension_start,
            src,
            host_layout,
            dst,
            dst_size,
            event,
        )
    end
end

csizeof(::Type{PJRT_Buffer_ToHostBuffer_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_Buffer_ToHostBuffer ( PJRT_Buffer_ToHostBuffer_Args * args )
const PJRT_Buffer_ToHostBuffer = Cvoid

struct PJRT_Buffer_OnDeviceSizeInBytes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    on_device_size_in_bytes::Csize_t
    function PJRT_Buffer_OnDeviceSizeInBytes_Args(
        extension_start, buffer, on_device_size_in_bytes
    )
        return new(
            csizeof(PJRT_Buffer_OnDeviceSizeInBytes_Args),
            extension_start,
            buffer,
            on_device_size_in_bytes,
        )
    end
end

csizeof(::Type{PJRT_Buffer_OnDeviceSizeInBytes_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Buffer_OnDeviceSizeInBytes ( PJRT_Buffer_OnDeviceSizeInBytes_Args * args )
const PJRT_Buffer_OnDeviceSizeInBytes = Cvoid

struct PJRT_Buffer_Delete_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    function PJRT_Buffer_Delete_Args(extension_start, buffer)
        return new(csizeof(PJRT_Buffer_Delete_Args), extension_start, buffer)
    end
end

csizeof(::Type{PJRT_Buffer_Delete_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Buffer_Delete ( PJRT_Buffer_Delete_Args * args )
const PJRT_Buffer_Delete = Cvoid

struct PJRT_Buffer_IsDeleted_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    is_deleted::Bool
    function PJRT_Buffer_IsDeleted_Args(extension_start, buffer, is_deleted)
        return new(csizeof(PJRT_Buffer_IsDeleted_Args), extension_start, buffer, is_deleted)
    end
end

csizeof(::Type{PJRT_Buffer_IsDeleted_Args}) = Csize_t(25)

# typedef PJRT_Error * PJRT_Buffer_IsDeleted ( PJRT_Buffer_IsDeleted_Args * args )
const PJRT_Buffer_IsDeleted = Cvoid

struct PJRT_Buffer_CopyRawToHost_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    dst::Ptr{Cvoid}
    offset::Int64
    transfer_size::Int64
    event::Ptr{PJRT_Event}
    function PJRT_Buffer_CopyRawToHost_Args(
        extension_start, buffer, dst, offset, transfer_size, event
    )
        return new(
            csizeof(PJRT_Buffer_CopyRawToHost_Args),
            extension_start,
            buffer,
            dst,
            offset,
            transfer_size,
            event,
        )
    end
end

csizeof(::Type{PJRT_Buffer_CopyRawToHost_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_Buffer_CopyRawToHost ( PJRT_Buffer_CopyRawToHost_Args * args )
const PJRT_Buffer_CopyRawToHost = Cvoid

struct PJRT_Buffer_CopyRawToHostFuture_Callback_Args
    struct_size::Csize_t
    callback_data::Ptr{Cvoid}
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
    dst::Ptr{Cvoid}
    function PJRT_Buffer_CopyRawToHostFuture_Callback_Args(
        callback_data, error_code, error_message, error_message_size, dst
    )
        return new(
            csizeof(PJRT_Buffer_CopyRawToHostFuture_Callback_Args),
            callback_data,
            error_code,
            error_message,
            error_message_size,
            dst,
        )
    end
end

csizeof(::Type{PJRT_Buffer_CopyRawToHostFuture_Callback_Args}) = Csize_t(48)

struct PJRT_Buffer_CopyRawToHostFuture_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    offset::Int64
    transfer_size::Int64
    event::Ptr{PJRT_Event}
    callback_data::Ptr{Cvoid}
    future_ready_callback::Ptr{Cvoid}
    function PJRT_Buffer_CopyRawToHostFuture_Args(
        extension_start,
        buffer,
        offset,
        transfer_size,
        event,
        callback_data,
        future_ready_callback,
    )
        return new(
            csizeof(PJRT_Buffer_CopyRawToHostFuture_Args),
            extension_start,
            buffer,
            offset,
            transfer_size,
            event,
            callback_data,
            future_ready_callback,
        )
    end
end

csizeof(::Type{PJRT_Buffer_CopyRawToHostFuture_Args}) = Csize_t(64)

# typedef PJRT_Error * PJRT_Buffer_CopyRawToHostFuture ( PJRT_Buffer_CopyRawToHostFuture_Args * args )
const PJRT_Buffer_CopyRawToHostFuture = Cvoid

struct PJRT_Buffer_CopyToDevice_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    dst_device::Ptr{PJRT_Device}
    dst_buffer::Ptr{PJRT_Buffer}
    function PJRT_Buffer_CopyToDevice_Args(extension_start, buffer, dst_device, dst_buffer)
        return new(
            csizeof(PJRT_Buffer_CopyToDevice_Args),
            extension_start,
            buffer,
            dst_device,
            dst_buffer,
        )
    end
end

csizeof(::Type{PJRT_Buffer_CopyToDevice_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Buffer_CopyToDevice ( PJRT_Buffer_CopyToDevice_Args * args )
const PJRT_Buffer_CopyToDevice = Cvoid

struct PJRT_Buffer_CopyToMemory_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    dst_memory::Ptr{PJRT_Memory}
    dst_buffer::Ptr{PJRT_Buffer}
    function PJRT_Buffer_CopyToMemory_Args(extension_start, buffer, dst_memory, dst_buffer)
        return new(
            csizeof(PJRT_Buffer_CopyToMemory_Args),
            extension_start,
            buffer,
            dst_memory,
            dst_buffer,
        )
    end
end

csizeof(::Type{PJRT_Buffer_CopyToMemory_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Buffer_CopyToMemory ( PJRT_Buffer_CopyToMemory_Args * args )
const PJRT_Buffer_CopyToMemory = Cvoid

struct PJRT_Buffer_Bitcast_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    element_type::PJRT_Buffer_Type
    dims::Ptr{Int64}
    num_dims::Csize_t
    device_layout::Ptr{PJRT_Buffer_MemoryLayout}
    out_buffer::Ptr{PJRT_Buffer}
    function PJRT_Buffer_Bitcast_Args(
        extension_start, buffer, element_type, dims, num_dims, device_layout, out_buffer
    )
        return new(
            csizeof(PJRT_Buffer_Bitcast_Args),
            extension_start,
            buffer,
            element_type,
            dims,
            num_dims,
            device_layout,
            out_buffer,
        )
    end
end

csizeof(::Type{PJRT_Buffer_Bitcast_Args}) = Csize_t(64)

# typedef PJRT_Error * PJRT_Buffer_Bitcast ( PJRT_Buffer_Bitcast_Args * args )
const PJRT_Buffer_Bitcast = Cvoid

struct PJRT_Buffer_IsOnCpu_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    is_on_cpu::Bool
    function PJRT_Buffer_IsOnCpu_Args(extension_start, buffer, is_on_cpu)
        return new(csizeof(PJRT_Buffer_IsOnCpu_Args), extension_start, buffer, is_on_cpu)
    end
end

csizeof(::Type{PJRT_Buffer_IsOnCpu_Args}) = Csize_t(25)

# typedef PJRT_Error * PJRT_Buffer_IsOnCpu ( PJRT_Buffer_IsOnCpu_Args * args )
const PJRT_Buffer_IsOnCpu = Cvoid

struct PJRT_Buffer_Device_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    device::Ptr{PJRT_Device}
    function PJRT_Buffer_Device_Args(extension_start, buffer, device)
        return new(csizeof(PJRT_Buffer_Device_Args), extension_start, buffer, device)
    end
end

csizeof(::Type{PJRT_Buffer_Device_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Buffer_Device ( PJRT_Buffer_Device_Args * args )
const PJRT_Buffer_Device = Cvoid

struct PJRT_Buffer_Memory_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    memory::Ptr{PJRT_Memory}
    function PJRT_Buffer_Memory_Args(extension_start, buffer, memory)
        return new(csizeof(PJRT_Buffer_Memory_Args), extension_start, buffer, memory)
    end
end

csizeof(::Type{PJRT_Buffer_Memory_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Buffer_Memory ( PJRT_Buffer_Memory_Args * args )
const PJRT_Buffer_Memory = Cvoid

struct PJRT_Buffer_ReadyEvent_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    event::Ptr{PJRT_Event}
    function PJRT_Buffer_ReadyEvent_Args(extension_start, buffer, event)
        return new(csizeof(PJRT_Buffer_ReadyEvent_Args), extension_start, buffer, event)
    end
end

csizeof(::Type{PJRT_Buffer_ReadyEvent_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Buffer_ReadyEvent ( PJRT_Buffer_ReadyEvent_Args * args )
const PJRT_Buffer_ReadyEvent = Cvoid

struct PJRT_Buffer_UnsafePointer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    buffer_pointer::Csize_t
    function PJRT_Buffer_UnsafePointer_Args(extension_start, buffer, buffer_pointer)
        return new(
            csizeof(PJRT_Buffer_UnsafePointer_Args), extension_start, buffer, buffer_pointer
        )
    end
end

csizeof(::Type{PJRT_Buffer_UnsafePointer_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Buffer_UnsafePointer ( PJRT_Buffer_UnsafePointer_Args * args )
const PJRT_Buffer_UnsafePointer = Cvoid

struct PJRT_Buffer_IncreaseExternalReferenceCount_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    function PJRT_Buffer_IncreaseExternalReferenceCount_Args(extension_start, buffer)
        return new(
            csizeof(PJRT_Buffer_IncreaseExternalReferenceCount_Args),
            extension_start,
            buffer,
        )
    end
end

csizeof(::Type{PJRT_Buffer_IncreaseExternalReferenceCount_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Buffer_IncreaseExternalReferenceCount ( PJRT_Buffer_IncreaseExternalReferenceCount_Args * args )
const PJRT_Buffer_IncreaseExternalReferenceCount = Cvoid

struct PJRT_Buffer_DecreaseExternalReferenceCount_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    function PJRT_Buffer_DecreaseExternalReferenceCount_Args(extension_start, buffer)
        return new(
            csizeof(PJRT_Buffer_DecreaseExternalReferenceCount_Args),
            extension_start,
            buffer,
        )
    end
end

csizeof(::Type{PJRT_Buffer_DecreaseExternalReferenceCount_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Buffer_DecreaseExternalReferenceCount ( PJRT_Buffer_DecreaseExternalReferenceCount_Args * args )
const PJRT_Buffer_DecreaseExternalReferenceCount = Cvoid

struct PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    device_memory_ptr::Ptr{Cvoid}
    function PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args(
        extension_start, buffer, device_memory_ptr
    )
        return new(
            csizeof(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args),
            extension_start,
            buffer,
            device_memory_ptr,
        )
    end
end

csizeof(::Type{PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Buffer_OpaqueDeviceMemoryDataPointer ( PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args * args )
const PJRT_Buffer_OpaqueDeviceMemoryDataPointer = Cvoid

struct PJRT_Buffer_DonateWithControlDependency_Callback_Args
    struct_size::Csize_t
    callback_data::Ptr{Cvoid}
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
    function PJRT_Buffer_DonateWithControlDependency_Callback_Args(
        callback_data, error_code, error_message, error_message_size
    )
        return new(
            csizeof(PJRT_Buffer_DonateWithControlDependency_Callback_Args),
            callback_data,
            error_code,
            error_message,
            error_message_size,
        )
    end
end

csizeof(::Type{PJRT_Buffer_DonateWithControlDependency_Callback_Args}) = Csize_t(40)

struct PJRT_Buffer_DonateWithControlDependency_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    callback_data::Ptr{Cvoid}
    dependency_ready_callback::Ptr{Cvoid}
    out_buffer::Ptr{PJRT_Buffer}
    function PJRT_Buffer_DonateWithControlDependency_Args(
        extension_start, buffer, callback_data, dependency_ready_callback, out_buffer
    )
        return new(
            csizeof(PJRT_Buffer_DonateWithControlDependency_Args),
            extension_start,
            buffer,
            callback_data,
            dependency_ready_callback,
            out_buffer,
        )
    end
end

csizeof(::Type{PJRT_Buffer_DonateWithControlDependency_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Buffer_DonateWithControlDependency ( PJRT_Buffer_DonateWithControlDependency_Args * args )
const PJRT_Buffer_DonateWithControlDependency = Cvoid

struct PJRT_CopyToDeviceStream_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    stream::Ptr{PJRT_CopyToDeviceStream}
    function PJRT_CopyToDeviceStream_Destroy_Args(extension_start, stream)
        return new(csizeof(PJRT_CopyToDeviceStream_Destroy_Args), extension_start, stream)
    end
end

csizeof(::Type{PJRT_CopyToDeviceStream_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_CopyToDeviceStream_Destroy ( PJRT_CopyToDeviceStream_Destroy_Args * args )
const PJRT_CopyToDeviceStream_Destroy = Cvoid

struct PJRT_CopyToDeviceStream_AddChunk_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    stream::Ptr{PJRT_CopyToDeviceStream}
    chunk::Ptr{PJRT_Chunk}
    transfer_complete::Ptr{PJRT_Event}
    function PJRT_CopyToDeviceStream_AddChunk_Args(
        extension_start, stream, chunk, transfer_complete
    )
        return new(
            csizeof(PJRT_CopyToDeviceStream_AddChunk_Args),
            extension_start,
            stream,
            chunk,
            transfer_complete,
        )
    end
end

csizeof(::Type{PJRT_CopyToDeviceStream_AddChunk_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_CopyToDeviceStream_AddChunk ( PJRT_CopyToDeviceStream_AddChunk_Args * args )
const PJRT_CopyToDeviceStream_AddChunk = Cvoid

struct PJRT_CopyToDeviceStream_TotalBytes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    stream::Ptr{PJRT_CopyToDeviceStream}
    total_bytes::Int64
    function PJRT_CopyToDeviceStream_TotalBytes_Args(extension_start, stream, total_bytes)
        return new(
            csizeof(PJRT_CopyToDeviceStream_TotalBytes_Args),
            extension_start,
            stream,
            total_bytes,
        )
    end
end

csizeof(::Type{PJRT_CopyToDeviceStream_TotalBytes_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_CopyToDeviceStream_TotalBytes ( PJRT_CopyToDeviceStream_TotalBytes_Args * args )
const PJRT_CopyToDeviceStream_TotalBytes = Cvoid

struct PJRT_CopyToDeviceStream_GranuleSize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    stream::Ptr{PJRT_CopyToDeviceStream}
    granule_size_in_bytes::Int64
    function PJRT_CopyToDeviceStream_GranuleSize_Args(
        extension_start, stream, granule_size_in_bytes
    )
        return new(
            csizeof(PJRT_CopyToDeviceStream_GranuleSize_Args),
            extension_start,
            stream,
            granule_size_in_bytes,
        )
    end
end

csizeof(::Type{PJRT_CopyToDeviceStream_GranuleSize_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_CopyToDeviceStream_GranuleSize ( PJRT_CopyToDeviceStream_GranuleSize_Args * args )
const PJRT_CopyToDeviceStream_GranuleSize = Cvoid

struct PJRT_CopyToDeviceStream_CurrentBytes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    stream::Ptr{PJRT_CopyToDeviceStream}
    current_bytes::Int64
    function PJRT_CopyToDeviceStream_CurrentBytes_Args(
        extension_start, stream, current_bytes
    )
        return new(
            csizeof(PJRT_CopyToDeviceStream_CurrentBytes_Args),
            extension_start,
            stream,
            current_bytes,
        )
    end
end

csizeof(::Type{PJRT_CopyToDeviceStream_CurrentBytes_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_CopyToDeviceStream_CurrentBytes ( PJRT_CopyToDeviceStream_CurrentBytes_Args * args )
const PJRT_CopyToDeviceStream_CurrentBytes = Cvoid

struct PJRT_TopologyDescription_Create_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology_name::Cstring
    topology_name_size::Csize_t
    create_options::Ptr{PJRT_NamedValue}
    num_options::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    function PJRT_TopologyDescription_Create_Args(
        extension_start,
        topology_name,
        topology_name_size,
        create_options,
        num_options,
        topology,
    )
        return new(
            csizeof(PJRT_TopologyDescription_Create_Args),
            extension_start,
            topology_name,
            topology_name_size,
            create_options,
            num_options,
            topology,
        )
    end
end

csizeof(::Type{PJRT_TopologyDescription_Create_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_TopologyDescription_Create ( PJRT_TopologyDescription_Create_Args * args )
const PJRT_TopologyDescription_Create = Cvoid

struct PJRT_TopologyDescription_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    function PJRT_TopologyDescription_Destroy_Args(extension_start, topology)
        return new(
            csizeof(PJRT_TopologyDescription_Destroy_Args), extension_start, topology
        )
    end
end

csizeof(::Type{PJRT_TopologyDescription_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_TopologyDescription_Destroy ( PJRT_TopologyDescription_Destroy_Args * args )
const PJRT_TopologyDescription_Destroy = Cvoid

struct PJRT_TopologyDescription_PlatformVersion_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    platform_version::Cstring
    platform_version_size::Csize_t
    function PJRT_TopologyDescription_PlatformVersion_Args(
        extension_start, topology, platform_version, platform_version_size
    )
        return new(
            csizeof(PJRT_TopologyDescription_PlatformVersion_Args),
            extension_start,
            topology,
            platform_version,
            platform_version_size,
        )
    end
end

csizeof(::Type{PJRT_TopologyDescription_PlatformVersion_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TopologyDescription_PlatformVersion ( PJRT_TopologyDescription_PlatformVersion_Args * args )
const PJRT_TopologyDescription_PlatformVersion = Cvoid

struct PJRT_TopologyDescription_PlatformName_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    platform_name::Cstring
    platform_name_size::Csize_t
    function PJRT_TopologyDescription_PlatformName_Args(
        extension_start, topology, platform_name, platform_name_size
    )
        return new(
            csizeof(PJRT_TopologyDescription_PlatformName_Args),
            extension_start,
            topology,
            platform_name,
            platform_name_size,
        )
    end
end

csizeof(::Type{PJRT_TopologyDescription_PlatformName_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TopologyDescription_PlatformName ( PJRT_TopologyDescription_PlatformName_Args * args )
const PJRT_TopologyDescription_PlatformName = Cvoid

struct PJRT_TopologyDescription_GetDeviceDescriptions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    descriptions::Ptr{Ptr{PJRT_DeviceDescription}}
    num_descriptions::Csize_t
    function PJRT_TopologyDescription_GetDeviceDescriptions_Args(
        extension_start, topology, descriptions, num_descriptions
    )
        return new(
            csizeof(PJRT_TopologyDescription_GetDeviceDescriptions_Args),
            extension_start,
            topology,
            descriptions,
            num_descriptions,
        )
    end
end

csizeof(::Type{PJRT_TopologyDescription_GetDeviceDescriptions_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TopologyDescription_GetDeviceDescriptions ( PJRT_TopologyDescription_GetDeviceDescriptions_Args * args )
const PJRT_TopologyDescription_GetDeviceDescriptions = Cvoid

mutable struct PJRT_SerializedTopology end

struct PJRT_TopologyDescription_Serialize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    serialized_bytes::Cstring
    serialized_bytes_size::Csize_t
    serialized_topology::Ptr{PJRT_SerializedTopology}
    serialized_topology_deleter::Ptr{Cvoid}
    function PJRT_TopologyDescription_Serialize_Args(
        extension_start,
        topology,
        serialized_bytes,
        serialized_bytes_size,
        serialized_topology,
        serialized_topology_deleter,
    )
        return new(
            csizeof(PJRT_TopologyDescription_Serialize_Args),
            extension_start,
            topology,
            serialized_bytes,
            serialized_bytes_size,
            serialized_topology,
            serialized_topology_deleter,
        )
    end
end

csizeof(::Type{PJRT_TopologyDescription_Serialize_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_TopologyDescription_Serialize ( PJRT_TopologyDescription_Serialize_Args * args )
const PJRT_TopologyDescription_Serialize = Cvoid

struct PJRT_TopologyDescription_Deserialize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    serialized_topology::Cstring
    serialized_topology_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    function PJRT_TopologyDescription_Deserialize_Args(
        extension_start, serialized_topology, serialized_topology_size, topology
    )
        return new(
            csizeof(PJRT_TopologyDescription_Deserialize_Args),
            extension_start,
            serialized_topology,
            serialized_topology_size,
            topology,
        )
    end
end

csizeof(::Type{PJRT_TopologyDescription_Deserialize_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TopologyDescription_Deserialize ( PJRT_TopologyDescription_Deserialize_Args * args )
const PJRT_TopologyDescription_Deserialize = Cvoid

struct PJRT_TopologyDescription_Attributes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    attributes::Ptr{PJRT_NamedValue}
    num_attributes::Csize_t
    function PJRT_TopologyDescription_Attributes_Args(
        extension_start, topology, attributes, num_attributes
    )
        return new(
            csizeof(PJRT_TopologyDescription_Attributes_Args),
            extension_start,
            topology,
            attributes,
            num_attributes,
        )
    end
end

csizeof(::Type{PJRT_TopologyDescription_Attributes_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TopologyDescription_Attributes ( PJRT_TopologyDescription_Attributes_Args * args )
const PJRT_TopologyDescription_Attributes = Cvoid

struct PJRT_TopologyDescription_Fingerprint_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    fingerprint::UInt64
    function PJRT_TopologyDescription_Fingerprint_Args(
        extension_start, topology, fingerprint
    )
        return new(
            csizeof(PJRT_TopologyDescription_Fingerprint_Args),
            extension_start,
            topology,
            fingerprint,
        )
    end
end

csizeof(::Type{PJRT_TopologyDescription_Fingerprint_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_TopologyDescription_Fingerprint ( PJRT_TopologyDescription_Fingerprint_Args * args )
const PJRT_TopologyDescription_Fingerprint = Cvoid

struct PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    memory_space_kind_id::Cint
    dims::Ptr{Int64}
    num_dims::Csize_t
    element_type::PJRT_Buffer_Type
    layout::Ptr{PJRT_Buffer_MemoryLayout}
    serialized_shape::Cstring
    serialized_shape_size::Csize_t
    serialized_shape_deleter::Ptr{Cvoid}
    function PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args(
        extension_start,
        topology,
        memory_space_kind_id,
        dims,
        num_dims,
        element_type,
        layout,
        serialized_shape,
        serialized_shape_size,
        serialized_shape_deleter,
    )
        return new(
            csizeof(PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args),
            extension_start,
            topology,
            memory_space_kind_id,
            dims,
            num_dims,
            element_type,
            layout,
            serialized_shape,
            serialized_shape_size,
            serialized_shape_deleter,
        )
    end
end

function csizeof(::Type{PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args})
    return Csize_t(88)
end

# typedef PJRT_Error * PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace ( PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args * args )
const PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace = Cvoid

struct PJRT_TopologyDescription_GetMemorySpaceKindIds_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    memory_space_kind_ids::Ptr{Cint}
    num_memory_space_kind_ids::Csize_t
    function PJRT_TopologyDescription_GetMemorySpaceKindIds_Args(
        extension_start, topology, memory_space_kind_ids, num_memory_space_kind_ids
    )
        return new(
            csizeof(PJRT_TopologyDescription_GetMemorySpaceKindIds_Args),
            extension_start,
            topology,
            memory_space_kind_ids,
            num_memory_space_kind_ids,
        )
    end
end

csizeof(::Type{PJRT_TopologyDescription_GetMemorySpaceKindIds_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TopologyDescription_GetMemorySpaceKindIds ( PJRT_TopologyDescription_GetMemorySpaceKindIds_Args * args )
const PJRT_TopologyDescription_GetMemorySpaceKindIds = Cvoid

struct PJRT_Compile_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    program::Ptr{PJRT_Program}
    compile_options::Cstring
    compile_options_size::Csize_t
    client::Ptr{PJRT_Client}
    executable::Ptr{PJRT_Executable}
    function PJRT_Compile_Args(
        extension_start,
        topology,
        program,
        compile_options,
        compile_options_size,
        client,
        executable,
    )
        return new(
            csizeof(PJRT_Compile_Args),
            extension_start,
            topology,
            program,
            compile_options,
            compile_options_size,
            client,
            executable,
        )
    end
end

csizeof(::Type{PJRT_Compile_Args}) = Csize_t(64)

# typedef PJRT_Error * PJRT_Compile ( PJRT_Compile_Args * args )
const PJRT_Compile = Cvoid

struct PJRT_Api
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    pjrt_api_version::PJRT_Api_Version
    PJRT_Error_Destroy::Ptr{PJRT_Error_Destroy}
    PJRT_Error_Message::Ptr{PJRT_Error_Message}
    PJRT_Error_GetCode::Ptr{PJRT_Error_GetCode}
    PJRT_Plugin_Initialize::Ptr{PJRT_Plugin_Initialize}
    PJRT_Plugin_Attributes::Ptr{PJRT_Plugin_Attributes}
    PJRT_Event_Destroy::Ptr{PJRT_Event_Destroy}
    PJRT_Event_IsReady::Ptr{PJRT_Event_IsReady}
    PJRT_Event_Error::Ptr{PJRT_Event_Error}
    PJRT_Event_Await::Ptr{PJRT_Event_Await}
    PJRT_Event_OnReady::Ptr{PJRT_Event_OnReady}
    PJRT_Client_Create::Ptr{PJRT_Client_Create}
    PJRT_Client_Destroy::Ptr{PJRT_Client_Destroy}
    PJRT_Client_PlatformName::Ptr{PJRT_Client_PlatformName}
    PJRT_Client_ProcessIndex::Ptr{PJRT_Client_ProcessIndex}
    PJRT_Client_PlatformVersion::Ptr{PJRT_Client_PlatformVersion}
    PJRT_Client_Devices::Ptr{PJRT_Client_Devices}
    PJRT_Client_AddressableDevices::Ptr{PJRT_Client_AddressableDevices}
    PJRT_Client_LookupDevice::Ptr{PJRT_Client_LookupDevice}
    PJRT_Client_LookupAddressableDevice::Ptr{PJRT_Client_LookupAddressableDevice}
    PJRT_Client_AddressableMemories::Ptr{PJRT_Client_AddressableMemories}
    PJRT_Client_Compile::Ptr{PJRT_Client_Compile}
    PJRT_Client_DefaultDeviceAssignment::Ptr{PJRT_Client_DefaultDeviceAssignment}
    PJRT_Client_BufferFromHostBuffer::Ptr{PJRT_Client_BufferFromHostBuffer}
    PJRT_DeviceDescription_Id::Ptr{PJRT_DeviceDescription_Id}
    PJRT_DeviceDescription_ProcessIndex::Ptr{PJRT_DeviceDescription_ProcessIndex}
    PJRT_DeviceDescription_Attributes::Ptr{PJRT_DeviceDescription_Attributes}
    PJRT_DeviceDescription_Kind::Ptr{PJRT_DeviceDescription_Kind}
    PJRT_DeviceDescription_DebugString::Ptr{PJRT_DeviceDescription_DebugString}
    PJRT_DeviceDescription_ToString::Ptr{PJRT_DeviceDescription_ToString}
    PJRT_Device_GetDescription::Ptr{PJRT_Device_GetDescription}
    PJRT_Device_IsAddressable::Ptr{PJRT_Device_IsAddressable}
    PJRT_Device_LocalHardwareId::Ptr{PJRT_Device_LocalHardwareId}
    PJRT_Device_AddressableMemories::Ptr{PJRT_Device_AddressableMemories}
    PJRT_Device_DefaultMemory::Ptr{PJRT_Device_DefaultMemory}
    PJRT_Device_MemoryStats::Ptr{PJRT_Device_MemoryStats}
    PJRT_Memory_Id::Ptr{PJRT_Memory_Id}
    PJRT_Memory_Kind::Ptr{PJRT_Memory_Kind}
    PJRT_Memory_DebugString::Ptr{PJRT_Memory_DebugString}
    PJRT_Memory_ToString::Ptr{PJRT_Memory_ToString}
    PJRT_Memory_AddressableByDevices::Ptr{PJRT_Memory_AddressableByDevices}
    PJRT_Executable_Destroy::Ptr{PJRT_Executable_Destroy}
    PJRT_Executable_Name::Ptr{PJRT_Executable_Name}
    PJRT_Executable_NumReplicas::Ptr{PJRT_Executable_NumReplicas}
    PJRT_Executable_NumPartitions::Ptr{PJRT_Executable_NumPartitions}
    PJRT_Executable_NumOutputs::Ptr{PJRT_Executable_NumOutputs}
    PJRT_Executable_SizeOfGeneratedCodeInBytes::Ptr{
        PJRT_Executable_SizeOfGeneratedCodeInBytes
    }
    PJRT_Executable_GetCostAnalysis::Ptr{PJRT_Executable_GetCostAnalysis}
    PJRT_Executable_OutputMemoryKinds::Ptr{PJRT_Executable_OutputMemoryKinds}
    PJRT_Executable_OptimizedProgram::Ptr{PJRT_Executable_OptimizedProgram}
    PJRT_Executable_Serialize::Ptr{PJRT_Executable_Serialize}
    PJRT_LoadedExecutable_Destroy::Ptr{PJRT_LoadedExecutable_Destroy}
    PJRT_LoadedExecutable_GetExecutable::Ptr{PJRT_LoadedExecutable_GetExecutable}
    PJRT_LoadedExecutable_AddressableDevices::Ptr{PJRT_LoadedExecutable_AddressableDevices}
    PJRT_LoadedExecutable_Delete::Ptr{PJRT_LoadedExecutable_Delete}
    PJRT_LoadedExecutable_IsDeleted::Ptr{PJRT_LoadedExecutable_IsDeleted}
    PJRT_LoadedExecutable_Execute::Ptr{PJRT_LoadedExecutable_Execute}
    PJRT_Executable_DeserializeAndLoad::Ptr{PJRT_Executable_DeserializeAndLoad}
    PJRT_LoadedExecutable_Fingerprint::Ptr{PJRT_LoadedExecutable_Fingerprint}
    PJRT_Buffer_Destroy::Ptr{PJRT_Buffer_Destroy}
    PJRT_Buffer_ElementType::Ptr{PJRT_Buffer_ElementType}
    PJRT_Buffer_Dimensions::Ptr{PJRT_Buffer_Dimensions}
    PJRT_Buffer_UnpaddedDimensions::Ptr{PJRT_Buffer_UnpaddedDimensions}
    PJRT_Buffer_DynamicDimensionIndices::Ptr{PJRT_Buffer_DynamicDimensionIndices}
    PJRT_Buffer_GetMemoryLayout::Ptr{PJRT_Buffer_GetMemoryLayout}
    PJRT_Buffer_OnDeviceSizeInBytes::Ptr{PJRT_Buffer_OnDeviceSizeInBytes}
    PJRT_Buffer_Device::Ptr{PJRT_Buffer_Device}
    PJRT_Buffer_Memory::Ptr{PJRT_Buffer_Memory}
    PJRT_Buffer_Delete::Ptr{PJRT_Buffer_Delete}
    PJRT_Buffer_IsDeleted::Ptr{PJRT_Buffer_IsDeleted}
    PJRT_Buffer_CopyToDevice::Ptr{PJRT_Buffer_CopyToDevice}
    PJRT_Buffer_ToHostBuffer::Ptr{PJRT_Buffer_ToHostBuffer}
    PJRT_Buffer_IsOnCpu::Ptr{PJRT_Buffer_IsOnCpu}
    PJRT_Buffer_ReadyEvent::Ptr{PJRT_Buffer_ReadyEvent}
    PJRT_Buffer_UnsafePointer::Ptr{PJRT_Buffer_UnsafePointer}
    PJRT_Buffer_IncreaseExternalReferenceCount::Ptr{
        PJRT_Buffer_IncreaseExternalReferenceCount
    }
    PJRT_Buffer_DecreaseExternalReferenceCount::Ptr{
        PJRT_Buffer_DecreaseExternalReferenceCount
    }
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer::Ptr{
        PJRT_Buffer_OpaqueDeviceMemoryDataPointer
    }
    PJRT_CopyToDeviceStream_Destroy::Ptr{PJRT_CopyToDeviceStream_Destroy}
    PJRT_CopyToDeviceStream_AddChunk::Ptr{PJRT_CopyToDeviceStream_AddChunk}
    PJRT_CopyToDeviceStream_TotalBytes::Ptr{PJRT_CopyToDeviceStream_TotalBytes}
    PJRT_CopyToDeviceStream_GranuleSize::Ptr{PJRT_CopyToDeviceStream_GranuleSize}
    PJRT_CopyToDeviceStream_CurrentBytes::Ptr{PJRT_CopyToDeviceStream_CurrentBytes}
    PJRT_TopologyDescription_Create::Ptr{PJRT_TopologyDescription_Create}
    PJRT_TopologyDescription_Destroy::Ptr{PJRT_TopologyDescription_Destroy}
    PJRT_TopologyDescription_PlatformName::Ptr{PJRT_TopologyDescription_PlatformName}
    PJRT_TopologyDescription_PlatformVersion::Ptr{PJRT_TopologyDescription_PlatformVersion}
    PJRT_TopologyDescription_GetDeviceDescriptions::Ptr{
        PJRT_TopologyDescription_GetDeviceDescriptions
    }
    PJRT_TopologyDescription_Serialize::Ptr{PJRT_TopologyDescription_Serialize}
    PJRT_TopologyDescription_Attributes::Ptr{PJRT_TopologyDescription_Attributes}
    PJRT_Compile::Ptr{PJRT_Compile}
    PJRT_Executable_OutputElementTypes::Ptr{PJRT_Executable_OutputElementTypes}
    PJRT_Executable_OutputDimensions::Ptr{PJRT_Executable_OutputDimensions}
    PJRT_Buffer_CopyToMemory::Ptr{PJRT_Buffer_CopyToMemory}
    PJRT_Client_CreateViewOfDeviceBuffer::Ptr{PJRT_Client_CreateViewOfDeviceBuffer}
    PJRT_Executable_Fingerprint::Ptr{PJRT_Executable_Fingerprint}
    PJRT_Client_TopologyDescription::Ptr{PJRT_Client_TopologyDescription}
    PJRT_Executable_GetCompiledMemoryStats::Ptr{PJRT_Executable_GetCompiledMemoryStats}
    PJRT_Memory_Kind_Id::Ptr{PJRT_Memory_Kind_Id}
    PJRT_ExecuteContext_Create::Ptr{PJRT_ExecuteContext_Create}
    PJRT_ExecuteContext_Destroy::Ptr{PJRT_ExecuteContext_Destroy}
    PJRT_Buffer_CopyRawToHost::Ptr{PJRT_Buffer_CopyRawToHost}
    PJRT_AsyncHostToDeviceTransferManager_Destroy::Ptr{
        PJRT_AsyncHostToDeviceTransferManager_Destroy
    }
    PJRT_AsyncHostToDeviceTransferManager_TransferData::Ptr{
        PJRT_AsyncHostToDeviceTransferManager_TransferData
    }
    PJRT_Client_CreateBuffersForAsyncHostToDevice::Ptr{
        PJRT_Client_CreateBuffersForAsyncHostToDevice
    }
    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer::Ptr{
        PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer
    }
    PJRT_AsyncHostToDeviceTransferManager_Device::Ptr{
        PJRT_AsyncHostToDeviceTransferManager_Device
    }
    PJRT_AsyncHostToDeviceTransferManager_BufferCount::Ptr{
        PJRT_AsyncHostToDeviceTransferManager_BufferCount
    }
    PJRT_AsyncHostToDeviceTransferManager_BufferSize::Ptr{
        PJRT_AsyncHostToDeviceTransferManager_BufferSize
    }
    PJRT_AsyncHostToDeviceTransferManager_SetBufferError::Ptr{
        PJRT_AsyncHostToDeviceTransferManager_SetBufferError
    }
    PJRT_AsyncHostToDeviceTransferManager_AddMetadata::Ptr{
        PJRT_AsyncHostToDeviceTransferManager_AddMetadata
    }
    PJRT_Client_DmaMap::Ptr{PJRT_Client_DmaMap}
    PJRT_Client_DmaUnmap::Ptr{PJRT_Client_DmaUnmap}
    PJRT_Client_CreateUninitializedBuffer::Ptr{PJRT_Client_CreateUninitializedBuffer}
    PJRT_Client_UpdateGlobalProcessInfo::Ptr{PJRT_Client_UpdateGlobalProcessInfo}
    PJRT_TopologyDescription_Deserialize::Ptr{PJRT_TopologyDescription_Deserialize}
    PJRT_Client_CreateAliasBuffer::Ptr{PJRT_Client_CreateAliasBuffer}
    PJRT_Client_FulfillAliasBuffer::Ptr{PJRT_Client_FulfillAliasBuffer}
    PJRT_LoadedExecutable_GetDeviceAssignment::Ptr{
        PJRT_LoadedExecutable_GetDeviceAssignment
    }
    PJRT_Client_CreateErrorBuffer::Ptr{PJRT_Client_CreateErrorBuffer}
    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral::Ptr{
        PJRT_AsyncHostToDeviceTransferManager_TransferLiteral
    }
    PJRT_Buffer_CopyRawToHostFuture::Ptr{PJRT_Buffer_CopyRawToHostFuture}
    PJRT_Device_PoisonExecution::Ptr{PJRT_Device_PoisonExecution}
    PJRT_Device_CreateAsyncTrackingEvent::Ptr{PJRT_Device_CreateAsyncTrackingEvent}
    PJRT_AsyncTrackingEvent_Destroy::Ptr{PJRT_AsyncTrackingEvent_Destroy}
    PJRT_Executable_GetCompileOptions::Ptr{PJRT_Executable_GetCompileOptions}
    PJRT_Buffer_DonateWithControlDependency::Ptr{PJRT_Buffer_DonateWithControlDependency}
    PJRT_Event_Create::Ptr{PJRT_Event_Create}
    PJRT_Event_Set::Ptr{PJRT_Event_Set}
    PJRT_Device_GetAttributes::Ptr{PJRT_Device_GetAttributes}
    PJRT_Client_Load::Ptr{PJRT_Client_Load}
    PJRT_LoadedExecutable_AddressableDeviceLogicalIds::Ptr{
        PJRT_LoadedExecutable_AddressableDeviceLogicalIds
    }
    PJRT_Buffer_Bitcast::Ptr{PJRT_Buffer_Bitcast}
    PJRT_Error_ForEachPayload::Ptr{PJRT_Error_ForEachPayload}
    PJRT_TopologyDescription_Fingerprint::Ptr{PJRT_TopologyDescription_Fingerprint}
    PJRT_Executable_ParameterMemoryKinds::Ptr{PJRT_Executable_ParameterMemoryKinds}
    PJRT_Device_ClearMemoryStats::Ptr{PJRT_Device_ClearMemoryStats}
    PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace::Ptr{
        PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace
    }
    PJRT_TopologyDescription_GetMemorySpaceKindIds::Ptr{
        PJRT_TopologyDescription_GetMemorySpaceKindIds
    }
end

csizeof(::Type{PJRT_Api}) = Csize_t(1144)

mutable struct PJRT_RuntimeAbiVersion end

mutable struct PJRT_ExecutableAbiVersion end

mutable struct PJRT_SerializedProto end

struct PJRT_Client_RuntimeAbiVersion_Args
    struct_size::Csize_t
    client::Ptr{PJRT_Client}
    abi_version::Ptr{PJRT_RuntimeAbiVersion}
    function PJRT_Client_RuntimeAbiVersion_Args(client, abi_version)
        return new(csizeof(PJRT_Client_RuntimeAbiVersion_Args), client, abi_version)
    end
end

csizeof(::Type{PJRT_Client_RuntimeAbiVersion_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Client_RuntimeAbiVersion ( PJRT_Client_RuntimeAbiVersion_Args * args )
const PJRT_Client_RuntimeAbiVersion = Cvoid

struct PJRT_Executable_GetAbiVersion_Args
    struct_size::Csize_t
    executable::Ptr{PJRT_Executable}
    abi_version::Ptr{PJRT_ExecutableAbiVersion}
    function PJRT_Executable_GetAbiVersion_Args(executable, abi_version)
        return new(csizeof(PJRT_Executable_GetAbiVersion_Args), executable, abi_version)
    end
end

csizeof(::Type{PJRT_Executable_GetAbiVersion_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Executable_GetAbiVersion ( PJRT_Executable_GetAbiVersion_Args * args )
const PJRT_Executable_GetAbiVersion = Cvoid

struct PJRT_RuntimeAbiVersion_Destroy_Args
    struct_size::Csize_t
    abi_version::Ptr{PJRT_RuntimeAbiVersion}
    function PJRT_RuntimeAbiVersion_Destroy_Args(abi_version)
        return new(csizeof(PJRT_RuntimeAbiVersion_Destroy_Args), abi_version)
    end
end

csizeof(::Type{PJRT_RuntimeAbiVersion_Destroy_Args}) = Csize_t(16)

# typedef PJRT_Error * PJRT_RuntimeAbiVersion_Destroy ( PJRT_RuntimeAbiVersion_Destroy_Args * args )
const PJRT_RuntimeAbiVersion_Destroy = Cvoid

struct PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args
    struct_size::Csize_t
    abi_version::Ptr{PJRT_RuntimeAbiVersion}
    other_abi_version::Ptr{PJRT_RuntimeAbiVersion}
    function PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args(
        abi_version, other_abi_version
    )
        return new(
            csizeof(PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args),
            abi_version,
            other_abi_version,
        )
    end
end

csizeof(::Type{PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime ( PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args * args )
const PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime = Cvoid

struct PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args
    struct_size::Csize_t
    abi_version::Ptr{PJRT_RuntimeAbiVersion}
    executable_abi_version::Ptr{PJRT_ExecutableAbiVersion}
    function PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args(
        abi_version, executable_abi_version
    )
        return new(
            csizeof(PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args),
            abi_version,
            executable_abi_version,
        )
    end
end

csizeof(::Type{PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable ( PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args * args )
const PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable = Cvoid

struct PJRT_RuntimeAbiVersion_ToProto_Args
    struct_size::Csize_t
    abi_version::Ptr{PJRT_RuntimeAbiVersion}
    serialized_proto::Cstring
    serialized_proto_size::Csize_t
    serialized_proto_holder::Ptr{PJRT_SerializedProto}
    serialized_proto_deleter::Ptr{Cvoid}
    function PJRT_RuntimeAbiVersion_ToProto_Args(
        abi_version,
        serialized_proto,
        serialized_proto_size,
        serialized_proto_holder,
        serialized_proto_deleter,
    )
        return new(
            csizeof(PJRT_RuntimeAbiVersion_ToProto_Args),
            abi_version,
            serialized_proto,
            serialized_proto_size,
            serialized_proto_holder,
            serialized_proto_deleter,
        )
    end
end

csizeof(::Type{PJRT_RuntimeAbiVersion_ToProto_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_RuntimeAbiVersion_ToProto ( PJRT_RuntimeAbiVersion_ToProto_Args * args )
const PJRT_RuntimeAbiVersion_ToProto = Cvoid

struct PJRT_RuntimeAbiVersion_PlatformId_Args
    struct_size::Csize_t
    abi_version::Ptr{PJRT_RuntimeAbiVersion}
    platform_id::UInt64
    function PJRT_RuntimeAbiVersion_PlatformId_Args(abi_version, platform_id)
        return new(
            csizeof(PJRT_RuntimeAbiVersion_PlatformId_Args), abi_version, platform_id
        )
    end
end

csizeof(::Type{PJRT_RuntimeAbiVersion_PlatformId_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_RuntimeAbiVersion_PlatformId ( PJRT_RuntimeAbiVersion_PlatformId_Args * args )
const PJRT_RuntimeAbiVersion_PlatformId = Cvoid

struct PJRT_ExecutableAbiVersion_Destroy_Args
    struct_size::Csize_t
    abi_version::Ptr{PJRT_ExecutableAbiVersion}
    function PJRT_ExecutableAbiVersion_Destroy_Args(abi_version)
        return new(csizeof(PJRT_ExecutableAbiVersion_Destroy_Args), abi_version)
    end
end

csizeof(::Type{PJRT_ExecutableAbiVersion_Destroy_Args}) = Csize_t(16)

# typedef PJRT_Error * PJRT_ExecutableAbiVersion_Destroy ( PJRT_ExecutableAbiVersion_Destroy_Args * args )
const PJRT_ExecutableAbiVersion_Destroy = Cvoid

struct PJRT_ExecutableAbiVersion_ToProto_Args
    struct_size::Csize_t
    abi_version::Ptr{PJRT_ExecutableAbiVersion}
    serialized_proto::Cstring
    serialized_proto_size::Csize_t
    serialized_proto_holder::Ptr{PJRT_SerializedProto}
    serialized_proto_deleter::Ptr{Cvoid}
    function PJRT_ExecutableAbiVersion_ToProto_Args(
        abi_version,
        serialized_proto,
        serialized_proto_size,
        serialized_proto_holder,
        serialized_proto_deleter,
    )
        return new(
            csizeof(PJRT_ExecutableAbiVersion_ToProto_Args),
            abi_version,
            serialized_proto,
            serialized_proto_size,
            serialized_proto_holder,
            serialized_proto_deleter,
        )
    end
end

csizeof(::Type{PJRT_ExecutableAbiVersion_ToProto_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_ExecutableAbiVersion_ToProto ( PJRT_ExecutableAbiVersion_ToProto_Args * args )
const PJRT_ExecutableAbiVersion_ToProto = Cvoid

struct PJRT_ExecutableAbiVersion_PlatformId_Args
    struct_size::Csize_t
    abi_version::Ptr{PJRT_ExecutableAbiVersion}
    platform_id::UInt64
    function PJRT_ExecutableAbiVersion_PlatformId_Args(abi_version, platform_id)
        return new(
            csizeof(PJRT_ExecutableAbiVersion_PlatformId_Args), abi_version, platform_id
        )
    end
end

csizeof(::Type{PJRT_ExecutableAbiVersion_PlatformId_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_ExecutableAbiVersion_PlatformId ( PJRT_ExecutableAbiVersion_PlatformId_Args * args )
const PJRT_ExecutableAbiVersion_PlatformId = Cvoid

struct PJRT_RuntimeAbiVersion_FromProto_Args
    struct_size::Csize_t
    serialized_proto::Cstring
    serialized_proto_size::Csize_t
    abi_version::Ptr{PJRT_RuntimeAbiVersion}
    function PJRT_RuntimeAbiVersion_FromProto_Args(
        serialized_proto, serialized_proto_size, abi_version
    )
        return new(
            csizeof(PJRT_RuntimeAbiVersion_FromProto_Args),
            serialized_proto,
            serialized_proto_size,
            abi_version,
        )
    end
end

csizeof(::Type{PJRT_RuntimeAbiVersion_FromProto_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_RuntimeAbiVersion_FromProto ( PJRT_RuntimeAbiVersion_FromProto_Args * args )
const PJRT_RuntimeAbiVersion_FromProto = Cvoid

struct PJRT_ExecutableAbiVersion_FromProto_Args
    struct_size::Csize_t
    serialized_proto::Cstring
    serialized_proto_size::Csize_t
    abi_version::Ptr{PJRT_ExecutableAbiVersion}
    function PJRT_ExecutableAbiVersion_FromProto_Args(
        serialized_proto, serialized_proto_size, abi_version
    )
        return new(
            csizeof(PJRT_ExecutableAbiVersion_FromProto_Args),
            serialized_proto,
            serialized_proto_size,
            abi_version,
        )
    end
end

csizeof(::Type{PJRT_ExecutableAbiVersion_FromProto_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_ExecutableAbiVersion_FromProto ( PJRT_ExecutableAbiVersion_FromProto_Args * args )
const PJRT_ExecutableAbiVersion_FromProto = Cvoid

struct PJRT_AbiVersion_Extension
    base::PJRT_Extension_Base
    client_runtime_abi_version::Ptr{PJRT_Client_RuntimeAbiVersion}
    executable_get_abi_version::Ptr{PJRT_Executable_GetAbiVersion}
    runtime_abi_version_destroy::Ptr{PJRT_RuntimeAbiVersion_Destroy}
    runtime_abi_version_is_compatible_with_runtime::Ptr{
        PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime
    }
    runtime_abi_version_is_compatible_with_executable::Ptr{
        PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable
    }
    runtime_abi_version_to_proto::Ptr{PJRT_RuntimeAbiVersion_ToProto}
    runtime_abi_version_platform_id::Ptr{PJRT_RuntimeAbiVersion_PlatformId}
    executable_abi_version_destroy::Ptr{PJRT_ExecutableAbiVersion_Destroy}
    executable_abi_version_to_proto::Ptr{PJRT_ExecutableAbiVersion_ToProto}
    executable_abi_version_platform_id::Ptr{PJRT_ExecutableAbiVersion_PlatformId}
    runtime_abi_version_from_proto::Ptr{PJRT_RuntimeAbiVersion_FromProto}
    executable_abi_version_from_proto::Ptr{PJRT_ExecutableAbiVersion_FromProto}
end

csizeof(::Type{PJRT_AbiVersion_Extension}) = Csize_t(120)

@cenum PJRT_Callback_Type::UInt32 begin
    PJRT_Callback_Type_Unknown = 0
    PJRT_Callback_Type_Tpu_SliceBuilder = 1
    PJRT_Callback_Type_Prefatal = 2
end

@cenum PJRT_Callback_Tpu_SliceFailureType::Int32 begin
    SLICE_FAILURE_UNKNOWN = 0
    SLICE_FAILURE_INIT_ERROR = 1
    SLICE_FAILURE_WORKER_UNAVAILABLE = 2
    SLICE_FAILURE_FLAPPING_TASK_ERROR = 3
    SLICE_FAILURE_SW_INJECT_ERROR = 4
    SLICE_FAILURE_CHIP_DRIVER_ERROR = 5
end

struct PJRT_Callback_Tpu_SliceBuilderArgs
    struct_size::Csize_t
    failure_type::PJRT_Callback_Tpu_SliceFailureType
end

csizeof(::Type{PJRT_Callback_Tpu_SliceBuilderArgs}) = Csize_t(12)

struct PJRT_Callback_PrefatalArgs
    struct_size::Csize_t
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
end

csizeof(::Type{PJRT_Callback_PrefatalArgs}) = Csize_t(32)

# typedef void PJRT_Callback_Function ( void * args , void * user_arg )
const PJRT_Callback_Function = Cvoid

struct PJRT_Callback_RegisterCallback_Args
    struct_size::Csize_t
    client::Ptr{PJRT_Client}
    type::PJRT_Callback_Type
    callback::Ptr{PJRT_Callback_Function}
    user_arg::Ptr{Cvoid}
    function PJRT_Callback_RegisterCallback_Args(client, type, callback, user_arg)
        return new(
            csizeof(PJRT_Callback_RegisterCallback_Args), client, type, callback, user_arg
        )
    end
end

csizeof(::Type{PJRT_Callback_RegisterCallback_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Register_Callback ( PJRT_Callback_RegisterCallback_Args * args )
const PJRT_Register_Callback = Cvoid

struct PJRT_Callback_InvokeCallback_Args
    struct_size::Csize_t
    client::Ptr{PJRT_Client}
    type::PJRT_Callback_Type
    args::Ptr{Cvoid}
    function PJRT_Callback_InvokeCallback_Args(client, type, args)
        return new(csizeof(PJRT_Callback_InvokeCallback_Args), client, type, args)
    end
end

csizeof(::Type{PJRT_Callback_InvokeCallback_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Callback_InvokeCallback ( PJRT_Callback_InvokeCallback_Args * args )
const PJRT_Callback_InvokeCallback = Cvoid

struct PJRT_Callback_Extension
    base::PJRT_Extension_Base
    register_callback::Ptr{PJRT_Register_Callback}
    invoke_callback::Ptr{PJRT_Callback_InvokeCallback}
end

csizeof(::Type{PJRT_Callback_Extension}) = Csize_t(40)

mutable struct PJRT_Collectives end

mutable struct PJRT_Collectives_Communicator end

mutable struct PJRT_Collectives_Communicators end

mutable struct PJRT_Collectives_ToString_Holder end

struct PJRT_Collectives_Destroy_Args
    struct_size::Csize_t
    collectives::Ptr{PJRT_Collectives}
    function PJRT_Collectives_Destroy_Args(collectives)
        return new(csizeof(PJRT_Collectives_Destroy_Args), collectives)
    end
end

csizeof(::Type{PJRT_Collectives_Destroy_Args}) = Csize_t(16)

# typedef PJRT_Error * PJRT_Collectives_Destroy ( PJRT_Collectives_Destroy_Args * args )
const PJRT_Collectives_Destroy = Cvoid

struct PJRT_Collectives_CreateCommunicators_Args
    struct_size::Csize_t
    collectives::Ptr{PJRT_Collectives}
    device_ids::Ptr{Int64}
    num_device_ids::Csize_t
    clique_ids::Ptr{Cstring}
    clique_id_sizes::Ptr{Csize_t}
    num_clique_ids::Csize_t
    rank_ids::Ptr{Int64}
    num_device_ranks::Csize_t
    communicators::Ptr{Ptr{PJRT_Collectives_Communicator}}
    num_communicators::Csize_t
    communicators_holder::Ptr{PJRT_Collectives_Communicators}
    communicators_holder_deleter::Ptr{Cvoid}
    function PJRT_Collectives_CreateCommunicators_Args(
        collectives,
        device_ids,
        num_device_ids,
        clique_ids,
        clique_id_sizes,
        num_clique_ids,
        rank_ids,
        num_device_ranks,
        communicators,
        num_communicators,
        communicators_holder,
        communicators_holder_deleter,
    )
        return new(
            csizeof(PJRT_Collectives_CreateCommunicators_Args),
            collectives,
            device_ids,
            num_device_ids,
            clique_ids,
            clique_id_sizes,
            num_clique_ids,
            rank_ids,
            num_device_ranks,
            communicators,
            num_communicators,
            communicators_holder,
            communicators_holder_deleter,
        )
    end
end

csizeof(::Type{PJRT_Collectives_CreateCommunicators_Args}) = Csize_t(104)

# typedef PJRT_Error * PJRT_Collectives_CreateCommunicators ( PJRT_Collectives_CreateCommunicators_Args * args )
const PJRT_Collectives_CreateCommunicators = Cvoid

struct PJRT_Collectives_Communicator_Destroy_Args
    struct_size::Csize_t
    communicator::Ptr{PJRT_Collectives_Communicator}
    function PJRT_Collectives_Communicator_Destroy_Args(communicator)
        return new(csizeof(PJRT_Collectives_Communicator_Destroy_Args), communicator)
    end
end

csizeof(::Type{PJRT_Collectives_Communicator_Destroy_Args}) = Csize_t(16)

# typedef PJRT_Error * PJRT_Collectives_Communicator_Destroy ( PJRT_Collectives_Communicator_Destroy_Args * args )
const PJRT_Collectives_Communicator_Destroy = Cvoid

@cenum PJRT_Collectives_ReductionKind::Int32 begin
    PJRT_COLLECTIVES_REDUCTION_SUM = 0
    PJRT_COLLECTIVES_REDUCTION_PRODUCT = 1
    PJRT_COLLECTIVES_REDUCTION_MIN = 2
    PJRT_COLLECTIVES_REDUCTION_MAX = 3
end

@cenum PJRT_Collectives_CollectiveOpKind::Int32 begin
    PJRT_COLLECTIVES_COLLECTIVE_OP_KIND_CROSS_MODULE = 0
    PJRT_COLLECTIVES_COLLECTIVE_OP_KIND_CROSS_REPLICA = 1
end

struct PJRT_Collectives_CpuExecutor
    run_id::Int64
    global_device_ids::Ptr{Int64}
    num_global_device_ids::Csize_t
    num_local_participants::Int32
    collective_op_kind::PJRT_Collectives_CollectiveOpKind
    op_id::Int64
    timeout_in_ns::Int64
end

struct PJRT_Collectives_Communicator_AllReduce_Args
    struct_size::Csize_t
    communicator::Ptr{PJRT_Collectives_Communicator}
    send_buffer_ptr::Ptr{Cvoid}
    send_buffer_size::Csize_t
    recv_buffer_ptr::Ptr{Cvoid}
    recv_buffer_size::Csize_t
    primitive_type::PJRT_Buffer_Type
    count::Csize_t
    reduction_kind::PJRT_Collectives_ReductionKind
    cpu_executor::Ptr{PJRT_Collectives_CpuExecutor}
    event::Ptr{PJRT_Event}
    function PJRT_Collectives_Communicator_AllReduce_Args(
        communicator,
        send_buffer_ptr,
        send_buffer_size,
        recv_buffer_ptr,
        recv_buffer_size,
        primitive_type,
        count,
        reduction_kind,
        cpu_executor,
        event,
    )
        return new(
            csizeof(PJRT_Collectives_Communicator_AllReduce_Args),
            communicator,
            send_buffer_ptr,
            send_buffer_size,
            recv_buffer_ptr,
            recv_buffer_size,
            primitive_type,
            count,
            reduction_kind,
            cpu_executor,
            event,
        )
    end
end

csizeof(::Type{PJRT_Collectives_Communicator_AllReduce_Args}) = Csize_t(88)

# typedef PJRT_Error * PJRT_Collectives_Communicator_AllReduce ( PJRT_Collectives_Communicator_AllReduce_Args * args )
const PJRT_Collectives_Communicator_AllReduce = Cvoid

struct PJRT_Collectives_Communicator_ReduceScatter_Args
    struct_size::Csize_t
    communicator::Ptr{PJRT_Collectives_Communicator}
    send_buffer_ptr::Ptr{Cvoid}
    send_buffer_size::Csize_t
    recv_buffer_ptr::Ptr{Cvoid}
    recv_buffer_size::Csize_t
    primitive_type::PJRT_Buffer_Type
    count::Csize_t
    reduction_kind::PJRT_Collectives_ReductionKind
    cpu_executor::Ptr{PJRT_Collectives_CpuExecutor}
    event::Ptr{PJRT_Event}
    function PJRT_Collectives_Communicator_ReduceScatter_Args(
        communicator,
        send_buffer_ptr,
        send_buffer_size,
        recv_buffer_ptr,
        recv_buffer_size,
        primitive_type,
        count,
        reduction_kind,
        cpu_executor,
        event,
    )
        return new(
            csizeof(PJRT_Collectives_Communicator_ReduceScatter_Args),
            communicator,
            send_buffer_ptr,
            send_buffer_size,
            recv_buffer_ptr,
            recv_buffer_size,
            primitive_type,
            count,
            reduction_kind,
            cpu_executor,
            event,
        )
    end
end

csizeof(::Type{PJRT_Collectives_Communicator_ReduceScatter_Args}) = Csize_t(88)

# typedef PJRT_Error * PJRT_Collectives_Communicator_ReduceScatter ( PJRT_Collectives_Communicator_ReduceScatter_Args * args )
const PJRT_Collectives_Communicator_ReduceScatter = Cvoid

struct PJRT_Collectives_Communicator_AllGather_Args
    struct_size::Csize_t
    communicator::Ptr{PJRT_Collectives_Communicator}
    send_buffer_ptr::Ptr{Cvoid}
    send_buffer_size::Csize_t
    recv_buffer_ptr::Ptr{Cvoid}
    recv_buffer_size::Csize_t
    primitive_type::PJRT_Buffer_Type
    count::Csize_t
    cpu_executor::Ptr{PJRT_Collectives_CpuExecutor}
    event::Ptr{PJRT_Event}
    function PJRT_Collectives_Communicator_AllGather_Args(
        communicator,
        send_buffer_ptr,
        send_buffer_size,
        recv_buffer_ptr,
        recv_buffer_size,
        primitive_type,
        count,
        cpu_executor,
        event,
    )
        return new(
            csizeof(PJRT_Collectives_Communicator_AllGather_Args),
            communicator,
            send_buffer_ptr,
            send_buffer_size,
            recv_buffer_ptr,
            recv_buffer_size,
            primitive_type,
            count,
            cpu_executor,
            event,
        )
    end
end

csizeof(::Type{PJRT_Collectives_Communicator_AllGather_Args}) = Csize_t(80)

# typedef PJRT_Error * PJRT_Collectives_Communicator_AllGather ( PJRT_Collectives_Communicator_AllGather_Args * args )
const PJRT_Collectives_Communicator_AllGather = Cvoid

struct PJRT_Collectives_Communicator_CollectivePermute_Args
    struct_size::Csize_t
    communicator::Ptr{PJRT_Collectives_Communicator}
    send_buffer_ptr::Ptr{Cvoid}
    send_buffer_size::Csize_t
    recv_buffer_ptr::Ptr{Cvoid}
    recv_buffer_size::Csize_t
    primitive_type::PJRT_Buffer_Type
    count::Csize_t
    has_source_rank::Bool
    source_rank::Int64
    target_ranks::Ptr{Int64}
    num_target_ranks::Csize_t
    cpu_executor::Ptr{PJRT_Collectives_CpuExecutor}
    event::Ptr{PJRT_Event}
    function PJRT_Collectives_Communicator_CollectivePermute_Args(
        communicator,
        send_buffer_ptr,
        send_buffer_size,
        recv_buffer_ptr,
        recv_buffer_size,
        primitive_type,
        count,
        has_source_rank,
        source_rank,
        target_ranks,
        num_target_ranks,
        cpu_executor,
        event,
    )
        return new(
            csizeof(PJRT_Collectives_Communicator_CollectivePermute_Args),
            communicator,
            send_buffer_ptr,
            send_buffer_size,
            recv_buffer_ptr,
            recv_buffer_size,
            primitive_type,
            count,
            has_source_rank,
            source_rank,
            target_ranks,
            num_target_ranks,
            cpu_executor,
            event,
        )
    end
end

csizeof(::Type{PJRT_Collectives_Communicator_CollectivePermute_Args}) = Csize_t(112)

# typedef PJRT_Error * PJRT_Collectives_Communicator_CollectivePermute ( PJRT_Collectives_Communicator_CollectivePermute_Args * args )
const PJRT_Collectives_Communicator_CollectivePermute = Cvoid

struct PJRT_Collectives_Communicator_AllToAll_Args
    struct_size::Csize_t
    communicator::Ptr{PJRT_Collectives_Communicator}
    send_buffers_ptrs::Ptr{Ptr{Cvoid}}
    send_buffers_sizes::Ptr{Csize_t}
    num_send_buffers::Csize_t
    recv_buffers_ptrs::Ptr{Ptr{Cvoid}}
    recv_buffers_sizes::Ptr{Csize_t}
    num_recv_buffers::Csize_t
    primitive_type::PJRT_Buffer_Type
    count::Csize_t
    cpu_executor::Ptr{PJRT_Collectives_CpuExecutor}
    event::Ptr{PJRT_Event}
    function PJRT_Collectives_Communicator_AllToAll_Args(
        communicator,
        send_buffers_ptrs,
        send_buffers_sizes,
        num_send_buffers,
        recv_buffers_ptrs,
        recv_buffers_sizes,
        num_recv_buffers,
        primitive_type,
        count,
        cpu_executor,
        event,
    )
        return new(
            csizeof(PJRT_Collectives_Communicator_AllToAll_Args),
            communicator,
            send_buffers_ptrs,
            send_buffers_sizes,
            num_send_buffers,
            recv_buffers_ptrs,
            recv_buffers_sizes,
            num_recv_buffers,
            primitive_type,
            count,
            cpu_executor,
            event,
        )
    end
end

csizeof(::Type{PJRT_Collectives_Communicator_AllToAll_Args}) = Csize_t(96)

# typedef PJRT_Error * PJRT_Collectives_Communicator_AllToAll ( PJRT_Collectives_Communicator_AllToAll_Args * args )
const PJRT_Collectives_Communicator_AllToAll = Cvoid

struct PJRT_Collectives_Communicator_ToString_Args
    struct_size::Csize_t
    communicator::Ptr{PJRT_Collectives_Communicator}
    str::Cstring
    str_size::Csize_t
    str_holder::Ptr{PJRT_Collectives_ToString_Holder}
    str_holder_deleter::Ptr{Cvoid}
    function PJRT_Collectives_Communicator_ToString_Args(
        communicator, str, str_size, str_holder, str_holder_deleter
    )
        return new(
            csizeof(PJRT_Collectives_Communicator_ToString_Args),
            communicator,
            str,
            str_size,
            str_holder,
            str_holder_deleter,
        )
    end
end

csizeof(::Type{PJRT_Collectives_Communicator_ToString_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Collectives_Communicator_ToString ( PJRT_Collectives_Communicator_ToString_Args * args )
const PJRT_Collectives_Communicator_ToString = Cvoid

struct PJRT_Collectives_Extension
    base::PJRT_Extension_Base
    collectives_destroy::Ptr{PJRT_Collectives_Destroy}
    collectives_create_communicators::Ptr{PJRT_Collectives_CreateCommunicators}
    communicator_destroy::Ptr{PJRT_Collectives_Communicator_Destroy}
    communicator_all_reduce::Ptr{PJRT_Collectives_Communicator_AllReduce}
    communicator_reduce_scatter::Ptr{PJRT_Collectives_Communicator_ReduceScatter}
    communicator_all_gather::Ptr{PJRT_Collectives_Communicator_AllGather}
    communicator_collective_permute::Ptr{PJRT_Collectives_Communicator_CollectivePermute}
    communicator_all_to_all::Ptr{PJRT_Collectives_Communicator_AllToAll}
    communicator_to_string::Ptr{PJRT_Collectives_Communicator_ToString}
end

csizeof(::Type{PJRT_Collectives_Extension}) = Csize_t(96)

struct JAX_CustomCallPartitioner_string
    data::Cstring
    size::Csize_t
end

struct JAX_CustomCallPartitioner_aval
    shape::JAX_CustomCallPartitioner_string
    has_sharding::Bool
    sharding::JAX_CustomCallPartitioner_string
end

struct JAX_CustomCallPartitioner_version_and_error
    api_version::Int64
    data::Ptr{Cvoid}
    cleanup_fn::Ptr{Cvoid}
    has_error::Bool
    code::PJRT_Error_Code
    error_msg::JAX_CustomCallPartitioner_string
end

struct JAX_CustomCallPartitioner_Partition_Args
    header::JAX_CustomCallPartitioner_version_and_error
    num_args::Csize_t
    op_args::Ptr{JAX_CustomCallPartitioner_aval}
    op_result::JAX_CustomCallPartitioner_aval
    backend_config::JAX_CustomCallPartitioner_string
    mlir_module::JAX_CustomCallPartitioner_string
    args_sharding::Ptr{JAX_CustomCallPartitioner_string}
    result_sharding::JAX_CustomCallPartitioner_string
end

struct JAX_CustomCallPartitioner_InferShardingFromOperands_Args
    header::JAX_CustomCallPartitioner_version_and_error
    num_args::Csize_t
    op_args::Ptr{JAX_CustomCallPartitioner_aval}
    result_shape::JAX_CustomCallPartitioner_string
    backend_config::JAX_CustomCallPartitioner_string
    has_result_sharding::Bool
    result_sharding::JAX_CustomCallPartitioner_string
end

struct JAX_CustomCallPartitioner_PropagateUserSharding_Args
    header::JAX_CustomCallPartitioner_version_and_error
    backend_config::JAX_CustomCallPartitioner_string
    result_shape::JAX_CustomCallPartitioner_string
    result_sharding::JAX_CustomCallPartitioner_string
end

struct JAX_CustomCallPartitioner_Callbacks
    version::Int64
    private_data::Ptr{Cvoid}
    dtor::Ptr{Cvoid}
    partition::Ptr{Cvoid}
    infer_sharding::Ptr{Cvoid}
    propagate_user_sharding::Ptr{Cvoid}
    can_side_effecting_have_replicated_sharding::Bool
end

struct PJRT_Register_Custom_Partitioner_Args
    struct_size::Csize_t
    name::Cstring
    name_size::Csize_t
    callbacks::Ptr{JAX_CustomCallPartitioner_Callbacks}
    function PJRT_Register_Custom_Partitioner_Args(name, name_size, callbacks)
        return new(
            csizeof(PJRT_Register_Custom_Partitioner_Args), name, name_size, callbacks
        )
    end
end

csizeof(::Type{PJRT_Register_Custom_Partitioner_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Register_Custom_Partitioner ( PJRT_Register_Custom_Partitioner_Args * args )
const PJRT_Register_Custom_Partitioner = Cvoid

struct PJRT_Register_Batch_Partitionable_Args
    struct_size::Csize_t
    name::Cstring
    name_size::Csize_t
    function PJRT_Register_Batch_Partitionable_Args(name, name_size)
        return new(csizeof(PJRT_Register_Batch_Partitionable_Args), name, name_size)
    end
end

csizeof(::Type{PJRT_Register_Batch_Partitionable_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Register_Batch_Partitionable ( PJRT_Register_Batch_Partitionable_Args * args )
const PJRT_Register_Batch_Partitionable = Cvoid

struct PJRT_Custom_Partitioner_Extension
    base::PJRT_Extension_Base
    register_custom_partitioner::Ptr{PJRT_Register_Custom_Partitioner}
    register_batch_partitionable::Ptr{PJRT_Register_Batch_Partitionable}
end

csizeof(::Type{PJRT_Custom_Partitioner_Extension}) = Csize_t(40)

# typedef void ( * PJRT_DeviceEvent_AndThen ) ( void * user_arg )
const PJRT_DeviceEvent_AndThen = Ptr{Cvoid}

@cenum PJRT_DeviceEvent_State::UInt32 begin
    PJRT_DeviceEvent_State_Unavailable = 0
    PJRT_DeviceEvent_State_Ready = 1
    PJRT_DeviceEvent_State_Error = 2
end

struct PJRT_DeviceEvent_FunctionTable
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    inc_ref::Ptr{Cvoid}
    dec_ref::Ptr{Cvoid}
    and_then::Ptr{Cvoid}
    get_error_if_present::Ptr{Cvoid}
    parent::Ptr{PJRT_DeviceEvent_FunctionTable}
    get_state::Ptr{Cvoid}
    get_definition_stream::Ptr{Cvoid}
end

csizeof(::Type{PJRT_DeviceEvent_FunctionTable}) = Csize_t(72)

struct PJRT_DeviceEvent
    vtable::Ptr{PJRT_DeviceEvent_FunctionTable}
    device_event::Ptr{Cvoid}
end

struct PJRT_DeviceEventVector
    data::Ptr{PJRT_DeviceEvent}
    size::Csize_t
    capacity::Csize_t
    destroy::Ptr{Cvoid}
end

struct PJRT_DeviceEventPromise_FunctionTable
    struct_size::Csize_t
    instance_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    inc_ref::Ptr{Cvoid}
    dec_ref::Ptr{Cvoid}
    event::Ptr{Cvoid}
    set::Ptr{Cvoid}
    set_error::Ptr{Cvoid}
    set_ready::Ptr{Cvoid}
end

struct PJRT_DeviceEventPromise
    vtable::Ptr{PJRT_DeviceEventPromise_FunctionTable}
end

csizeof(::Type{PJRT_DeviceEventPromise_FunctionTable}) = Csize_t(72)

csizeof(::Type{PJRT_DeviceEventPromise}) = Csize_t(8)

struct PJRT_FFI_Type_Info
    deleter::Ptr{Cvoid}
    serialize::Ptr{Cvoid}
    deserialize::Ptr{Cvoid}
end

struct PJRT_FFI_Type_Register_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    type_name::Cstring
    type_name_size::Csize_t
    type_id::Int64
    type_info::Ptr{PJRT_FFI_Type_Info}
    function PJRT_FFI_Type_Register_Args(
        extension_start, type_name, type_name_size, type_id, type_info
    )
        return new(
            csizeof(PJRT_FFI_Type_Register_Args),
            extension_start,
            type_name,
            type_name_size,
            type_id,
            type_info,
        )
    end
end

csizeof(::Type{PJRT_FFI_Type_Register_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_FFI_Type_Register ( PJRT_FFI_Type_Register_Args * args )
const PJRT_FFI_Type_Register = Cvoid

struct PJRT_FFI_UserData
    type_id::Int64
    data::Ptr{Cvoid}
end

struct PJRT_FFI_UserData_Add_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    context::Ptr{PJRT_ExecuteContext}
    user_data::PJRT_FFI_UserData
    function PJRT_FFI_UserData_Add_Args(extension_start, context, user_data)
        return new(csizeof(PJRT_FFI_UserData_Add_Args), extension_start, context, user_data)
    end
end

csizeof(::Type{PJRT_FFI_UserData_Add_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_FFI_UserData_Add ( PJRT_FFI_UserData_Add_Args * args )
const PJRT_FFI_UserData_Add = Cvoid

@cenum PJRT_FFI_Handler_TraitsBits::UInt32 begin
    PJRT_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE = 1
end

struct PJRT_FFI_Register_Handler_Args
    struct_size::Csize_t
    target_name::Cstring
    target_name_size::Csize_t
    handler::Ptr{Cvoid}
    platform_name::Cstring
    platform_name_size::Csize_t
    traits::PJRT_FFI_Handler_TraitsBits
    function PJRT_FFI_Register_Handler_Args(
        target_name, target_name_size, handler, platform_name, platform_name_size, traits
    )
        return new(
            csizeof(PJRT_FFI_Register_Handler_Args),
            target_name,
            target_name_size,
            handler,
            platform_name,
            platform_name_size,
            traits,
        )
    end
end

csizeof(::Type{PJRT_FFI_Register_Handler_Args}) = Csize_t(52)

# typedef PJRT_Error * PJRT_FFI_Register_Handler ( PJRT_FFI_Register_Handler_Args * args )
const PJRT_FFI_Register_Handler = Cvoid

struct PJRT_FFI_Extension
    base::PJRT_Extension_Base
    type_register::Ptr{PJRT_FFI_Type_Register}
    user_data_add::Ptr{PJRT_FFI_UserData_Add}
    register_handler::Ptr{PJRT_FFI_Register_Handler}
end

const PJRT_FFI = PJRT_FFI_Extension

csizeof(::Type{PJRT_FFI_Extension}) = Csize_t(48)

struct PJRT_Gpu_Register_Custom_Call_Args
    struct_size::Csize_t
    function_name::Cstring
    function_name_size::Csize_t
    api_version::Cint
    handler_instantiate::Ptr{Cvoid}
    handler_prepare::Ptr{Cvoid}
    handler_initialize::Ptr{Cvoid}
    handler_execute::Ptr{Cvoid}
    traits::UInt32
    function PJRT_Gpu_Register_Custom_Call_Args(
        function_name,
        function_name_size,
        api_version,
        handler_instantiate,
        handler_prepare,
        handler_initialize,
        handler_execute,
        traits,
    )
        return new(
            csizeof(PJRT_Gpu_Register_Custom_Call_Args),
            function_name,
            function_name_size,
            api_version,
            handler_instantiate,
            handler_prepare,
            handler_initialize,
            handler_execute,
            traits,
        )
    end
end

csizeof(::Type{PJRT_Gpu_Register_Custom_Call_Args}) = Csize_t(68)

# typedef PJRT_Error * PJRT_Gpu_Register_Custom_Call ( PJRT_Gpu_Register_Custom_Call_Args * args )
const PJRT_Gpu_Register_Custom_Call = Cvoid

struct PJRT_Gpu_Custom_Call
    base::PJRT_Extension_Base
    custom_call::Ptr{PJRT_Gpu_Register_Custom_Call}
    custom_call_handles_traits::Bool
end

csizeof(::Type{PJRT_Gpu_Custom_Call}) = Csize_t(33)

mutable struct PJRT_Layouts_MemoryLayout end

mutable struct PJRT_Layouts_SerializedLayout end

struct PJRT_Layouts_MemoryLayout_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    layout::Ptr{PJRT_Layouts_MemoryLayout}
    function PJRT_Layouts_MemoryLayout_Destroy_Args(extension_start, layout)
        return new(csizeof(PJRT_Layouts_MemoryLayout_Destroy_Args), extension_start, layout)
    end
end

csizeof(::Type{PJRT_Layouts_MemoryLayout_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Layouts_MemoryLayout_Destroy ( PJRT_Layouts_MemoryLayout_Destroy_Args * args )
const PJRT_Layouts_MemoryLayout_Destroy = Cvoid

struct PJRT_Layouts_MemoryLayout_Serialize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    layout::Ptr{PJRT_Layouts_MemoryLayout}
    serialized_bytes::Cstring
    serialized_bytes_size::Csize_t
    serialized_layout::Ptr{PJRT_Layouts_SerializedLayout}
    serialized_layout_deleter::Ptr{Cvoid}
    function PJRT_Layouts_MemoryLayout_Serialize_Args(
        extension_start,
        layout,
        serialized_bytes,
        serialized_bytes_size,
        serialized_layout,
        serialized_layout_deleter,
    )
        return new(
            csizeof(PJRT_Layouts_MemoryLayout_Serialize_Args),
            extension_start,
            layout,
            serialized_bytes,
            serialized_bytes_size,
            serialized_layout,
            serialized_layout_deleter,
        )
    end
end

csizeof(::Type{PJRT_Layouts_MemoryLayout_Serialize_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_Layouts_MemoryLayout_Serialize ( PJRT_Layouts_MemoryLayout_Serialize_Args * args )
const PJRT_Layouts_MemoryLayout_Serialize = Cvoid

struct PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    layout::Ptr{PJRT_Layouts_MemoryLayout}
    function PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args(extension_start, buffer, layout)
        return new(
            csizeof(PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args),
            extension_start,
            buffer,
            layout,
        )
    end
end

csizeof(::Type{PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Layouts_PJRT_Buffer_MemoryLayout ( PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args * args )
const PJRT_Layouts_PJRT_Buffer_MemoryLayout = Cvoid

struct PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    type::PJRT_Buffer_Type
    dims::Ptr{Int64}
    num_dims::Csize_t
    layout::Ptr{PJRT_Layouts_MemoryLayout}
    function PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args(
        extension_start, client, type, dims, num_dims, layout
    )
        return new(
            csizeof(PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args),
            extension_start,
            client,
            type,
            dims,
            num_dims,
            layout,
        )
    end
end

csizeof(::Type{PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_Layouts_PJRT_Client_GetDefaultLayout ( PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args * args )
const PJRT_Layouts_PJRT_Client_GetDefaultLayout = Cvoid

struct PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology_description::Ptr{PJRT_TopologyDescription}
    type::PJRT_Buffer_Type
    dims::Ptr{Int64}
    num_dims::Csize_t
    layout::Ptr{PJRT_Layouts_MemoryLayout}
    function PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args(
        extension_start, topology_description, type, dims, num_dims, layout
    )
        return new(
            csizeof(PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args),
            extension_start,
            topology_description,
            type,
            dims,
            num_dims,
            layout,
        )
    end
end

csizeof(::Type{PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_Layouts_PJRT_Topology_GetDefaultLayout ( PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args * args )
const PJRT_Layouts_PJRT_Topology_GetDefaultLayout = Cvoid

struct PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_outputs::Csize_t
    layouts::Ptr{Ptr{PJRT_Layouts_MemoryLayout}}
    function PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args(
        extension_start, executable, num_outputs, layouts
    )
        return new(
            csizeof(PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args),
            extension_start,
            executable,
            num_outputs,
            layouts,
        )
    end
end

csizeof(::Type{PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Layouts_PJRT_Executable_GetOutputLayouts ( PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args * args )
const PJRT_Layouts_PJRT_Executable_GetOutputLayouts = Cvoid

struct PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_parameters::Csize_t
    layouts::Ptr{Ptr{PJRT_Layouts_MemoryLayout}}
    function PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args(
        extension_start, executable, num_parameters, layouts
    )
        return new(
            csizeof(PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args),
            extension_start,
            executable,
            num_parameters,
            layouts,
        )
    end
end

csizeof(::Type{PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Layouts_PJRT_Executable_GetParameterLayouts ( PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args * args )
const PJRT_Layouts_PJRT_Executable_GetParameterLayouts = Cvoid

struct PJRT_Layouts_Extension
    base::PJRT_Extension_Base
    PJRT_Layouts_MemoryLayout_Destroy::Ptr{PJRT_Layouts_MemoryLayout_Destroy}
    PJRT_Layouts_MemoryLayout_Serialize::Ptr{PJRT_Layouts_MemoryLayout_Serialize}
    PJRT_Layouts_PJRT_Client_GetDefaultLayout::Ptr{
        PJRT_Layouts_PJRT_Client_GetDefaultLayout
    }
    PJRT_Layouts_PJRT_Buffer_MemoryLayout::Ptr{PJRT_Layouts_PJRT_Buffer_MemoryLayout}
    PJRT_Layouts_PJRT_Topology_GetDefaultLayout::Ptr{
        PJRT_Layouts_PJRT_Topology_GetDefaultLayout
    }
    PJRT_Layouts_PJRT_Executable_GetOutputLayouts::Ptr{
        PJRT_Layouts_PJRT_Executable_GetOutputLayouts
    }
    PJRT_Layouts_PJRT_Executable_GetParameterLayouts::Ptr{
        PJRT_Layouts_PJRT_Executable_GetParameterLayouts
    }
end

csizeof(::Type{PJRT_Layouts_Extension}) = Csize_t(80)

mutable struct PJRT_Megascale_ClientContext end

mutable struct PJRT_Megascale_NumDevicesPerSlice end

mutable struct PJRT_Megascale_SerializedConfig end

struct PJRT_Megascale_CreateClientContextFromPjRtClient_Args
    struct_size::Csize_t
    client::Ptr{PJRT_Client}
    client_context::Ptr{PJRT_Megascale_ClientContext}
    function PJRT_Megascale_CreateClientContextFromPjRtClient_Args(client, client_context)
        return new(
            csizeof(PJRT_Megascale_CreateClientContextFromPjRtClient_Args),
            client,
            client_context,
        )
    end
end

csizeof(::Type{PJRT_Megascale_CreateClientContextFromPjRtClient_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Megascale_CreateClientContextFromPjRtClient ( PJRT_Megascale_CreateClientContextFromPjRtClient_Args * args )
const PJRT_Megascale_CreateClientContextFromPjRtClient = Cvoid

struct PJRT_Megascale_CreateDefaultClientContext_Args
    struct_size::Csize_t
    client_context::Ptr{PJRT_Megascale_ClientContext}
    function PJRT_Megascale_CreateDefaultClientContext_Args(client_context)
        return new(csizeof(PJRT_Megascale_CreateDefaultClientContext_Args), client_context)
    end
end

csizeof(::Type{PJRT_Megascale_CreateDefaultClientContext_Args}) = Csize_t(16)

# typedef PJRT_Error * PJRT_Megascale_CreateDefaultClientContext ( PJRT_Megascale_CreateDefaultClientContext_Args * args )
const PJRT_Megascale_CreateDefaultClientContext = Cvoid

struct PJRT_Megascale_DeleteClientContext_Args
    struct_size::Csize_t
    client_context::Ptr{PJRT_Megascale_ClientContext}
    function PJRT_Megascale_DeleteClientContext_Args(client_context)
        return new(csizeof(PJRT_Megascale_DeleteClientContext_Args), client_context)
    end
end

csizeof(::Type{PJRT_Megascale_DeleteClientContext_Args}) = Csize_t(16)

# typedef PJRT_Error * PJRT_Megascale_DeleteClientContext ( PJRT_Megascale_DeleteClientContext_Args * args )
const PJRT_Megascale_DeleteClientContext = Cvoid

struct PJRT_Megascale_CreateAoTConfig_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    num_slices::Int32
    multi_slice_config::Ptr{PJRT_MultiSlice_Config}
    function PJRT_Megascale_CreateAoTConfig_Args(topology, num_slices, multi_slice_config)
        return new(
            csizeof(PJRT_Megascale_CreateAoTConfig_Args),
            topology,
            num_slices,
            multi_slice_config,
        )
    end
end

csizeof(::Type{PJRT_Megascale_CreateAoTConfig_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Megascale_CreateAoTConfig ( PJRT_Megascale_CreateAoTConfig_Args * args )
const PJRT_Megascale_CreateAoTConfig = Cvoid

struct PJRT_Megascale_CreateMultiSliceConfig_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    num_slices::Int32
    local_slice_id::Int32
    local_host_id::Int32
    endpoint_addresses::Cstring
    endpoint_addresses_size::Int32
    dcn_topology::Cstring
    dcn_topology_size::Int32
    client_context::Ptr{PJRT_Megascale_ClientContext}
    multi_slice_config::Ptr{PJRT_MultiSlice_Config}
    function PJRT_Megascale_CreateMultiSliceConfig_Args(
        topology,
        num_slices,
        local_slice_id,
        local_host_id,
        endpoint_addresses,
        endpoint_addresses_size,
        dcn_topology,
        dcn_topology_size,
        client_context,
        multi_slice_config,
    )
        return new(
            csizeof(PJRT_Megascale_CreateMultiSliceConfig_Args),
            topology,
            num_slices,
            local_slice_id,
            local_host_id,
            endpoint_addresses,
            endpoint_addresses_size,
            dcn_topology,
            dcn_topology_size,
            client_context,
            multi_slice_config,
        )
    end
end

csizeof(::Type{PJRT_Megascale_CreateMultiSliceConfig_Args}) = Csize_t(80)

# typedef PJRT_Error * PJRT_Megascale_CreateMultiSliceConfig ( PJRT_Megascale_CreateMultiSliceConfig_Args * args )
const PJRT_Megascale_CreateMultiSliceConfig = Cvoid

struct PJRT_Megascale_ClientContext_Initialize_Args
    struct_size::Csize_t
    client_context::Ptr{PJRT_Megascale_ClientContext}
    has_incarnation_id::Bool
    incarnation_id::UInt64
    function PJRT_Megascale_ClientContext_Initialize_Args(
        client_context, has_incarnation_id, incarnation_id
    )
        return new(
            csizeof(PJRT_Megascale_ClientContext_Initialize_Args),
            client_context,
            has_incarnation_id,
            incarnation_id,
        )
    end
end

csizeof(::Type{PJRT_Megascale_ClientContext_Initialize_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Megascale_ClientContext_Initialize ( PJRT_Megascale_ClientContext_Initialize_Args * args )
const PJRT_Megascale_ClientContext_Initialize = Cvoid

struct PJRT_Megascale_ClientContext_UnblockPendingWork_Args
    struct_size::Csize_t
    client_context::Ptr{PJRT_Megascale_ClientContext}
    launch_id::Int32
    expire_after_ms::Int64
    function PJRT_Megascale_ClientContext_UnblockPendingWork_Args(
        client_context, launch_id, expire_after_ms
    )
        return new(
            csizeof(PJRT_Megascale_ClientContext_UnblockPendingWork_Args),
            client_context,
            launch_id,
            expire_after_ms,
        )
    end
end

csizeof(::Type{PJRT_Megascale_ClientContext_UnblockPendingWork_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Megascale_ClientContext_UnblockPendingWork ( PJRT_Megascale_ClientContext_UnblockPendingWork_Args * args )
const PJRT_Megascale_ClientContext_UnblockPendingWork = Cvoid

struct PJRT_Megascale_ClientContext_MegascalePort_Args
    struct_size::Csize_t
    client_context::Ptr{PJRT_Megascale_ClientContext}
    port::Int32
    function PJRT_Megascale_ClientContext_MegascalePort_Args(client_context, port)
        return new(
            csizeof(PJRT_Megascale_ClientContext_MegascalePort_Args), client_context, port
        )
    end
end

csizeof(::Type{PJRT_Megascale_ClientContext_MegascalePort_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_Megascale_ClientContext_MegascalePort ( PJRT_Megascale_ClientContext_MegascalePort_Args * args )
const PJRT_Megascale_ClientContext_MegascalePort = Cvoid

struct PJRT_Megascale_ProcessesInfo
    addresses::Ptr{Cstring}
    address_sizes::Ptr{Csize_t}
    num_addresses::Csize_t
    slice_indexes::Ptr{Int32}
    num_slice_indexes::Csize_t
    per_slice_indexes::Ptr{Int32}
    num_per_slice_indexes::Csize_t
    num_devices_per_process::Int32
end

struct PJRT_Megascale_CreateMegascaleCollectives_Args
    struct_size::Csize_t
    client_context::Ptr{PJRT_Megascale_ClientContext}
    processes_info::Ptr{PJRT_Megascale_ProcessesInfo}
    dcn_topology::Cstring
    dcn_topology_size::Csize_t
    collectives::Ptr{PJRT_Collectives}
    function PJRT_Megascale_CreateMegascaleCollectives_Args(
        client_context, processes_info, dcn_topology, dcn_topology_size, collectives
    )
        return new(
            csizeof(PJRT_Megascale_CreateMegascaleCollectives_Args),
            client_context,
            processes_info,
            dcn_topology,
            dcn_topology_size,
            collectives,
        )
    end
end

csizeof(::Type{PJRT_Megascale_CreateMegascaleCollectives_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Megascale_CreateMegascaleCollectives ( PJRT_Megascale_CreateMegascaleCollectives_Args * args )
const PJRT_Megascale_CreateMegascaleCollectives = Cvoid

struct PJRT_Megascale_DeviceId_To_MegascaleId_Args
    struct_size::Csize_t
    slice_id::Int32
    per_slice_device_id::Int32
    megascale_id::Int64
    function PJRT_Megascale_DeviceId_To_MegascaleId_Args(
        slice_id, per_slice_device_id, megascale_id
    )
        return new(
            csizeof(PJRT_Megascale_DeviceId_To_MegascaleId_Args),
            slice_id,
            per_slice_device_id,
            megascale_id,
        )
    end
end

csizeof(::Type{PJRT_Megascale_DeviceId_To_MegascaleId_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Megascale_DeviceId_To_MegascaleId ( PJRT_Megascale_DeviceId_To_MegascaleId_Args * args )
const PJRT_Megascale_DeviceId_To_MegascaleId = Cvoid

struct PJRT_Megascale_MegascaleId_To_DeviceId_Args
    struct_size::Csize_t
    megascale_id::Int64
    slice_id::Int32
    per_slice_device_id::Int32
    function PJRT_Megascale_MegascaleId_To_DeviceId_Args(
        megascale_id, slice_id, per_slice_device_id
    )
        return new(
            csizeof(PJRT_Megascale_MegascaleId_To_DeviceId_Args),
            megascale_id,
            slice_id,
            per_slice_device_id,
        )
    end
end

csizeof(::Type{PJRT_Megascale_MegascaleId_To_DeviceId_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Megascale_MegascaleId_To_DeviceId ( PJRT_Megascale_MegascaleId_To_DeviceId_Args * args )
const PJRT_Megascale_MegascaleId_To_DeviceId = Cvoid

# typedef void ( * PJRT_Megascale_ErrorHandler ) ( const char * serialized_error , size_t serialized_error_size , void * user_data )
const PJRT_Megascale_ErrorHandler = Ptr{Cvoid}

struct PJRT_Megascale_RegisterErrorHandler_Args
    struct_size::Csize_t
    handler_name::Cstring
    handler_name_size::Csize_t
    handler::PJRT_Megascale_ErrorHandler
    user_data::Ptr{Cvoid}
    function PJRT_Megascale_RegisterErrorHandler_Args(
        handler_name, handler_name_size, handler, user_data
    )
        return new(
            csizeof(PJRT_Megascale_RegisterErrorHandler_Args),
            handler_name,
            handler_name_size,
            handler,
            user_data,
        )
    end
end

csizeof(::Type{PJRT_Megascale_RegisterErrorHandler_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_Megascale_RegisterErrorHandler ( PJRT_Megascale_RegisterErrorHandler_Args * args )
const PJRT_Megascale_RegisterErrorHandler = Cvoid

struct PJRT_Megascale_UnregisterErrorHandler_Args
    struct_size::Csize_t
    handler_name::Cstring
    handler_name_size::Csize_t
    function PJRT_Megascale_UnregisterErrorHandler_Args(handler_name, handler_name_size)
        return new(
            csizeof(PJRT_Megascale_UnregisterErrorHandler_Args),
            handler_name,
            handler_name_size,
        )
    end
end

csizeof(::Type{PJRT_Megascale_UnregisterErrorHandler_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Megascale_UnregisterErrorHandler ( PJRT_Megascale_UnregisterErrorHandler_Args * args )
const PJRT_Megascale_UnregisterErrorHandler = Cvoid

mutable struct PJRT_Megascale_ErrorAggregator end

mutable struct PJRT_Megascale_ErrorDigest end

struct PJRT_Megascale_ErrorAggregator_Create_Args
    struct_size::Csize_t
    app_type::Cstring
    app_type_size::Csize_t
    aggregator::Ptr{PJRT_Megascale_ErrorAggregator}
    function PJRT_Megascale_ErrorAggregator_Create_Args(app_type, app_type_size, aggregator)
        return new(
            csizeof(PJRT_Megascale_ErrorAggregator_Create_Args),
            app_type,
            app_type_size,
            aggregator,
        )
    end
end

csizeof(::Type{PJRT_Megascale_ErrorAggregator_Create_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_Megascale_ErrorAggregator_Create ( PJRT_Megascale_ErrorAggregator_Create_Args * args )
const PJRT_Megascale_ErrorAggregator_Create = Cvoid

struct PJRT_Megascale_ErrorAggregator_Delete_Args
    struct_size::Csize_t
    aggregator::Ptr{PJRT_Megascale_ErrorAggregator}
    function PJRT_Megascale_ErrorAggregator_Delete_Args(aggregator)
        return new(csizeof(PJRT_Megascale_ErrorAggregator_Delete_Args), aggregator)
    end
end

csizeof(::Type{PJRT_Megascale_ErrorAggregator_Delete_Args}) = Csize_t(16)

# typedef PJRT_Error * PJRT_Megascale_ErrorAggregator_Delete ( PJRT_Megascale_ErrorAggregator_Delete_Args * args )
const PJRT_Megascale_ErrorAggregator_Delete = Cvoid

struct PJRT_Megascale_ErrorDigest_Delete_Args
    struct_size::Csize_t
    digest::Ptr{PJRT_Megascale_ErrorDigest}
    function PJRT_Megascale_ErrorDigest_Delete_Args(digest)
        return new(csizeof(PJRT_Megascale_ErrorDigest_Delete_Args), digest)
    end
end

csizeof(::Type{PJRT_Megascale_ErrorDigest_Delete_Args}) = Csize_t(16)

# typedef PJRT_Error * PJRT_Megascale_ErrorDigest_Delete ( PJRT_Megascale_ErrorDigest_Delete_Args * args )
const PJRT_Megascale_ErrorDigest_Delete = Cvoid

struct PJRT_Megascale_ErrorAggregator_AddError_Args
    struct_size::Csize_t
    aggregator::Ptr{PJRT_Megascale_ErrorAggregator}
    worker_id::Cstring
    worker_id_size::Csize_t
    serialized_error::Cstring
    serialized_error_size::Csize_t
    function PJRT_Megascale_ErrorAggregator_AddError_Args(
        aggregator, worker_id, worker_id_size, serialized_error, serialized_error_size
    )
        return new(
            csizeof(PJRT_Megascale_ErrorAggregator_AddError_Args),
            aggregator,
            worker_id,
            worker_id_size,
            serialized_error,
            serialized_error_size,
        )
    end
end

csizeof(::Type{PJRT_Megascale_ErrorAggregator_AddError_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Megascale_ErrorAggregator_AddError ( PJRT_Megascale_ErrorAggregator_AddError_Args * args )
const PJRT_Megascale_ErrorAggregator_AddError = Cvoid

struct PJRT_Megascale_ErrorAggregator_ProcessAndShutdown_Args
    struct_size::Csize_t
    aggregator::Ptr{PJRT_Megascale_ErrorAggregator}
    digest::Ptr{PJRT_Megascale_ErrorDigest}
    function PJRT_Megascale_ErrorAggregator_ProcessAndShutdown_Args(aggregator, digest)
        return new(
            csizeof(PJRT_Megascale_ErrorAggregator_ProcessAndShutdown_Args),
            aggregator,
            digest,
        )
    end
end

csizeof(::Type{PJRT_Megascale_ErrorAggregator_ProcessAndShutdown_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Megascale_ErrorAggregator_ProcessAndShutdown ( PJRT_Megascale_ErrorAggregator_ProcessAndShutdown_Args * args )
const PJRT_Megascale_ErrorAggregator_ProcessAndShutdown = Cvoid

struct PJRT_Megascale_ErrorAggregator_LogErrorDigest_Args
    struct_size::Csize_t
    aggregator::Ptr{PJRT_Megascale_ErrorAggregator}
    digest::Ptr{PJRT_Megascale_ErrorDigest}
    function PJRT_Megascale_ErrorAggregator_LogErrorDigest_Args(aggregator, digest)
        return new(
            csizeof(PJRT_Megascale_ErrorAggregator_LogErrorDigest_Args), aggregator, digest
        )
    end
end

csizeof(::Type{PJRT_Megascale_ErrorAggregator_LogErrorDigest_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Megascale_ErrorAggregator_LogErrorDigest ( PJRT_Megascale_ErrorAggregator_LogErrorDigest_Args * args )
const PJRT_Megascale_ErrorAggregator_LogErrorDigest = Cvoid

struct PJRT_Megascale_ErrorAggregator_Size_Args
    struct_size::Csize_t
    aggregator::Ptr{PJRT_Megascale_ErrorAggregator}
    size::Csize_t
    function PJRT_Megascale_ErrorAggregator_Size_Args(aggregator, size)
        return new(csizeof(PJRT_Megascale_ErrorAggregator_Size_Args), aggregator, size)
    end
end

csizeof(::Type{PJRT_Megascale_ErrorAggregator_Size_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Megascale_ErrorAggregator_Size ( PJRT_Megascale_ErrorAggregator_Size_Args * args )
const PJRT_Megascale_ErrorAggregator_Size = Cvoid

struct PJRT_Megascale_ErrorAggregator_Active_Args
    struct_size::Csize_t
    aggregator::Ptr{PJRT_Megascale_ErrorAggregator}
    active::Bool
    function PJRT_Megascale_ErrorAggregator_Active_Args(aggregator, active)
        return new(csizeof(PJRT_Megascale_ErrorAggregator_Active_Args), aggregator, active)
    end
end

csizeof(::Type{PJRT_Megascale_ErrorAggregator_Active_Args}) = Csize_t(17)

# typedef PJRT_Error * PJRT_Megascale_ErrorAggregator_Active ( PJRT_Megascale_ErrorAggregator_Active_Args * args )
const PJRT_Megascale_ErrorAggregator_Active = Cvoid

mutable struct PJRT_Megascale_SerializedAddresses end

struct PJRT_Megascale_GetInterfaceAddressesHelper_Args
    struct_size::Csize_t
    megascale_port_name::Cstring
    megascale_port_name_size::Csize_t
    megascale_port::Int32
    interface_prefixes::Ptr{Cstring}
    interface_prefixes_sizes::Ptr{Csize_t}
    num_interface_prefixes::Csize_t
    use_all_interfaces::Bool
    limit_to_process_numa_local_interfaces::Bool
    serialized_addresses::Ptr{Cstring}
    serialized_addresses_sizes::Ptr{Csize_t}
    num_addresses::Csize_t
    addresses::Ptr{PJRT_Megascale_SerializedAddresses}
    addresses_deleter::Ptr{Cvoid}
    function PJRT_Megascale_GetInterfaceAddressesHelper_Args(
        megascale_port_name,
        megascale_port_name_size,
        megascale_port,
        interface_prefixes,
        interface_prefixes_sizes,
        num_interface_prefixes,
        use_all_interfaces,
        limit_to_process_numa_local_interfaces,
        serialized_addresses,
        serialized_addresses_sizes,
        num_addresses,
        addresses,
        addresses_deleter,
    )
        return new(
            csizeof(PJRT_Megascale_GetInterfaceAddressesHelper_Args),
            megascale_port_name,
            megascale_port_name_size,
            megascale_port,
            interface_prefixes,
            interface_prefixes_sizes,
            num_interface_prefixes,
            use_all_interfaces,
            limit_to_process_numa_local_interfaces,
            serialized_addresses,
            serialized_addresses_sizes,
            num_addresses,
            addresses,
            addresses_deleter,
        )
    end
end

csizeof(::Type{PJRT_Megascale_GetInterfaceAddressesHelper_Args}) = Csize_t(104)

# typedef PJRT_Error * PJRT_Megascale_GetInterfaceAddressesHelper ( PJRT_Megascale_GetInterfaceAddressesHelper_Args * args )
const PJRT_Megascale_GetInterfaceAddressesHelper = Cvoid

mutable struct PJRT_Megascale_SerializedError end

struct PJRT_Megascale_GetOrCreateRuntimeError_Args
    struct_size::Csize_t
    error_type::Int32
    start_time_nanos::Int64
    status_code::Int32
    status_message::Cstring
    status_message_size::Csize_t
    launch_id::Int32
    has_unrecoverable_error_type::Bool
    unrecoverable_error_type::Int32
    serialized_error::Cstring
    serialized_error_size::Csize_t
    is_new_error::Bool
    error::Ptr{PJRT_Megascale_SerializedError}
    error_deleter::Ptr{Cvoid}
    function PJRT_Megascale_GetOrCreateRuntimeError_Args(
        error_type,
        start_time_nanos,
        status_code,
        status_message,
        status_message_size,
        launch_id,
        has_unrecoverable_error_type,
        unrecoverable_error_type,
        serialized_error,
        serialized_error_size,
        is_new_error,
        error,
        error_deleter,
    )
        return new(
            csizeof(PJRT_Megascale_GetOrCreateRuntimeError_Args),
            error_type,
            start_time_nanos,
            status_code,
            status_message,
            status_message_size,
            launch_id,
            has_unrecoverable_error_type,
            unrecoverable_error_type,
            serialized_error,
            serialized_error_size,
            is_new_error,
            error,
            error_deleter,
        )
    end
end

csizeof(::Type{PJRT_Megascale_GetOrCreateRuntimeError_Args}) = Csize_t(104)

# typedef PJRT_Error * PJRT_Megascale_GetOrCreateRuntimeError ( PJRT_Megascale_GetOrCreateRuntimeError_Args * args )
const PJRT_Megascale_GetOrCreateRuntimeError = Cvoid

struct PJRT_Megascale_Extension
    base::PJRT_Extension_Base
    create_client_context_from_pjrt_client::Ptr{
        PJRT_Megascale_CreateClientContextFromPjRtClient
    }
    create_default_client_context::Ptr{PJRT_Megascale_CreateDefaultClientContext}
    delete_client_context::Ptr{PJRT_Megascale_DeleteClientContext}
    create_aot_config::Ptr{PJRT_Megascale_CreateAoTConfig}
    create_multi_slice_config::Ptr{PJRT_Megascale_CreateMultiSliceConfig}
    delete_multi_slice_config::Ptr{Cvoid}
    multi_slice_config_num_slices::Ptr{Cvoid}
    multi_slice_config_slice_id::Ptr{Cvoid}
    multi_slice_config_get_num_devices_per_slice::Ptr{Cvoid}
    multi_slice_config_serialize::Ptr{Cvoid}
    client_context_initialize::Ptr{PJRT_Megascale_ClientContext_Initialize}
    client_context_unblock_pending_work::Ptr{
        PJRT_Megascale_ClientContext_UnblockPendingWork
    }
    client_context_megascale_port::Ptr{PJRT_Megascale_ClientContext_MegascalePort}
    create_megascale_collectives::Ptr{PJRT_Megascale_CreateMegascaleCollectives}
    device_id_to_megascale_id::Ptr{PJRT_Megascale_DeviceId_To_MegascaleId}
    megascale_id_to_device_id::Ptr{PJRT_Megascale_MegascaleId_To_DeviceId}
    register_megascale_error_handler::Ptr{PJRT_Megascale_RegisterErrorHandler}
    unregister_megascale_error_handler::Ptr{PJRT_Megascale_UnregisterErrorHandler}
    error_aggregator_create::Ptr{PJRT_Megascale_ErrorAggregator_Create}
    error_aggregator_delete::Ptr{PJRT_Megascale_ErrorAggregator_Delete}
    error_digest_delete::Ptr{PJRT_Megascale_ErrorDigest_Delete}
    error_aggregator_add_error::Ptr{PJRT_Megascale_ErrorAggregator_AddError}
    error_aggregator_process_and_shutdown::Ptr{
        PJRT_Megascale_ErrorAggregator_ProcessAndShutdown
    }
    error_aggregator_log_error_digest::Ptr{PJRT_Megascale_ErrorAggregator_LogErrorDigest}
    error_aggregator_size::Ptr{PJRT_Megascale_ErrorAggregator_Size}
    error_aggregator_active::Ptr{PJRT_Megascale_ErrorAggregator_Active}
    get_interface_addresses_helper::Ptr{PJRT_Megascale_GetInterfaceAddressesHelper}
    get_or_create_runtime_error::Ptr{PJRT_Megascale_GetOrCreateRuntimeError}
end

csizeof(::Type{PJRT_Megascale_Extension}) = Csize_t(248)

mutable struct PJRT_MemoryDescription end

struct PJRT_DeviceDescription_MemoryDescriptions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    memory_descriptions::Ptr{Ptr{PJRT_MemoryDescription}}
    num_memory_descriptions::Csize_t
    default_memory_index::Csize_t
    function PJRT_DeviceDescription_MemoryDescriptions_Args(
        extension_start,
        device_description,
        memory_descriptions,
        num_memory_descriptions,
        default_memory_index,
    )
        return new(
            csizeof(PJRT_DeviceDescription_MemoryDescriptions_Args),
            extension_start,
            device_description,
            memory_descriptions,
            num_memory_descriptions,
            default_memory_index,
        )
    end
end

csizeof(::Type{PJRT_DeviceDescription_MemoryDescriptions_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_DeviceDescription_MemoryDescriptions ( PJRT_DeviceDescription_MemoryDescriptions_Args * args )
const PJRT_DeviceDescription_MemoryDescriptions = Cvoid

struct PJRT_MemoryDescription_Kind_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory_description::Ptr{PJRT_MemoryDescription}
    kind::Cstring
    kind_size::Csize_t
    kind_id::Cint
    function PJRT_MemoryDescription_Kind_Args(
        extension_start, memory_description, kind, kind_size, kind_id
    )
        return new(
            csizeof(PJRT_MemoryDescription_Kind_Args),
            extension_start,
            memory_description,
            kind,
            kind_size,
            kind_id,
        )
    end
end

csizeof(::Type{PJRT_MemoryDescription_Kind_Args}) = Csize_t(44)

# typedef PJRT_Error * PJRT_MemoryDescription_Kind ( PJRT_MemoryDescription_Kind_Args * args )
const PJRT_MemoryDescription_Kind = Cvoid

struct PJRT_MemoryDescriptions_Extension
    base::PJRT_Extension_Base
    PJRT_DeviceDescription_MemoryDescriptions::Ptr{
        PJRT_DeviceDescription_MemoryDescriptions
    }
    PJRT_MemoryDescription_Kind::Ptr{PJRT_MemoryDescription_Kind}
end

csizeof(::Type{PJRT_MemoryDescriptions_Extension}) = Csize_t(40)

mutable struct PJRT_MultiSlice_NumDevicesPerSlice end

mutable struct PJRT_MultiSlice_SerializedConfig end

struct PJRT_MultiSlice_Config_Destroy_Args
    struct_size::Csize_t
    config::Ptr{PJRT_MultiSlice_Config}
    function PJRT_MultiSlice_Config_Destroy_Args(config)
        return new(csizeof(PJRT_MultiSlice_Config_Destroy_Args), config)
    end
end

csizeof(::Type{PJRT_MultiSlice_Config_Destroy_Args}) = Csize_t(16)

# typedef PJRT_Error * PJRT_MultiSlice_Config_Destroy ( PJRT_MultiSlice_Config_Destroy_Args * args )
const PJRT_MultiSlice_Config_Destroy = Cvoid

struct PJRT_MultiSlice_Config_NumSlices_Args
    struct_size::Csize_t
    config::Ptr{PJRT_MultiSlice_Config}
    num_slices::Int32
    function PJRT_MultiSlice_Config_NumSlices_Args(config, num_slices)
        return new(csizeof(PJRT_MultiSlice_Config_NumSlices_Args), config, num_slices)
    end
end

csizeof(::Type{PJRT_MultiSlice_Config_NumSlices_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_MultiSlice_Config_NumSlices ( PJRT_MultiSlice_Config_NumSlices_Args * args )
const PJRT_MultiSlice_Config_NumSlices = Cvoid

struct PJRT_MultiSlice_Config_SliceId_Args
    struct_size::Csize_t
    config::Ptr{PJRT_MultiSlice_Config}
    slice_id::Int32
    function PJRT_MultiSlice_Config_SliceId_Args(config, slice_id)
        return new(csizeof(PJRT_MultiSlice_Config_SliceId_Args), config, slice_id)
    end
end

csizeof(::Type{PJRT_MultiSlice_Config_SliceId_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_MultiSlice_Config_SliceId ( PJRT_MultiSlice_Config_SliceId_Args * args )
const PJRT_MultiSlice_Config_SliceId = Cvoid

struct PJRT_MultiSlice_Config_NumDevicesPerSlice_Args
    struct_size::Csize_t
    config::Ptr{PJRT_MultiSlice_Config}
    num_devices_per_slice_map::Csize_t
    slice_ids::Ptr{Int32}
    num_devices::Ptr{Int32}
    devices_per_slice_map::Ptr{PJRT_MultiSlice_NumDevicesPerSlice}
    devices_per_slice_map_deleter::Ptr{Cvoid}
    function PJRT_MultiSlice_Config_NumDevicesPerSlice_Args(
        config,
        num_devices_per_slice_map,
        slice_ids,
        num_devices,
        devices_per_slice_map,
        devices_per_slice_map_deleter,
    )
        return new(
            csizeof(PJRT_MultiSlice_Config_NumDevicesPerSlice_Args),
            config,
            num_devices_per_slice_map,
            slice_ids,
            num_devices,
            devices_per_slice_map,
            devices_per_slice_map_deleter,
        )
    end
end

csizeof(::Type{PJRT_MultiSlice_Config_NumDevicesPerSlice_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_MultiSlice_Config_NumDevicesPerSlice ( PJRT_MultiSlice_Config_NumDevicesPerSlice_Args * args )
const PJRT_MultiSlice_Config_NumDevicesPerSlice = Cvoid

struct PJRT_MultiSlice_Config_Serialize_Args
    struct_size::Csize_t
    config::Ptr{PJRT_MultiSlice_Config}
    serialized::Cstring
    size::Csize_t
    serialized_config::Ptr{PJRT_MultiSlice_SerializedConfig}
    serialized_config_deleter::Ptr{Cvoid}
    function PJRT_MultiSlice_Config_Serialize_Args(
        config, serialized, size, serialized_config, serialized_config_deleter
    )
        return new(
            csizeof(PJRT_MultiSlice_Config_Serialize_Args),
            config,
            serialized,
            size,
            serialized_config,
            serialized_config_deleter,
        )
    end
end

csizeof(::Type{PJRT_MultiSlice_Config_Serialize_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_MultiSlice_Config_Serialize ( PJRT_MultiSlice_Config_Serialize_Args * args )
const PJRT_MultiSlice_Config_Serialize = Cvoid

struct PJRT_MultiSlice_Extension
    base::PJRT_Extension_Base
    config_destroy::Ptr{PJRT_MultiSlice_Config_Destroy}
    config_num_slices::Ptr{PJRT_MultiSlice_Config_NumSlices}
    config_slice_id::Ptr{PJRT_MultiSlice_Config_SliceId}
    config_num_devices_per_slice::Ptr{PJRT_MultiSlice_Config_NumDevicesPerSlice}
    config_serialize::Ptr{PJRT_MultiSlice_Config_Serialize}
end

csizeof(::Type{PJRT_MultiSlice_Extension}) = Csize_t(64)

struct PJRT_PhaseCompile_Get_Compiler_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    phase_compiler::Ptr{PJRT_PhaseCompiler}
    function PJRT_PhaseCompile_Get_Compiler_Args(extension_start, phase_compiler)
        return new(
            csizeof(PJRT_PhaseCompile_Get_Compiler_Args), extension_start, phase_compiler
        )
    end
end

csizeof(::Type{PJRT_PhaseCompile_Get_Compiler_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_PhaseCompile_Get_Compiler ( PJRT_PhaseCompile_Get_Compiler_Args * args )
const PJRT_PhaseCompile_Get_Compiler = Cvoid

struct PJRT_PhaseCompile_Destroy_Compiler_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    phase_compiler::Ptr{PJRT_PhaseCompiler}
    function PJRT_PhaseCompile_Destroy_Compiler_Args(extension_start, phase_compiler)
        return new(
            csizeof(PJRT_PhaseCompile_Destroy_Compiler_Args),
            extension_start,
            phase_compiler,
        )
    end
end

csizeof(::Type{PJRT_PhaseCompile_Destroy_Compiler_Args}) = Csize_t(24)

# typedef void PJRT_PhaseCompile_Destroy_Compiler ( PJRT_PhaseCompile_Destroy_Compiler_Args * args )
const PJRT_PhaseCompile_Destroy_Compiler = Cvoid

struct PJRT_PhaseCompile_Run_Phase_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    phase_compiler::Ptr{PJRT_PhaseCompiler}
    input_programs::Ptr{Cstring}
    input_programs_sizes::Ptr{Csize_t}
    num_input_programs::Csize_t
    phases_to_run::Ptr{Cstring}
    phases_to_run_sizes::Ptr{Csize_t}
    num_phases_to_run::Csize_t
    compile_options::Cstring
    compile_options_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    output_programs::Ptr{Cstring}
    output_programs_sizes::Ptr{Csize_t}
    num_output_programs::Csize_t
    function PJRT_PhaseCompile_Run_Phase_Args(
        extension_start,
        phase_compiler,
        input_programs,
        input_programs_sizes,
        num_input_programs,
        phases_to_run,
        phases_to_run_sizes,
        num_phases_to_run,
        compile_options,
        compile_options_size,
        topology,
        output_programs,
        output_programs_sizes,
        num_output_programs,
    )
        return new(
            csizeof(PJRT_PhaseCompile_Run_Phase_Args),
            extension_start,
            phase_compiler,
            input_programs,
            input_programs_sizes,
            num_input_programs,
            phases_to_run,
            phases_to_run_sizes,
            num_phases_to_run,
            compile_options,
            compile_options_size,
            topology,
            output_programs,
            output_programs_sizes,
            num_output_programs,
        )
    end
end

csizeof(::Type{PJRT_PhaseCompile_Run_Phase_Args}) = Csize_t(120)

# typedef PJRT_Error * PJRT_PhaseCompile_Run_Phase ( PJRT_PhaseCompile_Run_Phase_Args * args )
const PJRT_PhaseCompile_Run_Phase = Cvoid

struct PJRT_PhaseCompile_Get_PhaseNames_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    phase_compiler::Ptr{PJRT_PhaseCompiler}
    phase_names::Ptr{Cstring}
    phase_names_sizes::Ptr{Csize_t}
    num_phase_names::Csize_t
    function PJRT_PhaseCompile_Get_PhaseNames_Args(
        extension_start, phase_compiler, phase_names, phase_names_sizes, num_phase_names
    )
        return new(
            csizeof(PJRT_PhaseCompile_Get_PhaseNames_Args),
            extension_start,
            phase_compiler,
            phase_names,
            phase_names_sizes,
            num_phase_names,
        )
    end
end

csizeof(::Type{PJRT_PhaseCompile_Get_PhaseNames_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_PhaseCompile_Get_PhaseNames ( PJRT_PhaseCompile_Get_PhaseNames_Args * args )
const PJRT_PhaseCompile_Get_PhaseNames = Cvoid

struct PJRT_PhaseCompile_C_Buffers_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    char_buffers::Ptr{Cstring}
    char_buffer_sizes::Ptr{Csize_t}
    num_char_buffers::Csize_t
    function PJRT_PhaseCompile_C_Buffers_Destroy_Args(
        extension_start, char_buffers, char_buffer_sizes, num_char_buffers
    )
        return new(
            csizeof(PJRT_PhaseCompile_C_Buffers_Destroy_Args),
            extension_start,
            char_buffers,
            char_buffer_sizes,
            num_char_buffers,
        )
    end
end

csizeof(::Type{PJRT_PhaseCompile_C_Buffers_Destroy_Args}) = Csize_t(40)

# typedef void PJRT_PhaseCompile_C_Buffers_Destroy ( PJRT_PhaseCompile_C_Buffers_Destroy_Args * args )
const PJRT_PhaseCompile_C_Buffers_Destroy = Cvoid

struct PJRT_PhaseCompile_Extension
    base::PJRT_Extension_Base
    phase_compile_get_compiler::Ptr{PJRT_PhaseCompile_Get_Compiler}
    phase_compile_destroy_compiler::Ptr{PJRT_PhaseCompile_Destroy_Compiler}
    phase_compile_run_phases::Ptr{PJRT_PhaseCompile_Run_Phase}
    phase_compile_get_phase_names::Ptr{PJRT_PhaseCompile_Get_PhaseNames}
    phase_compile_c_buffers_destroy::Ptr{PJRT_PhaseCompile_C_Buffers_Destroy}
end

csizeof(::Type{PJRT_PhaseCompile_Extension}) = Csize_t(64)

mutable struct PLUGIN_Profiler end

mutable struct PLUGIN_Profiler_Error end

mutable struct PLUGIN_Profiler_ConsumeResult end

struct PLUGIN_Profiler_Error_Destroy_Args
    struct_size::Csize_t
    priv::Ptr{Cvoid}
    error::Ptr{PLUGIN_Profiler_Error}
    function PLUGIN_Profiler_Error_Destroy_Args(priv, error)
        return new(csizeof(PLUGIN_Profiler_Error_Destroy_Args), priv, error)
    end
end

csizeof(::Type{PLUGIN_Profiler_Error_Destroy_Args}) = Csize_t(24)

# typedef void PLUGIN_Profiler_Error_Destroy ( PLUGIN_Profiler_Error_Destroy_Args * args )
const PLUGIN_Profiler_Error_Destroy = Cvoid

struct PLUGIN_Profiler_Error_Message_Args
    struct_size::Csize_t
    priv::Ptr{Cvoid}
    error::Ptr{PLUGIN_Profiler_Error}
    message::Cstring
    message_size::Csize_t
    function PLUGIN_Profiler_Error_Message_Args(priv, error, message, message_size)
        return new(
            csizeof(PLUGIN_Profiler_Error_Message_Args), priv, error, message, message_size
        )
    end
end

csizeof(::Type{PLUGIN_Profiler_Error_Message_Args}) = Csize_t(40)

# typedef void PLUGIN_Profiler_Error_Message ( PLUGIN_Profiler_Error_Message_Args * args )
const PLUGIN_Profiler_Error_Message = Cvoid

struct PLUGIN_Profiler_Error_GetCode_Args
    struct_size::Csize_t
    priv::Ptr{Cvoid}
    error::Ptr{PLUGIN_Profiler_Error}
    code::Cint
    function PLUGIN_Profiler_Error_GetCode_Args(priv, error, code)
        return new(csizeof(PLUGIN_Profiler_Error_GetCode_Args), priv, error, code)
    end
end

csizeof(::Type{PLUGIN_Profiler_Error_GetCode_Args}) = Csize_t(28)

# typedef PLUGIN_Profiler_Error * PLUGIN_Profiler_Error_GetCode ( PLUGIN_Profiler_Error_GetCode_Args * args )
const PLUGIN_Profiler_Error_GetCode = Cvoid

struct PLUGIN_Profiler_Create_Args
    struct_size::Csize_t
    options::Cstring
    options_size::Csize_t
    profiler::Ptr{PLUGIN_Profiler}
    function PLUGIN_Profiler_Create_Args(options, options_size, profiler)
        return new(csizeof(PLUGIN_Profiler_Create_Args), options, options_size, profiler)
    end
end

csizeof(::Type{PLUGIN_Profiler_Create_Args}) = Csize_t(32)

# typedef PLUGIN_Profiler_Error * PLUGIN_Profiler_Create ( PLUGIN_Profiler_Create_Args * args )
const PLUGIN_Profiler_Create = Cvoid

struct PLUGIN_Profiler_Destroy_Args
    struct_size::Csize_t
    profiler::Ptr{PLUGIN_Profiler}
    function PLUGIN_Profiler_Destroy_Args(profiler)
        return new(csizeof(PLUGIN_Profiler_Destroy_Args), profiler)
    end
end

csizeof(::Type{PLUGIN_Profiler_Destroy_Args}) = Csize_t(16)

# typedef PLUGIN_Profiler_Error * PLUGIN_Profiler_Destroy ( PLUGIN_Profiler_Destroy_Args * args )
const PLUGIN_Profiler_Destroy = Cvoid

struct PLUGIN_Profiler_Start_Args
    struct_size::Csize_t
    profiler::Ptr{PLUGIN_Profiler}
    function PLUGIN_Profiler_Start_Args(profiler)
        return new(csizeof(PLUGIN_Profiler_Start_Args), profiler)
    end
end

csizeof(::Type{PLUGIN_Profiler_Start_Args}) = Csize_t(16)

# typedef PLUGIN_Profiler_Error * PLUGIN_Profiler_Start ( PLUGIN_Profiler_Start_Args * args )
const PLUGIN_Profiler_Start = Cvoid

struct PLUGIN_Profiler_Stop_Args
    struct_size::Csize_t
    profiler::Ptr{PLUGIN_Profiler}
    function PLUGIN_Profiler_Stop_Args(profiler)
        return new(csizeof(PLUGIN_Profiler_Stop_Args), profiler)
    end
end

csizeof(::Type{PLUGIN_Profiler_Stop_Args}) = Csize_t(16)

# typedef PLUGIN_Profiler_Error * PLUGIN_Profiler_Stop ( PLUGIN_Profiler_Stop_Args * args )
const PLUGIN_Profiler_Stop = Cvoid

struct PLUGIN_Profiler_CollectData_Args
    struct_size::Csize_t
    profiler::Ptr{PLUGIN_Profiler}
    buffer::Ptr{UInt8}
    buffer_size_in_bytes::Csize_t
    function PLUGIN_Profiler_CollectData_Args(profiler, buffer, buffer_size_in_bytes)
        return new(
            csizeof(PLUGIN_Profiler_CollectData_Args),
            profiler,
            buffer,
            buffer_size_in_bytes,
        )
    end
end

csizeof(::Type{PLUGIN_Profiler_CollectData_Args}) = Csize_t(32)

# typedef PLUGIN_Profiler_Error * PLUGIN_Profiler_CollectData ( PLUGIN_Profiler_CollectData_Args * args )
const PLUGIN_Profiler_CollectData = Cvoid

struct PLUGIN_Profiler_Consume_Args
    struct_size::Csize_t
    profiler::Ptr{PLUGIN_Profiler}
    result::Ptr{PLUGIN_Profiler_ConsumeResult}
    function PLUGIN_Profiler_Consume_Args(profiler, result)
        return new(csizeof(PLUGIN_Profiler_Consume_Args), profiler, result)
    end
end

csizeof(::Type{PLUGIN_Profiler_Consume_Args}) = Csize_t(24)

# typedef PLUGIN_Profiler_Error * PLUGIN_Profiler_Consume ( PLUGIN_Profiler_Consume_Args * args )
const PLUGIN_Profiler_Consume = Cvoid

struct PLUGIN_Profiler_ConsumeResult_Destroy_Args
    struct_size::Csize_t
    consume_result::Ptr{PLUGIN_Profiler_ConsumeResult}
    function PLUGIN_Profiler_ConsumeResult_Destroy_Args(consume_result)
        return new(csizeof(PLUGIN_Profiler_ConsumeResult_Destroy_Args), consume_result)
    end
end

csizeof(::Type{PLUGIN_Profiler_ConsumeResult_Destroy_Args}) = Csize_t(16)

# typedef void PLUGIN_Profiler_ConsumeResult_Destroy ( PLUGIN_Profiler_ConsumeResult_Destroy_Args * args )
const PLUGIN_Profiler_ConsumeResult_Destroy = Cvoid

struct PLUGIN_Profiler_Serialize_Args
    struct_size::Csize_t
    profiler::Ptr{PLUGIN_Profiler}
    consume_result::Ptr{PLUGIN_Profiler_ConsumeResult}
    serialized_bytes::Ptr{UInt8}
    serialized_size::Csize_t
    function PLUGIN_Profiler_Serialize_Args(
        profiler, consume_result, serialized_bytes, serialized_size
    )
        return new(
            csizeof(PLUGIN_Profiler_Serialize_Args),
            profiler,
            consume_result,
            serialized_bytes,
            serialized_size,
        )
    end
end

csizeof(::Type{PLUGIN_Profiler_Serialize_Args}) = Csize_t(40)

# typedef PLUGIN_Profiler_Error * PLUGIN_Profiler_Serialize ( PLUGIN_Profiler_Serialize_Args * args )
const PLUGIN_Profiler_Serialize = Cvoid

struct PLUGIN_Profiler_Api
    struct_size::Csize_t
    priv::Ptr{Cvoid}
    error_destroy::Ptr{PLUGIN_Profiler_Error_Destroy}
    error_message::Ptr{PLUGIN_Profiler_Error_Message}
    error_get_code::Ptr{PLUGIN_Profiler_Error_GetCode}
    create::Ptr{PLUGIN_Profiler_Create}
    destroy::Ptr{PLUGIN_Profiler_Destroy}
    start::Ptr{PLUGIN_Profiler_Start}
    stop::Ptr{PLUGIN_Profiler_Stop}
    collect_data::Ptr{PLUGIN_Profiler_CollectData}
    consume::Ptr{PLUGIN_Profiler_Consume}
    consume_result_destroy::Ptr{PLUGIN_Profiler_ConsumeResult_Destroy}
    serialize::Ptr{PLUGIN_Profiler_Serialize}
end

csizeof(::Type{PLUGIN_Profiler_Api}) = Csize_t(104)

struct PJRT_Profiler_Extension
    base::PJRT_Extension_Base
    profiler_api::Ptr{PLUGIN_Profiler_Api}
    traceme_context_id::Int64
end

csizeof(::Type{PJRT_Profiler_Extension}) = Csize_t(40)

struct PJRT_RawBuffer_FunctionTable
    struct_size::Csize_t
    instance_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    inc_ref::Ptr{Cvoid}
    dec_ref::Ptr{Cvoid}
    get_on_device_size_in_bytes::Ptr{Cvoid}
    get_memory_space::Ptr{Cvoid}
    get_host_pointer::Ptr{Cvoid}
    copy_raw_host_to_device_and_return_event::Ptr{Cvoid}
    copy_raw_device_to_host_and_return_event::Ptr{Cvoid}
    opaque_device_memory_data_pointer::Ptr{Cvoid}
    make_allocation_ready_event::Ptr{Cvoid}
    get_raw_buffer_async_value::Ptr{Cvoid}
    is_mutable::Ptr{Cvoid}
    slice::Ptr{Cvoid}
    schedule_copy_to::Ptr{Cvoid}
end

struct PJRT_RawBuffer
    vtable::Ptr{PJRT_RawBuffer_FunctionTable}
end

csizeof(::Type{PJRT_RawBuffer_FunctionTable}) = Csize_t(128)

csizeof(::Type{PJRT_RawBuffer}) = Csize_t(8)

struct PJRT_RawBuffer_CreateRawAliasOfBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    raw_buffer::Ptr{PJRT_RawBuffer}
    function PJRT_RawBuffer_CreateRawAliasOfBuffer_Args(extension_start, buffer, raw_buffer)
        return new(
            csizeof(PJRT_RawBuffer_CreateRawAliasOfBuffer_Args),
            extension_start,
            buffer,
            raw_buffer,
        )
    end
end

csizeof(::Type{PJRT_RawBuffer_CreateRawAliasOfBuffer_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_RawBuffer_CreateRawAliasOfBuffer ( PJRT_RawBuffer_CreateRawAliasOfBuffer_Args * args )
const PJRT_RawBuffer_CreateRawAliasOfBuffer = Cvoid

struct PJRT_RawBuffer_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_RawBuffer}
    function PJRT_RawBuffer_Destroy_Args(extension_start, buffer)
        return new(csizeof(PJRT_RawBuffer_Destroy_Args), extension_start, buffer)
    end
end

csizeof(::Type{PJRT_RawBuffer_Destroy_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_RawBuffer_Destroy ( PJRT_RawBuffer_Destroy_Args * args )
const PJRT_RawBuffer_Destroy = Cvoid

struct PJRT_RawBuffer_GetHostPointer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_RawBuffer}
    host_pointer::Ptr{Cvoid}
    function PJRT_RawBuffer_GetHostPointer_Args(extension_start, buffer, host_pointer)
        return new(
            csizeof(PJRT_RawBuffer_GetHostPointer_Args),
            extension_start,
            buffer,
            host_pointer,
        )
    end
end

csizeof(::Type{PJRT_RawBuffer_GetHostPointer_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_RawBuffer_GetHostPointer ( PJRT_RawBuffer_GetHostPointer_Args * args )
const PJRT_RawBuffer_GetHostPointer = Cvoid

struct PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_RawBuffer}
    on_device_size_in_bytes::Csize_t
    function PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args(
        extension_start, buffer, on_device_size_in_bytes
    )
        return new(
            csizeof(PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args),
            extension_start,
            buffer,
            on_device_size_in_bytes,
        )
    end
end

csizeof(::Type{PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_RawBuffer_GetOnDeviceSizeInBytes ( PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args * args )
const PJRT_RawBuffer_GetOnDeviceSizeInBytes = Cvoid

struct PJRT_RawBuffer_GetMemorySpace_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_RawBuffer}
    memory_space::Ptr{PJRT_Memory}
    function PJRT_RawBuffer_GetMemorySpace_Args(extension_start, buffer, memory_space)
        return new(
            csizeof(PJRT_RawBuffer_GetMemorySpace_Args),
            extension_start,
            buffer,
            memory_space,
        )
    end
end

csizeof(::Type{PJRT_RawBuffer_GetMemorySpace_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_RawBuffer_GetMemorySpace ( PJRT_RawBuffer_GetMemorySpace_Args * args )
const PJRT_RawBuffer_GetMemorySpace = Cvoid

struct PJRT_RawBuffer_CopyRawDeviceToHost_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_RawBuffer}
    dst::Ptr{Cvoid}
    offset::Int64
    transfer_size::Int64
    event::Ptr{PJRT_Event}
    function PJRT_RawBuffer_CopyRawDeviceToHost_Args(
        extension_start, buffer, dst, offset, transfer_size, event
    )
        return new(
            csizeof(PJRT_RawBuffer_CopyRawDeviceToHost_Args),
            extension_start,
            buffer,
            dst,
            offset,
            transfer_size,
            event,
        )
    end
end

csizeof(::Type{PJRT_RawBuffer_CopyRawDeviceToHost_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_RawBuffer_CopyRawDeviceToHost ( PJRT_RawBuffer_CopyRawDeviceToHost_Args * args )
const PJRT_RawBuffer_CopyRawDeviceToHost = Cvoid

struct PJRT_RawBuffer_CopyRawHostToDevice_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_RawBuffer}
    src::Ptr{Cvoid}
    offset::Int64
    transfer_size::Int64
    event::Ptr{PJRT_Event}
    function PJRT_RawBuffer_CopyRawHostToDevice_Args(
        extension_start, buffer, src, offset, transfer_size, event
    )
        return new(
            csizeof(PJRT_RawBuffer_CopyRawHostToDevice_Args),
            extension_start,
            buffer,
            src,
            offset,
            transfer_size,
            event,
        )
    end
end

csizeof(::Type{PJRT_RawBuffer_CopyRawHostToDevice_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_RawBuffer_CopyRawHostToDevice ( PJRT_RawBuffer_CopyRawHostToDevice_Args * args )
const PJRT_RawBuffer_CopyRawHostToDevice = Cvoid

struct PJRT_RawBuffer_Extension
    base::PJRT_Extension_Base
    PJRT_RawBuffer_CreateRawAliasOfBuffer::Ptr{PJRT_RawBuffer_CreateRawAliasOfBuffer}
    PJRT_RawBuffer_Destroy::Ptr{PJRT_RawBuffer_Destroy}
    PJRT_RawBuffer_GetOnDeviceSizeInBytes::Ptr{PJRT_RawBuffer_GetOnDeviceSizeInBytes}
    PJRT_RawBuffer_GetMemorySpace::Ptr{PJRT_RawBuffer_GetMemorySpace}
    PJRT_RawBuffer_CopyRawHostToDevice::Ptr{PJRT_RawBuffer_CopyRawHostToDevice}
    PJRT_RawBuffer_CopyRawDeviceToHost::Ptr{PJRT_RawBuffer_CopyRawDeviceToHost}
    PJRT_RawBuffer_GetHostPointer::Ptr{PJRT_RawBuffer_GetHostPointer}
end

csizeof(::Type{PJRT_RawBuffer_Extension}) = Csize_t(80)

struct PJRT_Shardings_PJRT_Executable_ParameterShardings_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_parameters::Csize_t
    shardings::Ptr{Cstring}
    sharding_sizes::Ptr{Csize_t}
    function PJRT_Shardings_PJRT_Executable_ParameterShardings_Args(
        extension_start, executable, num_parameters, shardings, sharding_sizes
    )
        return new(
            csizeof(PJRT_Shardings_PJRT_Executable_ParameterShardings_Args),
            extension_start,
            executable,
            num_parameters,
            shardings,
            sharding_sizes,
        )
    end
end

csizeof(::Type{PJRT_Shardings_PJRT_Executable_ParameterShardings_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Shardings_PJRT_Executable_ParameterShardings ( PJRT_Shardings_PJRT_Executable_ParameterShardings_Args * args )
const PJRT_Shardings_PJRT_Executable_ParameterShardings = Cvoid

struct PJRT_Shardings_PJRT_Executable_OutputShardings_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_outputs::Csize_t
    shardings::Ptr{Cstring}
    sharding_sizes::Ptr{Csize_t}
    function PJRT_Shardings_PJRT_Executable_OutputShardings_Args(
        extension_start, executable, num_outputs, shardings, sharding_sizes
    )
        return new(
            csizeof(PJRT_Shardings_PJRT_Executable_OutputShardings_Args),
            extension_start,
            executable,
            num_outputs,
            shardings,
            sharding_sizes,
        )
    end
end

csizeof(::Type{PJRT_Shardings_PJRT_Executable_OutputShardings_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_Shardings_PJRT_Executable_OutputShardings ( PJRT_Shardings_PJRT_Executable_OutputShardings_Args * args )
const PJRT_Shardings_PJRT_Executable_OutputShardings = Cvoid

struct PJRT_Shardings_Extension
    base::PJRT_Extension_Base
    PJRT_Shardings_PJRT_Executable_ParameterShardings::Ptr{
        PJRT_Shardings_PJRT_Executable_ParameterShardings
    }
    PJRT_Shardings_PJRT_Executable_OutputShardings::Ptr{
        PJRT_Shardings_PJRT_Executable_OutputShardings
    }
end

csizeof(::Type{PJRT_Shardings_Extension}) = Csize_t(40)

struct PJRT_Get_Stream_For_External_Ready_Events_Args
    struct_size::Csize_t
    device::Ptr{PJRT_Device}
    stream::Cptrdiff_t
    function PJRT_Get_Stream_For_External_Ready_Events_Args(device, stream)
        return new(csizeof(PJRT_Get_Stream_For_External_Ready_Events_Args), device, stream)
    end
end

csizeof(::Type{PJRT_Get_Stream_For_External_Ready_Events_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Get_Stream_For_External_Ready_Events ( PJRT_Get_Stream_For_External_Ready_Events_Args * args )
const PJRT_Get_Stream_For_External_Ready_Events = Cvoid

struct PJRT_Wait_Until_Buffer_Ready_On_Stream_Args
    struct_size::Csize_t
    stream::Cptrdiff_t
    buffer::Ptr{PJRT_Buffer}
    function PJRT_Wait_Until_Buffer_Ready_On_Stream_Args(stream, buffer)
        return new(csizeof(PJRT_Wait_Until_Buffer_Ready_On_Stream_Args), stream, buffer)
    end
end

csizeof(::Type{PJRT_Wait_Until_Buffer_Ready_On_Stream_Args}) = Csize_t(24)

# typedef PJRT_Error * PJRT_Wait_Until_Buffer_Ready_On_Stream ( PJRT_Wait_Until_Buffer_Ready_On_Stream_Args * args )
const PJRT_Wait_Until_Buffer_Ready_On_Stream = Cvoid

struct PJRT_Stream_Extension
    base::PJRT_Extension_Base
    get_stream::Ptr{PJRT_Get_Stream_For_External_Ready_Events}
    wait_stream::Ptr{PJRT_Wait_Until_Buffer_Ready_On_Stream}
end

csizeof(::Type{PJRT_Stream_Extension}) = Csize_t(40)

mutable struct PJRT_TpuExecutable_TargetArguments end

mutable struct PJRT_TpuExecutable_CompiledMemoryStats end

mutable struct PJRT_TpuExecutable_CompilationEnvironments end

mutable struct PJRT_TpuExecutable_CostAnalysisResult end

struct PJRT_TpuExecutable_GetTargetArguments_Args
    struct_size::Csize_t
    serialized_executable::Cstring
    serialized_executable_size::Csize_t
    target_arguments::Cstring
    target_arguments_size::Csize_t
    target_arguments_ptr::Ptr{PJRT_TpuExecutable_TargetArguments}
    target_arguments_deleter::Ptr{Cvoid}
    function PJRT_TpuExecutable_GetTargetArguments_Args(
        serialized_executable,
        serialized_executable_size,
        target_arguments,
        target_arguments_size,
        target_arguments_ptr,
        target_arguments_deleter,
    )
        return new(
            csizeof(PJRT_TpuExecutable_GetTargetArguments_Args),
            serialized_executable,
            serialized_executable_size,
            target_arguments,
            target_arguments_size,
            target_arguments_ptr,
            target_arguments_deleter,
        )
    end
end

csizeof(::Type{PJRT_TpuExecutable_GetTargetArguments_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_TpuExecutable_GetTargetArguments ( PJRT_TpuExecutable_GetTargetArguments_Args * args )
const PJRT_TpuExecutable_GetTargetArguments = Cvoid

mutable struct PJRT_TpuExecutable_HloModuleWithConfig end

struct PJRT_TpuExecutable_GetHloModuleWithConfig_Args
    struct_size::Csize_t
    serialized_executable::Cstring
    serialized_executable_size::Csize_t
    hlo_module_with_config::Cstring
    hlo_module_with_config_size::Csize_t
    hlo_module_with_config_ptr::Ptr{PJRT_TpuExecutable_HloModuleWithConfig}
    hlo_module_with_config_deleter::Ptr{Cvoid}
    function PJRT_TpuExecutable_GetHloModuleWithConfig_Args(
        serialized_executable,
        serialized_executable_size,
        hlo_module_with_config,
        hlo_module_with_config_size,
        hlo_module_with_config_ptr,
        hlo_module_with_config_deleter,
    )
        return new(
            csizeof(PJRT_TpuExecutable_GetHloModuleWithConfig_Args),
            serialized_executable,
            serialized_executable_size,
            hlo_module_with_config,
            hlo_module_with_config_size,
            hlo_module_with_config_ptr,
            hlo_module_with_config_deleter,
        )
    end
end

csizeof(::Type{PJRT_TpuExecutable_GetHloModuleWithConfig_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_TpuExecutable_GetHloModuleWithConfig ( PJRT_TpuExecutable_GetHloModuleWithConfig_Args * args )
const PJRT_TpuExecutable_GetHloModuleWithConfig = Cvoid

struct PJRT_TpuExecutable_RunHloCostAnalysis_Args
    struct_size::Csize_t
    target_arguments::Cstring
    target_arguments_size::Csize_t
    hlo_module::Ptr{Cstring}
    hlo_module_size::Ptr{Csize_t}
    num_hlo_modules::Csize_t
    analysis_results::Ptr{PJRT_NamedValue}
    num_analysis_results::Csize_t
    analysis_results_ptr::Ptr{PJRT_TpuExecutable_CostAnalysisResult}
    analysis_results_deleter::Ptr{Cvoid}
    function PJRT_TpuExecutable_RunHloCostAnalysis_Args(
        target_arguments,
        target_arguments_size,
        hlo_module,
        hlo_module_size,
        num_hlo_modules,
        analysis_results,
        num_analysis_results,
        analysis_results_ptr,
        analysis_results_deleter,
    )
        return new(
            csizeof(PJRT_TpuExecutable_RunHloCostAnalysis_Args),
            target_arguments,
            target_arguments_size,
            hlo_module,
            hlo_module_size,
            num_hlo_modules,
            analysis_results,
            num_analysis_results,
            analysis_results_ptr,
            analysis_results_deleter,
        )
    end
end

csizeof(::Type{PJRT_TpuExecutable_RunHloCostAnalysis_Args}) = Csize_t(80)

# typedef PJRT_Error * PJRT_TpuExecutable_RunHloCostAnalysis ( PJRT_TpuExecutable_RunHloCostAnalysis_Args * args )
const PJRT_TpuExecutable_RunHloCostAnalysis = Cvoid

struct PJRT_TpuExecutable_GetCompiledMemoryStats_Args
    struct_size::Csize_t
    serialized_executable::Cstring
    serialized_executable_size::Csize_t
    compiled_memory_stats::Cstring
    compiled_memory_stats_size::Csize_t
    compiled_memory_stats_ptr::Ptr{PJRT_TpuExecutable_CompiledMemoryStats}
    compiled_memory_stats_deleter::Ptr{Cvoid}
    function PJRT_TpuExecutable_GetCompiledMemoryStats_Args(
        serialized_executable,
        serialized_executable_size,
        compiled_memory_stats,
        compiled_memory_stats_size,
        compiled_memory_stats_ptr,
        compiled_memory_stats_deleter,
    )
        return new(
            csizeof(PJRT_TpuExecutable_GetCompiledMemoryStats_Args),
            serialized_executable,
            serialized_executable_size,
            compiled_memory_stats,
            compiled_memory_stats_size,
            compiled_memory_stats_ptr,
            compiled_memory_stats_deleter,
        )
    end
end

csizeof(::Type{PJRT_TpuExecutable_GetCompiledMemoryStats_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_TpuExecutable_GetCompiledMemoryStats ( PJRT_TpuExecutable_GetCompiledMemoryStats_Args * args )
const PJRT_TpuExecutable_GetCompiledMemoryStats = Cvoid

struct PJRT_TpuExecutable_SetTpuCompilationEnv_Args
    struct_size::Csize_t
    serialized_compilation_environments::Cstring
    serialized_compilation_environments_size::Csize_t
    result_compilation_environments::Cstring
    result_compilation_environments_size::Csize_t
    result_compilation_environments_ptr::Ptr{PJRT_TpuExecutable_CompilationEnvironments}
    result_compilation_environments_deleter::Ptr{Cvoid}
    function PJRT_TpuExecutable_SetTpuCompilationEnv_Args(
        serialized_compilation_environments,
        serialized_compilation_environments_size,
        result_compilation_environments,
        result_compilation_environments_size,
        result_compilation_environments_ptr,
        result_compilation_environments_deleter,
    )
        return new(
            csizeof(PJRT_TpuExecutable_SetTpuCompilationEnv_Args),
            serialized_compilation_environments,
            serialized_compilation_environments_size,
            result_compilation_environments,
            result_compilation_environments_size,
            result_compilation_environments_ptr,
            result_compilation_environments_deleter,
        )
    end
end

csizeof(::Type{PJRT_TpuExecutable_SetTpuCompilationEnv_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_TpuExecutable_SetTpuCompilationEnv ( PJRT_TpuExecutable_SetTpuCompilationEnv_Args * args )
const PJRT_TpuExecutable_SetTpuCompilationEnv = Cvoid

mutable struct PJRT_TpuExecutable_FieldValue end

struct PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString_Args
    struct_size::Csize_t
    serialized_compilation_environments::Cstring
    serialized_compilation_environments_size::Csize_t
    field_name::Cstring
    field_name_size::Csize_t
    field_value::Cstring
    field_value_size::Csize_t
    field_value_ptr::Ptr{PJRT_TpuExecutable_FieldValue}
    field_value_deleter::Ptr{Cvoid}
    function PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString_Args(
        serialized_compilation_environments,
        serialized_compilation_environments_size,
        field_name,
        field_name_size,
        field_value,
        field_value_size,
        field_value_ptr,
        field_value_deleter,
    )
        return new(
            csizeof(PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString_Args),
            serialized_compilation_environments,
            serialized_compilation_environments_size,
            field_name,
            field_name_size,
            field_value,
            field_value_size,
            field_value_ptr,
            field_value_deleter,
        )
    end
end

csizeof(::Type{PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString_Args}) = Csize_t(72)

# typedef PJRT_Error * PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString ( PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString_Args * args )
const PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString = Cvoid

struct PJRT_TpuExecutable_IsTpuPredeterminedError_Args
    struct_size::Csize_t
    serialized_status::Cstring
    serialized_status_size::Csize_t
    is_predetermined::Bool
    function PJRT_TpuExecutable_IsTpuPredeterminedError_Args(
        serialized_status, serialized_status_size, is_predetermined
    )
        return new(
            csizeof(PJRT_TpuExecutable_IsTpuPredeterminedError_Args),
            serialized_status,
            serialized_status_size,
            is_predetermined,
        )
    end
end

csizeof(::Type{PJRT_TpuExecutable_IsTpuPredeterminedError_Args}) = Csize_t(25)

# typedef PJRT_Error * PJRT_TpuExecutable_IsTpuPredeterminedError ( PJRT_TpuExecutable_IsTpuPredeterminedError_Args * args )
const PJRT_TpuExecutable_IsTpuPredeterminedError = Cvoid

struct PJRT_TpuExecutable_IsTpuCancellationPlaceholderError_Args
    struct_size::Csize_t
    serialized_status::Cstring
    serialized_status_size::Csize_t
    is_cancellation_placeholder::Bool
    function PJRT_TpuExecutable_IsTpuCancellationPlaceholderError_Args(
        serialized_status, serialized_status_size, is_cancellation_placeholder
    )
        return new(
            csizeof(PJRT_TpuExecutable_IsTpuCancellationPlaceholderError_Args),
            serialized_status,
            serialized_status_size,
            is_cancellation_placeholder,
        )
    end
end

csizeof(::Type{PJRT_TpuExecutable_IsTpuCancellationPlaceholderError_Args}) = Csize_t(25)

# typedef PJRT_Error * PJRT_TpuExecutable_IsTpuCancellationPlaceholderError ( PJRT_TpuExecutable_IsTpuCancellationPlaceholderError_Args * args )
const PJRT_TpuExecutable_IsTpuCancellationPlaceholderError = Cvoid

struct PJRT_TpuExecutable_Extension
    base::PJRT_Extension_Base
    get_target_arguments::Ptr{PJRT_TpuExecutable_GetTargetArguments}
    get_hlo_module_with_config::Ptr{PJRT_TpuExecutable_GetHloModuleWithConfig}
    get_core_program_abi_version::Ptr{Cvoid}
    get_compiled_memory_stats::Ptr{PJRT_TpuExecutable_GetCompiledMemoryStats}
    run_hlo_cost_analysis::Ptr{PJRT_TpuExecutable_RunHloCostAnalysis}
    set_tpu_compilation_env::Ptr{PJRT_TpuExecutable_SetTpuCompilationEnv}
    get_tpu_compilation_env_field_as_string::Ptr{
        PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString
    }
    is_tpu_predetermined_error::Ptr{PJRT_TpuExecutable_IsTpuPredeterminedError}
    is_tpu_cancellation_placeholder_error::Ptr{
        PJRT_TpuExecutable_IsTpuCancellationPlaceholderError
    }
end

csizeof(::Type{PJRT_TpuExecutable_Extension}) = Csize_t(96)

struct PJRT_TpuTopology_Subslice_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chips_per_host_bounds::Ptr{Int32}
    chips_per_host_bounds_num_dims::Csize_t
    host_bounds::Ptr{Int32}
    host_bounds_num_dims::Csize_t
    subslice_topology::Ptr{PJRT_TopologyDescription}
    function PJRT_TpuTopology_Subslice_Args(
        topology,
        chips_per_host_bounds,
        chips_per_host_bounds_num_dims,
        host_bounds,
        host_bounds_num_dims,
        subslice_topology,
    )
        return new(
            csizeof(PJRT_TpuTopology_Subslice_Args),
            topology,
            chips_per_host_bounds,
            chips_per_host_bounds_num_dims,
            host_bounds,
            host_bounds_num_dims,
            subslice_topology,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_Subslice_Args}) = Csize_t(56)

# typedef PJRT_Error * PJRT_TpuTopology_Subslice ( PJRT_TpuTopology_Subslice_Args * args )
const PJRT_TpuTopology_Subslice = Cvoid

struct PJRT_TpuTopology_IsSubsliceTopology_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    is_subslice_topology::Bool
    function PJRT_TpuTopology_IsSubsliceTopology_Args(topology, is_subslice_topology)
        return new(
            csizeof(PJRT_TpuTopology_IsSubsliceTopology_Args),
            topology,
            is_subslice_topology,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_IsSubsliceTopology_Args}) = Csize_t(17)

# typedef PJRT_Error * PJRT_TpuTopology_IsSubsliceTopology ( PJRT_TpuTopology_IsSubsliceTopology_Args * args )
const PJRT_TpuTopology_IsSubsliceTopology = Cvoid

struct PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args
    struct_size::Csize_t
    client_topology::Ptr{PJRT_TopologyDescription}
    subslice_topology::Ptr{PJRT_TopologyDescription}
    subslice_origin::Ptr{Int32}
    subslice_origin_dim_num::Csize_t
    full_device_id::Int32
    subslice_device_id::Int32
    function PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args(
        client_topology,
        subslice_topology,
        subslice_origin,
        subslice_origin_dim_num,
        full_device_id,
        subslice_device_id,
    )
        return new(
            csizeof(PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args),
            client_topology,
            subslice_topology,
            subslice_origin,
            subslice_origin_dim_num,
            full_device_id,
            subslice_device_id,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId ( PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args * args )
const PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId = Cvoid

struct PJRT_TpuTopology_ReplaceHostBounds_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    host_bounds::Ptr{Int32}
    host_bounds_dim_num::Csize_t
    new_topology::Ptr{PJRT_TopologyDescription}
    function PJRT_TpuTopology_ReplaceHostBounds_Args(
        topology, host_bounds, host_bounds_dim_num, new_topology
    )
        return new(
            csizeof(PJRT_TpuTopology_ReplaceHostBounds_Args),
            topology,
            host_bounds,
            host_bounds_dim_num,
            new_topology,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ReplaceHostBounds_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TpuTopology_ReplaceHostBounds ( PJRT_TpuTopology_ReplaceHostBounds_Args * args )
const PJRT_TpuTopology_ReplaceHostBounds = Cvoid

struct PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    is_enhanced_barrier_enabled::Bool
    function PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args(
        topology, is_enhanced_barrier_enabled
    )
        return new(
            csizeof(PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args),
            topology,
            is_enhanced_barrier_enabled,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args}) = Csize_t(17)

# typedef PJRT_Error * PJRT_TpuTopology_IsEnhancedBarrierEnabled ( PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args * args )
const PJRT_TpuTopology_IsEnhancedBarrierEnabled = Cvoid

struct PJRT_TpuTopology_HasLimitedIciConnectivity_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    has_limited_ici_connectivity::Bool
    function PJRT_TpuTopology_HasLimitedIciConnectivity_Args(
        topology, has_limited_ici_connectivity
    )
        return new(
            csizeof(PJRT_TpuTopology_HasLimitedIciConnectivity_Args),
            topology,
            has_limited_ici_connectivity,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_HasLimitedIciConnectivity_Args}) = Csize_t(17)

# typedef PJRT_Error * PJRT_TpuTopology_HasLimitedIciConnectivity ( PJRT_TpuTopology_HasLimitedIciConnectivity_Args * args )
const PJRT_TpuTopology_HasLimitedIciConnectivity = Cvoid

struct PJRT_TpuTopology_IsReachableOverLimitedIci_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    source_chip_id::Int32
    dest_chip_id::Int32
    is_reachable_over_limited_ici::Bool
    function PJRT_TpuTopology_IsReachableOverLimitedIci_Args(
        topology, source_chip_id, dest_chip_id, is_reachable_over_limited_ici
    )
        return new(
            csizeof(PJRT_TpuTopology_IsReachableOverLimitedIci_Args),
            topology,
            source_chip_id,
            dest_chip_id,
            is_reachable_over_limited_ici,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_IsReachableOverLimitedIci_Args}) = Csize_t(25)

# typedef PJRT_Error * PJRT_TpuTopology_IsReachableOverLimitedIci ( PJRT_TpuTopology_IsReachableOverLimitedIci_Args * args )
const PJRT_TpuTopology_IsReachableOverLimitedIci = Cvoid

struct PJRT_TpuTopology_ProcessCount_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    process_count::Int32
    function PJRT_TpuTopology_ProcessCount_Args(topology, process_count)
        return new(csizeof(PJRT_TpuTopology_ProcessCount_Args), topology, process_count)
    end
end

csizeof(::Type{PJRT_TpuTopology_ProcessCount_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_TpuTopology_ProcessCount ( PJRT_TpuTopology_ProcessCount_Args * args )
const PJRT_TpuTopology_ProcessCount = Cvoid

struct PJRT_TpuTopology_ChipsPerProcess_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chips_per_process::Int32
    function PJRT_TpuTopology_ChipsPerProcess_Args(topology, chips_per_process)
        return new(
            csizeof(PJRT_TpuTopology_ChipsPerProcess_Args), topology, chips_per_process
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ChipsPerProcess_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_TpuTopology_ChipsPerProcess ( PJRT_TpuTopology_ChipsPerProcess_Args * args )
const PJRT_TpuTopology_ChipsPerProcess = Cvoid

struct PJRT_TpuTopology_CoreCountPerChip_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    core_count_of_default_type_per_chip::Int32
    function PJRT_TpuTopology_CoreCountPerChip_Args(
        topology, core_count_of_default_type_per_chip
    )
        return new(
            csizeof(PJRT_TpuTopology_CoreCountPerChip_Args),
            topology,
            core_count_of_default_type_per_chip,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_CoreCountPerChip_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_TpuTopology_CoreCountPerChip ( PJRT_TpuTopology_CoreCountPerChip_Args * args )
const PJRT_TpuTopology_CoreCountPerChip = Cvoid

struct PJRT_TpuTopology_ChipCount_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chip_count::Int32
    function PJRT_TpuTopology_ChipCount_Args(topology, chip_count)
        return new(csizeof(PJRT_TpuTopology_ChipCount_Args), topology, chip_count)
    end
end

csizeof(::Type{PJRT_TpuTopology_ChipCount_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_TpuTopology_ChipCount ( PJRT_TpuTopology_ChipCount_Args * args )
const PJRT_TpuTopology_ChipCount = Cvoid

struct PJRT_TpuTopology_CoreCount_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    core_count_of_default_type::Int32
    function PJRT_TpuTopology_CoreCount_Args(topology, core_count_of_default_type)
        return new(
            csizeof(PJRT_TpuTopology_CoreCount_Args), topology, core_count_of_default_type
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_CoreCount_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_TpuTopology_CoreCount ( PJRT_TpuTopology_CoreCount_Args * args )
const PJRT_TpuTopology_CoreCount = Cvoid

struct PJRT_TpuTopology_LogiDeviceCount_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    logical_device_count_of_default_type::Int32
    function PJRT_TpuTopology_LogiDeviceCount_Args(
        topology, logical_device_count_of_default_type
    )
        return new(
            csizeof(PJRT_TpuTopology_LogiDeviceCount_Args),
            topology,
            logical_device_count_of_default_type,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_LogiDeviceCount_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_TpuTopology_LogiDeviceCount ( PJRT_TpuTopology_LogiDeviceCount_Args * args )
const PJRT_TpuTopology_LogiDeviceCount = Cvoid

struct PJRT_TpuTopology_LogiDeviceCountPerProcess_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    logical_device_count_of_default_type_per_process::Int32
    function PJRT_TpuTopology_LogiDeviceCountPerProcess_Args(
        topology, logical_device_count_of_default_type_per_process
    )
        return new(
            csizeof(PJRT_TpuTopology_LogiDeviceCountPerProcess_Args),
            topology,
            logical_device_count_of_default_type_per_process,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_LogiDeviceCountPerProcess_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_TpuTopology_LogiDeviceCountPerProcess ( PJRT_TpuTopology_LogiDeviceCountPerProcess_Args * args )
const PJRT_TpuTopology_LogiDeviceCountPerProcess = Cvoid

struct PJRT_TpuTopology_LogiDeviceCountPerChip_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    logical_device_count_of_default_type_per_chip::Int32
    function PJRT_TpuTopology_LogiDeviceCountPerChip_Args(
        topology, logical_device_count_of_default_type_per_chip
    )
        return new(
            csizeof(PJRT_TpuTopology_LogiDeviceCountPerChip_Args),
            topology,
            logical_device_count_of_default_type_per_chip,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_LogiDeviceCountPerChip_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_TpuTopology_LogiDeviceCountPerChip ( PJRT_TpuTopology_LogiDeviceCountPerChip_Args * args )
const PJRT_TpuTopology_LogiDeviceCountPerChip = Cvoid

struct PJRT_TpuTopology_CoreCountPerProcess_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    core_count_of_default_type_per_process::Int32
    function PJRT_TpuTopology_CoreCountPerProcess_Args(
        topology, core_count_of_default_type_per_process
    )
        return new(
            csizeof(PJRT_TpuTopology_CoreCountPerProcess_Args),
            topology,
            core_count_of_default_type_per_process,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_CoreCountPerProcess_Args}) = Csize_t(20)

# typedef PJRT_Error * PJRT_TpuTopology_CoreCountPerProcess ( PJRT_TpuTopology_CoreCountPerProcess_Args * args )
const PJRT_TpuTopology_CoreCountPerProcess = Cvoid

struct PJRT_TpuTopology_ProcessIds_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    max_process_ids::Int32
    process_ids::Ptr{Int32}
    num_process_ids::Csize_t
    function PJRT_TpuTopology_ProcessIds_Args(
        topology, max_process_ids, process_ids, num_process_ids
    )
        return new(
            csizeof(PJRT_TpuTopology_ProcessIds_Args),
            topology,
            max_process_ids,
            process_ids,
            num_process_ids,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ProcessIds_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TpuTopology_ProcessIds ( PJRT_TpuTopology_ProcessIds_Args * args )
const PJRT_TpuTopology_ProcessIds = Cvoid

struct PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    process_id::Int32
    max_logical_device_ids::Int32
    logical_device_of_default_type_ids::Ptr{Int32}
    num_logical_device_ids::Csize_t
    function PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args(
        topology,
        process_id,
        max_logical_device_ids,
        logical_device_of_default_type_ids,
        num_logical_device_ids,
    )
        return new(
            csizeof(PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args),
            topology,
            process_id,
            max_logical_device_ids,
            logical_device_of_default_type_ids,
            num_logical_device_ids,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TpuTopology_LogiDeviceIdsOnProcess ( PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args * args )
const PJRT_TpuTopology_LogiDeviceIdsOnProcess = Cvoid

struct PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chip_id::Int32
    process_id::Int32
    index_on_process::Int32
    function PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args(
        topology, chip_id, process_id, index_on_process
    )
        return new(
            csizeof(PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args),
            topology,
            chip_id,
            process_id,
            index_on_process,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args}) = Csize_t(28)

# typedef PJRT_Error * PJRT_TpuTopology_ProcIdAndIdxOnProcForChip ( PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args * args )
const PJRT_TpuTopology_ProcIdAndIdxOnProcForChip = Cvoid

struct PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    device_id::Int32
    process_id::Int32
    index_on_process::Int32
    function PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args(
        topology, device_id, process_id, index_on_process
    )
        return new(
            csizeof(PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args),
            topology,
            device_id,
            process_id,
            index_on_process,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args}) = Csize_t(28)

# typedef PJRT_Error * PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice ( PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args * args )
const PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice = Cvoid

struct PJRT_TpuTopology_ProcessCoordFromId_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    process_id::Int32
    coords_max_dims::Csize_t
    coords::Ptr{Int32}
    coords_num_dims::Csize_t
    function PJRT_TpuTopology_ProcessCoordFromId_Args(
        topology, process_id, coords_max_dims, coords, coords_num_dims
    )
        return new(
            csizeof(PJRT_TpuTopology_ProcessCoordFromId_Args),
            topology,
            process_id,
            coords_max_dims,
            coords,
            coords_num_dims,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ProcessCoordFromId_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_TpuTopology_ProcessCoordFromId ( PJRT_TpuTopology_ProcessCoordFromId_Args * args )
const PJRT_TpuTopology_ProcessCoordFromId = Cvoid

struct PJRT_TpuTopology_ChipIdFromCoord_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    coords::Ptr{Int32}
    coords_num_dims::Csize_t
    chip_id::Int32
    function PJRT_TpuTopology_ChipIdFromCoord_Args(
        topology, coords, coords_num_dims, chip_id
    )
        return new(
            csizeof(PJRT_TpuTopology_ChipIdFromCoord_Args),
            topology,
            coords,
            coords_num_dims,
            chip_id,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ChipIdFromCoord_Args}) = Csize_t(36)

# typedef PJRT_Error * PJRT_TpuTopology_ChipIdFromCoord ( PJRT_TpuTopology_ChipIdFromCoord_Args * args )
const PJRT_TpuTopology_ChipIdFromCoord = Cvoid

struct PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chip_coords::Ptr{Int32}
    chip_coords_num_dims::Csize_t
    logical_device_index_on_chip::Int32
    logical_device_of_default_type_id::Int32
    function PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args(
        topology,
        chip_coords,
        chip_coords_num_dims,
        logical_device_index_on_chip,
        logical_device_of_default_type_id,
    )
        return new(
            csizeof(PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args),
            topology,
            chip_coords,
            chip_coords_num_dims,
            logical_device_index_on_chip,
            logical_device_of_default_type_id,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx ( PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args * args )
const PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx = Cvoid

struct PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    device_id::Int32
    chip_coords_max_dims::Csize_t
    chip_coords::Ptr{Int32}
    chip_coords_num_dims::Csize_t
    device_index_on_chip::Int32
    function PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args(
        topology,
        device_id,
        chip_coords_max_dims,
        chip_coords,
        chip_coords_num_dims,
        device_index_on_chip,
    )
        return new(
            csizeof(PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args),
            topology,
            device_id,
            chip_coords_max_dims,
            chip_coords,
            chip_coords_num_dims,
            device_index_on_chip,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args}) = Csize_t(52)

# typedef PJRT_Error * PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice ( PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args * args )
const PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice = Cvoid

struct PJRT_TpuTopology_ChipsPerProcessBounds_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chip_per_process_bounds_max_dims::Csize_t
    chip_per_process_bounds::Ptr{Int32}
    chip_per_process_bounds_num_dims::Csize_t
    function PJRT_TpuTopology_ChipsPerProcessBounds_Args(
        topology,
        chip_per_process_bounds_max_dims,
        chip_per_process_bounds,
        chip_per_process_bounds_num_dims,
    )
        return new(
            csizeof(PJRT_TpuTopology_ChipsPerProcessBounds_Args),
            topology,
            chip_per_process_bounds_max_dims,
            chip_per_process_bounds,
            chip_per_process_bounds_num_dims,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ChipsPerProcessBounds_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TpuTopology_ChipsPerProcessBounds ( PJRT_TpuTopology_ChipsPerProcessBounds_Args * args )
const PJRT_TpuTopology_ChipsPerProcessBounds = Cvoid

struct PJRT_TpuTopology_ChipBounds_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chip_bounds_max_dims::Csize_t
    chip_bounds::Ptr{Int32}
    chip_bounds_num_dims::Csize_t
    function PJRT_TpuTopology_ChipBounds_Args(
        topology, chip_bounds_max_dims, chip_bounds, chip_bounds_num_dims
    )
        return new(
            csizeof(PJRT_TpuTopology_ChipBounds_Args),
            topology,
            chip_bounds_max_dims,
            chip_bounds,
            chip_bounds_num_dims,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ChipBounds_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TpuTopology_ChipBounds ( PJRT_TpuTopology_ChipBounds_Args * args )
const PJRT_TpuTopology_ChipBounds = Cvoid

struct PJRT_TpuTopology_ProcessBounds_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    process_bounds_max_dims::Csize_t
    process_bounds::Ptr{Int32}
    process_bounds_num_dims::Csize_t
    function PJRT_TpuTopology_ProcessBounds_Args(
        topology, process_bounds_max_dims, process_bounds, process_bounds_num_dims
    )
        return new(
            csizeof(PJRT_TpuTopology_ProcessBounds_Args),
            topology,
            process_bounds_max_dims,
            process_bounds,
            process_bounds_num_dims,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_ProcessBounds_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TpuTopology_ProcessBounds ( PJRT_TpuTopology_ProcessBounds_Args * args )
const PJRT_TpuTopology_ProcessBounds = Cvoid

struct PJRT_TpuTopology_GetRoutingStrategy_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    routing_strategy::Cstring
    routing_strategy_len::Csize_t
    function PJRT_TpuTopology_GetRoutingStrategy_Args(
        topology, routing_strategy, routing_strategy_len
    )
        return new(
            csizeof(PJRT_TpuTopology_GetRoutingStrategy_Args),
            topology,
            routing_strategy,
            routing_strategy_len,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_GetRoutingStrategy_Args}) = Csize_t(32)

# typedef PJRT_Error * PJRT_TpuTopology_GetRoutingStrategy ( PJRT_TpuTopology_GetRoutingStrategy_Args * args )
const PJRT_TpuTopology_GetRoutingStrategy = Cvoid

struct PJRT_TpuTopology_SliceConfig
    dim_size::Csize_t
    dimensions::NTuple{4,Int32}
    wrap::NTuple{4,Bool}
    twist::Bool
end

csizeof(::Type{PJRT_TpuTopology_SliceConfig}) = Csize_t(29)

struct PJRT_TpuTopology_GetSliceConfig_Args
    struct_size::Csize_t
    platform_type_name::Cstring
    platform_type_name_len::Csize_t
    slice_name::Cstring
    slice_name_len::Csize_t
    slice_config::Ptr{PJRT_TpuTopology_SliceConfig}
    function PJRT_TpuTopology_GetSliceConfig_Args(
        platform_type_name, platform_type_name_len, slice_name, slice_name_len, slice_config
    )
        return new(
            csizeof(PJRT_TpuTopology_GetSliceConfig_Args),
            platform_type_name,
            platform_type_name_len,
            slice_name,
            slice_name_len,
            slice_config,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_GetSliceConfig_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_TpuTopology_GetSliceConfig ( PJRT_TpuTopology_GetSliceConfig_Args * args )
const PJRT_TpuTopology_GetSliceConfig = Cvoid

struct PJRT_TpuTopology_GetSliceConfigs_Args
    struct_size::Csize_t
    platform_type_name::Cstring
    platform_type_name_len::Csize_t
    slice_configs::Ptr{PJRT_TpuTopology_SliceConfig}
    max_slice_configs::Csize_t
    num_slice_configs::Csize_t
    function PJRT_TpuTopology_GetSliceConfigs_Args(
        platform_type_name,
        platform_type_name_len,
        slice_configs,
        max_slice_configs,
        num_slice_configs,
    )
        return new(
            csizeof(PJRT_TpuTopology_GetSliceConfigs_Args),
            platform_type_name,
            platform_type_name_len,
            slice_configs,
            max_slice_configs,
            num_slice_configs,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_GetSliceConfigs_Args}) = Csize_t(48)

# typedef PJRT_Error * PJRT_TpuTopology_GetSliceConfigs ( PJRT_TpuTopology_GetSliceConfigs_Args * args )
const PJRT_TpuTopology_GetSliceConfigs = Cvoid

struct PJRT_TpuTopology_GetDefaultPlatformConfig_Args
    struct_size::Csize_t
    platform_type_name::Cstring
    platform_type_name_len::Csize_t
    num_chips_per_tray::Int64
    num_trays::Int64
    function PJRT_TpuTopology_GetDefaultPlatformConfig_Args(
        platform_type_name, platform_type_name_len, num_chips_per_tray, num_trays
    )
        return new(
            csizeof(PJRT_TpuTopology_GetDefaultPlatformConfig_Args),
            platform_type_name,
            platform_type_name_len,
            num_chips_per_tray,
            num_trays,
        )
    end
end

csizeof(::Type{PJRT_TpuTopology_GetDefaultPlatformConfig_Args}) = Csize_t(40)

# typedef PJRT_Error * PJRT_TpuTopology_GetDefaultPlatformConfig ( PJRT_TpuTopology_GetDefaultPlatformConfig_Args * args )
const PJRT_TpuTopology_GetDefaultPlatformConfig = Cvoid

struct PJRT_TpuTopology_Extension
    base::PJRT_Extension_Base
    subslice::Ptr{PJRT_TpuTopology_Subslice}
    is_subslice_topology::Ptr{PJRT_TpuTopology_IsSubsliceTopology}
    subslice_device_id_from_full_device_id::Ptr{
        PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId
    }
    replace_host_bounds::Ptr{PJRT_TpuTopology_ReplaceHostBounds}
    is_enhanced_barrier_enabled::Ptr{PJRT_TpuTopology_IsEnhancedBarrierEnabled}
    has_limited_ici_connectivity::Ptr{PJRT_TpuTopology_HasLimitedIciConnectivity}
    is_reachable_over_limited_ici::Ptr{PJRT_TpuTopology_IsReachableOverLimitedIci}
    process_count::Ptr{PJRT_TpuTopology_ProcessCount}
    chips_per_process::Ptr{PJRT_TpuTopology_ChipsPerProcess}
    core_count_per_chip::Ptr{PJRT_TpuTopology_CoreCountPerChip}
    chip_count::Ptr{PJRT_TpuTopology_ChipCount}
    core_count::Ptr{PJRT_TpuTopology_CoreCount}
    logical_device_count_per_process::Ptr{PJRT_TpuTopology_LogiDeviceCountPerProcess}
    logical_device_count::Ptr{PJRT_TpuTopology_LogiDeviceCount}
    logical_device_count_per_chip::Ptr{PJRT_TpuTopology_LogiDeviceCountPerChip}
    core_count_per_process::Ptr{PJRT_TpuTopology_CoreCountPerProcess}
    process_ids::Ptr{PJRT_TpuTopology_ProcessIds}
    logical_device_ids_on_process::Ptr{PJRT_TpuTopology_LogiDeviceIdsOnProcess}
    proc_id_and_idx_on_proc_for_chip::Ptr{PJRT_TpuTopology_ProcIdAndIdxOnProcForChip}
    proc_id_and_idx_on_proc_for_logi_device::Ptr{
        PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice
    }
    process_coord_from_id::Ptr{PJRT_TpuTopology_ProcessCoordFromId}
    chip_id_from_coord::Ptr{PJRT_TpuTopology_ChipIdFromCoord}
    logical_device_id_from_chip_coord_and_idx::Ptr{
        PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx
    }
    chip_coord_and_idx_for_logi_device::Ptr{PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice}
    chips_per_process_bounds::Ptr{PJRT_TpuTopology_ChipsPerProcessBounds}
    chip_bounds::Ptr{PJRT_TpuTopology_ChipBounds}
    process_bounds::Ptr{PJRT_TpuTopology_ProcessBounds}
    get_routing_strategy::Ptr{PJRT_TpuTopology_GetRoutingStrategy}
    get_slice_config::Ptr{PJRT_TpuTopology_GetSliceConfig}
    get_slice_configs::Ptr{PJRT_TpuTopology_GetSliceConfigs}
    get_default_platform_config::Ptr{PJRT_TpuTopology_GetDefaultPlatformConfig}
end

csizeof(::Type{PJRT_TpuTopology_Extension}) = Csize_t(272)

struct PJRT_XlaTransform_string
    data::Cstring
    size::Csize_t
end

struct PJRT_XlaTransform_version_and_error
    api_version::Int64
    data::Ptr{Cvoid}
    cleanup_fn::Ptr{Cvoid}
    has_error::Bool
    code::PJRT_Error_Code
    error_msg::PJRT_XlaTransform_string
end

struct PJRT_XlaTransform_Args
    struct_size::Csize_t
    header::PJRT_XlaTransform_version_and_error
    hlo_module::PJRT_XlaTransform_string
    transformed_hlo_module::PJRT_XlaTransform_string
    changed::Bool
    function PJRT_XlaTransform_Args(header, hlo_module, transformed_hlo_module, changed)
        return new(
            csizeof(PJRT_XlaTransform_Args),
            header,
            hlo_module,
            transformed_hlo_module,
            changed,
        )
    end
end

csizeof(::Type{PJRT_XlaTransform_Args}) = Csize_t(89)

struct PJRT_XlaTransform_Callbacks
    version::Int64
    dtor::Ptr{Cvoid}
    transform_hlo_module::Ptr{Cvoid}
end

@cenum PJRT_XlaTransform_PipelineStage::UInt32 begin
    PJRT_XlaTransform_PipelineStage_kPreScheduler = 0
    PJRT_XlaTransform_PipelineStage_kPostScheduler = 1
end

struct PJRT_Register_Xla_Transform_Args
    struct_size::Csize_t
    name::Cstring
    name_size::Csize_t
    stage::PJRT_XlaTransform_PipelineStage
    callbacks::Ptr{PJRT_XlaTransform_Callbacks}
    function PJRT_Register_Xla_Transform_Args(name, name_size, stage, callbacks)
        return new(
            csizeof(PJRT_Register_Xla_Transform_Args), name, name_size, stage, callbacks
        )
    end
end

csizeof(::Type{PJRT_Register_Xla_Transform_Args}) = Csize_t(40)

# typedef PJRT_Error * ( * PJRT_Register_Xla_Transform_Fn ) ( PJRT_Register_Xla_Transform_Args * args )
const PJRT_Register_Xla_Transform_Fn = Ptr{Cvoid}

struct PJRT_Clear_Xla_Transform_Args
    struct_size::Csize_t
    stage::PJRT_XlaTransform_PipelineStage
    name::Cstring
    name_size::Csize_t
    callbacks::Ptr{PJRT_XlaTransform_Callbacks}
    cleared::Bool
    function PJRT_Clear_Xla_Transform_Args(stage, name, name_size, callbacks, cleared)
        return new(
            csizeof(PJRT_Clear_Xla_Transform_Args),
            stage,
            name,
            name_size,
            callbacks,
            cleared,
        )
    end
end

csizeof(::Type{PJRT_Clear_Xla_Transform_Args}) = Csize_t(41)

# typedef PJRT_Error * ( * PJRT_Clear_Xla_Transform_Fn ) ( PJRT_Clear_Xla_Transform_Args * args )
const PJRT_Clear_Xla_Transform_Fn = Ptr{Cvoid}

struct PJRT_HloPassPipelineTrace
    serialized_trace::Cstring
    serialized_trace_size::Csize_t
end

struct PJRT_Xla_Transform_Get_Hlo_Pass_Pipeline_Trace_Args
    struct_size::Csize_t
    hlo_module::PJRT_XlaTransform_string
    trace::PJRT_HloPassPipelineTrace
    function PJRT_Xla_Transform_Get_Hlo_Pass_Pipeline_Trace_Args(hlo_module, trace)
        return new(
            csizeof(PJRT_Xla_Transform_Get_Hlo_Pass_Pipeline_Trace_Args), hlo_module, trace
        )
    end
end

csizeof(::Type{PJRT_Xla_Transform_Get_Hlo_Pass_Pipeline_Trace_Args}) = Csize_t(40)

# typedef PJRT_Error * ( * PJRT_Xla_Transform_Get_Hlo_Pass_Pipeline_Trace_Fn ) ( PJRT_Xla_Transform_Get_Hlo_Pass_Pipeline_Trace_Args * args )
const PJRT_Xla_Transform_Get_Hlo_Pass_Pipeline_Trace_Fn = Ptr{Cvoid}

struct PJRT_Xla_Transform_Destroy_Hlo_Pass_Pipeline_Trace_Args
    struct_size::Csize_t
    trace::Ptr{PJRT_HloPassPipelineTrace}
    function PJRT_Xla_Transform_Destroy_Hlo_Pass_Pipeline_Trace_Args(trace)
        return new(csizeof(PJRT_Xla_Transform_Destroy_Hlo_Pass_Pipeline_Trace_Args), trace)
    end
end

csizeof(::Type{PJRT_Xla_Transform_Destroy_Hlo_Pass_Pipeline_Trace_Args}) = Csize_t(16)

# typedef void ( * PJRT_Xla_Transform_Destroy_Hlo_Pass_Pipeline_Trace_Fn ) ( PJRT_Xla_Transform_Destroy_Hlo_Pass_Pipeline_Trace_Args * args )
const PJRT_Xla_Transform_Destroy_Hlo_Pass_Pipeline_Trace_Fn = Ptr{Cvoid}

struct PJRT_Xla_Transform_Extension
    base::PJRT_Extension_Base
    register_xla_transform::PJRT_Register_Xla_Transform_Fn
    clear_xla_transform::PJRT_Clear_Xla_Transform_Fn
    get_hlo_pass_pipeline_trace::PJRT_Xla_Transform_Get_Hlo_Pass_Pipeline_Trace_Fn
    destroy_hlo_pass_pipeline_trace::PJRT_Xla_Transform_Destroy_Hlo_Pass_Pipeline_Trace_Fn
end

csizeof(::Type{PJRT_Xla_Transform_Extension}) = Csize_t(56)

const PJRT_API_MAJOR = 0

const PJRT_API_MINOR = 117

const PJRT_API_ABI_VERSION_EXTENSION_VERSION = 1

const PJRT_API_CALLBACK_EXTENSION_VERSION = 1

const PJRT_API_COLLECTIVES_EXTENSION_VERSION = 1

const PJRT_API_CUSTOM_PARTITIONER_EXTENSION_VERSION = 1

const PJRT_API_FFI_EXTENSION_VERSION = 3

const PJRT_API_GPU_EXTENSION_VERSION = 3

const PJRT_API_LAYOUTS_EXTENSION_VERSION = 4

const PJRT_API_MEGASCALE_EXTENSION_VERSION = 1

const PJRT_API_MEMORY_DESCRIPTIONS_EXTENSION_VERSION = 1

const PJRT_API_MULTI_SLICE_EXTENSION_VERSION = 1

const PJRT_API_PHASE_COMPILE_EXTENSION_VERSION = 1

const PLUGIN_PROFILER_VERSION = 1

const PJRT_API_PROFILER_EXTENSION_VERSION = 1

const PJRT_API_RAW_BUFFER_EXTENSION_VERSION = 2

const PJRT_API_SHARDINGS_EXTENSION_VERSION = 1

const PJRT_API_STREAM_EXTENSION_VERSION = 0

const PJRT_API_TPU_EXECUTABLE_EXTENSION_VERSION = 1

const PJRT_API_TPU_TOPOLOGY_EXTENSION_VERSION = 1

const PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION = 1
