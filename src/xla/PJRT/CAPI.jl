using CEnum: CEnum, @cenum

const IS_LIBC_MUSL = occursin("musl", Base.MACHINE)

if Sys.islinux() && Sys.ARCH === :aarch64 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :aarch64 && IS_LIBC_MUSL
    const off_t = Clong
elseif Sys.islinux() && startswith(string(Sys.ARCH), "arm") && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && startswith(string(Sys.ARCH), "arm") && IS_LIBC_MUSL
    const off_t = Clonglong
elseif Sys.islinux() && Sys.ARCH === :i686 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :i686 && IS_LIBC_MUSL
    const off_t = Clonglong
elseif Sys.iswindows() && Sys.ARCH === :i686
    const off32_t = Clong
    const off_t = off32_t
elseif Sys.islinux() && Sys.ARCH === :powerpc64le
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.isapple()
    const __darwin_off_t = Int64
    const off_t = __darwin_off_t
elseif Sys.islinux() && Sys.ARCH === :x86_64 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :x86_64 && IS_LIBC_MUSL
    const off_t = Clong
elseif Sys.isbsd() && !Sys.isapple()
    const __off_t = Int64
    const off_t = __off_t
elseif Sys.iswindows() && Sys.ARCH === :x86_64
    const off32_t = Clong
    const off_t = off32_t
end


@cenum PJRT_Extension_Type::UInt32 begin
    PJRT_Extension_Type_Gpu_Custom_Call = 0x0000000000000000
    PJRT_Extension_Type_Profiler = 0x0000000000000001
    PJRT_Extension_Type_Custom_Partitioner = 0x0000000000000002
    PJRT_Extension_Type_Stream = 0x0000000000000003
    PJRT_Extension_Type_Layouts = 0x0000000000000004
    PJRT_Extension_Type_FFI = 0x0000000000000005
    PJRT_Extension_Type_MemoryDescriptions = 0x0000000000000006
    PJRT_Extension_Type_Triton = 0x0000000000000007
    PJRT_Extension_Type_RawBuffer = 0x0000000000000008
    PJRT_Extension_Type_PhaseCompile = 0x0000000000000009
    PJRT_Extension_Type_Example = 0x000000000000000a
    PJRT_Extension_Type_Unknown = 0x000000000000000b
    PJRT_Extension_Type_CrossHostTransfers = 0x000000000000000c
    PJRT_Extension_Type_ExecutableMetadata = 0x000000000000000d
    PJRT_Extension_Type_Callback = 0x000000000000000e
    PJRT_Extension_Type_HostAllocator = 0x000000000000000f
    PJRT_Extension_Type_TpuTopology = 0x0000000000000010
    PJRT_Extension_Type_TpuExecutable = 0x0000000000000011
    PJRT_Extension_Type_Megascale = 0x0000000000000012
end

struct PJRT_Extension_Base
    struct_size::Csize_t
    type::PJRT_Extension_Type
    next::Ptr{PJRT_Extension_Base}
end

@cenum __JL_Ctag_1::UInt32 begin
    PJRT_Extension_Base_STRUCT_SIZE = 0x0000000000000018
end

struct PJRT_Api_Version
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    major_version::Cint
    minor_version::Cint
end

@cenum __JL_Ctag_2::UInt32 begin
    PJRT_Api_Version_STRUCT_SIZE = 0x0000000000000018
end

mutable struct PJRT_Error end

struct PJRT_Error_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    error::Ptr{PJRT_Error}
end

@cenum __JL_Ctag_3::UInt32 begin
    PJRT_Error_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef void PJRT_Error_Destroy ( PJRT_Error_Destroy_Args * args )
const PJRT_Error_Destroy = Cvoid

struct PJRT_Error_Message_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    error::Ptr{PJRT_Error}
    message::Cstring
    message_size::Csize_t
end

@cenum __JL_Ctag_4::UInt32 begin
    PJRT_Error_Message_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef void PJRT_Error_Message ( PJRT_Error_Message_Args * args )
const PJRT_Error_Message = Cvoid

@cenum PJRT_Error_Code::UInt32 begin
    PJRT_Error_Code_OK = 0x0000000000000000
    PJRT_Error_Code_CANCELLED = 0x0000000000000001
    PJRT_Error_Code_UNKNOWN = 0x0000000000000002
    PJRT_Error_Code_INVALID_ARGUMENT = 0x0000000000000003
    PJRT_Error_Code_DEADLINE_EXCEEDED = 0x0000000000000004
    PJRT_Error_Code_NOT_FOUND = 0x0000000000000005
    PJRT_Error_Code_ALREADY_EXISTS = 0x0000000000000006
    PJRT_Error_Code_PERMISSION_DENIED = 0x0000000000000007
    PJRT_Error_Code_RESOURCE_EXHAUSTED = 0x0000000000000008
    PJRT_Error_Code_FAILED_PRECONDITION = 0x0000000000000009
    PJRT_Error_Code_ABORTED = 0x000000000000000a
    PJRT_Error_Code_OUT_OF_RANGE = 0x000000000000000b
    PJRT_Error_Code_UNIMPLEMENTED = 0x000000000000000c
    PJRT_Error_Code_INTERNAL = 0x000000000000000d
    PJRT_Error_Code_UNAVAILABLE = 0x000000000000000e
    PJRT_Error_Code_DATA_LOSS = 0x000000000000000f
    PJRT_Error_Code_UNAUTHENTICATED = 0x0000000000000010
end

struct PJRT_Error_GetCode_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    error::Ptr{PJRT_Error}
    code::PJRT_Error_Code
end

@cenum __JL_Ctag_5::UInt32 begin
    PJRT_Error_GetCode_Args_STRUCT_SIZE = 0x000000000000001c
end

# typedef PJRT_Error * PJRT_Error_GetCode ( PJRT_Error_GetCode_Args * args )
const PJRT_Error_GetCode = Cvoid

# typedef PJRT_Error * ( * PJRT_CallbackError ) ( PJRT_Error_Code code , const char * message , size_t message_size )
const PJRT_CallbackError = Ptr{Cvoid}

@cenum PJRT_NamedValue_Type::UInt32 begin
    PJRT_NamedValue_kString = 0x0000000000000000
    PJRT_NamedValue_kInt64 = 0x0000000000000001
    PJRT_NamedValue_kInt64List = 0x0000000000000002
    PJRT_NamedValue_kFloat = 0x0000000000000003
    PJRT_NamedValue_kBool = 0x0000000000000004
end

struct PJRT_NamedValue
    data::NTuple{56, UInt8}
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
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::PJRT_NamedValue, private::Bool = false)
    (:struct_size, :extension_start, :name, :name_size, :type, :string_value, :int64_value, :int64_array_value, :float_value, :bool_value, :value_size, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

@cenum __JL_Ctag_6::UInt32 begin
    PJRT_NamedValue_STRUCT_SIZE = 0x0000000000000038
end

struct PJRT_Plugin_Initialize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
end

@cenum __JL_Ctag_7::UInt32 begin
    PJRT_Plugin_Initialize_Args_STRUCT_SIZE = 0x0000000000000010
end

# typedef PJRT_Error * PJRT_Plugin_Initialize ( PJRT_Plugin_Initialize_Args * args )
const PJRT_Plugin_Initialize = Cvoid

struct PJRT_Plugin_Attributes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    attributes::Ptr{PJRT_NamedValue}
    num_attributes::Csize_t
end

@cenum __JL_Ctag_8::UInt32 begin
    PJRT_Plugin_Attributes_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Plugin_Attributes ( PJRT_Plugin_Attributes_Args * args )
const PJRT_Plugin_Attributes = Cvoid

mutable struct PJRT_Event end

struct PJRT_Event_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
end

@cenum __JL_Ctag_9::UInt32 begin
    PJRT_Event_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Event_Destroy ( PJRT_Event_Destroy_Args * args )
const PJRT_Event_Destroy = Cvoid

struct PJRT_Event_IsReady_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
    is_ready::Bool
end

@cenum __JL_Ctag_10::UInt32 begin
    PJRT_Event_IsReady_Args_STRUCT_SIZE = 0x0000000000000019
end

# typedef PJRT_Error * PJRT_Event_IsReady ( PJRT_Event_IsReady_Args * args )
const PJRT_Event_IsReady = Cvoid

struct PJRT_Event_Error_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
end

@cenum __JL_Ctag_11::UInt32 begin
    PJRT_Event_Error_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Event_Error ( PJRT_Event_Error_Args * args )
const PJRT_Event_Error = Cvoid

struct PJRT_Event_Await_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
end

@cenum __JL_Ctag_12::UInt32 begin
    PJRT_Event_Await_Args_STRUCT_SIZE = 0x0000000000000018
end

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
end

@cenum __JL_Ctag_13::UInt32 begin
    PJRT_Event_OnReady_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Event_OnReady ( PJRT_Event_OnReady_Args * args )
const PJRT_Event_OnReady = Cvoid

struct PJRT_Event_Create_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
end

@cenum __JL_Ctag_14::UInt32 begin
    PJRT_Event_Create_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Event_Create ( PJRT_Event_Create_Args * args )
const PJRT_Event_Create = Cvoid

struct PJRT_Event_Set_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_Event}
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
end

@cenum __JL_Ctag_15::UInt32 begin
    PJRT_Event_Set_Args_STRUCT_SIZE = 0x0000000000000030
end

# typedef PJRT_Error * PJRT_Event_Set ( PJRT_Event_Set_Args * args )
const PJRT_Event_Set = Cvoid

mutable struct PJRT_Client end

mutable struct PJRT_Device end

mutable struct PJRT_Memory end

@cenum PJRT_Buffer_Type::UInt32 begin
    PJRT_Buffer_Type_INVALID = 0x0000000000000000
    PJRT_Buffer_Type_PRED = 0x0000000000000001
    PJRT_Buffer_Type_S8 = 0x0000000000000002
    PJRT_Buffer_Type_S16 = 0x0000000000000003
    PJRT_Buffer_Type_S32 = 0x0000000000000004
    PJRT_Buffer_Type_S64 = 0x0000000000000005
    PJRT_Buffer_Type_U8 = 0x0000000000000006
    PJRT_Buffer_Type_U16 = 0x0000000000000007
    PJRT_Buffer_Type_U32 = 0x0000000000000008
    PJRT_Buffer_Type_U64 = 0x0000000000000009
    PJRT_Buffer_Type_F16 = 0x000000000000000a
    PJRT_Buffer_Type_F32 = 0x000000000000000b
    PJRT_Buffer_Type_F64 = 0x000000000000000c
    PJRT_Buffer_Type_BF16 = 0x000000000000000d
    PJRT_Buffer_Type_C64 = 0x000000000000000e
    PJRT_Buffer_Type_C128 = 0x000000000000000f
    PJRT_Buffer_Type_F8E5M2 = 0x0000000000000010
    PJRT_Buffer_Type_F8E4M3FN = 0x0000000000000011
    PJRT_Buffer_Type_F8E4M3B11FNUZ = 0x0000000000000012
    PJRT_Buffer_Type_F8E5M2FNUZ = 0x0000000000000013
    PJRT_Buffer_Type_F8E4M3FNUZ = 0x0000000000000014
    PJRT_Buffer_Type_S4 = 0x0000000000000015
    PJRT_Buffer_Type_U4 = 0x0000000000000016
    PJRT_Buffer_Type_TOKEN = 0x0000000000000017
    PJRT_Buffer_Type_S2 = 0x0000000000000018
    PJRT_Buffer_Type_U2 = 0x0000000000000019
    PJRT_Buffer_Type_F8E4M3 = 0x000000000000001a
    PJRT_Buffer_Type_F8E3M4 = 0x000000000000001b
    PJRT_Buffer_Type_F8E8M0FNU = 0x000000000000001c
    PJRT_Buffer_Type_F4E2M1FN = 0x000000000000001d
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
end

@cenum __JL_Ctag_16::UInt32 begin
    PJRT_KeyValueGetCallback_Args_STRUCT_SIZE = 0x0000000000000050
end

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
end

@cenum __JL_Ctag_17::UInt32 begin
    PJRT_KeyValueTryGetCallback_Args_STRUCT_SIZE = 0x0000000000000048
end

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
end

@cenum __JL_Ctag_18::UInt32 begin
    PJRT_KeyValuePutCallback_Args_STRUCT_SIZE = 0x0000000000000040
end

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
end

@cenum __JL_Ctag_19::UInt32 begin
    PJRT_Client_Create_Args_STRUCT_SIZE = 0x0000000000000058
end

# typedef PJRT_Error * PJRT_Client_Create ( PJRT_Client_Create_Args * args )
const PJRT_Client_Create = Cvoid

struct PJRT_Client_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
end

@cenum __JL_Ctag_20::UInt32 begin
    PJRT_Client_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Client_Destroy ( PJRT_Client_Destroy_Args * args )
const PJRT_Client_Destroy = Cvoid

struct PJRT_Client_PlatformName_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    platform_name::Cstring
    platform_name_size::Csize_t
end

@cenum __JL_Ctag_21::UInt32 begin
    PJRT_Client_PlatformName_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Client_PlatformName ( PJRT_Client_PlatformName_Args * args )
const PJRT_Client_PlatformName = Cvoid

struct PJRT_Client_ProcessIndex_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    process_index::Cint
end

@cenum __JL_Ctag_22::UInt32 begin
    PJRT_Client_ProcessIndex_Args_STRUCT_SIZE = 0x000000000000001c
end

# typedef PJRT_Error * PJRT_Client_ProcessIndex ( PJRT_Client_ProcessIndex_Args * args )
const PJRT_Client_ProcessIndex = Cvoid

struct PJRT_Client_PlatformVersion_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    platform_version::Cstring
    platform_version_size::Csize_t
end

@cenum __JL_Ctag_23::UInt32 begin
    PJRT_Client_PlatformVersion_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Client_PlatformVersion ( PJRT_Client_PlatformVersion_Args * args )
const PJRT_Client_PlatformVersion = Cvoid

struct PJRT_Client_TopologyDescription_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    topology::Ptr{PJRT_TopologyDescription}
end

@cenum __JL_Ctag_24::UInt32 begin
    PJRT_Client_TopologyDescription_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Client_TopologyDescription ( PJRT_Client_TopologyDescription_Args * args )
const PJRT_Client_TopologyDescription = Cvoid

struct PJRT_Client_Devices_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    devices::Ptr{Ptr{PJRT_Device}}
    num_devices::Csize_t
end

@cenum __JL_Ctag_25::UInt32 begin
    PJRT_Client_Devices_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Client_Devices ( PJRT_Client_Devices_Args * args )
const PJRT_Client_Devices = Cvoid

struct PJRT_Client_AddressableDevices_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    addressable_devices::Ptr{Ptr{PJRT_Device}}
    num_addressable_devices::Csize_t
end

@cenum __JL_Ctag_26::UInt32 begin
    PJRT_Client_AddressableDevices_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Client_AddressableDevices ( PJRT_Client_AddressableDevices_Args * args )
const PJRT_Client_AddressableDevices = Cvoid

struct PJRT_Client_LookupDevice_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    id::Cint
    device::Ptr{PJRT_Device}
end

@cenum __JL_Ctag_27::UInt32 begin
    PJRT_Client_LookupDevice_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Client_LookupDevice ( PJRT_Client_LookupDevice_Args * args )
const PJRT_Client_LookupDevice = Cvoid

struct PJRT_Client_LookupAddressableDevice_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    local_hardware_id::Cint
    addressable_device::Ptr{PJRT_Device}
end

@cenum __JL_Ctag_28::UInt32 begin
    PJRT_Client_LookupAddressableDevice_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Client_LookupAddressableDevice ( PJRT_Client_LookupAddressableDevice_Args * args )
const PJRT_Client_LookupAddressableDevice = Cvoid

@cenum PJRT_ProcessState::UInt32 begin
    PJRT_ProcessState_kUnspecified = 0x0000000000000000
    PJRT_ProcessState_kUninitialized = 0x0000000000000001
    PJRT_ProcessState_kDisconnected = 0x0000000000000002
    PJRT_ProcessState_kConnected = 0x0000000000000003
    PJRT_ProcessState_kError = 0x0000000000000004
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

@cenum __JL_Ctag_29::UInt32 begin
    PJRT_ProcessInfo_STRUCT_SIZE = 0x0000000000000030
end

struct PJRT_Client_UpdateGlobalProcessInfo_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    process_infos::Ptr{PJRT_ProcessInfo}
    num_process_infos::Csize_t
end

@cenum __JL_Ctag_30::UInt32 begin
    PJRT_Client_UpdateGlobalProcessInfo_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Client_UpdateGlobalProcessInfo ( PJRT_Client_UpdateGlobalProcessInfo_Args * args )
const PJRT_Client_UpdateGlobalProcessInfo = Cvoid

struct PJRT_Client_AddressableMemories_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    addressable_memories::Ptr{Ptr{PJRT_Memory}}
    num_addressable_memories::Csize_t
end

@cenum __JL_Ctag_31::UInt32 begin
    PJRT_Client_AddressableMemories_Args_STRUCT_SIZE = 0x0000000000000028
end

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

@cenum __JL_Ctag_32::UInt32 begin
    PJRT_Program_STRUCT_SIZE = 0x0000000000000030
end

struct PJRT_Client_Compile_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    program::Ptr{PJRT_Program}
    compile_options::Cstring
    compile_options_size::Csize_t
    executable::Ptr{PJRT_LoadedExecutable}
end

@cenum __JL_Ctag_33::UInt32 begin
    PJRT_Client_Compile_Args_STRUCT_SIZE = 0x0000000000000038
end

# typedef PJRT_Error * PJRT_Client_Compile ( PJRT_Client_Compile_Args * args )
const PJRT_Client_Compile = Cvoid

struct PJRT_Client_DefaultDeviceAssignment_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    num_replicas::Cint
    num_partitions::Cint
    default_assignment_size::Csize_t
    default_assignment::Ptr{Cint}
end

@cenum __JL_Ctag_34::UInt32 begin
    PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE = 0x0000000000000030
end

# typedef PJRT_Error * PJRT_Client_DefaultDeviceAssignment ( PJRT_Client_DefaultDeviceAssignment_Args * args )
const PJRT_Client_DefaultDeviceAssignment = Cvoid

struct PJRT_Client_DmaMap_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    data::Ptr{Cvoid}
    size::Csize_t
end

@cenum __JL_Ctag_35::UInt32 begin
    PJRT_Client_DmaMap_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Client_DmaMap ( PJRT_Client_DmaMap_Args * args )
const PJRT_Client_DmaMap = Cvoid

struct PJRT_Client_DmaUnmap_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    data::Ptr{Cvoid}
end

@cenum __JL_Ctag_36::UInt32 begin
    PJRT_Client_DmaUnmap_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Client_DmaUnmap ( PJRT_Client_DmaUnmap_Args * args )
const PJRT_Client_DmaUnmap = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
end

@cenum __JL_Ctag_37::UInt32 begin
    PJRT_AsyncHostToDeviceTransferManager_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

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
end

@cenum __JL_Ctag_38::UInt32 begin
    PJRT_AsyncHostToDeviceTransferManager_TransferData_Args_STRUCT_SIZE = 0x0000000000000048
end

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_TransferData ( PJRT_AsyncHostToDeviceTransferManager_TransferData_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_TransferData = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    buffer_index::Cint
    buffer_out::Ptr{PJRT_Buffer}
end

@cenum __JL_Ctag_39::UInt32 begin
    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer ( PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_Device_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    device_out::Ptr{PJRT_Device}
end

@cenum __JL_Ctag_40::UInt32 begin
    PJRT_AsyncHostToDeviceTransferManager_Device_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_Device ( PJRT_AsyncHostToDeviceTransferManager_Device_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_Device = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    buffer_count::Csize_t
end

@cenum __JL_Ctag_41::UInt32 begin
    PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_BufferCount ( PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_BufferCount = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    buffer_index::Cint
    buffer_size::Csize_t
end

@cenum __JL_Ctag_42::UInt32 begin
    PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args_STRUCT_SIZE = 0x0000000000000028
end

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
end

@cenum __JL_Ctag_43::UInt32 begin
    PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args_STRUCT_SIZE = 0x0000000000000030
end

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_SetBufferError ( PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_SetBufferError = Cvoid

struct PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    transfer_manager::Ptr{PJRT_AsyncHostToDeviceTransferManager}
    transfer_metadata::Ptr{PJRT_NamedValue}
    num_metadata::Csize_t
end

@cenum __JL_Ctag_44::UInt32 begin
    PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_AsyncHostToDeviceTransferManager_AddMetadata ( PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args * args )
const PJRT_AsyncHostToDeviceTransferManager_AddMetadata = Cvoid

@cenum PJRT_HostBufferSemantics::UInt32 begin
    PJRT_HostBufferSemantics_kImmutableOnlyDuringCall = 0x0000000000000000
    PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes = 0x0000000000000001
    PJRT_HostBufferSemantics_kImmutableZeroCopy = 0x0000000000000002
    PJRT_HostBufferSemantics_kMutableZeroCopy = 0x0000000000000003
end

@cenum PJRT_Buffer_MemoryLayout_Type::UInt32 begin
    PJRT_Buffer_MemoryLayout_Type_Tiled = 0x0000000000000000
    PJRT_Buffer_MemoryLayout_Type_Strides = 0x0000000000000001
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

@cenum __JL_Ctag_45::UInt32 begin
    PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE = 0x0000000000000038
end

struct PJRT_Buffer_MemoryLayout_Strides
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    byte_strides::Ptr{Int64}
    num_byte_strides::Csize_t
end

@cenum __JL_Ctag_46::UInt32 begin
    PJRT_Buffer_MemoryLayout_Strides_STRUCT_SIZE = 0x0000000000000020
end

struct PJRT_Buffer_MemoryLayout
    data::NTuple{80, UInt8}
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
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::PJRT_Buffer_MemoryLayout, private::Bool = false)
    (:struct_size, :extension_start, :tiled, :strides, :type, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

@cenum __JL_Ctag_47::UInt32 begin
    PJRT_Buffer_MemoryLayout_STRUCT_SIZE = 0x000000000000004c
end

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
end

@cenum __JL_Ctag_48::UInt32 begin
    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args_STRUCT_SIZE = 0x0000000000000050
end

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
end

@cenum __JL_Ctag_49::UInt32 begin
    PJRT_Client_CreateUninitializedBuffer_Args_STRUCT_SIZE = 0x0000000000000050
end

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
end

@cenum __JL_Ctag_50::UInt32 begin
    PJRT_Client_CreateErrorBuffer_Args_STRUCT_SIZE = 0x0000000000000060
end

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
end

@cenum __JL_Ctag_51::UInt32 begin
    PJRT_Client_CreateAliasBuffer_Args_STRUCT_SIZE = 0x0000000000000050
end

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
end

@cenum __JL_Ctag_52::UInt32 begin
    PJRT_Client_FulfillAliasBuffer_Args_STRUCT_SIZE = 0x0000000000000040
end

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
end

@cenum __JL_Ctag_53::UInt32 begin
    PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE = 0x0000000000000078
end

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
end

@cenum __JL_Ctag_54::UInt32 begin
    PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE = 0x0000000000000070
end

# typedef PJRT_Error * PJRT_Client_CreateViewOfDeviceBuffer ( PJRT_Client_CreateViewOfDeviceBuffer_Args * args )
const PJRT_Client_CreateViewOfDeviceBuffer = Cvoid

@cenum __JL_Ctag_55::UInt32 begin
    PJRT_ShapeSpec_STRUCT_SIZE = 0x0000000000000024
end

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
end

@cenum __JL_Ctag_56::UInt32 begin
    PJRT_Client_CreateBuffersForAsyncHostToDevice_Args_STRUCT_SIZE = 0x0000000000000048
end

# typedef PJRT_Error * PJRT_Client_CreateBuffersForAsyncHostToDevice ( PJRT_Client_CreateBuffersForAsyncHostToDevice_Args * args )
const PJRT_Client_CreateBuffersForAsyncHostToDevice = Cvoid

struct PJRT_DeviceDescription_Id_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    id::Cint
end

@cenum __JL_Ctag_57::UInt32 begin
    PJRT_DeviceDescription_Id_Args_STRUCT_SIZE = 0x000000000000001c
end

# typedef PJRT_Error * PJRT_DeviceDescription_Id ( PJRT_DeviceDescription_Id_Args * args )
const PJRT_DeviceDescription_Id = Cvoid

struct PJRT_DeviceDescription_ProcessIndex_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    process_index::Cint
end

@cenum __JL_Ctag_58::UInt32 begin
    PJRT_DeviceDescription_ProcessIndex_Args_STRUCT_SIZE = 0x000000000000001c
end

# typedef PJRT_Error * PJRT_DeviceDescription_ProcessIndex ( PJRT_DeviceDescription_ProcessIndex_Args * args )
const PJRT_DeviceDescription_ProcessIndex = Cvoid

struct PJRT_DeviceDescription_Attributes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    num_attributes::Csize_t
    attributes::Ptr{PJRT_NamedValue}
end

@cenum __JL_Ctag_59::UInt32 begin
    PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_DeviceDescription_Attributes ( PJRT_DeviceDescription_Attributes_Args * args )
const PJRT_DeviceDescription_Attributes = Cvoid

struct PJRT_DeviceDescription_Kind_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    device_kind::Cstring
    device_kind_size::Csize_t
end

@cenum __JL_Ctag_60::UInt32 begin
    PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_DeviceDescription_Kind ( PJRT_DeviceDescription_Kind_Args * args )
const PJRT_DeviceDescription_Kind = Cvoid

struct PJRT_DeviceDescription_DebugString_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    debug_string::Cstring
    debug_string_size::Csize_t
end

@cenum __JL_Ctag_61::UInt32 begin
    PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_DeviceDescription_DebugString ( PJRT_DeviceDescription_DebugString_Args * args )
const PJRT_DeviceDescription_DebugString = Cvoid

struct PJRT_DeviceDescription_ToString_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    to_string::Cstring
    to_string_size::Csize_t
end

@cenum __JL_Ctag_62::UInt32 begin
    PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_DeviceDescription_ToString ( PJRT_DeviceDescription_ToString_Args * args )
const PJRT_DeviceDescription_ToString = Cvoid

struct PJRT_Device_GetDescription_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    device_description::Ptr{PJRT_DeviceDescription}
end

@cenum __JL_Ctag_63::UInt32 begin
    PJRT_Device_GetDescription_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Device_GetDescription ( PJRT_Device_GetDescription_Args * args )
const PJRT_Device_GetDescription = Cvoid

struct PJRT_Device_IsAddressable_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    is_addressable::Bool
end

@cenum __JL_Ctag_64::UInt32 begin
    PJRT_Device_IsAddressable_Args_STRUCT_SIZE = 0x0000000000000019
end

# typedef PJRT_Error * PJRT_Device_IsAddressable ( PJRT_Device_IsAddressable_Args * args )
const PJRT_Device_IsAddressable = Cvoid

struct PJRT_Device_LocalHardwareId_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    local_hardware_id::Cint
end

@cenum __JL_Ctag_65::UInt32 begin
    PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE = 0x000000000000001c
end

# typedef PJRT_Error * PJRT_Device_LocalHardwareId ( PJRT_Device_LocalHardwareId_Args * args )
const PJRT_Device_LocalHardwareId = Cvoid

struct PJRT_Device_AddressableMemories_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    memories::Ptr{Ptr{PJRT_Memory}}
    num_memories::Csize_t
end

@cenum __JL_Ctag_66::UInt32 begin
    PJRT_Device_AddressableMemories_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Device_AddressableMemories ( PJRT_Device_AddressableMemories_Args * args )
const PJRT_Device_AddressableMemories = Cvoid

struct PJRT_Device_DefaultMemory_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    memory::Ptr{PJRT_Memory}
end

@cenum __JL_Ctag_67::UInt32 begin
    PJRT_Device_DefaultMemory_Args_STRUCT_SIZE = 0x0000000000000020
end

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
end

@cenum __JL_Ctag_68::UInt32 begin
    PJRT_Device_MemoryStats_Args_STRUCT_SIZE = 0x00000000000000b9
end

# typedef PJRT_Error * PJRT_Device_MemoryStats ( PJRT_Device_MemoryStats_Args * args )
const PJRT_Device_MemoryStats = Cvoid

struct PJRT_Device_PoisonExecution_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    launch_id::Int32
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
    poisoned::Bool
end

@cenum __JL_Ctag_69::UInt32 begin
    PJRT_Device_PoisonExecution_Args_STRUCT_SIZE = 0x0000000000000031
end

# typedef PJRT_Error * PJRT_Device_PoisonExecution ( PJRT_Device_PoisonExecution_Args * args )
const PJRT_Device_PoisonExecution = Cvoid

mutable struct PJRT_AsyncTrackingEvent end

struct PJRT_Device_CreateAsyncTrackingEvent_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device::Ptr{PJRT_Device}
    description::Cstring
    description_size::Csize_t
    event::Ptr{PJRT_AsyncTrackingEvent}
end

@cenum __JL_Ctag_70::UInt32 begin
    PJRT_Device_CreateAsyncTrackingEvent_Args_STRUCT_SIZE = 0x0000000000000030
end

# typedef PJRT_Error * PJRT_Device_CreateAsyncTrackingEvent ( PJRT_Device_CreateAsyncTrackingEvent_Args * args )
const PJRT_Device_CreateAsyncTrackingEvent = Cvoid

struct PJRT_AsyncTrackingEvent_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    event::Ptr{PJRT_AsyncTrackingEvent}
end

@cenum __JL_Ctag_71::UInt32 begin
    PJRT_AsyncTrackingEvent_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_AsyncTrackingEvent_Destroy ( PJRT_AsyncTrackingEvent_Destroy_Args * args )
const PJRT_AsyncTrackingEvent_Destroy = Cvoid

struct PJRT_Memory_Id_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    id::Cint
end

@cenum __JL_Ctag_72::UInt32 begin
    PJRT_Memory_Id_Args_STRUCT_SIZE = 0x000000000000001c
end

# typedef PJRT_Error * PJRT_Memory_Id ( PJRT_Memory_Id_Args * args )
const PJRT_Memory_Id = Cvoid

struct PJRT_Memory_Kind_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    kind::Cstring
    kind_size::Csize_t
end

@cenum __JL_Ctag_73::UInt32 begin
    PJRT_Memory_Kind_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Memory_Kind ( PJRT_Memory_Kind_Args * args )
const PJRT_Memory_Kind = Cvoid

struct PJRT_Memory_Kind_Id_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    kind_id::Cint
end

@cenum __JL_Ctag_74::UInt32 begin
    PJRT_Memory_Kind_Id_Args_STRUCT_SIZE = 0x000000000000001c
end

# typedef PJRT_Error * PJRT_Memory_Kind_Id ( PJRT_Memory_Kind_Id_Args * args )
const PJRT_Memory_Kind_Id = Cvoid

struct PJRT_Memory_DebugString_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    debug_string::Cstring
    debug_string_size::Csize_t
end

@cenum __JL_Ctag_75::UInt32 begin
    PJRT_Memory_DebugString_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Memory_DebugString ( PJRT_Memory_DebugString_Args * args )
const PJRT_Memory_DebugString = Cvoid

struct PJRT_Memory_ToString_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    to_string::Cstring
    to_string_size::Csize_t
end

@cenum __JL_Ctag_76::UInt32 begin
    PJRT_Memory_ToString_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Memory_ToString ( PJRT_Memory_ToString_Args * args )
const PJRT_Memory_ToString = Cvoid

struct PJRT_Memory_AddressableByDevices_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory::Ptr{PJRT_Memory}
    devices::Ptr{Ptr{PJRT_Device}}
    num_devices::Csize_t
end

@cenum __JL_Ctag_77::UInt32 begin
    PJRT_Memory_AddressableByDevices_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Memory_AddressableByDevices ( PJRT_Memory_AddressableByDevices_Args * args )
const PJRT_Memory_AddressableByDevices = Cvoid

mutable struct PJRT_ExecuteContext end

struct PJRT_ExecuteContext_Create_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    context::Ptr{PJRT_ExecuteContext}
end

@cenum __JL_Ctag_78::UInt32 begin
    PJRT_ExecuteContext_Create_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_ExecuteContext_Create ( PJRT_ExecuteContext_Create_Args * args )
const PJRT_ExecuteContext_Create = Cvoid

struct PJRT_ExecuteContext_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    context::Ptr{PJRT_ExecuteContext}
end

@cenum __JL_Ctag_79::UInt32 begin
    PJRT_ExecuteContext_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_ExecuteContext_Destroy ( PJRT_ExecuteContext_Destroy_Args * args )
const PJRT_ExecuteContext_Destroy = Cvoid

struct PJRT_Executable_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
end

@cenum __JL_Ctag_80::UInt32 begin
    PJRT_Executable_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Executable_Destroy ( PJRT_Executable_Destroy_Args * args )
const PJRT_Executable_Destroy = Cvoid

struct PJRT_LoadedExecutable_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
end

@cenum __JL_Ctag_81::UInt32 begin
    PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_LoadedExecutable_Destroy ( PJRT_LoadedExecutable_Destroy_Args * args )
const PJRT_LoadedExecutable_Destroy = Cvoid

struct PJRT_LoadedExecutable_GetExecutable_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    loaded_executable::Ptr{PJRT_LoadedExecutable}
    executable::Ptr{PJRT_Executable}
end

@cenum __JL_Ctag_82::UInt32 begin
    PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE = 0x0000000000000020
end

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
end

@cenum __JL_Ctag_83::UInt32 begin
    PJRT_LoadedExecutable_GetDeviceAssignment_Args_STRUCT_SIZE = 0x0000000000000038
end

# typedef PJRT_Error * PJRT_LoadedExecutable_GetDeviceAssignment ( PJRT_LoadedExecutable_GetDeviceAssignment_Args * args )
const PJRT_LoadedExecutable_GetDeviceAssignment = Cvoid

struct PJRT_Executable_Name_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    executable_name::Cstring
    executable_name_size::Csize_t
end

@cenum __JL_Ctag_84::UInt32 begin
    PJRT_Executable_Name_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Executable_Name ( PJRT_Executable_Name_Args * args )
const PJRT_Executable_Name = Cvoid

struct PJRT_Executable_NumReplicas_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_replicas::Csize_t
end

@cenum __JL_Ctag_85::UInt32 begin
    PJRT_Executable_NumReplicas_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Executable_NumReplicas ( PJRT_Executable_NumReplicas_Args * args )
const PJRT_Executable_NumReplicas = Cvoid

struct PJRT_Executable_NumPartitions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_partitions::Csize_t
end

@cenum __JL_Ctag_86::UInt32 begin
    PJRT_Executable_NumPartitions_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Executable_NumPartitions ( PJRT_Executable_NumPartitions_Args * args )
const PJRT_Executable_NumPartitions = Cvoid

struct PJRT_LoadedExecutable_AddressableDevices_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    addressable_devices::Ptr{Ptr{PJRT_Device}}
    num_addressable_devices::Csize_t
end

@cenum __JL_Ctag_87::UInt32 begin
    PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_LoadedExecutable_AddressableDevices ( PJRT_LoadedExecutable_AddressableDevices_Args * args )
const PJRT_LoadedExecutable_AddressableDevices = Cvoid

struct PJRT_Executable_OptimizedProgram_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    program::Ptr{PJRT_Program}
end

@cenum __JL_Ctag_88::UInt32 begin
    PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Executable_OptimizedProgram ( PJRT_Executable_OptimizedProgram_Args * args )
const PJRT_Executable_OptimizedProgram = Cvoid

struct PJRT_LoadedExecutable_Delete_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
end

@cenum __JL_Ctag_89::UInt32 begin
    PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_LoadedExecutable_Delete ( PJRT_LoadedExecutable_Delete_Args * args )
const PJRT_LoadedExecutable_Delete = Cvoid

struct PJRT_LoadedExecutable_IsDeleted_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    is_deleted::Bool
end

@cenum __JL_Ctag_90::UInt32 begin
    PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE = 0x0000000000000019
end

# typedef PJRT_Error * PJRT_LoadedExecutable_IsDeleted ( PJRT_LoadedExecutable_IsDeleted_Args * args )
const PJRT_LoadedExecutable_IsDeleted = Cvoid

struct PJRT_Chunk
    data::Ptr{Cvoid}
    size::Csize_t
    deleter::Ptr{Cvoid}
    deleter_arg::Ptr{Cvoid}
end

mutable struct PJRT_CopyToDeviceStream end

mutable struct PJRT_TransferMetadata end

# typedef PJRT_Error * ( * PJRT_SendCallback ) ( PJRT_Chunk * chunk , PJRT_CallbackError * callback_error , size_t total_size_in_bytes , bool done , void * user_arg )
const PJRT_SendCallback = Ptr{Cvoid}

# typedef void ( * PJRT_RecvCallback ) ( PJRT_CopyToDeviceStream * stream , void * user_arg )
const PJRT_RecvCallback = Ptr{Cvoid}

struct PJRT_SendCallbackInfo
    channel_id::Int64
    user_arg::Ptr{Cvoid}
    send_callback::PJRT_SendCallback
end

@cenum __JL_Ctag_91::UInt32 begin
    PJRT_SendCallbackInfo_STRUCT_SIZE = 0x0000000000000018
end

struct PJRT_RecvCallbackInfo
    channel_id::Int64
    user_arg::Ptr{Cvoid}
    recv_callback::PJRT_RecvCallback
end

@cenum __JL_Ctag_92::UInt32 begin
    PJRT_RecvCallbackInfo_STRUCT_SIZE = 0x0000000000000018
end

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
end

@cenum __JL_Ctag_93::UInt32 begin
    PJRT_ExecuteOptions_STRUCT_SIZE = 0x0000000000000070
end

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
end

@cenum __JL_Ctag_94::UInt32 begin
    PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE = 0x0000000000000050
end

# typedef PJRT_Error * PJRT_LoadedExecutable_Execute ( PJRT_LoadedExecutable_Execute_Args * args )
const PJRT_LoadedExecutable_Execute = Cvoid

struct PJRT_Executable_NumOutputs_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_outputs::Csize_t
end

@cenum __JL_Ctag_95::UInt32 begin
    PJRT_Executable_NumOutputs_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Executable_NumOutputs ( PJRT_Executable_NumOutputs_Args * args )
const PJRT_Executable_NumOutputs = Cvoid

struct PJRT_Executable_SizeOfGeneratedCodeInBytes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    size_in_bytes::Int64
end

@cenum __JL_Ctag_96::UInt32 begin
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Executable_SizeOfGeneratedCodeInBytes ( PJRT_Executable_SizeOfGeneratedCodeInBytes_Args * args )
const PJRT_Executable_SizeOfGeneratedCodeInBytes = Cvoid

struct PJRT_Executable_Fingerprint_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    executable_fingerprint::Cstring
    executable_fingerprint_size::Csize_t
end

@cenum __JL_Ctag_97::UInt32 begin
    PJRT_Executable_Fingerprint_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Executable_Fingerprint ( PJRT_Executable_Fingerprint_Args * args )
const PJRT_Executable_Fingerprint = Cvoid

struct PJRT_Executable_GetCostAnalysis_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_properties::Csize_t
    properties::Ptr{PJRT_NamedValue}
end

@cenum __JL_Ctag_98::UInt32 begin
    PJRT_Executable_GetCostAnalysis_Args_STRUCT_SIZE = 0x0000000000000028
end

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
end

@cenum __JL_Ctag_99::UInt32 begin
    PJRT_Executable_GetCompiledMemoryStats_Args_STRUCT_SIZE = 0x0000000000000078
end

# typedef PJRT_Error * PJRT_Executable_GetCompiledMemoryStats ( PJRT_Executable_GetCompiledMemoryStats_Args * args )
const PJRT_Executable_GetCompiledMemoryStats = Cvoid

struct PJRT_Executable_OutputElementTypes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    output_types::Ptr{PJRT_Buffer_Type}
    num_output_types::Csize_t
end

@cenum __JL_Ctag_100::UInt32 begin
    PJRT_Executable_OutputElementTypes_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Executable_OutputElementTypes ( PJRT_Executable_OutputElementTypes_Args * args )
const PJRT_Executable_OutputElementTypes = Cvoid

struct PJRT_Executable_OutputDimensions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_outputs::Csize_t
    dims::Ptr{Int64}
    dim_sizes::Ptr{Csize_t}
end

@cenum __JL_Ctag_101::UInt32 begin
    PJRT_Executable_OutputDimensions_Args_STRUCT_SIZE = 0x0000000000000030
end

# typedef PJRT_Error * PJRT_Executable_OutputDimensions ( PJRT_Executable_OutputDimensions_Args * args )
const PJRT_Executable_OutputDimensions = Cvoid

struct PJRT_Executable_OutputMemoryKinds_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_outputs::Csize_t
    memory_kinds::Ptr{Cstring}
    memory_kind_sizes::Ptr{Csize_t}
end

@cenum __JL_Ctag_102::UInt32 begin
    PJRT_Executable_OutputMemoryKinds_Args_STRUCT_SIZE = 0x0000000000000030
end

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
end

@cenum __JL_Ctag_103::UInt32 begin
    PJRT_Executable_Serialize_Args_STRUCT_SIZE = 0x0000000000000038
end

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
end

@cenum __JL_Ctag_104::UInt32 begin
    PJRT_Executable_GetCompileOptions_Args_STRUCT_SIZE = 0x0000000000000038
end

# typedef PJRT_Error * PJRT_Executable_GetCompileOptions ( PJRT_Executable_GetCompileOptions_Args * args )
const PJRT_Executable_GetCompileOptions = Cvoid

struct PJRT_Executable_DeserializeAndLoad_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    client::Ptr{PJRT_Client}
    serialized_executable::Cstring
    serialized_executable_size::Csize_t
    loaded_executable::Ptr{PJRT_LoadedExecutable}
    overridden_serialized_compile_options::Cstring
    overridden_serialized_compile_options_size::Csize_t
end

@cenum __JL_Ctag_105::UInt32 begin
    PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE = 0x0000000000000040
end

# typedef PJRT_Error * PJRT_Executable_DeserializeAndLoad ( PJRT_Executable_DeserializeAndLoad_Args * args )
const PJRT_Executable_DeserializeAndLoad = Cvoid

struct PJRT_LoadedExecutable_Fingerprint_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_LoadedExecutable}
    executable_fingerprint::Cstring
    executable_fingerprint_size::Csize_t
end

@cenum __JL_Ctag_106::UInt32 begin
    PJRT_LoadedExecutable_Fingerprint_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_LoadedExecutable_Fingerprint ( PJRT_LoadedExecutable_Fingerprint_Args * args )
const PJRT_LoadedExecutable_Fingerprint = Cvoid

struct PJRT_Buffer_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
end

@cenum __JL_Ctag_107::UInt32 begin
    PJRT_Buffer_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Buffer_Destroy ( PJRT_Buffer_Destroy_Args * args )
const PJRT_Buffer_Destroy = Cvoid

struct PJRT_Buffer_ElementType_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    type::PJRT_Buffer_Type
end

@cenum __JL_Ctag_108::UInt32 begin
    PJRT_Buffer_ElementType_Args_STRUCT_SIZE = 0x000000000000001c
end

# typedef PJRT_Error * PJRT_Buffer_ElementType ( PJRT_Buffer_ElementType_Args * args )
const PJRT_Buffer_ElementType = Cvoid

struct PJRT_Buffer_Dimensions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    dims::Ptr{Int64}
    num_dims::Csize_t
end

@cenum __JL_Ctag_109::UInt32 begin
    PJRT_Buffer_Dimensions_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Buffer_Dimensions ( PJRT_Buffer_Dimensions_Args * args )
const PJRT_Buffer_Dimensions = Cvoid

struct PJRT_Buffer_UnpaddedDimensions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    unpadded_dims::Ptr{Int64}
    num_dims::Csize_t
end

@cenum __JL_Ctag_110::UInt32 begin
    PJRT_Buffer_UnpaddedDimensions_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Buffer_UnpaddedDimensions ( PJRT_Buffer_UnpaddedDimensions_Args * args )
const PJRT_Buffer_UnpaddedDimensions = Cvoid

struct PJRT_Buffer_DynamicDimensionIndices_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    dynamic_dim_indices::Ptr{Csize_t}
    num_dynamic_dims::Csize_t
end

@cenum __JL_Ctag_111::UInt32 begin
    PJRT_Buffer_DynamicDimensionIndices_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Buffer_DynamicDimensionIndices ( PJRT_Buffer_DynamicDimensionIndices_Args * args )
const PJRT_Buffer_DynamicDimensionIndices = Cvoid

struct PJRT_Buffer_GetMemoryLayout_Args
    data::NTuple{104, UInt8}
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
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::PJRT_Buffer_GetMemoryLayout_Args, private::Bool = false)
    (:struct_size, :extension_start, :buffer, :layout, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

@cenum __JL_Ctag_112::UInt32 begin
    PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE = 0x0000000000000068
end

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
end

@cenum __JL_Ctag_113::UInt32 begin
    PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE = 0x0000000000000038
end

# typedef PJRT_Error * PJRT_Buffer_ToHostBuffer ( PJRT_Buffer_ToHostBuffer_Args * args )
const PJRT_Buffer_ToHostBuffer = Cvoid

struct PJRT_Buffer_OnDeviceSizeInBytes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    on_device_size_in_bytes::Csize_t
end

@cenum __JL_Ctag_114::UInt32 begin
    PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Buffer_OnDeviceSizeInBytes ( PJRT_Buffer_OnDeviceSizeInBytes_Args * args )
const PJRT_Buffer_OnDeviceSizeInBytes = Cvoid

struct PJRT_Buffer_Delete_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
end

@cenum __JL_Ctag_115::UInt32 begin
    PJRT_Buffer_Delete_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Buffer_Delete ( PJRT_Buffer_Delete_Args * args )
const PJRT_Buffer_Delete = Cvoid

struct PJRT_Buffer_IsDeleted_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    is_deleted::Bool
end

@cenum __JL_Ctag_116::UInt32 begin
    PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE = 0x0000000000000019
end

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
end

@cenum __JL_Ctag_117::UInt32 begin
    PJRT_Buffer_CopyRawToHost_Args_STRUCT_SIZE = 0x0000000000000038
end

# typedef PJRT_Error * PJRT_Buffer_CopyRawToHost ( PJRT_Buffer_CopyRawToHost_Args * args )
const PJRT_Buffer_CopyRawToHost = Cvoid

struct PJRT_Buffer_CopyRawToHostFuture_Callback_Args
    struct_size::Csize_t
    callback_data::Ptr{Cvoid}
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
    dst::Ptr{Cvoid}
end

@cenum __JL_Ctag_118::UInt32 begin
    PJRT_Buffer_CopyRawToHostFuture_Callback_Args_STRUCT_SIZE = 0x0000000000000030
end

struct PJRT_Buffer_CopyRawToHostFuture_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    offset::Int64
    transfer_size::Int64
    event::Ptr{PJRT_Event}
    callback_data::Ptr{Cvoid}
    future_ready_callback::Ptr{Cvoid}
end

@cenum __JL_Ctag_119::UInt32 begin
    PJRT_Buffer_CopyRawToHostFuture_Args_STRUCT_SIZE = 0x0000000000000040
end

# typedef PJRT_Error * PJRT_Buffer_CopyRawToHostFuture ( PJRT_Buffer_CopyRawToHostFuture_Args * args )
const PJRT_Buffer_CopyRawToHostFuture = Cvoid

struct PJRT_Buffer_CopyToDevice_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    dst_device::Ptr{PJRT_Device}
    dst_buffer::Ptr{PJRT_Buffer}
end

@cenum __JL_Ctag_120::UInt32 begin
    PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Buffer_CopyToDevice ( PJRT_Buffer_CopyToDevice_Args * args )
const PJRT_Buffer_CopyToDevice = Cvoid

struct PJRT_Buffer_CopyToMemory_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    dst_memory::Ptr{PJRT_Memory}
    dst_buffer::Ptr{PJRT_Buffer}
end

@cenum __JL_Ctag_121::UInt32 begin
    PJRT_Buffer_CopyToMemory_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Buffer_CopyToMemory ( PJRT_Buffer_CopyToMemory_Args * args )
const PJRT_Buffer_CopyToMemory = Cvoid

struct PJRT_Buffer_IsOnCpu_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    is_on_cpu::Bool
end

@cenum __JL_Ctag_122::UInt32 begin
    PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE = 0x0000000000000019
end

# typedef PJRT_Error * PJRT_Buffer_IsOnCpu ( PJRT_Buffer_IsOnCpu_Args * args )
const PJRT_Buffer_IsOnCpu = Cvoid

struct PJRT_Buffer_Device_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    device::Ptr{PJRT_Device}
end

@cenum __JL_Ctag_123::UInt32 begin
    PJRT_Buffer_Device_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Buffer_Device ( PJRT_Buffer_Device_Args * args )
const PJRT_Buffer_Device = Cvoid

struct PJRT_Buffer_Memory_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    memory::Ptr{PJRT_Memory}
end

@cenum __JL_Ctag_124::UInt32 begin
    PJRT_Buffer_Memory_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Buffer_Memory ( PJRT_Buffer_Memory_Args * args )
const PJRT_Buffer_Memory = Cvoid

struct PJRT_Buffer_ReadyEvent_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    event::Ptr{PJRT_Event}
end

@cenum __JL_Ctag_125::UInt32 begin
    PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Buffer_ReadyEvent ( PJRT_Buffer_ReadyEvent_Args * args )
const PJRT_Buffer_ReadyEvent = Cvoid

struct PJRT_Buffer_UnsafePointer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    buffer_pointer::Csize_t
end

@cenum __JL_Ctag_126::UInt32 begin
    PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Buffer_UnsafePointer ( PJRT_Buffer_UnsafePointer_Args * args )
const PJRT_Buffer_UnsafePointer = Cvoid

struct PJRT_Buffer_IncreaseExternalReferenceCount_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
end

@cenum __JL_Ctag_127::UInt32 begin
    PJRT_Buffer_IncreaseExternalReferenceCount_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Buffer_IncreaseExternalReferenceCount ( PJRT_Buffer_IncreaseExternalReferenceCount_Args * args )
const PJRT_Buffer_IncreaseExternalReferenceCount = Cvoid

struct PJRT_Buffer_DecreaseExternalReferenceCount_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
end

@cenum __JL_Ctag_128::UInt32 begin
    PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Buffer_DecreaseExternalReferenceCount ( PJRT_Buffer_DecreaseExternalReferenceCount_Args * args )
const PJRT_Buffer_DecreaseExternalReferenceCount = Cvoid

struct PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    device_memory_ptr::Ptr{Cvoid}
end

@cenum __JL_Ctag_129::UInt32 begin
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Buffer_OpaqueDeviceMemoryDataPointer ( PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args * args )
const PJRT_Buffer_OpaqueDeviceMemoryDataPointer = Cvoid

struct PJRT_Buffer_DonateWithControlDependency_Callback_Args
    struct_size::Csize_t
    callback_data::Ptr{Cvoid}
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
end

@cenum __JL_Ctag_130::UInt32 begin
    PJRT_Buffer_DonateWithControlDependency_Callback_Args_STRUCT_SIZE = 0x0000000000000028
end

struct PJRT_Buffer_DonateWithControlDependency_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    callback_data::Ptr{Cvoid}
    dependency_ready_callback::Ptr{Cvoid}
    out_buffer::Ptr{PJRT_Buffer}
end

@cenum __JL_Ctag_131::UInt32 begin
    PJRT_Buffer_DonateWithControlDependency_Args_STRUCT_SIZE = 0x0000000000000030
end

# typedef PJRT_Error * PJRT_Buffer_DonateWithControlDependency ( PJRT_Buffer_DonateWithControlDependency_Args * args )
const PJRT_Buffer_DonateWithControlDependency = Cvoid

struct PJRT_CopyToDeviceStream_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    stream::Ptr{PJRT_CopyToDeviceStream}
end

@cenum __JL_Ctag_132::UInt32 begin
    PJRT_CopyToDeviceStream_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_CopyToDeviceStream_Destroy ( PJRT_CopyToDeviceStream_Destroy_Args * args )
const PJRT_CopyToDeviceStream_Destroy = Cvoid

struct PJRT_CopyToDeviceStream_AddChunk_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    stream::Ptr{PJRT_CopyToDeviceStream}
    chunk::Ptr{PJRT_Chunk}
    transfer_complete::Ptr{PJRT_Event}
end

@cenum __JL_Ctag_133::UInt32 begin
    PJRT_CopyToDeviceStream_AddChunk_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_CopyToDeviceStream_AddChunk ( PJRT_CopyToDeviceStream_AddChunk_Args * args )
const PJRT_CopyToDeviceStream_AddChunk = Cvoid

struct PJRT_CopyToDeviceStream_TotalBytes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    stream::Ptr{PJRT_CopyToDeviceStream}
    total_bytes::Int64
end

@cenum __JL_Ctag_134::UInt32 begin
    PJRT_CopyToDeviceStream_TotalBytes_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_CopyToDeviceStream_TotalBytes ( PJRT_CopyToDeviceStream_TotalBytes_Args * args )
const PJRT_CopyToDeviceStream_TotalBytes = Cvoid

struct PJRT_CopyToDeviceStream_GranuleSize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    stream::Ptr{PJRT_CopyToDeviceStream}
    granule_size_in_bytes::Int64
end

@cenum __JL_Ctag_135::UInt32 begin
    PJRT_CopyToDeviceStream_GranuleSize_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_CopyToDeviceStream_GranuleSize ( PJRT_CopyToDeviceStream_GranuleSize_Args * args )
const PJRT_CopyToDeviceStream_GranuleSize = Cvoid

struct PJRT_CopyToDeviceStream_CurrentBytes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    stream::Ptr{PJRT_CopyToDeviceStream}
    current_bytes::Int64
end

@cenum __JL_Ctag_136::UInt32 begin
    PJRT_CopyToDeviceStream_CurrentBytes_Args_STRUCT_SIZE = 0x0000000000000020
end

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
end

@cenum __JL_Ctag_137::UInt32 begin
    PJRT_TopologyDescription_Create_Args_STRUCT_SIZE = 0x0000000000000038
end

# typedef PJRT_Error * PJRT_TopologyDescription_Create ( PJRT_TopologyDescription_Create_Args * args )
const PJRT_TopologyDescription_Create = Cvoid

struct PJRT_TopologyDescription_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
end

@cenum __JL_Ctag_138::UInt32 begin
    PJRT_TopologyDescription_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_TopologyDescription_Destroy ( PJRT_TopologyDescription_Destroy_Args * args )
const PJRT_TopologyDescription_Destroy = Cvoid

struct PJRT_TopologyDescription_PlatformVersion_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    platform_version::Cstring
    platform_version_size::Csize_t
end

@cenum __JL_Ctag_139::UInt32 begin
    PJRT_TopologyDescription_PlatformVersion_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_TopologyDescription_PlatformVersion ( PJRT_TopologyDescription_PlatformVersion_Args * args )
const PJRT_TopologyDescription_PlatformVersion = Cvoid

struct PJRT_TopologyDescription_PlatformName_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    platform_name::Cstring
    platform_name_size::Csize_t
end

@cenum __JL_Ctag_140::UInt32 begin
    PJRT_TopologyDescription_PlatformName_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_TopologyDescription_PlatformName ( PJRT_TopologyDescription_PlatformName_Args * args )
const PJRT_TopologyDescription_PlatformName = Cvoid

struct PJRT_TopologyDescription_GetDeviceDescriptions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    descriptions::Ptr{Ptr{PJRT_DeviceDescription}}
    num_descriptions::Csize_t
end

@cenum __JL_Ctag_141::UInt32 begin
    PJRT_TopologyDescription_GetDeviceDescriptions_Args_STRUCT_SIZE = 0x0000000000000028
end

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
end

@cenum __JL_Ctag_142::UInt32 begin
    PJRT_TopologyDescription_Serialize_Args_STRUCT_SIZE = 0x0000000000000038
end

# typedef PJRT_Error * PJRT_TopologyDescription_Serialize ( PJRT_TopologyDescription_Serialize_Args * args )
const PJRT_TopologyDescription_Serialize = Cvoid

struct PJRT_TopologyDescription_Deserialize_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    serialized_topology::Cstring
    serialized_topology_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
end

@cenum __JL_Ctag_143::UInt32 begin
    PJRT_TopologyDescription_Deserialize_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_TopologyDescription_Deserialize ( PJRT_TopologyDescription_Deserialize_Args * args )
const PJRT_TopologyDescription_Deserialize = Cvoid

struct PJRT_TopologyDescription_Attributes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    attributes::Ptr{PJRT_NamedValue}
    num_attributes::Csize_t
end

@cenum __JL_Ctag_144::UInt32 begin
    PJRT_TopologyDescription_Attributes_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_TopologyDescription_Attributes ( PJRT_TopologyDescription_Attributes_Args * args )
const PJRT_TopologyDescription_Attributes = Cvoid

struct PJRT_Compile_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    topology::Ptr{PJRT_TopologyDescription}
    program::Ptr{PJRT_Program}
    compile_options::Cstring
    compile_options_size::Csize_t
    client::Ptr{PJRT_Client}
    executable::Ptr{PJRT_Executable}
end

@cenum __JL_Ctag_145::UInt32 begin
    PJRT_Compile_Args_STRUCT_SIZE = 0x0000000000000040
end

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
    PJRT_Executable_SizeOfGeneratedCodeInBytes::Ptr{PJRT_Executable_SizeOfGeneratedCodeInBytes}
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
    PJRT_Buffer_IncreaseExternalReferenceCount::Ptr{PJRT_Buffer_IncreaseExternalReferenceCount}
    PJRT_Buffer_DecreaseExternalReferenceCount::Ptr{PJRT_Buffer_DecreaseExternalReferenceCount}
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer::Ptr{PJRT_Buffer_OpaqueDeviceMemoryDataPointer}
    PJRT_CopyToDeviceStream_Destroy::Ptr{PJRT_CopyToDeviceStream_Destroy}
    PJRT_CopyToDeviceStream_AddChunk::Ptr{PJRT_CopyToDeviceStream_AddChunk}
    PJRT_CopyToDeviceStream_TotalBytes::Ptr{PJRT_CopyToDeviceStream_TotalBytes}
    PJRT_CopyToDeviceStream_GranuleSize::Ptr{PJRT_CopyToDeviceStream_GranuleSize}
    PJRT_CopyToDeviceStream_CurrentBytes::Ptr{PJRT_CopyToDeviceStream_CurrentBytes}
    PJRT_TopologyDescription_Create::Ptr{PJRT_TopologyDescription_Create}
    PJRT_TopologyDescription_Destroy::Ptr{PJRT_TopologyDescription_Destroy}
    PJRT_TopologyDescription_PlatformName::Ptr{PJRT_TopologyDescription_PlatformName}
    PJRT_TopologyDescription_PlatformVersion::Ptr{PJRT_TopologyDescription_PlatformVersion}
    PJRT_TopologyDescription_GetDeviceDescriptions::Ptr{PJRT_TopologyDescription_GetDeviceDescriptions}
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
    PJRT_AsyncHostToDeviceTransferManager_Destroy::Ptr{PJRT_AsyncHostToDeviceTransferManager_Destroy}
    PJRT_AsyncHostToDeviceTransferManager_TransferData::Ptr{PJRT_AsyncHostToDeviceTransferManager_TransferData}
    PJRT_Client_CreateBuffersForAsyncHostToDevice::Ptr{PJRT_Client_CreateBuffersForAsyncHostToDevice}
    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer::Ptr{PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer}
    PJRT_AsyncHostToDeviceTransferManager_Device::Ptr{PJRT_AsyncHostToDeviceTransferManager_Device}
    PJRT_AsyncHostToDeviceTransferManager_BufferCount::Ptr{PJRT_AsyncHostToDeviceTransferManager_BufferCount}
    PJRT_AsyncHostToDeviceTransferManager_BufferSize::Ptr{PJRT_AsyncHostToDeviceTransferManager_BufferSize}
    PJRT_AsyncHostToDeviceTransferManager_SetBufferError::Ptr{PJRT_AsyncHostToDeviceTransferManager_SetBufferError}
    PJRT_AsyncHostToDeviceTransferManager_AddMetadata::Ptr{PJRT_AsyncHostToDeviceTransferManager_AddMetadata}
    PJRT_Client_DmaMap::Ptr{PJRT_Client_DmaMap}
    PJRT_Client_DmaUnmap::Ptr{PJRT_Client_DmaUnmap}
    PJRT_Client_CreateUninitializedBuffer::Ptr{PJRT_Client_CreateUninitializedBuffer}
    PJRT_Client_UpdateGlobalProcessInfo::Ptr{PJRT_Client_UpdateGlobalProcessInfo}
    PJRT_TopologyDescription_Deserialize::Ptr{PJRT_TopologyDescription_Deserialize}
    PJRT_Client_CreateAliasBuffer::Ptr{PJRT_Client_CreateAliasBuffer}
    PJRT_Client_FulfillAliasBuffer::Ptr{PJRT_Client_FulfillAliasBuffer}
    PJRT_LoadedExecutable_GetDeviceAssignment::Ptr{PJRT_LoadedExecutable_GetDeviceAssignment}
    PJRT_Client_CreateErrorBuffer::Ptr{PJRT_Client_CreateErrorBuffer}
    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral::Ptr{PJRT_AsyncHostToDeviceTransferManager_TransferLiteral}
    PJRT_Buffer_CopyRawToHostFuture::Ptr{PJRT_Buffer_CopyRawToHostFuture}
    PJRT_Device_PoisonExecution::Ptr{PJRT_Device_PoisonExecution}
    PJRT_Device_CreateAsyncTrackingEvent::Ptr{PJRT_Device_CreateAsyncTrackingEvent}
    PJRT_AsyncTrackingEvent_Destroy::Ptr{PJRT_AsyncTrackingEvent_Destroy}
    PJRT_Executable_GetCompileOptions::Ptr{PJRT_Executable_GetCompileOptions}
    PJRT_Buffer_DonateWithControlDependency::Ptr{PJRT_Buffer_DonateWithControlDependency}
    PJRT_Event_Create::Ptr{PJRT_Event_Create}
    PJRT_Event_Set::Ptr{PJRT_Event_Set}
end

@cenum __JL_Ctag_146::UInt32 begin
    PJRT_Api_STRUCT_SIZE = 0x0000000000000428
end

@cenum PJRT_Callback_Type::UInt32 begin
    PJRT_Callback_Type_Unknown = 0x0000000000000000
    PJRT_Callback_Type_Tpu_SliceBuilder = 0x0000000000000001
    PJRT_Callback_Type_Prefatal = 0x0000000000000002
end

@cenum PJRT_Callback_Tpu_SliceFailureType::Int32 begin
    SLICE_FAILURE_UNKNOWN = 0x0000000000000000
    SLICE_FAILURE_INIT_ERROR = 0x0000000000000001
    SLICE_FAILURE_WORKER_UNAVAILABLE = 0x0000000000000002
    SLICE_FAILURE_FLAPPING_TASK_ERROR = 0x0000000000000003
    SLICE_FAILURE_SW_INJECT_ERROR = 0x0000000000000004
    SLICE_FAILURE_CHIP_DRIVER_ERROR = 0x0000000000000005
end

struct PJRT_Callback_Tpu_SliceBuilderArgs
    struct_size::Csize_t
    failure_type::PJRT_Callback_Tpu_SliceFailureType
end

@cenum __JL_Ctag_147::UInt32 begin
    PJRT_Callback_Tpu_SliceBuilderArgs_STRUCT_SIZE = 0x000000000000000c
end

struct PJRT_Callback_PrefatalArgs
    struct_size::Csize_t
    error_code::PJRT_Error_Code
    error_message::Cstring
    error_message_size::Csize_t
end

@cenum __JL_Ctag_148::UInt32 begin
    PJRT_Callback_PrefatalArgs_STRUCT_SIZE = 0x0000000000000020
end

# typedef void PJRT_Callback_Function ( void * args , void * user_arg )
const PJRT_Callback_Function = Cvoid

struct PJRT_Callback_RegisterCallback_Args
    struct_size::Csize_t
    client::Ptr{PJRT_Client}
    type::PJRT_Callback_Type
    callback::Ptr{PJRT_Callback_Function}
    user_arg::Ptr{Cvoid}
end

@cenum __JL_Ctag_149::UInt32 begin
    PJRT_Callback_RegisterCallback_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Register_Callback ( PJRT_Callback_RegisterCallback_Args * args )
const PJRT_Register_Callback = Cvoid

struct PJRT_Callback_InvokeCallback_Args
    struct_size::Csize_t
    client::Ptr{PJRT_Client}
    type::PJRT_Callback_Type
    args::Ptr{Cvoid}
end

@cenum __JL_Ctag_150::UInt32 begin
    PJRT_Callback_InvokeCallback_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Callback_InvokeCallback ( PJRT_Callback_InvokeCallback_Args * args )
const PJRT_Callback_InvokeCallback = Cvoid

struct PJRT_Callback_Extension
    base::PJRT_Extension_Base
    register_callback::Ptr{PJRT_Register_Callback}
    invoke_callback::Ptr{PJRT_Callback_InvokeCallback}
end

@cenum __JL_Ctag_151::UInt32 begin
    PJRT_Callback_Extension_STRUCT_SIZE = 0x0000000000000028
end

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
end

@cenum __JL_Ctag_298::UInt32 begin
    PJRT_Register_Custom_Partitioner_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_Register_Custom_Partitioner ( PJRT_Register_Custom_Partitioner_Args * args )
const PJRT_Register_Custom_Partitioner = Cvoid

struct PJRT_Register_Batch_Partitionable_Args
    struct_size::Csize_t
    name::Cstring
    name_size::Csize_t
end

@cenum __JL_Ctag_299::UInt32 begin
    PJRT_Register_Batch_Partitionable_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Register_Batch_Partitionable ( PJRT_Register_Batch_Partitionable_Args * args )
const PJRT_Register_Batch_Partitionable = Cvoid

struct PJRT_Custom_Partitioner_Extension
    base::PJRT_Extension_Base
    register_custom_partitioner::Ptr{PJRT_Register_Custom_Partitioner}
    register_batch_partitionable::Ptr{PJRT_Register_Batch_Partitionable}
end

@cenum __JL_Ctag_300::UInt32 begin
    PJRT_Custom_Partitioner_Extension_STRUCT_SIZE = 0x0000000000000028
end

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
end

@cenum __JL_Ctag_447::UInt32 begin
    PJRT_FFI_Type_Register_Args_STRUCT_SIZE = 0x0000000000000030
end

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
end

@cenum __JL_Ctag_448::UInt32 begin
    PJRT_FFI_UserData_Add_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_FFI_UserData_Add ( PJRT_FFI_UserData_Add_Args * args )
const PJRT_FFI_UserData_Add = Cvoid

@cenum PJRT_FFI_Handler_TraitsBits::UInt32 begin
    PJRT_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE = 0x0000000000000001
end

struct PJRT_FFI_Register_Handler_Args
    struct_size::Csize_t
    target_name::Cstring
    target_name_size::Csize_t
    handler::Ptr{Cvoid}
    platform_name::Cstring
    platform_name_size::Csize_t
    traits::PJRT_FFI_Handler_TraitsBits
end

@cenum __JL_Ctag_449::UInt32 begin
    PJRT_FFI_Register_Handler_Args_STRUCT_SIZE = 0x0000000000000034
end

# typedef PJRT_Error * PJRT_FFI_Register_Handler ( PJRT_FFI_Register_Handler_Args * args )
const PJRT_FFI_Register_Handler = Cvoid

struct PJRT_FFI_Extension
    base::PJRT_Extension_Base
    type_register::Ptr{PJRT_FFI_Type_Register}
    user_data_add::Ptr{PJRT_FFI_UserData_Add}
    register_handler::Ptr{PJRT_FFI_Register_Handler}
end

const PJRT_FFI = PJRT_FFI_Extension

@cenum __JL_Ctag_450::UInt32 begin
    PJRT_FFI_Extension_STRUCT_SIZE = 0x0000000000000030
end

struct PJRT_Gpu_Register_Custom_Call_Args
    struct_size::Csize_t
    function_name::Cstring
    function_name_size::Csize_t
    api_version::Cint
    handler_instantiate::Ptr{Cvoid}
    handler_prepare::Ptr{Cvoid}
    handler_initialize::Ptr{Cvoid}
    handler_execute::Ptr{Cvoid}
end

@cenum __JL_Ctag_597::UInt32 begin
    PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE = 0x0000000000000040
end

# typedef PJRT_Error * PJRT_Gpu_Register_Custom_Call ( PJRT_Gpu_Register_Custom_Call_Args * args )
const PJRT_Gpu_Register_Custom_Call = Cvoid

struct PJRT_Gpu_Custom_Call
    base::PJRT_Extension_Base
    custom_call::Ptr{PJRT_Gpu_Register_Custom_Call}
end

@cenum __JL_Ctag_598::UInt32 begin
    PJRT_Gpu_Custom_Call_STRUCT_SIZE = 0x0000000000000020
end

mutable struct PJRT_Layouts_MemoryLayout end

mutable struct PJRT_Layouts_SerializedLayout end

struct PJRT_Layouts_MemoryLayout_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    layout::Ptr{PJRT_Layouts_MemoryLayout}
end

@cenum __JL_Ctag_745::UInt32 begin
    PJRT_Layouts_MemoryLayout_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

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
end

@cenum __JL_Ctag_746::UInt32 begin
    PJRT_Layouts_MemoryLayout_Serialize_Args_STRUCT_SIZE = 0x0000000000000038
end

# typedef PJRT_Error * PJRT_Layouts_MemoryLayout_Serialize ( PJRT_Layouts_MemoryLayout_Serialize_Args * args )
const PJRT_Layouts_MemoryLayout_Serialize = Cvoid

struct PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    layout::Ptr{PJRT_Layouts_MemoryLayout}
end

@cenum __JL_Ctag_747::UInt32 begin
    PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args_STRUCT_SIZE = 0x0000000000000020
end

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
end

@cenum __JL_Ctag_748::UInt32 begin
    PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args_STRUCT_SIZE = 0x0000000000000038
end

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
end

@cenum __JL_Ctag_749::UInt32 begin
    PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args_STRUCT_SIZE = 0x0000000000000038
end

# typedef PJRT_Error * PJRT_Layouts_PJRT_Topology_GetDefaultLayout ( PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args * args )
const PJRT_Layouts_PJRT_Topology_GetDefaultLayout = Cvoid

struct PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    executable::Ptr{PJRT_Executable}
    num_outputs::Csize_t
    layouts::Ptr{Ptr{PJRT_Layouts_MemoryLayout}}
end

@cenum __JL_Ctag_750::UInt32 begin
    PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_Layouts_PJRT_Executable_GetOutputLayouts ( PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args * args )
const PJRT_Layouts_PJRT_Executable_GetOutputLayouts = Cvoid

struct PJRT_Layouts_Extension
    base::PJRT_Extension_Base
    PJRT_Layouts_MemoryLayout_Destroy::Ptr{PJRT_Layouts_MemoryLayout_Destroy}
    PJRT_Layouts_MemoryLayout_Serialize::Ptr{PJRT_Layouts_MemoryLayout_Serialize}
    PJRT_Layouts_PJRT_Client_GetDefaultLayout::Ptr{PJRT_Layouts_PJRT_Client_GetDefaultLayout}
    PJRT_Layouts_PJRT_Buffer_MemoryLayout::Ptr{PJRT_Layouts_PJRT_Buffer_MemoryLayout}
    PJRT_Layouts_PJRT_Topology_GetDefaultLayout::Ptr{PJRT_Layouts_PJRT_Topology_GetDefaultLayout}
    PJRT_Layouts_PJRT_Executable_GetOutputLayouts::Ptr{PJRT_Layouts_PJRT_Executable_GetOutputLayouts}
end

@cenum __JL_Ctag_751::UInt32 begin
    PJRT_Layouts_Extension_STRUCT_SIZE = 0x0000000000000048
end

mutable struct PJRT_MemoryDescription end

struct PJRT_DeviceDescription_MemoryDescriptions_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    device_description::Ptr{PJRT_DeviceDescription}
    memory_descriptions::Ptr{Ptr{PJRT_MemoryDescription}}
    num_memory_descriptions::Csize_t
    default_memory_index::Csize_t
end

@cenum __JL_Ctag_898::UInt32 begin
    PJRT_DeviceDescription_MemoryDescriptions_Args_STRUCT_SIZE = 0x0000000000000030
end

# typedef PJRT_Error * PJRT_DeviceDescription_MemoryDescriptions ( PJRT_DeviceDescription_MemoryDescriptions_Args * args )
const PJRT_DeviceDescription_MemoryDescriptions = Cvoid

struct PJRT_MemoryDescription_Kind_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    memory_description::Ptr{PJRT_MemoryDescription}
    kind::Cstring
    kind_size::Csize_t
    kind_id::Cint
end

@cenum __JL_Ctag_899::UInt32 begin
    PJRT_MemoryDescription_Kind_Args_STRUCT_SIZE = 0x000000000000002c
end

# typedef PJRT_Error * PJRT_MemoryDescription_Kind ( PJRT_MemoryDescription_Kind_Args * args )
const PJRT_MemoryDescription_Kind = Cvoid

struct PJRT_MemoryDescriptions_Extension
    base::PJRT_Extension_Base
    PJRT_DeviceDescription_MemoryDescriptions::Ptr{PJRT_DeviceDescription_MemoryDescriptions}
    PJRT_MemoryDescription_Kind::Ptr{PJRT_MemoryDescription_Kind}
end

@cenum __JL_Ctag_900::UInt32 begin
    PJRT_MemoryDescriptions_Extension_STRUCT_SIZE = 0x0000000000000028
end

struct PJRT_PhaseCompile_Get_Compiler_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    phase_compiler::Ptr{PJRT_PhaseCompiler}
end

@cenum __JL_Ctag_1047::UInt32 begin
    PJRT_PhaseCompile_Get_Compiler_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_PhaseCompile_Get_Compiler ( PJRT_PhaseCompile_Get_Compiler_Args * args )
const PJRT_PhaseCompile_Get_Compiler = Cvoid

struct PJRT_PhaseCompile_Destroy_Compiler_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    phase_compiler::Ptr{PJRT_PhaseCompiler}
end

@cenum __JL_Ctag_1048::UInt32 begin
    PJRT_PhaseCompile_Destroy_Compiler_Args_STRUCT_SIZE = 0x0000000000000018
end

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
end

@cenum __JL_Ctag_1049::UInt32 begin
    PJRT_PhaseCompile_Run_Phase_Args_STRUCT_SIZE = 0x0000000000000078
end

# typedef PJRT_Error * PJRT_PhaseCompile_Run_Phase ( PJRT_PhaseCompile_Run_Phase_Args * args )
const PJRT_PhaseCompile_Run_Phase = Cvoid

struct PJRT_PhaseCompile_Get_PhaseNames_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    phase_compiler::Ptr{PJRT_PhaseCompiler}
    phase_names::Ptr{Cstring}
    phase_names_sizes::Ptr{Csize_t}
    num_phase_names::Csize_t
end

@cenum __JL_Ctag_1050::UInt32 begin
    PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE = 0x0000000000000030
end

# typedef PJRT_Error * PJRT_PhaseCompile_Get_PhaseNames ( PJRT_PhaseCompile_Get_PhaseNames_Args * args )
const PJRT_PhaseCompile_Get_PhaseNames = Cvoid

struct PJRT_PhaseCompile_C_Buffers_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    char_buffers::Ptr{Cstring}
    char_buffer_sizes::Ptr{Csize_t}
    num_char_buffers::Csize_t
end

@cenum __JL_Ctag_1051::UInt32 begin
    PJRT_PhaseCompile_C_Buffers_Destroy_Args_STRUCT_SIZE = 0x0000000000000028
end

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

@cenum __JL_Ctag_1052::UInt32 begin
    PJRT_PhaseCompile_Extension_STRUCT_SIZE = 0x0000000000000040
end

struct PJRT_Profiler_Extension
    base::PJRT_Extension_Base
    profiler_api::Ptr{Cint}
    traceme_context_id::Int64
end

@cenum __JL_Ctag_1199::UInt32 begin
    PJRT_Profiler_Extension_STRUCT_SIZE = 0x0000000000000000
end

mutable struct PJRT_RawBuffer end

struct PJRT_RawBuffer_CreateRawAliasOfBuffer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_Buffer}
    raw_buffer::Ptr{PJRT_RawBuffer}
end

@cenum __JL_Ctag_1346::UInt32 begin
    PJRT_RawBuffer_CreateRawAliasOfBuffer_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_RawBuffer_CreateRawAliasOfBuffer ( PJRT_RawBuffer_CreateRawAliasOfBuffer_Args * args )
const PJRT_RawBuffer_CreateRawAliasOfBuffer = Cvoid

struct PJRT_RawBuffer_Destroy_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_RawBuffer}
end

@cenum __JL_Ctag_1347::UInt32 begin
    PJRT_RawBuffer_Destroy_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_RawBuffer_Destroy ( PJRT_RawBuffer_Destroy_Args * args )
const PJRT_RawBuffer_Destroy = Cvoid

struct PJRT_RawBuffer_GetHostPointer_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_RawBuffer}
    host_pointer::Ptr{Cvoid}
end

@cenum __JL_Ctag_1348::UInt32 begin
    PJRT_RawBuffer_GetHostPointer_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_RawBuffer_GetHostPointer ( PJRT_RawBuffer_GetHostPointer_Args * args )
const PJRT_RawBuffer_GetHostPointer = Cvoid

struct PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_RawBuffer}
    on_device_size_in_bytes::Csize_t
end

@cenum __JL_Ctag_1349::UInt32 begin
    PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_RawBuffer_GetOnDeviceSizeInBytes ( PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args * args )
const PJRT_RawBuffer_GetOnDeviceSizeInBytes = Cvoid

struct PJRT_RawBuffer_GetMemorySpace_Args
    struct_size::Csize_t
    extension_start::Ptr{PJRT_Extension_Base}
    buffer::Ptr{PJRT_RawBuffer}
    memory_space::Ptr{PJRT_Memory}
end

@cenum __JL_Ctag_1350::UInt32 begin
    PJRT_RawBuffer_GetMemorySpace_Args_STRUCT_SIZE = 0x0000000000000020
end

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
end

@cenum __JL_Ctag_1351::UInt32 begin
    PJRT_RawBuffer_CopyRawDeviceToHost_Args_STRUCT_SIZE = 0x0000000000000038
end

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
end

@cenum __JL_Ctag_1352::UInt32 begin
    PJRT_RawBuffer_CopyRawHostToDevice_Args_STRUCT_SIZE = 0x0000000000000038
end

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

@cenum __JL_Ctag_1353::UInt32 begin
    PJRT_RawBuffer_Extension_STRUCT_SIZE = 0x0000000000000050
end

struct PJRT_Get_Stream_For_External_Ready_Events_Args
    struct_size::Csize_t
    device::Ptr{PJRT_Device}
    stream::Cptrdiff_t
end

@cenum __JL_Ctag_1500::UInt32 begin
    PJRT_Get_Stream_For_External_Ready_Events_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Get_Stream_For_External_Ready_Events ( PJRT_Get_Stream_For_External_Ready_Events_Args * args )
const PJRT_Get_Stream_For_External_Ready_Events = Cvoid

struct PJRT_Wait_Until_Buffer_Ready_On_Stream_Args
    struct_size::Csize_t
    stream::Cptrdiff_t
    buffer::Ptr{PJRT_Buffer}
end

@cenum __JL_Ctag_1501::UInt32 begin
    PJRT_Wait_Until_Buffer_Ready_On_Stream_Args_STRUCT_SIZE = 0x0000000000000018
end

# typedef PJRT_Error * PJRT_Wait_Until_Buffer_Ready_On_Stream ( PJRT_Wait_Until_Buffer_Ready_On_Stream_Args * args )
const PJRT_Wait_Until_Buffer_Ready_On_Stream = Cvoid

struct PJRT_Stream_Extension
    base::PJRT_Extension_Base
    get_stream::Ptr{PJRT_Get_Stream_For_External_Ready_Events}
    wait_stream::Ptr{PJRT_Wait_Until_Buffer_Ready_On_Stream}
end

@cenum __JL_Ctag_1502::UInt32 begin
    PJRT_Stream_Extension_STRUCT_SIZE = 0x0000000000000028
end

struct PJRT_TpuTopology_Subslice_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chips_per_host_bounds::Ptr{Int32}
    chips_per_host_bounds_num_dims::Csize_t
    host_bounds::Ptr{Int32}
    host_bounds_num_dims::Csize_t
    subslice_topology::Ptr{PJRT_TopologyDescription}
end

@cenum __JL_Ctag_1649::UInt32 begin
    PJRT_TpuTopology_Subslice_Args_STRUCT_SIZE = 0x0000000000000038
end

# typedef PJRT_Error * PJRT_TpuTopology_Subslice ( PJRT_TpuTopology_Subslice_Args * args )
const PJRT_TpuTopology_Subslice = Cvoid

struct PJRT_TpuTopology_IsSubsliceTopology_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    is_subslice_topology::Bool
end

@cenum __JL_Ctag_1650::UInt32 begin
    PJRT_TpuTopology_IsSubsliceTopology_Args_STRUCT_SIZE = 0x0000000000000011
end

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
end

@cenum __JL_Ctag_1651::UInt32 begin
    PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args_STRUCT_SIZE = 0x0000000000000030
end

# typedef PJRT_Error * PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId ( PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args * args )
const PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId = Cvoid

struct PJRT_TpuTopology_ReplaceHostBounds_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    host_bounds::Ptr{Int32}
    host_bounds_dim_num::Csize_t
    new_topology::Ptr{PJRT_TopologyDescription}
end

@cenum __JL_Ctag_1652::UInt32 begin
    PJRT_TpuTopology_ReplaceHostBounds_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_TpuTopology_ReplaceHostBounds ( PJRT_TpuTopology_ReplaceHostBounds_Args * args )
const PJRT_TpuTopology_ReplaceHostBounds = Cvoid

struct PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    is_enhanced_barrier_enabled::Bool
end

@cenum __JL_Ctag_1653::UInt32 begin
    PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args_STRUCT_SIZE = 0x0000000000000011
end

# typedef PJRT_Error * PJRT_TpuTopology_IsEnhancedBarrierEnabled ( PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args * args )
const PJRT_TpuTopology_IsEnhancedBarrierEnabled = Cvoid

struct PJRT_TpuTopology_HasLimitedIciConnectivity_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    has_limited_ici_connectivity::Bool
end

@cenum __JL_Ctag_1654::UInt32 begin
    PJRT_TpuTopology_HasLimitedIciConnectivity_Args_STRUCT_SIZE = 0x0000000000000011
end

# typedef PJRT_Error * PJRT_TpuTopology_HasLimitedIciConnectivity ( PJRT_TpuTopology_HasLimitedIciConnectivity_Args * args )
const PJRT_TpuTopology_HasLimitedIciConnectivity = Cvoid

struct PJRT_TpuTopology_IsReachableOverLimitedIci_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    source_chip_id::Int32
    dest_chip_id::Int32
    is_reachable_over_limited_ici::Bool
end

@cenum __JL_Ctag_1655::UInt32 begin
    PJRT_TpuTopology_IsReachableOverLimitedIci_Args_STRUCT_SIZE = 0x0000000000000019
end

# typedef PJRT_Error * PJRT_TpuTopology_IsReachableOverLimitedIci ( PJRT_TpuTopology_IsReachableOverLimitedIci_Args * args )
const PJRT_TpuTopology_IsReachableOverLimitedIci = Cvoid

struct PJRT_TpuTopology_ProcessCount_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    process_count::Int32
end

@cenum __JL_Ctag_1656::UInt32 begin
    PJRT_TpuTopology_ProcessCount_Args_STRUCT_SIZE = 0x0000000000000014
end

# typedef PJRT_Error * PJRT_TpuTopology_ProcessCount ( PJRT_TpuTopology_ProcessCount_Args * args )
const PJRT_TpuTopology_ProcessCount = Cvoid

struct PJRT_TpuTopology_ChipsPerProcess_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chips_per_process::Int32
end

@cenum __JL_Ctag_1657::UInt32 begin
    PJRT_TpuTopology_ChipsPerProcess_Args_STRUCT_SIZE = 0x0000000000000014
end

# typedef PJRT_Error * PJRT_TpuTopology_ChipsPerProcess ( PJRT_TpuTopology_ChipsPerProcess_Args * args )
const PJRT_TpuTopology_ChipsPerProcess = Cvoid

struct PJRT_TpuTopology_CoreCountPerChip_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    core_count_of_default_type_per_chip::Int32
end

@cenum __JL_Ctag_1658::UInt32 begin
    PJRT_TpuTopology_CoreCountPerChip_Args_STRUCT_SIZE = 0x0000000000000014
end

# typedef PJRT_Error * PJRT_TpuTopology_CoreCountPerChip ( PJRT_TpuTopology_CoreCountPerChip_Args * args )
const PJRT_TpuTopology_CoreCountPerChip = Cvoid

struct PJRT_TpuTopology_ChipCount_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chip_count::Int32
end

@cenum __JL_Ctag_1659::UInt32 begin
    PJRT_TpuTopology_ChipCount_Args_STRUCT_SIZE = 0x0000000000000014
end

# typedef PJRT_Error * PJRT_TpuTopology_ChipCount ( PJRT_TpuTopology_ChipCount_Args * args )
const PJRT_TpuTopology_ChipCount = Cvoid

struct PJRT_TpuTopology_CoreCount_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    core_count_of_default_type::Int32
end

@cenum __JL_Ctag_1660::UInt32 begin
    PJRT_TpuTopology_CoreCount_Args_STRUCT_SIZE = 0x0000000000000014
end

# typedef PJRT_Error * PJRT_TpuTopology_CoreCount ( PJRT_TpuTopology_CoreCount_Args * args )
const PJRT_TpuTopology_CoreCount = Cvoid

struct PJRT_TpuTopology_LogiDeviceCount_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    logical_device_count_of_default_type::Int32
end

@cenum __JL_Ctag_1661::UInt32 begin
    PJRT_TpuTopology_LogiDeviceCount_Args_STRUCT_SIZE = 0x0000000000000014
end

# typedef PJRT_Error * PJRT_TpuTopology_LogiDeviceCount ( PJRT_TpuTopology_LogiDeviceCount_Args * args )
const PJRT_TpuTopology_LogiDeviceCount = Cvoid

struct PJRT_TpuTopology_LogiDeviceCountPerProcess_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    logical_device_count_of_default_type_per_process::Int32
end

@cenum __JL_Ctag_1662::UInt32 begin
    PJRT_TpuTopology_LogiDeviceCountPerProcess_Args_STRUCT_SIZE = 0x0000000000000014
end

# typedef PJRT_Error * PJRT_TpuTopology_LogiDeviceCountPerProcess ( PJRT_TpuTopology_LogiDeviceCountPerProcess_Args * args )
const PJRT_TpuTopology_LogiDeviceCountPerProcess = Cvoid

struct PJRT_TpuTopology_LogiDeviceCountPerChip_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    logical_device_count_of_default_type_per_chip::Int32
end

@cenum __JL_Ctag_1663::UInt32 begin
    PJRT_TpuTopology_LogiDeviceCountPerChip_Args_STRUCT_SIZE = 0x0000000000000014
end

# typedef PJRT_Error * PJRT_TpuTopology_LogiDeviceCountPerChip ( PJRT_TpuTopology_LogiDeviceCountPerChip_Args * args )
const PJRT_TpuTopology_LogiDeviceCountPerChip = Cvoid

struct PJRT_TpuTopology_CoreCountPerProcess_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    core_count_of_default_type_per_process::Int32
end

@cenum __JL_Ctag_1664::UInt32 begin
    PJRT_TpuTopology_CoreCountPerProcess_Args_STRUCT_SIZE = 0x0000000000000014
end

# typedef PJRT_Error * PJRT_TpuTopology_CoreCountPerProcess ( PJRT_TpuTopology_CoreCountPerProcess_Args * args )
const PJRT_TpuTopology_CoreCountPerProcess = Cvoid

struct PJRT_TpuTopology_ProcessIds_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    max_process_ids::Int32
    process_ids::Ptr{Int32}
    num_process_ids::Csize_t
end

@cenum __JL_Ctag_1665::UInt32 begin
    PJRT_TpuTopology_ProcessIds_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_TpuTopology_ProcessIds ( PJRT_TpuTopology_ProcessIds_Args * args )
const PJRT_TpuTopology_ProcessIds = Cvoid

struct PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    process_id::Int32
    max_logical_device_ids::Int32
    logical_device_of_default_type_ids::Ptr{Int32}
    num_logical_device_ids::Csize_t
end

@cenum __JL_Ctag_1666::UInt32 begin
    PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_TpuTopology_LogiDeviceIdsOnProcess ( PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args * args )
const PJRT_TpuTopology_LogiDeviceIdsOnProcess = Cvoid

struct PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chip_id::Int32
    process_id::Int32
    index_on_process::Int32
end

@cenum __JL_Ctag_1667::UInt32 begin
    PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args_STRUCT_SIZE = 0x000000000000001c
end

# typedef PJRT_Error * PJRT_TpuTopology_ProcIdAndIdxOnProcForChip ( PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args * args )
const PJRT_TpuTopology_ProcIdAndIdxOnProcForChip = Cvoid

struct PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    device_id::Int32
    process_id::Int32
    index_on_process::Int32
end

@cenum __JL_Ctag_1668::UInt32 begin
    PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args_STRUCT_SIZE = 0x000000000000001c
end

# typedef PJRT_Error * PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice ( PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args * args )
const PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice = Cvoid

struct PJRT_TpuTopology_ProcessCoordFromId_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    process_id::Int32
    coords_max_dims::Csize_t
    coords::Ptr{Int32}
    coords_num_dims::Csize_t
end

@cenum __JL_Ctag_1669::UInt32 begin
    PJRT_TpuTopology_ProcessCoordFromId_Args_STRUCT_SIZE = 0x0000000000000030
end

# typedef PJRT_Error * PJRT_TpuTopology_ProcessCoordFromId ( PJRT_TpuTopology_ProcessCoordFromId_Args * args )
const PJRT_TpuTopology_ProcessCoordFromId = Cvoid

struct PJRT_TpuTopology_ChipIdFromCoord_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    coords::Ptr{Int32}
    coords_num_dims::Csize_t
    chip_id::Int32
end

@cenum __JL_Ctag_1670::UInt32 begin
    PJRT_TpuTopology_ChipIdFromCoord_Args_STRUCT_SIZE = 0x0000000000000024
end

# typedef PJRT_Error * PJRT_TpuTopology_ChipIdFromCoord ( PJRT_TpuTopology_ChipIdFromCoord_Args * args )
const PJRT_TpuTopology_ChipIdFromCoord = Cvoid

struct PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chip_coords::Ptr{Int32}
    chip_coords_num_dims::Csize_t
    logical_device_index_on_chip::Int32
    logical_device_of_default_type_id::Int32
end

@cenum __JL_Ctag_1671::UInt32 begin
    PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args_STRUCT_SIZE = 0x0000000000000028
end

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
end

@cenum __JL_Ctag_1672::UInt32 begin
    PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args_STRUCT_SIZE = 0x0000000000000034
end

# typedef PJRT_Error * PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice ( PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args * args )
const PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice = Cvoid

struct PJRT_TpuTopology_ChipsPerProcessBounds_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chip_per_process_bounds_max_dims::Csize_t
    chip_per_process_bounds::Ptr{Int32}
    chip_per_process_bounds_num_dims::Csize_t
end

@cenum __JL_Ctag_1673::UInt32 begin
    PJRT_TpuTopology_ChipsPerProcessBounds_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_TpuTopology_ChipsPerProcessBounds ( PJRT_TpuTopology_ChipsPerProcessBounds_Args * args )
const PJRT_TpuTopology_ChipsPerProcessBounds = Cvoid

struct PJRT_TpuTopology_ChipBounds_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    chip_bounds_max_dims::Csize_t
    chip_bounds::Ptr{Int32}
    chip_bounds_num_dims::Csize_t
end

@cenum __JL_Ctag_1674::UInt32 begin
    PJRT_TpuTopology_ChipBounds_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_TpuTopology_ChipBounds ( PJRT_TpuTopology_ChipBounds_Args * args )
const PJRT_TpuTopology_ChipBounds = Cvoid

struct PJRT_TpuTopology_ProcessBounds_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    process_bounds_max_dims::Csize_t
    process_bounds::Ptr{Int32}
    process_bounds_num_dims::Csize_t
end

@cenum __JL_Ctag_1675::UInt32 begin
    PJRT_TpuTopology_ProcessBounds_Args_STRUCT_SIZE = 0x0000000000000028
end

# typedef PJRT_Error * PJRT_TpuTopology_ProcessBounds ( PJRT_TpuTopology_ProcessBounds_Args * args )
const PJRT_TpuTopology_ProcessBounds = Cvoid

struct PJRT_TpuTopology_GetRoutingStrategy_Args
    struct_size::Csize_t
    topology::Ptr{PJRT_TopologyDescription}
    routing_strategy::Cstring
    routing_strategy_len::Csize_t
end

@cenum __JL_Ctag_1676::UInt32 begin
    PJRT_TpuTopology_GetRoutingStrategy_Args_STRUCT_SIZE = 0x0000000000000020
end

# typedef PJRT_Error * PJRT_TpuTopology_GetRoutingStrategy ( PJRT_TpuTopology_GetRoutingStrategy_Args * args )
const PJRT_TpuTopology_GetRoutingStrategy = Cvoid

struct PJRT_TpuTopology_Extension
    base::PJRT_Extension_Base
    subslice::Ptr{PJRT_TpuTopology_Subslice}
    is_subslice_topology::Ptr{PJRT_TpuTopology_IsSubsliceTopology}
    subslice_device_id_from_full_device_id::Ptr{PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId}
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
    proc_id_and_idx_on_proc_for_logi_device::Ptr{PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice}
    process_coord_from_id::Ptr{PJRT_TpuTopology_ProcessCoordFromId}
    chip_id_from_coord::Ptr{PJRT_TpuTopology_ChipIdFromCoord}
    logical_device_id_from_chip_coord_and_idx::Ptr{PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx}
    chip_coord_and_idx_for_logi_device::Ptr{PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice}
    chips_per_process_bounds::Ptr{PJRT_TpuTopology_ChipsPerProcessBounds}
    chip_bounds::Ptr{PJRT_TpuTopology_ChipBounds}
    process_bounds::Ptr{PJRT_TpuTopology_ProcessBounds}
    get_routing_strategy::Ptr{PJRT_TpuTopology_GetRoutingStrategy}
end

@cenum __JL_Ctag_1677::UInt32 begin
    PJRT_TpuTopology_Extension_STRUCT_SIZE = 0x00000000000000f8
end

struct PJRT_Triton_Compile_Args
    struct_size::Csize_t
    _module::Cstring
    module_size::Csize_t
    arch_name::Cstring
    arch_name_size::Csize_t
    num_warps::Cint
    num_ctas::Cint
    num_stages::Cint
    out_asm::Cstring
    out_asm_size::Csize_t
    out_smem_bytes::Int64
end

@cenum __JL_Ctag_1824::UInt32 begin
    PJRT_Triton_Compile_Args_STRUCT_SIZE = 0x0000000000000050
end

# typedef PJRT_Error * PJRT_Triton_Compile ( PJRT_Triton_Compile_Args * args )
const PJRT_Triton_Compile = Cvoid

struct PJRT_Triton_Extension
    base::PJRT_Extension_Base
    compile::Ptr{PJRT_Triton_Compile}
end

const PJRT_Triton = PJRT_Triton_Extension

@cenum __JL_Ctag_1825::UInt32 begin
    PJRT_Triton_Extension_STRUCT_SIZE = 0x0000000000000020
end

const PJRT_API_MAJOR = 0

const PJRT_API_MINOR = 90

const _PJRT_API_STRUCT_FIELD = fn_type(fn_type) * fn_type

const PJRT_API_CALLBACK_EXTENSION_VERSION = 1

const PJRT_API_CUSTOM_PARTITIONER_EXTENSION_VERSION = 1

const PJRT_API_FFI_EXTENSION_VERSION = 3

const PJRT_API_GPU_EXTENSION_VERSION = 2

const PJRT_API_LAYOUTS_EXTENSION_VERSION = 3

const PJRT_API_MEMORY_DESCRIPTIONS_EXTENSION_VERSION = 1

const PJRT_API_PHASE_COMPILE_EXTENSION_VERSION = 1

const PJRT_API_PROFILER_EXTENSION_VERSION = 1

const PJRT_API_RAW_BUFFER_EXTENSION_VERSION = 2

const PJRT_API_STREAM_EXTENSION_VERSION = 0

const PJRT_API_TPU_TOPOLOGY_EXTENSION_VERSION = 1

const PJRT_API_TRITON_EXTENSION_VERSION = 1

