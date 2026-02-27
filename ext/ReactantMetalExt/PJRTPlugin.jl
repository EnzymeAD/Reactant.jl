# PJRTPlugin.jl — PJRT plugin glue for Metal backend
#
# Design:
#   - All handles use Libc.malloc (C memory, stable across GC)
#   - All @cfunction pointers are module-level consts (stable LLVM stubs)
#   - PJRT_Api struct stored in C memory via Libc.malloc (stable pointer)
#   - PJRT_Api struct uses Reactant.XLA.PJRT.CAPI types (PJRT_Api, PJRT_Api_Version)
#   - PLATFORM_NAME = "METAL" (uppercase, matches test assertion)
#
# File organization:
#   - Error.jl:      Error handling callbacks
#   - Plugin.jl:     Plugin lifecycle callbacks
#   - Event.jl:      Event callbacks
#   - Client.jl:     Client lifecycle callbacks
#   - Device.jl:     Device + DeviceDescription callbacks
#   - Memory.jl:     Memory callbacks
#   - Buffer.jl:     MTLTensor buffer helpers + buffer callbacks
#   - Executable.jl: MetalExecutableData struct + executable callbacks
#   - PJRTPlugin.jl: Constants, stubs, @cfunction table, init_pjrt_handles!, make_client

# ============================================================
# Opaque handles (C memory — GC-stable, never moved)
# Initialized at runtime in init_pjrt_handles!() to avoid
# dangling pointers from precompilation serialization.
# ============================================================

CLIENT_HANDLE = Ptr{Cvoid}(0)
DEVICE_HANDLE = Ptr{Cvoid}(0)
DEVDESC_HANDLE = Ptr{Cvoid}(0)
MEMORY_HANDLE = Ptr{Cvoid}(0)

DEVICE_PTR_ARRAY = Ptr{Cvoid}(0)
MEMORY_PTR_ARRAY = Ptr{Cvoid}(0)

# Static strings (Julia global consts — stable pointers for the process lifetime)
const PLATFORM_NAME = "METAL"   # uppercase — test asserts == "METAL"
const PLATFORM_VERSION = "1.0.0"
const DEVICE_KIND = "Metal GPU"
const DEVICE_DEBUG_STRING = "Metal Apple GPU Device"
const DEVICE_TO_STRING = "Metal:0"
const MEMORY_KIND_STR = "device"
const MEMORY_DEBUG_STR = "Metal Device Memory"
const MEMORY_TO_STR = "Metal:0:device"
const UNIMPL_MESSAGE = "UNIMPLEMENTED: Operation not supported by Metal PJRT plugin"

# Sentinel values written into opaque handles for debuggability.
# PJRT requires non-NULL pointers for client/device/memory/event handles
# (the C++ side dereferences them). Since this is a single-device plugin
# (one Metal GPU, one memory space), the handles don't carry real state —
# they just need to be valid, distinct, non-NULL pointers. The hex values
# are conventional debug markers, easy to identify in a debugger/lldb.
const SENTINEL_CLIENT = Int64(0xDEADBEEF)
const SENTINEL_DEVICE = Int64(0xCAFEBABE)
const SENTINEL_DEVDESC = Int64(0xF00DCAFE)
const SENTINEL_MEMORY = Int64(0xFEEDFACE)
const SENTINEL_ERROR = Int64(0xDEAD)
const SENTINEL_EVENT = Int64(0x1234CAFE)

# Error handle for UNIMPLEMENTED responses
UNIMPL_ERROR_HANDLE = Ptr{Cvoid}(0)

# Pre-allocated "ready" event handle — event_is_ready always returns true for this
READY_EVENT_HANDLE = Ptr{Cvoid}(0)

# ============================================================
# Generic stubs
# ============================================================

# Returns C_NULL (success, no error). Safe for callbacks whose output fields are not read.
function _stub(args::Ptr{Cvoid})::Ptr{Cvoid}
    return C_NULL
end

# Returns non-NULL error handle. Used for PJRT_Client_TopologyDescription
# which is called without NULL-checking the function pointer.
function _unimpl(args::Ptr{Cvoid})::Ptr{Cvoid}
    return UNIMPL_ERROR_HANDLE
end

# ============================================================
# Callback implementations (one file per API topic)
# ============================================================

include("Error.jl")
include("Plugin.jl")
include("Event.jl")
include("Client.jl")
include("Device.jl")
include("Memory.jl")

# ============================================================
# @cfunction pointers — initialized at runtime in init_pjrt_handles!()
# @cfunction returns Ptr{Cvoid} which is just a number to Julia's serializer.
# Precompilation serializes the ADDRESS, not the trampoline code.
# Must be re-created at runtime.
# ============================================================

cfn_stub = Ptr{Cvoid}(0)
cfn_unimpl = Ptr{Cvoid}(0)
cfn_client_compile = Ptr{Cvoid}(0)
cfn_loaded_exec_destroy = Ptr{Cvoid}(0)
cfn_loaded_exec_get_exec = Ptr{Cvoid}(0)
cfn_loaded_exec_addr_devs = Ptr{Cvoid}(0)
cfn_loaded_exec_execute = Ptr{Cvoid}(0)
cfn_client_buffer_from_host = Ptr{Cvoid}(0)
cfn_buffer_destroy = Ptr{Cvoid}(0)
cfn_buffer_element_type = Ptr{Cvoid}(0)
cfn_buffer_dimensions = Ptr{Cvoid}(0)
cfn_buffer_on_device_size = Ptr{Cvoid}(0)
cfn_buffer_device = Ptr{Cvoid}(0)
cfn_buffer_memory = Ptr{Cvoid}(0)
cfn_buffer_is_on_cpu = Ptr{Cvoid}(0)
cfn_buffer_ready_event = Ptr{Cvoid}(0)
cfn_buffer_to_host = Ptr{Cvoid}(0)
cfn_error_destroy = Ptr{Cvoid}(0)
cfn_error_message = Ptr{Cvoid}(0)
cfn_error_getcode = Ptr{Cvoid}(0)
cfn_plugin_initialize = Ptr{Cvoid}(0)
cfn_plugin_attributes = Ptr{Cvoid}(0)
cfn_event_destroy = Ptr{Cvoid}(0)
cfn_event_is_ready = Ptr{Cvoid}(0)
cfn_event_await = Ptr{Cvoid}(0)
cfn_client_create = Ptr{Cvoid}(0)
cfn_client_destroy = Ptr{Cvoid}(0)
cfn_client_platform_name = Ptr{Cvoid}(0)
cfn_client_process_index = Ptr{Cvoid}(0)
cfn_client_platform_version = Ptr{Cvoid}(0)
cfn_client_devices = Ptr{Cvoid}(0)
cfn_client_addr_devices = Ptr{Cvoid}(0)
cfn_client_addr_memories = Ptr{Cvoid}(0)
cfn_device_get_desc = Ptr{Cvoid}(0)
cfn_device_is_addressable = Ptr{Cvoid}(0)
cfn_device_local_hw_id = Ptr{Cvoid}(0)
cfn_device_addr_memories = Ptr{Cvoid}(0)
cfn_device_default_memory = Ptr{Cvoid}(0)
cfn_devdesc_id = Ptr{Cvoid}(0)
cfn_devdesc_process_index = Ptr{Cvoid}(0)
cfn_devdesc_attributes = Ptr{Cvoid}(0)
cfn_devdesc_kind = Ptr{Cvoid}(0)
cfn_devdesc_debug_string = Ptr{Cvoid}(0)
cfn_devdesc_to_string = Ptr{Cvoid}(0)
cfn_memory_id = Ptr{Cvoid}(0)
cfn_memory_kind = Ptr{Cvoid}(0)
cfn_memory_kind_id = Ptr{Cvoid}(0)
cfn_memory_debug_string = Ptr{Cvoid}(0)
cfn_memory_to_string = Ptr{Cvoid}(0)
cfn_memory_addr_by_devices = Ptr{Cvoid}(0)
cfn_event_on_ready = Ptr{Cvoid}(0)
cfn_exec_name = Ptr{Cvoid}(0)
cfn_exec_num_replicas = Ptr{Cvoid}(0)
cfn_exec_num_partitions = Ptr{Cvoid}(0)
cfn_exec_num_outputs = Ptr{Cvoid}(0)
cfn_exec_code_size = Ptr{Cvoid}(0)
cfn_exec_optimized_program = Ptr{Cvoid}(0)
cfn_exec_output_memory_kinds = Ptr{Cvoid}(0)
cfn_loaded_exec_fingerprint = Ptr{Cvoid}(0)
cfn_exec_output_element_types = Ptr{Cvoid}(0)
cfn_exec_output_dimensions = Ptr{Cvoid}(0)
cfn_noop_free = Ptr{Cvoid}(0)
cfn_loaded_exec_get_device_assignment = Ptr{Cvoid}(0)
cfn_device_get_attributes = Ptr{Cvoid}(0)

# ============================================================
# PJRT_Api struct (function pointer table)
# ============================================================

# Pointer to C-managed PJRT_Api struct (initialized in init_pjrt_handles!())
_PJRT_API_MEM = Ptr{Cvoid}(0)

"""
    init_pjrt_handles!()

Allocate C memory for opaque handles and the PJRT_Api struct.
Must be called at runtime (from __init__), NOT at precompile time,
because Libc.malloc pointers become dangling after .ji deserialization.
"""
function init_pjrt_handles!()
    global CLIENT_HANDLE = Libc.malloc(64)
    global DEVICE_HANDLE = Libc.malloc(64)
    global DEVDESC_HANDLE = Libc.malloc(64)
    global MEMORY_HANDLE = Libc.malloc(64)

    unsafe_store!(Ptr{Int64}(CLIENT_HANDLE), SENTINEL_CLIENT)
    unsafe_store!(Ptr{Int64}(DEVICE_HANDLE), SENTINEL_DEVICE)
    unsafe_store!(Ptr{Int64}(DEVDESC_HANDLE), SENTINEL_DEVDESC)
    unsafe_store!(Ptr{Int64}(MEMORY_HANDLE), SENTINEL_MEMORY)

    global DEVICE_PTR_ARRAY = Libc.malloc(8)
    unsafe_store!(Ptr{Ptr{Cvoid}}(DEVICE_PTR_ARRAY), DEVICE_HANDLE)

    global MEMORY_PTR_ARRAY = Libc.malloc(8)
    unsafe_store!(Ptr{Ptr{Cvoid}}(MEMORY_PTR_ARRAY), MEMORY_HANDLE)

    global UNIMPL_ERROR_HANDLE = Libc.malloc(8)
    unsafe_store!(Ptr{Int64}(UNIMPL_ERROR_HANDLE), SENTINEL_ERROR)

    global READY_EVENT_HANDLE = Libc.malloc(8)
    unsafe_store!(Ptr{Int64}(READY_EVENT_HANDLE), SENTINEL_EVENT)

    # Create @cfunction trampolines at runtime (addresses are process-specific).
    global cfn_stub = @cfunction(_stub, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_unimpl = @cfunction(_unimpl, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_error_destroy = @cfunction(
        _error_destroy, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Error_Destroy_Args},)
    )
    global cfn_error_message = @cfunction(
        _error_message, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Error_Message_Args},)
    )
    global cfn_error_getcode = @cfunction(
        _error_getcode, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Error_GetCode_Args},)
    )
    global cfn_plugin_initialize = @cfunction(
        _plugin_initialize,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Plugin_Initialize_Args},)
    )
    global cfn_plugin_attributes = @cfunction(
        _plugin_attributes,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Plugin_Attributes_Args},)
    )
    global cfn_event_destroy = @cfunction(
        _event_destroy, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Event_Destroy_Args},)
    )
    global cfn_event_is_ready = @cfunction(
        _event_is_ready, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Event_IsReady_Args},)
    )
    global cfn_event_await = @cfunction(
        _event_await, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Event_Await_Args},)
    )
    global cfn_client_create = @cfunction(
        _client_create, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Client_Create_Args},)
    )
    global cfn_client_destroy = @cfunction(
        _client_destroy, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Client_Destroy_Args},)
    )
    global cfn_client_platform_name = @cfunction(
        _client_platform_name,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Client_PlatformName_Args},)
    )
    global cfn_client_process_index = @cfunction(
        _client_process_index,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Client_ProcessIndex_Args},)
    )
    global cfn_client_platform_version = @cfunction(
        _client_platform_version,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Client_PlatformVersion_Args},)
    )
    global cfn_client_devices = @cfunction(
        _client_devices, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Client_Devices_Args},)
    )
    global cfn_client_addr_devices = @cfunction(
        _client_addressable_devices,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Client_AddressableDevices_Args},)
    )
    global cfn_client_addr_memories = @cfunction(
        _client_addressable_memories,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Client_AddressableMemories_Args},)
    )
    global cfn_device_get_desc = @cfunction(
        _device_get_description,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Device_GetDescription_Args},)
    )
    global cfn_device_is_addressable = @cfunction(
        _device_is_addressable,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Device_IsAddressable_Args},)
    )
    global cfn_device_local_hw_id = @cfunction(
        _device_local_hardware_id,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Device_LocalHardwareId_Args},)
    )
    global cfn_device_addr_memories = @cfunction(
        _device_addressable_memories,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Device_AddressableMemories_Args},)
    )
    global cfn_device_default_memory = @cfunction(
        _device_default_memory,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Device_DefaultMemory_Args},)
    )
    global cfn_devdesc_id = @cfunction(
        _devdesc_id,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_DeviceDescription_Id_Args},)
    )
    global cfn_devdesc_process_index = @cfunction(
        _devdesc_process_index,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_DeviceDescription_ProcessIndex_Args},)
    )
    global cfn_devdesc_attributes = @cfunction(
        _devdesc_attributes,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_DeviceDescription_Attributes_Args},)
    )
    global cfn_devdesc_kind = @cfunction(
        _devdesc_kind,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_DeviceDescription_Kind_Args},)
    )
    global cfn_devdesc_debug_string = @cfunction(
        _devdesc_debug_string,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_DeviceDescription_DebugString_Args},)
    )
    global cfn_devdesc_to_string = @cfunction(
        _devdesc_to_string,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_DeviceDescription_ToString_Args},)
    )
    global cfn_memory_id = @cfunction(
        _memory_id, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Memory_Id_Args},)
    )
    global cfn_memory_kind = @cfunction(
        _memory_kind, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Memory_Kind_Args},)
    )
    global cfn_memory_kind_id = @cfunction(
        _memory_kind_id, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Memory_Kind_Id_Args},)
    )
    global cfn_memory_debug_string = @cfunction(
        _memory_debug_string,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Memory_DebugString_Args},)
    )
    global cfn_memory_to_string = @cfunction(
        _memory_to_string,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Memory_ToString_Args},)
    )
    global cfn_memory_addr_by_devices = @cfunction(
        _memory_addressable_by_devices,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Memory_AddressableByDevices_Args},)
    )
    global cfn_client_compile = @cfunction(
        _client_compile, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Client_Compile_Args},)
    )
    global cfn_loaded_exec_destroy = @cfunction(
        _loaded_exec_destroy,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_LoadedExecutable_Destroy_Args},)
    )
    global cfn_loaded_exec_get_exec = @cfunction(
        _loaded_exec_get_executable,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_LoadedExecutable_GetExecutable_Args},)
    )
    global cfn_loaded_exec_addr_devs = @cfunction(
        _loaded_exec_addressable_devices,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_LoadedExecutable_AddressableDevices_Args},)
    )
    global cfn_loaded_exec_execute = @cfunction(
        _loaded_exec_execute,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_LoadedExecutable_Execute_Args},)
    )
    global cfn_client_buffer_from_host = @cfunction(
        _client_buffer_from_host,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Client_BufferFromHostBuffer_Args},)
    )
    global cfn_buffer_destroy = @cfunction(
        _buffer_destroy, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Buffer_Destroy_Args},)
    )
    global cfn_buffer_element_type = @cfunction(
        _buffer_element_type,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Buffer_ElementType_Args},)
    )
    global cfn_buffer_dimensions = @cfunction(
        _buffer_dimensions,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Buffer_Dimensions_Args},)
    )
    global cfn_buffer_on_device_size = @cfunction(
        _buffer_on_device_size,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args},)
    )
    global cfn_buffer_device = @cfunction(
        _buffer_device, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Buffer_Device_Args},)
    )
    global cfn_buffer_memory = @cfunction(
        _buffer_memory, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Buffer_Memory_Args},)
    )
    global cfn_buffer_is_on_cpu = @cfunction(
        _buffer_is_on_cpu,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Buffer_IsOnCpu_Args},)
    )
    global cfn_buffer_ready_event = @cfunction(
        _buffer_ready_event,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Buffer_ReadyEvent_Args},)
    )
    global cfn_buffer_to_host = @cfunction(
        _buffer_to_host,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Buffer_ToHostBuffer_Args},)
    )
    global cfn_event_on_ready = @cfunction(
        _event_on_ready, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Event_OnReady_Args},)
    )
    global cfn_exec_name = @cfunction(
        _exec_name, Ptr{Cvoid}, (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Executable_Name_Args},)
    )
    global cfn_exec_num_replicas = @cfunction(
        _exec_num_replicas,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Executable_NumReplicas_Args},)
    )
    global cfn_exec_num_partitions = @cfunction(
        _exec_num_partitions,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Executable_NumPartitions_Args},)
    )
    global cfn_exec_num_outputs = @cfunction(
        _exec_num_outputs,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Executable_NumOutputs_Args},)
    )
    global cfn_exec_code_size = @cfunction(
        _exec_code_size,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Executable_SizeOfGeneratedCodeInBytes_Args},)
    )
    global cfn_exec_optimized_program = @cfunction(
        _exec_optimized_program,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Executable_OptimizedProgram_Args},)
    )
    global cfn_exec_output_memory_kinds = @cfunction(
        _exec_output_memory_kinds,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Executable_OutputMemoryKinds_Args},)
    )
    global cfn_loaded_exec_fingerprint = @cfunction(
        _loaded_exec_fingerprint,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_LoadedExecutable_Fingerprint_Args},)
    )
    global cfn_exec_output_element_types = @cfunction(
        _exec_output_element_types,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Executable_OutputElementTypes_Args},)
    )
    global cfn_exec_output_dimensions = @cfunction(
        _exec_output_dimensions,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_Executable_OutputDimensions_Args},)
    )
    # Must be initialized before cfn_loaded_exec_get_device_assignment (which embeds it at call time)
    global cfn_noop_free = @cfunction(_noop_free, Cvoid, (Ptr{Cvoid},))
    global cfn_loaded_exec_get_device_assignment = @cfunction(
        _loaded_exec_get_device_assignment,
        Ptr{Cvoid},
        (Ptr{Reactant.XLA.PJRT.CAPI.PJRT_LoadedExecutable_GetDeviceAssignment_Args},)
    )
    global cfn_device_get_attributes = @cfunction(
        _device_get_attributes,
        Ptr{Cvoid},
        (Ptr{PJRT_Device_GetAttributes_Args},)
    )

    # Rebuild the function pointer tuple and PJRT_Api struct at runtime.
    n_fn_ptrs = fieldcount(CAPI.PJRT_Api) - 3  # total fields minus struct_size, extension_start, pjrt_api_version
    fns = ntuple(Val(n_fn_ptrs)) do i
        if i == 1
            cfn_error_destroy
        elseif i == 2
            cfn_error_message
        elseif i == 3
            cfn_error_getcode
        elseif i == 4
            cfn_plugin_initialize
        elseif i == 5
            cfn_plugin_attributes
        elseif i == 6
            cfn_event_destroy
        elseif i == 7
            cfn_event_is_ready
        elseif i == 8
            cfn_stub
        elseif i == 9
            cfn_event_await
        elseif i == 10
            cfn_event_on_ready
        elseif i == 11
            cfn_client_create
        elseif i == 12
            cfn_client_destroy
        elseif i == 13
            cfn_client_platform_name
        elseif i == 14
            cfn_client_process_index
        elseif i == 15
            cfn_client_platform_version
        elseif i == 16
            cfn_client_devices
        elseif i == 17
            cfn_client_addr_devices
        elseif i == 18
            cfn_stub
        elseif i == 19
            cfn_stub
        elseif i == 20
            cfn_client_addr_memories
        elseif i == 21
            cfn_client_compile
        elseif i == 22
            cfn_stub
        elseif i == 23
            cfn_client_buffer_from_host
        elseif i == 24
            cfn_devdesc_id
        elseif i == 25
            cfn_devdesc_process_index
        elseif i == 26
            cfn_devdesc_attributes
        elseif i == 27
            cfn_devdesc_kind
        elseif i == 28
            cfn_devdesc_debug_string
        elseif i == 29
            cfn_devdesc_to_string
        elseif i == 30
            cfn_device_get_desc
        elseif i == 31
            cfn_device_is_addressable
        elseif i == 32
            cfn_device_local_hw_id
        elseif i == 33
            cfn_device_addr_memories
        elseif i == 34
            cfn_device_default_memory
        elseif i == 35
            cfn_stub
        elseif i == 36
            cfn_memory_id
        elseif i == 37
            cfn_memory_kind
        elseif i == 38
            cfn_memory_debug_string
        elseif i == 39
            cfn_memory_to_string
        elseif i == 40
            cfn_memory_addr_by_devices
        elseif i == 42
            cfn_exec_name
        elseif i == 43
            cfn_exec_num_replicas
        elseif i == 44
            cfn_exec_num_partitions
        elseif i == 45
            cfn_exec_num_outputs
        elseif i == 46
            cfn_exec_code_size
        elseif i == 49
            cfn_exec_optimized_program
        elseif i == 48
            cfn_exec_output_memory_kinds
        elseif i == 51
            cfn_loaded_exec_destroy
        elseif i == 52
            cfn_loaded_exec_get_exec
        elseif i == 53
            cfn_loaded_exec_addr_devs
        elseif i == 56
            cfn_loaded_exec_execute
        elseif i == 58
            cfn_loaded_exec_fingerprint
        elseif i == 59
            cfn_buffer_destroy
        elseif i == 60
            cfn_buffer_element_type
        elseif i == 61
            cfn_buffer_dimensions
        elseif i == 62
            cfn_buffer_dimensions  # UnpaddedDimensions — same layout, must set output fields
        elseif i == 65
            cfn_buffer_on_device_size
        elseif i == 66
            cfn_buffer_device
        elseif i == 67
            cfn_buffer_memory
        elseif i == 71
            cfn_buffer_to_host
        elseif i == 72
            cfn_buffer_is_on_cpu
        elseif i == 73
            cfn_buffer_ready_event
        elseif i == 91
            cfn_exec_output_element_types
        elseif i == 92
            cfn_exec_output_dimensions
        elseif i == 96
            cfn_unimpl
        elseif i == 98
            cfn_memory_kind_id
        elseif i == 118
            cfn_loaded_exec_get_device_assignment
        elseif i == 129
            cfn_device_get_attributes
        else
            cfn_stub
        end
    end

    api_val = CAPI.PJRT_Api(
        Csize_t(sizeof(CAPI.PJRT_Api)),
        Ptr{CAPI.PJRT_Extension_Base}(0),
        CAPI.PJRT_Api_Version(
            Csize_t(sizeof(CAPI.PJRT_Api_Version)),
            Ptr{CAPI.PJRT_Extension_Base}(0),
            Cint(CAPI.PJRT_API_MAJOR),
            Cint(CAPI.PJRT_API_MINOR),
        ),
        fns...,
    )

    global _PJRT_API_MEM = Libc.malloc(sizeof(CAPI.PJRT_Api))
    unsafe_store!(Ptr{CAPI.PJRT_Api}(_PJRT_API_MEM), api_val)

    return nothing
end

# ============================================================
# Public API
# ============================================================

"""
    make_client() -> Ptr{Cvoid}

Create a Metal PJRT client by registering our @cfunction callbacks with
libReactantExtra via MakeClientFromApi. Returns a PjRtClient*.
"""
function make_client()
    api_ptr = Ptr{Cvoid}(_PJRT_API_MEM)
    client_ptr = Reactant.XLA.PJRT.MakeClientFromApi(api_ptr, "metal", "METAL")
    return client_ptr
end
