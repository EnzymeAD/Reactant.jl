module Julia

# DESIGN NOTES
# when dealing with memory at the Julia - C FFI boundary, there are two directions we can follow:
# - rely on Julia's GC, which requires to...
#   - globally mark objects
#   - refer to them via `pointer_from_objref`, `unsafe_pointer_to_objref`
#   - use `mutable` objects only (immutable objects do not have a )
# - manually allocate/free memory, which requires to...
#   - use `Libc`
#   - `isbits` objects (i.e. not mutable nor immutable with mutable fields)
#
# for this first PoC, I have decided to manually allocate objects because...
# - it avoids any kind of global state which can be buggy and difficult to debug
# - it's more akin to be trim-able for AOT compilation
#
# note that we can change to the other strategy (global GC marks) at any time as it eases further Julia integration.
# while both strategies can be used at the same time (manual alloc for some objects, global GC marks for others),
# this is severely discouraged as it leads to a complex design with no apparent gaints.

using Base: unsafe_convert
using Reactant.XLA: CAPI
using Reactant.XLA.CAPI: csizeof

# helpers
@inline strdup(str::Union{Cstring,String}) = ccall(:strdup, Cstring, (Cstring,), str)
@inline strlen(str::Cstring) = ccall(:strlen, Csize_t, (Cstring,), str)

# malloc'd copy of `n` elements; allocates at least 1 element so that `C_NULL` always means out-of-memory
function memdup(src::Ptr{T}, n::Integer) where {T}
    dst = Ptr{T}(Libc.malloc(max(n, 1) * sizeof(T)))
    dst == C_NULL && return dst
    unsafe_copyto!(dst, src, n)
    return dst
end

## vtables
const vtable_api = Ref{CAPI.PJRT_Api}()
const vtable_error = Ref{CAPI.PJRT_Error_FunctionTable}()
const vtable_memory = Ref{CAPI.PJRT_Memory_FunctionTable}()

## Error
struct Error
    vtable::Ptr{CAPI.PJRT_Error_FunctionTable}
    code::CAPI.PJRT_Error_Code
    message::Cstring

    function Error(code::CAPI.PJRT_Error_Code, msg::Cstring)
        return Error(unsafe_convert(Ptr{CAPI.PJRT_Error_FunctionTable}, vtable_error), code, msg)
    end
end

Error(code, msg::AbstractString) = Error(code, Symbol(msg))
Error(code, msg::Symbol) = Error(code, Cstring(unsafe_convert(Ptr{UInt8}, msg)))

Error(code) = Error(code, Symbol())

# function ok()
#     ptr = unsafe_convert(Ptr{Error}, Libc.malloc(sizeof(Error)))
#     unsafe_store!(ptr, Error(CAPI.PJRT_Error_Code_OK))
#     return ptr
# end

# XLA considers `nullptr` to be OK as it avoids implementing stuff or 
ok() = C_NULL

function unimplemented()
    ptr = unsafe_convert(Ptr{Error}, Libc.malloc(sizeof(Error)))
    unsafe_store!(
        ptr,
        Error(CAPI.PJRT_Error_Code_UNIMPLEMENTED, "this feature is not implemented yet!"),
    )
    return ptr
end

function invalid_argument()
    ptr = unsafe_convert(Ptr{Error}, Libc.malloc(sizeof(Error)))
    unsafe_store!(ptr, Error(CAPI.PJRT_Error_Code_INVALID_ARGUMENT, "invalid argument"))
    return ptr
end

function resource_exhausted()
    ptr = unsafe_convert(Ptr{Error}, Libc.malloc(sizeof(Error)))
    unsafe_store!(ptr, Error(CAPI.PJRT_Error_Code_RESOURCE_EXHAUSTED, "out of memory"))
    return ptr
end

function error_destroy(args::Ptr{CAPI.PJRT_Error_Destroy_Args})::Cvoid
    args.error != C_NULL && Libc.free(args.error)
    return nothing
end

function error_message(args::Ptr{CAPI.PJRT_Error_Message_Args})::Ptr{}
    args.error == C_NULL && return nothing
    unsafe_store!(args.message, args.error.message)
    unsafe_store!(args.message_size, strlen(args.error.message))
    return ok()
end

# this is required by CAPI.PJRT_Api
function error_getcode(args::Ptr{CAPI.PJRT_Error_GetCode_Args})::Ptr{CAPI.PJRT_Error}
    error = unsafe_load(args.error)
    unsafe_store!(args.code, error.code)
    return ok()
end

# this is required by CAPI.PJRT_Error_FunctionTable
function error_getcode_obj(args::Ptr{CAPI.PJRT_Error_GetCode_Args})::CAPI.PJRT_Error_Code
    error = unsafe_load(args.error)
    return error.code
end

function error_for_each_payload(args::Ptr{CAPI.PJRT_Error_ForEachPayload_Args})::Cvoid
    # do nothing
    return nothing
end

## Plugin
function plugin_initialize(
    args::Ptr{CAPI.PJRT_Plugin_Initialize_Args}
)::Ptr{CAPI.PJRT_Error}
    # do nothing
    return ok()
end

function plugin_attributes(
    args::Ptr{CAPI.PJRT_Plugin_Attributes_Args}
)::Ptr{CAPI.PJRT_Error}
    # TODO return key-value pairs of `xla_version`, `stablehlo_current_version` and `stablehlo_minimum_version`
    return ok()
end

## Memory
struct Memory
    vtable::Ptr{CAPI.PJRT_Memory_FunctionTable}
    id::Cint
    kind::Symbol
    devices::Ptr{Ptr{CAPI.PJRT_Device}}
    num_devices::Csize_t

    function Memory(id, kind, devices, num_devices)
        return new(unsafe_convert(Ptr{CAPI.PJRT_Memory_FunctionTable}, vtable_memory), id, kind, devices, num_devices)
    end
end

function _memory_destroy(memory::Memory)
    # only the array is owned by the memory, the devices are owned by the client
    Libc.free(memory.devices)
    return nothing
end

function memory_id(args::Ptr{CAPI.PJRT_Memory_Id_Args})::Ptr{CAPI.PJRT_Error}
    unsafe_store!(args.id, args.memory.id)
    return ok()
end

function memory_kind(args::Ptr{CAPI.PJRT_Memory_Kind_Args})::Ptr{CAPI.PJRT_Error}
    sym = unsafe_load(unsafe_convert(Ptr{Symbol}, args.memory))
    cstr = Cstring(unsafe_convert(Ptr{UInt8}, sym))
    unsafe_store!(args.kind, cstr)
    unsafe_store!(args.kind_size, strlen(cstr))
    return ok()
end

function memory_debug_string(
    args::Ptr{CAPI.PJRT_Memory_DebugString_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_load(Ptr{Memory}(unsafe_load(args.memory)))
    # TODO move interpolated string to `Memory` field instead of interning
    str = "Memory(id=$(memory.id), kind=$(memory.kind), platform=julia)"
    cstr = Cstring(Base.unsafe_convert(Ptr{UInt8}, Symbol(str)))
    unsafe_store!(args.debug_string, cstr)
    unsafe_store!(args.debug_string_size, strlen(cstr))
    return ok()
end

function memory_to_string(args::Ptr{CAPI.PJRT_Memory_ToString_Args})::Ptr{CAPI.PJRT_Error}
    # if interpolation is used, we need to move the string to a `Memory` field
    cstr = Cstring(Base.unsafe_convert(Ptr{UInt8}, Symbol("Memory managed by Julia")))
    unsafe_store!(args.to_string, cstr)
    unsafe_store!(args.to_string_size, strlen(cstr))
    return ok()
end

function memory_addressable_by_devices(
    args::Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args}
)::Ptr{CAPI.PJRT_Error}
    # TODO
    return ok()
end

function memory_kind_id(args::Ptr{CAPI.PJRT_Memory_Kind_Args})::Ptr{CAPI.PJRT_Error}
    # arbitrary value
    unsafe_store!(args.kind_id, Cint(1))
    return ok()
end

## DeviceDescription
struct DeviceDescription
    id::Cint
    pid::Cint
    debug_string::Cstring
    to_string::Cstring
end

function DeviceDescription()
    id = 0
    pid = 0
    debug_string = strdup("julia: id=$id")
    to_string = strdup("JuliaDevice(id=$id)")
    return DeviceDescription(id, pid, debug_string, to_string)
end

# takes the value and not a pointer because `DeviceDescription` is embedded in `Device`, so only the strings are freed
function _device_description_destroy(description::DeviceDescription)
    Libc.free(description.debug_string)
    Libc.free(description.to_string)
    return nothing
end

function device_description_id(
    args::Ptr{CAPI.PJRT_DeviceDescription_Id_Args}
)::Ptr{CAPI.PJRT_Error}
    id = unsafe_load(unsafe_convert(Ptr{Device}, args.device_description).id)
    unsafe_store!(args.id, id)
    return ok()
end

function device_description_process_index(
    args::Ptr{CAPI.PJRT_DeviceDescription_ProcessIndex_Args}
)::Ptr{CAPI.PJRT_Error}
    pid = unsafe_load(unsafe_convert(Ptr{Device}, args.device_description).pid)
    unsafe_store!(args.process_index, pid)
    return ok()
end

function device_description_attributes(
    args::Ptr{CAPI.PJRT_DeviceDescription_Attributes_Args}
)::Ptr{CAPI.PJRT_Error}
    unsafe_store!(args.num_attributes, Cint(0))
    unsafe_store!(args.attributes, C_NULL)
    return ok()
end

function device_description_kind(
    args::Ptr{CAPI.PJRT_DeviceDescription_Kind_Args}
)::Ptr{CAPI.PJRT_Error}
    cstr = Cstring(unsafe_convert(Ptr{UInt8}, Symbol("julia")))
    unsafe_store!(args.device_kind, cstr)
    unsafe_store!(args.device_kind_size, strlen(cstr))
    return ok()
end

function device_description_debug_string(
    args::Ptr{CAPI.PJRT_DeviceDescription_DebugString_Args}
)::Ptr{CAPI.PJRT_Error}
    unsafe_store!(args.debug_string, unsafe_convert(Ptr{Device}, args.device_description).debug_string)
    unsafe_store!(args.debug_string_size, strlen(unsafe_convert(Ptr{Device}, args.device_description).debug_string))
    return ok()
end

function device_description_to_string(
    args::Ptr{CAPI.PJRT_DeviceDescription_ToString_Args}
)::Ptr{CAPI.PJRT_Error}
    unsafe_store!(args.to_string, unsafe_convert(Ptr{Device}, args.device_description).to_string)
    unsafe_store!(args.to_string_size, strlen(unsafe_convert(Ptr{Device}, args.device_description).to_string))
    return ok()
end

## Device
struct Device
    description::DeviceDescription
    local_hardware_id::Cint
    # the first memory is the default memory
    memories::Ptr{Ptr{Memory}}
    num_memories::Csize_t
end

function _device_destroy(device::Device)
    _device_description_destroy(device.description)

    # only the array is owned by the device, the memories are owned by the client
    Libc.free(device.memories)
    return nothing
end

function device_get_description(
    args::Ptr{CAPI.PJRT_Device_GetDescription_Args}
)::Ptr{CAPI.PJRT_Error}
    device = Ptr{Device}(unsafe_load(args.device))
    unsafe_store!(
        args.device_description, Ptr{CAPI.PJRT_DeviceDescription}(device.description)
    )
    return ok()
end

function device_is_addressable(
    args::Ptr{CAPI.PJRT_Device_IsAddressable_Args}
)::Ptr{CAPI.PJRT_Error}
    # single-process backend: the client can issue commands to all of its devices
    unsafe_store!(args.is_addressable, true)
    return ok()
end

function device_local_hardware_id(
    args::Ptr{CAPI.PJRT_Device_LocalHardwareId_Args}
)::Ptr{CAPI.PJRT_Error}
    device = Ptr{Device}(unsafe_load(args.device))
    unsafe_store!(args.local_hardware_id, unsafe_load(device.local_hardware_id))
    return ok()
end

function device_addressable_memories(
    args::Ptr{CAPI.PJRT_Device_AddressableMemories_Args}
)::Ptr{CAPI.PJRT_Error}
    device = Ptr{Device}(unsafe_load(args.device))
    unsafe_store!(args.memories, Ptr{Ptr{CAPI.PJRT_Memory}}(unsafe_load(device.memories)))
    unsafe_store!(args.num_memories, unsafe_load(device.num_memories))
    return ok()
end

function device_default_memory(
    args::Ptr{CAPI.PJRT_Device_DefaultMemory_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_load(Ptr{Device}(unsafe_load(args.device)))
    unsafe_store!(args.memory, Ptr{CAPI.PJRT_Memory}(unsafe_load(device.memories)))
    return ok()
end

function device_memory_stats(
    args::Ptr{CAPI.PJRT_Device_MemoryStats_Args}
)::Ptr{CAPI.PJRT_Error}
    # allocations are not tracked yet; `pjrt_c_api.h` allows returning `UNIMPLEMENTED` here
    return unimplemented()
end

## Buffer
struct Buffer
    # malloc'd array of `size_in_bytes` bytes, owned by the buffer
    data::Ptr{Cvoid}
    size_in_bytes::Csize_t
    type::CAPI.PJRT_Buffer_Type
    # malloc'd arrays of `num_dims` elements, owned by the buffer
    dims::Ptr{Int64}
    minor_to_major::Ptr{Int64}
    num_dims::Csize_t
    device::Ptr{Device}
end

function buffer_destroy(args::Ptr{CAPI.PJRT_Buffer_Destroy_Args})::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{Buffer}(unsafe_load(args.buffer))
    ptr == C_NULL && return ok()

    buffer = unsafe_load(ptr)
    Libc.free(buffer.data)
    Libc.free(buffer.dims)
    Libc.free(buffer.minor_to_major)
    Libc.free(ptr)
    return ok()
end

function buffer_element_type(
    args::Ptr{CAPI.PJRT_Buffer_ElementType_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    unsafe_store!(args.type, buffer.type)
    return ok()
end

function buffer_dimensions(
    args::Ptr{CAPI.PJRT_Buffer_Dimensions_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    unsafe_store!(args.dims, buffer.dims)
    unsafe_store!(args.num_dims, buffer.num_dims)
    return ok()
end

# deprecated, but required for backward compatibility
# use layouts extension instead
function buffer_get_memory_layout(
    args::Ptr{CAPI.PJRT_Buffer_GetMemoryLayout_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))

    # dense layout without tiling; `minor_to_major` has the lifetime of `buffer` as required by PJRT
    tiled = CAPI.PJRT_Buffer_MemoryLayout_Tiled(
        csizeof(CAPI.PJRT_Buffer_MemoryLayout_Tiled),
        C_NULL,
        buffer.minor_to_major,
        buffer.num_dims,
        C_NULL,
        C_NULL,
        0,
    )
    unsafe_store!(args.layout.struct_size, csizeof(CAPI.PJRT_Buffer_MemoryLayout))
    unsafe_store!(args.layout.extension_start, C_NULL)
    unsafe_store!(args.layout.tiled, tiled)
    unsafe_store!(args.layout.type, CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled)
    return ok()
end

function buffer_on_device_size_in_bytes(
    args::Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    unsafe_store!(args.on_device_size_in_bytes, buffer.size_in_bytes)
    return ok()
end

function buffer_copy_to_device(
    args::Ptr{CAPI.PJRT_Buffer_CopyToDevice_Args}
)::Ptr{CAPI.PJRT_Error}
    src = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    dst_device = Ptr{Device}(unsafe_load(args.dst_device))

    # `pjrt_c_api.h` requires an error if the buffer is already on `dst_device`
    src.device == dst_device && return invalid_argument()

    data = memdup(Ptr{UInt8}(src.data), src.size_in_bytes)
    dims = memdup(src.dims, src.num_dims)
    minor_to_major = memdup(src.minor_to_major, src.num_dims)
    dst = Ptr{Buffer}(Libc.malloc(sizeof(Buffer)))
    if data == C_NULL || dims == C_NULL || minor_to_major == C_NULL || dst == C_NULL
        Libc.free(data)
        Libc.free(dims)
        Libc.free(minor_to_major)
        Libc.free(dst)
        return resource_exhausted()
    end

    unsafe_store!(
        dst,
        Buffer(
            data,
            src.size_in_bytes,
            src.type,
            dims,
            minor_to_major,
            src.num_dims,
            dst_device,
        ),
    )

    # caller is responsible for freeing `dst_buffer` with `PJRT_Buffer_Destroy`
    unsafe_store!(args.dst_buffer, Ptr{CAPI.PJRT_Buffer}(dst))
    return ok()
end

function buffer_is_on_cpu(args::Ptr{CAPI.PJRT_Buffer_IsOnCpu_Args})::Ptr{CAPI.PJRT_Error}
    # buffers are allocated with `Libc.malloc` on host memory
    unsafe_store!(args.is_on_cpu, true)
    return ok()
end

## Client
struct Client
    devices::Ptr{Ptr{Device}}
    num_devices::Cint

    memories::Ptr{Ptr{Memory}}
    num_memories::Cint
end

function client_create(args::Ptr{CAPI.PJRT_Client_Create_Args})::Ptr{CAPI.PJRT_Error}
    # single device with a single memory
    num_devices = 1
    num_memories = 1

    client = Ptr{Client}(Libc.malloc(sizeof(Client)))
    devices = Ptr{Ptr{Device}}(Libc.malloc(sizeof(Ptr{Device}) * num_devices))
    memories = Ptr{Ptr{Memory}}(Libc.malloc(sizeof(Ptr{Memory}) * num_memories))
    device = Ptr{Device}(Libc.malloc(sizeof(Device)))
    memory = Ptr{Memory}(Libc.malloc(sizeof(Memory)))
    device_memories = Ptr{Ptr{Memory}}(Libc.malloc(sizeof(Ptr{Memory})))
    memory_devices = Ptr{Ptr{CAPI.PJRT_Device}}(Libc.malloc(sizeof(Ptr{CAPI.PJRT_Device})))
    if client == C_NULL ||
        devices == C_NULL ||
        memories == C_NULL ||
        device == C_NULL ||
        memory == C_NULL ||
        device_memories == C_NULL ||
        memory_devices == C_NULL
        Libc.free(client)
        Libc.free(devices)
        Libc.free(memories)
        Libc.free(device)
        Libc.free(memory)
        Libc.free(device_memories)
        Libc.free(memory_devices)
        return resource_exhausted()
    end

    unsafe_store!(memory_devices, Ptr{CAPI.PJRT_Device}(device))
    unsafe_store!(memory, Memory(0, :device, memory_devices, 1))

    unsafe_store!(device_memories, memory)
    unsafe_store!(device, Device(DeviceDescription(), -1, device_memories, 1))

    unsafe_store!(devices, device)
    unsafe_store!(memories, memory)
    unsafe_store!(client, Client(devices, num_devices, memories, num_memories))

    unsafe_store!(args.client, Ptr{CAPI.PJRT_Client}(client))
    return ok()
end

function client_destroy(args::Ptr{CAPI.PJRT_Client_Destroy_Args})::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{Client}(unsafe_load(args.client))
    if ptr != C_NULL
        client = unsafe_load(ptr)
        for i in 1:(client.num_devices)
            device = unsafe_load(client.devices, i)
            _device_destroy(unsafe_load(device))
            Libc.free(device)
        end
        Libc.free(client.devices)

        for i in 1:(client.num_memories)
            memory = unsafe_load(client.memories, i)
            _memory_destroy(unsafe_load(memory))
            Libc.free(memory)
        end
        Libc.free(client.memories)

        Libc.free(ptr)
        unsafe_store!(args.client, C_NULL)
    end
    return ok()
end

function client_platform_name(
    args::Ptr{CAPI.PJRT_Client_PlatformName_Args}
)::Ptr{CAPI.PJRT_Error}
    cstr = Cstring(unsafe_convert(Ptr{UInt8}, :julia))
    unsafe_store!(args.platform_name, cstr)
    unsafe_store!(args.platform_name_size, strlen(cstr))
    return ok()
end

function client_process_index(
    args::Ptr{CAPI.PJRT_Client_ProcessIndex_Args}
)::Ptr{CAPI.PJRT_Error}
    unsafe_store!(args.process_index, Cint(0))
    return ok()
end

function client_platform_version(
    args::Ptr{CAPI.PJRT_Client_PlatformVersion_Args}
)::Ptr{CAPI.PJRT_Error}
    cstr = Cstring(unsafe_convert(Ptr{UInt8}, Symbol(VERSION)))
    unsafe_store!(args.platform_version, cstr)
    unsafe_store!(args.platform_version_size, strlen(cstr))
    return ok()
end

function client_devices(args::Ptr{CAPI.PJRT_Client_Devices_Args})::Ptr{CAPI.PJRT_Error}
    client = unsafe_load(Ptr{Client}(unsafe_load(args.client)))
    unsafe_store!(args.devices, Ptr{Ptr{CAPI.PJRT_Device}}(client.devices))
    unsafe_store!(args.num_devices, client.num_devices)
    return ok()
end

function client_addressable_devices(
    args::Ptr{CAPI.PJRT_Client_AddressableDevices_Args}
)::Ptr{CAPI.PJRT_Error}
    # single-process backend: all devices are addressable
    client = unsafe_load(Ptr{Client}(unsafe_load(args.client)))
    unsafe_store!(args.addressable_devices, Ptr{Ptr{CAPI.PJRT_Device}}(client.devices))
    unsafe_store!(args.num_addressable_devices, client.num_devices)
    return ok()
end

function client_compile(args::Ptr{CAPI.PJRT_Client_Compile_Args})::Ptr{CAPI.PJRT_Error}
    # TODO
end

function client_buffer_from_host_buffer(
    args::Ptr{CAPI.PJRT_Client_BufferFromHostBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    # TODO
end

for T in [Error, Device, DeviceDescription, Memory, Buffer, Client]
    exprs =
        map(Iterators.zip(fieldnames(T), fieldtypes(DeviceDescription))) do (fname, ftype)
            :(
                f === $(QuoteNode(fname)) &&
                return Ptr{$ftype}(x + $(fieldoffset($T, fname)))
            )
        end
    @eval begin
        function Base.getproperty(x::Ptr{$T}, f::Symbol)
            $(exprs...)
            return getfield(x, f)
        end

        function Base.propertynames(x::Ptr{$T}; private=false)
            return ($(propertynames($T)...), Base.propertynames(Ptr{Cvoid}; private)...)
        end
    end
end

function __init__()
    # generic handler for unimplemented PJRT API functions
    c_unimplemented = @cfunction(unimplemented, Ptr{CAPI.PJRT_Error}, (Ptr{Cvoid},))

    #! format: off
    vtable_error[] = CAPI.PJRT_Error_FunctionTable(
        #= struct_size =# csizeof(PJRT_Error_FunctionTable),
        #= instance_size =# sizeof(Error),
        #= extension_start =# C_NULL,
        #= destroy =# @cfunction(error_destroy, (), (Ptr{CAPI.PJRT_Error_Destroy_Args},)),
        #= message =# @cfunction(error_message, (), (Ptr{CAPI.PJRT_Error_Message_Args},)),
        #= get_code =# @cfunction(error_getcode, (), (Ptr{CAPI.PJRT_Error_GetCode_Args},)),
        #= for_each_payload =# c_unimplemented,
    )

    vtable_memory[] = CAPI.PJRT_Memory_FunctionTable(
        #= struct_size =# csizeof(CAPI.PJRT_Memory_FunctionTable),
        #= extension_start =# C_NULL,
        #= instance_struct_size =# sizeof(Memory),
        #= get_user_data =# c_unimplemented,
        #= set_user_data =# c_unimplemented,
    )

    vtable_api[] = CAPI.PJRT_Api(
        #= struct_size =# csizeof(CAPI.PJRT_Api),
        #= extension_start =# C_NULL,
        #= pjrt_api_version =# CAPI.PJRT_Api_Version(
            #= struct_size =# csizeof(CAPI.PJRT_Api_Version),
            #= extension_start =# C_NULL,
            #= major_version =# CAPI.PJRT_API_MAJOR,
            #= minor_version =# CAPI.PJRT_API_MINOR,
        ),
        #= PJRT_Error_Destroy =# @cfunction(error_destroy, (), (Ptr{CAPI.PJRT_Error_Destroy_Args},)),
        #= PJRT_Error_Message =# @cfunction(error_message, (), (Ptr{CAPI.PJRT_Error_Message_Args},)),
        #= PJRT_Error_GetCode =# @cfunction(error_getcode, (), (Ptr{CAPI.PJRT_Error_GetCode_Args},)),
        #= PJRT_Plugin_Initialize =# @cfunction(plugin_initialize, (), (Ptr{CAPI.PJRT_Plugin_Attributes_Args},)),
        #= PJRT_Plugin_Attributes =# @cfunction(plugin_attributes, (), (Ptr{CAPI.PJRT_Plugin_Attributes_Args},)),
        #= PJRT_Event_Destroy =# c_unimplemented,
        #= PJRT_Event_IsReady =# c_unimplemented,
        #= PJRT_Event_Error =# c_unimplemented,
        #= PJRT_Event_Await =# c_unimplemented,
        #= PJRT_Event_OnReady =# c_unimplemented,
        #= PJRT_Client_Create =# @cfunction(client_create, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Create_Args},)),
        #= PJRT_Client_Destroy =# @cfunction(client_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Destroy_Args},)),
        #= PJRT_Client_PlatformName =# @cfunction(client_platform_name, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_PlatformName_Args},)),
        #= PJRT_Client_ProcessIndex =# @cfunction(client_process_index, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_ProcessIndex_Args},)),
        #= PJRT_Client_PlatformVersion =# @cfunction(client_platform_version, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_PlatformVersion_Args},)),
        #= PJRT_Client_Devices =# @cfunction(client_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Devices_Args},)),
        #= PJRT_Client_AddressableDevices =# @cfunction(client_addressable_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_AddressableDevices_Args},)),
        #= PJRT_Client_LookupDevice =# c_unimplemented,
        #= PJRT_Client_LookupAddressableDevice =# c_unimplemented,
        #= PJRT_Client_AddressableMemories =# c_unimplemented,
        #= PJRT_Client_Compile =# @cfunction(client_compile, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Compile_Args},)),
        #= PJRT_Client_DefaultDeviceAssignment =# c_unimplemented,
        #= PJRT_Client_BufferFromHostBuffer =# @cfunction(client_buffer_from_host_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_BufferFromHostBuffer_Args},)),
        #= PJRT_DeviceDescription_Id =# @cfunction(device_description_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Id_Args},)),
        #= PJRT_DeviceDescription_ProcessIndex =# @cfunction(device_description_process_index, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_ProcessIndex_Args},)),
        #= PJRT_DeviceDescription_Attributes =# @cfunction(device_description_attributes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Attributes_Args},)),
        #= PJRT_DeviceDescription_Kind =# @cfunction(device_description_kind, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Kind_Args},)),
        #= PJRT_DeviceDescription_DebugString =# @cfunction(device_description_debug_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_DebugString_Args},)),
        #= PJRT_DeviceDescription_ToString =# @cfunction(device_description_to_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_ToString_Args},)),
        #= PJRT_Device_GetDescription =# @cfunction(device_get_description, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_GetDescription_Args},)),
        #= PJRT_Device_IsAddressable =# @cfunction(device_is_addressable, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_IsAddressable_Args},)),
        #= PJRT_Device_LocalHardwareId =# @cfunction(device_local_hardware_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_LocalHardwareId_Args},)),
        #= PJRT_Device_AddressableMemories =# @cfunction(device_addressable_memories, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_AddressableMemories_Args},)),
        #= PJRT_Device_DefaultMemory =# @cfunction(device_default_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_DefaultMemory_Args},)),
        #= PJRT_Device_MemoryStats =# @cfunction(device_memory_stats, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_MemoryStats_Args},)),
        #= PJRT_Memory_Id =# @cfunction(memory_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Id_Args},)),
        #= PJRT_Memory_Kind =# @cfunction(memory_kind, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Kind_Args},)),
        #= PJRT_Memory_DebugString =# @cfunction(memory_debug_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_DebugString_Args},)),
        #= PJRT_Memory_ToString =# @cfunction(memory_to_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_ToString_Args},)),
        #= PJRT_Memory_AddressableByDevices =# @cfunction(memory_addressable_by_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args},)),
        #= PJRT_Executable_Destroy =# c_unimplemented,
        #= PJRT_Executable_Name =# c_unimplemented,
        #= PJRT_Executable_NumReplicas =# c_unimplemented,
        #= PJRT_Executable_NumPartitions =# c_unimplemented,
        #= PJRT_Executable_NumOutputs =# c_unimplemented,
        #= PJRT_Executable_SizeOfGeneratedCodeInBytes =# c_unimplemented,
        #= PJRT_Executable_GetCostAnalysis =# c_unimplemented,
        #= PJRT_Executable_OutputMemoryKinds =# c_unimplemented,
        #= PJRT_Executable_OptimizedProgram =# c_unimplemented,
        #= PJRT_Executable_Serialize =# c_unimplemented,
        #= PJRT_LoadedExecutable_Destroy =# c_unimplemented,
        #= PJRT_LoadedExecutable_GetExecutable =# c_unimplemented,
        #= PJRT_LoadedExecutable_AddressableDevices =# c_unimplemented,
        #= PJRT_LoadedExecutable_Delete =# c_unimplemented,
        #= PJRT_LoadedExecutable_IsDeleted =# c_unimplemented,
        #= PJRT_LoadedExecutable_Execute =# c_unimplemented,
        #= PJRT_Executable_DeserializeAndLoad =# c_unimplemented,
        #= PJRT_LoadedExecutable_Fingerprint =# c_unimplemented,
        #= PJRT_Buffer_Destroy =# @cfunction(buffer_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Destroy_Args},)),
        #= PJRT_Buffer_ElementType =# @cfunction(buffer_element_type, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ElementType_Args},)),
        #= PJRT_Buffer_Dimensions =# @cfunction(buffer_dimensions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Dimensions_Args},)),
        #= PJRT_Buffer_UnpaddedDimensions =# c_unimplemented,
        #= PJRT_Buffer_DynamicDimensionIndices =# c_unimplemented,
        #= PJRT_Buffer_GetMemoryLayout =# @cfunction(buffer_get_memory_layout, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_GetMemoryLayout_Args},)),
        #= PJRT_Buffer_OnDeviceSizeInBytes =# @cfunction(buffer_on_device_size_in_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args},)),
        #= PJRT_Buffer_Device =# c_unimplemented,
        #= PJRT_Buffer_Memory =# c_unimplemented,
        #= PJRT_Buffer_Delete =# c_unimplemented,
        #= PJRT_Buffer_IsDeleted =# c_unimplemented,
        #= PJRT_Buffer_CopyToDevice =# @cfunction(buffer_copy_to_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyToDevice_Args},)),
        #= PJRT_Buffer_ToHostBuffer =# c_unimplemented,
        #= PJRT_Buffer_IsOnCpu =# @cfunction(buffer_is_on_cpu, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IsOnCpu_Args},)),
        #= PJRT_Buffer_ReadyEvent =# c_unimplemented,
        #= PJRT_Buffer_UnsafePointer =# c_unimplemented,
        #= PJRT_Buffer_IncreaseExternalReferenceCount =# c_unimplemented,
        #= PJRT_Buffer_DecreaseExternalReferenceCount =# c_unimplemented,
        #= PJRT_Buffer_OpaqueDeviceMemoryDataPointer =# c_unimplemented,
        #= PJRT_CopyToDeviceStream_Destroy =# c_unimplemented,
        #= PJRT_CopyToDeviceStream_AddChunk =# c_unimplemented,
        #= PJRT_CopyToDeviceStream_TotalBytes =# c_unimplemented,
        #= PJRT_CopyToDeviceStream_GranuleSize =# c_unimplemented,
        #= PJRT_CopyToDeviceStream_CurrentBytes =# c_unimplemented,
        #= PJRT_TopologyDescription_Create =# c_unimplemented,
        #= PJRT_TopologyDescription_Destroy =# c_unimplemented,
        #= PJRT_TopologyDescription_PlatformName =# c_unimplemented,
        #= PJRT_TopologyDescription_PlatformVersion =# c_unimplemented,
        #= PJRT_TopologyDescription_GetDeviceDescriptions =# c_unimplemented,
        #= PJRT_TopologyDescription_Serialize =# c_unimplemented,
        #= PJRT_TopologyDescription_Attributes =# c_unimplemented,
        #= PJRT_Compile =# c_unimplemented,
        #= PJRT_Executable_OutputElementTypes =# c_unimplemented,
        #= PJRT_Executable_OutputDimensions =# c_unimplemented,
        #= PJRT_Buffer_CopyToMemory =# c_unimplemented,
        #= PJRT_Client_CreateViewOfDeviceBuffer =# c_unimplemented,
        #= PJRT_Executable_Fingerprint =# c_unimplemented,
        #= PJRT_Client_TopologyDescription =# c_unimplemented,
        #= PJRT_Executable_GetCompiledMemoryStats =# c_unimplemented,
        #= PJRT_Memory_Kind_Id =# c_unimplemented,
        #= PJRT_ExecuteContext_Create =# c_unimplemented,
        #= PJRT_ExecuteContext_Destroy =# c_unimplemented,
        #= PJRT_Buffer_CopyRawToHost =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_Destroy =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_TransferData =# c_unimplemented,
        #= PJRT_Client_CreateBuffersForAsyncHostToDevice =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_Device =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_BufferCount =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_BufferSize =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_SetBufferError =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_AddMetadata =# c_unimplemented,
        #= PJRT_Client_DmaMap =# c_unimplemented,
        #= PJRT_Client_DmaUnmap =# c_unimplemented,
        #= PJRT_Client_CreateUninitializedBuffer =# c_unimplemented,
        #= PJRT_Client_UpdateGlobalProcessInfo =# c_unimplemented,
        #= PJRT_TopologyDescription_Deserialize =# c_unimplemented,
        #= PJRT_Client_CreateAliasBuffer =# c_unimplemented,
        #= PJRT_Client_FulfillAliasBuffer =# c_unimplemented,
        #= PJRT_LoadedExecutable_GetDeviceAssignment =# c_unimplemented,
        #= PJRT_Client_CreateErrorBuffer =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_TransferLiteral =# c_unimplemented,
        #= PJRT_Buffer_CopyRawToHostFuture =# c_unimplemented,
        #= PJRT_Device_PoisonExecution =# c_unimplemented,
        #= PJRT_Device_CreateAsyncTrackingEvent =# c_unimplemented,
        #= PJRT_AsyncTrackingEvent_Destroy =# c_unimplemented,
        #= PJRT_Executable_GetCompileOptions =# c_unimplemented,
        #= PJRT_Buffer_DonateWithControlDependency =# c_unimplemented,
        #= PJRT_Event_Create =# c_unimplemented,
        #= PJRT_Event_Set =# c_unimplemented,
        #= PJRT_Device_GetAttributes =# c_unimplemented,
        #= PJRT_Client_Load =# c_unimplemented,
        #= PJRT_LoadedExecutable_AddressableDeviceLogicalIds =# c_unimplemented,
        #= PJRT_Buffer_Bitcast =# c_unimplemented,
        #= PJRT_Error_ForEachPayload =# c_unimplemented,
        #= PJRT_TopologyDescription_Fingerprint =# c_unimplemented,
        #= PJRT_Executable_ParameterMemoryKinds =# c_unimplemented,
        #= PJRT_Device_ClearMemoryStats =# c_unimplemented,
        #= PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace =# c_unimplemented,
        #= PJRT_TopologyDescription_GetMemorySpaceKindIds =# c_unimplemented,
    )
    #! format: on
end

function make_pjrt_client()
    # TODO
end

end
