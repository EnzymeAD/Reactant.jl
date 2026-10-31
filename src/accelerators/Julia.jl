module Julia

# DESIGN NOTES
# when dealing with memory at the Julia - C FFI boundary, there are two directions we can follow:
# - rely on Julia's GC, which requires to...
#   - globally mark objects
#   - refer to them via `pointer_from_objref`, `unsafe_pointer_to_objref`
#   - use `mutable` objects only (immutable objects do not have a stable memory address)
# - manually allocate/free memory, which requires to...
#   - use `Libc`
#   - only work with `isbits` objects (i.e. not mutable nor immutable with mutable fields)
#
# after trying both designs, i ultimately opted for the "globally GC rooted" design because
# - has a simpler design
# - it's easier to extend
# - safer against memory bugs
# - almost no overhead compared to manual allocations
#
# notes:
# - all objects passed through the XLA/PJRT boundary need to be GC rooted (i.e. use `(un)root!`), which requires objects to be `mutable` to have a stable pointer address
# - some objects require as a first field a pointer to their vtable
# - if for some PJRT object, the API requires it to return a string, store the string inside the object so that it's correctly freed (owner is the object)
# - if for some PJRT object, the API requires it to return a list, store the list as a `Vector{X}` where `X` is an abstract type
#   - the Julia array ABI layout states that an array with abstract eltype is represented as a pointer of pointers (no inline storage. check: https://docs.julialang.org/en/v1/manual/embedding/#Working-with-Arrays)
#   - the stored pointers are of type `jl_value_t*` which points to the object data (the type tag is stored before that pointer address. check https://docs.julialang.org/en/v1/devdocs/object/#Object-layout-(jl_value_t))
#   - once the object is created, DO NOT resize the array
# - DO NOT use moving GC (e.g. MMTk) as it would break `pointer_from_objref`

using Base: unsafe_convert
using Reactant.XLA.CAPI: CAPI, csizeof

# helpers
# malloc'd copy of `n` elements; allocates at least 1 element so that `C_NULL` always means out-of-memory
function memdup(src::Ptr{T}, n::Integer) where {T}
    dst = Ptr{T}(Libc.malloc(max(n, 1) * sizeof(T)))
    dst == C_NULL && return dst
    unsafe_copyto!(dst, src, n)
    return dst
end

# GC roots of objects whose pointers are handed to C; objects stay alive until they are unmarked
const gc_roots = IdSet{Any}()
const gc_roots_lock = ReentrantLock()

function root!(obj::T) where {T}
    ismutabletype(T) || error("Only mutable types can be GC rooted")
    @lock gc_roots_lock push!(gc_roots, obj)
    return Ptr{T}(pointer_from_objref(obj))
end

unroot!(ptr::Ptr) = unroot!(unsafe_pointer_to_objref(ptr))
function unroot!(obj)
    @lock gc_roots_lock delete!(gc_roots, obj)
    return nothing
end

function pjrt_elsize(T::CAPI.PJRT_Buffer_Type)
    if T in (
        CAPI.PJRT_Buffer_Type_PRED,
        CAPI.PJRT_Buffer_Type_S8,
        CAPI.PJRT_Buffer_Type_U8,
        CAPI.PJRT_Buffer_Type_F8E5M2,
        CAPI.PJRT_Buffer_Type_F8E4M3FN,
        CAPI.PJRT_Buffer_Type_F8E4M3B11FNUZ,
        CAPI.PJRT_Buffer_Type_F8E5M2FNUZ,
        CAPI.PJRT_Buffer_Type_F8E4M3FNUZ,
        CAPI.PJRT_Buffer_Type_F8E4M3,
        CAPI.PJRT_Buffer_Type_F8E3M4,
        CAPI.PJRT_Buffer_Type_F8E8M0FNU,
    )
        1
    elseif T in (
        CAPI.PJRT_Buffer_Type_S16,
        CAPI.PJRT_Buffer_Type_U16,
        CAPI.PJRT_Buffer_Type_F16,
        CAPI.PJRT_Buffer_Type_BF16,
    )
        2
    elseif T in (CAPI.PJRT_Buffer_Type_S32, CAPI.PJRT_Buffer_Type_U32, CAPI.PJRT_Buffer_Type_F32)
        4
    elseif T in (
        CAPI.PJRT_Buffer_Type_S64,
        CAPI.PJRT_Buffer_Type_U64,
        CAPI.PJRT_Buffer_Type_F64,
        CAPI.PJRT_Buffer_Type_C64,
    )
        8
    elseif type == CAPI.PJRT_Buffer_Type_C128
        16
    else
        return nothing
    end
end

## vtables
const vtable_api = Ref{CAPI.PJRT_Api}()
const vtable_error = Ref{CAPI.PJRT_Error_FunctionTable}()
const vtable_memory = Ref{CAPI.PJRT_Memory_FunctionTable}()

## Error
mutable struct Error
    const vtable::Ptr{CAPI.PJRT_Error_FunctionTable}
    const code::CAPI.PJRT_Error_Code
    const message::String

    function Error(code::CAPI.PJRT_Error_Code, message)
        return new(
            unsafe_convert(Ptr{CAPI.PJRT_Error_FunctionTable}, vtable_error),
            code,
            message,
        )
    end
end

# function ok()
#     ptr = unsafe_convert(Ptr{Error}, Libc.malloc(sizeof(Error)))
#     unsafe_store!(ptr, Error(CAPI.PJRT_Error_Code_OK))
#     return ptr
# end

# XLA considers `nullptr` to be OK as it avoids allocations
ok() = C_NULL

# the caller is responsible for freeing the error with `PJRT_Error_Destroy`, which unmarks it
function unimplemented()::Ptr{CAPI.PJRT_Error}
    return root!(
        Error(CAPI.PJRT_Error_Code_UNIMPLEMENTED, "this feature is not implemented yet!")
    )
end

function invalid_argument(reason="invalid argument")::Ptr{CAPI.PJRT_Error}
    return root!(Error(CAPI.PJRT_Error_Code_INVALID_ARGUMENT, reason))
end

function resource_exhausted(reason="out of memory")::Ptr{CAPI.PJRT_Error}
    return root!(Error(CAPI.PJRT_Error_Code_RESOURCE_EXHAUSTED, reason))
end

# these are required by CAPI.PJRT_Api
function error_destroy(args::Ptr{CAPI.PJRT_Error_Destroy_Args})::Cvoid
    error = Ptr{Error}(unsafe_load(args.error))
    error_destroy_obj(error)
    return nothing
end

function error_message(args::Ptr{CAPI.PJRT_Error_Message_Args})::Cvoid
    error = Ptr{Error}(unsafe_load(args.error))
    error_message_obj(error, args.message, args.message_size)
    return nothing
end

function error_getcode(args::Ptr{CAPI.PJRT_Error_GetCode_Args})::Ptr{CAPI.PJRT_Error}
    error = Ptr{Error}(unsafe_load(args.error))
    code = error_getcode_obj(error)
    unsafe_store!(args.code, code)
    return ok()
end

function error_for_each_payload(
    args::Ptr{CAPI.PJRT_Error_ForEachPayload_Args}
)::Ptr{CAPI.PJRT_Error}
    error = Ptr{Error}(unsafe_load(args.error))
    visitor = unsafe_load(args.visitor)
    user_arg = unsafe_load(args.user_arg)
    error_for_each_payload_obj(error, visitor, user_arg)
    return ok()
end

# this are required by CAPI.PJRT_Error_FunctionTable
function error_destroy_obj(error::Ptr{Error})::Cvoid
    error != C_NULL && unroot!(error)
    return nothing
end

function error_message_obj(
    error::Ptr{Error}, message::Ptr{Cstring}, message_size::Ptr{Csize_t}
)::Cvoid
    error == C_NULL && return nothing
    err = unsafe_pointer_to_objref(error)::Error
    msg = err.message
    unsafe_store!(message, Cstring(pointer(msg)))
    unsafe_store!(message_size, sizeof(msg))
    return nothing
end

function error_getcode_obj(error::Ptr{Error})::CAPI.PJRT_Error_Code
    error == C_NULL && return CAPI.PJRT_Error_Code_OK
    err = unsafe_pointer_to_objref(error)::Error
    return err.code
end

function error_for_each_payload_obj(
    error::Ptr{Error}, visitor::CAPI.PJRT_Error_PayloadVisitor, user_arg::Ptr{Cvoid}
)::Cvoid
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
    unsafe_store!(args.attributes, C_NULL)
    unsafe_store!(args.num_attributes, 0)
    return ok()
end

## interfaces
"""
    Memory

### Inteface

- `id`
- `kind`
- `debug_string`
- `to_string`
- `addressable_devices`
"""
abstract type Memory end

function checkinterface(T::Type{<:Memory})
    if !ismutabletype(T)
        error("$T <: Memory must be a mutable type to be GC rootable")
    end

    if fieldtype(T, 1) != Ptr{CAPI.PJRT_Memory_FunctionTable}
        error("First field of $T must be a $(CAPI.PJRT_Memory_FunctionTable)")
    end

    return nothing
end

function memory_id(args::Ptr{CAPI.PJRT_Memory_Id_Args})::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    unsafe_store!(args.id, id(memory))
    return ok()
end

function memory_kind(args::Ptr{CAPI.PJRT_Memory_Kind_Args})::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    str = kind(memory)
    unsafe_store!(args.kind, Cstring(pointer(str)))
    unsafe_store!(args.kind_size, length(str))
    return ok()
end

function memory_debug_string(
    args::Ptr{CAPI.PJRT_Memory_DebugString_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    str = debug_string(memory)::String
    unsafe_store!(args.debug_string, Cstring(pointer(str)))
    unsafe_store!(args.debug_string_size, length(str))
    return ok()
end

function memory_to_string(args::Ptr{CAPI.PJRT_Memory_ToString_Args})::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    str = to_string(memory)::String
    unsafe_store!(args.to_string, Cstring(pointer(str)))
    unsafe_store!(args.to_string_size, length(str))
    return ok()
end

function memory_addressable_by_devices(
    args::Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    devs = addressable_devices(memory)
    unsafe_store!(args.devices, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(devs)))
    unsafe_store!(args.num_devices, length(devs))
    return ok()
end

"""
    DeviceDescription

### Interface

- `id`
- `pid`
- `kind`
- `debug_string`
- `to_string`
- `attributes`
"""
abstract type DeviceDescription end

function device_description_id(
    args::Ptr{CAPI.PJRT_DeviceDescription_Id_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    unsafe_store!(args.id, id(desc))
    return ok()
end

function device_description_process_index(
    args::Ptr{CAPI.PJRT_DeviceDescription_ProcessIndex_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    unsafe_store!(args.process_index, pid(desc))
    return ok()
end

# TODO find the interface of `attributes`
function device_description_attributes(
    args::Ptr{CAPI.PJRT_DeviceDescription_Attributes_Args}
)::Ptr{CAPI.PJRT_Error}
    # ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    # desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    # attrs = attributes(desc)
    # unsafe_store!(args.num_attributes, length(attrs))
    # unsafe_store!(args.attributes, C_NULL)
    # return ok()
    return unimplemented()
end

function device_description_kind(
    args::Ptr{CAPI.PJRT_DeviceDescription_Kind_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    str = kind(desc)::String
    unsafe_store!(args.device_kind, Cstring(pointer(str)))
    unsafe_store!(args.device_kind_size, length(str))
    return ok()
end

function device_description_debug_string(
    args::Ptr{CAPI.PJRT_DeviceDescription_DebugString_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    str = debug_string(desc)::String
    unsafe_store!(args.debug_string, Cstring(pointer(desc)))
    unsafe_store!(args.debug_string_size, length(desc))
    return ok()
end

function device_description_to_string(
    args::Ptr{CAPI.PJRT_DeviceDescription_ToString_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    str = to_string(desc)
    unsafe_store!(args.to_string, Cstring(pointer(str)))
    unsafe_store!(args.to_string_size, length(str))
    return ok()
end

"""
    Device

### Interface

- `description`
- `is_addressable`
- `local_hardware_id`
- `addressable_memories`
- `default_memory`
- `memory_stats`
"""
abstract type Device end
function checkinterface(T::Type{<:Device})
    if !ismutabletype(T)
        error("$T <: Device must be a mutable type to be GC rootable")
    end
end

function device_get_description(
    args::Ptr{CAPI.PJRT_Device_GetDescription_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    desc = description(device)::DeviceDescription
    unsafe_store!(
        args.device_description,
        Ptr{CAPI.PJRT_DeviceDescription}(pointer_from_objref(device.description)),
    )
    return ok()
end

function device_is_addressable(
    args::Ptr{CAPI.PJRT_Device_IsAddressable_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    unsafe_store!(args.is_addressable, is_addressable(device)::Bool)
    return ok()
end

function device_local_hardware_id(
    args::Ptr{CAPI.PJRT_Device_LocalHardwareId_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    unsafe_store!(args.local_hardware_id, local_hardware_id(device))
    return ok()
end

function device_addressable_memories(
    args::Ptr{CAPI.PJRT_Device_AddressableMemories_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    mems = addressable_memories(device)::Vector{Memory}
    unsafe_store!(args.memories, Ptr{Ptr{CAPI.PJRT_Memory}}(pointer(mems)))
    unsafe_store!(args.num_memories, length(mems))
    return ok()
end

function device_default_memory(
    args::Ptr{CAPI.PJRT_Device_DefaultMemory_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    mem = default_memory(device)::Memory
    unsafe_store!(args.memory, Ptr{CAPI.PJRT_Memory}(pointer_from_objref(mem)))
    return ok()
end

function device_memory_stats(
    args::Ptr{CAPI.PJRT_Device_MemoryStats_Args}
)::Ptr{CAPI.PJRT_Error}
    # allocations are not tracked yet; `pjrt_c_api.h` allows returning `UNIMPLEMENTED` here
    return unimplemented()
end

"""
    Buffer

### Interface

- `destroy`
- `delete!`
- `isdeleted`
- `element_type` -> eltype
- `dimensions` -> size
- `on_device_size_in_bytes` -> sizeof?
- `device`
- `memory`
- `copy_to_device!` -> copyto!?
- `to_host_buffer` -> copyto!?
- `is_on_cpu`
- `unsafe_pointer`
- ...
"""

# TODO abstract interface
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

function buffer_delete(args::Ptr{CAPI.PJRT_Buffer_Delete_Args})::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{Buffer}(unsafe_load(args.buffer))
    Libc.free(unsafe_load(ptr.data))
    unsafe_store!(ptr.data, C_NULL)
    return ok()
end

function buffer_is_deleted(
    args::Ptr{CAPI.PJRT_Buffer_IsDeleted_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    unsafe_store!(args.is_deleted, buffer.data == C_NULL)
    return ok()
end

function buffer_element_type(
    args::Ptr{CAPI.PJRT_Buffer_ElementType_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    buffer.data == C_NULL && return invalid_argument("Buffer has been deleted or donated")
    unsafe_store!(args.type, buffer.type)
    return ok()
end

function buffer_dimensions(
    args::Ptr{CAPI.PJRT_Buffer_Dimensions_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    buffer.data == C_NULL && return invalid_argument("Buffer has been deleted or donated")
    unsafe_store!(args.dims, buffer.dims)
    unsafe_store!(args.num_dims, buffer.num_dims)
    return ok()
end

function buffer_on_device_size_in_bytes(
    args::Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    buffer.data == C_NULL && return invalid_argument("Buffer has been deleted or donated")
    unsafe_store!(args.on_device_size_in_bytes, buffer.size_in_bytes)
    return ok()
end

function buffer_device(args::Ptr{CAPI.PJRT_Buffer_Device_Args})::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    buffer.data == C_NULL && return invalid_argument("Buffer has been deleted or donated")
    unsafe_store!(args.device, Ptr{CAPI.PJRT_Device}(buffer.device))
    return ok()
end

function buffer_memory(args::Ptr{CAPI.PJRT_Buffer_Memory_Args})::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    buffer.data == C_NULL && return invalid_argument("Buffer has been deleted or donated")
    # buffers are always allocated on the default memory of their device
    device = unsafe_pointer_to_objref(buffer.device)::Device
    memory = first(device.memories)
    unsafe_store!(args.memory, Ptr{CAPI.PJRT_Memory}(pointer_from_objref(memory)))
    return ok()
end

function buffer_copy_to_device(
    args::Ptr{CAPI.PJRT_Buffer_CopyToDevice_Args}
)::Ptr{CAPI.PJRT_Error}
    src = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    src.data == C_NULL && return invalid_argument("Buffer has been deleted or donated")
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

function buffer_to_host_buffer(
    args::Ptr{CAPI.PJRT_Buffer_ToHostBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    src = unsafe_load(Ptr{Buffer}(unsafe_load(args.src)))
    src.data == C_NULL && return invalid_argument("Buffer has been deleted or donated")

    # relayout is not supported, so `host_layout` must match the dense layout of `src`
    host_layout = unsafe_load(args.host_layout)
    if host_layout != C_NULL
        type = unsafe_load(host_layout.type)
        type != CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled && return unimplemented()

        tiled = unsafe_load(host_layout.tiled)
        tiled.num_tiles != 0 && return unimplemented()
        tiled.minor_to_major_size != src.num_dims && return invalid_argument()
        for i in 1:(src.num_dims)
            m2m = unsafe_load(tiled.minor_to_major, i)
            m2m != unsafe_load(src.minor_to_major, i) && return unimplemented()
        end
    end

    # `dst == C_NULL` queries the required size
    dst = unsafe_load(args.dst)
    if dst == C_NULL
        unsafe_store!(args.dst_size, src.size_in_bytes)
        return ok()
    end
    unsafe_load(args.dst_size) < src.size_in_bytes && return invalid_argument()

    # the copy is synchronous
    unsafe_copyto!(Ptr{UInt8}(dst), Ptr{UInt8}(src.data), src.size_in_bytes)

    # TODO return an already-ready `PJRT_Event` once events are implemented
    unsafe_store!(args.event, C_NULL)
    return ok()
end

function buffer_is_on_cpu(args::Ptr{CAPI.PJRT_Buffer_IsOnCpu_Args})::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    buffer.data == C_NULL && return invalid_argument("Buffer has been deleted or donated")
    unsafe_store!(args.is_on_cpu, true)
    return ok()
end

function buffer_unsafe_pointer(
    args::Ptr{CAPI.PJRT_Buffer_UnsafePointer_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_load(Ptr{Buffer}(unsafe_load(args.buffer)))
    buffer.data == C_NULL && return invalid_argument("Buffer has been deleted or donated")
    unsafe_store!(args.buffer_pointer, UInt(buffer.data))
    return ok()
end

"""
    Client

## Interface

- `create`
- `destroy!`
- `platform_name`
- `platform_version`
- `process_index` -> pid?
- `devices`
- `addressable_devices`
- `addressable_memories`
- `compile!`
- `buffer_from_host_buffer`
"""
abstract type Client end

function client_create(args::Ptr{CAPI.PJRT_Client_Create})
    # TODO
end

function client_destroy(args::Ptr{CAPI.PJRT_Client_Destroy})
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    unroot!(client)
    return ok()
end

function client_platform_name(args::Ptr{CAPI.PJRT_Client_PlatformName})
    client = unsafe_pointer_to_objref(unsafe_load(Ptr{Client}(args.client)))::Client
    str = platform_name(client)::String
    unsafe_store!(args.platform_name, Cstring(pointer(str)))
    unsafe_store!(args.platform_name_size, length(str))
    return ok()
end

function client_platform_version(args::Ptr{CAPI.PJRT_Client_PlatformVersion})
    client = unsafe_pointer_to_objref(unsafe_load(Ptr{Client}(args.client)))::Client
    str = platform_version(client)::String
    unsafe_store!(args.platform_version, Cstring(pointer(str)))
    unsafe_store!(args.platform_version_size, length(str))
    return ok()
end

function client_process_index(args::Ptr{CAPI.PJRT_Client_ProcessIndex})
    client = unsafe_pointer_to_objref(unsafe_load(Ptr{Client}(args.client)))::Client
    pid = process_index(client)
    unsafe_store!(args.process_index, Cint(0))
    return ok()
end

function client_devices(args::Ptr{CAPI.PJRT_Client_Devices})
    client = unsafe_pointer_to_objref(unsafe_load(Ptr{Client}(args.client)))::Client
    devs = devices(client)::Vector{Device}
    unsafe_store!(args.devices, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(devs)))
    unsafe_store!(args.num_devices, length(devs))
    return ok()
end

function client_addressable_devices(args::Ptr{CAPI.PJRT_Client_AddressableDevices})
    client = unsafe_pointer_to_objref(unsafe_load(Ptr{Client}(args.client)))::Client
    devs = addressable_devices(client)::Vector{Device}
    unsafe_store!(args.addressable_devices, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(devs)))
    unsafe_store!(args.num_devices, length(devs))
    return ok()
end

function client_addressable_memories(args::Ptr{CAPI.PJRT_Client_AddressableMemories})
    client = unsafe_pointer_to_objref(unsafe_load(Ptr{Client}(args.client)))::Client
    mems = addressable_memories(client)::Vector{Memory}
    unsafe_store!(args.addressable_memories, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(mems)))
    unsafe_store!(args.num_memories, length(mems))
    return ok()
end

function client_compile(args::Ptr{CAPI.PJRT_Client_Compile})
    client = unsafe_pointer_to_objref(unsafe_load(Ptr{Client}(args.client)))::Client
    return unimplemented()
end

function client_buffer_from_host_buffer(args::Ptr{CAPI.PJRT_Client_BufferFromHostBuffer})::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(Ptr{Client}(args.client)))::Client

    _eltype = unsafe_load(args.type)

    _ndims = unsafe_load(args.num_dims)
    _dims_ptr = unsafe_load(args.dims)
    _size = unsafe_wrap(Vector{Cint}, _dims_ptr, _ndims)

    _nbytestrides = unsafe_load(args.num_byte_strides)
    _device_layout = unsafe_load(args.device_layout)
    _host_buffer_semantics = unsafe_load(args.host_buffer_semantics)

    mem_ptr = Ptr{Memory}(unsafe_load(args.memory))
    dev_ptr = Ptr{Device}(unsafe_load(args.device))

    # TODO error?
    buffer = if mem_ptr != C_NULL
        mem = unsafe_pointer_to_objref(mem_ptr)::Memory
        buffer_from_host_buffer(client, mem, _eltype, _size, _nbytestrides, _device_layout, _host_buffer_semantics)::Buffer
    elseif dev_ptr != C_NULL
        dev = unsafe_pointer_to_objref(dev_ptr)::Device
        buffer_from_host_buffer(client, dev, _eltype, _size, _nbytestrides, _device_layout, _host_buffer_semantics)::Buffer
    else
        return invalid_argument("No devive nor memory has been assigned")
    end

    # TODO return an already-ready `PJRT_Event` once events are implemented
    unsafe_store!(args.done_with_host_buffer, C_NULL)

    root!(buffer)
    unsafe_store!(args.buffer, Ptr{CAPI.PJRT_Buffer}(buffer))

    return ok()
end

## HostMemory
mutable struct HostMemory <: Memory
    const vtable::Ptr{CAPI.PJRT_Memory_FunctionTable}
    const id::Cint
    const kind::String
    const debug_string::String
    const to_string::String
    const devices::Vector{Device}

    function HostMemory(id, kind, devices)
        return new(
            unsafe_convert(Ptr{CAPI.PJRT_Memory_FunctionTable}, vtable_memory),
            id,
            kind,
            "Memory(id=$id, kind=$kind, platform=julia)",
            "Memory managed by Julia",
            devices,
        )
    end
end

id(mem::HostMemory) = mem.id
kind(mem::HostMemory) = mem.kind
debug_string(mem::HostMemory) = mem.debug_string
to_string(mem::HostMemory) = mem.to_string
addressable_devices(mem::HostMemory) = mem.devices

## DeviceDescription
mutable struct HostDeviceDescription
    const id::Cint
    const pid::Cint
    const debug_string::String
    const to_string::String
end

function HostDeviceDescription()
    id = 0
    pid = 0
    return HostDeviceDescription(id, pid, "julia: id=$id", "JuliaDevice(id=$id)")
end

id(desc::HostDeviceDescription) = desc.id
pid(desc::HostDeviceDescription) = desc.pid
# kind(desc::HostDeviceDescription) = desc.kind
debug_string(desc::HostDeviceDescription) = desc.debug_string
to_string(desc::HostDeviceDescription) = desc.to_string
# attributes(desc::HostDeviceDescription) = desc.

## Device
mutable struct HostDevice <: Device
    const description::DeviceDescription # TODO or HostDeviceDescription
    const local_hardware_id::Cint
    const memories::Vector{Memory}
end

description(dev::HostDevice) = dev.description
is_addressable(dev::HostDevice) = true
local_hardware_id(dev::HostDevice) = dev.local_hardware_id
addressable_memories(dev::HostDevice) = dev.memories
default_memory(dev::HostDevice) = dev.memories[1]
# memory_stats(dev::HostDevice) = 

## HostBuffer
struct HostBuffer
    data::Ptr{Cvoid}
    size_in_bytes::Csize_t
    type::CAPI.PJRT_Buffer_Type
    dims::Vector{Int64}
    minor_to_major::Vector{Int64}
    device::Device
end

## HostClient
mutable struct HostClient
    devices::Vector{Device}
    memories::Vector{Memory}
end

function HostClient(args::Ptr{CAPI.PJRT_Client_Create_Args})::Ptr{CAPI.PJRT_Error}
    devs = Vector{Device}(undef, 1)
    mems = Vector{Device}(undef, 1)

    dev = HostDevice(DeviceDescription(), -1, mems)
    mem = HostMemory(0, "device", devs)

    devs[1] = dev
    mems[1] = mem

    client = HostClient(devs, mems)
    unsafe_store!(args.client, Ptr{CAPI.PJRT_Client}(client))

    root!(client)

    return ok()
end

platform_name(::HostClient) = "Julia client"
platform_version(::HostClient) = string(VERSION)
process_index(::HostClient) = 0
devices(client::HostClient) = client.devices
addressable_devices(client::HostClient) = devices(client)
addressable_memories(client::HostClient) = client.memories

function buffer_from_host_buffer(client::HostClient, mem::HostMemory, _eltype, _size, _nbytestrides, _layout, _host_buffer_semantics)
    _elsize = pjrt_elsize(_eltype)
    _ndims = length(_size)

    size_in_bytes = prod(_size; init=_elsize)

    # only dense major-to-minor host data is supported; strides of size-1 dims are irrelevant
    num_byte_strides = Int(unsafe_load(args.num_byte_strides))
    if num_byte_strides != 0
        num_byte_strides != num_dims && return invalid_argument()
        byte_strides = unsafe_load(args.byte_strides)
        stride = elsize
        for i in num_dims:-1:1
            dim = unsafe_load(host_dims, i)
            dim != 1 && unsafe_load(byte_strides, i) != stride && return unimplemented()
            stride *= dim
        end
    end

    # relayout is not supported, so `device_layout` must be dense major-to-minor
    # if device_layout != C_NULL
    #     layout_type = unsafe_load(device_layout.type)
    #     layout_type != CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled && return unimplemented()

    #     tiled = unsafe_load(device_layout.tiled)
    #     tiled.num_tiles != 0 && return unimplemented()
    #     tiled.minor_to_major_size != num_dims && return invalid_argument()
    #     for i in 1:num_dims
    #         unsafe_load(tiled.minor_to_major, i) != num_dims - i && return unimplemented()
    #     end
    # end

    # the copy is synchronous, so all `host_buffer_semantics` are honored
    data = Libc.malloc(size_in_bytes)
    data == C_NULL && return resource_exhausted()

    minor_to_major = collect(_ndims:-1:1)

    # TODO device
    buffer = Buffer(data, size_in_bytes, _eltype, _size, minor_to_major, device)
    root!(buffer)
    return buffer
end

function buffer_from_host_buffer(client::HostClient, mem::HostDevice, _eltype, _size, _nbytestrides, _device_layout, _host_buffer_semantics)
    # TODO error
end

function __init__()
    # generic handler for unimplemented PJRT API functions
    c_unimplemented = @cfunction(unimplemented, Ptr{CAPI.PJRT_Error}, ())

    #! format: off
    vtable_error[] = CAPI.PJRT_Error_FunctionTable(
        #= struct_size =# csizeof(CAPI.PJRT_Error_FunctionTable),
        #= instance_size =# csizeof(CAPI.PJRT_Error),
        #= extension_start =# C_NULL,
        #= destroy =# @cfunction(error_destroy_obj, Cvoid, (Ptr{Error},)),
        #= message =# @cfunction(error_message_obj, Cvoid, (Ptr{Error}, Ptr{Cstring}, Ptr{Csize_t})),
        #= get_code =# @cfunction(error_getcode_obj, CAPI.PJRT_Error_Code, (Ptr{Error},)),
        #= for_each_payload =# @cfunction(error_for_each_payload_obj, Cvoid, (Ptr{Error}, CAPI.PJRT_Error_PayloadVisitor, Ptr{Cvoid})),
    )

    vtable_memory[] = CAPI.PJRT_Memory_FunctionTable(
        #= struct_size =# csizeof(CAPI.PJRT_Memory_FunctionTable),
        #= extension_start =# C_NULL,
        #= instance_struct_size =# csizeof(CAPI.PJRT_Memory),
        #= get_user_data =# C_NULL,
        #= set_user_data =# C_NULL,
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
        #= PJRT_Error_Destroy =# @cfunction(error_destroy, Cvoid, (Ptr{CAPI.PJRT_Error_Destroy_Args},)),
        #= PJRT_Error_Message =# @cfunction(error_message, Cvoid, (Ptr{CAPI.PJRT_Error_Message_Args},)),
        #= PJRT_Error_GetCode =# @cfunction(error_getcode, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Error_GetCode_Args},)),
        #= PJRT_Plugin_Initialize =# @cfunction(plugin_initialize, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Plugin_Initialize_Args},)),
        #= PJRT_Plugin_Attributes =# @cfunction(plugin_attributes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Plugin_Attributes_Args},)),
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
        #= PJRT_Client_AddressableMemories =# @cfunction(client_addressable_memories, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_AddressableMemories_Args},)),
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
        #= PJRT_Buffer_GetMemoryLayout =# c_unimplemented,
        #= PJRT_Buffer_OnDeviceSizeInBytes =# @cfunction(buffer_on_device_size_in_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args},)),
        #= PJRT_Buffer_Device =# @cfunction(buffer_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Device_Args},)),
        #= PJRT_Buffer_Memory =# @cfunction(buffer_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Memory_Args},)),
        #= PJRT_Buffer_Delete =# @cfunction(buffer_delete, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Delete_Args},)),
        #= PJRT_Buffer_IsDeleted =# @cfunction(buffer_is_deleted, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IsDeleted_Args},)),
        #= PJRT_Buffer_CopyToDevice =# @cfunction(buffer_copy_to_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyToDevice_Args},)),
        #= PJRT_Buffer_ToHostBuffer =# @cfunction(buffer_to_host_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ToHostBuffer_Args},)),
        #= PJRT_Buffer_IsOnCpu =# @cfunction(buffer_is_on_cpu, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IsOnCpu_Args},)),
        #= PJRT_Buffer_ReadyEvent =# c_unimplemented,
        #= PJRT_Buffer_UnsafePointer =# @cfunction(buffer_unsafe_pointer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_UnsafePointer_Args},)),
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
        #= PJRT_Error_ForEachPayload =# @cfunction(error_for_each_payload, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Error_ForEachPayload_Args},)),
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
