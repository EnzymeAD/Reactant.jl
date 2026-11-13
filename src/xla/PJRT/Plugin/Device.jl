"""
    Device

### Interface

- `description`
- `isaddressable`
- `local_hardware_id`
- `addressable_memories`
- `default_memory`
- `memory_stats`
- `clear_memory_stats!`
- `async_tracking_event` (defaults to `nothing`, i.e. not supported)
- `poison_execution!`
- `attributes` (defaults to `()`, i.e. no attributes)
"""
abstract type Device end

"""
    description(dev::Device) -> DeviceDescription

Return the description of `dev`. The description must be kept alive by `dev`.

Backs `PJRT_Device_GetDescription`.
"""
description(::Device)

@pjrt_catch_return function pjrt_device_get_description(
    args::Ptr{CAPI.PJRT_Device_GetDescription_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    desc = description(device)::DeviceDescription
    unsafe_store!(
        args.device_description, Ptr{CAPI.PJRT_DeviceDescription}(pointer_from_objref(desc))
    )
    return ok()
end

"""
    isaddressable(dev::Device) -> Bool

Return whether `dev` can be addressed by the client of this process.

Backs `PJRT_Device_IsAddressable`.
"""
isaddressable(::Device)

@pjrt_catch_return function pjrt_device_is_addressable(
    args::Ptr{CAPI.PJRT_Device_IsAddressable_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    unsafe_store!(args.is_addressable, isaddressable(device)::Bool)
    return ok()
end

"""
    local_hardware_id(dev::Device) -> Integer

Return an opaque hardware id of `dev` (e.g. the CUDA device number).
Use the vendor's handle for the physical device.
Not guaranteed to be dense (i.e. `∈(0:n-1)`), and `-1` if undefined.

Backs `PJRT_Device_LocalHardwareId`.
"""
local_hardware_id(::Device)

@pjrt_catch_return function pjrt_device_local_hardware_id(
    args::Ptr{CAPI.PJRT_Device_LocalHardwareId_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    unsafe_store!(args.local_hardware_id, local_hardware_id(device))
    return ok()
end

"""
    addressable_memories(dev::Device) -> Vector{Memory}

Return the memories that `dev` can address.

The returned value doesn't need to be owned by `dev`, but it must be the same on every call: the C wrapper calls it at
most once per `dev` and keeps a copy for as long as `dev` is alive, since C keeps pointers to it. The objects in it must
be the same ones on every call too, and the same ones that other functions return, since C identifies them by pointer.

Backs `PJRT_Device_AddressableMemories`.
"""
addressable_memories(::Device)

# C borrows the vector, so it is kept for as long as the device is alive
const device_addressable_memories_cache = WeakKeyIdDict{Device,Vector{Memory}}()

@pjrt_catch_return function pjrt_device_addressable_memories(
    args::Ptr{CAPI.PJRT_Device_AddressableMemories_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    mems = get!(device_addressable_memories_cache, device) do
        collect(Memory, addressable_memories(device))
    end
    unsafe_store!(args.memories, Ptr{Ptr{CAPI.PJRT_Memory}}(pointer(mems)))
    unsafe_store!(args.num_memories, length(mems))
    return ok()
end

"""
    default_memory(dev::Device) -> Memory

Return the memory where buffers are placed on `dev` when no memory is given. Must be one of `addressable_memories(dev)`.

Backs `PJRT_Device_DefaultMemory`.
"""
default_memory(::Device)

@pjrt_catch_return function pjrt_device_default_memory(
    args::Ptr{CAPI.PJRT_Device_DefaultMemory_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    mem = default_memory(device)::Memory
    unsafe_store!(args.memory, Ptr{CAPI.PJRT_Memory}(pointer_from_objref(mem)))
    return ok()
end

# TODO document and add interface function
@pjrt_catch_return function pjrt_device_memory_stats(
    args::Ptr{CAPI.PJRT_Device_MemoryStats_Args}
)::Ptr{CAPI.PJRT_Error}
    throw(unimplemented())
end

"""
    clear_memory_stats!(dev::Device)

Reset the memory statistics of `dev` (e.g. the peaks reported by `PJRT_Device_MemoryStats`).

Backs `PJRT_Device_ClearMemoryStats`.
"""
clear_memory_stats!(::Device)

@pjrt_catch_return function pjrt_device_clear_memory_stats(
    args::Ptr{CAPI.PJRT_Device_ClearMemoryStats_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    clear_memory_stats!(device)
    return ok()
end

"""
    poison_execution!(dev::Device, launch_id::Int32, error::Error) -> Bool

Make the outputs of the earliest unfinished execution on `dev` with launch id `launch_id` fail with `error`.

Return `true` if they have been poisoned, or `false` if there is no such execution (e.g. it already finished).

!!! todo
Error payloads are not supported yet: if C passes any, `PJRT_Device_PoisonExecution` returns `UNIMPLEMENTED`.

Backs `PJRT_Device_PoisonExecution`.
"""
poison_execution!(::Device, ::Int32, ::Error)

@pjrt_catch_return function pjrt_device_poison_execution(
    args::Ptr{CAPI.PJRT_Device_PoisonExecution_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    unsafe_load(args.num_payload) == 0 ||
        throw(unimplemented("error payloads are not supported yet"))
    error = Error(
        unsafe_load(args.error_code),
        unsafe_load(args.error_message),
        unsafe_load(args.error_message_size),
    )
    poisoned = poison_execution!(device, unsafe_load(args.launch_id), error)::Bool
    unsafe_store!(args.poisoned, poisoned)
    return ok()
end

"""
    async_tracking_event(dev::Device, description::String) -> Union{Nothing,AsyncTrackingEvent}

Return a new [`AsyncTrackingEvent`](@ref) (i.e. `AsyncTrackingEvent(description)`) that tracks asynchronous work
depending on the activity of `dev`, or `nothing` if `dev` doesn't support tracking it. Defaults to `nothing`.

The caller GC-roots the returned event.

Backs `PJRT_Device_CreateAsyncTrackingEvent`.
"""
async_tracking_event(::Device, ::String) = nothing

@pjrt_catch_return function pjrt_device_create_async_tracking_event(
    args::Ptr{CAPI.PJRT_Device_CreateAsyncTrackingEvent_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    desc = unsafe_string(
        Ptr{UInt8}(unsafe_load(args.description)), unsafe_load(args.description_size)
    )
    event = async_tracking_event(device, desc)::Union{Nothing,AsyncTrackingEvent}
    if isnothing(event)
        unsafe_store!(args.event, C_NULL)
        return ok()
    end
    root!(event)
    unsafe_store!(args.event, Ptr{CAPI.PJRT_AsyncTrackingEvent}(pointer_from_objref(event)))
    return ok()
end

"""
    attributes(dev::Device)

Return the device-specific attributes of `dev` as `name => value` pairs (e.g. a `Dict`). Values must be
`AbstractString`, `Bool`, `Integer`, `AbstractVector{<:Integer}` or `AbstractFloat`, which C receives as `String`,
`Bool`, `Int64`, `Vector{Int64}` and `Float32`. Defaults to `()`, i.e. no attributes.

C gets its own copy of the attributes on every call and frees it through the deleter, so the returned value doesn't need
to be the same on every call.

Backs `PJRT_Device_GetAttributes`.
"""
attributes(::Device) = ()

@pjrt_catch_return function pjrt_device_get_attributes(
    args::Ptr{CAPI.PJRT_Device_GetAttributes_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    # C owns the named values and the objects they point to until it calls the deleter
    values, roots = pjrt_named_values(attributes(device))
    push!(roots, values)
    root!(roots)
    unsafe_store!(args.attributes, pointer(values))
    unsafe_store!(args.num_attributes, length(values))
    unsafe_store!(
        args.device_attributes, Ptr{CAPI.PJRT_Device_Attributes}(pointer_from_objref(roots))
    )
    unsafe_store!(
        args.attributes_deleter,
        @cfunction(pjrt_serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    )
    return ok()
end
