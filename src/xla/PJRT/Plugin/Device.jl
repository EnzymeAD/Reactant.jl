"""
    Device

### Interface

- [`description`](@ref description(::Device))
- [`isaddressable`](@ref isaddressable(::Device))
- [`local_hardware_id`](@ref local_hardware_id(::Device))
- [`addressable_memories`](@ref addressable_memories(::Device))
- [`default_memory`](@ref default_memory(::Device))
- [`async_tracking_event`](@ref async_tracking_event(::Device, ::String))
- [`poison_execution!`](@ref poison_execution!(::Device, ::Int32, ::Error))
- [`attributes`](@ref attributes(::Device))

#### Memory stats

- [`bytes_in_use`](@ref bytes_in_use(::Device))
- [`peak_bytes_in_use`](@ref peak_bytes_in_use(::Device))
- [`nallocs`](@ref nallocs(::Device))
- [`largest_alloc_size`](@ref largest_alloc_size(::Device))
- [`bytes_limit`](@ref bytes_limit(::Device))
- [`bytes_reserved`](@ref bytes_reserved(::Device))
- [`peak_bytes_reserved`](@ref peak_bytes_reserved(::Device))
- [`bytes_reservable_limit`](@ref bytes_reservable_limit(::Device))
- [`largest_free_block_bytes`](@ref largest_free_block_bytes(::Device))
- [`pool_bytes`](@ref pool_bytes(::Device))
- [`peak_pool_bytes`](@ref peak_pool_bytes(::Device))
- [`peak_allocated_bytes`](@ref peak_allocated_bytes(::Device))
- [`clear_memory_stats!`](@ref clear_memory_stats!(::Device))
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

!!! info
The returned value doesn't need to be owned by `dev`: the C wrapper calls it at most once per `dev` and keeps a copy of
the pointers to its objects for as long as `dev` is alive, since C borrows that copy. The copy doesn't keep the objects
alive, so they must be kept alive elsewhere (e.g. by the client) for as long as `dev` is alive. They must also be the
same ones on every call, and the same ones that other functions return, since C identifies them by pointer.

Backs `PJRT_Device_AddressableMemories`.
"""
addressable_memories(::Device)

# C borrows the vector, so it is kept for as long as the device is alive
# (it stores pointers instead of objects so that it doesn't leak the device, see the notes in Plugin.jl)
const device_addressable_memories_cache = WeakKeyIdDict{
    Device,Vector{Ptr{CAPI.PJRT_Memory}}
}()

@pjrt_catch_return function pjrt_device_addressable_memories(
    args::Ptr{CAPI.PJRT_Device_AddressableMemories_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    mems = get!(device_addressable_memories_cache, device) do
        Ptr{CAPI.PJRT_Memory}[
            unsafe_convert(Ptr{CAPI.PJRT_Memory}, mem::Memory) for
            mem in addressable_memories(device)
        ]
    end
    unsafe_store!(args.memories, pointer(mems))
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
    unsafe_store!(args.memory, unsafe_convert(Ptr{CAPI.PJRT_Memory}, mem))
    return ok()
end

"""
    bytes_in_use(dev::Device) -> Integer

Return the number of bytes in use on `dev`, for diagnostics. Defaults to throwing an `unimplemented` error, which C
accepts as memory stats not being supported: implement it to support them.

Backs `PJRT_Device_MemoryStats`, together with the optional stats (e.g. [`peak_bytes_in_use`](@ref)).
"""
bytes_in_use(::Device) = throw(unimplemented("`bytes_in_use` is not implemented"))

"""
    peak_bytes_in_use(dev::Device) -> Union{Nothing,Integer}

Return the peak number of bytes in use on `dev`, or `nothing` if unknown. Defaults to `nothing`.

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
peak_bytes_in_use(::Device) = nothing

"""
    nallocs(dev::Device) -> Union{Nothing,Integer}

Return the number of allocations on `dev`, or `nothing` if unknown. Defaults to `nothing`.

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
nallocs(::Device) = nothing

"""
    largest_alloc_size(dev::Device) -> Union{Nothing,Integer}

Return the size in bytes of the largest single allocation seen on `dev`, or `nothing` if unknown. Defaults to `nothing`.

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
largest_alloc_size(::Device) = nothing

"""
    bytes_limit(dev::Device) -> Union{Nothing,Integer}

Return the most bytes that the user can allocate on `dev`, or `nothing` if unknown. Defaults to `nothing`.

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
bytes_limit(::Device) = nothing

"""
    bytes_reserved(dev::Device) -> Union{Nothing,Integer}

Return the number of bytes reserved on `dev`, or `nothing` if unknown. Defaults to `nothing`.

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
bytes_reserved(::Device) = nothing

"""
    peak_bytes_reserved(dev::Device) -> Union{Nothing,Integer}

Return the peak number of bytes reserved on `dev`, or `nothing` if unknown. Defaults to `nothing`.

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
peak_bytes_reserved(::Device) = nothing

"""
    bytes_reservable_limit(dev::Device) -> Union{Nothing,Integer}

Return the most bytes that can be reserved on `dev`, or `nothing` if unknown. Defaults to `nothing`.

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
bytes_reservable_limit(::Device) = nothing

"""
    largest_free_block_bytes(dev::Device) -> Union{Nothing,Integer}

Return the size in bytes of the largest free block on `dev`, or `nothing` if unknown. Defaults to `nothing`.

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
largest_free_block_bytes(::Device) = nothing

"""
    pool_bytes(dev::Device) -> Union{Nothing,Integer}

Return the number of bytes held by the allocator of `dev`, which may be more than [`bytes_in_use`](@ref) if it keeps a
pool of memory, or `nothing` if unknown. Defaults to `nothing`.

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
pool_bytes(::Device) = nothing

"""
    peak_pool_bytes(dev::Device) -> Union{Nothing,Integer}

Return the peak number of bytes held by the allocator of `dev` (see [`pool_bytes`](@ref)), or `nothing` if unknown.
Defaults to `nothing`.

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
peak_pool_bytes(::Device) = nothing

"""
    peak_allocated_bytes(dev::Device) -> Union{Nothing,Integer}

Return the peak number of bytes allocated on `dev`, or `nothing` if unknown. Defaults to `nothing`. C doesn't document
how it differs from [`peak_bytes_in_use`](@ref).

Backs `PJRT_Device_MemoryStats`, which only calls it if [`bytes_in_use`](@ref) is implemented.
"""
peak_allocated_bytes(::Device) = nothing

@pjrt_catch_return function pjrt_device_memory_stats(
    args::Ptr{CAPI.PJRT_Device_MemoryStats_Args}
)::Ptr{CAPI.PJRT_Error}
    # like XLA, require every stat but `peak_allocated_bytes`, which was added at the end of the struct later
    struct_size = unsafe_load(args.struct_size)
    if struct_size < UInt(args.peak_pool_bytes_is_set) - UInt(args) + sizeof(Bool)
        msg = "`PJRT_Device_MemoryStats_Args` is too small ($struct_size bytes)"
        throw(invalid_argument(msg))
    end

    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    unsafe_store!(args.bytes_in_use, bytes_in_use(device)::Integer)

    # C doesn't initialize the `_is_set` flags, so they are always written
    value = peak_bytes_in_use(device)::Union{Nothing,Integer}
    unsafe_store!(args.peak_bytes_in_use_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.peak_bytes_in_use, value)

    value = nallocs(device)::Union{Nothing,Integer}
    unsafe_store!(args.num_allocs_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.num_allocs, value)

    value = largest_alloc_size(device)::Union{Nothing,Integer}
    unsafe_store!(args.largest_alloc_size_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.largest_alloc_size, value)

    value = bytes_limit(device)::Union{Nothing,Integer}
    unsafe_store!(args.bytes_limit_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.bytes_limit, value)

    value = bytes_reserved(device)::Union{Nothing,Integer}
    unsafe_store!(args.bytes_reserved_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.bytes_reserved, value)

    value = peak_bytes_reserved(device)::Union{Nothing,Integer}
    unsafe_store!(args.peak_bytes_reserved_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.peak_bytes_reserved, value)

    value = bytes_reservable_limit(device)::Union{Nothing,Integer}
    unsafe_store!(args.bytes_reservable_limit_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.bytes_reservable_limit, value)

    value = largest_free_block_bytes(device)::Union{Nothing,Integer}
    unsafe_store!(args.largest_free_block_bytes_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.largest_free_block_bytes, value)

    value = pool_bytes(device)::Union{Nothing,Integer}
    unsafe_store!(args.pool_bytes_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.pool_bytes, value)

    value = peak_pool_bytes(device)::Union{Nothing,Integer}
    unsafe_store!(args.peak_pool_bytes_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.peak_pool_bytes, value)

    # older callers pass a struct without `peak_allocated_bytes`
    struct_size >= UInt(args.peak_allocated_bytes_is_set) - UInt(args) + sizeof(Bool) ||
        return ok()
    value = peak_allocated_bytes(device)::Union{Nothing,Integer}
    unsafe_store!(args.peak_allocated_bytes_is_set, !isnothing(value))
    isnothing(value) || unsafe_store!(args.peak_allocated_bytes, value)

    return ok()
end

"""
    clear_memory_stats!(dev::Device)

Reset the memory statistics of `dev`.

See also: [`bytes_in_use`](@ref), [`peak_bytes_in_use`](@ref), [`nallocs`](@ref), [`largest_alloc_size`](@ref),
[`bytes_limit`](@ref), [`bytes_reserved`](@ref), [`peak_bytes_reserved`](@ref), [`bytes_reservable_limit`](@ref),
[`largest_free_block_bytes`](@ref), [`pool_bytes`](@ref), [`peak_pool_bytes`](@ref), [`peak_allocated_bytes`](@ref).

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
