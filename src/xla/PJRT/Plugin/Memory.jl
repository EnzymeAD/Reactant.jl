"""
    Memory

### Inteface

- `id`
- `kind`
- `kind_id`
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

"""
    id(mem::Memory) -> Integer

Return the id of `mem`, unique among the memories of its client.

Backs `PJRT_Memory_Id`.
"""
id(::Memory)

@pjrt_catch_return function pjrt_memory_id(
    args::Ptr{CAPI.PJRT_Memory_Id_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    unsafe_store!(args.id, id(memory))
    return ok()
end

"""
    kind(mem::Memory) -> String

Return the kind of `mem` (e.g. `"device"`, `"pinned_host"`).

!!! warning
The string is handed to C by pointer, so it must be owned by `mem`.

Backs `PJRT_Memory_Kind`.
"""
kind(::Memory)

@pjrt_catch_return function pjrt_memory_kind(
    args::Ptr{CAPI.PJRT_Memory_Kind_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    str = kind(memory)
    unsafe_store!(args.kind, Cstring(pointer(str)))
    unsafe_store!(args.kind_size, sizeof(str))
    return ok()
end

"""
    kind_id(mem::Memory) -> Integer

Return a platform-dependent id that uniquely identifies the kind of `mem`.
Memories with the same `kind` must return the same `kind_id`.

Backs `PJRT_Memory_Kind_Id`.
"""
kind_id(::Memory)

@pjrt_catch_return function pjrt_memory_kind_id(
    args::Ptr{CAPI.PJRT_Memory_Kind_Id_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    kid = kind_id(memory)::Integer
    unsafe_store!(args.kind_id, kid)
    return ok()
end

"""
    debug_string(mem::Memory) -> String

Return a verbose description of `mem` for debugging.
    
!!! warning
The string is handed to C by pointer, so it must be owned by `mem`.

Backs `PJRT_Memory_DebugString`.
"""
debug_string(::Memory)

@pjrt_catch_return function pjrt_memory_debug_string(
    args::Ptr{CAPI.PJRT_Memory_DebugString_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    str = debug_string(memory)::String
    unsafe_store!(args.debug_string, Cstring(pointer(str)))
    unsafe_store!(args.debug_string_size, sizeof(str))
    return ok()
end

"""
    to_string(mem::Memory) -> String

Return a short, user-facing description of `mem`.

!!! warning
The string is handed to C by pointer, so it must be owned by `mem`.

Backs `PJRT_Memory_ToString`.
"""
to_string(::Memory)

@pjrt_catch_return function pjrt_memory_to_string(
    args::Ptr{CAPI.PJRT_Memory_ToString_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    str = to_string(memory)::String
    unsafe_store!(args.to_string, Cstring(pointer(str)))
    unsafe_store!(args.to_string_size, sizeof(str))
    return ok()
end

"""
    addressable_devices(mem::Memory) -> Vector{Device}

Return the devices that can address `mem`.

!!! warning
The vector is handed to C by pointer, so it must be owned by `mem` and never be resized.

Backs `PJRT_Memory_AddressableByDevices`.
"""
addressable_devices(::Memory)

@pjrt_catch_return function pjrt_memory_addressable_by_devices(
    args::Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    devs = addressable_devices(memory)
    unsafe_store!(args.devices, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(devs)))
    unsafe_store!(args.num_devices, length(devs))
    return ok()
end
