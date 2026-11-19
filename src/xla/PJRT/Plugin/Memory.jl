"""
    Memory

Represents a memory space of a [`Client`](@ref).

!!! info
Subtypes must be mutable, since the C wrapper keeps pointers to them, but they have no layout requirements: C never
sees them directly, only the shim that the C wrapper creates for each of them.

### Interface

- [`id`](@ref id(::Memory))
- [`kind`](@ref kind(::Memory))
- [`kind_id`](@ref kind_id(::Memory))
- [`debug_string`](@ref debug_string(::Memory))
- [`Base.print`](@ref Base.print(::IO, ::Memory))
- [`addressable_devices`](@ref addressable_devices(::Memory))
"""
abstract type Memory end

# Wraps a `Memory` into the layout of `PJRT_Memory*`.
mutable struct MemoryShim
    const vtable::Ptr{CAPI.PJRT_Memory_FunctionTable}
    # a pointer instead of a reference, since a reference would make the shim keep its memory alive forever
    const memory::Ptr{Memory}

    function MemoryShim(mem::Memory)
        vtable = unsafe_convert(Ptr{CAPI.PJRT_Memory_FunctionTable}, vtable_memory)
        return new(vtable, Ptr{Memory}(pointer_from_objref(mem)))
    end
end

# The values of a `WeakKeyIdDict` are strong references, so it keeps each shim alive for as long as its memory is,
# and drops it after the `Memory` is collected
const memory_shims = WeakKeyIdDict{Memory,MemoryShim}()

# Julia -> C: the pointer that C sees for `mem`, which is the same one on every call
function Base.unsafe_convert(::Type{Ptr{CAPI.PJRT_Memory}}, mem::Memory)
    shim = get!(() -> MemoryShim(mem), memory_shims, mem)
    return Ptr{CAPI.PJRT_Memory}(pointer_from_objref(shim))
end

# C -> Julia: the `Memory` behind a pointer returned by `unsafe_convert`
function unsafe_memory(ptr::Ptr{CAPI.PJRT_Memory})
    shim = unsafe_pointer_to_objref(ptr)::MemoryShim
    return unsafe_pointer_to_objref(shim.memory)::Memory
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
    memory = unsafe_memory(unsafe_load(args.memory))
    unsafe_store!(args.id, id(memory))
    return ok()
end

"""
    kind(mem::Memory) -> String

Return the kind of `mem` (e.g. `"device"`, `"pinned_host"`).

!!! info
The returned value doesn't need to be owned by `mem`, but it must be the same on every call: the C wrapper calls it at
most once per `mem` and keeps a copy for as long as `mem` is alive, since C keeps pointers to it.

Backs `PJRT_Memory_Kind`.
"""
kind(::Memory)

# C borrows the string, so it is kept for as long as the memory is alive
const memory_kind_cache = WeakKeyIdDict{Memory,String}()

@pjrt_catch_return function pjrt_memory_kind(
    args::Ptr{CAPI.PJRT_Memory_Kind_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_memory(unsafe_load(args.memory))
    str = get!(memory_kind_cache, memory) do
        String(kind(memory))
    end
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
    memory = unsafe_memory(unsafe_load(args.memory))
    kid = kind_id(memory)::Integer
    unsafe_store!(args.kind_id, kid)
    return ok()
end

"""
    debug_string(mem::Memory) -> String

Return a verbose description of `mem` for debugging. Defaults to `Base.show(io, mem)`, i.e. `sprint(show, mem)`.

!!! info
The returned value doesn't need to be owned by `mem`, but it must be the same on every call: the C wrapper calls it at
most once per `mem` and keeps a copy for as long as `mem` is alive, since C keeps pointers to it.

Backs `PJRT_Memory_DebugString`.
"""
debug_string(mem::Memory) = sprint(show, mem)

# C borrows the strings, so they are kept for as long as the memory is alive
const memory_debug_string_cache = WeakKeyIdDict{Memory,String}()

@pjrt_catch_return function pjrt_memory_debug_string(
    args::Ptr{CAPI.PJRT_Memory_DebugString_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_memory(unsafe_load(args.memory))
    str = get!(memory_debug_string_cache, memory) do
        debug_string(memory)::String
    end
    unsafe_store!(args.debug_string, Cstring(pointer(str)))
    unsafe_store!(args.debug_string_size, sizeof(str))
    return ok()
end

"""
    Base.print(io::IO, mem::Memory)

Write a short, user-facing description of `mem`. If not implemented, Base falls back to `Base.show(io, mem)`.

The text doesn't need to be owned by `mem`, but it must be the same on every call: the C wrapper calls it at
most once per `mem` and keeps a copy for as long as `mem` is alive, since C keeps pointers to it.

Backs `PJRT_Memory_ToString`.
"""
Base.print(::IO, ::Memory)

# C borrows the strings, so they are kept for as long as the memory is alive
const memory_to_string_cache = WeakKeyIdDict{Memory,String}()

@pjrt_catch_return function pjrt_memory_to_string(
    args::Ptr{CAPI.PJRT_Memory_ToString_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_memory(unsafe_load(args.memory))
    str = get!(memory_to_string_cache, memory) do
        sprint(print, memory)
    end
    unsafe_store!(args.to_string, Cstring(pointer(str)))
    unsafe_store!(args.to_string_size, sizeof(str))
    return ok()
end

"""
    addressable_devices(mem::Memory) -> Vector{Device}

Return the devices that can address `mem`.

!!! info
The returned value doesn't need to be owned by `mem`: the C wrapper calls it at most once per `mem` and keeps a copy of
the pointers to its objects for as long as `mem` is alive, since C borrows that copy. The copy doesn't keep the objects
alive, so they must be kept alive elsewhere (e.g. by the client) for as long as `mem` is alive. They must also be the
same ones on every call, and the same ones that other functions return, since C identifies them by pointer.

Backs `PJRT_Memory_AddressableByDevices`.
"""
addressable_devices(::Memory)

# C borrows the vector, so it is kept for as long as the memory is alive
# (it stores pointers instead of objects so that it doesn't leak the memory, see the notes in Plugin.jl)
const memory_addressable_devices_cache = WeakKeyIdDict{
    Memory,Vector{Ptr{CAPI.PJRT_Device}}
}()

@pjrt_catch_return function pjrt_memory_addressable_by_devices(
    args::Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_memory(unsafe_load(args.memory))
    devs = get!(memory_addressable_devices_cache, memory) do
        Ptr{CAPI.PJRT_Device}[
            pointer_from_objref(dev::Device) for dev in addressable_devices(memory)
        ]
    end
    unsafe_store!(args.devices, pointer(devs))
    unsafe_store!(args.num_devices, length(devs))
    return ok()
end
