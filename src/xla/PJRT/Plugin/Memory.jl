"""
    Memory

### Interface

- `id`
- `kind`
- `kind_id`
- `debug_string` (defaults to `Base.show`)
- `Base.print`
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
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
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
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    kid = kind_id(memory)::Integer
    unsafe_store!(args.kind_id, kid)
    return ok()
end

"""
    debug_string(mem::Memory) -> String

Return a verbose description of `mem` for debugging. Defaults to `Base.show(io, mem)`, i.e. `sprint(show, mem)`.

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
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
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
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
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

The returned value doesn't need to be owned by `mem`, but it must be the same on every call: the C wrapper calls it at
most once per `mem` and keeps a copy for as long as `mem` is alive, since C keeps pointers to it. The objects in it must
be the same ones on every call too, and the same ones that other functions return, since C identifies them by pointer.

Backs `PJRT_Memory_AddressableByDevices`.
"""
addressable_devices(::Memory)

# C borrows the vector, so it is kept for as long as the memory is alive
# (it is a `Vector{Device}`, but `Device` is defined after `Memory`)
const memory_addressable_devices_cache = WeakKeyIdDict{Memory,Vector}()

@pjrt_catch_return function pjrt_memory_addressable_by_devices(
    args::Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    devs = get!(memory_addressable_devices_cache, memory) do
        collect(Device, addressable_devices(memory))
    end
    unsafe_store!(args.devices, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(devs)))
    unsafe_store!(args.num_devices, length(devs))
    return ok()
end
