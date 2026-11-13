"""
    DeviceDescription

### Interface

- `id`
- `pid`
- `kind`
- `debug_string` (defaults to `Base.show`)
- `Base.print`
- `attributes`
"""
abstract type DeviceDescription end

"""
    id(desc::DeviceDescription) -> Integer

Return the id of the device described by `desc`, unique among the devices of its platform across all processes.

Backs `PJRT_DeviceDescription_Id`.
"""
id(::DeviceDescription)

@pjrt_catch_return function pjrt_device_description_id(
    args::Ptr{CAPI.PJRT_DeviceDescription_Id_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    unsafe_store!(args.id, id(desc))
    return ok()
end

"""
    process_index(desc::DeviceDescription) -> Integer

Return the index of the process that owns the device described by `desc`. Always `0` in single-process settings.

Backs `PJRT_DeviceDescription_ProcessIndex`.
"""
process_index(::DeviceDescription)

@pjrt_catch_return function pjrt_device_description_process_index(
    args::Ptr{CAPI.PJRT_DeviceDescription_ProcessIndex_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    unsafe_store!(args.process_index, process_index(desc))
    return ok()
end

"""
    kind(desc::DeviceDescription) -> String

Return a vendor-dependent string that uniquely identifies the kind of device (e.g. `"Tesla V100-SXM2-16GB"`).

!!! warning
The string is handed to C by pointer, so it must be owned by `desc`.

Backs `PJRT_DeviceDescription_Kind`.
"""
kind(::DeviceDescription)

@pjrt_catch_return function pjrt_device_description_kind(
    args::Ptr{CAPI.PJRT_DeviceDescription_Kind_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    str = kind(desc)::String
    unsafe_store!(args.device_kind, Cstring(pointer(str)))
    unsafe_store!(args.device_kind_size, sizeof(str))
    return ok()
end

# C views of the results of `attributes(::DeviceDescription)`, which C keeps for as long as the description is alive
const device_description_attributes_cache = WeakKeyIdDict{
    DeviceDescription,Tuple{Vector{CAPI.PJRT_NamedValue},Vector{Any}}
}()

"""
    attributes(desc::DeviceDescription)

Return the device-specific attributes of the device described by `desc` as `name => value` pairs (e.g. a `Dict`).
Values must be `AbstractString`, `Bool`, `Integer`, `AbstractVector{<:Integer}` or `AbstractFloat`, which C receives as
`String`, `Bool`, `Int64`, `Vector{Int64}` and `Float32`.

It is called at most once per `desc`: the C wrapper keeps a copy for as long as `desc` is alive, since C keeps pointers
to it.

!!! warning
`PjRtCApiClient` copies the attributes once, when it wraps the device description, and treats any error as fatal.

Backs `PJRT_DeviceDescription_Attributes`.
"""
attributes(::DeviceDescription)

@pjrt_catch_return function pjrt_device_description_attributes(
    args::Ptr{CAPI.PJRT_DeviceDescription_Attributes_Args}
)::Ptr{CAPI.PJRT_Error}
    desc = unsafe_pointer_to_objref(unsafe_load(args.device_description))::DeviceDescription
    values, _ = get!(device_description_attributes_cache, desc) do
        pjrt_named_values(attributes(desc))
    end
    unsafe_store!(args.num_attributes, length(values))
    unsafe_store!(args.attributes, pointer(values))
    return ok()
end

"""
    debug_string(desc::DeviceDescription) -> String

Return a verbose description of the device for debugging. Defaults to `Base.show(io, desc)`, i.e.
`sprint(show, desc)`.

It is called at most once per `desc`: the C wrapper keeps the string for as long as `desc` is alive, since C keeps
pointers to it.

Backs `PJRT_DeviceDescription_DebugString`.
"""
debug_string(desc::DeviceDescription) = sprint(show, desc)

# C borrows the strings, so they are kept for as long as the description is alive
const device_description_debug_string_cache = WeakKeyIdDict{DeviceDescription,String}()

@pjrt_catch_return function pjrt_device_description_debug_string(
    args::Ptr{CAPI.PJRT_DeviceDescription_DebugString_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    str = get!(device_description_debug_string_cache, desc) do
        debug_string(desc)::String
    end
    unsafe_store!(args.debug_string, Cstring(pointer(str)))
    unsafe_store!(args.debug_string_size, sizeof(str))
    return ok()
end

"""
    Base.print(io::IO, desc::DeviceDescription)

Write a short, user-facing description of the device. If not implemented, Base falls back to `Base.show(io, desc)`.

It is called at most once per `desc`: the C wrapper keeps the string for as long as `desc` is alive, since C keeps
pointers to it.

Backs `PJRT_DeviceDescription_ToString`.
"""
Base.print(::IO, ::DeviceDescription)

# C borrows the strings, so they are kept for as long as the description is alive
const device_description_to_string_cache = WeakKeyIdDict{DeviceDescription,String}()

@pjrt_catch_return function pjrt_device_description_to_string(
    args::Ptr{CAPI.PJRT_DeviceDescription_ToString_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    str = get!(device_description_to_string_cache, desc) do
        sprint(print, desc)
    end
    unsafe_store!(args.to_string, Cstring(pointer(str)))
    unsafe_store!(args.to_string_size, sizeof(str))
    return ok()
end
