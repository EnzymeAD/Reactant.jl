"""
    MemoryDescription

The description of a kind of memory space that a device supports, independently of the memory spaces that a client
has. XLA uses them to place data in non-default memories when compiling ahead of time (e.g. with `compile`).

### Interface

- [`kind`](@ref kind(::MemoryDescription))
- [`kind_id`](@ref kind_id(::MemoryDescription))

Additionally, [`DeviceDescription`](@ref) must implement:
- [`memory_descriptions`](@ref)
- [`default_memory_description`](@ref) (defaults to `nothing`)
"""
abstract type MemoryDescription end

"""
    kind(md::MemoryDescription) -> String

Return a platform-dependent string that identifies the kind of memory described by `md` (e.g. `"device"`,
`"pinned_host"`).

The returned value doesn't need to be owned by `md`, but it must be the same on every call: the C wrapper calls it at
most once per `md` and keeps a copy for as long as `md` is alive, since C keeps pointers to it.

Backs `PJRT_MemoryDescription_Kind`.
"""
kind(::MemoryDescription)

"""
    kind_id(md::MemoryDescription) -> Integer

Return the numeric id of the kind of memory described by `md`, unique among the kinds of memory of the platform.

Backs `PJRT_MemoryDescription_Kind`.
"""
kind_id(::MemoryDescription)

# C borrows the string, so it is kept for as long as the description is alive
const memory_description_kind_cache = WeakKeyIdDict{MemoryDescription,String}()

@pjrt_catch_return function pjrt_memory_description_kind(
    args::Ptr{CAPI.PJRT_MemoryDescription_Kind_Args}
)::Ptr{CAPI.PJRT_Error}
    md = unsafe_pointer_to_objref(unsafe_load(args.memory_description))::MemoryDescription
    str = get!(memory_description_kind_cache, md) do
        String(kind(md))
    end
    unsafe_store!(args.kind, Cstring(pointer(str)))
    unsafe_store!(args.kind_size, sizeof(str))
    unsafe_store!(args.kind_id, kind_id(md))
    return ok()
end

"""
    memory_descriptions(desc::DeviceDescription) -> Vector{MemoryDescription}

Return the descriptions of all the kinds of memory that the device described by `desc` supports, in any order.

The returned value doesn't need to be owned by `desc`, but it must be the same on every call: the C wrapper calls it at
most once per `desc` and keeps a copy for as long as `desc` is alive, since C keeps pointers to it. The objects in it
must be the same ones on every call too, and the same ones that other functions return, since C identifies them by
pointer.

Backs `PJRT_DeviceDescription_MemoryDescriptions`.
"""
memory_descriptions(::DeviceDescription)

"""
    default_memory_description(desc::DeviceDescription) -> Union{Nothing,MemoryDescription}

Return the element of `memory_descriptions(desc)` that describes the default memory of the device described by
`desc`, or `nothing` if it has none. Defaults to `nothing`.

Backs `PJRT_DeviceDescription_MemoryDescriptions`.
"""
default_memory_description(::DeviceDescription) = nothing

# C borrows the vector, so it is kept for as long as the description is alive
const memory_descriptions_cache = WeakKeyIdDict{
    DeviceDescription,Vector{MemoryDescription}
}()

@pjrt_catch_return function pjrt_device_description_memory_descriptions(
    args::Ptr{CAPI.PJRT_DeviceDescription_MemoryDescriptions_Args}
)::Ptr{CAPI.PJRT_Error}
    desc = unsafe_pointer_to_objref(unsafe_load(args.device_description))::DeviceDescription
    mds = get!(memory_descriptions_cache, desc) do
        collect(MemoryDescription, memory_descriptions(desc))
    end
    default = default_memory_description(desc)::Union{Nothing,MemoryDescription}

    # C's `(size_t)-1` means that there is no default memory
    default_index = if isnothing(default)
        typemax(Csize_t)
    else
        i = findfirst(md -> md === default, mds)
        if isnothing(i)
            msg = "`default_memory_description` returned a description that is not in `memory_descriptions`"
            throw(Error(CAPI.PJRT_Error_Code_INTERNAL, msg))
        end
        i - 1
    end

    unsafe_store!(
        args.memory_descriptions, Ptr{Ptr{CAPI.PJRT_MemoryDescription}}(pointer(mds))
    )
    unsafe_store!(args.num_memory_descriptions, length(mds))
    unsafe_store!(args.default_memory_index, default_index)
    return ok()
end

"""
    MemoryDescriptionsExtension

Advertises `PJRT_MemoryDescriptions_Extension`, through which XLA gets the kinds of memory that a device supports
without a client (e.g. to compile ahead of time). Pass it to [`make_api`](@ref).

The device descriptions of the plugin must implement `memory_descriptions` (and optionally
`default_memory_description`), and its [`MemoryDescription`](@ref)s `kind` and `kind_id`.
"""
mutable struct MemoryDescriptionsExtension <: Extension
    const ftable::CAPI.PJRT_MemoryDescriptions_Extension

    function MemoryDescriptionsExtension(
        ::AbstractPlugin, next::Ptr{CAPI.PJRT_Extension_Base}=C_NULL
    )
        base = CAPI.PJRT_Extension_Base(
            csizeof(CAPI.PJRT_MemoryDescriptions_Extension),
            extension_type(MemoryDescriptionsExtension),
            next,
        )
        #! format: off
        return new(CAPI.PJRT_MemoryDescriptions_Extension(
            base,
            #= PJRT_DeviceDescription_MemoryDescriptions =# @cfunction(pjrt_device_description_memory_descriptions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_MemoryDescriptions_Args},)),
            #= PJRT_MemoryDescription_Kind =# @cfunction(pjrt_memory_description_kind, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_MemoryDescription_Kind_Args},)),
        ))
        #! format: on
    end
end

function extension_type(::Type{MemoryDescriptionsExtension})
    return CAPI.PJRT_Extension_Type_MemoryDescriptions
end
