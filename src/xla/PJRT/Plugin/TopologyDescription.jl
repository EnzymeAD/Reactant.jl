"""
    TopologyDescription

The description of the devices of a platform and how they are connected, used by `compile` to compile programs without
a `Client`. C gets one from a client with `topology_description`, or without a client through
`TopologyDescription(plugin, name, options)` and `TopologyDescription(plugin, bytes)`.

### Interface

- `platform_name`
- `platform_version`
- `device_descriptions`
- `serialize`
- `attributes`
- `fingerprint`
- `canonical_shape`
- `memory_space_kind_ids`

!!! info
The `TopologyDescription(plugin, name, options)` and `TopologyDescription(plugin, bytes)` constructors (see
[`AbstractPlugin`](@ref)) return the plugin's own subtype of `TopologyDescription`, chosen by dispatch on their
`plugin` argument.
"""
abstract type TopologyDescription end

"""
    platform_name(topology::TopologyDescription) -> String

Return a string that identifies the platform of `topology` (e.g. `"cpu"`, `"gpu"`, `"tpu"`).

!!! warning
The string is handed to C by pointer, so it must be owned by `topology`.

Backs `PJRT_TopologyDescription_PlatformName`.
"""
platform_name(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_platform_name(
    args::Ptr{CAPI.PJRT_TopologyDescription_PlatformName_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    str = platform_name(topology)::String
    unsafe_store!(args.platform_name, Cstring(pointer(str)))
    unsafe_store!(args.platform_name_size, sizeof(str))
    return ok()
end

"""
    platform_version(topology::TopologyDescription) -> String

Return a string containing the platform version of `topology`.

!!! warning
The string is handed to C by pointer, so it must be owned by `topology`.

Backs `PJRT_TopologyDescription_PlatformVersion`.
"""
platform_version(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_platform_version(
    args::Ptr{CAPI.PJRT_TopologyDescription_PlatformVersion_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    str = platform_version(topology)::String
    unsafe_store!(args.platform_version, Cstring(pointer(str)))
    unsafe_store!(args.platform_version_size, sizeof(str))
    return ok()
end

"""
    device_descriptions(topology::TopologyDescription) -> Vector{DeviceDescription}

Return the descriptions of all the devices in `topology`, in any order but the same one on every call.

!!! warning
The vector is handed to C by pointer, so it must be owned by `topology` and never be resized.

Backs `PJRT_TopologyDescription_GetDeviceDescriptions`.
"""
device_descriptions(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_get_device_descriptions(
    args::Ptr{CAPI.PJRT_TopologyDescription_GetDeviceDescriptions_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    descs = device_descriptions(topology)::Vector{DeviceDescription}
    unsafe_store!(args.descriptions, Ptr{Ptr{CAPI.PJRT_DeviceDescription}}(pointer(descs)))
    unsafe_store!(args.num_descriptions, length(descs))
    return ok()
end

"""
    serialize(topology::TopologyDescription) -> Vector{UInt8}

Return a serialization of `topology` that `TopologyDescription(plugin, bytes)` can read back, also used in cache keys.

The returned vector is GC rooted until C calls its deleter, so it must be a new vector that `topology` doesn't hand out
anywhere else.

Backs `PJRT_TopologyDescription_Serialize`.
"""
serialize(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_serialize(
    args::Ptr{CAPI.PJRT_TopologyDescription_Serialize_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    bytes = root!(serialize(topology)::Vector{UInt8})
    unsafe_store!(args.serialized_bytes, Cstring(pointer(bytes)))
    unsafe_store!(args.serialized_bytes_size, length(bytes))
    unsafe_store!(
        args.serialized_topology,
        Ptr{CAPI.PJRT_SerializedTopology}(pointer_from_objref(bytes)),
    )
    unsafe_store!(
        args.serialized_topology_deleter,
        @cfunction(pjrt_serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    )
    return ok()
end

# C views of the results of `attributes(::TopologyDescription)`, which C keeps for as long as the topology is alive
const topology_attributes_cache = WeakKeyIdDict{
    TopologyDescription,Tuple{Vector{CAPI.PJRT_NamedValue},Vector{Any}}
}()

"""
    attributes(topology::TopologyDescription)

Return the platform-specific attributes of `topology` as `name => value` pairs (e.g. a `Dict`). Values must be
`AbstractString`, `Bool`, `Integer`, `AbstractVector{<:Integer}` or `AbstractFloat`, which C receives as `String`,
`Bool`, `Int64`, `Vector{Int64}` and `Float32`.

It is called at most once per `topology`: the C wrapper keeps a copy for as long as `topology` is alive, since C keeps
pointers to it.

Backs `PJRT_TopologyDescription_Attributes`.
"""
attributes(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_attributes(
    args::Ptr{CAPI.PJRT_TopologyDescription_Attributes_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    values, _ = get!(topology_attributes_cache, topology) do
        pjrt_named_values(attributes(topology))
    end
    unsafe_store!(args.attributes, pointer(values))
    unsafe_store!(args.num_attributes, length(values))
    return ok()
end

"""
    fingerprint(topology::TopologyDescription) -> UInt64

Return a fingerprint of `topology` for use in cache keys. It doesn't need to be stable across versions.

Backs `PJRT_TopologyDescription_Fingerprint`.
"""
fingerprint(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_fingerprint(
    args::Ptr{CAPI.PJRT_TopologyDescription_Fingerprint_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    unsafe_store!(args.fingerprint, fingerprint(topology)::UInt64)
    return ok()
end

"""
    canonical_shape(topology::TopologyDescription, kind_id::Integer, type, dims, layout)
        -> Reactant.Proto.xla.ShapeProto

Return the shape, including its layout, that a buffer with element type `type::CAPI.PJRT_Buffer_Type`, dimensions
`dims::Vector{Int64}` and layout `layout::Union{Nothing,MemoryLayout}` (`nothing` if none is given) takes once
transferred to a memory space of kind id `kind_id` (see `memory_space_kind_ids`).

Backs `PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace`.
"""
canonical_shape(::TopologyDescription, ::Integer, args...)

@pjrt_catch_return function pjrt_topology_description_make_canonical_shape_for_memory_space(
    args::Ptr{CAPI.PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    kind_id = unsafe_load(args.memory_space_kind_id)
    _eltype = unsafe_load(args.element_type)
    _size = copy(
        unsafe_wrap(Vector{Int64}, unsafe_load(args.dims), unsafe_load(args.num_dims))
    )
    c_layout = unsafe_load(args.layout)
    layout = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)

    shape = canonical_shape(
        topology, kind_id, _eltype, _size, layout
    )::Reactant.Proto.xla.ShapeProto
    bytes = Reactant.ProtoUtils.proto_to_bytes(shape)

    # the deleter only gets the pointer to the data, so it's malloc'ed and freed by `free` instead of GC rooted
    ptr = Ptr{UInt8}(Libc.malloc(max(length(bytes), 1)))
    ptr == C_NULL && throw(resource_exhausted())
    GC.@preserve bytes unsafe_copyto!(ptr, pointer(bytes), length(bytes))
    unsafe_store!(args.serialized_shape, Cstring(ptr))
    unsafe_store!(args.serialized_shape_size, length(bytes))
    unsafe_store!(args.serialized_shape_deleter, cglobal(:free))
    return ok()
end

"""
    memory_space_kind_ids(topology::TopologyDescription) -> Vector{Cint}

Return the kind ids of the memory spaces in `topology` (see `kind_id(::Memory)`). The first one must be the kind id
of the default memory space.

!!! warning
The vector is handed to C by pointer, so it must be owned by `topology` and never be resized.

Backs `PJRT_TopologyDescription_GetMemorySpaceKindIds`.
"""
memory_space_kind_ids(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_get_memory_space_kind_ids(
    args::Ptr{CAPI.PJRT_TopologyDescription_GetMemorySpaceKindIds_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    ids = memory_space_kind_ids(topology)::Vector{Cint}
    unsafe_store!(args.memory_space_kind_ids, pointer(ids))
    unsafe_store!(args.num_memory_space_kind_ids, length(ids))
    return ok()
end

"""
    topology_description(client::Client) -> TopologyDescription

Return the description of the topology of the devices of `client`.

The topology must be kept alive by `client`: C doesn't destroy it.

Backs `PJRT_Client_TopologyDescription`.
"""
topology_description(::Client)

@pjrt_catch_return function pjrt_client_topology_description(
    args::Ptr{CAPI.PJRT_Client_TopologyDescription_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    topology = topology_description(client)::TopologyDescription
    unsafe_store!(
        args.topology, Ptr{CAPI.PJRT_TopologyDescription}(pointer_from_objref(topology))
    )
    return ok()
end

"""
    compile(topology::TopologyDescription, code::MLIR.IR.Module, options, client::Union{Nothing,Client}) -> Executable

Compile `code` for the devices described by `topology` and return an executable, which must be loaded with `load`
before it can run.

`code` and `options::Reactant.Proto.xla.CompileOptionsProto` are like in `compile(::Client, ...)`. `client` is an
optional client of the same platform, for performance-guided optimizations.

The caller GC-roots the returned executable.

Backs `PJRT_Compile`.
"""
compile(::TopologyDescription, args...)

@pjrt_catch_return function pjrt_compile(
    args::Ptr{CAPI.PJRT_Compile_Args}
)::Ptr{CAPI.PJRT_Error}
    topology_ptr = unsafe_load(args.topology)
    topology_ptr == C_NULL && throw(invalid_argument("No topology has been given"))
    topology = unsafe_pointer_to_objref(topology_ptr)::TopologyDescription

    client_ptr = unsafe_load(args.client)
    client = client_ptr == C_NULL ? nothing : unsafe_pointer_to_objref(client_ptr)::Client

    mlir_mod = unsafe_load_program(Ptr{CAPI.PJRT_Program}(unsafe_load(args.program)))
    options = unsafe_load_compile_options(
        unsafe_load(args.compile_options), unsafe_load(args.compile_options_size)
    )
    exe = compile(topology, mlir_mod, options, client)::Executable
    root!(exe)
    unsafe_store!(args.executable, Ptr{CAPI.PJRT_Executable}(pointer_from_objref(exe)))
    return ok()
end
