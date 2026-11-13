"""
    AbstractPlugin

A type representing a PJRT plugin. Used for dispatching on plugin entrypoints, like client creation or the topologies
created without a client.

Pass an instance to [`make_api`](@ref) to build its `PJRT_Api`.

It must be stateless: its type `P` must be `Base.issingletontype(P)` with an empty constructor `P()`. Plugin-wide state
(e.g. a registry of FFI handlers) belongs in globals of the module of the plugin.

### Interface

- `initialize` (optional)
- `attributes` (optional)
- `create_client`
- `create_topology_description`
- `deserialize_topology_description`
"""
abstract type AbstractPlugin end

"""
    initialize(plugin::AbstractPlugin)

One-time setup of `plugin`, called by C before any other function of the plugin. Defaults to doing nothing.

Backs `PJRT_Plugin_Initialize`.
"""
initialize(::AbstractPlugin) = nothing

struct PluginInitialize{P<:AbstractPlugin} end

@pjrt_catch_return function (::PluginInitialize{P})(
    args::Ptr{CAPI.PJRT_Plugin_Initialize_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    initialize(P())
    return ok()
end

"""
    attributes(plugin::AbstractPlugin)

Return the attributes of `plugin` as `name => value` pairs (e.g. a `Dict`). Common names are `"xla_version"`,
`"stablehlo_current_version"` and `"stablehlo_minimum_version"`. Values must be `AbstractString`, `Bool`, `Integer`,
`AbstractVector{<:Integer}` or `AbstractFloat`, which C receives as `String`, `Bool`, `Int64`, `Vector{Int64}` and
`Float32`. Defaults to no attributes.

It is called at most once per plugin type: the C wrapper keeps a copy for the lifetime of the process, since C
keeps pointers to it.

Backs `PJRT_Plugin_Attributes`.
"""
attributes(::AbstractPlugin) = ()

# the C view of the attributes of each plugin type, computed on the first call since the attributes may depend on
# `initialize`, and never freed since C keeps pointers to it
const plugin_attributes = IdDict{DataType,Tuple{Vector{CAPI.PJRT_NamedValue},Vector{Any}}}()
const plugin_attributes_lock = ReentrantLock()

struct PluginAttributes{P<:AbstractPlugin} end

@pjrt_catch_return function (::PluginAttributes{P})(
    args::Ptr{CAPI.PJRT_Plugin_Attributes_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    values, _ = @lock plugin_attributes_lock begin
        get!(() -> pjrt_named_values(attributes(P())), plugin_attributes, P)
    end
    unsafe_store!(args.attributes, pointer(values))
    unsafe_store!(args.num_attributes, length(values))
    return ok()
end

"""
    create_client(plugin::AbstractPlugin, options::Dict{String,Any}) -> Client

Return a new client of `plugin`. `options` holds platform-specific options, whose values are `String`, `Int64`,
`Vector{Int64}`, `Float32` or `Bool`.

The caller GC-roots the returned client until `PJRT_Client_Destroy`.

Backs `PJRT_Client_Create`.
"""
create_client(::AbstractPlugin, ::Dict{String,Any})

struct ClientCreate{P<:AbstractPlugin} end

@pjrt_catch_return function (::ClientCreate{P})(
    args::Ptr{CAPI.PJRT_Client_Create_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    options = unsafe_load_pjrt_named_values(
        unsafe_load(args.create_options), unsafe_load(args.num_options)
    )
    client = root!(create_client(P(), options)::Client)
    unsafe_store!(args.client, Ptr{CAPI.PJRT_Client}(pointer_from_objref(client)))
    return ok()
end

"""
    create_topology_description(plugin::AbstractPlugin, name::String, options::Dict{String,Any}) -> TopologyDescription

Return a new description of the topology named `name` of the platform of `plugin`, without creating a client.
`options` holds platform-specific options, whose values are `String`, `Int64`, `Vector{Int64}`, `Float32` or `Bool`.

The caller GC-roots the returned topology until `PJRT_TopologyDescription_Destroy`.

Backs `PJRT_TopologyDescription_Create`.
"""
create_topology_description(::AbstractPlugin, ::String, ::Dict{String,Any})

struct TopologyDescriptionCreate{P<:AbstractPlugin} end

@pjrt_catch_return function (::TopologyDescriptionCreate{P})(
    args::Ptr{CAPI.PJRT_TopologyDescription_Create_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    name = unsafe_string(
        Ptr{UInt8}(unsafe_load(args.topology_name)), unsafe_load(args.topology_name_size)
    )
    options = unsafe_load_pjrt_named_values(
        unsafe_load(args.create_options), unsafe_load(args.num_options)
    )
    topology = create_topology_description(P(), name, options)::TopologyDescription
    root!(topology)
    unsafe_store!(
        args.topology, Ptr{CAPI.PJRT_TopologyDescription}(pointer_from_objref(topology))
    )
    return ok()
end

"""
    deserialize_topology_description(plugin::AbstractPlugin, bytes::Vector{UInt8}) -> TopologyDescription

Return the topology description of the platform of `plugin` serialized in `bytes` by `serialize`.

The caller GC-roots the returned topology until `PJRT_TopologyDescription_Destroy`.

Backs `PJRT_TopologyDescription_Deserialize`.
"""
deserialize_topology_description(::AbstractPlugin, ::Vector{UInt8})

struct TopologyDescriptionDeserialize{P<:AbstractPlugin} end

@pjrt_catch_return function (::TopologyDescriptionDeserialize{P})(
    args::Ptr{CAPI.PJRT_TopologyDescription_Deserialize_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    # the serialized topology is owned by C, so copy it in case the implementation stores it
    bytes_ptr = Ptr{UInt8}(unsafe_load(args.serialized_topology))
    bytes = copy(
        unsafe_wrap(Vector{UInt8}, bytes_ptr, unsafe_load(args.serialized_topology_size))
    )
    topology = deserialize_topology_description(P(), bytes)::TopologyDescription
    root!(topology)
    unsafe_store!(
        args.topology, Ptr{CAPI.PJRT_TopologyDescription}(pointer_from_objref(topology))
    )
    return ok()
end

@pjrt_catch_return function pjrt_topology_description_destroy(
    args::Ptr{CAPI.PJRT_TopologyDescription_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.topology)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end
