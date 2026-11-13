# A `PJRT_Layouts_MemoryLayout*` points to a `Ref{MemoryLayout}`. The caller owns the layouts of buffers and the default
# layouts, so each one is a new `Ref` that is GC rooted until `PJRT_Layouts_MemoryLayout_Destroy`, while the layouts of
# an executable are owned by the executable.

@pjrt_catch_return function pjrt_layouts_memory_layout_destroy(
    args::Ptr{CAPI.PJRT_Layouts_MemoryLayout_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.layout)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

@pjrt_catch_return function pjrt_layouts_memory_layout_serialize(
    args::Ptr{CAPI.PJRT_Layouts_MemoryLayout_Serialize_Args}
)::Ptr{CAPI.PJRT_Error}
    layout = (unsafe_pointer_to_objref(unsafe_load(args.layout))::Ref{MemoryLayout})[]
    layout isa TiledMemoryLayout || throw(
        unimplemented("only `TiledMemoryLayout` can be serialized, got $(typeof(layout))"),
    )
    # XLA parses it with `xla::ParseLayout`, so it uses the format of `xla::Layout::ToString` (e.g. `{1,0}`, or
    # `{1,0:T(8,128)(2,1)}` if tiled)
    tiles = if isempty(layout.tiles)
        ""
    else
        ":T" * join("(" * join(tile, ",") * ")" for tile in layout.tiles)
    end
    bytes = root!(Vector{UInt8}("{" * join(layout.minor_to_major, ",") * tiles * "}"))
    unsafe_store!(args.serialized_bytes, Cstring(pointer(bytes)))
    unsafe_store!(args.serialized_bytes_size, length(bytes))
    unsafe_store!(
        args.serialized_layout,
        Ptr{CAPI.PJRT_Layouts_SerializedLayout}(pointer_from_objref(bytes)),
    )
    unsafe_store!(
        args.serialized_layout_deleter,
        @cfunction(pjrt_serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    )
    return ok()
end

"""
    default_memory_layout(client::Client, type, dims::Vector{Int64}) -> MemoryLayout
    default_memory_layout(topology::TopologyDescription, type, dims::Vector{Int64}) -> MemoryLayout

Return the default [`MemoryLayout`](@ref) of `client` (or `topology`) for an array of element type
`type::CAPI.PJRT_Buffer_Type` and dimensions `dims` (e.g. `TiledMemoryLayout(n-1:-1:0)` for XLA's default row-major
layout, with `n = length(dims)`).

Backs `PJRT_Layouts_PJRT_Client_GetDefaultLayout` and `PJRT_Layouts_PJRT_Topology_GetDefaultLayout`.
"""
function default_memory_layout end

@pjrt_catch_return function pjrt_layouts_client_get_default_layout(
    args::Ptr{CAPI.PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    dims = copy(
        unsafe_wrap(Vector{Int64}, unsafe_load(args.dims), unsafe_load(args.num_dims))
    )
    layout = root!(
        Ref{MemoryLayout}(
            default_memory_layout(client, unsafe_load(args.type), dims)::MemoryLayout
        ),
    )
    unsafe_store!(
        args.layout, Ptr{CAPI.PJRT_Layouts_MemoryLayout}(pointer_from_objref(layout))
    )
    return ok()
end

@pjrt_catch_return function pjrt_layouts_topology_get_default_layout(
    args::Ptr{CAPI.PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(
        unsafe_load(args.topology_description)
    )::TopologyDescription
    dims = copy(
        unsafe_wrap(Vector{Int64}, unsafe_load(args.dims), unsafe_load(args.num_dims))
    )
    layout = root!(
        Ref{MemoryLayout}(
            default_memory_layout(topology, unsafe_load(args.type), dims)::MemoryLayout
        ),
    )
    unsafe_store!(
        args.layout, Ptr{CAPI.PJRT_Layouts_MemoryLayout}(pointer_from_objref(layout))
    )
    return ok()
end

# reuses `memory_layout(::Buffer)`, which backs `PJRT_Buffer_GetMemoryLayout`; unlike there, it isn't cached, since C
# doesn't borrow anything from it
@pjrt_catch_return function pjrt_layouts_buffer_memory_layout(
    args::Ptr{CAPI.PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    layout = root!(Ref{MemoryLayout}(memory_layout(buffer)::MemoryLayout))
    unsafe_store!(
        args.layout, Ptr{CAPI.PJRT_Layouts_MemoryLayout}(pointer_from_objref(layout))
    )
    return ok()
end

"""
    parameter_memory_layouts(exe::Executable) -> Vector{<:MemoryLayout}

Return the [`MemoryLayout`](@ref) of each parameter of `exe`.

The returned value doesn't need to be owned by `exe`, but it must be the same on every call: the C wrapper calls it at
most once per `exe` and keeps a copy for as long as `exe` is alive, since C keeps pointers to it.

Backs `PJRT_Layouts_PJRT_Executable_GetParameterLayouts`.
"""
function parameter_memory_layouts end

"""
    output_memory_layouts(exe::Executable) -> Vector{<:MemoryLayout}

Return the [`MemoryLayout`](@ref) of each output of `exe`.

The returned value doesn't need to be owned by `exe`, but it must be the same on every call: the C wrapper calls it at
most once per `exe` and keeps a copy for as long as `exe` is alive, since C keeps pointers to it.

Backs `PJRT_Layouts_PJRT_Executable_GetOutputLayouts`.
"""
function output_memory_layouts end

const parameter_memory_layouts_cache = WeakKeyIdDict{
    Executable,Tuple{Vector{Ref{MemoryLayout}},Vector{Ptr{CAPI.PJRT_Layouts_MemoryLayout}}}
}()

@pjrt_catch_return function pjrt_layouts_executable_get_parameter_layouts(
    args::Ptr{CAPI.PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    _, ptrs = get!(parameter_memory_layouts_cache, exe) do
        layouts = map(Ref{MemoryLayout}, parameter_memory_layouts(exe)::Vector{<:MemoryLayout})
        (
            layouts,
            [Ptr{CAPI.PJRT_Layouts_MemoryLayout}(pointer_from_objref(l)) for l in layouts],
        )
    end
    unsafe_store!(args.num_parameters, length(ptrs))
    unsafe_store!(args.layouts, pointer(ptrs))
    return ok()
end

const output_memory_layouts_cache = WeakKeyIdDict{
    Executable,Tuple{Vector{Ref{MemoryLayout}},Vector{Ptr{CAPI.PJRT_Layouts_MemoryLayout}}}
}()

@pjrt_catch_return function pjrt_layouts_executable_get_output_layouts(
    args::Ptr{CAPI.PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    _, ptrs = get!(output_memory_layouts_cache, exe) do
        layouts = map(Ref{MemoryLayout}, output_memory_layouts(exe)::Vector{<:MemoryLayout})
        (
            layouts,
            [Ptr{CAPI.PJRT_Layouts_MemoryLayout}(pointer_from_objref(l)) for l in layouts],
        )
    end
    unsafe_store!(args.num_outputs, length(ptrs))
    unsafe_store!(args.layouts, pointer(ptrs))
    return ok()
end

"""
    LayoutsExtension(plugin::AbstractPlugin, next::Ptr{CAPI.PJRT_Extension_Base} = C_NULL)

Advertises `PJRT_Layouts_Extension`, through which XLA gets the memory layouts of buffers, executables and default
layouts of clients and topologies. Without it, XLA assumes the default row-major layout everywhere. Pass it to
[`make_api`](@ref).

## Interface

- [`default_memory_layout(::Client)`](@ref default_memory_layout)
- [`default_memory_layout(::TopologyDescription)`](@ref default_memory_layout)
- [`parameter_memory_layouts`](@ref)
- [`output_memory_layouts`](@ref)
- [`memory_layout(::Buffer)`](@ref)

!!! warning
The layouts they return must be [`TiledMemoryLayout`](@ref)s, since `xla::Layout` can't describe strides. XLA aborts if
the layout of a buffer or executable can't be serialized, and returns an error for the default layouts.

!!! note
If this extension is advertised, frameworks (e.g. JAX) may assume that the input program can contain
`mhlo.layout_mode` attributes in its inputs and outputs, and that the layouts returned here reflect them.
"""
mutable struct LayoutsExtension <: Extension
    const ftable::CAPI.PJRT_Layouts_Extension

    function LayoutsExtension(::AbstractPlugin, next::Ptr{CAPI.PJRT_Extension_Base}=C_NULL)
        base = CAPI.PJRT_Extension_Base(
            csizeof(CAPI.PJRT_Layouts_Extension), extension_type(LayoutsExtension), next
        )
        #! format: off
        return new(CAPI.PJRT_Layouts_Extension(
            base,
            #= PJRT_Layouts_MemoryLayout_Destroy =# @cfunction(pjrt_layouts_memory_layout_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Layouts_MemoryLayout_Destroy_Args},)),
            #= PJRT_Layouts_MemoryLayout_Serialize =# @cfunction(pjrt_layouts_memory_layout_serialize, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Layouts_MemoryLayout_Serialize_Args},)),
            #= PJRT_Layouts_PJRT_Client_GetDefaultLayout =# @cfunction(pjrt_layouts_client_get_default_layout, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args},)),
            #= PJRT_Layouts_PJRT_Buffer_MemoryLayout =# @cfunction(pjrt_layouts_buffer_memory_layout, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args},)),
            #= PJRT_Layouts_PJRT_Topology_GetDefaultLayout =# @cfunction(pjrt_layouts_topology_get_default_layout, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args},)),
            #= PJRT_Layouts_PJRT_Executable_GetOutputLayouts =# @cfunction(pjrt_layouts_executable_get_output_layouts, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args},)),
            #= PJRT_Layouts_PJRT_Executable_GetParameterLayouts =# @cfunction(pjrt_layouts_executable_get_parameter_layouts, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args},)),
        ))
        #! format: on
    end
end

extension_type(::Type{LayoutsExtension}) = CAPI.PJRT_Extension_Type_Layouts
