"""
    Buffer

### Interface

- `Base.delete!`
- `isdeleted`
- `Base.eltype`
- `pjrt_eltype` (defaults to `pjrt_eltype(Base.eltype(buffer))`)
- `Base.size`
- `unpadded_size` (defaults to `Base.size`)
- `dynamic_dims`
- `memory_layout`
- `Base.sizeof`
- `device`
- `memory`
- `copyto(::Device, ::Buffer)`
- `copyto(::Memory, ::Buffer)`
- `to_host_buffer!`
- `Base.unsafe_copyto!`
- `bitcast!`
- `donate!`
- `opaque_pointer` (defaults to `Base.pointer`)
- `incref!`
- `decref!`
- `isoncpu`
- `ready_event`
- `Base.pointer`

!!! info
The `Buffer(client, ...)` constructors (see [`Client`](@ref)) return the plugin's own subtype of `Buffer`, chosen by
dispatch on their `client`, `dev` and `mem` arguments.
"""
abstract type Buffer end

@pjrt_catch_return function pjrt_buffer_destroy(
    args::Ptr{CAPI.PJRT_Buffer_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    delete!(buffer)
    unroot!(buffer)
    return ok()
end

"""
    Base.delete!(buffer::Buffer)

Free the data of `buffer` while keeping its metadata alive. Must be idempotent.
After this call, `isdeleted(buffer)` must return `true`.

Backs `PJRT_Buffer_Delete`.
"""
Base.delete!(::Buffer)

@pjrt_catch_return function pjrt_buffer_delete(
    args::Ptr{CAPI.PJRT_Buffer_Delete_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    delete!(buffer)
    return ok()
end

"""
    isdeleted(buffer::Buffer) -> Bool

Return whether the data of `buffer` has been freed with `Base.delete!`.

Backs `PJRT_Buffer_IsDeleted`.
"""
isdeleted(::Buffer)

@pjrt_catch_return function pjrt_buffer_is_deleted(
    args::Ptr{CAPI.PJRT_Buffer_IsDeleted_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    pred = isdeleted(buffer)::Bool
    unsafe_store!(args.is_deleted, pred)
    return ok()
end

"""
    Base.eltype(buffer::Buffer) -> Type

Return the Julia element type of `buffer`.

Backs `PJRT_Buffer_ElementType` through the default of [`pjrt_eltype`](@ref).
"""
Base.eltype(::Buffer)

"""
    pjrt_eltype(buffer::Buffer) -> CAPI.PJRT_Buffer_Type

Return the PJRT element type of `buffer`.

Defaults to `pjrt_eltype(Base.eltype(buffer))`. Override it if the element type has no Julia equivalent, or if `buffer`
already stores its PJRT element type.

Backs `PJRT_Buffer_ElementType`.
"""
pjrt_eltype(buffer::Buffer) = pjrt_eltype(eltype(buffer))

@pjrt_catch_return function pjrt_buffer_element_type(
    args::Ptr{CAPI.PJRT_Buffer_ElementType_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    _pjrt_eltype = pjrt_eltype(buffer)::CAPI.PJRT_Buffer_Type
    unsafe_store!(args.type, _pjrt_eltype)
    return ok()
end

"""
    Base.size(buffer::Buffer) -> Dims

Return the dimensions of `buffer`.

The returned value doesn't need to be owned by `buffer`, but it must be the same on every call: the C wrapper calls it
at most once per `buffer` and keeps a copy for as long as `buffer` is alive, since C keeps pointers to it.

Backs `PJRT_Buffer_Dimensions`.
"""
Base.size(::Buffer)

# C borrows the dimensions, so they are kept for as long as the buffer is alive
const size_cache = WeakKeyIdDict{Buffer,Vector{Int64}}()

@pjrt_catch_return function pjrt_buffer_dimensions(
    args::Ptr{CAPI.PJRT_Buffer_Dimensions_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dims = get!(size_cache, buffer) do
        collect(Int64, size(buffer)::Dims)
    end
    unsafe_store!(args.dims, pointer(dims))
    unsafe_store!(args.num_dims, length(dims))
    return ok()
end

"""
    unpadded_size(buffer::Buffer) -> Dims

Return the dimensions of `buffer` without padding. They usually are `Base.size(buffer)`, but dynamically-sized
dimensions implemented by padding to a fixed size may be smaller.

Defaults to `Base.size(buffer)`, i.e. no dimension is padded.

The returned value doesn't need to be owned by `buffer`, but it must be the same on every call: the C wrapper calls it
at most once per `buffer` and keeps a copy for as long as `buffer` is alive, since C keeps pointers to it.

Backs `PJRT_Buffer_UnpaddedDimensions`.
"""
unpadded_size(buffer::Buffer) = size(buffer)

# C borrows the dimensions, so they are kept for as long as the buffer is alive
const unpadded_size_cache = WeakKeyIdDict{Buffer,Vector{Int64}}()

@pjrt_catch_return function pjrt_buffer_unpadded_dimensions(
    args::Ptr{CAPI.PJRT_Buffer_UnpaddedDimensions_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dims = get!(unpadded_size_cache, buffer) do
        collect(Int64, unpadded_size(buffer)::Dims)
    end
    unsafe_store!(args.unpadded_dims, pointer(dims))
    unsafe_store!(args.num_dims, length(dims))
    return ok()
end

"""
    dynamic_dims(buffer::Buffer) -> Vector{Int}

Return the 1-based indices of the dynamically-sized dimensions of `buffer`, or an empty vector if all are static. The
C wrapper converts them to the 0-based indices that C expects.

The returned value doesn't need to be owned by `buffer`, but it must be the same on every call: the C wrapper calls it
at most once per `buffer` and keeps a copy for as long as `buffer` is alive, since C keeps pointers to it.

Backs `PJRT_Buffer_DynamicDimensionIndices`.
"""
dynamic_dims(::Buffer)

# C borrows the vector, so it is kept for as long as the buffer is alive
const dynamic_dims_cache = WeakKeyIdDict{Buffer,Vector{Csize_t}}()

@pjrt_catch_return function pjrt_buffer_dynamic_dimension_indices(
    args::Ptr{CAPI.PJRT_Buffer_DynamicDimensionIndices_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    indices = get!(dynamic_dims_cache, buffer) do
        dims = dynamic_dims(buffer)
        n = length(size(buffer))
        if !all(d -> 1 <= d <= n, dims)
            msg = "`dynamic_dims` returned $dims, but the dimensions of the buffer are 1:$n"
            throw(Error(CAPI.PJRT_Error_Code_INTERNAL, msg))
        end
        # C expects 0-based indices
        Csize_t[d - 1 for d in dims]
    end
    unsafe_store!(args.dynamic_dim_indices, pointer(indices))
    unsafe_store!(args.num_dynamic_dims, length(indices))
    return ok()
end

"""
    memory_layout(buffer::Buffer) -> MemoryLayout

Return the layout of `buffer` in memory (e.g. `TiledMemoryLayout(n-1:-1:0)` for XLA's default row-major layout of an
array with `n` dimensions).

If the plugin advertises [`LayoutsExtension`](@ref), it must return a [`TiledMemoryLayout`](@ref): XLA aborts if the
layout of a buffer can't be serialized.

The returned value doesn't need to be owned by `buffer`, but it must be the same on every call: the C wrapper calls it
at most once per `buffer` and keeps a copy for as long as `buffer` is alive, since C keeps pointers to it.

Backs `PJRT_Buffer_GetMemoryLayout` and `PJRT_Layouts_PJRT_Buffer_MemoryLayout`.
"""
memory_layout(::Buffer)

# C borrows the arrays of `PJRT_Buffer_MemoryLayout`, so they are kept for as long as the buffer is alive (like in XLA's
# reference plugin, which keeps them in the buffer)
const memory_layout_cache = WeakKeyIdDict{
    Buffer,Tuple{MemoryLayout,Vector{Int64},Vector{Csize_t}}
}()

@pjrt_catch_return function pjrt_buffer_get_memory_layout(
    args::Ptr{CAPI.PJRT_Buffer_GetMemoryLayout_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    # for tiled layouts, C expects the sizes of all tiles back to back, plus the number of dimensions of each tile
    layout, tile_dims, tile_dim_sizes = get!(memory_layout_cache, buffer) do
        buffer_layout = memory_layout(buffer)::MemoryLayout
        if buffer_layout isa StridedMemoryLayout
            return (buffer_layout, Int64[], Csize_t[])
        elseif buffer_layout isa TiledMemoryLayout
            (
                buffer_layout,
                Int64[d for tile in buffer_layout.tiles for d in tile],
                Csize_t[length(tile) for tile in buffer_layout.tiles],
            )
        else
            throw(unimplemented("unsupported layout type $(typeof(buffer_layout))"))
        end
    end
    c_layout = args.layout
    unsafe_store!(c_layout.struct_size, csizeof(CAPI.PJRT_Buffer_MemoryLayout))
    unsafe_store!(c_layout.extension_start, C_NULL)
    if layout isa StridedMemoryLayout
        unsafe_store!(c_layout.type, CAPI.PJRT_Buffer_MemoryLayout_Type_Strides)
        unsafe_store!(
            c_layout.strides,
            CAPI.PJRT_Buffer_MemoryLayout_Strides(
                csizeof(CAPI.PJRT_Buffer_MemoryLayout_Strides), #= struct_size =#
                C_NULL, #= extension_start =#
                pointer(layout.byte_strides), #= byte_strides =#
                length(layout.byte_strides), #= num_byte_strides =#
            ),
        )
    else
        unsafe_store!(c_layout.type, CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled)
        unsafe_store!(
            c_layout.tiled,
            CAPI.PJRT_Buffer_MemoryLayout_Tiled(
                csizeof(CAPI.PJRT_Buffer_MemoryLayout_Tiled), #= struct_size =#
                C_NULL, #= extension_start =#
                pointer(layout.minor_to_major), #= minor_to_major =#
                length(layout.minor_to_major), #= minor_to_major_size =#
                pointer(tile_dims), #= tile_dims =#
                pointer(tile_dim_sizes), #= tile_dim_sizes =#
                length(tile_dim_sizes), #= num_tiles =#
            ),
        )
    end
    return ok()
end

"""
    device(buffer::Buffer) -> Device

Return the device where `buffer` resides. The device must be kept alive by its client.

Backs `PJRT_Buffer_Device`.
"""
device(::Buffer)

@pjrt_catch_return function pjrt_buffer_device(
    args::Ptr{CAPI.PJRT_Buffer_Device_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dev = device(buffer)::Device
    ptr = Ptr{CAPI.PJRT_Device}(pointer_from_objref(dev))
    unsafe_store!(args.device, ptr)
    return ok()
end

"""
    memory(buffer::Buffer) -> Memory

Return the memory space where `buffer` resides. The memory must be kept alive by its client.

Backs `PJRT_Buffer_Memory`.
"""
memory(::Buffer)

@pjrt_catch_return function pjrt_buffer_memory(
    args::Ptr{CAPI.PJRT_Buffer_Memory_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    mem = memory(buffer)::Memory
    ptr = Ptr{CAPI.PJRT_Memory}(pointer_from_objref(mem))
    unsafe_store!(args.memory, ptr)
    return ok()
end

"""
    Base.sizeof(buffer::Buffer) -> Integer

Return the number of bytes that `buffer` occupies on its device.

Must be implemented: Base's fallback silently returns the size of the Julia object, so this one throws instead.

Backs `PJRT_Buffer_OnDeviceSizeInBytes`.
"""
function Base.sizeof(buffer::Buffer)
    throw(unimplemented("`sizeof` is not implemented for $(typeof(buffer))"))
end

@pjrt_catch_return function pjrt_buffer_on_device_size_in_bytes(
    args::Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    _size = sizeof(buffer)
    unsafe_store!(args.on_device_size_in_bytes, Csize_t(_size))
    return ok()
end

"""
    copyto(dst::Device, buffer::Buffer) -> Buffer

Return a new buffer with a copy of the data of `buffer` placed on `dst`.
Must fail if `buffer` already resides on `dst`.
The returned buffer is GC rooted by the caller.

Backs `PJRT_Buffer_CopyToDevice`.
"""
copyto(::Device, ::Buffer)

@pjrt_catch_return function pjrt_buffer_copy_to_device(
    args::Ptr{CAPI.PJRT_Buffer_CopyToDevice_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dst_dev = unsafe_pointer_to_objref(unsafe_load(args.dst_device))::Device

    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    dst_buffer = copyto(dst_dev, buffer)::Buffer
    root!(dst_buffer)
    ptr = Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(dst_buffer))
    unsafe_store!(args.dst_buffer, ptr)
    return ok()
end

"""
    copyto(dst::Memory, buffer::Buffer) -> Buffer

Return a new buffer with a copy of the data of `buffer` placed on memory space `dst`.
Unlike the PJRT C API header says, `dst` may be the memory of `buffer`: XLA's CPU client allows it and Reactant's
`CopyBufferToDevice` relies on it.
The returned buffer is GC rooted by the caller.

Backs `PJRT_Buffer_CopyToMemory`.
"""
copyto(::Memory, ::Buffer)

@pjrt_catch_return function pjrt_buffer_copy_to_memory(
    args::Ptr{CAPI.PJRT_Buffer_CopyToMemory_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dst_mem = unsafe_pointer_to_objref(unsafe_load(args.dst_memory))::Memory

    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    dst_buffer = copyto(dst_mem, buffer)::Buffer
    root!(dst_buffer)
    ptr = Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(dst_buffer))
    unsafe_store!(args.dst_buffer, ptr)
    return ok()
end

"""
    to_host_buffer!(dst::Ptr, buffer::Buffer)

Copy the data of `buffer` to the host memory pointed by `dst`, which must hold at least `Base.sizeof(buffer)` bytes.

!!! todo
The copy is _currently_ synchronous.

Backs `PJRT_Buffer_ToHostBuffer`.
"""
to_host_buffer!(::Ptr, ::Buffer)

@pjrt_catch_return function pjrt_buffer_to_host_buffer(
    args::Ptr{CAPI.PJRT_Buffer_ToHostBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.src))::Buffer
    dst = unsafe_load(args.dst)
    src_size = sizeof(buffer)

    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    # size query shortcut
    if dst == C_NULL
        _size = src_size
        unsafe_store!(args.dst_size, Csize_t(_size))
        return ok()
    end

    dst_size = unsafe_load(args.dst_size)
    if dst_size < src_size
        throw(
            invalid_argument(
                "passed a host buffer of $dst_size bytes when $src_size bytes at least are required",
            ),
        )
    end

    # TODO relayout is not supported, so `host_layout` must be the layout of `buffer`. If it's `C_NULL`, the layout of
    # `buffer` is used.
    c_host_layout = unsafe_load(args.host_layout)
    if c_host_layout != C_NULL
        host_layout = MemoryLayout(c_host_layout)
        buffer_layout = memory_layout(buffer)::MemoryLayout
        host_layout == buffer_layout || throw(
            unimplemented("relayout from $buffer_layout to $host_layout is not supported"),
        )
    end

    to_host_buffer!(dst, buffer)

    event = root!(Event())
    notify(event)
    unsafe_store!(args.event, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    return ok()
end

"""
    Base.unsafe_copyto!(dst::Ptr, buffer::Buffer, offset::Integer, nbytes::Integer) -> Ptr

Copy `nbytes` bytes of the raw on-device data of `buffer`, starting at byte `offset`, to the host memory pointed by
`dst`. Like in Base, return `dst`.

`offset` is 0-based, as in PJRT, unlike the 1-based offsets of Base. The C wrapper checks that the range is within
`Base.sizeof(buffer)`.

!!! todo
The copy is _currently_ synchronous.

Backs `PJRT_Buffer_CopyRawToHost`.
"""
Base.unsafe_copyto!(::Ptr, ::Buffer, ::Integer, ::Integer)

@pjrt_catch_return function pjrt_buffer_copy_raw_to_host(
    args::Ptr{CAPI.PJRT_Buffer_CopyRawToHost_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dst = unsafe_load(args.dst)
    offset = unsafe_load(args.offset)
    transfer_size = unsafe_load(args.transfer_size)

    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    src_size = sizeof(buffer)
    if offset < 0 || transfer_size < 0 || offset + transfer_size > src_size
        throw(
            invalid_argument(
                "copy range [$offset, $(offset + transfer_size)) is out of the $src_size bytes of the buffer",
            ),
        )
    end

    unsafe_copyto!(dst, buffer, offset, transfer_size)

    event = root!(Event())
    notify(event)
    unsafe_store!(args.event, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    return ok()
end

"""
    bitcast!(buffer::Buffer, type, dims, layout) -> Buffer

Return a new buffer that takes over the on-device data of `buffer`, reinterpreted with element type
`type::CAPI.PJRT_Buffer_Type`, dimensions `dims::Vector{Int64}` and layout `layout::Union{Nothing,MemoryLayout}`
(`nothing` for the default layout, not the layout of `buffer`).

It must not copy any data, and the new buffer must have the same `Base.sizeof` as `buffer`. `buffer` is
donated: afterwards, `isdeleted(buffer)` must return `true`, but its data must not be freed.

The caller GC-roots the returned buffer.

Backs `PJRT_Buffer_Bitcast`.
"""
bitcast!(::Buffer, args...)

@pjrt_catch_return function pjrt_buffer_bitcast(
    args::Ptr{CAPI.PJRT_Buffer_Bitcast_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    _eltype = unsafe_load(args.element_type)
    _size = copy(
        unsafe_wrap(Vector{Int64}, unsafe_load(args.dims), unsafe_load(args.num_dims))
    )
    c_layout = unsafe_load(args.device_layout)
    layout = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)

    out_buffer = bitcast!(buffer, _eltype, _size, layout)::Buffer
    root!(out_buffer)
    unsafe_store!(args.out_buffer, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(out_buffer)))
    return ok()
end

# called by C when the destination of a `PJRT_Buffer_CopyRawToHostFuture` is ready (or failed): finishes the copy from
# the staging copy of the data
function pjrt_buffer_copy_raw_to_host_future_ready(
    args::Ptr{CAPI.PJRT_Buffer_CopyRawToHostFuture_Callback_Args}
)::Cvoid
    data = unsafe_pointer_to_objref(unsafe_load(args.callback_data))
    unroot!(data)
    staging, event = data[]::Tuple{Vector{UInt8},Event}

    code = unsafe_load(args.error_code)
    if code != CAPI.PJRT_Error_Code_OK
        error = Error(
            code, unsafe_load(args.error_message), unsafe_load(args.error_message_size)
        )
        notify(event, error)
        return nothing
    end

    dst = Ptr{UInt8}(unsafe_load(args.dst))
    GC.@preserve staging unsafe_copyto!(dst, pointer(staging), length(staging))
    notify(event)
    return nothing
end

@pjrt_catch_return function pjrt_buffer_copy_raw_to_host_future(
    args::Ptr{CAPI.PJRT_Buffer_CopyRawToHostFuture_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    offset = unsafe_load(args.offset)
    transfer_size = unsafe_load(args.transfer_size)

    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    src_size = sizeof(buffer)
    if offset < 0 || transfer_size < 0 || offset + transfer_size > src_size
        throw(
            invalid_argument(
                "copy range [$offset, $(offset + transfer_size)) is out of the $src_size bytes of the buffer",
            ),
        )
    end

    # the copy must happen even if `buffer` is deleted before the destination is ready, so the data is copied now
    # and moved to the destination by `pjrt_buffer_copy_raw_to_host_future_ready`
    staging = Vector{UInt8}(undef, transfer_size)
    GC.@preserve staging unsafe_copyto!(pointer(staging), buffer, offset, transfer_size)

    event = root!(Event())
    data = root!(Ref((staging, event)))
    unsafe_store!(args.event, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    unsafe_store!(args.callback_data, pointer_from_objref(data))
    unsafe_store!(
        args.future_ready_callback,
        @cfunction(
            pjrt_buffer_copy_raw_to_host_future_ready,
            Cvoid,
            (Ptr{CAPI.PJRT_Buffer_CopyRawToHostFuture_Callback_Args},)
        ),
    )
    return ok()
end

"""
    donate!(buffer::Buffer, dependency::Event) -> Buffer

Return a new buffer that takes over the on-device data of `buffer`, and which is ready only once both `buffer` and
`dependency` are ready. If either of them fails, the new buffer fails too.

`buffer` is donated: afterwards, `isdeleted(buffer)` must return `true`, but its data must not be freed.

`dependency` is set by the C wrapper when C signals it, possibly after this returns, so it must not be waited on here.
The caller GC-roots the returned buffer.

Backs `PJRT_Buffer_DonateWithControlDependency`.
"""
donate!(::Buffer, ::Event)

# called by C when the dependency of a `PJRT_Buffer_DonateWithControlDependency` is ready (or failed)
function pjrt_buffer_donate_dependency_ready(
    args::Ptr{CAPI.PJRT_Buffer_DonateWithControlDependency_Callback_Args}
)::Cvoid
    data = unsafe_pointer_to_objref(unsafe_load(args.callback_data))
    unroot!(data)
    dependency = data[]::Event
    code = unsafe_load(args.error_code)
    if code == CAPI.PJRT_Error_Code_OK
        notify(dependency)
    else
        error = Error(
            code, unsafe_load(args.error_message), unsafe_load(args.error_message_size)
        )
        notify(dependency, error)
    end
    return nothing
end

@pjrt_catch_return function pjrt_buffer_donate_with_control_dependency(
    args::Ptr{CAPI.PJRT_Buffer_DonateWithControlDependency_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    dependency = Event()
    out_buffer = donate!(buffer, dependency)::Buffer
    root!(out_buffer)

    # rooted in a `Ref` of its own, so that the callback unroots it even if the plugin hands `dependency` out to C
    data = root!(Ref(dependency))
    unsafe_store!(args.out_buffer, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(out_buffer)))
    unsafe_store!(args.callback_data, pointer_from_objref(data))
    unsafe_store!(
        args.dependency_ready_callback,
        @cfunction(
            pjrt_buffer_donate_dependency_ready,
            Cvoid,
            (Ptr{CAPI.PJRT_Buffer_DonateWithControlDependency_Callback_Args},)
        ),
    )
    return ok()
end

"""
    opaque_pointer(buffer::Buffer) -> Ptr

Return the opaque pointer to the device memory of `buffer`. It may become invalid at any point unless `buffer` has an
external reference (see `incref!`).

Defaults to `Base.pointer(buffer)`.

Backs `PJRT_Buffer_OpaqueDeviceMemoryDataPointer`.
"""
opaque_pointer(buffer::Buffer) = pointer(buffer)

@pjrt_catch_return function pjrt_buffer_opaque_device_memory_data_pointer(
    args::Ptr{CAPI.PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))
    ptr = opaque_pointer(buffer)::Ptr
    unsafe_store!(args.device_memory_ptr, ptr)
    return ok()
end

"""
    incref!(buffer::Buffer)

Increment the number of external references to `buffer`, which mean that its data is shared with another framework
(e.g. through DLPack). While there is any, the data of `buffer` must be neither freed (even by `Base.delete!`) nor
moved.

Backs `PJRT_Buffer_IncreaseExternalReferenceCount`.
"""
incref!(::Buffer)

@pjrt_catch_return function pjrt_buffer_increase_external_reference_count(
    args::Ptr{CAPI.PJRT_Buffer_IncreaseExternalReferenceCount_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    incref!(buffer)
    return ok()
end

"""
    decref!(buffer::Buffer)

Decrement the number of external references to `buffer` (see `incref!`). Must throw
[`invalid_argument`](@ref) if there is none.

Backs `PJRT_Buffer_DecreaseExternalReferenceCount`.
"""
decref!(::Buffer)

@pjrt_catch_return function pjrt_buffer_decrease_external_reference_count(
    args::Ptr{CAPI.PJRT_Buffer_DecreaseExternalReferenceCount_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    decref!(buffer)
    return ok()
end

"""
    isoncpu(buffer::Buffer) -> Bool

Return whether the data of `buffer` resides in host memory.

Backs `PJRT_Buffer_IsOnCpu`.
"""
isoncpu(::Buffer)

@pjrt_catch_return function pjrt_buffer_is_on_cpu(
    args::Ptr{CAPI.PJRT_Buffer_IsOnCpu_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    pred = isoncpu(buffer)::Bool
    unsafe_store!(args.is_on_cpu, pred)
    return ok()
end

"""
    ready_event(buffer::Buffer) -> Event

Return a new [`Event`](@ref) that is set once the data of `buffer` is ready, or with an [`Error`](@ref) if
producing it failed. A new event is needed on each call because the caller owns it and destroys it.
The returned event is GC rooted by the caller.

If `buffer` is deleted, the C wrapper returns an event set with an error without calling `ready_event`.

Backs `PJRT_Buffer_ReadyEvent`.
"""
ready_event(::Buffer)

@pjrt_catch_return function pjrt_buffer_ready_event(
    args::Ptr{CAPI.PJRT_Buffer_ReadyEvent_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer

    if isdeleted(buffer)
        event = Event()
        notify(event, invalid_argument("buffer is deleted"))
    else
        event = ready_event(buffer)::Event
    end
    root!(event)
    unsafe_store!(args.event, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    return ok()
end

"""
    Base.pointer(buffer::Buffer) -> Ptr

Return the raw pointer to the data of `buffer`. The pointer is only valid until `buffer` is deleted or destroyed.

Backs `PJRT_Buffer_UnsafePointer`.
"""
Base.pointer(::Buffer)

@pjrt_catch_return function pjrt_buffer_unsafe_pointer(
    args::Ptr{CAPI.PJRT_Buffer_UnsafePointer_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    ptr = pointer(buffer)::Ptr
    unsafe_store!(args.buffer_pointer, UInt(ptr))
    return ok()
end
