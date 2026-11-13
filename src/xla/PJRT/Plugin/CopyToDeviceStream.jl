"""
    CopyToDeviceStream

A stream of chunks of host data to be copied to a device, which expects `total_bytes` bytes in total, in chunks whose
size is a multiple of `granule_size`. It is complete once `current_bytes == total_bytes`, and never changes afterwards.

The C wrapper checks those conditions before calling `add_chunk!`.

!!! todo
`execute` ignores the receive callbacks of `PJRT_ExecuteOptions`, which is how C gets a stream, so nothing can hand
one to C yet.

### Interface

- `add_chunk!`
- `total_bytes`
- `granule_size`
- `current_bytes`
"""
abstract type CopyToDeviceStream end

@pjrt_catch_return function pjrt_copy_to_device_stream_destroy(
    args::Ptr{CAPI.PJRT_CopyToDeviceStream_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.stream)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

"""
    add_chunk!(stream::CopyToDeviceStream, data::Ptr{Cvoid}, size::Integer)

Copy the `size` bytes of host memory at `data` to the device as the next chunk of `stream`, adding `size` to
`current_bytes(stream)`.

`data` is freed right after this returns, so the copy must be done by then.

!!! todo
The copy is _currently_ synchronous: the event returned to C is set as soon as this returns.

Backs `PJRT_CopyToDeviceStream_AddChunk`.
"""
add_chunk!(::CopyToDeviceStream, ::Ptr{Cvoid}, ::Integer)

@pjrt_catch_return function pjrt_copy_to_device_stream_add_chunk(
    args::Ptr{CAPI.PJRT_CopyToDeviceStream_AddChunk_Args}
)::Ptr{CAPI.PJRT_Error}
    stream = unsafe_pointer_to_objref(unsafe_load(args.stream))::CopyToDeviceStream
    chunk_ptr = unsafe_load(args.chunk)
    chunk_ptr == C_NULL && throw(invalid_argument("chunk is null"))
    # `PJRT_Chunk` has no field accessors on pointers, so it's loaded by value
    chunk = unsafe_load(chunk_ptr)

    # the stream takes ownership of the chunk, so the chunk is freed whatever happens; invalid chunks and failed
    # transfers are reported through the returned event, like in XLA
    error = try
        total = total_bytes(stream)
        current = current_bytes(stream)
        granule = granule_size(stream)
        if current == total
            throw(invalid_argument("stream is already complete"))
        elseif chunk.size % granule != 0
            throw(
                invalid_argument(
                    "chunk of $(chunk.size) bytes is not a multiple of the $granule-byte granule",
                ),
            )
        elseif current + chunk.size > total
            throw(
                invalid_argument(
                    "chunk of $(chunk.size) bytes exceeds the $(total - current) bytes left in the stream",
                ),
            )
        end
        add_chunk!(stream, chunk.data, chunk.size)
        nothing
    catch err
        err isa Error || rethrow()
        err
    finally
        if chunk.deleter != C_NULL
            ccall(chunk.deleter, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), chunk.data, chunk.deleter_arg)
        end
    end

    event = root!(Event())
    notify(event, error)
    unsafe_store!(args.transfer_complete, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    return ok()
end

"""
    total_bytes(stream::CopyToDeviceStream) -> Integer

Return the total number of bytes that `stream` expects to be transferred.

Backs `PJRT_CopyToDeviceStream_TotalBytes`.
"""
total_bytes(::CopyToDeviceStream)

@pjrt_catch_return function pjrt_copy_to_device_stream_total_bytes(
    args::Ptr{CAPI.PJRT_CopyToDeviceStream_TotalBytes_Args}
)::Ptr{CAPI.PJRT_Error}
    stream = unsafe_pointer_to_objref(unsafe_load(args.stream))::CopyToDeviceStream
    unsafe_store!(args.total_bytes, total_bytes(stream))
    return ok()
end

"""
    granule_size(stream::CopyToDeviceStream) -> Integer

Return the granule size in bytes of `stream`: the size of every chunk must be a multiple of it.

Backs `PJRT_CopyToDeviceStream_GranuleSize`.
"""
granule_size(::CopyToDeviceStream)

@pjrt_catch_return function pjrt_copy_to_device_stream_granule_size(
    args::Ptr{CAPI.PJRT_CopyToDeviceStream_GranuleSize_Args}
)::Ptr{CAPI.PJRT_Error}
    stream = unsafe_pointer_to_objref(unsafe_load(args.stream))::CopyToDeviceStream
    unsafe_store!(args.granule_size_in_bytes, granule_size(stream))
    return ok()
end

"""
    current_bytes(stream::CopyToDeviceStream) -> Integer

Return the number of bytes that `stream` has transferred or buffered to transfer so far.

Backs `PJRT_CopyToDeviceStream_CurrentBytes`.
"""
current_bytes(::CopyToDeviceStream)

@pjrt_catch_return function pjrt_copy_to_device_stream_current_bytes(
    args::Ptr{CAPI.PJRT_CopyToDeviceStream_CurrentBytes_Args}
)::Ptr{CAPI.PJRT_Error}
    stream = unsafe_pointer_to_objref(unsafe_load(args.stream))::CopyToDeviceStream
    unsafe_store!(args.current_bytes, current_bytes(stream))
    return ok()
end

"""
    create_buffers_for_async_host_to_device(client::Client, specs::Vector, layouts::Vector, mem::Memory)
        -> AsyncHostToDeviceTransferManager

Return a new [`AsyncHostToDeviceTransferManager`](@ref) for a set of new buffers placed on memory `mem`, one per
element of `specs`, which are `NamedTuple`s with fields `element_type::CAPI.PJRT_Buffer_Type` and `dims::Vector{Int64}`.

`layouts::Vector{Union{Nothing,MemoryLayout}}` is either empty, for the default layouts, or holds the layout of each
buffer (`nothing` for its default layout).

The caller GC-roots the returned manager.

Backs `PJRT_Client_CreateBuffersForAsyncHostToDevice`.
"""
create_buffers_for_async_host_to_device(::Client, ::Vector, ::Vector, ::Memory)

@pjrt_catch_return function pjrt_client_create_buffers_for_async_host_to_device(
    args::Ptr{CAPI.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    mem_ptr = Ptr{Memory}(unsafe_load(args.memory))
    mem_ptr == C_NULL && throw(invalid_argument("No memory has been assigned"))
    mem = unsafe_pointer_to_objref(mem_ptr)::Memory

    c_specs = unsafe_load(args.shape_specs)
    specs = map(1:unsafe_load(args.num_shape_specs)) do i
        spec = unsafe_load(c_specs, i)
        (;
            element_type=spec.element_type,
            dims=copy(unsafe_wrap(Vector{Int64}, spec.dims, spec.num_dims)),
        )
    end

    num_layouts = unsafe_load(args.num_device_layouts)
    if num_layouts != 0 && num_layouts != length(specs)
        throw(
            invalid_argument("got $num_layouts device layouts for $(length(specs)) buffers")
        )
    end
    c_layouts = unsafe_load(args.device_layouts)
    layouts = Vector{Union{Nothing,MemoryLayout}}(undef, num_layouts)
    for i in eachindex(layouts)
        c_layout = unsafe_load(c_layouts, i)
        layouts[i] = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)
    end

    manager = create_buffers_for_async_host_to_device(
        client, specs, layouts, mem
    )::AsyncHostToDeviceTransferManager
    root!(manager)

    unsafe_store!(
        args.transfer_manager,
        Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager}(pointer_from_objref(manager)),
    )
    return ok()
end
