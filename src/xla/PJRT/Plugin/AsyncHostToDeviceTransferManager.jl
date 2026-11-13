"""
    AsyncHostToDeviceTransferManager

Manages the transfer of host data to a set of new buffers, as returned by `create_buffers_for_async_host_to_device`.

Buffers are identified by their 1-based index (C's 0-based index + 1), which the C wrapper checks to be within
`1:buffercount(manager)`. Once a buffer has received its last transfer, or an error with `set_buffer_error!`, no more
transfers into it are made.

!!! todo
Transfers are _currently_ synchronous: the events returned to C are set as soon as the transfer methods return, so
they must be done reading the host data by then.

### Interface

- `buffercount`
- `device`
- `retrieve_buffer!`
- `buffer_size`
- `transfer_data!`
- `transfer_literal!`
- `set_buffer_error!`
- `add_metadata!`
"""
abstract type AsyncHostToDeviceTransferManager end

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_destroy(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

# convert C's 0-based `buffer_index` into the 1-based index of a buffer of `manager`, checking that it exists
function checked_buffer_index(manager::AsyncHostToDeviceTransferManager, c_index::Integer)
    n = buffercount(manager)
    if !(0 <= c_index < n)
        throw(invalid_argument("buffer index $c_index is out of range for $n buffers"))
    end
    return Int(c_index) + 1
end

"""
    buffercount(manager::AsyncHostToDeviceTransferManager) -> Integer

Return the number of buffers that `manager` transfers data into.

Backs `PJRT_AsyncHostToDeviceTransferManager_BufferCount`.
"""
buffercount(::AsyncHostToDeviceTransferManager)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_buffer_count(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    unsafe_store!(args.buffer_count, buffercount(manager))
    return ok()
end

"""
    device(manager::AsyncHostToDeviceTransferManager) -> Device

Return the device that `manager` transfers data to. The device must be kept alive by its client.

Backs `PJRT_AsyncHostToDeviceTransferManager_Device`.
"""
device(::AsyncHostToDeviceTransferManager)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_device(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_Device_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    dev = device(manager)::Device
    unsafe_store!(args.device_out, Ptr{CAPI.PJRT_Device}(pointer_from_objref(dev)))
    return ok()
end

"""
    retrieve_buffer!(manager::AsyncHostToDeviceTransferManager, index::Int) -> Buffer

Return the `index`-th buffer of `manager`, handing it over to the caller. It may be retrieved before its transfers are
done, in which case it becomes ready once they are. C retrieves each buffer at most once.

The caller GC-roots the returned buffer.

Backs `PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer`.
"""
retrieve_buffer!(::AsyncHostToDeviceTransferManager, ::Int)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_retrieve_buffer(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    index = checked_buffer_index(manager, unsafe_load(args.buffer_index))
    buffer = retrieve_buffer!(manager, index)::Buffer
    root!(buffer)
    unsafe_store!(args.buffer_out, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(buffer)))
    return ok()
end

"""
    buffer_size(manager::AsyncHostToDeviceTransferManager, index::Int) -> Integer

Return the on-device size in bytes of the `index`-th buffer of `manager`.

Backs `PJRT_AsyncHostToDeviceTransferManager_BufferSize`.
"""
buffer_size(::AsyncHostToDeviceTransferManager, ::Int)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_buffer_size(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    index = checked_buffer_index(manager, unsafe_load(args.buffer_index))
    unsafe_store!(args.buffer_size, buffer_size(manager, index))
    return ok()
end

"""
    transfer_data!(manager::AsyncHostToDeviceTransferManager, index::Int, data::Ptr{Cvoid}, offset, size, is_last::Bool)

Copy `size` bytes of host memory at `data` into the `index`-th buffer of `manager`, starting at byte `offset` of its
on-device data. `data` is already laid out in the on-device format (e.g. as copied by
`Base.unsafe_copyto!(dst, buffer, offset, nbytes)`). The C wrapper checks that the range is within
`buffer_size(manager, index)`.

If `is_last`, the buffer becomes ready once the copy is done, and no more transfers into it are made.

Backs `PJRT_AsyncHostToDeviceTransferManager_TransferData`.
"""
transfer_data!(::AsyncHostToDeviceTransferManager, ::Int, args...)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_transfer_data(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    index = checked_buffer_index(manager, unsafe_load(args.buffer_index))
    offset = unsafe_load(args.offset)
    transfer_size = unsafe_load(args.transfer_size)

    dst_size = buffer_size(manager, index)
    if offset < 0 || transfer_size < 0 || offset + transfer_size > dst_size
        throw(
            invalid_argument(
                "transfer range [$offset, $(offset + transfer_size)) is out of the $dst_size bytes of the buffer",
            ),
        )
    end

    data = unsafe_load(args.data)
    transfer_data!(
        manager, index, data, offset, transfer_size, unsafe_load(args.is_last_transfer)
    )

    event = root!(Event())
    notify(event)
    unsafe_store!(
        args.done_with_h2d_transfer, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event))
    )
    return ok()
end

"""
    transfer_literal!(manager::AsyncHostToDeviceTransferManager, index::Int, data::Ptr{Cvoid}, type, dims, layout)

Copy the host array at `data`, with element type `type::CAPI.PJRT_Buffer_Type`, dimensions `dims::Vector{Int64}` and
host layout `layout::Union{Nothing,MemoryLayout}` (`nothing` for the default layout), into the whole `index`-th
buffer of `manager`, converting it to the on-device format. The buffer becomes ready once the copy is done, and no more
transfers into it are made.

Backs `PJRT_AsyncHostToDeviceTransferManager_TransferLiteral`.
"""
transfer_literal!(::AsyncHostToDeviceTransferManager, ::Int, args...)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_transfer_literal(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    index = checked_buffer_index(manager, unsafe_load(args.buffer_index))

    _eltype = unsafe_load(args.shape_element_type)
    _size = copy(
        unsafe_wrap(
            Vector{Int64}, unsafe_load(args.shape_dims), unsafe_load(args.shape_num_dims)
        ),
    )
    c_layout = unsafe_load(args.shape_layout)
    layout = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)
    transfer_literal!(manager, index, unsafe_load(args.data), _eltype, _size, layout)

    event = root!(Event())
    notify(event)
    unsafe_store!(
        args.done_with_h2d_transfer, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event))
    )
    return ok()
end

"""
    set_buffer_error!(manager::AsyncHostToDeviceTransferManager, index::Int, error::Error)

Make the `index`-th buffer of `manager` fail with `error` instead of receiving data. No more transfers into it are
made.

Backs `PJRT_AsyncHostToDeviceTransferManager_SetBufferError`.
"""
set_buffer_error!(::AsyncHostToDeviceTransferManager, ::Int, ::Error)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_set_buffer_error(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    index = checked_buffer_index(manager, unsafe_load(args.buffer_index))
    error = Error(
        unsafe_load(args.error_code),
        unsafe_load(args.error_message),
        unsafe_load(args.error_message_size),
    )
    set_buffer_error!(manager, index, error)
    return ok()
end

"""
    add_metadata!(manager::AsyncHostToDeviceTransferManager, metadata::Dict{String,String})

Attach `metadata` to the transfers of `manager`, e.g. a handle to identify them when debugging.

Backs `PJRT_AsyncHostToDeviceTransferManager_AddMetadata`.
"""
add_metadata!(::AsyncHostToDeviceTransferManager, ::Dict{String,String})

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_add_metadata(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    values = unsafe_load_pjrt_named_values(
        unsafe_load(args.transfer_metadata), unsafe_load(args.num_metadata)
    )
    metadata = Dict{String,String}()
    for (key, value) in values
        value isa String ||
            throw(invalid_argument("transfer metadata `$key` is not a string"))
        metadata[key] = value
    end
    add_metadata!(manager, metadata)
    return ok()
end
