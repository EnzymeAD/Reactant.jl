# Buffer.jl — MTLBuffer-based buffer PJRT callbacks
#
# The PJRT buffer handle IS the retained MTLBuffer ObjC id (UInt64).
# Metadata (dims, dtype) is cached in Julia-side Dicts keyed by handle.
# Apple manages buffer memory via ObjC retain/release.

# Type conversion helpers — delegate to Reactant's canonical implementations
pjrt_type_to_julia(t::UInt32) = Reactant.XLA.julia_type(Int64(t))
julia_type_to_pjrt(T) = UInt32(Reactant.XLA.primitive_type(T))

const _ObjC = Metal.MTL.ObjectiveC

# ============================================================
# Buffer metadata caches
# ============================================================
# PJRT_Buffer_Dimensions needs a stable Ptr{Int64} for the buffer's lifetime.
# PJRT_Buffer_ElementType needs the dtype. Both are cached here,
# added on buffer creation and removed on destroy.

const _BUFFER_DIMS_CACHE = Dict{UInt64,Vector{Int64}}()
const _BUFFER_DTYPE_CACHE = Dict{UInt64,UInt32}()

# ============================================================
# MTLBuffer helpers
# ============================================================

"""
    create_mtl_buffer(dev, host_data_ptr, dims, pjrt_type, nbytes) -> MTLBuffer

Allocate a shared MTLBuffer, copy host data into it, and retain it.
Returns the MTLBuffer (caller must cache dims/dtype separately).
"""
function create_mtl_buffer(
    dev, host_data_ptr::Ptr{UInt8}, dims::Vector{Int64}, pjrt_type::UInt32, nbytes::Int
)
    mtl_buf = Metal.alloc(dev, max(nbytes, 1); storage=Metal.SharedStorage)
    gpu_ptr = contents(mtl_buf)
    if nbytes > 0
        unsafe_copyto!(Ptr{UInt8}(gpu_ptr), host_data_ptr, nbytes)
    end
    _ObjC.Foundation.retain(mtl_buf)
    return mtl_buf
end

"""
    destroy_mtl_buffer(raw_id::UInt64)

Free and release an MTLBuffer from its raw ObjC id.
"""
function destroy_mtl_buffer(raw_id::UInt64)
    mtl_buf = Metal.MTL.MTLBufferInstance(
        _ObjC.id{Metal.MTL.MTLBuffer}(raw_id)
    )
    Metal.free(mtl_buf)
    _ObjC.Foundation.release(mtl_buf)
    return nothing
end

# ============================================================
# Buffer callbacks
# ============================================================

function _client_buffer_from_host(
    args::Ptr{CAPI.PJRT_Client_BufferFromHostBuffer_Args}
)::Ptr{Cvoid}
    data_ptr = Ptr{UInt8}(Reactant.unsafe_load_field(args, Val{:data}()))
    type_val = UInt32(Reactant.unsafe_load_field(args, Val{:type}()))
    dims_ptr = Reactant.unsafe_load_field(args, Val{:dims}())
    num_dims = Int(Reactant.unsafe_load_field(args, Val{:num_dims}()))

    dims = Int64[unsafe_load(dims_ptr, i) for i in 1:num_dims]
    julia_dtype = pjrt_type_to_julia(type_val)
    n_elems = num_dims > 0 ? Int(prod(dims)) : 1
    nbytes = n_elems * sizeof(julia_dtype)

    dev = Metal.device()
    mtl_buf = create_mtl_buffer(dev, data_ptr, dims, type_val, nbytes)
    handle = UInt64(pointer(mtl_buf))
    _BUFFER_DIMS_CACHE[handle] = dims
    _BUFFER_DTYPE_CACHE[handle] = type_val

    Reactant.unsafe_store_field!(
        args, Ptr{CAPI.PJRT_Event}(C_NULL), Val{:done_with_host_buffer}()
    )
    Reactant.unsafe_store_field!(args, Ptr{CAPI.PJRT_Buffer}(handle), Val{:buffer}())
    return C_NULL
end

function _buffer_destroy(args::Ptr{CAPI.PJRT_Buffer_Destroy_Args})::Ptr{Cvoid}
    handle = Ptr{Cvoid}(Reactant.unsafe_load_field(args, Val{:buffer}()))
    raw_id = UInt64(handle)
    delete!(_BUFFER_DIMS_CACHE, raw_id)
    delete!(_BUFFER_DTYPE_CACHE, raw_id)
    destroy_mtl_buffer(raw_id)
    return C_NULL
end

function _buffer_element_type(args::Ptr{CAPI.PJRT_Buffer_ElementType_Args})::Ptr{Cvoid}
    handle = Ptr{Cvoid}(Reactant.unsafe_load_field(args, Val{:buffer}()))
    raw_id = UInt64(handle)
    pjrt_type = _BUFFER_DTYPE_CACHE[raw_id]
    Reactant.unsafe_store_field!(args, CAPI.PJRT_Buffer_Type(pjrt_type), Val{:type}())
    return C_NULL
end

function _buffer_dimensions(args::Ptr{CAPI.PJRT_Buffer_Dimensions_Args})::Ptr{Cvoid}
    handle = Ptr{Cvoid}(Reactant.unsafe_load_field(args, Val{:buffer}()))
    raw_id = UInt64(handle)
    dims = _BUFFER_DIMS_CACHE[raw_id]
    Reactant.unsafe_store_field!(args, pointer(dims), Val{:dims}())
    Reactant.unsafe_store_field!(args, Csize_t(length(dims)), Val{:num_dims}())
    return C_NULL
end

function _buffer_on_device_size(
    args::Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args}
)::Ptr{Cvoid}
    handle = Ptr{Cvoid}(Reactant.unsafe_load_field(args, Val{:buffer}()))
    raw_id = UInt64(handle)
    dims = _BUFFER_DIMS_CACHE[raw_id]
    pjrt_type = _BUFFER_DTYPE_CACHE[raw_id]
    julia_dtype = pjrt_type_to_julia(pjrt_type)
    n_elems = length(dims) > 0 ? prod(dims) : 1
    nbytes = Int(n_elems * sizeof(julia_dtype))
    Reactant.unsafe_store_field!(
        args, Csize_t(nbytes), Val{:on_device_size_in_bytes}()
    )
    return C_NULL
end

function _buffer_device(args::Ptr{CAPI.PJRT_Buffer_Device_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, DEVICE_HANDLE, Val{:device}())
    return C_NULL
end

function _buffer_memory(args::Ptr{CAPI.PJRT_Buffer_Memory_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, MEMORY_HANDLE, Val{:memory}())
    return C_NULL
end

function _buffer_is_on_cpu(args::Ptr{CAPI.PJRT_Buffer_IsOnCpu_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, false, Val{:is_on_cpu}())
    return C_NULL
end

function _buffer_ready_event(args::Ptr{CAPI.PJRT_Buffer_ReadyEvent_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, READY_EVENT_HANDLE, Val{:event}())
    return C_NULL
end

function _buffer_to_host(args::Ptr{CAPI.PJRT_Buffer_ToHostBuffer_Args})::Ptr{Cvoid}
    handle = Ptr{Cvoid}(Reactant.unsafe_load_field(args, Val{:src}()))
    dst_ptr = Ptr{UInt8}(Reactant.unsafe_load_field(args, Val{:dst}()))
    dst_size = Int(Reactant.unsafe_load_field(args, Val{:dst_size}()))

    if dst_ptr != C_NULL
        raw_id = UInt64(handle)
        mtl_buf = Metal.MTL.MTLBufferInstance(
            _ObjC.id{Metal.MTL.MTLBuffer}(raw_id)
        )
        data_ptr = Ptr{UInt8}(contents(mtl_buf))
        dims = _BUFFER_DIMS_CACHE[raw_id]
        pjrt_type = _BUFFER_DTYPE_CACHE[raw_id]
        julia_dtype = pjrt_type_to_julia(pjrt_type)
        n_elems = length(dims) > 0 ? prod(dims) : 1
        buf_nbytes = Int(n_elems * sizeof(julia_dtype))
        nbytes = min(dst_size, buf_nbytes)
        unsafe_copyto!(dst_ptr, data_ptr, nbytes)
    end

    Reactant.unsafe_store_field!(
        args, Ptr{CAPI.PJRT_Event}(READY_EVENT_HANDLE), Val{:event}()
    )
    return C_NULL
end
