# Buffer.jl — MTLBuffer-based buffer PJRT callbacks
#
# The PJRT buffer handle is a Libc.malloc'd MetalBufferMeta struct that owns a
# retained MTLBuffer ObjC id plus the metadata (dims, dtype) PJRT queries need.
# We control alloc/free: the struct is malloc'd on buffer creation and freed
# (together with the MTLBuffer release) on PJRT_Buffer_Destroy — no Julia-side
# registries and no GC involvement.

# Type conversion helpers — delegate to Reactant's canonical implementations
pjrt_type_to_julia(t::UInt32) = Reactant.XLA.julia_type(Int64(t))
julia_type_to_pjrt(T) = UInt32(Reactant.XLA.primitive_type(T))

const _ObjC = Metal.MTL.ObjectiveC

# ============================================================
# Buffer handle struct
# ============================================================

# XLA supports ranks well below this; fixed-size inline dims keep the struct
# POD so PJRT_Buffer_Dimensions can hand out a stable interior pointer.
const METAL_BUFFER_MAX_RANK = 16

struct MetalBufferMeta
    mtl_id::UInt64    # retained MTLBuffer ObjC id
    dtype::UInt32     # PJRT_Buffer_Type
    ndims::Int32
    dims::NTuple{METAL_BUFFER_MAX_RANK,Int64}
end

"""
    new_buffer_handle(mtl_buf, dims, pjrt_type) -> Ptr{Cvoid}

Allocate a `MetalBufferMeta` with `Libc.malloc` describing an already-retained
MTLBuffer and return it as the opaque PJRT buffer handle.
"""
function new_buffer_handle(
    mtl_buf::Metal.MTL.MTLBuffer, dims::Vector{Int64}, pjrt_type::UInt32
)
    ndims = length(dims)
    ndims <= METAL_BUFFER_MAX_RANK ||
        error("Buffer rank $ndims exceeds METAL_BUFFER_MAX_RANK")
    meta = Ptr{MetalBufferMeta}(Libc.malloc(sizeof(MetalBufferMeta)))
    Reactant.unsafe_store_field!(meta, UInt64(pointer(mtl_buf)), Val{:mtl_id}())
    Reactant.unsafe_store_field!(meta, pjrt_type, Val{:dtype}())
    Reactant.unsafe_store_field!(meta, Int32(ndims), Val{:ndims}())
    dims_ptr = Ptr{Int64}(UInt(meta) + fieldoffset(MetalBufferMeta, 4))
    for i in 1:ndims
        unsafe_store!(dims_ptr, dims[i], i)
    end
    return Ptr{Cvoid}(meta)
end

"""
    handle_mtl_buffer(handle) -> MTLBuffer

Reconstruct the retained MTLBuffer from a PJRT buffer handle.
"""
function handle_mtl_buffer(handle::Ptr{Cvoid})
    mtl_id = Reactant.unsafe_load_field(Ptr{MetalBufferMeta}(handle), Val{:mtl_id}())
    return Metal.MTL.MTLBuffer(_ObjC.id{Metal.MTL.MTLBuffer}(mtl_id))
end

function handle_dims(handle::Ptr{Cvoid})
    meta = Ptr{MetalBufferMeta}(handle)
    ndims = Int(Reactant.unsafe_load_field(meta, Val{:ndims}()))
    dims_ptr = Ptr{Int64}(UInt(meta) + fieldoffset(MetalBufferMeta, 4))
    return Int64[unsafe_load(dims_ptr, i) for i in 1:ndims]
end

function handle_dtype(handle::Ptr{Cvoid})
    return Reactant.unsafe_load_field(Ptr{MetalBufferMeta}(handle), Val{:dtype}())
end

function handle_nbytes(handle::Ptr{Cvoid})
    dims = handle_dims(handle)
    julia_dtype = pjrt_type_to_julia(handle_dtype(handle))
    n_elems = isempty(dims) ? 1 : prod(dims)
    return Int(n_elems * sizeof(julia_dtype))
end

# ============================================================
# MTLBuffer helpers
# ============================================================

"""
    create_mtl_buffer(dev, host_data_ptr, dims, pjrt_type, nbytes) -> MTLBuffer

Allocate a shared MTLBuffer, copy host data into it, and retain it.
Returns the MTLBuffer (wrap it with [`new_buffer_handle`](@ref) to get a PJRT handle).
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
    mtl_buf = Metal.MTL.MTLBuffer(
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
    handle = new_buffer_handle(mtl_buf, dims, type_val)

    Reactant.unsafe_store_field!(
        args, Ptr{CAPI.PJRT_Event}(C_NULL), Val{:done_with_host_buffer}()
    )
    Reactant.unsafe_store_field!(args, Ptr{CAPI.PJRT_Buffer}(handle), Val{:buffer}())
    return C_NULL
end

function _buffer_destroy(args::Ptr{CAPI.PJRT_Buffer_Destroy_Args})::Ptr{Cvoid}
    handle = Ptr{Cvoid}(Reactant.unsafe_load_field(args, Val{:buffer}()))
    mtl_id = Reactant.unsafe_load_field(Ptr{MetalBufferMeta}(handle), Val{:mtl_id}())
    destroy_mtl_buffer(mtl_id)
    Libc.free(handle)
    return C_NULL
end

function _buffer_element_type(args::Ptr{CAPI.PJRT_Buffer_ElementType_Args})::Ptr{Cvoid}
    handle = Ptr{Cvoid}(Reactant.unsafe_load_field(args, Val{:buffer}()))
    Reactant.unsafe_store_field!(
        args, CAPI.PJRT_Buffer_Type(handle_dtype(handle)), Val{:type}()
    )
    return C_NULL
end

function _buffer_dimensions(args::Ptr{CAPI.PJRT_Buffer_Dimensions_Args})::Ptr{Cvoid}
    handle = Ptr{Cvoid}(Reactant.unsafe_load_field(args, Val{:buffer}()))
    meta = Ptr{MetalBufferMeta}(handle)
    ndims = Int(Reactant.unsafe_load_field(meta, Val{:ndims}()))
    dims_ptr = Ptr{Int64}(UInt(meta) + fieldoffset(MetalBufferMeta, 4))
    Reactant.unsafe_store_field!(args, dims_ptr, Val{:dims}())
    Reactant.unsafe_store_field!(args, Csize_t(ndims), Val{:num_dims}())
    return C_NULL
end

function _buffer_on_device_size(
    args::Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args}
)::Ptr{Cvoid}
    handle = Ptr{Cvoid}(Reactant.unsafe_load_field(args, Val{:buffer}()))
    Reactant.unsafe_store_field!(
        args, Csize_t(handle_nbytes(handle)), Val{:on_device_size_in_bytes}()
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
        mtl_buf = handle_mtl_buffer(handle)
        data_ptr = Ptr{UInt8}(contents(mtl_buf))
        nbytes = min(dst_size, handle_nbytes(handle))
        unsafe_copyto!(dst_ptr, data_ptr, nbytes)
    end

    Reactant.unsafe_store_field!(
        args, Ptr{CAPI.PJRT_Event}(READY_EVENT_HANDLE), Val{:event}()
    )
    return C_NULL
end
