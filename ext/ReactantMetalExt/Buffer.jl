# Buffer.jl — MTLTensor-based buffer PJRT callbacks
#
# The PJRT buffer handle IS the retained MTLTensor ObjC id (UInt64).
# All buffer lifecycle callbacks reconstruct the MTLTensor from the handle.
# Apple manages tensor/buffer memory via ObjC retain/release (zero Libc.malloc).

# Type conversion helpers — delegate to Reactant's canonical implementations
pjrt_type_to_julia(t::UInt32) = Reactant.XLA.julia_type(Int64(t))
julia_type_to_pjrt(T) = UInt32(Reactant.XLA.primitive_type(T))

# ============================================================
# MTLTensor helpers (macOS 26+)
# ============================================================
#
# Wraps @objc calls for MTLTensor creation, destruction, and property access.
# Used by PJRT buffer callbacks to manage buffer handles.
# Apple manages tensor/buffer memory via ObjC retain/release (zero manual alloc).

const _ObjC = Metal.MTL.ObjectiveC

# --- dtype conversion: PJRT ↔ MTLTensorDataType ---

const _PJRT_TO_MTL_DTYPE = Dict{UInt32,MTLTensorDataType}(
    julia_type_to_pjrt(Float32) => MTLTensorDataTypeFloat32,
    julia_type_to_pjrt(Float16) => MTLTensorDataTypeFloat16,
    julia_type_to_pjrt(Int8)    => MTLTensorDataTypeInt8,
    julia_type_to_pjrt(UInt8)   => MTLTensorDataTypeUInt8,
    julia_type_to_pjrt(Int16)   => MTLTensorDataTypeInt16,
    julia_type_to_pjrt(UInt16)  => MTLTensorDataTypeUInt16,
    julia_type_to_pjrt(Int32)   => MTLTensorDataTypeInt32,
    julia_type_to_pjrt(UInt32)  => MTLTensorDataTypeUInt32,
)

# BFloat16 support (Julia >= 1.11)
@static if isdefined(Core, :BFloat16)
    _PJRT_TO_MTL_DTYPE[julia_type_to_pjrt(Core.BFloat16)] = MTLTensorDataTypeBFloat16
end

const _MTL_DTYPE_TO_PJRT = Dict{MTLTensorDataType,UInt32}(v => k for (k, v) in _PJRT_TO_MTL_DTYPE)

pjrt_type_to_mtl_tensor_dtype(pjrt_type::UInt32) = _PJRT_TO_MTL_DTYPE[pjrt_type]
mtl_tensor_dtype_to_pjrt_type(mtl_dtype::MTLTensorDataType) = _MTL_DTYPE_TO_PJRT[mtl_dtype]

# --- internal: create MTLTensorExtents from a dims vector ---

function _make_extents(dims::Vector{Int64})
    ext_raw = @objc [MTLTensorExtents alloc]::id{MTLTensorExtents}
    rk = Int64(length(dims))
    vp = pointer(dims)
    ext_id = @objc [ext_raw::id{MTLTensorExtents} initWithRank:(rk::Int64) values:(vp::Ptr{Int64})]::id{MTLTensorExtents}
    return MTLTensorExtents(ext_id)
end

# --- internal: column-major byte strides ---

function _col_major_byte_strides(dims::Vector{Int64}, elem_size::Int)
    n = length(dims)
    strides = Vector{Int64}(undef, n)
    if n == 0
        return strides
    end
    strides[1] = Int64(elem_size)
    for i in 2:n
        strides[i] = strides[i - 1] * dims[i - 1]
    end
    return strides
end

# --- _wrap_buffer_as_tensor: wrap existing MTLBuffer in retained MTLTensor ---

function _wrap_buffer_as_tensor(
    dev, mtl_buf, dims::Vector{Int64}, pjrt_type::UInt32
)
    julia_dtype = pjrt_type_to_julia(pjrt_type)
    strides = _col_major_byte_strides(dims, sizeof(julia_dtype))
    mtl_dtype = pjrt_type_to_mtl_tensor_dtype(pjrt_type)

    strides_ext = _make_extents(strides)

    desc_raw = @objc [MTLTensorDescriptor alloc]::id{MTLTensorDescriptor}
    desc_id = @objc [desc_raw::id{MTLTensorDescriptor} init]::id{MTLTensorDescriptor}
    desc = MTLTensorDescriptor(desc_id)
    desc.dimensions = _make_extents(dims)
    desc.strides = strides_ext
    desc.dataType = mtl_dtype
    desc.storageMode = MTLStorageModeShared
    desc.usage = MTLTensorUsageCompute

    err_ref = Ref{Ptr{Cvoid}}(C_NULL)
    off = UInt64(0)
    tid = @objc [dev::id{MTLDevice} newTensorWithBuffer:(mtl_buf::id{MTLBuffer}) descriptor:(desc::id{MTLTensorDescriptor}) offset:(off::UInt64) strides:(strides_ext::id{MTLTensorExtents}) error:(err_ref::Ptr{Ptr{Cvoid}})]::id{MTLTensor}
    tid == nil && error("Failed to create MTLTensor")
    tensor = MTLTensor(tid)

    _ObjC.Foundation.retain(tensor)
    return tensor
end

# --- create_mtl_tensor: allocate buffer, copy host data, wrap in MTLTensor ---

function create_mtl_tensor(
    dev, host_data_ptr::Ptr{UInt8}, dims::Vector{Int64}, pjrt_type::UInt32, nbytes::Int
)
    mtl_buf = Metal.alloc(dev, max(nbytes, 1); storage=Metal.SharedStorage)
    gpu_ptr = contents(mtl_buf)
    if nbytes > 0
        unsafe_copyto!(Ptr{UInt8}(gpu_ptr), host_data_ptr, nbytes)
    end
    return _wrap_buffer_as_tensor(dev, mtl_buf, dims, pjrt_type)
end

# --- destroy_mtl_tensor: free buffer + release tensor ---

function destroy_mtl_tensor(raw_id::UInt64)
    tensor = MTLTensor(id{MTLTensor}(raw_id))
    buf = tensor.buffer
    if buf !== nothing
        Metal.free(buf)
    end
    _ObjC.Foundation.release(tensor)
    return nothing
end

# --- query helpers (reconstruct tensor from raw ObjC id) ---

function mtl_tensor_dims(tensor)
    ext = tensor.dimensions
    r = Int(ext.rank)
    return [let idx = UInt(i)
        Int64(@objc [ext::id{MTLTensorExtents} extentAtDimensionIndex:(idx::NSUInteger)]::NSUInteger)
    end for i in 0:(r - 1)]
end

function mtl_tensor_dtype(tensor)
    return mtl_tensor_dtype_to_pjrt_type(tensor.dataType)
end

function mtl_tensor_data_ptr(tensor)
    return Ptr{Cvoid}(contents(tensor.buffer))
end

function mtl_tensor_nbytes(tensor)
    dims = mtl_tensor_dims(tensor)
    pjrt_type = mtl_tensor_dtype(tensor)
    julia_dtype = pjrt_type_to_julia(pjrt_type)
    n_elems = length(dims) > 0 ? prod(dims) : 1
    return Int(n_elems * sizeof(julia_dtype))
end

# ============================================================
# Buffer dims cache
# ============================================================
# PJRT_Buffer_Dimensions needs a stable Ptr{Int64} for the buffer's lifetime.
# MTLTensorExtents doesn't expose a contiguous Int64 pointer, so we cache dims
# in a Dict. Entries are added on buffer creation and removed on destroy.

const _BUFFER_DIMS_CACHE = Dict{UInt64,Vector{Int64}}()

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

    dev = Metal.current_device()
    tensor = create_mtl_tensor(dev, data_ptr, dims, type_val, nbytes)
    handle = UInt64(pointer(tensor))
    _BUFFER_DIMS_CACHE[handle] = dims

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
    destroy_mtl_tensor(raw_id)
    return C_NULL
end

function _buffer_element_type(args::Ptr{CAPI.PJRT_Buffer_ElementType_Args})::Ptr{Cvoid}
    handle = Ptr{Cvoid}(Reactant.unsafe_load_field(args, Val{:buffer}()))
    raw_id = UInt64(handle)
    tensor = MTLTensor(id{MTLTensor}(raw_id))
    pjrt_type = mtl_tensor_dtype(tensor)
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
    tensor = MTLTensor(id{MTLTensor}(raw_id))
    nbytes = mtl_tensor_nbytes(tensor)
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
        tensor = MTLTensor(id{MTLTensor}(raw_id))
        data_ptr = mtl_tensor_data_ptr(tensor)
        nbytes = min(dst_size, mtl_tensor_nbytes(tensor))
        unsafe_copyto!(dst_ptr, Ptr{UInt8}(data_ptr), nbytes)
    end

    Reactant.unsafe_store_field!(
        args, Ptr{CAPI.PJRT_Event}(READY_EVENT_HANDLE), Val{:event}()
    )
    return C_NULL
end
