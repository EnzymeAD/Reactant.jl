using Reactant,PyCall,Revise,DLPack,CUDA
const REACTANT_POOL = IdDict{WeakRef, Any}()

function DLPack.unsafe_share(
    A_reactant::Reactant.AnyConcretePJRTArray{T,N}; stream=nothing
) where {T,N}
    # note we don't keep A saved anywhere so if it gets deleted the capsule will crash. Likewise we don't set the deleter here.
    data_ptr = Reactant.XLA.unsafe_buffer_pointer(first(A_reactant.data).buffer)
    platform_name = Reactant.XLA.platform_name(Reactant.XLA.client(A_reactant))
    device_id = Reactant.XLA.device_ordinal(Reactant.XLA.device(A_reactant))
    if platform_name == "cuda"
        if !isnothing(stream)
            println("i haven't tested streams too well")
            @assert isa(stream, CUDA.CuStream) "stream must be a CUDA.CuStream"
            (@ccall Reactant.MLIR.API.mlir_c.WaitUntilBufferReadyOnStream(
                first(A_reactant.data).buffer.buffer::Ptr{Nothing},
                stream.handle::Ptr{Nothing},
            )::Ptr{Nothing})
        else
            Reactant.synchronize(A_reactant)
        end
        DL_device = DLPack.DLDevice(DLPack.kDLCUDA, Cint(device_id))
    elseif platform_name == "cpu"
        Reactant.synchronize(A_reactant)
        DL_device = DLPack.DLDevice(DLPack.kDLCPU, Cint(0))
    else
        @assert false "$platform_name not implemented"
    end

    size_ = Clonglong[(size)(A_reactant)...]
    # size_ = Clonglong[(reverse ∘ size)(A_reactant)...]
    strides_ = Clonglong[(strides)(A_reactant)...]
    # strides_ = Clonglong[(reverse ∘ strides)(A_reactant)...]
    ndim = Cint(N)
    dtype = DLPack.jltypes_to_dtypes()[T]
    size_ptr = pointer(size_)
    strides__ptr = pointer(strides_)
    dl_tensor = DLPack.DLTensor(
        data_ptr, DL_device, ndim, dtype, size_ptr, strides__ptr, Culonglong(0)
    )
    tensor = DLPack.DLManagedTensor(dl_tensor, C_NULL, C_NULL)

    return DLPack.Capsule(tensor, size_, strides_)
end
function DLPack.share(A::StridedArray, from_dlpack::PyCall.PyObject; stream=nothing)
    # I think I added this function because of the stream and because the try catch thing wasn't working with Jax
    capsule = DLPack.share(A; stream)
    tensor = capsule.tensor
    tensor_ptr = pointer_from_objref(tensor)

    # Prevent `A` and `tensor` from being `gc`ed while `o` is around.
    # For certain DLPack-compatible libraries, e.g. PyTorch, the tensor is
    # captured and the `deleter` referenced from it.
    DLPack.SHARES_POOL[tensor_ptr] = (capsule, A)
    tensor.deleter = DLPack.DELETER[]

    pycapsule = PyCall.PyObject(
        PyCall.@pycheck ccall(
            (PyCall.@pysym :PyCapsule_New),
            PyCall.PyPtr,
            (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cvoid}),
            tensor_ptr,
            DLPack.PYCAPSULE_NAME,
            C_NULL,
        )
    )
    return try
        from_dlpack(pycapsule)
    catch e
        if !(e isa KeyError && any(e.key .== (:__dlpack__, :__dlpack_device__))) &&
            !(occursin(
            "The array passed to from_dlpack must have __dlpack__ and __dlpack_device__ methods",
            string(e),
        ))
            rethrow()
        end

        dl_array = DLArray()
        ctx = DLPack.dldevice(tensor)
        dl_array.capsule = pycapsule
        dl_array.device = (Int(ctx.device_type), ctx.device_id)
        from_dlpack(dl_array)
    end
end
function DLPack.share(A::Reactant.AnyConcretePJRTArray{T,N}; stream=nothing) where {T,N}
    return DLPack.unsafe_share(A; stream=stream)
end

function to_julia(A_reactant::Reactant.AnyConcretePJRTArray{T,N}; stream=nothing) where {T,N}
    data_ptr = Reactant.XLA.unsafe_buffer_pointer(first(A_reactant.data).buffer)
    platform_name = Reactant.XLA.platform_name(Reactant.XLA.client(A_reactant))
    device_id = Reactant.XLA.device_ordinal(Reactant.XLA.device(A_reactant))

    if platform_name == "cuda"
        if !isnothing(stream)
            @assert isa(stream, CUDA.CuStream) "stream must be a CUDA.CuStream"
            (@ccall Reactant.MLIR.API.mlir_c.WaitUntilBufferReadyOnStream(
                first(A_reactant.data).buffer.buffer::Ptr{Nothing},
                stream.handle::Ptr{Nothing},
            )::Ptr{Nothing})
        else
            Reactant.synchronize(A_reactant)
        end
        addr = Int(data_ptr)
        array = DLPack.unsafe_wrap(CUDA.CuArray, CUDA.CuPtr{T}(addr), size(A_reactant))
    elseif platform_name == "cpu"
        Reactant.synchronize(A_reactant)
        addr = Int(data_ptr)
        array = unsafe_wrap(Array, Ptr{T}(addr), size(A_reactant))
    else
        @assert false "$platform_name not implemented"
    end
    wk_ref = WeakRef(array)
    REACTANT_POOL[wk_ref] = A_reactant
    finalizer(_ -> let
        delete!(REACTANT_POOL, wk_ref)
    end, array)
    return array
end
function strides_to_minor_to_major(dims, strides)
    @assert length(dims) == length(strides)

    minor_to_major = collect(eachindex(dims)) .- 1

    # Sort by stride ascending; break ties by choosing the *larger* axis index first
    sort!(minor_to_major; by=i -> (strides[i + 1], -i))

    # Validate compact striding
    stride = 1
    for d in minor_to_major
        dim = dims[d + 1]
        if dim > 1 && strides[d + 1] != stride
            dims_str = join(dims, ",")
            strides_str = join(strides, ",")
            throw(
                ErrorException(
                    "Only DLPack tensors with trivial (compact) striding are supported; " *
                    "i.e. tensors whose striding represents a transposition of the underlying buffer " *
                    "but not broadcasting. Dimensions were: [$dims_str], strides were: [$strides_str].",
                ),
            )
        end
        stride *= dim
    end

    return minor_to_major
end

function julia_array_to_reactant_client(A::StridedArray)
    return Reactant.XLA.global_backend_state.clients["cpu"]
end
function julia_array_to_reactant_client(A::CUDA.CuArray)
    return Reactant.XLA.global_backend_state.clients["cuda"]
end
function verify_alignment(A::StridedArray)
    let
        min_alignment = @ccall Reactant.MLIR.API.mlir_c.CpuMinAlignment()::Int
        return (Int(Ptr{Cvoid}(UInt(pointer(A)))) % min_alignment) == 0
    end
end
verify_alignment(A::CUDA.CuArray) = true
verify_pointer_alignment(A::CuPtr) = true
function verify_pointer_alignment(A::Ptr)
    let
        min_alignment = @ccall Reactant.MLIR.API.mlir_c.CpuMinAlignment()::Int
        return (Int(Ptr{Cvoid}(UInt(A))) % min_alignment) == zero(UInt)
    end
end
function data_ptr_to_view_buffer(
    data_ptr::Union{Ptr{T},CuPtr{T}}, device,client, shape, strides_; stream=C_NULL
) where {T}
    ndims = length(shape)
    
    xla_type = Reactant.XLA.primitive_type(T)
    if data_ptr isa Ptr
        @assert verify_pointer_alignment(data_ptr) "CPU data is not aligned; copy the array
        I think larger arrays might be guaranteed to aligned to 64 bits but smaller ones not.
        Also I read somewhere that XLA might accept 32 bits aligned arrays but I lost where
        any((reinterpret(UInt, pointer(Vector{Float64}(undef, 252))) % 64) ≠ zero(UInt) for i ∈ 1:1000)
        false
        any((reinterpret(UInt, pointer(Vector{Float64}(undef, 251))) % 64) ≠ zero(UInt) for i ∈ 1:1000)
        true
"
        return GC.@preserve shape begin
            @ccall Reactant.MLIR.API.mlir_c.MutableZeroCopyBufferFromHostBuffer(
                client.client::Ptr{Cvoid},
                data_ptr::Ptr{T}, #data
                Reactant.XLA.primitive_type(T)::UInt64, #type
                ndims::Csize_t, #dim
                pointer(shape)::Ptr{Int64}, #cshape
                device.device::Ptr{Cvoid},
            )::Ptr{Cvoid}
        end
    end
    data_ptr_cvoid = Ptr{Cvoid}(UInt(data_ptr))
    
    minor_to_major = strides_to_minor_to_major(shape, strides_)
    # println(minor_to_major)
    GC.@preserve shape minor_to_major begin
        buffer_ptr = @ccall Reactant.MLIR.API.mlir_c.DataPointerToViewBuffer(
            data_ptr_cvoid::Ptr{Cvoid},
            device.device::Ptr{Cvoid},
            xla_type::UInt64,
            pointer(shape)::Ptr{Int64},
            pointer(minor_to_major)::Ptr{Int64},
            ndims::Csize_t,
            stream::Clonglong,
        )::Ptr{Nothing}
        
    end
    return buffer_ptr
end
function MutableZeroCopyBufferFromHostBuffer(array::Array{T,N}) where {T,N}
    sizear = collect(Int64, reverse(size(array)))
    client = julia_array_to_reactant_client(array)
    device = Reactant.XLA.default_device(client)
    # https://github.com/openxla/xla/blob/af7ecacb3e6dd6d8dde0adf5aa181735283181fa/xla/pjrt/cpu/cpu_client.cc#L1435
    # https://github.com/openxla/xla/blob/580849e0a1fbc28db73add45e270b9d4176f18dc/xla/pjrt/c/pjrt_c_api.h#L918C3-L918C44
    buffer = GC.@preserve array sizear begin
        @ccall Reactant.MLIR.API.mlir_c.MutableZeroCopyBufferFromHostBuffer(
            client.client::Ptr{Cvoid},
            pointer(array)::Ptr{T}, #data
            Reactant.XLA.primitive_type(T)::UInt64, #type
            N::Csize_t, #dim
            pointer(sizear)::Ptr{Int64}, #cshape
            device.device::Ptr{Cvoid},
        )::Ptr{Cvoid}
    end
    async_buffer = Reactant.XLA.PJRT.AsyncBuffer(Reactant.XLA.PJRT.Buffer(buffer), nothing)
    arrayr = Reactant.ConcretePJRTArray{T,N}((async_buffer,), (size(array)))

    return arrayr
end
function array_to_view_rarray(A::StridedArray{T,N}; stream=C_NULL) where {T,N}
    client = julia_array_to_reactant_client(A)
    data_ptr = pointer(A)
    device = Reactant.XLA.default_device(client)
    shape = [reverse(size(A))...]
    strides_ = [reverse(strides(A))...]
    GC.@preserve A begin
        buffer_ptr = data_ptr_to_view_buffer(data_ptr, device, client, shape, strides_; stream)
    end
    async_buffer = Reactant.XLA.PJRT.AsyncBuffer(
        Reactant.XLA.PJRT.Buffer(buffer_ptr), nothing
    )
    array = Reactant.ConcretePJRTArray{T,N}((async_buffer,), (size(A)))
    return array
end
function from_julia(A::StridedArray; stream=C_NULL)
    array = array_to_view_rarray(A; stream)
    wkref = WeakRef(array)
    DLPack.WRAPS_POOL[wkref] = A
    finalizer(_ -> delete!(DLPack.WRAPS_POOL, wkref), array)
    return array
end

function Base.unsafe_wrap(
    ::Type{A}, managed_tensor::DLPack.DLManagedTensor, foreign; copy_=false, stream=0
) where {A<:Reactant.AnyConcretePJRTArray}
    N = Int(managed_tensor.dl_tensor.ndim)
    T = DLPack.dtypes_to_jltypes()[managed_tensor.dl_tensor.dtype]    
    typed_manager = DLPack.DLManager(managed_tensor)
    array = unsafe_wrap(Reactant.ConcretePJRTArray{T,N}, typed_manager; copy_, stream)
    wkref = WeakRef(array)
    DLPack.WRAPS_POOL[wkref] = (foreign)
    finalizer(_ -> delete!(DLPack.WRAPS_POOL, wkref), Reactant.ancestor(array))
    return array
end
function Base.unsafe_wrap(
    ::Type{<:Reactant.AnyConcretePJRTArray{T,N}},
    manager::DLPack.DLManager{T};
    copy_=false,
    stream=0,
) where {T,N}
    @assert !copy_ "friendly reminder to myself to implement this"
    managed_tensor = manager.manager
    # @assert !DLPack.is_col_major(managed_tensor, Val(N)) "Reactant assumes RowMajor, use Base.unsafe_wrap(::Type{A},managed_tensor::DLManagedTensor, foreign)"

    (; device_type, device_id) = managed_tensor.dl_tensor.ctx
    # device_type = DLPack
    device_type == DLPack.kDLCUDA
    client = if device_type == DLPack.kDLCPU
        data_ptr = Ptr{T}(managed_tensor.dl_tensor.data)
        Reactant.XLA.global_backend_state.clients["cpu"]
    elseif device_type == DLPack.kDLCUDA
        data_ptr = CuPtr{T}(UInt(managed_tensor.dl_tensor.data))
        Reactant.XLA.global_backend_state.clients["cuda"]

    elseif device_type == DLPack.kDLMetal
        an_error
        # Reactant.XLA.global_backend_state.clients["metal"]
    elseif device_type == DLPack.kDLExtDev
        # Reactant.XLA.global_backend_state.clients["tpu"]
        another_error
    else
        what
    end

    idx = nothing
    if idx isa Nothing
        device = Reactant.XLA.default_device(client)
    else
        device = Reactant.XLA.get_device(client, idx)
    end
    size_ = unsafe_wrap(Array, managed_tensor.dl_tensor.shape, N)
    strides_ = unsafe_wrap(Array, managed_tensor.dl_tensor.strides, N)
    
    column_major =
        length(size_) > 1 &&
        unsafe_load(
            Base.unsafe_convert(Ptr{NTuple{N,Int64}}, managed_tensor.dl_tensor.strides)
        ) == Base.size_to_strides(1, size_...)

    # if reverse with row major and remove the permutedimsarray after we get the "incorrect" reshape(xr,:)
    # but with permutedimsarray theres the allow scalar problem
    if column_major
        strides_ = reverse(strides_)
        size_ = reverse(size_)
    end
    GC.@preserve managed_tensor begin
        buffer_ptr = data_ptr_to_view_buffer(
            data_ptr, device, client, size_, strides_; stream=C_NULL
        )
        # managed tensor has pointers to stride and shape which are used to create the XLA buffer. A reference to the managed tensor is kept by calling Base.unsafe_wrap(::Type{A},managed_tensor::DLManagedTensor, foreign) but not here
        # buffer_ptr = @ccall Reactant.MLIR.API.mlir_c.DLManagedTensorToBufferInternal(device.device::Ptr{Nothing}, Ref(managed_tensor)::Ref{DLPack.DLManagedTensor},copy_::Bool,stream::Clonglong)::Ptr{Nothing} 
        asyncbuffer = Reactant.XLA.PJRT.AsyncBuffer(
            Reactant.XLA.PJRT.Buffer(buffer_ptr), nothing
        )
    end
    
    if column_major
        array = (Reactant.ConcretePJRTArray{T,N}(
            (asyncbuffer,), ntuple(i -> size_[N - i + 1], Val(N));
        ))
    else
        
        array = PermutedDimsArray(
            Reactant.ConcretePJRTArray{T,N}(
                (asyncbuffer,), ntuple(i -> size_[N - i + 1], Val(N));
            ),
            ntuple(i -> N - i + 1, Val(N)),
        )
        

    end
    return array
end
function DLPack.from_dlpack(::Type{A}, o::PyCall.PyObject) where {A<:AbstractArray}
    tensor = DLPack.DLManagedTensor(PyCall.@pycall o.__dlpack__()::PyCall.PyObject)
    return DLPack.unsafe_wrap(A, tensor, o)
end
