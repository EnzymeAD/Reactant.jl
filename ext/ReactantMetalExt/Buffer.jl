# `data` has shared storage, so the host can also access it through `contents`. Like in MLIR, it's dense in
# major-to-minor order
mutable struct MetalBuffer <: Buffer
    data::Union{Nothing,MTLBuffer}
    const size_in_bytes::Int
    const type::CAPI.PJRT_Buffer_Type
    const dims::Vector{Int64}
    const device::MetalDevice
    const memory::MetalMemory

    # allocate a buffer whose contents are left uninitialized
    function MetalBuffer(type, dims, dev, mem)
        elsize = pjrt_elsize(type)
        isnothing(elsize) && throw(unimplemented("buffers of $type are not supported"))
        size_in_bytes = prod(dims; init=elsize)

        # Metal can't allocate empty buffers
        data = try
            Metal.alloc(dev.mtl, max(size_in_bytes, 1); storage=Metal.SharedStorage)
        catch err
            err isa OutOfMemoryError || rethrow()
            throw(resource_exhausted())
        end
        return Base.finalizer(delete!, new(data, size_in_bytes, type, dims, dev, mem))
    end
end

function Base.delete!(buffer::MetalBuffer)
    isdeleted(buffer) && return nothing
    Metal.free(buffer.data)
    buffer.data = nothing
    return nothing
end

isdeleted(buffer::MetalBuffer) = isnothing(buffer.data)
pjrt_eltype(buffer::MetalBuffer) = buffer.type
Base.size(buffer::MetalBuffer) = Tuple(buffer.dims)
dynamic_dims(::MetalBuffer) = Int[]
memory_layout(buffer::MetalBuffer) = TiledMemoryLayout(length(buffer.dims):-1:1)
device(buffer::MetalBuffer) = buffer.device
memory(buffer::MetalBuffer) = buffer.memory
Base.sizeof(buffer::MetalBuffer) = buffer.size_in_bytes

function copyto(dev::MetalDevice, buffer::MetalBuffer)
    if dev === device(buffer)
        throw(invalid_argument("MetalBuffer is already on MetalDevice"))
    end
    new_buffer = MetalBuffer(buffer.type, buffer.dims, dev, default_memory(dev))
    Libc.memcpy(pointer(new_buffer), pointer(buffer), sizeof(buffer))
    return new_buffer
end

function copyto(mem::MetalMemory, buffer::MetalBuffer)
    devs = addressable_devices(mem)
    isempty(devs) && throw(invalid_argument("MetalMemory has no addressable devices"))
    dev = device(buffer) in devs ? device(buffer) : first(devs)
    new_buffer = MetalBuffer(buffer.type, buffer.dims, dev, mem)
    Libc.memcpy(pointer(new_buffer), pointer(buffer), sizeof(buffer))
    return new_buffer
end

function to_host_buffer!(dst::Ptr{Cvoid}, buffer::MetalBuffer)
    Libc.memcpy(dst, pointer(buffer), sizeof(buffer))
    return nothing
end

# the C wrapper checks that `buffer` is not deleted and that the range is within it; `offset` is 0-based
function Base.unsafe_copyto!(
    dst::Ptr, buffer::MetalBuffer, offset::Integer, nbytes::Integer
)
    Libc.memcpy(dst, pointer(buffer) + offset, nbytes)
    return dst
end

# shared storage lives in host memory, which lets Reactant read and write the buffer in place
isoncpu(::MetalBuffer) = true

# `execute` waits for the GPU and host data is copied synchronously, so the data of a `MetalBuffer` is always ready
function ready_event(::MetalBuffer)
    event = Event()
    notify(event)
    return event
end

Base.pointer(buffer::MetalBuffer) = MTL.contents(buffer.data)
