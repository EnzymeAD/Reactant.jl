mutable struct HostBuffer <: Buffer
    data::Ptr{Cvoid}
    const size_in_bytes::Csize_t
    const type::CAPI.PJRT_Buffer_Type
    const dims::Vector{Int64}
    const minor_to_major::Vector{Int64}
    const device::HostDevice
    const memory::HostMemory

    function HostBuffer(args...)
        obj = new(args...)
        notify_device_of_alloc(obj)
        return Base.finalizer(obj) do x
            if !isdeleted(x)
                Libc.free(x.data)
                notify_device_of_free(x)
            end
        end
    end
end

function Base.delete!(buffer::HostBuffer)
    if !isdeleted(buffer)
        Libc.free(buffer.data)
        buffer.data = C_NULL
        notify_device_of_free(buffer)
    end
end

isdeleted(buffer::HostBuffer) = buffer.data == C_NULL
# TODO implement Base.eltype
pjrt_eltype(buffer::HostBuffer) = buffer.type
Base.size(buffer::HostBuffer) = Tuple(buffer.dims)
dynamic_dims(::HostBuffer) = Int[]
memory_layout(buffer::HostBuffer) = TiledMemoryLayout(buffer.minor_to_major)
device(buffer::HostBuffer) = buffer.device
memory(buffer::HostBuffer) = buffer.memory
Base.sizeof(buffer::HostBuffer) = Int(buffer.size_in_bytes)

function notify_device_of_alloc(buffer::HostBuffer)
    siz = sizeof(buffer)
    in_use = @atomic buffer.device.bytes_in_use += siz
    @atomic buffer.device.peak_bytes_in_use max in_use
    @atomic buffer.device.nallocs += 1
    @atomic buffer.device.largest_alloc_size max siz
end

notify_device_of_free(buffer::HostBuffer) = @atomic buffer.device.bytes_in_use -= Int64(buffer.size_in_bytes)

function copyto(dev::HostDevice, buffer::HostBuffer)
    if dev === device(buffer)
        throw(invalid_argument("HostBuffer is already on HostDevice"))
    end
    new_data = Libc.malloc(buffer.size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())
    Libc.memcpy(new_data, buffer.data, buffer.size_in_bytes)
    new_buffer = HostBuffer(
        new_data,
        buffer.size_in_bytes,
        buffer.type,
        buffer.dims,
        buffer.minor_to_major,
        dev,
        default_memory(dev),
    )
    return new_buffer
end

function copyto(mem::HostMemory, buffer::HostBuffer)
    devs = addressable_devices(mem)
    isempty(devs) && throw(invalid_argument("HostMemory has no addressable devices"))
    dev = device(buffer) in devs ? device(buffer) : first(devs)
    new_data = Libc.malloc(buffer.size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())
    Libc.memcpy(new_data, buffer.data, buffer.size_in_bytes)
    return HostBuffer(
        new_data,
        buffer.size_in_bytes,
        buffer.type,
        buffer.dims,
        buffer.minor_to_major,
        dev,
        mem,
    )
end

function to_host_buffer!(dst::Ptr{Cvoid}, buffer::HostBuffer)
    Libc.memcpy(dst, buffer.data, buffer.size_in_bytes)
    return nothing
end

# the C wrapper checks that `buffer` is not deleted and that the range is within it; `offset` is 0-based
function Base.unsafe_copyto!(dst::Ptr, buffer::HostBuffer, offset::Integer, nbytes::Integer)
    Libc.memcpy(dst, buffer.data + offset, nbytes)
    return dst
end

isoncpu(::HostBuffer) = true

# `HostBuffer` data is written synchronously before the buffer is handed out, so it is always ready
function ready_event(::HostBuffer)
    event = Event()
    notify(event)
    return event
end

Base.pointer(buffer::HostBuffer) = buffer.data
