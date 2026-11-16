mutable struct JuliaBuffer <: Buffer
    data::Ptr{Cvoid}
    const size_in_bytes::Csize_t
    const type::CAPI.PJRT_Buffer_Type
    const dims::Vector{Int64}
    const minor_to_major::Vector{Int64}
    const device::JuliaDevice
    const memory::JuliaMemory

    function JuliaBuffer(args...)
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

function Base.delete!(buffer::JuliaBuffer)
    if !isdeleted(buffer)
        Libc.free(buffer.data)
        buffer.data = C_NULL
        notify_device_of_free(buffer)
    end
end

isdeleted(buffer::JuliaBuffer) = buffer.data == C_NULL
# TODO implement Base.eltype
pjrt_eltype(buffer::JuliaBuffer) = buffer.type
Base.size(buffer::JuliaBuffer) = Tuple(buffer.dims)
dynamic_dims(::JuliaBuffer) = Int[]
memory_layout(buffer::JuliaBuffer) = TiledMemoryLayout(buffer.minor_to_major)
device(buffer::JuliaBuffer) = buffer.device
memory(buffer::JuliaBuffer) = buffer.memory
Base.sizeof(buffer::JuliaBuffer) = Int(buffer.size_in_bytes)

function notify_device_of_alloc(buffer::JuliaBuffer)
    siz = sizeof(buffer)
    in_use = @atomic buffer.device.bytes_in_use += siz
    @atomic buffer.device.peak_bytes_in_use max in_use
    @atomic buffer.device.nallocs += 1
    @atomic buffer.device.largest_alloc_size max siz
end

function notify_device_of_free(buffer::JuliaBuffer)
    @atomic buffer.device.bytes_in_use -= Int64(buffer.size_in_bytes)
end

function copyto(dev::JuliaDevice, buffer::JuliaBuffer)
    if dev === device(buffer)
        throw(invalid_argument("JuliaBuffer is already on JuliaDevice"))
    end
    new_data = Libc.malloc(buffer.size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())
    Libc.memcpy(new_data, buffer.data, buffer.size_in_bytes)
    new_buffer = JuliaBuffer(
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

function copyto(mem::JuliaMemory, buffer::JuliaBuffer)
    devs = addressable_devices(mem)
    isempty(devs) && throw(invalid_argument("JuliaMemory has no addressable devices"))
    dev = device(buffer) in devs ? device(buffer) : first(devs)
    new_data = Libc.malloc(buffer.size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())
    Libc.memcpy(new_data, buffer.data, buffer.size_in_bytes)
    return JuliaBuffer(
        new_data,
        buffer.size_in_bytes,
        buffer.type,
        buffer.dims,
        buffer.minor_to_major,
        dev,
        mem,
    )
end

function to_host_buffer!(dst::Ptr{Cvoid}, buffer::JuliaBuffer)
    Libc.memcpy(dst, buffer.data, buffer.size_in_bytes)
    return nothing
end

# the C wrapper checks that `buffer` is not deleted and that the range is within it; `offset` is 0-based
function Base.unsafe_copyto!(
    dst::Ptr, buffer::JuliaBuffer, offset::Integer, nbytes::Integer
)
    Libc.memcpy(dst, buffer.data + offset, nbytes)
    return dst
end

isoncpu(::JuliaBuffer) = true

# `JuliaBuffer` data is written synchronously before the buffer is handed out, so it is always ready
function ready_event(::JuliaBuffer)
    event = Event()
    notify(event)
    return event
end

Base.pointer(buffer::JuliaBuffer) = buffer.data
