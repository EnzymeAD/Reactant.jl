mutable struct Buffer <: XLA.AbstractBuffer
    buffer::Ptr{Cvoid}

    function Buffer(buffer::Ptr{Cvoid})
        return finalizer(XLA.free_buffer, new(buffer))
    end
end

function Buffer(client::Client, array::Array{T,N}, device::Device) where {T,N}
    sizear = collect(Int64, reverse(size(array)))
    GC.@preserve client array device begin
        buffer = MLIR.API.ArrayFromHostBuffer(
            client.client, pointer(array), XLA.primitive_type(T), N, sizear, device.device
        )
    end
    return Buffer(buffer)
end

function Base.similar(a::Buffer)
    return Buffer(
        MLIR.API.UninitPJRTBuffer(
            XLA.client(a).client,
            XLA.device(a).device,
            MLIR.API.BufferPrimitiveType(a.buffer),
            MLIR.API.BufferNDimensions(a.buffer),
            MLIR.API.BufferShape(a.buffer),
        ),
    )
end

function Base.similar(a::Buffer, S::Type)
    client = XLA.client(a)
    device = XLA.device(a)
    GC.@preserve client device a begin
        return Buffer(
            MLIR.API.UninitPJRTBuffer(
                client.client,
                device.device,
                XLA.primitive_type(S),
                MLIR.API.BufferNDimensions(a.buffer),
                MLIR.API.BufferShape(a.buffer),
            ),
        )
    end
end

function Base.similar(a::Buffer, dims::Dims)
    sizear = collect(Int64, reverse(dims))
    client = XLA.client(a)
    device = XLA.device(a)
    GC.@preserve client device a begin
        return Buffer(
            MLIR.API.UninitPJRTBuffer(
                client.client,
                device.device,
                MLIR.API.BufferPrimitiveType(a.buffer),
                length(dims),
                sizear,
            ),
        )
    end
end

@inline function Base.similar(
    ::Type{Buffer},
    S::Type,
    dims::Dims;
    client::Union{Nothing,Client}=nothing,
    idx::Union{Int,Nothing}=nothing,
    device::Union{Nothing,Device}=nothing,
)
    client = client === nothing ? XLA.default_backend() : client

    if device === nothing
        if idx === nothing
            device = XLA.default_device(client)
        else
            device = XLA.get_device(client, idx)
        end
    else
        if idx !== nothing
            device_from_idx = XLA.get_device(client, idx)
            @assert device_from_idx == device "If both `idx` and `device` are \
                                               specified, `idx` must match `device`"
        end
    end

    GC.@preserve client device begin
        return Buffer(
            MLIR.API.UninitPJRTBuffer(
                client.client,
                device.device,
                XLA.primitive_type(S),
                length(dims),
                collect(Int64, reverse(dims)),
            ),
        )
    end
end

function Base.similar(a::Buffer, S::Type, dims::Dims)
    return Base.similar(Buffer, S, dims; client=XLA.client(a), device=XLA.device(a))
end

@inline function XLA.free_buffer(buffer::Buffer)
    if buffer.buffer != C_NULL && XLA.is_live[]
        GC.@preserve buffer MLIR.API.PjRtBufferFree(buffer.buffer)
    end
end

function Base.ndims(buffer::Buffer)
    return GC.@preserve buffer MLIR.API.BufferNDimensions(buffer.buffer)
end

function Base.size(buffer::Buffer)
    sz = GC.@preserve buffer MLIR.API.BufferShape(buffer.buffer)
    return Tuple(unsafe_wrap(Array, sz, ndims(buffer)))
end

function Base.eltype(buffer::Buffer)
    pt = GC.@preserve buffer MLIR.API.BufferPrimitiveType(buffer.buffer)
    return XLA.julia_type(pt)
end

function XLA.device(buffer::Buffer)
    return Device(GC.@preserve buffer MLIR.API.BufferToDevice(buffer.buffer))
end

function XLA.client(buffer::Buffer)
    return Client(GC.@preserve buffer MLIR.API.BufferToClient(buffer.buffer))
end

XLA.synced_buffer(buffer::Buffer) = buffer

function XLA.buffer_on_cpu(buffer::Buffer)
    res = GC.@preserve buffer MLIR.API.BufferOnCPU(buffer.buffer)
    return res == 1
end

function XLA.to_host(buffer::Buffer, data, sharding)
    @assert data !== C_NULL
    @assert buffer.buffer !== C_NULL
    GC.@preserve buffer MLIR.API.BufferToHost(buffer.buffer, data)
    return data
end

# TODO(#2235): users themselves need to gc preserve here
function XLA.unsafe_buffer_pointer(buffer::Buffer)
    return GC.@preserve buffer MLIR.API.UnsafeBufferPointer(buffer.buffer)
end

function Base.copy(buffer::Buffer)
    dev = XLA.device(buffer)
    return Buffer(
        GC.@preserve buffer dev MLIR.API.CopyBufferToDevice(buffer.buffer, dev.device)
    )
end

function XLA.copy_buffer_to_device(buffer::Buffer, dev::Device)
    XLA.device(buffer) == dev && return buffer
    return Buffer(
        GC.@preserve buffer dev MLIR.API.CopyBufferToDevice(buffer.buffer, dev.device)
    )
end

XLA.sharding(::Buffer) = Reactant.Sharding.NoSharding()
