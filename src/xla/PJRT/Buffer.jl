mutable struct Buffer <: XLA.AbstractBuffer
    buffer::Ptr{Cvoid}

    function Buffer(buffer::Ptr{Cvoid})
        return finalizer(XLA.free_buffer, new(buffer))
    end
end

function Buffer(client::Client, array::Array{T,N}, device::Device) where {T,N}
    sizear = collect(Int64, reverse(size(array)))
    buffer = begin
        MLIR.API.ArrayFromHostBuffer(
            client.client,
            pointer(array),
            XLA.primitive_type(T),
            N,
            pointer(sizear),
            device.device,
        )
    end
    return Buffer(buffer)
end

function Base.similar(a::Buffer)
    new_buffer = begin
        MLIR.API.UninitPJRTBuffer(
            XLA.client(a).client,
            XLA.device(a).device,
            (MLIR.API.BufferPrimitiveType(a.buffer)),
            (MLIR.API.BufferNDimensions(a.buffer)),
            (MLIR.API.BufferShape(a.buffer)),
        )
    end
    return Buffer(new_buffer)
end

function Base.similar(a::Buffer, S::Type)
    new_buffer = begin
        MLIR.API.UninitPJRTBuffer(
            XLA.client(a).client,
            XLA.device(a).device,
            XLA.primitive_type(S),
            (MLIR.API.BufferNDimensions(a.buffer)),
            (MLIR.API.BufferShape(a.buffer)),
        )
    end
    return Buffer(new_buffer)
end

function Base.similar(a::Buffer, dims::Dims)
    sizear = collect(Int64, reverse(dims))
    new_buffer = begin
        MLIR.API.UninitPJRTBuffer(
            XLA.client(a).client,
            XLA.device(a).device,
            (MLIR.API.BufferPrimitiveType(a.buffer)),
            length(dims),
            pointer(sizear),
        )
    end
    return Buffer(new_buffer)
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

    sizear = collect(Int64, reverse(dims))
    buffer = begin
        MLIR.API.UninitPJRTBuffer(
            client.client,
            device.device,
            XLA.primitive_type(S),
            length(dims),
            pointer(sizear),
        )
    end
    return Buffer(buffer)
end

function Base.similar(a::Buffer, S::Type, dims::Dims)
    return Base.similar(Buffer, S, dims; client=XLA.client(a), device=XLA.device(a))
end

@inline function XLA.free_buffer(buffer::Buffer)
    sbuffer = buffer.buffer
    if sbuffer != C_NULL && XLA.is_live[]
        MLIR.API.PjRtBufferFree(sbuffer)
    end
end

function Base.ndims(buffer::Buffer)
    return MLIR.API.BufferNDimensions(buffer.buffer)
end

function Base.size(buffer::Buffer)
    sz = MLIR.API.BufferShape(buffer.buffer)
    return Tuple(unsafe_wrap(Array, sz, ndims(buffer)))
end

function Base.eltype(buffer::Buffer)
    return XLA.julia_type(MLIR.API.BufferPrimitiveType(buffer.buffer))
end

function XLA.device(buffer::Buffer)
    return Device(MLIR.API.BufferToDevice(buffer.buffer))
end

function XLA.client(buffer::Buffer)
    return Client(MLIR.API.BufferToClient(buffer.buffer))
end

XLA.synced_buffer(buffer::Buffer) = buffer

function XLA.buffer_on_cpu(buffer::Buffer)
    return MLIR.API.BufferOnCPU(buffer.buffer)
end

function XLA.to_host(buffer::Buffer, data, sharding)
    @assert data !== C_NULL
    @assert buffer.buffer !== C_NULL
    MLIR.API.BufferToHost(buffer.buffer, data)
    return data
end

# TODO(#2235): users themselves need to gc preserve here
function XLA.unsafe_buffer_pointer(buffer::Buffer)
    return MLIR.API.UnsafeBufferPointer(buffer.buffer)
end

function Base.copy(buffer::Buffer)
    dev = XLA.device(buffer)
    return Buffer(MLIR.API.CopyBufferToDevice(buffer.buffer, dev.device))
end

function XLA.copy_buffer_to_device(buffer::Buffer, dev::Device)
    XLA.device(buffer) == dev && return buffer
    return Buffer(MLIR.API.CopyBufferToDevice(buffer.buffer, dev.device))
end

XLA.sharding(::Buffer) = Reactant.Sharding.NoSharding()
