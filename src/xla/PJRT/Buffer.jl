mutable struct Buffer <: XLA.AbstractBuffer
    buffer::Ptr{Cvoid}

    function Buffer(buffer::Ptr{Cvoid})
        return finalizer(XLA.free_buffer, new(buffer))
    end
end

function Buffer(client::Client, array::Array{T,N}, device::Device) where {T,N}
    sizear = collect(Int64, reverse(size(array)))
    buffer = GC.@preserve array sizear begin
        @ccall MLIR.API.mlir_c.ArrayFromHostBuffer(
            client.client::Ptr{Cvoid},
            pointer(array)::Ptr{T},
            XLA.primitive_type(T)::UInt64,
            N::Csize_t,
            pointer(sizear)::Ptr{Int64},
            device.device::Ptr{Cvoid},
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
end

function Base.similar(a::Buffer)
    buffer = GC.@preserve a begin
        @ccall MLIR.API.mlir_c.UninitPJRTBuffer(
            XLA.client(a).client::Ptr{Cvoid},
            XLA.device(a).device::Ptr{Cvoid},
            (@ccall MLIR.API.mlir_c.BufferPrimitiveType(
                buffer.buffer::Ptr{Cvoid}
            )::Cint)::UInt64,
            (@ccall MLIR.API.mlir_c.BufferNDimensions(
                buffer.buffer::Ptr{Cvoid}
            )::Cint)::UInt64,
            (@ccall MLIR.API.mlir_c.BufferShape(
                buffer.buffer::Ptr{Cvoid}
            )::Ptr{Int64})::Ptr{Int64},
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
end

function Base.similar(a::Buffer, S::Type)
    buffer = GC.@preserve a begin
        @ccall MLIR.API.mlir_c.UninitPJRTBuffer(
            XLA.client(a).client::Ptr{Cvoid},
            XLA.device(a).device::Ptr{Cvoid},
            XLA.primitive_type(S)::UInt64,
            (@ccall MLIR.API.mlir_c.BufferNDimensions(
                buffer.buffer::Ptr{Cvoid}
            )::Cint)::UInt64,
            (@ccall MLIR.API.mlir_c.BufferShape(
                buffer.buffer::Ptr{Cvoid}
            )::Ptr{Int64})::Ptr{Int64},
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
end

function Base.similar(a::Buffer, dims::Dims)
    sizear = collect(Int64, reverse(dims))
    buffer = GC.@preserve a sizear begin
        @ccall MLIR.API.mlir_c.UninitPJRTBuffer(
            XLA.client(a).client::Ptr{Cvoid},
            XLA.device(a).device::Ptr{Cvoid},
            (@ccall MLIR.API.mlir_c.BufferPrimitiveType(
                buffer.buffer::Ptr{Cvoid}
            )::Cint)::UInt64,
            length(dims)::UInt64,
            pointer(sizear)::Ptr{Int64},
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
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
    buffer = GC.@preserve sizear begin
        @ccall MLIR.API.mlir_c.UninitPJRTBuffer(
            client.client::Ptr{Cvoid},
            device.device::Ptr{Cvoid},
            XLA.primitive_type(S)::UInt64,
            length(dims)::UInt64,
            pointer(sizear)::Ptr{Int64},
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
end

function Base.similar(a::Buffer, S::Type, dims::Dims)
    return Base.similar(Buffer, S, dims; client=XLA.client(a), device=XLA.device(a))
end

@inline function XLA.free_buffer(buffer::Buffer)
    sbuffer = buffer.buffer
    if sbuffer != C_NULL
        @ccall MLIR.API.mlir_c.PjRtBufferFree(sbuffer::Ptr{Cvoid})::Cvoid
    end
end

function Base.ndims(buffer::Buffer)
    GC.@preserve buffer begin
        return @ccall MLIR.API.mlir_c.BufferNDimensions(buffer.buffer::Ptr{Cvoid})::Cint
    end
end

function Base.size(buffer::Buffer)
    GC.@preserve buffer begin
        sz = @ccall MLIR.API.mlir_c.BufferShape(buffer.buffer::Ptr{Cvoid})::Ptr{Int64}
    end
    return Tuple(unsafe_wrap(Array, sz, ndims(buffer)))
end

function Base.eltype(buffer::Buffer)
    GC.@preserve buffer begin
        return XLA.julia_type(
            @ccall MLIR.API.mlir_c.BufferPrimitiveType(buffer.buffer::Ptr{Cvoid})::Cint
        )
    end
end

function XLA.device(buffer::Buffer)
    GC.@preserve buffer begin
        return Device(
            @ccall MLIR.API.mlir_c.BufferToDevice(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
        )
    end
end

function XLA.client(buffer::Buffer)
    GC.@preserve buffer begin
        return Client(
            @ccall MLIR.API.mlir_c.BufferToClient(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
        )
    end
end

XLA.synced_buffer(buffer::Buffer) = buffer

function XLA.buffer_on_cpu(buffer::Buffer)
    GC.@preserve buffer begin
        return @ccall MLIR.API.mlir_c.BufferOnCPU(buffer.buffer::Ptr{Cvoid})::Bool
    end
end

function XLA.to_host(buffer::Buffer, data, sharding)
    @assert data !== C_NULL
    @assert buffer.buffer !== C_NULL
    GC.@preserve buffer data begin
        @ccall MLIR.API.mlir_c.BufferToHost(
            buffer.buffer::Ptr{Cvoid}, data::Ptr{Cvoid}
        )::Cvoid
    end
    return data
end

# TODO: users themselves need to gc preserve here
function XLA.unsafe_buffer_pointer(buffer::Buffer)
    @ccall MLIR.API.mlir_c.UnsafeBufferPointer(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
end

function Base.copy(buffer::Buffer)
    dev = XLA.device(buffer)
    GC.@preserve buffer dev begin
        Buffer(
            @ccall MLIR.API.mlir_c.CopyBufferToDevice(
                buffer.buffer::Ptr{Cvoid}, dev.device::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function XLA.copy_buffer_to_device(buffer::Buffer, dev::Device)
    XLA.device(buffer) == dev && return buffer
    GC.@preserve buffer dev begin
        Buffer(
            @ccall MLIR.API.mlir_c.CopyBufferToDevice(
                buffer.buffer::Ptr{Cvoid}, dev.device::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

XLA.sharding(::Buffer) = Reactant.Sharding.NoSharding()
