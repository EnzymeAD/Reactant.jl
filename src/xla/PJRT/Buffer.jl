mutable struct Buffer <: XLA.AbstractBuffer
    buffer::Ptr{Cvoid}

    function Buffer(buffer::Ptr{Cvoid})
        return finalizer(free_buffer, new(buffer))
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
    buffer = GC.@preserve buffer begin
        return @ccall MLIR.API.mlir_c.UninitPJRTBuffer(
            XLA.client(a)::Ptr{Cvoid},
            XLA.device(a)::Ptr{Cvoid},
            (@ccall MLIR.API.mlir_c.BufferPrimitiveType(buffer.buffer::Ptr{Cvoid})::Cint)::UInt64,
            (@ccall MLIR.API.mlir_c.BufferNDimensions(buffer.buffer::Ptr{Cvoid})::Cint)::UInt64,
            (@ccall MLIR.API.mlir_c.BufferShape(buffer.buffer::Ptr{Cvoid})::Ptr{Int64})::Ptr{Int64}
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
end

function Base.similar(a::Buffer, ::Type{S}) where S
    buffer = GC.@preserve buffer begin
        return @ccall MLIR.API.mlir_c.UninitPJRTBuffer(
            XLA.client(a)::Ptr{Cvoid},
            XLA.device(a)::Ptr{Cvoid},
            primitive_type(S)::UInt64,
            (@ccall MLIR.API.mlir_c.BufferNDimensions(buffer.buffer::Ptr{Cvoid})::Cint)::UInt64
            (@ccall MLIR.API.mlir_c.BufferShape(buffer.buffer::Ptr{Cvoid})::Ptr{Int64})::Ptr{Int64}
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
end

function Base.similar(a::Buffer, dims::Dims)
    sizear = collect(Int64, reverse(dims))
    buffer = GC.@preserve buffer sizear begin
        return @ccall MLIR.API.mlir_c.UninitPJRTBuffer(
            XLA.client(a)::Ptr{Cvoid},
            XLA.device(a)::Ptr{Cvoid},
            @ccall MLIR.API.mlir_c.BufferPrimitiveType(buffer.buffer::Ptr{Cvoid})::Cint,
            length(dims)::UInt64
            pointer(sizear)::Ptr{Int64}
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
end

function Base.similar(a::Buffer, ::Type{S}, dims::Dims) where S
    sizear = collect(Int64, reverse(dims))
    buffer = GC.@preserve buffer sizear begin
        return @ccall MLIR.API.mlir_c.UninitPJRTBuffer(
            XLA.client(a)::Ptr{Cvoid},
            XLA.device(a)::Ptr{Cvoid},
            primitive_type(S)::UInt64,
            length(dims)::UInt64
            pointer(sizear)::Ptr{Int64}
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
end

@inline function free_buffer(buffer::Buffer)
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
    GC.@preserve buffer begin
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
