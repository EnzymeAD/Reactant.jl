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
    return [unsafe_load(sz, i) for i in 1:ndims(buffer)]
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

function Base.convert(::Type{<:Array{T}}, buffer::Buffer) where {T}
    arr = zeros(T, reverse(size(buffer))...)
    XLA.to_host(buffer, arr)
    return arr
end

function XLA.to_host(buffer::Buffer, data)
    GC.@preserve buffer begin
        @ccall MLIR.API.mlir_c.BufferToHost(
            buffer.buffer::Ptr{Cvoid}, data::Ptr{Cvoid}
        )::Cvoid
    end
end

# TODO: users themselves need to gc preserve here
function XLA.unsafe_buffer_pointer(buffer::Buffer)
    @ccall MLIR.API.mlir_c.UnsafeBufferPointer(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
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
