# Buffer
mutable struct HeldBuffer
    ptr::Ptr{Cvoid}

    function HeldBuffer(ptr::Ptr{Cvoid})
        return finalizer(release_buffer, new(ptr))
    end
end

@inline function release_buffer(held_buffer::HeldBuffer)
    @ccall MLIR.API.mlir_c.reactant_release_pjrtbuffer(held_buffer.ptr::Ptr{Cvoid})::Cvoid
end

mutable struct Buffer
    buffer::Ptr{Cvoid}
    held::Union{Nothing,HeldBuffer}

    function Buffer(buffer::Ptr{Cvoid})
        return finalizer(free_buffer, new(buffer, nothing))
    end
end

@inline function free_buffer(buffer)
    if isnothing(buffer.holded) && buffer.buffer != C_NULL
        @ccall MLIR.API.mlir_c.PjRtBufferFree(buffer.buffer::Ptr{Cvoid})::Cvoid
    end
end

function hold!(buffer::Buffer)
    if buffer.holded == C_NULL
        sbuffer = buffer.buffer
        buffer.holded = HeldBuffer(
            @ccall MLIR.API.mlir_c.reactant_hold_pjrtbuffer(sbuffer::Ptr{Cvoid})::Ptr{Cvoid}
        )
    end
    return buffer
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

function device(buffer::Buffer)
    GC.@preserve buffer begin
        return Device(
            @ccall MLIR.API.mlir_c.BufferToDevice(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
        )
    end
end

function client(buffer::Buffer)
    GC.@preserve buffer begin
        return Client(
            @ccall MLIR.API.mlir_c.BufferToClient(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
        )
    end
end

@inline synced_buffer(buffer::Buffer) = buffer

# TODO users themselves need to gc preserve here
function UnsafeBufferPointer(buffer::Buffer)
    @ccall MLIR.API.mlir_c.UnsafeBufferPointer(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
end

function BufferOnCPU(buffer::Buffer)
    GC.@preserve buffer begin
        (@ccall MLIR.API.mlir_c.BufferOnCPU(buffer.buffer::Ptr{Cvoid})::UInt8) != 0
    end
end

function ArrayFromHostBuffer(client::Client, array::Array{T,N}, device) where {T,N}
    sizear = Int64[s for s in reverse(size(array))]
    buffer = GC.@preserve array sizear begin
        @ccall MLIR.API.mlir_c.ArrayFromHostBuffer(
            client.client::Ptr{Cvoid},
            pointer(array)::Ptr{T},
            primitive_type(T)::UInt64,
            N::Csize_t,
            pointer(sizear)::Ptr{Int64},
            device.device::Ptr{Cvoid},
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
end

function BufferToHost(buffer::Buffer, data)
    GC.@preserve buffer begin
        @ccall MLIR.API.mlir_c.BufferToHost(
            buffer.buffer::Ptr{Cvoid}, data::Ptr{Cvoid}
        )::Cvoid
    end
end

function CopyBufferToDevice(buffer::Buffer, dev::Device)
    device(buffer) == dev && return buffer
    GC.@preserve buffer dev begin
        Buffer(
            @ccall MLIR.API.mlir_c.CopyBufferToDevice(
                buffer.buffer::Ptr{Cvoid}, dev.device::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

# AsyncBuffer
mutable struct AsyncBuffer
    buffer::Buffer
    future::Union{Future,Nothing}
end

const AsyncEmptyBuffer = AsyncBuffer(Buffer(C_NULL), nothing)

for op in (:(Base.ndims), :(Base.size), :device, :client)
    @eval $op(buffer::AsyncBuffer) = $op(buffer.buffer)
end

function client(buffers::Union{Array{<:AsyncBuffer},NTuple{<:Any,AsyncBuffer}})
    all_clients = map(client, buffers)
    @assert allequal(all_clients) "All buffers must have the same client"
    return first(all_clients)
end

@inline function synced_buffer(buffer::AsyncBuffer)
    if buffer.future !== nothing
        future = buffer.future
        buffer.future = nothing
        await(future::Future)
    end
    return buffer.buffer
end

@inline function synced_buffer(
    buffers::Union{
        AbstractArray{<:Union{AsyncBuffer,Buffer}},NTuple{<:Any,<:Union{AsyncBuffer,Buffer}}
    },
)
    return map(synced_buffer, buffers)
end

@inline function await(buffer::AsyncBuffer)::Nothing
    if buffer.future === nothing
        return nothing
    else
        future = buffer.future
        buffer.future = nothing
        await(future::Future)
    end
    return nothing
end

function is_ready(buffer::AsyncBuffer)::Bool
    future = buffer.future
    if isnothing(future)
        return true
    else
        return is_ready(future)
    end
end
