mutable struct AsyncBuffer <: XLA.AbstractBuffer
    buffer::Buffer
    future::Union{Future,Nothing}
end

const AsyncEmptyBuffer = AsyncBuffer(Buffer(C_NULL), nothing)

function AsyncBuffer(client::Client, array::Array{T,N}, device::Device) where {T,N}
    return AsyncBuffer(Buffer(client, array, device), nothing)
end

Base.isempty(buffer::AsyncBuffer) = buffer == AsyncEmptyBuffer

function Base.convert(::Type{<:Array{T}}, buffer::AsyncBuffer) where {T}
    XLA.await(buffer)
    return convert(Array{T}, buffer.buffer)
end

for op in (:(Base.ndims), :(Base.size), :device, :client)
    @eval $op(buffer::AsyncBuffer) = $op(buffer.buffer)
end

function XLA.synced_buffer(buffer::AsyncBuffer)
    XLA.await(buffer)
    return buffer.buffer
end

function XLA.await(buffer::AsyncBuffer)
    buffer.future === nothing && return nothing
    future = buffer.future
    buffer.future = nothing
    XLA.await(future)
    return nothing
end

function XLA.is_ready(buffer::AsyncBuffer)
    buffer.future === nothing && return true
    return XLA.is_ready(buffer.future)
end

XLA.buffer_on_cpu(buffer::AsyncBuffer) = XLA.buffer_on_cpu(buffer.buffer)

XLA.client(buffer::AsyncBuffer) = XLA.client(buffer.buffer)
XLA.device(buffer::AsyncBuffer) = XLA.device(buffer.buffer)
