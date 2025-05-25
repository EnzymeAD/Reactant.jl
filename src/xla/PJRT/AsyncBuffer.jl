mutable struct AsyncBuffer <: XLA.AbstractAsyncBuffer
    buffer::Buffer
    future::Union{Future,Nothing}
end

const AsyncEmptyBuffer = AsyncBuffer(Buffer(C_NULL), nothing)

AsyncBuffer(args...; kwargs...) = AsyncBuffer(Buffer(args...; kwargs...), nothing)

function Base.copy(b::AsyncBuffer)
     Base.wait(b)
     b2 = XLA.copy_buffer_to_device(b.buffer, XLA.device(b.buffer))
     AsyncBuffer(b2, nothing)
end
