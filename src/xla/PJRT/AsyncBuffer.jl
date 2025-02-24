mutable struct AsyncBuffer <: XLA.AbstractAsyncBuffer
    buffer::Buffer
    future::Union{Future,Nothing}
end

const AsyncEmptyBuffer = AsyncBuffer(Buffer(C_NULL), nothing)

AsyncBuffer(args...; kwargs...) = AsyncBuffer(Buffer(args...; kwargs...), nothing)
