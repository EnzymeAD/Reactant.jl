mutable struct AsyncBuffer <: XLA.AbstractAsyncBuffer
    buffer::Buffer
    future::Union{Future,Nothing}
end

const AsyncEmptyBuffer = AsyncBuffer(Buffer(C_NULL), nothing)

AsyncBuffer(args...; kwargs...) = AsyncBuffer(Buffer(args...; kwargs...), nothing)

function Base.copy(b::AsyncBuffer)
    Base.wait(b)
    return AsyncBuffer(Base.copy(b.buffer), nothing)
end

function Base.similar(a::AsyncBuffer, args...)
    return AsyncBuffer(Base.similar(a.buffer, args...)::Buffer, nothing)
end

@inline function Base.similar(::Type{AsyncBuffer}, args...; kwargs...)
    return AsyncBuffer(Base.similar(Buffer, args...; kwargs...)::Buffer, nothing)
end
