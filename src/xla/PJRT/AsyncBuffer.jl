mutable struct AsyncBuffer <: XLA.AbstractAsyncBuffer
    buffer::Buffer
    future::Union{Future,Nothing}
end

const AsyncEmptyBuffer = AsyncBuffer(Buffer(C_NULL), nothing)

AsyncBuffer(args...; kwargs...) = AsyncBuffer(Buffer(args...; kwargs...), nothing)

function Base.copy(b::AsyncBuffer)
    Base.wait(b)
    # The device-to-device copy is asynchronous: the new buffer is allocated eagerly but
    # its contents arrive later. Track that with the buffer's ready future so that
    # `wait`/`synchronize` on the copy (and hence raw-pointer access to it) are correct.
    buffer = Base.copy(b.buffer)
    return AsyncBuffer(buffer, ready_future(buffer))
end

function Base.similar(a::AsyncBuffer, args...)
    return AsyncBuffer(Base.similar(a.buffer, args...)::Buffer, nothing)
end

@inline function Base.similar(::Type{AsyncBuffer}, args...; kwargs...)
    return AsyncBuffer(Base.similar(Buffer, args...; kwargs...)::Buffer, nothing)
end
