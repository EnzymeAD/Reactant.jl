mutable struct AsyncArray <: XLA.AbstractAsyncBuffer
    buffer::Array
    future::Union{Future,Nothing}
end

const AsyncEmptyArray = AsyncArray(Array(C_NULL), nothing)

AsyncArray(args...; kwargs...) = AsyncArray(Array(args...; kwargs...), nothing)
