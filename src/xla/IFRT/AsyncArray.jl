mutable struct AsyncArray <: XLA.AbstractAsyncBuffer
    buffer::Array
    future::Union{Future,Nothing}
end

const AsyncEmptyArray = AsyncArray(Array(C_NULL, false), nothing)

AsyncArray(args...; kwargs...) = AsyncArray(Array(args...; kwargs...), nothing)

function disassemble_into_single_device_arrays(
    x::AsyncArray, only_addressable_devices::Bool
)
    wait(x)
    return disassemble_into_single_device_arrays(x.buffer, only_addressable_devices)
end

function replicate_array_to_all_devices(array::AsyncArray, args...)
    wait(array)
    return replicate_array_to_all_devices(array.buffer, args...)
end

function XLA.to_host(array::AsyncArray, data, reactant_sharding)
    wait(array)
    return XLA.to_host(array.buffer, data, reactant_sharding)
end

XLA.sharding(x::AsyncArray) = XLA.sharding(x.buffer)

function Base.copy(b::AsyncArray)
    Base.wait(b)
    return AsyncArray(Base.copy(b.buffer), nothing)
end
