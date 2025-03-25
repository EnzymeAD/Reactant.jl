mutable struct AsyncArray <: XLA.AbstractAsyncBuffer
    buffer::Array
    future::Union{Future,Nothing}
end

const AsyncEmptyArray = AsyncArray(Array(C_NULL), nothing)

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
