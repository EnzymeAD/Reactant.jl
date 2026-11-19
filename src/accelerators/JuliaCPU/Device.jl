mutable struct JuliaDevice <: Device
    const description::JuliaDeviceDescription
    const local_hardware_id::Cint
    const memories::Vector{Memory}

    # memory stats
    @atomic bytes_in_use::Int64
    @atomic peak_bytes_in_use::Int64
    @atomic nallocs::Int64
    @atomic largest_alloc_size::Int64

    # `@kwdef` can't set defaults of `@atomic` fields
    function JuliaDevice(description, local_hardware_id, memories)
        return new(description, local_hardware_id, memories, 0, 0, 0, 0)
    end
end

description(dev::JuliaDevice) = dev.description
isaddressable(::JuliaDevice) = true
local_hardware_id(dev::JuliaDevice) = dev.local_hardware_id
addressable_memories(dev::JuliaDevice) = dev.memories
default_memory(dev::JuliaDevice) = dev.memories[1]

bytes_in_use(dev::JuliaDevice) = dev.bytes_in_use
peak_bytes_in_use(dev::JuliaDevice) = dev.peak_bytes_in_use
nallocs(dev::JuliaDevice) = dev.nallocs
largest_alloc_size(dev::JuliaDevice) = dev.largest_alloc_size

function clear_memory_stats!(dev::JuliaDevice)
    @atomic dev.peak_bytes_in_use = dev.bytes_in_use
    @atomic dev.nallocs = 0
    @atomic dev.largest_alloc_size = 0
end
