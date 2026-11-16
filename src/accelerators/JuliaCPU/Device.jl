mutable struct HostDevice <: Device
    const description::HostDeviceDescription
    const local_hardware_id::Cint
    const memories::Vector{Memory}

    # memory stats
    @atomic bytes_in_use::Int64
    @atomic peak_bytes_in_use::Int64
    @atomic nallocs::Int64
    @atomic largest_alloc_size::Int64

    # `@kwdef` can't set defaults of `@atomic` fields
    function HostDevice(description, local_hardware_id, memories)
        return new(description, local_hardware_id, memories, 0, 0, 0, 0)
    end
end

description(dev::HostDevice) = dev.description
isaddressable(::HostDevice) = true
local_hardware_id(dev::HostDevice) = dev.local_hardware_id
addressable_memories(dev::HostDevice) = dev.memories
default_memory(dev::HostDevice) = dev.memories[1]

bytes_in_use(dev::HostDevice) = dev.bytes_in_use
peak_bytes_in_use(dev::HostDevice) = dev.peak_bytes_in_use
nallocs(dev::HostDevice) = dev.nallocs
largest_alloc_size(dev::HostDevice) = dev.largest_alloc_size

function clear_memory_stats!(dev::HostDevice)
    @atomic dev.peak_bytes_in_use = dev.bytes_in_use
    @atomic dev.nallocs = 0
    @atomic dev.largest_alloc_size = 0
end
