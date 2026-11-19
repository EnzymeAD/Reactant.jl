mutable struct MetalDevice <: Device
    const description::MetalDeviceDescription
    const mtl::MTLDevice
    const memories::Vector{Memory}
end

description(dev::MetalDevice) = dev.description
isaddressable(::MetalDevice) = true
# Metal.jl only uses the first GPU of the system
local_hardware_id(::MetalDevice) = 0
addressable_memories(dev::MetalDevice) = dev.memories
default_memory(dev::MetalDevice) = dev.memories[1]

# memory allocated on the GPU by this process, which includes the allocations made by Metal.jl
bytes_in_use(dev::MetalDevice) = Int64(dev.mtl.currentAllocatedSize)
