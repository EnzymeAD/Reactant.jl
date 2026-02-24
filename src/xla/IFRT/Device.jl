struct Device <: XLA.AbstractDevice
    device::Ptr{Cvoid}
end

function XLA.client(device::Device)
    return Client(MLIR.API.ifrt_DeviceToClient(device.device))
end

function XLA.device_ordinal(device::Device)
    return MLIR.API.ifrt_DeviceGetGlobalDeviceId(device.device)
end

function XLA.device_kind(device::Device)
    str = MLIR.API.ifrt_DeviceGetKind(device.device)
    return XLA.unsafe_string_and_free(str)
end

function XLA.get_local_device_id(::Device)
    return error("Not implemented for ifrt devices")
end

function XLA.get_local_hardware_id(device::Device)
    return MLIR.API.ifrt_DeviceGetLocalHardwareId(device.device)
end

function XLA.default_memory(device::Device)
    return Memory(MLIR.API.ifrt_DeviceGetDefaultMemory(device.device))
end

function XLA.memories(device::Device)
    memories_size = Ref{Int32}(0)
    ptr = MLIR.API.ifrt_DeviceGetMemories(device.device, memories_size)
    return [Memory(unsafe_load(ptr, i)) for i in 1:memories_size[]]
end

# TODO(#2235): https://github.com/openxla/xla/blob/ad0814d221883609f784e57dd26914b17f92fbbc/xla/python/ifrt/sharding.cc#L60
function XLA.default_memory(device_list::AbstractVector{Device})
    default_memories = XLA.default_memory.(device_list)
    default_memory_kinds = convert.(MemoryKind, default_memories)
    @assert allequal(default_memory_kinds) "All devices must have the same default memory"
    return first(default_memories)
end

function XLA.client(device_list::AbstractVector{Device})
    clients = XLA.client.(device_list)
    @assert allequal(clients) "All devices must have the same client"
    return first(clients)
end

function XLA.is_addressable(device::Device)
    return MLIR.API.ifrt_DeviceIsAddressable(device.device)
end

function XLA.allocatorstats_internal(device::Device)
    ref = Ref{MLIR.API.JLAllocatorStats}()
    MLIR.API.ifrt_device_get_allocator_stats(device.device, ref)
    return ref[]
end
