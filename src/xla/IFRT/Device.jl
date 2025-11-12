struct Device <: XLA.AbstractDevice
    device::Ptr{Cvoid}
end

function XLA.client(device::Device)
    GC.@preserve device begin
        return Client(
            @ccall MLIR.API.mlir_c.ifrt_DeviceToClient(
                device.device::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function XLA.device_ordinal(device::Device)
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.ifrt_DeviceGetGlobalDeviceId(
            device.device::Ptr{Cvoid}
        )::Int64
    end
end

function XLA.device_kind(device::Device)
    GC.@preserve device begin
        str = @ccall MLIR.API.mlir_c.ifrt_DeviceGetKind(device.device::Ptr{Cvoid})::Cstring
    end
    return XLA.unsafe_string_and_free(str)
end

function XLA.get_local_device_id(::Device)
    return error("Not implemented for ifrt devices")
end

function XLA.get_local_hardware_id(device::Device)
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.ifrt_DeviceGetLocalHardwareId(
            device.device::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.default_memory(device::Device)
    GC.@preserve device begin
        return Memory(
            @ccall MLIR.API.mlir_c.ifrt_DeviceGetDefaultMemory(
                device.device::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function XLA.memories(device::Device)
    memories_size = Ref{Int32}(0)
    GC.@preserve device memories_size begin
        ptr = @ccall MLIR.API.mlir_c.ifrt_DeviceGetMemories(
            device.device::Ptr{Cvoid}, memories_size::Ptr{Int32}
        )::Ptr{Ptr{Cvoid}}
    end
    return [Memory(unsafe_load(ptr, i)) for i in 1:memories_size[]]
end

# TODO: https://github.com/openxla/xla/blob/ad0814d221883609f784e57dd26914b17f92fbbc/xla/python/ifrt/sharding.cc#L60
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
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.ifrt_DeviceIsAddressable(
            device.device::Ptr{Cvoid}
        )::Bool
    end
end

function XLA.allocatorstats_internal(device::Device)
    ref = Ref{XLA.JLAllocatorStats}()
    @ccall MLIR.API.mlir_c.ifrt_device_get_allocator_stats(
        device.device::Ptr{Cvoid}, ref::Ptr{Cvoid}
    )::Cvoid
    return ref[]
end
