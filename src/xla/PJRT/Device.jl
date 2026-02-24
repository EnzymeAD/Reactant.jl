struct Device <: XLA.AbstractDevice
    device::Ptr{Cvoid}
end

function XLA.client(device::Device)
    GC.@preserve device Client(MLIR.API.DeviceToClient(device.device))
end

function XLA.device_ordinal(device::Device)
    GC.@preserve device MLIR.API.PjRtDeviceGetLocalDeviceId(device.device)
end

function XLA.device_kind(device::Device)
    return GC.@preserve device XLA.unsafe_string_and_free(
        MLIR.API.DeviceGetKind(device.device)
    )
end

function XLA.get_local_device_id(device::Device)
    GC.@preserve device MLIR.API.PjRtDeviceGetLocalDeviceId(device.device)
end

function XLA.get_local_hardware_id(device::Device)
    return GC.@preserve device MLIR.API.PjRtDeviceGetLocalHardwareId(device.device)
end

function XLA.is_addressable(device::Device)
    GC.@preserve device MLIR.API.PjRtDeviceIsAddressable(device.device)
end

function XLA.allocatorstats_internal(device::Device)
    ref = Ref{MLIR.API.JLAllocatorStats}()
    GC.@preserve device MLIR.API.PjRtDeviceGetAllocatorStats(device.device, ref)
    return ref[]
end
