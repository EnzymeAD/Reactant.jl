struct Device <: XLA.AbstractDevice
    device::Ptr{Cvoid}
end

function XLA.client(device::Device)
    return Client(MLIR.API.DeviceToClient(device.device))
end

function XLA.device_ordinal(device::Device)
    return MLIR.API.PjRtDeviceGetLocalDeviceId(device.device)
end

function XLA.device_kind(device::Device)
    str = MLIR.API.DeviceGetKind(device.device)
    return XLA.unsafe_string_and_free(str)
end

function XLA.get_local_device_id(device::Device)
    return MLIR.API.PjRtDeviceGetLocalDeviceId(device.device)
end

function XLA.get_local_hardware_id(device::Device)
    return MLIR.API.PjRtDeviceGetLocalHardwareId(device.device)
end

function XLA.is_addressable(device::Device)
    return MLIR.API.pjrt_device_is_addressable(device.device)
end

function XLA.allocatorstats_internal(device::Device)
    ref = Ref{XLA.JLAllocatorStats}()
    MLIR.API.PjRtDeviceGetAllocatorStats(device.device, ref)
    return ref[]
end
