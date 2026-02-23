struct Device <: XLA.AbstractDevice
    device::Ptr{Cvoid}
end

XLA.client(device::Device) = Client(MLIR.API.DeviceToClient(device.device))

XLA.device_ordinal(device::Device) = MLIR.API.PjRtDeviceGetLocalDeviceId(device.device)

function XLA.device_kind(device::Device)
    return XLA.unsafe_string_and_free(MLIR.API.DeviceGetKind(device.device))
end

XLA.get_local_device_id(device::Device) = MLIR.API.PjRtDeviceGetLocalDeviceId(device.device)

function XLA.get_local_hardware_id(device::Device)
    return MLIR.API.PjRtDeviceGetLocalHardwareId(device.device)
end

XLA.is_addressable(device::Device) = MLIR.API.PjRtDeviceIsAddressable(device.device)

function XLA.allocatorstats_internal(device::Device)
    ref = Ref{MLIR.API.JLAllocatorStats}()
    MLIR.API.PjRtDeviceGetAllocatorStats(device.device, ref)
    return ref[]
end
