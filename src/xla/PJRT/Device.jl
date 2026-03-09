struct Device <: XLA.AbstractDevice
    device::Ptr{Cvoid}
end

function XLA.client(device::Device)
    GC.@preserve device begin
        return Client(MLIR.API.DeviceToClient(device.device))
    end
end

function XLA.device_ordinal(device::Device)
    GC.@preserve device begin
        return MLIR.API.PjRtDeviceGetLocalDeviceId(device.device)
    end
end

function XLA.device_kind(device::Device)
    GC.@preserve device begin
        str = MLIR.API.DeviceGetKind(device.device)
    end
    return XLA.unsafe_string_and_free(str)
end

function XLA.get_local_device_id(device::Device)
    GC.@preserve device begin
        return MLIR.API.PjRtDeviceGetLocalDeviceId(device.device)
    end
end

function XLA.get_local_hardware_id(device::Device)
    GC.@preserve device begin
        return MLIR.API.PjRtDeviceGetLocalHardwareId(device.device)
    end
end

function XLA.is_addressable(device::Device)
    GC.@preserve device begin
        return MLIR.API.PjRtDeviceIsAddressable(device.device)
    end
end

function XLA.allocatorstats_internal(device::Device)
    ref = Ref{MLIR.API.JLAllocatorStats}()
    GC.@preserve device begin
        MLIR.API.PjRtDeviceGetAllocatorStats(device.device, ref)
    end
    return ref[]
end
