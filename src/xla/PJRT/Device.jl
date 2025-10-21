struct Device <: XLA.AbstractDevice
    device::Ptr{Cvoid}
end

function XLA.client(device::Device)
    GC.@preserve device begin
        return Client(
            @ccall MLIR.API.mlir_c.DeviceToClient(device.device::Ptr{Cvoid})::Ptr{Cvoid}
        )
    end
end

function XLA.device_ordinal(device::Device)
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.PjRtDeviceGetLocalDeviceId(
            device.device::Ptr{Cvoid}
        )::Int64
    end
end

function XLA.device_kind(device::Device)
    GC.@preserve device begin
        str = @ccall MLIR.API.mlir_c.DeviceGetKind(device.device::Ptr{Cvoid})::Cstring
    end
    return XLA.unsafe_string_and_free(str)
end

function XLA.get_local_device_id(device::Device)
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.PjRtDeviceGetLocalDeviceId(
            device.device::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.get_local_hardware_id(device::Device)
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.PjRtDeviceGetLocalHardwareId(
            device.device::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.is_addressable(device::Device)
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.pjrt_device_is_addressable(
            device.device::Ptr{Cvoid}
        )::Bool
    end
end

function XLA.allocatorstats_internal(device::Device)
    ref = Ref{XLA.JLAllocatorStats}()
    @ccall MLIR.API.mlir_c.PjRtDeviceGetAllocatorStats(
        device.device::Ptr{Cvoid}, ref::Ptr{Cvoid}
    )::Cvoid
    return ref[]
end
