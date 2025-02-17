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
    str_jl = unsafe_string(str)
    @ccall free(str::Cstring)::Cvoid
    return str_jl
end

function XLA.get_local_device_id(device::Device)
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.PjRtDeviceGetLocalDeviceId(
            device.device::Ptr{Cvoid}
        )::Cint
    end
end
