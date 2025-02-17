struct Device <: XLA.AbstractDevice
    device::Ptr{Cvoid}
end

function XLA.client(device::Device)
    GC.@preserve device begin
        return Client(
            @ccall MLIR.API.mlir_c.DeviceToClient(
                device.device::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

# TODO: Can be defined on the AbstractDevice?
function XLA.device_ordinal(client::Client, device::Device)
    return XLA.device_ordinal(client, XLA.get_local_device_id(device))
end
function XLA.device_ordinal(client::Client, local_device_id::Integer)
    return client.global_ordinals[local_device_id + 1]
end

function Base.string(device::Device)
    client = XLA.client(device)
    platform_name = XLA.platform_name(client)
    return "$(uppercase(platform_name)):$(XLA.device_ordinal(client, device))"
end

function XLA.get_local_device_id(device::Device)
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.PjRtDeviceGetLocalDeviceId(
            device.device::Ptr{Cvoid}
        )::Cint
    end
end
