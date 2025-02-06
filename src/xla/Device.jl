struct Device
    device::Ptr{Cvoid}
end

function Base.show(io::IO, ::MIME"text/plain", device::Device)
    print(io, "Device($(device.device), name=\"$(DeviceToString(device))\")")
    return nothing
end

function client(device::Device)
    GC.@preserve device begin
        return Client(
            @ccall MLIR.API.mlir_c.DeviceToClient(device.device::Ptr{Cvoid})::Ptr{Cvoid}
        )
    end
end

"""
    device_ordinal(client::Client, device::Device)
    device_ordinal(client::Client, local_device_id::Int)

Given the device or local device id, return the corresponding global device ordinal in the client.
"""
function device_ordinal(client::Client, device::Device)
    return client.global_ordinals[DeviceGetLocalDeviceId(device) + 1]
end

function device_ordinal(client::Client, local_device_id::Int)
    return client.global_ordinals[local_device_id + 1]
end

function DeviceToString(device::Device)
    pjrtclient = client(device)
    platform_name = ClientGetPlatformName(pjrtclient)
    return "$(uppercase(platform_name)):$(device_ordinal(pjrtclient, device))"
end

function DeviceGetLocalDeviceId(device::Device)
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.PjRtDeviceGetLocalDeviceId(
            device.device::Ptr{Cvoid}
        )::Cint
    end
end
