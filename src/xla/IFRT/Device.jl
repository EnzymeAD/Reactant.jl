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

function XLA.device_ordinal(::Client, device::Device)
    GC.@preserve device begin
        return @ccall MLIR.API.mlir_c.ifrt_DeviceGetGlobalDeviceId(
            device.device::Ptr{Cvoid}
        )::Int64
    end
end
function XLA.device_ordinal(client::Client, device_id::Integer)
    return XLA.device_ordinal(client, XLA.get_addressable_device(client, device_id))
end

function XLA.device_kind(device::Device)
    GC.@preserve device begin
        str = @ccall MLIR.API.mlir_c.ifrt_DeviceGetKind(device.device::Ptr{Cvoid})::Cstring
    end
    str_jl = unsafe_string(str)
    @ccall free(str::Cstring)::Cvoid
    return str_jl
end

function XLA.get_local_device_id(::Device)
    return error("Not implemented for ifrt devices")
end
