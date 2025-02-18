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

# Device List
struct BasicDeviceList <: AbstractVector{Device}
    ptr::Ptr{Cvoid}

    function BasicDeviceList(devices::AbstractVector{Device})
        GC.@preserve devices begin
            ptr = @ccall MLIR.API.mlir_c.ifrt_CreateBasicDeviceListFromDevices(
                [d.device for d in devices]::Ptr{Ptr{Cvoid}}, length(devices)::Int32
            )::Ptr{Cvoid}
        end
        return new(ptr)
    end
end

function Base.getindex(device_list::BasicDeviceList, index::Integer)
    if !(1 ≤ index ≤ length(device_list))
        throw(BoundsError(device_list, index))
    end
    GC.@preserve device_list begin
        device_ptr = @ccall MLIR.API.mlir_c.ifrt_BasicDeviceListGetDevice(
            device_list.ptr::Ptr{Cvoid}, (index - 1)::Int32
        )::Ptr{Cvoid}
    end
    return Device(device_ptr)
end

function Base.size(device_list::BasicDeviceList)
    GC.@preserve device_list begin
        len = @ccall MLIR.API.mlir_c.ifrt_BasicDeviceListSize(
            device_list.ptr::Ptr{Cvoid}
        )::Int32
    end
    return (len,)
end

function Base.string(device_list::BasicDeviceList)
    GC.@preserve device_list begin
        str = @ccall MLIR.API.mlir_c.ifrt_BasicDeviceListToString(
            device_list.ptr::Ptr{Cvoid}
        )::Cstring
    end
    return XLA.unsafe_string_and_free(str)
end
