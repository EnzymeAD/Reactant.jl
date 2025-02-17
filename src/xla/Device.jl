abstract type AbstractDevice end

function Base.show(io::IO, ::MIME"text/plain", device::AbstractDevice)
    mod = parentmodule(device)
    print(io, "$(mod).Device($(device.device), name=\"$(string(device))\")")
    return nothing
end

function device end
function get_local_device_id end

"""
    device_ordinal(client::XLA.AbstractClient, device::Device)
    device_ordinal(client::XLA.AbstractClient, local_device_id::Int)

Given the device or local device id, return the corresponding global device ordinal in the client.
"""
function device_ordinal end
