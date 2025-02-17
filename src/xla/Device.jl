abstract type AbstractDevice end

function Base.show(io::IO, ::MIME"text/plain", device::D) where {D<:AbstractDevice}
    print(io, "$(parentmodule(D)).Device($(device.device), \"$(string(device))\")")
    return nothing
end

function device end
function get_local_device_id end
function device_kind end

"""
    device_ordinal(client::XLA.AbstractClient, device::Device)
    device_ordinal(client::XLA.AbstractClient, local_device_id::Int)

Given the device or local device id, return the corresponding global device ordinal in the client.
"""
function device_ordinal end

function Base.string(device::AbstractDevice)
    client = XLA.client(device)
    pname = XLA.platform_name(client)
    return "$(uppercase(pname)):$(device_ordinal(client, device)) $(device_kind(device))"
end
