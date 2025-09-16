abstract type AbstractDevice end

function Base.show(io::IO, ::MIME"text/plain", device::D) where {D<:AbstractDevice}
    print(io, "$(parentmodule(D)).Device($(device.device), \"$(string(device))\")")
    return nothing
end

function device end
function get_local_device_id end
function device_kind end
function default_memory end
function memories end
function is_addressable end

"""
    device_ordinal(device::Device)

Given the device, return the corresponding global device ordinal in the client.
"""
function device_ordinal end

function Base.string(device::AbstractDevice)
    client = client(device)
    pname = platform_name(client)
    return "$(uppercase(pname)):$(device_ordinal(device)) $(device_kind(device))"
end

# Fallback method, preferably all device implementations overload this
function is_addressable(device::AbstractDevice)
    return device ∈ addressable_devices(client(device))
end
