"""
    devices(backend::String)
    devices(backend::XLA.Client = XLA.default_backend[])

Return a list of devices available on the backend.
"""
devices(backend::String) = devices(XLA.backends[backend])

function devices(client::XLA.Client=XLA.default_backend[])
    ndevices = XLA.ClientNumDevices(client)
    return [XLA.ClientGetDevice(client, i - 1) for i in 1:ndevices]
end

"""
    addressable_devices(backend::String)
    addressable_devices(backend::XLA.Client = XLA.default_backend[])

Return a list of addressable devices available on the backend.
"""
addressable_devices(backend::String) = addressable_devices(XLA.backends[backend])

function addressable_devices(client::XLA.Client=XLA.default_backend[])
    ndevices = XLA.ClientNumAddressableDevices(client)
    return [XLA.ClientGetAddressableDevice(client, i - 1) for i in 1:ndevices]
end
