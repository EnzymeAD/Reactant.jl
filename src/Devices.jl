"""
    devices(backend::String)
    devices(backend::XLA.AbstractClient = XLA.default_backend())

Return a list of devices available for the given client.
"""
devices(backend::String) = devices(XLA.client(backend))

devices(client::XLA.AbstractClient=XLA.default_backend()) = XLA.devices(client)

"""
    addressable_devices(backend::String)
    addressable_devices(backend::XLA.AbstractClient = XLA.default_backend())

Return a list of addressable devices available for the given client.
"""
addressable_devices(backend::String) = addressable_devices(XLA.client(backend))

function addressable_devices(client::XLA.AbstractClient=XLA.default_backend())
    return XLA.addressable_devices(client)
end
