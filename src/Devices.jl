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

# https://github.com/jax-ml/jax/blob/152099ee0ef31119f16f4c2dac50d84fcb1575ef/jax/_src/hardware_utils.py#L19-L55
const _GOOGLE_PCI_VENDOR_ID = "0x1ae0"
const _TPU_PCI_DEVICE_IDS = (
    # TPU v2, v3
    "0x0027",
    # No public name (plc)
    "0x0056",
    # TPU v4
    "0x005e",
    # TPU v5p
    "0x0062",
    # TPU v5e
    "0x0063",
    # TPU v6e
    "0x006f",
)

function has_tpu()
    Sys.islinux() || return false

    devices_dir = "/sys/bus/pci/devices/"
    isdir(devices_dir) || return false

    try
        for path in readdir(devices_dir; join=true, sort=false)
            if trim(read(joinpath(path, "vendor"), String)) == _GOOGLE_PCI_VENDOR_ID &&
                trim(read(joinpath(path, "device"), String)) in _TPU_PCI_DEVICE_IDS
                return true
            end
        end
    catch ex
        @warn "failed to query PCI device information" maxlog = 1 exception = (
            ex, catch_backtrace()
        )
    end

    return false
end
