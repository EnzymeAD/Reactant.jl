abstract type AbstractDevice end

struct CPUDevice <: AbstractDevice
    idx::Int
    client::XLA.Client
end

struct CUDADevice <: AbstractDevice
    idx::Int
    client::XLA.Client
end

struct TPUDevice <: AbstractDevice
    idx::Int
    client::XLA.Client
end

XLA.Device(d::AbstractDevice) = convert(XLA.Device, d)
Base.convert(::Type{XLA.Device}, d::AbstractDevice) = XLA.ClientGetDevice(d.client, d.idx)

XLA.allocatorstats(d::AbstractDevice) = XLA.allocatorstats(convert(XLA.Device, d))

"""
    devices(backend::String)
    devices(backend::XLA.Client = XLA.default_backend[])

Return a list of devices available on the backend.
"""
devices(backend::String) = devices(XLA.backends[backend])

function devices(client::XLA.Client = XLA.default_backend[])
    platform_name = XLA.ClientGetPlatformName(client)
    dev = if platform_name == "cpu"
        CPUDevice
    elseif platform_name == "cuda"
        CUDADevice
    elseif platform_name == "tpu"
        TPUDevice
    else
        error("Unsupported platform: $(platform_name)")
    end
    return [dev(i - 1, client) for i in 1:XLA.ClientNumDevices(client)]
end
