mutable struct Client <: XLA.AbstractClient
    client::Ptr{Cvoid}

    function Client(client::Ptr{Cvoid}; skip_check::Bool=false)
        skip_check || (@assert client != C_NULL)
        return new(client)
    end
end

const NullClient = Client(C_NULL; skip_check=true)

function XLA.free_client(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        MLIR.API.FreeClient(client.client)
    end
    return nothing
end

function XLA.num_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return MLIR.API.ClientNumDevices(client.client)
    end
end

function XLA.num_addressable_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return MLIR.API.ClientNumAddressableDevices(client.client)
    end
end

function XLA.devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    ndevices = Int(XLA.num_devices(client))
    devices = Ref{NTuple{ndevices,Ptr{Cvoid}}}()
    GC.@preserve client devices begin
        MLIR.API.ClientGetDevices(client.client, devices)
    end
    return [Device(device) for device in devices[]]
end

function XLA.addressable_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    naddressable_devices = Int(XLA.num_addressable_devices(client))
    addressable_devices = Ref{NTuple{naddressable_devices,Ptr{Cvoid}}}()
    GC.@preserve client addressable_devices begin
        MLIR.API.ClientGetAddressableDevices(client.client, addressable_devices)
    end
    return [Device(device) for device in addressable_devices[]]
end

function XLA.process_index(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return MLIR.API.ClientProcessIndex(client.client)
    end
end

function XLA.get_device(client::Client, idx)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return Device(MLIR.API.ClientGetDevice(client.client, idx))
    end
end

function XLA.get_addressable_device(client::Client, idx)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return Device(MLIR.API.ClientGetAddressableDevice(client.client, idx))
    end
end

function XLA.platform_name(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        str = MLIR.API.ClientGetPlatformName(client.client)
    end
    return XLA.unsafe_string_and_free(str)
end

function XLA.cost_analysis(client::Client, hlo_module::XLA.HloModule)
    ref = Ref{MLIR.API.JLHloCostAnalysisProperties}()
    GC.@preserve client hlo_module begin
        MLIR.API.pjrt_hlo_module_cost_analysis_properties(
            client.client, hlo_module.ptr, ref
        )
    end
    return ref[]
end

function MakeClientUsingPluginAPI(
    library_path::String, device_type::String, client_name::String=uppercase(device_type)
)
    @assert isfile(library_path) "$(library_path) does not exist for $(device_type) PJRT \
                                  plugin."

    errstr = Ref{Cstring}()
    GC.@preserve errstr library_path device_type client_name begin
        client = MLIR.API.MakeClientUsingPluginAPI(
            device_type, library_path, client_name, errstr
        )
    end
    client == C_NULL && throw(AssertionError(unsafe_string(errstr[])))
    return client
end
