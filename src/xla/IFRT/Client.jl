struct Client <: XLA.AbstractClient
    client::Ptr{Cvoid}

    function Client(client::Ptr{Cvoid}; skip_check::Bool=false)
        skip_check || (@assert client != C_NULL)
        return new(client)
    end
end

const NullClient = Client(C_NULL; skip_check=true)

function XLA.free_client(client::Client)
    @assert client.client != C_NULL "Client is null"
    @debug "[GETPID $(getpid())] Freeing Client $client"
    return MLIR.API.ifrt_FreeClient(client.client)
end

function XLA.num_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    return MLIR.API.ifrt_client_device_count(client.client)
end

function XLA.num_addressable_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    return MLIR.API.ifrt_client_addressable_device_count(client.client)
end

function XLA.process_index(client::Client)
    @assert client.client != C_NULL "Client is null"
    return MLIR.API.ifrt_client_process_index(client.client)
end

function XLA.get_device(client::Client, idx)
    @assert client.client != C_NULL "Client is null"
    return Device(MLIR.API.ifrt_client_lookup_device(client.client, idx))
end

function XLA.get_addressable_device(client::Client, idx)
    @assert client.client != C_NULL "Client is null"
    return Device(MLIR.API.ifrt_client_lookup_addressable_device(client.client, idx))
end

function XLA.platform_name(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        str = MLIR.API.ifrt_ClientGetPlatformName(client.client)
    end
    return XLA.unsafe_string_and_free(str)
end

function XLA.devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    ndevices = Int(XLA.num_devices(client))
    devices = Ref{NTuple{ndevices,Ptr{Cvoid}}}()
    GC.@preserve client begin
        MLIR.API.ifrt_client_devices(client.client, devices)
    end
    return [Device(device) for device in devices[]]
end

function XLA.addressable_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    naddressable_devices = Int(XLA.num_addressable_devices(client))
    addressable_devices = Ref{NTuple{naddressable_devices,Ptr{Cvoid}}}()
    GC.@preserve client begin
        MLIR.API.ifrt_client_addressable_devices(client.client, addressable_devices)
    end
    return [Device(device) for device in addressable_devices[]]
end

function XLA.cost_analysis(client::Client, hlo_module::XLA.HloModule)
    @assert client.client != C_NULL "Client is null"
    ref = Ref{MLIR.API.JLHloCostAnalysisProperties}()
    GC.@preserve client hlo_module begin
        MLIR.API.ifrt_hlo_module_cost_analysis_properties(
            client.client, hlo_module.ptr, ref
        )
    end
    return ref[]
end

function MakeIFRTPJRTClientViaPluginAPI(
    library_path::String,
    device_type::String,
    client_name::String=uppercase(device_type);
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    pjrt_client = XLA.PJRT.MakeClientUsingPluginAPI(library_path, device_type, client_name)

    distributed_runtime_client =
        distributed_runtime_client === nothing ? C_NULL : distributed_runtime_client.client

    errstr = Ref{Cstring}()
    GC.@preserve pjrt_client errstr distributed_runtime_client device_type begin
        client = MLIR.API.ifrt_pjrt_make_client_with_default_kv_store(
            pjrt_client, node_id, num_nodes, distributed_runtime_client, errstr, device_type
        )
    end
    return client, errstr
end
