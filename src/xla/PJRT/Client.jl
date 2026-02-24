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
    return GC.@preserve client MLIR.API.FreeClient(client.client)
end

function XLA.num_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    return GC.@preserve client MLIR.API.ClientNumDevices(client.client)
end

function XLA.num_addressable_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    return GC.@preserve client MLIR.API.ClientNumAddressableDevices(client.client)
end

function XLA.devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    ndevices = Int(XLA.num_devices(client))
    devices = Ref{NTuple{ndevices,Ptr{Cvoid}}}()
    GC.@preserve client MLIR.API.ClientGetDevices(client.client, devices)
    return [Device(device) for device in devices[]]
end

function XLA.addressable_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    naddressable_devices = Int(XLA.num_addressable_devices(client))
    addressable_devices = Ref{NTuple{naddressable_devices,Ptr{Cvoid}}}()
    GC.@preserve client MLIR.API.ClientGetAddressableDevices(
        client.client, addressable_devices
    )
    return [Device(device) for device in addressable_devices[]]
end

function XLA.process_index(client::Client)
    @assert client.client != C_NULL "Client is null"
    return GC.@preserve client MLIR.API.ClientProcessIndex(client.client)
end

function XLA.get_device(client::Client, idx)
    @assert client.client != C_NULL "Client is null"
    return Device(GC.@preserve client MLIR.API.ClientGetDevice(client.client, idx))
end

function XLA.get_addressable_device(client::Client, idx)
    @assert client.client != C_NULL "Client is null"
    return Device(
        GC.@preserve client MLIR.API.ClientGetAddressableDevice(client.client, idx)
    )
end

function XLA.platform_name(client::Client)
    @assert client.client != C_NULL "Client is null"
    return GC.@preserve client XLA.unsafe_string_and_free(
        MLIR.API.ClientGetPlatformName(client.client)
    )
end

function XLA.cost_analysis(client::Client, hlo_module::XLA.HloModule)
    ref = Ref{MLIR.API.JLHloCostAnalysisProperties}()
    GC.@preserve client hlo_module MLIR.API.PjRtHloModuleCostAnalysisProperties(
        client.client, hlo_module.ptr, ref
    )
    return ref[]
end

# Different Backends
const cpu_client_count = Ref(0)
const cuda_client_count = Ref(0)
const tpu_client_count = Ref(0)
const metal_client_count = Ref(0)
const tt_client_count = Ref(0)

for (backend, counter) in (
    (:CPUClient, :cpu_client_count),
    (:CUDAClient, :cuda_client_count),
    (:TPUClient, :tpu_client_count),
    (:MetalClient, :metal_client_count),
    (:TTClient, :tt_client_count),
)
    main_fn = Symbol(:Make, backend)
    @eval function $(backend)(args...; checkcount::Bool=true, kwargs...)
        if checkcount
            @assert $(counter)[] == 0
        end
        client = Client($(main_fn)(args...; kwargs...))
        XLA.LLVMclopts("-nvptx-fma-level=1")
        if checkcount
            # Only increment the counter if we successfully created a client
            $(counter)[] += 1
        end
        return client
    end
end

function MakeCPUClient(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    asynchronous::Bool=true,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    @assert num_nodes == 1 "`PJRT.MakeCPUClient` does not support num_nodes > 1"
    @assert distributed_runtime_client === nothing "`PJRT.MakeCPUClient` does not support \
                                                    distributed_runtime_client"
    return MLIR.API.MakeCPUClient(asynchronous, node_id)
end

function MakeCUDAClient(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    platform::String="gpu",
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    refstr = Ref{Cstring}()

    num_allowed_devices = allowed_devices === nothing ? 0 : length(allowed_devices)
    allowed_devices = allowed_devices === nothing ? C_NULL : allowed_devices
    distributed_runtime_client =
        distributed_runtime_client === nothing ? C_NULL : distributed_runtime_client.client

    client = MLIR.API.MakeGPUClient(
        node_id,
        num_nodes,
        allowed_devices,
        num_allowed_devices,
        XLA.XLA_REACTANT_GPU_MEM_FRACTION[],
        XLA.XLA_REACTANT_GPU_PREALLOCATE[],
        platform,
        refstr,
        distributed_runtime_client,
    )

    client == C_NULL && throw(AssertionError(unsafe_string(refstr[])))
    return client
end

function MakeTPUClient(;
    tpu_path::String,
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    @assert node_id == 0 "`PJRT.MakeTPUClient` does not support node_id"
    @assert num_nodes == 1 "`PJRT.MakeTPUClient` does not support num_nodes > 1"
    @assert distributed_runtime_client === nothing "`PJRT.MakeTPUClient` does not support \
                                                    distributed_runtime_client"

    # LibTPU has its own internal copy of XLA which does not read the regular XLA flags
    if !haskey(ENV, "LIBTPU_INIT_ARGS")
        xla_flags = "--xla_enable_enzyme_comms_opt=true"
        if haskey(ENV, "XLA_FLAGS")
            xla_flags = xla_flags * " " * ENV["XLA_FLAGS"]
        end
        ENV["LIBTPU_INIT_ARGS"] = xla_flags
    end

    return MakeClientUsingPluginAPI(tpu_path, "tpu", "TPU")
end

function MakeMetalClient(;
    metal_pjrt_plugin_path::String,
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    @assert node_id == 0 "`PJRT.MakeMetalClient` does not support node_id"
    @assert num_nodes == 1 "`PJRT.MakeMetalClient` does not support num_nodes > 1"
    @assert distributed_runtime_client === nothing "`PJRT.MakeMetalClient` does not support \
                                                    distributed_runtime_client"

    return MakeClientUsingPluginAPI(metal_pjrt_plugin_path, "metal", "METAL")
end

function MakeTTClient(;
    tt_pjrt_plugin_path::String,
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    @assert node_id == 0 "`PJRT.MakeTTClient` does not support node_id"
    @assert num_nodes == 1 "`PJRT.MakeTTClient` does not support num_nodes > 1"
    @assert distributed_runtime_client === nothing "`PJRT.MakeTTClient` does not support \
                                                    distributed_runtime_client"

    return MakeClientUsingPluginAPI(tt_pjrt_plugin_path, "tt", "TT")
end

function MakeClientUsingPluginAPI(
    library_path::String, device_type::String, client_name::String=uppercase(device_type)
)
    @assert isfile(library_path) "$(library_path) does not exist for $(device_type) PJRT \
                                  plugin."

    errstr = Ref{Cstring}()
    client = MLIR.API.MakeClientUsingPluginAPI(
        device_type, library_path, client_name, errstr
    )
    client == C_NULL && throw(AssertionError(unsafe_string(errstr[])))
    return client
end
