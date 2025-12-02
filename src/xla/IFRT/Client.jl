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
        @ccall MLIR.API.mlir_c.ifrt_FreeClient(client.client::Ptr{Cvoid})::Cvoid
    end
end

function XLA.num_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ifrt_client_device_count(
            client.client::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.num_addressable_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ifrt_client_addressable_device_count(
            client.client::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.process_index(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ifrt_ClientProcessIndex(
            client.client::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.get_device(client::Client, idx)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ifrt_client_lookup_device(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function XLA.get_addressable_device(client::Client, idx)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ifrt_client_lookup_addressable_device(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function XLA.platform_name(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        str = @ccall MLIR.API.mlir_c.ifrt_ClientGetPlatformName(
            client.client::Ptr{Cvoid}
        )::Cstring
    end
    return XLA.unsafe_string_and_free(str)
end

function XLA.devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    ndevices = Int(XLA.num_devices(client))
    devices = Ref{NTuple{ndevices,Ptr{Cvoid}}}()
    GC.@preserve client devices begin
        @ccall MLIR.API.mlir_c.ifrt_client_devices(
            client.client::Ptr{Cvoid}, devices::Ptr{Ptr{Cvoid}}
        )::Cvoid
    end
    return [Device(device) for device in devices[]]
end

function XLA.addressable_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    naddressable_devices = Int(XLA.num_addressable_devices(client))
    addressable_devices = Ref{NTuple{naddressable_devices,Ptr{Cvoid}}}()
    GC.@preserve client addressable_devices begin
        @ccall MLIR.API.mlir_c.ifrt_client_addressable_devices(
            client.client::Ptr{Cvoid}, addressable_devices::Ptr{Ptr{Cvoid}}
        )::Cvoid
    end
    return [Device(device) for device in addressable_devices[]]
end

function XLA.cost_analysis(client::Client, hlo_module::XLA.HloModule)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client hlo_module begin
        ref = Ref{XLA.HloCostAnalysisProperties}()
        @ccall MLIR.API.mlir_c.ifrt_hlo_module_cost_analysis_properties(
            client.client::Ptr{Cvoid}, hlo_module.ptr::Ptr{Cvoid}, ref::Ptr{Cvoid}
        )::Cvoid
        return ref[]
    end
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
    main_fn = Symbol(:MakeIFRTPJRT, backend)
    @eval function $(backend)(args...; checkcount::Bool=true, kwargs...)
        if checkcount
            @assert $(counter)[] == 0
        end
        client, refstr = $(main_fn)(args...; kwargs...)
        client == C_NULL && throw(AssertionError(unsafe_string(refstr[])))
        XLA.LLVMclopts("-nvptx-fma-level=1")
        if checkcount
            # Only increment the counter if we successfully created a client
            $(counter)[] += 1
        end
        return Client(client)
    end
end

function MakeIFRTPJRTCPUClient(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    asynchronous::Bool=true,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    refstr = Ref{Cstring}()
    distributed_runtime_client =
        distributed_runtime_client === nothing ? C_NULL : distributed_runtime_client.client

    GC.@preserve refstr distributed_runtime_client begin
        client = @ccall MLIR.API.mlir_c.ifrt_make_pjrt_cpu_client(
            asynchronous::UInt8,
            node_id::Cint,
            num_nodes::Cint,
            distributed_runtime_client::Ptr{Cvoid},
            refstr::Ptr{Cstring},
        )::Ptr{Cvoid}
    end

    return client, refstr
end

function MakeIFRTPJRTCUDAClient(;
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

    GC.@preserve refstr allowed_devices distributed_runtime_client begin
        client = @ccall MLIR.API.mlir_c.ifrt_make_pjrt_gpu_client(
            node_id::Cint,
            num_nodes::Cint,
            allowed_devices::Ptr{Int64},
            num_allowed_devices::Int64,
            XLA.XLA_REACTANT_GPU_MEM_FRACTION[]::Cdouble,
            XLA.XLA_REACTANT_GPU_PREALLOCATE[]::Bool,
            platform::Cstring,
            refstr::Ptr{Cstring},
            distributed_runtime_client::Ptr{Cvoid},
        )::Ptr{Cvoid}
    end

    return client, refstr
end

function MakeIFRTPJRTTPUClient(;
    tpu_path::String,
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    return MakeIFRTPJRTClientViaPluginAPI(
        tpu_path, "tpu", "TPU"; node_id, num_nodes, distributed_runtime_client
    )
end

function MakeIFRTPJRTMetalClient(;
    metal_pjrt_plugin_path::String,
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    return MakeIFRTPJRTClientViaPluginAPI(
        metal_pjrt_plugin_path,
        "metal",
        "METAL";
        node_id,
        num_nodes,
        distributed_runtime_client,
    )
end

function MakeIFRTPJRTTTClient(;
    tt_pjrt_plugin_path::String,
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    return MakeIFRTPJRTClientViaPluginAPI(
        tt_pjrt_plugin_path, "tt", "TT"; node_id, num_nodes, distributed_runtime_client
    )
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
        client = @ccall MLIR.API.mlir_c.ifrt_pjrt_make_client_with_default_kv_store(
            pjrt_client::Ptr{Cvoid},
            node_id::Cint,
            num_nodes::Cint,
            distributed_runtime_client::Ptr{Cvoid},
            errstr::Ptr{Cstring},
            device_type::Cstring,
        )::Ptr{Cvoid}
    end

    return client, errstr
end
