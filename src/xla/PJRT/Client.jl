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
        @ccall MLIR.API.mlir_c.FreeClient(client.client::Ptr{Cvoid})::Cvoid
    end
end

function XLA.num_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientNumDevices(client.client::Ptr{Cvoid})::Cint
    end
end

function XLA.num_addressable_devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientNumAddressableDevices(
            client.client::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.devices(client::Client)
    @assert client.client != C_NULL "Client is null"
    ndevices = Int(XLA.num_devices(client))
    devices = Ref{NTuple{ndevices,Ptr{Cvoid}}}()
    GC.@preserve client devices begin
        @ccall MLIR.API.mlir_c.ClientGetDevices(
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
        @ccall MLIR.API.mlir_c.ClientGetAddressableDevices(
            client.client::Ptr{Cvoid}, addressable_devices::Ptr{Ptr{Cvoid}}
        )::Cvoid
    end
    return [Device(device) for device in addressable_devices[]]
end

function XLA.process_index(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientProcessIndex(client.client::Ptr{Cvoid})::Cint
    end
end

function XLA.get_device(client::Client, idx)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ClientGetDevice(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function XLA.get_addressable_device(client::Client, idx)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ClientGetAddressableDevice(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function XLA.platform_name(client::Client)
    @assert client.client != C_NULL "Client is null"
    GC.@preserve client begin
        str = @ccall MLIR.API.mlir_c.ClientGetPlatformName(
            client.client::Ptr{Cvoid}
        )::Cstring
    end
    return XLA.unsafe_string_and_free(str)
end

function XLA.cost_analysis(client::Client, hlo_module::XLA.HloModule)
    GC.@preserve client hlo_module begin
        ref = Ref{XLA.HloCostAnalysisProperties}()
        @ccall MLIR.API.mlir_c.pjrt_hlo_module_cost_analysis_properties(
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

    return @ccall MLIR.API.mlir_c.MakeCPUClient(
        asynchronous::UInt8, node_id::Cint
    )::Ptr{Cvoid}
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

    GC.@preserve refstr allowed_devices distributed_runtime_client begin
        client = @ccall MLIR.API.mlir_c.MakeGPUClient(
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
    GC.@preserve errstr library_path device_type client_name begin
        client = @ccall MLIR.API.mlir_c.MakeClientUsingPluginAPI(
            device_type::Cstring,
            library_path::Cstring,
            client_name::Cstring,
            errstr::Ptr{Cstring},
        )::Ptr{Cvoid}
    end

    client == C_NULL && throw(AssertionError(unsafe_string(errstr[])))
    return client
end
