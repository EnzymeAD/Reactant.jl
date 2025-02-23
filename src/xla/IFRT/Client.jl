mutable struct Client <: XLA.AbstractClient
    client::Ptr{Cvoid}

    function Client(client::Ptr{Cvoid}; skip_check::Bool=false)
        skip_check || (@assert client != C_NULL)
        return new(client)
    end
end

function XLA.free_client(client::Client)
    GC.@preserve client begin
        @ccall MLIR.API.mlir_c.ifrt_FreeClient(client.client::Ptr{Cvoid})::Cvoid
    end
end

function XLA.num_devices(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ifrt_client_device_count(
            client.client::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.num_addressable_devices(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ifrt_client_addressable_device_count(
            client.client::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.process_index(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ifrt_ClientProcessIndex(
            client.client::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.get_device(client::Client, idx)
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ifrt_client_lookup_device(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function XLA.get_addressable_device(client::Client, idx)
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ifrt_client_lookup_addressable_device(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function XLA.platform_name(client::Client)
    GC.@preserve client begin
        str = @ccall MLIR.API.mlir_c.ifrt_ClientGetPlatformName(
            client.client::Ptr{Cvoid}
        )::Cstring
    end
    return XLA.unsafe_string_and_free(str)
end

function XLA.devices(client::Client)
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
    naddressable_devices = Int(XLA.num_addressable_devices(client))
    addressable_devices = Ref{NTuple{naddressable_devices,Ptr{Cvoid}}}()
    GC.@preserve client addressable_devices begin
        @ccall MLIR.API.mlir_c.ifrt_client_addressable_devices(
            client.client::Ptr{Cvoid}, addressable_devices::Ptr{Ptr{Cvoid}}
        )::Cvoid
    end
    return [Device(device) for device in addressable_devices[]]
end

# Different Backends
const cpu_client_count = Ref(0)
const gpu_client_count = Ref(0)
const tpu_client_count = Ref(0)

for (backend, counter) in (
    (:CPUClient, :cpu_client_count),
    (:GPUClient, :gpu_client_count),
    (:TPUClient, :tpu_client_count),
)
    pjrt_fn = Symbol(:MakePJRT, backend)
    proxy_fn = Symbol(:MakeProxy, backend)

    # XXX: call it something other than `version`??
    @eval function $(backend)(
        args...; checkcount::Bool=true, version::Symbol=:PJRT, num_nodes=1, kwargs...
    )
        checkcount && (@assert iszero($(counter)[]))
        if version == :PJRT
            client, refstr = $(pjrt_fn)(args...; num_nodes, kwargs...)
        elseif version == :Proxy
            client, refstr = $(proxy_fn)(args...; num_nodes, kwargs...)
        else
            error("Invalid version: $version")
        end
        client == C_NULL && throw(AssertionError(unsafe_string(refstr[])))
        XLA.LLVMclopts("-nvptx-fma-level=1")
        # Only increment the counter if we successfully created a client
        checkcount && ($(counter)[] += 1)
        return Client(client)
    end
end

function MakePJRTCPUClient(;
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

function MakeProxyCPUClient(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    asynchronous::Bool=true,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
    coordinator_address::String=XLA.global_state.coordinator_address,
)
    refstr = Ref{Cstring}()
    distributed_runtime_client =
        distributed_runtime_client === nothing ? C_NULL : distributed_runtime_client.client

    GC.@preserve refstr distributed_runtime_client begin
        @ccall MLIR.API.mlir_c.ifrt_proxy_grpc_server_create_from_ifrt_client_factory_cpu(
            coordinator_address::Cstring,
            asynchronous::UInt8,
            node_id::Cint,
            num_nodes::Cint,
            distributed_runtime_client::Ptr{Cvoid},
            refstr::Ptr{Cstring},
        )::Ptr{Cvoid}

        client = @ccall MLIR.API.mlir_c.ifrt_proxy_create_client(
            coordinator_address::Cstring, 12::Cint
        )::Ptr{Cvoid}
    end

    return client, refstr
end

function MakePJRTGPUClient(;
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
            allowed_devices::Ptr{Cvoid},
            num_allowed_devices::Cint,
            XLA.XLA_REACTANT_GPU_MEM_FRACTION[]::Cdouble,
            XLA.XLA_REACTANT_GPU_PREALLOCATE[]::Bool,
            platform::Cstring,
            refstr::Ptr{Cstring},
            distributed_runtime_client::Ptr{Cvoid},
        )::Ptr{Cvoid}
    end

    return client, refstr
end

# TODO: Proxy GPU client

function MakePJRTTPUClient(;
    tpu_path::String,
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client::Union{Nothing,XLA.DistributedRuntimeClient}=nothing,
)
    refstr = Ref{Cstring}()
    distributed_runtime_client =
        distributed_runtime_client === nothing ? C_NULL : distributed_runtime_client.client

    GC.@preserve refstr distributed_runtime_client begin
        client = @ccall MLIR.API.mlir_c.ifrt_make_pjrt_tpu_client(
            tpu_path::Cstring,
            refstr::Ptr{Cstring},
            node_id::Cint,
            num_nodes::Cint,
            distributed_runtime_client::Ptr{Cvoid},
        )::Ptr{Cvoid}
    end

    return client, refstr
end

# TODO: Proxy TPU client
