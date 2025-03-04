mutable struct Client <: XLA.AbstractClient
    client::Ptr{Cvoid}

    function Client(client::Ptr{Cvoid}; skip_check::Bool=false)
        skip_check || (@assert client != C_NULL)
        return new(client)
    end
end

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

# Different Backends
const cpu_client_count = Ref(0)
const gpu_client_count = Ref(0)
const tpu_client_count = Ref(0)

for (backend, counter) in (
    (:CPUClient, :cpu_client_count),
    (:GPUClient, :gpu_client_count),
    (:TPUClient, :tpu_client_count),
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

function MakeIFRTPJRTGPUClient(;
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

function MakeIFRTPJRTTPUClient(;
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
