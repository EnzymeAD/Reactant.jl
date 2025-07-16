# Client
mutable struct DistributedRuntimeClient
    client::Ptr{Cvoid}

    function DistributedRuntimeClient(client::Ptr{Cvoid})
        @assert client != C_NULL
        return finalizer(free_distributed_runtime_client, new(client))
    end
end

function DistributedRuntimeClient(
    coordinator_bind_address::String,
    process_id::Integer;
    rpc_timeout_in_seconds::Integer=120,
    shutdown_timeout_in_minutes::Integer=5,
    heartbeat_interval_in_seconds::Integer=10,
    use_compression::Bool=true,
)
    GC.@preserve coordinator_bind_address begin
        client = @ccall MLIR.API.mlir_c.GetDistributedRuntimeClient(
            coordinator_bind_address::Cstring,
            Int32(process_id)::Int32,
            Int32(rpc_timeout_in_seconds)::Int32,
            Int32(shutdown_timeout_in_minutes)::Int32,
            Int32(heartbeat_interval_in_seconds)::Int32,
            use_compression::Bool,
        )::Ptr{Cvoid}
    end
    return DistributedRuntimeClient(client)
end

function free_distributed_runtime_client(client::DistributedRuntimeClient)
    GC.@preserve client begin
        @ccall MLIR.API.mlir_c.free_distributed_runtime_client(
            client.client::Ptr{Cvoid}
        )::Cvoid
    end
end

function connect(client::DistributedRuntimeClient)
    GC.@preserve client begin
        @ccall MLIR.API.mlir_c.distributed_runtime_client_connect(
            client.client::Ptr{Cvoid}
        )::Cvoid
    end
end

function shutdown(client::DistributedRuntimeClient)
    GC.@preserve client begin
        @ccall MLIR.API.mlir_c.distributed_runtime_client_shutdown(
            client.client::Ptr{Cvoid}
        )::Cvoid
    end
end

# Service
mutable struct DistributedRuntimeService
    service::Ptr{Cvoid}

    function DistributedRuntimeService(service::Ptr{Cvoid})
        @assert service != C_NULL
        return finalizer(free_distributed_runtime_service, new(service))
    end
end

function DistributedRuntimeService(
    coordinator_bind_address::String,
    num_nodes::Integer;
    heartbeat_interval_in_seconds::Integer=10,
    cluster_register_timeout_in_minutes::Integer=60,
    shutdown_timeout_in_minutes::Integer=5,
)
    GC.@preserve coordinator_bind_address begin
        service = @ccall MLIR.API.mlir_c.GetDistributedRuntimeService(
            coordinator_bind_address::Cstring,
            Cint(num_nodes)::Cint,
            Int32(heartbeat_interval_in_seconds)::Int32,
            Int32(cluster_register_timeout_in_minutes)::Int32,
            Int32(shutdown_timeout_in_minutes)::Int32,
        )::Ptr{Cvoid}
    end
    return DistributedRuntimeService(service)
end

function free_distributed_runtime_service(service::DistributedRuntimeService)
    GC.@preserve service begin
        @ccall MLIR.API.mlir_c.free_distributed_runtime_service(
            service.service::Ptr{Cvoid}
        )::Cvoid
    end
end

function shutdown(service::DistributedRuntimeService)
    GC.@preserve service begin
        @ccall MLIR.API.mlir_c.distributed_runtime_service_shutdown(
            service.service::Ptr{Cvoid}
        )::Cvoid
    end
end

# Global State
@kwdef mutable struct State
    process_id::Int = 0
    num_processes::Int = 1
    local_gpu_device_ids::Union{Nothing,Vector{Int}} = nothing
    service::Union{Nothing,DistributedRuntimeService} = nothing
    client::Union{Nothing,DistributedRuntimeClient} = nothing
    coordinator_address::Union{Nothing,String} = nothing
    coordinator_bind_address::Union{Nothing,String} = nothing
end

function shutdown(state::State)
    if state.service !== nothing
        shutdown(state.service)
        state.service = nothing
    end
    if state.client !== nothing
        shutdown(state.client)
        state.client = nothing
    end
end

function update!(
    state::State;
    coordinator_address::String,
    num_processes::Int,
    process_id::Int,
    local_gpu_device_ids::Union{Nothing,Vector{Int}},
    coordinator_bind_address::Union{Nothing,String}=nothing,
    cluster_register_timeout_in_minutes::Integer=60,
    rpc_timeout_in_seconds::Integer=120,
    shutdown_timeout_in_minutes::Integer=5,
    heartbeat_interval_in_seconds::Integer=10,
    use_compression::Bool=true,
)
    @assert 0 ≤ process_id < num_processes

    state.coordinator_address = coordinator_address
    if local_gpu_device_ids !== nothing
        state.local_gpu_device_ids = local_gpu_device_ids
    end
    state.process_id = process_id
    state.num_processes = num_processes

    if coordinator_bind_address === nothing
        if haskey(ENV, "REACTANT_COORDINATOR_BIND_ADDRESS")
            coordinator_bind_address = ENV["REACTANT_COORDINATOR_BIND_ADDRESS"]
        else
            coordinator_bind_address =
                "[::]:" * rsplit(coordinator_address, ":"; limit=2)[2]
        end
    end
    state.coordinator_bind_address = coordinator_bind_address

    if process_id == 0
        @assert state.service === nothing "`Reactant.Distributed.initialize` should only \
                                           be called once."
        @debug "[PID $(process_id)] Starting Reactant distributed service on \
                $(coordinator_bind_address)"
        state.service = DistributedRuntimeService(
            coordinator_bind_address,
            num_processes;
            heartbeat_interval_in_seconds,
            cluster_register_timeout_in_minutes,
            shutdown_timeout_in_minutes,
        )
    end

    # Check for proxy variables that might cause a hang
    proxy_vars = filter(Base.Fix1(occursin, "_proxy") ∘ lowercase, keys(ENV))
    if length(proxy_vars) > 0
        vars = join(proxy_vars, ", ")
        @warn "Reactant detected proxy variable(s) in the environment as distributed \
               setup: $(vars). On some systems, this may cause a hang of `XLA.update!` and \
               you may need to unset the proxy variables."
    end

    @assert state.client === nothing "`Reactant.Distributed.initialize` should only be \
                                      called once."
    state.client = DistributedRuntimeClient(
        coordinator_address,
        process_id;
        rpc_timeout_in_seconds,
        shutdown_timeout_in_minutes,
        heartbeat_interval_in_seconds,
        use_compression,
    )
    @debug "[PID $(process_id)] Connecting to Reactant distributed service on \
            $(coordinator_address)"
    connect(state.client)
    @debug "[PID $(process_id)] Connected to Reactant distributed service on \
            $(coordinator_address)"

    return nothing
end
