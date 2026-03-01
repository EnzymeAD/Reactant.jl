module CPU

using ..Reactant: Reactant

using ..Registration: register_backend

function make_pjrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    @assert num_nodes == 1 "`make_pjrt_client` does not support num_nodes > 1"
    @assert distributed_runtime_client === nothing "`make_pjrt_client` does not \
                                                    support distributed_runtime_client"

    if allowed_devices !== nothing
        @debug "CPUClient doesn't support allowed_devices. Ignoring the kwarg."
    end

    client = Reactant.MLIR.API.MakeCPUClient(true, node_id)
    return Reactant.XLA.PJRT.Client(client)
end

function make_ifrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    refstr = Ref{Cstring}()
    distributed_runtime_client =
        distributed_runtime_client === nothing ? C_NULL : distributed_runtime_client.client

    if allowed_devices !== nothing
        @debug "CPUClient doesn't support allowed_devices. Ignoring the kwarg."
    end

    GC.@preserve refstr distributed_runtime_client begin
        client = Reactant.MLIR.API.ifrt_make_pjrt_cpu_client(
            true, node_id, num_nodes, distributed_runtime_client, refstr
        )
    end

    client == C_NULL && throw(AssertionError(unsafe_string(refstr[])))
    return Reactant.XLA.IFRT.Client(client)
end

function __init__()
    register_backend(
        "cpu";
        priority=100,
        pjrt_initialize_function=make_pjrt_client,
        ifrt_initialize_function=make_ifrt_client,
    )
    return nothing
end

end
