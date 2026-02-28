module GPU

using Reactant_jll: Reactant_jll
using ..Reactant: Reactant
using ..Registration: register_backend

function make_pjrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    refstr = Ref{Cstring}()

    num_allowed_devices = allowed_devices === nothing ? 0 : length(allowed_devices)
    allowed_devices = allowed_devices === nothing ? C_NULL : allowed_devices
    distributed_runtime_client =
        distributed_runtime_client === nothing ? C_NULL : distributed_runtime_client.client

    GC.@preserve refstr allowed_devices distributed_runtime_client begin
        client = Reactant.MLIR.API.MakeGPUClient(
            node_id,
            num_nodes,
            allowed_devices,
            num_allowed_devices,
            Reactant.XLA.XLA_REACTANT_GPU_MEM_FRACTION[],
            Reactant.XLA.XLA_REACTANT_GPU_PREALLOCATE[],
            "gpu",
            refstr,
            distributed_runtime_client,
        )
    end

    client == C_NULL && throw(AssertionError(unsafe_string(refstr[])))
    Reactant.XLA.LLVMclopts("--nvptx-fma-level=1")
    return Reactant.XLA.PJRT.Client(client)
end

function make_ifrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    refstr = Ref{Cstring}()

    num_allowed_devices = allowed_devices === nothing ? 0 : length(allowed_devices)
    allowed_devices = allowed_devices === nothing ? C_NULL : allowed_devices
    distributed_runtime_client =
        distributed_runtime_client === nothing ? C_NULL : distributed_runtime_client.client

    GC.@preserve refstr allowed_devices distributed_runtime_client begin
        client = Reactant.MLIR.API.ifrt_make_pjrt_gpu_client(
            node_id,
            num_nodes,
            allowed_devices,
            num_allowed_devices,
            Reactant.XLA.XLA_REACTANT_GPU_MEM_FRACTION[],
            Reactant.XLA.XLA_REACTANT_GPU_PREALLOCATE[],
            "gpu",
            refstr,
            distributed_runtime_client,
        )
    end

    client == C_NULL && throw(AssertionError(unsafe_string(refstr[])))
    Reactant.XLA.LLVMclopts("--nvptx-fma-level=1")
    return Reactant.XLA.IFRT.Client(client)
end

function __init__()
    if (
        Sys.islinux() &&
        Reactant_jll.is_available() &&
        Reactant_jll.host_platform.tags["gpu"] != "none"
    )
        register_backend(
            "cuda"; # TODO: disambiguate between CUDA and ROCM
            priority=500,
            pjrt_initialize_function=make_pjrt_client,
            ifrt_initialize_function=make_ifrt_client,
        )
    end
    return nothing
end

end
