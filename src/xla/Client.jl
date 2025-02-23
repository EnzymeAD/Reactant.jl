abstract type AbstractClient end

Base.:(==)(a::AbstractClient, b::AbstractClient) = a.client == b.client

function client end
function free_client end
function num_devices end
function num_addressable_devices end
function process_index end
function devices end
function addressable_devices end
function get_device end
function get_addressable_device end
function platform_name end

default_device(client::AbstractClient) = first(addressable_devices(client))

# Clients for Different Backends
function CPUClient(cfunc, node_id=0, num_nodes=1; asynchronous=true)
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, string(cfunc))
    client = ccall(f, Ptr{Cvoid}, (UInt, Cint, Cint), asynchronous, node_id, num_nodes)
    LLVMclopts("-nvptx-fma-level=1")
    return client
end

function GPUClient(
    cfunc,
    node_id=0,
    num_nodes=1,
    platform="gpu";
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
    distributed_runtime_client::Union{Nothing,DistributedRuntimeClient}=nothing,
)
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, string(cfunc))
    refstr = Ref{Cstring}()

    num_allowed_devices = allowed_devices === nothing ? 0 : length(allowed_devices)
    allowed_devices = allowed_devices === nothing ? C_NULL : allowed_devices
    distributed_runtime_client =
        distributed_runtime_client === nothing ? C_NULL : distributed_runtime_client.client

    client = ccall(
        f,
        Ptr{Cvoid},
        (Cint, Cint, Ptr{Cvoid}, Cint, Cdouble, Bool, Cstring, Ptr{Cstring}, Ptr{Cvoid}),
        node_id,
        num_nodes,
        allowed_devices,
        num_allowed_devices,
        XLA_REACTANT_GPU_MEM_FRACTION[],
        false,
        platform,
        refstr,
        distributed_runtime_client,
    )
    client == C_NULL && throw(AssertionError(unsafe_string(refstr[])))
    LLVMclopts("-nvptx-fma-level=1")
    return client
end

function TPUClient(cfunc, tpu_path::String)
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, string(cfunc))
    refstr = Ref{Cstring}()
    client = ccall(f, Ptr{Cvoid}, (Cstring, Ptr{Cstring}), tpu_path, refstr)
    client == C_NULL && throw(AssertionError(unsafe_string(refstr[])))
    LLVMclopts("-nvptx-fma-level=1")
    return client
end
