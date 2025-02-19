abstract type AbstractClient end

Base.:(==)(a::AbstractClient, b::AbstractClient) = a.client == b.client

function client end
function free_client end
function num_devices end
function num_addressable_devices end
function process_index end
function get_device end
function get_addressable_device end
function platform_name end

# Clients for Different Backends
const cpu_pjrt_client_count = Ref(0)
const cpu_ifrt_client_count = Ref(0)

function CPUClient(cfunc, node_id=0, num_nodes=1; asynchronous=true)
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, string(cfunc))
    client = ccall(f, Ptr{Cvoid}, (UInt, Cint, Cint), asynchronous, node_id, num_nodes)
    LLVMclopts("-nvptx-fma-level=1")
    return client
end

function GPUClient(cfunc, node_id=0, num_nodes=1, platform="gpu")
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, string(cfunc))
    refstr = Ref{Cstring}()
    client = ccall(
        f,
        Ptr{Cvoid},
        (Cint, Cint, Ptr{Cvoid}, Cint, Cdouble, Bool, Cstring, Ptr{Cstring}),
        node_id,
        num_nodes,
        C_NULL,
        0,
        XLA_REACTANT_GPU_MEM_FRACTION[],
        false,
        platform,
        refstr,
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
