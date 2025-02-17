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
const cpuclientcount = Ref(0)

# TODO: Add the IfRtClient dispatches here

function CPUPjRtClient(node_id=0, num_nodes=1; checkcount=true, asynchronous=true)
    if checkcount
        @assert cpuclientcount[] == 0
        cpuclientcount[] += 1
    end
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "MakeCPUClient")
    client = ccall(f, Ptr{Cvoid}, (UInt, Cint, Cint), asynchronous, node_id, num_nodes)
    LLVMclopts("-nvptx-fma-level=1")
    #client = @ccall MLIR.API.mlir_c.MakeCPUClient(asynchronous::UInt8, node_id::Cint, num_nodes::Cint)::Ptr{Cvoid}
    return PJRT.Client(client)
end

function GPUPjRtClient(node_id=0, num_nodes=1, platform="gpu")
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "MakeGPUClient")
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
        XLA_REACTANT_GPU_PREALLOCATE[],
        platform,
        refstr,
    )
    if client == C_NULL
        throw(AssertionError(unsafe_string(refstr[])))
    end
    LLVMclopts("-nvptx-fma-level=1")
    return PJRT.Client(client)
end

function TPUPjRtClient(tpu_path::String)
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "MakeTPUClient")
    refstr = Ref{Cstring}()
    client = ccall(f, Ptr{Cvoid}, (Cstring, Ptr{Cstring}), tpu_path, refstr)
    if client == C_NULL
        throw(AssertionError(unsafe_string(refstr[])))
    end
    LLVMclopts("-nvptx-fma-level=1")
    return PJRT.Client(client)
end
