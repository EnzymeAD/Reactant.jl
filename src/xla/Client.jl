mutable struct Client
    client::Ptr{Cvoid}
    global_ordinals::Vector{Cint}
    holded::Ptr{Cvoid}

    function Client(client::Ptr{Cvoid})
        @assert client != C_NULL
        global_ordinals = Cint[]

        client = new(client, global_ordinals, C_NULL)

        # https://github.com/pytorch/xla/blob/8b2414094578e829b99a8383877c86d357eeb682/torch_xla/csrc/runtime/pjrt_computation_client.cc#L127
        devices = [
            ClientGetAddressableDevice(client, i - 1) for
            i in 1:ClientNumAddressableDevices(client)
        ]
        sort!(devices; lt=(a, b) -> DeviceGetLocalDeviceId(a) < DeviceGetLocalDeviceId(b))

        local_ids = [DeviceGetLocalDeviceId(device) + 1 for device in devices]
        max_local_id = maximum(local_ids)
        resize!(global_ordinals, max_local_id)
        global_ordinals .= -1
        for (i, device) in enumerate(devices)
            global_ordinals[local_ids[i]] = i - 1
        end
        return client
    end
end

Base.:(==)(a::Client, b::Client) = a.client == b.client

@inline function free_client(client::Client)
    if client.holded == C_NULL
        @ccall MLIR.API.mlir_c.FreeClient(client.client::Ptr{Cvoid})::Cvoid
    else
        @ccall MLIR.API.mlir_c.reactant_release_pjrtclient(client.holded::Ptr{Cvoid})::Cvoid
    end
end

function hold!(client::Client)
    if client.holded == C_NULL
        client.holded = @ccall MLIR.API.mlir_c.reactant_hold_pjrtclient(client.client::Ptr{Cvoid})::Ptr{Cvoid}
    end
    return client
end

function ClientNumDevices(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientNumDevices(client.client::Ptr{Cvoid})::Cint
    end
end

function ClientNumAddressableDevices(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientNumAddressableDevices(
            client.client::Ptr{Cvoid}
        )::Cint
    end
end

function ClientProcessIndex(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientProcessIndex(client.client::Ptr{Cvoid})::Cint
    end
end

function ClientGetDevice(client::Client, idx)
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ClientGetDevice(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function ClientGetAddressableDevice(client::Client, idx)
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ClientGetAddressableDevice(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function ClientGetPlatformName(client::Client)
    GC.@preserve client begin
        str = @ccall MLIR.API.mlir_c.ClientGetPlatformName(
            client.client::Ptr{Cvoid}
        )::Cstring
    end
    str_jl = unsafe_string(str)
    @ccall free(str::Cstring)::Cvoid
    return str_jl
end

# Clients for Different Backends
const cpuclientcount = Ref(0)

function CPUClient(node_id=0, num_nodes=1; checkcount=true, asynchronous=true)
    if checkcount
        @assert cpuclientcount[] == 0
        cpuclientcount[] += 1
    end
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "MakeCPUClient")
    client = ccall(f, Ptr{Cvoid}, (UInt, Cint, Cint), asynchronous, node_id, num_nodes)
    LLVMclopts("-nvptx-fma-level=1")
    #client = @ccall MLIR.API.mlir_c.MakeCPUClient(asynchronous::UInt8, node_id::Cint, num_nodes::Cint)::Ptr{Cvoid}
    return Client(client)
end

function GPUClient(node_id=0, num_nodes=1, platform="gpu")
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
    return Client(client)
end

function TPUClient(tpu_path::String)
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "MakeTPUClient")
    refstr = Ref{Cstring}()
    client = ccall(f, Ptr{Cvoid}, (Cstring, Ptr{Cstring}), tpu_path, refstr)
    if client == C_NULL
        throw(AssertionError(unsafe_string(refstr[])))
    end
    LLVMclopts("-nvptx-fma-level=1")
    return Client(client)
end
