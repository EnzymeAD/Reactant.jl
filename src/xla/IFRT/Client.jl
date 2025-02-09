# currently, only supports IFRT-PjRt
mutable struct Client
    ptr::Ptr{Cvoid}

    function Client(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return finalizer(free_client, new(ptr))
    end
end

function Client(pjrt_client::XLA.Client)
    # it needs a `std::shared_ptr<xla::PjRtClient>`
    hold!(pjrt_client)
    return Client(@ccall MLIR.API.mlir_c.ifrt_pjrt_MakeClient(pjrt_client.holded::Ptr{Cvoid})::Ptr{Cvoid})
end

function free_client(client)
    @ccall MLIR.API.mlir_c.ifrt_pjrt_FreeClient(client.ptr::Ptr{Cvoid})::Cvoid
end

function compile(client::Client, code::MLIR.IR.Module)
    return LoadedExecutable(@ccall MLIR.API.mlir_c.ifrt_pjrt_ClientCompile(client.ptr::Ptr{Cvoid}, mod.module_::MLIR.API.MlirModule)::Ptr{Cvoid})
end
