mutable struct Client <: XLA.AbstractClient
    client::Ptr{Cvoid}

    function Client(client::Ptr{Cvoid})
        @assert client != C_NULL
        # TODO: add finalizer, but I am getting segfaults
        # return finalizer(XLA.free_client, new(client))
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
        return @ccall MLIR.API.mlir_c.ifrt_client_device_count(client.client::Ptr{Cvoid})::Cint
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

# Different Backends
const cpu_client_count = Ref(0)
const gpu_client_count = Ref(0)
const tpu_client_count = Ref(0)

for (backend, fname, counter) in (
    (:CPUClient, "ifrt_make_cpu_client", :cpu_client_count),
    (:GPUClient, "ifrt_make_gpu_client", :gpu_client_count),
    (:TPUClient, "ifrt_make_tpu_client", :tpu_client_count),
)
    @eval function $(backend)(args...; checkcount::Bool=true, kwargs...)
        if checkcount
            @assert $(counter)[] == 0
            $(counter)[] += 1
        end
        return Client(XLA.$(backend)($(fname), args...; kwargs...))
    end
end
