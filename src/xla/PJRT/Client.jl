mutable struct Client <: XLA.AbstractClient
    client::Ptr{Cvoid}
    global_ordinals::Vector{Cint}

    function Client(client::Ptr{Cvoid})
        @assert client != C_NULL
        global_ordinals = Cint[]

        client = new(client, global_ordinals)

        # https://github.com/pytorch/xla/blob/8b2414094578e829b99a8383877c86d357eeb682/torch_xla/csrc/runtime/pjrt_computation_client.cc#L127
        devices = [
            XLA.get_addressable_device(client, i - 1) for
            i in 1:XLA.num_addressable_devices(client)
        ]
        sort!(devices; lt=(a, b) -> XLA.get_local_device_id(a) < XLA.get_local_device_id(b))

        local_ids = [XLA.get_local_device_id(device) + 1 for device in devices]
        max_local_id = maximum(local_ids)
        resize!(global_ordinals, max_local_id)
        global_ordinals .= -1
        for (i, device) in enumerate(devices)
            global_ordinals[local_ids[i]] = i - 1
        end
        return client
    end
end

function XLA.free_client(client::Client)
    @ccall MLIR.API.mlir_c.FreeClient(client.client::Ptr{Cvoid})::Cvoid
end

function XLA.num_devices(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientNumDevices(client.client::Ptr{Cvoid})::Cint
    end
end

function XLA.num_addressable_devices(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientNumAddressableDevices(
            client.client::Ptr{Cvoid}
        )::Cint
    end
end

function XLA.process_index(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientProcessIndex(client.client::Ptr{Cvoid})::Cint
    end
end

function XLA.get_device(client::Client, idx)
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ClientGetDevice(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function XLA.get_addressable_device(client::Client, idx)
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ClientGetAddressableDevice(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function XLA.platform_name(client::Client)
    GC.@preserve client begin
        str = @ccall MLIR.API.mlir_c.ClientGetPlatformName(
            client.client::Ptr{Cvoid}
        )::Cstring
    end
    str_jl = unsafe_string(str)
    @ccall free(str::Cstring)::Cvoid
    return str_jl
end
