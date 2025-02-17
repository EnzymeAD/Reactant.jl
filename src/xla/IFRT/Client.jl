mutable struct Client <: XLA.AbstractClient
    client::Ptr{Cvoid}

    function Client(client::Ptr{Cvoid})
        @assert client != C_NULL
        # TODO: Finalizer
        return new(client)
    end
end

# Different Backends
const cpu_client_count = Ref(0)
const gpu_client_count = Ref(0)
const tpu_client_count = Ref(0)

for (backend, fname, counter) in (
    (:CPUClient, "MakeCPUIfrtClient", :cpu_client_count),
    (:GPUClient, "MakeGPUIfrtClient", :gpu_client_count),
    (:TPUClient, "MakeTPUIfrtClient", :tpu_client_count),
)
    @eval function $(backend)(args...; checkcount::Bool=true, kwargs...)
        if checkcount
            @assert $(counter)[] == 0
            $(counter)[] += 1
        end
        return Client(XLA.$(backend)($(fname), args...; kwargs...))
    end
end
