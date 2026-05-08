mutable struct LoadedExecutable <: XLA.AbstractLoadedExecutable
    exec::Ptr{Cvoid}
    num_outputs::Int64
    num_parameters::Int64
    is_sharded::Bool
    num_replicas::Int64
    num_partitions::Int64

    function LoadedExecutable(exec::Ptr{Cvoid}, args...)
        @assert exec != C_NULL
        return finalizer(free_exec, new(exec, args...))
    end
end

function free_exec(exec::LoadedExecutable)
    if XLA.is_live[]
        GC.@preserve exec begin
            MLIR.API.ifrt_loaded_executable_dtor(exec.exec)
        end
    end
end

function XLA.client(exec::LoadedExecutable)
    GC.@preserve exec begin
        return Client(MLIR.API.ifrt_loaded_executable_client(exec.exec))
    end
end

XLA.num_partitions(exec::LoadedExecutable) = exec.num_partitions
XLA.num_replicas(exec::LoadedExecutable) = exec.num_replicas
XLA.num_devices(exec::LoadedExecutable) = XLA.num_replicas(exec) * XLA.num_partitions(exec)

for (jlop, xlaop, field) in (
    (:get_output_shardings, :ifrt_loaded_executable_get_output_shardings, :num_outputs),
    (
        :get_parameter_shardings,
        :ifrt_loaded_executable_get_parameter_shardings,
        :num_parameters,
    ),
)
    @eval function XLA.$(jlop)(exec::LoadedExecutable)
        if !exec.is_sharded || iszero(exec.$(field))
            return XLA.OpSharding[]
        end

        op_shardings = Ref{NTuple{exec.$(field),Ptr{Cvoid}}}()
        GC.@preserve exec begin
            MLIR.API.$(xlaop)(exec.exec, op_shardings, exec.$(field))
        end
        return [XLA.OpSharding(op_sharding) for op_sharding in op_shardings[]]
    end
end

function XLA.get_hlo_modules(exec::LoadedExecutable)
    # If we had compiled with MPMD then we would need all the partitions to get hlo_modules
    # but if we used SPMD we get only 1 module. To be safe we allocate for all the modules
    # and use the ones assigned to by XLA
    hlo_modules = Ref{NTuple{Int64(XLA.num_partitions(exec)),Ptr{Cvoid}}}()
    nmodules = Ref{Int32}(0)
    GC.@preserve exec hlo_modules begin
        MLIR.API.ifrt_loaded_executable_get_hlo_modules(exec.exec, hlo_modules, nmodules)
    end
    return map(XLA.HloModule, hlo_modules[][1:Int(nmodules[])])
end

function XLA.compile(
    client::Client,
    mod::MLIR.IR.Module;
    compile_options::Reactant.Proto.xla.CompileOptionsProto,
    num_parameters::Int64,
    num_outputs::Int64,
    is_sharded::Bool,
    num_replicas::Int64,
    num_partitions::Int64,
)
    compile_options_bytes = Reactant.ProtoUtils.proto_to_bytes(compile_options)
    GC.@preserve client mod compile_options_bytes begin
        exec = MLIR.IR.try_compile_dump_mlir(mod) do
            MLIR.API.ifrt_compile_with_proto(
                client.client, mod, compile_options_bytes, length(compile_options_bytes)
            )
        end
    end
    return LoadedExecutable(
        exec, num_outputs, num_parameters, is_sharded, num_replicas, num_partitions
    )
end

"""
    GpuTopology(client; num_partitions, num_hosts_per_partition=1,
                       num_devices_per_host, platform_version="12.3")

Create a GPU topology description for AOT compilation with mock devices.
The `client` is used to extract the GPU target config (compute capability,
memory, etc.) so the compiled executable matches the real hardware.
"""
mutable struct GpuTopology
    topology::Ptr{Cvoid}

    function GpuTopology(
        client=XLA.default_backend();
        num_partitions::Integer,
        num_hosts_per_partition::Integer=1,
        num_devices_per_host::Integer,
        platform_version::String="12.3",
    )
        ptr = MLIR.API.ifrt_gpu_topology_create(
            client.client,
            platform_version,
            Int32(num_partitions),
            Int32(num_hosts_per_partition),
            Int32(num_devices_per_host),
        )
        return finalizer(new(ptr)) do topo
            XLA.is_live[] && MLIR.API.ifrt_topology_dtor(topo.topology)
        end
    end
end

"""
    compile_to_executable(client, mod, topology; compile_options, kwargs...)

Compile an MLIR module against a GPU topology (AOT, no live devices needed).
Returns an unloaded `Executable` that can be serialized.
"""
function compile_to_executable(
    client::Client,
    mod::MLIR.IR.Module,
    topology::GpuTopology;
    compile_options::Reactant.Proto.xla.CompileOptionsProto,
)
    compile_options_bytes = Reactant.ProtoUtils.proto_to_bytes(compile_options)
    GC.@preserve client mod topology compile_options_bytes begin
        exec = MLIR.IR.try_compile_dump_mlir(mod) do
            MLIR.API.ifrt_compile_with_topology(
                client,
                mod,
                topology.topology,
                compile_options_bytes,
                length(compile_options_bytes),
            )
        end
    end
    return Executable(exec)
end

"""
    Executable

An unloaded IFRT executable (compiled but not yet loaded onto devices).
Can be serialized to bytes and later deserialized + loaded.
"""
mutable struct Executable
    exec::Ptr{Cvoid}

    function Executable(exec::Ptr{Cvoid})
        @assert exec != C_NULL
        return finalizer(new(exec)) do e
            XLA.is_live[] && MLIR.API.ifrt_executable_dtor(e.exec)
        end
    end
end

"""
    serialize_executable(exec::Executable) -> Vector{UInt8}

Serialize an unloaded executable to bytes for on-disk storage.
"""
function serialize_executable(exec::Executable)
    GC.@preserve exec begin
        return MLIR.API.ifrt_executable_serialize(exec.exec)
    end
end

"""
    deserialize_and_load(client::Client, bytes::Vector{UInt8}; kwargs...) -> LoadedExecutable

Deserialize and load an executable onto the devices of `client`.
"""
function deserialize_and_load(
    client::Client,
    bytes::Vector{UInt8};
    num_outputs::Int64,
    num_parameters::Int64,
    is_sharded::Bool,
    num_replicas::Int64,
    num_partitions::Int64,
    compile_options::Reactant.Proto.xla.CompileOptionsProto,
)
    compile_options_bytes = Reactant.ProtoUtils.proto_to_bytes(compile_options)
    GC.@preserve client bytes begin
        exec = MLIR.API.ifrt_deserialize_and_load(
            client,
            bytes,
            length(bytes),
            compile_options_bytes,
            length(compile_options_bytes),
        )
    end
    return LoadedExecutable(
        exec, num_outputs, num_parameters, is_sharded, num_replicas, num_partitions
    )
end

@inline function XLA.execute(
    exec::LoadedExecutable,
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{M,UInt8},
    ::Val{n_outs},
) where {N,M,n_outs}
    outputs = Ref{NTuple{n_outs,Ptr{Cvoid}}}()
    future_res = Ref{Ptr{Cvoid}}()
    futures = Ref{UInt8}(0)

    @info "EXECUTING" N l = length(inputs) m = M

    GC.@preserve exec outputs future_res futures begin
        MLIR.API.ifrt_loaded_executable_execute(
            exec.exec,
            N,
            Base.RefValue(inputs),
            Base.RefValue(donated_args),
            n_outs,
            outputs,
            futures,
            future_res,
        )
    end

    outputs = outputs[]
    future = futures[] != 0
    future && (future_res[] = future_res[])

    return ntuple(n_outs) do i
        Base.@_inline_meta
        AsyncArray(Array(outputs[i]), future ? Future(future_res[]) : nothing)
    end
end

# Convinience functions matching the signatures of the PJRT loaded executable
@inline function XLA.execute(
    exec::LoadedExecutable,
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{M,UInt8},
    ::Val{n_outs},
    ::Val{K},
) where {N,M,n_outs,K}
    return XLA.execute(exec, inputs, donated_args, Val(n_outs))
end

@inline function XLA.execute_sharded(
    exec::LoadedExecutable,
    ::Device,
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{N,UInt8},
    ::Val{n_outs},
) where {N,n_outs}
    return XLA.execute(exec, inputs, donated_args, Val(n_outs))
end
