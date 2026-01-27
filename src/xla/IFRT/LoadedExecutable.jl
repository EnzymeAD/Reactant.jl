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
    GC.@preserve exec begin
        @ccall MLIR.API.mlir_c.ifrt_loaded_executable_dtor(exec.exec::Ptr{Cvoid})::Cvoid
    end
end

function XLA.client(exec::LoadedExecutable)
    GC.@preserve exec begin
        return Client(
            @ccall MLIR.API.mlir_c.ifrt_loaded_executable_client(
                exec.exec::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
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

        GC.@preserve exec op_shardings begin
            @ccall MLIR.API.mlir_c.$(xlaop)(
                exec.exec::Ptr{Cvoid}, op_shardings::Ptr{Ptr{Cvoid}}, exec.$(field)::Cint
            )::Cvoid
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
        @ccall MLIR.API.mlir_c.ifrt_loaded_executable_get_hlo_modules(
            exec.exec::Ptr{Cvoid}, hlo_modules::Ptr{Ptr{Cvoid}}, nmodules::Ptr{Int32}
        )::Cvoid
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
            @ccall MLIR.API.mlir_c.ifrt_compile_with_proto(
                client.client::Ptr{Cvoid},
                mod::MLIR.API.MlirModule,
                compile_options_bytes::Ptr{UInt8},
                length(compile_options_bytes)::Csize_t,
            )::Ptr{Cvoid}
        end
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

    inputs = Base.RefValue(inputs)
    donated_args = Base.RefValue(donated_args)
    GC.@preserve exec outputs future_res futures begin
        @ccall MLIR.API.mlir_c.ifrt_loaded_executable_execute(
            exec.exec::Ptr{Cvoid},
            N::Cint,
            inputs::Ptr{Ptr{Cvoid}},
            donated_args::Ptr{UInt8},
            n_outs::Cint,
            Base.unsafe_convert(Ptr{Ptr{Cvoid}}, outputs)::Ptr{Ptr{Cvoid}},
            futures::Ptr{UInt8},
            future_res::Ptr{Ptr{Cvoid}},
        )::Cvoid
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
