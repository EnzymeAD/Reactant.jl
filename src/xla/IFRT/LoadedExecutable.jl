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
    device::Union{Device,Nothing},
    mod::MLIR.IR.Module;
    is_sharded::Bool=false,
    global_device_ids::Vector{Int64}=Int64[],
    num_outputs::Int64,
    num_parameters::Int64,
    num_replicas::Int64,
    num_partitions::Int64,
    use_shardy_partitioner::Bool,
)
    device_id = is_sharded ? Int64(-1) : Int64(XLA.device_ordinal(device))
    GC.@preserve client mod begin
        exec = MLIR.IR.try_compile_dump_mlir(mod) do
            @ccall MLIR.API.mlir_c.ifrt_compile(
                client.client::Ptr{Cvoid},
                mod.module_::MLIR.API.MlirModule,
                device_id::Clong,
                global_device_ids::Ptr{Clong},
                length(global_device_ids)::Clong,
                XLA.CUDA_DATA_DIR[]::Cstring,
                use_shardy_partitioner::Bool,
                num_replicas::Int64,
                num_partitions::Int64,
                is_sharded::Bool,
                Reactant.PersistentCompileCache.kernel_cache_enabled()::Bool,
                Reactant.PersistentCompileCache.get_kernel_cache_path()::Cstring,
                Reactant.PersistentCompileCache.autotune_cache_enabled()::Bool,
                Reactant.PersistentCompileCache.get_autotune_cache_directory()::Cstring,
                Reactant.Distributed.local_rank()::Cint,
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
