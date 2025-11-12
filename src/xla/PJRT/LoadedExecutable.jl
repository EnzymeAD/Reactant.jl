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

@inline function free_exec(exec::LoadedExecutable)
    @ccall MLIR.API.mlir_c.ExecutableFree(exec.exec::Ptr{Cvoid})::Cvoid
end

function XLA.client(exec::LoadedExecutable)
    GC.@preserve exec begin
        return Client(
            @ccall MLIR.API.mlir_c.PjRtLoadedExecutableGetClient(
                exec.exec::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

XLA.num_partitions(exec::LoadedExecutable) = exec.num_partitions
XLA.num_replicas(exec::LoadedExecutable) = exec.num_replicas
XLA.num_devices(exec::LoadedExecutable) = XLA.num_replicas(exec) * XLA.num_partitions(exec)

for (jlop, xlaop, field) in (
    (:get_output_shardings, :PjRtLoadedExecutableGetOuputShardings, :num_outputs),
    (:get_parameter_shardings, :PjRtLoadedExecutableGetParameterShardings, :num_parameters),
)
    @eval function XLA.$(jlop)(exec::LoadedExecutable)
        if !exec.is_sharded || iszero(exec.$(field))
            return XLA.OpSharding[]
        end

        op_shardings = Ref{NTuple{exec.$(field),Ptr{Cvoid}}}()

        GC.@preserve op_shardings begin
            @ccall MLIR.API.mlir_c.$(xlaop)(
                exec.exec::Ptr{Cvoid}, op_shardings::Ptr{Ptr{Cvoid}}, exec.$(field)::Int32
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
        @ccall MLIR.API.mlir_c.PjRtLoadedExecutableGetHloModules(
            exec.exec::Ptr{Cvoid}, hlo_modules::Ptr{Ptr{Cvoid}}, nmodules::Ptr{Cint}
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
            @ccall MLIR.API.mlir_c.ClientCompile(
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

function execute_ir(N, M, n_outs, with_device::Bool, nmesh_ids::Int64)
    ptr = @static if VERSION < v"1.12"
        sizeof(Int) == sizeof(Int64) ? "i64" : "i32"
    else
        "ptr"
    end
    cint = sizeof(Cint) == sizeof(Int64) ? "i64" : "i32"
    args = N > 0 ? ", [$N x $ptr] %inps, [$M x i8] %donated" : ""
    if with_device
        args = "$ptr %dev $args"
    else
        args = "[$nmesh_ids x $ptr] %mesh_ids $args"
    end

    stores = N > 0 ? """
   store [$N x $ptr] %inps, [$N x $ptr]* %inpa
   store [$M x i8] %donated, [$M x i8]* %dona
   	""" : ""

    if !with_device
        stores *= """
      store [$nmesh_ids x $ptr] %mesh_ids, [$nmesh_ids x $ptr]* %mesha
      		"""
    end

    extra_str1 = with_device ? "$ptr" : "[$nmesh_ids x $ptr]*, i64"
    extra_str2 = if with_device
        "$ptr %dev"
    else
        "[$(nmesh_ids) x $ptr]* nocapture readonly %mesha, i64 $(nmesh_ids)"
    end

    fn = if with_device
        "@XLAExecuteSharded"
    else
        "@XLAExecute"
    end

    res = """
declare void @XLAExecuteSharded($ptr %exec, $cint %num_args, [$N x $ptr]* readonly nocapture %op_args, $ptr %device, 
[$M x i8]* nocapture readonly %is_arg_donatable, $cint %num_results, [$n_outs x $ptr]* writeonly nocapture %op_results, i8* writeonly nocapture %futures, [$n_outs x $ptr]* writeonly nocapture %future_results)

declare void @XLAExecute($ptr %exec, $cint %op_args_len, [$N x $ptr]* readonly nocapture %op_args, [$M x i8]* nocapture readonly %is_arg_donatable, $cint %num_results, [$n_outs x $ptr]* writeonly nocapture %op_results, i8* writeonly nocapture %futures, [$n_outs x $ptr]* writeonly nocapture %future_results)

define { [$n_outs x $ptr], [$n_outs x $ptr], i8 } @f($ptr %exec, $args) alwaysinline {
   entry:
   	%inpa = alloca [$N x $ptr]
   	%dona = alloca [$M x i8]
   	%outa = alloca [$n_outs x $ptr]
   	%futpa = alloca [$n_outs x $ptr]
   	%mesha = alloca [$nmesh_ids x $ptr]
   	$stores
   	%futa = alloca i8
   	call void $fn($ptr %exec, $cint $N, [$N x $ptr]* nocapture readonly %inpa, $extra_str2, [$M x i8]* nocapture readonly %dona, $cint $n_outs, [$n_outs x $ptr]* nocapture writeonly %outa, i8* nocapture writeonly %futa, [$n_outs x $ptr]* nocapture writeonly %futpa)
   	%out = load [$n_outs x $ptr], [$n_outs x $ptr]* %outa
   	%fut = load i8, i8* %futa
   	%futp = load [$n_outs x $ptr], [$n_outs x $ptr]* %futpa
   	%fca.0.insert = insertvalue { [$n_outs x $ptr], [$n_outs x $ptr], i8 } undef, [$n_outs x $ptr] %out, 0
   	%fca.1.insert = insertvalue { [$n_outs x $ptr], [$n_outs x $ptr], i8 } %fca.0.insert, [$n_outs x $ptr] %futp, 1
   	%fca.2.insert = insertvalue { [$n_outs x $ptr], [$n_outs x $ptr], i8 } %fca.1.insert, i8 %fut, 2
   	ret { [$n_outs x $ptr], [$n_outs x $ptr], i8 } %fca.2.insert
}
   """
    return res
end

@generated function XLA.execute_sharded(
    exec::LoadedExecutable,
    device::Device,
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{N,UInt8},
    ::Val{n_outs},
) where {N,n_outs}
    ir = execute_ir(N, N, n_outs, true, 0)
    results = []
    for i in 1:n_outs
        push!(
            results,
            :((
                AsyncBuffer(Buffer(outputs[$i]), future ? Future(future_res[$i]) : nothing),
            )),
        )
    end

    args_type = if N > 0
        (Ptr{Cvoid}, Ptr{Cvoid}, NTuple{N,Ptr{Cvoid}}, NTuple{N,UInt8})
    else
        (Ptr{Cvoid}, Ptr{Cvoid})
    end
    args = N > 0 ? (:inputs, :donated_args) : ()

    if !Reactant.precompiling() || Sys.isapple()
        return quote
            Base.@_inline_meta
            exec = exec.exec
            device = device.device
            GC.@preserve exec device begin
                outputs, future_res, future = Base.llvmcall(
                    ($ir, "f"),
                    Tuple{NTuple{n_outs,Ptr{Cvoid}},NTuple{n_outs,Ptr{Cvoid}},Bool},
                    Tuple{$args_type...},
                    exec,
                    device,
                    $(args...),
                )
            end
            return ($(results...),)
        end
    else
        return quote
            Base.@_inline_meta
            exec = exec.exec
            device = device.device
            inputs = Base.RefValue(inputs)
            is_arg_donatable = Base.RefValue(donated_args)
            outputs_p = Ref{NTuple{$n_outs,Ptr{Cvoid}}}()
            futures = Ref{UInt8}(0)
            futures_res = Ref{NTuple{$n_outs,Ptr{Cvoid}}}()
            GC.@preserve exec device inputs is_arg_donatable outputs_p futures futures_res begin
                @ccall MLIR.API.mlir_c.XLAExecuteSharded(
                    exec::Ptr{Cvoid},
                    $N::Cuint,
                    inputs::Ptr{Cvoid},
                    device::Ptr{Cvoid},
                    is_arg_donatable::Ptr{Cvoid},
                    $n_outs::Cuint,
                    outputs_p::Ptr{Cvoid},
                    futures::Ptr{Cvoid},
                    futures_res::Ptr{Cvoid},
                )::Cvoid
            end
            outputs = outputs_p[]
            future_res = futures_res[]
            future = futures[] != 0
            return ($(results...),)
        end
    end
end

# XXX: Fix this
# @generated function XLA.execute(
#     exec::LoadedExecutable,
#     mesh_ids::Vector{Int64},
#     inputs::NTuple{N,Ptr{Cvoid}},
#     donated_args::NTuple{M,UInt8},
#     ::Val{n_outs},
#     ::Val{K},
# ) where {N,M,K,n_outs}
#     sym0 = dlsym(Reactant_jll.libReactantExtra_handle, "XLAExecute")
#     xla_execute_fn = reinterpret(UInt, sym0)

#     ir = execute_ir(N, M, n_outs * K, xla_execute_fn, false, K)
#     results = [Vector{Any}(undef, K) for i in 1:n_outs]
#     for i in 1:n_outs, j in 1:K
#         idx = (i - 1) * K + j
#         results[i][j] = :(AsyncBuffer(
#             Buffer(outputs[$idx]), future ? Future(future_res[$idx]) : nothing
#         ))
#     end

#     args_type = if N > 0
#         (Ptr{Cvoid}, Ptr{Clong}, NTuple{N,Ptr{Cvoid}}, NTuple{M,UInt8})
#     else
#         (Ptr{Cvoid}, Ptr{Clong})
#     end
#     args = N > 0 ? (:inputs, :donated_args) : ()
#     return quote
#         Base.@_inline_meta
#         exec = exec.exec
#         GC.@preserve exec begin
#             outputs, future_res, future = Base.llvmcall(
#                 ($ir, "f"),
#                 Tuple{NTuple{n_outs * K,Ptr{Cvoid}},NTuple{n_outs * K,Ptr{Cvoid}},Bool},
#                 Tuple{$args_type...},
#                 exec,
#                 mesh_ids,
#                 $(args...),
#             )
#         end
#         return ($(results...),)
#     end
# end

@inline function XLA.execute(
    exec::LoadedExecutable,
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{M,UInt8},
    ::Val{n_outs},
    ::Val{K},
) where {N,M,n_outs,K}
    outputs = Ref{NTuple{n_outs * K,Ptr{Cvoid}}}()
    future_res = Ref{NTuple{n_outs * K,Ptr{Cvoid}}}()
    futures = Ref{UInt8}(0)

    inputs = Base.RefValue(inputs)
    donated_args = Base.RefValue(donated_args)
    GC.@preserve inputs donated_args outputs futures future_res begin
        @ccall MLIR.API.mlir_c.XLAExecute(
            exec.exec::Ptr{Cvoid},
            N::Cint,
            inputs::Ptr{Cvoid},
            donated_args::Ptr{UInt8},
            n_outs::Cint,
            Base.unsafe_convert(Ptr{Cvoid}, outputs)::Ptr{Cvoid},
            Base.unsafe_convert(Ptr{UInt8}, futures)::Ptr{UInt8},
            Base.unsafe_convert(Ptr{Cvoid}, future_res)::Ptr{Cvoid},
        )::Cvoid
    end

    outputs = outputs[]
    future = futures[] != 0
    future && (future_res = future_res[])

    return ntuple(Val(n_outs)) do j
        ntuple(Val(K)) do i
            Base.@_inline_meta
            idx = (i - 1) * n_outs + j
            return AsyncBuffer(
                Buffer(outputs[idx]), future ? Future(future_res[idx]) : nothing
            )
        end
    end
end
