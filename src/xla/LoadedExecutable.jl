@inline function free_exec(exec)
    @ccall MLIR.API.mlir_c.ExecutableFree(exec.exec::Ptr{Cvoid})::Cvoid
end

mutable struct LoadedExecutable
    exec::Ptr{Cvoid}
    num_outputs::Int64
    num_parameters::Int64
    is_sharded::Bool

    function LoadedExecutable(
        exec::Ptr{Cvoid}, num_outputs::Int64, num_parameters::Int64, is_sharded::Bool
    )
        @assert exec != C_NULL
        return finalizer(free_exec, new(exec, num_outputs, num_parameters, is_sharded))
    end
end

for (jlop, xlaop) in (
    (:num_replicas, :PjRtLoadedExecutableNumReplicas),
    (:num_partitions, :PjRtLoadedExecutableNumPartitions),
)
    @eval function $(jlop)(exec::LoadedExecutable)
        GC.@preserve exec begin
            return @ccall MLIR.API.mlir_c.$(xlaop)(exec.exec::Ptr{Cvoid})::Cint
        end
    end
end

function client(exec::LoadedExecutable)
    GC.@preserve exec begin
        return Client(
            @ccall MLIR.API.mlir_c.PjRtLoadedExecutableGetClient(
                exec.exec::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function execute_ir(N, M, n_outs, fn, with_device::Bool, nmesh_ids::Int64)
    ptr = sizeof(Int) == sizeof(Int64) ? "i64" : "i32"
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

    res = """define { [$n_outs x $ptr], [$n_outs x $ptr], i8 } @f($ptr %exec, $args) alwaysinline {
   entry:
   	%inpa = alloca [$N x $ptr]
   	%dona = alloca [$M x i8]
   	%outa = alloca [$n_outs x $ptr]
   	%futpa = alloca [$n_outs x $ptr]
   	%mesha = alloca [$nmesh_ids x $ptr]
   	$stores
   	%futa = alloca i8
   	call void inttoptr ($ptr $fn to void ($ptr, $cint, [$N x $ptr]*, $extra_str1, [$M x i8]*, $cint, [$n_outs x $ptr]*, i8*, [$n_outs x $ptr]*)*)($ptr %exec, $cint $N, [$N x $ptr]* nocapture readonly %inpa, $extra_str2, [$M x i8]* nocapture readonly %dona, $cint $n_outs, [$n_outs x $ptr]* nocapture writeonly %outa, i8* nocapture writeonly %futa, [$n_outs x $ptr]* nocapture writeonly %futpa)
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

@generated function ExecutableCallSharded(
    exec::LoadedExecutable,
    device::Device,
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{N,UInt8},
    ::Val{n_outs},
) where {N,n_outs}
    sym0 = dlsym(Reactant_jll.libReactantExtra_handle, "XLAExecuteSharded")
    xla_execute_fn = reinterpret(UInt, sym0)
    ir = execute_ir(N, N, n_outs, xla_execute_fn, true, 0)
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
end

# XXX: Fix this
# @generated function ExecutableCall(
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

@inline function ExecutableCall(
    exec::LoadedExecutable,
    mesh_ids::Vector{Int64},
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{M,UInt8},
    ::Val{n_outs},
    ::Val{K},
) where {N,M,n_outs,K}
    @assert length(mesh_ids) == K
    outputs = Ref{NTuple{n_outs * K,Ptr{Cvoid}}}()
    future_res = Ref{NTuple{n_outs * K,Ptr{Cvoid}}}()
    futures = Ref{UInt8}(0)

    inputs = Base.RefValue(inputs)
    donated_args = Base.RefValue(donated_args)
    GC.@preserve inputs donated_args mesh_ids outputs futures future_res begin
        @ccall MLIR.API.mlir_c.XLAExecute(
            exec.exec::Ptr{Cvoid},
            N::Cint,
            inputs::Ptr{Cvoid},
            mesh_ids::Ptr{Clong},
            K::Clong,
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

function Compile(
    client::Client,
    device::Union{Device,Nothing},
    mod::MLIR.IR.Module;
    is_sharded::Bool=false,
    device_ids::Vector{Int64}=Int64[],
    num_outputs::Int64,
    num_parameters::Int64,
)
    device_id = is_sharded ? Int64(-1) : Int64(device_ordinal(client, device))
    mesh_ids = Int64.(device_ordinal.((client,), device_ids))
    GC.@preserve client mod begin
        exec = @ccall MLIR.API.mlir_c.ClientCompile(
            client.client::Ptr{Cvoid},
            mod.module_::MLIR.API.MlirModule,
            device_id::Clong,
            is_sharded::Bool,
            mesh_ids::Ptr{Clong},
            length(mesh_ids)::Clong,
            CUDA_DATA_DIR[]::Cstring,
        )::Ptr{Cvoid}
    end
    return LoadedExecutable(exec, num_outputs, num_parameters, is_sharded)
end

for (jlop, xlaop, field) in (
    (:get_output_shardings, :PjRtLoadedExecutableGetOuputShardings, :num_outputs),
    (:get_parameter_shardings, :PjRtLoadedExecutableGetParameterShardings, :num_parameters),
)
    @eval function $(jlop)(exec::LoadedExecutable)
        exec.is_sharded || return OpSharding[]

        jl_op_shardings = [Ref{JLOpSharding}() for _ in 1:(exec.$(field))]
        jl_op_shardings_ptr = [
            Base.unsafe_convert(Ptr{JLOpSharding}, sharding) for sharding in jl_op_shardings
        ]

        GC.@preserve jl_op_shardings begin
            @ccall MLIR.API.mlir_c.$(xlaop)(
                exec.exec::Ptr{Cvoid},
                jl_op_shardings_ptr::Ptr{Ptr{JLOpSharding}},
                exec.$(field)::Int32,
            )::Cvoid
        end

        return map(Base.Fix1(convert, OpSharding) ∘ getindex, jl_op_shardings)
    end
end

function get_hlo_modules(exec::LoadedExecutable)
    # If we had compiled with MPMD then we would need all the partitions to get hlo_modules
    # but if we used SPMD we get only 1 module. To be safe we allocate for all the modules
    # and use the ones assigned to by XLA
    hlo_modules = Ref{NTuple{Int64(num_partitions(exec)),Ptr{Cvoid}}}()
    nmodules = Ref{Int32}(0)
    GC.@preserve exec hlo_modules begin
        @ccall MLIR.API.mlir_c.PjRtLoadedExecutableGetHloModules(
            exec.exec::Ptr{Cvoid}, hlo_modules::Ptr{Ptr{Cvoid}}, nmodules::Ptr{Cint}
        )::Cvoid
    end
    return map(HloModule, hlo_modules[][1:Int(nmodules[])])
end
