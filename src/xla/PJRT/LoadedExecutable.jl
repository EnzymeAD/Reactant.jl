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
    if XLA.is_live[]
        GC.@preserve exec begin
            MLIR.API.ExecutableFree(exec.exec)
        end
    end
end

function XLA.client(exec::LoadedExecutable)
    GC.@preserve exec begin
        return Client(MLIR.API.PjRtLoadedExecutableGetClient(exec.exec))
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
        MLIR.API.PjRtLoadedExecutableGetHloModules(exec.exec, hlo_modules, nmodules)
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
            MLIR.API.ClientCompileWithProto(
                client.client, mod, compile_options_bytes, length(compile_options_bytes)
            )
        end
    end
    return LoadedExecutable(
        exec, num_outputs, num_parameters, is_sharded, num_replicas, num_partitions
    )
end

# Call `XLAExecuteSharded`, keeping its argument and result arrays on the stack. The function
# is called by name: `XLA.__init__` defines it in Julia's JIT.
LLVM.Interop.@llvmgenerated builder function xla_execute_sharded(
    exec::Ptr{Cvoid},
    device::Ptr{Cvoid},
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{N,UInt8},
    ::Val{n_outs},
)::Tuple{NTuple{n_outs,Ptr{Cvoid}},NTuple{n_outs,Ptr{Cvoid}},Bool} where {N,n_outs}
    mod = LLVM.Interop.current_module(builder)
    T_ptr = exec.value_type  # how Julia lowers a `Ptr` (an integer before Julia 1.12)
    T_i8 = LLVM.Int8Type()
    T_cint = convert(LLVM.LLVMType, Cint)
    T_buf = LLVM.PointerType(T_i8)

    # void XLAExecuteSharded(exec, num_args, op_args, device, is_arg_donatable,
    #                        num_results, op_results, futures, future_results)
    ft = LLVM.FunctionType(
        LLVM.VoidType(), [T_ptr, T_cint, T_buf, T_ptr, T_buf, T_cint, T_buf, T_buf, T_buf]
    )
    f = LLVM.Function(mod, "XLAExecuteSharded", ft)
    # the buffers aren't captured (LLVM 21 replaced `nocapture` by `captures(none)`)
    nocapture = if LLVM.version() < v"21"
        LLVM.EnumAttribute(:nocapture)
    else
        LLVM.EnumAttribute(:captures, 0)
    end
    for (i, access) in
        ((3, :readonly), (5, :readonly), (7, :writeonly), (8, :writeonly), (9, :writeonly))
        append!(f.parameter_attributes[i], [LLVM.EnumAttribute(access), nocapture])
    end

    inputs_buf = LLVM.alloca!(builder, LLVM.ArrayType(T_ptr, N))
    donated_buf = LLVM.alloca!(builder, LLVM.ArrayType(T_i8, N))
    if N > 0
        LLVM.store!(builder, inputs, inputs_buf)
        LLVM.store!(builder, donated_args, donated_buf)
    end
    future_buf = LLVM.alloca!(builder, T_i8)
    if n_outs > 0
        outputs_buf = LLVM.alloca!(builder, LLVM.ArrayType(T_ptr, n_outs))
        future_results_buf = LLVM.alloca!(builder, LLVM.ArrayType(T_ptr, n_outs))
    end
    as_buf(ptr) = LLVM.bitcast!(builder, ptr, T_buf)
    LLVM.call!(
        builder,
        ft,
        f,
        [
            exec,
            LLVM.ConstantInt(T_cint, N),
            as_buf(inputs_buf),
            device,
            as_buf(donated_buf),
            LLVM.ConstantInt(T_cint, n_outs),
            n_outs > 0 ? as_buf(outputs_buf) : LLVM.null(T_buf),
            future_buf,
            n_outs > 0 ? as_buf(future_results_buf) : LLVM.null(T_buf),
        ],
    )

    # the outputs and their futures (if any), and whether there are futures
    results = LLVM.Value[]
    if n_outs > 0
        push!(results, LLVM.load!(builder, LLVM.ArrayType(T_ptr, n_outs), outputs_buf))
        push!(
            results, LLVM.load!(builder, LLVM.ArrayType(T_ptr, n_outs), future_results_buf)
        )
    end
    push!(results, LLVM.load!(builder, T_i8, future_buf))
    T_ret = LLVM.Interop.current_function(builder).function_type.return_type
    T_ret isa LLVM.StructType || return only(results)
    ret = LLVM.UndefValue(T_ret)
    for (i, val) in enumerate(results)
        ret = LLVM.insert_value!(builder, ret, val, i - 1)
    end
    return ret
end

@generated function XLA.execute_sharded(
    exec::LoadedExecutable,
    device::Device,
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{N,UInt8},
    ::Val{n_outs},
) where {N,n_outs}
    results = []
    for i in 1:n_outs
        push!(
            results,
            :((
                AsyncBuffer(Buffer(outputs[$i]), future ? Future(future_res[$i]) : nothing),
            )),
        )
    end

    if !Reactant.precompiling() || Sys.isapple()
        return quote
            Base.@_inline_meta
            exec = exec.exec
            device = device.device
            GC.@preserve exec device begin
                outputs, future_res, future = xla_execute_sharded(
                    exec, device, inputs, donated_args, Val(n_outs)
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
                MLIR.API.XLAExecuteSharded(
                    exec,
                    $N,
                    Base.RefValue(inputs),
                    Base.RefValue(device),
                    Base.RefValue(is_arg_donatable),
                    $n_outs,
                    outputs_p,
                    futures,
                    futures_res,
                )
            end
            outputs = outputs_p[]
            future_res = futures_res[]
            future = futures[] != 0
            return ($(results...),)
        end
    end
end

# TODO(#2235): Fix this
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

    GC.@preserve exec inputs donated_args outputs future_res futures begin
        MLIR.API.XLAExecute(
            exec.exec,
            N,
            Base.RefValue(inputs),
            Base.RefValue(donated_args),
            n_outs,
            Base.unsafe_convert(Ptr{Cvoid}, outputs),
            Base.unsafe_convert(Ptr{UInt8}, futures),
            Base.unsafe_convert(Ptr{Cvoid}, future_res),
        )
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
