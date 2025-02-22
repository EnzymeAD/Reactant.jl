module Compiler

using Reactant_jll
using Libdl: dlsym

import ..Reactant:
    Reactant,
    MLIR,
    XLA,
    ConcreteRArray,
    ConcreteRNumber,
    TracedRArray,
    TracedRNumber,
    RArray,
    RNumber,
    OrderedIdDict,
    make_tracer,
    TracedToConcrete,
    append_path,
    ancestor,
    TracedType

import ..ReactantCore: correct_maybe_bcast_call

@inline function traced_getfield(@nospecialize(obj::Dict), field)
    return Base.getindex(obj, field)
end

@inline function traced_getfield(@nospecialize(obj), field)
    return Base.getfield(obj, field)
end

@inline function traced_getfield(@nospecialize(obj::AbstractArray{T}), field) where {T}
    (isbitstype(T) || ancestor(obj) isa RArray) && return Base.getfield(obj, field)
    return Base.getindex(obj, field)
end

@inline traced_setfield!(@nospecialize(obj), field, val) = Base.setfield!(obj, field, val)
@inline function traced_setfield!(
    @nospecialize(obj::AbstractArray{T}), field, val
) where {T}
    ancestor_obj = ancestor(obj)
    (isbitstype(T) || ancestor_obj isa RArray) && return Base.setfield!(obj, field, val)
    return Base.setindex!(obj, val, field)
end

@inline function traced_setfield!(@nospecialize(obj::Dict), field, val)
    return Base.setindex!(obj, field, val)
end

function create_result(
    tocopy::T, path, result_stores, path_to_shard_info, sharding_mesh
) where {T}
    if !isstructtype(typeof(tocopy))
        error("cannot copy $tocopy of type $(Core.Typeof(tocopy))")
    end

    elems = Union{Symbol,Expr}[]

    for i in 1:fieldcount(T)
        # If the field is undefined we don't set it. A common example for this is `du2`
        # for Tridiagonal
        isdefined(tocopy, i) || continue
        ev = create_result(
            getfield(tocopy, i),
            append_path(path, i),
            result_stores,
            path_to_shard_info,
            sharding_mesh,
        )
        push!(elems, ev)
    end

    return Expr(:new, T, elems...)
end

function __reconstruct_shardinfo(path, path_to_shard_info, sharding_mesh)
    device_to_array_slices, partition_spec = path_to_shard_info[path]
    delete!(path_to_shard_info, path)
    sharding = Reactant.Sharding.NamedSharding(sharding_mesh, partition_spec)
    return Reactant.Sharding.ShardInfo(sharding, device_to_array_slices)
end

function create_result(
    tocopy::ConcreteRNumber{T,D,S}, path, result_stores, path_to_shard_info, sharding_mesh
) where {T,D,S}
    if haskey(result_stores, path)
        restore = result_stores[path]
        delete!(result_stores, path)
        if path_to_shard_info !== nothing # restore sharding
            sharding = __reconstruct_shardinfo(path, path_to_shard_info, sharding_mesh)
            return :(ConcreteRNumber{$T,length($(restore)),$(typeof(sharding))}(
                ($(restore)...,), $sharding
            ))
        else
            return :(ConcreteRNumber{$T}($restore))
        end
    end

    if path_to_shard_info !== nothing # restore sharding
        sharding = __reconstruct_shardinfo(path, path_to_shard_info, sharding_mesh)
        return :(ConcreteRNumber{$T,length($(tocopy.data)),$(typeof(sharding))}(
            ($(tocopy.data...,)), $sharding
        ))
    end
    # We will set the data for this later
    return :(ConcreteRNumber{$T}($(tocopy.data)))
end

function create_result(
    tocopy::ConcreteRArray{T,N,D,S}, path, result_stores, path_to_shard_info, sharding_mesh
) where {T,N,D,S}
    if haskey(result_stores, path)
        restore = result_stores[path]
        delete!(result_stores, path)
        if path_to_shard_info !== nothing # restore sharding
            sharding = __reconstruct_shardinfo(path, path_to_shard_info, sharding_mesh)
            return :(ConcreteRArray{$T,$N,length($(restore)),$(typeof(sharding))}(
                ($(restore)...,), $(tocopy.shape), $sharding
            ))
        else
            return :(ConcreteRArray{$T,$N}($restore, $(tocopy.shape)))
        end
    end

    if path_to_shard_info !== nothing # restore sharding
        sharding = __reconstruct_shardinfo(path, path_to_shard_info, sharding_mesh)
        return :(ConcreteRArray{$T,$N,length($(tocopy.data)),$(typeof(sharding))}(
            ($(tocopy.data)...,), $(tocopy.shape), $sharding
        ))
    end
    # We will set the data for this later
    return :(ConcreteRArray{$T,$N,$D,$S}(
        $(tocopy.data), $(tocopy.shape), $(tocopy.sharding)
    ))
end

function create_result(
    tocopy::Array{T,N}, path, result_stores, path_to_shard_info, sharding_mesh
) where {T,N}
    elems = Expr[]
    for (i, v) in enumerate(tocopy)
        push!(
            elems,
            create_result(
                v, append_path(path, i), result_stores, path_to_shard_info, sharding_mesh
            ),
        )
    end
    # TODO is there a way to not call `reshape` here? what expr is used for array literals?
    return :(reshape($T[$(elems...)], $(size(tocopy))...))
end

function create_result(
    tocopy::Tuple, path, result_stores, path_to_shard_info, sharding_mesh
)
    elems = Union{Symbol,Expr}[]
    for (k, v) in pairs(tocopy)
        push!(
            elems,
            create_result(
                v, append_path(path, k), result_stores, path_to_shard_info, sharding_mesh
            ),
        )
    end
    return :(($(elems...),))
end

function create_result(
    tocopy::NamedTuple{K,T}, path, result_stores, path_to_shard_info, sharding_mesh
) where {K,T}
    elems = Union{Symbol,Expr}[]
    for (i, (k, v)) in enumerate(pairs(tocopy))
        push!(
            elems,
            create_result(
                v, append_path(path, i), result_stores, path_to_shard_info, sharding_mesh
            ),
        )
    end
    return :(NamedTuple{$K}(($(elems...),)))
end

function create_result(
    tocopy::D, path, result_stores, path_to_shard_info, sharding_mesh
) where {K,V,D<:AbstractDict{K,V}}
    elems = Expr[]
    for (i, p) in enumerate(pairs(tocopy))
        push!(
            elems,
            create_result(
                p, append_path(path, i), result_stores, path_to_shard_info, sharding_mesh
            ),
        )
    end
    return :($D([$(elems...)]))
end

function create_result(
    tocopy::Union{Integer,AbstractFloat,AbstractString,Nothing,Type,Symbol,Char},
    path,
    result_stores,
    path_to_shard_info,
    sharding_mesh,
)
    return Meta.quot(tocopy)
end

# Optimization passes via transform dialect
function optimization_passes(; no_nan::Bool=false, sroa::Bool=false, inline::Bool=true)
    transform_passes_list = [
        "patterns=compare_op_canon<16>",
        "transpose_transpose<16>",
        "broadcast_in_dim_op_canon<16>",
        "convert_op_canon<16>",
        "dynamic_broadcast_in_dim_op_not_actually_dynamic<16>",
        "chained_dynamic_broadcast_in_dim_canonicalization<16>",
        "dynamic_broadcast_in_dim_all_dims_non_expanding<16>",
        "noop_reduce_op_canon<16>",
        "empty_reduce_op_canon<16>",
        "dynamic_reshape_op_canon<16>",
        "get_tuple_element_op_canon<16>",
        "real_op_canon<16>",
        "imag_op_canon<16>",
        "conj_complex_negate<16>",
        "get_dimension_size_op_canon<16>",
        "gather_op_canon<16>",
        "reshape_op_canon<16>",
        "merge_consecutive_reshapes<16>",
        "transpose_is_reshape<16>",
        "zero_extent_tensor_canon<16>",
        "reorder_elementwise_and_shape_op<16>",
        "chlo_inf_const_prop<16>",
        "gamma_const_prop<16>",
        "cse_broadcast_in_dim<16>",
        "cse_slice<16>",
        "cse_transpose<16>",
        "cse_convert<16>",
        "cse_pad<16>",
        "cse_dot_general<16>",
        "cse_reshape<16>",
        "cse_mul<16>",
        "cse_div<16>",
        "cse_add<16>",
        "cse_subtract<16>",
        "cse_min<16>",
        "cse_max<16>",
        "cse_neg<16>",
        "cse_concatenate<16>",
        "concatenate_op_canon<16>(1024)",
        "select_op_canon<16>(1024)",
        "add_simplify<16>",
        "sub_simplify<16>",
        "and_simplify<16>",
        "max_simplify<16>",
        "min_simplify<16>",
        "or_simplify<16>",
        "negate_simplify<16>",
        "mul_simplify<16>",
        "div_simplify<16>",
        "rem_simplify<16>",
        "pow_simplify<16>",
        "sqrt_simplify<16>",
        "cos_simplify<16>",
        "sin_simplify<16>",
        "noop_slice<16>",
        "noop_reverse<16>",
        "const_prop_through_barrier<16>",
        "slice_slice<16>",
        "shift_right_logical_simplify<16>",
        "pad_simplify<16>",
        "negative_pad_to_slice<16>",
        "tanh_simplify<16>",
        "exp_simplify<16>",
        "slice_simplify<16>",
        "convert_simplify<16>",
        "dynamic_slice_to_static<16>",
        "dynamic_update_slice_elim<16>",
        "concat_to_broadcast<16>",
        "reduce_to_reshape<16>",
        "broadcast_to_reshape<16>",
        "gather_simplify<16>",
        "iota_simplify<16>(1024)",
        "broadcast_in_dim_simplify<16>(1024)",
        "convert_concat<1>",
        "dynamic_update_to_concat<1>",
        "slice_of_dynamic_update<1>",
        "slice_elementwise<1>",
        "slice_pad<1>",
        "dot_reshape_dot<1>",
        "concat_const_prop<1>",
        "concat_fuse<1>",
        "pad_reshape_pad<1>",
        "pad_pad<1>",
        "concat_push_binop_add<1>",
        "concat_push_binop_mul<1>",
        "scatter_to_dynamic_update_slice<1>",
        "reduce_concat<1>",
        "slice_concat<1>",
        "concat_slice<1>",
        "select_op_used_within_if<1>",
        "bin_broadcast_splat_add<1>",
        "bin_broadcast_splat_subtract<1>",
        "bin_broadcast_splat_div<1>",
        "bin_broadcast_splat_mul<1>",
        "reshape_iota<16>",
        "slice_reshape_slice<1>",
        "dot_general_simplify<16>",
        "transpose_simplify<16>",
        "reshape_empty_broadcast<1>",
        "add_pad_pad_to_concat<1>",
        "broadcast_reshape<1>",
        "slice_reshape_concat<1>",
        "slice_reshape_elementwise<1>",
        "slice_reshape_transpose<1>",
        "slice_reshape_dot_general<1>",
        "concat_pad<1>",
        "reduce_pad<1>",
        "broadcast_pad<1>",
        "zero_product_reshape_pad<1>",
        "mul_zero_pad<1>",
        "div_zero_pad<1>",
        "binop_const_reshape_pad<1>",
        "binop_const_pad_add<1>",
        "binop_const_pad_subtract<1>",
        "binop_const_pad_mul<1>",
        "binop_const_pad_div<1>",
        "slice_reshape_pad<1>",
        "binop_binop_pad_pad_add<1>",
        "binop_binop_pad_pad_mul<1>",
        "binop_pad_pad_add<1>",
        "binop_pad_pad_subtract<1>",
        "binop_pad_pad_mul<1>",
        "binop_pad_pad_div<1>",
        "binop_pad_pad_min<1>",
        "binop_pad_pad_max<1>",
        "unary_pad_push_convert<1>",
        "unary_pad_push_tanh<1>",
        "unary_pad_push_exp<1>",
        "transpose_pad<1>",
        "transpose_dot_reorder<1>",
        "dot_transpose<1>",
        "transpose_einsum<1>",
        "einsum_transpose<1>",
        "transpose_convolution<1>",
        "convolution_transpose<1>",
        "convert_convert_float<1>",
        "concat_to_pad<1>",
        "concat_appending_reshape<1>",
        "reshape_iota<1>",
        "broadcast_reduce<1>",
        "slice_dot_general<1>",
        "dot_reshape_pad<1>",
        "pad_dot_general<1>(0)",
        "dot_reshape_pad<1>",
        "pad_dot_general<1>(1)",
        "if_inline<1>",
        "if_to_select<1>",
        "dynamic_update_slice_const_prop",
        "dynamic_gather_op_is_not_dynamic<16>",
        "divide_sqrt_to_multiply_rsqrt<16>",
        "binary_op_transpose_simplify_add",
        "binary_op_transpose_simplify_sub",
        "binary_op_transpose_simplify_mul",
        "binary_op_transpose_simplify_div",
        "binary_op_transpose_simplify_min",
        "binary_op_transpose_simplify_max",
        "binary_op_transpose_simplify_pow",
        "binary_op_transpose_simplify_rem",
        "binary_op_transpose_simplify_or",
        "binary_op_transpose_simplify_and",
        "binary_op_transpose_simplify_xor",
        "associative_binary_op_reordering<1>",
        "transpose_unary_transpose_abs",
        "transpose_unary_transpose_neg",
        "transpose_unary_transpose_sqrt",
        "transpose_unary_transpose_rsqrt",
        "transpose_unary_transpose_ceil",
        "transpose_unary_transpose_convert",
        "transpose_unary_transpose_cosine",
        "transpose_unary_transpose_exp",
        "transpose_unary_transpose_expm1",
        "transpose_unary_transpose_log",
        "transpose_unary_transpose_log1p",
        "transpose_unary_transpose_sign",
        "transpose_unary_transpose_sine",
        "transpose_unary_transpose_tanh",
        "transpose_broadcast_in_dim_to_broadcast_in_dim<16>",
        "scatter_indices_are_unique",
        "transpose_reduce_simplify",
        "replace_neg_add_with_subtract",
        "log_const_prop<1>",
        "log_plus_one_const_prop<1>",
        "binop_const_simplify",
        "transpose_broadcast_in_dim_to_broadcast_in_dim",
        "not_select_simplify",
        "scatter_update_computation_const_prop",
        "common_compare_expression_rewrite",
        "compare_select_simplify",
        "while_simplify<1>",
        "scatter_update_computation_const_prop",
        "if_remove_unused",
    ]
    if no_nan
        append!(
            transform_passes_list,
            ["no_nan", "no_nan_self_sub_simplify", "no_nan_add_sub_simplify(1)"],
        )
    else
        push!(transform_passes_list, "no_nan_add_sub_simplify(0)")
    end
    transform_passes = join(
        [
            "enzyme-hlo-generate-td{" * join(transform_passes_list, ';') * "}",
            "transform-interpreter",
            "enzyme-hlo-remove-transform",
        ],
        ",",
    )
    func_passes = join(["canonicalize", "cse", "canonicalize", transform_passes], ",")
    passes = String[]
    if inline
        push!(passes, "inline{default-pipeline=canonicalize max-iterations=4}")
    end
    if sroa
        push!(passes, "propagate-constant-bounds")
        if DUMP_LLVMIR[]
            push!(
                passes,
                "sroa-wrappers{dump_prellvm=true dump_postllvm=true instcombine=false instsimplify=true}",
            )
        else
            push!(passes, "sroa-wrappers{instcombine=false instsimplify=true}")
        end
        push!(passes, "canonicalize")
        push!(passes, "sroa-wrappers{instcombine=false instsimplify=true}")
        push!(passes, "libdevice-funcs-raise")
        push!(passes, "canonicalize")
        push!(passes, "remove-duplicate-func-def")
    end
    push!(passes, func_passes)
    return join(passes, ',')
end

# TODO we want to be able to run the more advanced passes via transform dialect as an enzyme intermediate
# However, this errs as we cannot attach the transform with to the funcop itself [as we run a functionpass].
const enzyme_pass::String = "enzyme{postpasses=\"arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize\"}"

function run_pass_pipeline!(mod, pass_pipeline; enable_verifier=true)
    pm = MLIR.IR.PassManager()
    MLIR.IR.enable_verifier!(pm, enable_verifier)
    opm = MLIR.IR.OpPassManager(pm)
    MLIR.IR.add_pipeline!(opm, pass_pipeline)
    MLIR.IR.run!(pm, mod)
    return mod
end

const context_gc_vector = Dict{MLIR.IR.Context,Vector{TracedRArray}}()

# helper for debug purposes: String -> Text
function run_pass_pipeline_on_source(source, pass_pipeline; enable_verifier=true)
    ctx = MLIR.IR.Context(Reactant.registry[], false)
    context_gc_vector[ctx] = Vector{TracedRArray}(undef, 0)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
    result = MLIR.IR.context!(ctx) do
        mod = parse(MLIR.IR.Module, source)
        run_pass_pipeline!(mod, pass_pipeline; enable_verifier)
        MLIR.IR.verifyall(MLIR.IR.Operation(mod); debug=true)
        Text(repr(mod))
    end
    Base.delete!(context_gc_vector, ctx)
    return result
end

function compile_mlir(f, args; client=nothing, kwargs...)
    ctx = MLIR.IR.Context(Reactant.registry[], false)
    context_gc_vector[ctx] = Vector{TracedRArray}(undef, 0)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid

    if client !== nothing
        backend = XLA.platform_name(client)
    else
        backend = XLA.platform_name(XLA.default_backend[])
    end
    if backend == "CUDA"
        backend = "GPU"
    elseif backend == "CPU"
        backend = "cpu"
    end

    results = MLIR.IR.context!(ctx) do
        mod = MLIR.IR.Module(MLIR.IR.Location())

        mlir_fn_res = compile_mlir!(mod, f, args; backend, kwargs...)

        client, _ = __resolve_device_and_client(
            client,
            mlir_fn_res.seen_args,
            mlir_fn_res.linear_args,
            mlir_fn_res.is_sharded,
        )

        # Attach a name, and partitioning attributes to the module
        __add_mhlo_attributes_and_name!(
            mod, f; mlir_fn_res.num_partitions, mlir_fn_res.num_replicas
        )

        return mod, mlir_fn_res
    end
    Base.delete!(context_gc_vector, ctx)

    return results
end

const PartitionKA = Ref{Bool}(true)

const cubinChip = Ref{String}("sm_60")
const cubinFormat = Ref{String}("bin")
const cuindexBitWidth = Ref{Int}(32)
const cuOptLevel = Ref{Int}(2)
# Wgatever the relevant highest version from our LLVM is within NVPTX.td
# Or more specifically looking at clang/lib/Driver/ToolChains/Cuda.cpp:684
#  We see relevant ptx version is CUDA 12.6 -> 85
#                                      12.2 -> 82
#                                      11.8 -> 78
function cubinFeatures()
    ver = @ccall MLIR.API.mlir_c.ReactantCudaDriverGetVersion()::UInt32
    # No cuda available
    if ver == 0
        return "+ptx86"
    end
    ver2 = @ccall MLIR.API.mlir_c.ReactantHermeticCudaGetVersion()::UInt32
    ver = min(ver, ver2)
    major, ver = divrem(ver, 1000)
    minor, patch = divrem(ver, 10)
    version = VersionNumber(major, minor, patch)
    # From https://github.com/llvm/llvm-project/blob/106c483a102e1328f11e2b1d9398f4ad2826b59f/clang/lib/Driver/ToolChains/Cuda.cpp#L685
    cuver_map = Dict([
        (126, 85),
        (125, 85),
        (124, 84),
        (123, 83),
        (122, 82),
        (121, 81),
        (120, 80),
        (118, 78),
        (117, 77),
        (116, 76),
        (115, 75),
        (114, 74),
        (113, 73),
        (112, 72),
        (111, 71),
        (110, 70),
        (102, 65),
        (101, 64),
        (100, 63),
        (92, 61),
        (91, 61),
        (90, 60),
    ])
    mver = major * 10 + minor
    if mver > 126
        return 86
    end
    ptx = cuver_map[mver]
    return "+ptx$ptx"
end

const DEBUG_KERNEL = Ref{Bool}(false)
const DUMP_LLVMIR = Ref{Bool}(false)

const Raise = Ref{Bool}(false)

function compile_mlir!(
    mod,
    f,
    args,
    callcache=Dict{
        Vector,
        @NamedTuple{
            f_name::String,
            mlir_result_types::Vector{MLIR.IR.Type},
            traced_result::Any,
            mutated_args::Vector{Int},
        }
    }();
    optimize::Union{Bool,Symbol}=true,
    no_nan::Bool=false,
    backend="gpu",
)
    # Explicitly don't use block! to avoid creating a closure, which creates
    # both compile-time and relocatability issues

    MLIR.IR.activate!(mod)
    MLIR.IR.activate!(MLIR.IR.body(mod))
    activate_callcache!(callcache)

    mlir_fn_res = try
        Reactant.TracedUtils.make_mlir_fn(f, args, (), "main", true)
    finally
        deactivate_callcache!(callcache)
        MLIR.IR.deactivate!(MLIR.IR.body(mod))
        MLIR.IR.deactivate!(mod)
    end
    (; fnwrapped, traced_result, seen_args, ret, linear_args, in_tys, linear_results) =
        mlir_fn_res
    compiled_f = mlir_fn_res.f

    concrete_seen = OrderedIdDict()

    concrete_result = make_tracer(
        concrete_seen, traced_result, ("result",), TracedToConcrete
    )

    optimize isa Bool && (optimize = ifelse(optimize, :all, :none))

    toolkit = ""
    if isdefined(Reactant_jll, :ptxas_path)
        toolkit = Reactant_jll.ptxas_path[1:(end - length("/bin/ptxas"))]
    end

    if backend == "cpu"
        kern = "lower-kernel{backend=cpu},canonicalize"
        jit = "lower-jit{openmp=true backend=cpu},symbol-dce"
    elseif DEBUG_KERNEL[]
        curesulthandler = dlsym(
            Reactant_jll.libReactantExtra_handle, "ReactantHandleCuResult"
        )
        @assert curesulthandler !== nothing
        curesulthandler = Base.reinterpret(UInt, curesulthandler)
        kern = if Raise[]
            "lower-kernel{backend=cpu},symbol-dce,canonicalize"
        else
            "lower-kernel,canonicalize"
        end
        jit = "lower-jit{debug=true cuResultHandlerPtr=$curesulthandler cuOptLevel=$(cuOptLevel[]) cubinFormat=$(cubinFormat[]) indexBitWidth=$(cuindexBitWidth[])  cubinChip=$(cubinChip[]) cubinFeatures=$(cubinFeatures()) run_init=true toolkitPath=$toolkit},symbol-dce"
    else
        kern = if Raise[]
            "lower-kernel{backend=cpu},symbol-dce,canonicalize"
        else
            "lower-kernel,canonicalize"
        end
        jit = "lower-jit{cuOptLevel=$(cuOptLevel[]) indexBitWidth=$(cuindexBitWidth[]) cubinFormat=$(cubinFormat[]) cubinChip=$(cubinChip[]) cubinFeatures=$(cubinFeatures()) run_init=true toolkitPath=$toolkit},symbol-dce"
    end

    opt_passes = optimization_passes(; no_nan, sroa=true)
    opt_passes2 = optimization_passes(; no_nan, sroa=false)

    raise = if Raise[]
        "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,raise-affine-to-stablehlo,arith-raise{stablehlo=true}," *
        opt_passes2
    else
        "canonicalize"
    end

    if optimize === :all
        run_pass_pipeline!(mod, join([opt_passes, "enzyme-batch", opt_passes2], ","))
        run_pass_pipeline!(
            mod, "$enzyme_pass,arith-raise{stablehlo=true}"; enable_verifier=false
        )
        run_pass_pipeline!(
            mod,
            join(
                [
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    opt_passes2,
                    kern,
                    raise,
		    jit
                ],
                ',',
            ),
        )
    elseif optimize === :before_kernel
        run_pass_pipeline!(mod, join([opt_passes, "enzyme-batch", opt_passes2], ","))
        run_pass_pipeline!(
            mod, "$enzyme_pass,arith-raise{stablehlo=true}"; enable_verifier=false
        )
        run_pass_pipeline!(
            mod,
            join(
                [
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    opt_passes2,
                ],
                ',',
            ),
        )
    elseif optimize === :before_jit
        run_pass_pipeline!(mod, join([opt_passes, "enzyme-batch", opt_passes2], ","))
        run_pass_pipeline!(
            mod, "$enzyme_pass,arith-raise{stablehlo=true}"; enable_verifier=false
        )
        run_pass_pipeline!(
            mod,
            join(
                [
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    opt_passes2,
                    kern,
                    raise,
                ],
                ',',
            ),
        )
    elseif optimize === :before_raise
        run_pass_pipeline!(mod, join([opt_passes, "enzyme-batch", opt_passes2], ","))
        run_pass_pipeline!(
            mod, "$enzyme_pass,arith-raise{stablehlo=true}"; enable_verifier=false
        )
        run_pass_pipeline!(
            mod,
            join(
                [
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    opt_passes2,
                    kern,
                ],
                ',',
            ),
        )
    elseif optimize === :no_enzyme
        run_pass_pipeline!(mod, join([opt_passes, "enzyme-batch", opt_passes2], ","))
        run_pass_pipeline!(mod, "arith-raise{stablehlo=true}"; enable_verifier=false)
        run_pass_pipeline!(
            mod,
            join(
                [
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    opt_passes2,
                ],
                ',',
            ),
        )
    elseif optimize === :only_enzyme
        run_pass_pipeline!(mod, "enzyme-batch")
        run_pass_pipeline!(
            mod, "$enzyme_pass,arith-raise{stablehlo=true}"; enable_verifier=false
        )
        run_pass_pipeline!(
            mod,
            join(
                ["canonicalize", "remove-unnecessary-enzyme-ops", "enzyme-simplify-math"],
                ',',
            ),
        )
    elseif optimize === :after_enzyme
        run_pass_pipeline!(mod, "enzyme-batch")
        run_pass_pipeline!(
            mod, "$enzyme_pass,arith-raise{stablehlo=true}"; enable_verifier=false
        )
        run_pass_pipeline!(
            mod,
            join(
                [
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    opt_passes2,
                    kern,
                    raise,
                    jit,
                ],
                ',',
            ),
        )
    elseif optimize === :before_enzyme
        run_pass_pipeline!(mod, join([opt_passes, "enzyme-batch", opt_passes2], ","))
        run_pass_pipeline!(
            mod, "$enzyme_pass,arith-raise{stablehlo=true}"; enable_verifier=false
        )
        run_pass_pipeline!(
            mod,
            join(
                [
                    "canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math",
                    kern,
                    raise,
                    jit,
                ],
                ',',
            ),
        )
    elseif optimize === :canonicalize
        run_pass_pipeline!(mod, "canonicalize")
    elseif optimize === :just_batch
        run_pass_pipeline!(mod, "enzyme-batch")
    elseif optimize !== :none
        error("Invalid optimize option: $(Meta.quot(optimize))")
    end

    preserved_args = Tuple{TracedType,Int}[]
    results = [MLIR.IR.operand(ret, i) for i in 1:MLIR.IR.noperands(ret)]
    nresults = MLIR.IR.Value[]
    linear_results2 = TracedType[]
    results_mask = falses(length(results))
    for (i, op) in enumerate(results)
        if !MLIR.IR.is_block_arg(op)
            push!(nresults, op)
            push!(linear_results2, linear_results[i])
            results_mask[i] = true
            continue
        end
        push!(preserved_args, (linear_results[i], MLIR.IR.block_arg_num(op)))
    end

    fnbody = MLIR.IR.block(ret)
    MLIR.API.mlirOperationDestroy(ret.operation)
    ret.operation = MLIR.API.MlirOperation(C_NULL)
    MLIR.IR.block!(fnbody) do
        return MLIR.Dialects.func.return_(nresults)
    end

    out_tys2 = [MLIR.IR.type(a) for a in nresults]

    res_attrs = MLIR.IR.attr(compiled_f, "res_attrs")
    if res_attrs isa MLIR.IR.Attribute
        res_attrs = [
            res_attrs[i - 1] for (i, present) in enumerate(results_mask) if present
        ]
    end

    func3 = MLIR.Dialects.func.func_(;
        sym_name="main",
        function_type=MLIR.IR.FunctionType(in_tys, out_tys2),
        arg_attrs=MLIR.IR.attr(compiled_f, "arg_attrs"),
        res_attrs,
        no_inline=MLIR.IR.attr(compiled_f, "no_inline"),
        body=MLIR.IR.Region(),
    )
    MLIR.API.mlirRegionTakeBody(MLIR.IR.region(func3, 1), MLIR.IR.region(compiled_f, 1))

    push!(MLIR.IR.body(mod), func3)

    MLIR.API.mlirOperationDestroy(compiled_f.operation)
    compiled_f.operation = MLIR.API.MlirOperation(C_NULL)

    return Reactant.TracedUtils.CompiledMlirFnResult(
        fnwrapped,
        func3,
        traced_result,
        mlir_fn_res.result,
        seen_args,
        ret,
        linear_args,
        in_tys,
        linear_results2,
        mlir_fn_res.num_partitions,
        mlir_fn_res.num_replicas,
        mlir_fn_res.is_sharded,
        preserved_args,
        concrete_result,
        mlir_fn_res.sharding_mesh,
        mlir_fn_res.mutated_args,
    )
end

"""
    @code_hlo [optimize = ...] [no_nan = <true/false>] f(args...)

See also [`@code_xla`](@ref), [`@code_mhlo`](@ref).
"""
macro code_hlo(args...)
    default_options = Dict{Symbol,Any}(
        :optimize => true, :no_nan => false, :client => nothing
    )
    compile_expr, (; compiled) = compile_call_expr(
        __module__, compile_mlir, default_options, args...
    )
    #! format: off
    return esc(
        :(
            $(compile_expr);
            $(first)($(compiled))
        )
    )
    #! format: on
end

"""
    @code_mhlo [optimize = ...] [no_nan = <true/false>] f(args...)

Similar to `@code_hlo`, but prints the module after running the XLA compiler.

See also [`@code_xla`](@ref), [`@code_hlo`](@ref).
"""
macro code_mhlo(args...)
    default_options = Dict{Symbol,Any}(
        :optimize => true, :no_nan => false, :client => nothing
    )
    compile_expr, (; compiled) = compile_call_expr(
        __module__, compile_xla, default_options, args...
    )
    #! format: off
    return esc(
        :(
            $(compile_expr);
            $(first)($(compiled))
        )
    )
    #! format: on
end

"""
    @code_xla [optimize = ...] [no_nan = <true/false>] f(args...)

Similar to `@code_hlo`, but prints the HLO module.

See also [`@code_mhlo`](@ref), [`@code_hlo`](@ref).
"""
macro code_xla(args...)
    default_options = Dict{Symbol,Any}(
        :optimize => true, :no_nan => false, :client => nothing
    )
    compile_expr, (; compiled) = compile_call_expr(
        __module__, compile_xla, default_options, args...
    )
    #! format: off
    return esc(
        :(
            $(compile_expr);
            exec = $(compiled)[2];
            hlo_modules = $(XLA.get_hlo_modules)(exec);
            length(hlo_modules) == 1 ? only(hlo_modules) : hlo_modules
        )
    )
    #! format: on
end

"""
    @compile [optimize = ...] [no_nan = <true/false>] [sync = <true/false>] f(args...)
"""
macro compile(args...)
    default_options = Dict{Symbol,Any}(
        :optimize => true, :sync => false, :no_nan => false, :client => nothing
    )
    return esc(first(compile_call_expr(__module__, compile, default_options, args...)))
end

"""
    @jit [optimize = ...] [no_nan = <true/false>] [sync = <true/false>] f(args...)

Run @compile f(args..) then immediately execute it
"""
macro jit(args...)
    default_options = Dict{Symbol,Any}(
        :optimize => true, :sync => false, :no_nan => false, :client => nothing
    )
    compile_expr, (; compiled, args) = compile_call_expr(
        __module__, compile, default_options, args...
    )
    #! format: off
    return esc(
        :(
            $(compile_expr);
            $(compiled)($(args)...)
        )
    )
    #! format: on
end

function compile_call_expr(mod, compiler, options, args...)
    while length(args) > 1
        option, args = args[1], args[2:end]
        if !Meta.isexpr(option, :(=))
            error("Invalid option $(option)")
        else
            option_name = option.args[1]
            @assert haskey(options, option_name) "Invalid option $(option_name)"
            options[option_name] = option.args[2]
        end
    end
    call = only(args)
    f_symbol = gensym(:f)
    args_symbol = gensym(:args)
    compiled_symbol = gensym(:compiled)

    if Meta.isexpr(call, :call)
        bcast, fname, fname_full = correct_maybe_bcast_call(call.args[1])
        fname = if bcast
            quote
                if isdefined(mod, $(Meta.quot(fname_full)))
                    $(fname_full)
                else
                    Base.Broadcast.BroadcastFunction($(fname))
                end
            end
        else
            :($(fname))
        end
        args_rhs = Expr(:tuple, call.args[2:end]...)
    elseif Meta.isexpr(call, :(.), 2) && Meta.isexpr(call.args[2], :tuple)
        fname = :($(Base.Broadcast.BroadcastFunction)($(call.args[1])))
        args_rhs = only(call.args[2:end])
    else
        error("Invalid function call: $(call)")
    end

    return quote
        $(f_symbol) = $(fname)
        $(args_symbol) = $(args_rhs)
        $(compiled_symbol) = $(compiler)(
            $(f_symbol), $(args_symbol); $(Expr.(:kw, keys(options), values(options))...)
        )
    end,
    (; compiled=compiled_symbol, args=args_symbol)
end

"""
    codegen_flatten!

Generate Julia code to extract the XLA buffers from input arguments.
The name is due to its similarity to the `flatten` function in `jax.tree_util.register_pytree_node`.

# Arguments

- `linear_args`: A list of arguments to be flattened.

# Returns

- `flatten_names`: A list of `Symbol`s representing the names of the flattened arguments.
- `flatten_code`: A list of `Expr`s to extract the XLA buffers from the input arguments.

# Note

The _linearized arguments_ do not directly refer to the  are the arguments that have been flattened into a single list.
"""
function codegen_flatten!(
    linear_args,
    seen_args,
    result_stores,
    is_sharded::Bool,
    mesh,
    linear_parameter_shardings,
    client,
)
    flatten_names = Symbol[]
    flatten_code = Expr[]

    if is_sharded
        inv_seen_args = Reactant.OrderedIdDict()
        for (k, v) in seen_args
            inv_seen_args[v] = k
        end
    end

    for (i, arg) in enumerate(linear_args)
        paths = (
            (
                p for
                p in Reactant.TracedUtils.get_paths(arg) if length(p) > 0 && p[1] == :args
            )...,
        )
        path = if length(paths) == 1
            paths[1]
        else
            throw(
                "Invalid path duplication $(Reactant.TracedUtils.get_paths(arg)) into $(paths)",
            )
        end

        usbuf = Symbol(:usbuf_, i)

        flatcode = :(getindex(args, $(path[2])))
        for p in path[3:end]
            flatcode = :(traced_getfield($flatcode, $(Meta.quot(p))))
        end

        if is_sharded
            carg = inv_seen_args[arg]
            device_ids = mesh.sorted_device_ids
            if Reactant.Sharding.is_sharded(carg)
                # Currently disabling the error since we roundtrip from MHLO to generate
                # the shardings
                # # Check if the sharding provided is same as the one we have
                # arg_condensed_op_sharding = Reactant.Sharding.XLA.CondensedOpSharding(
                #     Reactant.Sharding.ShardingWithShape(carg.sharding, size(carg))
                # )
                # @assert arg_condensed_op_sharding == condensed_op_sharding "Sharding provided by the user ($arg_condensed_op_sharding) does not match the sharding computed by XLA ($condensed_op_sharding). This generally means that Reactant.jl made an error in generating the executable. Please open an issue with the error message and an MWE."

                push!(flatten_code, :($usbuf = $flatcode.data))
                for j in 1:length(mesh)
                    sbuf = Symbol(:sbuf_, i, "_", device_ids[j])
                    push!(flatten_names, sbuf)
                    push!(flatten_code, :($sbuf = XLA.synced_buffer(getindex($usbuf, $j))))
                end
            else
                condensed_op_sharding = convert(
                    Reactant.Sharding.XLA.CondensedOpSharding, linear_parameter_shardings[i]
                )
                push!(flatten_code, :($usbuf = $flatcode))
                device_to_array_slices = XLA.sharding_to_concrete_array_indices(
                    condensed_op_sharding, size(carg), mesh
                )
                for j in 1:length(mesh)
                    local_device_id = device_ids[j]
                    buf = Symbol(:buf_, i, :_, local_device_id)
                    slice = device_to_array_slices[j]
                    push!(
                        flatten_code,
                        :($buf = XLA.synced_buffer(only($usbuf[$(slice)...].data))),
                    )
                    sbuf = Symbol(:sbuf_, i, :_, local_device_id)
                    device = XLA.get_addressable_device(client, local_device_id)
                    push!(flatten_names, sbuf)
                    push!(flatten_code, :($sbuf = XLA.copy_buffer_to_device($buf, $device)))
                end
            end
        else
            push!(flatten_code, :($usbuf = $flatcode.data))
            sbuf = Symbol(:sbuf_, i)
            push!(flatten_names, sbuf)
            if arg isa TracedRArray || arg isa TracedRNumber
                push!(flatten_code, :($sbuf = only(XLA.synced_buffer($usbuf))))
            else
                error("Unsupported type $(typeof(arg))")
            end
        end
    end

    # We reorder how the buffers are passed to the XLA call
    is_sharded &&
        (flatten_names = vcat(eachrow(reshape(flatten_names, length(mesh), :))...))

    return flatten_names, flatten_code
end

"""
    codegen_unflatten!

Generate Julia code to wrap the XLA buffers back into the output result datatypes.
The name is due to its similarity to the `unflatten` function in `jax.tree_util.register_pytree_node`.
"""
function codegen_unflatten!(
    linear_args,
    preserved_args,
    concretized_res_names,
    linear_results,
    concrete_result,
    result_stores,
    path_to_shard_info,
    linear_result_shard_info,
    sharding_mesh,
)
    cache_dict = gensym("cache_dict")
    has_cache_dict = false
    unflatten_code = Expr[]

    # mutate the result stores to point to the correct concrete results
    for (concrete_res_name, result, shard_info) in
        zip(concretized_res_names, linear_results, linear_result_shard_info)
        paths = (
            (
                p for p in Reactant.TracedUtils.get_paths(result) if
                length(p) > 0 && (p[1] == :result || p[1] == :resargs)
            )...,
        )
        for path in paths
            if path[1] == :result
                unflatcode = :result
                path = path[2:end]
                result_stores[path] = concrete_res_name
                if path_to_shard_info !== nothing
                    path_to_shard_info[path] = shard_info
                end
                continue
            else
                @assert path[1] == :resargs
                unflatcode = :(args[$(path[2])])
                path = path[3:end]

                for p in path[1:(end - 1)]
                    unflatcode = :(traced_getfield($unflatcode, $(Meta.quot(p))))
                end

                if length(path) > 0
                    final_val = gensym("final_val")
                    clocal = gensym("clocal")
                    if !has_cache_dict
                        has_cache_dict = true
                        push!(
                            unflatten_code,
                            :(
                                $cache_dict = $(IdDict{
                                    Union{TracedRArray,TracedRNumber},
                                    Union{ConcreteRArray,ConcreteRNumber},
                                }())
                            ),
                        )
                    end
                    unflatcode = quote
                        $final_val = traced_getfield($unflatcode, $(Meta.quot(path[end])))
                        if $final_val isa TracedRArray
                            $clocal = if haskey($cache_dict, $final_val)
                                $cache_dict[$final_val]
                            else
                                $cache_dict[$final_val] = ConcreteRArray{
                                    $(Reactant.unwrapped_eltype)($final_val),
                                    ndims($final_val),
                                }(
                                    $concrete_res_name, size($final_val)
                                )
                                $cache_dict[$final_val]
                            end
                            traced_setfield!($unflatcode, $(Meta.quot(path[end])), $clocal)
                        elseif $final_val isa TracedRNumber
                            $clocal = if haskey($cache_dict, $final_val)
                                $cache_dict[$final_val]
                            else
                                $cache_dict[$final_val] = ConcreteRNumber{
                                    $(Reactant.unwrapped_eltype)($final_val)
                                }(
                                    $concrete_res_name
                                )
                                $cache_dict[$final_val]
                            end
                            traced_setfield!($unflatcode, $(Meta.quot(path[end])), $clocal)
                        else
                            traced_setfield!($final_val, :data, $concrete_res_name)
                        end
                    end
                else
                    unflatcode = :(traced_setfield!($unflatcode, :data, $concrete_res_name))
                end
                push!(unflatten_code, unflatcode)
            end
        end
    end

    prevkeys = collect(keys(result_stores))
    result_code = create_result(
        concrete_result, (), result_stores, path_to_shard_info, sharding_mesh
    )
    postkeys = collect(keys(result_stores))
    used = [t for t in prevkeys if !in(t, postkeys)]

    # if some argument is mutated, change them to point to the correct concrete results
    for (result, arg_idx) in preserved_args
        paths = (
            (
                p for p in Reactant.TracedUtils.get_paths(result) if
                length(p) > 0 && (p[1] == :result || p[1] == :resargs || p[1] == :args)
            )...,
        )

        for path in paths
            arg = linear_args[arg_idx + 1]
            argpath = only((
                p for
                p in Reactant.TracedUtils.get_paths(arg) if length(p) > 0 && p[1] == :args
            ))

            if path[1] == :result
                res = :result
                path = path[2:end]
                if in(path, used) # TODO
                    continue
                end
            else
                @assert path[1] == :resargs || path[1] == :args "Expected :resargs or :args, got $(path[1])"
                # We can optimize cases where we set the arg to itself
                if path[2:end] == argpath[2:end]
                    continue
                end
                res = :(args[$(path[2])])
                path = path[3:end]
            end
            for p in path
                res = :(traced_getfield($res, $(Meta.quot(p))))
            end

            argres = :(args[$(argpath[2])])
            for p in argpath[3:end]
                argres = :(traced_getfield($argres, $(Meta.quot(p))))
            end

            res = :($res.data = $argres.data)
            push!(unflatten_code, res)
        end
    end

    # generate return object which stores the concrete results in some arbitrary way
    pushfirst!(unflatten_code, :(result = $result_code))
    # push!(unflatten_code, :(return result))

    return unflatten_code
end

"""
    codegen_xla_call

Generate Julia code to call the XLA executable.

# Arguments

- `exec`: The XLA executable to call.
- `flatten_names`: A list of `Symbol`s representing the names of the flattened linear arguments.
- `donated_args_mask`: A list of `UInt8`s representing whether the argument is donated.
- `nresults`: The number of results to expect.
"""
function codegen_xla_call(
    exec,
    device,
    flatten_names,
    donated_args_mask,
    nresults,
    is_sharded::Bool,
    ndevices::Int,
)
    flatten_buffer_refs = map(n -> :($n.buffer), flatten_names)

    base_symbol_name = is_sharded ? Symbol(:result_buffer_m, ndevices, :_) : :result_buffer_
    concretized_res_names = Symbol[Symbol(base_symbol_name, i) for i in 1:nresults]
    concretized_res_code = map(enumerate(concretized_res_names)) do (i, varname)
        :($varname = linearized_results[$i])
    end

    xla_call_code = if nresults == 0
        :()
    else
        if is_sharded
            quote
                GC.@preserve $(flatten_names...) begin
                    linearized_results = XLA.execute(
                        $exec,
                        ($(flatten_buffer_refs...),),
                        $(Tuple(donated_args_mask)),
                        Val($nresults),
                        Val($ndevices),
                    )
                end
                $(concretized_res_code...)
            end
        else
            quote
                GC.@preserve $(flatten_names...) begin
                    linearized_results = XLA.execute_sharded(
                        $exec,
                        $(device),
                        ($(flatten_buffer_refs...),),
                        $(Tuple(donated_args_mask)),
                        Val($nresults),
                    )
                end
                $(concretized_res_code...)
            end
        end
    end

    return concretized_res_names, xla_call_code
end

function __add_mhlo_attributes_and_name!(mod::MLIR.IR.Module, f; kwargs...)
    fname = string(f)
    length(fname) > 10 && (fname = fname[1:7] * "...")
    __add_mhlo_attributes_and_name!(mod, fname; kwargs...)
    return nothing
end

function __add_mhlo_attributes_and_name!(
    mod::MLIR.IR.Module, fname::String; num_partitions=1, num_replicas=1
)
    moduleop = MLIR.IR.Operation(mod)
    module_name = Reactant.TracedUtils.__lookup_unique_name_in_module(
        mod, "reactant_" * fname
    )
    module_name = MLIR.IR.Attribute(module_name)
    MLIR.IR.attr!(moduleop, "mhlo.num_partitions", MLIR.IR.Attribute(num_partitions))
    MLIR.IR.attr!(moduleop, "mhlo.num_replicas", MLIR.IR.Attribute(num_replicas))
    MLIR.IR.attr!(
        moduleop, String(MLIR.API.mlirSymbolTableGetSymbolAttributeName()), module_name
    )
    return nothing
end

function __resolve_device_and_client(client, seen_args, linear_args, is_sharded)
    if is_sharded
        client === nothing && (client = XLA.default_backend[])
        return client, nothing
    end

    device = nothing
    if length(linear_args) > 0
        devices_list = [
            XLA.device(only(k.data)) for (k, v) in seen_args if v isa TracedRArray
        ]
        if !isempty(devices_list)
            if !allequal(devices_list)
                msg = "Expected all arguments to be on the same device, got:\n"
                for (i, device) in enumerate(devices_list)
                    msg *= "    Device $(i): $(string(device))\n"
                end
                throw(ArgumentError(msg))
            end
            @assert allequal(devices_list) "All arguments must be on the same device: $(devices_list)"
            device = first(devices_list)
        end
    end

    if client === nothing
        if device !== nothing
            client = XLA.client(device)
        else
            client = XLA.default_backend[]
            device = XLA.get_addressable_device(client, XLA.default_device_idx[])
        end
    else
        if device !== nothing
            @assert client == XLA.client(device) "client ($(client)) and XLA.client(device) ($(XLA.client(device))) must be the same"
        else
            device = XLA.get_addressable_device(client, XLA.default_device_idx[])
        end
    end

    return (client, device)
end

function compile_xla(f, args; client=nothing, kwargs...)
    # register MLIR dialects
    ctx = MLIR.IR.Context(Reactant.registry[], false)
    context_gc_vector[ctx] = Vector{TracedRArray}(undef, 0)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid

    if client !== nothing
        backend = XLA.platform_name(client)
    else
        backend = XLA.platform_name(XLA.default_backend[])
    end
    if backend == "CUDA"
        backend = "GPU"
    elseif backend == "CPU"
        backend = "cpu"
    end

    MLIR.IR.activate!(ctx)
    results = try
        # compile function to MLIR module
        mod = MLIR.IR.Module(MLIR.IR.Location())
        mlir_fn_res = compile_mlir!(mod, f, args; backend, kwargs...)

        # Resolve client and device
        client, device = __resolve_device_and_client(
            client,
            mlir_fn_res.seen_args,
            mlir_fn_res.linear_args,
            mlir_fn_res.is_sharded,
        )

        # Attach a name, and partitioning attributes to the module
        __add_mhlo_attributes_and_name!(
            mod, f; mlir_fn_res.num_partitions, mlir_fn_res.num_replicas
        )

        # compile MLIR module to XLA executable
        local_device_ids = if mlir_fn_res.is_sharded
            collect(Int64, mlir_fn_res.sharding_mesh.sorted_device_ids)
        else
            Int64[]
        end
        mlir_fn_res.is_sharded && (device = nothing)

        exec = XLA.compile(
            client,
            device,
            mod;
            num_outputs=length(mlir_fn_res.linear_results),
            num_parameters=length(mlir_fn_res.linear_args),
            mlir_fn_res.is_sharded,
            local_device_ids,
        )

        return mod, exec, mlir_fn_res, device, client
    finally
        MLIR.IR.deactivate!(ctx)
    end

    Base.delete!(context_gc_vector, ctx)
    return results
end

function compile(f, args; sync=false, kwargs...)
    _, exec, mlir_fn_res, device, client = compile_xla(f, args; kwargs...)
    (; linear_args, seen_args, linear_results, preserved_args, concrete_result) =
        mlir_fn_res

    preserved_args_idx = last.(preserved_args)
    donated_args_mask = map(1:length(linear_args)) do i
        UInt8(i ∉ preserved_args_idx)
    end

    result_stores = Dict{Tuple,Symbol}()
    path_to_shard_info = mlir_fn_res.is_sharded ? Dict{Tuple,Tuple}() : nothing

    # generate Julia `Thunk` code
    flatten_arg_names, flatten_code = codegen_flatten!(
        linear_args,
        seen_args,
        result_stores,
        mlir_fn_res.is_sharded,
        mlir_fn_res.sharding_mesh,
        XLA.get_parameter_shardings(exec),
        client,
    )

    concretized_res_names, xla_call_code = codegen_xla_call(
        exec,
        device,
        flatten_arg_names,
        donated_args_mask,
        length(linear_results),
        mlir_fn_res.is_sharded,
        mlir_fn_res.is_sharded ? length(mlir_fn_res.sharding_mesh) : 1,
    )

    linear_result_shard_info = if mlir_fn_res.is_sharded
        output_shardings = XLA.get_output_shardings(exec)
        XLA.compute_array_indices_and_partition_spec.(
            output_shardings,
            size.(mlir_fn_res.linear_results),
            (mlir_fn_res.sharding_mesh,),
        )
    else
        ntuple(Returns(nothing), length(linear_results))
    end

    unflatten_code = codegen_unflatten!(
        linear_args,
        preserved_args,
        concretized_res_names,
        linear_results,
        concrete_result,
        result_stores,
        path_to_shard_info,
        linear_result_shard_info,
        mlir_fn_res.sharding_mesh,
    )

    sync_call = if sync
        calls = []
        for name in concretized_res_names
            push!(calls, :(XLA.synced_buffer($(name))))
        end
        Expr(:block, calls...)
    else
        :()
    end

    fname = gensym(Symbol(Symbol(f), :_reactant))

    body = quote
        $(flatten_code...)
        $(xla_call_code)
        $(sync_call)
        $(unflatten_code...)
        return result
    end

    return register_thunk(
        fname, Tuple{map(Core.Typeof, args)...}, body, f, mlir_fn_res.fnwrapped
    )
end

# inspired by RuntimeGeneratedFunction.jl
const __thunk_body_cache = Dict{Symbol,Expr}()

struct Thunk{FTy,tag,IsClosure,ArgTypes}
    f::FTy
end

struct MisMatchedThunkTypeError{ThunkTy,FoundTypes} <: Base.Exception end

function Base.showerror(
    io::IO, ece::MisMatchedThunkTypeError{Thunk{FTy,tag,ArgTypes,IsClosure},FoundTypes}
) where {FTy,tag,ArgTypes,FoundTypes,IsClosure}
    print(
        io,
        "\nThe Reactant-compiled function `$(Thunk{FTy, tag, ArgTypes, IsClosure})` exists, but no method is defined for this combination of argument types.",
    )
    print(
        io,
        "\nYou passed in arguments with types\n\t(" *
        join(FoundTypes.parameters, ", ") *
        ")",
    )
    return print(
        io,
        "\nHowever the method you are calling was compiled for arguments with types\n\t(" *
        join(ArgTypes.parameters, ", ") *
        ")",
    )
end

@generated function (thunk::Thunk{FTy,tag,ArgTypes,IsClosure})(
    args...
) where {FTy,tag,ArgTypes,IsClosure}
    FoundTypes = Tuple{args...}
    if ArgTypes != FoundTypes
        return quote
            throw(
                $(MisMatchedThunkTypeError{Thunk{FTy,tag,ArgTypes,IsClosure},FoundTypes}())
            )
        end
    end
    body = __thunk_body_cache[tag]
    if IsClosure
        return quote
            args = (thunk.f, args...)
            $body
        end
    else
        return body
    end
end

function register_thunk(
    tag::Symbol, @nospecialize(argtys::Type), body::Expr, @nospecialize(f), isclosure::Bool
)
    __thunk_body_cache[tag] = body
    return Thunk{Core.Typeof(f),tag,argtys,isclosure}(f)
end

function activate_callcache!(callcache)
    stack = get!(task_local_storage(), :callcache) do
        return []
    end
    push!(stack, callcache)
    return nothing
end

function deactivate_callcache!(callcache)
    callcache === last(task_local_storage(:callcache)) ||
        error("Deactivating wrong callcache")
    return pop!(task_local_storage(:callcache))
end

function _has_callcache()
    return haskey(task_local_storage(), :callcache) &&
           !Base.isempty(task_local_storage(:callcache))
end

function callcache(; throw_error::Bool=true)
    if !_has_callcache()
        throw_error && error("No callcache is active")
        return nothing
    end
    return last(task_local_storage(:callcache))
end

function callcache!(f, callcache)
    activate_callcache!(callcache)
    try
        return f()
    finally
        deactivate_callcache!(callcache)
    end
end

end
