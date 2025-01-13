module Compiler

using Reactant_jll

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
    (isbitstype(T) || ancestor(obj) isa RArray) && return Base.setfield!(obj, field, val)
    return Base.setindex!(obj, val, field)
end

function create_result(tocopy::T, path, result_stores) where {T}
    if !isstructtype(typeof(tocopy))
        error("cannot copy $tocopy of type $(Core.Typeof(tocopy))")
    end

    elems = Union{Symbol,Expr}[]

    for i in 1:fieldcount(T)
        # If the field is undefined we don't set it. A common example for this is `du2`
        # for Tridiagonal
        isdefined(tocopy, i) || continue
        ev = create_result(getfield(tocopy, i), append_path(path, i), result_stores)
        push!(elems, ev)
    end

    return Expr(:new, T, elems...)
end

function create_result(tocopy::ConcreteRNumber{T}, path, result_stores) where {T}
    if haskey(result_stores, path)
        restore = result_stores[path]
        delete!(result_stores, path)
        return :(ConcreteRNumber{$T}($restore))
    end
    # We will set the data for this later
    return :(ConcreteRNumber{$T}($(tocopy.data)))
end

function create_result(tocopy::ConcreteRArray{T,N}, path, result_stores) where {T,N}
    if haskey(result_stores, path)
        restore = result_stores[path]
        delete!(result_stores, path)
        return :(ConcreteRArray{$T,$N}($restore, $(tocopy.shape)))
    end
    # We will set the data for this later
    return :(ConcreteRArray{$T,$N}($(tocopy.data), $(tocopy.shape)))
end

function create_result(tocopy::Array{T,N}, path, result_stores) where {T,N}
    elems = Expr[]
    for (i, v) in enumerate(tocopy)
        push!(elems, create_result(v, append_path(path, i), result_stores))
    end
    # TODO is there a way to not call `reshape` here? what expr is used for array literals?
    return :(reshape($T[$(elems...)], $(size(tocopy))...))
end

function create_result(tocopy::Tuple, path, result_stores)
    elems = Union{Symbol,Expr}[]
    for (k, v) in pairs(tocopy)
        push!(elems, create_result(v, append_path(path, k), result_stores))
    end
    return :(($(elems...),))
end

function create_result(tocopy::NamedTuple{K,T}, path, result_stores) where {K,T}
    elems = Union{Symbol,Expr}[]
    for (i, (k, v)) in enumerate(pairs(tocopy))
        push!(elems, create_result(v, append_path(path, i), result_stores))
    end
    return :(NamedTuple{$K}(($(elems...),)))
end

function create_result(tocopy::D, path, result_stores) where {K,V,D<:AbstractDict{K,V}}
    elems = Expr[]
    for (i, p) in enumerate(pairs(tocopy))
        push!(elems, create_result(p, append_path(path, i), result_stores))
    end
    return :($D([$(elems...)]))
end

function create_result(
    tocopy::Union{Integer,AbstractFloat,AbstractString,Nothing,Type,Symbol,Char},
    path,
    result_stores,
)
    return Meta.quot(tocopy)
end

# Optimization passes via transform dialect
function optimization_passes(; no_nan::Bool=false)
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
        "replace_neg_add_with_subtract",
        "log_const_prop<1>",
        "log_plus_one_const_prop<1>",
    ]
    if no_nan
        append!(
            transform_passes_list,
            ["no_nan", "no_nan_self_sub_simplify", "no_nan_add_sub_simplify"],
        )
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
    return join(
        [
            "inline{default-pipeline=canonicalize max-iterations=4}",
            "libdevice-funcs-raise",
            func_passes,
        ],
        ',',
    )
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

function compile_mlir(f, args; kwargs...)
    ctx = MLIR.IR.Context(Reactant.registry[], false)
    context_gc_vector[ctx] = Vector{TracedRArray}(undef, 0)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
    results = MLIR.IR.context!(ctx) do
        mod = MLIR.IR.Module(MLIR.IR.Location())
        evalinfo = compile_mlir!(mod, f, args; kwargs...)
        return (mod, evalinfo...)
    end
    Base.delete!(context_gc_vector, ctx)
    return results
end

const cuLaunch = Ref{UInt}(0)
const cuFunc = Ref{UInt}(0)
const cuModule = Ref{UInt}(0)

function compile_mlir!(mod, f, args; optimize::Union{Bool,Symbol}=true, no_nan::Bool=false)
    # Explicitly don't use block! to avoid creating a closure, which creates
    # both compile-time and relocatability issues

    MLIR.IR.activate!(mod)
    MLIR.IR.activate!(MLIR.IR.body(mod))
    fnwrapped,
    func2, traced_result, result, seen_args, ret, linear_args, in_tys,
    linear_results = try
        Reactant.TracedUtils.make_mlir_fn(f, args, (), "main", true)
    finally
        MLIR.IR.deactivate!(MLIR.IR.body(mod))
        MLIR.IR.deactivate!(mod)
    end

    concrete_seen = OrderedIdDict()

    concrete_result = make_tracer(
        concrete_seen, traced_result, ("result",), TracedToConcrete
    )

    optimize isa Bool && (optimize = ifelse(optimize, :all, :none))

    toolkit = ""
    if isdefined(Reactant_jll, :ptxas_path)
        toolkit = Reactant_jll.ptxas_path[1:(end - length("/bin/ptxas"))]
    end
    kern = "lower-kernel{run_init=true toolkitPath=$toolkit cuLaunchKernelPtr=$(cuLaunch[]) cuModuleLoadDataPtr=$(cuModule[]) cuModuleGetFunctionPtr=$(cuFunc[])},symbol-dce"

    opt_passes = optimization_passes(; no_nan)

    if optimize === :all
        run_pass_pipeline!(mod, join([opt_passes, "enzyme-batch", opt_passes], ","))
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
                    opt_passes,
                    kern,
                ],
                ',',
            ),
        )
    elseif optimize === :before_kernel
        run_pass_pipeline!(mod, join([opt_passes, "enzyme-batch", opt_passes], ","))
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
                    opt_passes,
                ],
                ',',
            ),
        )
    elseif optimize === :no_enzyme
        run_pass_pipeline!(mod, join([opt_passes, "enzyme-batch", opt_passes], ","))
        run_pass_pipeline!(mod, "arith-raise{stablehlo=true}"; enable_verifier=false)
        run_pass_pipeline!(
            mod,
            join(
                [
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    opt_passes,
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
                    opt_passes,
                    kern,
                ],
                ',',
            ),
        )
    elseif optimize === :before_enzyme
        run_pass_pipeline!(mod, join([opt_passes, "enzyme-batch", opt_passes], ","))
        run_pass_pipeline!(
            mod, "$enzyme_pass,arith-raise{stablehlo=true}"; enable_verifier=false
        )
        run_pass_pipeline!(
            mod, "canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math," * kern
        )
    elseif optimize !== :none
        error("Invalid optimize option: $(Meta.quot(optimize))")
    end

    preserved_args = Tuple{TracedType,Int}[]
    results = [MLIR.IR.operand(ret, i) for i in 1:MLIR.IR.noperands(ret)]
    nresults = MLIR.IR.Value[]
    linear_results2 = TracedType[]
    for (i, op) in enumerate(results)
        if !MLIR.IR.is_block_arg(op)
            push!(nresults, op)
            push!(linear_results2, linear_results[i])
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

    func3 = MLIR.Dialects.func.func_(;
        sym_name="main",
        function_type=MLIR.IR.FunctionType(in_tys, out_tys2),
        body=MLIR.IR.Region(),
    )
    MLIR.API.mlirRegionTakeBody(MLIR.IR.region(func3, 1), MLIR.IR.region(func2, 1))

    push!(MLIR.IR.body(mod), func3)

    MLIR.API.mlirOperationDestroy(func2.operation)
    func2.operation = MLIR.API.MlirOperation(C_NULL)

    return linear_args,
    linear_results2, preserved_args, seen_args, concrete_result,
    fnwrapped
end

"""
    @code_hlo [optimize = ...] [no_nan = <true/false>] f(args...)
"""
macro code_hlo(args...)
    default_options = Dict{Symbol,Any}(:optimize => true, :no_nan => false)
    compile_expr, (; compiled) = compile_call_expr(
        __module__, compile_mlir, default_options, args...
    )
    return esc(:($(compile_expr);
    $(first)($(compiled))))
end

"""
    @compile [optimize = ...] [no_nan = <true/false>] [sync = <true/false>] f(args...)
"""
macro compile(args...)
    default_options = Dict{Symbol,Any}(:optimize => true, :sync => false, :no_nan => false)
    return esc(first(compile_call_expr(__module__, compile, default_options, args...)))
end

"""
    @jit [optimize = ...] [no_nan = <true/false>] [sync = <true/false>] f(args...)

Run @compile f(args..) then immediately execute it
"""
macro jit(args...)
    default_options = Dict{Symbol,Any}(:optimize => true, :sync => false, :no_nan => false)
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

function correct_maybe_bcast_call(fname)
    startswith(string(fname), '.') || return false, fname, fname
    return true, Symbol(string(fname)[2:end]), fname
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
function codegen_flatten!(linear_args, result_stores)
    flatten_names = Symbol[]
    flatten_code = Expr[]
    # resarg_code = Expr[]

    for (i, arg) in enumerate(linear_args)
        paths = ((p for p in Reactant.TracedUtils.get_paths(arg) if p[1] == :args)...,)
        path = if length(paths) == 1
            paths[1]
        else
            throw(
                "Invalid path duplication $(Reactant.TracedUtils.get_paths(arg)) into $(paths)",
            )
        end

        usbuf = Symbol(:usbuf_, i)
        sbuf = Symbol(:sbuf_, i)
        push!(flatten_names, sbuf)

        flatcode = :(getindex(args, $(path[2])))
        for p in path[3:end]
            flatcode = :(traced_getfield($flatcode, $(Meta.quot(p))))
        end
        push!(flatten_code, :($usbuf = $flatcode.data))
        push!(flatten_code, :($sbuf = XLA.synced_buffer($usbuf)))

        # TODO: unused for the time being
        # respaths = ((p for p in Reactant.TracedUtils.get_paths(arg) if p[1] == :result || p[1] == :resargs)...,)

        # resarg = false
        # for respath in respaths
        #     if respath[1] == :result
        #         flatcode = :result
        #         respath = respath[2:end]
        #         result_stores[respath] = usbuf
        #         resarg = true
        #     else
        #         @assert respath[1] == :resargs
        #         if respath[2] != path[2]
        #             continue
        #         end
        #         # flatcode = :(args[$(respath[2])])
        #         path = path[3:end]
        #     end
        #     # for p in path
        #     #     flatcode = :(traced_getfield($flatcode, $(Meta.quot(p))))
        #     # end
        #     # resarg = true
        #     # flatcode = :($flatcode.data = $usbuf)
        #     # @show flatcode
        #     # push!(flatten_code, res)
        # end
        # if resarg
        #     push!(resarg_code, :($usbuf = $flatcode.data))
        # end
    end
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
)
    cache_dict = gensym("cache_dict")
    unflatten_code = Expr[:(
        $cache_dict = $(IdDict{
            Union{TracedRArray,TracedRNumber},Union{ConcreteRArray,ConcreteRNumber}
        }())
    ),]

    # mutate the result stores to point to the correct concrete results
    for (concrete_res_name, result) in zip(concretized_res_names, linear_results)
        paths = (
            (
                p for p in Reactant.TracedUtils.get_paths(result) if
                p[1] == :result || p[1] == :resargs
            )...,
        )
        for path in paths
            if path[1] == :result
                unflatcode = :result
                path = path[2:end]
                result_stores[path] = concrete_res_name
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
    result_code = create_result(concrete_result, (), result_stores)
    postkeys = collect(keys(result_stores))
    used = [t for t in prevkeys if !in(t, postkeys)]

    # if some argument is mutated, change them to point to the correct concrete results
    for (result, arg_idx) in preserved_args
        paths = (
            (
                p for p in Reactant.TracedUtils.get_paths(result) if
                p[1] == :result || p[1] == :resargs || p[1] == :args
            )...,
        )

        for path in paths
            arg = linear_args[arg_idx + 1]
            argpath = only((
                p for p in Reactant.TracedUtils.get_paths(arg) if p[1] == :args
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
function codegen_xla_call(exec, flatten_names, donated_args_mask, nresults)
    flatten_buffer_refs = map(n -> :($n.buffer), flatten_names)

    concretized_res_names = Symbol[Symbol(:concrete_res_, i) for i in 1:nresults]
    concretized_res_code = map(enumerate(concretized_res_names)) do (i, varname)
        :($varname = linearized_results[$i])
    end

    xla_call_code = if nresults == 0
        :()
    else
        quote
            GC.@preserve $(flatten_names...) begin
                linearized_results = XLA.ExecutableCall(
                    $exec,
                    ($(flatten_buffer_refs...),),
                    $(Tuple(donated_args_mask)),
                    Val($nresults),
                )
            end
            $(concretized_res_code...)
        end
    end

    return concretized_res_names, xla_call_code
end

function compile_xla(f, args; client=nothing, optimize=true, no_nan=false)
    # register MLIR dialects
    ctx = MLIR.IR.Context(Reactant.registry[], false)
    context_gc_vector[ctx] = Vector{TracedRArray}(undef, 0)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid

    MLIR.IR.activate!(ctx)
    results = try
        # compile function to MLIR module
        mod = MLIR.IR.Module(MLIR.IR.Location())
        linear_args, linear_results, preserved_args, seen_args, concrete_result, isclosure = compile_mlir!(
            mod, f, args; optimize, no_nan
        )

        if isnothing(client)
            if length(linear_args) > 0
                for (k, _) in Iterators.filter(((_, v),) -> v isa TracedRArray, seen_args)
                    client = XLA.client(k.data)
                end
            end
            if isnothing(client)
                client = XLA.default_backend[]
            end
        end

        # compile MLIR module to XLA executable
        exec = XLA.Compile(client, mod)
        return exec,
        linear_args, linear_results, preserved_args, seen_args, concrete_result,
        isclosure
    finally
        MLIR.IR.deactivate!(ctx)
    end
    Base.delete!(context_gc_vector, ctx)
    return results
end

function compile(f, args; client=nothing, optimize=true, sync=false, no_nan=false)
    exec, linear_args, linear_results, preserved_args, seen_args, concrete_result, isclosure = compile_xla(
        f, args; client, optimize, no_nan
    )

    preserved_args_idx = last.(preserved_args)
    donated_args_mask = map(1:length(linear_args)) do i
        UInt8(i ∉ preserved_args_idx)
    end

    fnwrap = isclosure ? f : nothing
    closure_ty = typeof(fnwrap)

    result_stores = Dict{Tuple,Symbol}()

    # generate Julia `Thunk` code
    flatten_arg_names, flatten_code = codegen_flatten!(linear_args, result_stores)

    concretized_res_names, xla_call_code = codegen_xla_call(
        exec, flatten_arg_names, donated_args_mask, length(linear_results)
    )

    unflatten_code = codegen_unflatten!(
        linear_args,
        preserved_args,
        concretized_res_names,
        linear_results,
        concrete_result,
        result_stores,
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

    return register_thunk(fname, Tuple{map(Core.Typeof, args)...}, body, f, isclosure)
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

end
