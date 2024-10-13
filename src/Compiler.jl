module Compiler

import ..Reactant:
    Reactant,
    MLIR,
    XLA,
    ConcreteRArray,
    TracedRArray,
    TracedRNumber,
    OrderedIdDict,
    make_tracer,
    TracedToConcrete,
    append_path,
    TracedType

@inline traced_getfield(@nospecialize(obj), field) = Base.getfield(obj, field)

function create_result(tocopy::T, path, result_stores) where {T}
    if !isstructtype(typeof(tocopy))
        error("cannot copy $tocopy of type $(Core.Typeof(tocopy))")
    end

    elems = Union{Symbol,Expr}[]

    for i in 1:fieldcount(T)
        ev = create_result(getfield(tocopy, i), append_path(path, i), result_stores)
        push!(elems, ev)
    end

    return Expr(:new, T, elems...)
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
    return :($T[$(elems...)])
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
    tocopy::Union{Integer,AbstractFloat,AbstractString,Nothing,Type,Symbol},
    path,
    result_stores,
)
    return Meta.quot(tocopy)
end

const opt_passes::String = join(
    [
        "inline{default-pipeline=canonicalize max-iterations=4}",
        "canonicalize,cse",
        "canonicalize",
        "enzyme-hlo-generate-td{" *
        join(
            [
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
            ],
            ';',
        ) *
        "}",
        "transform-interpreter",
        "enzyme-hlo-remove-transform",
    ],
    ',',
)

function run_pass_pipeline!(mod, pass_pipeline)
    pm = MLIR.IR.PassManager()
    opm = MLIR.IR.OpPassManager(pm)
    MLIR.IR.add_pipeline!(opm, pass_pipeline)
    MLIR.IR.run!(pm, mod)
    return mod
end

function compile_mlir(f, args; kwargs...)
    ctx = MLIR.IR.Context()
    Base.append!(Reactant.registry[]; context=ctx)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
    MLIR.IR.context!(ctx) do
        mod = MLIR.IR.Module(MLIR.IR.Location())
        evalinfo = compile_mlir!(mod, f, args; kwargs...)
        return mod, evalinfo...
    end
end

function compile_mlir!(mod, f, args; optimize=true)
    fnwrapped,
    func2, traced_result, result, seen_args, ret, linear_args, in_tys,
    linear_results = MLIR.IR.mmodule!(mod) do
        MLIR.IR.block!(MLIR.IR.body(mod)) do
            return Reactant.make_mlir_fn(f, args, (), "main", true)
        end
    end

    concrete_seen = OrderedIdDict()

    concrete_result = make_tracer(
        concrete_seen, traced_result, ("result",), TracedToConcrete
    )

    if optimize
        run_pass_pipeline!(
            mod,
            join(
                [
                    opt_passes,
                    "enzyme-batch",
                    opt_passes,
                    "enzyme",
                    "arith-raise{stablehlo=true}",
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    opt_passes,
                ],
                ',',
            ),
        )
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
    @code_hlo [optimize = ...] f(args...)
"""
macro code_hlo(options, maybe_call=nothing)
    call = something(maybe_call, options)
    options = isnothing(maybe_call) ? :(optimize = true) : options
    Meta.isexpr(call, :call) || error("@code_hlo: expected call, got $call")
    if !Meta.isexpr(options, :(=)) || options.args[1] != :optimize
        error("@code_hlo: expected options in format optimize=value, got $options")
    end

    options = Expr(:tuple, Expr(:parameters, Expr(:kw, options.args...)))

    quote
        options = $(esc(options))
        f = $(esc(call.args[1]))
        args = $(esc(Expr(:vect, call.args[2:end]...)))

        mod, _... = compile_mlir(f, args; optimize=options.optimize)
        return mod
    end
end

"""
    @compile f(args...)
"""
macro compile(options, maybe_call=nothing)
    call = something(maybe_call, options)
    options = isnothing(maybe_call) ? :(optimize = true) : options
    Meta.isexpr(call, :call) || error("@compile: expected call, got $call")
    if !Meta.isexpr(options, :(=)) || options.args[1] != :optimize
        error("@compile: expected options in format optimize=value, got $options")
    end

    options = Expr(:tuple, Expr(:parameters, Expr(:kw, options.args...)))

    quote
        f = $(esc(call.args[1]))
        args = $(esc(Expr(:tuple, call.args[2:end]...)))
        compile(f, args)
    end
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
        paths = ((p for p in arg.paths if p[1] == :args)...,)
        path = if length(paths) == 1
            paths[1]
        else
            throw("Invalid path duplication $(arg.paths) into $(paths)")
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

        # TODO
        respaths = ((p for p in arg.paths if p[1] != :args)...,)

        # resarg = false
        for respath in respaths
            if respath[1] == :result
                flatcode = :result
                respath = respath[2:end]
                result_stores[respath] = usbuf
                resarg = true
            else
                @assert respath[1] == :resargs
                if respath[2] != path[2]
                    continue
                end
                # flatcode = :(args[$(respath[2])])
                path = path[3:end]
            end
            # for p in path
            #     flatcode = :(traced_getfield($flatcode, $(Meta.quot(p))))
            # end
            # resarg = true
            # flatcode = :($flatcode.data = $usbuf)
            # @show flatcode
            # push!(flatten_code, res)
        end
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
    unflatten_code = Expr[]

    # mutate the result stores to point to the correct concrete results
    for (concrete_res_name, result) in zip(concretized_res_names, linear_results)
        paths = ((p for p in result.paths if p[1] != :args)...,)
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
            end

            # unroll path tree
            for p in path
                unflatcode = :(traced_getfield($unflatcode, $(Meta.quot(p))))
            end
            unflatcode = :($unflatcode.data = $concrete_res_name)

            push!(unflatten_code, unflatcode)
        end
    end

    prevkeys = collect(keys(result_stores))
    result_code = create_result(concrete_result, (), result_stores)
    postkeys = collect(keys(result_stores))
    used = [t for t in prevkeys if !in(t, postkeys)]

    # if some argument is mutated, change them to point to the correct concrete results
    for (result, arg_idx) in preserved_args
        for path in result.paths
            arg = linear_args[arg_idx + 1]
            argpath = only((p for p in arg.paths if p[1] == :args))

            if path[1] == :result
                res = :result
                path = path[2:end]
                if in(path, used) # TODO
                    continue
                end
            else
                @assert path[1] == :resargs || path[1] == :args
                # We can optimize cases where we set the arg to itself
                if path[2:end] == argpath[2:end]
                    continue
                end
                res = :(args[path[2]])
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

function compile_xla(f, args; client=nothing)
    # register MLIR dialects
    ctx = MLIR.IR.Context()
    append!(Reactant.registry[]; context=ctx)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid

    return MLIR.IR.context!(ctx) do
        # compile function to MLIR module
        mod = MLIR.IR.Module(MLIR.IR.Location())
        linear_args, linear_results, preserved_args, seen_args, concrete_result, isclosure = compile_mlir!(
            mod, f, args; optimize=true
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
    end
end

function compile(f, args; client=nothing)
    exec, linear_args, linear_results, preserved_args, seen_args, concrete_result, isclosure = compile_xla(
        f, args
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

    fname = gensym(Symbol(Symbol(f), :_reactant))
    expr = :(function $fname(args...)
        $(
            # if `f` is a closure, then prepend the closure into `args`
            # the closure fields will be correctly extracted from it as the tracer has already passed through it
            if !(closure_ty <: Nothing)
                :(args = ($fnwrap, args...))
            end
        )
        $(flatten_code...)
        $xla_call_code
        $(unflatten_code...)
        return result
    end)

    body = expr.args[2]
    return register_thunk(fname, body)
end

# inspired by RuntimeGeneratedFunction.jl
const __thunk_body_cache = Dict{Symbol,Expr}()

struct Thunk{tag} end

@generated function (thunk::Thunk{tag})(args...) where {tag}
    return __thunk_body_cache[tag]
end

function register_thunk(tag, body)
    __thunk_body_cache[tag] = body
    return Thunk{tag}()
end

end
