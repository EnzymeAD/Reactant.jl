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
                "reshape_simplify<16>",
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

struct CompiledModule
    mod::MLIR.IR.Module
    ctx::MLIR.IR.Context
end

Base.show(io::IO, cm::CompiledModule) = show(io, cm.mod)

function compile_to_module!(mod, f, args; optimize=true)
    fnwrapped,
    func2, traced_result, result, seen_args, ret, linear_args, in_tys,
    linear_results = MLIR.IR.mmodule!(mod) do
        MLIR.IR.block!(MLIR.IR.body(mod)) do
            return make_mlir_fn(f, args, (), "main", true)
        end
    end

    concrete_seen = IdDict()

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

    preserved_args = Tuple{TracedRArray,Int}[]
    results = [MLIR.IR.operand(ret, i) for i in 1:MLIR.IR.noperands(ret)]
    nresults = MLIR.IR.Value[]
    linear_results2 = TracedRArray[]
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
    Meta.isexpr(call, :call) || error("@code_mlir: expected call, got $call")
    if !Meta.isexpr(options, :(=)) || options.args[1] != :optimize
        error("@code_mlir: expected options in format optimize=value, got $options")
    end

    options = Expr(:tuple, Expr(:parameters, Expr(:kw, options.args...)))

    quote
        options = $(esc(options))
        f = $(esc(call.args[1]))
        args = $(esc(Expr(:vect, call.args[2:end]...)))

        ctx = MLIR.IR.Context()
        Base.append!(registry[]; context=ctx)
        @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
        MLIR.IR.context!(ctx) do
            mod = MLIR.IR.Module(MLIR.IR.Location())
            compile_to_module!(mod, f, args; optimize=options.optimize)
            CompiledModule(mod, ctx)
        end
    end
end

traced_getfield(obj, field) = Base.getfield(obj, field)

struct MakeConcreteRArray{T,N}
    shape::NTuple{N,Int}
end
struct MakeArray{AT,Vals} end
struct MakeString{AT,Val} end
struct MakeStruct{AT,Val} end
struct MakeVal{AT} end
struct MakeSymbol{AT} end

function make_valable(tocopy)
    if tocopy isa ConcreteRArray
        return MakeConcreteRArray{eltype(tocopy),ndims(tocopy)}(size(tocopy))
    end
    if tocopy isa Array
        return MakeArray{Core.Typeof(tocopy),Tuple{map(make_valable, tocopy)...}}()
    end
    if tocopy isa Symbol
        return tocopy
    end
    if tocopy isa Int || tocopy isa AbstractFloat || tocopy isa Nothing || tocopy isa Type
        return MakeVal{Val{tocopy}}()
    end
    if tocopy isa AbstractString
        return MakeString{Core.Typeof(tocopy),Symbol(string)}() || T <: Nothing
    end
    T = Core.Typeof(tocopy)
    if tocopy isa Tuple || tocopy isa NamedTuple || isstructtype(T)
        elems = []
        nf = fieldcount(T)
        for i in 1:nf
            push!(elems, make_valable(getfield(tocopy, i)))
        end
        return MakeStruct{Core.Typeof(tocopy),Tuple{elems...}}()
    end

    return error("cannot copy $tocopy of type $(Core.Typeof(tocopy))")
end

function create_result(tocopy::MakeConcreteRArray{T,N}, path, result_stores) where {T,N}
    return :(ConcreteRArray{$T,$N}($(result_stores[path]), $(tocopy.shape)))
end

function create_result(tocopy::Tuple, path, result_stores)
    elems = Union{Symbol,Expr}[]
    for (k, v) in pairs(tocopy)
        push!(elems, create_result(v, (path..., k), result_stores))
    end
    return quote
        ($(elems...),)
    end
end

function create_result(tocopy::NamedTuple, path, result_stores)
    elems = Union{Symbol,Expr}[]
    for (k, v) in pairs(tocopy)
        push!(elems, create_result(v, (path..., k), result_Stores))
    end
    return quote
        NamedTuple{$(keys(tocopy))}($elems)
    end
end

function create_result(::MakeArray{AT,tocopy}, path, result_stores) where {AT,tocopy}
    elems = Expr[]
    for (i, v) in enumerate(tocopy.parameters)
        push!(elems, create_result(v, (path..., i), result_stores))
    end
    return quote
        $(eltype(AT))[$(elems...)]
    end
end

function create_result(::MakeVal{Val{nothing}}, path, result_stores)
    return :(nothing)
end

function create_result(::MakeVal{Val{elem}}, path, result_stores) where {elem}
    return :($elem)
end

function create_result(tocopy::Symbol, path, result_stores)
    return Meta.quot(tocopy)
end

function create_result(::MakeString{AT,Val}, path, result_stores) where {AT,Val}
    return :($(AT(Val)))
end

function create_result(::MakeStruct{AT,tocopy}, path, result_stores) where {AT,tocopy}
    # @info "create_result" AT tocopy path tocopy.parameters result_stores
    elems = Union{Symbol,Expr}[]
    for (i, v) in enumerate(tocopy.parameters)
        ev = create_result(v, (path..., i), result_stores)
        push!(elems, ev)
    end
    return Expr(:new, AT, elems...)
end

function compile(f, args; pipeline_options="", client=nothing)
    N = length(args)
    ctx = MLIR.IR.Context()
    Base.append!(registry[]; context=ctx)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
    MLIR.IR.context!(ctx) do
        mod = MLIR.IR.Module(MLIR.IR.Location())
        linear_args, linear_results, preserved_args, seen_args, concrete_result, isfwrapped = compile_to_module!(
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

        # return generate_jlfunc(
        #     concrete_result,
        #     client,
        #     mod,
        #     linear_args,
        #     linear_results,
        #     preserved_args,
        #     isfwrapped ? f : nothing,
        # )

        # @show linear_args linear_results
        # @show map(size, linear_results)
        # @show preserved_args seen_args
        # @show concrete_result typeof(concrete_result)

        linear_results_paths = (map(x -> x.paths, linear_results)...,)
        linear_args_paths = (map(x -> x.paths, linear_args)...,)
        preserved_args_paths = (map(x -> (x[1].paths, x[2]), preserved_args)...,)

        # @show linear_results_paths
        # @show linear_args_paths
        # @show preserved_args_paths

        exec = XLA.Compile(client, mod)
        fnwrap = isfwrapped ? f : nothing
        closure_ty = typeof(fnwrap)
        concrete_result_ty = make_valable(concrete_result)

        # Thunk code
        arg_syncs = Expr[]
        topres = Symbol[]
        linearized_args = Union{Symbol,Expr}[]

        for (i, argpaths) in enumerate(linear_args_paths)
            paths = ((p for p in argpaths if p[1] == :args)...,)
            path = if length(paths) == 1
                paths[1]
            else
                throw("Invalid path duplication $(argpaths) into $(paths)")
            end
            res = :(args[$(path[2])])
            for p in path[3:end]
                res = :(traced_getfield($res, $(Meta.quot(p))))
            end
            sym = Symbol("sbuf_$i")
            sbuf = :($sym = XLA.synced_buffer($res.data))
            push!(arg_syncs, sbuf)

            push!(topres, sym)

            res = :($sym.buffer)
            push!(linearized_args, res)
        end

        concretize = Expr[]
        for idx in 1:length(linear_results_paths)
            push!(
                concretize, :($(Symbol("concrete_res_$(idx)")) = linearized_results[$idx])
            )
        end

        delinearized_results = Expr[]

        result_stores = Dict{Tuple,Symbol}()

        for (idx, result_paths) in enumerate(linear_results_paths)
            paths = ((p for p in result_paths if p[1] != :args)...,)
            for path in paths
                if path[1] == :result
                    res = Symbol("result")
                    path = path[2:end]
                    result_stores[path] = Symbol("concrete_res_$(idx)")
                    continue
                else
                    if path[1] != :resargs
                        @show idx #, result
                        @show paths
                        @show path
                    end
                    @assert path[1] == :resargs
                    res = :(args[$(path[2])])
                    path = path[3:end]
                end
                for p in path
                    res = :(traced_getfield($res, $(Meta.quot(p))))
                end
                res = :($res.data = $(Symbol("concrete_res_$(idx)")))
                push!(delinearized_results, res)
            end
        end

        for (result_paths, arg_idx) in preserved_args_paths
            for path in result_paths
                argpaths = linear_args_paths[arg_idx + 1]
                argpath = only((p for p in argpaths if p[1] == :args))

                if path[1] == :result
                    res = Symbol("result")
                    path = path[2:end]
                else
                    @assert path[1] == :resargs || path[1] == :args
                    # We can optimize cases where we set the arg to itself
                    if path[2:end] == argpath[2:end]
                        continue
                    end
                    @show path, argpath
                    res = :(args[path[2]])
                    path = path[3:end]
                end
                for p in path
                    res = :(traced_getfield($res, $(Meta.quot(p))))
                end

                argres = :(args[argpath[2]])
                for p in argpath[3:end]
                    argres = :(traced_getfield($argres, $(Meta.quot(p))))
                end

                res = :($res.data = $argres.data)
                push!(delinearized_results, res)
            end
        end

        donated_args_set = zeros(UInt8, length(linearized_args))
        preserved_argnums = [i for (_, i) in preserved_args_paths]
        for i in 1:length(linear_args_paths)
            if !in(i, preserved_argnums)
                donated_args_set[i] = 1
            end
        end
        donated_args_set = (donated_args_set...,)

        exec_call = if length(linear_results_paths) == 0
            :()
        else
            quote
                $(arg_syncs...)
                GC.@preserve $(topres...) begin
                    linearized_results = XLA.ExecutableCall(
                        $exec, # thunk.exec,
                        ($(linearized_args...),),
                        $donated_args_set,
                        Val($(length(linear_results_paths))),
                    )
                end
            end
        end

        resexpr = create_result(concrete_result_ty, (), result_stores)

        fname = gensym(Symbol(Symbol(f), :_reactant))

        expr = quote
            function $fname(args...)
                Base.@_inline_meta
                $(
                    # if `f` is a closure, then prepend the closure into `args`
                    # the closure fields will be correctly extracted from it as the tracer has already passed through it
                    if !(closure_ty <: Nothing)
                        :(args = ($fnwrap, $args...))
                    end
                )
                $exec_call
                $(concretize...)
                # Needs to store into result
                result = $resexpr
                $(delinearized_results...)
                return result
            end
        end

        return eval(expr)
    end
end

# function generate_jlfunc(
#     concrete_result,
#     client,
#     mod,
#     linear_args,
#     linear_results,
#     preserved_args,
#     fnwrap::closure_ty,
# ) where {closure_ty}
#     linear_results_paths = (map(x -> x.paths, linear_results)...,)
#     linear_args_paths = (map(x -> x.paths, linear_args)...,)
#     preserved_args_paths = (map(x -> (x[1].paths, x[2]), preserved_args)...,)
#     exec = XLA.Compile(client, mod)
#     v = make_valable(concrete_result)

#     @info "generate_jlfunc" concrete_result v
#     return Thunk{
#         Val{linear_results_paths},
#         Val{linear_args_paths},
#         Val{preserved_args_paths},
#         v,
#         closure_ty,
#         ndims(concrete_result),
#     }(
#         exec, fnwrap, size(concrete_result)
#     )
# end
