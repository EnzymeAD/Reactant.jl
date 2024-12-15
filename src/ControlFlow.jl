function ReactantCore.traced_if(
    cond::TracedRNumber{Bool}, true_fn::TFn, false_fn::FFn, args
) where {TFn,FFn}
    (_, true_branch_compiled, true_branch_results, _, _, _, _, _, true_linear_results) = Reactant.TracedUtils.make_mlir_fn(
        true_fn,
        args,
        (),
        string(gensym("true_branch")),
        false;
        return_dialect=:stablehlo,
        no_args_in_result=true,
        construct_function_without_args=true,
    )

    (_, false_branch_compiled, false_branch_results, _, _, _, _, _, false_linear_results) = Reactant.TracedUtils.make_mlir_fn(
        false_fn,
        args,
        (),
        string(gensym("false_branch")),
        false;
        return_dialect=:stablehlo,
        no_args_in_result=true,
        construct_function_without_args=true,
    )

    @assert length(true_branch_results) == length(false_branch_results) "true branch returned $(length(true_branch_results)) results, false branch returned $(length(false_branch_results)). This shouldn't happen."

    result_types = MLIR.IR.Type[]
    linear_results = []
    true_block_insertions = []
    false_block_insertions = []
    for (i, (tr, fr)) in enumerate(zip(true_branch_results, false_branch_results))
        if typeof(tr) != typeof(fr)
            if !(tr isa MissingTracedValue) && !(fr isa MissingTracedValue)
                error("Result #$(i) for the branches have different types: true branch \
                       returned `$(typeof(tr))`, false branch returned `$(typeof(fr))`.")
            elseif tr isa MissingTracedValue
                push!(result_types, MLIR.IR.type(fr.mlir_data))
                push!(linear_results, TracedUtils.new_traced_value(false_linear_results[i]))
                push!(true_block_insertions, (i => linear_results[end]))
            else
                push!(result_types, MLIR.IR.type(tr.mlir_data))
                push!(linear_results, TracedUtils.new_traced_value(true_linear_results[i]))
                push!(false_block_insertions, (i => linear_results[end]))
            end
        else
            push!(result_types, MLIR.IR.type(tr.mlir_data))
            push!(linear_results, TracedUtils.new_traced_value(tr))
        end
    end

    # Replace all uses of missing values with the correct values
    true_branch_region = get_region_removing_missing_values(
        true_branch_compiled, true_block_insertions
    )

    false_branch_region = get_region_removing_missing_values(
        false_branch_compiled, false_block_insertions
    )

    MLIR.IR.rmfromparent!(true_branch_compiled)
    MLIR.IR.rmfromparent!(false_branch_compiled)

    if_compiled = MLIR.Dialects.stablehlo.if_(
        cond.mlir_data;
        true_branch=true_branch_region,
        false_branch=false_branch_region,
        result_0=result_types,
    )

    return map(enumerate(linear_results)) do (i, res)
        res.mlir_data = MLIR.IR.result(if_compiled, i)
        return res
    end
end

function ReactantCore.traced_while(
    cond_fn::CFn, body_fn::BFn, args
) where {CFn<:Function,BFn<:Function}
    # TODO: detect and prevent mutation within the condition

    # We promote all incoming args (is there a better way to do this?)
    traced_args = [
        if v isa Number && !(v isa TracedType)
            Reactant.TracedUtils.promote_to(TracedRNumber{typeof(v)}, v)
        else
            v
        end for v in args
    ]

    (_, cond_fn_compiled, cond_fn_results, _, _, _, _, in_tys, cond_fn_linear_results) = Reactant.TracedUtils.make_mlir_fn(
        cond_fn,
        traced_args,
        (),
        string(gensym("cond_fn")),
        false;
        no_args_in_result=true,
        return_dialect=:stablehlo,
        do_transpose=false,
    )

    (_, body_fn_compiled, body_fn_results, _, _, _, _, _, body_fn_linear_results) = Reactant.TracedUtils.make_mlir_fn(
        body_fn,
        traced_args,
        (),
        string(gensym("body_fn")),
        false;
        no_args_in_result=true,
        return_dialect=:stablehlo,
        do_transpose=false,
    )

    cond_reg = take_region(cond_fn_compiled)
    body_reg = take_region(body_fn_compiled)

    MLIR.IR.rmfromparent!(cond_fn_compiled)
    MLIR.IR.rmfromparent!(body_fn_compiled)

    result_0 = in_tys

    operands = MLIR.IR.Value[v.mlir_data for v in traced_args]

    while_compiled = MLIR.Dialects.stablehlo.while_(
        operands; result_0, cond=cond_reg, body=body_reg
    )

    return map(enumerate(traced_args)) do (i, res)
        res.mlir_data = MLIR.IR.result(while_compiled, i)
        return res
    end
end

function ReactantCore.traced_call(f, args...)
    seen_cache = Reactant.OrderedIdDict()
    make_tracer(
        seen_cache,
        args,
        (), # we have to insert something here, but we remove it immediately below.
        TracedTrack;
        toscalar=false,
        track_numbers=(), # TODO: track_numbers?
    )
    linear_args = Reactant.MLIR.IR.Value[]
    for (k, v) in seen_cache
        v isa TracedType || continue
        push!(linear_args, v.mlir_data)
        # make tracer inserted `()` into the path, here we remove it:
        v.paths = v.paths[1:end-1]
    end

    cache_key = Cached((f, args...))
    if haskey(Reactant.Compiler.callcache[], cache_key)
        # cache lookup:
        (; f_name, mlir_result_types, traced_result) = Reactant.Compiler.callcache[][cache_key]
    else
        f_name = String(gensym(Symbol(f)))
        temp = Reactant.make_mlir_fn(
            f,
            args,
            (),
            f_name,
            false;
            no_args_in_result=true,
            do_transpose=false,
        )
        traced_result, ret = temp[[3, 6]]
        mlir_result_types = [MLIR.IR.type(MLIR.IR.operand(ret, i)) for i in 1:MLIR.IR.noperands(ret)]
        Reactant.Compiler.callcache[][cache_key] = (; f_name, mlir_result_types, traced_result)
    end

    call_op = MLIR.Dialects.func.call(
        linear_args;
        result_0=mlir_result_types,
        callee=MLIR.IR.FlatSymbolRefAttribute(f_name),
    )

    seen_results = Reactant.OrderedIdDict()
    traced_result = make_tracer(
        seen_results,
        traced_result,
        (), # we have to insert something here, but we remove it immediately below.
        TracedSetPath;
        toscalar=false,
        track_numbers=(),
    )
    i = 1
    for (k, v) in seen_results
        v isa TracedType || continue
        # this mutates `traced_result`, which is what we want:
        v.mlir_data = MLIR.IR.result(call_op, i)
        # make tracer inserted `()` into the path, here we remove it:
        v.paths = v.paths[1:end-1]
        i += 1
    end

    return traced_result
end

function take_region(compiled_fn)
    region = MLIR.IR.Region()
    MLIR.API.mlirRegionTakeBody(region, MLIR.API.mlirOperationGetRegion(compiled_fn, 0))
    return region
end

function get_region_removing_missing_values(compiled_fn, insertions)
    region = take_region(compiled_fn)
    block = MLIR.IR.Block(MLIR.API.mlirRegionGetFirstBlock(region), false)
    return_op = MLIR.IR.terminator(block)
    for (i, rt) in insertions
        if rt isa TracedRNumber
            attr = MLIR.IR.DenseElementsAttribute(Array{unwrapped_eltype(rt)}(undef, ()))
            op = MLIR.Dialects.stablehlo.constant(; value=attr)
        elseif rt isa TracedRArray
            attr = MLIR.IR.DenseElementsAttribute(
                Array{unwrapped_eltype(rt)}(undef, size(rt))
            )
            op = MLIR.Dialects.stablehlo.constant(; value=attr)
        else
            error("Unknown type $(typeof(rt))")
        end
        MLIR.IR.rmfromparent!(op)
        insert!(block, 1, op)
        val = MLIR.IR.result(op, 1)
        MLIR.API.mlirValueReplaceAllUsesOfWith(MLIR.IR.operand(return_op, i), val)
    end
    return region
end

struct Cached
    obj
end
Base.:(==)(a::Cached, b::Cached) = recursive_equal(a.obj, b.obj)
Base.hash(a::Cached, h::UInt) = recursive_hash(a.obj, h)

recursive_equal(a, b) = false
function recursive_equal(a::T, b::T) where {T}
    fn = fieldnames(T)
    isempty(fn) && return a == b
    for name in fn
        !recursive_equal(getfield(a, name), getfield(b, name)) && return false
    end
    return true
end
function recursive_equal(a::T, b::T) where {T<:AbstractArray}
    for (el_a, el_b) in zip(a, b)
        !recursive_equal(el_a, el_b) && return false
    end
    return true
end
recursive_equal(a::T, b::T) where {T<:TracedRArray} = MLIR.IR.type(a.mlir_data) == MLIR.IR.type(b.mlir_data)


function recursive_hash(a::T, h::UInt) where T
    fn = fieldnames(T)
    isempty(fn) && return hash(a, h)
    h = hash(T, h) # include type in the hash
    for name in fn
        h = recursive_hash(getfield(a, name), h)
    end
    return h
end
function recursive_hash(a::AbstractArray, h::UInt)
    for el in a
        h = recursive_hash(el, h)
    end
    return h
end
recursive_hash(a::TracedRArray, h::UInt) = hash(MLIR.IR.type(a.mlir_data), h)
