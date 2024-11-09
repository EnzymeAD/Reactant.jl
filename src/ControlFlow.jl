function ReactantCore.traced_if(
    cond::TracedRNumber{Bool}, true_fn::TFn, false_fn::FFn, args
) where {TFn,FFn}
    true_region = MLIR.IR.Region()
    fnbody = MLIR.IR.Block()
    true_ret_values = MLIR.IR.block!(fnbody) do
        res = true_fn(args...)
        ret_values = map(res) do x
            hasfield(typeof(x), :mlir_data) && return x
            x isa Number && return promote_to(TracedRNumber{typeof(x)}, x)
            error("Unknown type $(typeof(x))")
        end
        MLIR.Dialects.stablehlo.return_([x.mlir_data for x in ret_values])
        return ret_values
    end
    push!(true_region, fnbody)

    false_region = MLIR.IR.Region()
    fnbody = MLIR.IR.Block()
    false_ret_values = MLIR.IR.block!(fnbody) do
        res = false_fn(args...)
        ret_values = map(res) do x
            hasfield(typeof(x), :mlir_data) && return x
            x isa Number && return promote_to(TracedRNumber{typeof(x)}, x)
            error("Unknown type $(typeof(x))")
        end
        MLIR.Dialects.stablehlo.return_([x.mlir_data for x in ret_values])
        return ret_values
    end
    push!(false_region, fnbody)

    @assert length(true_ret_values) == length(false_ret_values) "true branch returned $(length(true_ret_values)) results, false branch returned $(length(false_ret_values)). This shouldn't happen."

    result_types = MLIR.IR.Type[]
    true_block_insertions = []
    false_block_insertions = []
    for (i, (tr, fr)) in enumerate(zip(true_branch_results, false_branch_results))
        if typeof(tr) != typeof(fr)
            if !(tr isa MissingTracedValue) && !(fr isa MissingTracedValue)
                error("Result #$(i) for the branches have different types: true branch \
                       returned `$(typeof(tr))`, false branch returned `$(typeof(fr))`.")
            elseif tr isa MissingTracedValue
                push!(result_types, MLIR.IR.type(fr.mlir_data))
                push!(linear_results, new_traced_value(false_linear_results[i]))
                push!(true_block_insertions, (i => linear_results[end]))
            else
                push!(result_types, MLIR.IR.type(tr.mlir_data))
                push!(linear_results, new_traced_value(true_linear_results[i]))
                push!(false_block_insertions, (i => linear_results[end]))
            end
        else
            push!(result_types, MLIR.IR.type(tr.mlir_data))
            push!(linear_results, new_traced_value(tr))
        end
    end


    # if_compiled = MLIR.Dialects.stablehlo.if_(
    #     cond.mlir_data;
    #     true_branch=true_region,
    #     false_branch=false_region,
    #     result_0=result_types,
    # )
    # @show if_compiled

    error(1)


    (_, true_branch_compiled, true_branch_results, _, _, _, _, _, true_linear_results) = Reactant.make_mlir_fn(
        true_fn,
        args,
        (),
        string(gensym("true_branch")),
        false;
        return_dialect=:stablehlo,
        no_args_in_result=true,
        construct_function_without_args=true,
    )

    @show true_branch_compiled

    (_, false_branch_compiled, false_branch_results, _, _, _, _, _, false_linear_results) = Reactant.make_mlir_fn(
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
                push!(linear_results, new_traced_value(false_linear_results[i]))
                push!(true_block_insertions, (i => linear_results[end]))
            else
                push!(result_types, MLIR.IR.type(tr.mlir_data))
                push!(linear_results, new_traced_value(true_linear_results[i]))
                push!(false_block_insertions, (i => linear_results[end]))
            end
        else
            push!(result_types, MLIR.IR.type(tr.mlir_data))
            push!(linear_results, new_traced_value(tr))
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

function get_region_removing_missing_values(compiled_fn, insertions)
    region = MLIR.IR.Region()
    MLIR.API.mlirRegionTakeBody(region, MLIR.API.mlirOperationGetRegion(compiled_fn, 0))
    block = MLIR.IR.Block(MLIR.API.mlirRegionGetFirstBlock(region), false)
    return_op = MLIR.IR.terminator(block)
    for (i, rt) in insertions
        if rt isa TracedRNumber
            attr = MLIR.IR.DenseElementsAttribute(Array{eltype(rt)}(undef, ()))
            op = MLIR.Dialects.stablehlo.constant(; value=attr)
        elseif rt isa TracedRArray
            attr = MLIR.IR.DenseElementsAttribute(Array{eltype(rt)}(undef, size(rt)))
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
