function ReactantCore.traced_if(cond::TracedRNumber{Bool}, true_fn, false_fn, args)
    # NOTE: This behavior is different from how we compile other functions, i.e., we keep
    #       things as constants if possible, but from a block we do need to return a
    #       traced value, so we force a conversion to a TracedType.
    # XXX: Eventually we would want to support nested structures as block arguments
    #      but we will have to do a flatten/unflatten pass to make this work.
    args_traced = map(args) do arg
        arg isa TracedType && return arg
        arg isa Number && return promote_to(TracedRNumber{eltype(arg)}, arg)
        arg isa AbstractArray &&
            return promote_to(TracedRArray{eltype(arg),ndims(arg)}, arg)
        @warn "Argument $(arg) is not a TracedType, TracedRNumber, or TracedRArray. It \
               will be promoted to a TracedRNumber. Please open an issue in Reactant.jl \
               with an example of this behavior."
        return arg
    end

    true_block = MLIR.IR.Block()
    true_res = MLIR.IR.block!(true_block) do
        results = map(true_fn(args_traced...)) do r
            r isa TracedType && return r
            r isa Number && return promote_to(TracedRNumber{typeof(r)}, r)
            r isa AbstractArray &&
                return promote_to(TracedRArray{eltype(r),ndims(r)}, r)
            error("Unsupported return type $(typeof(r))")
        end
        MLIR.Dialects.stablehlo.return_([x.mlir_data for x in results])
        return results
    end

    false_block = MLIR.IR.Block()
    false_res = MLIR.IR.block!(false_block) do
        results = map(false_fn(args_traced...)) do r
            r isa TracedType && return r
            r isa Number && return promote_to(TracedRNumber{typeof(r)}, r)
            r isa AbstractArray &&
                return promote_to(TracedRArray{eltype(r),ndims(r)}, r)
            error("Unsupported return type $(typeof(r))")
        end
        MLIR.Dialects.stablehlo.return_([x.mlir_data for x in results])
        return results
    end

    @assert length(true_res) == length(false_res) "true branch returned $(length(true_res)) results, false branch returned $(length(false_res)). This shouldn't happen."

    result_types = MLIR.IR.Type[]
    true_block_insertions = []
    false_block_insertions = []
    for (i, (tr, fr)) in enumerate(zip(true_res, false_res))
        if typeof(tr) != typeof(fr)
            if !(tr isa MissingTracedValue) && !(fr isa MissingTracedValue)
                error("Result #$(i) for the branches have different types: true branch \
                       returned `$(typeof(tr))`, false branch returned `$(typeof(fr))`.")
            elseif tr isa MissingTracedValue
                push!(result_types, MLIR.IR.type(fr.mlir_data))
                push!(true_block_insertions, (i => new_traced_value(false_res[i])))
            else
                push!(result_types, MLIR.IR.type(tr.mlir_data))
                push!(false_block_insertions, (i => new_traced_value(true_res[i])))
            end
        else
            push!(result_types, MLIR.IR.type(tr.mlir_data))
        end
    end

    true_branch_region = get_region_removing_missing_values(
        true_block, true_block_insertions
    )
    false_branch_region = get_region_removing_missing_values(
        false_block, false_block_insertions
    )

    if_compiled = MLIR.Dialects.stablehlo.if_(
        cond.mlir_data;
        true_branch=true_branch_region,
        false_branch=false_branch_region,
        result_0=result_types,
    )

    return map(1:MLIR.IR.nresults(if_compiled)) do i
        res = MLIR.IR.result(if_compiled, i)
        sz = size(MLIR.IR.type(res))
        T = MLIR.IR.julia_type(eltype(MLIR.IR.type(res)))
        isempty(sz) && return TracedRNumber{T}((), res)
        return TracedRArray{T,length(sz)}((), res, sz)
    end
end

function get_region_removing_missing_values(block, insertions)
    region = MLIR.IR.Region()
    push!(region, block)
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
