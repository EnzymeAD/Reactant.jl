module ControlFlow

using ..Reactant: Reactant, TracedRNumber, TracedRArray
using ..MLIR: MLIR

using ExpressionExplorer: ExpressionExplorer
using MacroTools: MacroTools

macro trace(expr)
    expr.head == :if && return esc(trace_if(__module__, expr))
    return error("Only `if-elseif-else` blocks are currently supported by `@trace`")
end

function cleanup_expr_to_avoid_boxing(expr, prepend::Symbol, all_vars)
    return MacroTools.postwalk(expr) do x
        if x isa Symbol && x ∈ all_vars
            return Symbol(prepend, x)
        end
        return x
    end
end

mutable struct MissingTracedValue
    paths
end

MissingTracedValue() = MissingTracedValue(())

function trace_if(mod, expr)
    true_branch_symbols = ExpressionExplorer.compute_symbols_state(expr.args[2])
    true_branch_input_list = [true_branch_symbols.references...]
    true_branch_assignments = [true_branch_symbols.assignments...]
    all_true_branch_vars = true_branch_input_list ∪ true_branch_assignments
    true_branch_fn_name = gensym(:true_branch)

    else_block, discard_vars = if length(expr.args) == 3
        if expr.args[3].head != :elseif
            expr.args[3], nothing
        else
            trace_if(mod, expr.args[3])
        end
    elseif length(expr.args) == 2
        :(), nothing
    else
        dump(expr)
        error("This shouldn't happen")
    end

    false_branch_symbols = ExpressionExplorer.compute_symbols_state(else_block)
    false_branch_input_list = [false_branch_symbols.references...]
    false_branch_assignments = [false_branch_symbols.assignments...]
    all_false_branch_vars = false_branch_input_list ∪ false_branch_assignments
    false_branch_fn_name = gensym(:false_branch)

    all_input_vars = true_branch_input_list ∪ false_branch_input_list
    all_output_vars = true_branch_assignments ∪ false_branch_assignments
    discard_vars !== nothing && setdiff!(all_output_vars, discard_vars)

    all_vars = all_input_vars ∪ all_output_vars

    non_existant_true_branch_vars = setdiff(all_output_vars, all_true_branch_vars)
    true_branch_extras = Expr(
        :block,
        [:($(var) = $(MissingTracedValue())) for var in non_existant_true_branch_vars]...,
    )

    true_branch_fn = quote
        $(true_branch_fn_name) =
            ($(all_input_vars...),) -> begin
                $(expr.args[2])
                $(true_branch_extras)
                return ($(all_output_vars...),)
            end
    end
    true_branch_fn = cleanup_expr_to_avoid_boxing(
        true_branch_fn, true_branch_fn_name, all_vars
    )

    non_existant_false_branch_vars = setdiff(all_output_vars, all_false_branch_vars)
    false_branch_extras = Expr(
        :block,
        [:($(var) = $(MissingTracedValue())) for var in non_existant_false_branch_vars]...,
    )

    false_branch_fn = quote
        $(false_branch_fn_name) =
            ($(all_input_vars...),) -> begin
                $(else_block)
                $(false_branch_extras)
                return ($(all_output_vars...),)
            end
    end
    false_branch_fn = cleanup_expr_to_avoid_boxing(
        false_branch_fn, false_branch_fn_name, all_vars
    )

    reactant_code_block = quote
        $(true_branch_fn)
        $(false_branch_fn)
        ($(all_output_vars...),) = $(traced_if)(
            $(expr.args[1]),
            $(true_branch_fn_name),
            $(false_branch_fn_name),
            ($(all_input_vars...),),
        )
    end

    expr.head != :if &&
        return reactant_code_block, (true_branch_fn_name, false_branch_fn_name)

    return quote
        if any($(is_traced), ($(all_input_vars...),))
            $(reactant_code_block)
        else
            $(expr)
        end
    end
end

is_traced(x) = false
is_traced(::TracedRArray) = true
is_traced(::TracedRNumber) = true

makelet(x, prepend::Symbol) = :($(Symbol(prepend, x)) = $(x))

new_traced_value(::TracedRNumber{T}) where {T} = TracedRNumber{T}((), nothing)
new_traced_value(res::TracedRArray) = similar(res)

# Generate this dummy function and later we remove it during tracing
function traced_if(cond, true_fn::TFn, false_fn::FFn, args) where {TFn,FFn}
    return cond ? true_fn(args) : false_fn(args)
end

function traced_if(
    cond::TracedRNumber{Bool}, true_fn::TFn, false_fn::FFn, args
) where {TFn,FFn}
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
            attr = MLIR.IR.DenseElementsAttribute(zeros(eltype(rt)))
            op = MLIR.Dialects.stablehlo.constant(; value=attr)
        elseif rt isa TracedRArray
            attr = MLIR.IR.DenseElementsAttribute(zeros(eltype(rt), size(rt)))
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

export @trace

end
