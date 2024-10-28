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

function trace_if(mod, expr)
    @assert expr.head == :if
    @assert length(expr.args) == 3 "`@trace` expects an `else` block for `if` blocks."
    # XXX: support `elseif` blocks
    @assert expr.args[3].head == :block "`elseif` blocks are not supported yet."

    true_branch_symbols = ExpressionExplorer.compute_symbols_state(expr.args[2])
    true_branch_input_list = [true_branch_symbols.references...]
    true_branch_assignments = [true_branch_symbols.assignments...]
    true_branch_fn_name = gensym(:true_branch)

    false_branch_symbols = ExpressionExplorer.compute_symbols_state(expr.args[3])
    false_branch_input_list = [false_branch_symbols.references...]
    false_branch_assignments = [false_branch_symbols.assignments...]
    false_branch_fn_name = gensym(:false_branch)

    all_input_vars = true_branch_input_list ∪ false_branch_input_list
    all_output_vars = true_branch_assignments ∪ false_branch_assignments

    all_vars = all_input_vars ∪ all_output_vars

    true_branch_fn = quote
        $(true_branch_fn_name) =
            ($(all_input_vars...),) -> begin
                $(Expr(:meta, :inline))
                $(expr.args[2])
                return ($(all_output_vars...),)
            end
    end
    true_branch_fn = cleanup_expr_to_avoid_boxing(
        true_branch_fn, true_branch_fn_name, all_vars
    )

    false_branch_fn = quote
        $(false_branch_fn_name) =
            ($(all_input_vars...),) -> begin
                $(Expr(:meta, :inline))
                $(expr.args[3])
                return ($(all_output_vars...),)
            end
    end
    false_branch_fn = cleanup_expr_to_avoid_boxing(
        false_branch_fn, false_branch_fn_name, all_vars
    )

    return quote
        if any($(is_traced), ($(all_input_vars...),))
            $(true_branch_fn)
            $(false_branch_fn)
            ($(all_output_vars...),) = $(traced_if)(
                $(expr.args[1]),
                $(true_branch_fn_name),
                $(false_branch_fn_name),
                ($(all_input_vars...),),
            )
        else
            $(expr)
        end
    end
end

is_traced(x) = false
is_traced(::TracedRArray) = true
is_traced(::TracedRNumber) = true

makelet(x, prepend::Symbol) = :($(Symbol(prepend, x)) = $(x))

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

    for (i, (tr, fr)) in enumerate(zip(true_branch_results, false_branch_results))
        @assert typeof(tr) == typeof(fr) "Result #$(i) for the branches have different \
                                          types: true branch returned `$(typeof(tr))`, \
                                          false branch returned `$(typeof(fr))`."
    end

    results = [MLIR.IR.type(tr.mlir_data) for tr in true_linear_results]

    true_branch_region = let reg = MLIR.IR.Region()
        MLIR.API.mlirRegionTakeBody(
            reg, MLIR.API.mlirOperationGetRegion(true_branch_compiled, 0)
        )
        reg
    end

    false_branch_region = let reg = MLIR.IR.Region()
        MLIR.API.mlirRegionTakeBody(
            reg, MLIR.API.mlirOperationGetRegion(false_branch_compiled, 0)
        )
        reg
    end

    if_compiled = MLIR.Dialects.stablehlo.if_(
        cond.mlir_data;
        true_branch=true_branch_region,
        false_branch=false_branch_region,
        result_0=results,
    )

    return map(enumerate(true_linear_results)) do (i, res)
        res = copy(res)
        res.mlir_data = MLIR.IR.result(if_compiled, i)
        return res
    end
end

export @trace

end
