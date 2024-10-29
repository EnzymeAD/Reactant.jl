module ReactantCore

using ExpressionExplorer: ExpressionExplorer
using MacroTools: MacroTools

export @trace, MissingTracedValue

# Traits
is_traced(x) = false

# New Type signifying that a value is missing
mutable struct MissingTracedValue
    paths::Tuple
end

MissingTracedValue() = MissingTracedValue(())

# Code generation
macro trace(expr)
    expr.head == :if && return esc(trace_if(__module__, expr))
    return error("Only `if-elseif-else` blocks are currently supported by `@trace`")
end

function trace_if(mod, expr)
    expr.head == :if && error_if_return(expr)

    condition_vars = [ExpressionExplorer.compute_symbols_state(expr.args[1]).references...]

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
        tmp_expr = []
        for var in true_branch_assignments
            push!(tmp_expr, :($(var) = $(var)))
        end
        Expr(:block, tmp_expr...), nothing
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

    all_check_vars = [all_input_vars..., condition_vars...]
    return quote
        if any($(is_traced), ($(all_check_vars...),))
            $(reactant_code_block)
        else
            $(expr)
        end
    end
end

# Generate this dummy function and later we remove it during tracing
function traced_if(cond, true_fn::TFn, false_fn::FFn, args) where {TFn,FFn}
    return cond ? true_fn(args) : false_fn(args)
end

function cleanup_expr_to_avoid_boxing(expr, prepend::Symbol, all_vars)
    return MacroTools.postwalk(expr) do x
        if x isa Symbol && x ∈ all_vars
            return Symbol(prepend, x)
        end
        return x
    end
end

function error_if_return(expr)
    return MacroTools.postwalk(expr) do x
        if x isa Expr && x.head == :return
            error("Cannot use @trace on a block that contains a return statement")
        end
        return x
    end
end

end
