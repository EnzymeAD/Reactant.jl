module ControlFlow

using ..Reactant: Reactant, TracedRNumber, TracedRArray
using ..MLIR: MLIR

using MacroTools: MacroTools

macro trace(expr)
    expr.head == :if && return esc(trace_if(__module__, expr))
    return error("Only `if-elseif-else` blocks are currently supported by `@trace`")
end

function trace_if(mod, expr)
    @assert expr.head == :if
    @assert length(expr.args) == 3 "`@trace` expects an `else` block for `if` blocks."
    # XXX: support `elseif` blocks
    @assert expr.args[3].head == :block "`elseif` blocks are not supported yet."

    # `var_list` is a list of input variables that are used in the `if` block
    # `bound_vars` is a list of variables that are bound in the `if` block
    true_branch_var_list = Symbol[]
    true_branch_bound_vars = Symbol[]
    find_var_uses!(true_branch_var_list, true_branch_bound_vars, expr.args[2])
    true_branch_fn_name = gensym(:true_branch)

    false_branch_var_list = Symbol[]
    false_branch_bound_vars = Symbol[]
    find_var_uses!(false_branch_var_list, false_branch_bound_vars, expr.args[3])
    false_branch_fn_name = gensym(:false_branch)

    all_input_vars = true_branch_var_list ∪ false_branch_var_list
    all_output_vars = true_branch_bound_vars ∪ false_branch_bound_vars

    true_branch_fn = quote
        $(true_branch_fn_name) =
            ($(all_input_vars...),) -> begin
                $(Expr(:meta, :inline))
                $(expr.args[2])
                return ($(all_output_vars...),)
            end
    end

    true_branch_fn = MacroTools.prewalk(true_branch_fn) do x
        if x isa Symbol && x ∈ all_output_vars
            return Symbol(:true_branch₋, x)
        end
        return x
    end

    false_branch_fn = quote
        $(false_branch_fn_name) =
            ($(all_input_vars...),) -> begin
                $(Expr(:meta, :inline))
                $(expr.args[3])
                return ($(all_output_vars...),)
            end
    end

    false_branch_fn = MacroTools.prewalk(false_branch_fn) do x
        if x isa Symbol && x ∈ all_output_vars
            return Symbol(:false_branch₋, x)
        end
        return x
    end

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

makelet(x) = :($(x) = $(x))

# Generate this dummy function and later we remove it during tracing
function traced_if(cond, true_fn::TFn, false_fn::FFn, args) where {TFn,FFn}
    if cond
        return true_fn(args...)
    else
        return false_fn(args...)
    end
end

function traced_if(
    cond::TracedRNumber{Bool}, true_fn::TFn, false_fn::FFn, args
) where {TFn,FFn}
    _, true_branch_compiled, true_branch_results, _, _, _, _, _, true_linear_results = Reactant.make_mlir_fn(
        true_fn, args, (), string(gensym("true_branch")), false
    )

    _, false_branch_compiled, false_branch_results, _, _, _, _, _, false_linear_results = Reactant.make_mlir_fn(
        false_fn, args, (), string(gensym("false_branch")), false
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

    @show if_compiled

    return error("WIP")
end

# XXX: Use `ExpressionExplorer.jl` instead
# NOTE: Adapted from https://github.com/c42f/FastClosures.jl/blob/master/src/FastClosures.jl
function find_var_uses!(varlist, bound_vars, ex)
    if isa(ex, Symbol)
        var = ex
        if !(var in bound_vars)
            var ∈ varlist || push!(varlist, var)
        end
        return varlist
    elseif isa(ex, Expr)
        if ex.head == :quote || ex.head == :line || ex.head == :inbounds
            return varlist
        end
        if ex.head == :(=)
            find_var_uses_lhs!(varlist, bound_vars, ex.args[1])
            find_var_uses!(varlist, bound_vars, ex.args[2])
        elseif ex.head == :kw
            find_var_uses!(varlist, bound_vars, ex.args[2])
        elseif ex.head == :for ||
            ex.head == :while ||
            ex.head == :comprehension ||
            ex.head == :let
            # New scopes
            inner_bindings = copy(bound_vars)
            find_var_uses!(varlist, inner_bindings, ex.args)
        elseif ex.head == :try
            # New scope + ex.args[2] is a new binding
            find_var_uses!(varlist, copy(bound_vars), ex.args[1])
            catch_bindings = copy(bound_vars)
            !isa(ex.args[2], Symbol) || push!(catch_bindings, ex.args[2])
            find_var_uses!(varlist, catch_bindings, ex.args[3])
            if length(ex.args) > 3
                finally_bindings = copy(bound_vars)
                find_var_uses!(varlist, finally_bindings, ex.args[4])
            end
        elseif ex.head == :call
            find_var_uses!(varlist, bound_vars, ex.args[2:end])
        elseif ex.head == :local
            foreach(ex.args) do e
                if !isa(e, Symbol)
                    find_var_uses!(varlist, bound_vars, e)
                end
            end
        elseif ex.head == :(::)
            find_var_uses_lhs!(varlist, bound_vars, ex)
        else
            find_var_uses!(varlist, bound_vars, ex.args)
        end
    end
    return varlist
end

function find_var_uses!(varlist, bound_vars, exs::Vector)
    return foreach(e -> find_var_uses!(varlist, bound_vars, e), exs)
end

# Find variable uses on the left hand side of an assignment.  Some of what may
# be variable uses turn into bindings in this context (cf. tuple unpacking).
function find_var_uses_lhs!(varlist, bound_vars, ex)
    if isa(ex, Symbol)
        var = ex
        var ∈ bound_vars || push!(bound_vars, var)
    elseif isa(ex, Expr)
        if ex.head == :tuple
            find_var_uses_lhs!(varlist, bound_vars, ex.args)
        elseif ex.head == :(::)
            find_var_uses!(varlist, bound_vars, ex.args[2])
            find_var_uses_lhs!(varlist, bound_vars, ex.args[1])
        else
            find_var_uses!(varlist, bound_vars, ex.args)
        end
    end
end

function find_var_uses_lhs!(varlist, bound_vars, exs::Vector)
    return foreach(e -> find_var_uses_lhs!(varlist, bound_vars, e), exs)
end

export @trace

end
