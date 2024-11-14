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

const SPECIAL_SYMBOLS = [
    :(:), :nothing, :missing, :Inf, :Inf16, :Inf32, :Inf64, :Base, :Core
]

# Code generation
"""
    @trace <expr>

Converts certain expressions like control flow into a Reactant friendly form. Importantly,
if no traced value is found inside the expression, then there is no overhead.

## Currently Supported

- `if` conditions (with `elseif` and other niceties) (`@trace if ...`)
- `if` statements with a preceeding assignment (`@trace a = if ...`) (note the positioning
  of the macro needs to be before the assignment and not before the `if`)
- `for` statements with a single induction variable iterating over a syntactic `StepRange` of integers.

## Special Considerations

- Apply `@trace` only at the outermost `if`. Nested `if` statements will be automatically
  expanded into the correct form.

# Extended Help

## Caveats (Deviations from Core Julia Semantics)

### New variables introduced

```julia
@trace if x > 0
    y = x + 1
    p = 1
else
    y = x - 1
end
```

In the outer scope `p` is not defined if `x ≤ 0`. However, for the traced version, it is
defined and set to a dummy value.

### Short Circuiting Operations

```julia
@trace if x > 0 && z > 0
    y = x + 1
else
    y = x - 1
end
```

`&&` and `||` are short circuiting operations. In the traced version, we replace them with
`&` and `|` respectively.

### Type-Unstable Branches

```julia
@trace if x > 0
    y = 1.0f0
else
    y = 1.0
end
```

This will not compile since `y` is a `Float32` in one branch and a `Float64` in the other.
You need to ensure that all branches have the same type.

Another example is the following for loop which changes the type of `x` between iterations.

```julia
x = ... # ConcreteRArray{Int64, 1}
for i in 1f0:0.5f0:10f0
    x = x .+ i # ConcreteRArray{Float32, 1}
end
```

### Certain Symbols are Reserved

Symbols like $(SPECIAL_SYMBOLS) are not allowed as variables in `@trace` expressions. While certain cases might work but these are not guaranteed to work. For
example, the following will not work:

```julia
function fn(x)
    nothing = sum(x)
    @trace if nothing > 0
        y = 1.0
    else
        y = 2.0
    end
    return y, nothing
end
```
"""
macro trace(expr)
    expr = macroexpand(__module__, expr)
    if Meta.isexpr(expr, :(=))
        if Meta.isexpr(expr.args[2], :if)
            return esc(trace_if_with_returns(__module__, expr))
        end
    end
    Meta.isexpr(expr, :if) && return esc(trace_if(__module__, expr))
    Meta.isexpr(expr, :for) && return (esc(trace_for(__module__, expr)))
    return error("Only `if-elseif-else` blocks are currently supported by `@trace`")
end

function trace_for(mod, expr)
    Meta.isexpr(expr, :for, 2) || error("expected for expr")
    assign, body = expr.args

    error_if_any_control_flow(body)
    if !Meta.isexpr(assign, :(=)) ||
        !(assign.args[1] isa Symbol) ||
        !Meta.isexpr(assign.args[2], :call) ||
        assign.args[2].args[1] !== :(:)
        error("malformed for loop assignment")
    end

    induction, range = assign.args

    counter = gensym(:i)
    num_iters = gensym(:num_iters)

    start = range.args[2]
    step = length(range.args) == 3 ? 1 : range.args[3]
    limit = range.args[end]

    body_symbols = ExpressionExplorer.compute_symbols_state(
        quote
            $(Expr(:local, assign))
            $body
        end,
    )

    external_syms = body_symbols.assignments ∪ body_symbols.references
    filter!(∉(SPECIAL_SYMBOLS), external_syms)

    all_syms = Expr(:tuple, counter, external_syms...)
    args_init = Expr(
        :tuple, :(Reactant.promote_to(Reactant.TracedRNumber{Int}, 0)), external_syms...
    )

    reactant_code_block = quote
        let args = $(args_init)
            cond_fn =
                $(all_syms) -> begin
                    local num_iters = div($limit - $start, $step, RoundDown)
                    local num_iters = Reactant.promote_to(
                        Reactant.TracedRNumber{Int64}, num_iters
                    )
                    $counter < num_iters + 1
                end
            body_fn =
                $(all_syms) -> begin
                    local step_ = $step
                    local start_ = $start
                    local $induction = start_ + $counter * step_
                    $body
                    ($counter + 1, $(all_syms.args[(begin + 1):end]...))
                end

            $(ReactantCore).traced_while(cond_fn, body_fn, args)
        end
    end

    return quote
        if any($(is_traced), $(Expr(:tuple, all_syms.args[(begin + 1):end]...)))
            $(reactant_code_block)
        else
            $(expr)
        end
    end
end

# ... = if ... style expressions
function trace_if_with_returns(mod, expr)
    new_expr, _, all_check_vars = trace_if(
        mod, expr.args[2]; store_last_line=expr.args[1], depth=1
    )
    return quote
        if any($(is_traced), ($(all_check_vars...),))
            $(new_expr)
        else
            $(expr)
        end
    end
end

function trace_if(mod, expr; store_last_line=nothing, depth=0)
    discard_vars_from_expansion = []
    original_expr = expr

    if depth == 0
        error_if_any_control_flow(expr)

        counter = 0
        expr = MacroTools.prewalk(expr) do x
            counter += 1
            if x isa Expr && x.head == :if && counter > 1
                ex_new, dv, _ = trace_if(mod, x; store_last_line, depth=depth + 1)
                append!(discard_vars_from_expansion, dv)
                return ex_new
            end
            return x
        end
    end

    cond_expr = remove_shortcircuiting(expr.args[1])
    condition_vars = [ExpressionExplorer.compute_symbols_state(cond_expr).references...]

    true_block = if store_last_line !== nothing
        @assert expr.args[2].head == :block "currently we only support blocks"
        true_last_line = expr.args[2].args[end]
        remaining_lines = expr.args[2].args[1:(end - 1)]
        quote
            $(remaining_lines...)
            $(store_last_line) = $(true_last_line)
        end
    else
        expr.args[2]
    end

    true_branch_symbols = ExpressionExplorer.compute_symbols_state(true_block)
    true_branch_input_list = [true_branch_symbols.references...]
    filter!(x -> x ∉ SPECIAL_SYMBOLS, true_branch_input_list)
    true_branch_assignments = [true_branch_symbols.assignments...]
    all_true_branch_vars = true_branch_input_list ∪ true_branch_assignments
    true_branch_fn_name = gensym(:true_branch)

    else_block, discard_vars, _ = if length(expr.args) == 3
        if expr.args[3].head != :elseif
            expr.args[3], [], nothing
        else
            trace_if(mod, expr.args[3]; store_last_line, depth=depth + 1)
        end
    elseif length(expr.args) == 2
        tmp_expr = []
        for var in true_branch_assignments
            push!(tmp_expr, :($(var) = $(var)))
        end
        Expr(:block, tmp_expr...), [], nothing
    else
        dump(expr)
        error("This shouldn't happen")
    end

    discard_vars = unique(discard_vars_from_expansion) ∪ discard_vars

    false_block = if store_last_line !== nothing
        @assert else_block.head == :block "currently we only support blocks"
        false_last_line = else_block.args[end]
        remaining_lines = else_block.args[1:(end - 1)]
        quote
            $(remaining_lines...)
            $(store_last_line) = $(false_last_line)
        end
    else
        else_block
    end

    false_branch_symbols = ExpressionExplorer.compute_symbols_state(false_block)
    false_branch_input_list = [false_branch_symbols.references...]
    filter!(x -> x ∉ SPECIAL_SYMBOLS, false_branch_input_list)
    false_branch_assignments = [false_branch_symbols.assignments...]
    all_false_branch_vars = false_branch_input_list ∪ false_branch_assignments
    false_branch_fn_name = gensym(:false_branch)

    all_input_vars = true_branch_input_list ∪ false_branch_input_list
    all_output_vars = all_true_branch_vars ∪ all_false_branch_vars
    discard_vars !== nothing && setdiff!(all_output_vars, discard_vars)

    all_vars = all_input_vars ∪ all_output_vars

    non_existant_true_branch_vars = setdiff(all_output_vars, all_true_branch_vars)
    true_branch_extras = Expr(
        :block,
        [:($(var) = $(MissingTracedValue())) for var in non_existant_true_branch_vars]...,
    )

    true_branch_fn = :(($(all_input_vars...),) -> begin
        $(true_block)
        $(true_branch_extras)
        return ($(all_output_vars...),)
    end)
    true_branch_fn = cleanup_expr_to_avoid_boxing(
        true_branch_fn, true_branch_fn_name, all_vars
    )
    true_branch_fn = :($(true_branch_fn_name) = $(true_branch_fn))

    non_existant_false_branch_vars = setdiff(
        setdiff(all_output_vars, all_false_branch_vars), all_input_vars
    )
    false_branch_extras = Expr(
        :block,
        [:($(var) = $(MissingTracedValue())) for var in non_existant_false_branch_vars]...,
    )

    false_branch_fn = :(($(all_input_vars...),) -> begin
        $(false_block)
        $(false_branch_extras)
        return ($(all_output_vars...),)
    end)
    false_branch_fn = cleanup_expr_to_avoid_boxing(
        false_branch_fn, false_branch_fn_name, all_vars
    )
    false_branch_fn = :($(false_branch_fn_name) = $(false_branch_fn))

    reactant_code_block = quote
        $(true_branch_fn)
        $(false_branch_fn)
        ($(all_output_vars...),) = $(traced_if)(
            $(cond_expr),
            $(true_branch_fn_name),
            $(false_branch_fn_name),
            ($(all_input_vars...),),
        )
    end

    all_check_vars = [all_input_vars..., condition_vars...]
    unique!(all_check_vars)

    depth > 0 && return (
        reactant_code_block, (true_branch_fn_name, false_branch_fn_name), all_check_vars
    )

    return quote
        if any($(is_traced), ($(all_check_vars...),))
            $(reactant_code_block)
        else
            $(original_expr)
        end
    end
end

function remove_shortcircuiting(expr)
    return MacroTools.prewalk(expr) do x
        if MacroTools.@capture(x, a_ && b_)
            return :($a & $b)
        elseif MacroTools.@capture(x, a_ || b_)
            return :($a | $b)
        end
        return x
    end
end

# Generate this dummy function and later we remove it during tracing
function traced_if(cond, true_fn, false_fn, args)
    return cond ? true_fn(args) : false_fn(args)
end

function traced_while(cond_fn, body_fn, args)
    while cond_fn(args...)
        args = body_fn(args...)
    end
    return args
end

function cleanup_expr_to_avoid_boxing(expr, prepend::Symbol, all_vars)
    return MacroTools.postwalk(expr) do x
        if x isa Symbol && x ∈ all_vars
            return Symbol(prepend, x)
        end
        return x
    end
end

const CONTROL_FLOW_EXPRS = [:return, :break, :continue, :symbolicgoto]

function error_if_any_control_flow(expr)
    return MacroTools.postwalk(expr) do x
        for head in CONTROL_FLOW_EXPRS
            if Meta.isexpr(x, head)
                error("Cannot use @trace on a block that contains a $head statement")
            end
        end
        return x
    end
end

end
