module ReactantCore

using ExpressionExplorer: ExpressionExplorer
using MacroTools: MacroTools, @capture

export @trace, within_compile, MissingTracedValue, promote_to_traced

# Traits
function is_traced((@nospecialize x::T), seen=Base.IdSet()) where {T}
    if !isprimitivetype(x)
        for fn in fieldnames(T)
            f = getfield(x, fn)
            if !(f in seen)
                push!(seen, f)
                is_traced(f, seen) && return true
            end
        end
    end
    return false
end

is_traced(T::Type) = false

# New Type signifying that a value is missing
mutable struct MissingTracedValue
    paths::Tuple
end

MissingTracedValue() = MissingTracedValue(())

Base.zero(::MissingTracedValue) = MissingTracedValue()

const SPECIAL_SYMBOLS = [
    :(:), :nothing, :missing, :Inf, :Inf16, :Inf32, :Inf64, :Base, :Core
]

# Function definitions needed to be defined for loop tracing
# Defined in TracedUtils.jl
function promote_to_traced end

"""
    within_compile()

Returns true if this function is executed in a Reactant compilation context, otherwise false.
"""
@inline within_compile() = false # behavior is overwritten in Interpreter.jl

# Code generation
"""
    @trace [key = val,...] <expr>

Converts certain expressions like control flow into a Reactant friendly form. Importantly,
if no traced value is found inside the expression, then there is no overhead.

## Currently Supported

- `if` conditions (with `elseif` and other niceties) (`@trace if ...`)
- `if` statements with a preceeding assignment (`@trace a = if ...`) (note the positioning
  of the macro needs to be before the assignment and not before the `if`)
- `for` statements with a single induction variable iterating over integers with known `step`
- `while` statements

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

Symbols like `$(SPECIAL_SYMBOLS)` are not allowed as variables in `@trace` expressions. While certain cases might work but these are not guaranteed to work. For
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

### Configuration

The behavior of loops can be configured with the following configuration options:

 - `track_numbers::Union{Bool,Datatype}` - whether Julia numbers should be automatically promoted to traced numbers upon entering the loop.
 - `checkpointing::Bool` - whether or not to enable checkpointing when performing reverse mode differentiation (default: `false`).
 - `mincut::Bool` - whether or not to enable the mincut algorithm when performing reverse mode differentiation (default: `false`).
"""
macro trace(args...)
    track_numbers = true
    checkpointing = false
    mincut = false

    expr = first(args)
    while length(args) > 1
        if Meta.isexpr(args[1], :(=))
            tn_expr = args[1]
            key, val = tn_expr.args
            key ∈ (:track_numbers, :checkpointing, :mincut) || error(
                "@trace supports setting track_numbers, checkpointing or mincut, but got $(tn_expr)",
            )

            if key === :track_numbers
                track_numbers = val
            elseif key === :checkpointing
                checkpointing = val
            elseif key === :mincut
                mincut = val
            end
            args = args[2:end]
        else
            break
        end
    end
    expr = only(args)

    track_numbers = track_numbers ? Number : Union{}
    expr = macroexpand(__module__, expr)

    #! format: off
    if @capture(
        expr,
        (
            fnname_(call_args__) where {Typs__} = fnbody_
        ) | (
            fnname_(call_args__) = fnbody_
        ) | (
            function fnname_(call_args__)
                fnbody_
            end
        ) | (
            function fnname_(call_args__) where {Typs__}
                fnbody_
            end
        )
    )
        return esc(trace_function_definition(__module__, expr))
    end
    #! format: on

    if Meta.isexpr(expr, :(=))
        if Meta.isexpr(expr.args[2], :if)
            return esc(trace_if_with_returns(expr; track_numbers))
        end
    end

    Meta.isexpr(expr, :call) && return esc(trace_call(__module__, expr))

    if Meta.isexpr(expr, :(.), 2) && Meta.isexpr(expr.args[2], :tuple)
        fname = :($(Base.Broadcast.BroadcastFunction)($(expr.args[1])))
        args = only(expr.args[2:end]).args
        call = Expr(:call, fname, args...)
        return esc(trace_call(__module__, call))
    end

    Meta.isexpr(expr, :if) && return esc(trace_if(expr; track_numbers))

    Meta.isexpr(expr, :for) &&
        return (esc(trace_for(expr; track_numbers, checkpointing, mincut)))

    Meta.isexpr(expr, :while) &&
        return (esc(trace_while(expr; track_numbers, checkpointing, mincut)))

    return error(
        "Only `if-elseif-else` blocks, function definitions, `function calls`, `for` and \
        `while` loops are currently supported by `@trace`"
    )
end

function get_argname(expr)
    if Meta.isexpr(expr, :(::))
        length(expr.args) == 2 && return expr.args[1], expr
        @assert length(expr.args) == 1
        var = gensym(:_)
        return var, Expr(:(::), var, expr.args[1])
    end
    Meta.isexpr(expr, :kw) && return get_argname(expr.args[1])[1], expr
    Meta.isexpr(expr, :(...)) && return expr, expr
    @assert expr isa Symbol
    if expr == :_
        var = gensym(:_)
        return var, var
    end
    return expr, expr
end

function trace_function_definition(mod, expr)
    internal_fn = MacroTools.splitdef(expr)
    orig_fname = internal_fn[:name]

    isfunctor = Meta.isexpr(orig_fname, :(::))
    fname = gensym(Symbol(orig_fname, :internal))
    internal_fn[:name] = fname

    if isfunctor
        if length(orig_fname.args) == 1
            sym_name = gensym("functor")
            orig_fname = Expr(:(::), sym_name, orig_fname.args[1])
        end
        @assert length(orig_fname.args) == 2
        insert!(internal_fn[:args], 1, :($orig_fname))
    end

    new_fn = MacroTools.splitdef(expr)

    standardized_argnames = get_argname.(new_fn[:args])
    argnames = first.(standardized_argnames)
    new_fn[:args] = last.(standardized_argnames)

    if isfunctor
        insert!(argnames, 1, orig_fname.args[1])
    end

    if isempty(new_fn[:kwargs])
        traced_call_expr = :($(traced_call)($(fname), $(argnames...)))
        untraced_call_expr = :($(fname)($(argnames...)))
    else
        kws = first.(get_argname.(new_fn[:kwargs]))
        traced_call_expr =
            :($(traced_call)(Core.kwcall, (; $(kws...)), $(fname), $(argnames...)))
        untraced_call_expr = :(Core.kwcall((; $(kws...)), $(fname), $(argnames...)))
    end

    new_fn[:name] = orig_fname
    new_fn[:body] = :(
        if $(within_compile)() && $(any)($(is_traced), ($(argnames...),))
            return $(traced_call_expr)
        else
            return $(untraced_call_expr)
        end
    )

    return quote
        $(MacroTools.combinedef(new_fn))
        $(MacroTools.combinedef(internal_fn))
    end
end

function trace_while(expr; track_numbers, mincut, checkpointing, first_arg=nothing)
    Meta.isexpr(expr, :while, 2) || error("expected while expr")
    cond, body = expr.args

    error_if_any_control_flow(body)

    cond_symbols = ExpressionExplorer.compute_symbols_state(cond)
    body_symbols = ExpressionExplorer.compute_symbols_state(body)

    external_syms = Symbol[]
    if !isnothing(first_arg)
        push!(external_syms, first_arg)
    end
    union!(external_syms, cond_symbols.references)
    union!(external_syms, cond_symbols.assignments)
    union!(external_syms, body_symbols.references)
    union!(external_syms, body_symbols.assignments)
    filter!(∉(SPECIAL_SYMBOLS), external_syms)

    all_syms = Expr(:tuple, external_syms...)
    args_names = Expr(:tuple, external_syms...)

    cond_val(s) = :(@isdefined($s) ? $s : nothing)
    args_init = Expr(:tuple, (:(Ref($(cond_val(s)))) for s in external_syms)...)

    ref_syms = Symbol[Symbol(string(sym), "_ref") for sym in external_syms]
    arg_syms = Expr(:tuple, ref_syms...)

    to_locals = [:(local $s = $ref[]) for (s, ref) in zip(external_syms, ref_syms)]
    from_locals = [(
        quote
            if !isnothing($ref[])
                $ref[] = $s
            end
        end
    ) for (s, ref) in zip(external_syms, ref_syms)]

    body_fn_sym = gensym(:body_fn)
    cond_fn_sym = gensym(:cond_fn)
    args_sym = gensym(:args)
    verify_arg_names_sym = gensym(:verify_arg_names)

    reactant_code_block = quote
        let $args_sym = $(args_init)
            $cond_fn_sym = $(arg_syms) -> begin
                $(to_locals...)
                $cond
            end
            $body_fn_sym = $(arg_syms) -> begin
                $(to_locals...)
                $body
                $(from_locals...)
                nothing
            end

            $(verify_arg_names_sym) = if sizeof($(cond_fn_sym)) != 0
                (Symbol($cond_fn_sym), $(QuoteNode.(args_names.args)...))
            else
                ($(QuoteNode.(args_names.args)...),)
            end

            $(ReactantCore).traced_while(
                $(cond_fn_sym),
                $(body_fn_sym),
                $(args_sym);
                track_numbers=($(track_numbers)),
                verify_arg_names=($(verify_arg_names_sym)),
                mincut=($(mincut)),
                checkpointing=($(checkpointing)),
            )
        end
    end

    return quote
        if $(within_compile)() &&
            $(any)($(is_traced), $(Expr(:tuple, cond_val.(all_syms.args)...)))
            $(reactant_code_block)
        else
            $(expr)
        end
    end
end

function trace_for(expr; track_numbers, checkpointing, mincut)
    Meta.isexpr(expr, :for, 2) || error("expected for expr")
    assign, body = expr.args

    error_if_any_control_flow(body)
    if !Meta.isexpr(assign, :(=)) || !(assign.args[1] isa Symbol)
        error(
            "malformed for loop assignment, expected a single induction variable, got $assign",
        )
    end

    induction, range = assign.args

    counter = gensym(:i)
    num_iters = gensym(:num_iters)
    range_sym = gensym(:range)

    start_sym = gensym(:start)
    step_sym = gensym(:step)
    limit_sym = gensym(:limit)

    # Unwrap the start:step:limit syntax since we cannot create such a range with tracedrnumbers
    # because its length would be unknown.
    bounds_defs = if Meta.isexpr(range, :call) && range.args[begin] == :(:)
        local start = range.args[2]
        local step = length(range.args) == 3 ? :(one($start_sym)) : range.args[3]
        local limit = range.args[end]
        quote
            $start_sym = $start
            $step_sym = $step
            $limit_sym = $limit
        end
    else
        quote
            local $range_sym = $range

            $start_sym = first($range_sym)
            $step_sym = step($range_sym)
            $limit_sym = last($range_sym)
        end
    end

    return quote
        local $start_sym, $limit_sym, $step_sym
        $bounds_defs

        if $(within_compile)()
            $start_sym = $(ReactantCore.promote_to_traced)($start_sym)
            $limit_sym = $(ReactantCore.promote_to_traced)($limit_sym)
            $step_sym = $(ReactantCore.promote_to_traced)($step_sym)
        end

        local $counter = zero($start_sym)

        $(trace_while(
            Expr(
                :while,
                quote
                    local $num_iters = div($limit_sym - $start_sym, $step_sym)
                    $counter < $num_iters + one($num_iters)
                end,
                quote
                    local $induction = $start_sym + $counter * $step_sym
                    $counter = $counter + one($counter)
                    $body
                end,
            );
            track_numbers,
            first_arg=counter,
            checkpointing,
            mincut,
        ))
    end
end

# ... = if ... style expressions
function trace_if_with_returns(expr; track_numbers)
    new_expr, _, all_check_vars = trace_if(
        expr.args[2]; store_last_line=expr.args[1], depth=1, track_numbers
    )
    cond_name = first(all_check_vars)
    original_cond = expr.args[2].args[1]
    expr.args[2].args[1] = cond_name
    return quote
        $(cond_name) = $(original_cond)
        if $(within_compile)() && $(any)($(is_traced), ($(all_check_vars...),))
            $(new_expr)
        else
            $(expr)
        end
    end
end

function trace_if(expr; store_last_line=nothing, depth=0, track_numbers)
    discard_vars_from_expansion = []
    original_expr = expr

    if depth == 0
        error_if_any_control_flow(expr)

        counter = 0
        expr = MacroTools.prewalk(expr) do x
            counter += 1
            if x isa Expr && x.head == :if && counter > 1
                ex_new, dv, _ = trace_if(x; store_last_line, depth=depth + 1, track_numbers)
                append!(discard_vars_from_expansion, dv)
                return ex_new
            end
            return x
        end
    end

    cond_expr = remove_shortcircuiting(expr.args[1])
    condition_vars = [ExpressionExplorer.compute_symbols_state(cond_expr).references...]

    true_block = if store_last_line !== nothing
        if expr.args[2] isa Expr
            @assert expr.args[2].head == :block "currently we only support blocks"
            expr.args[2] = Expr(:block, expr.args[2].args...)
            true_last_line = expr.args[2].args[end]
            remaining_lines = expr.args[2].args[1:(end - 1)]
        else
            true_last_line = expr.args[2]
            remaining_lines = []
        end
        quote
            $(remaining_lines...)
            $(store_last_line) = $(true_last_line)
        end
    else
        quote
            $(expr.args[2])
            nothing # explicitly return nothing to prevent branches from returning different types
        end
    end

    true_branch_symbols = ExpressionExplorer.compute_symbols_state(true_block)
    true_branch_input_list = [true_branch_symbols.references...]
    filter!(x -> x ∉ SPECIAL_SYMBOLS, true_branch_input_list)
    true_branch_assignments = [true_branch_symbols.assignments...]
    all_true_branch_vars = true_branch_input_list ∪ true_branch_assignments
    true_branch_fn_name = gensym(:true_branch)

    else_block, discard_vars, _ = if length(expr.args) == 3
        if !(expr.args[3] isa Expr) || expr.args[3].head != :elseif
            expr.args[3], [], nothing
        else
            trace_if(expr.args[3]; store_last_line, depth=depth + 1, track_numbers)
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
        if else_block isa Expr
            @assert else_block.head == :block "currently we only support blocks"
            false_last_line = else_block.args[end]
            remaining_lines = else_block.args[1:(end - 1)]
        else
            false_last_line = else_block
            remaining_lines = []
        end
        quote
            $(remaining_lines...)
            $(store_last_line) = $(false_last_line)
        end
    else
        quote
            $else_block
            nothing # explicitly return nothing to prevent branches from returning different types
        end
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

    non_existent_true_branch_vars = setdiff(
        all_output_vars, all_true_branch_vars, all_input_vars
    )
    true_branch_extras = Expr(
        :block,
        [:($(var) = $(MissingTracedValue)()) for var in non_existent_true_branch_vars]...,
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

    non_existent_false_branch_vars = setdiff(
        all_output_vars, all_false_branch_vars, all_input_vars
    )
    false_branch_extras = Expr(
        :block,
        [:($(var) = $(MissingTracedValue)()) for var in non_existent_false_branch_vars]...,
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

    cond_name = gensym(:cond)

    reactant_code_block = quote
        $(true_branch_fn)
        $(false_branch_fn)
        ($(all_output_vars...),) = $(traced_if)(
            $(cond_name),
            $(true_branch_fn_name),
            $(false_branch_fn_name),
            ($(all_input_vars...),);
            track_numbers=($(track_numbers)),
        )
    end

    non_reactant_code_block = Expr(:if, cond_name, original_expr.args[2])
    if length(original_expr.args) > 2 # has else block
        append!(non_reactant_code_block.args, original_expr.args[3:end])
    end

    all_check_vars = [cond_name, all_input_vars..., condition_vars...]
    unique!(all_check_vars)

    depth > 0 && return (
        quote
            $(cond_name) = $(cond_expr)
            $(reactant_code_block)
        end,
        (true_branch_fn_name, false_branch_fn_name),
        all_check_vars,
    )

    return quote
        $(cond_name) = $(cond_expr)
        if $(within_compile)() && $(any)($(is_traced), ($(all_check_vars...),))
            $(reactant_code_block)
        else
            $(non_reactant_code_block)
        end
    end
end

function correct_maybe_bcast_call(fname)
    startswith(string(fname), '.') || return false, fname, fname
    return true, Symbol(string(fname)[2:end]), fname
end

function trace_call(mod, call)
    bcast, fname, fname_full = correct_maybe_bcast_call(call.args[1])
    f = if bcast
        quote
            if isdefined(mod, $(Meta.quot(fname_full)))
                $(fname_full)
            else
                Base.Broadcast.BroadcastFunction($(fname))
            end
        end
    else
        :($(fname))
    end
    return quote
        if $(within_compile)()
            $(traced_call)($f, $(call.args[2:end]...))
        else
            $(call)
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
function traced_if(cond, true_fn, false_fn, args; track_numbers)
    return cond ? true_fn(args) : false_fn(args)
end

function traced_while end # defined inside Reactant.jl

traced_call(f, args...; kwargs...) = f(args...; kwargs...)

function cleanup_expr_to_avoid_boxing(expr, prepend::Symbol, all_vars)
    return MacroTools.postwalk(expr) do x
        if Meta.isexpr(x, :kw) # undo lhs rewriting
            if startswith(string(x.args[1]), string(prepend))
                return Expr(
                    :kw,
                    Symbol(string(x.args[1])[(length(string(prepend)) + 1):end]),
                    x.args[2],
                )
            end
        end
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

"""
    materialize_traced_array(AbstractArray{<:TracedRNumber})::TracedRArray

Given an AbstractArray{TracedRNumber}, return or create an equivalent TracedRArray.

"""
function materialize_traced_array end

end
