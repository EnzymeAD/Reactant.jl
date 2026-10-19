"""
    Reactant.Tessera

Julia frontend for [Tessera](https://github.com/EnzymeAD/Enzyme-JAX) annotations. Tessera
lets you name functions as "Tessera ops" and declare term-rewriting rules over calls to
them.

  - [`@tessera_op`](@ref) marks a function as a Tessera op. While tracing, every call to it
    becomes a `func.call` to a separate `func.func` that carries a `tessera_op` (or
    `pure_tessera_op`) attribute, in the same format the C++ frontend produces.
  - [`@tessera_optimize`](@ref) declares a rewrite rule, the equivalent of
    `#pragma optimize "..."` in C++. Rules are emitted into the module as a
    `tessera.optimizations` region, i.e. the IR that `lift-tessera-annotations` produces for
    C++ code, and applied by the Enzyme-JAX Tessera passes before the optimization
    pipeline runs (unless `optimize = false`).
"""
module Tessera

using ..Reactant: Reactant, MLIR
using ..ReactantCore: ReactantCore, MacroTools

"""
    TesseraOp(name, argnames, pure)

Describes the `tessera_op` attribute to attach to a traced function. `name` is either a bare
op name (e.g. `"tessera.pow"`), in which case the argument list is generated from
`argnames`, or a full signature (e.g. `"tessera.pow(x, n)"`), which is used verbatim.
"""
struct TesseraOp
    name::String
    argnames::Tuple{Vararg{String}}
    pure::Bool
end

has_signature(op::TesseraOp) = occursin('(', op.name)

attribute_name(op::TesseraOp) = op.pure ? "pure_tessera_op" : "tessera_op"

const ATTRIBUTE_NAMES = ("tessera_op", "pure_tessera_op")

# Count the arguments in a signature like `"tessera.pow(x, n:val=in):globals=0"`.
function signature_nargs(sig::AbstractString)
    open, close = findfirst('(', sig), findlast(')', sig)
    (isnothing(open) || isnothing(close) || close < open) &&
        error("Malformed Tessera op signature: $(repr(sig))")
    inner = strip(sig[nextind(sig, open):prevind(sig, close)])
    return isempty(inner) ? 0 : count(==(','), inner) + 1
end

function signature(op::TesseraOp, nargs::Integer)
    if has_signature(op)
        n = signature_nargs(op.name)
        n == nargs || error(
            "Tessera op signature $(repr(op.name)) has $n arguments, but the traced \
             function has $nargs MLIR arguments. Tessera ops on Reactant functions see \
             one argument per traced array (struct arguments and captured variables are \
             flattened).",
        )
        return op.name
    end
    names = if length(op.argnames) == nargs
        op.argnames
    else
        # struct arguments / closures are flattened into several MLIR arguments.
        ntuple(i -> "arg$(i - 1)", nargs)
    end
    return "$(op.name)($(join(names, ", ")))"
end

"""
    annotate!(func, op::TesseraOp)

Attach the `tessera_op` / `pure_tessera_op` attribute to the `func.func` operation `func`.
"""
function annotate!(func::MLIR.IR.Operation, op::TesseraOp)
    nargs = MLIR.IR.nargs(MLIR.IR.first_block(MLIR.IR.region(func, 1)))
    MLIR.IR.setattr!(func, attribute_name(op), MLIR.IR.Attribute(signature(op, nargs)))
    return func
end

# Reactant traces an argument passed twice as a single function argument, but a rule sees one
# operand per argument (`tessera.mul(x, x)` has two), so give each argument its own traced
# value. Mutations made through a copy are written back to the argument it copies.
unalias(x::Reactant.TracedRArray) = copy(x)
unalias(x::Reactant.TracedRNumber{T}) where {T} = Reactant.TracedRNumber{T}((), x.mlir_data)
unalias(x) = x

function with_unaliased_args(f, args::Tuple)
    unaliased = ntuple(length(args)) do i
        any(Base.Fix1(===, args[i]), args[1:(i - 1)]) ? unalias(args[i]) : args[i]
    end
    before = map(a -> a isa Reactant.TracedType ? a.mlir_data : nothing, args)
    res = f(unaliased)
    for (a, u, data) in zip(args, unaliased, before)
        a === u && continue
        if u.mlir_data != data
            Reactant.TracedUtils.set_mlir_data!(a, u.mlir_data)
        end
    end
    return res
end

function is_annotated(op::MLIR.IR.Operation)
    return any(a -> !isnothing(MLIR.IR.getattr(op, a)), ATTRIBUTE_NAMES)
end

# Rules are stored as methods of `rule` (one per `@tessera_optimize`, dispatched on a unique
# `Val`), so that rules declared at the top level of a package survive precompilation.
function rule end

"""
    rules()

Return all rewrite rules registered with [`@tessera_optimize`](@ref).
"""
function rules()
    rs = String[]
    for m in methods(rule)
        T = Base.unwrap_unionall(m.sig).parameters[2]
        (T isa DataType && T <: Val) || continue
        push!(rs, Base.invokelatest(rule, T())::String)
    end
    return unique!(rs)
end

"""
    emit_rules!(mod::MLIR.IR.Module)

If `mod` contains functions annotated with [`@tessera_op`](@ref), insert a
`tessera.optimizations` op holding every registered rewrite rule at the start of the module.
"""
function emit_rules!(mod::MLIR.IR.Module)
    body = MLIR.IR.body(mod)
    any(is_annotated, body) || return mod
    rs = rules()
    isempty(rs) && return mod

    # the dialect is registered, but only loaded on demand
    MLIR.IR.get_or_load_dialect!("tessera")
    loc = MLIR.IR.Location()
    block = MLIR.IR.Block()
    for r in rs
        push!(
            block,
            MLIR.IR.create_operation_common(
                "tessera.optimization",
                loc;
                attributes=[MLIR.IR.NamedAttribute("rule", MLIR.IR.Attribute(r))],
                results=MLIR.IR.Type[],
            ),
        )
    end
    region = MLIR.IR.Region()
    push!(region, block)
    optimizations = MLIR.IR.create_operation_common(
        "tessera.optimizations", loc; owned_regions=[region], results=MLIR.IR.Type[]
    )
    pushfirst!(body, optimizations)
    return mod
end

# The function implementing each Tessera op, as `(f, op::TesseraOp)`, registered by
# `@tessera_op` as methods (like `rule`) so that they survive precompilation.
implementation(::Val) = nothing

op_symbol(op::TesseraOp) = Symbol(first(split(op.name, '('; limit=2)))

function for_each_op(f, op::MLIR.IR.Operation)
    for region in op, block in region, inner in block
        f(inner)
        for_each_op(f, inner)
    end
end

"""
    trace_missing_ops!(mod::MLIR.IR.Module)

Trace the function of every Tessera op that a rewrite introduced a `tessera.call` to, but
that is not yet defined in `mod` (e.g. `tessera.mul` in `tessera.pow(x, 2) ->
tessera.mul(x, x)` when the program never calls `tessera.mul` itself). The function is
traced for the types of the first call's operands, so this must run while `mod` is being
traced.
"""
function trace_missing_ops!(mod::MLIR.IR.Module)
    defined = Set{String}()
    for op in MLIR.IR.body(mod)
        sym = MLIR.IR.getattr(op, "sym_name")
        isnothing(sym) || push!(defined, String(sym))
    end
    missing_calls = Dict{String,MLIR.IR.Operation}()
    for_each_op(MLIR.IR.Operation(mod)) do op
        MLIR.IR.name(op) == "tessera.call" || return nothing
        callee = MLIR.IR.flatsymbol(MLIR.IR.getattr(op, "callee"))
        callee in defined || get!(missing_calls, callee, op)
        return nothing
    end

    for (callee, call) in missing_calls
        impl = Base.invokelatest(implementation, Val(Symbol(callee)))
        isnothing(impl) &&
            error("A Tessera rule calls `$callee`, but no `@tessera_op` defines it.")
        f, tessera_op = impl
        args = map(1:MLIR.IR.noperands(call)) do i
            value = MLIR.IR.operand(call, i)
            T = eltype(MLIR.IR.julia_type(MLIR.IR.type(value)))
            if ndims(MLIR.IR.type(value)) == 0
                Reactant.TracedRNumber{T}((), value)
            else
                Reactant.TracedRArray{T}(value)
            end
        end
        # Only the function is needed: the call to it is traced into a scratch block.
        block = MLIR.IR.Block()
        MLIR.IR.activate(block)
        try
            Reactant.Ops.call(f, args...; tessera_op)
        finally
            MLIR.IR.deactivate(block)
            MLIR.IR.dispose(block)
        end
    end
    return mod
end

"""
    apply_rules!(mod::MLIR.IR.Module)

Rewrite calls to [`@tessera_op`](@ref) functions in `mod` with the rules that
[`emit_rules!`](@ref) inserted, then [`strip!`](@ref) the Tessera annotations. The rules
match calls, so this must run before the annotated functions are inlined, and it may trace
the functions of Tessera ops that only a rule calls, so `mod` must still be being traced.
"""
function apply_rules!(mod::MLIR.IR.Module)
    any(op -> MLIR.IR.name(op) == "tessera.optimizations", MLIR.IR.body(mod)) ||
        return mod
    # Until `trace_missing_ops!` runs, a rewrite may have left a call to a function that
    # does not exist yet, so the module does not verify.
    Reactant.Compiler.run_pass_pipeline!(
        mod,
        "parse-optimization-rules,llvm-to-tessera,tessera-apply-pdl",
        "tessera";
        enable_verifier=false,
    )
    trace_missing_ops!(mod)
    Reactant.Compiler.run_pass_pipeline!(
        mod, "llvm-to-tessera,tessera-to-func", "tessera_lower"
    )
    return strip!(mod)
end

"""
    strip!(mod::MLIR.IR.Module)

Remove all Tessera annotations from `mod` (rules, generated PDL patterns, and `tessera_op`
attributes), so that it can be handed to XLA.
"""
function strip!(mod::MLIR.IR.Module)
    to_erase = MLIR.IR.Operation[]
    for op in MLIR.IR.body(mod)
        opname = MLIR.IR.name(op)
        if opname == "tessera.optimizations"
            push!(to_erase, op)
        elseif opname == "builtin.module" &&
            MLIR.IR.getattr(op, "sym_name") == MLIR.IR.Attribute("patterns")
            push!(to_erase, op)
        else
            for a in ATTRIBUTE_NAMES
                MLIR.IR.rmattr!(op, a)
            end
        end
    end
    foreach(op -> MLIR.IR.dispose(MLIR.IR.rmfromparent!(op)), to_erase)
    return mod
end

"""
    @tessera_op [pure=true] [name] function f(args...) ... end

Mark `f` as a Tessera op. When `f` is called on traced values, it is traced into a separate
`func.func` (like `@trace function`) that carries a `pure_tessera_op` attribute
(`tessera_op` if `pure=false`).

`name` is the Tessera op name and defaults to the function name. It can either be a bare
name (`"tessera.pow"`), in which case the argument names are taken from the Julia
definition, or a full signature (`"tessera.pow(x, n)"`) that is used verbatim. Names in rules
written with [`@tessera_optimize`](@ref) must match these op names exactly.

Reactant functions have value semantics on tensors, so ops are marked pure by default.

# Example

```julia
@tessera_op "tessera.pow" function power(x, n)
    return x .^ n
end

@tessera_op "tessera.mul" function mul(x, y)
    return x .* y
end

@tessera_optimize "tessera.pow(x, 2) -> tessera.mul(x, x)"

@code_hlo optimize = false power(Reactant.to_rarray(rand(3)), Reactant.to_rarray(rand(3)))
```
"""
macro tessera_op(args...)
    isempty(args) && error("@tessera_op expects a function definition")
    expr = args[end]
    pure = true
    name = nothing
    for opt in args[1:(end - 1)]
        if Meta.isexpr(opt, :(=)) && opt.args[1] === :pure
            opt.args[2] isa Bool || error("@tessera_op: `pure` must be a literal Bool")
            pure = opt.args[2]
        elseif opt isa String
            name = opt
        else
            error("@tessera_op: unsupported option $(opt)")
        end
    end

    def = MacroTools.splitdef(expr)
    fname = def[:name]
    Meta.isexpr(fname, :(::)) &&
        error("@tessera_op does not support callable structs; use a named function")
    name = something(name, string(fname))
    argnames = Tuple(string(first(ReactantCore.get_argname(a))) for a in def[:args])
    op = TesseraOp(name, argnames, pure)
    internal_name = gensym(Symbol(fname, :internal))

    return esc(
        quote
            f = $(ReactantCore.trace_function_definition(
                __module__,
                expr;
                traced_call_kwargs=(Expr(:kw, :tessera_op, op),),
                internal_name,
            ))
            function $(GlobalRef(Tessera, :implementation))(
                ::$(GlobalRef(Base, :Val)){$(QuoteNode(op_symbol(op)))}
            )
                return ($internal_name, $op)
            end
            f
        end,
    )
end

"""
    @tessera_optimize "lhs -> rhs"

Register a Tessera rewrite rule, the Julia equivalent of `#pragma optimize "lhs -> rhs"`.
Rules are emitted (as a `tessera.optimizations` op) into every module that contains a
function marked with [`@tessera_op`](@ref).

# Example

```julia
@tessera_optimize "tessera.pow(x, 2) -> tessera.mul(x, x)"
```
"""
macro tessera_optimize(r)
    r isa String || error("@tessera_optimize expects a string literal, got $(r)")
    occursin("->", r) || error("Tessera rule $(repr(r)) must have the form \"lhs -> rhs\"")
    id = QuoteNode(gensym(:tessera_rule))
    return esc(:($(GlobalRef(Tessera, :rule))(::$(GlobalRef(Base, :Val)){$id}) = $r))
end

end
