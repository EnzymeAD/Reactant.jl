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
    C++ code.
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

    return esc(
        ReactantCore.trace_function_definition(
            __module__, expr; traced_call_kwargs=(Expr(:kw, :tessera_op, op),)
        ),
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
