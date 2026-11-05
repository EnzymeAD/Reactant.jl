# The emitter: an interpreter over structured IR. Control flow is in regions.jl.

# The iterator state of a `TracedUnitRange`, whose `iterate` cannot decide
# termination on the host; the `=== nothing` test becomes a traced comparison.
struct Iteration{I,S}
    i::I
    stop::S
end

# A traced range crosses a region as its endpoints; Reactant's tracer does not
# handle the range type.
struct RangeCapture{S,T}
    start::S
    stop::T
    length::Int
end

# Emission state of one method activation; regions fork it.
mutable struct Frame
    const code::Code
    const arguments::Vector{Any}   # Argument(n) => arguments[n]; arguments[1] is the callee
    const ssa::Vector{Any}         # SSAValue(id) => ssa[id]
    const blockargs::Vector{Any}   # BlockArgument(id) => blockargs[id]
    const parent::Union{Nothing,Frame}
    pc::Int                        # SSA index being emitted, for diagnostics
    exits::Bool                    # in a general loop body: continue/break yield (done, values...)
end

function Frame(
    code::Code, @nospecialize(f), args::Vector{Any}, parent::Union{Nothing,Frame}
)
    sci = code.sci
    return Frame(
        code,
        pack_arguments(code.method, f, args),
        Vector{Any}(undef, sci.max_ssa_idx),
        Vector{Any}(undef, sci.max_arg_idx),
        parent,
        0,
        false,
    )
end

function fork(fr::Frame; exits::Bool=fr.exits)
    return Frame(
        fr.code,
        copy(fr.arguments),
        copy(fr.ssa),
        copy(fr.blockargs),
        fr.parent,
        fr.pc,
        exits,
    )
end

# Makes `dest` a fresh fork of `source` again, for the next host loop iteration.
function reset_frame!(dest::Frame, source::Frame; exits::Bool=source.exits)
    copyto!(dest.arguments, source.arguments)
    copyto!(dest.ssa, source.ssa)
    copyto!(dest.blockargs, source.blockargs)
    dest.pc = source.pc
    dest.exits = exits
    return dest
end

# A vararg method receives its trailing arguments as one tuple.
function pack_arguments(method::Method, @nospecialize(f), args::Vector{Any})
    method.isva || return Any[f, args...]
    fixed = Int(method.nargs) - 2
    return Any[f, args[1:fixed]..., Tuple(@view args[(fixed + 1):end])]
end

@inline function operand(fr::Frame, x::PreparedOperand)
    kind = x.kind
    kind === SSA && return fr.ssa[x.index]
    kind === ARGUMENT && return fr.arguments[x.index]
    kind === BLOCK_ARGUMENT && return fr.blockargs[x.index]
    if kind === GLOBAL
        ref = x.value::GlobalRef
        return getglobal(ref.mod, ref.name)
    elseif kind === UNDEFINED
        return operand(fr, x.value::Undef)
    end
    return x.value
end

operand(::Frame, @nospecialize(x)) = x
operand(fr::Frame, x::Core.SSAValue) = fr.ssa[x.id]
operand(fr::Frame, x::Core.Argument) = fr.arguments[x.n]
operand(fr::Frame, x::BlockArgument) = fr.blockargs[x.id]
operand(fr::Frame, x::Core.PiNode) = operand(fr, x.val)
operand(::Frame, x::QuoteNode) = x.value
operand(::Frame, x::GlobalRef) = getglobal(x.mod, x.name)
# An `undef` operand is never read on its path; a number is materialized so that
# a slot undefined in both branches of a traced `if` needs no filling.
function operand(::Frame, u::Undef)
    T = u.type
    T isa DataType && T <: Number && return zero(T)
    return MissingTracedValue()
end

operands(fr::Frame, xs) = Any[operand(fr, x) for x in xs]

# `any`, `all` and `map` that do not specialize on the tuple's type, which the
# Base functions do, once per tuple type of the program.
function tuple_any(f, @nospecialize(t::Union{Tuple,NamedTuple}))
    for i in 1:nfields(t)
        f(getfield(t, i)) && return true
    end
    return false
end
function tuple_all(f, @nospecialize(t::Union{Tuple,NamedTuple}))
    for i in 1:nfields(t)
        f(getfield(t, i)) || return false
    end
    return true
end
function tuple_map(f, @nospecialize(t::Union{Tuple,NamedTuple}))
    values = Vector{Any}(undef, nfields(t))
    for i in 1:nfields(t)
        values[i] = f(getfield(t, i))
    end
    return Tuple(values)
end

bind!(fr::Frame, x::Core.SSAValue, @nospecialize(v)) = (fr.ssa[x.id] = v)
bind!(fr::Frame, x::Core.Argument, @nospecialize(v)) = (fr.arguments[x.n] = v)
bind!(fr::Frame, x::BlockArgument, @nospecialize(v)) = (fr.blockargs[x.id] = v)

function bind!(fr::Frame, keys::Vector, @nospecialize(values::Tuple))
    length(keys) == nfields(values) || unsupported(
        fr, "a region receiving $(nfields(values)) values for $(length(keys)) arguments"
    )
    for i in eachindex(keys)
        bind!(fr, keys[i], uncapture(getfield(values, i)))
    end
    return fr
end

uncapture(@nospecialize(v)) = v
function uncapture(v::RangeCapture)
    T = typeof(v.start)
    return Reactant.TracedUnitRange{T}(v.start, convert(T, v.stop), v.length)
end

unsupported(::Frame, what) = throw(FrontendError(string(what, " is not supported")))

# Whether `x` is or contains a traced value.
function has_traced(@nospecialize(x), seen::Base.IdSet{Any}=Base.IdSet{Any}())
    x isa TracedType && return true
    x isa Union{Type,Module,Symbol,AbstractString,Core.MethodInstance,Method} &&
        return false
    T = typeof(x)
    isprimitivetype(T) && return false
    if x isa Array   # other array types (wrappers, ranges) are searched by field
        isbitstype(eltype(x)) && return false
        for i in eachindex(x)
            isassigned(x, i) && has_traced(x[i], seen) && return true
        end
        return false
    end
    if ismutabletype(T)
        x in seen && return false
        push!(seen, x)
    end
    for i in 1:fieldcount(T)
        isdefined(x, i) && has_traced(getfield(x, i), seen) && return true
    end
    return false
end

function describe(fr::Frame)
    method = fr.code.method
    location = try
        scopes = IRStructurizer.source_location(fr.code.sci, fr.pc)
        isempty(scopes) ? "" : string(" at ", last(scopes).file, ":", last(scopes).line)
    catch
        ""
    end
    return string(method.name, " in ", method.module, location)
end

# How a block ends. Regions must yield; the function body must return.
struct Returned
    value::Any
end
struct Yielded
    values::Tuple
end
struct Conditioned
    condition::Any
    values::Tuple
end

# The rest of a block after an `if` containing a `return`: regions cannot return
# from the function, so the rest is emitted inside each branch that yields.
struct Continuation
    block::Block
    position::Int
    next::Union{Nothing,Continuation}
end

function resume(k::Continuation, fr::Frame, yielded::Tuple)
    fr.ssa[k.block.body.ssa_idxes[k.position]] = yielded
    return emit_block(fr, k.block, k.position + 1, k.next)
end

function emit_block(
    fr::Frame, block::Block, from::Int=1, k::Union{Nothing,Continuation}=nothing
)
    body = block.body
    prepared = fr.code.blocks[block]
    for position in from:length(body.ssa_idxes)
        idx = body.ssa_idxes[position]
        stmt = body.stmts[position]
        fr.pc = idx
        if stmt isa IfOp && (has_return(fr, stmt) || (fr.exits && exits_loop(fr, stmt)))
            return emit_if(fr, stmt, Continuation(block, position, k))
        end
        fr.ssa[idx] = emit_stmt(fr, prepared.statements[position], stmt)
    end
    return emit_terminator(fr, prepared.terminator, k)
end

function emit_terminator(fr::Frame, t::PreparedTerminator, k::Union{Nothing,Continuation})
    kind = t.kind
    if kind === RETURN
        return Returned(operand(fr, t.value))
    elseif kind === YIELD || kind === CONTINUE
        values = operands(fr, t.operands)
        kind === CONTINUE && fr.exits && return Yielded((false, values...))
        yielded = Tuple(values)
        return k === nothing ? Yielded(yielded) : resume(k, fr, yielded)
    elseif kind === BREAK
        fr.exits || unsupported(fr, "`break` outside a general loop")
        return Yielded((true, operands(fr, t.operands)...))
    elseif kind === CONDITION
        return Conditioned(operand(fr, t.value), Tuple(operands(fr, t.operands)))
    elseif kind === UNREACHABLE
        unsupported(fr, "code after a call that always throws")
    end
    return unsupported(fr, "a block without a terminator")
end

# Memoized in the `Code`: a block is emitted once per call and unrolled iteration.
has_return(fr::Frame, op::ControlFlowOp) = has_return(fr.code.returns, op)
function has_return(memo::IdDict{Any,Bool}, op::ControlFlowOp)
    return get!(memo, op) do
        return any(b -> has_return(memo, b), blocks(op))
    end
end
function has_return(memo::IdDict{Any,Bool}, block::Block)
    block.terminator isa Core.ReturnNode && return true
    return any(values(block.body)) do e
        return e.stmt isa ControlFlowOp && has_return(memo, e.stmt)
    end
end

# Does a branch end the enclosing loop's iteration (`continue`/`break`) on some path?
exits_loop(fr::Frame, op::IfOp) = exits_loop(fr.code.exits, op)
function exits_loop(memo::IdDict{Any,Bool}, op::IfOp)
    return get!(memo, op) do
        return any(b -> exits_loop(memo, b), blocks(op))
    end
end
function exits_loop(memo::IdDict{Any,Bool}, block::Block)
    block.terminator isa Union{BreakOp,ContinueOp} && return true
    return any(values(block.body)) do e
        return e.stmt isa IfOp && exits_loop(memo, e.stmt)
    end
end

emit_stmt(fr::Frame, @nospecialize(stmt)) = operand(fr, stmt)
emit_stmt(::Frame, ::Nothing) = nothing
emit_stmt(fr::Frame, op::IfOp) = emit_if(fr, op, nothing).values
emit_stmt(fr::Frame, op::Union{WhileOp,ForOp,LoopOp}) = emit_loop(fr, op)
emit_stmt(fr::Frame, ::Core.PhiNode) = unsupported(fr, "unstructured control flow")
function emit_stmt(fr::Frame, ::Union{Core.GotoNode,Core.GotoIfNot})
    return unsupported(fr, "unstructured control flow")
end
function emit_stmt(fr::Frame, ::Union{Core.PhiCNode,Core.UpsilonNode,Core.EnterNode})
    return unsupported(fr, "`try`/`catch`")
end
function emit_stmt(fr::Frame, ::Core.ReturnNode)
    return unsupported(fr, "code after a call that always throws")
end

const SILENT_EXPRESSIONS = (
    :meta,
    :inbounds,
    :loopinfo,
    :code_coverage_effect,
    :gc_preserve_begin,
    :gc_preserve_end,
    :aliasscope,
    :popaliasscope,
)

function emit_stmt(fr::Frame, p::PreparedStatement, @nospecialize(stmt))
    kind = p.kind
    kind === VALUE && return operand(fr, p.value)
    kind === CALL &&
        return emit_call(fr, operand(fr, p.value), operands(fr, p.operands), p.traced)
    kind === NEW && return construct(fr, operand(fr, p.value), operands(fr, p.operands))
    return emit_stmt(fr, stmt)
end

function emit_stmt(fr::Frame, ex::Expr)
    head = ex.head
    if head === :splatnew
        fields = Any[operand(fr, ex.args[2])...]
        return construct(fr, operand(fr, ex.args[1]), fields)
    elseif head === :static_parameter
        return fr.code.sparams[ex.args[1]::Int]
    elseif head === :boundscheck
        return true
    elseif head === :throw_undef_if_not
        defined = operand(fr, ex.args[2])
        defined === true && return nothing
        defined === false && throw(UndefVarError(ex.args[1]::Symbol))
        unsupported(
            fr, "reading `$(ex.args[1])`, which may be unassigned on the traced path"
        )
    elseif head in SILENT_EXPRESSIONS
        return nothing
    end
    return unsupported(fr, "the `$(head)` expression")
end

# A field that became traced after inference no longer matches the type `:new`
# was given, so the object is rebuilt through its constructor.
function construct(fr::Frame, @nospecialize(T), fields::Vector{Any})
    T isa DataType || unsupported(fr, "constructing a value of type $(T)")
    if fields_match(T, fields) || !any(has_traced, fields)
        return ccall(
            :jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, fields, length(fields)
        )
    end
    T <: NamedTuple && return NamedTuple{fieldnames(T)}(Tuple(fields))
    # `a:b` inlined to a `UnitRange`; with traced endpoints it is Reactant's range.
    T <: UnitRange && return Reactant.call_with_reactant(:, fields[1], fields[2])
    wrapper = T.name.wrapper
    if isempty(methods(wrapper))   # a closure type has no constructor
        R = reparameterize(T, fields)
        return ccall(
            :jl_new_structv, Any, (Any, Ptr{Any}, UInt32), R, fields, length(fields)
        )
    end
    return wrapper(fields...)
end

# A loop: on Julia 1.12 a closure over `T` has the type `Type{T}` in it, so `all`
# would be compiled once per struct type.
function fields_match(@nospecialize(T::DataType), fields::Vector{Any})
    for i in eachindex(fields)
        fields[i] isa fieldtype(T, i) || return false
    end
    return true
end

function reparameterize(T::DataType, fields::Vector{Any})
    body = Base.unwrap_unionall(T.name.wrapper)
    params = Any[T.parameters...]
    for i in 1:fieldcount(body)
        ft = fieldtype(body, i)
        ft isa TypeVar || continue
        j = findfirst(p -> p === ft, body.parameters)
        j === nothing || (params[j] = Core.Typeof(fields[i]))
    end
    return T.name.wrapper{params...}
end

# Builtins are handled here. Calls without traced arguments run through Reactant,
# since a host helper can still depend on its overlays or `within_compile()`,
# unless inferred to return a traced value: such a callee makes traced values from
# host ones (Reactant's `rand` on a host RNG) and may branch on them.
function emit_call(fr::Frame, @nospecialize(f), args::Vector{Any}, traced::Bool=false)
    if f isa Core.IntrinsicFunction   # intrinsics are builtins too; test them first
        return emit_intrinsic(fr, f, args)
    elseif f isa Core.Builtin
        return emit_builtin(fr, f, args, traced)
    elseif traced || has_traced(f) || any(has_traced, args)
        return emit_method(f, args, fr)
    end
    return Reactant.call_with_reactant(f, args...)
end

# Counting leaf calls tells emitting loop iterations from host ones.
const EMISSIONS = Ref(0)

function leaf(@nospecialize(f), args::Vector{Any})
    EMISSIONS[] += 1
    # `Any[...]`: `map` would type its result by the arguments, compiling for each.
    return Reactant.call_with_reactant(f, Any[structure_callback(a) for a in args]...)
end

# Dispatch on the runtime argument types: a leaf is called through Reactant with
# user callbacks among its arguments wrapped; any other method is emitted.
function emit_method(
    @nospecialize(f), @nospecialize(args::Tuple), parent::Union{Nothing,Frame}
)
    return emit_method(f, Any[args...], parent)
end
function emit_method(@nospecialize(f), args::Vector{Any}, parent::Union{Nothing,Frame})
    f === Base.iterate && iterates_traced_range(args) && return traced_iterate(args...)
    Reactant.should_rewrite_call(Core.Typeof(f)) || return leaf(f, args)
    sig = Tuple{Core.Typeof(f),Any[Core.Typeof(a) for a in args]...}
    # The trace's world, unchanged by method definitions while it yields.
    resolution = resolve(sig, Base.tls_world_age())
    resolution === nothing && return f(args...)   # no method: Julia raises the MethodError
    code = resolution.code
    code === nothing && return leaf(f, args)

    fr = Frame(code, f, args, parent)
    ancestor = parent
    while ancestor !== nothing
        ancestor.code === code && throw(
            FrontendError("recursion on traced values is not supported", [describe(fr)])
        )
        ancestor = ancestor.parent
    end
    outcome = try
        emit_block(fr, code.sci.entry)
    catch err
        err isa FrontendError && push!(err.context, describe(fr))
        rethrow()
    end
    outcome isa Returned || unsupported(fr, "a function body that does not return")
    return outcome.value
end

# Leaves compare Base functions by identity (`op === Base.add_sum`).
function structure_callback(@nospecialize(x))
    # `Enzyme.autodiff(mode, Const(loss), ...)`, as Lux's training API writes it.
    x isa Reactant.EnzymeCore.Const &&
        return Reactant.EnzymeCore.Const(structure_callback(x.val))
    x isa Function || return x
    x isa Program && return x
    root = Base.moduleroot(parentmodule(typeof(x)))
    (root in LEAF_ROOTS || isstdlib(root)) && return x
    return structured(x)
end

function emit_builtin(fr::Frame, @nospecialize(f), args::Vector{Any}, traced::Bool=false)
    if f === Core._apply_iterate
        flat = Any[]
        for i in 3:length(args)
            splat!(fr, flat, args[i])
        end
        return emit_call(fr, args[2], flat, traced)
    elseif f === Core.ifelse && args[1] isa TracedRNumber{Bool}
        return Reactant.call_with_reactant(Base.ifelse, args...)
    elseif f === Core.getfield && length(args) >= 2 && args[1] isa Iteration
        return iteration_field(fr, args[1], args[2])
    elseif f === Core.:(===) && length(args) == 2 && any(a -> a isa Iteration, args)
        return iteration_done(fr, args[1], args[2])
    elseif f === Core.typeassert && args[1] isa TracedRNumber
        # A traced number stands in for its element type.
        Reactant.unwrapped_eltype(args[1]) <: args[2] && return args[1]
    elseif f === Core.:(===) && length(args) == 2 && any(a -> a isa TracedRNumber, args)
        all(a -> a isa Number, args) && return identical(fr, args[1], args[2])
    elseif f === Core.:(===) && length(args) == 2 && any(a -> a isa TracedEnum, args)
        # Inference saw a host enum, whose `==` inlines to `===`.
        return Reactant.call_with_reactant(==, args[1], args[2])
    elseif f === Core.getfield && length(args) >= 2 && args[2] isa TracedRNumber
        unsupported(fr, "indexing a tuple or struct with a traced integer")
    end
    return f(args...)
end

# Structurization synthesizes `===` on integer discriminators, which turn traced
# through a traced `if`: `==` within one integer type, `false` across types.
function identical(fr::Frame, a::Number, b::Number)
    element(x) = x isa TracedRNumber ? Reactant.unwrapped_eltype(x) : typeof(x)
    element(a) === element(b) || return false
    element(a) <: Integer && return Reactant.call_with_reactant(==, a, b)
    return unsupported(fr, "`===` on traced floating-point values")
end

function splat!(::Frame, flat, x::Union{Tuple,NamedTuple,AbstractArray,Core.SimpleVector})
    return append!(flat, x)
end
splat!(fr::Frame, flat, ::TracedRArray) = unsupported(fr, "splatting a traced array")
function splat!(::Frame, flat, x)
    for v in x
        push!(flat, v)
    end
    return flat
end
