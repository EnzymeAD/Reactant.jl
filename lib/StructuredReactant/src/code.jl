# The structured IR and prepared operands of one inferred MethodInstance.
struct Code
    sci::StructuredIRCode
    method::Method
    sparams::Core.SimpleVector
    blocks::IdDict{Block,PreparedBlock}
    # Memoized per control-flow op: does it return, does it leave the loop.
    returns::IdDict{Any,Bool}
    exits::IdDict{Any,Bool}
end
function Code(sci, method, sparams)
    return Code(sci, method, sparams, prepare(sci), IdDict{Any,Bool}(), IdDict{Any,Bool}())
end

# Attached to the CodeInstance, so it is invalidated with the inferred source.
mutable struct PreparedCode
    code::Union{Nothing,Code}
end
PreparedCode() = PreparedCode(nothing)

# What a signature dispatches to in one world: a new overload can select another
# method without invalidating the old one. The prepared code outlives it on `ci`.
struct Resolution
    code::Union{Nothing,Code}   # `nothing` marks a leaf
    world::UInt
    ci::Union{Nothing,Core.CodeInstance}
end

# An `IdDict`: storing into a `Dict{Type}` compiles once per signature.
const RESOLUTIONS = IdDict{Any,Resolution}()
const RESOLUTIONS_LOCK = ReentrantLock()

function covers(r::Resolution, world::UInt)
    return r.world == world &&
           (r.ci === nothing || r.ci.min_world <= world <= r.ci.max_world)
end

# What `sig` dispatches to in `world`: `nothing` without a method, a leaf, or
# the method's structured IR; cached until a method definition invalidates it.
function resolve(@nospecialize(sig::Type), world::UInt)
    # Not `lock(f, l)`: Julia 1.12 types a closure by the exact type of what it
    # captures, `Type{sig}` here, and would compile `lock` once per signature.
    lock(RESOLUTIONS_LOCK)
    try
        return Base.invoke_in_world(
            COMPILER_WORLD[], resolve_locked, sig, world
        )::Union{Nothing,Resolution}
    finally
        unlock(RESOLUTIONS_LOCK)
    end
end

function resolve_locked(@nospecialize(sig::Type), world::UInt)
    cached = get(RESOLUTIONS, sig, nothing)
    cached !== nothing && covers(cached, world) && return cached

    interp = Interpreter(world)
    match, _ = CC.findsup(sig, CC.method_table(interp))
    match === nothing && return nothing

    resolution = if isleaf(match.method, sig)
        Resolution(nothing, world, nothing)
    else
        ci, code = prepared_code(interp, match, specialization(interp, match, sig), sig)
        Resolution(code, world, ci)
    end
    RESOLUTIONS[sig] = resolution
    return resolution
end

# The CodeInstance of `mi` and its structured IR, built once per CodeInstance.
function prepared_code(
    interp::Interpreter,
    match::Core.MethodMatch,
    mi::Core.MethodInstance,
    @nospecialize(sig)
)
    cache = CacheView{PreparedCode}(CC.cache_owner(interp), interp.world)
    ci = get(cache, mi, nothing)
    ci === nothing && (ci = CompilerCaching.typeinf!(interp, mi))
    ci === nothing && throw(FrontendError("inference of $(sig) failed"))
    prepared = CompilerCaching.results(cache, ci)
    prepared.code === nothing || return ci, prepared.code

    src = CompilerCaching.get_source(ci)
    if src === nothing   # cached without its source: infer again
        ci = CompilerCaching.typeinf!(interp, mi)
        src = ci === nothing ? nothing : CompilerCaching.get_source(ci)
        src === nothing && throw(FrontendError("inferred source of $(sig) unavailable"))
        prepared = CompilerCaching.results(cache, ci)
    end
    prepared.code = structure(CC.inflate_ir(src, mi), match.method, mi, interp.world)
    return ci, prepared.code
end

# `ir` is a copy (`inflate_ir` makes one): inference and other resolutions still
# use the cached source. Cleanup reads globals in the traced program's `world`.
function structure(ir::CC.IRCode, method::Method, mi::Core.MethodInstance, world::UInt)
    ir = strip_exception_handling!(ir, world)
    sci = try
        StructuredIRCode(ir)
    catch err
        message = sprint(showerror, err)
        throw(
            FrontendError(
                "the control flow of $(method) could not be structured: " * message
            ),
        )
    end
    return Code(sci, method, mi.sparam_vals)
end

# Traced Bool arguments are admitted as host Bools, as leaf results are; the
# concrete signature stays the cache key.
function specialization(interp::Interpreter, match::Core.MethodMatch, @nospecialize(sig))
    widened = Tuple{map(widen_traced_bool, sig.parameters)...}
    widened === sig && return CC.specialize_method(match)
    rematch, _ = CC.findsup(widened, CC.method_table(interp))
    rematch !== nothing &&
        rematch.method === match.method &&
        return CC.specialize_method(rematch)
    return CC.specialize_method(match)
end

function widen_traced_bool(@nospecialize(T))
    return T === TracedRNumber{Bool} ? Union{Bool,TracedRNumber{Bool}} : T
end

# Handlers are dropped: nothing throws in the compiled program, and an exception
# while tracing aborts the compile. The normal path keeps its `finally` copy.
function strip_exception_handling!(ir::CC.IRCode, world::UInt)
    found = false
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
        if stmt isa Core.EnterNode
            found = true
            CC.kill_edge!(ir, CC.block_for_inst(ir.cfg, i), stmt.catch_dest)
            setstmt!(ir, i, nothing)
        elseif stmt isa Core.UpsilonNode ||
            (stmt isa Expr && stmt.head in (:leave, :pop_exception))
            setstmt!(ir, i, nothing)
        end
    end
    found || return ir
    return fold_trivial_phis!(decide_literal_branches!(CC.compact!(ir, true), world))
end

# Julia 1.11 lowers `finally` to one body dispatching on a state value; with the
# handler gone the rethrow branch compares two literals, and left undecided it
# would become a throwing exit carried out of an enclosing loop.
function decide_literal_branches!(ir::CC.IRCode, world::UInt)
    decided = false
    for (b, block) in enumerate(ir.cfg.blocks)
        i = last(block.stmts)
        t = ir.stmts[i][:stmt]
        t isa Core.GotoIfNot || continue
        taken = literal_condition(ir, t.cond, world)
        taken === nothing && continue
        if taken
            setstmt!(ir, i, nothing)
            CC.kill_edge!(ir, b, t.dest)
        else
            setstmt!(ir, i, Core.GotoNode(t.dest))
            CC.kill_edge!(ir, b, b + 1)
        end
        decided = true
    end
    return decided ? CC.compact!(ir, true) : ir
end

function literal_condition(ir::CC.IRCode, @nospecialize(cond), world::UInt)
    cond isa Bool && return cond
    cond isa Core.SSAValue || return nothing
    stmt = ir.stmts[cond.id][:stmt]
    is_call(stmt, ===, 2, world) || return nothing
    a, b = literal(stmt.args[2]), literal(stmt.args[3])
    (a === nothing || b === nothing) && return nothing
    return a[] === b[]
end

# A literal operand, boxed so that `nothing` itself can be one.
function literal(@nospecialize(x))
    x isa QuoteNode && (x = x.value)
    x isa Union{Core.SSAValue,Core.Argument,GlobalRef,Expr,Core.PhiNode} && return nothing
    return Ref{Any}(x)
end

# Julia 1.11 saves every variable a `try` reads in a slot, which leaves loop phis
# merging a value with itself once the catch edge is gone; such a phi would be a
# loop carry that rolls the loop from its first iteration.
function fold_trivial_phis!(ir::CC.IRCode)
    rename = Dict{Int,Any}()
    function target(@nospecialize(v))
        seen = 0
        while v isa Core.SSAValue && haskey(rename, v.id) && (seen += 1) <= length(rename)
            v = rename[v.id]
        end
        return v
    end
    # A phi turns trivial once the phis it merges with fold: fixed point.
    changed = true
    while changed
        changed = false
        for i in 1:length(ir.stmts)
            haskey(rename, i) && continue
            phi = ir.stmts[i][:stmt]
            phi isa Core.PhiNode || continue
            value = Ref{Any}()
            trivial = all(eachindex(phi.values)) do k
                isassigned(phi.values, k) || return false
                v = target(phi.values[k])
                v === Core.SSAValue(i) && return true
                isassigned(value) || (value[] = v)
                return v === value[]
            end
            trivial && isassigned(value) || continue
            rename[i] = value[]
            changed = true
        end
    end
    isempty(rename) && return ir
    for i in 1:length(ir.stmts)
        setstmt!(ir, i, haskey(rename, i) ? nothing : CC.ssamap(target, ir.stmts[i][:stmt]))
    end
    return ir
end

# On Julia 1.11 `Base.setindex!` has no method for an `Instruction`.
function setstmt!(ir::CC.IRCode, i::Int, @nospecialize(stmt))
    return CC.setindex!(ir.stmts[i], stmt, :stmt)
end
