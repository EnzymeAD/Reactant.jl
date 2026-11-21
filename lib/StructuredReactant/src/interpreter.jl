# Inference on traced types: a native interpreter over Reactant's overlay table,
# with `within_compile()` constant `true` and leaf calls (Reactant's own methods,
# Base applied to Reactant types) typed by Reactant's interpreter, not inlined.

struct CacheOwner end

struct Interpreter <: CC.AbstractInterpreter
    world::UInt
    inference_params::CC.InferenceParams
    optimization_params::CC.OptimizationParams
    inference_cache::Vector{CC.InferenceResult}
    method_table::CC.CachedMethodTable{CC.OverlayMethodTable}
end

function Interpreter(world::UInt=Base.get_world_counter())
    table = CC.OverlayMethodTable(world, Reactant.REACTANT_METHOD_TABLE)
    return Interpreter(
        world,
        CC.InferenceParams(),
        CC.OptimizationParams(),
        CC.InferenceResult[],
        CC.CachedMethodTable(table),
    )
end

CC.InferenceParams(interp::Interpreter) = interp.inference_params
CC.OptimizationParams(interp::Interpreter) = interp.optimization_params
CC.get_inference_cache(interp::Interpreter) = interp.inference_cache
CC.get_inference_world(interp::Interpreter) = interp.world
CC.method_table(interp::Interpreter) = interp.method_table
CC.cache_owner(::Interpreter) = CacheOwner()
CC.lock_mi_inference(::Interpreter, ::Core.MethodInstance) = nothing
CC.unlock_mi_inference(::Interpreter, ::Core.MethodInstance) = nothing
CC.may_discard_trees(::Interpreter) = false

# A leaf is called through `Reactant.call_with_reactant`, never emitted from IR.
function isleaf(method::Method, @nospecialize(sig))
    if isdefined(method, :external_mt)
        method.external_mt === Reactant.REACTANT_METHOD_TABLE && return true
    end
    root = Base.moduleroot(method.module)
    root in LEAF_ROOTS && return true
    return isstdlib(root) && involves_reactant(sig)
end

const LEAF_ROOTS = Module[Reactant, ReactantCore, Reactant.Enzyme, Reactant.EnzymeCore]

function isstdlib(root::Module)
    (root === Base || root === Core) && return true
    path = pathof(root)
    return path !== nothing && startswith(path, Sys.STDLIB)
end

function involves_reactant(@nospecialize(T))
    T = Base.unwrap_unionall(T)
    if T isa DataType
        Base.moduleroot(T.name.module) === Reactant && return true
        return any(involves_reactant, T.parameters)
    elseif T isa Union
        return involves_reactant(T.a) || involves_reactant(T.b)
    elseif T isa Core.TypeofVararg
        return isdefined(T, :T) && involves_reactant(T.T)
    end
    return false
end

# Leaves, foreign calls, exception handlers and host callees with loops stay out
# of line, where they run natively rather than one interpreted statement at a
# time; small host helpers still inline. Sources stay uncompressed for the policy.
CC.may_compress(::Interpreter) = false

function stays_out_of_line(@nospecialize(src), @nospecialize(info::CC.CallInfo))
    # Julia 1.13 may supply a local optimization state before publishing source.
    src isa CC.OptimizationState && (src = src.src)
    return any_call_match(leaf_match, info) ||
           opaque_to_emitter(src) ||
           (contains_loop(src) && any_call_match(host_match, info))
end

# Julia 1.12 turned `inlining_policy`, which returns the source to inline or
# `nothing`, into `src_inlining_policy`, which returns whether to inline.
@static if isdefined(CC, :src_inlining_policy)
    function CC.src_inlining_policy(
        interp::Interpreter,
        @nospecialize(src),
        @nospecialize(info::CC.CallInfo),
        stmt_flag::UInt32,
    )
        stays_out_of_line(src, info) && return false
        return @invoke CC.src_inlining_policy(
            interp::CC.AbstractInterpreter, src::Any, info::CC.CallInfo, stmt_flag::UInt32
        )
    end
else
    function CC.inlining_policy(
        interp::Interpreter,
        @nospecialize(src),
        @nospecialize(info::CC.CallInfo),
        stmt_flag::UInt32,
    )
        stays_out_of_line(src, info) && return nothing
        return @invoke CC.inlining_policy(
            interp::CC.AbstractInterpreter, src::Any, info::CC.CallInfo, stmt_flag::UInt32
        )
    end
end

function opaque_to_emitter(@nospecialize(src))
    stmts = if src isa Core.CodeInfo
        src.code
    elseif src isa CC.IRCode
        src.stmts.stmt
    else
        return false
    end
    return any(stmts) do stmt
        return (stmt isa Expr && stmt.head === :foreigncall) || stmt isa Core.EnterNode
    end
end

function contains_loop(@nospecialize(src))
    if src isa Core.CodeInfo
        for (i, stmt) in enumerate(src.code)
            stmt isa Core.GotoNode && stmt.label <= i && return true
            stmt isa Core.GotoIfNot && stmt.dest <= i && return true
        end
    elseif src isa CC.IRCode
        for (i, block) in enumerate(src.cfg.blocks)
            any(succ -> succ <= i, block.succs) && return true
        end
    end
    return false
end

function any_call_match(predicate, info::CC.MethodMatchInfo)
    return any(predicate, info.results.matches)
end
function any_call_match(predicate, info::CC.UnionSplitInfo)
    return any(Base.Fix1(any_call_match, predicate), union_split(info))
end
any_call_match(predicate, info::CC.ConstCallInfo) = any_call_match(predicate, info.call)
any_call_match(predicate, info::CC.ApplyCallInfo) = any_call_match(predicate, info.call)
function any_call_match(predicate, info::CC.UnionSplitApplyCallInfo)
    return any(Base.Fix1(any_call_match, predicate), info.infos)
end
any_call_match(predicate, info::CC.InvokeCallInfo) = predicate(info.match)
any_call_match(predicate, @nospecialize(info::CC.CallInfo)) = false

leaf_match(match::Core.MethodMatch) = isleaf(match.method, match.spec_types)
host_match(match::Core.MethodMatch) = !involves_reactant(match.spec_types)

# Julia 1.12 renamed the field holding a union split's matches.
@static if hasfield(CC.UnionSplitInfo, :split)
    union_split(info::CC.UnionSplitInfo) = info.split
else
    union_split(info::CC.UnionSplitInfo) = info.matches
end

# Type a leaf with Reactant's interpreter and stop there.
function CC.abstract_call_method(
    interp::Interpreter,
    method::Method,
    @nospecialize(sig),
    sparams::Core.SimpleVector,
    hardlimit::Bool,
    si::CC.StmtInfo,
    sv::CC.AbsIntState,
)
    if isleaf(method, sig)
        rt = leaf_return_type(method, sig, sparams, interp.world)
        # A structured callback returns where Reactant's sees it always throw.
        rt === Union{} && (rt = Any)
        # Admitting a host Bool keeps branches on traced Bools alive.
        if TracedRNumber{Bool} in Base.uniontypes(rt)
            rt = Union{rt,Bool}
        end
        return deferred(leaf_call_result(rt))
    end
    return @invoke CC.abstract_call_method(
        interp::CC.AbstractInterpreter,
        method::Method,
        sig::Any,
        sparams::Core.SimpleVector,
        hardlimit::Bool,
        si::CC.StmtInfo,
        sv::CC.AbsIntState,
    )
end

# Julia 1.12 hands inference results around as `Future`s, and reordered the
# fields of `MethodCallResult`.
@static if isdefined(CC, :Future)
    deferred(x::T) where {T} = CC.Future{T}(x)
    function leaf_call_result(@nospecialize(rt))
        return CC.MethodCallResult(rt, Any, CC.Effects(), nothing, false, false)
    end
else
    deferred(x) = x
    function leaf_call_result(@nospecialize(rt))
        return CC.MethodCallResult(rt, Any, false, false, nothing, CC.Effects())
    end
end

# The matched method, which may be an overlay: `_return_type` on the signature
# dispatches natively, and would type `Random.default_rng()` as a `TaskLocalRNG`
# that inference then folds into a constant.
function leaf_return_type(
    method::Method, @nospecialize(sig), sparams::Core.SimpleVector, world::UInt
)
    rt = CC.typeinf_type(Reactant.ReactantInterpreter(; world), method, sig, sparams)
    return rt === nothing ? Any : rt
end

function CC.abstract_call_known(
    interp::Interpreter,
    @nospecialize(f),
    arginfo::CC.ArgInfo,
    si::CC.StmtInfo,
    sv::CC.InferenceState,
    max_methods::Int=CC.get_max_methods(interp, f, sv),
)
    if f === ReactantCore.within_compile && length(arginfo.argtypes) == 1
        return deferred(
            CC.CallMeta(Core.Const(true), Union{}, CC.EFFECTS_TOTAL, CC.MethodResultPure())
        )
    end
    return @invoke CC.abstract_call_known(
        interp::CC.AbstractInterpreter,
        f::Any,
        arginfo::CC.ArgInfo,
        si::CC.StmtInfo,
        sv::CC.InferenceState,
        max_methods::Int,
    )
end
