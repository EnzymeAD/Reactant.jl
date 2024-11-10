# Taken from https://github.com/JuliaLang/julia/pull/52964/files#diff-936d33e524bcd097015043bd6410824119be5c210d43185c4d19634eb4912708
# Other references:
# - https://github.com/JuliaLang/julia/blob/0fd1f04dc7d4b905b0172b7130e9b1beab9bc4c9/test/compiler/AbstractInterpreter.jl#L228-L234
# - https://github.com/JuliaLang/julia/blob/v1.10.4/test/compiler/newinterp.jl#L9

const CC = Core.Compiler
using Enzyme

import Core.Compiler:
    AbstractInterpreter,
    abstract_call,
    abstract_call_known,
    ArgInfo,
    StmtInfo,
    AbsIntState,
    get_max_methods,
    CallMeta,
    Effects,
    NoCallInfo,
    widenconst,
    mapany,
    MethodResultPure

function set_reactant_abi(
    interp,
    @nospecialize(f),
    arginfo::ArgInfo,
    si::StmtInfo,
    sv::AbsIntState,
    max_methods::Int = get_max_methods(interp, f, sv),
)

    (; fargs, argtypes) = arginfo

    if ((f === Enzyme.autodiff) || (f === Enzyme.autodiff_deferred) || (f === Enzyme.gradient) || (f === Enzyme.jacobian)) && (length(argtypes) >= 2)
        if widenconst(argtypes[2]) <: Enzyme.Mode
            newmode = Enzyme.set_abi(widenconst(argtypes[2]), ReactantABI)
            if newmode != widenconst(argtypes[2])
                newmodev = newmode()
                arginfo2 = ArgInfo(
                    fargs isa Nothing ? nothing :
                    [fargs[1], :($(newmodev)), fargs[3:end]...],
                    [argtypes[1], Core.Const(newmodev), argtypes[3:end]...],
                )
                return abstract_call_known(
                    interp,
                    f,
                    arginfo2,
                    si,
                    sv,
                    max_methods,
                )
            end
        end
    end
    return Base.@invoke abstract_call_known(
        interp::AbstractInterpreter,
        f,
        arginfo::ArgInfo,
        si::StmtInfo,
        sv::AbsIntState,
        max_methods::Int,
    )
end

function set_reactant_abi end

@static if Enzyme.GPUCompiler.HAS_INTEGRATED_CACHE
    struct ReactantCacheToken
    end

    function ReactantInterpreter(;
        world::UInt=Base.get_world_counter(),
    )
        return Enzyme.Compiler.Interpreter.EnzymeInterpreter(
            ReactantCacheToken(),
            #=mt=#nothing,
            world,
            #=forward_rules=#true,
            #=reverse_rules=#true,
            #=deferred_lower=#true,
            set_reactant_abi
        )
    end
else
    const REACTANT_CACHE = Enzyme.GPUCompiler.CodeCache()

    function ReactantInterpreter(;
        world::UInt=Base.get_world_counter(),
        code_cache=REACTANT_CACHE
    )
        return Enzyme.Compiler.Interpreter.EnzymeInterpreter(
            REACTANT_CACHE,
            #=mt=#nothing,
            world,
            #=forward_rules=#true,
            #=forward_rules=#true,
            #=deferred_lower=#true,
            set_reactant_abi
        )
    end
end


const enzyme_out = 0
const enzyme_dup = 1
const enzyme_const = 2
const enzyme_dupnoneed = 3
const enzyme_outnoneed = 4
const enzyme_constnoneed = 5

@inline act_from_type(x, reverse, needs_primal=true) =
    throw(AssertionError("Unhandled activity $(typeof(x))"))
@inline act_from_type(::Enzyme.Const, reverse, needs_primal=true) =
    act_from_type(Enzyme.Const, reverse, needs_primal)
@inline act_from_type(::Enzyme.Duplicated, reverse, needs_primal=true) =
    act_from_type(Enzyme.Duplicated, reverse, needs_primal)
@inline act_from_type(::Enzyme.DuplicatedNoNeed, reverse, needs_primal=true) =
    reverse ? enzyme_out : enzyme_dupnoneed
@inline act_from_type(::Enzyme.Active, reverse, needs_primal=true) =
    act_from_tuple(Enzyme.Active, reverse, needs_primal)
@inline act_from_type(::Type{<:Enzyme.Const}, reverse, needs_primal) =
    if needs_primal
        enzyme_const
    else
        enzyme_constnoneed
    end
@inline act_from_type(::Type{<:Enzyme.Duplicated}, reverse, needs_primal) =
    if reverse
        if needs_primal
            enzyme_out
        else
            enzyme_outnoneed
        end
    else
        if needs_primal
            enzyme_dup
        else
            enzyme_dupnoneed
        end
    end
@inline act_from_type(::Type{<:Enzyme.Active}, reverse, needs_primal) =
    if needs_primal
        enzyme_out
    else
        enzyme_outnoneed
    end

function push_val!(ad_inputs, x, path)
    for p in path
        x = traced_getfield(x, p)
    end
    x = x.mlir_data
    return push!(ad_inputs, x)
end

function push_acts!(ad_inputs, x::Const, path, reverse)
    return push_val!(ad_inputs, x.val, path)
end

function push_acts!(ad_inputs, x::Active, path, reverse)
    return push_val!(ad_inputs, x.val, path)
end

function push_acts!(ad_inputs, x::Duplicated, path, reverse)
    push_val!(ad_inputs, x.val, path)
    if !reverse
        push_val!(ad_inputs, x.dval, path)
    end
end

function push_acts!(ad_inputs, x::DuplicatedNoNeed, path, reverse)
    push_val!(ad_inputs, x.val, path)
    if !reverse
        push_val!(ad_inputs, x.dval, path)
    end
end

function set_act!(inp, path, reverse, tostore; emptypath=false)
    x = if inp isa Enzyme.Active
        inp.val
    else
        inp.dval
    end

    for p in path
        x = traced_getfield(x, p)
    end

    #if inp isa Enzyme.Active || !reverse
    x.mlir_data = tostore
    #else
    #    x.mlir_data = MLIR.IR.result(MLIR.Dialects.stablehlo.add(x.mlir_data, tostore), 1)
    #end

    if emptypath
        x.paths = ()
    end
end

function set!(x, path, tostore; emptypath=false)
    for p in path
        x = traced_getfield(x, p)
    end

    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

function get_argidx(x)
    for path in x.paths
        if length(path) == 0
            continue
        end
        if path[1] == :args
            return path[2]::Int, path
        end
    end
    throw(AssertionError("No path found for $x"))
end
function get_residx(x)
    for path in x.paths
        if length(path) == 0
            continue
        end
        if path[1] == :result
            return path
        end
    end
    throw(AssertionError("No path found $x"))
end

function has_residx(x)
    for path in x.paths
        if length(path) == 0
            continue
        end
        if path[1] == :result
            return true
        end
    end
    return false
end

function get_attribute_by_name(operation, name)
    return MLIR.IR.Attribute(MLIR.API.mlirOperationGetAttributeByName(operation, name))
end

function overload_autodiff(
    ::CMode, f::FA, ::Type{A}, args::Vararg{Enzyme.Annotation,Nargs}
) where {CMode<:Enzyme.Mode,FA<:Enzyme.Annotation,A<:Enzyme.Annotation,Nargs}
    reverse = CMode <: Enzyme.ReverseMode

    primf = f.val
    primargs = ((v.val for v in args)...,)

    fnwrap, func2, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results = make_mlir_fn(
        primf, primargs, (), string(f) * "_autodiff", false
    )

    activity = Int32[]
    ad_inputs = MLIR.IR.Value[]

    for a in linear_args
        idx, path = get_argidx(a)
        if idx == 1 && fnwrap
            push!(activity, act_from_type(f, reverse))
            push_acts!(ad_inputs, f, path[3:end], reverse)
        else
            if fnwrap
                idx -= 1
            end
            push!(activity, act_from_type(args[idx], reverse))
            push_acts!(ad_inputs, args[idx], path[3:end], reverse)
        end
    end

    outtys = MLIR.IR.Type[]
    @inline needs_primal(::Type{<:Enzyme.ReverseMode{ReturnPrimal}}) where {ReturnPrimal} =
        ReturnPrimal
    @inline needs_primal(::Type{<:Enzyme.ForwardMode{ReturnPrimal}}) where {ReturnPrimal} =
        ReturnPrimal
    for a in linear_results
        if has_residx(a)
            if needs_primal(CMode)
                push!(outtys, transpose_ty(MLIR.IR.type(a.mlir_data)))
            end
        else
            push!(outtys, transpose_ty(MLIR.IR.type(a.mlir_data)))
        end
    end
    for (i, act) in enumerate(activity)
        if act == enzyme_out || (reverse && (act == enzyme_dup || act == enzyme_dupnoneed))
            push!(outtys, in_tys[i])# transpose_ty(MLIR.IR.type(MLIR.IR.operand(ret, i))))
        end
    end

    ret_activity = Int32[]
    for a in linear_results
        if has_residx(a)
            act = act_from_type(A, reverse, needs_primal(CMode))
            push!(ret_activity, act)
            if act == enzyme_out || act == enzyme_outnoneed
                attr = fill(MLIR.IR.Attribute(eltype(a)(1)), mlir_type(a))
                cst = MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
                push!(ad_inputs, cst)
            end
        else
            idx, path = get_argidx(a)
            if idx == 1 && fnwrap
                act = act_from_type(f, reverse, true)
                push!(ret_activity, act)
                if act != enzyme_out && act != enzyme_outnoneed
                    continue
                end
                push_val!(ad_inputs, f.dval, path[3:end])
            else
                if fnwrap
                    idx -= 1
                end
                act = act_from_type(args[idx], reverse, true)
                push!(ret_activity, act)
                if act != enzyme_out && act != enzyme_outnoneed
                    continue
                end
                push_val!(ad_inputs, args[idx].dval, path[3:end])
            end
        end
    end

    function act_attr(val)
        val = @ccall MLIR.API.mlir_c.enzymeActivityAttrGet(
            MLIR.IR.context()::MLIR.API.MlirContext, val::Int32
        )::MLIR.API.MlirAttribute
        return MLIR.IR.Attribute(val)
    end
    fname = get_attribute_by_name(func2, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))
    res = (reverse ? MLIR.Dialects.enzyme.autodiff : MLIR.Dialects.enzyme.fwddiff)(
        [transpose_val(v) for v in ad_inputs];
        outputs=outtys,
        fn=fname,
        activity=MLIR.IR.Attribute([act_attr(a) for a in activity]),
        ret_activity=MLIR.IR.Attribute([act_attr(a) for a in ret_activity]),
    )

    residx = 1

    for a in linear_results
        if has_residx(a)
            if needs_primal(CMode)
                path = get_residx(a)
                set!(result, path[2:end], transpose_val(MLIR.IR.result(res, residx)))
                residx += 1
            end
        else
            idx, path = get_argidx(a)
            if idx == 1 && fnwrap
                set!(f.val, path[3:end], transpose_val(MLIR.IR.result(res, residx)))
                residx += 1
            else
                if fnwrap
                    idx -= 1
                end
                set!(args[idx].val, path[3:end], transpose_val(MLIR.IR.result(res, residx)))
                residx += 1
            end
        end
    end

    restup = Any[(a isa Active) ? copy(a) : nothing for a in args]
    for a in linear_args
        idx, path = get_argidx(a)
        if idx == 1 && fnwrap
            if act_from_type(f, reverse) != enzyme_out
                continue
            end
            if f isa Enzyme.Active
                @assert false
                residx += 1
                continue
            end
            set_act!(f, path[3:end], reverse, transpose_val(MLIR.IR.result(res, residx)))
        else
            if fnwrap
                idx -= 1
            end
            if act_from_type(args[idx], reverse) != enzyme_out
                continue
            end
            if args[idx] isa Enzyme.Active
                set_act!(
                    args[idx],
                    path[3:end],
                    false,
                    transpose_val(MLIR.IR.result(res, residx));
                    emptypaths=true,
                ) #=reverse=#
                residx += 1
                continue
            end
            set_act!(
                args[idx], path[3:end], reverse, transpose_val(MLIR.IR.result(res, residx))
            )
        end
        residx += 1
    end

    func2.operation = MLIR.API.MlirOperation(C_NULL)

    if reverse
        resv = if needs_primal(CMode)
            result
        else
            nothing
        end
        return ((restup...,), resv)
    else
        if A <: Const
            return result
        else
            dres = copy(result)
            throw(AssertionError("TODO implement forward mode handler"))
            if A <: Duplicated
                return ()
            end
        end
    end
end


@inline function Enzyme.autodiff_deferred(
    rmode::Enzyme.ReverseMode{ReturnPrimal,RuntimeActivity,ReactantABI,Holomorphic,ErrIfFuncWritten},
    f::FA,
    rt::Type{A},
    args::Vararg{Annotation,Nargs},
) where {
    FA<:Annotation,
    A<:Annotation,
    ReturnPrimal,
    RuntimeActivity,
    Holomorphic,
    Nargs,
    ErrIfFuncWritten,
}
    overload_autodiff(rmode, f, rt, args...)
end

@inline function Enzyme.autodiff_deferred(
    rmode::ForwardMode{ReturnPrimal,ReactantABI,ErrIfFuncWritten,RuntimeActivity},
    f::FA,
    rt::Type{A},
    args::Vararg{Annotation,Nargs},
) where {
    FA<:Annotation,
    A<:Annotation,
    ReturnPrimal,
    Nargs,
    ErrIfFuncWritten,
    RuntimeActivity
}
    overload_autodiff(rmode, f, rt, args...)
end
