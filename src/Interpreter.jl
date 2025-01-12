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

Base.Experimental.@MethodTable(REACTANT_METHOD_TABLE)

function var"@reactant_overlay"(__source__::LineNumberNode, __module__::Module, def)
    return Base.Experimental.var"@overlay"(
        __source__, __module__, :(Reactant.REACTANT_METHOD_TABLE), def
    )
end

function set_reactant_abi(
    interp,
    @nospecialize(f),
    arginfo::ArgInfo,
    si::StmtInfo,
    sv::AbsIntState,
    max_methods::Int=get_max_methods(interp, f, sv),
)
    (; fargs, argtypes) = arginfo

    if f === ReactantCore.within_compile
        if length(argtypes) != 1
            @static if VERSION < v"1.11.0-"
                return CallMeta(Union{}, Effects(), NoCallInfo())
            else
                return CallMeta(Union{}, Union{}, Effects(), NoCallInfo())
            end
        end
        @static if VERSION < v"1.11.0-"
            return CallMeta(
                Core.Const(true), Core.Compiler.EFFECTS_TOTAL, MethodResultPure()
            )
        else
            return CallMeta(
                Core.Const(true), Union{}, Core.Compiler.EFFECTS_TOTAL, MethodResultPure()
            )
        end
    end

    # Improve inference by considering call_with_reactant as having the same results as
    # the original call
    if f === Reactant.call_with_reactant
        arginfo2 = ArgInfo(fargs isa Nothing ? nothing : fargs[2:end], argtypes[2:end])
        return abstract_call(interp, arginfo2::ArgInfo, si, sv, max_methods)
    end

    return Base.@invoke abstract_call_known(
        interp::AbstractInterpreter,
        f::Any,
        arginfo::ArgInfo,
        si::StmtInfo,
        sv::AbsIntState,
        max_methods::Int,
    )
end

@static if Enzyme.GPUCompiler.HAS_INTEGRATED_CACHE
    struct ReactantCacheToken end

    function ReactantInterpreter(; world::UInt=Base.get_world_counter())
        return Enzyme.Compiler.Interpreter.EnzymeInterpreter(
            ReactantCacheToken(),
            REACTANT_METHOD_TABLE,
            world,
            false,            #=forward_rules=#
            false,            #=reverse_rules=#
            false,            #=inactive_rules=#
            false,            #=broadcast_rewrite=#
            set_reactant_abi,
        )
    end
else
    const REACTANT_CACHE = Enzyme.GPUCompiler.CodeCache()

    function ReactantInterpreter(;
        world::UInt=Base.get_world_counter(), code_cache=REACTANT_CACHE
    )
        return Enzyme.Compiler.Interpreter.EnzymeInterpreter(
            REACTANT_CACHE,
            REACTANT_METHOD_TABLE,
            world,
            false,            #=forward_rules=#
            false,            #=reverse_rules=#
            false,            #=inactive_rules=#
            false,            #=broadcast_rewrite=#
            set_reactant_abi,
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
@inline act_from_type(::Enzyme.BatchDuplicated, reverse, needs_primal=true) =
    act_from_type(Enzyme.Duplicated, reverse, needs_primal)
@inline act_from_type(::Enzyme.BatchDuplicatedNoNeed, reverse, needs_primal=true) =
    reverse ? enzyme_out : enzyme_dupnoneed
@inline act_from_type(::Enzyme.Active, reverse, needs_primal=true) =
    act_from_type(Enzyme.Active, reverse, needs_primal)
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

@inline act_from_type(::Type{<:Enzyme.BatchDuplicated}, reverse, needs_primal) =
    act_from_type(Enzyme.Duplicated, reverse, needs_primal)
@inline act_from_type(::Type{<:Enzyme.BatchDuplicatedNoNeed}, reverse, needs_primal) =
    act_from_type(Enzyme.DuplicatedNoNeed, Reverse, needs_primal)

@inline act_from_type(::Type{<:Enzyme.Active}, reverse, needs_primal) =
    if needs_primal
        enzyme_out
    else
        enzyme_outnoneed
    end

function push_acts!(ad_inputs, x::Const, path, reverse)
    return TracedUtils.push_val!(ad_inputs, x.val, path)
end

function push_acts!(ad_inputs, x::Active, path, reverse)
    return TracedUtils.push_val!(ad_inputs, x.val, path)
end

function push_acts!(ad_inputs, x::Duplicated, path, reverse)
    TracedUtils.push_val!(ad_inputs, x.val, path)
    if !reverse
        TracedUtils.push_val!(ad_inputs, x.dval, path)
    end
end

function push_acts!(ad_inputs, x::DuplicatedNoNeed, path, reverse)
    TracedUtils.push_val!(ad_inputs, x.val, path)
    if !reverse
        TracedUtils.push_val!(ad_inputs, x.dval, path)
    end
end

function push_acts!(ad_inputs, x::BatchDuplicated, path, reverse)
    TracedUtils.push_val!(ad_inputs, x.val, path)
    if !reverse
        ET = unwrapped_eltype(x.val)
        predims = size(x.val)
        cval = MLIR.IR.result(
            MLIR.Dialects.stablehlo.concatenate(
                [
                    TracedUtils.get_mlir_data(Ops.reshape(v, Int64[1, predims...])) for
                    v in x.dval
                ];
                dimension=Int64(0),
            ),
        )
        tval = TracedRArray{ET,length(predims) + 1}((), cval, (length(x.dval), predims...))
        TracedUtils.push_val!(ad_inputs, tval, path)
    end
end

function push_acts!(ad_inputs, x::BatchDuplicatedNoNeed, path, reverse)
    TracedUtils.push_val!(ad_inputs, x.val, path)
    if !reverse
        ET = unwrapped_eltype(x.val)
        predims = size(x.val)
        cval = MLIR.IR.result(
            MLIR.Dialects.stablehlo.concatenate(
                [Ops.reshape(v, Int64[1, predims...]) for v in x.dval]; dimension=Int64(0)
            ),
        )
        tval = TracedRArray{ET,length(predims) + 1}((), cval, (length(x.dval), predims...))
        TracedUtils.push_val!(ad_inputs, tval, path)
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
    TracedUtils.set_mlir_data!(x, tostore)
    #else
    #    x.mlir_data = MLIR.IR.result(MLIR.Dialects.stablehlo.add(x.mlir_data, tostore), 1)
    #end

    emptypath && TracedUtils.set_paths!(x, ())
    return nothing
end

function overload_autodiff(
    ::CMode, f::FA, ::Type{A}, args::Vararg{Enzyme.Annotation,Nargs}
) where {CMode<:Enzyme.Mode,FA<:Enzyme.Annotation,A<:Enzyme.Annotation,Nargs}
    reverse = CMode <: Enzyme.ReverseMode

    width = Enzyme.same_or_one(1, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end

    primf = f.val
    primargs = ((v.val for v in args)...,)

    argprefix::Symbol = gensym("autodiffarg")
    resprefix::Symbol = gensym("autodiffresult")
    resargprefix::Symbol = gensym("autodiffresarg")

    mlir_fn_res = TracedUtils.make_mlir_fn(
        primf,
        primargs,
        (),
        string(f) * "_autodiff",
        false;
        argprefix,
        resprefix,
        resargprefix,
    )
    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    activity = Int32[]
    ad_inputs = MLIR.IR.Value[]

    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
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
        if TracedUtils.has_idx(a, resprefix)
            if needs_primal(CMode)
                push!(
                    outtys,
                    TracedUtils.transpose_ty(MLIR.IR.type(TracedUtils.get_mlir_data(a))),
                )
            end
            if CMode <: Enzyme.ForwardMode && !(A <: Enzyme.Const)
                if width == 1
                    push!(
                        outtys,
                        TracedUtils.transpose_ty(
                            MLIR.IR.type(TracedUtils.get_mlir_data(a))
                        ),
                    )
                else
                    push!(
                        outtys,
                        TracedUtils.batch_ty(
                            width,
                            TracedUtils.transpose_ty(
                                MLIR.IR.type(TracedUtils.get_mlir_data(a))
                            ),
                        ),
                    )
                end
            end
        else
            push!(
                outtys, TracedUtils.transpose_ty(MLIR.IR.type(TracedUtils.get_mlir_data(a)))
            )
        end
    end
    for (i, act) in enumerate(activity)
        if act == enzyme_out || act == enzyme_dup || act == enzyme_dupnoneed
            if width == 1
                push!(outtys, in_tys[i])
            else
                push!(outtys, TracedUtils.batch_ty(width, in_tys[i]))
            end
        end
    end

    ret_activity = Int32[]
    for a in linear_results
        if TracedUtils.has_idx(a, resprefix)
            act = act_from_type(A, reverse, needs_primal(CMode))
            push!(ret_activity, act)
            if act == enzyme_out || act == enzyme_outnoneed
                attr = MLIR.IR.DenseElementsAttribute(
                    fill(one(unwrapped_eltype(a)), size(a))
                )
                cst = MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
                push!(ad_inputs, cst)
            end
        elseif TracedUtils.has_idx(a, argprefix)
            idx, path = TracedUtils.get_argidx(a, argprefix)
            if idx == 1 && fnwrap
                act = act_from_type(f, reverse, true)
                push!(ret_activity, act)
                if act != enzyme_out && act != enzyme_outnoneed
                    continue
                end
                TracedUtils.push_val!(ad_inputs, f.dval, path[3:end])
            else
                if fnwrap
                    idx -= 1
                end
                act = act_from_type(args[idx], reverse, true)
                push!(ret_activity, act)
                if act != enzyme_out && act != enzyme_outnoneed
                    continue
                end
                TracedUtils.push_val!(ad_inputs, args[idx].dval, path[3:end])
            end
        else
            act = act_from_type(Enzyme.Const, reverse, true)
            push!(ret_activity, act)
            if act != enzyme_out && act != enzyme_outnoneed
                continue
            end
        end
    end

    function act_attr(val)
        val = @ccall MLIR.API.mlir_c.enzymeActivityAttrGet(
            MLIR.IR.context()::MLIR.API.MlirContext, val::Int32
        )::MLIR.API.MlirAttribute
        return MLIR.IR.Attribute(val)
    end
    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))
    res = (reverse ? MLIR.Dialects.enzyme.autodiff : MLIR.Dialects.enzyme.fwddiff)(
        [TracedUtils.transpose_val(v) for v in ad_inputs];
        outputs=outtys,
        fn=fname,
        width,
        activity=MLIR.IR.Attribute([act_attr(a) for a in activity]),
        ret_activity=MLIR.IR.Attribute([act_attr(a) for a in ret_activity]),
    )

    residx = 1

    dresult = if CMode <: Enzyme.ForwardMode && !(A <: Enzyme.Const)
        if width == 1
            deepcopy(result)
        else
            ntuple(Val(width)) do i
                Base.@_inline_meta
                deepcopy(result)
            end
        end
    else
        nothing
    end

    for a in linear_results
        if TracedUtils.has_idx(a, resprefix)
            if needs_primal(CMode)
                path = TracedUtils.get_idx(a, resprefix)
                tval = TracedUtils.transpose_val(MLIR.IR.result(res, residx))
                TracedUtils.set!(result, path[2:end], tval)
                residx += 1
            end
            if CMode <: Enzyme.ForwardMode && !(A <: Enzyme.Const)
                path = TracedUtils.get_idx(a, resprefix)
                if width == 1
                    tval = TracedUtils.transpose_val(MLIR.IR.result(res, residx))
                    TracedUtils.set!(dresult, path[2:end], tval)
                else
                    tval = TracedUtils.transpose_val(MLIR.IR.result(res, residx))
                    for i in 1:width
                        sz = size(a)
                        starts = Int64[i]
                        limits = Int64[i]
                        for v in sz
                            push!(starts, 0)
                            push!(limits, v)
                        end
                        sval = Ops.slice(TracedRArray(tval), starts, limits)
                        TracedUtils.set!(
                            dresult[i], path[2:end], TracedUtils.get_mlir_data(sval)
                        )
                    end
                end
                residx += 1
            end
        elseif TracedUtils.has_idx(a, argprefix)
            idx, path = TracedUtils.get_argidx(a, argprefix)
            if idx == 1 && fnwrap
                TracedUtils.set!(
                    f.val,
                    path[3:end],
                    TracedUtils.transpose_val(MLIR.IR.result(res, residx)),
                )
                residx += 1
            else
                if fnwrap
                    idx -= 1
                end
                TracedUtils.set!(
                    args[idx].val,
                    path[3:end],
                    TracedUtils.transpose_val(MLIR.IR.result(res, residx)),
                )
                residx += 1
            end
        else
            TracedUtils.set!(a, (), TracedUtils.transpose_val(MLIR.IR.result(res, residx)))
            residx += 1
        end
    end

    restup = Any[(a isa Active) ? copy(a) : nothing for a in args]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 1 && fnwrap
            if act_from_type(f, reverse) != enzyme_out
                continue
            end
            if f isa Enzyme.Active
                @assert false
                residx += 1
                continue
            end
            set_act!(
                f,
                path[3:end],
                reverse,
                TracedUtils.transpose_val(MLIR.IR.result(res, residx)),
            )
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
                    TracedUtils.transpose_val(MLIR.IR.result(res, residx));
                    emptypaths=true,
                ) #=reverse=#
                residx += 1
                continue
            end
            set_act!(
                args[idx],
                path[3:end],
                reverse,
                TracedUtils.transpose_val(MLIR.IR.result(res, residx)),
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
        if needs_primal(CMode)
            if CMode <: Enzyme.ForwardMode && !(A <: Enzyme.Const)
                (dresult, result)
            else
                (result,)
            end
        else
            if CMode <: Enzyme.ForwardMode && !(A <: Enzyme.Const)
                (dresult,)
            else
                ()
            end
        end
    end
end
