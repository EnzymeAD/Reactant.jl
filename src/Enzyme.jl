const enzyme_out = 0
const enzyme_dup = 1
const enzyme_const = 2
const enzyme_dupnoneed = 3
const enzyme_outnoneed = 4
const enzyme_constnoneed = 5

function Enzyme.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
)::RT where {copy_if_inactive,RT<:Union{RArray,RNumber}}
    if haskey(seen, prev)
        return seen[prev]
    end
    if Enzyme.Compiler.guaranteed_const_nongen(eltype(RT), nothing)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    res = zero(prev)
    seen[prev] = res
    return res
end

# The default `onehot` will lead to scalar indexing
function Enzyme.onehot(x::TracedRArray{T,N}) where {T,N}
    # TODO: Ideally we do it as a scatter -> slice but we don't implement constant
    #       folding for scatter yet.
    results = Vector{TracedRArray{T,N}}(undef, length(x))
    pad_value = TracedUtils.promote_to(TracedRNumber{T}, 0)
    base_value = TracedUtils.broadcast_to_size(T(1), (1,))
    for i in eachindex(x)
        results[i] = Ops.reshape(
            Ops.pad(base_value, pad_value; low=Int64[i - 1], high=Int64[length(x) - i]),
            collect(Int64, size(x)),
        )
    end
    return Tuple(results)
end

function Enzyme.EnzymeRules.inactive_noinl(::typeof(XLA.buffer_on_cpu), args...)
    return nothing
end

function Enzyme.EnzymeRules.inactive_noinl(::typeof(XLA.addressable_devices), args...)
    return nothing
end

function Enzyme.EnzymeRules.noalias(
    ::typeof(Base.similar), a::ConcretePJRTArray, ::Type, args...
)
    return nothing
end

function Enzyme.EnzymeRules.noalias(
    ::typeof(Base.similar), a::ConcreteIFRTArray, ::Type, args...
)
    return nothing
end

function Enzyme.EnzymeRules.augmented_primal(
    config,
    ofn::Const{typeof(Base.similar)},
    ::Type{RT},
    uval::Enzyme.Annotation{<:ConcretePJRTArray},
    T::Enzyme.Const{<:Type},
    args...,
) where {RT}
    primargs = ntuple(Val(length(args))) do i
        Base.@_inline_meta
        args[i].val
    end

    primal = if EnzymeRules.needs_primal(config)
        ofn.val(uval.val, T.val, primargs...)
    else
        nothing
    end

    shadow = if EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            ConcretePJRTArray(
                zeros(T.val, primargs...);
                client=XLA.client(uval.val),
                device=XLA.device(uval.val),
                uval.val.sharding,
            )
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                ConcretePJRTArray(
                    zeros(T.val, primargs...);
                    client=XLA.client(uval.val),
                    device=XLA.device(uval.val),
                    uval.val.sharding,
                )
            end
        end
    else
        nothing
    end

    return EnzymeRules.AugmentedReturn{
        EnzymeRules.primal_type(config, RT),EnzymeRules.shadow_type(config, RT),Nothing
    }(
        primal, shadow, nothing
    )
end

function Enzyme.EnzymeRules.reverse(
    config,
    ofn::Const{typeof(Base.similar)},
    ::Type{RT},
    tape,
    uval::Enzyme.Annotation{<:ConcretePJRTArray},
    T::Enzyme.Const{<:Type},
    args::Vararg{Enzyme.Annotation,N},
) where {RT,N}
    ntuple(Val(N + 2)) do i
        Base.@_inline_meta
        nothing
    end
end

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
        [TracedUtils.transpose_val(v; keep_first_intact=width > 1) for v in ad_inputs];
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
                        sval = Ops.reshape(sval, collect(Int64, sz))
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
