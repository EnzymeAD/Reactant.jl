const enzyme_out = 0
const enzyme_dup = 1
const enzyme_const = 2
const enzyme_dupnoneed = 3
const enzyme_outnoneed = 4
const enzyme_constnoneed = 5

@inline function Enzyme.make_zero(x::RNumber)
    return zero(Core.Typeof(x))
end

@inline function Enzyme.make_zero(x::RArray{FT,N})::RArray{FT,N} where {FT<:AbstractFloat,N}
    return Base.zero(x)
end

@inline function Enzyme.make_zero(
    x::RArray{Complex{FT},N}
)::RArray{Complex{FT},N} where {FT<:AbstractFloat,N}
    return Base.zero(x)
end

macro register_make_zero_inplace(sym)
    quote
        @inline function $sym(prev::RArray{T,N})::Nothing where {T<:AbstractFloat,N}
            $sym(prev, nothing)
            return nothing
        end

        @inline function $sym(prev::RArray{T,N}, seen::ST)::Nothing where {T,N,ST}
            if Enzyme.Compiler.guaranteed_const(T)
                return nothing
            end
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            fill!(prev, zero(T))
            return nothing
        end
    end
end

@register_make_zero_inplace(Enzyme.make_zero!)
@register_make_zero_inplace(Enzyme.remake_zero!)

function Enzyme.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
)::RT where {copy_if_inactive,RT<:Union{RArray,RNumber}}
    if haskey(seen, prev)
        return seen[prev]
    end
    if Enzyme.Compiler.guaranteed_const(eltype(RT))
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    res = zero(prev)
    seen[prev] = res
    return res
end

function Enzyme.onehot(x::TracedRArray{T,N}) where {T,N}
    onehot_matrix = promote_to(TracedRArray{T,2}, LinearAlgebra.I(length(x)))
    return Tuple(
        materialize_traced_array(reshape(y, size(x))) for y in eachcol(onehot_matrix)
    )
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
        TracedUtils.push_val!(ad_inputs, stack(x.dval; dims=1), path)
    end
end

function push_acts!(ad_inputs, x::BatchDuplicatedNoNeed, path, reverse)
    TracedUtils.push_val!(ad_inputs, x.val, path)
    if !reverse
        TracedUtils.push_val!(ad_inputs, stack(tval; dims=1), path)
    end
end

function set_act!(inp, path, reverse, tostore; emptypath=false, width=1)
    x = if inp isa Enzyme.Active
        inp.val
    else
        inp.dval
    end

    for p in path
        x = traced_getfield(x, p)
    end

    if !reverse || width == 1
        TracedUtils.set_mlir_data!(x, tostore)
    else
        tostore_traced = TracedRArray(tostore)
        @assert length(x) == size(tostore_traced, ndims(tostore_traced))
        for (i, sl) in enumerate(eachslice(tostore_traced; dims=ndims(tostore_traced)))
            TracedUtils.set_mlir_data!(x[i], TracedUtils.get_mlir_data(sl))
        end
    end

    emptypath && TracedUtils.set_paths!(x, ())
    return nothing
end

function act_attr(val)
    val = @ccall MLIR.API.mlir_c.enzymeActivityAttrGet(
        MLIR.IR.context()::MLIR.API.MlirContext, val::Int32
    )::MLIR.API.MlirAttribute
    return MLIR.IR.Attribute(val)
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
        arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]
        push!(activity, act_from_type(arg, reverse))
        push_acts!(ad_inputs, arg, path[3:end], reverse)
    end

    outtys = MLIR.IR.Type[]
    ret_activity = Int32[]

    for a in linear_results
        if TracedUtils.has_idx(a, resprefix)
            if Enzyme.needs_primal(CMode)
                push!(
                    outtys,
                    TracedUtils.transpose_ty(MLIR.IR.type(TracedUtils.get_mlir_data(a))),
                )
            end

            if CMode <: Enzyme.ForwardMode && !(A <: Enzyme.Const)
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

            act = act_from_type(A, reverse, Enzyme.needs_primal(CMode))
            push!(ret_activity, act)
            if act == enzyme_out || act == enzyme_outnoneed
                if width == 1 || CMode <: Enzyme.ForwardMode
                    cst = @opcall fill(one(unwrapped_eltype(a)), size(a))
                else
                    cst = @opcall fill(one(unwrapped_eltype(a)), (width, size(a)...))
                end
                push!(ad_inputs, cst.mlir_data)
            end
        else
            if TracedUtils.has_idx(a, argprefix)
                idx, path = TracedUtils.get_argidx(a, argprefix)
                arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]
                act = act_from_type(arg, reverse, true)
                push!(ret_activity, act)

                if act == enzyme_out || act == enzyme_outnoneed
                    if width == 1 || CMode <: Enzyme.ForwardMode
                        TracedUtils.push_val!(ad_inputs, arg.dval, path[3:end])
                    else
                        TracedUtils.push_val!(
                            ad_inputs, stack(arg.dval; dims=1), path[3:end]
                        )
                    end
                end
            else
                act = act_from_type(Enzyme.Const, reverse, true)
                push!(ret_activity, act)
            end

            push!(
                outtys, TracedUtils.transpose_ty(MLIR.IR.type(TracedUtils.get_mlir_data(a)))
            )
        end
    end

    for (i, act) in enumerate(activity)
        if act == enzyme_out || act == enzyme_dup || act == enzyme_dupnoneed
            push!(outtys, TracedUtils.batch_ty(width, in_tys[i]))
        end
    end

    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))
    res = (reverse ? MLIR.Dialects.enzyme.autodiff : MLIR.Dialects.enzyme.fwddiff)(
        [TracedUtils.transpose_val(v; keep_first_intact=width > 1) for v in ad_inputs];
        outputs=outtys,
        fn=fname,
        width,
        strong_zero=Enzyme.strong_zero(CMode),
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
            if Enzyme.needs_primal(CMode)
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
                        sval = @opcall slice(TracedRArray(tval), starts, limits)
                        sval = @opcall reshape(sval, collect(Int64, sz))
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
                TracedUtils.transpose_val(MLIR.IR.result(res, residx));
                width,
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
                    width,
                    emptypath=true,
                ) #=reverse=#
                residx += 1
                continue
            end
            set_act!(
                args[idx],
                path[3:end],
                reverse,
                TracedUtils.transpose_val(MLIR.IR.result(res, residx));
                width,
            )
        end
        residx += 1
    end

    func2.operation = MLIR.API.MlirOperation(C_NULL)

    if reverse
        resv = if Enzyme.needs_primal(CMode)
            result
        else
            nothing
        end
        return ((restup...,), resv)
    else
        if Enzyme.needs_primal(CMode)
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

const ignore_derivatives = EnzymeCore.ignore_derivatives
