@enumx EnzymeActivity begin
    OUT = 0
    DUPLICATED = 1
    CONST = 2
    DUPLICATED_NO_NEED = 3
    OUT_NO_NEED = 4
    CONST_NO_NEED = 5
end

struct StackedBatchDuplicated{T,N,M,V<:AbstractArray{T,N},W<:AbstractArray{T,M}} <:
       Annotation{V}
    val::V
    dval::W

    function StackedBatchDuplicated(
        val::V, dval::W
    ) where {T,N,M,V<:AbstractArray{T,N},W<:AbstractArray{T,M}}
        @assert N == M - 1
        @assert size(val) == size(dval)[1:(end - 1)]
        return new{T,N,M,V,W}(val, dval)
    end
end

@inline function Enzyme.same_or_one_rec(current, arg::StackedBatchDuplicated, args...)
    return Enzyme.same_or_one_rec(
        Enzyme.same_or_one_helper(current, size(arg.dval, ndims(arg.dval))), args...
    )
end

@inline function Enzyme.same_or_one_rec(current, ::Type{<:StackedBatchDuplicated}, args...)
    throw(AssertionError("BatchDuplicatedNoNeed not yet supported"))
end

struct StackedBatchDuplicatedNoNeed{T,N,M,V<:AbstractArray{T,N},W<:AbstractArray{T,M}} <:
       Annotation{V}
    val::V
    dval::W

    function StackedBatchDuplicatedNoNeed(
        val::V, dval::W
    ) where {T,N,M,V<:AbstractArray{T,N},W<:AbstractArray{T,M}}
        @assert N == M - 1
        @assert size(val) == size(dval)[1:(end - 1)]
        return new{T,N,M,V,W}(val, dval)
    end
end

@inline function Enzyme.same_or_one_rec(current, arg::StackedBatchDuplicatedNoNeed, args...)
    return Enzyme.same_or_one_rec(
        Enzyme.same_or_one_helper(current, size(arg.dval, ndims(arg.dval))), args...
    )
end

@inline function Enzyme.same_or_one_rec(
    current, ::Type{<:StackedBatchDuplicatedNoNeed}, args...
)
    throw(AssertionError("BatchDuplicatedNoNeed not yet supported"))
end

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

function EnzymeRules.inactive_noinl(::typeof(XLA.buffer_on_cpu), args...)
    return nothing
end

function EnzymeRules.inactive_noinl(::typeof(XLA.addressable_devices), args...)
    return nothing
end

function EnzymeRules.noalias(::typeof(Base.similar), a::ConcretePJRTArray, ::Type, args...)
    return nothing
end

function EnzymeRules.noalias(::typeof(Base.similar), a::ConcreteIFRTArray, ::Type, args...)
    return nothing
end

function EnzymeRules.augmented_primal(
    config,
    ofn::Const{typeof(Base.similar)},
    ::Type{RT},
    uval::Annotation{<:ConcretePJRTArray},
    T::Const{<:Type},
    args...,
) where {RT}
    primargs = ntuple(Val(length(args))) do i
        Base.@_inline_meta
        args[i].val
    end

    primal = if EnzymeCore.needs_primal(config)
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

function EnzymeRules.reverse(
    config,
    ofn::Const{typeof(Base.similar)},
    ::Type{RT},
    tape,
    uval::Annotation{<:ConcretePJRTArray},
    T::Const{<:Type},
    args::Vararg{Annotation,N},
) where {RT,N}
    ntuple(Val(N + 2)) do i
        Base.@_inline_meta
        nothing
    end
end

@inline function act_from_type(::A, reverse, needs_primal=true) where {A<:Annotation}
    return act_from_type(A, reverse, needs_primal)
end

@inline function act_from_type(::Type{<:Active}, reverse, needs_primal)
    return needs_primal ? EnzymeActivity.OUT : EnzymeActivity.OUT_NO_NEED
end
@inline function act_from_type(::Type{<:Const}, reverse, needs_primal)
    return needs_primal ? EnzymeActivity.CONST : EnzymeActivity.CONST_NO_NEED
end

@inline function act_from_type(::Type{<:Duplicated}, reverse, needs_primal)
    if reverse
        return needs_primal ? EnzymeActivity.OUT : EnzymeActivity.OUT_NO_NEED
    else
        return needs_primal ? EnzymeActivity.DUPLICATED : EnzymeActivity.DUPLICATED_NO_NEED
    end
end
@inline function act_from_type(
    ::Type{<:Union{BatchDuplicated,StackedBatchDuplicated}}, reverse, needs_primal
)
    return act_from_type(Duplicated, reverse, needs_primal)
end

@inline function act_from_type(::Type{<:DuplicatedNoNeed}, reverse, needs_primal)
    return reverse ? EnzymeActivity.OUT : EnzymeActivity.DUPLICATED_NO_NEED
end
@inline function act_from_type(
    ::Type{<:Union{BatchDuplicatedNoNeed,StackedBatchDuplicatedNoNeed}},
    reverse,
    needs_primal,
)
    return act_from_type(DuplicatedNoNeed, reverse, needs_primal)
end

function push_acts!(ad_inputs, x::Union{Const,Active}, path, reverse)
    TracedUtils.push_val!(ad_inputs, x.val, path)
    return nothing
end

function push_acts!(ad_inputs, x::Union{Duplicated,DuplicatedNoNeed}, path, reverse)
    TracedUtils.push_val!(ad_inputs, x.val, path)
    if !reverse
        TracedUtils.push_val!(ad_inputs, x.dval, path)
    end
end

function push_acts!(
    ad_inputs, x::Union{BatchDuplicated,BatchDuplicatedNoNeed}, path, reverse
)
    TracedUtils.push_val!(ad_inputs, x.val, path)
    if !reverse
        TracedUtils.push_val!(ad_inputs, call_with_reactant(stack, x.dval), path)
    end
end

function push_acts!(
    ad_inputs, x::Union{StackedBatchDuplicated,StackedBatchDuplicatedNoNeed}, path, reverse
)
    TracedUtils.push_val!(ad_inputs, x.val, path)
    if !reverse
        TracedUtils.push_val!(ad_inputs, x.dval, path)
    end
end

function set_act!(inp, path, reverse, tostore; emptypath=false, width=1)
    x = if inp isa Active
        inp.val
    else
        inp.dval
    end

    for p in path
        x = traced_getfield(x, p)
    end

    if width == 1
        TracedUtils.set_mlir_data!(x, tostore)
    elseif x isa AbstractArray
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
    return MLIR.IR.Attribute(
        MLIR.API.enzymeActivityAttrGet(MLIR.IR.current_context(), Int32(val))
    )
end

function infer_activity(
    mode::CMode, ::FA, args::Vararg{Annotation,Nargs}
) where {CMode<:Mode,FA<:Annotation,Nargs}
    return Enzyme.guess_activity(
        call_with_native(
            primal_return_type,
            mode isa ForwardMode ? Enzyme.Forward : Enzyme.Reverse,
            eltype(FA),
            Enzyme.vaEltypeof(args...),
        ),
        mode,
    )
end

function overload_autodiff(
    mode::CMode, f::FA, args::Vararg{Annotation,Nargs}
) where {CMode<:Mode,FA<:Annotation,Nargs}
    return overload_autodiff(mode, f, infer_activity(mode, f, args...), args...)
end

function overload_autodiff(
    ::CMode, f::FA, ::Type{A}, args::Vararg{Annotation,Nargs}
) where {CMode<:Mode,FA<:Annotation,A<:Annotation,Nargs}
    reverse = CMode <: ReverseMode

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

    activity = EnzymeActivity.T[]
    ad_inputs = MLIR.IR.Value[]

    reverse_seeds = Dict{Tuple,MLIR.IR.Value}()

    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]
        push!(activity, act_from_type(arg, reverse))
        push_acts!(ad_inputs, arg, path[3:end], reverse)

        if CMode <: ReverseMode && act_from_type(arg, false) == EnzymeActivity.DUPLICATED
            x = if width == 1
                arg.dval
            elseif arg.dval isa AbstractArray
                arg.dval
            else
                call_with_reactant(stack, arg.dval)
            end
            for p in path[3:end]
                x = Compiler.traced_getfield(x, p)
            end
            x = TracedUtils.get_mlir_data(x)
            reverse_seeds[path] = x
        end
    end

    outtys = MLIR.IR.Type[]
    ret_activity = EnzymeActivity.T[]

    for a in linear_results
        if TracedUtils.has_idx(a, resprefix)
            if EnzymeCore.needs_primal(CMode)
                push!(
                    outtys,
                    TracedUtils.transpose_ty(MLIR.IR.type(TracedUtils.get_mlir_data(a))),
                )
            end

            if CMode <: ForwardMode && !(A <: Const)
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

            act = act_from_type(A, reverse, EnzymeCore.needs_primal(CMode))
            cst = nothing
            if act == EnzymeActivity.OUT || act == EnzymeActivity.OUT_NO_NEED
                if width == 1
                    cst = @opcall fill(one(unwrapped_eltype(a)), size(a))
                else
                    cst = @opcall fill(one(unwrapped_eltype(a)), (size(a)..., width))
                end
                cst = cst.mlir_data
            end

            if CMode <: ReverseMode && TracedUtils.has_idx(a, argprefix)
                idx, path = TracedUtils.get_argidx(a, argprefix)
                arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]
                if act_from_type(arg, false) == EnzymeActivity.DUPLICATED
                    seed = reverse_seeds[path]
                    if cst === nothing
                        if act == EnzymeActivity.CONST
                            act = EnzymeActivity.OUT
                        elseif act == EnzymeActivity.CONST_NO_NEED
                            act = EnzymeActivity.OUT_NO_NEED
                        else
                            @assert false
                        end
                        cst = seed
                    else
                        @assert (
                            act == EnzymeActivity.OUT || act == EnzymeActivity.OUT_NO_NEED
                        )
                        cst = MLIR.IR.result(MLIR.Dialects.stablehlo.add(cst, seed), 1)
                    end
                end
            end

            push!(ret_activity, act)
            if cst !== nothing
                push!(ad_inputs, cst)
            end
        else
            if TracedUtils.has_idx(a, argprefix)
                idx, path = TracedUtils.get_argidx(a, argprefix)
                arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]

                act = act_from_type(arg, reverse, true)
                push!(ret_activity, act)

                if act == EnzymeActivity.OUT || act == EnzymeActivity.OUT_NO_NEED
                    seed = reverse_seeds[path]
                    push!(ad_inputs, seed)
                end
            else
                act = act_from_type(Const, reverse, true)
                push!(ret_activity, act)
            end

            push!(
                outtys, TracedUtils.transpose_ty(MLIR.IR.type(TracedUtils.get_mlir_data(a)))
            )
        end
    end

    for (i, act) in enumerate(activity)
        if (
            act == EnzymeActivity.OUT ||
            act == EnzymeActivity.DUPLICATED ||
            act == EnzymeActivity.DUPLICATED_NO_NEED
        )
            push!(outtys, TracedUtils.batch_ty(width, in_tys[i]))
        end
    end

    fname = TracedUtils.get_attribute_by_name(mlir_fn_res.f, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))
    res = (reverse ? MLIR.Dialects.enzyme.autodiff : MLIR.Dialects.enzyme.fwddiff)(
        [TracedUtils.transpose_val(v) for v in ad_inputs];
        outputs=outtys,
        fn=fname,
        width,
        strong_zero=EnzymeCore.strong_zero(CMode),
        activity=MLIR.IR.Attribute([act_attr(a) for a in activity]),
        ret_activity=MLIR.IR.Attribute([act_attr(a) for a in ret_activity]),
    )

    residx = 1

    dresult = if CMode <: ForwardMode && !(A <: Const)
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
            if EnzymeCore.needs_primal(CMode)
                path = TracedUtils.get_idx(a, resprefix)
                tval = TracedUtils.transpose_val(MLIR.IR.result(res, residx))
                TracedUtils.set!(result, path[2:end], tval)
                residx += 1
            end
            if CMode <: ForwardMode && !(A <: Const)
                path = TracedUtils.get_idx(a, resprefix)
                tval = TracedUtils.transpose_val(MLIR.IR.result(res, residx))
                if width == 1
                    TracedUtils.set!(dresult, path[2:end], tval)
                else
                    ttval = TracedRArray(tval)
                    for (i, sl) in enumerate(eachslice(ttval; dims=ndims(ttval)))
                        TracedUtils.set!(
                            dresult[i],
                            path[2:end],
                            @allowscalar(TracedUtils.get_mlir_data(sl))
                        )
                    end
                end
                residx += 1
            end
        elseif TracedUtils.has_idx(a, argprefix)
            idx, path = TracedUtils.get_argidx(a, argprefix)
            arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]
            TracedUtils.set!(
                arg.val, path[3:end], TracedUtils.transpose_val(MLIR.IR.result(res, residx))
            )
            residx += 1
        else
            TracedUtils.set!(a, (), TracedUtils.transpose_val(MLIR.IR.result(res, residx)))
            residx += 1
        end
    end

    restup = Any[(a isa Active) ? copy(a) : nothing for a in args]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)

        arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]
        act_from_type(arg, reverse) != EnzymeActivity.OUT && continue

        if idx == 1 && fnwrap && arg isa Active
            @assert false
        end

        set_act!(
            arg,
            path[3:end],
            reverse,
            TracedUtils.transpose_val(MLIR.IR.result(res, residx));
            width,
            emptypath=arg isa Active,
        )
        residx += 1
    end

    if reverse
        if EnzymeCore.needs_primal(CMode)
            return ((restup...,), result)
        else
            return ((restup...,),)
        end
    else
        if EnzymeCore.needs_primal(CMode)
            if CMode <: ForwardMode && !(A <: Const)
                return (dresult, result)
            else
                return (result,)
            end
        else
            if CMode <: ForwardMode && !(A <: Const)
                return (dresult,)
            else
                return ()
            end
        end
    end
end

function lower_jacobian(
    mode::CMode, f::FA, args::Vararg{Annotation,Nargs}
) where {CMode<:Mode,FA<:Annotation,Nargs}
    return lower_jacobian(mode, f, infer_activity(mode, f, args...), args...)
end

function lower_jacobian(
    ::CMode, f::FA, ::Type{A}, args::Vararg{Annotation,Nargs}
) where {CMode<:Mode,FA<:Annotation,A<:Annotation,Nargs}
    reverse = CMode <: ReverseMode

    width = Enzyme.same_or_one(1, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end

    primf = f.val
    primargs = ((v.val for v in args)...,)

    argprefix::Symbol = gensym("jacobianarg")
    resprefix::Symbol = gensym("jacobianresult")
    resargprefix::Symbol = gensym("jacobianresarg")

    mlir_fn_res = TracedUtils.make_mlir_fn(
        primf,
        primargs,
        (),
        string(f) * "_jacobian",
        false;
        argprefix,
        resprefix,
        resargprefix,
    )
    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped

    activity = Int32[]
    ad_inputs = MLIR.IR.Value[]

    reverse_seeds = Dict{Tuple,MLIR.IR.Value}()

    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]
        push!(activity, act_from_type(arg, reverse))
        push_acts!(ad_inputs, arg, path[3:end], reverse)

        if CMode <: ReverseMode && act_from_type(arg, false) == enzyme_dup
            x = if width == 1
                arg.dval
            elseif arg.dval isa AbstractArray
                arg.dval
            else
                call_with_reactant(stack, arg.dval)
            end
            for p in path[3:end]
                x = Compiler.traced_getfield(x, p)
            end
            x = TracedUtils.get_mlir_data(x)
            reverse_seeds[path] = x
        end
    end

    outtys = MLIR.IR.Type[]
    ret_activity = Int32[]

    for a in linear_results
        if TracedUtils.has_idx(a, resprefix)
            if EnzymeCore.needs_primal(CMode)
                push!(
                    outtys,
                    TracedUtils.transpose_ty(MLIR.IR.type(TracedUtils.get_mlir_data(a))),
                )
            end

            if CMode <: ForwardMode && !(A <: Const)
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

            act = act_from_type(A, reverse, EnzymeCore.needs_primal(CMode))
            cst = nothing
            if act == enzyme_out || act == enzyme_outnoneed
                if width == 1
                    cst = @opcall fill(one(unwrapped_eltype(a)), size(a))
                else
                    cst = @opcall fill(one(unwrapped_eltype(a)), (size(a)..., width))
                end
                cst = cst.mlir_data
            end

            if CMode <: ReverseMode && TracedUtils.has_idx(a, argprefix)
                idx, path = TracedUtils.get_argidx(a, argprefix)
                arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]
                if act_from_type(arg, false) == enzyme_dup
                    seed = reverse_seeds[path]
                    if cst == nothing
                        if act == enzyme_const
                            act = enzyme_out
                        elseif act == enzyme_constnoneed
                            act = enzyme_outnoneed
                        else
                            @assert false
                        end
                        cst = seed
                    else
                        @assert act == enzyme_out || act == enzyme_outnoneed
                        cst = MLIR.IR.result(MLIR.Dialects.stablehlo.add(cst, seed), 1)
                    end
                end
            end

            push!(ret_activity, act)
            if cst != nothing
                push!(ad_inputs, cst)
            end
        else
            if TracedUtils.has_idx(a, argprefix)
                idx, path = TracedUtils.get_argidx(a, argprefix)
                arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]

                act = act_from_type(arg, reverse, true)
                push!(ret_activity, act)

                if act == enzyme_out || act == enzyme_outnoneed
                    seed = reverse_seeds[path]
                    push!(ad_inputs, seed)
                end
            else
                act = act_from_type(Const, reverse, true)
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

    fname = TracedUtils.get_attribute_by_name(mlir_fn_res.f, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))
    res = MLIR.Dialects.enzyme.jacobian(
        [TracedUtils.transpose_val(v) for v in ad_inputs];
        outputs=outtys,
        fn=fname,
        width,
        strong_zero=EnzymeCore.strong_zero(CMode),
        activity=MLIR.IR.Attribute([act_attr(a) for a in activity]),
        ret_activity=MLIR.IR.Attribute([act_attr(a) for a in ret_activity]),
    )

    residx = 1

    dresult = if CMode <: ForwardMode && !(A <: Const)
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
            if EnzymeCore.needs_primal(CMode)
                path = TracedUtils.get_idx(a, resprefix)
                tval = TracedUtils.transpose_val(MLIR.IR.result(res, residx))
                TracedUtils.set!(result, path[2:end], tval)
                residx += 1
            end
            if CMode <: ForwardMode && !(A <: Const)
                path = TracedUtils.get_idx(a, resprefix)
                tval = TracedUtils.transpose_val(MLIR.IR.result(res, residx))
                if width == 1
                    TracedUtils.set!(dresult, path[2:end], tval)
                else
                    ttval = TracedRArray(tval)
                    for (i, sl) in enumerate(eachslice(ttval; dims=ndims(ttval)))
                        TracedUtils.set!(
                            dresult[i],
                            path[2:end],
                            @allowscalar(TracedUtils.get_mlir_data(sl))
                        )
                    end
                end
                residx += 1
            end
        elseif TracedUtils.has_idx(a, argprefix)
            idx, path = TracedUtils.get_argidx(a, argprefix)
            arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]
            TracedUtils.set!(
                arg.val, path[3:end], TracedUtils.transpose_val(MLIR.IR.result(res, residx))
            )
            residx += 1
        else
            TracedUtils.set!(a, (), TracedUtils.transpose_val(MLIR.IR.result(res, residx)))
            residx += 1
        end
    end

    restup = Any[(a isa Active) ? copy(a) : nothing for a in args]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)

        arg = idx == 1 && fnwrap ? f : args[idx - fnwrap]
        act_from_type(arg, reverse) != enzyme_out && continue

        if idx == 1 && fnwrap && arg isa Active
            @assert false
        end

        set_act!(
            arg,
            path[3:end],
            reverse,
            TracedUtils.transpose_val(MLIR.IR.result(res, residx));
            width,
            emptypath=arg isa Active,
        )
        residx += 1
    end

    if reverse
        if EnzymeCore.needs_primal(CMode)
            return ((restup...,), result)
        else
            return ((restup...,),)
        end
    else
        if EnzymeCore.needs_primal(CMode)
            if CMode <: ForwardMode && !(A <: Const)
                return (dresult, result)
            else
                return (result,)
            end
        else
            if CMode <: ForwardMode && !(A <: Const)
                return (dresult,)
            else
                return ()
            end
        end
    end
end

@inline _jacobian_unwrap_const(x::Const) = x.val
@inline _jacobian_unwrap_const(x) = x

function _jacobian_wrap_const(x)
    if x isa Const
        return x
    end
    x isa Annotation &&
        error(
            "Reactant jacobian overlay v1 expects raw values or Const-wrapped values for non-differentiated arguments.",
        )
    return Const(x)
end

function _jacobian_active_arg_index(args::Tuple)
    idx = 0
    for i in eachindex(args)
        if !(args[i] isa Const)
            if idx != 0
                error(
                    "Reactant jacobian overlay v1 supports exactly one differentiable argument; wrap all other arguments with `Const(...)`.",
                )
            end
            idx = i
        end
    end
    idx == 0 &&
        error(
            "Reactant jacobian overlay v1 requires exactly one differentiable argument; all arguments were Const.",
        )
    return idx
end

function _jacobian_parse_chunk(chunk)
    chunk === nothing && return nothing
    chunk isa Val || error("Reactant jacobian overlay v1 expects `chunk` to be `nothing` or `Val{N}()`.")
    c = typeof(chunk).parameters[1]
    c == 0 && error("Cannot differentiate with a batch size of 0")
    c < 0 && error("Reactant jacobian overlay v1 requires `chunk` to be positive.")
    return c
end

function _jacobian_parse_nouts(n_outs)
    n_outs === nothing &&
        error(
            "Reactant jacobian overlay v1 requires explicit `n_outs` in reverse mode.",
        )
    n_outs isa Val ||
        error("Reactant jacobian overlay v1 expects `n_outs` to be a tuple `Val((... ,))`.")
    dims = typeof(n_outs).parameters[1]
    dims isa Tuple ||
        error("Reactant jacobian overlay v1 expects `n_outs` to be a tuple `Val((... ,))`.")
    for d in dims
        (d isa Integer && d >= 0) ||
            error("Reactant jacobian overlay v1 requires non-negative integer `n_outs` dimensions.")
    end
    return dims
end

function _jacobian_num_elements(dims::Tuple)
    if isempty(dims)
        return 1
    end
    return prod(dims)
end

function _jacobian_forward_seed_groups(x, chunk)
    c = _jacobian_parse_chunk(chunk)
    if x isa AbstractFloat
        return ((one(x),),)
    elseif x isa AbstractArray
        if c === nothing
            return (Enzyme.onehot(x),)
        else
            return Enzyme.chunkedonehot(x, Val(c))
        end
    else
        error(
            "Reactant jacobian overlay v1 currently supports differentiating only `AbstractArray` or `AbstractFloat` arguments.",
        )
    end
end

function _jacobian_forward_assemble(rows::Vector{Any}, x::AbstractArray)
    isempty(rows) && error("Reactant jacobian overlay v1 produced no Jacobian rows.")
    first_row = first(rows)
    first_row isa AbstractArray &&
        error(
            "Reactant jacobian overlay v1 currently supports only scalar-output functions in forward mode.",
        )
    length(rows) == length(x) ||
        error("Reactant jacobian overlay v1 expected $(length(x)) Jacobian rows, got $(length(rows)).")
    stacked = call_with_reactant(stack, Tuple(rows))
    return call_with_reactant(reshape, stacked, size(x)...)
end

function _jacobian_forward_assemble(rows::Vector{Any}, ::AbstractFloat)
    length(rows) == 1 ||
        error(
            "Reactant jacobian overlay v1 expected a single Jacobian entry for scalar input.",
        )
    return only(rows)
end

function _jacobian_forward_assemble(rows::Vector{Any}, x)
    error(
        "Reactant jacobian overlay v1 currently supports differentiating only `AbstractArray` or `AbstractFloat` arguments (got $(Core.Typeof(x))).",
    )
end

function overload_jacobian(
    mode::Enzyme.ForwardMode, f, x, xs...; chunk=nothing, shadows=nothing, kwargs...
)
    isempty(kwargs) ||
        error(
            "Reactant jacobian overlay v1 only supports `chunk` and `shadows` keywords in forward mode.",
        )
    shadows === nothing ||
        error(
            "Reactant jacobian overlay v1 does not support explicit `shadows`; omit the keyword to use internal Jacobian seeds.",
        )

    all_args = (x, xs...)
    active_idx = _jacobian_active_arg_index(all_args)
    active_arg = all_args[active_idx]

    active_arg isa Annotation &&
        !(active_arg isa Const) &&
        error(
            "Reactant jacobian overlay v1 expects a raw differentiable argument value, not an Enzyme annotation wrapper.",
        )

    active_val = _jacobian_unwrap_const(active_arg)
    seed_groups = _jacobian_forward_seed_groups(active_val, chunk)

    f_ann = f isa Annotation ? f : Const(f)

    mode_iter = mode
    primal = nothing
    rows = Any[]

    for seeds in seed_groups
        seeds_tuple = Tuple(seeds)
        active_ann = if length(seeds_tuple) == 1
            Duplicated(active_val, only(seeds_tuple))
        else
            BatchDuplicated(active_val, seeds_tuple)
        end
        rt = length(seeds_tuple) == 1 ? Duplicated : BatchDuplicated

        ann_args = Any[]
        for i in eachindex(all_args)
            if i == active_idx
                push!(ann_args, active_ann)
            else
                push!(ann_args, _jacobian_wrap_const(all_args[i]))
            end
        end

        res = lower_jacobian(mode_iter, f_ann, rt, ann_args...)
        dpart = res[1]

        if EnzymeCore.needs_primal(mode_iter)
            primal = res[2]
            mode_iter = EnzymeCore.NoPrimal(mode_iter)
        end

        if length(seeds_tuple) == 1
            push!(rows, dpart)
        else
            append!(rows, collect(dpart))
        end
    end

    jac = _jacobian_forward_assemble(rows, active_val)
    derivs = ntuple(i -> i == active_idx ? jac : nothing, length(all_args))

    if EnzymeCore.needs_primal(mode)
        return (; derivs, val=primal)
    end
    return derivs
end

function overload_jacobian(
    mode::Enzyme.ReverseMode, f, x, xs...; n_outs=nothing, chunk=nothing
)
    n_out_dims = _jacobian_parse_nouts(n_outs)
    n_out_elems = _jacobian_num_elements(n_out_dims)
    n_out_elems == 1 ||
        error(
            "Reactant jacobian overlay v1 currently supports only scalar-output functions in reverse mode (`prod(n_outs) == 1`).",
        )

    c = _jacobian_parse_chunk(chunk)
    (c === nothing || c == 1) ||
        error(
            "Reactant jacobian overlay v1 reverse mode currently supports only `chunk=nothing` or `chunk=Val(1)`.",
        )

    all_args = (x, xs...)
    active_idx = _jacobian_active_arg_index(all_args)
    active_arg = all_args[active_idx]

    active_arg isa Annotation &&
        !(active_arg isa Const) &&
        error(
            "Reactant jacobian overlay v1 expects a raw differentiable argument value, not an Enzyme annotation wrapper.",
        )

    active_val = _jacobian_unwrap_const(active_arg)
    active_val isa Union{AbstractArray,AbstractFloat} ||
        error(
            "Reactant jacobian overlay v1 currently supports differentiating only `AbstractArray` or `AbstractFloat` arguments.",
        )

    active_seed = Enzyme.make_zero(active_val)
    active_ann = Duplicated(active_val, active_seed)
    f_ann = f isa Annotation ? f : Const(f)

    ann_args = Any[]
    for i in eachindex(all_args)
        if i == active_idx
            push!(ann_args, active_ann)
        else
            push!(ann_args, _jacobian_wrap_const(all_args[i]))
        end
    end

    res = lower_jacobian(mode, f_ann, Active, ann_args...)

    derivs = ntuple(i -> i == active_idx ? active_seed : nothing, length(all_args))
    if EnzymeCore.needs_primal(mode)
        return (; derivs, val=res[2])
    end
    return derivs
end

const ignore_derivatives = EnzymeCore.ignore_derivatives
