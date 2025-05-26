# TODO: move the overload_autodiff here as well

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
