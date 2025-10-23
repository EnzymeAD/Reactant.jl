struct CallWithReactant{F} <: Function
    f::F
end

function Base.reducedim_init(f::F, op::CallWithReactant, A::AbstractArray, region) where {F}
    return Base.reducedim_init(f, op.f, A, region)
end

function (f::CallWithReactant{F})(args...; kwargs...) where {F}
    if isempty(kwargs)
        return call_with_reactant(f.f, args...)
    else
        return call_with_reactant(Core.kwcall, NamedTuple(kwargs), f.f, args...)
    end
end

function apply(f::F, args...; kwargs...) where {F}
    return f(args...; kwargs...)
end

function call_with_reactant end

# Defined in KernelAbstractions Ext
function ka_with_reactant end

@static if isdefined(Core, :BFloat16)
    nmantissa(::Type{Core.BFloat16}) = 7
end
nmantissa(::Type{Float16}) = 10
nmantissa(::Type{Float32}) = 23
nmantissa(::Type{Float64}) = 52

_unwrap_val(::Val{T}) where {T} = T
