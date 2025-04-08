module ReactantKernelAbstractionsExt

using Reactant

import KernelAbstractions as KA

using Adapt: Adapt

## back-end

export ReactantBackend

struct ReactantBackend <: KA.GPU end

function Base.getproperty(x::ReactantBackend, sym::Symbol)
    if sym === :always_inline
        return true
    elseif sym === :prefer_blocks
        return false
    else
        return Base.getfield(x, sym)
    end
end

KA.allocate(n::ReactantBackend, ::Type{T}, dims::Tuple) where {T} = KA.zeros(b, T, dims)
function KA.zeros(::ReactantBackend, ::Type{T}, dims::Tuple) where {T}
    return Reactant.to_rarray(zeros(T, dims))
end
function KA.ones(::ReactantBackend, ::Type{T}, dims::Tuple) where {T}
    return Reactant.to_rarray(ones(T, dims))
end

KA.get_backend(::Reactant.AnyTracedRArray) = ReactantBackend()
KA.get_backend(::Reactant.AnyConcretePJRTArray) = ReactantBackend()
function KA.synchronize(::ReactantBackend) end

Adapt.adapt_storage(::ReactantBackend, a::Array) = a
Adapt.adapt_storage(::ReactantBackend, a::Reactant.AnyTracedRArray) = a
Adapt.adapt_storage(::ReactantBackend, a::Reactant.AnyConcretePJRTArray) = a
Adapt.adapt_storage(::ReactantBackend, a::Reactant.AnyConcreteIFRTArray) = a
Adapt.adapt_storage(::KA.CPU, a::Reactant.AnyConcretePJRTArray) = convert(Array, a)
Adapt.adapt_storage(::KA.CPU, a::Reactant.AnyConcreteIFRTArray) = convert(Array, a)

## memory operations

function KA.copyto!(::ReactantBackend, A, B)
    Base.copyto!(A, B)
    return A
end

## kernel launch

function KA.mkcontext(kernel::KA.Kernel{ReactantBackend}, _ndrange, iterspace)
    return KA.CompilerMetadata{KA.ndrange(kernel),KA.DynamicCheck}(_ndrange, iterspace)
end

function KA.launch_config(kernel::KA.Kernel{ReactantBackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize,)
    end

    # partition checked that the ndrange's agreed
    if KA.ndrange(kernel) <: KA.StaticSize
        ndrange = nothing
    end

    iterspace, dynamic =
        if KA.workgroupsize(kernel) <: KA.DynamicSize && workgroupsize === nothing
            # use ndrange as preliminary workgroupsize for autotuning
            KA.partition(kernel, ndrange, ndrange)
        else
            KA.partition(kernel, ndrange, workgroupsize)
        end

    return ndrange, workgroupsize, iterspace, dynamic
end

KA.argconvert(k::KA.Kernel{ReactantBackend}, arg) = arg

function KA.priority!(::ReactantBackend, prio::Symbol)
    if !(prio in (:high, :normal, :low))
        error("priority must be one of :high, :normal, :low")
    end
    return nothing
end

function tokw(ndrange, workgroupsize, obj, args...)
    @inline obj(args...; ndrange, workgroupsize)
end

function (obj::KA.Kernel{ReactantBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    if Reactant.precompiling()
        @code_hlo optimize = false tokw(ndrange, workgroupsize, obj, args...)
    else
        @jit tokw(ndrange, workgroupsize, obj, args...)
    end
    return nothing
end

function ka_with_reactant end # defined in the CUDA extension

Reactant.@reactant_overlay @noinline Base.@nospecializeinfer function (
    obj::KA.Kernel{ReactantBackend}
)
    (args...; ndrange=nothing, workgroupsize=nothing)
    @nospecialize
    return Reactant.call_with_reactant(
        ka_with_reactant, ndrange, workgroupsize, obj, args...
    )
end

end
