module ReactantKernelAbstractionsExt

using Reactant: Reactant

using Adapt: Adapt
using KernelAbstractions: KernelAbstractions

const KA = KernelAbstractions

## back-end

# ToDo: Include XLA client, device and sharding in ReactantBackend struct, to
# support more complex applications? If so, need to adapt implementation of
# `KA.get_backend` and `KA.allocate` accordingly.
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

function KA.allocate(::ReactantBackend, ::Type{T}, dims::Tuple) where {T}
    return Reactant.ConcreteRArray{T}(undef, dims)
end

function KA.zeros(b::ReactantBackend, ::Type{T}, dims::Tuple) where {T}
    A = KA.allocate(b, T, dims)
    isempty(A) || fill!(A, zero(T))
    return A
end
function KA.ones(b::ReactantBackend, ::Type{T}, dims::Tuple) where {T}
    A = KA.allocate(b, T, dims)
    isempty(A) || fill!(A, one(T))
    return A
end

KA.get_backend(::Reactant.AnyTracedRArray) = ReactantBackend()
KA.get_backend(::Reactant.AnyConcreteRArray) = ReactantBackend()
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
        Reactant.@code_hlo optimize = false tokw(ndrange, workgroupsize, obj, args...)
    else
        Reactant.@jit tokw(ndrange, workgroupsize, obj, args...)
    end
    return nothing
end

@static if VERSION < v"1.12-"
    Reactant.@reactant_overlay Base.@nospecializeinfer @noinline function (
        obj::KA.Kernel{ReactantBackend}
    )(
        @nospecialize args...; ndrange=nothing, workgroupsize=nothing
    )
        return Reactant.call_with_reactant(
            Reactant.ka_with_reactant, ndrange, workgroupsize, obj, args...
        )
    end
else
    Reactant.@reactant_overlay function (obj::KA.Kernel{ReactantBackend})(
        args...; ndrange=nothing, workgroupsize=nothing
    )
        Base.@_noinline_meta
        Base.@_nospecializeinfer_meta
        return Reactant.call_with_reactant(
            Reactant.ka_with_reactant, ndrange, workgroupsize, obj, args...
        )
    end
end

end
