module ReactantKernelAbstractionsExt

using Reactant: Reactant, ReactantBackend

using Adapt: Adapt
using KernelAbstractions: KernelAbstractions

const KA = KernelAbstractions

## back-end

Adapt.adapt_storage(::KA.CPU, a::Reactant.AnyConcretePJRTArray) = convert(Array, a)
Adapt.adapt_storage(::KA.CPU, a::Reactant.AnyConcreteIFRTArray) = convert(Array, a)

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


function KA.priority!(::ReactantBackend, prio::Symbol)
    if !(prio in (:high, :normal, :low))
        error("priority must be one of :high, :normal, :low")
    end
    return nothing
end

end
