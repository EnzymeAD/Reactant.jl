module ReactantTensorOperationsExt

using Reactant
using Reactant:
    @reactant_overlay,
    use_overlayed_version,
    call_with_native,
    TracedRArray,
    TracedRNumber,
    unwrapped_eltype,
    promote_to
using TensorOperations: TensorOperations as TO

# allocation
function TO.tensoradd_type(TC, A::ConcreteRArray, pA::Index2Tuple, conjA::Bool)
    return ConcreteRArray{TC,TO.numind(pA)}
end

function TO.tensoradd_type(TC, A::TracedRArray, pA::Index2Tuple, conjA::Bool)
    return TracedRArray{unwrapped_eltype(TC),TO.numind(pA)}
end

# backend selection
@reactant_overlay function TO.select_backend(
    ::typeof(TO.tensoradd!), C::AbstractArray, A::AbstractArray
)
    if use_overlayed_version(C) || use_overlayed_version(A)
        TO.BaseCopy()
    else
        call_with_native(TO.select_backend, TO.tensoradd!, C, A)
    end
end

@reactant_overlay function TO.select_backend(
    ::typeof(TO.tensortrace!), C::AbstractArray, A::AbstractArray
)
    if use_overlayed_version(C) || use_overlayed_version(A)
        TO.BaseCopy()
    else
        call_with_native(TO.select_backend, TO.tensortrace!, C, A)
    end
end

@reactant_overlay function TO.select_backend(
    ::typeof(TO.tensorcontract!), C::AbstractArray, A::AbstractArray, B::AbstractArray
)
    if use_overlayed_version(C) || use_overlayed_version(A) || use_overlayed_version(B)
        TO.BaseCopy()
    else
        call_with_native(TO.select_backend, TO.tensorcontract!, C, A, B)
    end
end

# implementation
function TO.tensorscalar(C::TracedRArray)
    return ndims(C) == 0 ? @allowscalar(C[]) : throw(DimensionMismatch())
end

end
