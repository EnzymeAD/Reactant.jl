module ReactantTensorOperationsExt

using LinearAlgebra
using Reactant
using Reactant:
    @reactant_overlay,
    use_overlayed_version,
    call_with_native,
    TracedRArray,
    TracedRNumber,
    unwrapped_eltype,
    promote_to
using TensorOperations: TensorOperations as TO, StridedView, TupleTools, Index2Tuple, IndexTuple, stridedtensoradd!, _unsafe_blas_contract!

# allocation
function TO.tensoradd_type(TC, A::ConcreteRArray, pA::Index2Tuple, conjA::Bool)
    return ConcreteRArray{TC,TO.numind(pA)}
end

function TO.tensoradd_type(TC, A::TracedRArray, pA::Index2Tuple, conjA::Bool)
    return TracedRArray{unwrapped_eltype(TC),TO.numind(pA)}
end

@reactant_overlay function TO._unsafe_blas_contract!(
        C::StridedView, A::StridedView, pA, B::StridedView, pB, pAB::IndexTuple, α::Number, β::Number,
    )
    if use_overlayed_version(C) || use_overlayed_version(A) || use_overlayed_version(B)
        sizeA = size(A)
        sizeB = size(B)
        csizeA = TupleTools.getindices(sizeA, pA[2])
        csizeB = TupleTools.getindices(sizeB, pB[1])
        osizeA = TupleTools.getindices(sizeA, pA[1])
        osizeB = TupleTools.getindices(sizeB, pB[2])

        LinearAlgebra.mul!(
            TO.sreshape(permutedims(C, pAB), (prod(osizeA), prod(osizeB))),
            TO.sreshape(permutedims(A, TO.linearize(pA)), (prod(osizeA), prod(csizeA))),
            TO.sreshape(permutedims(B, TO.linearize(pB)), (prod(csizeB), prod(osizeB))),
            α, β
        )
        return C
    else
        return Reactant.call_with_native(TO._unsafe_blas_contract!, C, A, pA, B, pB, pAB, α, β)
    end
end

@reactant_overlay function TO.stridedtensoradd!(
        C::StridedView, A::StridedView, pA::IndexTuple, α::Number, β::Number,
    )
    if use_overlayed_version(C) || use_overlayed_version(A)
        TO.argcheck_tensoradd(C, A, pA)
        TO.dimcheck_tensoradd(C, A, pA)
        !TO.istrivialpermutation(pA) && Base.mightalias(C, A) &&
            throw(ArgumentError("output tensor must not be aliased with input tensor"))
        Ap = permutedims(A, pA)
        TO.Strided._mapreducedim!(TO.Scaler(α), TO.Adder(), TO.Scaler(β), size(C), (C, Ap))
        return C
    else
        return Reactant.call_with_native(TO.stridedtensoradd!, C, A, pA, α, β)
    end
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
