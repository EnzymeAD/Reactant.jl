module ReactantTensorOperationsExt

using Reactant
using Reactant: @reactant_overlay, use_overlayed_version, call_with_native, TracedRArray, TracedRNumber, unwrapped_eltype
using Reactant.Ops: @opcall
using Reactant.TracedUtils: materialize_traced_array, get_mlir_data, set_mlir_data!
using TensorOperations
using TensorOperations: TensorOperations as TO, ReactantBackend, ReactantAllocator

# allocation
function TO.tensoradd_type(TC, A::ConcreteRArray, pA::Index2Tuple, conjA::Bool)
    return ConcreteRArray{TC, TO.numind(pA)}
end

function TO.tensoradd_type(TC, A::TracedRArray, pA::Index2Tuple, conjA::Bool)
    return TracedRArray{unwrapped_eltype(TC), TO.numind(pA)}
end

function TO.tensoralloc_add(
    TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
    istemp::Val, allocator::TO.ReactantAllocator{true}
)
    T = unwrapped_eltype(TC)
    ttype = TO.tensoradd_type(T, A, pA, conjA)
    structure = TO.tensoradd_structure(A, pA, conjA)
    return similar(ttype, structure)
end

# backend selection
@reactant_overlay function TO.select_backend(::typeof(TO.tensoradd!), C::AbstractArray, A::AbstractArray)
    if use_overlayed_version(C) || use_overlayed_version(A)
        ReactantBackend()
    else
        call_with_native(TO.select_backend, TO.tensoradd!, C, A)
    end
end

@reactant_overlay function TO.select_backend(::typeof(TO.tensortrace!), C::AbstractArray, A::AbstractArray)
    if use_overlayed_version(C) || use_overlayed_version(A)
        ReactantBackend()
    else
        call_with_native(TO.select_backend, TO.tensortrace!, C, A)
    end
end

@reactant_overlay function TO.select_backend(::typeof(TO.tensorcontract!), C::AbstractArray, A::AbstractArray, B::AbstractArray)
    if use_overlayed_version(C) || use_overlayed_version(A) || use_overlayed_version(B)
        ReactantBackend()
    else
        call_with_native(TO.select_backend, TO.tensorcontract!, C, A, B)
    end
end

# implementation
function TO.tensorscalar(C::TracedRArray)
    return ndims(C) == 0 ? @allowscalar(C[]) : throw(DimensionMismatch())
end

function TO.tensoradd!(
    Ct::TracedRArray,
    A::AbstractArray, pA::Index2Tuple, conjA::Bool,
    α::Number, β::Number,
    ::ReactantBackend, allocator = ReactantAllocator{true}()
)
    TO.argcheck_tensoradd(Ct, A, pA)
    TO.dimcheck_tensoradd(Ct, A, pA)

    At = materialize_traced_array(A)
    At = permutedims(At, linearize(pA))
    if conjA
        At = conj(At)
    end

    Ctmp = α * At
    if β isa TracedRNumber || !iszero(β)
        Ctmp += β * Ct
    end
    set_mlir_data!(Ct, get_mlir_data(Ctmp))
    return Ct
end

# TODO tensortrace!

function tensorcontract!(
    Ct::TracedRArray,
    A::AbstractArray, pA::Index2Tuple, conjA::Bool,
    B::AbstractArray, pB::Index2Tuple, conjB::Bool,
    pAB::Index2Tuple,
    α::Number, β::Number,
    ::ReactantBackend, allocator = ReactantAllocator{true}()
)
    TO.argcheck_tensorcontract(Ct, A, pA, B, pB, pAB)
    TO.dimcheck_tensorcontract(Ct, A, pA, B, pB, pAB)

    At = materialize_traced_array(A)
    Bt = materialize_traced_array(B)

    if conjA
        At = conj(At)
    end

    if conjB
        Bt = conj(Bt)
    end

    contracting_dimensions = (collect(pA[2]), collect(pB[1]))
    Ctmp = α * @opcall dot_general(At, Bt; contracting_dimensions)
    if !iszero(β)
        Ctmp += β * Ct
    end
    set_mlir_data!(Ct, get_mlir_data(Ctmp))
    return Ct
end

end
