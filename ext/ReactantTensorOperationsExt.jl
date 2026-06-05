module ReactantTensorOperationsExt

using Reactant
using Reactant:
    @reactant_overlay,
    use_overlayed_version,
    call_with_native,
    TracedRArray,
    TracedRNumber,
    unwrapped_eltype
using Reactant.Ops: @opcall
using Reactant.TracedUtils: materialize_traced_array, get_mlir_data, set_mlir_data!
using TensorOperations
using TensorOperations: TensorOperations as TO, ReactantBackend, ReactantAllocator

# allocation
function TO.tensoradd_type(TC, A::ConcreteRArray, pA::Index2Tuple, conjA::Bool)
    return ConcreteRArray{TC,TO.numind(pA)}
end

function TO.tensoradd_type(TC, A::TracedRArray, pA::Index2Tuple, conjA::Bool)
    return TracedRArray{unwrapped_eltype(TC),TO.numind(pA)}
end

function TO.tensoralloc_add(
    TC,
    A::AbstractArray,
    pA::Index2Tuple,
    conjA::Bool,
    istemp::Val,
    allocator::TO.ReactantAllocator{true},
)
    T = unwrapped_eltype(TC)
    ttype = TO.tensoradd_type(T, A, pA, conjA)
    structure = TO.tensoradd_structure(A, pA, conjA)
    return similar(ttype, structure)
end

# backend selection
@reactant_overlay function TO.select_backend(
    ::typeof(TO.tensoradd!), C::AbstractArray, A::AbstractArray
)
    if use_overlayed_version(C) || use_overlayed_version(A)
        ReactantBackend()
    else
        call_with_native(TO.select_backend, TO.tensoradd!, C, A)
    end
end

@reactant_overlay function TO.select_backend(
    ::typeof(TO.tensortrace!), C::AbstractArray, A::AbstractArray
)
    if use_overlayed_version(C) || use_overlayed_version(A)
        ReactantBackend()
    else
        call_with_native(TO.select_backend, TO.tensortrace!, C, A)
    end
end

@reactant_overlay function TO.select_backend(
    ::typeof(TO.tensorcontract!), C::AbstractArray, A::AbstractArray, B::AbstractArray
)
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
    A::AbstractArray,
    pA::Index2Tuple,
    conjA::Bool,
    α::Number,
    β::Number,
    ::ReactantBackend,
    allocator=ReactantAllocator{true}(),
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

function TO.tensortrace!(
    C::AbstractArray,
    A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
    α::Number, β::Number,
    ::ReactantBackend, allocator = ReactantAllocator{true}()
)
    TO.argcheck_tensortrace(C, A, p, q)
    TO.dimcheck_tensortrace(C, A, p, q)

    At = materialize_traced_array(A)

    if conjA
        At = conj(At)
    end

    # `p` contains the batching dims, `q` contains the left/right dims for partial trace
    start_indices = zeros(Int, prod(d->size(At, d), q[1]), TO.numind(q))
    for (i, inds) in enumerate(Iterators.product([1:size(At,d) for d in q[1]]...))
        start_indices[i,:] = repeat(collect(Int, inds), inner=2)
    end
    start_indices = Reactant.promote_to(TracedRArray{Int,2}, start_indices)
    offset_dims = collect(Int, 1:TO.numind(p))
    collapsed_slice_dims = collect(Iterators.flatten(zip(q...)))
    operand_batching_dims = Int[]
    start_indices_batching_dims = Int[]
    start_index_map = collect(Iterators.flatten(zip(q...)))
    index_vector_dim = 1
    slice_sizes = Int[d ∈ q[1] || d ∈ q[2] ? 1 : size(At,d) for d in 1:ndims(At)]
    indices_are_sorted = false
    Ctmp = @opcall gather(
        At,
        start_indices;
        offset_dims,
        collapsed_slice_dims,
        operand_batching_dims,
        start_indices_batching_dims,
        start_index_map,
        index_vector_dim,
        slice_sizes,
        indices_are_sorted
    )

    Ctmp = dropdims(sum(Ctmp; dims=ndims(Ctmp)); dims=ndims(Ctmp))

    if α isa TracedRNumber || !isone(α)
        Ctmp *= α
    end

    if β isa TracedRNumber || !iszero(β)
        Ctmp += β * C
    end

    set_mlir_data!(C, get_mlir_data(Ctmp))
    return C
end

function TO.tensorcontract!(
    Ct::TracedRArray,
    A::AbstractArray,
    pA::Index2Tuple,
    conjA::Bool,
    B::AbstractArray,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple,
    α::Number,
    β::Number,
    ::ReactantBackend,
    allocator=ReactantAllocator{true}(),
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
    Ctmp = @opcall dot_general(At, Bt; contracting_dimensions)
    if α isa TracedRNumber || !isone(α)
        Ctmp *= α
    end
    if β isa TracedRNumber || !iszero(β)
        Ctmp += β * Ct
    end
    set_mlir_data!(Ct, get_mlir_data(Ctmp))
    return Ct
end

end
