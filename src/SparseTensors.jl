# Minimal sparse-matrix support lowered through the MLIR `sparse_tensor` dialect.
#
# `CSRMatrix` is an opaque wrapper (deliberately not an `AbstractArray`, so it is
# never swept into the dense `AnyTracedRArray` overloads) holding the three CSR
# buffers. At trace time `A * x` / `A * B` / `mul!` emit a
# `sparse_tensor.assemble` producing a `tensor<m x n x T, #sparse_tensor.encoding>`
# value consumed by an `enzymexla.sparse.spmm` (alpha * A * B + beta * C). The
# Enzyme-JAX `lower-sparse-csr` pass rewrites that pair into a
# `stablehlo.custom_call @reactant_csr_matmul[_acc]` handled by
# cuSPARSE/hipSPARSE (see deps/ReactantExtra/xla_ffi.cpp) before XLA sees the
# sparse-encoded types; constant alpha/beta (e.g. the scalars of a 5-arg
# `mul!`) are fused into a single library call.

"""
    CSRMatrix{T,Ti}(m, n, rowptr, colind, nzval)

An opaque `m × n` sparse matrix in CSR format with element type `T` and index
type `Ti`. `rowptr` has length `m + 1` and `colind`/`nzval` have length `nnz`;
all indices are 0-based, as required by the MLIR `sparse_tensor` dialect.

Construct one from a `SparseArrays.SparseMatrixCSC` via `CSRMatrix(A)` (requires
loading SparseArrays), and pass it through [`Reactant.to_rarray`](@ref) like any
other array. Inside traced functions only `A * x`, `A * B`, and
`LinearAlgebra.mul!` are supported, and execution requires a CUDA or ROCm
backend.
"""
struct CSRMatrix{T,Ti,V<:AbstractVector,Vi<:AbstractVector}
    m::Int
    n::Int
    rowptr::Vi
    colind::Vi
    nzval::V
end

function CSRMatrix(
    m::Integer,
    n::Integer,
    rowptr::AbstractVector,
    colind::AbstractVector,
    nzval::AbstractVector,
)
    length(rowptr) == m + 1 ||
        throw(ArgumentError("rowptr must have length m + 1 = $(m + 1)"))
    length(colind) == length(nzval) ||
        throw(ArgumentError("colind and nzval must have the same length"))
    return CSRMatrix{
        unwrapped_eltype(eltype(nzval)),
        unwrapped_eltype(eltype(colind)),
        typeof(nzval),
        typeof(colind),
    }(
        m, n, rowptr, colind, nzval
    )
end

const TracedCSRMatrix{T,Ti} = CSRMatrix{T,Ti,TracedRArray{T,1},TracedRArray{Ti,1}}

Base.size(A::CSRMatrix) = (A.m, A.n)
Base.size(A::CSRMatrix, i::Integer) = i <= 2 ? size(A)[i] : 1
Base.eltype(::Core.Type{<:CSRMatrix{T}}) where {T} = T
Base.eltype(::CSRMatrix{T}) where {T} = T

function Base.show(io::IO, A::CSRMatrix{T,Ti}) where {T,Ti}
    return print(
        io, "$(A.m)×$(A.n) CSRMatrix{$T,$Ti} with $(length(A.colind)) stored entries"
    )
end

# Tracing
Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(_::Core.Type{CSRMatrix{T,Ti,V,Vi}}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Core.Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
) where {T,Ti,V,Vi}
    V2 = traced_type_inner(V, seen, mode, track_numbers, ndevices, runtime)
    Vi2 = traced_type_inner(Vi, seen, mode, track_numbers, ndevices, runtime)
    return CSRMatrix{T,Ti,V2,Vi2}
end

Base.@nospecializeinfer function make_tracer(
    seen, @nospecialize(prev::CSRMatrix), @nospecialize(path), mode; kwargs...
)
    return make_tracer_via_immutable_constructor(seen, prev, path, mode; kwargs...)
end

function use_overlayed_version(A::CSRMatrix)
    return use_overlayed_version((A.rowptr, A.colind, A.nzval))
end

# IR emission
function _csr_encoding(::Core.Type{Ti}) where {Ti}
    width = 8 * sizeof(Ti)
    return Base.parse(
        MLIR.IR.Attribute,
        "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = $width, crdWidth = $width }>",
    )
end

function _with_nzval_eltype(::Core.Type{T}, A::TracedCSRMatrix{T}) where {T}
    return A
end
function _with_nzval_eltype(::Core.Type{T}, A::TracedCSRMatrix{T2,Ti}) where {T,T2,Ti}
    nzval = promote_to(TracedRArray{T,1}, A.nzval)
    return CSRMatrix{T,Ti,TracedRArray{T,1},TracedRArray{Ti,1}}(
        A.m, A.n, A.rowptr, A.colind, nzval
    )
end

"""
    sparse_csr_spmm(alpha, A::TracedCSRMatrix, B, beta, C)

Emits `sparse_tensor.assemble` + `enzymexla.sparse.spmm` computing
`alpha * A * B + beta * C` (spmv for vector `B`, spmm for matrix `B`) and
returns the dense result. The emitted pair is lowered to a library call by the
Enzyme-JAX `lower-sparse-csr` pass; when `alpha`/`beta` trace to constants the
scaling and accumulation are fused into that call.
"""
function sparse_csr_spmm(
    alpha::TracedRNumber{T},
    A::TracedCSRMatrix{T,Ti},
    B::TracedRArray{T},
    beta::TracedRNumber{T},
    C::TracedRArray{T};
    location=Ops.mlir_stacktrace("sparse_csr_spmm", @__FILE__, @__LINE__),
) where {T,Ti}
    ndims(B) in (1, 2) ||
        throw(ArgumentError("Only vectors and matrices can be multiplied by a CSRMatrix"))
    size(B, 1) == A.n ||
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B))"))
    ressize = ndims(B) == 1 ? Int[A.m] : Int[A.m, size(B, 2)]
    ndims(C) == ndims(B) && size(C) == Tuple(ressize) || throw(
        DimensionMismatch("C has size $(size(C)), expected $(Tuple(ressize))")
    )

    sparse_type = MLIR.IR.TensorType(Int[A.m, A.n], MLIR.IR.Type(T), _csr_encoding(Ti))
    asm = MLIR.Dialects.sparse_tensor.assemble(
        MLIR.IR.Value[A.rowptr.mlir_data, A.colind.mlir_data],
        A.nzval.mlir_data;
        result=sparse_type,
        location,
    )

    res = MLIR.IR.result(
        MLIR.Dialects.enzymexla.sparse_spmm(
            alpha.mlir_data,
            MLIR.IR.result(asm, 1),
            B.mlir_data,
            beta.mlir_data,
            C.mlir_data;
            output=MLIR.IR.TensorType(ressize, MLIR.IR.Type(T)),
            location,
        ),
    )
    return TracedRArray{T,length(ressize)}((), res, Tuple(ressize))
end

# LinearAlgebra surface
function LinearAlgebra.mul!(
    C::TracedRArray{T},
    A::TracedCSRMatrix,
    B::AbstractVecOrMat,
    α::Number=true,
    β::Number=false,
) where {T}
    ndims(C) in (1, 2) || throw(ArgumentError("C must be a vector or a matrix"))
    B = promote_to(TracedRArray{T}, B)

    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B))"))
    size(C, 1) == size(A, 1) ||
        throw(DimensionMismatch("C has size $(size(C)), A has size $(size(A))"))
    size(C, 2) == size(B, 2) ||
        throw(DimensionMismatch("C has size $(size(C)), B has size $(size(B))"))

    # Non-traced α/β become constants that `lower-sparse-csr` fuses into a
    # single library call; in particular a constant β == 0 never reads C.
    alpha = promote_to(TracedRNumber{T}, α)
    beta = promote_to(TracedRNumber{T}, β)

    ressize = ndims(B) == 1 ? (size(A, 1),) : (size(A, 1), size(B, 2))
    C_arr = ReactantCore.materialize_traced_array(C)
    if ndims(C_arr) != ndims(B)
        C_arr = ReactantCore.materialize_traced_array(reshape(C_arr, ressize...))
    end

    res = sparse_csr_spmm(alpha, _with_nzval_eltype(T, A), B, beta, C_arr)
    if size(res) != size(C)
        res = ReactantCore.materialize_traced_array(reshape(res, size(C)))
    end

    TracedUtils.set_mlir_data!(C, TracedUtils.get_mlir_data(res))
    return C
end

function _sparse_mul(A::TracedCSRMatrix{T}, B::AbstractVecOrMat) where {T}
    T2 = Base.promote_op(*, T, unwrapped_eltype(eltype(B)))
    B2 = promote_to(TracedRArray{T2}, B)
    ressize = ndims(B2) == 1 ? (A.m,) : (A.m, size(B2, 2))
    return sparse_csr_spmm(
        promote_to(TracedRNumber{T2}, 1),
        _with_nzval_eltype(T2, A),
        B2,
        promote_to(TracedRNumber{T2}, 0),
        Ops.fill(zero(T2), ressize),
    )
end

Base.:*(A::TracedCSRMatrix, x::AbstractVector) = _sparse_mul(A, x)
Base.:*(A::TracedCSRMatrix, B::AbstractMatrix) = _sparse_mul(A, B)

for f in (:adjoint, :transpose)
    @eval function Base.$(f)(::CSRMatrix)
        return error(
            "`$($(QuoteNode(f)))` of a `Reactant.CSRMatrix` is not supported yet; only `A * x`, `A * B` and `mul!` are implemented.",
        )
    end
end
