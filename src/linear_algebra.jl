module TracedLinearAlgebra

using ..Reactant
import ..TracedRArray
import ..TracedRNumber
import ..AnyTracedRArray
import ..AnyTracedRMatrix
import ..AnyTracedRVector

import ..TracedUtils
using ..TracedUtils:
    get_mlir_data,
    materialize_traced_array,
    set_mlir_data!

import ..Ops
import ..MLIR
using LinearAlgebra

function LinearAlgebra.mul!(
    @nospecialize(C::TracedRArray{T,1}),
    @nospecialize(A::AnyTracedRMatrix),
    @nospecialize(B::AnyTracedRVector),
    α::Number=true,
    β::Number=false,
) where {T}
    # TODO: The reshape operations are not getting optimized, we should directly call dot_general
    rC = Ops.reshape(C, length(C), 1)
    LinearAlgebra.mul!(rC, A, reshape(B, :, 1), α, β)
    C.mlir_data = get_mlir_data(vec(rC))
    return C
end

function LinearAlgebra.mul!(
    @nospecialize(C::TracedRArray{T,2}),
    @nospecialize(A::AnyTracedRMatrix),
    @nospecialize(B::AnyTracedRVector),
    α::Number=true,
    β::Number=false,
) where {T}
    LinearAlgebra.mul!(C, A, reshape(B, :, 1), α, β)
    return C
end

function LinearAlgebra.mul!(
    @nospecialize(C::TracedRArray{T,2}),
    @nospecialize(A::AnyTracedRMatrix),
    @nospecialize(B::AnyTracedRMatrix),
    α::Number=true,
    β::Number=false,
) where {T}
    if size(C) != (size(A, 1), size(B, 2))
        throw(
            DimensionMismatch(
                "C has size $(size(C)), A has size $(size(A)), B has size $(size(B))"
            ),
        )
    end
    if size(A, 2) != size(B, 1)
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B))"))
    end

    tmp = Ops.dot_general(
        T.(materialize_traced_array(A)),
        T.(materialize_traced_array(B));
        contracting_dimensions=([2], [1]),
    )

    res = if iszero(β)
        isone(α) ? tmp : Ops.multiply(tmp, TracedUtils.broadcast_to_size(T(α), size(C)))
    else
        α_res = Ops.multiply(tmp, TracedUtils.broadcast_to_size(T(α), size(C)))
        β_C = Ops.multiply(C, TracedUtils.broadcast_to_size(T(β), size(C)))
        Ops.add(α_res, β_C)
    end
    set_mlir_data!(C, get_mlir_data(res))
    return C
end

function LinearAlgebra.triu!(@nospecialize(X::TracedRArray{T,2}), k::Integer) where {T}
    iota_1 = Ops.iota(Int64, [size(X)...]; iota_dimension=1)
    iota_2 = Ops.subtract(
        Ops.iota(Int64, [size(X)...]; iota_dimension=2), TracedUtils.broadcast_to_size(k, size(X))
    )
    idxs = Ops.compare(iota_1, iota_2; comparison_direction="LE")
    X.mlir_data = Ops.select(idxs, X, zero(X)).mlir_data
    return X
end

function LinearAlgebra.tril!(@nospecialize(X::TracedRArray{T,2}), k::Integer) where {T}
    iota_1 = Ops.iota(Int64, [size(X)...]; iota_dimension=1)
    iota_2 = Ops.subtract(
        Ops.iota(Int64, [size(X)...]; iota_dimension=2), TracedUtils.broadcast_to_size(k, size(X))
    )
    idxs = Ops.compare(iota_1, iota_2; comparison_direction="GE")
    X.mlir_data = Ops.select(idxs, X, zero(X)).mlir_data
    return X
end

# LinearAlgebra defines norm with some conditionals which cannot be traced directly
function LinearAlgebra.norm(x::TracedRArray{T,N}, p::Real=2) where {T,N}
    isinf(p) && return maximum(abs, x)
    return mapreduce(Base.Fix2(^, p), +, x)^(1 / p)
end

function LinearAlgebra.diag(x::AnyTracedRArray{T,2}, k::Integer=0) where {T}
    y = materialize_traced_array(x)

    rows, cols = size(y)
    (start_row, start_col) = k ≥ 0 ? (0, k) : (-k, 0)
    diag_length = min(rows - start_row, cols - start_col)

    indices = stack((
        start_row:(start_row + diag_length - 1), start_col:(start_col + diag_length - 1)
    ))

    # XXX: creating an empty array causes
    # terminate called after throwing an instance of 'xla::XlaRuntimeError'
    #   what():  UNKNOWN: <unknown>:0: error: 'tensor.empty' op unsupported op for export to XLA
    #   <unknown>:0: note: see current operation: %0 = "tensor.empty"() : () -> tensor<0xf64>
    length(indices) ≤ 0 && return TracedUtils.promote_to(TracedRArray{T,1}, T[])

    idxs = get_mlir_data(TracedUtils.promote_to(TracedRArray{Int,2}, indices))

    #! format: off
    dimension_numbers = MLIR.API.stablehloGatherDimensionNumbersGet(
        MLIR.IR.context(),
        Int64(0), Int64[],
        Int64(2), Int64[0, 1],
        Int64(0), Int64[],
        Int64(0), Int64[],
        Int64(2), Int64[0, 1],
        Int64(1)
    )
    #! format: on

    slice_sizes = get_mlir_data(Reactant.TracedUtils.promote_to(TracedRArray{Int,1}, [1, 1]))
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dynamic_gather(
            get_mlir_data(y), idxs, slice_sizes; dimension_numbers
        ),
        1,
    )
    return TracedRArray{T,1}((), res, (diag_length,))
end

function LinearAlgebra.diagm(v::AnyTracedRArray{T,1}) where {T}
    return LinearAlgebra.diagm(length(v), length(v), v)
end
function LinearAlgebra.diagm(m::Integer, n::Integer, v::AnyTracedRArray{T,1}) where {T}
    m, n = LinearAlgebra.diagm_size((m, n), 0 => v) # size check

    v = materialize_traced_array(v)
    D = length(v)
    row_idxs = Ops.iota(Int, [D, D]; iota_dimension=1)
    col_idxs = Ops.iota(Int, [D, D]; iota_dimension=2)
    diag_indicator = Ops.compare(row_idxs, col_idxs; comparison_direction="EQ")

    mat = (v .+ zero(v)') .* diag_indicator
    return Ops.pad(
        mat, TracedUtils.promote_to(TracedRNumber{T}, 0); high=[m - length(v), n - length(v)]
    )
end

end
