module TracedLinearAlgebra

using ..Reactant:
    TracedRArray,
    TracedRNumber,
    AnyTracedRArray,
    AnyTracedRMatrix,
    AnyTracedRVector,
    Ops,
    MLIR

using ..TracedUtils: TracedUtils, get_mlir_data, materialize_traced_array, set_mlir_data!

using LinearAlgebra

# Various Wrapper Arrays defined in LinearAlgebra
function materialize_traced_array(
    x::Transpose{TracedRNumber{T},TracedRArray{T,N}}
) where {T,N}
    px = parent(x)
    A = ndims(px) == 1 ? reshape(px, :, 1) : px
    return permutedims(A, (2, 1))
end

function materialize_traced_array(
    x::Adjoint{TracedRNumber{T},TracedRArray{T,N}}
) where {T,N}
    return conj(materialize_traced_array(transpose(parent(x))))
end

function materialize_traced_array(
    x::LinearAlgebra.Diagonal{TracedRNumber{T},TracedRArray{T,1}}
) where {T}
    return LinearAlgebra.diagm(parent(x))
end

function TracedUtils.materialize_traced_array(x::Tridiagonal{T,TracedRArray{T,1}}) where {T}
    return diagm(-1 => x.dl, 0 => x.d, 1 => x.du)
end

for (AT, comp) in ((:LowerTriangular, "GE"), (:UpperTriangular, "LE"))
    uAT = Symbol(:Unit, AT)
    @eval begin
        function TracedUtils.materialize_traced_array(
            x::$(AT){T,TracedRArray{T,2}}
        ) where {T}
            m, n = size(x)
            row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
            col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
            indicator = Ops.compare(row_idxs, col_idxs; comparison_direction=$(comp))
            return Ops.select(indicator, parent(x), zero(parent(x)))
        end

        function TracedUtils.materialize_traced_array(
            x::$(uAT){T,TracedRArray{T,2}}
        ) where {T}
            m, n = size(x)
            row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
            col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
            nondiag_indicator = Ops.compare(row_idxs, col_idxs; comparison_direction="NE")
            x = materialize_traced_array($(AT)(parent(x)))
            return Ops.select(nondiag_indicator, x, one.(x))
        end
    end
end

function TracedUtils.materialize_traced_array(x::Symmetric{T,TracedRArray{T,2}}) where {T}
    m, n = size(x)
    row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
    col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
    if x.uplo == 'L'
        indicator = Ops.compare(row_idxs, col_idxs; comparison_direction="GT")
        x_lt = Ops.select(indicator, parent(x), zero(parent(x)))
        x_ltd = materialize_traced_array(LowerTriangular(parent(x)))
        return Ops.add(x_lt, Ops.transpose(x_ltd, [2, 1]))
    else
        indicator = Ops.compare(row_idxs, col_idxs; comparison_direction="LT")
        x_ut = Ops.select(indicator, parent(x), zero(parent(x)))
        x_utd = materialize_traced_array(UpperTriangular(parent(x)))
        return Ops.add(Ops.transpose(x_utd, [2, 1]), x_ut)
    end
end

function TracedUtils.set_mlir_data!(
    x::Transpose{TracedRNumber{T},TracedRArray{T,N}}, data
) where {T,N}
    tdata = TracedRArray{T}(data)
    px = parent(x)
    px.mlir_data = (
        if ndims(px) == 1
            Ops.reshape(tdata, length(tdata))
        else
            Ops.transpose(tdata, [2, 1])
        end
    ).mlir_data
    return x
end

function TracedUtils.set_mlir_data!(
    x::Adjoint{TracedRNumber{T},TracedRArray{T,N}}, data
) where {T,N}
    tdata = TracedRArray{T}(data)
    px = parent(x)
    transposed_data =
        ndims(px) == 1 ? Ops.reshape(tdata, length(tdata)) : Ops.transpose(tdata, [2, 1])
    px.mlir_data = (T <: Real ? transposed_data : Ops.conj(transposed_data)).mlir_data
    return x
end

function TracedUtils.set_mlir_data!(x::Diagonal{TracedRNumber{T},TracedRArray{T,1}}, data) where {T}
    parent(x).mlir_data = diag(TracedRArray{T}(data)).mlir_data
    return x
end

for (AT, dcomp, ocomp) in (
    (:LowerTriangular, "GE", "LT"),
    (:UnitLowerTriangular, "GT", "LE"),
    (:UpperTriangular, "LE", "GT"),
    (:UnitUpperTriangular, "LT", "GE"),
)
    @eval function TracedUtils.set_mlir_data!(
        x::LinearAlgebra.$(AT){T,TracedRArray{T,2}}, data
    ) where {T}
        tdata = TracedRArray{T}(data)
        z = zero(tdata)
        m, n = size(x)
        row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
        col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
        data_indicator = Ops.compare(row_idxs, col_idxs; comparison_direction=$(dcomp))
        original_indicator = Ops.compare(row_idxs, col_idxs; comparison_direction=$(ocomp))
        res = Ops.add(
            Ops.select(data_indicator, tdata, z), Ops.select(original_indicator, x.data, z)
        )
        set_mlir_data!(x.data, res.mlir_data)
        return x
    end
end

function TracedUtils.set_mlir_data!(
    x::LinearAlgebra.Symmetric{T,TracedRArray{T,2}}, data
) where {T}
    if x.uplo == 'L'
        set_mlir_data!(LinearAlgebra.LowerTriangular(parent(x)), data)
    else
        set_mlir_data!(LinearAlgebra.UpperTriangular(parent(x)), data)
    end
    return x
end

function TracedUtils.set_mlir_data!(x::Tridiagonal{T,TracedRArray{T,1}}, data) where {T}
    tdata = TracedRArray{T}(data)
    set_mlir_data!(x.dl, diag(tdata, -1).mlir_data)
    set_mlir_data!(x.d, diag(tdata, 0).mlir_data)
    set_mlir_data!(x.du, diag(tdata, 1).mlir_data)
    return x
end

# Core functions
function LinearAlgebra.mul!(
    @nospecialize(C::TracedRArray{T,1}),
    @nospecialize(A::AnyTracedRMatrix),
    @nospecialize(B::AnyTracedRVector),
    α::Number=true,
    β::Number=false,
) where {T}
    # TODO: The reshape operations are not getting optimized, we should directly call dot_general
    rC = Ops.reshape(C, length(C), 1)
    mul!(rC, A, reshape(B, :, 1), α, β)
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
    mul!(C, A, reshape(B, :, 1), α, β)
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
        Ops.iota(Int64, [size(X)...]; iota_dimension=2),
        TracedUtils.broadcast_to_size(k, size(X)),
    )
    idxs = Ops.compare(iota_1, iota_2; comparison_direction="LE")
    X.mlir_data = Ops.select(idxs, X, zero(X)).mlir_data
    return X
end

function LinearAlgebra.tril!(@nospecialize(X::TracedRArray{T,2}), k::Integer) where {T}
    iota_1 = Ops.iota(Int64, [size(X)...]; iota_dimension=1)
    iota_2 = Ops.subtract(
        Ops.iota(Int64, [size(X)...]; iota_dimension=2),
        TracedUtils.broadcast_to_size(k, size(X)),
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

    return Ops.gather_getindex(x, TracedUtils.promote_to(TracedRArray{Int,2}, indices))
end

function LinearAlgebra._diagm(
    shape, kv::Pair{<:Integer,<:AnyTracedRArray{T,1}}...
) where {T}
    m, n = LinearAlgebra.diagm_size(shape, kv...)
    scatter_indices = Matrix{Int64}[]
    concat_inputs = MLIR.IR.Value[]
    for (k, v) in kv
        push!(scatter_indices, diagonal_indices_zero_indexed(m, n, k)[1:length(v), :])
        push!(concat_inputs, get_mlir_data(v))
    end
    scatter_indices = Ops.constant(reduce(vcat, scatter_indices))
    values = TracedRArray{T,1}(
        (),
        MLIR.IR.result(MLIR.Dialects.stablehlo.concatenate(concat_inputs; dimension=0), 1),
        (size(scatter_indices, 1),),
    )
    return Ops.scatter_setindex(
        Ops.constant(fill(zero(T), (m, n))), scatter_indices, values
    )
end

# Common Utilities
## The cartesian version doesn't exist in julia 1.10
function diagonal_indices_zero_indexed(m::Integer, n::Integer, k::Integer=0)
    idx1, idx2 = 1 + max(0, -k), 1 + max(0, k)
    L = max(0, k ≤ 0 ? min(m + k, n) : min(m, n - k))
    indices = Matrix{Int}(undef, (L, 2))
    for i in axes(indices, 1)
        indices[i, 1] = idx1 + i - 2
        indices[i, 2] = idx2 + i - 2
    end
    return indices
end

end
