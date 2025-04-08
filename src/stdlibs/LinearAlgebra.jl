module TracedLinearAlgebra

using ..Reactant:
    Reactant,
    TracedRArray,
    TracedRNumber,
    AnyTracedRArray,
    AnyTracedRMatrix,
    AnyTracedRVector,
    AnyTracedRVecOrMat,
    unwrapped_eltype,
    Ops,
    MLIR

using ReactantCore: ReactantCore
using ReactantCore: materialize_traced_array

using ..TracedUtils: TracedUtils, get_mlir_data, set_mlir_data!

using LinearAlgebra

# Various Wrapper Arrays defined in LinearAlgebra
function ReactantCore.materialize_traced_array(
    x::Transpose{TracedRNumber{T},<:AnyTracedRArray}
) where {T}
    px = materialize_traced_array(parent(x))
    A = ndims(px) == 1 ? reshape(px, :, 1) : px
    return permutedims(A, (2, 1))
end

function ReactantCore.materialize_traced_array(
    x::Adjoint{TracedRNumber{T},<:AnyTracedRArray}
) where {T}
    return Ops.conj(
        materialize_traced_array(transpose(materialize_traced_array(parent(x))))
    )
end

function ReactantCore.materialize_traced_array(
    x::Diagonal{TracedRNumber{T},<:AnyTracedRVector}
) where {T}
    return diagm(materialize_traced_array(parent(x)))
end

function ReactantCore.materialize_traced_array(
    x::Tridiagonal{TracedRNumber{T},<:AnyTracedRVector}
) where {T}
    return diagm(-1 => x.dl, 0 => x.d, 1 => x.du)
end

for (AT, comp) in ((:LowerTriangular, "GE"), (:UpperTriangular, "LE"))
    uAT = Symbol(:Unit, AT)
    @eval begin
        function ReactantCore.materialize_traced_array(
            x::$(AT){TracedRNumber{T},<:AnyTracedRMatrix}
        ) where {T}
            m, n = size(x)
            px = materialize_traced_array(parent(x))
            row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
            col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
            indicator = Ops.compare(row_idxs, col_idxs; comparison_direction=($(comp)))
            return Ops.select(indicator, px, zero(px))
        end

        function ReactantCore.materialize_traced_array(
            x::$(uAT){TracedRNumber{T},<:AnyTracedRMatrix}
        ) where {T}
            m, n = size(x)
            px = materialize_traced_array(parent(x))
            row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
            col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
            nondiag_indicator = Ops.compare(row_idxs, col_idxs; comparison_direction="NE")
            x = materialize_traced_array($(AT)(px))
            return Ops.select(nondiag_indicator, x, one.(x))
        end
    end
end

function ReactantCore.materialize_traced_array(
    x::Symmetric{TracedRNumber{T},<:AnyTracedRMatrix}
) where {T}
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

function TracedUtils.set_mlir_data!(
    x::Diagonal{TracedRNumber{T},TracedRArray{T,1}}, data
) where {T}
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
        x::$(AT){TracedRNumber{T},<:AnyTracedRMatrix}, data
    ) where {T}
        tdata = TracedRArray{T}(data)
        z = zero(tdata)
        m, n = size(x)
        row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
        col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
        data_indicator = Ops.compare(row_idxs, col_idxs; comparison_direction=($(dcomp)))
        original_indicator = Ops.compare(
            row_idxs, col_idxs; comparison_direction=($(ocomp))
        )
        res = Ops.add(
            Ops.select(data_indicator, tdata, z), Ops.select(original_indicator, x.data, z)
        )
        set_mlir_data!(parent(x), res.mlir_data)
        return x
    end
end

function TracedUtils.set_mlir_data!(
    x::Symmetric{TracedRNumber{T},<:AnyTracedRMatrix}, data
) where {T}
    if x.uplo == 'L'
        set_mlir_data!(LowerTriangular(parent(x)), data)
    else
        set_mlir_data!(UpperTriangular(parent(x)), data)
    end
    return x
end

function TracedUtils.set_mlir_data!(
    x::Tridiagonal{TracedRNumber{T},<:AnyTracedRVector}, data
) where {T}
    tdata = TracedRArray{T}(data)
    set_mlir_data!(x.dl, materialize_traced_array(diag(tdata, -1)).mlir_data)
    set_mlir_data!(x.d, materialize_traced_array(diag(tdata, 0)).mlir_data)
    set_mlir_data!(x.du, materialize_traced_array(diag(tdata, 1)).mlir_data)
    return x
end

Reactant.aos_to_soa(x::Tridiagonal{TracedRNumber{T}}) where {T} = x

# Core functions
function overloaded_mul!(
    @nospecialize(C::TracedRArray{T,1}),
    @nospecialize(A::AbstractMatrix),
    @nospecialize(B::AbstractVector),
    α::Number=true,
    β::Number=false,
) where {T}
    # TODO: The reshape operations are not getting optimized, we should directly call
    #       dot_general
    rC = Ops.reshape(C, length(C), 1)
    overloaded_mul!(rC, A, reshape(B, :, 1), α, β)
    C.mlir_data = get_mlir_data(vec(rC))
    return C
end

function overloaded_mul!(
    @nospecialize(C::TracedRArray{T,2}),
    @nospecialize(A::AbstractMatrix),
    @nospecialize(B::AbstractVector),
    α::Number=true,
    β::Number=false,
) where {T}
    overloaded_mul!(C, A, reshape(B, :, 1), α, β)
    return C
end

function overloaded_mul!(
    @nospecialize(C::TracedRArray{T,2} where {T}),
    @nospecialize(A::AbstractMatrix),
    @nospecialize(B::AbstractMatrix),
    α::Number=true,
    β::Number=false,
)
    A = TracedUtils.promote_to(TracedRArray{unwrapped_eltype(A),2}, A)
    B = TracedUtils.promote_to(TracedRArray{unwrapped_eltype(B),2}, B)

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

    T = unwrapped_eltype(C)
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

    return Ops.gather_getindex(y, TracedUtils.promote_to(TracedRArray{Int,2}, indices))
end

function LinearAlgebra._diagm(
    shape, kv::Pair{<:Integer,<:AnyTracedRArray{T,1}}...
) where {T}
    m, n = LinearAlgebra.diagm_size(shape, kv...)

    # For repeated indices we need to aggregate the values
    kv_updated = Dict{Integer,AnyTracedRArray{T,1}}()
    for (k, v) in kv
        if haskey(kv_updated, k)
            kv_updated[k] = kv_updated[k] + v
        else
            kv_updated[k] = v
        end
    end

    scatter_indices = Matrix{Int64}[]
    concat_inputs = MLIR.IR.Value[]
    for (k, v) in pairs(kv_updated)
        push!(scatter_indices, diagonal_indices_zero_indexed(m, n, k)[1:length(v), :])
        push!(concat_inputs, get_mlir_data(v))
    end
    scatter_indices = Ops.constant(reduce(vcat, scatter_indices))
    values = TracedRArray{T,1}(
        (),
        MLIR.IR.result(MLIR.Dialects.stablehlo.concatenate(concat_inputs; dimension=0), 1),
        (size(scatter_indices, 1),),
    )
    return Ops.scatter_setindex(Ops.fill(zero(T), (m, n)), scatter_indices, values)
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

function LinearAlgebra.ldiv!(
    B::Union{AnyTracedRArray{T,1},AnyTracedRArray{T,2}}, D::Diagonal, A::AbstractVecOrMat
) where {T}
    LinearAlgebra.require_one_based_indexing(A, B)
    dd = D.diag
    d = length(dd)
    m, n = size(A, 1), size(A, 2)
    m′, n′ = size(B, 1), size(B, 2)
    m == d || throw(DimensionMismatch("right hand side has $m rows but D is $d by $d"))
    (m, n) == (m′, n′) ||
        throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    B .= dd .\ A
    # OG implementation below, we don't currently support the conditional throw exception
    #j = findfirst(iszero, D.diag)
    #isnothing(j) || throw(SingularException(j))
    #@inbounds for j = 1:n, i = 1:m
    #    B[i, j] = dd[i] \ A[i, j]
    #end
    return B
end

# Kronecker Product
function LinearAlgebra.kron(
    x::AnyTracedRVecOrMat{T1}, y::AnyTracedRVecOrMat{T2}
) where {T1,T2}
    x = materialize_traced_array(x)
    y = materialize_traced_array(y)
    z = similar(x, Base.promote_op(*, T1, T2), LinearAlgebra._kronsize(x, y))
    LinearAlgebra.kron!(z, x, y)
    return z
end

function LinearAlgebra.kron(x::AnyTracedRVector{T1}, y::AnyTracedRVector{T2}) where {T1,T2}
    x = materialize_traced_array(x)
    y = materialize_traced_array(y)
    z = similar(x, Base.promote_op(*, T1, T2), length(x) * length(y))
    LinearAlgebra.kron!(z, x, y)
    return z
end

function LinearAlgebra.kron!(C::AnyTracedRVector, A::AnyTracedRVector, B::AnyTracedRVector)
    LinearAlgebra.kron!(
        reshape(C, length(B), length(A)), reshape(A, 1, length(A)), reshape(B, length(B), 1)
    )
    return C
end

function LinearAlgebra._kron!(C::AnyTracedRMatrix, A::AnyTracedRMatrix, B::AnyTracedRMatrix)
    A = materialize_traced_array(A)
    B = materialize_traced_array(B)

    final_shape = Int64[size(B, 1), size(A, 1), size(B, 2), size(A, 2)]

    A = Ops.broadcast_in_dim(A, Int64[2, 4], final_shape)
    B = Ops.broadcast_in_dim(B, Int64[1, 3], final_shape)

    C_tmp = Ops.reshape(Ops.multiply(A, B), size(C)...)
    set_mlir_data!(C, get_mlir_data(C_tmp))

    return C
end

function LinearAlgebra._kron!(C::AnyTracedRMatrix, A::AnyTracedRVector, B::AnyTracedRMatrix)
    LinearAlgebra._kron!(C, reshape(A, length(A), 1), B)
    return C
end

function LinearAlgebra._kron!(C::AnyTracedRMatrix, A::AnyTracedRMatrix, B::AnyTracedRVector)
    LinearAlgebra._kron!(C, A, reshape(B, length(B), 1))
    return C
end

function LinearAlgebra.axpy!(α::Number, x::TracedRArray{T}, y::TracedRArray{T}) where {T}
    if length(x) != length(y)
        throw(
            DimensionMismatch(
                lazy"x has length $(length(x)), but y has length $(length(y))"
            ),
        )
    end
    ax = Ops.multiply(x, TracedUtils.broadcast_to_size(T(α), size(x)))

    set_mlir_data!(y, get_mlir_data(Ops.add(y, ax)))
    return y
end

function LinearAlgebra.axpby!(
    α::Number, x::TracedRArray{T}, β::Number, y::TracedRArray{T}
) where {T}
    if length(x) != length(y)
        throw(
            DimensionMismatch(
                lazy"x has length $(length(x)), but y has length $(length(y))"
            ),
        )
    end
    ax = Ops.multiply(x, TracedUtils.broadcast_to_size(T(α), size(x)))
    by = Ops.multiply(y, TracedUtils.broadcast_to_size(T(β), size(y)))

    set_mlir_data!(y, get_mlir_data(Ops.add(ax, by)))
    return y
end

# -------------
# TODO: The following currently drop several safety checks that are present in LinearAlgebra
#       Once we have auto if tracing we can remove them.

# Base.fill!
function Base.fill!(
    A::Union{
        Diagonal{<:TracedRNumber},
        Bidiagonal{<:TracedRNumber},
        Tridiagonal{<:TracedRNumber},
        SymTridiagonal{<:TracedRNumber},
    },
    x,
)
    xT = convert(eltype(A), x)
    LinearAlgebra.fillstored!(A, xT)
    return A
end

# Structured Broadcast
function Base.copyto!(
    dest::Union{
        Diagonal{<:TracedRNumber},
        Bidiagonal{<:TracedRNumber},
        Tridiagonal{<:TracedRNumber},
        SymTridiagonal{<:TracedRNumber},
        LowerTriangular{<:TracedRNumber},
        UpperTriangular{<:TracedRNumber},
    },
    bc::Broadcast.Broadcasted{<:LinearAlgebra.StructuredMatrixStyle},
)
    copyto!(dest, convert(Broadcast.Broadcasted{Nothing}, bc))
    return dest
end

#-------------

end
