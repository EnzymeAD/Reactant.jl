module TracedLinearAlgebra

using ..MLIR: MLIR
using ..Reactant: Reactant, Ops
using ..Reactant:
    TracedRArray, TracedRNumber, AnyTracedRArray, AnyTracedRMatrix, AnyTracedRVector
using ..Reactant: call_with_reactant
using ReactantCore: ReactantCore
using ReactantCore: materialize_traced_array
using Reactant_jll: Reactant_jll

using ..TracedUtils: TracedUtils, get_mlir_data, set_mlir_data!
using ..Ops: @opcall

using LinearAlgebra: LinearAlgebra, BLAS
using LinearAlgebra: Adjoint, Transpose, Factorization, RowMaximum
using LinearAlgebra: SymTridiagonal, Symmetric, Bidiagonal, Diagonal, Tridiagonal
using LinearAlgebra: LowerTriangular, UnitLowerTriangular, UpperTriangular
using LinearAlgebra: diag, diagm, dot, ldiv!
using Libdl: Libdl

function __init__()
    if Reactant_jll.is_available()
        libblastrampoline_handle = Libdl.dlopen(BLAS.libblas)

        for (cname, enzymexla_name) in [
            (BLAS.@blasfunc(sgetrf_), :enzymexla_lapack_sgetrf_),
            (BLAS.@blasfunc(dgetrf_), :enzymexla_lapack_dgetrf_),
            (BLAS.@blasfunc(cgetrf_), :enzymexla_lapack_cgetrf_),
            (BLAS.@blasfunc(zgetrf_), :enzymexla_lapack_zgetrf_),
        ]
            sym = Libdl.dlsym(libblastrampoline_handle, cname)
            @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
                enzymexla_name::Cstring, sym::Ptr{Cvoid}
            )::Cvoid
        end
    end

    return nothing
end

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
    return @opcall conj(
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
            x::LinearAlgebra.$(AT){TracedRNumber{T},<:AnyTracedRMatrix}
        ) where {T}
            m, n = size(x)
            px = materialize_traced_array(parent(x))
            row_idxs = @opcall iota(Int, [m, n]; iota_dimension=1)
            col_idxs = @opcall iota(Int, [m, n]; iota_dimension=2)
            indicator = @opcall compare(row_idxs, col_idxs; comparison_direction=$(comp))
            return @opcall select(indicator, px, zero(px))
        end

        function ReactantCore.materialize_traced_array(
            x::LinearAlgebra.$(uAT){TracedRNumber{T},<:AnyTracedRMatrix}
        ) where {T}
            m, n = size(x)
            px = materialize_traced_array(parent(x))
            row_idxs = @opcall iota(Int, [m, n]; iota_dimension=1)
            col_idxs = @opcall iota(Int, [m, n]; iota_dimension=2)
            nondiag_indicator = @opcall compare(
                row_idxs, col_idxs; comparison_direction="NE"
            )
            x = materialize_traced_array($(AT)(px))
            return @opcall select(nondiag_indicator, x, one.(x))
        end
    end
end

function ReactantCore.materialize_traced_array(
    x::Symmetric{TracedRNumber{T},<:AnyTracedRMatrix}
) where {T}
    m, n = size(x)
    row_idxs = @opcall iota(Int, [m, n]; iota_dimension=1)
    col_idxs = @opcall iota(Int, [m, n]; iota_dimension=2)
    if x.uplo == 'L'
        indicator = @opcall compare(row_idxs, col_idxs; comparison_direction="GT")
        x_lt = @opcall select(indicator, parent(x), zero(parent(x)))
        x_ltd = materialize_traced_array(LowerTriangular(parent(x)))
        return @opcall add(x_lt, @opcall(transpose(x_ltd, [2, 1])))
    else
        indicator = @opcall compare(row_idxs, col_idxs; comparison_direction="LT")
        x_ut = @opcall select(indicator, parent(x), zero(parent(x)))
        x_utd = materialize_traced_array(UpperTriangular(parent(x)))
        return @opcall add(@opcall(transpose(x_utd, [2, 1])), x_ut)
    end
end

function TracedUtils.set_mlir_data!(
    x::Transpose{TracedRNumber{T},TracedRArray{T,N}}, data
) where {T,N}
    tdata = TracedRArray{T}(data)
    px = parent(x)
    px.mlir_data = (
        if ndims(px) == 1
            @opcall reshape(tdata, length(tdata))
        else
            @opcall transpose(tdata, [2, 1])
        end
    ).mlir_data
    return x
end

function TracedUtils.set_mlir_data!(
    x::Adjoint{TracedRNumber{T},TracedRArray{T,N}}, data
) where {T,N}
    tdata = TracedRArray{T}(data)
    px = parent(x)
    transposed_data = if ndims(px) == 1
        @opcall(reshape(tdata, length(tdata)))
    else
        @opcall(transpose(tdata, [2, 1]))
    end
    px.mlir_data = (T <: Real ? transposed_data : @opcall(conj(transposed_data))).mlir_data
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
        x::LinearAlgebra.$(AT){TracedRNumber{T},<:AnyTracedRMatrix}, data
    ) where {T}
        tdata = TracedRArray{T}(data)
        z = zero(tdata)
        m, n = size(x)
        row_idxs = @opcall iota(Int, [m, n]; iota_dimension=1)
        col_idxs = @opcall iota(Int, [m, n]; iota_dimension=2)
        data_indicator = @opcall compare(row_idxs, col_idxs; comparison_direction=$(dcomp))
        original_indicator = @opcall compare(
            row_idxs, col_idxs; comparison_direction=$(ocomp)
        )
        res = @opcall add(
            @opcall(select(data_indicator, tdata, z)),
            @opcall(select(original_indicator, x.data, z)),
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
    rC = @opcall reshape(C, length(C), 1)
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
    A = call_with_reactant(Reactant.promote_to, TracedRArray, A)
    B = call_with_reactant(Reactant.promote_to, TracedRArray, B)

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

    T = Reactant.unwrapped_eltype(C)
    tmp = @opcall dot_general(
        T.(materialize_traced_array(A)),
        T.(materialize_traced_array(B));
        contracting_dimensions=([2], [1]),
    )

    res = if iszero(β)
        if isone(α)
            tmp
        else
            @opcall(multiply(tmp, Reactant.broadcast_to_size(T(α), size(C))))
        end
    else
        α_res = @opcall multiply(tmp, Reactant.broadcast_to_size(T(α), size(C)))
        β_C = @opcall multiply(C, Reactant.broadcast_to_size(T(β), size(C)))
        @opcall add(α_res, β_C)
    end
    set_mlir_data!(C, get_mlir_data(res))
    return C
end

function LinearAlgebra.triu!(@nospecialize(X::TracedRArray{T,2}), k::Integer) where {T}
    iota_1 = @opcall iota(Int64, [size(X)...]; iota_dimension=1)
    iota_2 = @opcall subtract(
        @opcall(iota(Int64, [size(X)...]; iota_dimension=2)),
        Reactant.broadcast_to_size(k, size(X)),
    )
    idxs = @opcall compare(iota_1, iota_2; comparison_direction="LE")
    X.mlir_data = @opcall(select(idxs, X, zero(X))).mlir_data
    return X
end

function LinearAlgebra.tril!(@nospecialize(X::TracedRArray{T,2}), k::Integer) where {T}
    iota_1 = @opcall iota(Int64, [size(X)...]; iota_dimension=1)
    iota_2 = @opcall subtract(
        @opcall(iota(Int64, [size(X)...]; iota_dimension=2)),
        Reactant.broadcast_to_size(k, size(X)),
    )
    idxs = @opcall compare(iota_1, iota_2; comparison_direction="GE")
    X.mlir_data = @opcall(select(idxs, X, zero(X))).mlir_data
    return X
end

# LinearAlgebra defines norm with some conditionals which cannot be traced directly
function LinearAlgebra.norm(x::TracedRArray{T,N}, p::Real=2) where {T,N}
    isinf(p) && return maximum(abs, x)
    return mapreduce(Base.Fix2(^, p), +, x)^(1 / p)
end

function LinearAlgebra._diagm(shape, kv::Pair{<:Integer,<:AnyTracedRVector}...)
    T = Reactant.unwrapped_eltype(last(first(kv)))
    m, n = LinearAlgebra.diagm_size(shape, kv...)

    # For repeated indices we need to aggregate the values
    kv_updated = Dict{Integer,AnyTracedRVector}()
    for (k, v) in kv
        if haskey(kv_updated, k)
            kv_updated[k] = kv_updated[k] + v
        else
            kv_updated[k] = v
        end
    end

    scatter_inds = TracedRArray{Int,2}[]
    concat_inputs = MLIR.IR.Value[]
    for (k, v) in pairs(kv_updated)
        ind = diagonal_indices(m, n, k, length(v))
        push!(scatter_inds, ind)
        push!(concat_inputs, get_mlir_data(v))
    end
    scatter_indices = @opcall concatenate(scatter_inds, 1)
    values = TracedRArray{T,1}(
        (),
        MLIR.IR.result(MLIR.Dialects.stablehlo.concatenate(concat_inputs; dimension=0), 1),
        (size(scatter_indices, 1),),
    )
    return @opcall scatter_setindex(@opcall(fill(zero(T), (m, n))), scatter_indices, values)
end

# Common Utilities
## The cartesian version doesn't exist in julia 1.10
function diagonal_indices(m::Integer, n::Integer, k::Integer, v::Integer)
    idx1, idx2 = 1 + max(0, -k), 1 + max(0, k)
    L = max(0, k ≤ 0 ? min(m + k, n) : min(m, n - k))
    L = min(L, v)

    if idx1 == idx2
        iota = @opcall iota(Int, [L, 2]; iota_dimension=1)
        op1 = @opcall add(iota, @opcall(fill(idx1, (L, 2))))
        return op1
    else
        iota = @opcall iota(Int, [L, 1]; iota_dimension=1)
        op1 = @opcall add(iota, @opcall(fill(idx1, (L, 1))))
        op2 = @opcall add(iota, @opcall(fill(idx2, (L, 1))))
        return @opcall concatenate([op1, op2], 2)
    end
end

function LinearAlgebra.ldiv!(
    B::Union{AnyTracedRArray{T,1},AnyTracedRArray{T,2}}, D::Diagonal, A::AbstractVecOrMat
) where {T}
    Base.require_one_based_indexing(A, B)
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
    x::Reactant.AnyTracedRVecOrMat{T1}, y::Reactant.AnyTracedRVecOrMat{T2}
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

    A = @opcall broadcast_in_dim(A, Int64[2, 4], final_shape)
    B = @opcall broadcast_in_dim(B, Int64[1, 3], final_shape)

    C_tmp = @opcall reshape(@opcall(multiply(A, B)), size(C)...)
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
    ax = @opcall multiply(x, Reactant.broadcast_to_size(T(α), size(x)))

    set_mlir_data!(y, get_mlir_data(@opcall add(y, ax)))
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
    ax = @opcall multiply(x, Reactant.broadcast_to_size(T(α), size(x)))
    by = @opcall multiply(y, Reactant.broadcast_to_size(T(β), size(y)))

    set_mlir_data!(y, get_mlir_data(@opcall add(ax, by)))
    return y
end

# -------------
# TODO: The following currently drop several safety checks that are present in LinearAlgebra
#       Once we have auto if tracing we can remove them.

for xType in (Any, TracedRNumber)
    # Base.fill!
    @eval function Base.fill!(
        A::Union{
            Diagonal{<:TracedRNumber},
            Bidiagonal{<:TracedRNumber},
            Tridiagonal{<:TracedRNumber},
            SymTridiagonal{<:TracedRNumber},
            LowerTriangular{<:TracedRNumber},
            UpperTriangular{<:TracedRNumber},
        },
        x::$(xType),
    )
        xT = convert(eltype(A), x)
        LinearAlgebra.fillstored!(A, xT)
        return A
    end
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

# LinearAlgebra overloads dot too many times for each of its types. Hence the overlay
function overloaded_dot(x::AbstractVector, y::AbstractVector)
    if length(x) != length(y)
        throw(
            DimensionMismatch(
                lazy"x has length $(length(x)), but y has length $(length(y))"
            ),
        )
    end
    res = @opcall dot_general(
        @opcall(conj(materialize_traced_array(x))),
        materialize_traced_array(y);
        contracting_dimensions=([1], [1]),
    )
    return TracedRNumber{Reactant.unwrapped_eltype(res)}((), res.mlir_data)
end

function overloaded_dot(x::AbstractArray, y::AbstractArray)
    return overloaded_dot(call_with_reactant(vec, x), call_with_reactant(vec, y))
end

function overloaded_dot(x::AbstractVector, A::AbstractMatrix, y::AbstractVector)
    return overloaded_dot(x, call_with_reactant(*, A, y))
end

# ldiv & rdiv interfaces
tfun_to_char(::typeof(identity)) = 'N'
tfun_to_char(::typeof(transpose)) = 'T'
tfun_to_char(::typeof(adjoint)) = 'C'

function LinearAlgebra.generic_trimatdiv!(
    C::AbstractVecOrMat{TracedRNumber{T}},
    uploc,
    isunitc,
    tfun::Function,
    A::AbstractMatrix,
    B::AbstractVecOrMat,
) where {T}
    @assert uploc in ('L', 'U')
    @assert isunitc in ('N', 'U')

    res = @opcall triangular_solve(
        Reactant.promote_to(TracedRArray{T}, A),
        Reactant.promote_to(TracedRArray{T}, B);
        left_side=true,
        lower=(uploc == 'L'),
        transpose_a=tfun_to_char(tfun),
        unit_diagonal=(isunitc == 'U'),
    )
    set_mlir_data!(C, get_mlir_data(res))
    return C
end

function LinearAlgebra.generic_trimatdiv!(
    C::AbstractVecOrMat{TracedRNumber{T}},
    uploc,
    isunitc,
    tfun::Function,
    A::Union{Adjoint,Transpose},
    B::AbstractVecOrMat,
) where {T}
    # our passes will simplify Adjoint/Transpose before triangular_solve
    return LinearAlgebra.generic_mattridiv!(
        C, uploc, isunitc, tfun, materialize_traced_array(A), B
    )
end

function LinearAlgebra.generic_mattridiv!(
    C::AbstractMatrix{TracedRNumber{T}},
    uploc,
    isunitc,
    tfun::Function,
    A::AbstractMatrix,
    B::AbstractMatrix,
) where {T}
    @assert uploc in ('L', 'U')
    @assert isunitc in ('N', 'U')

    res = @opcall triangular_solve(
        Reactant.promote_to(TracedRArray{T}, B),
        Reactant.promote_to(TracedRArray{T}, A);
        left_side=false,
        lower=(uploc == 'L'),
        transpose_a=tfun_to_char(tfun),
        unit_diagonal=(isunitc == 'U'),
    )
    set_mlir_data!(C, get_mlir_data(res))
    return C
end

function LinearAlgebra.generic_mattridiv!(
    C::AbstractMatrix{TracedRNumber{T}},
    uploc,
    isunitc,
    tfun::Function,
    A::AbstractMatrix,
    B::Union{Adjoint,Transpose},
) where {T}
    # our passes will simplify Adjoint/Transpose before triangular_solve
    return LinearAlgebra.generic_mattridiv!(
        C, uploc, isunitc, tfun, A, materialize_traced_array(B)
    )
end

LinearAlgebra.transpose!(B::AnyTracedRMatrix, A::AnyTracedRMatrix) = copy!(B, transpose(A))

LinearAlgebra.adjoint!(B::AnyTracedRMatrix, A::AnyTracedRMatrix) = copy!(B, adjoint(A))

# Supports batched factorization
abstract type GeneralizedFactorization{T} <: Factorization{T} end

function LinearAlgebra.TransposeFactorization(f::GeneralizedFactorization)
    return LinearAlgebra.TransposeFactorization{eltype(f),typeof(f)}(f)
end

function LinearAlgebra.AdjointFactorization(f::GeneralizedFactorization)
    return LinearAlgebra.AdjointFactorization{eltype(f),typeof(f)}(f)
end

const GeneralizedTransposeFactorization{T} =
    LinearAlgebra.TransposeFactorization{T,<:GeneralizedFactorization{T}} where {T}
const GeneralizedAdjointFactorization{T} =
    LinearAlgebra.AdjointFactorization{T,<:GeneralizedFactorization{T}} where {T}

# LU Factorization
struct GeneralizedLU{T,S<:AbstractArray,P<:AbstractArray,I<:Union{AbstractArray,Number}} <:
       GeneralizedFactorization{T}
    factors::S
    ipiv::P
    perm::P
    info::I
end

Base.ndims(lu::GeneralizedLU) = ndims(lu.factors)

function GeneralizedLU(factors::S, ipiv::P, perm::P, info::I) where {S,P,I}
    @assert ndims(ipiv) == ndims(perm) == ndims(factors) - 1
    @assert ndims(info) == ndims(factors) - 2
    return GeneralizedLU{eltype(factors),S,P,I}(factors, ipiv, perm, info)
end

function overloaded_lu(
    A::AnyTracedRArray{T,N}, ::RowMaximum; check::Bool=false, allowsingular::Bool=false
) where {T,N}
    # TODO: don't ignore the check and allowsingular flags
    # Batching here is in the last dimensions. `Ops.lu` expects the last dimensions
    permdims = vcat(collect(Int64, 3:N), 1, 2)
    A = @opcall transpose(materialize_traced_array(A), permdims)
    factors, ipiv, perm, info = @opcall lu(A)

    # Permute back to the original dimensions
    perm_perm = vcat(N - 1, collect(Int64, 1:(N - 2)))
    factors = @opcall transpose(factors, invperm(permdims))
    ipiv = @opcall transpose(ipiv, perm_perm)
    perm = @opcall transpose(perm, perm_perm)
    return GeneralizedLU(factors, ipiv, perm, info)
end

function LinearAlgebra.ldiv!(
    lu::GeneralizedLU{T,<:AbstractArray{T,N},P,I}, B::AbstractArray{T,M}
) where {T,P,I,N,M}
    @assert N == M + 1
    ldiv!(lu, reshape(B, size(B, 1), 1, size(B)[2:end]...))
    return B
end

function LinearAlgebra.ldiv!(
    lu::GeneralizedLU{T,<:AbstractArray{T,2},P,I}, B::AbstractArray{T,2}
) where {T,P,I}
    B .= _lu_solve_core(lu.factors, B, lu.perm)
    return B
end

function LinearAlgebra.ldiv!(
    lu::GeneralizedLU{T,<:AbstractArray{T,N},P,I}, B::AbstractArray{T,N}
) where {T,P,I,N}
    batch_shape = size(lu.factors)[3:end]
    @assert batch_shape == size(B)[3:end]

    permutation = vcat(collect(Int64, 3:N), 1, 2)

    factors = @opcall transpose(materialize_traced_array(lu.factors), permutation)
    B_permuted = @opcall transpose(materialize_traced_array(B), permutation)
    perm = @opcall transpose(
        materialize_traced_array(lu.perm), vcat(collect(Int64, 2:(N - 1)), 1)
    )

    res = @opcall transpose(
        only(
            @opcall(
                batch(
                    _lu_solve_core, [factors, B_permuted, perm], collect(Int64, batch_shape)
                )
            ),
        ),
        invperm(permutation),
    )
    B .= res
    return B
end

for f_wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization),
    aType in (:AbstractVecOrMat, :AbstractArray)

    @eval function LinearAlgebra.ldiv!(lu::$(f_wrapper){<:Any,<:GeneralizedLU}, B::$aType)
        # TODO: implement this
        error("`$(f_wrapper)` is not supported yet for LU.")
        return nothing
    end
end

function _lu_solve_core(factors::AbstractMatrix, B::AbstractMatrix, perm::AbstractVector)
    permuted_B = B[Int64.(perm), :]
    return UpperTriangular(factors) \ (UnitLowerTriangular(factors) \ permuted_B)
end

# Overload \ to support batched factorization
for FT in (
    :GeneralizedFactorization,
    :GeneralizedTransposeFactorization,
    :GeneralizedAdjointFactorization,
)
    for aType in (:AbstractVecOrMat, :AbstractArray)
        @eval Base.:(\)(F::$FT, B::$aType) = _overloaded_backslash(F, B)
    end

    @eval Base.:(\)(
        F::$FT{T}, B::Union{Array{Complex{T},1},Array{Complex{T},2}}
    ) where {T<:Union{Float32,Float64}} = _overloaded_backslash(F, B)
end

function _overloaded_backslash(F::GeneralizedFactorization, B::AbstractArray)
    return ldiv!(
        F, LinearAlgebra.copy_similar(B, typeof(oneunit(eltype(F)) \ oneunit(eltype(B))))
    )
end

function _overloaded_backslash(F::GeneralizedTransposeFactorization, B::AbstractArray)
    return conj!(adjoint(F.parent) \ conj.(B))
end

function _overloaded_backslash(F::GeneralizedAdjointFactorization, B::AbstractArray)
    return ldiv!(
        F, LinearAlgebra.copy_similar(B, typeof(oneunit(eltype(F)) \ oneunit(eltype(B))))
    )
end

end
