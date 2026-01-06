module TracedLinearAlgebra

using ..MLIR: MLIR
using ..Reactant: Reactant, Ops
using ..Reactant:
    TracedRArray, TracedRNumber, AnyTracedRArray, AnyTracedRMatrix, AnyTracedRVector
using ..Reactant: call_with_reactant, unwrapped_eltype, promote_to
using ReactantCore: ReactantCore, materialize_traced_array, @trace
using Reactant_jll: Reactant_jll

using ..TracedUtils: TracedUtils, get_mlir_data, set_mlir_data!
using ..Ops: @opcall

using LinearAlgebra: LinearAlgebra, BLAS
using LinearAlgebra: Adjoint, Transpose, Factorization, RowMaximum, NoPivot
using LinearAlgebra: SymTridiagonal, Symmetric, Bidiagonal, Diagonal, Tridiagonal
using LinearAlgebra: LowerTriangular, UnitLowerTriangular, UpperTriangular
using LinearAlgebra: I, diag, diagm, ldiv!, det, logabsdet, istriu, istril, triu!, tril!
using LinearAlgebra: inv!, rmul!, normalize
using LinearAlgebra: svd, lu
using Libdl: Libdl
using GPUArraysCore: @allowscalar

function __init__()
    if Reactant_jll.is_available()
        libblastrampoline_handle = Libdl.dlopen(BLAS.libblas)

        for (cname, enzymexla_name) in [
            # LU
            (BLAS.@blasfunc(sgetrf_), :enzymexla_lapack_sgetrf_),
            (BLAS.@blasfunc(dgetrf_), :enzymexla_lapack_dgetrf_),
            (BLAS.@blasfunc(cgetrf_), :enzymexla_lapack_cgetrf_),
            (BLAS.@blasfunc(zgetrf_), :enzymexla_lapack_zgetrf_),
            # SVD QR Iteration
            (BLAS.@blasfunc(sgesvd_), :enzymexla_lapack_sgesvd_),
            (BLAS.@blasfunc(dgesvd_), :enzymexla_lapack_dgesvd_),
            (BLAS.@blasfunc(cgesvd_), :enzymexla_lapack_cgesvd_),
            (BLAS.@blasfunc(zgesvd_), :enzymexla_lapack_zgesvd_),
            # SVD Divide and Conquer
            (BLAS.@blasfunc(sgesdd_), :enzymexla_lapack_sgesdd_),
            (BLAS.@blasfunc(dgesdd_), :enzymexla_lapack_dgesdd_),
            (BLAS.@blasfunc(cgesdd_), :enzymexla_lapack_cgesdd_),
            (BLAS.@blasfunc(zgesdd_), :enzymexla_lapack_zgesdd_),
            # SVD Jacobi
            (BLAS.@blasfunc(sgesvj_), :enzymexla_lapack_sgesvj_),
            (BLAS.@blasfunc(dgesvj_), :enzymexla_lapack_dgesvj_),
            (BLAS.@blasfunc(cgesvj_), :enzymexla_lapack_cgesvj_),
            (BLAS.@blasfunc(zgesvj_), :enzymexla_lapack_zgesvj_),
            # syrk
            (BLAS.@blasfunc(ssyrk_), :enzymexla_blas_ssyrk_),
            (BLAS.@blasfunc(dsyrk_), :enzymexla_blas_dsyrk_),
            (BLAS.@blasfunc(csyrk_), :enzymexla_blas_csyrk_),
            (BLAS.@blasfunc(zsyrk_), :enzymexla_blas_zsyrk_),
            # trmm
            (BLAS.@blasfunc(strmm_), :enzymexla_blas_strmm_),
            (BLAS.@blasfunc(dtrmm_), :enzymexla_blas_dtrmm_),
            (BLAS.@blasfunc(ctrmm_), :enzymexla_blas_ctrmm_),
            (BLAS.@blasfunc(ztrmm_), :enzymexla_blas_ztrmm_),
            # symm
            (BLAS.@blasfunc(ssymm_), :enzymexla_blas_ssymm_),
            (BLAS.@blasfunc(dsymm_), :enzymexla_blas_dsymm_),
            (BLAS.@blasfunc(csymm_), :enzymexla_blas_csymm_),
            (BLAS.@blasfunc(zsymm_), :enzymexla_blas_zsymm_),
        ]
            sym = Libdl.dlsym(libblastrampoline_handle, cname)
            @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
                enzymexla_name::Cstring, sym::Ptr{Cvoid}
            )::Cvoid
        end
    end

    return nothing
end

include("factorization/Factorization.jl")

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
    indicator = @opcall compare(
        row_idxs, col_idxs; comparison_direction=x.uplo == 'L' ? "GT" : "LT"
    )
    x_transposed = @opcall transpose(parent(x), [2, 1])
    return @opcall select(indicator, parent(x), x_transposed)
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

@static if isdefined(LinearAlgebra, :_triu)
    function LinearAlgebra._triu(A::AnyTracedRArray{T,2}, ::Val{true}, k::Integer) where {T}
        return overloaded_triu(materialize_traced_array(A), k)
    end
    function LinearAlgebra._triu(
        A::AnyTracedRArray{T,2}, ::Val{false}, k::Integer
    ) where {T}
        return overloaded_triu(materialize_traced_array(A), k)
    end
end

@static if isdefined(LinearAlgebra, :_tril)
    function LinearAlgebra._tril(A::AnyTracedRArray{T,2}, ::Val{true}, k::Integer) where {T}
        return overloaded_tril(materialize_traced_array(A), k)
    end
    function LinearAlgebra._tril(
        A::AnyTracedRArray{T,2}, ::Val{false}, k::Integer
    ) where {T}
        return overloaded_tril(materialize_traced_array(A), k)
    end
end

function LinearAlgebra.triu!(X::AnyTracedRArray{T,2}, k::Integer) where {T}
    set_mlir_data!(X, get_mlir_data(overloaded_triu(materialize_traced_array(X), k)))
    return X
end

function LinearAlgebra.tril!(X::AnyTracedRArray{T,2}, k::Integer) where {T}
    set_mlir_data!(X, get_mlir_data(overloaded_tril(materialize_traced_array(X), k)))
    return X
end

function overloaded_triu(X::TracedRArray{T,2}, k::Integer) where {T}
    iota_1 = @opcall iota(Int64, [size(X)...]; iota_dimension=1)
    iota_2 = @opcall subtract(
        @opcall(iota(Int64, [size(X)...]; iota_dimension=2)),
        Reactant.broadcast_to_size(k, size(X)),
    )
    idxs = @opcall compare(iota_1, iota_2; comparison_direction="LE")
    return @opcall select(idxs, X, zero(X))
end

function overloaded_tril(X::TracedRArray{T,2}, k::Integer) where {T}
    iota_1 = @opcall iota(Int64, [size(X)...]; iota_dimension=1)
    iota_2 = @opcall subtract(
        @opcall(iota(Int64, [size(X)...]; iota_dimension=2)),
        Reactant.broadcast_to_size(k, size(X)),
    )
    idxs = @opcall compare(iota_1, iota_2; comparison_direction="GE")
    return @opcall select(idxs, X, zero(X))
end

# LinearAlgebra defines norm with some conditionals which cannot be traced directly
function LinearAlgebra.norm(x::TracedRArray{T,N}, p::Real=2) where {T,N}
    isinf(p) && return maximum(abs, x)
    return mapreduce(Base.Fix2(^, p), +, x)^(T(1 / p))
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

# indexing into specific wrapepd array types
# TODO: specialize these ones. We don't need to make the arrays dense (though our passes
#       should be able to optimize them out)
for AT in (
    Bidiagonal{<:TracedRNumber},
    LowerTriangular{<:TracedRNumber},
    UpperTriangular{<:TracedRNumber},
    LinearAlgebra.Hermitian{<:TracedRNumber},
    SymTridiagonal{<:TracedRNumber},
    Tridiagonal{<:TracedRNumber},
    Symmetric{<:TracedRNumber},
    LinearAlgebra.UnitUpperTriangular{<:TracedRNumber},
    LinearAlgebra.UnitLowerTriangular{<:TracedRNumber},
    LinearAlgebra.UpperHessenberg{<:TracedRNumber},
)
    @eval function Base.getindex(A::$AT, i::Int, j::Int)
        return getindex(materialize_traced_array(A), i, j)
    end
end

LinearAlgebra._istriu(A::AnyTracedRMatrix, k) = all(iszero, overloaded_tril(A, k - 1))
LinearAlgebra._istril(A::AnyTracedRMatrix, k) = all(iszero, overloaded_triu(A, k + 1))

# Only needed because we lack automatic if tracing
function LinearAlgebra.det(A::AnyTracedRMatrix)
    @trace if istriu(A) || istril(A)
        _det = det(UpperTriangular(A))
    else
        _det = det(lu(A; check=false))
    end
    return _det
end

function LinearAlgebra.logabsdet(A::AnyTracedRMatrix)
    @trace if istriu(A) || istril(A)
        _logabsdet = logabsdet(UpperTriangular(A))
    else
        _logabsdet = logabsdet(lu(A; check=false))
    end
    return _logabsdet
end

function LinearAlgebra.logabsdet(
    A::Union{UpperTriangular{T,<:AnyTracedRMatrix},LowerTriangular{T,<:AnyTracedRMatrix}}
) where {T}
    d = LinearAlgebra.diag(A)
    sgn = prod(sign, d)
    abs_det = sum(log ∘ abs, d)
    return abs_det, sgn
end

function Base.inv(A::TracedRArray{T,2}) where {T} # don't overload Any* here
    LinearAlgebra.checksquare(A)
    @trace if istriu(A)
        Ai = triu!(parent(inv(UpperTriangular(A))))
    elseif istril(A)
        Ai = tril!(parent(inv(LowerTriangular(A))))
    else
        Ai = inv!(lu(A; check=false))
    end
    return Ai
end

for (wT, lower, ud) in (
    (:UpperTriangular, false, false),
    (:LowerTriangular, true, false),
    (:UnitUpperTriangular, false, true),
    (:UnitLowerTriangular, true, true),
)
    @eval function Base.inv(A::LinearAlgebra.$(wT){T,<:AnyTracedRMatrix}) where {T}
        S = typeof(inv(oneunit(Reactant.unwrapped_eltype(T))))
        rhs = Reactant.promote_to(TracedRArray{S,2}, LinearAlgebra.I(size(A, 1)))
        return @opcall triangular_solve(
            parent(A),
            rhs;
            left_side=false,
            lower=$(lower),
            transpose_a='N',
            unit_diagonal=$(ud),
        )
    end
end

function LinearAlgebra.cross(x::AnyTracedRVector, y::AbstractVector)
    return LinearAlgebra.cross(x, Reactant.promote_to(TracedRArray{eltype(y),1}, y))
end

function LinearAlgebra.cross(x::AbstractVector, y::AnyTracedRVector)
    return LinearAlgebra.cross(Reactant.promote_to(TracedRArray{eltype(x),1}, x), y)
end

function LinearAlgebra.cross(x::AnyTracedRVector, y::AnyTracedRVector)
    x_ = materialize_traced_array(x)
    y_ = materialize_traced_array(y)
    @allowscalar a1, a2, a3 = x_
    @allowscalar b1, b2, b3 = y_
    return Reactant.aos_to_soa([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])
end

function LinearAlgebra.issymmetric(A::AnyTracedRMatrix)
    axes(A, 1) == axes(A, 2) || return false
    return all(A .== transpose(A))
end

function LinearAlgebra.ishermitian(A::AnyTracedRMatrix)
    axes(A, 1) == axes(A, 2) || return false
    return all(A .== adjoint(A))
end

function LinearAlgebra.isbanded(A::AnyTracedRMatrix, kl::Integer, ku::Integer)
    return istriu(A, kl) & istril(A, ku)
end

function LinearAlgebra.normalize(a::AnyTracedRArray{T}, p::Real=2) where {T}
    nrm = LinearAlgebra.norm(a, p)
    if !isempty(a)
        aa = LinearAlgebra.copymutable_oftype(a, typeof(zero(T) / nrm))
        return LinearAlgebra.__normalize!(aa, nrm)
    else
        return typeof(zero(T) / nrm)[]
    end
end

@static if isdefined(LinearAlgebra, :__normalize!)
    function LinearAlgebra.__normalize!(a::AnyTracedRArray, nrm)
        # The largest positive floating point number whose inverse is less than infinity
        δ = inv(prevfloat(typemax(nrm)))
        @trace if nrm ≥ δ # Safe to multiply with inverse
            invnrm = inv(nrm)
            rmul!(a, invnrm)
        else # scale elements to avoid overflow
            εδ = eps(one(nrm)) / δ
            rmul!(a, εδ)
            rmul!(a, inv(nrm * εδ))
        end
        return a
    end
end

function LinearAlgebra.rmul!(A::AnyTracedRArray, b::Number)
    @. A *= b
    return A
end

function LinearAlgebra.lmul!(b::Number, A::AnyTracedRArray)
    @. A = b * A
    return A
end

function LinearAlgebra.rdiv!(A::AnyTracedRArray, b::Number)
    @. A /= b
    return A
end

function LinearAlgebra.ldiv!(b::Number, A::AnyTracedRArray)
    @. A = b \ A
    return A
end

# uniform scaling
function Base.:+(
    A::AnyTracedRMatrix{T1}, B::LinearAlgebra.UniformScaling{T2}
) where {T1,T2<:Number}
    m = LinearAlgebra.checksquare(A)
    return A + diagm(@opcall(fill(B.λ * oneunit(promote_type(T1, T2)), m)))
end

function Base.:+(
    A::LinearAlgebra.UniformScaling{T1}, B::AnyTracedRMatrix{T2}
) where {T1<:Number,T2}
    m = LinearAlgebra.checksquare(B)
    return diagm(@opcall(fill(A.λ * oneunit(promote_type(T1, T2)), m))) + B
end

function Base.:-(
    A::AnyTracedRMatrix{T1}, B::LinearAlgebra.UniformScaling{T2}
) where {T1,T2<:Number}
    m = LinearAlgebra.checksquare(A)
    return A - diagm(@opcall(fill(B.λ * oneunit(promote_type(T1, T2)), m)))
end

function Base.:-(
    A::LinearAlgebra.UniformScaling{T1}, B::AnyTracedRMatrix{T2}
) where {T1<:Number,T2}
    m = LinearAlgebra.checksquare(B)
    return diagm(@opcall(fill(A.λ * oneunit(promote_type(T1, T2)), m))) - B
end

end
