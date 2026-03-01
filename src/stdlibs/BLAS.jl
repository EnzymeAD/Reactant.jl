module BLASOverloads

using Reactant: Reactant, AnyTracedRArray, AnyTracedRVector, AnyTracedRMatrix, Ops
using ReactantCore: materialize_traced_array
using LinearAlgebra: LinearAlgebra, BLAS
using .Ops: @opcall
using LinearAlgebra:
    Symmetric,
    Hermitian,
    UpperTriangular,
    LowerTriangular,
    UnitUpperTriangular,
    UnitLowerTriangular

__get_indices(n::Integer, incx::Integer) = range(; length=n, step=incx, start=1)

function __extract_strided_view(X::AbstractArray, n::Integer, incx::Integer)
    return materialize_traced_array(vec(X))[__get_indices(n, incx)]
end

# Level 1

# Level 1

# See https://github.com/EnzymeAD/Enzyme-JAX/issues/2141
function BLAS.asum(n::Integer, X::AnyTracedRArray{<:Real}, incx::Integer)
    return sum(abs, __extract_strided_view(X, n, incx))
end

function BLAS.asum(n::Integer, X::AnyTracedRArray{<:Complex}, incx::Integer)
    X_ = __extract_strided_view(X, n, incx)
    return sum(abs, real(X_)) + sum(abs, imag(X_))
end

function BLAS.dotu(x::AnyTracedRArray{<:Complex}, y::AnyTracedRArray{<:Complex})
    n, m = length(x), length(y)
    n == m || throw(DimensionMismatch(lazy"dot product arguments have lengths $n and $m"))
    return BLAS.dotu(n, vec(x), 1, vec(y), 1)
end

function BLAS.dotu(
    n::Integer,
    x::AnyTracedRArray{<:Complex},
    incx::Integer,
    y::AnyTracedRArray{<:Complex},
    incy::Integer,
)
    return sum(__extract_strided_view(x, n, incx) .* __extract_strided_view(y, n, incy))
end

function BLAS.dotc(x::AnyTracedRArray{<:Complex}, y::AnyTracedRArray{<:Complex})
    n, m = length(x), length(y)
    n == m || throw(DimensionMismatch(lazy"dot product arguments have lengths $n and $m"))
    return BLAS.dotc(
        n, materialize_traced_array(vec(x)), 1, materialize_traced_array(vec(y)), 1
    )
end

function BLAS.dotc(
    n::Integer,
    x::AnyTracedRArray{<:Complex},
    incx::Integer,
    y::AnyTracedRArray{<:Complex},
    incy::Integer,
)
    return sum(
        conj(__extract_strided_view(x, n, incx)) .* __extract_strided_view(y, n, incy)
    )
end

function BLAS.dot(x::AnyTracedRArray{<:Real}, y::AnyTracedRArray{<:Real})
    n, m = length(x), length(y)
    n == m || throw(DimensionMismatch(lazy"dot product arguments have lengths $n and $m"))
    return BLAS.dot(
        n, materialize_traced_array(vec(x)), 1, materialize_traced_array(vec(y)), 1
    )
end

function BLAS.dot(
    n::Integer,
    x::AnyTracedRArray{<:Real},
    incx::Integer,
    y::AnyTracedRArray{<:Real},
    incy::Integer,
)
    return sum(__extract_strided_view(x, n, incx) .* __extract_strided_view(y, n, incy))
end

function BLAS.scal!(n::Integer, a::Number, x::AnyTracedRArray, incx::Integer)
    view(vec(x), __get_indices(n, incx)) .*= a
    return x
end

function BLAS.scal!(a::Number, x::AnyTracedRArray)
    return BLAS.scal!(length(x), a, materialize_traced_array(vec(x)), 1)
end

function BLAS.nrm2(n::Integer, X::AnyTracedRArray, incx::Integer)
    return LinearAlgebra.norm(__extract_strided_view(X, n, incx), 2)
end

BLAS.nrm2(x::AnyTracedRArray) = BLAS.nrm2(length(x), materialize_traced_array(vec(x)), 1)

function BLAS.blascopy!(
    n::Integer, x::AnyTracedRArray, incx::Integer, y::AnyTracedRArray, incy::Integer
)
    view(vec(y), __get_indices(n, incy)) .= __extract_strided_view(x, n, incx)
    return y
end

function BLAS.copy!(
    n::Integer, x::AnyTracedRArray, incx::Integer, y::AnyTracedRArray, incy::Integer
)
    return BLAS.blascopy!(n, x, incx, y, incy)
end

function BLAS.iamax(n::Integer, x::AnyTracedRArray, incx::Integer)
    return findmax(abs, __extract_strided_view(x, n, incx))[2]
end

function BLAS.iamax(x::AnyTracedRArray)
    return BLAS.iamax(length(x), materialize_traced_array(vec(x)), 1)
end

function BLAS.rot!(
    n::Integer,
    X::AnyTracedRArray,
    incx::Integer,
    Y::AnyTracedRArray,
    incy::Integer,
    c::Number,
    s::Number,
)
    x_indices = __get_indices(n, incx)
    y_indices = __get_indices(n, incy)
    x_view = view(vec(X), x_indices)
    y_view = view(vec(Y), y_indices)

    x_old = copy(x_view)
    x_view .= c .* x_view .+ s .* y_view
    y_view .= -conj(s) .* x_old .+ c .* y_view
    return X, Y
end

function BLAS.rot!(X::AnyTracedRArray, Y::AnyTracedRArray, c::Number, s::Number)
    length(X) == length(Y) || throw(
        DimensionMismatch(lazy"rot! arguments have lengths $(length(X)) and $(length(Y))"),
    )
    return BLAS.rot!(
        length(X),
        materialize_traced_array(vec(X)),
        1,
        materialize_traced_array(vec(Y)),
        1,
        c,
        s,
    )
end

# Level 2

function BLAS.gemv!(
    trans::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    x::AnyTracedRVector,
    beta::Number,
    y::AnyTracedRVector,
)
    A_ = (trans == 'N') ? A : (trans == 'T' ? transpose(A) : adjoint(A))
    LinearAlgebra.mul!(y, A_, x, alpha, beta)
    return y
end

function BLAS.gemv(trans::Char, alpha::Number, A::AnyTracedRMatrix, x::AnyTracedRVector)
    A_ = (trans == 'N') ? A : (trans == 'T' ? transpose(A) : adjoint(A))
    return alpha .* (A_ * x)
end

function BLAS.gemv(trans::Char, A::AnyTracedRMatrix, x::AnyTracedRVector)
    return BLAS.gemv(trans, true, A, x)
end

function BLAS.ger!(
    alpha::Number, x::AnyTracedRVector, y::AnyTracedRVector, A::AnyTracedRMatrix
)
    A .+= alpha .* (x * y')
    return A
end

@static if isdefined(BLAS, :geru!)
    function BLAS.geru!(
        alpha::Number, x::AnyTracedRVector, y::AnyTracedRVector, A::AnyTracedRMatrix
    )
        A .+= alpha .* (x * transpose(y))
        return A
    end
end

function BLAS.symv(uplo::Char, alpha::Number, A::AnyTracedRMatrix, x::AnyTracedRVector)
    return alpha .* (Symmetric(A, Symbol(uplo)) * x)
end

function BLAS.symv(uplo::Char, A::AnyTracedRMatrix, x::AnyTracedRVector)
    return BLAS.symv(uplo, true, A, x)
end

function BLAS.hemv(uplo::Char, alpha::Number, A::AnyTracedRMatrix, x::AnyTracedRVector)
    return alpha .* (Hermitian(A, Symbol(uplo)) * x)
end

function BLAS.hemv(uplo::Char, A::AnyTracedRMatrix, x::AnyTracedRVector)
    return BLAS.hemv(uplo, true, A, x)
end

function BLAS.syr!(uplo::Char, alpha::Number, x::AnyTracedRVector, A::AnyTracedRMatrix)
    A = materialize_traced_array(A)
    x = materialize_traced_array(x)
    res = alpha .* (x * transpose(x)) .+ A
    if uplo == 'U'
        UpperTriangular(A) .= UpperTriangular(res)
    else
        LowerTriangular(A) .= LowerTriangular(res)
    end
    return A
end

function BLAS.her!(uplo::Char, alpha::Number, x::AnyTracedRVector, A::AnyTracedRMatrix)
    A = materialize_traced_array(A)
    x = materialize_traced_array(x)
    res = alpha .* (x * x') .+ A
    if uplo == 'U'
        Hermitian(A, :U) .= Hermitian(res, :U)
    else
        Hermitian(A, :L) .= Hermitian(res, :L)
    end
    return A
end

function BLAS.trmv(
    uplo::Char, trans::Char, diag::Char, A::AnyTracedRMatrix, x::AnyTracedRVector
)
    return BLAS.trmv!(uplo, trans, diag, A, copy(x))
end

function BLAS.symv!(
    uplo::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    x::AnyTracedRVector,
    beta::Number,
    y::AnyTracedRVector,
)
    LinearAlgebra.mul!(
        y, Symmetric(materialize_traced_array(A), Symbol(uplo)), x, alpha, beta
    )
    return y
end

function BLAS.hemv!(
    uplo::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    x::AnyTracedRVector,
    beta::Number,
    y::AnyTracedRVector,
)
    LinearAlgebra.mul!(
        y, Hermitian(materialize_traced_array(A), Symbol(uplo)), x, alpha, beta
    )
    return y
end

function BLAS.trmv!(
    uplo::Char, trans::Char, diag::Char, A::AnyTracedRMatrix, x::AnyTracedRVector
)
    A = materialize_traced_array(A)
    x = materialize_traced_array(x)
    A_ = (uplo == 'U') ? UpperTriangular(A) : LowerTriangular(A)
    if diag == 'U'
        A_ = (uplo == 'U') ? UnitUpperTriangular(A) : UnitLowerTriangular(A)
    end
    A_op = (trans == 'N') ? A_ : (trans == 'T' ? transpose(A_) : adjoint(A_))
    copyto!(x, A_op * x)
    return x
end

function BLAS.trsv!(
    uplo::Char, trans::Char, diag::Char, A::AnyTracedRMatrix, x::AnyTracedRVector
)
    res = @opcall triangular_solve(
        materialize_traced_array(A),
        materialize_traced_array(x);
        left_side=true,
        lower=(uplo == 'L'),
        transpose_a=trans,
        unit_diagonal=(diag == 'U'),
    )
    copyto!(x, res)
    return x
end

function BLAS.trsv(
    uplo::Char, trans::Char, diag::Char, A::AnyTracedRMatrix, x::AnyTracedRVector
)
    return BLAS.trsv!(uplo, trans, diag, A, copy(x))
end

# Level 3

function BLAS.gemm!(
    transA::Char,
    transB::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    B::AnyTracedRMatrix,
    beta::Number,
    C::AnyTracedRMatrix,
)
    A = materialize_traced_array(A)
    B = materialize_traced_array(B)
    A_ = (transA == 'N') ? A : (transA == 'T' ? transpose(A) : adjoint(A))
    B_ = (transB == 'N') ? B : (transB == 'T' ? transpose(B) : adjoint(B))
    LinearAlgebra.mul!(C, A_, B_, alpha, beta)
    return C
end

function BLAS.gemm(
    transA::Char, transB::Char, alpha::Number, A::AnyTracedRMatrix, B::AnyTracedRMatrix
)
    A = materialize_traced_array(A)
    B = materialize_traced_array(B)
    A_ = (transA == 'N') ? A : (transA == 'T' ? transpose(A) : adjoint(A))
    B_ = (transB == 'N') ? B : (transB == 'T' ? transpose(B) : adjoint(B))
    return alpha .* (A_ * B_)
end

function BLAS.gemm(transA::Char, transB::Char, A::AnyTracedRMatrix, B::AnyTracedRMatrix)
    return BLAS.gemm(transA, transB, one(eltype(A)), A, B)
end

@static if isdefined(BLAS, :gemmt!)
    function BLAS.gemmt!(
        uplo::Char,
        transA::Char,
        transB::Char,
        alpha::Number,
        A::AnyTracedRMatrix,
        B::AnyTracedRMatrix,
        beta::Number,
        C::AnyTracedRMatrix,
    )
        A = materialize_traced_array(A)
        B = materialize_traced_array(B)
        C = materialize_traced_array(C)
        A_op = (transA == 'N') ? A : (transA == 'T' ? transpose(A) : adjoint(A))
        B_op = (transB == 'N') ? B : (transB == 'T' ? transpose(B) : adjoint(B))
        res = alpha .* (A_op * B_op) .+ beta .* C
        if uplo == 'U'
            UpperTriangular(C) .= UpperTriangular(res)
        else
            LowerTriangular(C) .= LowerTriangular(res)
        end
        return C
    end
end

@static if isdefined(BLAS, :gemmt)
    function BLAS.gemmt(
        uplo::Char,
        transA::Char,
        transB::Char,
        alpha::Number,
        A::AnyTracedRMatrix,
        B::AnyTracedRMatrix,
    )
        T = promote_type(eltype(A), eltype(B), typeof(alpha))
        n = (uplo == 'U' || uplo == 'L') ? size(A, (transA == 'N' ? 1 : 2)) : 0
        C = zeros(T, n, n)
        return BLAS.gemmt!(uplo, transA, transB, alpha, A, B, zero(T), C)
    end

    function BLAS.gemmt(
        uplo::Char, transA::Char, transB::Char, A::AnyTracedRMatrix, B::AnyTracedRMatrix
    )
        return BLAS.gemmt(uplo, transA, transB, true, A, B)
    end
end

function BLAS.symm!(
    side::Char,
    uplo::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    B::AnyTracedRMatrix,
    beta::Number,
    C::AnyTracedRMatrix,
)
    As = Symmetric(A, Symbol(uplo))
    if side == 'L'
        LinearAlgebra.mul!(C, As, B, alpha, beta)
    else
        LinearAlgebra.mul!(C, B, As, alpha, beta)
    end
    return C
end

function BLAS.symm(
    side::Char, uplo::Char, alpha::Number, A::AnyTracedRMatrix, B::AnyTracedRMatrix
)
    T = promote_type(eltype(A), eltype(B), typeof(alpha))
    C = zeros(T, size(B)...)
    return BLAS.symm!(side, uplo, alpha, A, B, zero(T), C)
end

function BLAS.symm(side::Char, uplo::Char, A::AnyTracedRMatrix, B::AnyTracedRMatrix)
    return BLAS.symm(side, uplo, true, A, B)
end

function BLAS.hemm!(
    side::Char,
    uplo::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    B::AnyTracedRMatrix,
    beta::Number,
    C::AnyTracedRMatrix,
)
    As = Hermitian(A, Symbol(uplo))
    if side == 'L'
        LinearAlgebra.mul!(C, As, B, alpha, beta)
    else
        LinearAlgebra.mul!(C, B, As, alpha, beta)
    end
    return C
end

function BLAS.hemm(
    side::Char, uplo::Char, alpha::Number, A::AnyTracedRMatrix, B::AnyTracedRMatrix
)
    T = promote_type(eltype(A), eltype(B), typeof(alpha))
    C = zeros(T, size(B)...)
    return BLAS.hemm!(side, uplo, alpha, A, B, zero(T), C)
end

function BLAS.hemm(side::Char, uplo::Char, A::AnyTracedRMatrix, B::AnyTracedRMatrix)
    return BLAS.hemm(side, uplo, true, A, B)
end

function BLAS.trmm!(
    side::Char,
    uplo::Char,
    transa::Char,
    diag::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    B::AnyTracedRMatrix,
)
    A_ = (uplo == 'U') ? UpperTriangular(A) : LowerTriangular(A)
    if diag == 'U'
        A_ = (uplo == 'U') ? UnitUpperTriangular(A) : UnitLowerTriangular(A)
    end
    A_op = (transa == 'N') ? A_ : (transa == 'T' ? transpose(A_) : adjoint(A_))

    if side == 'L'
        copyto!(B, alpha .* (A_op * B))
    else
        copyto!(B, alpha .* (B * A_op))
    end
    return B
end

function BLAS.trmm(
    side::Char,
    uplo::Char,
    transa::Char,
    diag::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    B::AnyTracedRMatrix,
)
    return BLAS.trmm!(side, uplo, transa, diag, alpha, A, copy(B))
end

function BLAS.trsm!(
    side::Char,
    uplo::Char,
    transA::Char,
    diag::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    B::AnyTracedRMatrix,
)
    res = @opcall triangular_solve(
        materialize_traced_array(A),
        materialize_traced_array(B);
        left_side=(side == 'L'),
        lower=(uplo == 'L'),
        transpose_a=transA,
        unit_diagonal=(diag == 'U'),
    )
    copyto!(B, alpha .* res)
    return B
end

function BLAS.trsm(
    side::Char,
    uplo::Char,
    transA::Char,
    diag::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    B::AnyTracedRMatrix,
)
    return BLAS.trsm!(side, uplo, transA, diag, alpha, A, copy(B))
end

function BLAS.syrk!(
    uplo::Char,
    trans::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    beta::Number,
    C::AnyTracedRMatrix,
)
    res = @opcall syrk(
        materialize_traced_array(A),
        materialize_traced_array(C),
        alpha,
        beta;
        uplo=uplo,
        transpose_a=trans,
    )
    copyto!(C, res)
    return C
end

function BLAS.syrk(uplo::Char, trans::Char, alpha::Number, A::AnyTracedRMatrix)
    T = eltype(A)
    n = (trans == 'N') ? size(A, 1) : size(A, 2)
    C = zeros(T, n, n)
    return BLAS.syrk!(uplo, trans, alpha, A, zero(T), C)
end

function BLAS.syr2k!(
    uplo::Char,
    trans::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    B::AnyTracedRMatrix,
    beta::Number,
    C::AnyTracedRMatrix,
)
    A = materialize_traced_array(A)
    B = materialize_traced_array(B)
    C = materialize_traced_array(C)
    A_op = (trans == 'N') ? A : (trans == 'T' ? transpose(A) : adjoint(A))
    B_op = (trans == 'N') ? B : (trans == 'T' ? transpose(B) : adjoint(B))
    res =
        alpha .* (A_op * transpose(B_op)) .+ alpha .* (B_op * transpose(A_op)) .+ beta .* C
    if uplo == 'U'
        UpperTriangular(C) .= UpperTriangular(res)
    else
        LowerTriangular(C) .= LowerTriangular(res)
    end
    return C
end

function BLAS.syr2k(
    uplo::Char, trans::Char, alpha::Number, A::AnyTracedRMatrix, B::AnyTracedRMatrix
)
    T = promote_type(eltype(A), eltype(B), typeof(alpha))
    n = (trans == 'N') ? size(A, 1) : size(A, 2)
    C = zeros(T, n, n)
    return BLAS.syr2k!(uplo, trans, alpha, A, B, zero(T), C)
end

function BLAS.syr2k(uplo::Char, trans::Char, A::AnyTracedRMatrix, B::AnyTracedRMatrix)
    return BLAS.syr2k(uplo, trans, true, A, B)
end

function BLAS.herk!(
    uplo::Char,
    trans::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    beta::Number,
    C::AnyTracedRMatrix,
)
    res = alpha .* (trans == 'N' ? A * adjoint(A) : adjoint(A) * A) .+ beta .* C
    if uplo == 'U'
        Hermitian(C, :U) .= Hermitian(res, :U)
    else
        Hermitian(C, :L) .= Hermitian(res, :L)
    end
    return C
end

function BLAS.herk(uplo::Char, trans::Char, alpha::Number, A::AnyTracedRMatrix)
    T = eltype(A)
    n = (trans == 'N') ? size(A, 1) : size(A, 2)
    C = zeros(T, n, n)
    return BLAS.herk!(uplo, trans, alpha, A, zero(T), C)
end

function BLAS.her2k!(
    uplo::Char,
    trans::Char,
    alpha::Number,
    A::AnyTracedRMatrix,
    B::AnyTracedRMatrix,
    beta::Number,
    C::AnyTracedRMatrix,
)
    A = materialize_traced_array(A)
    B = materialize_traced_array(B)
    C = materialize_traced_array(C)
    A_op = (trans == 'N') ? A : (trans == 'T' ? transpose(A) : adjoint(A))
    B_op = (trans == 'N') ? B : (trans == 'T' ? transpose(B) : adjoint(B))
    res = alpha .* (A_op * B_op') .+ conj(alpha) .* (B_op * A_op') .+ beta .* C
    if uplo == 'U'
        Hermitian(C, :U) .= Hermitian(res, :U)
    else
        Hermitian(C, :L) .= Hermitian(res, :L)
    end
    return C
end

function BLAS.her2k(
    uplo::Char, trans::Char, alpha::Number, A::AnyTracedRMatrix, B::AnyTracedRMatrix
)
    T = promote_type(eltype(A), eltype(B), typeof(alpha))
    n = (trans == 'N') ? size(A, 1) : size(A, 2)
    C = zeros(T, n, n)
    return BLAS.her2k!(uplo, trans, alpha, A, B, zero(T), C)
end

function BLAS.her2k(uplo::Char, trans::Char, A::AnyTracedRMatrix, B::AnyTracedRMatrix)
    return BLAS.her2k(uplo, trans, true, A, B)
end

end
