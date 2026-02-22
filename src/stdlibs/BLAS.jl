module BLASOverloads

using Reactant: Reactant, AnyTracedRArray
using ReactantCore: materialize_traced_array
using LinearAlgebra: LinearAlgebra, BLAS, dot

__get_indices(n::Integer, incx::Integer) = range(; length=n, step=incx, start=1)

function __extract_strided_view(X::AbstractArray, n::Integer, incx::Integer)
    return materialize_traced_array(vec(X))[__get_indices(n, incx)]
end

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
    return BLAS.dotc(n, vec(x), 1, vec(y), 1)
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
    return BLAS.dot(n, vec(x), 1, vec(y), 1)
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

BLAS.scal!(a::Number, x::AnyTracedRArray) = BLAS.scal!(length(x), a, x, 1)

end
