module BLASOverloads

using Reactant: Reactant, AnyTracedRArray
using ReactantCore: materialize_traced_array
using LinearAlgebra: LinearAlgebra, BLAS, dot

# See https://github.com/EnzymeAD/Enzyme-JAX/issues/2141
function BLAS.asum(n::Integer, X::AnyTracedRArray{<:Real}, incx::Integer)
    X_ = materialize_traced_array(vec(X))[range(; length=n, step=incx, start=1)]
    return sum(abs, X_)
end

function BLAS.asum(n::Integer, X::AnyTracedRArray{<:Complex}, incx::Integer)
    X_ = materialize_traced_array(vec(X))[range(; length=n, step=incx, start=1)]
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
    return sum(
        materialize_traced_array(vec(x))[range(; length=n, step=incx, start=1)] .*
        materialize_traced_array(vec(y))[range(; length=n, step=incy, start=1)],
    )
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
        materialize_traced_array(conj(vec(x)))[range(; length=n, step=incx, start=1)] .*
        materialize_traced_array(vec(y))[range(; length=n, step=incy, start=1)],
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
    return sum(
        materialize_traced_array(vec(x))[range(; length=n, step=incx, start=1)] .*
        materialize_traced_array(vec(y))[range(; length=n, step=incy, start=1)],
    )
end

end
