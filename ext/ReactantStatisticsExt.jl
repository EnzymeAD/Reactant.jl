module ReactantStatisticsExt

using Reactant: AnyTracedRArray, AnyTracedRMatrix, AnyTracedRVector, TracedRNumber
using ReactantCore: materialize_traced_array
using GPUArraysCore: @allowscalar
using LinearAlgebra: diag, tril!, Diagonal
using Statistics: Statistics, middle

function Statistics._mean(f::F, A::AnyTracedRArray{T,N}, dims) where {F,T,N}
    denom = dims isa Colon ? length(A) : prod(Base.Fix1(size, A), Tuple(dims))
    return mapreduce(f, +, A; dims) / denom
end

function Statistics._var(
    A::AnyTracedRArray{T,N}, corrected::Bool, mean, ::Colon
) where {T,N}
    mean === nothing && (mean = Statistics.mean(A))
    denom = length(A) - corrected
    return mapreduce(abs2, +, A .- mean; dims=:) / denom
end

function Statistics._var(A::AnyTracedRArray{T,N}, corrected::Bool, mean, dims) where {T,N}
    mean === nothing && (mean = Statistics.mean(A; dims))
    denom = prod(Base.Fix1(size, A), Tuple(dims)) - corrected
    return mapreduce(abs2, +, A .- mean; dims) / denom
end

function Statistics.covzm(x::AnyTracedRMatrix, vardim::Int=1; corrected::Bool=true)
    C = Statistics.unscaled_covzm(materialize_traced_array(x), vardim)
    T = promote_type(typeof(@allowscalar(first(C)) / 1), eltype(C))
    A = convert(AbstractMatrix{T}, C)
    return A ./ (size(x, vardim) - corrected)
end

for (xT, yT) in Iterators.product(
    [AnyTracedRMatrix, AnyTracedRVector], [AnyTracedRMatrix, AnyTracedRVector]
)
    @eval function Statistics.covzm(x::$xT, y::$yT, vardim::Int=1; corrected::Bool=true)
        x = materialize_traced_array(x)
        y = materialize_traced_array(y)
        C = if ndims(x) == ndims(y) == 1
            Statistics.unscaled_covzm(x, y)
        else
            Statistics.unscaled_covzm(x, y, vardim)
        end
        if C isa TracedRNumber || C isa Number
            return C / (size(x, vardim) - corrected)
        end
        T = promote_type(typeof(@allowscalar(first(C)) / 1), eltype(C))
        A = convert(AbstractMatrix{T}, C)
        return A ./ (size(x, vardim) - corrected)
    end
end

function Statistics.corzm(x::AnyTracedRMatrix, vardim::Int=1)
    c = Statistics.unscaled_covzm(x, vardim)
    return Statistics.cov2cor!(c, sqrt.(diag(c)))
end

Statistics.clampcor(x::TracedRNumber{<:Real}) = clamp(x, -1, 1)

function Statistics.cov2cor!(C::AnyTracedRMatrix{T}, xsd::AnyTracedRArray) where {T}
    nx = length(xsd)
    size(C) == (nx, nx) || throw(DimensionMismatch("inconsistent dimensions"))
    C .= C ./ (xsd .* transpose(xsd))
    C .= Statistics.clampcor.(C)
    L = tril!(C, -1)
    C .= L .+ adjoint(L) .+ Diagonal(fill(oneunit(T), nx))
    return C
end
function Statistics.cov2cor!(C::AnyTracedRMatrix, xsd, ysd::AnyTracedRArray)
    nx, ny = size(C)
    length(ysd) == ny || throw(DimensionMismatch("inconsistent dimensions"))
    C .= Statistics.clampcor.(C ./ (xsd .* transpose(ysd)))
    return C
end
function Statistics.cov2cor!(C::AnyTracedRMatrix, xsd::AnyTracedRArray, ysd)
    nx, ny = size(C)
    length(xsd) == nx || throw(DimensionMismatch("inconsistent dimensions"))
    C .= Statistics.clampcor.(C ./ (xsd .* transpose(ysd)))
    return C
end
function Statistics.cov2cor!(
    C::AnyTracedRMatrix, xsd::AnyTracedRArray, ysd::AnyTracedRArray
)
    nx, ny = size(C)
    (length(xsd) == nx && length(ysd) == ny) ||
        throw(DimensionMismatch("inconsistent dimensions"))
    C .= Statistics.clampcor.(C ./ (xsd .* transpose(ysd)))
    return C
end

function Statistics._median(v::AnyTracedRArray{T}, ::Colon) where {T}
    return Statistics.median!(copy(materialize_traced_array(vec(v))))
end

function Statistics.median!(v::AnyTracedRVector)
    isempty(v) && throw(ArgumentError("median of an empty array is undefined, $(repr(v))"))
    inds = axes(v, 1)
    n = length(inds)
    mid = div(first(inds) + last(inds), 2)
    nan_res = convert(eltype(v), NaN)
    anynan = any(isnan, v)

    if isodd(n)
        return ifelse(anynan, nan_res, middle(partialsort!(v, mid)))
    else
        m = partialsort!(v, mid:(mid + 1))
        return ifelse(anynan, nan_res, middle(@allowscalar(m[1]), @allowscalar(m[2])))
    end
end

for (xT, yT) in Iterators.product(
    [AnyTracedRVector, AbstractVector], [AnyTracedRVector, AbstractVector]
)
    xT == AbstractVector && yT == AbstractVector && continue

    @eval function Statistics.corm(x::$(xT), mx, y::$(yT), my)
        n = length(x)
        length(y) == n || throw(DimensionMismatch("inconsistent lengths"))
        n > 0 || throw(ArgumentError("correlation only defined for non-empty vectors"))

        X = x .- mx
        Y = y .- my
        xx = mapreduce(abs2, +, X)
        yy = mapreduce(abs2, +, Y)
        xy = mapreduce(*, +, X, conj.(Y))

        return Statistics.clampcor(xy / max(xx, yy) / sqrt(min(xx, yy) / max(xx, yy)))
    end
end

end
