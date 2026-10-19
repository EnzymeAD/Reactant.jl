abstract type BatchedFactorization{T} <: Factorization{T} end

function LinearAlgebra.TransposeFactorization(f::BatchedFactorization)
    return LinearAlgebra.TransposeFactorization{eltype(f),typeof(f)}(f)
end

function LinearAlgebra.AdjointFactorization(f::BatchedFactorization)
    return LinearAlgebra.AdjointFactorization{eltype(f),typeof(f)}(f)
end

const BatchedTransposeFactorization{T} =
    LinearAlgebra.TransposeFactorization{T,<:BatchedFactorization{T}} where {T}
const BatchedAdjointFactorization{T} =
    LinearAlgebra.AdjointFactorization{T,<:BatchedFactorization{T}} where {T}

include("Cholesky.jl")
include("LU.jl")
include("SVD.jl")

# Overload \ to support batched factorization
for FT in
    (:BatchedFactorization, :BatchedTransposeFactorization, :BatchedAdjointFactorization)
    for aType in (:AbstractVecOrMat, :AbstractArray)
        @eval Base.:(\)(F::$FT, B::$aType) = _overloaded_backslash(F, B)
    end

    @eval Base.:(\)(
        F::$FT{T}, B::Union{Array{Complex{T},1},Array{Complex{T},2}}
    ) where {T<:Union{Float32,Float64}} = _overloaded_backslash(F, B)
end

function __get_B(F::Factorization, B::AbstractArray)
    m, n = size(F, 1), size(F, 2)
    if m != size(B, 1)
        throw(DimensionMismatch("arguments must have the same number of rows"))
    end

    TFB = typeof(oneunit(eltype(F)) \ oneunit(eltype(B)))

    BB = similar(B, TFB, max(size(B, 1), n), size(B)[2:end]...)
    if n > size(B, 1)
        BB[1:m, ntuple(Returns(Colon()), ndims(B) - 1)...] = B
    else
        copyto!(BB, B)
    end

    return BB
end

function _overloaded_backslash(F::BatchedFactorization, B::AbstractArray)
    BB = __get_B(F, B)
    ldiv!(F, BB)
    return BB[1:size(F, 2), ntuple(Returns(Colon()), ndims(B) - 1)...]
end

function _overloaded_backslash(F::BatchedTransposeFactorization, B::AbstractArray)
    return conj!(adjoint(F.parent) \ conj.(B))
end

function _overloaded_backslash(F::BatchedAdjointFactorization, B::AbstractArray)
    BB = __get_B(F, B)
    ldiv!(F, BB)
    return BB[1:size(F)[2], ntuple(Returns(Colon()), ndims(B) - 1)...]
end

function Base.:(\)(
    A::TracedRArray{<:Any,2}, B::Union{TracedRArray{<:Any,2},TracedRArray{<:Any,1}}
)
    m, n = size(A)
    if m == n
        A_is_tril = istril(A)
        A_is_triu = istriu(A)
        @trace if A_is_tril & A_is_triu
            res = Diagonal(A) \ B
        elseif A_is_tril
            res = LowerTriangular(A) \ B
        elseif A_is_triu
            res = UpperTriangular(A) \ B
        else
            res = lu(A) \ B
        end
        return res
    end
    return qr(A, ColumnNorm()) \ B
end

function Base.:(\)(D::Diagonal{<:TracedRNumber}, B::AnyTracedRVector)
    # TODO: errors
    # j = findfirst(iszero, D.diag)
    # isnothing(j) || throw(SingularException(j))
    return D.diag .\ B
end
