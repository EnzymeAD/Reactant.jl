abstract type BatchedFactorization{T} <: Factorization{T} end

function LinearAlgebra.TransposeFactorization(f::BatchedFactorization)
    return LinearAlgebra.TransposeFactorization{eltype(f),typeof(f)}(f)
end

function LinearAlgebra.AdjointFactorization(f::BatchedFactorization)
    return LinearAlgebra.AdjointFactorization{eltype(f),typeof(f)}(f)
end

const GeneralizedTransposeFactorization{T} =
    LinearAlgebra.TransposeFactorization{T,<:BatchedFactorization{T}} where {T}
const GeneralizedAdjointFactorization{T} =
    LinearAlgebra.AdjointFactorization{T,<:BatchedFactorization{T}} where {T}

include("Cholesky.jl")
include("LU.jl")
include("SVD.jl")

# Overload \ to support batched factorization
for FT in (
    :BatchedFactorization,
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

function _overloaded_backslash(F::BatchedFactorization, B::AbstractArray)
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
