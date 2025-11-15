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

include("Cholesky.jl")
include("LU.jl")
include("QR.jl")
include("SVD.jl")

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
