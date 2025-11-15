struct GeneralizedSVD{T,Tr,M<:AbstractArray{T},C<:AbstractArray{T}} <: Factorization{T}
    U::M
    S::C
    Vt::M

    function GeneralizedSVD{T,Tr,M,C}(U::M, S::C, Vt::M) where {T,Tr,M,C}
        @assert ndims(S) == ndims(U) - 1
        return new{T,Tr,M,C}(U, S, Vt)
    end
end

function overloaded_svd(A::AbstractArray; kwargs...)
    return overloaded_svd(Reactant.promote_to(TracedRArray, A); kwargs...)
end

function overloaded_svd(
    A::AnyTracedRArray{T,N}; full::Bool=false, algorithm=nothing
) where {T,N}
    # TODO: don't ignore the algorithm kwarg
    return error("TODO: Not implemented yet")
end

function overloaded_svd(
    A::AnyTracedRVector{T}; full::Bool=false, algorithm=nothing
) where {T}
    # TODO: don't ignore the algorithm kwarg
    m = length(A)
    normA = LinearAlgebra.norm(A)

    return error("TODO: Not implemented yet")
end

# TODO: compute svdvals without computing the full svd. In principle we should
#       simple dce the U and Vt inside the compiler itself and simply compute Σ
