struct GeneralizedSVD{T,Tr,M<:AbstractArray,C<:AbstractArray} <: Factorization{T}
    U::M
    S::C
    Vt::M

    function GeneralizedSVD{T,Tr,M,C}(U::M, S::C, Vt::M) where {T,Tr,M,C}
        @assert ndims(S) == ndims(U) - 1
        return new{T,Tr,M,C}(U, S, Vt)
    end
end

function GeneralizedSVD(
    U::AbstractArray{T}, S::AbstractArray{Tr}, Vt::AbstractArray{T}
) where {T,Tr}
    return GeneralizedSVD{T,Tr,typeof(U),typeof(S)}(U, S, Vt)
end

function overloaded_svd(A::AbstractArray; kwargs...)
    return overloaded_svd(Reactant.promote_to(TracedRArray, A); kwargs...)
end

function overloaded_svd(
    A::AnyTracedRArray{T,N}; full::Bool=false, algorithm=nothing
) where {T,N}
    # TODO: don't ignore the algorithm kwarg
    U, S, Vt = @opcall svd(A; full)
    return GeneralizedSVD(U, S, Vt)
end

function overloaded_svd(
    A::AnyTracedRVector{T}; full::Bool=false, algorithm=nothing
) where {T}
    # TODO: don't ignore the algorithm kwarg
    normA = Reactant.call_with_reactant(LinearAlgebra.norm, A)
    U, S, Vt = if full
        ReactantCore.traced_if(
            iszero(normA), zeronorm_vector_svd_full, vector_svd_full, (A, normA)
        )
    else
        ReactantCore.traced_if(iszero(normA), zeronorm_vector_svd, vector_svd, (A, normA))
    end
    return GeneralizedSVD(U, S, Vt)
end

function zeronorm_vector_svd(A::AbstractVector{T}, normA) where {T}
    return zeronorm_vector_svd(A, false, normA)
end
function zeronorm_vector_svd_full(A::AbstractVector{T}, normA) where {T}
    return zeronorm_vector_svd(A, true, normA)
end

function zeronorm_vector_svd(A::AbstractVector{T}, full::Bool, normA) where {T}
    U = Reactant.promote_to(
        TracedRArray,
        Matrix{Reactant.unwrapped_eltype(T)}(
            LinearAlgebra.I, length(A), full ? length(A) : 1
        ),
    )
    return U, fill(normA, 1), ones(T, 1, 1)
end

vector_svd(A::AbstractVector{T}, normA) where {T} = vector_svd(A, false, normA)
function vector_svd_full(A::AbstractVector{T}, normA) where {T}
    return vector_svd(A, true, normA)
end

function vector_svd(A::AbstractVector{T}, full::Bool, normA) where {T}
    if !full
        U = materialize_traced_array(reshape(normalize(A), length(A), 1))
        return U, fill(normA, 1), ones(T, 1, 1)
    end
    return @opcall svd(materialize_traced_array(reshape(normalize(A), length(A), 1)); full)
end

# TODO: compute svdvals without computing the full svd. In principle we should
#       simple dce the U and Vt inside the compiler itself and simply compute Σ
LinearAlgebra.svdvals(x::AnyTracedRArray{T,N}) where {T,N} = overloaded_svd(x).S
LinearAlgebra.svdvals!(x::AnyTracedRArray{T,N}) where {T,N} = overloaded_svd(x).S
LinearAlgebra.svdvals(x::AnyTracedRVector{T}) where {T} = overloaded_svd(x).S
LinearAlgebra.svdvals!(x::AnyTracedRVector{T}) where {T} = overloaded_svd(x).S
