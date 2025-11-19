struct BatchedSVD{T,Tr,M<:AbstractArray,C<:AbstractArray} <: Factorization{T}
    U::M
    S::C
    Vt::M

    function BatchedSVD{T,Tr,M,C}(U::M, S::C, Vt::M) where {T,Tr,M,C}
        @assert ndims(S) == ndims(U) - 1
        return new{T,Tr,M,C}(U, S, Vt)
    end
end

function BatchedSVD(U::M, S::C, Vt::M) where {M,C}
    @assert ndims(S) == ndims(U) - 1
    return BatchedSVD{eltype(U),eltype(S),M,C}(U, S, Vt)
end

struct DefaultEnzymeXLASVDAlgorithm <: LinearAlgebra.Algorithm end
struct JacobiAlgorithm <: LinearAlgebra.Algorithm end

_jlalg_to_enzymexla_alg(::Nothing) = "DEFAULT"
_jlalg_to_enzymexla_alg(alg::DefaultEnzymeXLASVDAlgorithm) = "DEFAULT"
_jlalg_to_enzymexla_alg(alg::LinearAlgebra.DivideAndConquer) = "DivideAndConquer"
_jlalg_to_enzymexla_alg(alg::LinearAlgebra.QRIteration) = "QRIteration"
_jlalg_to_enzymexla_alg(alg::JacobiAlgorithm) = "Jacobi"
_jlalg_to_enzymexla_alg(alg::String) = alg
_jlalg_to_enzymexla_alg(alg::Symbol) = _jlalg_to_enzymexla_alg(string(alg))
_jlalg_to_enzymexla_alg(alg) = error("Unsupported SVD algorithm: $alg")

# default relies on the backend to select the best algorithm
LinearAlgebra.default_svd_alg(::AnyTracedRArray) = DefaultEnzymeXLASVDAlgorithm()

function overloaded_svd(A::AbstractArray; kwargs...)
    return overloaded_svd(Reactant.promote_to(TracedRArray, A); kwargs...)
end

function overloaded_svd(
    A::AnyTracedRArray{T,N}; full::Bool=false, algorithm=LinearAlgebra.default_svd_alg(A)
) where {T,N}
    # Batching here is in the last dimensions. `Ops.svd` expects the last dimensions
    permdims = vcat(collect(Int64, 3:N), 1, 2)
    A = @opcall transpose(materialize_traced_array(A), permdims)

    U, S, Vt = @opcall svd(A; full, algorithm=_jlalg_to_enzymexla_alg(algorithm))

    # Permute back to the original dimensions
    S_perm = vcat(N - 1, collect(Int64, 1:(N - 2)))

    U = @opcall transpose(U, invperm(permdims))
    S = @opcall transpose(S, S_perm)
    Vt = @opcall transpose(Vt, invperm(permdims))

    return BatchedSVD(U, S, Vt)
end

struct __InnerVectorSVDDispatch{A} <: Function
    full::Bool
    algorithm::A
end

struct __ZeroNormVectorSVDDispatch{A} <: Function
    full::Bool
    algorithm::A
end

function overloaded_svd(
    A::AnyTracedRVector; full::Bool=false, algorithm=LinearAlgebra.default_svd_alg(A)
)
    normA = Reactant.call_with_reactant(LinearAlgebra.norm, A)
    U, S, Vt = ReactantCore.traced_if(
        iszero(normA),
        __ZeroNormVectorSVDDispatch(full, algorithm),
        __InnerVectorSVDDispatch(full, algorithm),
        (A, normA),
    )
    return BatchedSVD(U, S, Vt)
end

function (fn::__ZeroNormVectorSVDDispatch)(A::AbstractVector{T}, normA) where {T}
    U = promote_to(
        TracedRArray, Matrix{unwrapped_eltype(A)}(I, length(A), fn.full ? length(A) : 1)
    )
    return U, fill(normA, 1), ones(T, 1, 1)
end

function (fn::__InnerVectorSVDDispatch)(A::AbstractVector{T}, normA) where {T}
    if !fn.full
        normalizedA = normalize(A)
        U = materialize_traced_array(reshape(normalizedA, length(A), 1))
        return U, fill(normA, 1), ones(T, 1, 1)
    end
    (; U, S, Vt) = overloaded_svd(reshape(A, :, 1); full=true, algorithm=fn.algorithm)
    return U, S, Vt
end

function LinearAlgebra.svdvals(x::AnyTracedRArray{T,N}; kwargs...) where {T,N}
    return overloaded_svd(x; kwargs..., full=false).S
end
function LinearAlgebra.svdvals!(x::AnyTracedRArray{T,N}; kwargs...) where {T,N}
    return overloaded_svd(x; kwargs..., full=false).S
end
function LinearAlgebra.svdvals(x::AnyTracedRVector{T}; kwargs...) where {T}
    return overloaded_svd(x; kwargs..., full=false).S
end
function LinearAlgebra.svdvals!(x::AnyTracedRVector{T}; kwargs...) where {T}
    return overloaded_svd(x; kwargs..., full=false).S
end
