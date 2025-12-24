struct BatchedSVD{T,Tr,M<:AbstractArray,C<:AbstractArray} <: BatchedFactorization{T}
    U::M
    S::C
    Vt::M

    function BatchedSVD{T,Tr,M,C}(U::M, S::C, Vt::M) where {T,Tr,M,C}
        @assert ndims(S) == ndims(U) - 1
        return new{T,Tr,M,C}(U, S, Vt)
    end
end

function Base.size(svd::BatchedSVD)
    return (size(svd.U, 1), size(svd.Vt, 2), size(svd.U)[3:end]...)
end
Base.size(svd::BatchedSVD, i::Integer) = i == 2 ? size(svd.Vt, 2) : size(svd.U, i)

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
    A::AnyTracedRArray{T,N}; full::Bool=false, alg=LinearAlgebra.default_svd_alg(A)
) where {T,N}
    # Batching here is in the last dimensions. `Ops.svd` expects the last dimensions
    permdims = vcat(collect(Int64, 3:N), 1, 2)
    A = @opcall transpose(materialize_traced_array(A), permdims)

    U, S, Vt = @opcall svd(A; full, algorithm=_jlalg_to_enzymexla_alg(alg))

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
    A::AnyTracedRVector; full::Bool=false, alg=LinearAlgebra.default_svd_alg(A)
)
    normA = Reactant.call_with_reactant(LinearAlgebra.norm, A)
    U, S, Vt = ReactantCore.traced_if(
        iszero(normA),
        __ZeroNormVectorSVDDispatch(full, alg),
        __InnerVectorSVDDispatch(full, alg),
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
    (; U, S, Vt) = overloaded_svd(reshape(A, :, 1); full=true, alg=fn.algorithm)
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

# Ideally we want to slice based on near zero singular values, but this will
# produce dynamically sized slices. Instead we zero out slices and proceed
function _svd_solve_core(
    U::AbstractMatrix, S::AbstractVector{Tr}, Vt::AbstractMatrix, B::AbstractMatrix
) where {Tr}
    mask = S .> eps(real(Tr)) * @allowscalar(S[1])
    m, n = size(U, 1), size(Vt, 2)
    rhs = S .\ (U' * LinearAlgebra._cut_B(B, 1:m))
    rhs = ifelse.(mask, rhs, zero(eltype(rhs)))
    return (Vt[1:length(S), :])' * rhs
end

function LinearAlgebra.ldiv!(
    svd::BatchedSVD{T,Tr,<:AbstractArray{T,N}}, B::AbstractArray{T,M}
) where {T,Tr,N,M}
    @assert N == M + 1
    ldiv!(svd, reshape(B, size(B, 1), 1, size(B)[2:end]...))
    return B
end

function LinearAlgebra.ldiv!(
    svd::BatchedSVD{T,Tr,<:AbstractArray{T,2}}, B::AbstractArray{T,2}
) where {T,Tr}
    n = size(svd, 2)
    sol = _svd_solve_core(svd.U, svd.S, svd.Vt, B)
    B[1:n, :] .= sol
    return B
end

function LinearAlgebra.ldiv!(
    svd::BatchedSVD{T,Tr,<:AbstractArray{T,N}}, B::AbstractArray{T,N}
) where {T,Tr,N}
    batch_shape = size(svd.U)[3:end]
    @assert batch_shape == size(B)[3:end]

    n = size(svd, 2)
    permutation = vcat(collect(Int64, 3:N), 1, 2)
    S_perm = vcat(collect(Int64, 2:(N - 1)), 1)

    U = @opcall transpose(materialize_traced_array(svd.U), permutation)
    S = @opcall transpose(materialize_traced_array(svd.S), S_perm)
    Vt = @opcall transpose(materialize_traced_array(svd.Vt), permutation)

    B_permuted = @opcall transpose(materialize_traced_array(B), permutation)

    res = @opcall transpose(
        only(
            @opcall(
                batch(_svd_solve_core, [U, S, Vt, B_permuted], collect(Int64, batch_shape))
            ),
        ),
        invperm(permutation),
    )
    B[1:n, :, ntuple(Returns(Colon()), length(batch_shape))...] .= res
    return B
end
