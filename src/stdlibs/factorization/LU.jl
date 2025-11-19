struct BatchedLU{T,S<:AbstractArray,P<:AbstractArray,I<:Union{AbstractArray,Number}} <:
       BatchedFactorization{T}
    factors::S
    ipiv::P
    perm::P
    info::I
end

Base.size(lu::BatchedLU) = size(lu.factors)
Base.size(lu::BatchedLU, i::Integer) = size(lu.factors, i)
Base.ndims(lu::BatchedLU) = ndims(lu.factors)
function Base.copy(lu::BatchedLU)
    return BatchedLU(copy(lu.factors), copy(lu.ipiv), copy(lu.perm), copy(lu.info))
end

function BatchedLU(factors::S, ipiv::P, perm::P, info::I) where {S,P,I}
    @assert ndims(ipiv) == ndims(perm) == ndims(factors) - 1
    @assert ndims(info) == ndims(factors) - 2
    return BatchedLU{eltype(factors),S,P,I}(factors, ipiv, perm, info)
end

function overloaded_lu(x::AbstractArray, args...; kwargs...)
    return overloaded_lu(Reactant.promote_to(TracedRArray, x), args...; kwargs...)
end

function overloaded_lu(
    A::AnyTracedRArray{T,N}, ::RowMaximum; check::Bool=false, allowsingular::Bool=false
) where {T,N}
    # TODO: don't ignore the check and allowsingular flags
    # Batching here is in the last dimensions. `Ops.lu` expects the last dimensions
    permdims = vcat(collect(Int64, 3:N), 1, 2)
    A = @opcall transpose(materialize_traced_array(A), permdims)
    factors, ipiv, perm, info = @opcall lu(A)

    # Permute back to the original dimensions
    perm_perm = vcat(N - 1, collect(Int64, 1:(N - 2)))
    factors = @opcall transpose(factors, invperm(permdims))
    ipiv = @opcall transpose(ipiv, perm_perm)
    perm = @opcall transpose(perm, perm_perm)
    return BatchedLU(factors, ipiv, perm, info)
end

function LinearAlgebra.ldiv!(
    lu::BatchedLU{T,<:AbstractArray{T,N},P,I}, B::AbstractArray{T,M}
) where {T,P,I,N,M}
    @assert N == M + 1
    ldiv!(lu, reshape(B, size(B, 1), 1, size(B)[2:end]...))
    return B
end

function LinearAlgebra.ldiv!(
    lu::BatchedLU{T,<:AbstractArray{T,2},P,I}, B::AbstractArray{T,2}
) where {T,P,I}
    B .= _lu_solve_core(lu.factors, B, lu.perm)
    return B
end

function LinearAlgebra.ldiv!(
    lu::BatchedLU{T,<:AbstractArray{T,N},P,I}, B::AbstractArray{T,N}
) where {T,P,I,N}
    batch_shape = size(lu.factors)[3:end]
    @assert batch_shape == size(B)[3:end]

    permutation = vcat(collect(Int64, 3:N), 1, 2)

    factors = @opcall transpose(materialize_traced_array(lu.factors), permutation)
    B_permuted = @opcall transpose(materialize_traced_array(B), permutation)
    perm = @opcall transpose(
        materialize_traced_array(lu.perm), vcat(collect(Int64, 2:(N - 1)), 1)
    )

    res = @opcall transpose(
        only(
            @opcall(
                batch(
                    _lu_solve_core, [factors, B_permuted, perm], collect(Int64, batch_shape)
                )
            ),
        ),
        invperm(permutation),
    )
    B .= res
    return B
end

function LinearAlgebra.det(lu::BatchedLU{T,<:AbstractMatrix}) where {T}
    n = LinearAlgebra.checksquare(lu)
    # TODO: check for non-singular matrices

    P = prod(LinearAlgebra.diag(lu.factors))
    return ifelse(isodd(sum(lu.ipiv[1:n] .!= (1:n))), -one(T), one(T)) * P
end

function LinearAlgebra.logabsdet(lu::BatchedLU{T,<:AbstractMatrix}) where {T}
    n = LinearAlgebra.checksquare(lu)
    Treal = real(T)
    # TODO: check for non-singular matrices

    d = LinearAlgebra.diag(lu.factors)
    absdet = sum(log ∘ abs, d)
    P = prod(sign, d)
    s = ifelse(isodd(sum(lu.ipiv[1:n] .!= (1:n))), -one(Treal), one(Treal)) * P
    return absdet, s
end

for f_wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization),
    aType in (:AbstractVecOrMat, :AbstractArray)

    @eval function LinearAlgebra.ldiv!(lu::$(f_wrapper){<:Any,<:BatchedLU}, B::$aType)
        # TODO: implement this
        error("`$(f_wrapper)` is not supported yet for LU.")
        return nothing
    end
end

# currently we lower inverse to lu decomposition + triangular solve. we should
# instead emit getri and lower that to a fallback if the backend doesn't support
# it.
function LinearAlgebra.inv!(lu::BatchedLU)
    @assert ndims(lu) == 2 "Only implemented for 2D tensors"
    rhs = Reactant.promote_to(
        TracedRArray{Reactant.unwrapped_eltype(eltype(lu)),2}, LinearAlgebra.I(size(lu, 1))
    )
    ldiv!(lu, rhs)
    return rhs
end

function _lu_solve_core(factors::AbstractMatrix, B::AbstractMatrix, perm::AbstractVector)
    permuted_B = B[Int64.(perm), :]
    return UpperTriangular(factors) \ (UnitLowerTriangular(factors) \ permuted_B)
end
