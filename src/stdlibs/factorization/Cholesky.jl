struct BatchedCholesky{T,S<:AbstractArray,I<:Union{AbstractArray,Number}} <:
       BatchedFactorization{T}
    factors::S
    uplo::Char
    info::I
end

function BatchedCholesky(factors::S, uplo::Char, info::I) where {S,I}
    @assert ndims(info) == ndims(factors) - 2
    return BatchedCholesky{eltype(factors),S,I}(factors, uplo, info)
end

Base.size(c::BatchedCholesky) = size(c.factors)
Base.size(c::BatchedCholesky, i::Integer) = size(c.factors, i)
Base.ndims(c::BatchedCholesky) = ndims(c.factors)

function overloaded_cholesky(A::AbstractArray, ::NoPivot; check::Bool=false)
    return overloaded_cholesky(Reactant.promote_to(TracedRArray, A), NoPivot(); check)
end

function overloaded_cholesky(
    A::AnyTracedRArray{T,N}, ::NoPivot; check::Bool=false
) where {T,N}
    # TODO: dont ignore check
    # move the batching dims to the front
    permdims = vcat(collect(Int64, 3:N), 1, 2)
    A = @opcall transpose(materialize_traced_array(A), permdims)

    factors = @opcall cholesky(A; lower=false)
    factors = @opcall transpose(factors, invperm(permdims))

    # stablehlo doesn't return the info
    info = materialize_traced_array(
        dropdims(
            Reactant.CallWithReactant(mapreduce)(
                isfinite, &, mapslices(LinearAlgebra.triu, factors; dims=(1, 2)); dims=1:2
            );
            dims=(1, 2),
        ),
    )
    if N == 2
        info = TracedRNumber{Bool}((), info.mlir_data)
    end

    return BatchedCholesky(factors, 'U', info)
end

function LinearAlgebra.ldiv!(
    F::BatchedCholesky{T,<:AbstractArray{T,N}}, B::AbstractArray{T,M}
) where {T,N,M}
    @assert N == M + 1
    ldiv!(F, reshape(B, size(B, 1), 1, size(B)[2:end]...))
    return B
end

function LinearAlgebra.ldiv!(
    F::BatchedCholesky{T,<:AbstractArray{T,2}}, B::AbstractArray{T,2}
) where {T}
    B .= _cholesky_solve_core(F.factors, B, F.uplo)
    return B
end

function LinearAlgebra.ldiv!(
    F::BatchedCholesky{T,<:AbstractArray{T,N}}, B::AbstractArray{T,N}
) where {T,N}
    batch_shape = size(F.factors)[3:end]
    @assert batch_shape == size(B)[3:end]

    base_fn = F.uplo == 'U' ? _cholesky_solve_core_upper : _cholesky_solve_core_lower

    permutation = vcat(collect(Int64, 3:N), 1, 2)

    factors = @opcall transpose(materialize_traced_array(F.factors), permutation)
    B_permuted = @opcall transpose(materialize_traced_array(B), permutation)

    res = @opcall transpose(
        only(@opcall(batch(base_fn, [factors, B_permuted], collect(Int64, batch_shape)))),
        invperm(permutation),
    )
    B .= res
    return B
end

function _cholesky_solve_core(factors::AbstractMatrix, B::AbstractMatrix, uplo::Char)
    if uplo == 'U'
        return _cholesky_solve_core_upper(factors, B)
    else
        return _cholesky_solve_core_lower(factors, B)
    end
end

function _cholesky_solve_core_lower(factors::AbstractMatrix, B::AbstractMatrix)
    return adjoint(LowerTriangular(factors)) \ (LowerTriangular(factors) \ B)
end
function _cholesky_solve_core_upper(factors::AbstractMatrix, B::AbstractMatrix)
    return UpperTriangular(factors) \ (adjoint(UpperTriangular(factors)) \ B)
end
