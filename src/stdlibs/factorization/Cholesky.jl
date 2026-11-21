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

function LinearAlgebra.copytrito!(
    B::AnyTracedRMatrix, A::AbstractMatrix, uplo::AbstractChar
)
    m, n = size(A)
    if uplo == 'U'
        LinearAlgebra.LAPACK.lacpy_size_check(size(B), (min(m, n), n))
        r, c = min(m, n), n
        rows = @opcall iota(Int64, [r, c]; iota_dimension=1)
        cols = @opcall iota(Int64, [r, c]; iota_dimension=2)
        mask = rows .<= cols
    elseif uplo == 'L'
        LinearAlgebra.LAPACK.lacpy_size_check(size(B), (m, min(m, n)))
        r, c = m, min(m, n)
        rows = @opcall iota(Int64, [r, c]; iota_dimension=1)
        cols = @opcall iota(Int64, [r, c]; iota_dimension=2)
        mask = rows .>= cols
    else
        return LinearAlgebra.throw_uplo()
    end
    TB = unwrapped_eltype(B)
    A_sub = promote_to(TracedRArray{TB,2}, materialize_traced_array(view(A, 1:r, 1:c)))
    B_sub = materialize_traced_array(view(B, 1:r, 1:c))
    B[1:r, 1:c] = ifelse.(mask, A_sub, B_sub)
    return B
end

function _zero_hermitian_diag_imag(A::AnyTracedRMatrix{T}) where {T<:Complex}
    m, n = size(A)
    rows = @opcall iota(Int64, [m, n]; iota_dimension=1)
    cols = @opcall iota(Int64, [m, n]; iota_dimension=2)
    raw = materialize_traced_array(A)
    diag_real = complex.(real.(raw), zero(real(T)))
    return ifelse.(rows .== cols, diag_real, raw)
end
_zero_hermitian_diag_imag(A::AnyTracedRMatrix) = materialize_traced_array(A)

function _cholesky_factors_and_info(A::AnyTracedRArray{T,N}, uplo::Char) where {T,N}
    LinearAlgebra.checksquare(A)
    lower = uplo == 'L'
    permdims = vcat(collect(Int64, 3:N), 1, 2)
    A_perm = @opcall transpose(materialize_traced_array(A), permdims)

    factors = @opcall cholesky(A_perm; lower)
    factors = @opcall transpose(factors, invperm(permdims))

    tri_fn = lower ? LinearAlgebra.tril : LinearAlgebra.triu
    # stablehlo doesn't return the info
    info = materialize_traced_array(
        dropdims(
            Reactant.CallWithReactant(mapreduce)(
                isfinite, &, mapslices(tri_fn, factors; dims=(1, 2)); dims=1:2
            );
            dims=(1, 2),
        ),
    )
    if N == 2
        info = TracedRNumber{Bool}((), info.mlir_data)
    end
    return factors, info
end

function _copy_cholesky_triangle!(dest::AnyTracedRArray{T,2}, factors, uplo::Char) where {T}
    return LinearAlgebra.copytrito!(dest, factors, uplo)
end

function _copy_cholesky_triangle!(
    dest::AnyTracedRArray{T,N}, factors, uplo::Char
) where {T,N}
    m = size(dest, 1)
    n = size(dest, 2)
    bshape = collect(Int64, size(dest))
    rows = @opcall iota(Int64, bshape; iota_dimension=1)
    cols = @opcall iota(Int64, bshape; iota_dimension=2)
    mask = uplo == 'L' ? (rows .>= cols) : (rows .<= cols)
    dest .= ifelse.(mask, factors, materialize_traced_array(dest))
    return dest
end

function overloaded_cholesky(A::Union{Symmetric,Hermitian}, ::NoPivot; check::Bool=false)
    # TODO: dont ignore check
    uplo = A.uplo
    data = promote_to(TracedRArray, A.data)
    if A isa Hermitian
        data = _zero_hermitian_diag_imag(data)
    end
    factors, info = _cholesky_factors_and_info(data, uplo)
    return BatchedCholesky(factors, uplo, info)
end

function overloaded_cholesky(
    A::AnyTracedRArray{T,N}, ::NoPivot; check::Bool=false
) where {T,N}
    # TODO: dont ignore check
    factors, info = _cholesky_factors_and_info(A, 'U')
    return BatchedCholesky(factors, 'U', info)
end

function _check_cholesky_inplace_eltype(::Type{T}) where {T}
    if !(T <: Union{AbstractFloat,Complex{<:AbstractFloat}})
        throw(InexactError(:cholesky!, T, 0))
    end
    return nothing
end

function overloaded_cholesky!(A::AbstractArray, ::NoPivot; check::Bool=false)
    _check_cholesky_inplace_eltype(unwrapped_eltype(A))
    return overloaded_cholesky!(promote_to(TracedRArray, A), NoPivot(); check)
end

function overloaded_cholesky!(A::Union{Symmetric,Hermitian}, ::NoPivot; check::Bool=false)
    # TODO: dont ignore check
    _check_cholesky_inplace_eltype(unwrapped_eltype(A))
    uplo = A.uplo
    if !(A.data isa AnyTracedRArray)
        return overloaded_cholesky(A, NoPivot(); check)
    end
    input_data = A isa Hermitian ? _zero_hermitian_diag_imag(A.data) : A.data
    factors, info = _cholesky_factors_and_info(input_data, uplo)
    _copy_cholesky_triangle!(A.data, factors, uplo)
    return BatchedCholesky(A.data, uplo, info)
end

function overloaded_cholesky!(
    A::AnyTracedRArray{T,N}, ::NoPivot; check::Bool=false
) where {T,N}
    # TODO: dont ignore check
    _check_cholesky_inplace_eltype(T)
    factors, info = _cholesky_factors_and_info(A, 'U')
    _copy_cholesky_triangle!(A, factors, 'U')
    return BatchedCholesky(A, 'U', info)
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
