"""
    BatchedQR

Unpivoted QR factorization of a (batch of) matrix(es), stored in LAPACK's compact form:
the packed `factors` and the reflector scalings `τ`. This mirrors `LinearAlgebra.QRCompactWY`,
so `F.factors`, `F.τ`, `F.Q` and `F.R` are all available -- `F.R` is free, and `F.Q` only
emits the LAPACK `orgqr`/`ungqr` call when it is actually requested.

For an `m × n` input with `k = min(m, n)`, `F.Q` is the **thin** `m × k` factor and `F.R` is
`k × n`, so `F.Q * F.R ≈ A` and `F.R` agrees with `LinearAlgebra.qr(A).R`. Note that Julia's
own `qr(A).Q` is instead a square `m × m` `AbstractQ` operator.

Like the other Reactant factorizations, this is meant to be used from inside traced code;
access `F.Q` / `F.R` within the function you compile.

!!! note

    Only unpivoted QR is supported -- the backend has no `geqp3`. Batched (`ndims > 2`)
    inputs are supported on the CUDA and TPU backends only.
"""
struct BatchedQR{T,S<:AbstractArray,Tau<:AbstractArray,I} <: BatchedFactorization{T}
    factors::S
    tau::Tau
    info::I
end

function BatchedQR(factors::S, tau::Tau, info::I) where {S,Tau,I}
    @assert ndims(tau) == ndims(factors) - 1
    return BatchedQR{eltype(factors),S,Tau,I}(factors, tau, info)
end

# `__get_B` / `_overloaded_backslash` in Factorization.jl read `size(F, 1)` and `size(F, 2)`
# as the dimensions of the original matrix, so these must track `factors`.
Base.size(F::BatchedQR) = size(getfield(F, :factors))
Base.size(F::BatchedQR, i::Integer) = size(getfield(F, :factors), i)
Base.ndims(F::BatchedQR) = ndims(getfield(F, :factors))

function Base.copy(F::BatchedQR)
    return BatchedQR(
        copy(getfield(F, :factors)), copy(getfield(F, :tau)), copy(getfield(F, :info))
    )
end

function Base.getproperty(F::BatchedQR, d::Symbol)
    d === :τ && return getfield(F, :tau)
    d === :Q && return _qr_Q(F)
    d === :R && return _qr_R(F)
    return getfield(F, d)
end

Base.propertynames(::BatchedQR) = (:factors, :tau, :τ, :Q, :R, :info)

Base.iterate(F::BatchedQR) = (F.Q, Val(:R))
Base.iterate(F::BatchedQR, ::Val{:R}) = (F.R, Val(:done))
Base.iterate(::BatchedQR, ::Val{:done}) = nothing

# The `Ops` take the matrix in the trailing dimensions with the batch dimensions leading,
# while the factorization stores everything in the usual Julia order.
_qr_matrix_last(N::Int) = vcat(collect(Int64, 3:N), 1, 2)
_qr_tau_matrix_last(N::Int) = vcat(collect(Int64, 2:(N - 1)), 1)
_qr_tau_matrix_first(N::Int) = vcat(N - 1, collect(Int64, 1:(N - 2)))

_qr_trailing_colons(N::Int) = ntuple(Returns(Colon()), N - 2)

function _qr_Q(F::BatchedQR)
    factors = materialize_traced_array(getfield(F, :factors))
    tau = materialize_traced_array(getfield(F, :tau))
    N = ndims(factors)
    m, n = size(factors, 1), size(factors, 2)

    if m < n
        # `orgqr` derives the number of reflectors from the column count of its input, so a
        # wide packed matrix has to be squared off first. The result is the `m × m` Q.
        factors = materialize_traced_array(factors[:, 1:m, _qr_trailing_colons(N)...])
    end

    permdims = _qr_matrix_last(N)
    Q = @opcall orgqr(
        @opcall(transpose(factors, permdims)),
        @opcall(transpose(tau, _qr_tau_matrix_last(N))),
    )
    return @opcall transpose(Q, invperm(permdims))
end

function _qr_R(F::BatchedQR)
    factors = materialize_traced_array(getfield(F, :factors))
    N = ndims(factors)
    k = min(size(factors, 1), size(factors, 2))
    return _qr_triu(materialize_traced_array(factors[1:k, :, _qr_trailing_colons(N)...]))
end

# `overloaded_triu` is matrix-only; this is the same iota/select trick over the leading two
# dimensions, which keeps the emitted IR independent of the batch size.
function _qr_triu(X::TracedRArray{T,N}) where {T,N}
    shape = collect(Int64, size(X))
    idxs = @opcall compare(
        @opcall(iota(Int64, shape; iota_dimension=1)),
        @opcall(iota(Int64, shape; iota_dimension=2));
        comparison_direction="LE",
    )
    return @opcall select(idxs, X, zero(X))
end

function overloaded_qr(A::AbstractArray, args...; kwargs...)
    return overloaded_qr(Reactant.promote_to(TracedRArray, A), args...; kwargs...)
end

function overloaded_qr(
    A::AnyTracedRArray{T,N}, ::NoPivot; blocksize::Integer=36
) where {T,N}
    # blocksize is ignored: the backend picks its own blocking for `geqrf`
    permdims = _qr_matrix_last(N)
    factors, tau, info = @opcall geqrf(
        @opcall(transpose(materialize_traced_array(A), permdims))
    )
    return BatchedQR(
        (@opcall transpose(factors, invperm(permdims))),
        (@opcall transpose(tau, _qr_tau_matrix_first(N))),
        info,
    )
end

function overloaded_qr(::AnyTracedRArray, ::ColumnNorm; kwargs...)
    throw(
        ArgumentError(
            "Column-pivoted QR is not supported by the Reactant backend (there is no \
             `geqp3` lowering). Use `qr(A)` for the unpivoted factorization, or `svd(A)` \
             if `A` may be rank-deficient."
        ),
    )
end

# Least squares for `m ≥ n`, and the same basic solution Julia's unpivoted `ldiv!(::QR, B)`
# produces for `m < n` (the caller zeroes the trailing rows).
function _qr_solve_core(Q::AbstractMatrix, R::AbstractMatrix, B::AbstractMatrix)
    m, k = size(Q)
    rhs = adjoint(Q) * LinearAlgebra._cut_B(B, 1:m)
    return UpperTriangular(materialize_traced_array(R[:, 1:k])) \ rhs
end

function LinearAlgebra.ldiv!(
    F::BatchedQR{T,<:AbstractArray{T,N}}, B::AbstractArray{T,M}
) where {T,N,M}
    @assert N == M + 1
    ldiv!(F, reshape(B, size(B, 1), 1, size(B)[2:end]...))
    return B
end

function LinearAlgebra.ldiv!(
    F::BatchedQR{T,<:AbstractArray{T,2}}, B::AbstractArray{T,2}
) where {T}
    m, n = size(F, 1), size(F, 2)
    k = min(m, n)
    B[1:k, :] .= _qr_solve_core(F.Q, F.R, B)
    # `__get_B` allocates the right-hand side with `similar`, so the trailing rows of an
    # under-determined system have to be zeroed explicitly.
    n > k && (B[(k + 1):n, :] .= zero(T))
    return B
end

function LinearAlgebra.ldiv!(
    F::BatchedQR{T,<:AbstractArray{T,N}}, B::AbstractArray{T,N}
) where {T,N}
    batch_shape = size(F)[3:end]
    @assert batch_shape == size(B)[3:end]

    m, n = size(F, 1), size(F, 2)
    k = min(m, n)
    permutation = _qr_matrix_last(N)

    # Q and R are materialized once, outside the batched region, so only the solve is
    # batched.
    Q = @opcall transpose(materialize_traced_array(F.Q), permutation)
    R = @opcall transpose(materialize_traced_array(F.R), permutation)
    B_permuted = @opcall transpose(materialize_traced_array(B), permutation)

    res = @opcall transpose(
        only(
            @opcall(batch(_qr_solve_core, [Q, R, B_permuted], collect(Int64, batch_shape)))
        ),
        invperm(permutation),
    )
    B[1:k, :, _qr_trailing_colons(N)...] .= res
    n > k && (B[(k + 1):n, :, _qr_trailing_colons(N)...] .= zero(T))
    return B
end

for f_wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization),
    aType in (:AbstractVecOrMat, :AbstractArray)

    # the message has to be built here rather than interpolated into the `@eval`'d body,
    # where `f_wrapper` would only be looked up at call time
    msg = "`$(nameof(f_wrapper))` is not supported yet for QR."

    @eval function LinearAlgebra.ldiv!(F::$(f_wrapper){<:Any,<:BatchedQR}, B::$aType)
        # TODO: implement this
        return error($msg)
    end
end
