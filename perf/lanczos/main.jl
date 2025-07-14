using Reactant
using Reactant: Ops, TracedRNumber
using LinearAlgebra
using Random
using Statistics
using BenchmarkTools

# setup
Random.seed!(0)

A = rand(Float64, 512, 512)
A = A' * A # make it hermitian
@assert ishermitian(A)

b = normalize!(rand(Float64, 512))

A_re = Reactant.to_rarray(A)
b_re = Reactant.to_rarray(b)

# fixes
# TODO move to Reactant
function Base.zeros(::Type{TracedRNumber{T}}, dims::NTuple{N,<:Integer}) where {T,N}
    _zero = Ops.constant(zero(T))
    return Ops.broadcast_in_dim(_zero, Int[], collect(dims))
end

# algorithm
# - A: matrix to (partially) decompose. lanczos requires it to be symmetric/hermitian.
# - v0: initial vector, should be normalized.
# - m: decomposition rank
function lanczos(A, v0, m)
    n = size(A, 1)
    V = zeros(eltype(A), n, m + 1)
    T = zeros(eltype(A), m, m)

    v = v0 / norm(v0)
    V[:, 1] = v
    beta = 0.0
    w = similar(v)

    @allowscalar for j in 1:m
        w .= A * v
        if j > 1
            w .-= beta * V[:, j - 1]
        end
        alpha = dot(w, v)
        w .-= alpha * v
        beta = norm(w)

        T[j, j] = alpha
        if j < m
            T[j, j + 1] = beta
            T[j + 1, j] = beta
        end

        # early termination if Krylov subspace is reached
        # TODO Reactant.@trace doesn't support return statements yet
        # @trace if beta < eps(eltype(A))
        #     return V[:, 1:j], T[1:j, 1:j]
        # end

        # full reorthogonalization via modified Gram-Schmidt to avoid spurious duplicate eigenvalues
        # TODO implicitly restarted Lanczos? available at KrylovKit
        for k in 1:j
            w .-= dot(V[:, k], w) * V[:, k]
        end

        v = w / beta
        V[:, j + 1] = v
    end

    return V, T
end

V, T = lanczos(A, b, 512)
eigvals(T)

l1_error = sum(abs.(eigvals(A) - eigvals(T)))
l2_error = sqrt(sum(abs2.(eigvals(A) - eigvals(T))))
linf_error = maximum(abs.(eigvals(A) - eigvals(T)))
@info "Error" l1 = l1_error l2 = l2_error linf = linf_error

# benchmarking

# @benchmark lanczos($A, $b, 16) setup = (GC.gc())
@benchmark lanczos($A, $b, 16)

# compile with Reactant
f = @compile sync = true lanczos(A_re, b_re, 16)
@benchmark $f($A_re, $b_re, 16)
