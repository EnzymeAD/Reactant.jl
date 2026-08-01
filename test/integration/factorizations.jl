using LinearAlgebra, Reactant, Test

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

solve_with_fact(f::F, A, b) where {F} = f(A) \ b
function solve_with_fact_batched(
    f::F, A::AbstractArray{T,N}, B::AbstractArray{T,N}
) where {F,T,N}
    A2 = reshape(A, size(A, 1), size(A, 2), prod(size(A)[3:end]))
    B2 = reshape(B, size(B, 1), size(B, 2), prod(size(B)[3:end]))
    @assert size(A2, 3) == size(B2, 3)
    return reshape(
        stack(f(view(A2, :, :, i)) \ view(B2, :, :, i) for i in axes(A2, 3)),
        size(A2, 1),
        size(B2, 2),
        size(A)[3:end]...,
    )
end
function solve_with_fact_batched(
    f::F, A::AbstractArray{T,N}, b::AbstractArray{T,M}
) where {F,T,N,M}
    @assert N == M + 1
    B = reshape(b, size(b, 1), 1, size(b)[2:end]...)
    return dropdims(solve_with_fact_batched(f, A, B); dims=2)
end

solve_with_lu(A, b) = solve_with_fact(lu, A, b)
solve_with_lu_batched(A, b) = solve_with_fact_batched(lu, A, b)

@testset "LU Factorization" begin
    @testset "Un-batched" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
            (T == ComplexF64 || T == Float64) && RunningOnTPU && continue

            A = rand(T, 4, 4)
            A_ra = Reactant.to_rarray(A)

            b = rand(T, 4)
            b_ra = Reactant.to_rarray(b)

            B = rand(T, 4, 3)
            B_ra = Reactant.to_rarray(B)

            @test @jit(solve_with_lu(A_ra, b_ra)) ≈ solve_with_lu(A, b) atol = 1e-4 rtol =
                1e-2
            @test @jit(solve_with_lu(A_ra, B_ra)) ≈ solve_with_lu(A, B) atol = 1e-4 rtol =
                1e-2
        end
    end

    @testset "Batched" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
            (T == ComplexF64 || T == Float64) && RunningOnTPU && continue

            A = rand(T, 4, 4, 3, 2)
            A_ra = Reactant.to_rarray(A)

            b = rand(T, 4, 3, 2)
            b_ra = Reactant.to_rarray(b)

            B = rand(T, 4, 5, 3, 2)
            B_ra = Reactant.to_rarray(B)

            @test @jit(solve_with_lu(A_ra, b_ra)) ≈ solve_with_lu_batched(A, b) atol = 1e-4 rtol =
                1e-2
            @test @jit(solve_with_lu(A_ra, B_ra)) ≈ solve_with_lu_batched(A, B) atol = 1e-4 rtol =
                1e-2
        end
    end

    @testset "transpose! and adjoint!" begin
        A = Reactant.TestUtils.construct_test_array(Complex{Float32}, 7, 13)
        B = similar(A')::Matrix
        A_ra = Reactant.to_rarray(A)

        B_ra = Reactant.to_rarray(B)
        @test Array(@jit(transpose!(B_ra, A_ra))) ≈ transpose(A)

        B_ra = Reactant.to_rarray(B)
        @test Array(@jit(adjoint!(B_ra, A_ra))) ≈ adjoint(A)
    end

    @testset "Input Permutation" begin
        A = rand(Float32, 10, 10, 32)
        B = rand(Float32, 10, 32)
        A_ra = Reactant.to_rarray(A)
        B_ra = Reactant.to_rarray(B)

        @test @jit(solve_with_lu(A_ra, B_ra)) ≈ solve_with_lu_batched(A, B) atol = 1e-4 rtol =
            1e-2
    end
end

solve_with_cholesky(A, b) = solve_with_fact(cholesky, A, b)
solve_with_cholesky_batched(A, b) = solve_with_fact_batched(cholesky, A, b)

function random_matrix_with_cond(
    ::Type{T}, rows::Int, cols::Int, cond_number::Float64
) where {T}
    # Generate random orthogonal matrices U and V
    U = (
        LinearAlgebra.qr(randn(rows, rows)).Q *
        Diagonal(sign.(diag(LinearAlgebra.qr(randn(rows, rows)).R)))
    )
    V = (
        LinearAlgebra.qr(randn(cols, cols)).Q *
        Diagonal(sign.(diag(LinearAlgebra.qr(randn(cols, cols)).R)))
    )

    min_dim = min(rows, cols)
    singular_values = exp.(range(log(1.0), log(1.0 / cond_number); length=min_dim))

    S = zeros(Float64, rows, cols)
    @inbounds for i in 1:min_dim
        S[i, i] = singular_values[i]
    end

    return T.(U * S * V')
end

@testset "Cholesky Factorization" begin
    @testset "Un-batched" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
            (T == ComplexF64 || T == Float64) && RunningOnTPU && continue

            A = random_matrix_with_cond(T, 4, 4, 1.001) # avoid ill conditioned
            A = A * A'
            A_ra = Reactant.to_rarray(A)

            b = rand(T, 4)
            b_ra = Reactant.to_rarray(b)

            B = rand(T, 4, 3)
            B_ra = Reactant.to_rarray(B)

            @test @jit(solve_with_cholesky(A_ra, b_ra)) ≈ solve_with_cholesky(A, b) atol =
                1e-4 rtol = 1e-2
            @test @jit(solve_with_cholesky(A_ra, B_ra)) ≈ solve_with_cholesky(A, B) atol =
                1e-4 rtol = 1e-2
        end
    end

    @testset "Batched" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
            (T == ComplexF64 || T == Float64) && RunningOnTPU && continue

            A = stack(random_matrix_with_cond(T, 4, 4, 1.001) for _ in 1:6)
            A = reshape(stack((r * r' for r in eachslice(A; dims=3))), 4, 4, 3, 2)
            A_ra = Reactant.to_rarray(A)

            b = rand(T, 4, 3, 2)
            b_ra = Reactant.to_rarray(b)

            B = rand(T, 4, 5, 3, 2)
            B_ra = Reactant.to_rarray(B)

            @test @jit(solve_with_cholesky(A_ra, b_ra)) ≈ solve_with_cholesky_batched(A, b) atol =
                1e-4 rtol = 1e-2
            @test @jit(solve_with_cholesky(A_ra, B_ra)) ≈ solve_with_cholesky_batched(A, B) atol =
                1e-4 rtol = 1e-2
        end
    end
end

const RunningOnCPU = contains(lowercase(string(Reactant.devices()[1])), "cpu")

function qr_factors(A)
    F = qr(A)
    return F.Q, F.R
end

function qr_destructure(A)
    Q, R = qr(A)
    return Q, R
end

solve_with_qr(A, b) = solve_with_fact(qr, A, b)
solve_with_qr!(A, b) = solve_with_fact(qr!, A, b)
qr_adjoint_solve(A, b) = qr(A)' \ b
# `\` on a transposed factorization delegates to the adjoint one, so reach the transpose
# method through `ldiv!` directly
qr_transpose_ldiv(A, b) = ldiv!(transpose(qr(A)), b)

# `qr` on a plain host array inside a traced function must fall through to LinearAlgebra
const QR_HOST_MATRIX = Matrix{Float32}(reshape(1:16, 4, 4)) + 8 * I
qr_of_host_array(x) = x .+ sum(qr(QR_HOST_MATRIX).R)

@testset "QR Factorization" begin
    @testset "$(T) $(m)x$(n)" for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((6, 4), (4, 4), (4, 6))

        (T == ComplexF64 || T == Float64) && RunningOnTPU && continue

        k = min(m, n)
        A = random_matrix_with_cond(T, m, n, 10.0)
        A_ra = Reactant.to_rarray(A)

        Q, R = @jit qr_factors(A_ra)
        Q, R = Array(Q), Array(R)

        @test size(Q) == (m, k)
        @test size(R) == (k, n)
        @test Q' * Q ≈ Matrix{T}(I, k, k) atol = 1e-4 rtol = 1e-2
        @test Q * R ≈ A atol = 1e-4 rtol = 1e-2
        @test R ≈ triu(R)

        # same LAPACK routine on both sides, so the sign convention matches too
        RunningOnCPU && @test R ≈ LinearAlgebra.qr(A).R atol = 1e-4 rtol = 1e-2
    end

    @testset "destructuring" begin
        A = random_matrix_with_cond(Float32, 6, 4, 10.0)
        A_ra = Reactant.to_rarray(A)

        Q, R = @jit qr_destructure(A_ra)
        @test Array(Q) * Array(R) ≈ A atol = 1e-4 rtol = 1e-2
    end

    @testset "least squares: $(T) $(m)x$(n)" for T in (Float32, ComplexF32),
        (m, n) in ((6, 4), (4, 4), (4, 6))

        A = random_matrix_with_cond(T, m, n, 10.0)
        A_ra = Reactant.to_rarray(A)

        b = rand(T, m)
        b_ra = Reactant.to_rarray(b)

        B = rand(T, m, 3)
        B_ra = Reactant.to_rarray(B)

        if m >= n
            # over-determined: the least squares solution is unique, so we can compare
            # against Julia directly
            @test @jit(solve_with_qr(A_ra, b_ra)) ≈ solve_with_qr(A, b) atol = 1e-4 rtol =
                1e-2
            @test @jit(solve_with_qr(A_ra, B_ra)) ≈ solve_with_qr(A, B) atol = 1e-4 rtol =
                1e-2
        else
            # under-determined: unpivoted QR returns a basic solution, which need not be
            # the one Julia's pivoted `\` picks, so check the residual instead
            x = Array(@jit(solve_with_qr(A_ra, b_ra)))
            @test maximum(abs, A * x .- b) < 1e-3
            @test all(iszero, x[(m + 1):n])

            X = Array(@jit(solve_with_qr(A_ra, B_ra)))
            @test maximum(abs, A * X .- B) < 1e-3
            @test all(iszero, X[(m + 1):n, :])
        end
    end

    @testset "pivoted QR is unsupported" begin
        A_ra = Reactant.to_rarray(rand(Float32, 4, 4))
        @test_throws ArgumentError @jit(qr(A_ra, ColumnNorm()))
    end

    @testset "qr!" begin
        A = random_matrix_with_cond(Float32, 6, 4, 10.0)
        b = rand(Float32, 6)
        A_ra = Reactant.to_rarray(A)
        b_ra = Reactant.to_rarray(b)

        @test @jit(solve_with_qr!(A_ra, b_ra)) ≈ solve_with_qr(A, b) atol = 1e-4 rtol = 1e-2
    end

    @testset "non-traced fallback" begin
        # `use_overlayed_version` is false here, so the overlay must defer to LinearAlgebra
        x_ra = Reactant.to_rarray(rand(Float32, 3))
        @test @jit(qr_of_host_array(x_ra)) ≈ qr_of_host_array(Array(x_ra))
    end

    @testset "factorization interface" begin
        A = random_matrix_with_cond(Float32, 6, 4, 10.0)
        A_ra = Reactant.to_rarray(A)
        F = @jit qr(A_ra)

        @test size(F) == (6, 4)
        @test size(F, 1) == 6
        @test size(F, 2) == 4
        @test ndims(F) == 2
        @test propertynames(F) == (:factors, :tau, :τ, :Q, :R, :info)
        @test F.τ === F.tau
        @test size(F.tau) == (4,)
        @test F.info == 0

        F2 = copy(F)
        @test Array(F2.factors) == Array(F.factors)
        @test Array(F2.tau) == Array(F.tau)
    end

    @testset "adjoint/transpose factorizations are unsupported" begin
        A = random_matrix_with_cond(Float32, 6, 4, 10.0)
        A_ra = Reactant.to_rarray(A)
        # `Adjoint`/`Transpose` reverse the reported size, so the RHS needs `size(F, 2)`
        # rows -- otherwise `__get_B` throws `DimensionMismatch` before `ldiv!` is reached
        b_ra = Reactant.to_rarray(rand(Float32, 4))

        @test_throws ErrorException @jit(qr_adjoint_solve(A_ra, b_ra))
        @test_throws ErrorException @jit(qr_transpose_ldiv(A_ra, b_ra))
    end
end

function get_svd_algorithms(backend::String, size=nothing)
    backend = lowercase(backend)
    algorithms = ["DEFAULT"]
    if occursin("cpu", backend)
        append!(algorithms, ["QRIteration", "DivideAndConquer"])
    elseif occursin("cuda", backend)
        if size === nothing || size[1] ≥ size[2]
            append!(algorithms, ["QRIteration", "Jacobi"])
        else
            append!(algorithms, ["Jacobi"])
        end
    elseif occursin("tpu", backend)
        append!(algorithms, ["Jacobi"])
    end
    return algorithms
end

least_squares_with_svd(A, b, full, alg) = svd(A; full, alg) \ b

function compute_ls_solution_error(A, sol, b, bsize)
    b = reshape(b, size(b, 1), bsize, :)
    A = reshape(A, size(A, 1), size(A, 2), :)
    sol = reshape(sol, size(sol, 1), bsize, :)
    mul = stack((A[:, :, i] * sol[:, :, i] for i in axes(A, 3)); dims=3)
    return maximum(abs, mul .- b)
end

@testset "svd factorization" begin
    A = Reactant.TestUtils.construct_test_array(Float32, 4, 8)

    algs = get_svd_algorithms(string(Reactant.devices()[1]), size(A))

    tmp = rand(Float32, 8, 5)
    B = A * tmp
    b = B[:, 1]

    A_ra = Reactant.to_rarray(A)
    B_ra = Reactant.to_rarray(B)
    b_ra = Reactant.to_rarray(b)

    # test least squares error
    @testset "least squares error: $(alg) | full=$(full)" for alg in algs,
        full in (true, false)

        # FIXME(#2314): fix TPU lowering
        @test begin
            sol1 = @jit least_squares_with_svd(A_ra, b_ra, full, alg)
            err1 = maximum(abs, A * Array(sol1) .- b)
            err1 < 1e-3
        end broken = RunningOnTPU

        # FIXME(#2314): fix TPU lowering
        @test begin
            sol2 = @jit least_squares_with_svd(A_ra, B_ra, full, alg)
            err2 = maximum(abs, A * Array(sol2) .- B)
            err2 < 1e-3
        end broken = RunningOnTPU
    end

    A = Reactant.TestUtils.construct_test_array(Float32, 4, 8, 3, 2)
    A_ra = Reactant.to_rarray(A)

    tmp = rand(Float32, 8, 5, 3, 2)
    B = similar(A, Float32, 4, 5, 3, 2)
    for i in 1:3, j in 1:2
        B[:, :, i, j] = A[:, :, i, j] * tmp[:, :, i, j]
    end
    B_ra = Reactant.to_rarray(B)
    b = B[:, 1, :, :]
    b_ra = Reactant.to_rarray(b)

    @testset "[batched] least squares error: $(alg) | full=$(full)" for alg in algs,
        full in (true, false)

        # FIXME(#2314): fix TPU lowering
        @test begin
            sol1 = @jit least_squares_with_svd(A_ra, b_ra, full, alg)
            err1 = compute_ls_solution_error(A, Array(sol1), b, 1)
            err1 < 1e-3
        end broken = RunningOnTPU

        # FIXME(#2314): fix TPU lowering
        @test begin
            sol2 = @jit least_squares_with_svd(A_ra, B_ra, full, alg)
            err2 = compute_ls_solution_error(A, Array(sol2), B, 5)
            err2 < 1e-3
        end broken = RunningOnTPU
    end
end

@testset "svdvals" begin
    algs = get_svd_algorithms(string(Reactant.devices()[1]))

    @testset "Un-batched: $(alg)" for alg in algs
        A = Reactant.TestUtils.construct_test_array(Float32, 4, 4)
        _svdvals = svdvals(A)
        A_ra = Reactant.to_rarray(A)

        # FIXME(#2314): fix TPU lowering
        @test begin
            _svdvals_ra = @jit svdvals(A_ra; alg=alg)
            _svdvals_ra ≈ _svdvals
        end broken = RunningOnTPU
    end

    @testset "Batched: $(alg)" for alg in algs
        A = Reactant.TestUtils.construct_test_array(Float32, 4, 4, 3, 2)
        _svdvals = reshape(mapslices(svdvals, A; dims=(1, 2)), 4, 3, 2)
        A_ra = Reactant.to_rarray(A)

        # FIXME(#2314): fix TPU lowering
        @test begin
            _svdvals_ra = @jit svdvals(A_ra; alg=alg)
            _svdvals_ra ≈ _svdvals
        end broken = RunningOnTPU
    end
end
