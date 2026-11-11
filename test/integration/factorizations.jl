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

solve_tridiag_2707(dl, d, du, b) = Tridiagonal(dl, d, du) \ b
solve_tridiag_trans_2707(dl, d, du, b) = transpose(Tridiagonal(dl, d, du)) \ b
solve_tridiag_adj_2707(dl, d, du, b) = adjoint(Tridiagonal(dl, d, du)) \ b
solve_symtridiag_2707(dv, ev, b) = SymTridiagonal(dv, ev) \ b
function solve_tridiag_ldiv_2707(dl, d, du, b)
    out = copy(b)
    ldiv!(Tridiagonal(dl, d, du), out)
    return out
end
function solve_symtridiag_shift_2707(dv, ev, b, s)
    out = copy(b)
    ldiv!(SymTridiagonal(dv, ev), out; shift=s)
    return out
end

@testset "Tridiagonal & SymTridiagonal Solve (#2707)" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        (T == ComplexF64 || T == Float64) && RunningOnTPU && continue

        dl = T[1, -0.5, 0.25, 0.5]
        d = T[4, 5, 4, 6, 5]
        du = T[0.5, 1, -0.25, 0.75]
        if T <: Complex
            dl = dl .+ im .* T[0.1, -0.2, 0.3, -0.1]
            d = d .+ im .* T[0.5, -0.4, 0.2, 0.3, -0.5]
            du = du .+ im .* T[-0.2, 0.1, 0.4, -0.3]
        end
        b = T[1, 2, 3, 4, 5]
        B = reshape(T[1:15...], 5, 3)

        dl_r = Reactant.to_rarray(dl)
        d_r = Reactant.to_rarray(d)
        du_r = Reactant.to_rarray(du)
        b_r = Reactant.to_rarray(b)
        B_r = Reactant.to_rarray(B)

        @test Array(@jit(solve_tridiag_2707(dl_r, d_r, du_r, b_r))) ≈
            solve_tridiag_2707(dl, d, du, b) atol = 1e-4 rtol = 1e-3
        @test Array(@jit(solve_tridiag_2707(dl_r, d_r, du_r, B_r))) ≈
            solve_tridiag_2707(dl, d, du, B) atol = 1e-4 rtol = 1e-3
        @test Array(@jit(solve_tridiag_2707(dl_r, d_r, du_r, b))) ≈
            solve_tridiag_2707(dl, d, du, b) atol = 1e-4 rtol = 1e-3
        @test Array(@jit(solve_tridiag_2707(dl, d, du, b_r))) ≈
            solve_tridiag_2707(dl, d, du, b) atol = 1e-4 rtol = 1e-3
        @test Array(@jit(solve_tridiag_trans_2707(dl_r, d_r, du_r, b_r))) ≈
            solve_tridiag_trans_2707(dl, d, du, b) atol = 1e-4 rtol = 1e-3
        @test Array(@jit(solve_tridiag_adj_2707(dl_r, d_r, du_r, b_r))) ≈
            solve_tridiag_adj_2707(dl, d, du, b) atol = 1e-4 rtol = 1e-3
        @test Array(@jit(solve_tridiag_ldiv_2707(dl_r, d_r, du_r, b_r))) ≈
            solve_tridiag_2707(dl, d, du, b) atol = 1e-4 rtol = 1e-3

        @test Array(@jit(solve_symtridiag_2707(d_r, dl_r, b_r))) ≈
            solve_symtridiag_2707(d, dl, b) atol = 1e-4 rtol = 1e-3
        @test Array(@jit(solve_symtridiag_2707(d_r, dl_r, B_r))) ≈
            solve_symtridiag_2707(d, dl, B) atol = 1e-4 rtol = 1e-3
        @test Array(@jit(solve_symtridiag_shift_2707(d_r, dl_r, b_r, real(T)(0.5)))) ≈
            solve_symtridiag_shift_2707(d, dl, b, real(T)(0.5)) atol = 1e-4 rtol = 1e-3
        @test Array(
            @jit(solve_symtridiag_shift_2707(d_r, dl_r, b_r, ConcreteRNumber(real(T)(0.5))))
        ) ≈ solve_symtridiag_shift_2707(d, dl, b, real(T)(0.5)) atol = 1e-4 rtol = 1e-3
    end
end

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

@testset "svd factorization ($T)" for T in (Float32, ComplexF32)
    A = Reactant.TestUtils.construct_test_array(T, 4, 8)

    algs = get_svd_algorithms(string(Reactant.devices()[1]), size(A))

    tmp = rand(T, 8, 5)
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

    A = Reactant.TestUtils.construct_test_array(T, 4, 8, 3, 2)
    A_ra = Reactant.to_rarray(A)

    tmp = rand(T, 8, 5, 3, 2)
    B = similar(A, T, 4, 5, 3, 2)
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

    @testset "Un-batched: $(alg) - $T" for alg in algs, T in (Float32, ComplexF32)
        A = Reactant.TestUtils.construct_test_array(T, 4, 4)
        _svdvals = svdvals(A)
        A_ra = Reactant.to_rarray(A)

        # FIXME(#2314): fix TPU lowering
        @test begin
            _svdvals_ra = @jit svdvals(A_ra; alg=alg)
            _svdvals_ra ≈ _svdvals
        end broken = RunningOnTPU
    end

    @testset "Batched: $(alg) - $T" for alg in algs, T in (Float32, ComplexF32)
        A = Reactant.TestUtils.construct_test_array(T, 4, 4, 3, 2)
        _svdvals = reshape(mapslices(svdvals, A; dims=(1, 2)), 4, 3, 2)
        A_ra = Reactant.to_rarray(A)

        # FIXME(#2314): fix TPU lowering
        @test begin
            _svdvals_ra = @jit svdvals(A_ra; alg=alg)
            _svdvals_ra ≈ _svdvals
        end broken = RunningOnTPU
    end
end
