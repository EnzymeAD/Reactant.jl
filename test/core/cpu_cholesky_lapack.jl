using Reactant
using LinearAlgebra
using Enzyme
using Test

chol_factors_fn(S) = cholesky(Hermitian(S)).factors
chol_factors_lower_fn(S) = cholesky(Hermitian(S, :L)).factors
chol_solve_fn(S, b) = cholesky(Hermitian(S)) \ b
chol_raw_fn(S) = cholesky(S; check=false)
ops_chol_lower(A) = Reactant.Ops.cholesky(A; lower=true)
ops_chol_upper(A) = Reactant.Ops.cholesky(A; lower=false)
chol_loss_fn(A, b) = sum(abs2, (cholesky(A * A' + 4.0 * I) \ b))

function make_spd_matrix(::Type{T}, n::Int, scale::Real) where {T}
    M = if T <: Complex
        R = real(T)
        Complex{
            R
        }.(
            reshape(collect(R, 1:(n * n)), n, n) ./ R(n),
            reshape(collect(R, (n * n):-1:1), n, n) ./ R(2 * n),
        )
    else
        reshape(collect(T, 1:(n * n)), n, n) ./ T(n)
    end
    return Matrix{T}(M * M' + T(scale) * I)
end

@testset "CPU Cholesky LAPACK lowering (#3411)" begin
    if contains(string(Reactant.devices()[1]), "CPU")
        @testset "Lowers to LAPACK potrf without XLA Cholesky while-loop" begin
            S = make_spd_matrix(Float64, 16, 16.0)
            rS = Reactant.to_rarray(S)
            xla_ir = sprint(show, @code_xla chol_factors_fn(rS))
            @test occursin("enzymexla_compile_cpu", xla_ir)
            @test !occursin("xla.cholesky", xla_ir)
        end

        @testset "2D Cholesky and solve across element types ($T)" for T in (
            Float32, Float64, ComplexF32, ComplexF64
        )
            S = make_spd_matrix(T, 12, 12.0)
            b = if T <: Complex
                Complex{real(T)}.(collect(1:12), collect(12:-1:1))
            else
                collect(T, 1:12)
            end
            B = hcat(b, T(2) .* b)

            rS = Reactant.to_rarray(S)
            rb = Reactant.to_rarray(b)
            rB = Reactant.to_rarray(B)

            ref_U = Matrix(cholesky(Hermitian(S)).U)
            got_factors = Array(@jit chol_factors_fn(rS))
            @test triu(got_factors) ≈ ref_U rtol = sqrt(eps(real(T)))
            @test all(iszero, tril(got_factors, -1))

            got_vec = Array(@jit chol_solve_fn(rS, rb))
            @test got_vec ≈ (cholesky(Hermitian(S)) \ b) rtol = 10 * sqrt(eps(real(T)))

            got_mat = Array(@jit chol_solve_fn(rS, rB))
            @test got_mat ≈ (cholesky(Hermitian(S)) \ B) rtol = 10 * sqrt(eps(real(T)))

            # Direct Ops.cholesky lower=true and lower=false with garbage in opposite triangle
            S_noisy_for_lower_mlir = copy(S)
            # Julia (row, col) -> MLIR (col-1, row-1); Ops.cholesky(lower=false) reads MLIR upper
            # triangle (col-1 <= row-1, i.e. Julia row >= col = Julia lower triangle).
            ref_ops_u = Array(@jit optimize = :none ops_chol_upper(rS))
            got_ops_u = Array(@jit ops_chol_upper(rS))
            @test got_ops_u ≈ ref_ops_u rtol = sqrt(eps(real(T)))

            ref_ops_l = Array(@jit optimize = :none ops_chol_lower(rS))
            got_ops_l = Array(@jit ops_chol_lower(rS))
            @test got_ops_l ≈ ref_ops_l rtol = sqrt(eps(real(T)))
        end

        @testset "Non-positive-definite matrix produces NaN and issuccess == false" begin
            bad = Float64[-1.0 0.0; 0.0 2.0]
            rbad = Reactant.to_rarray(bad)
            F = @jit chol_raw_fn(rbad)
            @test all(isnan, Array(F.factors))
            @test !Bool(F.info)
        end

        @testset "Batched 3D and 4D Cholesky on CPU" begin
            n = 6
            S1 = make_spd_matrix(Float64, n, 8.0)
            S2 = make_spd_matrix(Float64, n, 14.0)
            S3 = make_spd_matrix(Float64, n, 20.0)
            batch3 = cat(S1, S2, S3; dims=3)
            b3 = reshape(collect(Float64, 1:(n * 2 * 3)), n, 2, 3)

            rbatch3 = Reactant.to_rarray(batch3)
            rb3 = Reactant.to_rarray(b3)

            F3 = @jit chol_raw_fn(rbatch3)
            fac3 = Array(F3.factors)
            for k in 1:3
                @test triu(fac3[:, :, k]) ≈ Matrix(cholesky(Hermitian(batch3[:, :, k])).U)
                @test all(iszero, tril(fac3[:, :, k], -1))
            end
            @test Array(F3.info) == Bool[true, true, true]

            sol3 = Array(@jit(\(F3, rb3)))
            for k in 1:3
                @test sol3[:, :, k] ≈ (cholesky(Hermitian(batch3[:, :, k])) \ b3[:, :, k])
            end

            # 4D batch with one non-SPD slice
            batch4 = reshape(cat(S1, S2, -S1, S3; dims=3), n, n, 2, 2)
            F4 = @jit chol_raw_fn(Reactant.to_rarray(batch4))
            info4 = Array(F4.info)
            @test info4[1, 1] == true
            @test info4[2, 1] == true
            @test info4[1, 2] == false
            @test info4[2, 2] == true
            @test all(isnan, Array(F4.factors)[:, :, 1, 2])
        end

        @testset "Solve inside differentiable pipeline" begin
            A = make_spd_matrix(Float64, 8, 8.0)
            b = Float64[1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.5, 1.5]
            db = Float64[0.1, -0.2, 0.3, -0.1, 0.2, 0.05, -0.15, 0.25]

            rA = Reactant.to_rarray(A)
            rb = Reactant.to_rarray(b)
            rdb = Reactant.to_rarray(db)

            grad_b_ref = zeros(size(b))
            Enzyme.autodiff(
                Enzyme.Reverse,
                chol_loss_fn,
                Enzyme.Active,
                Enzyme.Const(A),
                Enzyme.Duplicated(copy(b), grad_b_ref),
            )
            grad_b_got = @jit Enzyme.gradient(
                Enzyme.Reverse, Const(chol_loss_fn), Const(rA), rb
            )
            @test Array(grad_b_got[2]) ≈ grad_b_ref rtol = 1e-6

            fwd_b_ref = Enzyme.autodiff(
                Enzyme.Forward,
                chol_loss_fn,
                Enzyme.Duplicated,
                Enzyme.Const(A),
                Enzyme.Duplicated(b, db),
            )
            fwd_b_got = @jit Enzyme.autodiff(
                Enzyme.Forward,
                Const(chol_loss_fn),
                Enzyme.Duplicated,
                Const(rA),
                Enzyme.Duplicated(rb, rdb),
            )
            @test Float64(fwd_b_got[1]) ≈ fwd_b_ref[1] rtol = 1e-6
        end
    end
end
