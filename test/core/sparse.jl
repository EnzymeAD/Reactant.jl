using Reactant, Test, LinearAlgebra, SparseArrays, FileCheck
using Random: Random

const RunningOnGPU =
    contains(string(Reactant.devices()[1]), "CUDA") ||
    contains(string(Reactant.devices()[1]), "ROCM")

spmv(A, x) = A * x
spmm(A, B) = A * B
spmv_mul!(C, A, B, α, β) = LinearAlgebra.mul!(C, A, B, α, β)

@testset "CSRMatrix construction and tracing" begin
    rng = Random.MersenneTwister(0)
    A = sprand(rng, Float64, 10, 8, 0.3)

    Acsr = Reactant.CSRMatrix(A)
    @test size(Acsr) == (10, 8)
    @test eltype(Acsr) == Float64
    @test nnz(Acsr) == nnz(A)
    @test SparseMatrixCSC(Acsr) == A

    A_ra = Reactant.to_rarray(A)
    @test A_ra isa Reactant.CSRMatrix
    @test A_ra.rowptr isa ConcreteRArray
    @test A_ra.colind isa ConcreteRArray
    @test A_ra.nzval isa ConcreteRArray
    A_rt = SparseMatrixCSC(A_ra)
    @test nnz(A_rt) == nnz(A)
    # TPUs emulate Float64, so the value round trip is only approximate there.
    @test A_rt ≈ A
end

@testset "sparse_tensor IR" begin
    rng = Random.MersenneTwister(0)
    A_ra = Reactant.to_rarray(sprand(rng, Float64, 10, 8, 0.3))
    x_ra = Reactant.to_rarray(rand(rng, 8))
    B_ra = Reactant.to_rarray(rand(rng, 8, 3))

    hlo = @code_hlo optimize = :none spmv(A_ra, x_ra)
    @test @filecheck begin
        @check "#sparse_tensor.encoding"
        @check "sparse_tensor.assemble"
        @check "enzymexla.sparse.spmm"
        hlo
    end

    hlo = @code_hlo optimize = :none spmm(A_ra, B_ra)
    @test @filecheck begin
        @check "sparse_tensor.assemble"
        @check "enzymexla.sparse.spmm"
        hlo
    end
end

@testset "lowering to custom_call" begin
    rng = Random.MersenneTwister(0)
    A_ra = Reactant.to_rarray(sprand(rng, Float64, 10, 8, 0.3))
    x_ra = Reactant.to_rarray(rand(rng, 8))
    B_ra = Reactant.to_rarray(rand(rng, 8, 3))
    C_ra = Reactant.to_rarray(rand(rng, 10))

    for hlo in (@code_hlo(spmv(A_ra, x_ra)), @code_hlo(spmm(A_ra, B_ra)))
        @test @filecheck begin
            @check "stablehlo.custom_call"
            @check "reactant_csr_matmul"
            hlo
        end
        @test !contains(repr(hlo), "sparse_tensor")
    end

    # Constant α/β are fused into a single accumulating library call with C
    # aliased to the output.
    hlo = @code_hlo spmv_mul!(C_ra, A_ra, x_ra, 2.0, 3.0)
    @test @filecheck begin
        @check "stablehlo.custom_call"
        @check "reactant_csr_matmul_acc"
        @check "output_operand_alias"
        hlo
    end
    @test !contains(repr(hlo), "sparse_tensor")

    # Traced (runtime) α/β fall back to explicit scaling around the plain
    # product.
    α_rn = Reactant.ConcreteRNumber(2.0)
    β_rn = Reactant.ConcreteRNumber(3.0)
    hlo = @code_hlo spmv_mul!(C_ra, A_ra, x_ra, α_rn, β_rn)
    @test @filecheck begin
        @check "stablehlo.custom_call"
        @check "reactant_csr_matmul"
        @check "stablehlo.multiply"
        @check "stablehlo.add"
        hlo
    end
    @test !contains(repr(hlo), "reactant_csr_matmul_acc")
    @test !contains(repr(hlo), "sparse_tensor")
end

@testset "numerical correctness" begin
    if !RunningOnGPU
        @test_skip "CSR matmul execution requires a CUDA or ROCm backend"
    else
        rng = Random.MersenneTwister(0)
        @testset for T in (Float32, Float64), Ti in (Int32, Int64)
            A = SparseMatrixCSC{T,Ti}(sprand(rng, T, 10, 8, 0.3))
            x = rand(rng, T, 8)
            B = rand(rng, T, 8, 3)
            C = rand(rng, T, 10)

            A_ra = Reactant.to_rarray(A)
            x_ra = Reactant.to_rarray(x)
            B_ra = Reactant.to_rarray(B)

            @test Array(@jit(spmv(A_ra, x_ra))) ≈ A * x atol = 1e-5 rtol = 1e-5
            @test Array(@jit(spmm(A_ra, B_ra))) ≈ A * B atol = 1e-5 rtol = 1e-5

            C_ra = Reactant.to_rarray(C)
            @jit spmv_mul!(C_ra, A_ra, x_ra, T(2), T(3))
            @test Array(C_ra) ≈ 2 .* (A * x) .+ 3 .* C atol = 1e-5 rtol = 1e-5

            # runtime α/β
            C_ra = Reactant.to_rarray(C)
            α_rn = Reactant.ConcreteRNumber(T(2))
            β_rn = Reactant.ConcreteRNumber(T(3))
            @jit spmv_mul!(C_ra, A_ra, x_ra, α_rn, β_rn)
            @test Array(C_ra) ≈ 2 .* (A * x) .+ 3 .* C atol = 1e-5 rtol = 1e-5
        end
    end
end
