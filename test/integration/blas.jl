using LinearAlgebra, Reactant, Test
using LinearAlgebra: BLAS

@testset "Level 1" begin
    @testset "asum" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 32)
        x_ra = Reactant.to_rarray(x)

        @test @jit(BLAS.asum(length(x), x_ra, 1)) ≈ BLAS.asum(length(x), x, 1)
        @test @jit(BLAS.asum(3, x_ra, 5)) ≈ BLAS.asum(3, x, 5)

        y = Reactant.TestUtils.construct_test_array(Complex{Float32}, 32)
        y_ra = Reactant.to_rarray(y)

        @test @jit(BLAS.asum(length(y), y_ra, 1)) ≈ BLAS.asum(length(y), y, 1)
        @test @jit(BLAS.asum(3, y_ra, 5)) ≈ BLAS.asum(3, y, 5)
    end

    @testset "dot" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 32)
        y = Reactant.TestUtils.construct_test_array(Float32, 32)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(dot(x_ra, y_ra)) ≈ dot(x, y)
        @test @jit(BLAS.dot(length(x), x_ra, 1, y_ra, 1)) ≈ BLAS.dot(length(x), x, 1, y, 1)

        z1 = Reactant.TestUtils.construct_test_array(Complex{Float32}, 32)
        z2 = Reactant.TestUtils.construct_test_array(Complex{Float32}, 32) .+ 3.0f0
        z1_ra = Reactant.to_rarray(z1)
        z2_ra = Reactant.to_rarray(z2)

        @test @jit(BLAS.dotu(z1_ra, z2_ra)) ≈ BLAS.dotu(z1, z2)
        @test @jit(BLAS.dotc(z1_ra, z2_ra)) ≈ BLAS.dotc(z1, z2)
    end

    @testset "scal!" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 32)
        x_ra = Reactant.to_rarray(x)

        @test @jit(BLAS.scal(2.0f0, x_ra)) ≈ BLAS.scal(length(x), 2.0f0, x, 1)

        x_ra = Reactant.to_rarray(x)
        @jit BLAS.scal!(3, 2.0f0, x_ra, 5)
        x_target = copy(x)
        BLAS.scal!(3, 2.0f0, x_target, 5)
        @test x_ra ≈ x_target
    end

    @testset "nrm2" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 32)
        x_ra = Reactant.to_rarray(x)

        @test @jit(BLAS.nrm2(length(x), x_ra, 1)) ≈ BLAS.nrm2(length(x), x, 1)
        @test @jit(BLAS.nrm2(x_ra)) ≈ BLAS.nrm2(x)
    end

    @testset "rot!" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 32)
        y = Reactant.TestUtils.construct_test_array(Float32, 32)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)
        c, s = 0.6f0, 0.8f0

        @jit BLAS.rot!(x_ra, y_ra, c, s)
        x_target, y_target = copy(x), copy(y)
        BLAS.rot!(length(x_target), x_target, 1, y_target, 1, c, s)
        @test x_ra ≈ x_target
        @test y_ra ≈ y_target
    end

    @testset "iamax" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 32)
        x_ra = Reactant.to_rarray(x)
        @test @jit(BLAS.iamax(x_ra)) == BLAS.iamax(x)
    end

    @testset "copy!" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 32)
        y = Reactant.TestUtils.construct_test_array(Float32, 32)
        x_ra, y_ra = Reactant.to_rarray(x), Reactant.to_rarray(y)
        @jit BLAS.blascopy!(length(x), x_ra, 1, y_ra, 1)
        @test y_ra ≈ x
    end
end

@testset "Level 2" begin
    @testset "gemv!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        x = Reactant.TestUtils.construct_test_array(Float32, 16)
        y = Reactant.TestUtils.construct_test_array(Float32, 16)
        A_ra, x_ra, y_ra = Reactant.to_rarray(A),
        Reactant.to_rarray(x),
        Reactant.to_rarray(y)

        @jit BLAS.gemv!('N', 2.0f0, A_ra, x_ra, 3.0f0, y_ra)
        y_target = copy(y)
        BLAS.gemv!('N', 2.0f0, A, x, 3.0f0, y_target)
        @test y_ra ≈ y_target
    end

    @testset "ger!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        x = Reactant.TestUtils.construct_test_array(Float32, 16)
        y = Reactant.TestUtils.construct_test_array(Float32, 16)
        A_ra, x_ra, y_ra = Reactant.to_rarray(A),
        Reactant.to_rarray(x),
        Reactant.to_rarray(y)

        @jit BLAS.ger!(2.0f0, x_ra, y_ra, A_ra)
        A_target = copy(A)
        BLAS.ger!(2.0f0, x, y, A_target)
        @test A_ra ≈ A_target
    end

    @testset "symv!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A = A + A'
        x = Reactant.TestUtils.construct_test_array(Float32, 16)
        y = Reactant.TestUtils.construct_test_array(Float32, 16)
        A_ra, x_ra, y_ra = Reactant.to_rarray(A),
        Reactant.to_rarray(x),
        Reactant.to_rarray(y)

        @jit BLAS.symv!('U', 2.0f0, A_ra, x_ra, 3.0f0, y_ra)
        y_target = copy(y)
        BLAS.symv!('U', 2.0f0, A, x, 3.0f0, y_target)
        @test y_ra ≈ y_target
    end

    @testset "hemv!" begin
        A = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        A = A + A'
        x = Reactant.TestUtils.construct_test_array(ComplexF32, 16)
        y = Reactant.TestUtils.construct_test_array(ComplexF32, 16)
        A_ra, x_ra, y_ra = Reactant.to_rarray(A),
        Reactant.to_rarray(x),
        Reactant.to_rarray(y)

        @jit BLAS.hemv!('U', ComplexF32(2.0f0), A_ra, x_ra, ComplexF32(3.0f0), y_ra)
        y_target = copy(y)
        BLAS.hemv!('U', ComplexF32(2.0f0), A, x, ComplexF32(3.0f0), y_target)
        @test y_ra ≈ y_target
    end

    @testset "syr!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A = A + A'
        x = Reactant.TestUtils.construct_test_array(Float32, 16)
        A_ra, x_ra = Reactant.to_rarray(A), Reactant.to_rarray(x)

        @jit BLAS.syr!('U', 2.0f0, x_ra, A_ra)
        A_target = copy(A)
        BLAS.syr!('U', 2.0f0, x, A_target)
        @test UpperTriangular(A_ra) ≈ UpperTriangular(A_target)
    end

    @testset "her!" begin
        A = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        A = A + A'
        x = Reactant.TestUtils.construct_test_array(ComplexF32, 16)
        A_ra, x_ra = Reactant.to_rarray(A), Reactant.to_rarray(x)

        @jit BLAS.her!('U', 2.0f0, x_ra, A_ra)
        A_target = copy(A)
        BLAS.her!('U', 2.0f0, x, A_target)
        @test Hermitian(A_ra, :U) ≈ Hermitian(A_target, :U)
    end

    if isdefined(BLAS, :geru!)
        @testset "geru!" begin
            A = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
            x = Reactant.TestUtils.construct_test_array(ComplexF32, 16)
            y = Reactant.TestUtils.construct_test_array(ComplexF32, 16)
            A_ra, x_ra, y_ra = Reactant.to_rarray(A),
            Reactant.to_rarray(x),
            Reactant.to_rarray(y)

            @jit BLAS.geru!(ComplexF32(2.0f0), x_ra, y_ra, A_ra)
            A_target = copy(A)
            BLAS.geru!(ComplexF32(2.0f0), x, y, A_target)
            @test A_ra ≈ A_target
        end
    end

    @testset "symv/hemv" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A = A + A'
        x = Reactant.TestUtils.construct_test_array(Float32, 16)
        A_ra, x_ra = Reactant.to_rarray(A), Reactant.to_rarray(x)
        @test @jit(BLAS.symv('U', 2.0f0, A_ra, x_ra)) ≈ BLAS.symv('U', 2.0f0, A, x)

        Ac = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        Ac = Ac + Ac'
        xc = Reactant.TestUtils.construct_test_array(ComplexF32, 16)
        Ac_ra, xc_ra = Reactant.to_rarray(Ac), Reactant.to_rarray(xc)
        @test @jit(BLAS.hemv('U', ComplexF32(2.0f0), Ac_ra, xc_ra)) ≈
            BLAS.hemv('U', ComplexF32(2.0f0), Ac, xc)
    end

    @testset "trmv!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        x = Reactant.TestUtils.construct_test_array(Float32, 16)
        A_ra, x_ra = Reactant.to_rarray(A), Reactant.to_rarray(x)

        @jit BLAS.trmv!('U', 'N', 'N', A_ra, x_ra)
        x_target = copy(x)
        BLAS.trmv!('U', 'N', 'N', A, x_target)
        @test x_ra ≈ x_target
    end

    @testset "trsv!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A = A' * A + I # ensure invertible
        x = Reactant.TestUtils.construct_test_array(Float32, 16)
        A_ra, x_ra = Reactant.to_rarray(A), Reactant.to_rarray(x)

        @jit BLAS.trsv!('U', 'N', 'N', A_ra, x_ra)
        x_target = copy(x)
        BLAS.trsv!('U', 'N', 'N', A, x_target)
        @test x_ra ≈ x_target
    end

    @testset "gemv/trmv/trsv" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        x = Reactant.TestUtils.construct_test_array(Float32, 16)
        A_ra, x_ra = Reactant.to_rarray(A), Reactant.to_rarray(x)

        @test @jit(BLAS.gemv('N', 2.0f0, A_ra, x_ra)) ≈ BLAS.gemv('N', 2.0f0, A, x)
        @test @jit(BLAS.trmv('U', 'N', 'N', A_ra, x_ra)) ≈ BLAS.trmv('U', 'N', 'N', A, x)

        Ainv = A' * A + I
        Ainv_ra = Reactant.to_rarray(Ainv)
        @test @jit(BLAS.trsv('U', 'N', 'N', Ainv_ra, x_ra)) ≈
            BLAS.trsv('U', 'N', 'N', Ainv, x)
    end
end

@testset "Level 3" begin
    @testset "gemm!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        B = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        C = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A_ra, B_ra, C_ra = Reactant.to_rarray(A),
        Reactant.to_rarray(B),
        Reactant.to_rarray(C)

        @jit BLAS.gemm!('N', 'N', 2.0f0, A_ra, B_ra, 3.0f0, C_ra)
        C_target = copy(C)
        BLAS.gemm!('N', 'N', 2.0f0, A, B, 3.0f0, C_target)
        @test C_ra ≈ C_target
    end

    @testset "trsm!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        B = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A = A' * A + I # ensure invertible
        A_ra, B_ra = Reactant.to_rarray(A), Reactant.to_rarray(B)

        @jit BLAS.trsm!('L', 'U', 'N', 'N', 2.0f0, A_ra, B_ra)
        B_target = copy(B)
        BLAS.trsm!('L', 'U', 'N', 'N', 2.0f0, A, B_target)
        @test B_ra ≈ B_target
    end

    @testset "syrk!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        C = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        C = C + C' # ensure symmetric
        A_ra, C_ra = Reactant.to_rarray(A), Reactant.to_rarray(C)

        @jit BLAS.syrk!('U', 'N', 2.0f0, A_ra, 3.0f0, C_ra)
        C_target = copy(C)
        BLAS.syrk!('U', 'N', 2.0f0, A, 3.0f0, C_target)
        @test UpperTriangular(C_ra) ≈ UpperTriangular(C_target)

        # test 'L' and 'T'
        A2 = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        C2 = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        C2 = C2 + C2' # ensure symmetric
        A2_ra, C2_ra = Reactant.to_rarray(A2), Reactant.to_rarray(C2)

        @jit BLAS.syrk!('L', 'T', 2.0f0, A2_ra, 3.0f0, C2_ra)
        C2_target = copy(C2)
        BLAS.syrk!('L', 'T', 2.0f0, A2, 3.0f0, C2_target)
        @test LowerTriangular(C2_ra) ≈ LowerTriangular(C2_target)
    end

    @testset "gemmt!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        B = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        C = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A_ra, B_ra, C_ra = Reactant.to_rarray(A),
        Reactant.to_rarray(B),
        Reactant.to_rarray(C)

        @jit BLAS.gemmt!('U', 'N', 'N', 2.0f0, A_ra, B_ra, 3.0f0, C_ra)
        C_target = copy(C)
        BLAS.gemmt!('U', 'N', 'N', 2.0f0, A, B, 3.0f0, C_target)
        @test UpperTriangular(C_ra) ≈ UpperTriangular(C_target)
    end

    @testset "symm!/hemm!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A = A + A'
        B = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        C = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A_ra, B_ra, C_ra = Reactant.to_rarray(A),
        Reactant.to_rarray(B),
        Reactant.to_rarray(C)

        @jit BLAS.symm!('L', 'U', 2.0f0, A_ra, B_ra, 3.0f0, C_ra)
        C_target = copy(C)
        BLAS.symm!('L', 'U', 2.0f0, A, B, 3.0f0, C_target)
        @test C_ra ≈ C_target

        Ac = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        Ac = Ac + Ac'
        Bc = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        Cc = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        Ac_ra, Bc_ra, Cc_ra = Reactant.to_rarray(Ac),
        Reactant.to_rarray(Bc),
        Reactant.to_rarray(Cc)

        @jit BLAS.hemm!('L', 'U', ComplexF32(2.0f0), Ac_ra, Bc_ra, ComplexF32(3.0f0), Cc_ra)
        Cc_target = copy(Cc)
        BLAS.hemm!('L', 'U', ComplexF32(2.0f0), Ac, Bc, ComplexF32(3.0f0), Cc_target)
        @test Cc_ra ≈ Cc_target
    end

    @testset "trmm!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        B = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A_ra, B_ra = Reactant.to_rarray(A), Reactant.to_rarray(B)

        @jit BLAS.trmm!('L', 'U', 'N', 'N', 2.0f0, A_ra, B_ra)
        B_target = copy(B)
        BLAS.trmm!('L', 'U', 'N', 'N', 2.0f0, A, B_target)
        @test B_ra ≈ B_target
    end

    @testset "syr2k!" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        B = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        C = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        C = C + C'
        A_ra, B_ra, C_ra = Reactant.to_rarray(A),
        Reactant.to_rarray(B),
        Reactant.to_rarray(C)

        @jit BLAS.syr2k!('U', 'N', 2.0f0, A_ra, B_ra, 3.0f0, C_ra)
        C_target = copy(C)
        BLAS.syr2k!('U', 'N', 2.0f0, A, B, 3.0f0, C_target)
        @test UpperTriangular(C_ra) ≈ UpperTriangular(C_target)
    end

    @testset "herk!/her2k!" begin
        Ac = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        Bc = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        Cc = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        Cc = Cc + Cc'
        Ac_ra, Bc_ra, Cc_ra = Reactant.to_rarray(Ac),
        Reactant.to_rarray(Bc),
        Reactant.to_rarray(Cc)

        @jit BLAS.herk!('U', 'N', 2.0f0, Ac_ra, 3.0f0, Cc_ra)
        Cc_target = copy(Cc)
        BLAS.herk!('U', 'N', 2.0f0, Ac, 3.0f0, Cc_target)
        @test Hermitian(Cc_ra, :U) ≈ Hermitian(Cc_target, :U)

        Cc2_ra = Reactant.to_rarray(Cc) # reset
        @jit BLAS.her2k!('U', 'N', ComplexF32(2.0f0), Ac_ra, Bc_ra, 3.0f0, Cc2_ra)
        Cc2_target = copy(Cc)
        BLAS.her2k!('U', 'N', ComplexF32(2.0f0), Ac, Bc, 3.0f0, Cc2_target)
        @test Hermitian(Cc2_ra, :U) ≈ Hermitian(Cc2_target, :U)
    end

    @testset "Non-mutating Level 3" begin
        A = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        B = Reactant.TestUtils.construct_test_array(Float32, 16, 16)
        A_ra, B_ra = Reactant.to_rarray(A), Reactant.to_rarray(B)

        @test @jit(BLAS.gemm('N', 'N', 2.0f0, A_ra, B_ra)) ≈
            BLAS.gemm('N', 'N', 2.0f0, A, B)
        @test UpperTriangular(@jit(BLAS.gemmt('U', 'N', 'N', 2.0f0, A_ra, B_ra))) ≈
            UpperTriangular(BLAS.gemmt('U', 'N', 'N', 2.0f0, A, B))

        Ainv = A' * A + I
        Ainv_ra = Reactant.to_rarray(Ainv)
        @test @jit(BLAS.symm('L', 'U', 2.0f0, Ainv_ra, B_ra)) ≈
            BLAS.symm('L', 'U', 2.0f0, Ainv, B)
        @test @jit(BLAS.trmm('L', 'U', 'N', 'N', 2.0f0, Ainv_ra, B_ra)) ≈
            BLAS.trmm('L', 'U', 'N', 'N', 2.0f0, Ainv, B)
        @test @jit(BLAS.trsm('L', 'U', 'N', 'N', 2.0f0, Ainv_ra, B_ra)) ≈
            BLAS.trsm('L', 'U', 'N', 'N', 2.0f0, Ainv, B)

        @test UpperTriangular(@jit(BLAS.syrk('U', 'N', 2.0f0, A_ra))) ≈
            UpperTriangular(BLAS.syrk('U', 'N', 2.0f0, A))
        @test UpperTriangular(@jit(BLAS.syr2k('U', 'N', 2.0f0, A_ra, B_ra))) ≈
            UpperTriangular(BLAS.syr2k('U', 'N', 2.0f0, A, B))

        Ac = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        Bc = Reactant.TestUtils.construct_test_array(ComplexF32, 16, 16)
        Acinv = Ac' * Ac + I
        Ac_ra, Bc_ra, Acinv_ra = Reactant.to_rarray(Ac),
        Reactant.to_rarray(Bc),
        Reactant.to_rarray(Acinv)

        @test @jit(BLAS.hemm('L', 'U', ComplexF32(2.0f0), Acinv_ra, Bc_ra)) ≈
            BLAS.hemm('L', 'U', ComplexF32(2.0f0), Acinv, Bc)
        @test Hermitian(@jit(BLAS.herk('U', 'N', 2.0f0, Ac_ra)), :U) ≈
            Hermitian(BLAS.herk('U', 'N', 2.0f0, Ac), :U)
        @test Hermitian(@jit(BLAS.her2k('U', 'N', ComplexF32(2.0f0), Ac_ra, Bc_ra)), :U) ≈
            Hermitian(BLAS.her2k('U', 'N', ComplexF32(2.0f0), Ac, Bc), :U)
    end
end
