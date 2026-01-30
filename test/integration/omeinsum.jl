using Test
using Reactant
using OMEinsum

@testset "sum" begin
    a = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3)
    a_re = Reactant.to_rarray(a)

    f = ein"ij->"
    c = f(a)
    c_re = @jit f(a_re)
    @test c ≈ c_re

    f = ein"ij->i"
    c = f(a)
    c_re = @jit f(a_re)
    @test c ≈ c_re

    f = ein"ij->j"
    c = f(a)
    c_re = @jit f(a_re)
    @test c ≈ c_re
end

@testset "trace" begin
    a = Reactant.TestUtils.construct_test_array(ComplexF32, 3, 3)
    a_re = Reactant.to_rarray(a)

    f = ein"ii->"
    c = f(a)
    c_re = @jit f(a_re)
    @test c ≈ c_re
end

@testset "transpose" begin
    a = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3)
    a_re = Reactant.to_rarray(a)

    f = ein"ij->ji"
    c = f(a)
    c_re = @jit f(a_re)
    @test c ≈ c_re

    a = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 4, 5)
    a_re = Reactant.to_rarray(a)

    f = ein"ijkl->jilk"
    c = f(a)
    c_re = @jit f(a_re)
    @test c ≈ c_re
end

@testset "matmul" begin
    a = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3)
    b = Reactant.TestUtils.construct_test_array(ComplexF32, 3, 4)
    a_re = Reactant.to_rarray(a)
    b_re = Reactant.to_rarray(b)

    f = ein"ij,jk->ik"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re
end

@testset "hadamard product" begin
    a = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3)
    b = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3)
    a_re = Reactant.to_rarray(a)
    b_re = Reactant.to_rarray(b)

    f = ein"ij,ij->ij"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re

    b = Reactant.TestUtils.construct_test_array(ComplexF32, 3, 2)
    b_re = Reactant.to_rarray(b)
    f = ein"ij,ji->ij"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re
end

@testset "inner product" begin
    a = Reactant.TestUtils.construct_test_array(ComplexF32, 4, 3)
    b = Reactant.TestUtils.construct_test_array(ComplexF32, 3, 4)
    a_re = Reactant.to_rarray(a)
    b_re = Reactant.to_rarray(b)

    f = ein"ij,ji->"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re
end

@testset "outer product" begin
    a = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3)
    b = Reactant.TestUtils.construct_test_array(ComplexF32, 4, 5)
    a_re = Reactant.to_rarray(a)
    b_re = Reactant.to_rarray(b)

    f = ein"ij,kl->ijkl"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re

    f = ein"ij,kl->klij"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re

    f = ein"ij,kl->ikjl"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re
end

@testset "scale" begin
    a = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3)
    b = fill(ComplexF32(2.0))
    a_re = Reactant.to_rarray(a)
    b_re = Reactant.to_rarray(b)

    f = ein"ij,->ij"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re
end

@testset "batch matmul" begin
    a = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 3, 6)
    b = Reactant.TestUtils.construct_test_array(ComplexF32, 3, 4, 6)
    a_re = Reactant.to_rarray(a)
    b_re = Reactant.to_rarray(b)

    f = ein"ijb,jkb->ikb"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re

    f = ein"ijb,jkb->bik"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re
end

@testset "tensor contraction" begin
    a = Reactant.TestUtils.construct_test_array(ComplexF32, 2, 4, 3)
    b = Reactant.TestUtils.construct_test_array(ComplexF32, 3, 5, 4)
    a_re = Reactant.to_rarray(a)
    b_re = Reactant.to_rarray(b)

    f = ein"ijk,klj->il"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re

    # contraction of NOT all common indices
    f = ein"ijk,klj->ikl"
    c = f(a, b)
    c_re = @jit f(a_re, b_re)
    @test c ≈ c_re
end
