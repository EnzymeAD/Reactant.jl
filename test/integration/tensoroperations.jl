using Test
using Reactant
using Reactant.TestUtils: construct_test_array
using TensorOperations

fadd(α, X) = @tensor Y[a, b, c] := α * X[c, a, b]

@testset "tensoradd!" begin
    α = 1.32
    x = construct_test_array(Float64, 2, 3, 4)

    α_re = ConcreteRNumber(α)
    x_re = Reactant.to_rarray(x)
    @test fadd(α, x) ≈ @jit fadd(α, x_re)
    @test fadd(α, x) ≈ @jit fadd(α_re, x_re)
end

ftrace(X) = @tensor Y[l, k, m] := X[l, i, i, k, j, j, m]

@testset "tensortrace!" begin
    a = ones(3, 2, 2, 3, 2, 2, 3)
    a_re = Reactant.to_rarray(a)
end

fcontract(A, B) = @tensor C[] := A[a, b, c] * B[c, a, b]

@testset "tensorcontract!" begin
    a = construct_test_array(Float64, 5, 5, 5)
    b = construct_test_array(Float64, 5, 5, 5)

    a_re = Reactant.to_rarray(a)
    b_re = Reactant.to_rarray(b)
    @test fcontract(a, b) ≈ @jit fcontract(a_re, b_re)
end

fall(α, A, B, C, D) = @tensor begin
    D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
end

@testset "all" begin
    α = 1.32
    a = construct_test_array(Float64, 5, 5, 5, 5, 5, 5)
    b = construct_test_array(Float64, 5, 5, 5)
    c = construct_test_array(Float64, 5, 5, 5)
    d = zeros(5, 5, 5)

    α_re = ConcreteRNumber(α)
    a_re = Reactant.to_rarray(a)
    b_re = Reactant.to_rarray(b)
    c_re = Reactant.to_rarray(c)
    d_re = Reactant.to_rarray(d)

    @test fall(α, a, b, c, d) ≈ @jit fall(α_re, a_re, b_re, c_re, d_re)
    @test d ≈ d_re
end
