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
    @test ftrace(a) ≈ @jit ftrace(a_re)
end

fcontract1(A, B) = @tensor C[] := A[a, b, c] * B[c, a, b]
fcontract2(A, B) = @tensor C[d,e] := A[a, d, b, c] * B[c, a, e, b]

@testset "tensorcontract!" begin
    @testset let
        a = construct_test_array(Float64, 5, 5, 5)
        b = construct_test_array(Float64, 5, 5, 5)
        a_re = Reactant.to_rarray(a)
        b_re = Reactant.to_rarray(b)
        @test fcontract1(a, b) ≈ @jit fcontract1(a_re, b_re)
    end

    @testset let
        a = construct_test_array(Float64, 5, 2, 5, 5)
        b = construct_test_array(Float64, 5, 5, 3, 5)
        a_re = Reactant.to_rarray(a)
        b_re = Reactant.to_rarray(b)
        @test fcontract2(a, b) ≈ @jit fcontract2(a_re, b_re)
    end
end

function fall(α, A, B, C, D)
    E = @tensor begin
        D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
        E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    end
    return E
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
