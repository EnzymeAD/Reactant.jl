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
fcontract2(A, B) = @tensor C[d, e] := A[a, d, b, c] * B[c, a, e, b]

function backend_contract(b, A, B)
    @tensor backend = b C1[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
end

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

    @testset let
        a = construct_test_array(Float64, 3, 20, 5, 3, 4)
        b = construct_test_array(Float64, 5, 6, 20, 3)
        backendlist = (
            TensorOperations.BaseCopy(),
            TensorOperations.BaseView(),
            TensorOperations.StridedNative(),
            TensorOperations.StridedBLAS(),
        )
        for backend in backendlist
            @test backend_contract(backend, a, b) ≈ @jit backend_contract(backend, a, b)
        end
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

# Writing through a StridedView of a traced array has to reach its parent, as when TensorKit
# hands TensorOperations a view of a tensor's flat storage as the output
function fwrite_view!(x, y, makeview)
    Reactant.TracedUtils.set_mlir_data!(makeview(x), Reactant.TracedUtils.get_mlir_data(y))
    return x
end

@testset "writes through StridedViews" begin
    x = construct_test_array(Float64, 24)
    for (name, makeview, sz) in (
        ("sreshape", x -> TensorOperations.sreshape(TensorOperations.StridedView(x), (2, 3, 4)), (2, 3, 4)),
        # offset and non-unit strides, like a TensorKit subblock
        ("strided", x -> TensorOperations.StridedView(x, (2, 3), (1, 4), 1), (2, 3)),
    )
        @testset "$name" begin
            y = construct_test_array(Float64, sz...) .+ 100
            expected = copy(x)
            v = makeview(expected)
            for I in CartesianIndices(v)
                v[I] = y[I]
            end
            x_re = Reactant.to_rarray(x)
            @test @jit(fwrite_view!(x_re, Reactant.to_rarray(y), makeview)) ≈ expected
        end
    end

    @testset "conj" begin
        z = complex.(construct_test_array(Float64, 6), construct_test_array(Float64, 6) .+ 1)
        w = complex.(construct_test_array(Float64, 6) .+ 2, construct_test_array(Float64, 6))
        makeview(z) = conj(TensorOperations.StridedView(z))
        @test @jit(fwrite_view!(Reactant.to_rarray(z), Reactant.to_rarray(w), makeview)) ≈ conj(w)
    end
end
