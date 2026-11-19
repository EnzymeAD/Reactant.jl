using Reactant
using GPUArraysCore: @allowscalar
using Test

const GLOBAL_TEST_RARRAY = Reactant.to_rarray(collect(1.0:8.0))
const GLOBAL_TEST_RNUMBER = Reactant.ConcreteRNumber(2.5)

function subfn_issue_1687!(u, i, r)
    @trace if @allowscalar(GLOBAL_TEST_RARRAY[i]^2) < r^2
        @allowscalar u[i] = 5.0
    else
        @allowscalar u[i] = 1.0
    end
    return nothing
end

function foo_issue_1687!(u, r)
    @trace for i in 1:8
        @trace subfn_issue_1687!(u, i, r)
    end
    return nothing
end

function global_slice_read(y)
    return GLOBAL_TEST_RARRAY[1:4] .+ y
end

function global_scalar_setindex!(y)
    @allowscalar GLOBAL_TEST_RARRAY[1] = 42.0
    return y
end

function global_traced_setindex!(y)
    idx = @allowscalar y[1]
    @allowscalar GLOBAL_TEST_RARRAY[idx] = 42.0
    return y
end

function global_fill!(y)
    fill!(GLOBAL_TEST_RARRAY, 0.0)
    return y
end

function global_copyto!(y)
    copyto!(GLOBAL_TEST_RARRAY, y)
    return y
end

function global_bcast_assign!(y)
    GLOBAL_TEST_RARRAY .= y
    return y
end

function global_number_add_rhs(x)
    return x + GLOBAL_TEST_RNUMBER
end

function global_number_mul_lhs(x)
    return GLOBAL_TEST_RNUMBER * x
end

@testset "Untraced global ConcreteRArray / ConcreteRNumber error messages (#1687)" begin
    u = Reactant.to_rarray(zeros(8))
    y4 = Reactant.to_rarray(ones(4))
    y8 = Reactant.to_rarray(ones(8))
    yi = Reactant.to_rarray(Int[2])
    xn = Reactant.ConcreteRNumber(3.0)

    @test_throws ArgumentError @jit foo_issue_1687!(u, 3.0)
    err_1687 = try
        @jit foo_issue_1687!(u, 3.0)
        nothing
    catch e
        e
    end
    @test err_1687 isa ArgumentError
    @test occursin("global variable", sprint(showerror, err_1687))
    @test occursin("getindex", sprint(showerror, err_1687))

    @test_throws ArgumentError @jit global_slice_read(y4)
    @test_throws ArgumentError @jit global_scalar_setindex!(y8)
    @test_throws ArgumentError @jit global_traced_setindex!(yi)
    @test_throws ArgumentError @jit global_fill!(y8)
    @test_throws ArgumentError @jit global_copyto!(y8)
    @test_throws ArgumentError @jit global_bcast_assign!(y8)
    @test_throws ArgumentError @jit global_number_add_rhs(xn)
    @test_throws ArgumentError @jit global_number_mul_lhs(xn)

    # Ensure host values of GLOBAL_TEST_RARRAY were never mutated during failed compilation
    @test Array(GLOBAL_TEST_RARRAY) == collect(1.0:8.0)

    # Eager operations outside @compile / @jit still work normally
    eager_copy = copy(GLOBAL_TEST_RARRAY)
    @test Array(eager_copy[1:4]) == collect(1.0:4.0)
    @allowscalar eager_copy[2] = 99.0
    @test @allowscalar(eager_copy[2]) == 99.0
    fill!(eager_copy, 7.0)
    @test Array(eager_copy) == fill(7.0, 8)
    @test GLOBAL_TEST_RNUMBER + 1.5 == 4.0
end
