using Enzyme, Reactant, Test, FileCheck

reuse_square(x) = x .* x

function reuse_forward(f, x, dx1, dx2)
    d1 = only(Enzyme.autodiff(Forward, f, Duplicated, Duplicated(x, dx1)))
    d2 = only(Enzyme.autodiff(Forward, f, Duplicated, Duplicated(x, dx2)))
    return d1, d2, x, dx1, dx2
end

function reuse_square!(x)
    x .*= x
    return nothing
end

function reuse_mutating_forward(x, dx1, dx2)
    Enzyme.autodiff(Forward, reuse_square!, Const, Duplicated(x, dx1))
    Enzyme.autodiff(Forward, reuse_square!, Const, Duplicated(x, dx2))
    return x, dx1, dx2
end

reuse_scale(x, scale) = x .* scale

function reuse_static_arguments(x, dx)
    d1 = only(Enzyme.autodiff(Forward, reuse_scale, Duplicated(x, dx), Const(2)))
    d2 = only(Enzyme.autodiff(Forward, reuse_scale, Duplicated(x, dx), Const(3)))
    return d1, d2
end

@testset "AD primal callee reuse and mutation tracking" begin
    # A nonsquare matrix exercises the transpose inserted at the tracing boundary.
    # That transpose must not make the primal argument appear to have been mutated.
    x = reshape(Float32[1, 2, 3, 4, 5, 6], 2, 3)
    dx1 = fill(2.0f0, size(x))
    dx2 = reshape(Float32[6, 5, 4, 3, 2, 1], size(x))
    inputs = Reactant.to_rarray.((x, dx1, dx2))

    hlo = @code_hlo optimize = false reuse_forward(reuse_square, inputs...)
    @test @filecheck begin
        @check "enzyme.fwddiff @[[CALLEE:(\"[^\"]+\"|[^ (]+)]]("
        @check "enzyme.fwddiff @[[CALLEE]]("
        @check_not "enzyme.fwddiff"
        hlo
    end

    @testset "batching=$batching" for batching in (false, true)
        options = CompileOptions(; ad_optimization_passes=batching)
        result = @jit compile_options = options reuse_forward(reuse_square, inputs...)
        @test Array(result[1]) ≈ 2 .* x .* dx1
        @test Array(result[2]) ≈ 2 .* x .* dx2
        @test Array(result[3]) == x
        @test Array(result[4]) == dx1
        @test Array(result[5]) == dx2

        # Returning an unchanged argument must still produce a derivative result.
        result = @jit compile_options = options reuse_forward(identity, inputs...)
        @test Array(result[1]) == dx1
        @test Array(result[2]) == dx2

        # Reusing the callee must preserve real writes between derivative requests.
        mutated_inputs = Reactant.to_rarray.((x, dx1, dx2))
        result = @jit compile_options = options reuse_mutating_forward(mutated_inputs...)
        @test Array(result[1]) ≈ x .^ 4
        @test Array(result[2]) ≈ 2 .* x .* dx1
        @test Array(result[3]) ≈ 2 .* x .^ 2 .* dx2

        result = @jit compile_options = options reuse_static_arguments(inputs[1], inputs[2])
        @test Array(result[1]) == 2 .* dx1
        @test Array(result[2]) == 3 .* dx1
    end
end
