using Reactant
using Test
using Enzyme

assume_sub_without(x) = x .- x
function assume_sub_nonan(x)
    y = Reactant.assume(x; no_nan=true)
    return y .- y
end

function assume_scalar_nonan(x)
    y = Reactant.assume(x; no_nan=true)
    return y - y
end

assume_mul_zero_without(x) = 0.0f0 .* (x .+ 1.0f0)
function assume_mul_zero_finite_arg(x)
    y = Reactant.assume(x; finite=true)
    return 0.0f0 .* (y .+ 1.0f0)
end

function assume_mul_zero_finite_intermediate(x)
    y = Reactant.assume(x .+ 1.0f0; finite=true)
    return 0.0f0 .* y
end

assume_log_sq_without(x) = log.(x .* x)
function assume_log_sq_nonneg(x)
    y = Reactant.assume(x; non_negative=true)
    return log.(y .* y)
end

function assume_chained(x)
    y1 = Reactant.assume(x; finite=true)
    y2 = Reactant.assume(y1; non_negative=true)
    return (y2 .- y2) .+ log.(y2 .* y2)
end

assume_transpose_without(A) = permutedims(A, (2, 1))
function assume_transpose_symmetric(A)
    B = Reactant.assume(A; symmetric=true)
    return permutedims(B, (2, 1))
end

assume_imag_without(z) = imag.(z)
function assume_imag_purely_real(z)
    w = Reactant.assume(z; purely_real=true)
    return imag.(w)
end

assume_real_without(z) = real.(z)
function assume_real_purely_imag(z)
    w = Reactant.assume(z; purely_imaginary=true)
    return real.(w)
end

function assume_view_slice(x)
    v = @view x[1:2]
    vy = Reactant.assume(v; no_nan=true)
    return (vy .- vy), isnan.(x)
end

function assume_after_prior_use(x)
    t = x .+ 1.0f0
    has_nan = isnan.(t)
    t_safe = Reactant.assume(t; no_nan=true)
    diff = t_safe .- t_safe
    return ifelse.(has_nan, 1.0f0, diff)
end

function assume_grad_primal(x)
    y = Reactant.assume(x; finite=true, non_negative=true)
    return sum(y .* y .+ (y .- y))
end

function assume_grad_reverse(x)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(Reverse, assume_grad_primal, Active, Duplicated(x, dx))
    return dx
end

function assume_grad_forward(x, dx)
    return Enzyme.autodiff(Forward, assume_grad_primal, Duplicated(x, dx))[1]
end

@testset "Reactant.assume" begin
    @testset "Host / untraced validation and pass-through" begin
        a = Float32[1.0, 2.0, 3.0]
        @test Reactant.assume(a; no_nan=true, finite=true, non_negative=true) === a
        @test Reactant.assume(3.5f0; finite=true) === 3.5f0
        m = Float32[1.0 2.0; 2.0 3.0]
        @test Reactant.assume(m; symmetric=true) === m
        @test Reactant.assume(a) === a

        @test_throws ArgumentError Reactant.assume(a; finite=true, no_nan=false)
        @test_throws ArgumentError Reactant.assume(a; symmetric=true)
        @test_throws ArgumentError Reactant.assume(zeros(Float32, 2, 3); symmetric=true)
    end

    @testset "no_nan assumption (scalar, 1D, 2D)" begin
        x1 = Reactant.to_rarray(Float32[1.0, 2.0, 3.0, 4.0])
        hlo_without = repr(Reactant.@code_hlo assume_sub_without(x1))
        hlo_with = repr(Reactant.@code_hlo assume_sub_nonan(x1))
        @test occursin("stablehlo.subtract", hlo_without)
        @test !occursin("stablehlo.subtract", hlo_with)
        @test Array(Reactant.@jit assume_sub_nonan(x1)) == zeros(Float32, 4)

        x2 = Reactant.to_rarray(Float32[1.0 2.0; 3.0 4.0])
        hlo_with_2d = repr(Reactant.@code_hlo assume_sub_nonan(x2))
        @test !occursin("stablehlo.subtract", hlo_with_2d)
        @test Array(Reactant.@jit assume_sub_nonan(x2)) == zeros(Float32, 2, 2)

        s = ConcreteRNumber(2.5f0)
        hlo_scalar = repr(Reactant.@code_hlo assume_scalar_nonan(s))
        @test !occursin("stablehlo.subtract", hlo_scalar)
        @test Number(Reactant.@jit assume_scalar_nonan(s)) == 0.0f0
    end

    @testset "finite assumption (argument and enzyme.batch intermediate)" begin
        x = Reactant.to_rarray(Float32[1.0, -2.0, 3.5, 0.0])
        hlo_without = repr(Reactant.@code_hlo assume_mul_zero_without(x))
        hlo_arg = repr(Reactant.@code_hlo assume_mul_zero_finite_arg(x))
        hlo_inter = repr(Reactant.@code_hlo assume_mul_zero_finite_intermediate(x))
        @test occursin("stablehlo.multiply", hlo_without)
        @test !occursin("stablehlo.multiply", hlo_arg)
        @test !occursin("stablehlo.multiply", hlo_inter)
        @test Array(Reactant.@jit assume_mul_zero_finite_arg(x)) == zeros(Float32, 4)
        @test Array(Reactant.@jit assume_mul_zero_finite_intermediate(x)) ==
            zeros(Float32, 4)
    end

    @testset "non_negative and chained assumptions" begin
        x = Reactant.to_rarray(Float32[1.0, 2.0, 4.0, 8.0])
        hlo_without = repr(Reactant.@code_hlo assume_log_sq_without(x))
        hlo_with = repr(Reactant.@code_hlo assume_log_sq_nonneg(x))
        # Without non_negative, log(x * x) keeps `multiply %arg0, %arg0` inside `log`.
        # With non_negative, log(x * x) simplifies to `2 * log(x)`.
        @test occursin("stablehlo.multiply %arg0, %arg0", hlo_without)
        @test !occursin("stablehlo.multiply %arg0, %arg0", hlo_with)
        @test Array(Reactant.@jit assume_log_sq_nonneg(x)) ≈
            2.0f0 .* log.(Float32[1, 2, 4, 8])

        hlo_chained = repr(Reactant.@code_hlo assume_chained(x))
        @test !occursin("stablehlo.subtract", hlo_chained)
        @test !occursin("stablehlo.multiply %arg0, %arg0", hlo_chained)
        @test Array(Reactant.@jit assume_chained(x)) ≈ 2.0f0 .* log.(Float32[1, 2, 4, 8])
    end

    @testset "symmetric assumption" begin
        m_host = Float32[1.0 2.0 3.0; 2.0 4.0 5.0; 3.0 5.0 6.0]
        m = Reactant.to_rarray(m_host)
        hlo_without = repr(Reactant.@code_hlo assume_transpose_without(m))
        hlo_with = repr(Reactant.@code_hlo assume_transpose_symmetric(m))
        @test occursin("stablehlo.transpose", hlo_without)
        @test !occursin("stablehlo.transpose", hlo_with)
        @test Array(Reactant.@jit assume_transpose_symmetric(m)) == m_host
    end

    @testset "purely_real and purely_imaginary assumptions" begin
        zr_host = ComplexF32[1.0 + 0.0im, -2.5 + 0.0im, 3.0 + 0.0im]
        zr = Reactant.to_rarray(zr_host)
        hlo_imag_without = repr(Reactant.@code_hlo assume_imag_without(zr))
        hlo_imag_with = repr(Reactant.@code_hlo assume_imag_purely_real(zr))
        @test occursin("stablehlo.imag", hlo_imag_without)
        @test !occursin("stablehlo.imag", hlo_imag_with)
        @test Array(Reactant.@jit assume_imag_purely_real(zr)) == zeros(Float32, 3)

        zi_host = ComplexF32[0.0 + 1.5im, 0.0 - 4.0im]
        zi = Reactant.to_rarray(zi_host)
        hlo_real_without = repr(Reactant.@code_hlo assume_real_without(zi))
        hlo_real_with = repr(Reactant.@code_hlo assume_real_purely_imag(zi))
        @test occursin("stablehlo.real", hlo_real_without)
        @test !occursin("stablehlo.real", hlo_real_with)
        @test Array(Reactant.@jit assume_real_purely_imag(zi)) == zeros(Float32, 2)
    end

    @testset "SubArray view and prior-use isolation" begin
        x_host = Float32[1.0, 2.0, NaN32, 4.0]
        x = Reactant.to_rarray(x_host)
        zeros_slice, nan_mask = Reactant.@jit assume_view_slice(x)
        @test Array(zeros_slice) == zeros(Float32, 2)
        @test Array(nan_mask) == Bool[false, false, true, false]

        res = Array(Reactant.@jit assume_after_prior_use(x))
        @test res == Float32[0.0, 0.0, 1.0, 0.0]
    end

    @testset "Differentiation and attribute stripping" begin
        x_host = Float32[1.0, 2.0, 3.0]
        dx_host = Float32[0.5, -1.0, 2.0]
        x = Reactant.to_rarray(x_host)
        dx = Reactant.to_rarray(dx_host)

        grad = Array(Reactant.@jit assume_grad_reverse(x))
        @test grad ≈ 2.0f0 .* x_host

        fgrad = Number(Reactant.@jit assume_grad_forward(x, dx))
        @test fgrad ≈ sum(2.0f0 .* x_host .* dx_host)

        ctx = Reactant.ReactantContext()
        mod, _ = Reactant.Compiler.compile_mlir(
            ctx, assume_sub_nonan, (x,); drop_unsupported_attributes=true
        )
        ir_str = repr(mod)
        @test !occursin("enzymexla.no_nan", ir_str)
        @test !occursin("#enzymexla.guaranteed", ir_str)
    end
end
