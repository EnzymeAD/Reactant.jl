using NNlib, Reactant, Enzyme, Statistics, Test, LuxLib

@testset "Batched Matrix Multiplication" begin
    Reactant.with_config(;
        convolution_precision=PrecisionConfig.HIGH,
        dot_general_precision=PrecisionConfig.HIGH,
    ) do
        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3, 5)
        y = Reactant.TestUtils.construct_test_array(Float32, 3, 2, 5)

        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(batched_mul(x_ra, y_ra)) ≈ batched_mul(x, y)

        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3, 1)
        y = Reactant.TestUtils.construct_test_array(Float32, 3, 2, 5)

        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(batched_mul(x_ra, y_ra)) ≈ batched_mul(x, y)

        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3, 5)
        y = Reactant.TestUtils.construct_test_array(Float32, 3, 2, 1)

        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(batched_mul(x_ra, y_ra)) ≈ batched_mul(x, y)
    end
end

@testset "Constant Padding: NNlib.pad_constant" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 4, 4)
    x_ra = Reactant.to_rarray(x)

    # Symmetric Padding
    @test @jit(NNlib.pad_constant(x_ra, (1, 1))) ≈ NNlib.pad_constant(x, (1, 1))
    @test @jit(NNlib.pad_constant(x_ra, (1, 1, 1, 1))) ≈ NNlib.pad_constant(x, (1, 1, 1, 1))

    # Asymmetric Padding
    @test @jit(NNlib.pad_constant(x_ra, (1, 3, 2, 1))) ≈ NNlib.pad_constant(x, (1, 3, 2, 1))
    @test @jit(NNlib.pad_constant(x_ra, (1, 0))) ≈ NNlib.pad_constant(x, (1, 0))

    # Symmetric Padding with value (test type-casting)
    @test @jit(NNlib.pad_constant(x_ra, (1, 1), 2)) ≈ NNlib.pad_constant(x, (1, 1), 2)
    @test @jit(NNlib.pad_constant(x_ra, (1, 1, 1, 1), 2)) ≈
        NNlib.pad_constant(x, (1, 1, 1, 1), 2)

    # Asymmetric Padding with value (test type-casting)
    @test @jit(NNlib.pad_constant(x_ra, (1, 3, 2, 1), 2)) ≈
        NNlib.pad_constant(x, (1, 3, 2, 1), 2)
    @test @jit(NNlib.pad_constant(x_ra, (1, 0), 2)) ≈ NNlib.pad_constant(x, (1, 0), 2)

    # pad_zeros just forward to pad_constant
    @test @jit(NNlib.pad_zeros(x_ra, (1, 1))) ≈ NNlib.pad_zeros(x, (1, 1))
    @test @jit(NNlib.pad_zeros(x_ra, (1, 1, 1, 1))) ≈ NNlib.pad_zeros(x, (1, 1, 1, 1))

    sumabs2(f, x) = sum(abs2, f(x))
    ∇sumabs2(f, x) = Enzyme.gradient(Reverse, sumabs2, Const(f), x)[2]

    pad_fn = Base.Fix2(NNlib.pad_constant, (1, 1, 1, 1))
    @test @jit(∇sumabs2(pad_fn, x_ra)) ≈ 2 .* x

    pad_fn2 = Base.Fix2(NNlib.pad_constant, (1, 0, 1, 3))
    @test @jit(∇sumabs2(pad_fn2, x_ra)) ≈ 2 .* x

    x = Reactant.TestUtils.construct_test_array(ComplexF32, 4, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(NNlib.pad_constant(x_ra, (1, 1))) ≈ NNlib.pad_constant(x, (1, 1))
end

@testset "make_causal_mask" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(NNlib.make_causal_mask(x_ra)) ≈ NNlib.make_causal_mask(x)

    causal_mask2(x) = NNlib.make_causal_mask(x; dims=1)
    @test @jit(causal_mask2(x_ra)) ≈ causal_mask2(x)
end

@testset "softmax/logsoftmax reshaped input" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 3, 4, 5)
    x_ra = reshape(Reactant.to_rarray(x), 12, 5)
    x = reshape(x, 12, 5)

    @test @jit(NNlib.softmax(x_ra)) ≈ NNlib.softmax(x)
    @test @jit(NNlib.logsoftmax(x_ra)) ≈ NNlib.logsoftmax(x)
end

@testset "logsumexp #1593" begin
    x = collect(Float32, 1:16)
    x_ra = Reactant.to_rarray(x)

    y = logsumexp(x)
    y_ra = @jit(logsumexp(x_ra))
    @test Float32(y_ra) ≈ y
end

@testset "Fused Dense" begin
    sumabs2fuseddense(act, weight, x, bias) =
        sum(abs2, fused_dense_bias_activation(act, weight, x, bias))

    function ∇fuseddense(act, weight, x, bias)
        return Enzyme.gradient(Reverse, sumabs2fuseddense, Const(act), weight, x, bias)[2:end]
    end

    function ∇fuseddense(actgradfn, act, weight, x, bias)
        z = weight * x
        if bias !== nothing
            z .+= bias
        end
        Ω = act.(z)

        δ = 2 .* actgradfn.(Ω, z) .* Ω

        ∂weight = δ * x'
        if bias !== nothing
            ∂bias = vec(sum(δ; dims=2))
        else
            ∂bias = nothing
        end
        ∂x = weight' * δ
        return ∂weight, ∂x, ∂bias
    end

    @testset "Activation: $act | bias=$has_bias" for (act, gradfn) in (
            (identity, (Ω, x) -> one(Ω)),
            (relu, (Ω, x) -> (Ω > 0)),
            (sigmoid, (Ω, x) -> conj((1 - Ω) * Ω)),
            (tanh, (Ω, x) -> conj(1 - Ω^2)),
            (gelu, (Ω, x) -> NNlib.deriv_gelu_tanh(x)),
        ),
        has_bias in (true, false)

        weight = Reactant.TestUtils.construct_test_array(Float32, 9, 10)
        x = Reactant.TestUtils.construct_test_array(Float32, 10, 12)
        bias = has_bias ? Reactant.TestUtils.construct_test_array(Float32, 9) : nothing

        weight_ra = Reactant.to_rarray(weight)
        x_ra = Reactant.to_rarray(x)
        bias_ra = Reactant.to_rarray(bias)

        y_compile = @jit fused_dense_bias_activation(act, weight_ra, x_ra, bias_ra)
        y_res = fused_dense_bias_activation(act, weight, x, bias)
        @test y_res ≈ y_compile atol = 1e-5 rtol = 1e-2

        @testset "Enzyme: fused_dense_bias_activation" begin
            dw_compile, dx_compile, db_compile = @jit ∇fuseddense(
                act, weight_ra, x_ra, bias_ra
            )

            dw_gt, dx_gt, db_gt = @jit ∇fuseddense(gradfn, act, weight_ra, x_ra, bias_ra)

            @test dw_gt ≈ dw_compile atol = 1e-5 rtol = 1e-2
            @test dx_gt ≈ dx_compile atol = 1e-5 rtol = 1e-2
            has_bias && @test db_gt ≈ db_compile atol = 1e-5 rtol = 1e-2
        end
    end
end
