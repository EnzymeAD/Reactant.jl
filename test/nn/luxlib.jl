using LuxLib, Reactant, Enzyme, NNlib, Test

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

@testset "Bias Activation" begin
    biasact(act, x, b) = bias_activation(act, x, b)
    sumabs2biasact(act, x, b) = sum(abs2, biasact(act, x, b))
    biasact!!(act, x, b) = bias_activation!!(act, copy(x), b)
    sumabs2biasact!!(act, x, b) = sum(abs2, biasact!!(act, x, b))

    function ∇biasact(act, x, b)
        return Enzyme.gradient(Reverse, sumabs2biasact, Const(act), x, b)[2:end]
    end

    function ∇biasact!!(act, x, b)
        return Enzyme.gradient(Reverse, sumabs2biasact!!, Const(act), x, b)[2:end]
    end

    function ∇biasact(gradfn, act, x, b)
        xb = x .+ b
        Ω = act.(xb)
        ∂x = 2 .* gradfn.(Ω, xb) .* Ω
        ∂b = vec(sum(∂x; dims=2))
        return ∂x, ∂b
    end

    @testset "Activation: $act" for (act, gradfn) in (
        (identity, (Ω, x) -> one(Ω)),
        (relu, (Ω, x) -> (Ω > 0)),
        (sigmoid, (Ω, x) -> conj((1 - Ω) * Ω)),
        (tanh, (Ω, x) -> conj(1 - Ω^2)),
        (gelu, (Ω, x) -> NNlib.deriv_gelu_tanh(x)),
    )
        x = Reactant.TestUtils.construct_test_array(Float32, 10, 10)
        b = Reactant.TestUtils.construct_test_array(Float32, 10)

        x_ra = Reactant.to_rarray(x)
        b_ra = Reactant.to_rarray(b)

        y_compile = @jit biasact(act, x_ra, b_ra)
        y_compile!! = @jit biasact!!(act, x_ra, b_ra)

        y_simple = biasact(act, x, b)
        y_simple!! = biasact!!(act, x, b)

        @test y_simple ≈ y_compile atol = 1e-5 rtol = 1e-2
        @test y_simple!! ≈ y_compile!! atol = 1e-5 rtol = 1e-2

        ∂x_gt, ∂b_gt = @jit ∇biasact(gradfn, act, x_ra, b_ra)

        @testset "Enzyme: bias_activation" begin
            ∂x_enz, ∂b_enz = @jit ∇biasact(act, x_ra, b_ra)

            @test ∂x_enz ≈ ∂x_gt atol = 1e-5 rtol = 1e-2
            @test ∂b_enz ≈ ∂b_gt atol = 1e-5 rtol = 1e-2
        end

        @testset "Enzyme: bias_activation!!" begin
            ∂x_enz!!, ∂b_enz!! = @jit ∇biasact!!(act, x_ra, b_ra)

            @test ∂x_enz!! ≈ ∂x_gt atol = 1e-5 rtol = 1e-2
            @test ∂b_enz!! ≈ ∂b_gt atol = 1e-5 rtol = 1e-2
        end
    end
end

@testset "Fast Activation" begin
    # Here we are testing that fast_activation doesn't switch to the faster versions
    sumabs2(f, x) = sum(abs2, fast_activation(f, x))
    sumabs2!!(f, x) = sum(abs2, fast_activation!!(f, copy(x)))

    ∇sumabs2(f, x) = Enzyme.gradient(Reverse, sumabs2, Const(f), x)[2]
    ∇sumabs2!!(f, x) = Enzyme.gradient(Reverse, sumabs2!!, Const(f), x)[2]

    function ∇sumabs2(gradfn, f, x)
        Ω = f.(x)
        return 2 .* gradfn.(Ω, x) .* Ω
    end

    x_act = Reactant.TestUtils.construct_test_array(Float32, 10, 10)
    x_act_ca = Reactant.to_rarray(x_act)

    @testset "Activation: $act" for (act, gradfn) in (
        (identity, (Ω, x) -> one(Ω)),
        (relu, (Ω, x) -> (Ω > 0)),
        (sigmoid, (Ω, x) -> conj((1 - Ω) * Ω)),
        (tanh, (Ω, x) -> conj(1 - Ω^2)),
        (tanh_fast, (Ω, x) -> conj(1 - Ω^2)),
        (sigmoid_fast, (Ω, x) -> conj((1 - Ω) * Ω)),
        (gelu, (Ω, x) -> NNlib.deriv_gelu_tanh(x)),
        (abs2, (Ω, x) -> (2 * x)),
        (relu6, (Ω, x) -> (Ω > 0) & (Ω < 6)),
    )
        y_simple = sumabs2(act, x_act)
        y_simple!! = sumabs2!!(act, x_act)
        y_compile = @jit sumabs2(act, x_act_ca)
        y_compile!! = @jit sumabs2!!(act, x_act_ca)

        @test y_simple ≈ y_compile atol = 1e-5 rtol = 1e-2
        @test y_simple!! ≈ y_compile!! atol = 1e-5 rtol = 1e-2

        ∂x_enz = @jit ∇sumabs2(act, x_act_ca)
        ∂x_enz!! = @jit ∇sumabs2!!(act, x_act_ca)
        ∂x_gt = @jit ∇sumabs2(gradfn, act, x_act_ca)

        @test ∂x_enz ≈ ∂x_gt atol = 1e-5 rtol = 1e-2
        @test ∂x_enz!! ≈ ∂x_gt atol = 1e-5 rtol = 1e-2
    end
end

@testset "Fused Conv" begin
    @testset for groups in (1, 2), has_bias in (true, false), act in (identity, relu, tanh)
        weight = Reactant.TestUtils.construct_test_array(Float32, 4, 4, 8 ÷ groups, 4)
        x = Reactant.TestUtils.construct_test_array(Float32, 16, 16, 8, 2)
        bias = has_bias ? Reactant.TestUtils.construct_test_array(Float32, 4) : nothing

        weight_reactant = Reactant.to_rarray(weight)
        x_reactant = Reactant.to_rarray(x)
        bias_reactant = Reactant.to_rarray(bias)

        @testset for stride in ((1, 1), (3, 3)),
            padding in ((0, 0), (2, 2), (2, 0)),
            dilation in ((1, 1), (1, 2))

            conv_dims = DenseConvDims(x, weight; stride, padding, dilation, groups)

            reactant_res = @jit fused_conv_bias_activation(
                act, weight_reactant, x_reactant, bias_reactant, conv_dims
            )

            luxlib_res = fused_conv_bias_activation(act, weight, x, bias, conv_dims)

            @test reactant_res ≈ luxlib_res atol = 1e-5 rtol = 1e-2
        end

        # TODO: test for gradients
    end
end
