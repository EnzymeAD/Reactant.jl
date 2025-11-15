using LuxLib, Reactant, Enzyme, NNlib, Test

@testset "Fused Dense" begin
    sumabs2fuseddense(act, weight, x, bias) =
        sum(abs2, fused_dense_bias_activation(act, weight, x, bias))

    function ∇fuseddense(act, weight, x, bias)
        return Enzyme.gradient(Reverse, sumabs2fuseddense, Const(act), weight, x, bias)[2:end]
    end

    function ∇fuseddense_fd(act, weight, x, bias)
        return Reactant.TestUtils.finite_difference_gradient(
            (w, x, b) -> sumabs2fuseddense(act, w, x, b), weight, x, bias
        )
    end

    @testset for act in (identity, relu, sigmoid, tanh, gelu), has_bias in (true, false)
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

            dw_fd, dx_fd, db_fd = @jit ∇fuseddense_fd(act, weight_ra, x_ra, bias_ra)

            @test dw_fd ≈ dw_compile atol = 1e-5 rtol = 1e-2
            @test dx_fd ≈ dx_compile atol = 1e-5 rtol = 1e-2
            has_bias && @test db_fd ≈ db_compile atol = 1e-5 rtol = 1e-2
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

    function ∇biasact_fd(act, x, b)
        return Reactant.TestUtils.finite_difference_gradient(
            (x, b) -> sumabs2biasact(act, x, b), x, b
        )
    end

    function ∇biasact!!(act, x, b)
        return Enzyme.gradient(Reverse, sumabs2biasact!!, Const(act), x, b)[2:end]
    end
    function ∇biasact!!_fd(act, x, b)
        return Reactant.TestUtils.finite_difference_gradient(
            (x, b) -> sumabs2biasact!!(act, x, b), x, b
        )
    end

    @testset for act in (identity, relu, sigmoid, tanh, gelu)
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

        @testset "Enzyme: bias_activation" begin
            ∂x_enz, ∂b_enz = @jit ∇biasact_fd(act, x_ra, b_ra)
            ∂x_compile, ∂b_compile = @jit ∇biasact(act, x_ra, b_ra)

            @test ∂x_enz ≈ ∂x_compile atol = 1e-5 rtol = 1e-2
            @test ∂b_enz ≈ ∂b_compile atol = 1e-5 rtol = 1e-2
        end

        @testset "Enzyme: bias_activation!!" begin
            ∂x_enz!!, ∂b_enz!! = @jit ∇biasact!!_fd(act, x_ra, b_ra)
            ∂x_compile!!, ∂b_compile!! = @jit ∇biasact!!(act, x_ra, b_ra)

            @test ∂x_enz!! ≈ ∂x_compile!! atol = 1e-5 rtol = 1e-2
            @test ∂b_enz!! ≈ ∂b_compile!! atol = 1e-5 rtol = 1e-2
        end
    end
end

@testset "Fast Activation" begin
    # Here we are testing that fast_activation doesn't switch to the faster versions
    sumabs2(f, x) = sum(abs2, fast_activation(f, x))
    sumabs2!!(f, x) = sum(abs2, fast_activation!!(f, copy(x)))

    ∇sumabs2(f, x) = Enzyme.gradient(Reverse, sumabs2, Const(f), x)[2]
    ∇sumabs2_fd(f, x) = Reactant.TestUtils.finite_difference_gradient(x -> sumabs2(f, x), x)
    ∇sumabs2!!(f, x) = Enzyme.gradient(Reverse, sumabs2!!, Const(f), x)[2]
    ∇sumabs2!!_fd(f, x) =
        Reactant.TestUtils.finite_difference_gradient(x -> sumabs2!!(f, x), x)

    x_act = Reactant.TestUtils.construct_test_array(Float32, 10, 10)
    x_act_ca = Reactant.to_rarray(x_act)

    @testset "Activation: $act" for act in (
        identity, relu, sigmoid, tanh, tanh_fast, sigmoid_fast, gelu, abs2
    )
        y_simple = sumabs2(act, x_act)
        y_simple!! = sumabs2!!(act, x_act)
        y_compile = @jit sumabs2(act, x_act_ca)
        y_compile!! = @jit sumabs2!!(act, x_act_ca)

        @test y_simple ≈ y_compile atol = 1e-5 rtol = 1e-2
        @test y_simple!! ≈ y_compile!! atol = 1e-5 rtol = 1e-2

        ∂x_enz = @jit ∇sumabs2_fd(act, x_act_ca)
        ∂x_compile = @jit ∇sumabs2(act, x_act_ca)
        ∂x_enz!! = @jit ∇sumabs2!!_fd(act, x_act_ca)
        ∂x_compile!! = @jit ∇sumabs2!!(act, x_act_ca)

        @test ∂x_enz ≈ ∂x_compile atol = 1e-5 rtol = 1e-2
        @test ∂x_enz!! ≈ ∂x_compile!! atol = 1e-5 rtol = 1e-2
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
