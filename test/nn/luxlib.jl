using LuxLib, Reactant, Enzyme, NNlib

@testset "Fused Dense" begin
    sumabs2fuseddense(act, weight, x, bias) =
        sum(abs2, fused_dense_bias_activation(act, weight, x, bias))

    function ∇fuseddense(act, weight, x, bias)
        dw = Enzyme.make_zero(weight)
        dx = Enzyme.make_zero(x)
        db = bias === nothing ? nothing : Enzyme.make_zero(bias)
        b_dup = bias === nothing ? Const(bias) : Duplicated(bias, db)
        Enzyme.autodiff(
            Reverse,
            sumabs2fuseddense,
            Active,
            Const(act),
            Duplicated(weight, dw),
            Duplicated(x, dx),
            b_dup,
        )
        return dw, dx, db
    end

    @testset for act in (identity, relu, sigmoid, tanh, gelu), has_bias in (true, false)
        weight = randn(Float32, 9, 10)
        x = randn(Float32, 10, 12)
        bias = has_bias ? randn(Float32, 9) : nothing

        weight_ra = Reactant.to_rarray(weight)
        x_ra = Reactant.to_rarray(x)
        bias_ra = Reactant.to_rarray(bias)

        y_compile = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGHEST,
            convolution_precision=PrecisionConfig.HIGHEST,
        ) do
            @jit fused_dense_bias_activation(act, weight_ra, x_ra, bias_ra)
        end

        y_res = fused_dense_bias_activation(act, weight, x, bias)

        @test y_res ≈ y_compile atol = 1e-5 rtol = 1e-2

        @testset "Enzyme: fused_dense_bias_activation" begin
            dw, dx, db = ∇fuseddense(act, weight, x, bias)

            dw_compile, dx_compile, db_compile = Reactant.with_config(;
                dot_general_precision=PrecisionConfig.HIGHEST,
                convolution_precision=PrecisionConfig.HIGHEST,
            ) do
                @jit ∇fuseddense(act, weight_ra, x_ra, bias_ra)
            end

            @test dw ≈ dw_compile atol = 1e-5 rtol = 1e-2
            @test dx ≈ dx_compile atol = 1e-5 rtol = 1e-2
            has_bias && @test db ≈ db_compile atol = 1e-5 rtol = 1e-2
        end
    end
end

@testset "Bias Activation" begin
    biasact(act, x, b) = bias_activation(act, x, b)
    sumabs2biasact(act, x, b) = sum(abs2, biasact(act, x, b))
    biasact!!(act, x, b) = bias_activation!!(act, copy(x), b)
    sumabs2biasact!!(act, x, b) = sum(abs2, biasact!!(act, x, b))

    function ∇biasact(act, x, b)
        dx = Enzyme.make_zero(x)
        db = Enzyme.make_zero(b)
        Enzyme.autodiff(
            Reverse,
            sumabs2biasact,
            Active,
            Const(act),
            Duplicated(x, dx),
            Duplicated(b, db),
        )
        return dx, db
    end

    function ∇biasact!!(act, x, b)
        dx = Enzyme.make_zero(x)
        db = Enzyme.make_zero(b)
        Enzyme.autodiff(
            Reverse,
            sumabs2biasact!!,
            Active,
            Const(act),
            Duplicated(x, dx),
            Duplicated(b, db),
        )
        return dx, db
    end

    @testset for act in (identity, relu, sigmoid, tanh, gelu)
        x = randn(Float32, 10, 10)
        b = randn(Float32, 10)

        x_ra = Reactant.to_rarray(x)
        b_ra = Reactant.to_rarray(b)

        y_compile = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGHEST,
            convolution_precision=PrecisionConfig.HIGHEST,
        ) do
            @jit biasact(act, x_ra, b_ra)
        end

        y_compile!! = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGHEST,
            convolution_precision=PrecisionConfig.HIGHEST,
        ) do
            @jit biasact!!(act, x_ra, b_ra)
        end

        y_simple = biasact(act, x, b)
        y_simple!! = biasact!!(act, x, b)

        @test y_simple ≈ y_compile atol = 1e-5 rtol = 1e-2
        @test y_simple!! ≈ y_compile!! atol = 1e-5 rtol = 1e-2

        @testset "Enzyme: bias_activation" begin
            ∂x_enz, ∂b_enz = ∇biasact(act, x, b)
            ∂x_compile, ∂b_compile = Reactant.with_config(;
                dot_general_precision=PrecisionConfig.HIGHEST,
                convolution_precision=PrecisionConfig.HIGHEST,
            ) do
                @jit ∇biasact(act, x_ra, b_ra)
            end

            @test ∂x_enz ≈ ∂x_compile atol = 1e-5 rtol = 1e-2
            @test ∂b_enz ≈ ∂b_compile atol = 1e-5 rtol = 1e-2
        end

        @testset "Enzyme: bias_activation!!" begin
            ∂x_enz!!, ∂b_enz!! = ∇biasact!!(act, x, b)
            ∂x_compile!!, ∂b_compile!! = Reactant.with_config(;
                dot_general_precision=PrecisionConfig.HIGHEST,
                convolution_precision=PrecisionConfig.HIGHEST,
            ) do
                @jit ∇biasact!!(act, x_ra, b_ra)
            end

            @test ∂x_enz!! ≈ ∂x_compile!! atol = 1e-5 rtol = 1e-2
            @test ∂b_enz!! ≈ ∂b_compile!! atol = 1e-5 rtol = 1e-2
        end
    end
end

@testset "Fast Activation" begin
    # Here we are testing that fast_activation doesn't switch to the faster versions
    sumabs2(f, x) = sum(abs2, fast_activation(f, x))
    sumabs2!!(f, x) = sum(abs2, fast_activation!!(f, copy(x)))

    function ∇sumabs2(f, x)
        dx = Enzyme.make_zero(x)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(f), Duplicated(x, dx))
        return dx
    end

    function ∇sumabs2!!(f, x)
        dx = Enzyme.make_zero(x)
        Enzyme.autodiff(Reverse, sumabs2!!, Active, Const(f), Duplicated(x, dx))
        return dx
    end

    x_act = randn(Float32, 10, 10)
    x_act_ca = Reactant.to_rarray(x_act)

    @testset "Activation: $act" for act in (
        identity, relu, sigmoid, tanh, tanh_fast, sigmoid_fast, gelu, abs2
    )
        y_simple = sumabs2(act, x_act)
        y_simple!! = sumabs2!!(act, x_act)
        y_compile = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGHEST,
            convolution_precision=PrecisionConfig.HIGHEST,
        ) do
            @jit sumabs2(act, x_act_ca)
        end
        y_compile!! = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGHEST,
            convolution_precision=PrecisionConfig.HIGHEST,
        ) do
            @jit sumabs2!!(act, x_act_ca)
        end

        @test y_simple ≈ y_compile atol = 1e-5 rtol = 1e-2
        @test y_simple!! ≈ y_compile!! atol = 1e-5 rtol = 1e-2

        ∂x_enz = Enzyme.make_zero(x_act)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(act), Duplicated(x_act, ∂x_enz))

        ∂x_enz!! = Enzyme.make_zero(x_act)
        Enzyme.autodiff(Reverse, sumabs2!!, Active, Const(act), Duplicated(x_act, ∂x_enz!!))

        ∂x_compile = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGHEST,
            convolution_precision=PrecisionConfig.HIGHEST,
        ) do
            @jit ∇sumabs2(act, x_act_ca)
        end

        ∂x_compile!! = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGHEST,
            convolution_precision=PrecisionConfig.HIGHEST,
        ) do
            @jit ∇sumabs2!!(act, x_act_ca)
        end

        @test ∂x_enz ≈ ∂x_compile atol = 1e-5 rtol = 1e-2
        @test ∂x_enz!! ≈ ∂x_compile!! atol = 1e-5 rtol = 1e-2
    end
end

@testset "Fused Conv" begin
    @testset for groups in (1, 2), has_bias in (true, false), act in (identity, relu, tanh)
        weight = randn(Float32, 4, 4, 8 ÷ groups, 4)
        x = randn(Float32, 16, 16, 8, 2)
        bias = has_bias ? randn(Float32, 4) : nothing

        weight_reactant = Reactant.to_rarray(weight)
        x_reactant = Reactant.to_rarray(x)
        bias_reactant = Reactant.to_rarray(bias)

        @testset for stride in ((1, 1), (3, 3)),
            padding in ((0, 0), (2, 2), (2, 0)),
            dilation in ((1, 1), (1, 2))

            conv_dims = DenseConvDims(x, weight; stride, padding, dilation, groups)

            reactant_res = Reactant.with_config(;
                dot_general_precision=PrecisionConfig.HIGHEST,
                convolution_precision=PrecisionConfig.HIGHEST,
            ) do
                @jit fused_conv_bias_activation(
                    act, weight_reactant, x_reactant, bias_reactant, conv_dims
                )
            end

            luxlib_res = fused_conv_bias_activation(act, weight, x, bias, conv_dims)

            @test reactant_res ≈ luxlib_res atol = 1e-5 rtol = 1e-2
        end

        # TODO: test for gradients
    end
end
