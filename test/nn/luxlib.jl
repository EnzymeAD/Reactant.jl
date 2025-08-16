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

        f_compile = Reactant.compile(
            fused_dense_bias_activation, (act, weight_ra, x_ra, bias_ra)
        )

        y_res = fused_dense_bias_activation(act, weight, x, bias)
        y_compile = f_compile(act, weight_ra, x_ra, bias_ra)

        @test y_res ≈ y_compile atol = 1e-5 rtol = 1e-2

        @testset "Enzyme: fused_dense_bias_activation" begin
            dw, dx, db = ∇fuseddense(act, weight, x, bias)
            ∇fuseddense_compiled = Reactant.compile(
                ∇fuseddense, (act, weight_ra, x_ra, bias_ra)
            )
            dw_compile, dx_compile, db_compile = ∇fuseddense_compiled(
                act, weight_ra, x_ra, bias_ra
            )

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

        f_compile = Reactant.compile(biasact, (act, x_ra, b_ra))
        f_compile!! = Reactant.compile(biasact!!, (act, x_ra, b_ra))

        y_simple = biasact(act, x, b)
        y_simple!! = biasact!!(act, x, b)
        y_compile = f_compile(act, x_ra, b_ra)
        y_compile!! = f_compile!!(act, x_ra, b_ra)

        @test y_simple ≈ y_compile atol = 1e-5 rtol = 1e-2
        @test y_simple!! ≈ y_compile!! atol = 1e-5 rtol = 1e-2

        @testset "Enzyme: bias_activation" begin
            ∂x_enz, ∂b_enz = ∇biasact(act, x, b)
            ∇biasact_compiled = Reactant.compile(∇biasact, (act, x_ra, b_ra))
            ∂x_compile, ∂b_compile = ∇biasact_compiled(act, x_ra, b_ra)

            @test ∂x_enz ≈ ∂x_compile atol = 1e-5 rtol = 1e-2
            @test ∂b_enz ≈ ∂b_compile atol = 1e-5 rtol = 1e-2
        end

        @testset "Enzyme: bias_activation!!" begin
            ∂x_enz!!, ∂b_enz!! = ∇biasact!!(act, x, b)
            ∇biasact!!_compiled = Reactant.compile(∇biasact!!, (act, x_ra, b_ra))
            ∂x_compile!!, ∂b_compile!! = ∇biasact!!_compiled(act, x_ra, b_ra)

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
        f_compile = Reactant.compile(sumabs2, (act, x_act_ca))
        f_compile!! = Reactant.compile(sumabs2!!, (act, x_act_ca))

        y_simple = sumabs2(act, x_act)
        y_simple!! = sumabs2!!(act, x_act)
        y_compile = f_compile(act, x_act_ca)
        y_compile!! = f_compile!!(act, x_act_ca)

        @test y_simple ≈ y_compile atol = 1e-5 rtol = 1e-2
        @test y_simple!! ≈ y_compile!! atol = 1e-5 rtol = 1e-2

        ∂x_enz = Enzyme.make_zero(x_act)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(act), Duplicated(x_act, ∂x_enz))

        ∂x_enz!! = Enzyme.make_zero(x_act)
        Enzyme.autodiff(Reverse, sumabs2!!, Active, Const(act), Duplicated(x_act, ∂x_enz!!))

        ∇sumabs2_compiled = Reactant.compile(∇sumabs2, (act, x_act_ca))
        ∂x_compile = ∇sumabs2_compiled(act, x_act_ca)

        ∇sumabs2!!_compiled = Reactant.compile(∇sumabs2!!, (act, x_act_ca))
        ∂x_compile!! = ∇sumabs2!!_compiled(act, x_act_ca)

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

            fused_conv_compiled = Reactant.compile(
                fused_conv_bias_activation,
                (act, weight_reactant, x_reactant, bias_reactant, conv_dims),
            )

            reactant_res = fused_conv_compiled(
                act, weight_reactant, x_reactant, bias_reactant, conv_dims
            )
            luxlib_res = fused_conv_bias_activation(act, weight, x, bias, conv_dims)

            @test reactant_res ≈ luxlib_res atol = 1e-5 rtol = 1e-2
        end

        # TODO: test for gradients
    end
end
