using NNlib, Reactant, Enzyme

@testset "Activation Functions" begin
    sumabs2(f, x) = sum(abs2, f.(x))

    function ∇sumabs2(f, x)
        dx = Enzyme.make_zero(x)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(f), Duplicated(x, dx))
        return dx
    end

    x_act = randn(Float32, 10, 10)
    x_act_ca = Reactant.ConcreteRArray(x_act)

    @testset "Activation: $act" for act in (
        identity, relu, sigmoid, tanh, tanh_fast, sigmoid_fast, gelu, abs2
    )
        f_compile = Reactant.compile(sumabs2, (act, x_act))

        y_simple = sumabs2(act, x_act)
        y_compile = f_compile(act, x_act_ca)

        ∂x_enz = Enzyme.make_zero(x_act)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(act), Duplicated(x_act, ∂x_enz))

        ∇sumabs2_compiled = Reactant.compile(∇sumabs2, (act, x_act_ca))

        ∂x_compile = ∇sumabs2_compiled(act, x_act_ca)

        @test y_simple ≈ y_compile
        # Mathematically the gelu definition here is slightly different from the one in NNlib
        @test ∂x_enz ≈ ∂x_compile broken=(act === gelu)
    end
end

@testset "Pooling" begin
    @testset for f in (NNlib.meanpool, NNlib.maxpool)
        x = randn(Float32, 32, 32, 3, 2)
        x_reactant = Reactant.ConcreteRArray(x)

        @testset for window in ((2, 2), (3, 3), (4, 4)),
            stride in ((1, 1), (2, 2)),
            padding in ((0, 0), (1, 1), (2, 2), (0, 2), (2, 0))

            pool_dims = PoolDims(x, window; stride, padding)

            f_reactant = Reactant.compile(f, (x_reactant, pool_dims))

            broken = any(==(2), padding) && f === NNlib.maxpool && window == (2, 2)
            @test f_reactant(x_reactant, pool_dims) ≈ f(x, pool_dims) broken = broken
        end

        # TODO: test for gradients
    end
end

@testset "Convolution" begin
    @testset for groups in (1, 2, 4)
        weight = randn(Float32, 4, 4, 8 ÷ groups, groups)
        x = randn(Float32, 16, 16, 8, 2)

        weight_reactant = Reactant.ConcreteRArray(weight)
        x_reactant = Reactant.ConcreteRArray(x)

        @testset for stride in ((1, 1), (2, 2), (3, 3)),
            padding in ((0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (0, 1), (1, 0)),
            dilation in ((1, 1), (2, 2), (1, 2), (2, 1))

            conv_dims = DenseConvDims(x, weight; stride, padding, dilation, groups)

            conv_compiled = Reactant.compile(
                NNlib.conv, (x_reactant, weight_reactant, conv_dims)
            )

            @test conv_compiled(x_reactant, weight_reactant, conv_dims) ≈
                NNlib.conv(x, weight, conv_dims)
        end

        # TODO: test for gradients
    end

    @testset "conv 1d: flip" begin
        x = [1.0f0; 2.0f0; 3.0f0;;;]
        W = [1.0f0; 2.0f0; 3.0f0;;;]
        xx = Reactant.ConcreteRArray(x)
        WW = Reactant.ConcreteRArray(W)
        conv_noflip(x, W) = NNlib.conv(x, W; pad=1, flipped=true)
        conv_flip(x, W) = NNlib.conv(x, W; pad=1, flipped=false)
        @test Reactant.compile(conv_noflip, (xx, WW))(xx, WW) ==
            [0*1+1*2+2*3; 1*1+2*2+3*3; 1*2+2*3+3*0;;;]
        @test Reactant.compile(conv_flip, (xx, WW))(xx, WW) ==
            [3*0+2*1+1*2; 3*1+2*2+1*3; 3*2+2*3+1*0;;;]
    end
end
