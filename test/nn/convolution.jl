using LuxLib, Reactant, Enzyme, NNlib, Test, Statistics

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

        # TODO(#2253): test for gradients
    end
end

@testset "unfold/fold" begin
    @testset "unfold wrapper" begin
        x = Reactant.to_rarray(
            Reactant.TestUtils.construct_test_array(Float32, 16, 16, 3, 10)
        )
        w = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 5, 5, 3, 2))
        @test size(@jit(NNlib.unfold(x, size(w)))) == (144, 75, 10)
        @test size(@jit(NNlib.unfold(x, size(w); pad=2))) == (256, 75, 10)
        @test size(@jit(NNlib.unfold(x, size(w); stride=2))) == (36, 75, 10)
        @test size(@jit(NNlib.unfold(x, size(w); dilation=2))) == (64, 75, 10)
    end

    @testset "spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
        x = Reactant.TestUtils.construct_test_array(
            Float32, repeat([8], spatial_rank)..., 3, 2
        )
        x_ra = Reactant.to_rarray(x)
        w = Reactant.TestUtils.construct_test_array(
            Float32, repeat([3], spatial_rank)..., 3, 3
        )
        w_ra = Reactant.to_rarray(w)

        cdims = DenseConvDims(x, w; padding=1)
        y = NNlib.unfold(x, cdims)
        z = NNlib.fold(y, size(x), cdims)

        y_ra = @jit NNlib.unfold(x_ra, cdims)
        z_ra = @jit NNlib.fold(y_ra, size(x_ra), cdims)

        @test y ≈ y_ra atol = 1e-5 rtol = 1e-2
        @test z ≈ z_ra atol = 1e-5 rtol = 1e-2

        # introduce stride
        cdims = DenseConvDims(x, w; padding=1, stride=2)
        y = NNlib.unfold(x, cdims)
        z = NNlib.fold(y, size(x), cdims)

        y_ra = @jit NNlib.unfold(x_ra, cdims)
        z_ra = @jit NNlib.fold(y_ra, size(x_ra), cdims)

        @test y ≈ y_ra atol = 1e-5 rtol = 1e-2
        @test z ≈ z_ra atol = 1e-5 rtol = 1e-2
    end
end

@testset "Upsampling" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 4, 4, 3, 2)
    x_ra = Reactant.to_rarray(x)

    @testset "Nearest" begin
        @test @jit(NNlib.upsample_nearest(x_ra, (2, 2))) ≈ NNlib.upsample_nearest(x, (2, 2))
    end

    @testset "Linear" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 4, 3, 2)
        x_ra = Reactant.to_rarray(x)

        @test @jit(NNlib.upsample_linear(x_ra, (2,))) ≈ NNlib.upsample_linear(x, (2,))

        @test @jit(NNlib.upsample_linear(x_ra, (2,); align_corners=false)) ≈
            NNlib.upsample_linear(x, (2,); align_corners=false)
    end

    @testset "Bi-Linear" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 4, 4, 3, 2)
        x_ra = Reactant.to_rarray(x)

        @test @jit(NNlib.upsample_bilinear(x_ra, (2, 2))) ≈
            NNlib.upsample_bilinear(x, (2, 2))

        @test @jit(NNlib.upsample_bilinear(x_ra, (2, 2); align_corners=false)) ≈
            NNlib.upsample_bilinear(x, (2, 2); align_corners=false)
    end

    @testset "Tri-Linear" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 4, 4, 4, 3, 2)
        x_ra = Reactant.to_rarray(x)

        @test @jit(NNlib.upsample_trilinear(x_ra, (2, 2, 2))) ≈
            NNlib.upsample_trilinear(x, (2, 2, 2))

        @test @jit(NNlib.upsample_trilinear(x_ra, (2, 2, 2); align_corners=false)) ≈
            NNlib.upsample_trilinear(x, (2, 2, 2); align_corners=false)
    end
end

@testset "Pixel shuffle" begin
    x = [10i + j + channel / 10 for i in 1:2, j in 1:3, channel in 1:4, batch in 1:1]
    x_ra = Reactant.to_rarray(x)

    @test @jit(NNlib.pixel_shuffle(x_ra, 2)) ≈ NNlib.pixel_shuffle(x, 2)

    y = [i + channel / 10 for i in 1:3, channel in 1:6, batch in 1:1]
    y_ra = Reactant.to_rarray(y)

    @test @jit(NNlib.pixel_shuffle(y_ra, 2)) ≈ NNlib.pixel_shuffle(y, 2)
end

@testset "Pooling" begin
    @testset for f in (NNlib.meanpool, NNlib.maxpool)
        x = Reactant.TestUtils.construct_test_array(Float32, 32, 32, 3, 2)
        x_reactant = Reactant.to_rarray(x)

        @testset for window in ((2, 2), (3, 3), (4, 4)),
            stride in ((1, 1), (2, 2)),
            padding in ((0, 0), (1, 1), (2, 2), (0, 2), (2, 0))

            pool_dims = PoolDims(x, window; stride, padding)

            f_reactant = Reactant.compile(f, (x_reactant, pool_dims))

            broken = any(==(2), padding) && f === NNlib.maxpool && window == (2, 2)
            @test f_reactant(x_reactant, pool_dims) ≈ f(x, pool_dims) broken = broken
        end

        # TODO(#2253): test for gradients
    end
end

function ∇conv_data_filter(x, weight, conv_dims)
    dx, dweight = Enzyme.make_zero(x), Enzyme.make_zero(weight)
    Enzyme.autodiff(
        Reverse,
        NNlib.conv,
        Duplicated(x, dx),
        Duplicated(weight, dweight),
        Const(conv_dims),
    )
    return dx, dweight
end

@testset "Convolution" begin
    @testset for groups in (1, 2, 4)
        weight = Reactant.TestUtils.construct_test_array(Float32, 4, 4, 8 ÷ groups, 4)
        x = Reactant.TestUtils.construct_test_array(Float32, 16, 16, 8, 2)

        weight_reactant = Reactant.to_rarray(weight)
        x_reactant = Reactant.to_rarray(x)

        @testset for stride in ((1, 1), (2, 2), (3, 3)),
            padding in ((0, 0), (2, 2), (0, 2), (2, 0)),
            dilation in ((1, 1), (1, 2))

            conv_dims = DenseConvDims(x, weight; stride, padding, dilation, groups)

            output_size = (
                NNlib.output_size(conv_dims)...,
                size(weight, ndims(weight)),
                size(x, ndims(x)),
            )
            dy = ones(Float32, output_size)
            dy_reactant = Reactant.to_rarray(dy)

            Reactant.with_config(; convolution_precision=PrecisionConfig.HIGH) do
                @test @jit(NNlib.conv(x_reactant, weight_reactant, conv_dims)) ≈
                    NNlib.conv(x, weight, conv_dims)

                ∇data = NNlib.∇conv_data(dy, weight, conv_dims)
                @test @jit(NNlib.∇conv_data(dy_reactant, weight_reactant, conv_dims)) ≈
                    ∇data

                ∇filter = NNlib.∇conv_filter(x, dy, conv_dims)
                @test @jit(NNlib.∇conv_filter(x_reactant, dy_reactant, conv_dims)) ≈ ∇filter

                ∇data_enzyme, ∇filter_enzyme = @jit ∇conv_data_filter(
                    x_reactant, weight_reactant, conv_dims
                )
                @test ∇data_enzyme ≈ ∇data
                @test ∇filter_enzyme ≈ ∇filter
            end
        end
    end

    @testset "conv 1d: flip" begin
        x = [1.0f0; 2.0f0; 3.0f0;;;]
        W = [1.0f0; 2.0f0; 3.0f0;;;]
        xx = Reactant.to_rarray(x)
        WW = Reactant.to_rarray(W)
        conv_noflip(x, W) = NNlib.conv(x, W; pad=1, flipped=true)
        conv_flip(x, W) = NNlib.conv(x, W; pad=1, flipped=false)
        @test Reactant.compile(conv_noflip, (xx, WW))(xx, WW) ==
            [0*1+1*2+2*3; 1*1+2*2+3*3; 1*2+2*3+3*0;;;]
        @test Reactant.compile(conv_flip, (xx, WW))(xx, WW) ==
            [3*0+2*1+1*2; 3*1+2*2+1*3; 3*2+2*3+1*0;;;]
    end
end
