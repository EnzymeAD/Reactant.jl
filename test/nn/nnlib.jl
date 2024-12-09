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
        identity, relu, sigmoid, tanh, tanh_fast, sigmoid_fast, gelu, abs2, relu6
    )
        f_compile = Reactant.compile(sumabs2, (act, x_act))

        y_simple = sumabs2(act, x_act)
        y_compile = f_compile(act, x_act_ca)

        ∂x_enz = Enzyme.make_zero(x_act)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(act), Duplicated(x_act, ∂x_enz))

        ∇sumabs2_compiled = Reactant.compile(∇sumabs2, (act, x_act_ca))

        ∂x_compile = ∇sumabs2_compiled(act, x_act_ca)

        @test y_simple ≈ y_compile
        @test ∂x_enz ≈ ∂x_compile
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
        weight = randn(Float32, 4, 4, 8 ÷ groups, 4)
        x = randn(Float32, 16, 16, 8, 2)

        weight_reactant = Reactant.ConcreteRArray(weight)
        x_reactant = Reactant.ConcreteRArray(x)

        @testset for stride in ((1, 1), (2, 2), (3, 3)),
            padding in ((0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (0, 1), (1, 0)),
            dilation in ((1, 1), (2, 2), (1, 2), (2, 1))

            conv_dims = DenseConvDims(x, weight; stride, padding, dilation, groups)

            output_size = (
                NNlib.output_size(conv_dims)...,
                size(weight, ndims(weight)),
                size(x, ndims(x)),
            )
            dy = randn(Float32, output_size)
            dy_reactant = Reactant.to_rarray(dy)

            conv_compiled = Reactant.compile(
                NNlib.conv, (x_reactant, weight_reactant, conv_dims)
            )

            @test conv_compiled(x_reactant, weight_reactant, conv_dims) ≈
                NNlib.conv(x, weight, conv_dims)

            @test Reactant.@jit(NNlib.∇conv_data(dy_reactant, weight_reactant, conv_dims)) ≈
                NNlib.∇conv_data(dy, weight, conv_dims)
            @test Reactant.@jit(NNlib.∇conv_filter(x_reactant, dy_reactant, conv_dims)) ≈
                NNlib.∇conv_filter(x, dy, conv_dims)
        end
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

@testset "Batched Matrix Multiplication" begin
    x = rand(Float32, 4, 3, 5)
    y = rand(Float32, 3, 2, 5)

    x_ra = Reactant.ConcreteRArray(x)
    y_ra = Reactant.ConcreteRArray(y)

    @test @jit(batched_mul(x_ra, y_ra)) ≈ batched_mul(x, y)

    x = rand(Float32, 4, 3, 1)
    y = rand(Float32, 3, 2, 5)

    x_ra = Reactant.ConcreteRArray(x)
    y_ra = Reactant.ConcreteRArray(y)

    @test @jit(batched_mul(x_ra, y_ra)) ≈ batched_mul(x, y)

    x = rand(Float32, 4, 3, 5)
    y = rand(Float32, 3, 2, 1)

    x_ra = Reactant.ConcreteRArray(x)
    y_ra = Reactant.ConcreteRArray(y)

    @test @jit(batched_mul(x_ra, y_ra)) ≈ batched_mul(x, y)
end

@testset "Constant Padding: NNlib.pad_constant" begin
    x = rand(Float32, 4, 4)
    x_ra = Reactant.ConcreteRArray(x)

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

    function ∇sumabs2(f, x)
        dx = Enzyme.make_zero(x)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(f), Duplicated(x, dx))
        return dx
    end

    pad_fn = Base.Fix2(NNlib.pad_constant, (1, 1, 1, 1))
    @test @jit(∇sumabs2(pad_fn, x_ra)) ≈ ∇sumabs2(pad_fn, x)

    pad_fn2 = Base.Fix2(NNlib.pad_constant, (1, 0, 1, 3))
    @test @jit(∇sumabs2(pad_fn2, x_ra)) ≈ ∇sumabs2(pad_fn2, x)

    x = rand(ComplexF32, 4, 4)
    x_ra = Reactant.ConcreteRArray(x)

    @test @jit(NNlib.pad_constant(x_ra, (1, 1))) ≈ NNlib.pad_constant(x, (1, 1))
end

@testset "make_causal_mask" begin
    x = rand(2, 10)
    x_ra = Reactant.ConcreteRArray(x)

    @test @jit(NNlib.make_causal_mask(x_ra)) ≈ NNlib.make_causal_mask(x)

    causal_mask2(x) = NNlib.make_causal_mask(x; dims=1)
    @test @jit(causal_mask2(x_ra)) ≈ causal_mask2(x)
end

# Adapted from https://github.com/FluxML/NNlib.jl/blob/02138682a4fc5ca019759218be50e59907d4527c/test/testsuite/gather.jl#L5
@testset "NNlib gather" begin
    @testset "gather scalar index" begin
        ## 1d src, 2d index of ints -> 2d output
        src = Float32[3, 4, 5, 6, 7]
        index = [
            1 2 3 4
            4 2 1 3
            3 5 5 3
        ]
        output = Float32[
            3 4 5 6
            6 4 3 5
            5 7 7 5
        ]

        y1 = @jit(NNlib.gather(Reactant.to_rarray(src), Reactant.to_rarray(index)))
        @test y1 ≈ output
        @test y1 isa ConcreteRArray{Float32,2}
        @test size(y1) == size(index)

        y2 = @jit(NNlib.gather(Reactant.to_rarray(src), index))
        @test y2 ≈ output
        @test y2 isa ConcreteRArray{Float32,2}
        @test size(y2) == size(index)

        dst = Float32.(zero.(index))
        @test @jit(
            NNlib.gather!(
                Reactant.to_rarray(dst), Reactant.to_rarray(src), Reactant.to_rarray(index)
            )
        ) ≈ output

        dst = zeros(Float32, 3, 5)
        @test_throws ArgumentError @jit(
            NNlib.gather!(
                Reactant.to_rarray(dst), Reactant.to_rarray(src), Reactant.to_rarray(index)
            )
        )

        ## 1d src, 3d index of ints -> 3d output
        src = Float32[3, 4, 5, 6, 7]
        index = [
            1 2 3 4
            4 2 1 3
            3 5 5 3
        ][:, :, 1:1]
        output = Float32[
            3 4 5 6
            6 4 3 5
            5 7 7 5
        ][:, :, 1:1]
        y = @jit(NNlib.gather(Reactant.to_rarray(src), Reactant.to_rarray(index)))
        @test y ≈ output
        @test y isa ConcreteRArray{Float32,3}
        @test size(y) == size(index)

        y = @jit(NNlib.gather(Reactant.to_rarray(src), index))
        @test y ≈ output
        @test y isa ConcreteRArray{Float32,3}
        @test size(y) == size(index)

        ## 2d src, 2d index of ints -> 3d output
        src = Float32[
            3 5 7
            4 6 8
        ]
        index = [
            1 2 3
            2 2 1
            3 1 3
        ]

        output = zeros(Float32, 2, 3, 3)
        output[:, :, 1] = [
            3 5 7
            4 6 8
        ]
        output[:, :, 2] = [
            5 5 3
            6 6 4
        ]
        output[:, :, 3] = [
            7 3 7
            8 4 8
        ]

        y = @jit(NNlib.gather(Reactant.to_rarray(src), Reactant.to_rarray(index)))
        @test y ≈ output
        @test y isa ConcreteRArray{Float32,3}
        @test size(y) == (size(src)[1:(end - 1)]..., size(index)...)

        y = @jit(NNlib.gather(Reactant.to_rarray(src), index))
        @test y ≈ output
        @test y isa ConcreteRArray{Float32,3}
        @test size(y) == (size(src)[1:(end - 1)]..., size(index)...)
    end

    @testset "gather tuple index" begin
        ## 2d src, 1d index of 2-tuples -> 1d output
        src = Float32[
            3 5 7
            4 6 8
        ]
        index = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
        output = Float32[3, 5, 7, 4, 6, 8]

        y = @jit(NNlib.gather(Reactant.to_rarray(src), Reactant.to_rarray(index)))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa ConcreteRArray{Float32,1}
        @test size(y) == (size(src)[1:(Nsrc - M)]..., size(index)...)
        @test y ≈ output

        y = @jit(NNlib.gather(Reactant.to_rarray(src), index))
        @test y ≈ output
        @test y isa ConcreteRArray{Float32,1}
        @test size(y) == (size(src)[1:(Nsrc - M)]..., size(index)...)
        @test y ≈ output

        ## 3d src, 2d index of 2-tuples -> 3d output
        n1, nsrc, nidx = 2, 3, 6
        src = rand(Float32, n1, nsrc, nsrc)
        index = [(rand(1:nsrc), rand(1:nsrc)) for i in 1:nidx, j in 1:nidx]

        y = @jit(NNlib.gather(Reactant.to_rarray(src), Reactant.to_rarray(index)))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa ConcreteRArray{Float32,3}
        @test size(y) == (size(src)[1:(Nsrc - M)]..., size(index)...)

        y = @jit(NNlib.gather(Reactant.to_rarray(src), index))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa ConcreteRArray{Float32,3}
        @test size(y) == (size(src)[1:(Nsrc - M)]..., size(index)...)
    end

    @testset "gather cartesian index" begin
        ## 2d src, 1d index of 2-tuples -> 1d output
        src = Float32[
            3 5 7
            4 6 8
        ]
        index = CartesianIndex.([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)])
        output = Float32[3, 5, 7, 4, 6, 8]

        y = @jit(NNlib.gather(Reactant.to_rarray(src), Reactant.to_rarray(index)))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa ConcreteRArray{Float32,1}
        @test size(y) == (size(src)[1:(Nsrc - M)]..., size(index)...)
        @test y ≈ output

        y = @jit(NNlib.gather(Reactant.to_rarray(src), index))
        @test y ≈ output
        @test y isa ConcreteRArray{Float32,1}
        @test size(y) == (size(src)[1:(Nsrc - M)]..., size(index)...)

        ## 3d src, 2d index of 2-tuples -> 3d output
        n1, nsrc, nidx = 2, 3, 6
        src = rand(Float32, n1, nsrc, nsrc)
        index = [CartesianIndex((rand(1:nsrc), rand(1:nsrc))) for i in 1:nidx, j in 1:nidx]

        y = @jit(NNlib.gather(Reactant.to_rarray(src), Reactant.to_rarray(index)))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa ConcreteRArray{Float32,3}
        @test size(y) == (size(src)[1:(Nsrc - M)]..., size(index)...)

        y = @jit(NNlib.gather(Reactant.to_rarray(src), index))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa ConcreteRArray{Float32,3}
        @test size(y) == (size(src)[1:(Nsrc - M)]..., size(index)...)
    end
end

@testset "∇conv(D = $ndim)" for ndim in 1:3
    x_spatial_dim = 4
    batch_size = 2
    n_in_features = 3
    n_out_features = 4
    kernel_size = Tuple((2 for _ in 1:ndim))

    x = randn(Float32, (x_spatial_dim for _ in 1:ndim)..., n_in_features, batch_size)
    x_reactant = Reactant.to_rarray(x)

    w = randn(Float32, kernel_size..., n_in_features, n_out_features)
    w_reactant = Reactant.to_rarray(w)

    @testset "conv: padding=$padding stride=$stride dilation=$dilation groups=$groups" for (
        padding, stride, dilation, groups
    ) in Iterators.product(
        (0, 2), (1, 2), (1,), (1,)
    )
        conv_dims = NNlib.DenseConvDims(x, w; padding, stride, dilation, groups)

        output_size = (NNlib.output_size(conv_dims)..., n_out_features, batch_size)
        dy = randn(Float32, output_size)
        dy_reactant = Reactant.to_rarray(dy)

        @test Reactant.@jit(NNlib.∇conv_data(dy_reactant, w_reactant, conv_dims)) ≈
            NNlib.∇conv_data(dy, w, conv_dims)
        @test Reactant.@jit(NNlib.∇conv_filter(x_reactant, dy_reactant, conv_dims)) ≈
            NNlib.∇conv_filter(x, dy, conv_dims)
    end
end
