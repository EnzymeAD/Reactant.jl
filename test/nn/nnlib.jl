using NNlib, Reactant, Enzyme, Statistics, Test

@testset "Activation Functions" begin
    sumabs2(f, x) = sum(abs2, f.(x))

    function ∇sumabs2(gradfn, f, x)
        Ω = f.(x)
        return 2 .* gradfn.(Ω, x) .* Ω
    end

    x_act = Reactant.TestUtils.construct_test_array(Float32, 10, 10) .- 0.5f0
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
        y_compile = @jit sumabs2(act, x_act_ca)

        ∂x_compile = @jit(Enzyme.gradient(Reverse, sumabs2, Const(act), x_act_ca))[2]
        ∂x_compile_gt = @jit ∇sumabs2(gradfn, act, x_act_ca)

        @test y_simple ≈ y_compile
        @test ∂x_compile ≈ ∂x_compile_gt atol = 1e-3 rtol = 1e-3
    end
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

        # TODO: test for gradients
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
        src = Reactant.TestUtils.construct_test_array(Float32, n1, nsrc, nsrc)
        index = [(mod1(i, nsrc), mod1(j, nsrc)) for i in 1:nidx, j in 1:nidx]

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
        src = Reactant.TestUtils.construct_test_array(Float32, n1, nsrc, nsrc)
        index = [
            CartesianIndex((mod1(i, nsrc), mod1(j, nsrc))) for i in 1:nidx, j in 1:nidx
        ]

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

# Adapted from https://github.com/FluxML/NNlib.jl/blob/1468582c4db5f18149cc8fff6fb4633c5debe5c5/test/testsuite/scatter.jl#L108
@testset "NNlib scatter" begin
    function test_scatter(dsts, srcs, idxs, res; dims)
        @testset "scatter Float32 $op" for op in (+, -, max, min, *, /, mean)
            for idx in values(idxs), dim in dims
                dst = copy(dsts[dim])
                target_y = res[(op, dim, true)]
                src = srcs[(dim, true)]
                if op == /
                    src = src .* 2.0f0
                end

                y1 = @jit(
                    NNlib.scatter!(
                        op, Reactant.to_rarray(dst), Reactant.to_rarray(src), idx
                    )
                )
                @test y1 ≈ target_y
                @test y1 isa ConcreteRArray{Float32,ndims(dst)}
                @test size(y1) == size(dsts[dim])
                dst = copy(dsts[dim])
                y2 = @jit(
                    NNlib.scatter!(
                        op,
                        Reactant.to_rarray(dst),
                        Reactant.to_rarray(src),
                        Reactant.to_rarray(idx),
                    )
                )
                @test y2 ≈ target_y
                @test y2 isa ConcreteRArray{Float32,ndims(dst)}
                @test size(y2) == size(dsts[dim])

                target_y = res[(op, dim, false)]
                src = srcs[(dim, false)]
                if op == /
                    src = src .* 2.0f0
                end

                y3 = @jit(NNlib.scatter(op, Reactant.to_rarray(src), idx))
                @test y3 ≈ target_y
                @test y3 isa ConcreteRArray{Float32,ndims(dst)}
                @test size(y3) == size(dsts[dim])
                y4 = @jit(
                    NNlib.scatter(
                        op,
                        Reactant.to_rarray(src),
                        Reactant.to_rarray(idx);
                        dstsize=size(dsts[dim]),
                    )
                )
                @test y4 ≈ target_y
                @test y4 isa ConcreteRArray{Float32,ndims(dst)}
                @test size(y4) == size(dsts[dim])

                ridx = Reactant.to_rarray(idx)
                if ridx isa Reactant.AbstractConcreteArray
                    @test_throws ArgumentError @jit(
                        NNlib.scatter(op, Reactant.to_rarray(src), ridx)
                    )
                else
                    y5 = @jit(NNlib.scatter(op, Reactant.to_rarray(src), ridx))
                    @test y5 ≈ target_y
                    @test y5 isa ConcreteRArray{Float32,ndims(dst)}
                    @test size(y5) == size(dsts[dim])
                end
            end
        end
    end

    @testset "scatter 1d src, 1d index => 1d output" begin
        #! format: off
        dsts = Dict(
            0 => Float32[3, 4, 5, 6, 7]
        )

        srcs = Dict(
            (0, true) => ones(Float32, 5),
            (0, false) => collect(Float32, 1:5),
        )

        idxs = Dict(
            :int => [4, 2, 1, 5, 3],
            :tup => [(4,), (2,), (1,), (5,), (3,)],
            :car => CartesianIndex.([(4,), (2,), (1,), (5,), (3,)]),
        )

        res = Dict(
            (+, 0, true) => Float32[4, 5, 6, 7, 8],
            (+, 0, false) => Float32[3, 2, 5, 1, 4],

            (-, 0, true) => Float32[2, 3, 4, 5, 6],
            (-, 0, false) => Float32[-3, -2, -5, -1, -4],

            (max, 0, true) => Float32[3, 4, 5, 6, 7],
            (max, 0, false) => Float32[3, 2, 5, 1, 4],

            (min, 0, true) => Float32[1, 1, 1, 1, 1],
            (min, 0, false) => Float32[3, 2, 5, 1, 4],

            (*, 0, true) => Float32[3, 4, 5, 6, 7],
            (*, 0, false) => Float32[3, 2, 5, 1, 4],

            (/, 0, true) => Float32[1.5, 2.0, 2.5, 3.0, 3.5],
            (/, 0, false) => Float32[1//6, 1//4, 1//10, 1//2, 1//8],

            (mean, 0, true) => Float32[4, 5, 6, 7, 8],
            (mean, 0, false) => Float32[3, 2, 5, 1, 4],
        )
        #! format: on
        test_scatter(dsts, srcs, idxs, res; dims=[0])
    end

    @testset "scatter 2d src, 1d index => 2d output" begin
        #! format: off
        dsts = Dict(
                0 => Float32[3 3 4 4 5
                            5 5 6 6 7]
        )

        srcs = Dict(
            (0, true) => ones(Float32, 2, 5),
            (0, false) => ones(Float32, 2) * collect(1:5)',
        )

        idxs = Dict(
            :int => [4, 2, 1, 5, 3],
            :tup => [(4,), (2,), (1,), (5,), (3,)],
            :car => CartesianIndex.([(4,), (2,), (1,), (5,), (3,)]),
        )

        res = Dict(
            (+, 0, true) => Float32[4 4 5 5 6;
                                    6 6 7 7 8],
            (+, 0, false) => Float32[3 2 5 1 4;
                                        3 2 5 1 4],

            (-, 0, true) => Float32[2 2 3 3 4;
                                    4 4 5 5 6],
            (-, 0, false) => Float32[-3 -2 -5 -1 -4;
                                        -3 -2 -5 -1 -4],

            (max, 0, true) => Float32[3 3 4 4 5;
                                        5 5 6 6 7],
            (max, 0, false) => Float32[3 2 5 1 4;
                                        3 2 5 1 4],

            (min, 0, true) => Float32[1 1 1 1 1;
                                        1 1 1 1 1],
            (min, 0, false) => Float32[3 2 5 1 4;
                                        3 2 5 1 4],

            (*, 0, true) => Float32[3 3 4 4 5;
                                    5 5 6 6 7],
            (*, 0, false) => Float32[3 2 5 1 4;
                                        3 2 5 1 4],

            (/, 0, true) => Float32[1.5 1.5 2.0 2.0 2.5;
                                    2.5 2.5 3.0 3.0 3.5],
            (/, 0, false) => Float32[1//6 1//4 1//10 1//2 1//8;
                                        1//6 1//4 1//10 1//2 1//8],

            (mean, 0, true) => Float32[4 4 5 5 6;
                                        6 6 7 7 8],
            (mean, 0, false) => Float32[3 2 5 1 4;
                                        3 2 5 1 4],
        )
        #! format: on
        test_scatter(dsts, srcs, idxs, res; dims=[0])
    end

    @testset "scatter 2d+3d src, 2d index => 1d+2d output" begin
        #! format: off
        dsts = Dict(
            0 => Float32[3, 4, 5, 6, 7],
            1 => Float32[3 3 4 4 5;
                         5 5 6 6 7],
        )

        srcs = Dict(
            (0, true) => ones(Float32, 3, 4),
            (0, false) => ones(Float32, 3) * collect(1:4)',
            (1, true) => ones(Float32, 2, 3, 4),
            (1, false) => Float32[1, 2] .* reshape(ones(Float32, 3) * collect(1:4)', 1,3,4),
        )

        idxs = Dict(
            :int => [1 2 3 4;
                     4 2 1 3;
                     3 5 5 3],
            :tup => [(1,) (2,) (3,) (4,);
                     (4,) (2,) (1,) (3,);
                     (3,) (5,) (5,) (3,)],
            :car => CartesianIndex.(
                    [(1,) (2,) (3,) (4,);
                     (4,) (2,) (1,) (3,);
                     (3,) (5,) (5,) (3,)]),
        )

        res = Dict(
            (+, 0, true) => Float32[5, 6, 9, 8, 9],
            (+, 1, true) => Float32[5 5 8 6 7;
                                    7 7 10 8 9],
            (+, 0, false) => Float32[4, 4, 12, 5, 5],
            (+, 1, false) => Float32[4 4 12 5 5;
                                     8 8 24 10 10],
            (-, 0, true) => Float32[1, 2, 1, 4, 5],
            (-, 1, true) => Float32[1 1 0 2 3;
                                    3 3 2 4 5],
            (-, 0, false) => Float32[-4, -4, -12, -5, -5],
            (-, 1, false) => Float32[-4 -4 -12 -5 -5;
                                     -8 -8 -24 -10 -10],
            (max, 0, true) => Float32[3, 4, 5, 6, 7],
            (max, 1, true) => Float32[3 3 4 4 5;
                                      5 5 6 6 7],
            (max, 0, false) => Float32[3, 2, 4, 4, 3],
            (max, 1, false) => Float32[3 2 4 4 3;
                                       6 4 8 8 6],
            (min, 0, true) => Float32[1, 1, 1, 1, 1],
            (min, 1, true) => Float32[1 1 1 1 1;
                                      1 1 1 1 1],
            (min, 0, false) => Float32[1, 2, 1, 1, 2],
            (min, 1, false) => Float32[1 2 1 1 2;
                                       2 4 2 2 4],
            (*, 0, true) => Float32[3, 4, 5, 6, 7],
            (*, 1, true) => Float32[3 3 4 4 5;
                                    5 5 6 6 7],
            (*, 0, false) => Float32[3, 4, 48, 4, 6],
            (*, 1, false) => Float32[3 4 48 4 6;
                                     12 16 768 16 24],
            (/, 0, true) => Float32[0.75, 1., 0.3125, 1.5, 1.75],
            (/, 1, true) => Float32[0.75 0.75 0.25 1. 1.25;
                                    1.25 1.25 0.375 1.5 1.75],
            (/, 0, false) => Float32[1//12, 1//16, 1//768, 1//16, 1//24],
            (/, 1, false) => Float32[1//12 1//16 1//768 1//16 1//24;
                                     1//48 1//64 1//12288 1//64 1//96],
            (mean, 0, true) => Float32[4., 5., 6., 7., 8.],
            (mean, 1, true) => Float32[4. 4. 5. 5. 6.;
                                       6. 6. 7. 7. 8.],
            (mean, 0, false) => Float32[2, 2, 3, 2.5, 2.5],
            (mean, 1, false) => Float32[2. 2. 3. 2.5 2.5;
                                        4. 4. 6. 5. 5.],
        )
        #! format: on

        test_scatter(dsts, srcs, idxs, res; dims=[0, 1])
    end

    @testset "scatter gradient" begin
        dst = Float32[
            3 3 4 4 5
            5 5 6 6 7
        ]
        dst_ca = Reactant.to_rarray(dst)

        src = ones(Float32, 2, 5)
        src_ca = Reactant.to_rarray(src)

        idx = [4, 2, 1, 5, 3]
        idx_ca = Reactant.to_rarray(idx)

        test_scatter(dsts, srcs, idxs) = sum(NNlib.scatter!(+, dsts, srcs, idxs))

        grads_ca, loss_ca = @jit Enzyme.gradient(
            Enzyme.ReverseWithPrimal, Const(test_scatter), dst_ca, src_ca, idx_ca
        )
        loss = test_scatter(dst, src, idx)

        @test grads_ca[1] ≈ ones(Float32, size(dst)...)
        @test grads_ca[2] ≈ ones(Float32, size(src)...)
        @test grads_ca[3] === nothing
        @test loss ≈ loss_ca
    end
end

@testset "∇conv(D = $ndim)" for ndim in 1:3
    x_spatial_dim = 4
    batch_size = 2
    n_in_features = 3
    n_out_features = 4
    kernel_size = Tuple((2 for _ in 1:ndim))

    x = Reactant.TestUtils.construct_test_array(
        Float32, (x_spatial_dim for _ in 1:ndim)..., n_in_features, batch_size
    )
    x_reactant = Reactant.to_rarray(x)

    w = Reactant.TestUtils.construct_test_array(
        Float32, kernel_size..., n_in_features, n_out_features
    )
    w_reactant = Reactant.to_rarray(w)

    @testset "conv: padding=$padding stride=$stride dilation=$dilation groups=$groups" for (
        padding, stride, dilation, groups
    ) in Iterators.product(
        (0, 2), (1, 2), (1,), (1,)
    )
        Reactant.with_config(; convolution_precision=PrecisionConfig.HIGH) do
            conv_dims = NNlib.DenseConvDims(x, w; padding, stride, dilation, groups)

            output_size = (NNlib.output_size(conv_dims)..., n_out_features, batch_size)
            dy = Reactant.TestUtils.construct_test_array(Float32, output_size...)
            dy_reactant = Reactant.to_rarray(dy)

            @test @jit(NNlib.∇conv_data(dy_reactant, w_reactant, conv_dims)) ≈
                NNlib.∇conv_data(dy, w, conv_dims)
            @test @jit(NNlib.∇conv_filter(x_reactant, dy_reactant, conv_dims)) ≈
                NNlib.∇conv_filter(x, dy, conv_dims)
        end
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

@testset "gather 32bit indexing" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 10, 10)
    x_ra = Reactant.to_rarray(x)

    idxs = Int32.(mod1.(collect(1:32), 10))
    idxs_ra = Reactant.to_rarray(idxs)

    @test @jit(NNlib.gather(x_ra, idxs_ra)) ≈ NNlib.gather(x, idxs)
    hlo = repr(@code_hlo(NNlib.gather(x_ra, idxs_ra)))
    @test !contains(hlo, "i64>")
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
