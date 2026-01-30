using NNlib, Reactant, Enzyme, Statistics, Test

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

@testset "gather 32bit indexing" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 10, 10)
    x_ra = Reactant.to_rarray(x)

    idxs = Int32.(mod1.(collect(1:32), 10))
    idxs_ra = Reactant.to_rarray(idxs)

    @test @jit(NNlib.gather(x_ra, idxs_ra)) ≈ NNlib.gather(x, idxs)
    hlo = repr(@code_hlo(NNlib.gather(x_ra, idxs_ra)))
    @test !contains(hlo, "i64>")
end
