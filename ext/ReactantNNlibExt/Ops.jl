function reduce_window(
    f, x::AnyTracedRArray{T,N}; init, dilation, kernel_size, padding, stride
) where {T,N}
    x = materialize_traced_array(x)

    num_spatial_dims = N - 2
    input_spatial_dims = 1:num_spatial_dims

    window_dimensions = [kernel_size..., 1, 1]
    window_strides = [stride..., 1, 1]
    window_dilations = [dilation..., 1, 1]

    output_spatial_shapes = map(input_spatial_dims) do i
        K = kernel_size[i]
        pl, pr = padding[2i - 1], padding[2i]
        d = dilation[i]
        s = stride[i]

        (size(x, i) + pl + pr - d * (K - 1) - 1) ÷ s + 1
    end

    padding = Reactant.MLIR.IR.DenseElementsAttribute(
        reshape([padding..., 0, 0, 0, 0], (2, N))'
    )

    output_shape = Int[output_spatial_shapes..., size(x, N - 1), size(x, N)]
    result_type = Reactant.MLIR.IR.TensorType(output_shape, Reactant.MLIR.IR.Type(T))

    unranked = Reactant.MLIR.IR.TensorType(
        Int[], eltype(Reactant.MLIR.IR.type(get_mlir_data(x)))
    )
    body =
        let body = Reactant.MLIR.IR.Region(),
            loc = Reactant.MLIR.IR.Location(),
            block = Reactant.MLIR.IR.Block([unranked, unranked], [loc, loc])

            Reactant.MLIR.IR.block!(block) do
                red = f(
                    Reactant.MLIR.IR.argument(block, 1),
                    Reactant.MLIR.IR.argument(block, 2);
                    result=nothing,
                )
                Reactant.MLIR.Dialects.stablehlo.return_([Reactant.MLIR.IR.result(red)])
            end
            push!(body, block)

            body
        end

    attr = fill(Reactant.MLIR.IR.Attribute(init), unranked)
    init_value = Reactant.MLIR.IR.result(
        Reactant.MLIR.Dialects.stablehlo.constant(; value=attr)
    )
    reduction = Reactant.MLIR.Dialects.stablehlo.reduce_window(
        [get_mlir_data(x)],
        [init_value];
        result_0=[result_type],
        window_dimensions,
        window_strides,
        window_dilations,
        padding,
        body,
    )

    return TracedRArray{T,N}((), Reactant.MLIR.IR.result(reduction), size(result_type))
end
