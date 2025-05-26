function reduce_window(
    f, x::AnyTracedRArray{T,N}; init, dilation, kernel_size, padding, stride
) where {T,N}
    output_spatial_shapes = map(1:(N - 2)) do i
        K = kernel_size[i]
        pl, pr = padding[2i - 1], padding[2i]
        d = dilation[i]
        s = stride[i]

        return (size(x, i) + pl + pr - d * (K - 1) - 1) ÷ s + 1
    end

    padding = collect(Int64, reshape([padding..., 0, 0, 0, 0], (2, N))')

    return Ops.reduce_window(
        f,
        [materialize_traced_array(x)],
        [Ops.constant(T(init))];
        window_dimensions=[kernel_size..., 1, 1],
        window_strides=[stride..., 1, 1],
        window_dilations=[dilation..., 1, 1],
        padding_low=padding[:, 1],
        padding_high=padding[:, 2],
        output_shape=Int[output_spatial_shapes..., size(x, N - 1), size(x, N)],
        base_dilations=ones(Int, N),
    )[1]
end
