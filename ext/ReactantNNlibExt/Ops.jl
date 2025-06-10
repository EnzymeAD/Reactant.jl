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

function upsample_linear(
    x::AnyTracedRArray{T,3}, out_size::Tuple{Int}, rwidth, align_corners::Bool
) where {T}
    W, C, B = size(x)

    out_idxs = Ops.iota(Int32, [out_size[1]]; iota_dimension=1)
    iw0, iw1, w0_λ, w1_λ = source_idx_and_λ(rwidth, out_idxs, align_corners, W)

    x0 = x[iw0, :, :]
    x1 = x[iw1, :, :]

    return w0_λ .* x0 .+ w1_λ .* x1
end

@inline function source_idx_and_λ(
    ratio::T, out_idx::AbstractVector, align::Bool, in_width::Int
) where {T}
    real_index = ifelse(
        align, ratio .* out_idx, max.(zero(T), ratio .* (out_idx .+ T(0.5)) .- T(0.5))
    )

    iw0 = Base.Fix1(floor, Int).(real_index)
    offset = ifelse.(iw0 .< in_width - 1, 1, 0)
    iw1 = iw0 .+ offset .+ 1

    w1lambda = real_index .- iw0
    w0lambda = one(T) .- w1lambda
    return iw0 .+ 1, iw1, w0lambda, w1lambda
end
