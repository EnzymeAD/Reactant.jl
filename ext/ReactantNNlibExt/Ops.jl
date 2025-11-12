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

    return @opcall(
        reduce_window(
            f,
            [materialize_traced_array(x)],
            [@opcall(constant(T(init)))];
            window_dimensions=[kernel_size..., 1, 1],
            window_strides=[stride..., 1, 1],
            window_dilations=[dilation..., 1, 1],
            padding_low=padding[:, 1],
            padding_high=padding[:, 2],
            output_shape=Int[output_spatial_shapes..., size(x, N - 1), size(x, N)],
            base_dilations=ones(Int, N),
        )
    )[1]
end

function upsample_linear(
    x::AnyTracedRArray{T,3}, out_size::Tuple{Int}, rwidth, align_corners::Bool
) where {T}
    W, _, _ = size(x)

    out_idxs = @opcall iota(Int32, [out_size[1]]; iota_dimension=1)
    iw0, iw1, w0_λ, w1_λ = source_idx_and_λ(rwidth, out_idxs, align_corners, W)

    x0 = x[iw0, :, :]
    x1 = x[iw1, :, :]

    return w0_λ .* x0 .+ w1_λ .* x1
end

function upsample_linear(
    x::AnyTracedRArray{T,4}, out_size::Tuple{Int,Int}, rwidth, rheight, align_corners::Bool
) where {T}
    W, H, _, _ = size(x)

    out_width = @opcall iota(Int32, [out_size[1]]; iota_dimension=1)
    out_height = @opcall iota(Int32, [out_size[2]]; iota_dimension=1)

    iw0, iw1, w0_λ, w1_λ = source_idx_and_λ(rwidth, out_width, align_corners, W)
    ih0, ih1, h0_λ, h1_λ = source_idx_and_λ(rheight, out_height, align_corners, H)

    w0_λ, w1_λ = reshape(w0_λ, (:, 1, 1, 1)), reshape(w1_λ, (:, 1, 1, 1))
    h0_λ, h1_λ = reshape(h0_λ, (1, :, 1, 1)), reshape(h1_λ, (1, :, 1, 1))

    x00 = x[iw0, ih0, :, :]
    x10 = x[iw1, ih0, :, :]
    x01 = x[iw0, ih1, :, :]
    x11 = x[iw1, ih1, :, :]

    return h0_λ .* (w0_λ .* x00 .+ w1_λ .* x10) .+ h1_λ .* (w0_λ .* x01 .+ w1_λ .* x11)
end

function upsample_linear(
    x::AnyTracedRArray{T,5},
    out_size::Tuple{Int,Int,Int},
    rwidth,
    rheight,
    rdepth,
    align_corners::Bool,
) where {T}
    W, H, D, _, _ = size(x)

    out_width = @opcall iota(Int32, [out_size[1]]; iota_dimension=1)
    out_height = @opcall iota(Int32, [out_size[2]]; iota_dimension=1)
    out_depth = @opcall iota(Int32, [out_size[3]]; iota_dimension=1)

    iw0, iw1, w0_λ, w1_λ = source_idx_and_λ(rwidth, out_width, align_corners, W)
    ih0, ih1, h0_λ, h1_λ = source_idx_and_λ(rheight, out_height, align_corners, H)
    id0, id1, d0_λ, d1_λ = source_idx_and_λ(rdepth, out_depth, align_corners, D)

    w0_λ = reshape(w0_λ, (:, 1, 1, 1))
    w1_λ = reshape(w1_λ, (:, 1, 1, 1))
    h0_λ = reshape(h0_λ, (1, :, 1, 1))
    h1_λ = reshape(h1_λ, (1, :, 1, 1))
    d0_λ = reshape(d0_λ, (1, 1, :, 1))
    d1_λ = reshape(d1_λ, (1, 1, :, 1))

    x000 = x[iw0, ih0, id0, :, :]
    x100 = x[iw1, ih0, id0, :, :]
    x010 = x[iw0, ih1, id0, :, :]
    x110 = x[iw1, ih1, id0, :, :]

    x001 = x[iw0, ih0, id1, :, :]
    x101 = x[iw1, ih0, id1, :, :]
    x011 = x[iw0, ih1, id1, :, :]
    x111 = x[iw1, ih1, id1, :, :]

    return (
        (
            d0_λ .* (
                h0_λ .* (w0_λ .* x000 .+ w1_λ .* x100) .+
                h1_λ .* (w0_λ .* x010 .+ w1_λ .* x110)
            )
        ) .+ (
            d1_λ .* (
                h0_λ .* (w0_λ .* x001 .+ w1_λ .* x101) .+
                h1_λ .* (w0_λ .* x011 .+ w1_λ .* x111)
            )
        )
    )
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
