for (jlop, hloop) in (
    (:(NNlib.tanh_fast), :tanh),
    (:(NNlib.sigmoid_fast), :logistic),
    (:(NNlib.sigmoid), :logistic),
    (:(NNlib.relu), :math_relu),
    (:(NNlib.softplus), :math_softplus),
)
    @eval $(jlop)(x::TracedRNumber) = @opcall $(hloop)(x)
end

# See https://github.com/EnzymeAD/Reactant.jl/issues/1420
# Without this we will never fuse the gelu into gemm
if isdefined(NNlib, :gelu_tanh)
    if isdefined(NNlib, :gelu_sigmoid)
        NNlib.gelu_sigmoid(x::TracedRNumber) = @opcall math_gelu(x, "SIGMOID")
        NNlib.gelu_tanh(x::TracedRNumber) = @opcall math_gelu(x, "TANH")
    else
        # In this version NNlib used sigmoid approximation in the tanh variant
        NNlib.gelu_tanh(x::TracedRNumber) = @opcall math_gelu(x, "SIGMOID")
    end

    NNlib.gelu_erf(x::TracedRNumber) = @opcall math_gelu(x, "NONE")
else
    # Older versions of NNlib do not have gelu_tanh (gelu refers to the tanh version)
    NNlib.gelu(x::TracedRNumber) = @opcall math_gelu(x, "TANH")
end

function NNlib.softmax!(out::AnyTracedRArray{T,N}, x::AbstractArray; dims=1) where {T,N}
    x = T.(materialize_traced_array(x))
    res = @opcall softmax(x; dims=vec(collect(Int64, dims)))
    copyto!(out, res)
    return out
end

function NNlib.logsoftmax!(out::AnyTracedRArray{T}, x::AbstractArray; dims=1) where {T}
    x = T.(materialize_traced_array(x))
    res = @opcall logsoftmax(x; dims=vec(collect(Int64, dims)))
    copyto!(out, res)
    return out
end

# Convolution
function overloaded_conv!(
    y::AnyTracedRArray{T,N},
    x::AnyTracedRArray{T2,N},
    W::AnyTracedRArray{T3,N},
    cdims::DenseConvDims;
) where {T,T2,T3,N}
    # StableHLO expects matching element types
    x = T.(materialize_traced_array(x))
    weight = T.(materialize_traced_array(W))

    if !NNlib.flipkernel(cdims)
        weight = @opcall reverse(weight; dimensions=collect(Int64, 1:(N - 2)))
    end

    result = @opcall convolution(
        collect(Int64, size(y)),
        x,
        weight;
        window_strides=collect(Int64, NNlib.stride(cdims)),
        input_batch_dim=N,
        input_feature_dim=N - 1,
        input_spatial_dims=collect(Int64, 1:(N - 2)),
        kernel_input_dim=N - 1,
        kernel_output_dim=N,
        kernel_spatial_dims=collect(Int64, 1:(N - 2)),
        output_batch_dim=N,
        output_feature_dim=N - 1,
        output_spatial_dims=collect(Int64, 1:(N - 2)),
        padding=reshape(collect(Int64, NNlib.padding(cdims)), 2, :),
        rhs_dilation=collect(Int64, NNlib.dilation(cdims)),
        feature_group_count=NNlib.groupcount(cdims),
        batch_group_count=1,
    )

    set_mlir_data!(y, result.mlir_data)
    return y
end

dilate_shape(s, d) = max(0, 1 + d * (s - 1))

# see lax._conv_general_dilated_transpose_rhs
# https://github.com/jax-ml/jax/blob/a1dfdc1d6164ad49afb337da9effd269d430d68b/jax/_src/lax/convolution.py#L495
function overloaded_∇conv_filter!(
    dw::AnyTracedRArray{T,N},
    x::AnyTracedRArray,
    dy::AnyTracedRArray,
    cdims::NNlib.DenseConvDims,
) where {T,N}
    # (w, h, cin, b)
    # (w, h, cout, b)
    # -> (w, h, cin, cout)

    x = T.(materialize_traced_array(x))
    dy = T.(materialize_traced_array(dy))

    num_spatial_dims = N - 2

    padding = reshape(collect(NNlib.padding(cdims)), (2, num_spatial_dims))

    lhs_shape = size(x)[1:num_spatial_dims]
    rhs_shape = dilate_shape.(size(dw)[1:num_spatial_dims], NNlib.dilation(cdims))
    out_shape = dilate_shape.(size(dy)[1:num_spatial_dims], NNlib.stride(cdims))

    padding = reduce(
        hcat,
        (
            let pad_before = padding[1, i],
                pad_after = (out_shape[i] - lhs_shape[i] + rhs_shape[i] - pad_before - 1)

                [pad_before, pad_after]
            end for i in 1:num_spatial_dims
        ),
    )
    padding = reshape(padding, 2, :)

    result = @opcall convolution(
        collect(Int64, size(dw)),
        x,
        dy;
        window_strides=collect(Int64, NNlib.dilation(cdims)),
        input_batch_dim=N - 1,
        input_feature_dim=N,
        input_spatial_dims=collect(Int64, 1:(N - 2)),
        kernel_input_dim=N,
        kernel_output_dim=N - 1,
        kernel_spatial_dims=collect(Int64, 1:(N - 2)),
        output_batch_dim=N - 1,
        output_feature_dim=N,
        output_spatial_dims=collect(Int64, 1:(N - 2)),
        padding,
        rhs_dilation=collect(Int64, NNlib.stride(cdims)),
        feature_group_count=1,
        batch_group_count=NNlib.groupcount(cdims),
    )
    set_mlir_data!(dw, result.mlir_data)

    if !NNlib.flipkernel(cdims)
        set_mlir_data!(
            dw, get_mlir_data(@opcall(reverse(dw; dimensions=collect(Int64, 1:(N - 2)))))
        )
    end

    return dw
end

# see lax._conv_general_dilated_transpose_lhs
# https://github.com/jax-ml/jax/blob/a1dfdc1d6164ad49afb337da9effd269d430d68b/jax/_src/lax/convolution.py#L457
function overloaded_∇conv_data!(
    dx::AnyTracedRArray{T,N},
    dy::AnyTracedRArray,
    w::AnyTracedRArray,
    cdims::NNlib.DenseConvDims,
) where {T,N}
    # (w, h, cout, b)
    # (w, h, cin, cout)
    # -> (w, h, cin, b)

    dy = T.(materialize_traced_array(dy))
    w = T.(materialize_traced_array(w))

    padding = reshape(collect(NNlib.padding(cdims)), 2, :)
    stride = NNlib.stride(cdims)
    dilation = NNlib.dilation(cdims)
    feature_group_count = NNlib.groupcount(cdims)
    kernel_spatial_dims = collect(Int64, 1:(N - 2))

    # jax does
    # (cout, cin, h, w) -> (group, cout ÷ group, cin , h, w) -> (cout ÷ group, group, cin, h, w) -> (cout, cin * group, h, w)
    # we perform the same operation but in transposed form
    # (w, h, cin, cout) -> (w, h, cin, cout ÷ group, group) -> (w, h, cin, group, cout ÷ group) -> (w, h, cin * group, cout ÷ group)
    if feature_group_count > 1
        w = reshape(
            w,
            (size(w, i) for i in kernel_spatial_dims)...,
            size(w, N - 1),
            size(w, N) ÷ feature_group_count,
            feature_group_count,
        )
        w = permutedims(w, (kernel_spatial_dims..., N - 1, N + 1, N))
        w = materialize_traced_array(
            reshape(
                w,
                (size(w, i) for i in kernel_spatial_dims)...,
                size(w, N - 1) * feature_group_count,
                size(w, N + 1),
            ),
        )
    end

    lhs_shape = size(dx)[1:(N - 2)]
    rhs_shape = dilate_shape.(size(w)[1:(N - 2)], dilation)
    out_shape = dilate_shape.(size(dy)[1:(N - 2)], stride)

    padding = reduce(
        hcat,
        (
            let pad_before = rhs_shape[i] - padding[2i - 1] - 1,
                pad_after = lhs_shape[i] + rhs_shape[i] - 1 - out_shape[i] - pad_before

                [pad_before, pad_after]
            end for i in 1:(N - 2)
        ),
    )
    padding = reshape(padding, 2, :)

    if NNlib.flipkernel(cdims)
        w = @opcall reverse(w; dimensions=kernel_spatial_dims)
    end

    result = @opcall convolution(
        collect(Int64, size(dx)),
        dy,
        w;
        input_batch_dim=N,
        input_feature_dim=N - 1,
        input_spatial_dims=collect(Int64, 1:(N - 2)),
        kernel_input_dim=N,
        kernel_output_dim=N - 1,
        kernel_spatial_dims,
        output_batch_dim=N,
        output_feature_dim=N - 1,
        output_spatial_dims=collect(Int64, 1:(N - 2)),
        padding,
        lhs_dilation=collect(NNlib.stride(cdims)),
        rhs_dilation=collect(NNlib.dilation(cdims)),
        feature_group_count,
        batch_group_count=1,
    )
    set_mlir_data!(dx, result.mlir_data)

    return dx
end

# Fold / Unfold
function NNlib.unfold!(
    y::AnyTracedRArray{yT,3}, x::AbstractArray{xT,5}, cdims::DenseConvDims
) where {yT,xT}
    unfold_impl!(y, x, cdims)
    return y
end

function NNlib.unfold!(
    y::AnyTracedRArray{yT,3}, x::AbstractArray{xT,N}, cdims::DenseConvDims
) where {yT,xT,N}
    unfold_impl!(y, x, cdims)
    return y
end

function unfold_impl!(
    y::AnyTracedRArray{yT,3}, x::AbstractArray{xT,N}, cdims::DenseConvDims
) where {yT,xT,N}
    @assert Reactant.unwrapped_eltype(yT) <: AbstractFloat "XLA doesn't support non-float \
                                                            unfold (got $(yT))."

    x = yT.(materialize_traced_array(x))

    C_in = NNlib.channels_in(cdims)
    K = NNlib.kernel_size(cdims)
    C_out = prod(K) * C_in

    weight = reshape(
        Reactant.promote_to(TracedRArray{yT,2}, LinearAlgebra.I(C_out)), (K..., C_in, C_out)
    )

    spatial_out = NNlib.output_size(cdims)
    conv_out_size = (spatial_out..., C_out, size(x, N))
    y_temp = similar(y, conv_out_size)
    overloaded_conv!(y_temp, x, weight, cdims)
    result = reshape(y_temp, prod(spatial_out), C_out, size(x, N))

    set_mlir_data!(y, materialize_traced_array(result).mlir_data)
    return y
end

function NNlib.fold!(
    x::AnyTracedRArray{T,5}, y::AnyTracedRArray{T2,3}, cdims::DenseConvDims;
) where {T,T2}
    fold_impl!(x, y, cdims)
    return x
end

function NNlib.fold!(
    x::AnyTracedRArray{T,N}, y::AnyTracedRArray{T2,3}, cdims::DenseConvDims;
) where {T,T2,N}
    fold_impl!(x, y, cdims)
    return x
end

function fold_impl!(
    x::AnyTracedRArray{T,N}, y::AnyTracedRArray{T2,3}, cdims::DenseConvDims;
) where {T,T2,N}
    @assert Reactant.unwrapped_eltype(T) <: AbstractFloat "XLA doesn't support non-float \
                                                            fold (got $(T))."
    y = T.(materialize_traced_array(y))

    C_in = NNlib.channels_in(cdims)
    K = NNlib.kernel_size(cdims)
    C_out = prod(K) * C_in

    weight = reshape(
        Reactant.promote_to(TracedRArray{T,2}, LinearAlgebra.I(C_out)), (K..., C_in, C_out)
    )

    spatial_out = NNlib.output_size(cdims)

    dy = materialize_traced_array(reshape(y, (spatial_out..., C_out, size(y, 3))))
    overloaded_∇conv_data!(x, dy, weight, cdims)
    return x
end

# Pooling

# With `ceil_mode` (NNlib ≥ 0.9.53), the last window may run past the padded input: extend
# the high padding by that overhang.
function pool_padding(pdims::NNlib.PoolDims)
    I, K = NNlib.input_size(pdims), NNlib.kernel_size(pdims)
    S, D = NNlib.stride(pdims), NNlib.dilation(pdims)
    padding = collect(NNlib.padding(pdims))
    for (i, o) in enumerate(NNlib.output_size(pdims))
        n = I[i] + padding[2i - 1] + padding[2i]
        padding[2i] += max(0, (o - 1) * S[i] + D[i] * (K[i] - 1) + 1 - n)
    end
    return Tuple(padding)
end

# Number of positions of each window inside the input, or inside the padded input with
# `count_include_pad` (so `ceil_mode` overhangs are never counted).
function meanpool_divisor(pdims::NNlib.PoolDims, count_include_pad::Bool)
    I, K = NNlib.input_size(pdims), NNlib.kernel_size(pdims)
    S, D = NNlib.stride(pdims), NNlib.dilation(pdims)
    P = NNlib.padding(pdims)
    counts = map(enumerate(NNlib.output_size(pdims))) do (i, o)
        lo, hi = count_include_pad ? (1 - P[2i - 1], I[i] + P[2i]) : (1, I[i])
        return map(1:o) do j
            start = (j - 1) * S[i] - P[2i - 1] + 1
            first_k = max(0, cld(lo - start, D[i]))
            last_k = min(K[i] - 1, fld(hi - start, D[i]))
            return max(0, last_k - first_k + 1)
        end
    end
    return prod.(Iterators.product(counts...))
end

function overloaded_maxpool!(
    y::AnyTracedRArray{T,N}, x::AnyTracedRArray{T2,N}, pdims::NNlib.PoolDims;
) where {T,T2,N}
    res = reduce_window(
        max,
        T.(x);
        init=typemin(T),
        dilation=NNlib.dilation(pdims),
        kernel_size=NNlib.kernel_size(pdims),
        padding=pool_padding(pdims),
        stride=NNlib.stride(pdims),
    )
    set_mlir_data!(y, get_mlir_data(res))
    return y
end

function overloaded_meanpool!(
    y::AnyTracedRArray{T,N},
    x::AnyTracedRArray{T2,N},
    pdims::NNlib.PoolDims;
    count_include_pad::Bool=true,
) where {T,T2,N}
    res = reduce_window(
        +,
        T.(x);
        init=zero(T),
        dilation=NNlib.dilation(pdims),
        kernel_size=NNlib.kernel_size(pdims),
        padding=pool_padding(pdims),
        stride=NNlib.stride(pdims),
    )
    # Windows with no input element sum to zero, and NNlib returns zero for them.
    divisor = max.(meanpool_divisor(pdims, count_include_pad), 1)
    divisor = if allequal(divisor)
        T(first(divisor))
    else
        Reactant.promote_to(TracedRArray{T,N}, reshape(T.(divisor), size(divisor)..., 1, 1))
    end
    set_mlir_data!(y, get_mlir_data(res ./ divisor))
    return y
end

# Batched Matrix Multiplication

NNlib.batched_transpose(x::AnyTracedRArray{T,3}) where {T} = PermutedDimsArray(x, (2, 1, 3))
function NNlib.batched_adjoint(x::AnyTracedRArray{T,3}) where {T}
    y = NNlib.batched_transpose(x)
    conj!(y)
    return y
end

function NNlib.batched_mul!(
    res::AnyTracedRArray{T1,3}, x::AnyTracedRArray{T2,3}, y::AnyTracedRArray{T3,3}
) where {T1,T2,T3}
    if (size(x, 3) != size(y, 3) && size(x, 3) != 1 && size(y, 3) != 1) ||
        (size(x, 2) != size(y, 1))
        throw(
            DimensionMismatch(
                lazy"size(x) = $(size(x)), size(y) = $(size(y)) inconsistent for batched_mul.",
            ),
        )
    end

    x = @opcall convert(TracedRArray{T2,3}, materialize_traced_array(x))
    y = @opcall convert(TracedRArray{T3,3}, materialize_traced_array(y))

    if size(x, 3) != size(y, 3)
        B = max(size(x, 3), size(y, 3))
        if size(x, 3) == 1
            x = @opcall broadcast_in_dim(x, [1, 2, 3], [size(x, 1), size(x, 2), B])
        elseif size(y, 3) == 1
            y = @opcall broadcast_in_dim(y, [1, 2, 3], [size(y, 1), size(y, 2), B])
        end
    end

    tmp = @opcall dot_general(
        x, y; contracting_dimensions=([2], [1]), batching_dimensions=([3], [3])
    )
    set_mlir_data!(res, get_mlir_data(permutedims(tmp, (2, 3, 1))))

    return res
end

# Padding
function NNlib.pad_constant(
    x::AnyTracedRArray{T,N}, pad::NTuple{N,Tuple{Int,Int}}, value
) where {T,N}
    return @opcall pad(
        materialize_traced_array(x),
        Reactant.promote_to(TracedRNumber{T}, value);
        low=[i[1] for i in pad],
        high=[i[2] for i in pad],
        interior=[0 for i in pad],
    )
end

# Gather
function NNlib.gather!(
    dst::AnyTracedRArray{T}, src::AnyTracedRArray, idxs::AbstractArray
) where {T}
    n_dims = NNlib.scatter_dims(src, dst, idxs)
    res = _nnlib_gather_impl(src, _stack_indices(idxs), n_dims)
    set_mlir_data!(dst, get_mlir_data(Reactant.promote_to(TracedRArray{T}, res)))
    return dst
end

function NNlib.gather!(
    dst::AnyTracedRArray{T}, src::AnyTracedRArray, idxs::AbstractArray{<:Number}
) where {T}
    n_dims = NNlib.scatter_dims(src, dst, idxs)
    res = _nnlib_gather_impl(src, reshape(idxs, 1, size(idxs)...), n_dims)
    set_mlir_data!(dst, get_mlir_data(Reactant.promote_to(TracedRArray{T}, res)))
    return dst
end

_stack_indices(idxs::AbstractArray) = stack(idxs)
function _stack_indices(idxs::AbstractArray{<:CartesianIndex})
    stacked_idxs = similar(idxs, Int, length(first(idxs)), size(idxs)...)
    for k in CartesianIndices(idxs)
        stacked_idxs[:, k.I...] .= idxs[k].I
    end
    return stacked_idxs
end

function _nnlib_gather_impl(src::AnyTracedRArray, idxs::AbstractArray, n_dims::Int)
    src = materialize_traced_array(src)
    return @opcall gather(
        src,
        Reactant.promote_to(TracedRArray, idxs);
        offset_dims=collect(Int64, 1:n_dims),
        collapsed_slice_dims=collect(Int64, (n_dims + 1):ndims(src)),
        operand_batching_dims=Int64[],
        start_indices_batching_dims=Int64[],
        start_index_map=collect(Int64, (ndims(src) - size(idxs, 1) + 1):ndims(src)),
        index_vector_dim=1,
        slice_sizes=Int64[size(src)[1:n_dims]..., ones(Int64, ndims(src) - n_dims)...],
    )
end

function NNlib.upsample_linear_kernel!(
    y::AnyTracedRArray{T,N}, x::AnyTracedRArray{T,N}; align_corners::Bool=true
) where {T,N}
    wT = real(Reactant.unwrapped_eltype(T))
    ratios = if align_corners
        ntuple(i -> wT((size(x, i) - 1) / (size(y, i) - 1)), N - 2)
    else
        ntuple(i -> wT(size(x, i) / size(y, i)), N - 2)
    end
    copyto!(y, upsample_linear(x, size(y)[1:(end - 2)], ratios..., align_corners))
    return y
end

# Scatter
function NNlib.scatter(
    op::OP, src::AnyTracedRArray{T}, idx::AbstractArray; init=nothing, dstsize=nothing
) where {OP,T}
    dims = ndims(src) - ndims(idx)
    dstsz = if isnothing(dstsize)
        (size(src)[1:dims]..., NNlib.maximum_dims(idx)...)
    else
        dstsize
    end
    if any(d -> d isa TracedRNumber, dstsz)
        throw(
            ArgumentError(
                "dstsize must be specified when idx is a TracedRArray or contains a TracedRNumber.",
            ),
        )
    end
    xinit = isnothing(init) ? NNlib.scatter_empty(op, T) : init
    dst = Reactant.promote_to(TracedRArray{T}, @opcall(fill(xinit, dstsz)))

    NNlib.scatter!(op, dst, src, idx)
    return dst
end

function NNlib.scatter!(
    op::OP, dst::AnyTracedRArray, src::AnyTracedRArray, idx::AbstractArray
) where {OP}
    dims = NNlib.scatter_dims(dst, src, idx)
    res = _nnlib_scatter_impl(op, dst, src, _stack_indices(idx), dims)
    set_mlir_data!(dst, get_mlir_data(res))
    return dst
end

function NNlib.scatter!(
    op::OP, dst::AnyTracedRArray, src::AnyTracedRArray, idx::AbstractArray{<:Number}
) where {OP}
    dims = NNlib.scatter_dims(dst, src, idx)
    res = _nnlib_scatter_impl(op, dst, src, reshape(idx, 1, size(idx)...), dims)
    set_mlir_data!(dst, get_mlir_data(res))
    return dst
end

for AT in (AbstractArray, AbstractArray{<:Number})
    @eval function NNlib.scatter!(
        ::typeof(mean), dst::AnyTracedRArray{T}, src::AnyTracedRArray, idx::$AT
    ) where {T}
        Ns = NNlib.scatter!(+, zero(dst), one.(src), idx)
        dst_ = NNlib.scatter!(+, zero(dst), src, idx)
        res = dst .+ NNlib.safe_div.(dst_, Ns)
        set_mlir_data!(dst, get_mlir_data(Reactant.promote_to(TracedRArray{T}, res)))
        return dst
    end
end

function _nnlib_scatter_impl(
    op::OP, dst::AnyTracedRArray{T}, src::AnyTracedRArray, idx::AbstractArray, n_dims::Int
) where {OP,T}
    return @opcall(
        scatter(
            op,
            [materialize_traced_array(dst)],
            Reactant.promote_to(TracedRArray, idx),
            [Reactant.promote_to(TracedRArray{T}, materialize_traced_array(src))];
            update_window_dims=collect(Int64, 1:n_dims),
            inserted_window_dims=collect(Int64, (n_dims + 1):ndims(dst)),
            input_batching_dims=Int64[],
            scatter_indices_batching_dims=Int64[],
            scatter_dims_to_operand_dims=collect(
                Int64, (ndims(dst) - size(idx, 1) + 1):ndims(dst)
            ),
            index_vector_dim=Int64(1),
        )
    )[1]
end

function NNlib.maximum_dims(dims::AnyTracedRArray{<:Integer})
    return (maximum(dims),)
end
function NNlib.maximum_dims(dims::AnyTracedRArray{NTuple{N,T}}) where {N,T}
    return ntuple(i -> maximum(x -> x[i], dims), N)
end
function NNlib.maximum_dims(dims::AnyTracedRArray{CartesianIndex{N}}) where {N}
    return ntuple(i -> maximum(x -> x[i], dims), N)
end

function _to_traced_array(::Type{ElT}, x::AbstractArray) where {ElT}
    if Reactant.ancestor(x) isa AnyTracedRArray
        return Reactant.promote_to(TracedRArray{ElT,ndims(x)}, materialize_traced_array(x))
    else
        return Reactant.promote_to(TracedRArray{ElT,ndims(x)}, Array{ElT}(x))
    end
end

# Scaled Dot-Product Attention
function _prepare_attention_bias_or_mask(
    ::Type{ElT}, x, target_sz::NTuple{4,Int}
) where {ElT}
    x === nothing && return nothing
    ndims(x) <= 4 && all(i -> size(x, i) in (1, target_sz[i]), (1, 2, 3, 4)) || throw(
        DimensionMismatch(
            "A bias or mask of size $(size(x)) cannot be broadcast to the attention score dimensions $target_sz.",
        ),
    )
    x_traced = _to_traced_array(ElT, x)
    return materialize_traced_array(
        reshape(x_traced, size(x, 1), size(x, 2), size(x, 3), size(x, 4))
    )
end

function _prepare_attention_lengths(lengths, B::Int, max_len::Int)
    lengths === nothing && return nothing
    max_len_i32 = Int32(max_len)
    zero_i32 = Int32(0)
    if lengths isa Number
        if lengths isa TracedRNumber
            c = clamp(
                Reactant.promote_to(TracedRNumber{Int32}, lengths), zero_i32, max_len_i32
            )
            return @opcall fill(c, (B,))
        else
            c = clamp(Int32(lengths), zero_i32, max_len_i32)
            return @opcall fill(c, (B,))
        end
    else
        length(lengths) == B || throw(
            ArgumentError(
                "Expected a sequence length for each of the $B sequences, got $(length(lengths)).",
            ),
        )
        l_traced = materialize_traced_array(reshape(_to_traced_array(Int32, lengths), B))
        return materialize_traced_array(clamp.(l_traced, zero_i32, max_len_i32))
    end
end

if isdefined(NNlib, :attention_lengths)
    function NNlib.attention_lengths(
        x::AnyTracedRArray{<:Any,4}, lengths::Union{Number,AbstractArray}, n::Int=size(x, 4)
    )
        l_vec = _prepare_attention_lengths(lengths, n, size(x, 3))
        return reshape(l_vec, (1, 1, 1, n))
    end
end

if isdefined(NNlib, :_is_fused_attention)
    function NNlib._is_fused_attention(
        ::AnyTracedRArray{<:Any,4},
        ::AnyTracedRArray{<:Any,4},
        ::AnyTracedRArray{<:Any,4},
        bias;
        fdrop=identity,
        mask=nothing,
        scale=nothing,
        is_causal=false,
        window=nothing,
        q_offset=0,
        q_lengths=nothing,
        kv_lengths=nothing,
        segments=nothing,
        kws...,
    )
        return fdrop === identity &&
               window === nothing &&
               is_causal isa Bool &&
               (q_offset isa Integer && iszero(q_offset)) &&
               segments === nothing &&
               isempty(kws)
    end
end

if isdefined(NNlib, :_scaled_dot_product_attention)
    function NNlib._scaled_dot_product_attention(
        q::AnyTracedRArray{Tq,4},
        k::AnyTracedRArray{Tk,4},
        v::AnyTracedRArray{Tv,4},
        bias=nothing;
        fdrop=identity,
        mask=nothing,
        scale=nothing,
        is_causal::Union{Bool,Symbol}=false,
        window=nothing,
        q_offset=0,
        q_lengths=nothing,
        kv_lengths=nothing,
        segments=nothing,
        kws...,
    ) where {Tq,Tk,Tv}
        if !(
            fdrop === identity &&
            window === nothing &&
            is_causal isa Bool &&
            (q_offset isa Integer && iszero(q_offset)) &&
            segments === nothing &&
            isempty(kws)
        )
            if isdefined(NNlib, :generic_attention)
                return NNlib.generic_attention(
                    q,
                    k,
                    v,
                    bias;
                    fdrop,
                    mask,
                    scale,
                    is_causal,
                    window,
                    q_offset,
                    q_lengths,
                    kv_lengths,
                    segments,
                    kws...,
                    unsegmented=NNlib.unsegmented_attention,
                )
            elseif hasmethod(
                NNlib._attention,
                Tuple{typeof(q),typeof(k),typeof(v),typeof(bias)},
                (:fdrop, :mask, :scale, :is_causal, :q_lengths, :kv_lengths),
            )
                x, _ = NNlib._attention(
                    q, k, v, bias; fdrop, mask, scale, is_causal, q_lengths, kv_lengths
                )
                return x
            else
                x, _ = NNlib._attention(q, k, v, bias; fdrop, mask, scale, is_causal)
                return x
            end
        end

        T = promote_type(
            Reactant.unwrapped_eltype(Tq),
            Reactant.unwrapped_eltype(Tk),
            Reactant.unwrapped_eltype(Tv),
        )
        q_traced = Reactant.promote_to(TracedRArray{T,4}, materialize_traced_array(q))
        k_traced = Reactant.promote_to(TracedRArray{T,4}, materialize_traced_array(k))
        v_traced = Reactant.promote_to(TracedRArray{T,4}, materialize_traced_array(v))

        if scale isa TracedRNumber
            s_traced = Reactant.promote_to(TracedRNumber{T}, scale)
            q_traced = materialize_traced_array(q_traced .* s_traced)
            scale_val = 1.0
        elseif scale === nothing
            scale_val = Float64(inv(sqrt(T(size(q, 1)))))
        else
            scale_val = Float64(T(scale))
        end

        score_sz = (size(k, 3), size(q, 3), size(q, 2), size(q, 4))
        bias_traced = _prepare_attention_bias_or_mask(T, bias, score_sz)
        mask_traced = _prepare_attention_bias_or_mask(Bool, mask, score_sz)
        q_len_traced = _prepare_attention_lengths(q_lengths, size(q, 4), size(q, 3))
        kv_len_traced = _prepare_attention_lengths(kv_lengths, size(k, 4), size(k, 3))

        return @opcall attention(
            q_traced,
            k_traced,
            v_traced,
            bias_traced;
            mask=mask_traced,
            q_lengths=q_len_traced,
            kv_lengths=kv_len_traced,
            scale=scale_val,
            is_causal,
        )
    end
end

# Grid Sample

@inline function _gs_validate_padding_mode(padding_mode::Symbol)
    padding_mode in (:zeros, :border) || throw(
        ArgumentError(
            lazy"Unsupported padding_mode: $(padding_mode). Expected :zeros or :border."
        ),
    )
    return nothing
end

@inline function _gs_unnormalize(coord::TracedRArray{V,N}, dim_size::Int) where {V,N}
    return ((coord .+ one(V)) .* V(0.5)) .* V(dim_size - 1) .+ one(V)
end

@inline function _gs_compute_source_index(
    coord::TracedRArray{V,N}, dim_size::Int, padding_mode::Symbol
) where {V,N}
    u = _gs_unnormalize(coord, dim_size)
    if padding_mode === :zeros
        return u
    else
        lo = @opcall fill(one(V), size(u))
        hi = @opcall fill(V(dim_size), size(u))
        return ifelse.(u .<= lo, lo, ifelse.(u .>= hi, hi, u))
    end
end

@inline function _gs_∇compute_source_index(
    coord::TracedRArray{V,N}, dim_size::Int, padding_mode::Symbol
) where {V,N}
    u = _gs_unnormalize(coord, dim_size)
    scale = V((dim_size - 1) * 0.5)
    if padding_mode === :zeros
        return u, @opcall(fill(scale, size(u)))
    else
        lo = @opcall fill(one(V), size(u))
        hi = @opcall fill(V(dim_size), size(u))
        clipped = ifelse.(u .<= lo, lo, ifelse.(u .>= hi, hi, u))
        oob = (u .<= lo) .| (u .>= hi)
        gmult = ifelse.(oob, zero(V), scale)
        return clipped, gmult
    end
end

function _gs_gather_2d_corner(
    input::TracedRArray{T,4},
    ix::TracedRArray{Int64,3},
    iy::TracedRArray{Int64,3},
    n_idx::TracedRArray{Int64,3},
) where {T}
    iW, iH, iC, iN = size(input)
    gW, gH = size(ix, 1), size(ix, 2)
    mask = (Int64(1) .<= ix .<= Int64(iW)) .& (Int64(1) .<= iy .<= Int64(iH))
    idxs = @opcall concatenate(
        [
            materialize_traced_array(reshape(ix, 1, gW, gH, iN)),
            materialize_traced_array(reshape(iy, 1, gW, gH, iN)),
            materialize_traced_array(reshape(n_idx, 1, gW, gH, iN)),
        ],
        1,
    )
    gathered = @opcall gather(
        input,
        idxs;
        offset_dims=Int64[3],
        collapsed_slice_dims=Int64[1, 2, 4],
        operand_batching_dims=Int64[],
        start_indices_batching_dims=Int64[],
        start_index_map=Int64[1, 2, 4],
        index_vector_dim=Int64(1),
        slice_sizes=Int64[1, 1, iC, 1],
    )
    return ifelse.(reshape(mask, gW, gH, 1, iN), gathered, zero(T)), mask, idxs
end

function _gs_gather_3d_corner(
    input::TracedRArray{T,5},
    ix::TracedRArray{Int64,4},
    iy::TracedRArray{Int64,4},
    iz::TracedRArray{Int64,4},
    n_idx::TracedRArray{Int64,4},
) where {T}
    iW, iH, iD, iC, iN = size(input)
    gW, gH, gD = size(ix, 1), size(ix, 2), size(ix, 3)
    mask =
        (Int64(1) .<= ix .<= Int64(iW)) .&
        (Int64(1) .<= iy .<= Int64(iH)) .&
        (Int64(1) .<= iz .<= Int64(iD))
    idxs = @opcall concatenate(
        [
            materialize_traced_array(reshape(ix, 1, gW, gH, gD, iN)),
            materialize_traced_array(reshape(iy, 1, gW, gH, gD, iN)),
            materialize_traced_array(reshape(iz, 1, gW, gH, gD, iN)),
            materialize_traced_array(reshape(n_idx, 1, gW, gH, gD, iN)),
        ],
        1,
    )
    gathered = @opcall gather(
        input,
        idxs;
        offset_dims=Int64[4],
        collapsed_slice_dims=Int64[1, 2, 3, 5],
        operand_batching_dims=Int64[],
        start_indices_batching_dims=Int64[],
        start_index_map=Int64[1, 2, 3, 5],
        index_vector_dim=Int64(1),
        slice_sizes=Int64[1, 1, 1, iC, 1],
    )
    return ifelse.(reshape(mask, gW, gH, gD, 1, iN), gathered, zero(T)), mask, idxs
end

function _gs_scatter_add_2d_corner(
    dx::TracedRArray{T,4},
    Δ::TracedRArray{T,4},
    weight::TracedRArray{T,3},
    mask::TracedRArray{Bool,3},
    idxs::TracedRArray{Int64,4},
) where {T}
    gW, gH, _, iN = size(Δ)
    upd = materialize_traced_array(
        ifelse.(reshape(mask, gW, gH, 1, iN), Δ .* reshape(weight, gW, gH, 1, iN), zero(T))
    )
    return @opcall(
        scatter(
            +,
            [dx],
            idxs,
            [upd];
            update_window_dims=Int64[3],
            inserted_window_dims=Int64[1, 2, 4],
            input_batching_dims=Int64[],
            scatter_indices_batching_dims=Int64[],
            scatter_dims_to_operand_dims=Int64[1, 2, 4],
            index_vector_dim=Int64(1),
        )
    )[1]
end

function _gs_scatter_add_3d_corner(
    dx::TracedRArray{T,5},
    Δ::TracedRArray{T,5},
    weight::TracedRArray{T,4},
    mask::TracedRArray{Bool,4},
    idxs::TracedRArray{Int64,5},
) where {T}
    gW, gH, gD, _, iN = size(Δ)
    upd = materialize_traced_array(
        ifelse.(
            reshape(mask, gW, gH, gD, 1, iN),
            Δ .* reshape(weight, gW, gH, gD, 1, iN),
            zero(T),
        ),
    )
    return @opcall(
        scatter(
            +,
            [dx],
            idxs,
            [upd];
            update_window_dims=Int64[4],
            inserted_window_dims=Int64[1, 2, 3, 5],
            input_batching_dims=Int64[],
            scatter_indices_batching_dims=Int64[],
            scatter_dims_to_operand_dims=Int64[1, 2, 3, 5],
            index_vector_dim=Int64(1),
        )
    )[1]
end

function _grid_sample_2d_impl(
    ::Type{OutT}, input::AbstractArray, grid::AbstractArray, padding_mode::Symbol
) where {OutT}
    _gs_validate_padding_mode(padding_mode)
    ndims(grid) == 4 || throw(
        DimensionMismatch(lazy"Expected 4D grid for 4D input, got $(ndims(grid))D grid."),
    )
    iW, iH, iC, iN = size(input)
    gC, gW, gH, gN = size(grid)
    gC == 2 ||
        throw(DimensionMismatch(lazy"Expected size(grid, 1) == 2 for 4D input, got $(gC)."))
    gN == iN || throw(
        DimensionMismatch(
            lazy"Batch dimension mismatch: size(input, 4) == $(iN) vs size(grid, 4) == $(gN).",
        ),
    )

    if iW == 0 || iH == 0 || iC == 0 || iN == 0 || gW == 0 || gH == 0
        return @opcall fill(zero(OutT), (gW, gH, iC, iN))
    end

    WT = promote_type(
        float(OutT),
        float(Reactant.unwrapped_eltype(input)),
        float(Reactant.unwrapped_eltype(grid)),
    )
    input_t = _to_traced_array(WT, input)
    grid_t = _to_traced_array(WT, grid)

    gx = materialize_traced_array(
        reshape(@opcall(slice(grid_t, [1, 1, 1, 1], [1, gW, gH, iN])), (gW, gH, iN))
    )
    gy = materialize_traced_array(
        reshape(@opcall(slice(grid_t, [2, 1, 1, 1], [2, gW, gH, iN])), (gW, gH, iN))
    )

    ix = _gs_compute_source_index(gx, iW, padding_mode)
    iy = _gs_compute_source_index(gy, iH, padding_mode)

    ix_0_f = @opcall floor(ix)
    iy_0_f = @opcall floor(iy)
    ix_1_f = ix_0_f .+ one(WT)
    iy_1_f = iy_0_f .+ one(WT)

    wx_1 = ix_1_f .- ix
    wx_0 = ix .- ix_0_f
    wy_1 = iy_1_f .- iy
    wy_0 = iy .- iy_0_f

    nw = reshape(wx_1 .* wy_1, gW, gH, 1, iN)
    ne = reshape(wx_0 .* wy_1, gW, gH, 1, iN)
    sw = reshape(wx_1 .* wy_0, gW, gH, 1, iN)
    se = reshape(wx_0 .* wy_0, gW, gH, 1, iN)

    ix_0 = @opcall convert(TracedRArray{Int64,3}, ix_0_f)
    iy_0 = @opcall convert(TracedRArray{Int64,3}, iy_0_f)
    ix_1 = ix_0 .+ Int64(1)
    iy_1 = iy_0 .+ Int64(1)

    n_idx = (@opcall iota(Int64, [gW, gH, iN]; iota_dimension=3)) .+ Int64(1)

    val_00, _, _ = _gs_gather_2d_corner(input_t, ix_0, iy_0, n_idx)
    val_10, _, _ = _gs_gather_2d_corner(input_t, ix_1, iy_0, n_idx)
    val_01, _, _ = _gs_gather_2d_corner(input_t, ix_0, iy_1, n_idx)
    val_11, _, _ = _gs_gather_2d_corner(input_t, ix_1, iy_1, n_idx)

    res = val_00 .* nw .+ val_10 .* ne .+ val_01 .* sw .+ val_11 .* se
    return Reactant.promote_to(TracedRArray{OutT,4}, materialize_traced_array(res))
end

function _grid_sample_3d_impl(
    ::Type{OutT}, input::AbstractArray, grid::AbstractArray, padding_mode::Symbol
) where {OutT}
    _gs_validate_padding_mode(padding_mode)
    ndims(grid) == 5 || throw(
        DimensionMismatch(lazy"Expected 5D grid for 5D input, got $(ndims(grid))D grid."),
    )
    iW, iH, iD, iC, iN = size(input)
    gC, gW, gH, gD, gN = size(grid)
    gC == 3 ||
        throw(DimensionMismatch(lazy"Expected size(grid, 1) == 3 for 5D input, got $(gC)."))
    gN == iN || throw(
        DimensionMismatch(
            lazy"Batch dimension mismatch: size(input, 5) == $(iN) vs size(grid, 5) == $(gN).",
        ),
    )

    if iW == 0 || iH == 0 || iD == 0 || iC == 0 || iN == 0 || gW == 0 || gH == 0 || gD == 0
        return @opcall fill(zero(OutT), (gW, gH, gD, iC, iN))
    end

    WT = promote_type(
        float(OutT),
        float(Reactant.unwrapped_eltype(input)),
        float(Reactant.unwrapped_eltype(grid)),
    )
    input_t = _to_traced_array(WT, input)
    grid_t = _to_traced_array(WT, grid)

    gx = materialize_traced_array(
        reshape(
            @opcall(slice(grid_t, [1, 1, 1, 1, 1], [1, gW, gH, gD, iN])), (gW, gH, gD, iN)
        ),
    )
    gy = materialize_traced_array(
        reshape(
            @opcall(slice(grid_t, [2, 1, 1, 1, 1], [2, gW, gH, gD, iN])), (gW, gH, gD, iN)
        ),
    )
    gz = materialize_traced_array(
        reshape(
            @opcall(slice(grid_t, [3, 1, 1, 1, 1], [3, gW, gH, gD, iN])), (gW, gH, gD, iN)
        ),
    )

    ix = _gs_compute_source_index(gx, iW, padding_mode)
    iy = _gs_compute_source_index(gy, iH, padding_mode)
    iz = _gs_compute_source_index(gz, iD, padding_mode)

    ix_0_f = @opcall floor(ix)
    iy_0_f = @opcall floor(iy)
    iz_0_f = @opcall floor(iz)
    ix_1_f = ix_0_f .+ one(WT)
    iy_1_f = iy_0_f .+ one(WT)
    iz_1_f = iz_0_f .+ one(WT)

    wx_1 = ix_1_f .- ix
    wx_0 = ix .- ix_0_f
    wy_1 = iy_1_f .- iy
    wy_0 = iy .- iy_0_f
    wz_1 = iz_1_f .- iz
    wz_0 = iz .- iz_0_f

    w_000 = reshape(wx_1 .* wy_1 .* wz_1, gW, gH, gD, 1, iN)
    w_100 = reshape(wx_0 .* wy_1 .* wz_1, gW, gH, gD, 1, iN)
    w_010 = reshape(wx_1 .* wy_0 .* wz_1, gW, gH, gD, 1, iN)
    w_110 = reshape(wx_0 .* wy_0 .* wz_1, gW, gH, gD, 1, iN)
    w_001 = reshape(wx_1 .* wy_1 .* wz_0, gW, gH, gD, 1, iN)
    w_101 = reshape(wx_0 .* wy_1 .* wz_0, gW, gH, gD, 1, iN)
    w_011 = reshape(wx_1 .* wy_0 .* wz_0, gW, gH, gD, 1, iN)
    w_111 = reshape(wx_0 .* wy_0 .* wz_0, gW, gH, gD, 1, iN)

    ix_0 = @opcall convert(TracedRArray{Int64,4}, ix_0_f)
    iy_0 = @opcall convert(TracedRArray{Int64,4}, iy_0_f)
    iz_0 = @opcall convert(TracedRArray{Int64,4}, iz_0_f)
    ix_1 = ix_0 .+ Int64(1)
    iy_1 = iy_0 .+ Int64(1)
    iz_1 = iz_0 .+ Int64(1)

    n_idx = (@opcall iota(Int64, [gW, gH, gD, iN]; iota_dimension=4)) .+ Int64(1)

    val_000, _, _ = _gs_gather_3d_corner(input_t, ix_0, iy_0, iz_0, n_idx)
    val_100, _, _ = _gs_gather_3d_corner(input_t, ix_1, iy_0, iz_0, n_idx)
    val_010, _, _ = _gs_gather_3d_corner(input_t, ix_0, iy_1, iz_0, n_idx)
    val_110, _, _ = _gs_gather_3d_corner(input_t, ix_1, iy_1, iz_0, n_idx)
    val_001, _, _ = _gs_gather_3d_corner(input_t, ix_0, iy_0, iz_1, n_idx)
    val_101, _, _ = _gs_gather_3d_corner(input_t, ix_1, iy_0, iz_1, n_idx)
    val_011, _, _ = _gs_gather_3d_corner(input_t, ix_0, iy_1, iz_1, n_idx)
    val_111, _, _ = _gs_gather_3d_corner(input_t, ix_1, iy_1, iz_1, n_idx)

    res =
        val_000 .* w_000 .+ val_100 .* w_100 .+ val_010 .* w_010 .+ val_110 .* w_110 .+
        val_001 .* w_001 .+ val_101 .* w_101 .+ val_011 .* w_011 .+ val_111 .* w_111
    return Reactant.promote_to(TracedRArray{OutT,5}, materialize_traced_array(res))
end

function overloaded_grid_sample(input::AbstractArray, grid; padding_mode=:zeros)
    N = ndims(input)
    if N == 4
        OutT = Reactant.unwrapped_eltype(input)
        return _grid_sample_2d_impl(OutT, input, grid, padding_mode)
    elseif N == 5
        OutT = Reactant.unwrapped_eltype(input)
        return _grid_sample_3d_impl(OutT, input, grid, padding_mode)
    else
        error("grid_sample is only supported for 4D and 5D arrays.")
    end
end

function overloaded_grid_sample!(
    output::AnyTracedRArray{T,4}, input::AbstractArray, grid, padding_mode=:zeros
) where {T}
    ndims(input) == 4 || throw(
        DimensionMismatch(
            lazy"Expected 4D input for 4D output, got $(ndims(input))D input."
        ),
    )
    OutT = Reactant.unwrapped_eltype(T)
    res = _grid_sample_2d_impl(OutT, input, grid, padding_mode)
    size(output) == size(res) || throw(
        DimensionMismatch(lazy"Expected output size $(size(res)), got $(size(output)).")
    )
    copyto!(output, res)
    return output
end

function overloaded_grid_sample!(
    output::AnyTracedRArray{T,5}, input::AbstractArray, grid, padding_mode=:zeros
) where {T}
    ndims(input) == 5 || throw(
        DimensionMismatch(
            lazy"Expected 5D input for 5D output, got $(ndims(input))D input."
        ),
    )
    OutT = Reactant.unwrapped_eltype(T)
    res = _grid_sample_3d_impl(OutT, input, grid, padding_mode)
    size(output) == size(res) || throw(
        DimensionMismatch(lazy"Expected output size $(size(res)), got $(size(output)).")
    )
    copyto!(output, res)
    return output
end

function _∇grid_sample_2d_impl(
    ::Type{DxT},
    ::Type{DgT},
    Δ::AbstractArray,
    input::AbstractArray,
    grid::AbstractArray,
    padding_mode::Symbol,
) where {DxT,DgT}
    _gs_validate_padding_mode(padding_mode)
    ndims(Δ) == 4 && ndims(input) == 4 && ndims(grid) == 4 ||
        throw(DimensionMismatch("Expected 4D Δ, input, and grid for 4D ∇grid_sample."))
    iW, iH, iC, iN = size(input)
    gC, gW, gH, gN = size(grid)
    gC == 2 ||
        throw(DimensionMismatch(lazy"Expected size(grid, 1) == 2 for 4D input, got $(gC)."))
    gN == iN || throw(
        DimensionMismatch(
            lazy"Batch dimension mismatch: size(input, 4) == $(iN) vs size(grid, 4) == $(gN).",
        ),
    )
    size(Δ) == (gW, gH, iC, iN) || throw(
        DimensionMismatch(lazy"Expected Δ of size $((gW, gH, iC, iN)), got $(size(Δ)).")
    )

    if iW == 0 || iH == 0 || iC == 0 || iN == 0 || gW == 0 || gH == 0
        return (
            @opcall(fill(zero(DxT), (iW, iH, iC, iN))),
            @opcall(fill(zero(DgT), (2, gW, gH, iN))),
        )
    end

    WT = promote_type(
        float(DxT),
        float(DgT),
        float(Reactant.unwrapped_eltype(Δ)),
        float(Reactant.unwrapped_eltype(input)),
        float(Reactant.unwrapped_eltype(grid)),
    )
    Δ_t = _to_traced_array(WT, Δ)
    input_t = _to_traced_array(WT, input)
    grid_t = _to_traced_array(WT, grid)

    gx = materialize_traced_array(
        reshape(@opcall(slice(grid_t, [1, 1, 1, 1], [1, gW, gH, iN])), (gW, gH, iN))
    )
    gy = materialize_traced_array(
        reshape(@opcall(slice(grid_t, [2, 1, 1, 1], [2, gW, gH, iN])), (gW, gH, iN))
    )

    ix, gix_mult = _gs_∇compute_source_index(gx, iW, padding_mode)
    iy, giy_mult = _gs_∇compute_source_index(gy, iH, padding_mode)

    ix_0_f = @opcall floor(ix)
    iy_0_f = @opcall floor(iy)
    ix_1_f = ix_0_f .+ one(WT)
    iy_1_f = iy_0_f .+ one(WT)

    wx_1 = ix_1_f .- ix
    wx_0 = ix .- ix_0_f
    wy_1 = iy_1_f .- iy
    wy_0 = iy .- iy_0_f

    w_00 = wx_1 .* wy_1
    w_10 = wx_0 .* wy_1
    w_01 = wx_1 .* wy_0
    w_11 = wx_0 .* wy_0

    ix_0 = @opcall convert(TracedRArray{Int64,3}, ix_0_f)
    iy_0 = @opcall convert(TracedRArray{Int64,3}, iy_0_f)
    ix_1 = ix_0 .+ Int64(1)
    iy_1 = iy_0 .+ Int64(1)

    n_idx = (@opcall iota(Int64, [gW, gH, iN]; iota_dimension=3)) .+ Int64(1)

    val_00, mask_00, idxs_00 = _gs_gather_2d_corner(input_t, ix_0, iy_0, n_idx)
    val_10, mask_10, idxs_10 = _gs_gather_2d_corner(input_t, ix_1, iy_0, n_idx)
    val_01, mask_01, idxs_01 = _gs_gather_2d_corner(input_t, ix_0, iy_1, n_idx)
    val_11, mask_11, idxs_11 = _gs_gather_2d_corner(input_t, ix_1, iy_1, n_idx)

    dx = @opcall fill(zero(WT), (iW, iH, iC, iN))
    dx = _gs_scatter_add_2d_corner(dx, Δ_t, w_00, mask_00, idxs_00)
    dx = _gs_scatter_add_2d_corner(dx, Δ_t, w_10, mask_10, idxs_10)
    dx = _gs_scatter_add_2d_corner(dx, Δ_t, w_01, mask_01, idxs_01)
    dx = _gs_scatter_add_2d_corner(dx, Δ_t, w_11, mask_11, idxs_11)

    init_zero = Reactant.promote_to(TracedRNumber{WT}, zero(WT))
    c_00 = @opcall reduce(val_00 .* Δ_t, init_zero, Int[3], +)
    c_10 = @opcall reduce(val_10 .* Δ_t, init_zero, Int[3], +)
    c_01 = @opcall reduce(val_01 .* Δ_t, init_zero, Int[3], +)
    c_11 = @opcall reduce(val_11 .* Δ_t, init_zero, Int[3], +)

    gix = gix_mult .* ((c_10 .- c_00) .* wy_1 .+ (c_11 .- c_01) .* wy_0)
    giy = giy_mult .* ((c_01 .- c_00) .* wx_1 .+ (c_11 .- c_10) .* wx_0)

    dgrid = @opcall concatenate(
        [
            materialize_traced_array(reshape(gix, 1, gW, gH, iN)),
            materialize_traced_array(reshape(giy, 1, gW, gH, iN)),
        ],
        1,
    )
    return (
        Reactant.promote_to(TracedRArray{DxT,4}, materialize_traced_array(dx)),
        Reactant.promote_to(TracedRArray{DgT,4}, materialize_traced_array(dgrid)),
    )
end

function _∇grid_sample_3d_impl(
    ::Type{DxT},
    ::Type{DgT},
    Δ::AbstractArray,
    input::AbstractArray,
    grid::AbstractArray,
    padding_mode::Symbol,
) where {DxT,DgT}
    _gs_validate_padding_mode(padding_mode)
    ndims(Δ) == 5 && ndims(input) == 5 && ndims(grid) == 5 ||
        throw(DimensionMismatch("Expected 5D Δ, input, and grid for 5D ∇grid_sample."))
    iW, iH, iD, iC, iN = size(input)
    gC, gW, gH, gD, gN = size(grid)
    gC == 3 ||
        throw(DimensionMismatch(lazy"Expected size(grid, 1) == 3 for 5D input, got $(gC)."))
    gN == iN || throw(
        DimensionMismatch(
            lazy"Batch dimension mismatch: size(input, 5) == $(iN) vs size(grid, 5) == $(gN).",
        ),
    )
    size(Δ) == (gW, gH, gD, iC, iN) || throw(
        DimensionMismatch(
            lazy"Expected Δ of size $((gW, gH, gD, iC, iN)), got $(size(Δ))."
        ),
    )

    if iW == 0 || iH == 0 || iD == 0 || iC == 0 || iN == 0 || gW == 0 || gH == 0 || gD == 0
        return (
            @opcall(fill(zero(DxT), (iW, iH, iD, iC, iN))),
            @opcall(fill(zero(DgT), (3, gW, gH, gD, iN))),
        )
    end

    WT = promote_type(
        float(DxT),
        float(DgT),
        float(Reactant.unwrapped_eltype(Δ)),
        float(Reactant.unwrapped_eltype(input)),
        float(Reactant.unwrapped_eltype(grid)),
    )
    Δ_t = _to_traced_array(WT, Δ)
    input_t = _to_traced_array(WT, input)
    grid_t = _to_traced_array(WT, grid)

    gx = materialize_traced_array(
        reshape(
            @opcall(slice(grid_t, [1, 1, 1, 1, 1], [1, gW, gH, gD, iN])), (gW, gH, gD, iN)
        ),
    )
    gy = materialize_traced_array(
        reshape(
            @opcall(slice(grid_t, [2, 1, 1, 1, 1], [2, gW, gH, gD, iN])), (gW, gH, gD, iN)
        ),
    )
    gz = materialize_traced_array(
        reshape(
            @opcall(slice(grid_t, [3, 1, 1, 1, 1], [3, gW, gH, gD, iN])), (gW, gH, gD, iN)
        ),
    )

    ix, gix_mult = _gs_∇compute_source_index(gx, iW, padding_mode)
    iy, giy_mult = _gs_∇compute_source_index(gy, iH, padding_mode)
    iz, giz_mult = _gs_∇compute_source_index(gz, iD, padding_mode)

    ix_0_f = @opcall floor(ix)
    iy_0_f = @opcall floor(iy)
    iz_0_f = @opcall floor(iz)
    ix_1_f = ix_0_f .+ one(WT)
    iy_1_f = iy_0_f .+ one(WT)
    iz_1_f = iz_0_f .+ one(WT)

    wx_1 = ix_1_f .- ix
    wx_0 = ix .- ix_0_f
    wy_1 = iy_1_f .- iy
    wy_0 = iy .- iy_0_f
    wz_1 = iz_1_f .- iz
    wz_0 = iz .- iz_0_f

    w_000 = wx_1 .* wy_1 .* wz_1
    w_100 = wx_0 .* wy_1 .* wz_1
    w_010 = wx_1 .* wy_0 .* wz_1
    w_110 = wx_0 .* wy_0 .* wz_1
    w_001 = wx_1 .* wy_1 .* wz_0
    w_101 = wx_0 .* wy_1 .* wz_0
    w_011 = wx_1 .* wy_0 .* wz_0
    w_111 = wx_0 .* wy_0 .* wz_0

    ix_0 = @opcall convert(TracedRArray{Int64,4}, ix_0_f)
    iy_0 = @opcall convert(TracedRArray{Int64,4}, iy_0_f)
    iz_0 = @opcall convert(TracedRArray{Int64,4}, iz_0_f)
    ix_1 = ix_0 .+ Int64(1)
    iy_1 = iy_0 .+ Int64(1)
    iz_1 = iz_0 .+ Int64(1)

    n_idx = (@opcall iota(Int64, [gW, gH, gD, iN]; iota_dimension=4)) .+ Int64(1)

    val_000, mask_000, idxs_000 = _gs_gather_3d_corner(input_t, ix_0, iy_0, iz_0, n_idx)
    val_100, mask_100, idxs_100 = _gs_gather_3d_corner(input_t, ix_1, iy_0, iz_0, n_idx)
    val_010, mask_010, idxs_010 = _gs_gather_3d_corner(input_t, ix_0, iy_1, iz_0, n_idx)
    val_110, mask_110, idxs_110 = _gs_gather_3d_corner(input_t, ix_1, iy_1, iz_0, n_idx)
    val_001, mask_001, idxs_001 = _gs_gather_3d_corner(input_t, ix_0, iy_0, iz_1, n_idx)
    val_101, mask_101, idxs_101 = _gs_gather_3d_corner(input_t, ix_1, iy_0, iz_1, n_idx)
    val_011, mask_011, idxs_011 = _gs_gather_3d_corner(input_t, ix_0, iy_1, iz_1, n_idx)
    val_111, mask_111, idxs_111 = _gs_gather_3d_corner(input_t, ix_1, iy_1, iz_1, n_idx)

    dx = @opcall fill(zero(WT), (iW, iH, iD, iC, iN))
    dx = _gs_scatter_add_3d_corner(dx, Δ_t, w_000, mask_000, idxs_000)
    dx = _gs_scatter_add_3d_corner(dx, Δ_t, w_100, mask_100, idxs_100)
    dx = _gs_scatter_add_3d_corner(dx, Δ_t, w_010, mask_010, idxs_010)
    dx = _gs_scatter_add_3d_corner(dx, Δ_t, w_110, mask_110, idxs_110)
    dx = _gs_scatter_add_3d_corner(dx, Δ_t, w_001, mask_001, idxs_001)
    dx = _gs_scatter_add_3d_corner(dx, Δ_t, w_101, mask_101, idxs_101)
    dx = _gs_scatter_add_3d_corner(dx, Δ_t, w_011, mask_011, idxs_011)
    dx = _gs_scatter_add_3d_corner(dx, Δ_t, w_111, mask_111, idxs_111)

    init_zero = Reactant.promote_to(TracedRNumber{WT}, zero(WT))
    c_000 = @opcall reduce(val_000 .* Δ_t, init_zero, Int[4], +)
    c_100 = @opcall reduce(val_100 .* Δ_t, init_zero, Int[4], +)
    c_010 = @opcall reduce(val_010 .* Δ_t, init_zero, Int[4], +)
    c_110 = @opcall reduce(val_110 .* Δ_t, init_zero, Int[4], +)
    c_001 = @opcall reduce(val_001 .* Δ_t, init_zero, Int[4], +)
    c_101 = @opcall reduce(val_101 .* Δ_t, init_zero, Int[4], +)
    c_011 = @opcall reduce(val_011 .* Δ_t, init_zero, Int[4], +)
    c_111 = @opcall reduce(val_111 .* Δ_t, init_zero, Int[4], +)

    gix =
        gix_mult .* (
            ((c_100 .- c_000) .* wy_1 .+ (c_110 .- c_010) .* wy_0) .* wz_1 .+
            ((c_101 .- c_001) .* wy_1 .+ (c_111 .- c_011) .* wy_0) .* wz_0
        )
    giy =
        giy_mult .* (
            ((c_010 .- c_000) .* wx_1 .+ (c_110 .- c_100) .* wx_0) .* wz_1 .+
            ((c_011 .- c_001) .* wx_1 .+ (c_111 .- c_101) .* wx_0) .* wz_0
        )
    giz =
        giz_mult .* (
            (c_001 .- c_000) .* (wx_1 .* wy_1) .+ (c_011 .- c_010) .* (wx_1 .* wy_0) .+
            (c_101 .- c_100) .* (wx_0 .* wy_1) .+ (c_111 .- c_110) .* (wx_0 .* wy_0)
        )

    dgrid = @opcall concatenate(
        [
            materialize_traced_array(reshape(gix, 1, gW, gH, gD, iN)),
            materialize_traced_array(reshape(giy, 1, gW, gH, gD, iN)),
            materialize_traced_array(reshape(giz, 1, gW, gH, gD, iN)),
        ],
        1,
    )
    return (
        Reactant.promote_to(TracedRArray{DxT,5}, materialize_traced_array(dx)),
        Reactant.promote_to(TracedRArray{DgT,5}, materialize_traced_array(dgrid)),
    )
end

function overloaded_∇grid_sample(
    Δ::AbstractArray, input::AbstractArray, grid; padding_mode=:zeros
)
    N = ndims(input)
    DxT = Reactant.unwrapped_eltype(input)
    DgT = Reactant.unwrapped_eltype(grid)
    if N == 4
        return _∇grid_sample_2d_impl(DxT, DgT, Δ, input, grid, padding_mode)
    elseif N == 5
        return _∇grid_sample_3d_impl(DxT, DgT, Δ, input, grid, padding_mode)
    else
        error("∇grid_sample is only supported for 4D and 5D arrays.")
    end
end

function overloaded_∇grid_sample!(
    dx::AnyTracedRArray{T1,4},
    dgrid::AnyTracedRArray{T2,4},
    Δ::AbstractArray,
    input::AbstractArray,
    grid::AbstractArray,
    padding_mode,
) where {T1,T2}
    DxT = Reactant.unwrapped_eltype(T1)
    DgT = Reactant.unwrapped_eltype(T2)
    res_dx, res_dgrid = _∇grid_sample_2d_impl(DxT, DgT, Δ, input, grid, padding_mode)
    size(dx) == size(res_dx) || throw(
        DimensionMismatch(lazy"Expected dx of size $(size(res_dx)), got $(size(dx)).")
    )
    size(dgrid) == size(res_dgrid) || throw(
        DimensionMismatch(
            lazy"Expected dgrid of size $(size(res_dgrid)), got $(size(dgrid))."
        ),
    )
    # NNlib.∇grid_sample! accumulates into dx via _safe_add! and overwrites dgrid.
    copyto!(dx, dx .+ res_dx)
    copyto!(dgrid, res_dgrid)
    return dx, dgrid
end

function overloaded_∇grid_sample!(
    dx::AnyTracedRArray{T1,5},
    dgrid::AnyTracedRArray{T2,5},
    Δ::AbstractArray,
    input::AbstractArray,
    grid::AbstractArray,
    padding_mode,
) where {T1,T2}
    DxT = Reactant.unwrapped_eltype(T1)
    DgT = Reactant.unwrapped_eltype(T2)
    res_dx, res_dgrid = _∇grid_sample_3d_impl(DxT, DgT, Δ, input, grid, padding_mode)
    size(dx) == size(res_dx) || throw(
        DimensionMismatch(lazy"Expected dx of size $(size(res_dx)), got $(size(dx)).")
    )
    size(dgrid) == size(res_dgrid) || throw(
        DimensionMismatch(
            lazy"Expected dgrid of size $(size(res_dgrid)), got $(size(dgrid))."
        ),
    )
    copyto!(dx, dx .+ res_dx)
    copyto!(dgrid, res_dgrid)
    return dx, dgrid
end
