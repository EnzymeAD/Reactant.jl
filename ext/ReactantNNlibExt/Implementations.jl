for (jlop, hloop) in (
    (:(NNlib.tanh_fast), :tanh),
    (:(NNlib.sigmoid_fast), :logistic),
    (:(NNlib.sigmoid), :logistic),
)
    @eval $(jlop)(x::TracedRNumber) = @opcall $(hloop)(x)
end

function NNlib.softmax!(out::AnyTracedRArray{T,N}, x::AbstractArray; dims=1) where {T,N}
    x = T.(materialize_traced_array(x))
    max_ = maximum(x; dims)
    diff = exp.(x .- max_)
    # TOOD: re-enable conditional once https://github.com/EnzymeAD/Reactant.jl/issues/1581
    # fixed
    # @trace if all(isfinite, max_)
    @. out = diff
    # else
    #     @. out = ifelse(isinf(max_), ifelse(isinf(x), T(1), T(0)), diff)
    # end
    out ./= sum(out; dims)
    return out
end

function NNlib.logsoftmax!(out::AnyTracedRArray{T}, x::AbstractArray; dims=1) where {T}
    x = T.(materialize_traced_array(x))
    max_ = maximum(x; dims)
    diff = x .- max_
    # TOOD: re-enable conditional once https://github.com/EnzymeAD/Reactant.jl/issues/1581
    # fixed
    # @trace if all(isfinite, max_)
    @. out = diff
    # else
    #     @. out = ifelse(isinf(max_), ifelse(isinf(x), T(0), -T(Inf)), diff)
    # end
    out .-= log.(sum(exp, out; dims))
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
function overloaded_maxpool!(
    y::AnyTracedRArray{T,N}, x::AnyTracedRArray{T2,N}, pdims::NNlib.PoolDims;
) where {T,T2,N}
    res = reduce_window(
        max,
        T.(x);
        init=typemin(T),
        dilation=NNlib.dilation(pdims),
        kernel_size=NNlib.kernel_size(pdims),
        padding=NNlib.padding(pdims),
        stride=NNlib.stride(pdims),
    )
    set_mlir_data!(y, get_mlir_data(res))
    return y
end

function overloaded_meanpool!(
    y::AnyTracedRArray{T,N}, x::AnyTracedRArray{T2,N}, pdims::NNlib.PoolDims;
) where {T,T2,N}
    res = reduce_window(
        +,
        T.(x);
        init=zero(T),
        dilation=NNlib.dilation(pdims),
        kernel_size=NNlib.kernel_size(pdims),
        padding=NNlib.padding(pdims),
        stride=NNlib.stride(pdims),
    )
    set_mlir_data!(y, get_mlir_data(res ./ T(prod(NNlib.kernel_size(pdims)))))
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
function NNlib.gather!(dst::AnyTracedRArray, src::AnyTracedRArray, idxs::AbstractArray)
    n_dims = NNlib.scatter_dims(src, dst, idxs)
    res = _nnlib_gather_impl(src, _stack_indices(idxs), n_dims)
    set_mlir_data!(dst, get_mlir_data(res))
    return dst
end

function NNlib.gather!(
    dst::AnyTracedRArray, src::AnyTracedRArray, idxs::AbstractArray{<:Number}
)
    n_dims = NNlib.scatter_dims(src, dst, idxs)
    res = _nnlib_gather_impl(src, reshape(idxs, 1, size(idxs)...), n_dims)
    set_mlir_data!(dst, get_mlir_data(res))
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
    dst = @opcall fill(xinit, dstsz)

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
        ::typeof(mean), dst::AnyTracedRArray, src::AnyTracedRArray, idx::$AT
    )
        Ns = NNlib.scatter!(+, zero(dst), one.(src), idx)
        dst_ = NNlib.scatter!(+, zero(dst), src, idx)
        res = dst .+ NNlib.safe_div.(dst_, Ns)
        set_mlir_data!(dst, get_mlir_data(res))
        return dst
    end
end

function _nnlib_scatter_impl(
    op::OP,
    dst::AnyTracedRArray{T},
    src::AnyTracedRArray{T},
    idx::AbstractArray,
    n_dims::Int,
) where {OP,T}
    return @opcall(
        scatter(
            op,
            [dst],
            Reactant.promote_to(TracedRArray, idx),
            [src];
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
