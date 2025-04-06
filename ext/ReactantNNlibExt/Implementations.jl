for (jlop, hloop) in (
    (:(NNlib.tanh_fast), :tanh),
    (:(NNlib.sigmoid_fast), :logistic),
    (:(NNlib.sigmoid), :logistic),
)
    @eval $(jlop)(x::TracedRNumber) = Ops.$(hloop)(x)
end

function NNlib.softmax!(out::AnyTracedRArray{T,N}, x::AbstractArray; dims=1) where {T,N}
    max_ = NNlib.fast_maximum(x; dims)
    # XXX: Once reverse mode of if is properly supported, we can make it @trace
    # zero_num = TracedUtils.promote_to(TracedRNumber{T}, 0)
    # one_num = TracedUtils.promote_to(TracedRNumber{T}, 1)
    # @trace if all(isfinite, max_)
    @. out = exp(x - max_)
    # else
    #     cond = max_ .== Inf
    #     true_pred = ifelse.(x .== Inf, one_num, zero_num)
    #     @. out = ifelse(cond, true_pred, exp(x - max_))
    # end
    tmp = dims isa Colon ? sum(out) : sum!(max_, out)
    out ./= tmp
    return out
end

function NNlib.logsoftmax!(out::AnyTracedRArray{T}, x::AbstractArray; dims=1) where {T}
    max_ = NNlib.fast_maximum(x; dims)
    # XXX: Once reverse mode of if is properly supported, we can make it @trace
    # inf_num = TracedUtils.promote_to(TracedRNumber{T}, Inf)
    # zero_num = TracedUtils.promote_to(TracedRNumber{T}, 0)
    # @trace if all(isfinite, max_)
    @. out = x - max_
    # else
    #     cond = max_ .== Inf
    #     true_pred = ifelse.(x .== Inf, zero_num, -inf_num)
    #     @. out = ifelse(cond, true_pred, x - max_)
    # end
    @fastmath log_ = log.(sum(exp, out; dims))
    out .-= log_
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
    W = T.(materialize_traced_array(W))

    padding = NNlib.padding(cdims)
    stride = NNlib.stride(cdims)
    dilation = NNlib.dilation(cdims)
    flipkernel = NNlib.flipkernel(cdims)
    feature_group_count = NNlib.groupcount(cdims)

    input_rank = ndims(x)

    num_spatial_dims = input_rank - 2

    input_spatial_dims = 1:num_spatial_dims
    input_feature_dim = N - 1
    input_batch_dim = N

    kernel_spatial_dims = input_spatial_dims
    kernel_input_dim = N - 1
    kernel_output_dim = N

    output_batch_dim = input_batch_dim
    output_feature_dim = input_feature_dim
    output_spatial_dims = input_spatial_dims

    #! format: off
    dimension_numbers = MLIR.API.stablehloConvDimensionNumbersGet(
        MLIR.IR.context(),
        Int64(input_batch_dim - 1),
        Int64(input_feature_dim - 1),
        length(input_spatial_dims), Int64[i - 1 for i in input_spatial_dims],
        Int64(kernel_input_dim - 1),
        Int64(kernel_output_dim - 1),
        length(kernel_spatial_dims), Int64[i - 1 for i in kernel_spatial_dims],
        Int64(output_batch_dim - 1),
        Int64(output_feature_dim - 1),
        length(output_spatial_dims), Int64[i - 1 for i in output_spatial_dims],
    )
    #! format: on

    padding = Reactant.MLIR.IR.DenseElementsAttribute(
        reshape(collect(padding), (2, num_spatial_dims))'
    )
    result_type = Reactant.MLIR.IR.TensorType(
        collect(Int, size(y)), Reactant.MLIR.IR.Type(T)
    )

    conv = Reactant.MLIR.Dialects.stablehlo.convolution(
        get_mlir_data(x),
        get_mlir_data(W);
        result_0=result_type,
        window_reversal=if flipkernel
            nothing
        else
            [true for _ in 1:length(kernel_spatial_dims)]
        end,
        window_strides=collect(stride),
        padding,
        dimension_numbers,
        lhs_dilation=1,
        rhs_dilation=collect(dilation),
        feature_group_count,
        batch_group_count=1,
    )
    set_mlir_data!(y, Reactant.MLIR.IR.result(conv))
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
    input_batch_dim = N - 1
    input_feature_dim = N

    kernel_input_dim = N
    kernel_output_dim = N - 1

    output_batch_dim = N - 1
    output_feature_dim = N

    output_spatial_dims = kernel_spatial_dims = input_spatial_dims = 1:num_spatial_dims

    padding = reshape(collect(NNlib.padding(cdims)), (2, num_spatial_dims))
    stride = NNlib.stride(cdims)
    dilation = NNlib.dilation(cdims)
    feature_group_count = NNlib.groupcount(cdims)

    padding =
        let lhs_shape = first(size(x), num_spatial_dims),
            rhs_shape = dilate_shape.(first(size(dw), num_spatial_dims), dilation),
            out_shape = dilate_shape.(first(size(dy), num_spatial_dims), stride),

            padding = reduce(
                hcat,
                (
                    let pad_before = padding[1, i],
                        pad_after = (
                            out_shape[i] - lhs_shape[i] + rhs_shape[i] - pad_before - 1
                        )

                        [pad_before, pad_after]
                    end for i in 1:num_spatial_dims
                ),
            )

            Reactant.MLIR.IR.DenseElementsAttribute(padding')
        end

    batch_group_count = 1
    if feature_group_count > 1
        batch_group_count = feature_group_count
        feature_group_count = 1
    end

    dimension_numbers = MLIR.API.stablehloConvDimensionNumbersGet(
        MLIR.IR.context(),
        Int64(input_batch_dim - 1),
        Int64(input_feature_dim - 1),
        length(input_spatial_dims),
        Int64[i - 1 for i in input_spatial_dims],
        Int64(kernel_input_dim - 1),
        Int64(kernel_output_dim - 1),
        length(kernel_spatial_dims),
        Int64[i - 1 for i in kernel_spatial_dims],
        Int64(output_batch_dim - 1),
        Int64(output_feature_dim - 1),
        length(output_spatial_dims),
        Int64[i - 1 for i in output_spatial_dims],
    )

    result_type = Reactant.MLIR.IR.TensorType(
        collect(Int, size(dw)), Reactant.MLIR.IR.Type(T)
    )
    conv = MLIR.Dialects.stablehlo.convolution(
        get_mlir_data(x),
        get_mlir_data(dy);
        result_0=result_type,
        window_strides=collect(dilation),
        padding,
        dimension_numbers,
        rhs_dilation=collect(stride),
        feature_group_count,
        batch_group_count,
    )
    set_mlir_data!(dw, MLIR.IR.result(conv))

    if !NNlib.flipkernel(cdims)
        set_mlir_data!(
            dw, get_mlir_data(Reactant.Ops.reverse(dw; dimensions=output_spatial_dims))
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

    num_spatial_dims = N - 2
    input_batch_dim = N
    input_feature_dim = N - 1

    kernel_input_dim = N
    kernel_output_dim = N - 1

    output_batch_dim = N
    output_feature_dim = N - 1

    output_spatial_dims = kernel_spatial_dims = input_spatial_dims = 1:num_spatial_dims

    padding = reshape(collect(NNlib.padding(cdims)), (2, num_spatial_dims))
    stride = NNlib.stride(cdims)
    dilation = NNlib.dilation(cdims)
    feature_group_count = NNlib.groupcount(cdims)

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
        w = reshape(
            w,
            (size(w, i) for i in kernel_spatial_dims)...,
            size(w, N - 1) * feature_group_count,
            size(w, N + 1),
        )
    end

    padding =
        let lhs_shape = first(size(dx), num_spatial_dims),
            rhs_shape = dilate_shape.(first(size(w), num_spatial_dims), dilation),
            out_shape = dilate_shape.(first(size(dy), num_spatial_dims), stride),

            padding = reduce(
                hcat,
                (
                    let pad_before = rhs_shape[i] - padding[2i - 1] - 1,
                        pad_after =
                            lhs_shape[i] + rhs_shape[i] - 1 - out_shape[i] - pad_before

                        [pad_before, pad_after]
                    end for i in input_spatial_dims
                ),
            )

            Reactant.MLIR.IR.DenseElementsAttribute(padding')
        end

    dimension_numbers = MLIR.API.stablehloConvDimensionNumbersGet(
        MLIR.IR.context(),
        Int64(input_batch_dim - 1),
        Int64(input_feature_dim - 1),
        length(input_spatial_dims),
        Int64[i - 1 for i in input_spatial_dims],
        Int64(kernel_input_dim - 1),
        Int64(kernel_output_dim - 1),
        length(kernel_spatial_dims),
        Int64[i - 1 for i in kernel_spatial_dims],
        Int64(output_batch_dim - 1),
        Int64(output_feature_dim - 1),
        length(output_spatial_dims),
        Int64[i - 1 for i in output_spatial_dims],
    )

    result_type = Reactant.MLIR.IR.TensorType(
        collect(Int, size(dx)), Reactant.MLIR.IR.Type(T)
    )

    if NNlib.flipkernel(cdims)
        w = Reactant.Ops.reverse(w; dimensions=kernel_spatial_dims)
    end

    conv = MLIR.Dialects.stablehlo.convolution(
        get_mlir_data(dy),
        get_mlir_data(w);
        result_0=result_type,
        window_strides=1,
        padding,
        lhs_dilation=collect(stride),
        rhs_dilation=collect(dilation),
        dimension_numbers,
        feature_group_count,
        batch_group_count=1,
    )
    set_mlir_data!(dx, MLIR.IR.result(conv))

    return dx
end

# Pooling
function overloaded_maxpool!(
    y::AnyTracedRArray{T,N}, x::AnyTracedRArray{T2,N}, pdims::NNlib.PoolDims;
) where {T,T2,N}
    res = reduce_window(
        Reactant.MLIR.Dialects.stablehlo.maximum,
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
        Reactant.MLIR.Dialects.stablehlo.add,
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

    if size(x, 3) != size(y, 3)
        B = max(size(x, 3), size(y, 3))
        if size(x, 3) == 1
            x = TracedUtils.broadcast_to_size(x, (size(x, 1), size(x, 2), B))
        elseif size(y, 3) == 1
            y = TracedUtils.broadcast_to_size(y, (size(y, 1), size(y, 2), B))
        end
    end

    x = permutedims(x, (3, 1, 2))
    y = permutedims(y, (3, 1, 2))

    if size(x, 1) != size(y, 1)
        B = max(size(x, 1), size(y, 1))
        if size(x, 1) == 1
            x = TracedUtils.broadcast_to_size(x, (B, size(x, 2), size(x, 3)))
        elseif size(y, 1) == 1
            y = TracedUtils.broadcast_to_size(y, (B, size(y, 2), size(y, 3)))
        end
    end

    tmp = Ops.dot_general(
        T1.(materialize_traced_array(x)),
        T1.(materialize_traced_array(y));
        contracting_dimensions=([3], [2]),
        batching_dimensions=([1], [1]),
    )
    set_mlir_data!(res, get_mlir_data(permutedims(tmp, (2, 3, 1))))

    return res
end

# Padding
function NNlib.pad_constant(
    x::AnyTracedRArray{T,N}, pad::NTuple{N,Tuple{Int,Int}}, value
) where {T,N}
    value = TracedUtils.promote_to(TracedRNumber{T}, value)
    low = [i[1] for i in pad]
    high = [i[2] for i in pad]
    interior = [0 for i in pad]
    return Ops.pad(materialize_traced_array(x), value; low, high, interior)
end

# Gather
# XXX: reevaluate this manual optimization once
#      https://github.com/EnzymeAD/Enzyme-JAX/issues/164 is handled
function NNlib.gather!(
    dst::AnyTracedRArray{T1,2},
    src::AnyTracedRArray{T2,2},
    idxs::Union{AbstractUnitRange{<:Number}},
) where {T1,T2}
    set_mlir_data!(dst, get_mlir_data(src[:, idxs]))
    return dst
end

function NNlib.gather!(
    dst::AnyTracedRArray{T1,2}, src::AnyTracedRArray{T2,2}, idxs::AbstractVector{<:Number}
) where {T1,T2}
    dims = NNlib.scatter_dims(src, dst, idxs)
    @assert dims == 1  # scatter_dims lets us do some size checks so we call that function
    idxs = get_mlir_data(TracedUtils.promote_to(TracedRArray{Int,1}, idxs) .- 1)
    slice_sizes = get_mlir_data(
        TracedUtils.promote_to(TracedRArray{Int,1}, [size(src, 1), 1])
    )

    #! format: off
    dimension_numbers = MLIR.API.stablehloGatherDimensionNumbersGet(
        MLIR.IR.context(),
        Int64(1), Int64[0],
        Int64(1), Int64[1],
        Int64(0), Int64[],
        Int64(0), Int64[],
        Int64(1), Int64[1],
        Int64(1)
    )
    #! format: on

    res = MLIR.IR.result(
        Reactant.MLIR.Dialects.stablehlo.dynamic_gather(
            get_mlir_data(src), idxs, slice_sizes; dimension_numbers
        ),
        1,
    )
    set_mlir_data!(dst, res)
    return dst
end

# XXX: For performance to use `stablehlo.dynamic_gather` or atleast use traced loop
#      instead of unrolling the loop (the case for AbstractArray can just use
#      `stablehlo.gather`). See above for the special case implementation that is optimized.
function NNlib.gather!(dst::AnyTracedRArray, src::AnyTracedRArray, idxs::AbstractArray)
    @warn "Using fallback implementation of `gather!` for using `stablehlo.dynamic_slice`. \
           This case is not optimized and will be slow." maxlog = 1
    dims = NNlib.scatter_dims(src, dst, idxs)
    colons = ntuple(Returns(Colon()), dims)
    start_sizes = ntuple(Base.Fix1(size, src), dims)
    results = map(CartesianIndices(idxs)) do k
        res = @allowscalar src[colons..., Tuple(idxs[k])...]
        res isa TracedRNumber && (res = TracedUtils.broadcast_to_size(res, (1,)))
        return reshape(res, start_sizes..., :)
    end
    res = reshape(cat(results...; dims=(dims + 1)), size(dst))
    set_mlir_data!(dst, get_mlir_data(res))
    return dst
end
