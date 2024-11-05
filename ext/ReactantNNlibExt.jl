module ReactantNNlibExt

using NNlib
using Reactant:
    Reactant, TracedRArray, AnyTracedRArray, materialize_traced_array, MLIR, TracedRNumber

for (jlop, hloop) in (
    (:(NNlib.tanh_fast), :tanh),
    (:(NNlib.sigmoid_fast), :logistic),
    (:(NNlib.sigmoid), :logistic),
)
    @eval function $(jlop)(x::TracedRNumber{T}) where {T}
        return TracedRNumber{T}(
            (),
            Reactant.MLIR.IR.result(
                Reactant.MLIR.Dialects.stablehlo.$(hloop)(x.mlir_data), 1
            ),
        )
    end
end

# TODO handle non finite cases
function NNlib.softmax!(out::TracedRArray{T,N}, x::AbstractArray; dims=1) where {T,N}
    max_ = NNlib.fast_maximum(x; dims)
    #if all(isfinite, max_)
    @fastmath out .= exp.(x .- max_)
    #else
    #    _zero, _one, _inf = T(0), T(1), T(Inf)
    #    @fastmath @. out = ifelse(isequal(max_,_inf), ifelse(isequal(x,_inf), _one, _zero), exp(x - max_))
    #end
    tmp = dims isa Colon ? sum(out) : sum!(max_, out)
    return out ./= tmp
end

function NNlib.logsoftmax!(out::TracedRArray{T}, x::AbstractArray; dims=1) where {T}
    max_ = NNlib.fast_maximum(x; dims)
    # if all(isfinite, max_)
    @fastmath out .= x .- max_
    # else
    #     _zero, _minf, _inf = T(0), T(-Inf), T(Inf)
    #     @. out = ifelse(
    #         isequal(max_, _inf), ifelse(isequal(x, _inf), _zero, _minf), x - max_
    #     )
    # end
    @fastmath log_ = log.(sum(exp, out; dims))
    return out .-= log_
end

function NNlib.conv(
    x::AnyTracedRArray{T,N}, W::AnyTracedRArray{T}, cdims::DenseConvDims
) where {T,N}
    x = materialize_traced_array(x)
    W = materialize_traced_array(W)

    kernel_size = NNlib.kernel_size(cdims)
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

    output_spatial_shapes = map(input_spatial_dims) do i
        K = kernel_size[i]
        pl, pr = padding[2i - 1], padding[2i]
        d = dilation[i]
        s = stride[i]

        (size(x, i) + pl + pr - d * (K - 1) - 1) ÷ s + 1
    end
    output_batch_dim = input_batch_dim
    output_feature_dim = input_feature_dim
    output_spatial_dims = input_spatial_dims

    output_shape = (output_spatial_shapes..., size(W, kernel_output_dim), size(x, N))

    dimension_numbers = """
                #stablehlo.conv<raw
                  input_batch_dimension = $(input_batch_dim - 1),
                  input_feature_dimension = $(input_feature_dim - 1),
                  input_spatial_dimensions = [$(join(input_spatial_dims .- 1, ", "))],
                  kernel_output_feature_dimension = $(kernel_output_dim - 1),
                  kernel_input_feature_dimension = $(kernel_input_dim - 1),
                  kernel_spatial_dimensions = [$(join(kernel_spatial_dims .- 1, ", "))],
                  output_batch_dimension = $( output_batch_dim - 1 ),
                  output_feature_dimension = $( output_feature_dim  - 1),
                  output_spatial_dimensions = [$(join(output_spatial_dims .- 1, ", "))],
                >"""
    dimension_numbers = parse(Reactant.MLIR.IR.Attribute, dimension_numbers)

    padding = Reactant.MLIR.IR.DenseElementsAttribute(
        reshape(collect(padding), (num_spatial_dims, 2))
    )
    result_type = Reactant.MLIR.IR.TensorType(output_shape, Reactant.MLIR.IR.Type(T))

    weight = W.mlir_data
    if !flipkernel
        weight = Reactant.MLIR.IR.result(
            Reactant.MLIR.Dialects.stablehlo.reverse(
                weight; dimensions=collect(kernel_spatial_dims .- 1)
            ),
        )
    end

    conv = Reactant.MLIR.Dialects.stablehlo.convolution(
        x.mlir_data,
        weight;
        result_0=result_type,
        window_strides=collect(stride),
        padding,
        dimension_numbers,
        lhs_dilation=1,
        rhs_dilation=collect(dilation),
        feature_group_count,
        batch_group_count=1,
    )

    return TracedRArray{T,N}((), Reactant.MLIR.IR.result(conv), output_shape)
end

function reduce_window(f, x::AnyTracedRArray{T,N}, pdims; init) where {T,N}
    x = materialize_traced_array(x)

    num_spatial_dims = N - 2
    input_spatial_dims = 1:num_spatial_dims

    dilation = NNlib.dilation(pdims)
    kernel_size = NNlib.kernel_size(pdims)
    stride = NNlib.stride(pdims)
    padding = NNlib.padding(pdims)

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
        reshape([padding..., 0, 0, 0, 0], (N, 2))
    )

    output_shape = (output_spatial_shapes..., size(x, N - 1), size(x, N))
    result_type = Reactant.MLIR.IR.TensorType(output_shape, Reactant.MLIR.IR.Type(T))

    unranked = Reactant.MLIR.IR.TensorType((), eltype(Reactant.MLIR.IR.type(x.mlir_data)))
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
        [x.mlir_data],
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

function NNlib.maxpool(x::AnyTracedRArray{T}, pdims::NNlib.PoolDims) where {T}
    return reduce_window(
        Reactant.MLIR.Dialects.stablehlo.maximum, x, pdims; init=typemin(T)
    )
end

function NNlib.meanpool(x::AnyTracedRArray{T}, pdims::NNlib.PoolDims) where {T}
    numel = prod(NNlib.kernel_size(pdims))
    return reduce_window(Reactant.MLIR.Dialects.stablehlo.add, x, pdims; init=zero(T)) ./
           T(numel)
end

NNlib.batched_transpose(x::AnyTracedRArray{T,3}) where {T} = permutedims(x, (2, 1, 3))
NNlib.batched_adjoint(x::AnyTracedRArray{<:Real,3}) = NNlib.batched_transpose(x)

function NNlib.batched_mul(x::AnyTracedRArray{T,3}, y::AnyTracedRArray{T,3}) where {T}
    if (size(x, 3) != size(y, 3) && size(x, 3) != 1 && size(y, 3) != 1) ||
        (size(x, 2) != size(y, 1))
        throw(
            DimensionMismatch(
                lazy"size(x) = $(size(x)), size(y) = $(size(y)) inconsistent for batched_matmul.",
            ),
        )
    end
    x = permutedims(x, (3, 1, 2))
    y = permutedims(y, (3, 1, 2))

    B = max(size(x, 1), size(y, 1))
    out_shape = (B, size(x, 2), size(y, 3))
    resty = MLIR.IR.TensorType(out_shape, eltype(MLIR.IR.type(x.mlir_data)))

    if size(x, 1) != size(y, 1)
        if size(x, 1) == 1
            x = Reactant.broadcast_to_size(x, (B, size(x, 2), size(x, 3)))
        elseif size(y, 1) == 1
            y = Reactant.broadcast_to_size(y, (B, size(y, 2), size(y, 3)))
        end
    end

    dot_dimension_numbers = MLIR.API.stablehloDotDimensionNumbersGet(
        MLIR.IR.context(), 1, [0], 1, [0], 1, [2], 1, [1]
    )

    prec = MLIR.IR.Attribute(
        MLIR.API.stablehloPrecisionAttrGet(MLIR.IR.context(), "DEFAULT")
    )
    res = TracedRArray{T,3}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.dot_general(
                x.mlir_data,
                y.mlir_data;
                result_0=resty,
                dot_dimension_numbers=dot_dimension_numbers,
                precision_config=prec,
            ),
            1,
        ),
        size(resty),
    )
    return permutedims(res, (2, 3, 1))
end

function NNlib.pad_constant(
    x::TracedRArray{T,N}, pad::NTuple{N,Tuple{Int,Int}}, value
) where {T,N}
    value = Reactant.promote_to(TracedRNumber{T}, value)
    edge_padding_low = [i[1] for i in pad]
    edge_padding_high = [i[2] for i in pad]
    interior_padding = [0 for i in pad]
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.pad(
            x.mlir_data,
            value.mlir_data;
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        ),
        1,
    )
    return TracedRArray{T,N}((), res, size(MLIR.IR.type(res)))
end

end # module ReactantNNlibExt
