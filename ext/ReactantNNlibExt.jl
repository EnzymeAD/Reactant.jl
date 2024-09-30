module ReactantNNlibExt

using NNlib
using Reactant: Reactant, TracedRArray, AnyTracedRArray, materialize_traced_array

for (jlop, hloop) in (
    (:(NNlib.tanh_fast), :tanh),
    (:(NNlib.sigmoid_fast), :logistic),
    (:(NNlib.sigmoid), :logistic),
)
    @eval function $(jlop)(x::TracedRArray{T,0}) where {T}
        return TracedRArray{T,0}(
            (),
            Reactant.MLIR.IR.result(
                Reactant.MLIR.Dialects.stablehlo.$(hloop)(x.mlir_data), 1
            ),
            (),
        )
    end
end

NNlib.relu(x::TracedRArray{T,0}) where {T} = max(x, zero(T))

function NNlib.gelu(x::TracedRArray{T,0}) where {T}
    α = T(0.044715)
    λλ = T(√(8 / π))
    return x * sigmoid(λλ * x * muladd(x^2, α, one(T)))
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
    if size(x, 3) == size(y, 3)
        return cat([x[:, :, i] * y[:, :, i] for i in axes(x, 3)]...; dims=Val(3))
    elseif size(x, 3) == 1
        return cat([x[:, :, i] * y[:, :, 1] for i in axes(x, 3)]...; dims=Val(3))
    elseif size(y, 3) == 1
        return cat([x[:, :, 1] * y[:, :, i] for i in axes(y, 3)]...; dims=Val(3))
    end
end

end # module ReactantNNlibExt
