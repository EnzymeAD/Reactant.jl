module ReactantNNlibExt

using NNlib
using Reactant

for (jlop, hloop) in (
    (:(NNlib.tanh_fast), :tanh),
    (:(NNlib.sigmoid_fast), :logistic),
    (:(NNlib.sigmoid), :logistic),
)
    @eval function $(jlop)(x::Reactant.TracedRArray{T,0}) where {T}
        return Reactant.TracedRArray{T,0}(
            (),
            Reactant.MLIR.IR.result(
                Reactant.MLIR.Dialects.stablehlo.$(hloop)(x.mlir_data), 1
            ),
            (),
        )
    end
end

NNlib.relu(x::Reactant.TracedRArray{T,0}) where {T} = max(x, zero(T))

NNlib.gelu(x::Reactant.TracedRArray{T,0}) where {T} = x * sigmoid(T(1.702) * x)

# TODO handle non finite cases
function NNlib.softmax!(
    out::Reactant.TracedRArray{T,N}, x::AbstractArray; dims=1
) where {T,N}
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
    x::Reactant.TracedRArray{T,N}, W::Reactant.TracedRArray{T}, cdims::DenseConvDims
) where {T,N}
    kernel_size = NNlib.kernel_size(cdims)
    padding = NNlib.padding(cdims)
    stride = NNlib.stride(cdims)
    dilation = NNlib.dilation(cdims)
    flipkernel = NNlib.flipkernel(cdims)

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

    conv = Reactant.MLIR.Dialects.stablehlo.convolution(
        x.mlir_data,
        W.mlir_data;
        result_0=result_type,
        window_strides=collect(stride),
        window_reversal=collect(fill(flipkernel, num_spatial_dims)),
        padding,
        dimension_numbers,
        lhs_dilation=1,
        rhs_dilation=collect(dilation),
        feature_group_count=1,
        batch_group_count=1,
    )

    return Reactant.TracedRArray{T,N}((), Reactant.MLIR.IR.result(conv), output_shape)
end

end
