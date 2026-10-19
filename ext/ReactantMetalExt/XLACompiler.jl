# XLACompiler.jl — ObjectiveC bindings for MPSGraph operations not wrapped in Metal.jl
#
# Ported from MetalPJRT/src/XLACompiler.jl.
# Provides @objc call wrappers for MPSGraph ops that Metal.jl doesn't expose,
# plus the julia_to_mps_dtype converter used by the MLIR walker.
#
# Depends on: Metal, Metal.MPS, Metal.MTL (@objc, id, NSString, etc.)
# These are imported in the parent ReactantMetalExt module.

# Helper to wrap ObjectiveC id return to proper MPSGraphTensor type
wrap_tensor(obj) = Metal.MPSGraphs.MPSGraphTensor(obj)

"""Maximum of two tensors (element-wise)"""
function mps_maximum(graph, primary, secondary, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} maximumWithPrimaryTensor:(primary::id{Metal.MPSGraphs.MPSGraphTensor})
        secondaryTensor:(secondary::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Subtraction of two tensors"""
function mps_subtract(graph, primary, secondary, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} subtractionWithPrimaryTensor:(primary::id{Metal.MPSGraphs.MPSGraphTensor})
        secondaryTensor:(secondary::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Division of two tensors"""
function mps_divide(graph, primary, secondary, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} divisionWithPrimaryTensor:(primary::id{Metal.MPSGraphs.MPSGraphTensor})
        secondaryTensor:(secondary::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Negation of a tensor"""
function mps_negate(graph, tensor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} negativeWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Exponential function"""
function mps_exp(graph, tensor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} exponentWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Natural logarithm"""
function mps_log(graph, tensor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} logarithmWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Square root"""
function mps_sqrt(graph, tensor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} squareRootWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Reciprocal square root"""
function mps_rsqrt(graph, tensor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} reciprocalSquareRootWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Hyperbolic tangent"""
function mps_tanh(graph, tensor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} tanhWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Sine"""
function mps_sin(graph, tensor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} sinWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Cosine"""
function mps_cos(graph, tensor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} cosWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Absolute value"""
function mps_abs(graph, tensor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} absoluteWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Reverse elements along specified axes"""
function mps_reverse(graph, tensor, axes::Vector{Int}, name::String)
    ns_axes = NSArray(Metal.MTL.NSNumber.(axes))
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} reverseTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        axes:(ns_axes::id{NSArray})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Reduce sum across axes (axes given as Vector{Int} in MPSGraph convention)"""
function mps_reduce_sum(graph, tensor, axes::Vector{Int}, name::String)
    ns_axes = NSArray(Metal.MTL.NSNumber.(axes))
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} reductionSumWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        axes:(ns_axes::id{NSArray})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Reduce max across axes (axes given as Vector{Int} in MPSGraph convention)"""
function mps_reduce_max(graph, tensor, axes::Vector{Int}, name::String)
    ns_axes = NSArray(Metal.MTL.NSNumber.(axes))
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} reductionMaximumWithTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        axes:(ns_axes::id{NSArray})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Reshape tensor to new shape (shape given as Vector{Int})"""
function mps_reshape(graph, tensor, shape, name::String)
    ns_shape = NSArray(Metal.MTL.NSNumber.(shape))
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} reshapeTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        withShape:(ns_shape::id{NSArray})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Concatenate tensors along a dimension"""
function mps_concatenate(
    graph, tensors::Vector{Metal.MPSGraphs.MPSGraphTensor}, dimension::Int, name::String
)
    ns_tensors = NSArray(tensors)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} concatTensors:(ns_tensors::id{NSArray})
        dimension:(dimension::Int)
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""2D Convolution with descriptor"""
function mps_convolution2d(graph, input, weights, descriptor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} convolution2DWithSourceTensor:(input::id{Metal.MPSGraphs.MPSGraphTensor})
        weightsTensor:(weights::id{Metal.MPSGraphs.MPSGraphTensor})
        descriptor:(descriptor::id{Metal.MPSGraphs.MPSGraphConvolution2DOpDescriptor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Max Pooling 2D with descriptor"""
function mps_max_pooling2d(graph, input, descriptor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} maxPooling2DWithSourceTensor:(input::id{Metal.MPSGraphs.MPSGraphTensor})
        descriptor:(descriptor::id{Metal.MPSGraphs.MPSGraphPooling2DOpDescriptor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Average Pooling 2D with descriptor"""
function mps_avg_pooling2d(graph, input, descriptor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} avgPooling2DWithSourceTensor:(input::id{Metal.MPSGraphs.MPSGraphTensor})
        descriptor:(descriptor::id{Metal.MPSGraphs.MPSGraphPooling2DOpDescriptor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""3D Convolution with descriptor"""
function mps_convolution3d(graph, input, weights, descriptor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} convolution3DWithSourceTensor:(input::id{Metal.MPSGraphs.MPSGraphTensor})
        weightsTensor:(weights::id{Metal.MPSGraphs.MPSGraphTensor})
        descriptor:(descriptor::id{Metal.MPSGraphs.MPSGraphConvolution3DOpDescriptor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Create an MPSGraphConvolution3DOpDescriptor via Apple's factory method."""
function mps_create_conv3d_descriptor(;
    stride_x=1,
    stride_y=1,
    stride_z=1,
    dilation_x=1,
    dilation_y=1,
    dilation_z=1,
    groups=1,
    padding_front=0,
    padding_back=0,
    padding_top=0,
    padding_bottom=0,
    padding_left=0,
    padding_right=0,
    data_layout=Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutNCDHW,
    weights_layout=Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutOIDHW,
)
    sx = UInt(stride_x)
    sy = UInt(stride_y)
    sz = UInt(stride_z)
    dx = UInt(dilation_x)
    dy = UInt(dilation_y)
    dz = UInt(dilation_z)
    g = UInt(groups)
    pl = UInt(padding_left)
    pr = UInt(padding_right)
    pt = UInt(padding_top)
    pb = UInt(padding_bottom)
    pf = UInt(padding_front)
    pk = UInt(padding_back)
    ps = UInt(Metal.MPSGraphs.MPSGraphPaddingStyleExplicit)
    dl = UInt(data_layout)
    wl = UInt(weights_layout)
    desc = @objc [
        MPSGraphConvolution3DOpDescriptor descriptorWithStrideInX:(sx::UInt)
        strideInY:(sy::UInt)
        strideInZ:(sz::UInt)
        dilationRateInX:(dx::UInt)
        dilationRateInY:(dy::UInt)
        dilationRateInZ:(dz::UInt)
        groups:(g::UInt)
        paddingLeft:(pl::UInt)
        paddingRight:(pr::UInt)
        paddingTop:(pt::UInt)
        paddingBottom:(pb::UInt)
        paddingFront:(pf::UInt)
        paddingBack:(pk::UInt)
        paddingStyle:(ps::UInt)
        dataLayout:(dl::UInt)
        weightsLayout:(wl::UInt)
    ]::id{MPSGraphConvolution3DOpDescriptor}
    return MPSGraphConvolution3DOpDescriptor(desc)
end

"""Create an MPSGraphConvolution2DOpDescriptor via Apple's factory method."""
function mps_create_conv2d_descriptor(;
    stride_x=1,
    stride_y=1,
    dilation_x=1,
    dilation_y=1,
    groups=1,
    padding_top=0,
    padding_bottom=0,
    padding_left=0,
    padding_right=0,
    data_layout=Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutNHWC,
    weights_layout=Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutHWIO,
)
    sx = UInt(stride_x)
    sy = UInt(stride_y)
    dx = UInt(dilation_x)
    dy = UInt(dilation_y)
    g = UInt(groups)
    pl = UInt(padding_left)
    pr = UInt(padding_right)
    pt = UInt(padding_top)
    pb = UInt(padding_bottom)
    ps = UInt(Metal.MPSGraphs.MPSGraphPaddingStyleExplicit)
    dl = UInt(data_layout)
    wl = UInt(weights_layout)
    desc = @objc [
        MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(sx::UInt)
        strideInY:(sy::UInt)
        dilationRateInX:(dx::UInt)
        dilationRateInY:(dy::UInt)
        groups:(g::UInt)
        paddingLeft:(pl::UInt)
        paddingRight:(pr::UInt)
        paddingTop:(pt::UInt)
        paddingBottom:(pb::UInt)
        paddingStyle:(ps::UInt)
        dataLayout:(dl::UInt)
        weightsLayout:(wl::UInt)
    ]::id{MPSGraphConvolution2DOpDescriptor}
    return MPSGraphConvolution2DOpDescriptor(desc)
end

"""Create an MPSGraphPooling2DOpDescriptor via Apple's factory method."""
function mps_create_pooling2d_descriptor(;
    kernel_w=2,
    kernel_h=2,
    stride_x=2,
    stride_y=2,
    dilation_x=1,
    dilation_y=1,
    padding_top=0,
    padding_bottom=0,
    padding_left=0,
    padding_right=0,
    data_layout=Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutNHWC,
)
    kw = UInt(kernel_w)
    kh = UInt(kernel_h)
    sx = UInt(stride_x)
    sy = UInt(stride_y)
    dx = UInt(dilation_x)
    dy = UInt(dilation_y)
    pl = UInt(padding_left)
    pr = UInt(padding_right)
    pt = UInt(padding_top)
    pb = UInt(padding_bottom)
    ps = UInt(Metal.MPSGraphs.MPSGraphPaddingStyleExplicit)
    dl = UInt(data_layout)
    desc = @objc [
        MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(kw::UInt)
        kernelHeight:(kh::UInt)
        strideInX:(sx::UInt)
        strideInY:(sy::UInt)
        dilationRateInX:(dx::UInt)
        dilationRateInY:(dy::UInt)
        paddingLeft:(pl::UInt)
        paddingRight:(pr::UInt)
        paddingTop:(pt::UInt)
        paddingBottom:(pb::UInt)
        paddingStyle:(ps::UInt)
        dataLayout:(dl::UInt)
    ]::id{MPSGraphPooling2DOpDescriptor}
    return MPSGraphPooling2DOpDescriptor(desc)
end

"""Create an MPSGraphPooling4DOpDescriptor via Apple's factory method.
kernel_sizes, strides, dilation_rates: length-N arrays (one per spatial dim).
padding_values: length-2N array [low0, high0, low1, high1, ..., lowN-1, highN-1]."""
function mps_create_pooling4d_descriptor(;
    kernel_sizes::Vector{Int},
    strides::Vector{Int},
    dilation_rates::Vector{Int}=ones(Int, length(kernel_sizes)),
    padding_values::Vector{Int}=zeros(Int, 2 * length(kernel_sizes)),
)
    ns_kernels = NSArray(Metal.MTL.NSNumber.(kernel_sizes))
    ns_strides = NSArray(Metal.MTL.NSNumber.(strides))
    ns_dilations = NSArray(Metal.MTL.NSNumber.(dilation_rates))
    ns_padding = NSArray(Metal.MTL.NSNumber.(padding_values))
    ps = UInt(Metal.MPSGraphs.MPSGraphPaddingStyleExplicit)
    desc = @objc [
        MPSGraphPooling4DOpDescriptor descriptorWithKernelSizes:(ns_kernels::id{NSArray})
        strides:(ns_strides::id{NSArray})
        dilationRates:(ns_dilations::id{NSArray})
        paddingValues:(ns_padding::id{NSArray})
        paddingStyle:(ps::UInt)
    ]::id{MPSGraphPooling4DOpDescriptor}
    return MPSGraphPooling4DOpDescriptor(desc)
end

"""Max Pooling 4D (N-D spatial) with descriptor"""
function mps_max_pooling4d(graph, input, descriptor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} maxPooling4DWithSourceTensor:(input::id{Metal.MPSGraphs.MPSGraphTensor})
        descriptor:(descriptor::id{Metal.MPSGraphs.MPSGraphPooling4DOpDescriptor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Average Pooling 4D (N-D spatial) with descriptor"""
function mps_avg_pooling4d(graph, input, descriptor, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} avgPooling4DWithSourceTensor:(input::id{Metal.MPSGraphs.MPSGraphTensor})
        descriptor:(descriptor::id{Metal.MPSGraphs.MPSGraphPooling4DOpDescriptor})
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Scatter ND with data tensor — scatterNDWithDataTensor:updatesTensor:indicesTensor:batchDimensions:mode:name:
Mode: 0 = Set, 1 = Add, 2 = Max, 3 = Min."""
function mps_scatter_nd(
    graph, data, updates, indices, batch_dims::Int, mode::Int, name::String
)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} scatterNDWithDataTensor:(data::id{Metal.MPSGraphs.MPSGraphTensor})
        updatesTensor:(updates::id{Metal.MPSGraphs.MPSGraphTensor})
        indicesTensor:(indices::id{Metal.MPSGraphs.MPSGraphTensor})
        batchDimensions:(UInt(batch_dims)::UInt)
        mode:(Int(mode)::Int)
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""Slice tensor along one dimension — sliceTensor:dimension:start:length:name:"""
function mps_slice_dim(graph, tensor, dim::Int, start::Int, len::Int, name::String)
    obj = @objc [
        graph::id{Metal.MPSGraphs.MPSGraph} sliceTensor:(tensor::id{Metal.MPSGraphs.MPSGraphTensor})
        dimension:(UInt(dim)::UInt)
        start:(Int(start)::Int)
        length:(Int(len)::Int)
        name:(name::id{NSString})
    ]::id{Metal.MPSGraphs.MPSGraphTensor}
    return wrap_tensor(obj)
end

"""
    julia_to_mps_dtype(dtype::DataType) -> DataType

Convert a Julia element type to the closest MPS-compatible type.
MPSGraph does not support Float64 or Int64, so those are downcast.
"""
function julia_to_mps_dtype(dtype::DataType)
    if dtype == Float32
        return Float32
    elseif dtype == Float16
        return Float16
    elseif dtype == Float64
        # MPSGraph does not support Float64 — use Float32
        return Float32
    elseif dtype == Int32
        return Int32
    elseif dtype == Int64
        # Use Int32 for Metal compatibility
        return Int32
    elseif dtype == Bool
        return Bool
    else
        return Float32
    end
end
