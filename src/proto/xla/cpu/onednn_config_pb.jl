import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export OneDnnWindowProto, OneDnnOptimizationConfig, OneDnnDataLayoutProto
export OneDnnFactorLayoutProto, OneDnnFilterLayoutProto
export var"OneDnnNormConfig.ScaleAndShift", var"OneDnnFusionConfig.FusionKind"
export OneDnnSoftmaxConfig, OneDnnTensorLayoutProto, OneDnnFusionConfig, OneDnnMatMulConfig
export OneDnnNormConfig, OneDnnConvolutionConfig


struct OneDnnWindowProto
    size::Vector{UInt64}
    pad_left::Vector{UInt64}
    pad_right::Vector{UInt64}
    strides::Vector{UInt64}
    window_dilations::Vector{UInt64}
end
PB.default_values(::Type{OneDnnWindowProto}) = (;size = Vector{UInt64}(), pad_left = Vector{UInt64}(), pad_right = Vector{UInt64}(), strides = Vector{UInt64}(), window_dilations = Vector{UInt64}())
PB.field_numbers(::Type{OneDnnWindowProto}) = (;size = 1, pad_left = 2, pad_right = 3, strides = 4, window_dilations = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnWindowProto}, _endpos::Int=0, _group::Bool=false)
    size = PB.BufferedVector{UInt64}()
    pad_left = PB.BufferedVector{UInt64}()
    pad_right = PB.BufferedVector{UInt64}()
    strides = PB.BufferedVector{UInt64}()
    window_dilations = PB.BufferedVector{UInt64}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, size)
        elseif field_number == 2
            PB.decode!(d, wire_type, pad_left)
        elseif field_number == 3
            PB.decode!(d, wire_type, pad_right)
        elseif field_number == 4
            PB.decode!(d, wire_type, strides)
        elseif field_number == 5
            PB.decode!(d, wire_type, window_dilations)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnWindowProto(size[], pad_left[], pad_right[], strides[], window_dilations[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnWindowProto)
    initpos = position(e.io)
    !isempty(x.size) && PB.encode(e, 1, x.size)
    !isempty(x.pad_left) && PB.encode(e, 2, x.pad_left)
    !isempty(x.pad_right) && PB.encode(e, 3, x.pad_right)
    !isempty(x.strides) && PB.encode(e, 4, x.strides)
    !isempty(x.window_dilations) && PB.encode(e, 5, x.window_dilations)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnWindowProto)
    encoded_size = 0
    !isempty(x.size) && (encoded_size += PB._encoded_size(x.size, 1))
    !isempty(x.pad_left) && (encoded_size += PB._encoded_size(x.pad_left, 2))
    !isempty(x.pad_right) && (encoded_size += PB._encoded_size(x.pad_right, 3))
    !isempty(x.strides) && (encoded_size += PB._encoded_size(x.strides, 4))
    !isempty(x.window_dilations) && (encoded_size += PB._encoded_size(x.window_dilations, 5))
    return encoded_size
end

struct OneDnnOptimizationConfig
    weights_prepacked::Bool
    user_scratchpad::Bool
    bias_broadcast::Bool
end
PB.default_values(::Type{OneDnnOptimizationConfig}) = (;weights_prepacked = false, user_scratchpad = false, bias_broadcast = false)
PB.field_numbers(::Type{OneDnnOptimizationConfig}) = (;weights_prepacked = 1, user_scratchpad = 2, bias_broadcast = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnOptimizationConfig}, _endpos::Int=0, _group::Bool=false)
    weights_prepacked = false
    user_scratchpad = false
    bias_broadcast = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            weights_prepacked = PB.decode(d, Bool)
        elseif field_number == 2
            user_scratchpad = PB.decode(d, Bool)
        elseif field_number == 3
            bias_broadcast = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnOptimizationConfig(weights_prepacked, user_scratchpad, bias_broadcast)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnOptimizationConfig)
    initpos = position(e.io)
    x.weights_prepacked != false && PB.encode(e, 1, x.weights_prepacked)
    x.user_scratchpad != false && PB.encode(e, 2, x.user_scratchpad)
    x.bias_broadcast != false && PB.encode(e, 3, x.bias_broadcast)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnOptimizationConfig)
    encoded_size = 0
    x.weights_prepacked != false && (encoded_size += PB._encoded_size(x.weights_prepacked, 1))
    x.user_scratchpad != false && (encoded_size += PB._encoded_size(x.user_scratchpad, 2))
    x.bias_broadcast != false && (encoded_size += PB._encoded_size(x.bias_broadcast, 3))
    return encoded_size
end

struct OneDnnDataLayoutProto
    batch_dim::UInt64
    feature_dim::UInt64
    spatial_dims::Vector{UInt64}
end
PB.default_values(::Type{OneDnnDataLayoutProto}) = (;batch_dim = zero(UInt64), feature_dim = zero(UInt64), spatial_dims = Vector{UInt64}())
PB.field_numbers(::Type{OneDnnDataLayoutProto}) = (;batch_dim = 1, feature_dim = 2, spatial_dims = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnDataLayoutProto}, _endpos::Int=0, _group::Bool=false)
    batch_dim = zero(UInt64)
    feature_dim = zero(UInt64)
    spatial_dims = PB.BufferedVector{UInt64}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            batch_dim = PB.decode(d, UInt64)
        elseif field_number == 2
            feature_dim = PB.decode(d, UInt64)
        elseif field_number == 3
            PB.decode!(d, wire_type, spatial_dims)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnDataLayoutProto(batch_dim, feature_dim, spatial_dims[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnDataLayoutProto)
    initpos = position(e.io)
    x.batch_dim != zero(UInt64) && PB.encode(e, 1, x.batch_dim)
    x.feature_dim != zero(UInt64) && PB.encode(e, 2, x.feature_dim)
    !isempty(x.spatial_dims) && PB.encode(e, 3, x.spatial_dims)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnDataLayoutProto)
    encoded_size = 0
    x.batch_dim != zero(UInt64) && (encoded_size += PB._encoded_size(x.batch_dim, 1))
    x.feature_dim != zero(UInt64) && (encoded_size += PB._encoded_size(x.feature_dim, 2))
    !isempty(x.spatial_dims) && (encoded_size += PB._encoded_size(x.spatial_dims, 3))
    return encoded_size
end

struct OneDnnFactorLayoutProto
    dimensions::Vector{UInt64}
    shape::Vector{UInt64}
end
PB.default_values(::Type{OneDnnFactorLayoutProto}) = (;dimensions = Vector{UInt64}(), shape = Vector{UInt64}())
PB.field_numbers(::Type{OneDnnFactorLayoutProto}) = (;dimensions = 1, shape = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnFactorLayoutProto}, _endpos::Int=0, _group::Bool=false)
    dimensions = PB.BufferedVector{UInt64}()
    shape = PB.BufferedVector{UInt64}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, dimensions)
        elseif field_number == 2
            PB.decode!(d, wire_type, shape)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnFactorLayoutProto(dimensions[], shape[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnFactorLayoutProto)
    initpos = position(e.io)
    !isempty(x.dimensions) && PB.encode(e, 1, x.dimensions)
    !isempty(x.shape) && PB.encode(e, 2, x.shape)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnFactorLayoutProto)
    encoded_size = 0
    !isempty(x.dimensions) && (encoded_size += PB._encoded_size(x.dimensions, 1))
    !isempty(x.shape) && (encoded_size += PB._encoded_size(x.shape, 2))
    return encoded_size
end

struct OneDnnFilterLayoutProto
    input_feature_dim::UInt64
    output_feature_dim::UInt64
    spatial_dims::Vector{UInt64}
    shape::Vector{UInt64}
end
PB.default_values(::Type{OneDnnFilterLayoutProto}) = (;input_feature_dim = zero(UInt64), output_feature_dim = zero(UInt64), spatial_dims = Vector{UInt64}(), shape = Vector{UInt64}())
PB.field_numbers(::Type{OneDnnFilterLayoutProto}) = (;input_feature_dim = 1, output_feature_dim = 2, spatial_dims = 3, shape = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnFilterLayoutProto}, _endpos::Int=0, _group::Bool=false)
    input_feature_dim = zero(UInt64)
    output_feature_dim = zero(UInt64)
    spatial_dims = PB.BufferedVector{UInt64}()
    shape = PB.BufferedVector{UInt64}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            input_feature_dim = PB.decode(d, UInt64)
        elseif field_number == 2
            output_feature_dim = PB.decode(d, UInt64)
        elseif field_number == 3
            PB.decode!(d, wire_type, spatial_dims)
        elseif field_number == 4
            PB.decode!(d, wire_type, shape)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnFilterLayoutProto(input_feature_dim, output_feature_dim, spatial_dims[], shape[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnFilterLayoutProto)
    initpos = position(e.io)
    x.input_feature_dim != zero(UInt64) && PB.encode(e, 1, x.input_feature_dim)
    x.output_feature_dim != zero(UInt64) && PB.encode(e, 2, x.output_feature_dim)
    !isempty(x.spatial_dims) && PB.encode(e, 3, x.spatial_dims)
    !isempty(x.shape) && PB.encode(e, 4, x.shape)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnFilterLayoutProto)
    encoded_size = 0
    x.input_feature_dim != zero(UInt64) && (encoded_size += PB._encoded_size(x.input_feature_dim, 1))
    x.output_feature_dim != zero(UInt64) && (encoded_size += PB._encoded_size(x.output_feature_dim, 2))
    !isempty(x.spatial_dims) && (encoded_size += PB._encoded_size(x.spatial_dims, 3))
    !isempty(x.shape) && (encoded_size += PB._encoded_size(x.shape, 4))
    return encoded_size
end

@enumx var"OneDnnNormConfig.ScaleAndShift" UNDEFINED=0 SCALE=1 SHIFT=2 SCALE_AND_SHIFT=3

@enumx var"OneDnnFusionConfig.FusionKind" UNDEFINED=0 BIAS=1 RELU=2 TANH=3 GELU_ERF=4 GELU_TANH=5 BINARY_ADD=6 LINEAR=7 ELU=8 RELU6=9 SIGMOID=10 SUM=11 SWISH=12

struct OneDnnSoftmaxConfig
    softmax_axis::Int32
end
PB.default_values(::Type{OneDnnSoftmaxConfig}) = (;softmax_axis = zero(Int32))
PB.field_numbers(::Type{OneDnnSoftmaxConfig}) = (;softmax_axis = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnSoftmaxConfig}, _endpos::Int=0, _group::Bool=false)
    softmax_axis = zero(Int32)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            softmax_axis = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnSoftmaxConfig(softmax_axis)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnSoftmaxConfig)
    initpos = position(e.io)
    x.softmax_axis != zero(Int32) && PB.encode(e, 1, x.softmax_axis)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnSoftmaxConfig)
    encoded_size = 0
    x.softmax_axis != zero(Int32) && (encoded_size += PB._encoded_size(x.softmax_axis, 1))
    return encoded_size
end

struct OneDnnTensorLayoutProto
    dims::UInt64
    layout::Union{Nothing,OneOf{<:Union{OneDnnDataLayoutProto,OneDnnFilterLayoutProto,OneDnnFactorLayoutProto}}}
end
PB.oneof_field_types(::Type{OneDnnTensorLayoutProto}) = (;
    layout = (;data=OneDnnDataLayoutProto, filter=OneDnnFilterLayoutProto, tensor=OneDnnFactorLayoutProto),
)
PB.default_values(::Type{OneDnnTensorLayoutProto}) = (;dims = zero(UInt64), data = nothing, filter = nothing, tensor = nothing)
PB.field_numbers(::Type{OneDnnTensorLayoutProto}) = (;dims = 1, data = 2, filter = 3, tensor = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnTensorLayoutProto}, _endpos::Int=0, _group::Bool=false)
    dims = zero(UInt64)
    layout = nothing
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            dims = PB.decode(d, UInt64)
        elseif field_number == 2
            layout = OneOf(:data, PB.decode(d, Ref{OneDnnDataLayoutProto}))
        elseif field_number == 3
            layout = OneOf(:filter, PB.decode(d, Ref{OneDnnFilterLayoutProto}))
        elseif field_number == 4
            layout = OneOf(:tensor, PB.decode(d, Ref{OneDnnFactorLayoutProto}))
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnTensorLayoutProto(dims, layout)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnTensorLayoutProto)
    initpos = position(e.io)
    x.dims != zero(UInt64) && PB.encode(e, 1, x.dims)
    if isnothing(x.layout);
    elseif x.layout.name === :data
        PB.encode(e, 2, x.layout[]::OneDnnDataLayoutProto)
    elseif x.layout.name === :filter
        PB.encode(e, 3, x.layout[]::OneDnnFilterLayoutProto)
    elseif x.layout.name === :tensor
        PB.encode(e, 4, x.layout[]::OneDnnFactorLayoutProto)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnTensorLayoutProto)
    encoded_size = 0
    x.dims != zero(UInt64) && (encoded_size += PB._encoded_size(x.dims, 1))
    if isnothing(x.layout);
    elseif x.layout.name === :data
        encoded_size += PB._encoded_size(x.layout[]::OneDnnDataLayoutProto, 2)
    elseif x.layout.name === :filter
        encoded_size += PB._encoded_size(x.layout[]::OneDnnFilterLayoutProto, 3)
    elseif x.layout.name === :tensor
        encoded_size += PB._encoded_size(x.layout[]::OneDnnFactorLayoutProto, 4)
    end
    return encoded_size
end

struct OneDnnFusionConfig
    ops::Vector{var"OneDnnFusionConfig.FusionKind".T}
    alpha::Vector{Float32}
end
PB.reserved_fields(::Type{OneDnnFusionConfig}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[2])
PB.default_values(::Type{OneDnnFusionConfig}) = (;ops = Vector{var"OneDnnFusionConfig.FusionKind".T}(), alpha = Vector{Float32}())
PB.field_numbers(::Type{OneDnnFusionConfig}) = (;ops = 1, alpha = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnFusionConfig}, _endpos::Int=0, _group::Bool=false)
    ops = PB.BufferedVector{var"OneDnnFusionConfig.FusionKind".T}()
    alpha = PB.BufferedVector{Float32}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, ops)
        elseif field_number == 3
            PB.decode!(d, wire_type, alpha)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnFusionConfig(ops[], alpha[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnFusionConfig)
    initpos = position(e.io)
    !isempty(x.ops) && PB.encode(e, 1, x.ops)
    !isempty(x.alpha) && PB.encode(e, 3, x.alpha)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnFusionConfig)
    encoded_size = 0
    !isempty(x.ops) && (encoded_size += PB._encoded_size(x.ops, 1))
    !isempty(x.alpha) && (encoded_size += PB._encoded_size(x.alpha, 3))
    return encoded_size
end

struct OneDnnMatMulConfig
    transpose_a::Bool
    transpose_b::Bool
    fusions::Union{Nothing,OneDnnFusionConfig}
    optimization_config::Union{Nothing,OneDnnOptimizationConfig}
    lhs::Union{Nothing,OneDnnTensorLayoutProto}
    rhs::Union{Nothing,OneDnnTensorLayoutProto}
    result::Union{Nothing,OneDnnTensorLayoutProto}
end
PB.reserved_fields(::Type{OneDnnMatMulConfig}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[4, 5, 6, 7])
PB.default_values(::Type{OneDnnMatMulConfig}) = (;transpose_a = false, transpose_b = false, fusions = nothing, optimization_config = nothing, lhs = nothing, rhs = nothing, result = nothing)
PB.field_numbers(::Type{OneDnnMatMulConfig}) = (;transpose_a = 1, transpose_b = 2, fusions = 3, optimization_config = 8, lhs = 9, rhs = 10, result = 11)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnMatMulConfig}, _endpos::Int=0, _group::Bool=false)
    transpose_a = false
    transpose_b = false
    fusions = Ref{Union{Nothing,OneDnnFusionConfig}}(nothing)
    optimization_config = Ref{Union{Nothing,OneDnnOptimizationConfig}}(nothing)
    lhs = Ref{Union{Nothing,OneDnnTensorLayoutProto}}(nothing)
    rhs = Ref{Union{Nothing,OneDnnTensorLayoutProto}}(nothing)
    result = Ref{Union{Nothing,OneDnnTensorLayoutProto}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            transpose_a = PB.decode(d, Bool)
        elseif field_number == 2
            transpose_b = PB.decode(d, Bool)
        elseif field_number == 3
            PB.decode!(d, fusions)
        elseif field_number == 8
            PB.decode!(d, optimization_config)
        elseif field_number == 9
            PB.decode!(d, lhs)
        elseif field_number == 10
            PB.decode!(d, rhs)
        elseif field_number == 11
            PB.decode!(d, result)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnMatMulConfig(transpose_a, transpose_b, fusions[], optimization_config[], lhs[], rhs[], result[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnMatMulConfig)
    initpos = position(e.io)
    x.transpose_a != false && PB.encode(e, 1, x.transpose_a)
    x.transpose_b != false && PB.encode(e, 2, x.transpose_b)
    !isnothing(x.fusions) && PB.encode(e, 3, x.fusions)
    !isnothing(x.optimization_config) && PB.encode(e, 8, x.optimization_config)
    !isnothing(x.lhs) && PB.encode(e, 9, x.lhs)
    !isnothing(x.rhs) && PB.encode(e, 10, x.rhs)
    !isnothing(x.result) && PB.encode(e, 11, x.result)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnMatMulConfig)
    encoded_size = 0
    x.transpose_a != false && (encoded_size += PB._encoded_size(x.transpose_a, 1))
    x.transpose_b != false && (encoded_size += PB._encoded_size(x.transpose_b, 2))
    !isnothing(x.fusions) && (encoded_size += PB._encoded_size(x.fusions, 3))
    !isnothing(x.optimization_config) && (encoded_size += PB._encoded_size(x.optimization_config, 8))
    !isnothing(x.lhs) && (encoded_size += PB._encoded_size(x.lhs, 9))
    !isnothing(x.rhs) && (encoded_size += PB._encoded_size(x.rhs, 10))
    !isnothing(x.result) && (encoded_size += PB._encoded_size(x.result, 11))
    return encoded_size
end

struct OneDnnNormConfig
    rescale::var"OneDnnNormConfig.ScaleAndShift".T
    epsilon_typecast::Int32
    fusions::Union{Nothing,OneDnnFusionConfig}
end
PB.default_values(::Type{OneDnnNormConfig}) = (;rescale = var"OneDnnNormConfig.ScaleAndShift".UNDEFINED, epsilon_typecast = zero(Int32), fusions = nothing)
PB.field_numbers(::Type{OneDnnNormConfig}) = (;rescale = 1, epsilon_typecast = 2, fusions = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnNormConfig}, _endpos::Int=0, _group::Bool=false)
    rescale = var"OneDnnNormConfig.ScaleAndShift".UNDEFINED
    epsilon_typecast = zero(Int32)
    fusions = Ref{Union{Nothing,OneDnnFusionConfig}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            rescale = PB.decode(d, var"OneDnnNormConfig.ScaleAndShift".T)
        elseif field_number == 2
            epsilon_typecast = PB.decode(d, Int32)
        elseif field_number == 3
            PB.decode!(d, fusions)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnNormConfig(rescale, epsilon_typecast, fusions[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnNormConfig)
    initpos = position(e.io)
    x.rescale != var"OneDnnNormConfig.ScaleAndShift".UNDEFINED && PB.encode(e, 1, x.rescale)
    x.epsilon_typecast != zero(Int32) && PB.encode(e, 2, x.epsilon_typecast)
    !isnothing(x.fusions) && PB.encode(e, 3, x.fusions)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnNormConfig)
    encoded_size = 0
    x.rescale != var"OneDnnNormConfig.ScaleAndShift".UNDEFINED && (encoded_size += PB._encoded_size(x.rescale, 1))
    x.epsilon_typecast != zero(Int32) && (encoded_size += PB._encoded_size(x.epsilon_typecast, 2))
    !isnothing(x.fusions) && (encoded_size += PB._encoded_size(x.fusions, 3))
    return encoded_size
end

struct OneDnnConvolutionConfig
    dims::UInt64
    input::Union{Nothing,OneDnnTensorLayoutProto}
    kernel::Union{Nothing,OneDnnTensorLayoutProto}
    output::Union{Nothing,OneDnnTensorLayoutProto}
    window::Union{Nothing,OneDnnWindowProto}
    fusions::Union{Nothing,OneDnnFusionConfig}
    feature_groups::UInt64
    optimization_config::Union{Nothing,OneDnnOptimizationConfig}
end
PB.default_values(::Type{OneDnnConvolutionConfig}) = (;dims = zero(UInt64), input = nothing, kernel = nothing, output = nothing, window = nothing, fusions = nothing, feature_groups = zero(UInt64), optimization_config = nothing)
PB.field_numbers(::Type{OneDnnConvolutionConfig}) = (;dims = 1, input = 2, kernel = 3, output = 4, window = 5, fusions = 6, feature_groups = 7, optimization_config = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OneDnnConvolutionConfig}, _endpos::Int=0, _group::Bool=false)
    dims = zero(UInt64)
    input = Ref{Union{Nothing,OneDnnTensorLayoutProto}}(nothing)
    kernel = Ref{Union{Nothing,OneDnnTensorLayoutProto}}(nothing)
    output = Ref{Union{Nothing,OneDnnTensorLayoutProto}}(nothing)
    window = Ref{Union{Nothing,OneDnnWindowProto}}(nothing)
    fusions = Ref{Union{Nothing,OneDnnFusionConfig}}(nothing)
    feature_groups = zero(UInt64)
    optimization_config = Ref{Union{Nothing,OneDnnOptimizationConfig}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            dims = PB.decode(d, UInt64)
        elseif field_number == 2
            PB.decode!(d, input)
        elseif field_number == 3
            PB.decode!(d, kernel)
        elseif field_number == 4
            PB.decode!(d, output)
        elseif field_number == 5
            PB.decode!(d, window)
        elseif field_number == 6
            PB.decode!(d, fusions)
        elseif field_number == 7
            feature_groups = PB.decode(d, UInt64)
        elseif field_number == 8
            PB.decode!(d, optimization_config)
        else
            Base.skip(d, wire_type)
        end
    end
    return OneDnnConvolutionConfig(dims, input[], kernel[], output[], window[], fusions[], feature_groups, optimization_config[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OneDnnConvolutionConfig)
    initpos = position(e.io)
    x.dims != zero(UInt64) && PB.encode(e, 1, x.dims)
    !isnothing(x.input) && PB.encode(e, 2, x.input)
    !isnothing(x.kernel) && PB.encode(e, 3, x.kernel)
    !isnothing(x.output) && PB.encode(e, 4, x.output)
    !isnothing(x.window) && PB.encode(e, 5, x.window)
    !isnothing(x.fusions) && PB.encode(e, 6, x.fusions)
    x.feature_groups != zero(UInt64) && PB.encode(e, 7, x.feature_groups)
    !isnothing(x.optimization_config) && PB.encode(e, 8, x.optimization_config)
    return position(e.io) - initpos
end
function PB._encoded_size(x::OneDnnConvolutionConfig)
    encoded_size = 0
    x.dims != zero(UInt64) && (encoded_size += PB._encoded_size(x.dims, 1))
    !isnothing(x.input) && (encoded_size += PB._encoded_size(x.input, 2))
    !isnothing(x.kernel) && (encoded_size += PB._encoded_size(x.kernel, 3))
    !isnothing(x.output) && (encoded_size += PB._encoded_size(x.output, 4))
    !isnothing(x.window) && (encoded_size += PB._encoded_size(x.window, 5))
    !isnothing(x.fusions) && (encoded_size += PB._encoded_size(x.fusions, 6))
    x.feature_groups != zero(UInt64) && (encoded_size += PB._encoded_size(x.feature_groups, 7))
    !isnothing(x.optimization_config) && (encoded_size += PB._encoded_size(x.optimization_config, 8))
    return encoded_size
end
