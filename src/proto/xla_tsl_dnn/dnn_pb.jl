import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export ConvolutionKind, ActivationMode, FMHAMaskKind, var"#DataType", DataLayout, NormKind
export ConvolutionMode, var"AlgorithmProto.MathType", FusedMHAKind, FilterLayout
export ConvolutionDescriptorProto, AlgorithmProto, TensorDescriptorProto
export AlgorithmConfigProto


@enumx ConvolutionKind INVALID=0 FORWARD=1 BACKWARD_FILTER=2 BACKWARD_DATA=3 FORWARD_BIAS_ACTIVATION=4 FORWARD_GRAPH=5

@enumx ActivationMode kNone=0 kSigmoid=1 kRelu=2 kRelu6=3 kReluX=4 kTanh=5 kBandPass=6 kElu=7 kLeakyRelu=8 kGeluExact=9

@enumx FMHAMaskKind NO_MASK=0 PADDING=1 CAUSAL=2 PADDING_CAUSAL=3 ALIBI=4

@enumx var"#DataType" kFloat=0 kDouble=1 kHalf=2 kInt8=3 kInt32=4 kComplexFloat=5 kComplexDouble=6 kBF16=7 kF8E5M2=8 kF8E4M3FN=9 kF8E5M2FNUZ=10 kF8E4M3FNUZ=11 kInt64=12 kF8E4M3=13 kF8E3M4=14 kF4E2M1FN=15 kF8E8M0FNU=16

@enumx DataLayout kYXDepthBatch=0 kYXBatchDepth=1 kBatchYXDepth=2 kBatchDepthYX=3 kBatchDepthYX4=4 kBatchDepthYX32=5

@enumx NormKind LAYER_FWD_INFER=0 LAYER_FWD_TRAIN=1 LAYER_BWD=2

@enumx ConvolutionMode CROSS_CORRELATION=0 CONVOLUTION=1

@enumx var"AlgorithmProto.MathType" DEFAULT_MATH=0 TENSOR_OP_MATH=1

@enumx FusedMHAKind BMM1_OUTPUT_UNKNOWN=0 BMM1_OUTPUT_INPUT_TYPE=1 BMM1_OUTPUT_FLOAT=2

@enumx FilterLayout kOutputInputYX=0 kOutputYXInput=1 kOutputInputYX4=2 kOutputInputYX32=5 kOutputInputYX32_CudnnReordered=6 kInputYXOutput=3 kYXInputOutput=4

struct ConvolutionDescriptorProto
    paddings::Vector{Int64}
    strides::Vector{Int64}
    dilations::Vector{Int64}
    compute_mode::var"#DataType".T
    group_count::Int32
    convolution_mode::ConvolutionMode.T
    name::String
end
ConvolutionDescriptorProto(;paddings = Vector{Int64}(), strides = Vector{Int64}(), dilations = Vector{Int64}(), compute_mode = var"#DataType".kFloat, group_count = zero(Int32), convolution_mode = ConvolutionMode.CROSS_CORRELATION, name = "") = ConvolutionDescriptorProto(paddings, strides, dilations, compute_mode, group_count, convolution_mode, name)
PB.default_values(::Type{ConvolutionDescriptorProto}) = (;paddings = Vector{Int64}(), strides = Vector{Int64}(), dilations = Vector{Int64}(), compute_mode = var"#DataType".kFloat, group_count = zero(Int32), convolution_mode = ConvolutionMode.CROSS_CORRELATION, name = "")
PB.field_numbers(::Type{ConvolutionDescriptorProto}) = (;paddings = 1, strides = 2, dilations = 3, compute_mode = 4, group_count = 5, convolution_mode = 6, name = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ConvolutionDescriptorProto})
    paddings = PB.BufferedVector{Int64}()
    strides = PB.BufferedVector{Int64}()
    dilations = PB.BufferedVector{Int64}()
    compute_mode = var"#DataType".kFloat
    group_count = zero(Int32)
    convolution_mode = ConvolutionMode.CROSS_CORRELATION
    name = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, paddings)
        elseif field_number == 2
            PB.decode!(d, wire_type, strides)
        elseif field_number == 3
            PB.decode!(d, wire_type, dilations)
        elseif field_number == 4
            compute_mode = PB.decode(d, var"#DataType".T)
        elseif field_number == 5
            group_count = PB.decode(d, Int32)
        elseif field_number == 6
            convolution_mode = PB.decode(d, ConvolutionMode.T)
        elseif field_number == 7
            name = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return ConvolutionDescriptorProto(paddings[], strides[], dilations[], compute_mode, group_count, convolution_mode, name)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ConvolutionDescriptorProto)
    initpos = position(e.io)
    !isempty(x.paddings) && PB.encode(e, 1, x.paddings)
    !isempty(x.strides) && PB.encode(e, 2, x.strides)
    !isempty(x.dilations) && PB.encode(e, 3, x.dilations)
    x.compute_mode != var"#DataType".kFloat && PB.encode(e, 4, x.compute_mode)
    x.group_count != zero(Int32) && PB.encode(e, 5, x.group_count)
    x.convolution_mode != ConvolutionMode.CROSS_CORRELATION && PB.encode(e, 6, x.convolution_mode)
    !isempty(x.name) && PB.encode(e, 7, x.name)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ConvolutionDescriptorProto)
    encoded_size = 0
    !isempty(x.paddings) && (encoded_size += PB._encoded_size(x.paddings, 1))
    !isempty(x.strides) && (encoded_size += PB._encoded_size(x.strides, 2))
    !isempty(x.dilations) && (encoded_size += PB._encoded_size(x.dilations, 3))
    x.compute_mode != var"#DataType".kFloat && (encoded_size += PB._encoded_size(x.compute_mode, 4))
    x.group_count != zero(Int32) && (encoded_size += PB._encoded_size(x.group_count, 5))
    x.convolution_mode != ConvolutionMode.CROSS_CORRELATION && (encoded_size += PB._encoded_size(x.convolution_mode, 6))
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 7))
    return encoded_size
end

struct AlgorithmProto
    algo_id::Int64
    math_type::var"AlgorithmProto.MathType".T
    tuning_knobs::Dict{Int64,Int64}
    is_cudnn_frontend::Bool
    workspace_size::Union{Nothing,google.protobuf.UInt64Value}
end
AlgorithmProto(;algo_id = zero(Int64), math_type = var"AlgorithmProto.MathType".DEFAULT_MATH, tuning_knobs = Dict{Int64,Int64}(), is_cudnn_frontend = false, workspace_size = nothing) = AlgorithmProto(algo_id, math_type, tuning_knobs, is_cudnn_frontend, workspace_size)
PB.reserved_fields(::Type{AlgorithmProto}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[3])
PB.default_values(::Type{AlgorithmProto}) = (;algo_id = zero(Int64), math_type = var"AlgorithmProto.MathType".DEFAULT_MATH, tuning_knobs = Dict{Int64,Int64}(), is_cudnn_frontend = false, workspace_size = nothing)
PB.field_numbers(::Type{AlgorithmProto}) = (;algo_id = 1, math_type = 2, tuning_knobs = 4, is_cudnn_frontend = 5, workspace_size = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AlgorithmProto})
    algo_id = zero(Int64)
    math_type = var"AlgorithmProto.MathType".DEFAULT_MATH
    tuning_knobs = Dict{Int64,Int64}()
    is_cudnn_frontend = false
    workspace_size = Ref{Union{Nothing,google.protobuf.UInt64Value}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            algo_id = PB.decode(d, Int64)
        elseif field_number == 2
            math_type = PB.decode(d, var"AlgorithmProto.MathType".T)
        elseif field_number == 4
            PB.decode!(d, tuning_knobs)
        elseif field_number == 5
            is_cudnn_frontend = PB.decode(d, Bool)
        elseif field_number == 6
            PB.decode!(d, workspace_size)
        else
            Base.skip(d, wire_type)
        end
    end
    return AlgorithmProto(algo_id, math_type, tuning_knobs, is_cudnn_frontend, workspace_size[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AlgorithmProto)
    initpos = position(e.io)
    x.algo_id != zero(Int64) && PB.encode(e, 1, x.algo_id)
    x.math_type != var"AlgorithmProto.MathType".DEFAULT_MATH && PB.encode(e, 2, x.math_type)
    !isempty(x.tuning_knobs) && PB.encode(e, 4, x.tuning_knobs)
    x.is_cudnn_frontend != false && PB.encode(e, 5, x.is_cudnn_frontend)
    !isnothing(x.workspace_size) && PB.encode(e, 6, x.workspace_size)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AlgorithmProto)
    encoded_size = 0
    x.algo_id != zero(Int64) && (encoded_size += PB._encoded_size(x.algo_id, 1))
    x.math_type != var"AlgorithmProto.MathType".DEFAULT_MATH && (encoded_size += PB._encoded_size(x.math_type, 2))
    !isempty(x.tuning_knobs) && (encoded_size += PB._encoded_size(x.tuning_knobs, 4))
    x.is_cudnn_frontend != false && (encoded_size += PB._encoded_size(x.is_cudnn_frontend, 5))
    !isnothing(x.workspace_size) && (encoded_size += PB._encoded_size(x.workspace_size, 6))
    return encoded_size
end

struct TensorDescriptorProto
    dimensions::Vector{Int64}
    data_type::var"#DataType".T
    layout_oneof::Union{Nothing,OneOf{<:Union{DataLayout.T,FilterLayout.T}}}
end
TensorDescriptorProto(;dimensions = Vector{Int64}(), data_type = var"#DataType".kFloat, layout_oneof = nothing) = TensorDescriptorProto(dimensions, data_type, layout_oneof)
PB.oneof_field_types(::Type{TensorDescriptorProto}) = (;
    layout_oneof = (;data_layout=DataLayout.T, filter_layout=FilterLayout.T),
)
PB.default_values(::Type{TensorDescriptorProto}) = (;dimensions = Vector{Int64}(), data_type = var"#DataType".kFloat, data_layout = DataLayout.kYXDepthBatch, filter_layout = FilterLayout.kOutputInputYX)
PB.field_numbers(::Type{TensorDescriptorProto}) = (;dimensions = 1, data_type = 2, data_layout = 3, filter_layout = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TensorDescriptorProto})
    dimensions = PB.BufferedVector{Int64}()
    data_type = var"#DataType".kFloat
    layout_oneof = nothing
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, dimensions)
        elseif field_number == 2
            data_type = PB.decode(d, var"#DataType".T)
        elseif field_number == 3
            layout_oneof = OneOf(:data_layout, PB.decode(d, DataLayout.T))
        elseif field_number == 4
            layout_oneof = OneOf(:filter_layout, PB.decode(d, FilterLayout.T))
        else
            Base.skip(d, wire_type)
        end
    end
    return TensorDescriptorProto(dimensions[], data_type, layout_oneof)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TensorDescriptorProto)
    initpos = position(e.io)
    !isempty(x.dimensions) && PB.encode(e, 1, x.dimensions)
    x.data_type != var"#DataType".kFloat && PB.encode(e, 2, x.data_type)
    if isnothing(x.layout_oneof);
    elseif x.layout_oneof.name === :data_layout
        PB.encode(e, 3, x.layout_oneof[]::DataLayout.T)
    elseif x.layout_oneof.name === :filter_layout
        PB.encode(e, 4, x.layout_oneof[]::FilterLayout.T)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::TensorDescriptorProto)
    encoded_size = 0
    !isempty(x.dimensions) && (encoded_size += PB._encoded_size(x.dimensions, 1))
    x.data_type != var"#DataType".kFloat && (encoded_size += PB._encoded_size(x.data_type, 2))
    if isnothing(x.layout_oneof);
    elseif x.layout_oneof.name === :data_layout
        encoded_size += PB._encoded_size(x.layout_oneof[]::DataLayout.T, 3)
    elseif x.layout_oneof.name === :filter_layout
        encoded_size += PB._encoded_size(x.layout_oneof[]::FilterLayout.T, 4)
    end
    return encoded_size
end

struct AlgorithmConfigProto
    optional_algorithm::Union{Nothing,OneOf{AlgorithmProto}}
    optional_algorithm_no_scratch::Union{Nothing,OneOf{AlgorithmProto}}
    optional_scratch_size::Union{Nothing,OneOf{Int64}}
end
AlgorithmConfigProto(;optional_algorithm = nothing, optional_algorithm_no_scratch = nothing, optional_scratch_size = nothing) = AlgorithmConfigProto(optional_algorithm, optional_algorithm_no_scratch, optional_scratch_size)
PB.oneof_field_types(::Type{AlgorithmConfigProto}) = (;
    optional_algorithm = (;algorithm=AlgorithmProto),
    optional_algorithm_no_scratch = (;algorithm_no_scratch=AlgorithmProto),
    optional_scratch_size = (;scratch_size=Int64),
)
PB.default_values(::Type{AlgorithmConfigProto}) = (;algorithm = nothing, algorithm_no_scratch = nothing, scratch_size = zero(Int64))
PB.field_numbers(::Type{AlgorithmConfigProto}) = (;algorithm = 1, algorithm_no_scratch = 2, scratch_size = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AlgorithmConfigProto})
    optional_algorithm = nothing
    optional_algorithm_no_scratch = nothing
    optional_scratch_size = nothing
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            optional_algorithm = OneOf(:algorithm, PB.decode(d, Ref{AlgorithmProto}))
        elseif field_number == 2
            optional_algorithm_no_scratch = OneOf(:algorithm_no_scratch, PB.decode(d, Ref{AlgorithmProto}))
        elseif field_number == 3
            optional_scratch_size = OneOf(:scratch_size, PB.decode(d, Int64))
        else
            Base.skip(d, wire_type)
        end
    end
    return AlgorithmConfigProto(optional_algorithm, optional_algorithm_no_scratch, optional_scratch_size)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AlgorithmConfigProto)
    initpos = position(e.io)
    if isnothing(x.optional_algorithm);
    elseif x.optional_algorithm.name === :algorithm
        PB.encode(e, 1, x.optional_algorithm[]::AlgorithmProto)
    end
    if isnothing(x.optional_algorithm_no_scratch);
    elseif x.optional_algorithm_no_scratch.name === :algorithm_no_scratch
        PB.encode(e, 2, x.optional_algorithm_no_scratch[]::AlgorithmProto)
    end
    if isnothing(x.optional_scratch_size);
    elseif x.optional_scratch_size.name === :scratch_size
        PB.encode(e, 3, x.optional_scratch_size[]::Int64)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::AlgorithmConfigProto)
    encoded_size = 0
    if isnothing(x.optional_algorithm);
    elseif x.optional_algorithm.name === :algorithm
        encoded_size += PB._encoded_size(x.optional_algorithm[]::AlgorithmProto, 1)
    end
    if isnothing(x.optional_algorithm_no_scratch);
    elseif x.optional_algorithm_no_scratch.name === :algorithm_no_scratch
        encoded_size += PB._encoded_size(x.optional_algorithm_no_scratch[]::AlgorithmProto, 2)
    end
    if isnothing(x.optional_scratch_size);
    elseif x.optional_scratch_size.name === :scratch_size
        encoded_size += PB._encoded_size(x.optional_scratch_size[]::Int64, 3)
    end
    return encoded_size
end
