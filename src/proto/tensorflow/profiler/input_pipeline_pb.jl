import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export InputTimeBreakdown, PerGenericStepDetails, BottleneckAnalysis
export InputPipelineAnalysisRecommendation, InputOpDetails, StepSummary
export GenericStepTimeBreakdown, InputPipelineAnalysisResult


struct InputTimeBreakdown
    demanded_file_read_us::Float64
    advanced_file_read_us::Float64
    preprocessing_us::Float64
    enqueue_us::Float64
    unclassified_non_enqueue_us::Float64
end
InputTimeBreakdown(;demanded_file_read_us = zero(Float64), advanced_file_read_us = zero(Float64), preprocessing_us = zero(Float64), enqueue_us = zero(Float64), unclassified_non_enqueue_us = zero(Float64)) = InputTimeBreakdown(demanded_file_read_us, advanced_file_read_us, preprocessing_us, enqueue_us, unclassified_non_enqueue_us)
PB.default_values(::Type{InputTimeBreakdown}) = (;demanded_file_read_us = zero(Float64), advanced_file_read_us = zero(Float64), preprocessing_us = zero(Float64), enqueue_us = zero(Float64), unclassified_non_enqueue_us = zero(Float64))
PB.field_numbers(::Type{InputTimeBreakdown}) = (;demanded_file_read_us = 1, advanced_file_read_us = 2, preprocessing_us = 3, enqueue_us = 4, unclassified_non_enqueue_us = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:InputTimeBreakdown})
    demanded_file_read_us = zero(Float64)
    advanced_file_read_us = zero(Float64)
    preprocessing_us = zero(Float64)
    enqueue_us = zero(Float64)
    unclassified_non_enqueue_us = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            demanded_file_read_us = PB.decode(d, Float64)
        elseif field_number == 2
            advanced_file_read_us = PB.decode(d, Float64)
        elseif field_number == 3
            preprocessing_us = PB.decode(d, Float64)
        elseif field_number == 4
            enqueue_us = PB.decode(d, Float64)
        elseif field_number == 5
            unclassified_non_enqueue_us = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return InputTimeBreakdown(demanded_file_read_us, advanced_file_read_us, preprocessing_us, enqueue_us, unclassified_non_enqueue_us)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::InputTimeBreakdown)
    initpos = position(e.io)
    x.demanded_file_read_us !== zero(Float64) && PB.encode(e, 1, x.demanded_file_read_us)
    x.advanced_file_read_us !== zero(Float64) && PB.encode(e, 2, x.advanced_file_read_us)
    x.preprocessing_us !== zero(Float64) && PB.encode(e, 3, x.preprocessing_us)
    x.enqueue_us !== zero(Float64) && PB.encode(e, 4, x.enqueue_us)
    x.unclassified_non_enqueue_us !== zero(Float64) && PB.encode(e, 5, x.unclassified_non_enqueue_us)
    return position(e.io) - initpos
end
function PB._encoded_size(x::InputTimeBreakdown)
    encoded_size = 0
    x.demanded_file_read_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.demanded_file_read_us, 1))
    x.advanced_file_read_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.advanced_file_read_us, 2))
    x.preprocessing_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.preprocessing_us, 3))
    x.enqueue_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.enqueue_us, 4))
    x.unclassified_non_enqueue_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.unclassified_non_enqueue_us, 5))
    return encoded_size
end

struct PerGenericStepDetails
    step_number::Int32
    step_name::String
    step_time_ms::Float64
    unknown_time_ms::Float64
    host_wait_input_ms::Float64
    host_to_device_ms::Float64
    output_ms::Float64
    device_compute_ms::Float64
    device_to_device_ms::Float64
    device_collectives_ms::Float64
    host_compute_ms::Float64
    host_prepare_ms::Float64
    host_compile_ms::Float64
end
PerGenericStepDetails(;step_number = zero(Int32), step_name = "", step_time_ms = zero(Float64), unknown_time_ms = zero(Float64), host_wait_input_ms = zero(Float64), host_to_device_ms = zero(Float64), output_ms = zero(Float64), device_compute_ms = zero(Float64), device_to_device_ms = zero(Float64), device_collectives_ms = zero(Float64), host_compute_ms = zero(Float64), host_prepare_ms = zero(Float64), host_compile_ms = zero(Float64)) = PerGenericStepDetails(step_number, step_name, step_time_ms, unknown_time_ms, host_wait_input_ms, host_to_device_ms, output_ms, device_compute_ms, device_to_device_ms, device_collectives_ms, host_compute_ms, host_prepare_ms, host_compile_ms)
PB.reserved_fields(::Type{PerGenericStepDetails}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[4])
PB.default_values(::Type{PerGenericStepDetails}) = (;step_number = zero(Int32), step_name = "", step_time_ms = zero(Float64), unknown_time_ms = zero(Float64), host_wait_input_ms = zero(Float64), host_to_device_ms = zero(Float64), output_ms = zero(Float64), device_compute_ms = zero(Float64), device_to_device_ms = zero(Float64), device_collectives_ms = zero(Float64), host_compute_ms = zero(Float64), host_prepare_ms = zero(Float64), host_compile_ms = zero(Float64))
PB.field_numbers(::Type{PerGenericStepDetails}) = (;step_number = 1, step_name = 14, step_time_ms = 2, unknown_time_ms = 3, host_wait_input_ms = 11, host_to_device_ms = 12, output_ms = 5, device_compute_ms = 6, device_to_device_ms = 7, device_collectives_ms = 13, host_compute_ms = 8, host_prepare_ms = 9, host_compile_ms = 10)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PerGenericStepDetails})
    step_number = zero(Int32)
    step_name = ""
    step_time_ms = zero(Float64)
    unknown_time_ms = zero(Float64)
    host_wait_input_ms = zero(Float64)
    host_to_device_ms = zero(Float64)
    output_ms = zero(Float64)
    device_compute_ms = zero(Float64)
    device_to_device_ms = zero(Float64)
    device_collectives_ms = zero(Float64)
    host_compute_ms = zero(Float64)
    host_prepare_ms = zero(Float64)
    host_compile_ms = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            step_number = PB.decode(d, Int32)
        elseif field_number == 14
            step_name = PB.decode(d, String)
        elseif field_number == 2
            step_time_ms = PB.decode(d, Float64)
        elseif field_number == 3
            unknown_time_ms = PB.decode(d, Float64)
        elseif field_number == 11
            host_wait_input_ms = PB.decode(d, Float64)
        elseif field_number == 12
            host_to_device_ms = PB.decode(d, Float64)
        elseif field_number == 5
            output_ms = PB.decode(d, Float64)
        elseif field_number == 6
            device_compute_ms = PB.decode(d, Float64)
        elseif field_number == 7
            device_to_device_ms = PB.decode(d, Float64)
        elseif field_number == 13
            device_collectives_ms = PB.decode(d, Float64)
        elseif field_number == 8
            host_compute_ms = PB.decode(d, Float64)
        elseif field_number == 9
            host_prepare_ms = PB.decode(d, Float64)
        elseif field_number == 10
            host_compile_ms = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return PerGenericStepDetails(step_number, step_name, step_time_ms, unknown_time_ms, host_wait_input_ms, host_to_device_ms, output_ms, device_compute_ms, device_to_device_ms, device_collectives_ms, host_compute_ms, host_prepare_ms, host_compile_ms)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PerGenericStepDetails)
    initpos = position(e.io)
    x.step_number != zero(Int32) && PB.encode(e, 1, x.step_number)
    !isempty(x.step_name) && PB.encode(e, 14, x.step_name)
    x.step_time_ms !== zero(Float64) && PB.encode(e, 2, x.step_time_ms)
    x.unknown_time_ms !== zero(Float64) && PB.encode(e, 3, x.unknown_time_ms)
    x.host_wait_input_ms !== zero(Float64) && PB.encode(e, 11, x.host_wait_input_ms)
    x.host_to_device_ms !== zero(Float64) && PB.encode(e, 12, x.host_to_device_ms)
    x.output_ms !== zero(Float64) && PB.encode(e, 5, x.output_ms)
    x.device_compute_ms !== zero(Float64) && PB.encode(e, 6, x.device_compute_ms)
    x.device_to_device_ms !== zero(Float64) && PB.encode(e, 7, x.device_to_device_ms)
    x.device_collectives_ms !== zero(Float64) && PB.encode(e, 13, x.device_collectives_ms)
    x.host_compute_ms !== zero(Float64) && PB.encode(e, 8, x.host_compute_ms)
    x.host_prepare_ms !== zero(Float64) && PB.encode(e, 9, x.host_prepare_ms)
    x.host_compile_ms !== zero(Float64) && PB.encode(e, 10, x.host_compile_ms)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PerGenericStepDetails)
    encoded_size = 0
    x.step_number != zero(Int32) && (encoded_size += PB._encoded_size(x.step_number, 1))
    !isempty(x.step_name) && (encoded_size += PB._encoded_size(x.step_name, 14))
    x.step_time_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.step_time_ms, 2))
    x.unknown_time_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.unknown_time_ms, 3))
    x.host_wait_input_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_wait_input_ms, 11))
    x.host_to_device_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_to_device_ms, 12))
    x.output_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.output_ms, 5))
    x.device_compute_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_compute_ms, 6))
    x.device_to_device_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_to_device_ms, 7))
    x.device_collectives_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.device_collectives_ms, 13))
    x.host_compute_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_compute_ms, 8))
    x.host_prepare_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_prepare_ms, 9))
    x.host_compile_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.host_compile_ms, 10))
    return encoded_size
end

struct BottleneckAnalysis
    input_percent::Float64
    output_percent::Float64
    idle_percent::Float64
    compute_percent::Float64
    input_classification::String
    input_statement::String
    kernel_launch_classification::String
    kernel_launch_statement::String
    all_other_classification::String
    all_other_statement::String
    device_collectives_classification::String
    device_collectives_statement::String
end
BottleneckAnalysis(;input_percent = zero(Float64), output_percent = zero(Float64), idle_percent = zero(Float64), compute_percent = zero(Float64), input_classification = "", input_statement = "", kernel_launch_classification = "", kernel_launch_statement = "", all_other_classification = "", all_other_statement = "", device_collectives_classification = "", device_collectives_statement = "") = BottleneckAnalysis(input_percent, output_percent, idle_percent, compute_percent, input_classification, input_statement, kernel_launch_classification, kernel_launch_statement, all_other_classification, all_other_statement, device_collectives_classification, device_collectives_statement)
PB.default_values(::Type{BottleneckAnalysis}) = (;input_percent = zero(Float64), output_percent = zero(Float64), idle_percent = zero(Float64), compute_percent = zero(Float64), input_classification = "", input_statement = "", kernel_launch_classification = "", kernel_launch_statement = "", all_other_classification = "", all_other_statement = "", device_collectives_classification = "", device_collectives_statement = "")
PB.field_numbers(::Type{BottleneckAnalysis}) = (;input_percent = 7, output_percent = 8, idle_percent = 9, compute_percent = 10, input_classification = 1, input_statement = 2, kernel_launch_classification = 3, kernel_launch_statement = 4, all_other_classification = 5, all_other_statement = 6, device_collectives_classification = 11, device_collectives_statement = 12)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:BottleneckAnalysis})
    input_percent = zero(Float64)
    output_percent = zero(Float64)
    idle_percent = zero(Float64)
    compute_percent = zero(Float64)
    input_classification = ""
    input_statement = ""
    kernel_launch_classification = ""
    kernel_launch_statement = ""
    all_other_classification = ""
    all_other_statement = ""
    device_collectives_classification = ""
    device_collectives_statement = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 7
            input_percent = PB.decode(d, Float64)
        elseif field_number == 8
            output_percent = PB.decode(d, Float64)
        elseif field_number == 9
            idle_percent = PB.decode(d, Float64)
        elseif field_number == 10
            compute_percent = PB.decode(d, Float64)
        elseif field_number == 1
            input_classification = PB.decode(d, String)
        elseif field_number == 2
            input_statement = PB.decode(d, String)
        elseif field_number == 3
            kernel_launch_classification = PB.decode(d, String)
        elseif field_number == 4
            kernel_launch_statement = PB.decode(d, String)
        elseif field_number == 5
            all_other_classification = PB.decode(d, String)
        elseif field_number == 6
            all_other_statement = PB.decode(d, String)
        elseif field_number == 11
            device_collectives_classification = PB.decode(d, String)
        elseif field_number == 12
            device_collectives_statement = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return BottleneckAnalysis(input_percent, output_percent, idle_percent, compute_percent, input_classification, input_statement, kernel_launch_classification, kernel_launch_statement, all_other_classification, all_other_statement, device_collectives_classification, device_collectives_statement)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::BottleneckAnalysis)
    initpos = position(e.io)
    x.input_percent !== zero(Float64) && PB.encode(e, 7, x.input_percent)
    x.output_percent !== zero(Float64) && PB.encode(e, 8, x.output_percent)
    x.idle_percent !== zero(Float64) && PB.encode(e, 9, x.idle_percent)
    x.compute_percent !== zero(Float64) && PB.encode(e, 10, x.compute_percent)
    !isempty(x.input_classification) && PB.encode(e, 1, x.input_classification)
    !isempty(x.input_statement) && PB.encode(e, 2, x.input_statement)
    !isempty(x.kernel_launch_classification) && PB.encode(e, 3, x.kernel_launch_classification)
    !isempty(x.kernel_launch_statement) && PB.encode(e, 4, x.kernel_launch_statement)
    !isempty(x.all_other_classification) && PB.encode(e, 5, x.all_other_classification)
    !isempty(x.all_other_statement) && PB.encode(e, 6, x.all_other_statement)
    !isempty(x.device_collectives_classification) && PB.encode(e, 11, x.device_collectives_classification)
    !isempty(x.device_collectives_statement) && PB.encode(e, 12, x.device_collectives_statement)
    return position(e.io) - initpos
end
function PB._encoded_size(x::BottleneckAnalysis)
    encoded_size = 0
    x.input_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.input_percent, 7))
    x.output_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.output_percent, 8))
    x.idle_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.idle_percent, 9))
    x.compute_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.compute_percent, 10))
    !isempty(x.input_classification) && (encoded_size += PB._encoded_size(x.input_classification, 1))
    !isempty(x.input_statement) && (encoded_size += PB._encoded_size(x.input_statement, 2))
    !isempty(x.kernel_launch_classification) && (encoded_size += PB._encoded_size(x.kernel_launch_classification, 3))
    !isempty(x.kernel_launch_statement) && (encoded_size += PB._encoded_size(x.kernel_launch_statement, 4))
    !isempty(x.all_other_classification) && (encoded_size += PB._encoded_size(x.all_other_classification, 5))
    !isempty(x.all_other_statement) && (encoded_size += PB._encoded_size(x.all_other_statement, 6))
    !isempty(x.device_collectives_classification) && (encoded_size += PB._encoded_size(x.device_collectives_classification, 11))
    !isempty(x.device_collectives_statement) && (encoded_size += PB._encoded_size(x.device_collectives_statement, 12))
    return encoded_size
end

struct InputPipelineAnalysisRecommendation
    details::Vector{String}
    bottleneck_analysis::Union{Nothing,google.protobuf.var"#Any"}
    summary_next_step::String
end
InputPipelineAnalysisRecommendation(;details = Vector{String}(), bottleneck_analysis = nothing, summary_next_step = "") = InputPipelineAnalysisRecommendation(details, bottleneck_analysis, summary_next_step)
PB.default_values(::Type{InputPipelineAnalysisRecommendation}) = (;details = Vector{String}(), bottleneck_analysis = nothing, summary_next_step = "")
PB.field_numbers(::Type{InputPipelineAnalysisRecommendation}) = (;details = 1, bottleneck_analysis = 2, summary_next_step = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:InputPipelineAnalysisRecommendation})
    details = PB.BufferedVector{String}()
    bottleneck_analysis = Ref{Union{Nothing,google.protobuf.var"#Any"}}(nothing)
    summary_next_step = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, details)
        elseif field_number == 2
            PB.decode!(d, bottleneck_analysis)
        elseif field_number == 3
            summary_next_step = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return InputPipelineAnalysisRecommendation(details[], bottleneck_analysis[], summary_next_step)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::InputPipelineAnalysisRecommendation)
    initpos = position(e.io)
    !isempty(x.details) && PB.encode(e, 1, x.details)
    !isnothing(x.bottleneck_analysis) && PB.encode(e, 2, x.bottleneck_analysis)
    !isempty(x.summary_next_step) && PB.encode(e, 3, x.summary_next_step)
    return position(e.io) - initpos
end
function PB._encoded_size(x::InputPipelineAnalysisRecommendation)
    encoded_size = 0
    !isempty(x.details) && (encoded_size += PB._encoded_size(x.details, 1))
    !isnothing(x.bottleneck_analysis) && (encoded_size += PB._encoded_size(x.bottleneck_analysis, 2))
    !isempty(x.summary_next_step) && (encoded_size += PB._encoded_size(x.summary_next_step, 3))
    return encoded_size
end

struct InputOpDetails
    op_name::String
    count::UInt64
    time_in_ms::Float64
    time_in_percent::Float64
    self_time_in_ms::Float64
    self_time_in_percent::Float64
    category::String
end
InputOpDetails(;op_name = "", count = zero(UInt64), time_in_ms = zero(Float64), time_in_percent = zero(Float64), self_time_in_ms = zero(Float64), self_time_in_percent = zero(Float64), category = "") = InputOpDetails(op_name, count, time_in_ms, time_in_percent, self_time_in_ms, self_time_in_percent, category)
PB.default_values(::Type{InputOpDetails}) = (;op_name = "", count = zero(UInt64), time_in_ms = zero(Float64), time_in_percent = zero(Float64), self_time_in_ms = zero(Float64), self_time_in_percent = zero(Float64), category = "")
PB.field_numbers(::Type{InputOpDetails}) = (;op_name = 1, count = 2, time_in_ms = 3, time_in_percent = 4, self_time_in_ms = 5, self_time_in_percent = 6, category = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:InputOpDetails})
    op_name = ""
    count = zero(UInt64)
    time_in_ms = zero(Float64)
    time_in_percent = zero(Float64)
    self_time_in_ms = zero(Float64)
    self_time_in_percent = zero(Float64)
    category = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            op_name = PB.decode(d, String)
        elseif field_number == 2
            count = PB.decode(d, UInt64)
        elseif field_number == 3
            time_in_ms = PB.decode(d, Float64)
        elseif field_number == 4
            time_in_percent = PB.decode(d, Float64)
        elseif field_number == 5
            self_time_in_ms = PB.decode(d, Float64)
        elseif field_number == 6
            self_time_in_percent = PB.decode(d, Float64)
        elseif field_number == 7
            category = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return InputOpDetails(op_name, count, time_in_ms, time_in_percent, self_time_in_ms, self_time_in_percent, category)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::InputOpDetails)
    initpos = position(e.io)
    !isempty(x.op_name) && PB.encode(e, 1, x.op_name)
    x.count != zero(UInt64) && PB.encode(e, 2, x.count)
    x.time_in_ms !== zero(Float64) && PB.encode(e, 3, x.time_in_ms)
    x.time_in_percent !== zero(Float64) && PB.encode(e, 4, x.time_in_percent)
    x.self_time_in_ms !== zero(Float64) && PB.encode(e, 5, x.self_time_in_ms)
    x.self_time_in_percent !== zero(Float64) && PB.encode(e, 6, x.self_time_in_percent)
    !isempty(x.category) && PB.encode(e, 7, x.category)
    return position(e.io) - initpos
end
function PB._encoded_size(x::InputOpDetails)
    encoded_size = 0
    !isempty(x.op_name) && (encoded_size += PB._encoded_size(x.op_name, 1))
    x.count != zero(UInt64) && (encoded_size += PB._encoded_size(x.count, 2))
    x.time_in_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.time_in_ms, 3))
    x.time_in_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.time_in_percent, 4))
    x.self_time_in_ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.self_time_in_ms, 5))
    x.self_time_in_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.self_time_in_percent, 6))
    !isempty(x.category) && (encoded_size += PB._encoded_size(x.category, 7))
    return encoded_size
end

struct StepSummary
    average::Float64
    standard_deviation::Float64
    minimum::Float64
    maximum::Float64
end
StepSummary(;average = zero(Float64), standard_deviation = zero(Float64), minimum = zero(Float64), maximum = zero(Float64)) = StepSummary(average, standard_deviation, minimum, maximum)
PB.default_values(::Type{StepSummary}) = (;average = zero(Float64), standard_deviation = zero(Float64), minimum = zero(Float64), maximum = zero(Float64))
PB.field_numbers(::Type{StepSummary}) = (;average = 1, standard_deviation = 2, minimum = 3, maximum = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:StepSummary})
    average = zero(Float64)
    standard_deviation = zero(Float64)
    minimum = zero(Float64)
    maximum = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            average = PB.decode(d, Float64)
        elseif field_number == 2
            standard_deviation = PB.decode(d, Float64)
        elseif field_number == 3
            minimum = PB.decode(d, Float64)
        elseif field_number == 4
            maximum = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return StepSummary(average, standard_deviation, minimum, maximum)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::StepSummary)
    initpos = position(e.io)
    x.average !== zero(Float64) && PB.encode(e, 1, x.average)
    x.standard_deviation !== zero(Float64) && PB.encode(e, 2, x.standard_deviation)
    x.minimum !== zero(Float64) && PB.encode(e, 3, x.minimum)
    x.maximum !== zero(Float64) && PB.encode(e, 4, x.maximum)
    return position(e.io) - initpos
end
function PB._encoded_size(x::StepSummary)
    encoded_size = 0
    x.average !== zero(Float64) && (encoded_size += PB._encoded_size(x.average, 1))
    x.standard_deviation !== zero(Float64) && (encoded_size += PB._encoded_size(x.standard_deviation, 2))
    x.minimum !== zero(Float64) && (encoded_size += PB._encoded_size(x.minimum, 3))
    x.maximum !== zero(Float64) && (encoded_size += PB._encoded_size(x.maximum, 4))
    return encoded_size
end

struct GenericStepTimeBreakdown
    unknown_time_ms_summary::Union{Nothing,StepSummary}
    host_wait_input_ms_summary::Union{Nothing,StepSummary}
    host_to_device_ms_summary::Union{Nothing,StepSummary}
    input_ms_summary::Union{Nothing,StepSummary}
    output_ms_summary::Union{Nothing,StepSummary}
    device_compute_ms_summary::Union{Nothing,StepSummary}
    device_to_device_ms_summary::Union{Nothing,StepSummary}
    device_collectives_ms_summary::Union{Nothing,StepSummary}
    host_compute_ms_summary::Union{Nothing,StepSummary}
    host_prepare_ms_summary::Union{Nothing,StepSummary}
    host_compile_ms_summary::Union{Nothing,StepSummary}
end
GenericStepTimeBreakdown(;unknown_time_ms_summary = nothing, host_wait_input_ms_summary = nothing, host_to_device_ms_summary = nothing, input_ms_summary = nothing, output_ms_summary = nothing, device_compute_ms_summary = nothing, device_to_device_ms_summary = nothing, device_collectives_ms_summary = nothing, host_compute_ms_summary = nothing, host_prepare_ms_summary = nothing, host_compile_ms_summary = nothing) = GenericStepTimeBreakdown(unknown_time_ms_summary, host_wait_input_ms_summary, host_to_device_ms_summary, input_ms_summary, output_ms_summary, device_compute_ms_summary, device_to_device_ms_summary, device_collectives_ms_summary, host_compute_ms_summary, host_prepare_ms_summary, host_compile_ms_summary)
PB.reserved_fields(::Type{GenericStepTimeBreakdown}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[2])
PB.default_values(::Type{GenericStepTimeBreakdown}) = (;unknown_time_ms_summary = nothing, host_wait_input_ms_summary = nothing, host_to_device_ms_summary = nothing, input_ms_summary = nothing, output_ms_summary = nothing, device_compute_ms_summary = nothing, device_to_device_ms_summary = nothing, device_collectives_ms_summary = nothing, host_compute_ms_summary = nothing, host_prepare_ms_summary = nothing, host_compile_ms_summary = nothing)
PB.field_numbers(::Type{GenericStepTimeBreakdown}) = (;unknown_time_ms_summary = 1, host_wait_input_ms_summary = 9, host_to_device_ms_summary = 10, input_ms_summary = 11, output_ms_summary = 3, device_compute_ms_summary = 4, device_to_device_ms_summary = 5, device_collectives_ms_summary = 12, host_compute_ms_summary = 6, host_prepare_ms_summary = 7, host_compile_ms_summary = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GenericStepTimeBreakdown})
    unknown_time_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    host_wait_input_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    host_to_device_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    input_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    output_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    device_compute_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    device_to_device_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    device_collectives_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    host_compute_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    host_prepare_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    host_compile_ms_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, unknown_time_ms_summary)
        elseif field_number == 9
            PB.decode!(d, host_wait_input_ms_summary)
        elseif field_number == 10
            PB.decode!(d, host_to_device_ms_summary)
        elseif field_number == 11
            PB.decode!(d, input_ms_summary)
        elseif field_number == 3
            PB.decode!(d, output_ms_summary)
        elseif field_number == 4
            PB.decode!(d, device_compute_ms_summary)
        elseif field_number == 5
            PB.decode!(d, device_to_device_ms_summary)
        elseif field_number == 12
            PB.decode!(d, device_collectives_ms_summary)
        elseif field_number == 6
            PB.decode!(d, host_compute_ms_summary)
        elseif field_number == 7
            PB.decode!(d, host_prepare_ms_summary)
        elseif field_number == 8
            PB.decode!(d, host_compile_ms_summary)
        else
            Base.skip(d, wire_type)
        end
    end
    return GenericStepTimeBreakdown(unknown_time_ms_summary[], host_wait_input_ms_summary[], host_to_device_ms_summary[], input_ms_summary[], output_ms_summary[], device_compute_ms_summary[], device_to_device_ms_summary[], device_collectives_ms_summary[], host_compute_ms_summary[], host_prepare_ms_summary[], host_compile_ms_summary[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GenericStepTimeBreakdown)
    initpos = position(e.io)
    !isnothing(x.unknown_time_ms_summary) && PB.encode(e, 1, x.unknown_time_ms_summary)
    !isnothing(x.host_wait_input_ms_summary) && PB.encode(e, 9, x.host_wait_input_ms_summary)
    !isnothing(x.host_to_device_ms_summary) && PB.encode(e, 10, x.host_to_device_ms_summary)
    !isnothing(x.input_ms_summary) && PB.encode(e, 11, x.input_ms_summary)
    !isnothing(x.output_ms_summary) && PB.encode(e, 3, x.output_ms_summary)
    !isnothing(x.device_compute_ms_summary) && PB.encode(e, 4, x.device_compute_ms_summary)
    !isnothing(x.device_to_device_ms_summary) && PB.encode(e, 5, x.device_to_device_ms_summary)
    !isnothing(x.device_collectives_ms_summary) && PB.encode(e, 12, x.device_collectives_ms_summary)
    !isnothing(x.host_compute_ms_summary) && PB.encode(e, 6, x.host_compute_ms_summary)
    !isnothing(x.host_prepare_ms_summary) && PB.encode(e, 7, x.host_prepare_ms_summary)
    !isnothing(x.host_compile_ms_summary) && PB.encode(e, 8, x.host_compile_ms_summary)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GenericStepTimeBreakdown)
    encoded_size = 0
    !isnothing(x.unknown_time_ms_summary) && (encoded_size += PB._encoded_size(x.unknown_time_ms_summary, 1))
    !isnothing(x.host_wait_input_ms_summary) && (encoded_size += PB._encoded_size(x.host_wait_input_ms_summary, 9))
    !isnothing(x.host_to_device_ms_summary) && (encoded_size += PB._encoded_size(x.host_to_device_ms_summary, 10))
    !isnothing(x.input_ms_summary) && (encoded_size += PB._encoded_size(x.input_ms_summary, 11))
    !isnothing(x.output_ms_summary) && (encoded_size += PB._encoded_size(x.output_ms_summary, 3))
    !isnothing(x.device_compute_ms_summary) && (encoded_size += PB._encoded_size(x.device_compute_ms_summary, 4))
    !isnothing(x.device_to_device_ms_summary) && (encoded_size += PB._encoded_size(x.device_to_device_ms_summary, 5))
    !isnothing(x.device_collectives_ms_summary) && (encoded_size += PB._encoded_size(x.device_collectives_ms_summary, 12))
    !isnothing(x.host_compute_ms_summary) && (encoded_size += PB._encoded_size(x.host_compute_ms_summary, 6))
    !isnothing(x.host_prepare_ms_summary) && (encoded_size += PB._encoded_size(x.host_prepare_ms_summary, 7))
    !isnothing(x.host_compile_ms_summary) && (encoded_size += PB._encoded_size(x.host_compile_ms_summary, 8))
    return encoded_size
end

struct InputPipelineAnalysisResult
    tag::Bool
    hardware_type::String
    step_time_summary::Union{Nothing,StepSummary}
    input_percent_summary::Union{Nothing,StepSummary}
    input_percent::Float64
    output_percent::Float64
    idle_percent::Float64
    compute_percent::Float64
    step_details::Vector{google.protobuf.var"#Any"}
    input_time_breakdown::Union{Nothing,InputTimeBreakdown}
    input_op_details::Vector{InputOpDetails}
    recommendation::Union{Nothing,InputPipelineAnalysisRecommendation}
    step_time_breakdown::Union{Nothing,google.protobuf.var"#Any"}
    diagnostics::Union{Nothing,Diagnostics}
end
InputPipelineAnalysisResult(;tag = false, hardware_type = "", step_time_summary = nothing, input_percent_summary = nothing, input_percent = zero(Float64), output_percent = zero(Float64), idle_percent = zero(Float64), compute_percent = zero(Float64), step_details = Vector{google.protobuf.var"#Any"}(), input_time_breakdown = nothing, input_op_details = Vector{InputOpDetails}(), recommendation = nothing, step_time_breakdown = nothing, diagnostics = nothing) = InputPipelineAnalysisResult(tag, hardware_type, step_time_summary, input_percent_summary, input_percent, output_percent, idle_percent, compute_percent, step_details, input_time_breakdown, input_op_details, recommendation, step_time_breakdown, diagnostics)
PB.reserved_fields(::Type{InputPipelineAnalysisResult}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[1, 10])
PB.default_values(::Type{InputPipelineAnalysisResult}) = (;tag = false, hardware_type = "", step_time_summary = nothing, input_percent_summary = nothing, input_percent = zero(Float64), output_percent = zero(Float64), idle_percent = zero(Float64), compute_percent = zero(Float64), step_details = Vector{google.protobuf.var"#Any"}(), input_time_breakdown = nothing, input_op_details = Vector{InputOpDetails}(), recommendation = nothing, step_time_breakdown = nothing, diagnostics = nothing)
PB.field_numbers(::Type{InputPipelineAnalysisResult}) = (;tag = 16, hardware_type = 9, step_time_summary = 2, input_percent_summary = 3, input_percent = 11, output_percent = 13, idle_percent = 14, compute_percent = 15, step_details = 4, input_time_breakdown = 5, input_op_details = 6, recommendation = 7, step_time_breakdown = 8, diagnostics = 12)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:InputPipelineAnalysisResult})
    tag = false
    hardware_type = ""
    step_time_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    input_percent_summary = Ref{Union{Nothing,StepSummary}}(nothing)
    input_percent = zero(Float64)
    output_percent = zero(Float64)
    idle_percent = zero(Float64)
    compute_percent = zero(Float64)
    step_details = PB.BufferedVector{google.protobuf.var"#Any"}()
    input_time_breakdown = Ref{Union{Nothing,InputTimeBreakdown}}(nothing)
    input_op_details = PB.BufferedVector{InputOpDetails}()
    recommendation = Ref{Union{Nothing,InputPipelineAnalysisRecommendation}}(nothing)
    step_time_breakdown = Ref{Union{Nothing,google.protobuf.var"#Any"}}(nothing)
    diagnostics = Ref{Union{Nothing,Diagnostics}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 16
            tag = PB.decode(d, Bool)
        elseif field_number == 9
            hardware_type = PB.decode(d, String)
        elseif field_number == 2
            PB.decode!(d, step_time_summary)
        elseif field_number == 3
            PB.decode!(d, input_percent_summary)
        elseif field_number == 11
            input_percent = PB.decode(d, Float64)
        elseif field_number == 13
            output_percent = PB.decode(d, Float64)
        elseif field_number == 14
            idle_percent = PB.decode(d, Float64)
        elseif field_number == 15
            compute_percent = PB.decode(d, Float64)
        elseif field_number == 4
            PB.decode!(d, step_details)
        elseif field_number == 5
            PB.decode!(d, input_time_breakdown)
        elseif field_number == 6
            PB.decode!(d, input_op_details)
        elseif field_number == 7
            PB.decode!(d, recommendation)
        elseif field_number == 8
            PB.decode!(d, step_time_breakdown)
        elseif field_number == 12
            PB.decode!(d, diagnostics)
        else
            Base.skip(d, wire_type)
        end
    end
    return InputPipelineAnalysisResult(tag, hardware_type, step_time_summary[], input_percent_summary[], input_percent, output_percent, idle_percent, compute_percent, step_details[], input_time_breakdown[], input_op_details[], recommendation[], step_time_breakdown[], diagnostics[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::InputPipelineAnalysisResult)
    initpos = position(e.io)
    x.tag != false && PB.encode(e, 16, x.tag)
    !isempty(x.hardware_type) && PB.encode(e, 9, x.hardware_type)
    !isnothing(x.step_time_summary) && PB.encode(e, 2, x.step_time_summary)
    !isnothing(x.input_percent_summary) && PB.encode(e, 3, x.input_percent_summary)
    x.input_percent !== zero(Float64) && PB.encode(e, 11, x.input_percent)
    x.output_percent !== zero(Float64) && PB.encode(e, 13, x.output_percent)
    x.idle_percent !== zero(Float64) && PB.encode(e, 14, x.idle_percent)
    x.compute_percent !== zero(Float64) && PB.encode(e, 15, x.compute_percent)
    !isempty(x.step_details) && PB.encode(e, 4, x.step_details)
    !isnothing(x.input_time_breakdown) && PB.encode(e, 5, x.input_time_breakdown)
    !isempty(x.input_op_details) && PB.encode(e, 6, x.input_op_details)
    !isnothing(x.recommendation) && PB.encode(e, 7, x.recommendation)
    !isnothing(x.step_time_breakdown) && PB.encode(e, 8, x.step_time_breakdown)
    !isnothing(x.diagnostics) && PB.encode(e, 12, x.diagnostics)
    return position(e.io) - initpos
end
function PB._encoded_size(x::InputPipelineAnalysisResult)
    encoded_size = 0
    x.tag != false && (encoded_size += PB._encoded_size(x.tag, 16))
    !isempty(x.hardware_type) && (encoded_size += PB._encoded_size(x.hardware_type, 9))
    !isnothing(x.step_time_summary) && (encoded_size += PB._encoded_size(x.step_time_summary, 2))
    !isnothing(x.input_percent_summary) && (encoded_size += PB._encoded_size(x.input_percent_summary, 3))
    x.input_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.input_percent, 11))
    x.output_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.output_percent, 13))
    x.idle_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.idle_percent, 14))
    x.compute_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.compute_percent, 15))
    !isempty(x.step_details) && (encoded_size += PB._encoded_size(x.step_details, 4))
    !isnothing(x.input_time_breakdown) && (encoded_size += PB._encoded_size(x.input_time_breakdown, 5))
    !isempty(x.input_op_details) && (encoded_size += PB._encoded_size(x.input_op_details, 6))
    !isnothing(x.recommendation) && (encoded_size += PB._encoded_size(x.recommendation, 7))
    !isnothing(x.step_time_breakdown) && (encoded_size += PB._encoded_size(x.step_time_breakdown, 8))
    !isnothing(x.diagnostics) && (encoded_size += PB._encoded_size(x.diagnostics, 12))
    return encoded_size
end
