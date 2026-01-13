import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"SourceStats.Metric", var"SourceStats.FileMetrics", SourceStats


struct var"SourceStats.Metric"
    occurrences::UInt64
    self_time_ps::UInt64
    time_ps::UInt64
    flops::UInt64
    flops_utilization::Float64
end
var"SourceStats.Metric"(;occurrences = zero(UInt64), self_time_ps = zero(UInt64), time_ps = zero(UInt64), flops = zero(UInt64), flops_utilization = zero(Float64)) = var"SourceStats.Metric"(occurrences, self_time_ps, time_ps, flops, flops_utilization)
PB.default_values(::Type{var"SourceStats.Metric"}) = (;occurrences = zero(UInt64), self_time_ps = zero(UInt64), time_ps = zero(UInt64), flops = zero(UInt64), flops_utilization = zero(Float64))
PB.field_numbers(::Type{var"SourceStats.Metric"}) = (;occurrences = 1, self_time_ps = 2, time_ps = 3, flops = 4, flops_utilization = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"SourceStats.Metric"})
    occurrences = zero(UInt64)
    self_time_ps = zero(UInt64)
    time_ps = zero(UInt64)
    flops = zero(UInt64)
    flops_utilization = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            occurrences = PB.decode(d, UInt64)
        elseif field_number == 2
            self_time_ps = PB.decode(d, UInt64)
        elseif field_number == 3
            time_ps = PB.decode(d, UInt64)
        elseif field_number == 4
            flops = PB.decode(d, UInt64)
        elseif field_number == 5
            flops_utilization = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"SourceStats.Metric"(occurrences, self_time_ps, time_ps, flops, flops_utilization)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"SourceStats.Metric")
    initpos = position(e.io)
    x.occurrences != zero(UInt64) && PB.encode(e, 1, x.occurrences)
    x.self_time_ps != zero(UInt64) && PB.encode(e, 2, x.self_time_ps)
    x.time_ps != zero(UInt64) && PB.encode(e, 3, x.time_ps)
    x.flops != zero(UInt64) && PB.encode(e, 4, x.flops)
    x.flops_utilization !== zero(Float64) && PB.encode(e, 5, x.flops_utilization)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"SourceStats.Metric")
    encoded_size = 0
    x.occurrences != zero(UInt64) && (encoded_size += PB._encoded_size(x.occurrences, 1))
    x.self_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.self_time_ps, 2))
    x.time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.time_ps, 3))
    x.flops != zero(UInt64) && (encoded_size += PB._encoded_size(x.flops, 4))
    x.flops_utilization !== zero(Float64) && (encoded_size += PB._encoded_size(x.flops_utilization, 5))
    return encoded_size
end

struct var"SourceStats.FileMetrics"
    line_number_to_metric::Dict{Int32,var"SourceStats.Metric"}
end
var"SourceStats.FileMetrics"(;line_number_to_metric = Dict{Int32,var"SourceStats.Metric"}()) = var"SourceStats.FileMetrics"(line_number_to_metric)
PB.default_values(::Type{var"SourceStats.FileMetrics"}) = (;line_number_to_metric = Dict{Int32,var"SourceStats.Metric"}())
PB.field_numbers(::Type{var"SourceStats.FileMetrics"}) = (;line_number_to_metric = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"SourceStats.FileMetrics"})
    line_number_to_metric = Dict{Int32,var"SourceStats.Metric"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, line_number_to_metric)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"SourceStats.FileMetrics"(line_number_to_metric)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"SourceStats.FileMetrics")
    initpos = position(e.io)
    !isempty(x.line_number_to_metric) && PB.encode(e, 1, x.line_number_to_metric)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"SourceStats.FileMetrics")
    encoded_size = 0
    !isempty(x.line_number_to_metric) && (encoded_size += PB._encoded_size(x.line_number_to_metric, 1))
    return encoded_size
end

struct SourceStats
    file_name_to_metrics::Dict{String,var"SourceStats.FileMetrics"}
end
SourceStats(;file_name_to_metrics = Dict{String,var"SourceStats.FileMetrics"}()) = SourceStats(file_name_to_metrics)
PB.default_values(::Type{SourceStats}) = (;file_name_to_metrics = Dict{String,var"SourceStats.FileMetrics"}())
PB.field_numbers(::Type{SourceStats}) = (;file_name_to_metrics = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SourceStats})
    file_name_to_metrics = Dict{String,var"SourceStats.FileMetrics"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, file_name_to_metrics)
        else
            Base.skip(d, wire_type)
        end
    end
    return SourceStats(file_name_to_metrics)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SourceStats)
    initpos = position(e.io)
    !isempty(x.file_name_to_metrics) && PB.encode(e, 1, x.file_name_to_metrics)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SourceStats)
    encoded_size = 0
    !isempty(x.file_name_to_metrics) && (encoded_size += PB._encoded_size(x.file_name_to_metrics, 1))
    return encoded_size
end
