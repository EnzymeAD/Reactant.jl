import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export PowerComponentMetrics, PowerMetrics


struct PowerComponentMetrics
    component_name::String
    max_power::Float64
    avg_power::Float64
    max_moving_avg_power_100us::Float64
    max_moving_avg_power_1ms::Float64
    max_moving_avg_power_10ms::Float64
    timescale_us::UInt32
    sample_count::UInt64
    max_moving_avg_power_1s::Float64
end
PowerComponentMetrics(;component_name = "", max_power = zero(Float64), avg_power = zero(Float64), max_moving_avg_power_100us = zero(Float64), max_moving_avg_power_1ms = zero(Float64), max_moving_avg_power_10ms = zero(Float64), timescale_us = zero(UInt32), sample_count = zero(UInt64), max_moving_avg_power_1s = zero(Float64)) = PowerComponentMetrics(component_name, max_power, avg_power, max_moving_avg_power_100us, max_moving_avg_power_1ms, max_moving_avg_power_10ms, timescale_us, sample_count, max_moving_avg_power_1s)
PB.default_values(::Type{PowerComponentMetrics}) = (;component_name = "", max_power = zero(Float64), avg_power = zero(Float64), max_moving_avg_power_100us = zero(Float64), max_moving_avg_power_1ms = zero(Float64), max_moving_avg_power_10ms = zero(Float64), timescale_us = zero(UInt32), sample_count = zero(UInt64), max_moving_avg_power_1s = zero(Float64))
PB.field_numbers(::Type{PowerComponentMetrics}) = (;component_name = 1, max_power = 2, avg_power = 3, max_moving_avg_power_100us = 4, max_moving_avg_power_1ms = 5, max_moving_avg_power_10ms = 6, timescale_us = 7, sample_count = 8, max_moving_avg_power_1s = 9)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PowerComponentMetrics})
    component_name = ""
    max_power = zero(Float64)
    avg_power = zero(Float64)
    max_moving_avg_power_100us = zero(Float64)
    max_moving_avg_power_1ms = zero(Float64)
    max_moving_avg_power_10ms = zero(Float64)
    timescale_us = zero(UInt32)
    sample_count = zero(UInt64)
    max_moving_avg_power_1s = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            component_name = PB.decode(d, String)
        elseif field_number == 2
            max_power = PB.decode(d, Float64)
        elseif field_number == 3
            avg_power = PB.decode(d, Float64)
        elseif field_number == 4
            max_moving_avg_power_100us = PB.decode(d, Float64)
        elseif field_number == 5
            max_moving_avg_power_1ms = PB.decode(d, Float64)
        elseif field_number == 6
            max_moving_avg_power_10ms = PB.decode(d, Float64)
        elseif field_number == 7
            timescale_us = PB.decode(d, UInt32)
        elseif field_number == 8
            sample_count = PB.decode(d, UInt64)
        elseif field_number == 9
            max_moving_avg_power_1s = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return PowerComponentMetrics(component_name, max_power, avg_power, max_moving_avg_power_100us, max_moving_avg_power_1ms, max_moving_avg_power_10ms, timescale_us, sample_count, max_moving_avg_power_1s)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PowerComponentMetrics)
    initpos = position(e.io)
    !isempty(x.component_name) && PB.encode(e, 1, x.component_name)
    x.max_power !== zero(Float64) && PB.encode(e, 2, x.max_power)
    x.avg_power !== zero(Float64) && PB.encode(e, 3, x.avg_power)
    x.max_moving_avg_power_100us !== zero(Float64) && PB.encode(e, 4, x.max_moving_avg_power_100us)
    x.max_moving_avg_power_1ms !== zero(Float64) && PB.encode(e, 5, x.max_moving_avg_power_1ms)
    x.max_moving_avg_power_10ms !== zero(Float64) && PB.encode(e, 6, x.max_moving_avg_power_10ms)
    x.timescale_us != zero(UInt32) && PB.encode(e, 7, x.timescale_us)
    x.sample_count != zero(UInt64) && PB.encode(e, 8, x.sample_count)
    x.max_moving_avg_power_1s !== zero(Float64) && PB.encode(e, 9, x.max_moving_avg_power_1s)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PowerComponentMetrics)
    encoded_size = 0
    !isempty(x.component_name) && (encoded_size += PB._encoded_size(x.component_name, 1))
    x.max_power !== zero(Float64) && (encoded_size += PB._encoded_size(x.max_power, 2))
    x.avg_power !== zero(Float64) && (encoded_size += PB._encoded_size(x.avg_power, 3))
    x.max_moving_avg_power_100us !== zero(Float64) && (encoded_size += PB._encoded_size(x.max_moving_avg_power_100us, 4))
    x.max_moving_avg_power_1ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.max_moving_avg_power_1ms, 5))
    x.max_moving_avg_power_10ms !== zero(Float64) && (encoded_size += PB._encoded_size(x.max_moving_avg_power_10ms, 6))
    x.timescale_us != zero(UInt32) && (encoded_size += PB._encoded_size(x.timescale_us, 7))
    x.sample_count != zero(UInt64) && (encoded_size += PB._encoded_size(x.sample_count, 8))
    x.max_moving_avg_power_1s !== zero(Float64) && (encoded_size += PB._encoded_size(x.max_moving_avg_power_1s, 9))
    return encoded_size
end

struct PowerMetrics
    power_component_metrics::Vector{PowerComponentMetrics}
end
PowerMetrics(;power_component_metrics = Vector{PowerComponentMetrics}()) = PowerMetrics(power_component_metrics)
PB.default_values(::Type{PowerMetrics}) = (;power_component_metrics = Vector{PowerComponentMetrics}())
PB.field_numbers(::Type{PowerMetrics}) = (;power_component_metrics = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PowerMetrics})
    power_component_metrics = PB.BufferedVector{PowerComponentMetrics}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, power_component_metrics)
        else
            Base.skip(d, wire_type)
        end
    end
    return PowerMetrics(power_component_metrics[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PowerMetrics)
    initpos = position(e.io)
    !isempty(x.power_component_metrics) && PB.encode(e, 1, x.power_component_metrics)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PowerMetrics)
    encoded_size = 0
    !isempty(x.power_component_metrics) && (encoded_size += PB._encoded_size(x.power_component_metrics, 1))
    return encoded_size
end
