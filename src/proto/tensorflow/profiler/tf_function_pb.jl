import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export TfFunctionMetrics, TfFunctionCompiler, TfFunctionExecutionMode, TfFunction
export TfFunctionDb


struct TfFunctionMetrics
    count::UInt64
    self_time_ps::UInt64
end
TfFunctionMetrics(;count = zero(UInt64), self_time_ps = zero(UInt64)) = TfFunctionMetrics(count, self_time_ps)
PB.default_values(::Type{TfFunctionMetrics}) = (;count = zero(UInt64), self_time_ps = zero(UInt64))
PB.field_numbers(::Type{TfFunctionMetrics}) = (;count = 1, self_time_ps = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TfFunctionMetrics})
    count = zero(UInt64)
    self_time_ps = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            count = PB.decode(d, UInt64)
        elseif field_number == 2
            self_time_ps = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return TfFunctionMetrics(count, self_time_ps)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TfFunctionMetrics)
    initpos = position(e.io)
    x.count != zero(UInt64) && PB.encode(e, 1, x.count)
    x.self_time_ps != zero(UInt64) && PB.encode(e, 2, x.self_time_ps)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TfFunctionMetrics)
    encoded_size = 0
    x.count != zero(UInt64) && (encoded_size += PB._encoded_size(x.count, 1))
    x.self_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.self_time_ps, 2))
    return encoded_size
end

@enumx TfFunctionCompiler INVALID_COMPILER=0 OTHER_COMPILER=1 MIXED_COMPILER=2 XLA_COMPILER=3 MLIR_COMPILER=4

@enumx TfFunctionExecutionMode INVALID_MODE=0 EAGER_MODE=1 TRACED_MODE=2 NOT_TRACED_MODE=3 CONCRETE_MODE=4

struct TfFunction
    metrics::Dict{Int32,TfFunctionMetrics}
    total_tracing_count::Int64
    compiler::TfFunctionCompiler.T
    expensive_call_percent::Float64
end
TfFunction(;metrics = Dict{Int32,TfFunctionMetrics}(), total_tracing_count = zero(Int64), compiler = TfFunctionCompiler.INVALID_COMPILER, expensive_call_percent = zero(Float64)) = TfFunction(metrics, total_tracing_count, compiler, expensive_call_percent)
PB.default_values(::Type{TfFunction}) = (;metrics = Dict{Int32,TfFunctionMetrics}(), total_tracing_count = zero(Int64), compiler = TfFunctionCompiler.INVALID_COMPILER, expensive_call_percent = zero(Float64))
PB.field_numbers(::Type{TfFunction}) = (;metrics = 1, total_tracing_count = 2, compiler = 3, expensive_call_percent = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TfFunction})
    metrics = Dict{Int32,TfFunctionMetrics}()
    total_tracing_count = zero(Int64)
    compiler = TfFunctionCompiler.INVALID_COMPILER
    expensive_call_percent = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, metrics)
        elseif field_number == 2
            total_tracing_count = PB.decode(d, Int64)
        elseif field_number == 3
            compiler = PB.decode(d, TfFunctionCompiler.T)
        elseif field_number == 4
            expensive_call_percent = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return TfFunction(metrics, total_tracing_count, compiler, expensive_call_percent)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TfFunction)
    initpos = position(e.io)
    !isempty(x.metrics) && PB.encode(e, 1, x.metrics)
    x.total_tracing_count != zero(Int64) && PB.encode(e, 2, x.total_tracing_count)
    x.compiler != TfFunctionCompiler.INVALID_COMPILER && PB.encode(e, 3, x.compiler)
    x.expensive_call_percent !== zero(Float64) && PB.encode(e, 4, x.expensive_call_percent)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TfFunction)
    encoded_size = 0
    !isempty(x.metrics) && (encoded_size += PB._encoded_size(x.metrics, 1))
    x.total_tracing_count != zero(Int64) && (encoded_size += PB._encoded_size(x.total_tracing_count, 2))
    x.compiler != TfFunctionCompiler.INVALID_COMPILER && (encoded_size += PB._encoded_size(x.compiler, 3))
    x.expensive_call_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.expensive_call_percent, 4))
    return encoded_size
end

struct TfFunctionDb
    tf_functions::Dict{String,TfFunction}
end
TfFunctionDb(;tf_functions = Dict{String,TfFunction}()) = TfFunctionDb(tf_functions)
PB.default_values(::Type{TfFunctionDb}) = (;tf_functions = Dict{String,TfFunction}())
PB.field_numbers(::Type{TfFunctionDb}) = (;tf_functions = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TfFunctionDb})
    tf_functions = Dict{String,TfFunction}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, tf_functions)
        else
            Base.skip(d, wire_type)
        end
    end
    return TfFunctionDb(tf_functions)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TfFunctionDb)
    initpos = position(e.io)
    !isempty(x.tf_functions) && PB.encode(e, 1, x.tf_functions)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TfFunctionDb)
    encoded_size = 0
    !isempty(x.tf_functions) && (encoded_size += PB._encoded_size(x.tf_functions, 1))
    return encoded_size
end
