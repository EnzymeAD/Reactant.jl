import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export AutotuningLogs, var"AutotuneResults.Entry", AutotuneResults


struct AutotuningLogs
    logs::Vector{AutotuningLog}
end
AutotuningLogs(;logs = Vector{AutotuningLog}()) = AutotuningLogs(logs)
PB.default_values(::Type{AutotuningLogs}) = (;logs = Vector{AutotuningLog}())
PB.field_numbers(::Type{AutotuningLogs}) = (;logs = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AutotuningLogs})
    logs = PB.BufferedVector{AutotuningLog}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, logs)
        else
            Base.skip(d, wire_type)
        end
    end
    return AutotuningLogs(logs[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AutotuningLogs)
    initpos = position(e.io)
    !isempty(x.logs) && PB.encode(e, 1, x.logs)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AutotuningLogs)
    encoded_size = 0
    !isempty(x.logs) && (encoded_size += PB._encoded_size(x.logs, 1))
    return encoded_size
end

struct var"AutotuneResults.Entry"
    device::String
    hlo::String
    result::Union{Nothing,AutotuneResult}
    version::Int32
end
var"AutotuneResults.Entry"(;device = "", hlo = "", result = nothing, version = zero(Int32)) = var"AutotuneResults.Entry"(device, hlo, result, version)
PB.default_values(::Type{var"AutotuneResults.Entry"}) = (;device = "", hlo = "", result = nothing, version = zero(Int32))
PB.field_numbers(::Type{var"AutotuneResults.Entry"}) = (;device = 1, hlo = 2, result = 3, version = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"AutotuneResults.Entry"})
    device = ""
    hlo = ""
    result = Ref{Union{Nothing,AutotuneResult}}(nothing)
    version = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            device = PB.decode(d, String)
        elseif field_number == 2
            hlo = PB.decode(d, String)
        elseif field_number == 3
            PB.decode!(d, result)
        elseif field_number == 4
            version = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"AutotuneResults.Entry"(device, hlo, result[], version)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"AutotuneResults.Entry")
    initpos = position(e.io)
    !isempty(x.device) && PB.encode(e, 1, x.device)
    !isempty(x.hlo) && PB.encode(e, 2, x.hlo)
    !isnothing(x.result) && PB.encode(e, 3, x.result)
    x.version != zero(Int32) && PB.encode(e, 4, x.version)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"AutotuneResults.Entry")
    encoded_size = 0
    !isempty(x.device) && (encoded_size += PB._encoded_size(x.device, 1))
    !isempty(x.hlo) && (encoded_size += PB._encoded_size(x.hlo, 2))
    !isnothing(x.result) && (encoded_size += PB._encoded_size(x.result, 3))
    x.version != zero(Int32) && (encoded_size += PB._encoded_size(x.version, 4))
    return encoded_size
end

struct AutotuneResults
    version::Int32
    results::Vector{var"AutotuneResults.Entry"}
end
AutotuneResults(;version = zero(Int32), results = Vector{var"AutotuneResults.Entry"}()) = AutotuneResults(version, results)
PB.reserved_fields(::Type{AutotuneResults}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[2, 3])
PB.default_values(::Type{AutotuneResults}) = (;version = zero(Int32), results = Vector{var"AutotuneResults.Entry"}())
PB.field_numbers(::Type{AutotuneResults}) = (;version = 1, results = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AutotuneResults})
    version = zero(Int32)
    results = PB.BufferedVector{var"AutotuneResults.Entry"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            version = PB.decode(d, Int32)
        elseif field_number == 4
            PB.decode!(d, results)
        else
            Base.skip(d, wire_type)
        end
    end
    return AutotuneResults(version, results[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AutotuneResults)
    initpos = position(e.io)
    x.version != zero(Int32) && PB.encode(e, 1, x.version)
    !isempty(x.results) && PB.encode(e, 4, x.results)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AutotuneResults)
    encoded_size = 0
    x.version != zero(Int32) && (encoded_size += PB._encoded_size(x.version, 1))
    !isempty(x.results) && (encoded_size += PB._encoded_size(x.results, 4))
    return encoded_size
end
