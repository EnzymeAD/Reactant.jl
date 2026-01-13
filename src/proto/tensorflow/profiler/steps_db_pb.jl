import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export DeviceMemoryTransfer, GenericStepBreakdown, AllReduceInfo, SparseCoreStepBreakdown
export TpuStepBreakdown, StepInfoResult, AllReduceDbResult, PerCoreStepInfo
export StepDatabaseResult


struct DeviceMemoryTransfer
    occurrence::UInt64
    time_us::Float64
    bytes_transferred::UInt64
end
DeviceMemoryTransfer(;occurrence = zero(UInt64), time_us = zero(Float64), bytes_transferred = zero(UInt64)) = DeviceMemoryTransfer(occurrence, time_us, bytes_transferred)
PB.default_values(::Type{DeviceMemoryTransfer}) = (;occurrence = zero(UInt64), time_us = zero(Float64), bytes_transferred = zero(UInt64))
PB.field_numbers(::Type{DeviceMemoryTransfer}) = (;occurrence = 1, time_us = 2, bytes_transferred = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:DeviceMemoryTransfer})
    occurrence = zero(UInt64)
    time_us = zero(Float64)
    bytes_transferred = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            occurrence = PB.decode(d, UInt64)
        elseif field_number == 2
            time_us = PB.decode(d, Float64)
        elseif field_number == 3
            bytes_transferred = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return DeviceMemoryTransfer(occurrence, time_us, bytes_transferred)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::DeviceMemoryTransfer)
    initpos = position(e.io)
    x.occurrence != zero(UInt64) && PB.encode(e, 1, x.occurrence)
    x.time_us !== zero(Float64) && PB.encode(e, 2, x.time_us)
    x.bytes_transferred != zero(UInt64) && PB.encode(e, 3, x.bytes_transferred)
    return position(e.io) - initpos
end
function PB._encoded_size(x::DeviceMemoryTransfer)
    encoded_size = 0
    x.occurrence != zero(UInt64) && (encoded_size += PB._encoded_size(x.occurrence, 1))
    x.time_us !== zero(Float64) && (encoded_size += PB._encoded_size(x.time_us, 2))
    x.bytes_transferred != zero(UInt64) && (encoded_size += PB._encoded_size(x.bytes_transferred, 3))
    return encoded_size
end

struct GenericStepBreakdown
    type_ps::Dict{Int32,UInt64}
    category_ps::Dict{String,UInt64}
end
GenericStepBreakdown(;type_ps = Dict{Int32,UInt64}(), category_ps = Dict{String,UInt64}()) = GenericStepBreakdown(type_ps, category_ps)
PB.default_values(::Type{GenericStepBreakdown}) = (;type_ps = Dict{Int32,UInt64}(), category_ps = Dict{String,UInt64}())
PB.field_numbers(::Type{GenericStepBreakdown}) = (;type_ps = 1, category_ps = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GenericStepBreakdown})
    type_ps = Dict{Int32,UInt64}()
    category_ps = Dict{String,UInt64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, type_ps)
        elseif field_number == 2
            PB.decode!(d, category_ps)
        else
            Base.skip(d, wire_type)
        end
    end
    return GenericStepBreakdown(type_ps, category_ps)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GenericStepBreakdown)
    initpos = position(e.io)
    !isempty(x.type_ps) && PB.encode(e, 1, x.type_ps)
    !isempty(x.category_ps) && PB.encode(e, 2, x.category_ps)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GenericStepBreakdown)
    encoded_size = 0
    !isempty(x.type_ps) && (encoded_size += PB._encoded_size(x.type_ps, 1))
    !isempty(x.category_ps) && (encoded_size += PB._encoded_size(x.category_ps, 2))
    return encoded_size
end

struct AllReduceInfo
    id::UInt64
    name::String
    all_reduce_id::UInt64
    start_time_ps::UInt64
    end_time_ps::UInt64
    byte_size::UInt64
end
AllReduceInfo(;id = zero(UInt64), name = "", all_reduce_id = zero(UInt64), start_time_ps = zero(UInt64), end_time_ps = zero(UInt64), byte_size = zero(UInt64)) = AllReduceInfo(id, name, all_reduce_id, start_time_ps, end_time_ps, byte_size)
PB.default_values(::Type{AllReduceInfo}) = (;id = zero(UInt64), name = "", all_reduce_id = zero(UInt64), start_time_ps = zero(UInt64), end_time_ps = zero(UInt64), byte_size = zero(UInt64))
PB.field_numbers(::Type{AllReduceInfo}) = (;id = 1, name = 2, all_reduce_id = 3, start_time_ps = 4, end_time_ps = 5, byte_size = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AllReduceInfo})
    id = zero(UInt64)
    name = ""
    all_reduce_id = zero(UInt64)
    start_time_ps = zero(UInt64)
    end_time_ps = zero(UInt64)
    byte_size = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            id = PB.decode(d, UInt64)
        elseif field_number == 2
            name = PB.decode(d, String)
        elseif field_number == 3
            all_reduce_id = PB.decode(d, UInt64)
        elseif field_number == 4
            start_time_ps = PB.decode(d, UInt64)
        elseif field_number == 5
            end_time_ps = PB.decode(d, UInt64)
        elseif field_number == 6
            byte_size = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return AllReduceInfo(id, name, all_reduce_id, start_time_ps, end_time_ps, byte_size)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AllReduceInfo)
    initpos = position(e.io)
    x.id != zero(UInt64) && PB.encode(e, 1, x.id)
    !isempty(x.name) && PB.encode(e, 2, x.name)
    x.all_reduce_id != zero(UInt64) && PB.encode(e, 3, x.all_reduce_id)
    x.start_time_ps != zero(UInt64) && PB.encode(e, 4, x.start_time_ps)
    x.end_time_ps != zero(UInt64) && PB.encode(e, 5, x.end_time_ps)
    x.byte_size != zero(UInt64) && PB.encode(e, 6, x.byte_size)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AllReduceInfo)
    encoded_size = 0
    x.id != zero(UInt64) && (encoded_size += PB._encoded_size(x.id, 1))
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 2))
    x.all_reduce_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.all_reduce_id, 3))
    x.start_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.start_time_ps, 4))
    x.end_time_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.end_time_ps, 5))
    x.byte_size != zero(UInt64) && (encoded_size += PB._encoded_size(x.byte_size, 6))
    return encoded_size
end

struct SparseCoreStepBreakdown
    sc_compute_ps::UInt64
    sc_infeed_ps::UInt64
    sc_outfeed_ps::UInt64
    sc_idle_ps::UInt64
    sc_busy_ps::UInt64
end
SparseCoreStepBreakdown(;sc_compute_ps = zero(UInt64), sc_infeed_ps = zero(UInt64), sc_outfeed_ps = zero(UInt64), sc_idle_ps = zero(UInt64), sc_busy_ps = zero(UInt64)) = SparseCoreStepBreakdown(sc_compute_ps, sc_infeed_ps, sc_outfeed_ps, sc_idle_ps, sc_busy_ps)
PB.default_values(::Type{SparseCoreStepBreakdown}) = (;sc_compute_ps = zero(UInt64), sc_infeed_ps = zero(UInt64), sc_outfeed_ps = zero(UInt64), sc_idle_ps = zero(UInt64), sc_busy_ps = zero(UInt64))
PB.field_numbers(::Type{SparseCoreStepBreakdown}) = (;sc_compute_ps = 1, sc_infeed_ps = 2, sc_outfeed_ps = 3, sc_idle_ps = 4, sc_busy_ps = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SparseCoreStepBreakdown})
    sc_compute_ps = zero(UInt64)
    sc_infeed_ps = zero(UInt64)
    sc_outfeed_ps = zero(UInt64)
    sc_idle_ps = zero(UInt64)
    sc_busy_ps = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            sc_compute_ps = PB.decode(d, UInt64)
        elseif field_number == 2
            sc_infeed_ps = PB.decode(d, UInt64)
        elseif field_number == 3
            sc_outfeed_ps = PB.decode(d, UInt64)
        elseif field_number == 4
            sc_idle_ps = PB.decode(d, UInt64)
        elseif field_number == 5
            sc_busy_ps = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return SparseCoreStepBreakdown(sc_compute_ps, sc_infeed_ps, sc_outfeed_ps, sc_idle_ps, sc_busy_ps)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SparseCoreStepBreakdown)
    initpos = position(e.io)
    x.sc_compute_ps != zero(UInt64) && PB.encode(e, 1, x.sc_compute_ps)
    x.sc_infeed_ps != zero(UInt64) && PB.encode(e, 2, x.sc_infeed_ps)
    x.sc_outfeed_ps != zero(UInt64) && PB.encode(e, 3, x.sc_outfeed_ps)
    x.sc_idle_ps != zero(UInt64) && PB.encode(e, 4, x.sc_idle_ps)
    x.sc_busy_ps != zero(UInt64) && PB.encode(e, 5, x.sc_busy_ps)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SparseCoreStepBreakdown)
    encoded_size = 0
    x.sc_compute_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.sc_compute_ps, 1))
    x.sc_infeed_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.sc_infeed_ps, 2))
    x.sc_outfeed_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.sc_outfeed_ps, 3))
    x.sc_idle_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.sc_idle_ps, 4))
    x.sc_busy_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.sc_busy_ps, 5))
    return encoded_size
end

struct TpuStepBreakdown
    infeed_duration_ps::UInt64
    host_outfeed_ps::UInt64
    wait_for_scv0_duration_ps::UInt64
    scv0_infeed_transform_ps::UInt64
    scv0_outfeed_ps::UInt64
    crs_duration_ps::UInt64
    scv0_infeed_percent::Float64
    send_duration_ps::UInt64
    recv_duration_ps::UInt64
    host_send_duration_ps::UInt64
    host_recv_duration_ps::UInt64
    wait_for_megacore_fusion_peer_duration_ps::UInt64
    overlay_wait_duration_ps::UInt64
    high_flops_compute_ps::UInt64
    tc_idle_ps::UInt64
    tc_busy_ps::UInt64
    scv0_busy_ps::UInt64
    scv0_step_ps::UInt64
end
TpuStepBreakdown(;infeed_duration_ps = zero(UInt64), host_outfeed_ps = zero(UInt64), wait_for_scv0_duration_ps = zero(UInt64), scv0_infeed_transform_ps = zero(UInt64), scv0_outfeed_ps = zero(UInt64), crs_duration_ps = zero(UInt64), scv0_infeed_percent = zero(Float64), send_duration_ps = zero(UInt64), recv_duration_ps = zero(UInt64), host_send_duration_ps = zero(UInt64), host_recv_duration_ps = zero(UInt64), wait_for_megacore_fusion_peer_duration_ps = zero(UInt64), overlay_wait_duration_ps = zero(UInt64), high_flops_compute_ps = zero(UInt64), tc_idle_ps = zero(UInt64), tc_busy_ps = zero(UInt64), scv0_busy_ps = zero(UInt64), scv0_step_ps = zero(UInt64)) = TpuStepBreakdown(infeed_duration_ps, host_outfeed_ps, wait_for_scv0_duration_ps, scv0_infeed_transform_ps, scv0_outfeed_ps, crs_duration_ps, scv0_infeed_percent, send_duration_ps, recv_duration_ps, host_send_duration_ps, host_recv_duration_ps, wait_for_megacore_fusion_peer_duration_ps, overlay_wait_duration_ps, high_flops_compute_ps, tc_idle_ps, tc_busy_ps, scv0_busy_ps, scv0_step_ps)
PB.reserved_fields(::Type{TpuStepBreakdown}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[10])
PB.default_values(::Type{TpuStepBreakdown}) = (;infeed_duration_ps = zero(UInt64), host_outfeed_ps = zero(UInt64), wait_for_scv0_duration_ps = zero(UInt64), scv0_infeed_transform_ps = zero(UInt64), scv0_outfeed_ps = zero(UInt64), crs_duration_ps = zero(UInt64), scv0_infeed_percent = zero(Float64), send_duration_ps = zero(UInt64), recv_duration_ps = zero(UInt64), host_send_duration_ps = zero(UInt64), host_recv_duration_ps = zero(UInt64), wait_for_megacore_fusion_peer_duration_ps = zero(UInt64), overlay_wait_duration_ps = zero(UInt64), high_flops_compute_ps = zero(UInt64), tc_idle_ps = zero(UInt64), tc_busy_ps = zero(UInt64), scv0_busy_ps = zero(UInt64), scv0_step_ps = zero(UInt64))
PB.field_numbers(::Type{TpuStepBreakdown}) = (;infeed_duration_ps = 1, host_outfeed_ps = 2, wait_for_scv0_duration_ps = 3, scv0_infeed_transform_ps = 4, scv0_outfeed_ps = 5, crs_duration_ps = 6, scv0_infeed_percent = 7, send_duration_ps = 8, recv_duration_ps = 9, host_send_duration_ps = 15, host_recv_duration_ps = 16, wait_for_megacore_fusion_peer_duration_ps = 14, overlay_wait_duration_ps = 11, high_flops_compute_ps = 12, tc_idle_ps = 13, tc_busy_ps = 17, scv0_busy_ps = 18, scv0_step_ps = 19)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TpuStepBreakdown})
    infeed_duration_ps = zero(UInt64)
    host_outfeed_ps = zero(UInt64)
    wait_for_scv0_duration_ps = zero(UInt64)
    scv0_infeed_transform_ps = zero(UInt64)
    scv0_outfeed_ps = zero(UInt64)
    crs_duration_ps = zero(UInt64)
    scv0_infeed_percent = zero(Float64)
    send_duration_ps = zero(UInt64)
    recv_duration_ps = zero(UInt64)
    host_send_duration_ps = zero(UInt64)
    host_recv_duration_ps = zero(UInt64)
    wait_for_megacore_fusion_peer_duration_ps = zero(UInt64)
    overlay_wait_duration_ps = zero(UInt64)
    high_flops_compute_ps = zero(UInt64)
    tc_idle_ps = zero(UInt64)
    tc_busy_ps = zero(UInt64)
    scv0_busy_ps = zero(UInt64)
    scv0_step_ps = zero(UInt64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            infeed_duration_ps = PB.decode(d, UInt64)
        elseif field_number == 2
            host_outfeed_ps = PB.decode(d, UInt64)
        elseif field_number == 3
            wait_for_scv0_duration_ps = PB.decode(d, UInt64)
        elseif field_number == 4
            scv0_infeed_transform_ps = PB.decode(d, UInt64)
        elseif field_number == 5
            scv0_outfeed_ps = PB.decode(d, UInt64)
        elseif field_number == 6
            crs_duration_ps = PB.decode(d, UInt64)
        elseif field_number == 7
            scv0_infeed_percent = PB.decode(d, Float64)
        elseif field_number == 8
            send_duration_ps = PB.decode(d, UInt64)
        elseif field_number == 9
            recv_duration_ps = PB.decode(d, UInt64)
        elseif field_number == 15
            host_send_duration_ps = PB.decode(d, UInt64)
        elseif field_number == 16
            host_recv_duration_ps = PB.decode(d, UInt64)
        elseif field_number == 14
            wait_for_megacore_fusion_peer_duration_ps = PB.decode(d, UInt64)
        elseif field_number == 11
            overlay_wait_duration_ps = PB.decode(d, UInt64)
        elseif field_number == 12
            high_flops_compute_ps = PB.decode(d, UInt64)
        elseif field_number == 13
            tc_idle_ps = PB.decode(d, UInt64)
        elseif field_number == 17
            tc_busy_ps = PB.decode(d, UInt64)
        elseif field_number == 18
            scv0_busy_ps = PB.decode(d, UInt64)
        elseif field_number == 19
            scv0_step_ps = PB.decode(d, UInt64)
        else
            Base.skip(d, wire_type)
        end
    end
    return TpuStepBreakdown(infeed_duration_ps, host_outfeed_ps, wait_for_scv0_duration_ps, scv0_infeed_transform_ps, scv0_outfeed_ps, crs_duration_ps, scv0_infeed_percent, send_duration_ps, recv_duration_ps, host_send_duration_ps, host_recv_duration_ps, wait_for_megacore_fusion_peer_duration_ps, overlay_wait_duration_ps, high_flops_compute_ps, tc_idle_ps, tc_busy_ps, scv0_busy_ps, scv0_step_ps)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TpuStepBreakdown)
    initpos = position(e.io)
    x.infeed_duration_ps != zero(UInt64) && PB.encode(e, 1, x.infeed_duration_ps)
    x.host_outfeed_ps != zero(UInt64) && PB.encode(e, 2, x.host_outfeed_ps)
    x.wait_for_scv0_duration_ps != zero(UInt64) && PB.encode(e, 3, x.wait_for_scv0_duration_ps)
    x.scv0_infeed_transform_ps != zero(UInt64) && PB.encode(e, 4, x.scv0_infeed_transform_ps)
    x.scv0_outfeed_ps != zero(UInt64) && PB.encode(e, 5, x.scv0_outfeed_ps)
    x.crs_duration_ps != zero(UInt64) && PB.encode(e, 6, x.crs_duration_ps)
    x.scv0_infeed_percent !== zero(Float64) && PB.encode(e, 7, x.scv0_infeed_percent)
    x.send_duration_ps != zero(UInt64) && PB.encode(e, 8, x.send_duration_ps)
    x.recv_duration_ps != zero(UInt64) && PB.encode(e, 9, x.recv_duration_ps)
    x.host_send_duration_ps != zero(UInt64) && PB.encode(e, 15, x.host_send_duration_ps)
    x.host_recv_duration_ps != zero(UInt64) && PB.encode(e, 16, x.host_recv_duration_ps)
    x.wait_for_megacore_fusion_peer_duration_ps != zero(UInt64) && PB.encode(e, 14, x.wait_for_megacore_fusion_peer_duration_ps)
    x.overlay_wait_duration_ps != zero(UInt64) && PB.encode(e, 11, x.overlay_wait_duration_ps)
    x.high_flops_compute_ps != zero(UInt64) && PB.encode(e, 12, x.high_flops_compute_ps)
    x.tc_idle_ps != zero(UInt64) && PB.encode(e, 13, x.tc_idle_ps)
    x.tc_busy_ps != zero(UInt64) && PB.encode(e, 17, x.tc_busy_ps)
    x.scv0_busy_ps != zero(UInt64) && PB.encode(e, 18, x.scv0_busy_ps)
    x.scv0_step_ps != zero(UInt64) && PB.encode(e, 19, x.scv0_step_ps)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TpuStepBreakdown)
    encoded_size = 0
    x.infeed_duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.infeed_duration_ps, 1))
    x.host_outfeed_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.host_outfeed_ps, 2))
    x.wait_for_scv0_duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.wait_for_scv0_duration_ps, 3))
    x.scv0_infeed_transform_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.scv0_infeed_transform_ps, 4))
    x.scv0_outfeed_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.scv0_outfeed_ps, 5))
    x.crs_duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.crs_duration_ps, 6))
    x.scv0_infeed_percent !== zero(Float64) && (encoded_size += PB._encoded_size(x.scv0_infeed_percent, 7))
    x.send_duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.send_duration_ps, 8))
    x.recv_duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.recv_duration_ps, 9))
    x.host_send_duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.host_send_duration_ps, 15))
    x.host_recv_duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.host_recv_duration_ps, 16))
    x.wait_for_megacore_fusion_peer_duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.wait_for_megacore_fusion_peer_duration_ps, 14))
    x.overlay_wait_duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.overlay_wait_duration_ps, 11))
    x.high_flops_compute_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.high_flops_compute_ps, 12))
    x.tc_idle_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.tc_idle_ps, 13))
    x.tc_busy_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.tc_busy_ps, 17))
    x.scv0_busy_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.scv0_busy_ps, 18))
    x.scv0_step_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.scv0_step_ps, 19))
    return encoded_size
end

struct StepInfoResult
    step_num::UInt32
    step_name::String
    duration_ps::UInt64
    begin_ps::UInt64
    step_breakdown::Union{Nothing,google.protobuf.var"#Any"}
    collectives::Union{Nothing,DeviceMemoryTransfer}
end
StepInfoResult(;step_num = zero(UInt32), step_name = "", duration_ps = zero(UInt64), begin_ps = zero(UInt64), step_breakdown = nothing, collectives = nothing) = StepInfoResult(step_num, step_name, duration_ps, begin_ps, step_breakdown, collectives)
PB.default_values(::Type{StepInfoResult}) = (;step_num = zero(UInt32), step_name = "", duration_ps = zero(UInt64), begin_ps = zero(UInt64), step_breakdown = nothing, collectives = nothing)
PB.field_numbers(::Type{StepInfoResult}) = (;step_num = 1, step_name = 5, duration_ps = 2, begin_ps = 3, step_breakdown = 4, collectives = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:StepInfoResult})
    step_num = zero(UInt32)
    step_name = ""
    duration_ps = zero(UInt64)
    begin_ps = zero(UInt64)
    step_breakdown = Ref{Union{Nothing,google.protobuf.var"#Any"}}(nothing)
    collectives = Ref{Union{Nothing,DeviceMemoryTransfer}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            step_num = PB.decode(d, UInt32)
        elseif field_number == 5
            step_name = PB.decode(d, String)
        elseif field_number == 2
            duration_ps = PB.decode(d, UInt64)
        elseif field_number == 3
            begin_ps = PB.decode(d, UInt64)
        elseif field_number == 4
            PB.decode!(d, step_breakdown)
        elseif field_number == 6
            PB.decode!(d, collectives)
        else
            Base.skip(d, wire_type)
        end
    end
    return StepInfoResult(step_num, step_name, duration_ps, begin_ps, step_breakdown[], collectives[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::StepInfoResult)
    initpos = position(e.io)
    x.step_num != zero(UInt32) && PB.encode(e, 1, x.step_num)
    !isempty(x.step_name) && PB.encode(e, 5, x.step_name)
    x.duration_ps != zero(UInt64) && PB.encode(e, 2, x.duration_ps)
    x.begin_ps != zero(UInt64) && PB.encode(e, 3, x.begin_ps)
    !isnothing(x.step_breakdown) && PB.encode(e, 4, x.step_breakdown)
    !isnothing(x.collectives) && PB.encode(e, 6, x.collectives)
    return position(e.io) - initpos
end
function PB._encoded_size(x::StepInfoResult)
    encoded_size = 0
    x.step_num != zero(UInt32) && (encoded_size += PB._encoded_size(x.step_num, 1))
    !isempty(x.step_name) && (encoded_size += PB._encoded_size(x.step_name, 5))
    x.duration_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.duration_ps, 2))
    x.begin_ps != zero(UInt64) && (encoded_size += PB._encoded_size(x.begin_ps, 3))
    !isnothing(x.step_breakdown) && (encoded_size += PB._encoded_size(x.step_breakdown, 4))
    !isnothing(x.collectives) && (encoded_size += PB._encoded_size(x.collectives, 6))
    return encoded_size
end

struct AllReduceDbResult
    all_reduce_info::Vector{AllReduceInfo}
end
AllReduceDbResult(;all_reduce_info = Vector{AllReduceInfo}()) = AllReduceDbResult(all_reduce_info)
PB.default_values(::Type{AllReduceDbResult}) = (;all_reduce_info = Vector{AllReduceInfo}())
PB.field_numbers(::Type{AllReduceDbResult}) = (;all_reduce_info = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AllReduceDbResult})
    all_reduce_info = PB.BufferedVector{AllReduceInfo}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, all_reduce_info)
        else
            Base.skip(d, wire_type)
        end
    end
    return AllReduceDbResult(all_reduce_info[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AllReduceDbResult)
    initpos = position(e.io)
    !isempty(x.all_reduce_info) && PB.encode(e, 1, x.all_reduce_info)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AllReduceDbResult)
    encoded_size = 0
    !isempty(x.all_reduce_info) && (encoded_size += PB._encoded_size(x.all_reduce_info, 1))
    return encoded_size
end

struct PerCoreStepInfo
    step_num::UInt32
    step_info_per_core::Dict{UInt32,StepInfoResult}
    hlo_metrics_db::Union{Nothing,OpMetricsDb}
    core_id_to_replica_id_map::Dict{UInt32,UInt32}
    all_reduce_db_per_core::Dict{UInt32,AllReduceDbResult}
    device_memory_transfers::Vector{DeviceMemoryTransfer}
end
PerCoreStepInfo(;step_num = zero(UInt32), step_info_per_core = Dict{UInt32,StepInfoResult}(), hlo_metrics_db = nothing, core_id_to_replica_id_map = Dict{UInt32,UInt32}(), all_reduce_db_per_core = Dict{UInt32,AllReduceDbResult}(), device_memory_transfers = Vector{DeviceMemoryTransfer}()) = PerCoreStepInfo(step_num, step_info_per_core, hlo_metrics_db, core_id_to_replica_id_map, all_reduce_db_per_core, device_memory_transfers)
PB.reserved_fields(::Type{PerCoreStepInfo}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[4])
PB.default_values(::Type{PerCoreStepInfo}) = (;step_num = zero(UInt32), step_info_per_core = Dict{UInt32,StepInfoResult}(), hlo_metrics_db = nothing, core_id_to_replica_id_map = Dict{UInt32,UInt32}(), all_reduce_db_per_core = Dict{UInt32,AllReduceDbResult}(), device_memory_transfers = Vector{DeviceMemoryTransfer}())
PB.field_numbers(::Type{PerCoreStepInfo}) = (;step_num = 1, step_info_per_core = 2, hlo_metrics_db = 3, core_id_to_replica_id_map = 5, all_reduce_db_per_core = 6, device_memory_transfers = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PerCoreStepInfo})
    step_num = zero(UInt32)
    step_info_per_core = Dict{UInt32,StepInfoResult}()
    hlo_metrics_db = Ref{Union{Nothing,OpMetricsDb}}(nothing)
    core_id_to_replica_id_map = Dict{UInt32,UInt32}()
    all_reduce_db_per_core = Dict{UInt32,AllReduceDbResult}()
    device_memory_transfers = PB.BufferedVector{DeviceMemoryTransfer}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            step_num = PB.decode(d, UInt32)
        elseif field_number == 2
            PB.decode!(d, step_info_per_core)
        elseif field_number == 3
            PB.decode!(d, hlo_metrics_db)
        elseif field_number == 5
            PB.decode!(d, core_id_to_replica_id_map)
        elseif field_number == 6
            PB.decode!(d, all_reduce_db_per_core)
        elseif field_number == 7
            PB.decode!(d, device_memory_transfers)
        else
            Base.skip(d, wire_type)
        end
    end
    return PerCoreStepInfo(step_num, step_info_per_core, hlo_metrics_db[], core_id_to_replica_id_map, all_reduce_db_per_core, device_memory_transfers[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PerCoreStepInfo)
    initpos = position(e.io)
    x.step_num != zero(UInt32) && PB.encode(e, 1, x.step_num)
    !isempty(x.step_info_per_core) && PB.encode(e, 2, x.step_info_per_core)
    !isnothing(x.hlo_metrics_db) && PB.encode(e, 3, x.hlo_metrics_db)
    !isempty(x.core_id_to_replica_id_map) && PB.encode(e, 5, x.core_id_to_replica_id_map)
    !isempty(x.all_reduce_db_per_core) && PB.encode(e, 6, x.all_reduce_db_per_core)
    !isempty(x.device_memory_transfers) && PB.encode(e, 7, x.device_memory_transfers)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PerCoreStepInfo)
    encoded_size = 0
    x.step_num != zero(UInt32) && (encoded_size += PB._encoded_size(x.step_num, 1))
    !isempty(x.step_info_per_core) && (encoded_size += PB._encoded_size(x.step_info_per_core, 2))
    !isnothing(x.hlo_metrics_db) && (encoded_size += PB._encoded_size(x.hlo_metrics_db, 3))
    !isempty(x.core_id_to_replica_id_map) && (encoded_size += PB._encoded_size(x.core_id_to_replica_id_map, 5))
    !isempty(x.all_reduce_db_per_core) && (encoded_size += PB._encoded_size(x.all_reduce_db_per_core, 6))
    !isempty(x.device_memory_transfers) && (encoded_size += PB._encoded_size(x.device_memory_transfers, 7))
    return encoded_size
end

struct StepDatabaseResult
    step_sequence::Vector{PerCoreStepInfo}
    use_incomplete_step::Bool
    num_steps_dropped::UInt32
    empty_intersect::Bool
end
StepDatabaseResult(;step_sequence = Vector{PerCoreStepInfo}(), use_incomplete_step = false, num_steps_dropped = zero(UInt32), empty_intersect = false) = StepDatabaseResult(step_sequence, use_incomplete_step, num_steps_dropped, empty_intersect)
PB.default_values(::Type{StepDatabaseResult}) = (;step_sequence = Vector{PerCoreStepInfo}(), use_incomplete_step = false, num_steps_dropped = zero(UInt32), empty_intersect = false)
PB.field_numbers(::Type{StepDatabaseResult}) = (;step_sequence = 1, use_incomplete_step = 2, num_steps_dropped = 3, empty_intersect = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:StepDatabaseResult})
    step_sequence = PB.BufferedVector{PerCoreStepInfo}()
    use_incomplete_step = false
    num_steps_dropped = zero(UInt32)
    empty_intersect = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, step_sequence)
        elseif field_number == 2
            use_incomplete_step = PB.decode(d, Bool)
        elseif field_number == 3
            num_steps_dropped = PB.decode(d, UInt32)
        elseif field_number == 4
            empty_intersect = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return StepDatabaseResult(step_sequence[], use_incomplete_step, num_steps_dropped, empty_intersect)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::StepDatabaseResult)
    initpos = position(e.io)
    !isempty(x.step_sequence) && PB.encode(e, 1, x.step_sequence)
    x.use_incomplete_step != false && PB.encode(e, 2, x.use_incomplete_step)
    x.num_steps_dropped != zero(UInt32) && PB.encode(e, 3, x.num_steps_dropped)
    x.empty_intersect != false && PB.encode(e, 4, x.empty_intersect)
    return position(e.io) - initpos
end
function PB._encoded_size(x::StepDatabaseResult)
    encoded_size = 0
    !isempty(x.step_sequence) && (encoded_size += PB._encoded_size(x.step_sequence, 1))
    x.use_incomplete_step != false && (encoded_size += PB._encoded_size(x.use_incomplete_step, 2))
    x.num_steps_dropped != zero(UInt32) && (encoded_size += PB._encoded_size(x.num_steps_dropped, 3))
    x.empty_intersect != false && (encoded_size += PB._encoded_size(x.empty_intersect, 4))
    return encoded_size
end
