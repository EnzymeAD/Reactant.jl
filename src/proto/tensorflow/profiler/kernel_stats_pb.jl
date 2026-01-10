import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export KernelReport, KernelStatsDb


struct KernelReport
    name::String
    registers_per_thread::UInt32
    static_shmem_bytes::UInt32
    dynamic_shmem_bytes::UInt32
    block_dim::Vector{UInt32}
    grid_dim::Vector{UInt32}
    total_duration_ns::UInt64
    min_duration_ns::UInt64
    max_duration_ns::UInt64
    is_kernel_using_tensor_core::Bool
    is_op_tensor_core_eligible::Bool
    op_name::String
    occurrences::UInt32
    occupancy_pct::Float32
end
KernelReport(;name = "", registers_per_thread = zero(UInt32), static_shmem_bytes = zero(UInt32), dynamic_shmem_bytes = zero(UInt32), block_dim = Vector{UInt32}(), grid_dim = Vector{UInt32}(), total_duration_ns = zero(UInt64), min_duration_ns = zero(UInt64), max_duration_ns = zero(UInt64), is_kernel_using_tensor_core = false, is_op_tensor_core_eligible = false, op_name = "", occurrences = zero(UInt32), occupancy_pct = zero(Float32)) = KernelReport(name, registers_per_thread, static_shmem_bytes, dynamic_shmem_bytes, block_dim, grid_dim, total_duration_ns, min_duration_ns, max_duration_ns, is_kernel_using_tensor_core, is_op_tensor_core_eligible, op_name, occurrences, occupancy_pct)
PB.default_values(::Type{KernelReport}) = (;name = "", registers_per_thread = zero(UInt32), static_shmem_bytes = zero(UInt32), dynamic_shmem_bytes = zero(UInt32), block_dim = Vector{UInt32}(), grid_dim = Vector{UInt32}(), total_duration_ns = zero(UInt64), min_duration_ns = zero(UInt64), max_duration_ns = zero(UInt64), is_kernel_using_tensor_core = false, is_op_tensor_core_eligible = false, op_name = "", occurrences = zero(UInt32), occupancy_pct = zero(Float32))
PB.field_numbers(::Type{KernelReport}) = (;name = 1, registers_per_thread = 2, static_shmem_bytes = 3, dynamic_shmem_bytes = 4, block_dim = 5, grid_dim = 6, total_duration_ns = 7, min_duration_ns = 8, max_duration_ns = 9, is_kernel_using_tensor_core = 10, is_op_tensor_core_eligible = 11, op_name = 12, occurrences = 13, occupancy_pct = 14)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:KernelReport})
    name = ""
    registers_per_thread = zero(UInt32)
    static_shmem_bytes = zero(UInt32)
    dynamic_shmem_bytes = zero(UInt32)
    block_dim = PB.BufferedVector{UInt32}()
    grid_dim = PB.BufferedVector{UInt32}()
    total_duration_ns = zero(UInt64)
    min_duration_ns = zero(UInt64)
    max_duration_ns = zero(UInt64)
    is_kernel_using_tensor_core = false
    is_op_tensor_core_eligible = false
    op_name = ""
    occurrences = zero(UInt32)
    occupancy_pct = zero(Float32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            registers_per_thread = PB.decode(d, UInt32)
        elseif field_number == 3
            static_shmem_bytes = PB.decode(d, UInt32)
        elseif field_number == 4
            dynamic_shmem_bytes = PB.decode(d, UInt32)
        elseif field_number == 5
            PB.decode!(d, wire_type, block_dim)
        elseif field_number == 6
            PB.decode!(d, wire_type, grid_dim)
        elseif field_number == 7
            total_duration_ns = PB.decode(d, UInt64)
        elseif field_number == 8
            min_duration_ns = PB.decode(d, UInt64)
        elseif field_number == 9
            max_duration_ns = PB.decode(d, UInt64)
        elseif field_number == 10
            is_kernel_using_tensor_core = PB.decode(d, Bool)
        elseif field_number == 11
            is_op_tensor_core_eligible = PB.decode(d, Bool)
        elseif field_number == 12
            op_name = PB.decode(d, String)
        elseif field_number == 13
            occurrences = PB.decode(d, UInt32)
        elseif field_number == 14
            occupancy_pct = PB.decode(d, Float32)
        else
            Base.skip(d, wire_type)
        end
    end
    return KernelReport(name, registers_per_thread, static_shmem_bytes, dynamic_shmem_bytes, block_dim[], grid_dim[], total_duration_ns, min_duration_ns, max_duration_ns, is_kernel_using_tensor_core, is_op_tensor_core_eligible, op_name, occurrences, occupancy_pct)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::KernelReport)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    x.registers_per_thread != zero(UInt32) && PB.encode(e, 2, x.registers_per_thread)
    x.static_shmem_bytes != zero(UInt32) && PB.encode(e, 3, x.static_shmem_bytes)
    x.dynamic_shmem_bytes != zero(UInt32) && PB.encode(e, 4, x.dynamic_shmem_bytes)
    !isempty(x.block_dim) && PB.encode(e, 5, x.block_dim)
    !isempty(x.grid_dim) && PB.encode(e, 6, x.grid_dim)
    x.total_duration_ns != zero(UInt64) && PB.encode(e, 7, x.total_duration_ns)
    x.min_duration_ns != zero(UInt64) && PB.encode(e, 8, x.min_duration_ns)
    x.max_duration_ns != zero(UInt64) && PB.encode(e, 9, x.max_duration_ns)
    x.is_kernel_using_tensor_core != false && PB.encode(e, 10, x.is_kernel_using_tensor_core)
    x.is_op_tensor_core_eligible != false && PB.encode(e, 11, x.is_op_tensor_core_eligible)
    !isempty(x.op_name) && PB.encode(e, 12, x.op_name)
    x.occurrences != zero(UInt32) && PB.encode(e, 13, x.occurrences)
    x.occupancy_pct !== zero(Float32) && PB.encode(e, 14, x.occupancy_pct)
    return position(e.io) - initpos
end
function PB._encoded_size(x::KernelReport)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    x.registers_per_thread != zero(UInt32) && (encoded_size += PB._encoded_size(x.registers_per_thread, 2))
    x.static_shmem_bytes != zero(UInt32) && (encoded_size += PB._encoded_size(x.static_shmem_bytes, 3))
    x.dynamic_shmem_bytes != zero(UInt32) && (encoded_size += PB._encoded_size(x.dynamic_shmem_bytes, 4))
    !isempty(x.block_dim) && (encoded_size += PB._encoded_size(x.block_dim, 5))
    !isempty(x.grid_dim) && (encoded_size += PB._encoded_size(x.grid_dim, 6))
    x.total_duration_ns != zero(UInt64) && (encoded_size += PB._encoded_size(x.total_duration_ns, 7))
    x.min_duration_ns != zero(UInt64) && (encoded_size += PB._encoded_size(x.min_duration_ns, 8))
    x.max_duration_ns != zero(UInt64) && (encoded_size += PB._encoded_size(x.max_duration_ns, 9))
    x.is_kernel_using_tensor_core != false && (encoded_size += PB._encoded_size(x.is_kernel_using_tensor_core, 10))
    x.is_op_tensor_core_eligible != false && (encoded_size += PB._encoded_size(x.is_op_tensor_core_eligible, 11))
    !isempty(x.op_name) && (encoded_size += PB._encoded_size(x.op_name, 12))
    x.occurrences != zero(UInt32) && (encoded_size += PB._encoded_size(x.occurrences, 13))
    x.occupancy_pct !== zero(Float32) && (encoded_size += PB._encoded_size(x.occupancy_pct, 14))
    return encoded_size
end

struct KernelStatsDb
    reports::Vector{KernelReport}
end
KernelStatsDb(;reports = Vector{KernelReport}()) = KernelStatsDb(reports)
PB.default_values(::Type{KernelStatsDb}) = (;reports = Vector{KernelReport}())
PB.field_numbers(::Type{KernelStatsDb}) = (;reports = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:KernelStatsDb})
    reports = PB.BufferedVector{KernelReport}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, reports)
        else
            Base.skip(d, wire_type)
        end
    end
    return KernelStatsDb(reports[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::KernelStatsDb)
    initpos = position(e.io)
    !isempty(x.reports) && PB.encode(e, 1, x.reports)
    return position(e.io) - initpos
end
function PB._encoded_size(x::KernelStatsDb)
    encoded_size = 0
    !isempty(x.reports) && (encoded_size += PB._encoded_size(x.reports, 1))
    return encoded_size
end
