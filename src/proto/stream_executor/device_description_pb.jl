import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export RocmComputeCapabilityProto, DnnVersionInfoProto, RuntimeVersionProto
export GpuDeviceInfoProto, GpuComputeCapabilityProto, GpuTargetConfigProto


struct RocmComputeCapabilityProto
    gcn_arch_name::String
end
RocmComputeCapabilityProto(;gcn_arch_name = "") = RocmComputeCapabilityProto(gcn_arch_name)
PB.default_values(::Type{RocmComputeCapabilityProto}) = (;gcn_arch_name = "")
PB.field_numbers(::Type{RocmComputeCapabilityProto}) = (;gcn_arch_name = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:RocmComputeCapabilityProto})
    gcn_arch_name = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            gcn_arch_name = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return RocmComputeCapabilityProto(gcn_arch_name)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::RocmComputeCapabilityProto)
    initpos = position(e.io)
    !isempty(x.gcn_arch_name) && PB.encode(e, 1, x.gcn_arch_name)
    return position(e.io) - initpos
end
function PB._encoded_size(x::RocmComputeCapabilityProto)
    encoded_size = 0
    !isempty(x.gcn_arch_name) && (encoded_size += PB._encoded_size(x.gcn_arch_name, 1))
    return encoded_size
end

struct DnnVersionInfoProto
    major::Int32
    minor::Int32
    patch::Int32
end
DnnVersionInfoProto(;major = zero(Int32), minor = zero(Int32), patch = zero(Int32)) = DnnVersionInfoProto(major, minor, patch)
PB.default_values(::Type{DnnVersionInfoProto}) = (;major = zero(Int32), minor = zero(Int32), patch = zero(Int32))
PB.field_numbers(::Type{DnnVersionInfoProto}) = (;major = 1, minor = 2, patch = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:DnnVersionInfoProto})
    major = zero(Int32)
    minor = zero(Int32)
    patch = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            major = PB.decode(d, Int32)
        elseif field_number == 2
            minor = PB.decode(d, Int32)
        elseif field_number == 3
            patch = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return DnnVersionInfoProto(major, minor, patch)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::DnnVersionInfoProto)
    initpos = position(e.io)
    x.major != zero(Int32) && PB.encode(e, 1, x.major)
    x.minor != zero(Int32) && PB.encode(e, 2, x.minor)
    x.patch != zero(Int32) && PB.encode(e, 3, x.patch)
    return position(e.io) - initpos
end
function PB._encoded_size(x::DnnVersionInfoProto)
    encoded_size = 0
    x.major != zero(Int32) && (encoded_size += PB._encoded_size(x.major, 1))
    x.minor != zero(Int32) && (encoded_size += PB._encoded_size(x.minor, 2))
    x.patch != zero(Int32) && (encoded_size += PB._encoded_size(x.patch, 3))
    return encoded_size
end

struct RuntimeVersionProto
    major::Int32
    minor::Int32
    patch::Int32
end
RuntimeVersionProto(;major = zero(Int32), minor = zero(Int32), patch = zero(Int32)) = RuntimeVersionProto(major, minor, patch)
PB.default_values(::Type{RuntimeVersionProto}) = (;major = zero(Int32), minor = zero(Int32), patch = zero(Int32))
PB.field_numbers(::Type{RuntimeVersionProto}) = (;major = 1, minor = 2, patch = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:RuntimeVersionProto})
    major = zero(Int32)
    minor = zero(Int32)
    patch = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            major = PB.decode(d, Int32)
        elseif field_number == 2
            minor = PB.decode(d, Int32)
        elseif field_number == 3
            patch = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return RuntimeVersionProto(major, minor, patch)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::RuntimeVersionProto)
    initpos = position(e.io)
    x.major != zero(Int32) && PB.encode(e, 1, x.major)
    x.minor != zero(Int32) && PB.encode(e, 2, x.minor)
    x.patch != zero(Int32) && PB.encode(e, 3, x.patch)
    return position(e.io) - initpos
end
function PB._encoded_size(x::RuntimeVersionProto)
    encoded_size = 0
    x.major != zero(Int32) && (encoded_size += PB._encoded_size(x.major, 1))
    x.minor != zero(Int32) && (encoded_size += PB._encoded_size(x.minor, 2))
    x.patch != zero(Int32) && (encoded_size += PB._encoded_size(x.patch, 3))
    return encoded_size
end

struct GpuDeviceInfoProto
    threads_per_block_limit::Int32
    threads_per_warp::Int32
    shared_memory_per_block::Int32
    shared_memory_per_core::Int32
    threads_per_core_limit::Int32
    core_count::Int32
    fpus_per_core::Int64
    block_dim_limit_x::Int32
    block_dim_limit_y::Int32
    block_dim_limit_z::Int32
    memory_bandwidth::Int64
    l2_cache_size::Int64
    clock_rate_ghz::Float32
    device_memory_size::Int64
    shared_memory_per_block_optin::Int32
    compute_capability::Union{Nothing,OneOf{<:Union{CudaComputeCapabilityProto,RocmComputeCapabilityProto}}}
    registers_per_core_limit::Int64
    registers_per_block_limit::Int64
end
GpuDeviceInfoProto(;threads_per_block_limit = zero(Int32), threads_per_warp = zero(Int32), shared_memory_per_block = zero(Int32), shared_memory_per_core = zero(Int32), threads_per_core_limit = zero(Int32), core_count = zero(Int32), fpus_per_core = zero(Int64), block_dim_limit_x = zero(Int32), block_dim_limit_y = zero(Int32), block_dim_limit_z = zero(Int32), memory_bandwidth = zero(Int64), l2_cache_size = zero(Int64), clock_rate_ghz = zero(Float32), device_memory_size = zero(Int64), shared_memory_per_block_optin = zero(Int32), compute_capability = nothing, registers_per_core_limit = zero(Int64), registers_per_block_limit = zero(Int64)) = GpuDeviceInfoProto(threads_per_block_limit, threads_per_warp, shared_memory_per_block, shared_memory_per_core, threads_per_core_limit, core_count, fpus_per_core, block_dim_limit_x, block_dim_limit_y, block_dim_limit_z, memory_bandwidth, l2_cache_size, clock_rate_ghz, device_memory_size, shared_memory_per_block_optin, compute_capability, registers_per_core_limit, registers_per_block_limit)
PB.oneof_field_types(::Type{GpuDeviceInfoProto}) = (;
    compute_capability = (;cuda_compute_capability=CudaComputeCapabilityProto, rocm_compute_capability=RocmComputeCapabilityProto),
)
PB.default_values(::Type{GpuDeviceInfoProto}) = (;threads_per_block_limit = zero(Int32), threads_per_warp = zero(Int32), shared_memory_per_block = zero(Int32), shared_memory_per_core = zero(Int32), threads_per_core_limit = zero(Int32), core_count = zero(Int32), fpus_per_core = zero(Int64), block_dim_limit_x = zero(Int32), block_dim_limit_y = zero(Int32), block_dim_limit_z = zero(Int32), memory_bandwidth = zero(Int64), l2_cache_size = zero(Int64), clock_rate_ghz = zero(Float32), device_memory_size = zero(Int64), shared_memory_per_block_optin = zero(Int32), cuda_compute_capability = nothing, rocm_compute_capability = nothing, registers_per_core_limit = zero(Int64), registers_per_block_limit = zero(Int64))
PB.field_numbers(::Type{GpuDeviceInfoProto}) = (;threads_per_block_limit = 1, threads_per_warp = 2, shared_memory_per_block = 3, shared_memory_per_core = 4, threads_per_core_limit = 5, core_count = 6, fpus_per_core = 7, block_dim_limit_x = 8, block_dim_limit_y = 9, block_dim_limit_z = 10, memory_bandwidth = 11, l2_cache_size = 12, clock_rate_ghz = 13, device_memory_size = 14, shared_memory_per_block_optin = 15, cuda_compute_capability = 16, rocm_compute_capability = 17, registers_per_core_limit = 18, registers_per_block_limit = 19)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GpuDeviceInfoProto})
    threads_per_block_limit = zero(Int32)
    threads_per_warp = zero(Int32)
    shared_memory_per_block = zero(Int32)
    shared_memory_per_core = zero(Int32)
    threads_per_core_limit = zero(Int32)
    core_count = zero(Int32)
    fpus_per_core = zero(Int64)
    block_dim_limit_x = zero(Int32)
    block_dim_limit_y = zero(Int32)
    block_dim_limit_z = zero(Int32)
    memory_bandwidth = zero(Int64)
    l2_cache_size = zero(Int64)
    clock_rate_ghz = zero(Float32)
    device_memory_size = zero(Int64)
    shared_memory_per_block_optin = zero(Int32)
    compute_capability = nothing
    registers_per_core_limit = zero(Int64)
    registers_per_block_limit = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            threads_per_block_limit = PB.decode(d, Int32)
        elseif field_number == 2
            threads_per_warp = PB.decode(d, Int32)
        elseif field_number == 3
            shared_memory_per_block = PB.decode(d, Int32)
        elseif field_number == 4
            shared_memory_per_core = PB.decode(d, Int32)
        elseif field_number == 5
            threads_per_core_limit = PB.decode(d, Int32)
        elseif field_number == 6
            core_count = PB.decode(d, Int32)
        elseif field_number == 7
            fpus_per_core = PB.decode(d, Int64)
        elseif field_number == 8
            block_dim_limit_x = PB.decode(d, Int32)
        elseif field_number == 9
            block_dim_limit_y = PB.decode(d, Int32)
        elseif field_number == 10
            block_dim_limit_z = PB.decode(d, Int32)
        elseif field_number == 11
            memory_bandwidth = PB.decode(d, Int64)
        elseif field_number == 12
            l2_cache_size = PB.decode(d, Int64)
        elseif field_number == 13
            clock_rate_ghz = PB.decode(d, Float32)
        elseif field_number == 14
            device_memory_size = PB.decode(d, Int64)
        elseif field_number == 15
            shared_memory_per_block_optin = PB.decode(d, Int32)
        elseif field_number == 16
            compute_capability = OneOf(:cuda_compute_capability, PB.decode(d, Ref{CudaComputeCapabilityProto}))
        elseif field_number == 17
            compute_capability = OneOf(:rocm_compute_capability, PB.decode(d, Ref{RocmComputeCapabilityProto}))
        elseif field_number == 18
            registers_per_core_limit = PB.decode(d, Int64)
        elseif field_number == 19
            registers_per_block_limit = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return GpuDeviceInfoProto(threads_per_block_limit, threads_per_warp, shared_memory_per_block, shared_memory_per_core, threads_per_core_limit, core_count, fpus_per_core, block_dim_limit_x, block_dim_limit_y, block_dim_limit_z, memory_bandwidth, l2_cache_size, clock_rate_ghz, device_memory_size, shared_memory_per_block_optin, compute_capability, registers_per_core_limit, registers_per_block_limit)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GpuDeviceInfoProto)
    initpos = position(e.io)
    x.threads_per_block_limit != zero(Int32) && PB.encode(e, 1, x.threads_per_block_limit)
    x.threads_per_warp != zero(Int32) && PB.encode(e, 2, x.threads_per_warp)
    x.shared_memory_per_block != zero(Int32) && PB.encode(e, 3, x.shared_memory_per_block)
    x.shared_memory_per_core != zero(Int32) && PB.encode(e, 4, x.shared_memory_per_core)
    x.threads_per_core_limit != zero(Int32) && PB.encode(e, 5, x.threads_per_core_limit)
    x.core_count != zero(Int32) && PB.encode(e, 6, x.core_count)
    x.fpus_per_core != zero(Int64) && PB.encode(e, 7, x.fpus_per_core)
    x.block_dim_limit_x != zero(Int32) && PB.encode(e, 8, x.block_dim_limit_x)
    x.block_dim_limit_y != zero(Int32) && PB.encode(e, 9, x.block_dim_limit_y)
    x.block_dim_limit_z != zero(Int32) && PB.encode(e, 10, x.block_dim_limit_z)
    x.memory_bandwidth != zero(Int64) && PB.encode(e, 11, x.memory_bandwidth)
    x.l2_cache_size != zero(Int64) && PB.encode(e, 12, x.l2_cache_size)
    x.clock_rate_ghz !== zero(Float32) && PB.encode(e, 13, x.clock_rate_ghz)
    x.device_memory_size != zero(Int64) && PB.encode(e, 14, x.device_memory_size)
    x.shared_memory_per_block_optin != zero(Int32) && PB.encode(e, 15, x.shared_memory_per_block_optin)
    if isnothing(x.compute_capability);
    elseif x.compute_capability.name === :cuda_compute_capability
        PB.encode(e, 16, x.compute_capability[]::CudaComputeCapabilityProto)
    elseif x.compute_capability.name === :rocm_compute_capability
        PB.encode(e, 17, x.compute_capability[]::RocmComputeCapabilityProto)
    end
    x.registers_per_core_limit != zero(Int64) && PB.encode(e, 18, x.registers_per_core_limit)
    x.registers_per_block_limit != zero(Int64) && PB.encode(e, 19, x.registers_per_block_limit)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GpuDeviceInfoProto)
    encoded_size = 0
    x.threads_per_block_limit != zero(Int32) && (encoded_size += PB._encoded_size(x.threads_per_block_limit, 1))
    x.threads_per_warp != zero(Int32) && (encoded_size += PB._encoded_size(x.threads_per_warp, 2))
    x.shared_memory_per_block != zero(Int32) && (encoded_size += PB._encoded_size(x.shared_memory_per_block, 3))
    x.shared_memory_per_core != zero(Int32) && (encoded_size += PB._encoded_size(x.shared_memory_per_core, 4))
    x.threads_per_core_limit != zero(Int32) && (encoded_size += PB._encoded_size(x.threads_per_core_limit, 5))
    x.core_count != zero(Int32) && (encoded_size += PB._encoded_size(x.core_count, 6))
    x.fpus_per_core != zero(Int64) && (encoded_size += PB._encoded_size(x.fpus_per_core, 7))
    x.block_dim_limit_x != zero(Int32) && (encoded_size += PB._encoded_size(x.block_dim_limit_x, 8))
    x.block_dim_limit_y != zero(Int32) && (encoded_size += PB._encoded_size(x.block_dim_limit_y, 9))
    x.block_dim_limit_z != zero(Int32) && (encoded_size += PB._encoded_size(x.block_dim_limit_z, 10))
    x.memory_bandwidth != zero(Int64) && (encoded_size += PB._encoded_size(x.memory_bandwidth, 11))
    x.l2_cache_size != zero(Int64) && (encoded_size += PB._encoded_size(x.l2_cache_size, 12))
    x.clock_rate_ghz !== zero(Float32) && (encoded_size += PB._encoded_size(x.clock_rate_ghz, 13))
    x.device_memory_size != zero(Int64) && (encoded_size += PB._encoded_size(x.device_memory_size, 14))
    x.shared_memory_per_block_optin != zero(Int32) && (encoded_size += PB._encoded_size(x.shared_memory_per_block_optin, 15))
    if isnothing(x.compute_capability);
    elseif x.compute_capability.name === :cuda_compute_capability
        encoded_size += PB._encoded_size(x.compute_capability[]::CudaComputeCapabilityProto, 16)
    elseif x.compute_capability.name === :rocm_compute_capability
        encoded_size += PB._encoded_size(x.compute_capability[]::RocmComputeCapabilityProto, 17)
    end
    x.registers_per_core_limit != zero(Int64) && (encoded_size += PB._encoded_size(x.registers_per_core_limit, 18))
    x.registers_per_block_limit != zero(Int64) && (encoded_size += PB._encoded_size(x.registers_per_block_limit, 19))
    return encoded_size
end

struct GpuComputeCapabilityProto
    compute_capability::Union{Nothing,OneOf{<:Union{CudaComputeCapabilityProto,RocmComputeCapabilityProto}}}
end
GpuComputeCapabilityProto(;compute_capability = nothing) = GpuComputeCapabilityProto(compute_capability)
PB.oneof_field_types(::Type{GpuComputeCapabilityProto}) = (;
    compute_capability = (;cuda_compute_capability=CudaComputeCapabilityProto, rocm_compute_capability=RocmComputeCapabilityProto),
)
PB.default_values(::Type{GpuComputeCapabilityProto}) = (;cuda_compute_capability = nothing, rocm_compute_capability = nothing)
PB.field_numbers(::Type{GpuComputeCapabilityProto}) = (;cuda_compute_capability = 1, rocm_compute_capability = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GpuComputeCapabilityProto})
    compute_capability = nothing
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            compute_capability = OneOf(:cuda_compute_capability, PB.decode(d, Ref{CudaComputeCapabilityProto}))
        elseif field_number == 2
            compute_capability = OneOf(:rocm_compute_capability, PB.decode(d, Ref{RocmComputeCapabilityProto}))
        else
            Base.skip(d, wire_type)
        end
    end
    return GpuComputeCapabilityProto(compute_capability)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GpuComputeCapabilityProto)
    initpos = position(e.io)
    if isnothing(x.compute_capability);
    elseif x.compute_capability.name === :cuda_compute_capability
        PB.encode(e, 1, x.compute_capability[]::CudaComputeCapabilityProto)
    elseif x.compute_capability.name === :rocm_compute_capability
        PB.encode(e, 2, x.compute_capability[]::RocmComputeCapabilityProto)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::GpuComputeCapabilityProto)
    encoded_size = 0
    if isnothing(x.compute_capability);
    elseif x.compute_capability.name === :cuda_compute_capability
        encoded_size += PB._encoded_size(x.compute_capability[]::CudaComputeCapabilityProto, 1)
    elseif x.compute_capability.name === :rocm_compute_capability
        encoded_size += PB._encoded_size(x.compute_capability[]::RocmComputeCapabilityProto, 2)
    end
    return encoded_size
end

struct GpuTargetConfigProto
    gpu_device_info::Union{Nothing,GpuDeviceInfoProto}
    platform_name::String
    dnn_version_info::Union{Nothing,DnnVersionInfoProto}
    runtime_version::Union{Nothing,RuntimeVersionProto}
    autotune_results::Union{Nothing,xla_autotuning.AutotuneResults}
    device_description_str::String
end
GpuTargetConfigProto(;gpu_device_info = nothing, platform_name = "", dnn_version_info = nothing, runtime_version = nothing, autotune_results = nothing, device_description_str = "") = GpuTargetConfigProto(gpu_device_info, platform_name, dnn_version_info, runtime_version, autotune_results, device_description_str)
PB.reserved_fields(::Type{GpuTargetConfigProto}) = (names = ["cuda_compute_capability", "rocm_compute_capability"], numbers = Union{Int,UnitRange{Int}}[2, 3])
PB.default_values(::Type{GpuTargetConfigProto}) = (;gpu_device_info = nothing, platform_name = "", dnn_version_info = nothing, runtime_version = nothing, autotune_results = nothing, device_description_str = "")
PB.field_numbers(::Type{GpuTargetConfigProto}) = (;gpu_device_info = 1, platform_name = 4, dnn_version_info = 5, runtime_version = 8, autotune_results = 6, device_description_str = 7)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GpuTargetConfigProto})
    gpu_device_info = Ref{Union{Nothing,GpuDeviceInfoProto}}(nothing)
    platform_name = ""
    dnn_version_info = Ref{Union{Nothing,DnnVersionInfoProto}}(nothing)
    runtime_version = Ref{Union{Nothing,RuntimeVersionProto}}(nothing)
    autotune_results = Ref{Union{Nothing,xla_autotuning.AutotuneResults}}(nothing)
    device_description_str = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, gpu_device_info)
        elseif field_number == 4
            platform_name = PB.decode(d, String)
        elseif field_number == 5
            PB.decode!(d, dnn_version_info)
        elseif field_number == 8
            PB.decode!(d, runtime_version)
        elseif field_number == 6
            PB.decode!(d, autotune_results)
        elseif field_number == 7
            device_description_str = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return GpuTargetConfigProto(gpu_device_info[], platform_name, dnn_version_info[], runtime_version[], autotune_results[], device_description_str)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GpuTargetConfigProto)
    initpos = position(e.io)
    !isnothing(x.gpu_device_info) && PB.encode(e, 1, x.gpu_device_info)
    !isempty(x.platform_name) && PB.encode(e, 4, x.platform_name)
    !isnothing(x.dnn_version_info) && PB.encode(e, 5, x.dnn_version_info)
    !isnothing(x.runtime_version) && PB.encode(e, 8, x.runtime_version)
    !isnothing(x.autotune_results) && PB.encode(e, 6, x.autotune_results)
    !isempty(x.device_description_str) && PB.encode(e, 7, x.device_description_str)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GpuTargetConfigProto)
    encoded_size = 0
    !isnothing(x.gpu_device_info) && (encoded_size += PB._encoded_size(x.gpu_device_info, 1))
    !isempty(x.platform_name) && (encoded_size += PB._encoded_size(x.platform_name, 4))
    !isnothing(x.dnn_version_info) && (encoded_size += PB._encoded_size(x.dnn_version_info, 5))
    !isnothing(x.runtime_version) && (encoded_size += PB._encoded_size(x.runtime_version, 8))
    !isnothing(x.autotune_results) && (encoded_size += PB._encoded_size(x.autotune_results, 6))
    !isempty(x.device_description_str) && (encoded_size += PB._encoded_size(x.device_description_str, 7))
    return encoded_size
end
