import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export HardwareType, GPUComputeCapability, DeviceCapabilities


@enumx HardwareType UNKNOWN_HARDWARE=0 CPU_ONLY=1 GPU=2 TPU=3

struct GPUComputeCapability
    major::UInt32
    minor::UInt32
end
GPUComputeCapability(;major = zero(UInt32), minor = zero(UInt32)) = GPUComputeCapability(major, minor)
PB.default_values(::Type{GPUComputeCapability}) = (;major = zero(UInt32), minor = zero(UInt32))
PB.field_numbers(::Type{GPUComputeCapability}) = (;major = 1, minor = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GPUComputeCapability})
    major = zero(UInt32)
    minor = zero(UInt32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            major = PB.decode(d, UInt32)
        elseif field_number == 2
            minor = PB.decode(d, UInt32)
        else
            Base.skip(d, wire_type)
        end
    end
    return GPUComputeCapability(major, minor)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GPUComputeCapability)
    initpos = position(e.io)
    x.major != zero(UInt32) && PB.encode(e, 1, x.major)
    x.minor != zero(UInt32) && PB.encode(e, 2, x.minor)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GPUComputeCapability)
    encoded_size = 0
    x.major != zero(UInt32) && (encoded_size += PB._encoded_size(x.major, 1))
    x.minor != zero(UInt32) && (encoded_size += PB._encoded_size(x.minor, 2))
    return encoded_size
end

struct DeviceCapabilities
    clock_rate_in_ghz::Float64
    num_cores::UInt32
    memory_size_in_bytes::UInt64
    memory_bandwidth::UInt64
    compute_capability::Union{Nothing,GPUComputeCapability}
    device_vendor::String
end
DeviceCapabilities(;clock_rate_in_ghz = zero(Float64), num_cores = zero(UInt32), memory_size_in_bytes = zero(UInt64), memory_bandwidth = zero(UInt64), compute_capability = nothing, device_vendor = "") = DeviceCapabilities(clock_rate_in_ghz, num_cores, memory_size_in_bytes, memory_bandwidth, compute_capability, device_vendor)
PB.default_values(::Type{DeviceCapabilities}) = (;clock_rate_in_ghz = zero(Float64), num_cores = zero(UInt32), memory_size_in_bytes = zero(UInt64), memory_bandwidth = zero(UInt64), compute_capability = nothing, device_vendor = "")
PB.field_numbers(::Type{DeviceCapabilities}) = (;clock_rate_in_ghz = 1, num_cores = 2, memory_size_in_bytes = 3, memory_bandwidth = 4, compute_capability = 5, device_vendor = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:DeviceCapabilities})
    clock_rate_in_ghz = zero(Float64)
    num_cores = zero(UInt32)
    memory_size_in_bytes = zero(UInt64)
    memory_bandwidth = zero(UInt64)
    compute_capability = Ref{Union{Nothing,GPUComputeCapability}}(nothing)
    device_vendor = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            clock_rate_in_ghz = PB.decode(d, Float64)
        elseif field_number == 2
            num_cores = PB.decode(d, UInt32)
        elseif field_number == 3
            memory_size_in_bytes = PB.decode(d, UInt64)
        elseif field_number == 4
            memory_bandwidth = PB.decode(d, UInt64)
        elseif field_number == 5
            PB.decode!(d, compute_capability)
        elseif field_number == 6
            device_vendor = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return DeviceCapabilities(clock_rate_in_ghz, num_cores, memory_size_in_bytes, memory_bandwidth, compute_capability[], device_vendor)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::DeviceCapabilities)
    initpos = position(e.io)
    x.clock_rate_in_ghz !== zero(Float64) && PB.encode(e, 1, x.clock_rate_in_ghz)
    x.num_cores != zero(UInt32) && PB.encode(e, 2, x.num_cores)
    x.memory_size_in_bytes != zero(UInt64) && PB.encode(e, 3, x.memory_size_in_bytes)
    x.memory_bandwidth != zero(UInt64) && PB.encode(e, 4, x.memory_bandwidth)
    !isnothing(x.compute_capability) && PB.encode(e, 5, x.compute_capability)
    !isempty(x.device_vendor) && PB.encode(e, 6, x.device_vendor)
    return position(e.io) - initpos
end
function PB._encoded_size(x::DeviceCapabilities)
    encoded_size = 0
    x.clock_rate_in_ghz !== zero(Float64) && (encoded_size += PB._encoded_size(x.clock_rate_in_ghz, 1))
    x.num_cores != zero(UInt32) && (encoded_size += PB._encoded_size(x.num_cores, 2))
    x.memory_size_in_bytes != zero(UInt64) && (encoded_size += PB._encoded_size(x.memory_size_in_bytes, 3))
    x.memory_bandwidth != zero(UInt64) && (encoded_size += PB._encoded_size(x.memory_bandwidth, 4))
    !isnothing(x.compute_capability) && (encoded_size += PB._encoded_size(x.compute_capability, 5))
    !isempty(x.device_vendor) && (encoded_size += PB._encoded_size(x.device_vendor, 6))
    return encoded_size
end
