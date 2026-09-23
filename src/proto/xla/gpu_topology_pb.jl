import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export GpuTopologyProto


struct GpuTopologyProto
    platform_version::String
    num_partitions::Int32
    num_hosts_per_partition::Int32
    num_devices_per_host::Int32
    gpu_target_config::Union{Nothing,stream_executor.GpuTargetConfigProto}
    host_target_machine_options::Union{Nothing,xla.cpu.TargetMachineOptionsProto}
    num_devices_per_process::Int32
end
PB.reserved_fields(::Type{GpuTopologyProto}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[1, 2, 7])
PB.default_values(::Type{GpuTopologyProto}) = (;platform_version = "", num_partitions = zero(Int32), num_hosts_per_partition = zero(Int32), num_devices_per_host = zero(Int32), gpu_target_config = nothing, host_target_machine_options = nothing, num_devices_per_process = zero(Int32))
PB.field_numbers(::Type{GpuTopologyProto}) = (;platform_version = 3, num_partitions = 4, num_hosts_per_partition = 5, num_devices_per_host = 6, gpu_target_config = 8, host_target_machine_options = 9, num_devices_per_process = 10)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:GpuTopologyProto}, _endpos::Int=0, _group::Bool=false)
    platform_version = ""
    num_partitions = zero(Int32)
    num_hosts_per_partition = zero(Int32)
    num_devices_per_host = zero(Int32)
    gpu_target_config = Ref{Union{Nothing,stream_executor.GpuTargetConfigProto}}(nothing)
    host_target_machine_options = Ref{Union{Nothing,xla.cpu.TargetMachineOptionsProto}}(nothing)
    num_devices_per_process = zero(Int32)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 3
            platform_version = PB.decode(d, String)
        elseif field_number == 4
            num_partitions = PB.decode(d, Int32)
        elseif field_number == 5
            num_hosts_per_partition = PB.decode(d, Int32)
        elseif field_number == 6
            num_devices_per_host = PB.decode(d, Int32)
        elseif field_number == 8
            PB.decode!(d, gpu_target_config)
        elseif field_number == 9
            PB.decode!(d, host_target_machine_options)
        elseif field_number == 10
            num_devices_per_process = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return GpuTopologyProto(platform_version, num_partitions, num_hosts_per_partition, num_devices_per_host, gpu_target_config[], host_target_machine_options[], num_devices_per_process)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::GpuTopologyProto)
    initpos = position(e.io)
    !isempty(x.platform_version) && PB.encode(e, 3, x.platform_version)
    x.num_partitions != zero(Int32) && PB.encode(e, 4, x.num_partitions)
    x.num_hosts_per_partition != zero(Int32) && PB.encode(e, 5, x.num_hosts_per_partition)
    x.num_devices_per_host != zero(Int32) && PB.encode(e, 6, x.num_devices_per_host)
    !isnothing(x.gpu_target_config) && PB.encode(e, 8, x.gpu_target_config)
    !isnothing(x.host_target_machine_options) && PB.encode(e, 9, x.host_target_machine_options)
    x.num_devices_per_process != zero(Int32) && PB.encode(e, 10, x.num_devices_per_process)
    return position(e.io) - initpos
end
function PB._encoded_size(x::GpuTopologyProto)
    encoded_size = 0
    !isempty(x.platform_version) && (encoded_size += PB._encoded_size(x.platform_version, 3))
    x.num_partitions != zero(Int32) && (encoded_size += PB._encoded_size(x.num_partitions, 4))
    x.num_hosts_per_partition != zero(Int32) && (encoded_size += PB._encoded_size(x.num_hosts_per_partition, 5))
    x.num_devices_per_host != zero(Int32) && (encoded_size += PB._encoded_size(x.num_devices_per_host, 6))
    !isnothing(x.gpu_target_config) && (encoded_size += PB._encoded_size(x.gpu_target_config, 8))
    !isnothing(x.host_target_machine_options) && (encoded_size += PB._encoded_size(x.host_target_machine_options, 9))
    x.num_devices_per_process != zero(Int32) && (encoded_size += PB._encoded_size(x.num_devices_per_process, 10))
    return encoded_size
end
