import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export MemoryActivity, MemoryAggregationStats, ActiveAllocation, MemoryActivityMetadata
export MemoryProfileSummary, MemoryProfileSnapshot, PerAllocatorMemoryProfile
export MemoryProfile


@enumx MemoryActivity UNKNOWN_ACTIVITY=0 ALLOCATION=1 DEALLOCATION=2 RESERVATION=3 EXPANSION=4

struct MemoryAggregationStats
    stack_reserved_bytes::Int64
    heap_allocated_bytes::Int64
    free_memory_bytes::Int64
    fragmentation::Float64
    peak_bytes_in_use::Int64
end
MemoryAggregationStats(;stack_reserved_bytes = zero(Int64), heap_allocated_bytes = zero(Int64), free_memory_bytes = zero(Int64), fragmentation = zero(Float64), peak_bytes_in_use = zero(Int64)) = MemoryAggregationStats(stack_reserved_bytes, heap_allocated_bytes, free_memory_bytes, fragmentation, peak_bytes_in_use)
PB.default_values(::Type{MemoryAggregationStats}) = (;stack_reserved_bytes = zero(Int64), heap_allocated_bytes = zero(Int64), free_memory_bytes = zero(Int64), fragmentation = zero(Float64), peak_bytes_in_use = zero(Int64))
PB.field_numbers(::Type{MemoryAggregationStats}) = (;stack_reserved_bytes = 1, heap_allocated_bytes = 2, free_memory_bytes = 3, fragmentation = 4, peak_bytes_in_use = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:MemoryAggregationStats})
    stack_reserved_bytes = zero(Int64)
    heap_allocated_bytes = zero(Int64)
    free_memory_bytes = zero(Int64)
    fragmentation = zero(Float64)
    peak_bytes_in_use = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            stack_reserved_bytes = PB.decode(d, Int64)
        elseif field_number == 2
            heap_allocated_bytes = PB.decode(d, Int64)
        elseif field_number == 3
            free_memory_bytes = PB.decode(d, Int64)
        elseif field_number == 4
            fragmentation = PB.decode(d, Float64)
        elseif field_number == 5
            peak_bytes_in_use = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return MemoryAggregationStats(stack_reserved_bytes, heap_allocated_bytes, free_memory_bytes, fragmentation, peak_bytes_in_use)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::MemoryAggregationStats)
    initpos = position(e.io)
    x.stack_reserved_bytes != zero(Int64) && PB.encode(e, 1, x.stack_reserved_bytes)
    x.heap_allocated_bytes != zero(Int64) && PB.encode(e, 2, x.heap_allocated_bytes)
    x.free_memory_bytes != zero(Int64) && PB.encode(e, 3, x.free_memory_bytes)
    x.fragmentation !== zero(Float64) && PB.encode(e, 4, x.fragmentation)
    x.peak_bytes_in_use != zero(Int64) && PB.encode(e, 5, x.peak_bytes_in_use)
    return position(e.io) - initpos
end
function PB._encoded_size(x::MemoryAggregationStats)
    encoded_size = 0
    x.stack_reserved_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.stack_reserved_bytes, 1))
    x.heap_allocated_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.heap_allocated_bytes, 2))
    x.free_memory_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.free_memory_bytes, 3))
    x.fragmentation !== zero(Float64) && (encoded_size += PB._encoded_size(x.fragmentation, 4))
    x.peak_bytes_in_use != zero(Int64) && (encoded_size += PB._encoded_size(x.peak_bytes_in_use, 5))
    return encoded_size
end

struct ActiveAllocation
    snapshot_index::Int64
    special_index::Int64
    num_occurrences::Int64
end
ActiveAllocation(;snapshot_index = zero(Int64), special_index = zero(Int64), num_occurrences = zero(Int64)) = ActiveAllocation(snapshot_index, special_index, num_occurrences)
PB.default_values(::Type{ActiveAllocation}) = (;snapshot_index = zero(Int64), special_index = zero(Int64), num_occurrences = zero(Int64))
PB.field_numbers(::Type{ActiveAllocation}) = (;snapshot_index = 1, special_index = 2, num_occurrences = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ActiveAllocation})
    snapshot_index = zero(Int64)
    special_index = zero(Int64)
    num_occurrences = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            snapshot_index = PB.decode(d, Int64)
        elseif field_number == 2
            special_index = PB.decode(d, Int64)
        elseif field_number == 3
            num_occurrences = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return ActiveAllocation(snapshot_index, special_index, num_occurrences)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ActiveAllocation)
    initpos = position(e.io)
    x.snapshot_index != zero(Int64) && PB.encode(e, 1, x.snapshot_index)
    x.special_index != zero(Int64) && PB.encode(e, 2, x.special_index)
    x.num_occurrences != zero(Int64) && PB.encode(e, 3, x.num_occurrences)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ActiveAllocation)
    encoded_size = 0
    x.snapshot_index != zero(Int64) && (encoded_size += PB._encoded_size(x.snapshot_index, 1))
    x.special_index != zero(Int64) && (encoded_size += PB._encoded_size(x.special_index, 2))
    x.num_occurrences != zero(Int64) && (encoded_size += PB._encoded_size(x.num_occurrences, 3))
    return encoded_size
end

struct MemoryActivityMetadata
    memory_activity::MemoryActivity.T
    requested_bytes::Int64
    allocation_bytes::Int64
    address::UInt64
    tf_op_name::String
    step_id::Int64
    region_type::String
    data_type::String
    tensor_shape::String
end
MemoryActivityMetadata(;memory_activity = MemoryActivity.UNKNOWN_ACTIVITY, requested_bytes = zero(Int64), allocation_bytes = zero(Int64), address = zero(UInt64), tf_op_name = "", step_id = zero(Int64), region_type = "", data_type = "", tensor_shape = "") = MemoryActivityMetadata(memory_activity, requested_bytes, allocation_bytes, address, tf_op_name, step_id, region_type, data_type, tensor_shape)
PB.default_values(::Type{MemoryActivityMetadata}) = (;memory_activity = MemoryActivity.UNKNOWN_ACTIVITY, requested_bytes = zero(Int64), allocation_bytes = zero(Int64), address = zero(UInt64), tf_op_name = "", step_id = zero(Int64), region_type = "", data_type = "", tensor_shape = "")
PB.field_numbers(::Type{MemoryActivityMetadata}) = (;memory_activity = 1, requested_bytes = 2, allocation_bytes = 3, address = 4, tf_op_name = 5, step_id = 6, region_type = 7, data_type = 8, tensor_shape = 9)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:MemoryActivityMetadata})
    memory_activity = MemoryActivity.UNKNOWN_ACTIVITY
    requested_bytes = zero(Int64)
    allocation_bytes = zero(Int64)
    address = zero(UInt64)
    tf_op_name = ""
    step_id = zero(Int64)
    region_type = ""
    data_type = ""
    tensor_shape = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            memory_activity = PB.decode(d, MemoryActivity.T)
        elseif field_number == 2
            requested_bytes = PB.decode(d, Int64)
        elseif field_number == 3
            allocation_bytes = PB.decode(d, Int64)
        elseif field_number == 4
            address = PB.decode(d, UInt64)
        elseif field_number == 5
            tf_op_name = PB.decode(d, String)
        elseif field_number == 6
            step_id = PB.decode(d, Int64)
        elseif field_number == 7
            region_type = PB.decode(d, String)
        elseif field_number == 8
            data_type = PB.decode(d, String)
        elseif field_number == 9
            tensor_shape = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return MemoryActivityMetadata(memory_activity, requested_bytes, allocation_bytes, address, tf_op_name, step_id, region_type, data_type, tensor_shape)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::MemoryActivityMetadata)
    initpos = position(e.io)
    x.memory_activity != MemoryActivity.UNKNOWN_ACTIVITY && PB.encode(e, 1, x.memory_activity)
    x.requested_bytes != zero(Int64) && PB.encode(e, 2, x.requested_bytes)
    x.allocation_bytes != zero(Int64) && PB.encode(e, 3, x.allocation_bytes)
    x.address != zero(UInt64) && PB.encode(e, 4, x.address)
    !isempty(x.tf_op_name) && PB.encode(e, 5, x.tf_op_name)
    x.step_id != zero(Int64) && PB.encode(e, 6, x.step_id)
    !isempty(x.region_type) && PB.encode(e, 7, x.region_type)
    !isempty(x.data_type) && PB.encode(e, 8, x.data_type)
    !isempty(x.tensor_shape) && PB.encode(e, 9, x.tensor_shape)
    return position(e.io) - initpos
end
function PB._encoded_size(x::MemoryActivityMetadata)
    encoded_size = 0
    x.memory_activity != MemoryActivity.UNKNOWN_ACTIVITY && (encoded_size += PB._encoded_size(x.memory_activity, 1))
    x.requested_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.requested_bytes, 2))
    x.allocation_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.allocation_bytes, 3))
    x.address != zero(UInt64) && (encoded_size += PB._encoded_size(x.address, 4))
    !isempty(x.tf_op_name) && (encoded_size += PB._encoded_size(x.tf_op_name, 5))
    x.step_id != zero(Int64) && (encoded_size += PB._encoded_size(x.step_id, 6))
    !isempty(x.region_type) && (encoded_size += PB._encoded_size(x.region_type, 7))
    !isempty(x.data_type) && (encoded_size += PB._encoded_size(x.data_type, 8))
    !isempty(x.tensor_shape) && (encoded_size += PB._encoded_size(x.tensor_shape, 9))
    return encoded_size
end

struct MemoryProfileSummary
    peak_bytes_usage_lifetime::Int64
    peak_stats::Union{Nothing,MemoryAggregationStats}
    peak_stats_time_ps::Int64
    memory_capacity::Int64
end
MemoryProfileSummary(;peak_bytes_usage_lifetime = zero(Int64), peak_stats = nothing, peak_stats_time_ps = zero(Int64), memory_capacity = zero(Int64)) = MemoryProfileSummary(peak_bytes_usage_lifetime, peak_stats, peak_stats_time_ps, memory_capacity)
PB.default_values(::Type{MemoryProfileSummary}) = (;peak_bytes_usage_lifetime = zero(Int64), peak_stats = nothing, peak_stats_time_ps = zero(Int64), memory_capacity = zero(Int64))
PB.field_numbers(::Type{MemoryProfileSummary}) = (;peak_bytes_usage_lifetime = 1, peak_stats = 2, peak_stats_time_ps = 3, memory_capacity = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:MemoryProfileSummary})
    peak_bytes_usage_lifetime = zero(Int64)
    peak_stats = Ref{Union{Nothing,MemoryAggregationStats}}(nothing)
    peak_stats_time_ps = zero(Int64)
    memory_capacity = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            peak_bytes_usage_lifetime = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, peak_stats)
        elseif field_number == 3
            peak_stats_time_ps = PB.decode(d, Int64)
        elseif field_number == 4
            memory_capacity = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return MemoryProfileSummary(peak_bytes_usage_lifetime, peak_stats[], peak_stats_time_ps, memory_capacity)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::MemoryProfileSummary)
    initpos = position(e.io)
    x.peak_bytes_usage_lifetime != zero(Int64) && PB.encode(e, 1, x.peak_bytes_usage_lifetime)
    !isnothing(x.peak_stats) && PB.encode(e, 2, x.peak_stats)
    x.peak_stats_time_ps != zero(Int64) && PB.encode(e, 3, x.peak_stats_time_ps)
    x.memory_capacity != zero(Int64) && PB.encode(e, 4, x.memory_capacity)
    return position(e.io) - initpos
end
function PB._encoded_size(x::MemoryProfileSummary)
    encoded_size = 0
    x.peak_bytes_usage_lifetime != zero(Int64) && (encoded_size += PB._encoded_size(x.peak_bytes_usage_lifetime, 1))
    !isnothing(x.peak_stats) && (encoded_size += PB._encoded_size(x.peak_stats, 2))
    x.peak_stats_time_ps != zero(Int64) && (encoded_size += PB._encoded_size(x.peak_stats_time_ps, 3))
    x.memory_capacity != zero(Int64) && (encoded_size += PB._encoded_size(x.memory_capacity, 4))
    return encoded_size
end

struct MemoryProfileSnapshot
    time_offset_ps::Int64
    aggregation_stats::Union{Nothing,MemoryAggregationStats}
    activity_metadata::Union{Nothing,MemoryActivityMetadata}
end
MemoryProfileSnapshot(;time_offset_ps = zero(Int64), aggregation_stats = nothing, activity_metadata = nothing) = MemoryProfileSnapshot(time_offset_ps, aggregation_stats, activity_metadata)
PB.default_values(::Type{MemoryProfileSnapshot}) = (;time_offset_ps = zero(Int64), aggregation_stats = nothing, activity_metadata = nothing)
PB.field_numbers(::Type{MemoryProfileSnapshot}) = (;time_offset_ps = 1, aggregation_stats = 2, activity_metadata = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:MemoryProfileSnapshot})
    time_offset_ps = zero(Int64)
    aggregation_stats = Ref{Union{Nothing,MemoryAggregationStats}}(nothing)
    activity_metadata = Ref{Union{Nothing,MemoryActivityMetadata}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            time_offset_ps = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, aggregation_stats)
        elseif field_number == 3
            PB.decode!(d, activity_metadata)
        else
            Base.skip(d, wire_type)
        end
    end
    return MemoryProfileSnapshot(time_offset_ps, aggregation_stats[], activity_metadata[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::MemoryProfileSnapshot)
    initpos = position(e.io)
    x.time_offset_ps != zero(Int64) && PB.encode(e, 1, x.time_offset_ps)
    !isnothing(x.aggregation_stats) && PB.encode(e, 2, x.aggregation_stats)
    !isnothing(x.activity_metadata) && PB.encode(e, 3, x.activity_metadata)
    return position(e.io) - initpos
end
function PB._encoded_size(x::MemoryProfileSnapshot)
    encoded_size = 0
    x.time_offset_ps != zero(Int64) && (encoded_size += PB._encoded_size(x.time_offset_ps, 1))
    !isnothing(x.aggregation_stats) && (encoded_size += PB._encoded_size(x.aggregation_stats, 2))
    !isnothing(x.activity_metadata) && (encoded_size += PB._encoded_size(x.activity_metadata, 3))
    return encoded_size
end

struct PerAllocatorMemoryProfile
    memory_profile_snapshots::Vector{MemoryProfileSnapshot}
    profile_summary::Union{Nothing,MemoryProfileSummary}
    active_allocations::Vector{ActiveAllocation}
    special_allocations::Vector{MemoryActivityMetadata}
    sampled_timeline_snapshots::Vector{MemoryProfileSnapshot}
end
PerAllocatorMemoryProfile(;memory_profile_snapshots = Vector{MemoryProfileSnapshot}(), profile_summary = nothing, active_allocations = Vector{ActiveAllocation}(), special_allocations = Vector{MemoryActivityMetadata}(), sampled_timeline_snapshots = Vector{MemoryProfileSnapshot}()) = PerAllocatorMemoryProfile(memory_profile_snapshots, profile_summary, active_allocations, special_allocations, sampled_timeline_snapshots)
PB.default_values(::Type{PerAllocatorMemoryProfile}) = (;memory_profile_snapshots = Vector{MemoryProfileSnapshot}(), profile_summary = nothing, active_allocations = Vector{ActiveAllocation}(), special_allocations = Vector{MemoryActivityMetadata}(), sampled_timeline_snapshots = Vector{MemoryProfileSnapshot}())
PB.field_numbers(::Type{PerAllocatorMemoryProfile}) = (;memory_profile_snapshots = 1, profile_summary = 2, active_allocations = 3, special_allocations = 4, sampled_timeline_snapshots = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:PerAllocatorMemoryProfile})
    memory_profile_snapshots = PB.BufferedVector{MemoryProfileSnapshot}()
    profile_summary = Ref{Union{Nothing,MemoryProfileSummary}}(nothing)
    active_allocations = PB.BufferedVector{ActiveAllocation}()
    special_allocations = PB.BufferedVector{MemoryActivityMetadata}()
    sampled_timeline_snapshots = PB.BufferedVector{MemoryProfileSnapshot}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, memory_profile_snapshots)
        elseif field_number == 2
            PB.decode!(d, profile_summary)
        elseif field_number == 3
            PB.decode!(d, active_allocations)
        elseif field_number == 4
            PB.decode!(d, special_allocations)
        elseif field_number == 5
            PB.decode!(d, sampled_timeline_snapshots)
        else
            Base.skip(d, wire_type)
        end
    end
    return PerAllocatorMemoryProfile(memory_profile_snapshots[], profile_summary[], active_allocations[], special_allocations[], sampled_timeline_snapshots[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::PerAllocatorMemoryProfile)
    initpos = position(e.io)
    !isempty(x.memory_profile_snapshots) && PB.encode(e, 1, x.memory_profile_snapshots)
    !isnothing(x.profile_summary) && PB.encode(e, 2, x.profile_summary)
    !isempty(x.active_allocations) && PB.encode(e, 3, x.active_allocations)
    !isempty(x.special_allocations) && PB.encode(e, 4, x.special_allocations)
    !isempty(x.sampled_timeline_snapshots) && PB.encode(e, 5, x.sampled_timeline_snapshots)
    return position(e.io) - initpos
end
function PB._encoded_size(x::PerAllocatorMemoryProfile)
    encoded_size = 0
    !isempty(x.memory_profile_snapshots) && (encoded_size += PB._encoded_size(x.memory_profile_snapshots, 1))
    !isnothing(x.profile_summary) && (encoded_size += PB._encoded_size(x.profile_summary, 2))
    !isempty(x.active_allocations) && (encoded_size += PB._encoded_size(x.active_allocations, 3))
    !isempty(x.special_allocations) && (encoded_size += PB._encoded_size(x.special_allocations, 4))
    !isempty(x.sampled_timeline_snapshots) && (encoded_size += PB._encoded_size(x.sampled_timeline_snapshots, 5))
    return encoded_size
end

struct MemoryProfile
    memory_profile_per_allocator::Dict{String,PerAllocatorMemoryProfile}
    num_hosts::Int32
    memory_ids::Vector{String}
    version::Int32
end
MemoryProfile(;memory_profile_per_allocator = Dict{String,PerAllocatorMemoryProfile}(), num_hosts = zero(Int32), memory_ids = Vector{String}(), version = zero(Int32)) = MemoryProfile(memory_profile_per_allocator, num_hosts, memory_ids, version)
PB.reserved_fields(::Type{MemoryProfile}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[4])
PB.default_values(::Type{MemoryProfile}) = (;memory_profile_per_allocator = Dict{String,PerAllocatorMemoryProfile}(), num_hosts = zero(Int32), memory_ids = Vector{String}(), version = zero(Int32))
PB.field_numbers(::Type{MemoryProfile}) = (;memory_profile_per_allocator = 1, num_hosts = 2, memory_ids = 3, version = 5)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:MemoryProfile})
    memory_profile_per_allocator = Dict{String,PerAllocatorMemoryProfile}()
    num_hosts = zero(Int32)
    memory_ids = PB.BufferedVector{String}()
    version = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, memory_profile_per_allocator)
        elseif field_number == 2
            num_hosts = PB.decode(d, Int32)
        elseif field_number == 3
            PB.decode!(d, memory_ids)
        elseif field_number == 5
            version = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return MemoryProfile(memory_profile_per_allocator, num_hosts, memory_ids[], version)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::MemoryProfile)
    initpos = position(e.io)
    !isempty(x.memory_profile_per_allocator) && PB.encode(e, 1, x.memory_profile_per_allocator)
    x.num_hosts != zero(Int32) && PB.encode(e, 2, x.num_hosts)
    !isempty(x.memory_ids) && PB.encode(e, 3, x.memory_ids)
    x.version != zero(Int32) && PB.encode(e, 5, x.version)
    return position(e.io) - initpos
end
function PB._encoded_size(x::MemoryProfile)
    encoded_size = 0
    !isempty(x.memory_profile_per_allocator) && (encoded_size += PB._encoded_size(x.memory_profile_per_allocator, 1))
    x.num_hosts != zero(Int32) && (encoded_size += PB._encoded_size(x.num_hosts, 2))
    !isempty(x.memory_ids) && (encoded_size += PB._encoded_size(x.memory_ids, 3))
    x.version != zero(Int32) && (encoded_size += PB._encoded_size(x.version, 5))
    return encoded_size
end
