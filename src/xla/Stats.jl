# To keep in sync with JLAllocatorStats in ReactantExtra/API.cpp
struct JLAllocatorStats
    num_allocs::Int64
    bytes_in_use::Int64
    peak_bytes_in_use::Int64
    largest_alloc_size::Int64
    bytes_limit::Int64
    bytes_reserved::Int64
    peak_bytes_reserved::Int64
    bytes_reservable_limit::Int64
    largest_free_block_bytes::Int64
    pool_bytes::Int64
    peak_pool_bytes::Int64
end

"""
  AllocatorStats()

Contains the following fields:
  - `num_allocs`
  - `bytes_in_use`
  - `peak_bytes_in_use`
  - `largest_alloc_size`
  - `bytes_limit`
  - `bytes_reserved`
  - `peak_bytes_reserved`
  - `bytes_reservable_limit`
  - `largest_free_block_bytes`
  - `pool_bytes`
  - `peak_pool_bytes`

It should be constructed using the [`allocatorstats`](@ref) function.
"""
struct AllocatorStats
    num_allocs::Int64
    bytes_in_use::Int64
    peak_bytes_in_use::Int64
    largest_alloc_size::Int64
    bytes_limit::Union{Nothing,Int64}
    bytes_reserved::Int64
    peak_bytes_reserved::Int64
    bytes_reservable_limit::Union{Nothing,Int64}
    largest_free_block_bytes::Int64
    pool_bytes::Union{Nothing,Int64}
    peak_pool_bytes::Union{Nothing,Int64}
end

function Base.show(io::IO, ::MIME"text/plain", stats::AllocatorStats)
    return print(
        io,
        """
        AllocatorStats
        --------------
        num_allocs: $(stats.num_allocs)
        bytes_in_use: $(stats.bytes_in_use)
        peak_bytes_in_use: $(stats.peak_bytes_in_use)
        largest_alloc_size: $(stats.largest_alloc_size)
        bytes_limit: $(stats.bytes_limit)
        bytes_reserved: $(stats.bytes_reserved)
        peak_bytes_reserved: $(stats.peak_bytes_reserved)
        bytes_reservable_limit: $(stats.bytes_reservable_limit)
        largest_free_block_bytes: $(stats.largest_free_block_bytes)
        pool_bytes: $(stats.pool_bytes)
        peak_pool_bytes: $(stats.peak_pool_bytes)
        """,
    )
end

"""
  allocatorstats([device])

Return an [`AllocatorStats`](@ref) instance with information about the device specific allocator.

!!! warning

    This method is currently not implemented for the CPU device.
"""
function allocatorstats(device::AbstractDevice=XLA.default_device(XLA.default_backend()))
    stats = allocatorstats_internal(device)
    nullopt = typemin(Int64)
    return AllocatorStats(
        stats.num_allocs,
        stats.bytes_in_use,
        stats.peak_bytes_in_use,
        stats.largest_alloc_size,
        stats.bytes_limit == nullopt ? nothing : stats.bytes_limit,
        stats.bytes_reserved,
        stats.peak_bytes_reserved,
        stats.bytes_reservable_limit == nullopt ? nothing : stats.bytes_reservable_limit,
        stats.largest_free_block_bytes,
        stats.pool_bytes == nullopt ? nothing : stats.pool_bytes,
        stats.peak_pool_bytes == nullopt ? nothing : stats.peak_pool_bytes,
    )
end

function allocatorstats_internal end

# To keep in sync with JLHloCostAnalysisProperties in ReactantExtra/API.cpp
struct HloCostAnalysisProperties
    flops::Cfloat
    transcendentals::Cfloat
    bytes_accessed::Cfloat
    optimal_seconds::Cfloat
    utilization::Cfloat
    operand0_utilization::Cfloat
    operand1_utilization::Cfloat
    operand0_bytes_accessed::Cfloat
    operand1_bytes_accessed::Cfloat
    output_root_bytes_accessed::Cfloat
    reserved0::Cfloat
end

function Base.show(io::IO, ::MIME"text/plain", cost_analysis::HloCostAnalysisProperties)
    return print(
        io,
        """
        HloCostAnalysisProperties
        -------------------------
        FLOPS: $(cost_analysis.flops)
        Transcendentals: $(cost_analysis.transcendentals)
        Bytes Accessed: $(cost_analysis.bytes_accessed)
        Optimal Seconds: $(cost_analysis.optimal_seconds)
        Utilization: $(cost_analysis.utilization)
        Operand 0 Utilization: $(cost_analysis.operand0_utilization)
        Operand 1 Utilization: $(cost_analysis.operand1_utilization)
        Operand 0 Bytes Accessed: $(cost_analysis.operand0_bytes_accessed)
        Operand 1 Bytes Accessed: $(cost_analysis.operand1_bytes_accessed)
        Output Root Bytes Accessed: $(cost_analysis.output_root_bytes_accessed)
        Reserved 0: $(cost_analysis.reserved0)
        """,
    )
end

"""
    cost_analysis(::AbstractLoadedExecutable)
    cost_analysis(::Reactant.Thunk)

Returns a HloCostAnalysisProperties object with the cost analysis of the loaded executable.
"""
function cost_analysis end
