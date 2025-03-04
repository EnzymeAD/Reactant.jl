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

"""
  allocatorstats([device])

Return an [`AllocatorStats`](@ref) instance with information about the device specific allocator.

!!! warning
    This method is currently not implemented for the CPU device.
"""
function allocatorstats(device::AbstractDevice=XLA.default_device(XLA.default_backend()))
    ref = Ref{JLAllocatorStats}()
    @ccall MLIR.API.mlir_c.PjRtDeviceGetAllocatorStats(
        device.device::Ptr{Cvoid}, ref::Ptr{Cvoid}
    )::Cvoid
    stats = ref[]

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
