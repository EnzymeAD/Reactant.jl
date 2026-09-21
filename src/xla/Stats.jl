_format_bytes(x) = x < 0 ? nothing : Base.format_bytes(x)
_format_bytes(x::Nothing) = x

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
        Num Allocs: $(stats.num_allocs)
        In Use: $(_format_bytes(stats.bytes_in_use))
        Peak In Use: $(_format_bytes(stats.peak_bytes_in_use))
        Largest Alloc Size: $(_format_bytes(stats.largest_alloc_size))
        Limit: $(_format_bytes(stats.bytes_limit))
        Reserved: $(_format_bytes(stats.bytes_reserved))
        Peak Reserved: $(_format_bytes(stats.peak_bytes_reserved))
        Reservable Limit: $(_format_bytes(stats.bytes_reservable_limit))
        Largest Free Block: $(_format_bytes(stats.largest_free_block_bytes))
        Pool: $(_format_bytes(stats.pool_bytes))
        Peak Pool: $(_format_bytes(stats.peak_pool_bytes))
        """,
    )
end

"""
  allocatorstats([device])

Return an [`AllocatorStats`](@ref) instance with information about the device specific allocator.

!!! warning

    This method is currently not implemented for the CPU device.
"""
function allocatorstats(device::AbstractDevice=default_device(default_backend()))
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

"""
    clear_memory_stats!([device])

Reset the high-water marks reported by [`allocatorstats`](@ref) (`peak_bytes_in_use`,
`peak_bytes_reserved` and `peak_pool_bytes`) to the current usage, so that the peak
reached by a following region of code can be measured on its own.

!!! warning

    Only devices that track allocator statistics support this (the CUDA and ROCm devices).
    Calling it on any other device, including the CPU device, throws.
"""
function clear_memory_stats!(device::AbstractDevice=default_device(default_backend()))
    clear_memory_stats_internal!(device)
    return nothing
end

function clear_memory_stats_internal! end

"""
    CompiledMemoryStats

The memory XLA's buffer assignment reserved for an executable, in bytes. Construct it with
[`compiled_memory_stats`](@ref). Fields:
  - `generated_code_size_in_bytes`, `argument_size_in_bytes`, `output_size_in_bytes`,
    `alias_size_in_bytes`, `temp_size_in_bytes`: device memory
  - the `host_` prefixed counterparts of the above: host memory
  - `peak_memory_in_bytes`: peak device memory while the program runs
"""
struct CompiledMemoryStats
    generated_code_size_in_bytes::Int64
    argument_size_in_bytes::Int64
    output_size_in_bytes::Int64
    alias_size_in_bytes::Int64
    temp_size_in_bytes::Int64
    host_generated_code_size_in_bytes::Int64
    host_argument_size_in_bytes::Int64
    host_output_size_in_bytes::Int64
    host_alias_size_in_bytes::Int64
    host_temp_size_in_bytes::Int64
    peak_memory_in_bytes::Int64
end

function Base.show(io::IO, ::MIME"text/plain", stats::CompiledMemoryStats)
    return print(
        io,
        """
        CompiledMemoryStats
        -------------------
        Generated Code: $(_format_bytes(stats.generated_code_size_in_bytes))
        Arguments: $(_format_bytes(stats.argument_size_in_bytes))
        Outputs: $(_format_bytes(stats.output_size_in_bytes))
        Aliases: $(_format_bytes(stats.alias_size_in_bytes))
        Temporaries: $(_format_bytes(stats.temp_size_in_bytes))
        Host Generated Code: $(_format_bytes(stats.host_generated_code_size_in_bytes))
        Host Arguments: $(_format_bytes(stats.host_argument_size_in_bytes))
        Host Outputs: $(_format_bytes(stats.host_output_size_in_bytes))
        Host Aliases: $(_format_bytes(stats.host_alias_size_in_bytes))
        Host Temporaries: $(_format_bytes(stats.host_temp_size_in_bytes))
        Peak Memory: $(_format_bytes(stats.peak_memory_in_bytes))
        """,
    )
end

"""
    compiled_memory_stats(exec::AbstractLoadedExecutable)
    compiled_memory_stats(thunk::Reactant.Thunk)

Return a [`CompiledMemoryStats`](@ref) with the memory XLA's buffer assignment reserved for
the executable. The numbers come from the compiler and are available without running the
program. `temp_size_in_bytes` is the scratch space the program needs; the runtime allocator
rounds allocations up and may add its own, so it is a lower bound on the scratch actually
allocated at run time.
"""
function compiled_memory_stats(exec::AbstractLoadedExecutable)
    stats = compiled_memory_stats_internal(exec)
    return CompiledMemoryStats(
        (getfield(stats, field) for field in fieldnames(CompiledMemoryStats))...
    )
end

function compiled_memory_stats_internal end

function Base.show(
    io::IO, ::MIME"text/plain", cost_analysis::MLIR.API.JLHloCostAnalysisProperties
)
    return print(
        io,
        """
        HloCostAnalysisProperties
        -------------------------
        FLOPS: $(cost_analysis.flops)
        Transcendentals: $(cost_analysis.transcendentals)
        Bytes Accessed: $(_format_bytes(cost_analysis.bytes_accessed))
        Optimal Seconds: $(cost_analysis.optimal_seconds)
        Utilization: $(cost_analysis.utilization)
        Operand 0 Utilization: $(cost_analysis.operand0_utilization)
        Operand 1 Utilization: $(cost_analysis.operand1_utilization)
        Operand 0 Bytes Accessed: $(_format_bytes(cost_analysis.operand0_bytes_accessed))
        Operand 1 Bytes Accessed: $(_format_bytes(cost_analysis.operand1_bytes_accessed))
        Output Root Bytes Accessed: $(_format_bytes(cost_analysis.output_root_bytes_accessed))
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
