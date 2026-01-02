module Profiler

import ..Reactant
using Sockets: Sockets
using JSON3: JSON3
using PrettyTables: PrettyTables, pretty_table
using Crayons: Crayon

const GRPC_SERVER_STARTED = Ref{Bool}(false)

"""
    with_profiler(f, trace_output_dir::String; trace_device=true, trace_host=true, create_perfetto_link=false)

Runs the provided function under a profiler for XLA (similar to [JAX's profiler](https://jax.readthedocs.io/en/latest/profiling.html)).
The traces will be exported in the provided folder and can be seen
using tools like [perfetto.dev](https://ui.perfetto.dev). It will return the return values
from the function. The `create_perfetto_link` parameter can be used
to automatically generate a perfetto url to visualize the trace.

```julia
compiled_func = with_profiler("./traces") do
    @compile sync=true myfunc(x, y, z)
end

with_profiler("./traces/") do
    compiled_func(x, y, z)
end
```

!!! note
    When profiling compiled functions make sure to [`Reactant.Compiler.@compile`](@ref) with the `sync=true` option so that the compiled execution is captured by the profiler.

"""
function with_profiler(
    f,
    trace_output_dir::String;
    trace_device=true,
    trace_host=true,
    create_perfetto_link=false,
)
    device_tracer_level =
        trace_device isa Bool ? UInt32(trace_device ? 1 : 0) : UInt32(trace_device)
    host_tracer_level =
        trace_host isa Bool ? UInt32(trace_host ? 2 : 0) : UInt32(trace_host)
    profiler = @ccall Reactant.MLIR.API.mlir_c.CreateProfilerSession(
        device_tracer_level::UInt32, host_tracer_level::UInt32
    )::Ptr{Cvoid}

    results = try
        f()
    finally
        @ccall Reactant.MLIR.API.mlir_c.ProfilerSessionCollectData(
            profiler::Ptr{Cvoid}, trace_output_dir::Cstring
        )::Cvoid
        @ccall Reactant.MLIR.API.mlir_c.ProfilerSessionDelete(profiler::Ptr{Cvoid})::Cvoid
    end

    if create_perfetto_link
        traces_path = joinpath(trace_output_dir, "plugins", "profile")
        date = maximum(readdir(traces_path))
        traces_path = joinpath(traces_path, date)

        filename = first(f for f in readdir(traces_path) if endswith(f, ".trace.json.gz"))
        serve_to_perfetto(joinpath(traces_path, filename))
    end

    return results
end

# https://github.com/google/tsl/blob/ffeadbc9111309a845ab07df3ff41d59cb005afb/tsl/profiler/lib/traceme.h#L49-L53
const TRACE_ME_LEVEL_CRITICAL = Cint(1)
const TRACE_ME_LEVEL_INFO = Cint(2)
const TRACE_ME_LEVEL_VERBOSE = Cint(3)

function traceme_encode(name::String, metadata::Dict{String,<:Any})
    isempty(metadata) && return name
    encoded = IOBuffer()
    print(encoded, name, '#')
    first = true
    for (k, v) in metadata
        first || print(encoded, ',')
        print(encoded, k, '=', string(v))
        first = false
    end
    print(encoded, '#')
    return String(take!(encoded))
end

"""
    profiler_activity_start(name::String, level::Cint[, metadata::Dict{String, <:Any}])
    profiler_activity_start(name::String, level::Cint[, metadata::Pair{String, <:Any}...])

Start a profiler activity with metadata (key-value pairs). The metadata will be encoded
into the trace event and can be viewed in profiling tools like Perfetto.

Returns an activity ID that should be passed to `profiler_activity_end` when the activity ends.

# Example
```julia
id = profiler_activity_start("my_operation", TRACE_ME_LEVEL_INFO,
                             "key1" => "value1", "key2" => 42)
# ... do work ...
profiler_activity_end(id)
```
"""
function profiler_activity_start(name::String, level::Cint)
    return @ccall Reactant.MLIR.API.mlir_c.ProfilerActivityStart(
        name::Cstring, level::Cint
    )::Int64
end

function profiler_activity_start(name::String, level::Cint, ::Nothing)
    return profiler_activity_start(name, level)
end

function profiler_activity_start(name::String, level::Cint, metadata::Dict{String,<:Any})
    return profiler_activity_start(traceme_encode(name, metadata), level)
end

function profiler_activity_start(name::String, level::Cint, metadata::Pair{String,<:Any}...)
    return profiler_activity_start(name, level, Dict(metadata...))
end

"""
    profiler_activity_end(id::Int64)

End a profiler activity. See [`profiler_activity_start`](@ref) for more information.
"""
function profiler_activity_end(id::Int64)
    return @ccall Reactant.MLIR.API.mlir_c.ProfilerActivityEnd(id::Int64)::Cvoid
end

"""
    annotate(f, name, [level=TRACE_ME_LEVEL_CRITICAL]; [metadata])

Generate an annotation in the current trace. Optionally include metadata as key-value pairs.

# Example
```julia
annotate("my_operation") do
    # ... do work ...
end

annotate("my_operation"; metadata=Dict("key1" => "value1", "key2" => 42)) do
    # ... do work ...
end
```
"""
function annotate(
    f,
    name,
    level=TRACE_ME_LEVEL_CRITICAL;
    metadata::Union{Dict{String,<:Any},Nothing}=nothing,
)
    id = profiler_activity_start(name, level, metadata)
    try
        f()
    finally
        profiler_activity_end(id)
    end
end

"""
    @annotate [name] function foo(a, b, c)
        ...
    end

The created function will generate an annotation in the captured XLA profiles.
"""
macro annotate(name, func_def=nothing)
    noname = isnothing(func_def)
    func_def = something(func_def, name)

    if !Meta.isexpr(func_def, :function)
        error("not a function definition: $func_def")
    end

    name = noname ? string(func_def.args[1].args[1]) : name
    code = func_def.args[2]

    code = quote
        annotate(() -> $(esc(code)), $(esc(name)))
    end

    return Expr(:function, esc(func_def.args[1]), code)
end

function serve_to_perfetto(path_to_trace_file)
    port_hint = 9001
    port, server = Sockets.listenany(port_hint)

    try
        url = "https://ui.perfetto.dev/#!/?url=http://127.0.0.1:$(port)/$(basename(path_to_trace_file))"
        @info "Open $url"
        # open_in_default_browser(url)

        while true
            isopen(server) || break

            io = Sockets.accept(server)
            @debug "Got connection"
            msg = String(readuntil(io, UInt8['\r', '\n', '\r', '\n']))
            @debug "Got request" msg
            if startswith(msg, "OPTIONS")
                isopen(io) || continue
                write(
                    io,
                    """
          HTTP/1.1 501
          Server: Reactant.jl
          Access-Control-Allow-Origin: *
          Content-Length: 0

          """,
                )
                close(io)
                continue
            end
            if startswith(msg, "POST")
                isopen(io) || continue
                write(
                    io,
                    """
          HTTP/1.1 404
          Server: Reactant.jl
          Access-Control-Allow-Origin: *
          Content-Length: 0

          """,
                )
                close(io)
                continue
            end

            file = read(path_to_trace_file)
            file_size = length(file)

            isopen(io) || continue
            write(
                io,
                """
      HTTP/1.1 200
      Server: Reactant.jl
      Access-Control-Allow-Origin: *
      Content-Length: $(file_size)
      Content-Type: application/gzip

      """,
            )

            write(io, file)
            break
        end
    finally
        isopen(server) && close(server)
    end
end

@inline function free_profiler(exec)
    @ccall MLIR.API.mlir_c.ProfilerServerStop(exec.exec::Ptr{Cvoid})::Cvoid
end

mutable struct ProfileServer
    exec::Ptr{Cvoid}

    function ProfileServer(port)
        exec = @ccall Reactant.MLIR.API.mlir_c.ProfilerServerStart(port::Int32)::Ptr{Cvoid}
        @assert exec != C_NULL
        return finalizer(free_profiler, new(exec))
    end
end

"""
    initialize_xprof_stubs(worker_service_address::String)

Initialize XProf stubs for remote profiling. This sets up the worker service address
for connecting to the XProf profiler service.

# Arguments

  - `worker_service_address`: The address of the worker service (e.g., "localhost:9001")
"""
function initialize_xprof_stubs(worker_service_address::String)
    @ccall Reactant.MLIR.API.mlir_c.InitializeXProfStubs(
        worker_service_address::Cstring
    )::Cvoid
    return nothing
end

"""
    start_xprof_grpc_server(port::Integer)

Start an XProf GRPC server on the specified port. This allows remote profiling
connections from tools like TensorBoard.

# Arguments

  - `port`: The port number to start the GRPC server on
"""
function start_xprof_grpc_server(port::Integer)
    @ccall Reactant.MLIR.API.mlir_c.StartGrpcServer(port::Cint)::Cvoid
    return nothing
end

"""
    xspace_to_tools_data(
        xspace_paths::Vector{String}, tool_name::String; options::Dict=Dict()
    )

Convert XSpace profile data to a specific tool format.

# Arguments

  - `xspace_paths`: Vector of paths to XSpace profile directories
  - `tool_name`: Name of the tool to convert to (e.g., "trace_viewer", "tensorflow_stats",
    "overview_page")
  - `options`: Optional dictionary of tool-specific options. Values can be Bool, Int, or
    String.

# Returns

  - `Tuple{Vector{UInt8}, Bool}`: A tuple of (data, is_binary) where data is the converted
    profile data and is_binary indicates whether the data is in binary format.

# Example

```julia
data, is_binary = xspace_to_tools_data(["/path/to/xspace"], "trace_viewer")
```
"""
function xspace_to_tools_data(
    xspace_paths::Vector{String}, tool_name::String; options::Dict=Dict()
)
    # we need to initialize this before using any xprof apis
    initialize_xprof_stubs_and_server()

    # Separate options by type
    bool_keys = String[]
    bool_values = Bool[]
    int_keys = String[]
    int_values = Cint[]
    str_keys = String[]
    str_values = String[]

    for (k, v) in options
        key = string(k)
        if v isa Bool
            push!(bool_keys, key)
            push!(bool_values, v)
        elseif v isa Integer
            push!(int_keys, key)
            push!(int_values, Cint(v))
        elseif v isa AbstractString
            push!(str_keys, key)
            push!(str_values, string(v))
        else
            error("Only Bool/Int/String values are supported. Got $(typeof(v))")
        end
    end

    # Prepare output parameters
    result_data = Ref{Ptr{Cchar}}(C_NULL)
    result_size = Ref{Int64}(0)
    is_binary = Ref{Bool}(false)
    error_ptr = Ref{Ptr{Cchar}}(C_NULL)

    # Convert string arrays to pointer arrays
    xspace_paths_ptrs = Base.unsafe_convert.(Cstring, xspace_paths)
    bool_keys_ptrs =
        isempty(bool_keys) ? Ptr{Cstring}(C_NULL) : Base.unsafe_convert.(Cstring, bool_keys)
    int_keys_ptrs =
        isempty(int_keys) ? Ptr{Cstring}(C_NULL) : Base.unsafe_convert.(Cstring, int_keys)
    str_keys_ptrs =
        isempty(str_keys) ? Ptr{Cstring}(C_NULL) : Base.unsafe_convert.(Cstring, str_keys)
    str_values_ptrs = if isempty(str_values)
        Ptr{Cstring}(C_NULL)
    else
        Base.unsafe_convert.(Cstring, str_values)
    end

    GC.@preserve xspace_paths bool_keys bool_values int_keys int_values str_keys str_values begin
        ret = @ccall Reactant.MLIR.API.mlir_c.XSpaceToToolsData(
            xspace_paths_ptrs::Ptr{Cstring},
            length(xspace_paths)::Int64,
            tool_name::Cstring,
            (isempty(bool_keys) ? C_NULL : bool_keys_ptrs)::Ptr{Cstring},
            (isempty(bool_values) ? C_NULL : bool_values)::Ptr{Bool},
            length(bool_keys)::Int64,
            (isempty(int_keys) ? C_NULL : int_keys_ptrs)::Ptr{Cstring},
            (isempty(int_values) ? C_NULL : int_values)::Ptr{Cint},
            length(int_keys)::Int64,
            (isempty(str_keys) ? C_NULL : str_keys_ptrs)::Ptr{Cstring},
            (isempty(str_values) ? C_NULL : str_values_ptrs)::Ptr{Cstring},
            length(str_keys)::Int64,
            result_data::Ptr{Ptr{Cchar}},
            result_size::Ptr{Int64},
            is_binary::Ptr{Bool},
            error_ptr::Ptr{Ptr{Cchar}},
        )::Cint
    end

    if ret != 0
        if error_ptr[] != C_NULL
            error_msg = unsafe_string(error_ptr[])
            Libc.free(error_ptr[])
            error(error_msg)
        else
            error("XSpaceToToolsData failed with unknown error")
        end
    end

    # Copy data and free C memory
    data = Vector{UInt8}(undef, result_size[])
    if result_size[] > 0 && result_data[] != C_NULL
        unsafe_copyto!(pointer(data), Ptr{UInt8}(result_data[]), result_size[])
        Libc.free(result_data[])
    end
    return data, is_binary[]
end

# Internal APIs to query XProf
function extract_mean_step_time(xplane_file::String, nrepeat::Int)
    @assert ispath(xplane_file) "xplane_file $xplane_file does not exist."

    # try from overview_page first
    try
        res = extract_mean_step_time_from_overview_page(xplane_file, nrepeat)
        res !== nothing && return res
    catch
    end
    @debug "Failed to extract mean step time from overview_page"
    try
        res = extract_mean_step_time_from_hlo_op_profile(xplane_file, nrepeat)
        res !== nothing && return res
    catch
    end
    @debug "Failed to extract mean step time from hlo_op_profile"
    return -1
end

function extract_mean_step_time_from_overview_page(xplane_file::String, ::Int)
    overview_data = JSON3.read(xspace_to_tools_data([xplane_file], "overview_page")[1])
    step_table = overview_data[2]
    cols = step_table["cols"]
    rows = step_table["rows"]

    col_indices = Dict{String,Int}(col["id"] => i for (i, col) in enumerate(cols))
    step_time_idx = get(col_indices, "stepTimeMs", nothing)

    (step_time_idx === nothing || length(rows) == 0) && return nothing

    step_times_ms = Float64[]
    for row in rows
        cells = get(row, "c", [])
        if step_time_idx <= length(cells)
            step_time = get(cells[step_time_idx], "v", 0.0)
            if step_time > 0
                push!(step_times_ms, step_time)
            end
        end
    end

    length(step_times_ms) == 0 && return nothing
    time_nanosec = (sum(step_times_ms) * 1_000_000) / length(step_times_ms)
    return ceil(Int64, time_nanosec)
end

function extract_mean_step_time_from_hlo_op_profile(xplane_file::String, nrepeat::Int)
    data = JSON3.read(xspace_to_tools_data([xplane_file], "op_profile")[1])
    picosec = data["byProgram"]["metrics"]["normalizedTimePs"]
    return (picosec ÷ 1000) ÷ nrepeat
end

# Bind to port 0 to let the OS assign a free port
function get_free_port()
    server = Sockets.listen(Sockets.IPv4(0), 0)
    _, port = Sockets.getsockname(server)
    close(server)
    return port
end

function initialize_xprof_stubs_and_server()
    GRPC_SERVER_STARTED[] && return nothing

    grpc_port = get_free_port()
    initialize_xprof_stubs("0.0.0.0:$(grpc_port)")
    start_xprof_grpc_server(grpc_port)
    GRPC_SERVER_STARTED[] = true
    return nothing
end

function profile_and_get_xplane_file(
    fn::F,
    args...;
    nrepeat::Int=1,
    warmup::Int=1,
    profile_dir::Union{String,Nothing}=nothing,
    kwargs...,
) where {F}
    @assert warmup >= 1 "Warmup must be non-negative."
    @assert nrepeat >= 1 "Nrepeat must be non-negative."
    @assert fn isa Reactant.Compiler.Thunk "Input function was not a compiled thunk."
    @assert fn.compiled_with_sync "Input function was not compiled with `sync=true`. This \
                                   will produce incorrect profiling results, and hence is \
                                   disable."

    profile_dir === nothing && (profile_dir = joinpath(tempdir(), "reactant_profile"))
    mkpath(profile_dir)

    # warmup
    val = fn(args...; kwargs...)
    for _ in 1:(warmup - 1)
        fn(args...; kwargs...)
    end

    # profile
    with_profiler(profile_dir) do
        for i in 1:nrepeat
            annotate("bench"; metadata=Dict("step_num" => i, "_r" => 1)) do
                fn(args...; kwargs...)
            end
        end
    end

    trace_output_dir = joinpath(profile_dir, "plugins", "profile")
    date = maximum(readdir(trace_output_dir))
    traces_path = joinpath(trace_output_dir, date)
    filename = first(f for f in readdir(traces_path) if endswith(f, ".xplane.pb"))
    @assert filename !== nothing "No xplane file found in $traces_path"
    xplane_file = joinpath(traces_path, filename)

    return (; val=val, xplane_file=xplane_file)
end

# https://github.com/openxla/xprof/blob/e2f03b3f236c581ec2ce70a548b753546f587c3d/plugin/xprof/protobuf/memory_profile.proto#L75
struct MemoryAggregationStats
    stack_reserved_bytes::Int64
    heap_allocated_bytes::Int64
    free_memory_bytes::Int64
    fragmentation::Float64
    peak_bytes_in_use::Int64
end

function _show_with_indent(io, stats::MemoryAggregationStats, indent=0)
    print(
        io,
        "    "^indent *
        "stack_reserved_bytes = $(Base.format_bytes(stats.stack_reserved_bytes)), ",
    )
    Base.printstyled(" # memory usage by stack reservation\n"; color=:light_black)
    print(
        io,
        "    "^indent *
        "heap_allocated_bytes = $(Base.format_bytes(stats.heap_allocated_bytes)), ",
    )
    Base.printstyled(" # memory usage by heap allocation\n"; color=:light_black)
    print(
        io,
        "    "^indent *
        "free_memory_bytes = $(Base.format_bytes(stats.free_memory_bytes)), ",
    )
    Base.printstyled(
        " # free memory available for allocation or reservation\n"; color=:light_black
    )
    print(io, "    "^indent * "fragmentation = $(stats.fragmentation), ")
    Base.printstyled(" # fragmentation of memory within [0, 1]\n"; color=:light_black)
    print(
        io,
        "    "^indent * "peak_bytes_in_use = $(Base.format_bytes(stats.peak_bytes_in_use))",
    )
    Base.printstyled(
        " # The peak memory usage over the entire program\n"; color=:light_black
    )
    return nothing
end

function Base.show(io::IO, stats::MemoryAggregationStats)
    println(io, "MemoryAggregationStats(")
    _show_with_indent(io, stats, 1)
    print(io, ")")
    return nothing
end

struct MemoryProfileSummary
    peak_bytes_usage_lifetime::Int64
    peak_stats::MemoryAggregationStats
    peak_stats_time_ps::Int64
    memory_capacity::Int64
end

function _show_with_indent(io, summary::MemoryProfileSummary, indent=0)
    print(
        io,
        "    "^indent *
        "peak_bytes_usage_lifetime = $(Base.format_bytes(summary.peak_bytes_usage_lifetime)), ",
    )
    Base.printstyled(
        " # peak memory usage over the entire program (lifetime of memory allocator)\n";
        color=:light_black,
    )
    println(io, "    "^indent * "peak_stats = MemoryAggregationStats(")
    _show_with_indent(io, summary.peak_stats, indent + 1)
    println(io, "    "^indent * ")")
    println(
        io,
        "    "^indent *
        "peak_stats_time = $(_timestr(summary.peak_stats_time_ps * 1e-3))s, ",
    )
    print(
        io,
        "    "^indent * "memory_capacity = $(Base.format_bytes(summary.memory_capacity))",
    )
    Base.printstyled(" # memory capacity of the allocator\n"; color=:light_black)
    return nothing
end

function Base.show(io::IO, summary::MemoryProfileSummary)
    println(io, "MemoryProfileSummary(")
    _show_with_indent(io, summary, 1)
    println(io, ")")
    return nothing
end

function get_aggregate_memory_statistics(xplane_file::String)
    data = JSON3.read(xspace_to_tools_data([xplane_file], "memory_profile")[1])
    memory_data = Dict{Symbol,MemoryProfileSummary}()
    for (k, v) in data[:memoryProfilePerAllocator]
        profile_summary = v[:profileSummary]
        memory_data[k] = MemoryProfileSummary(
            parse(Int64, profile_summary[:peakBytesUsageLifetime]),
            MemoryAggregationStats(
                parse(Int64, profile_summary[:peakStats][:stackReservedBytes]),
                parse(Int64, profile_summary[:peakStats][:heapAllocatedBytes]),
                parse(Int64, profile_summary[:peakStats][:freeMemoryBytes]),
                profile_summary[:peakStats][:fragmentation],
                parse(Int64, profile_summary[:peakStats][:peakBytesInUse]),
            ),
            parse(Int64, profile_summary[:peakStatsTimePs]),
            parse(Int64, profile_summary[:memoryCapacity]),
        )
    end
    return memory_data
end

struct FlopsSummary
    Flops::Float64
    UncappedFlops::Float64
    RawFlops::Float64
    BF16Flops::Float64
end

function _show_with_indent(io, summary::FlopsSummary, indent=0)
    print(io, "    "^indent * "Flops = $(summary.Flops), ")
    Base.printstyled(
        " # [flops / (peak flops * program time)], capped at 1.0\n"; color=:light_black
    )
    println(io, "    "^indent * "UncappedFlops = $(summary.UncappedFlops), ")
    print(io, "    "^indent * "RawFlops = $(summary.RawFlops), ")
    Base.printstyled(" # Total FLOPs performed\n"; color=:light_black)
    print(io, "    "^indent * "BF16Flops = $(summary.BF16Flops), ")
    Base.printstyled(
        " # Total FLOPs Normalized to the bf16 (default) devices peak bandwidth\n";
        color=:light_black,
    )
    return nothing
end

function Base.show(io::IO, flops::FlopsSummary)
    println(io, "FlopsSummary(")
    _show_with_indent(io, flops, 1)
    println(io, ")")
    return nothing
end

function get_aggregate_flops_statistics(xplane_file::String, nrepeat::Int)
    data = JSON3.read(xspace_to_tools_data([xplane_file], "op_profile")[1])
    if !haskey(data, :byProgram) || !haskey(data[:byProgram], :metrics)
        return nothing
    end
    return FlopsSummary(
        data[:byProgram][:metrics][:flops],
        data[:byProgram][:metrics][:uncappedFlops],
        data[:byProgram][:metrics][:rawFlops] / nrepeat,
        data[:byProgram][:metrics][:bf16Flops] / nrepeat,
    )
end

struct AggregateProfilingResult
    runtime_ns::Int64
    compile_time_ns::Int64
    memory_data::Dict{Symbol,MemoryProfileSummary}
    flops_data::Union{Nothing,FlopsSummary}
end

_timestr(time_ns) = Base.Ryu.writefixed(Float64(time_ns / 1e9), 8)

function Base.show(io::IO, result::AggregateProfilingResult)
    println(io, "AggregateProfilingResult(")
    println(io, "    runtime = $(_timestr(result.runtime_ns))s, ")
    if !iszero(result.compile_time_ns)
        print(io, "    compile_time = $(_timestr(result.compile_time_ns))s, ")
        Base.printstyled(" # time spent compiling by Reactant\n"; color=:light_black)
    end
    for (k, v) in result.memory_data
        println(io, "    $k = MemoryProfileSummary(")
        _show_with_indent(io, v, 2)
        println(io, "    )")
    end
    if result.flops_data !== nothing
        println(io, "    flops = FlopsSummary(")
        _show_with_indent(io, result.flops_data, 2)
        println(io, "    )")
    end
    print(io, ")")
    return nothing
end

function profile_with_xprof(
    fn::F,
    args...;
    nrepeat::Int=1,
    warmup::Int=1,
    profile_dir::Union{String,Nothing}=nothing,
    compile_options=nothing,
    kwargs...,
) where {F}
    if fn isa Reactant.Compiler.Thunk
        return profile_thunk_with_xprof(
            fn, args...; nrepeat, warmup, profile_dir, compile_time_ns, kwargs...
        )
    end

    compile_options === nothing && (compile_options = Reactant.Compiler.CompileOptions())
    compile_options = Reactant.__compile_options_with_updated_sync(compile_options, true)
    time_start = time_ns()
    compiled_fn = Reactant.compile(fn, args; fn_kwargs=(; kwargs...), compile_options)
    compile_time_ns = Int64(time_ns() - time_start)
    return profile_thunk_with_xprof(
        compiled_fn, args...; nrepeat, warmup, profile_dir, compile_time_ns
    )
end

function profile_thunk_with_xprof(
    fn,
    args...;
    nrepeat::Int=1,
    warmup::Int=1,
    profile_dir::Union{String,Nothing}=nothing,
    compile_time_ns::Int64=0,
    kwargs...,
)
    (; val, xplane_file) = profile_and_get_xplane_file(
        fn, args...; nrepeat, warmup, profile_dir, kwargs...
    )
    memory_data = get_aggregate_memory_statistics(xplane_file)
    flops_data = get_aggregate_flops_statistics(xplane_file, nrepeat)
    runtime_ns = extract_mean_step_time(xplane_file, nrepeat)
    return (;
        val,
        profiling_result=AggregateProfilingResult(
            runtime_ns, compile_time_ns, memory_data, flops_data
        ),
        xplane_file,
    )
end

function _extract_kwargs_from_expr(args...)
    nrepeat = 1
    warmup = 1
    compile_options = nothing
    while length(args) > 1
        if Meta.isexpr(args[1], :(=))
            tn_expr = args[1]
            key, val = tn_expr.args
            key ∈ (:nrepeat, :warmup, :compile_options) || error(
                "@timed supports setting nrepeat, warmup, or compile_options, but got $(tn_expr)",
            )

            if key === :nrepeat
                nrepeat = val
            elseif key === :warmup
                warmup = val
            elseif key === :compile_options
                compile_options = val
            end
            args = args[2:end]
        else
            break
        end
    end

    expr = only(args)
    @assert Meta.isexpr(expr, :call)

    fname = expr.args[1]
    args = expr.args[2:end]
    kwargs = []
    if length(args) ≥ 1 && Meta.isexpr(args[1], :parameters)
        kwargs = args[1].args
        args = args[2:end]
    end
    kw_idxs = findall(Base.Fix2(Meta.isexpr, :kw), args)
    arg_idxs = setdiff(1:length(args), kw_idxs)

    kwargs = (kwargs..., args[kw_idxs]...)
    args = args[arg_idxs]

    return fname, args, kwargs, nrepeat, warmup, compile_options
end

"""
    @timed [nrepeat=1] [warmup=1] [compile_options=nothing] fn(args...; kwargs...)

Profiles the given function and returns the runtime, compile time, and memory data.
`fn` will be compiled with `compile_options` if it is not already a reactant
compiled function.
"""
macro timed(args...)
    fname, args, kwargs, nrepeat, warmup, compile_options = _extract_kwargs_from_expr(
        args...
    )

    return esc(
        quote
            $(profile_with_xprof)(
                $(fname),
                $(args...);
                nrepeat=$(nrepeat),
                warmup=$(warmup),
                compile_options=$(compile_options),
                $(kwargs...),
            ).profiling_result
        end,
    )
end

"""
    @time [nrepeat=1] [warmup=1] [compile_options=nothing] fn(args...; kwargs...)

Profiles the given function and prints the runtime and compile time.
`fn` will be compiled with `compile_options` if it is not already a reactant
compiled function.
"""
macro time(args...)
    fname, args, kwargs, nrepeat, warmup, compile_options = _extract_kwargs_from_expr(
        args...
    )

    return esc(
        quote
            local timed_data = $(profile_with_xprof)(
                $(fname),
                $(args...);
                nrepeat=$(nrepeat),
                warmup=$(warmup),
                compile_options=$(compile_options),
                $(kwargs...),
            )
            println("  runtime: $($(_timestr)(timed_data.profiling_result.runtime_ns))s")
            if !iszero(timed_data.profiling_result.compile_time_ns)
                println("  compile time: \
                         $($(_timestr)(timed_data.profiling_result.compile_time_ns))s")
            end
            timed_data.val
        end,
    )
end

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

function get_kernel_stats(xplane_file::String)
    data = JSON3.read(xspace_to_tools_data([xplane_file], "kernel_stats")[1])

    cols = data[:cols]
    rows = data[:rows]

    # Build column index mapping: column_id => position (1-indexed)
    col_indices = Dict{String,Int}(col[:id] => i for (i, col) in enumerate(cols))

    # Helper to parse "x,y,z" dim strings into Vector{UInt32}
    function parse_dim(dim_str::String)
        parts = split(dim_str, ',')
        return UInt32[parse(UInt32, strip(p)) for p in parts]
    end

    # Helper to get value with fallback column ID
    function get_val_with_fallback(cells, primary_id, fallback_id)
        if haskey(col_indices, primary_id)
            return cells[col_indices[primary_id]][:v]
        elseif haskey(col_indices, fallback_id)
            return cells[col_indices[fallback_id]][:v]
        else
            error("Neither $primary_id nor $fallback_id found in columns")
        end
    end

    # Helper to get duration in ns (check _ns first, then _us with conversion)
    function get_duration_ns(cells, base_name)
        ns_id = "$(base_name)_ns"
        us_id = "$(base_name)_us"
        if haskey(col_indices, ns_id)
            return UInt64(round(cells[col_indices[ns_id]][:v]))
        elseif haskey(col_indices, us_id)
            return UInt64(round(cells[col_indices[us_id]][:v] * 1000))
        else
            error("Neither $ns_id nor $us_id found in columns")
        end
    end

    reports = KernelReport[]
    for row in rows
        cells = row[:c]
        # Extract values by column ID
        get_val(id) = cells[col_indices[id]][:v]

        push!(
            reports,
            KernelReport(
                String(get_val("kernel_name")),
                UInt32(get_val("registers_per_thread")),
                UInt32(get_val_with_fallback(cells, "static_shmem_bytes", "shmem_bytes")),
                UInt32(
                    if get(col_indices, "dynamic_shmem_bytes", 0) != 0
                        get_val("dynamic_shmem_bytes")
                    else
                        0
                    end,
                ),
                parse_dim(String(get_val("block_dim"))),
                parse_dim(String(get_val("grid_dim"))),
                get_duration_ns(cells, "total_duration"),
                get_duration_ns(cells, "min_duration"),
                get_duration_ns(cells, "max_duration"),
                Bool(get_val("is_kernel_using_tensor_core")),
                Bool(get_val("is_op_tensor_core_eligible")),
                String(get_val("op_name")),
                UInt32(get_val("occurrences")),
                Float32(get_val("occupancy_pct")),
            ),
        )
    end

    return reports
end

_clip_str(x, N::Int=50) = length(x) > N ? x[1:N] * "..." : x

function print_kernel_report(reports::Vector{KernelReport}; io::IO=stdout)
    isempty(reports) && return nothing

    # Calculate quantiles based on total_duration_ns
    durations = [r.total_duration_ns for r in reports]
    sorted_durations = sort(durations)
    n = length(sorted_durations)
    q90 = sorted_durations[max(1, ceil(Int, 0.90 * n))]
    q75 = sorted_durations[max(1, ceil(Int, 0.75 * n))]

    # Check which optional columns have data
    has_static_shmem = any(r -> r.static_shmem_bytes > 0, reports)
    has_dynamic_shmem = any(r -> r.dynamic_shmem_bytes > 0, reports)

    # Build column definitions: (header, extractor, is_duration)
    columns = Tuple{String,Function,Bool}[]
    push!(columns, ("Kernel Name", r -> _clip_str(r.name), false))
    push!(columns, ("Occurrences", r -> string(r.occurrences), false))
    push!(columns, ("Total Duration", r -> _timestr(r.total_duration_ns) * "s", true))
    push!(
        columns,
        (
            "Avg Duration",
            r -> begin
                avg = r.occurrences > 0 ? r.total_duration_ns ÷ r.occurrences : UInt64(0)
                _timestr(avg) * "s"
            end,
            true,
        ),
    )
    push!(columns, ("Min Duration", r -> _timestr(r.min_duration_ns) * "s", true))
    push!(columns, ("Max Duration", r -> _timestr(r.max_duration_ns) * "s", true))
    if has_static_shmem
        push!(
            columns, ("Static Shmem", r -> Base.format_bytes(r.static_shmem_bytes), false)
        )
    end
    if has_dynamic_shmem
        push!(
            columns, ("Dynamic Shmem", r -> Base.format_bytes(r.dynamic_shmem_bytes), false)
        )
    end
    push!(columns, ("Block Dim", r -> join(r.block_dim, ","), false))
    push!(columns, ("Grid Dim", r -> join(r.grid_dim, ","), false))
    push!(columns, ("TensorCore", r -> r.is_kernel_using_tensor_core ? "✓" : "✗", false))
    push!(
        columns, ("Occupancy %", r -> string(round(r.occupancy_pct; digits=1)) * "%", false)
    )

    header = [c[1] for c in columns]
    duration_cols = findall(c -> c[3], columns)

    # Build data matrix
    data = Matrix{String}(undef, length(reports), length(columns))
    raw_durations = Matrix{UInt64}(undef, length(reports), 4)

    for (i, r) in enumerate(reports)
        avg_dur_ns = r.occurrences > 0 ? r.total_duration_ns ÷ r.occurrences : UInt64(0)
        raw_durations[i, 1] = r.total_duration_ns
        raw_durations[i, 2] = avg_dur_ns
        raw_durations[i, 3] = r.min_duration_ns
        raw_durations[i, 4] = r.max_duration_ns

        for (j, (_, extractor, _)) in enumerate(columns)
            data[i, j] = extractor(r)
        end
    end

    # Map duration column indices to raw_durations indices
    dur_col_to_raw = Dict(zip(duration_cols, 1:4))

    # Create highlighters using PrettyTables
    hl_top90 = PrettyTables.TextHighlighter(
        (data, i, j) -> j in duration_cols && raw_durations[i, dur_col_to_raw[j]] >= q90,
        Crayon(; foreground=:red, bold=true),
    )
    hl_top75 = PrettyTables.TextHighlighter(
        (data, i, j) ->
            j in duration_cols &&
                raw_durations[i, dur_col_to_raw[j]] >= q75 &&
                raw_durations[i, dur_col_to_raw[j]] < q90,
        Crayon(; foreground=:yellow),
    )

    pretty_table(io, data; column_labels=[header], highlighters=[hl_top90, hl_top75])
    return nothing
end

struct FrameworkOpStats
    host_or_device::String
    op_type::String
    op_name::String
    occurrences::UInt32
    total_time_ns::UInt64
    avg_time_ns::UInt64
    total_self_time_ns::UInt64
    avg_self_time_ns::UInt64
    device_total_self_time_pct::Float64
    device_cumulative_total_self_time_pct::Float64
    host_total_self_time_pct::Float64
    host_cumulative_total_self_time_pct::Float64
    measured_flop_rate::Float64
    model_flop_rate_gflops::Float64
    measured_memory_bw_gbps::Float64
    operational_intensity::Float64
    gpu_tensorcore_utilization::Float64
    bound_by::String
    execution_mode::String
end

function get_framework_op_stats(xplane_file::String; include_idle::Bool=false)
    raw_data = JSON3.read(xspace_to_tools_data([xplane_file], "framework_op_stats")[1])
    length(raw_data) == 2 || return FrameworkOpStats[]

    data = include_idle ? raw_data[1] : raw_data[2]

    cols = data[:cols]
    rows = data[:rows]

    # Build column index mapping: column_id => position (1-indexed)
    col_indices = Dict{String,Int}(col[:id] => i for (i, col) in enumerate(cols))

    reports = FrameworkOpStats[]
    for row in rows
        cells = row[:c]
        function get_val(id, allowmissing=false)
            allowmissing && !haskey(col_indices, id) && return 0
            return cells[col_indices[id]][:v]
        end

        # Convert μs to ns for time fields
        push!(
            reports,
            FrameworkOpStats(
                String(get_val("host_or_device")),
                String(get_val("type")),
                String(get_val("operation")),
                UInt32(get_val("occurrences")),
                UInt64(round(get_val("total_time") * 1000)),      # μs to ns
                UInt64(round(get_val("avg_time") * 1000)),        # μs to ns
                UInt64(round(get_val("total_self_time") * 1000)), # μs to ns
                UInt64(round(get_val("avg_self_time") * 1000)),   # μs to ns
                Float64(get_val("device_total_self_time_percent")),
                Float64(get_val("device_cumulative_total_self_time_percent")),
                Float64(get_val("host_total_self_time_percent")),
                Float64(get_val("Host_cumulative_total_self_time_percent")),
                Float64(get_val("measured_flop_rate")),
                Float64(get_val("model_flop_rate")),
                Float64(get_val("measured_memory_bw")),
                Float64(get_val("operational_intensity")),
                Float64(get_val("gpu_tensorcore_utilization", true)),
                String(get_val("bound_by")),
                String(get_val("eager")),
            ),
        )
    end

    return reports
end

function print_framework_op_stats(reports::Vector{FrameworkOpStats}; io::IO=stdout)
    isempty(reports) && return nothing

    # Calculate quantiles based on total_self_time_ns
    durations = [r.total_self_time_ns for r in reports]
    sorted_durations = sort(durations)
    n = length(sorted_durations)
    q90 = sorted_durations[max(1, ceil(Int, 0.90 * n))]
    q75 = sorted_durations[max(1, ceil(Int, 0.75 * n))]

    # Check which optional columns have data
    has_host_stats = any(r -> r.host_total_self_time_pct > 0, reports)
    has_tensorcore = any(r -> r.gpu_tensorcore_utilization > 0, reports)

    # Build column definitions: (header, extractor, is_duration)
    columns = Tuple{String,Function,Bool}[]
    push!(columns, ("Operation", r -> _clip_str(r.op_name), false))
    push!(columns, ("Type", r -> r.op_type, false))
    push!(columns, ("Host/Device", r -> r.host_or_device, false))
    push!(columns, ("Occurrences", r -> string(r.occurrences), false))
    push!(columns, ("Total Self-Time", r -> _timestr(r.total_self_time_ns) * "s", true))
    push!(columns, ("Avg Self-Time", r -> _timestr(r.avg_self_time_ns) * "s", true))
    push!(
        columns,
        (
            "Device %",
            r -> string(round(r.device_total_self_time_pct * 100; digits=2)) * "%",
            false,
        ),
    )
    if has_host_stats
        push!(
            columns,
            (
                "Host %",
                r -> string(round(r.host_total_self_time_pct * 100; digits=2)) * "%",
                false,
            ),
        )
    end
    push!(
        columns,
        (
            "Memory BW",
            r -> string(round(r.measured_memory_bw_gbps; digits=2)) * " GB/s",
            false,
        ),
    )
    push!(
        columns,
        (
            "FLOP Rate",
            r -> string(round(r.model_flop_rate_gflops; digits=2)) * " GFLOP/s",
            false,
        ),
    )
    if has_tensorcore
        push!(
            columns,
            (
                "TensorCore",
                r -> string(round(r.gpu_tensorcore_utilization * 100; digits=1)) * "%",
                false,
            ),
        )
    end
    push!(columns, ("Bound By", r -> r.bound_by, false))

    header = [c[1] for c in columns]
    duration_cols = findall(c -> c[3], columns)

    # Build data matrix
    data = Matrix{String}(undef, length(reports), length(columns))
    raw_durations = Matrix{UInt64}(undef, length(reports), 2)

    for (i, r) in enumerate(reports)
        raw_durations[i, 1] = r.total_self_time_ns
        raw_durations[i, 2] = r.avg_self_time_ns

        for (j, (_, extractor, _)) in enumerate(columns)
            data[i, j] = extractor(r)
        end
    end

    # Map duration column indices to raw_durations indices
    dur_col_to_raw = Dict(zip(duration_cols, 1:2))

    # Create highlighters using PrettyTables
    hl_top90 = PrettyTables.TextHighlighter(
        (data, i, j) -> j in duration_cols && raw_durations[i, dur_col_to_raw[j]] >= q90,
        Crayon(; foreground=:red, bold=true),
    )
    hl_top75 = PrettyTables.TextHighlighter(
        (data, i, j) ->
            j in duration_cols &&
                raw_durations[i, dur_col_to_raw[j]] >= q75 &&
                raw_durations[i, dur_col_to_raw[j]] < q90,
        Crayon(; foreground=:yellow),
    )

    pretty_table(io, data; column_labels=[header], highlighters=[hl_top90, hl_top75])
    return nothing
end

function _print_summary_header(header::String)
    println("\n╔", "="^80 * "╗")
    println("║ " * header * " "^(79 - length(header)) * "║")
    return println("╚" * "="^80 * "╝")
end

"""
    @profile [nrepeat=1] [warmup=1] [compile_options=nothing] fn(args...; kwargs...)

Profiles the given function and prints detailed kernel and framework op statistics.
`fn` will be compiled with `compile_options` if it is not already a reactant
compiled function.

Returns the result of the function call.
"""
macro profile(args...)
    fname, args, kwargs, nrepeat, warmup, compile_options = _extract_kwargs_from_expr(
        args...
    )

    return esc(
        quote
            local profile_result = $(profile_with_xprof)(
                $(fname),
                $(args...);
                nrepeat=$(nrepeat),
                warmup=$(warmup),
                compile_options=$(compile_options),
                $(kwargs...),
            )

            # Get the xplane file for detailed stats
            local xplane_file = profile_result.xplane_file
            local kernel_stats = $(get_kernel_stats)(xplane_file)
            local framework_stats = $(get_framework_op_stats)(xplane_file)

            if !isempty(kernel_stats)
                $(_print_summary_header)("KERNEL STATISTICS")
                println()
                $(print_kernel_report)(kernel_stats)
            end

            if !isempty(framework_stats)
                $(_print_summary_header)("FRAMEWORK OP STATISTICS")
                println()
                $(print_framework_op_stats)(framework_stats)
            end

            $(_print_summary_header)("SUMMARY")
            println()
            println(profile_result.profiling_result)
            println()

            profile_result.val
        end,
    )
end

export with_profiler, annotate, @annotate, @time, @timed, @profile

end # module Profiler
