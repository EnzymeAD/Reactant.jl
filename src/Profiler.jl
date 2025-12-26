module Profiler

import ..Reactant
using Sockets: Sockets
using JSON3: JSON3

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

function _timestr(time_ns)
    return Base.Ryu.writefixed(Float64(time_ns / 1e9), 8)
end

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

export with_profiler, annotate, @annotate, @time, @timed

end # module Profiler
