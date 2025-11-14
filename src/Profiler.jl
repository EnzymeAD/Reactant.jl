module Profiler

import ..Reactant
using Sockets: Sockets
using PrettyTables: PrettyTables
using JSON3: JSON3

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

"""
    annotate(f, name, [level=TRACE_ME_LEVEL_CRITICAL])

Generate an annotation in the current trace.
"""
function annotate(f, name, level=TRACE_ME_LEVEL_CRITICAL)
    id = @ccall Reactant.MLIR.API.mlir_c.ProfilerActivityStart(
        name::Cstring, level::Cint
    )::Int64
    try
        f()
    finally
        @ccall Reactant.MLIR.API.mlir_c.ProfilerActivityEnd(id::Int64)::Cvoid
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

export with_profiler, annotate, @annotate

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

function wrap_string(s; width=20)
    s_str = string(s)
    return join([s_str[i:min(i + width - 1, end)] for i in 1:width:length(s_str)], "\n")
end

struct KernelStatsProfileResults
    data
end

function KernelStatsProfileResults(data::JSON3.Object)
    cols = data["cols"]
    rows = data["rows"]
    keys = Tuple(Symbol.(get.(cols, "id")))
    table = Vector{NamedTuple}(undef, length(rows))

    for (i, row) in enumerate(rows)
        vals = get.(row["c"], "v")
        table[i] = NamedTuple{keys}(vals)
    end

    return KernelStatsProfileResults(table)
end

function Base.show(io::IO, r::KernelStatsProfileResults)
    tbl = r.data

    println(io, "╔════════════════╗")
    println(io, "║  Kernel Stats  ║")
    println(io, "╚════════════════╝")

    isempty(tbl) && return nothing

    fields = fieldnames(typeof(tbl[1]))
    wrapped = split.(wrap_string.(fields; width=10), "\n")
    nrows = maximum(length.(wrapped))
    column_labels = [[get(wrapped[j], i, "") for j in 1:length(wrapped)] for i in 1:nrows]

    PrettyTables.pretty_table(
        io,
        tbl;
        line_breaks=true,
        maximum_data_column_widths=10,
        auto_wrap=true,
        column_labels,
    )
    return nothing
end

struct FrameworkStatsProfileResults
    data
end

function FrameworkStatsProfileResults(data::JSON3.Array{JSON3.Object})
    results = Vector{Vector{NamedTuple}}()

    for table in data
        local_result = Vector{NamedTuple}()

        # Extract column information
        cols = table["cols"]
        col_ids = [Symbol(col["id"]) for col in cols]

        # Extract rows
        rows = table["rows"]

        # Parse each row into a NamedTuple
        for row in rows
            values = [cell["v"] for cell in row["c"]]
            nt = NamedTuple{Tuple(col_ids)}(Tuple(values))
            push!(local_result, nt)
        end

        push!(results, local_result)
    end

    return FrameworkStatsProfileResults(results)
end

function Base.show(io::IO, r::FrameworkStatsProfileResults)
    println(io, "╔══════════════════════════════╗")
    println(io, "║  FrameworkOpStatsResults     ║")
    println(io, "╚══════════════════════════════╝")

    isempty(r.data) && return nothing

    for tbl in r.data
        fields = fieldnames(typeof(tbl[1]))
        wrapped = split.(wrap_string.(fields; width=10), "\n")
        nrows = maximum(length.(wrapped))
        column_labels = [
            [get(wrapped[j], i, "") for j in 1:length(wrapped)] for i in 1:nrows
        ]

        PrettyTables.pretty_table(
            io,
            tbl;
            line_breaks=true,
            auto_wrap=true,
            maximum_data_column_widths=10,
            column_labels,
        )
    end

    return nothing
end

struct ReactantProfileResults
    kernel_stats::KernelStatsProfileResults
    framework_stats::FrameworkStatsProfileResults
end

function Base.show(io::IO, r::ReactantProfileResults)
    println(io, "╔═══════════════════════════════════════════════════════╗")
    println(io, "║ Reactant Profile Results                              ║")
    println(io, "╚═══════════════════════════════════════════════════════╝")
    show(io, r.kernel_stats)
    println(io)
    show(io, r.framework_stats)
    return nothing
end

function parse_xprof_profile_data(data)
    extmod = Base.get_extension(Reactant, :ReactantPythonCallExt)
    if extmod === nothing
        error("Currently we require `PythonCall` to be loaded to parse xprof data.")
    end
    kernel_stats = KernelStatsProfileResults(
        JSON3.read(extmod.xspace_to_tools_data(data, "kernel_stats"))
    )
    framework_stats = FrameworkStatsProfileResults(
        JSON3.read(extmod.xspace_to_tools_data(data, "framework_op_stats"))
    )
    return ReactantProfileResults(kernel_stats, framework_stats)
    return nothing
end

macro profile(ex)
    profile_dir = joinpath(tempdir(), "reactant_profile")
    mkpath(profile_dir)

    quote
        # TODO: optionally compile the code first and profile

        $(with_profiler)($(esc(profile_dir))) do
            $(esc(ex))
        end

        trace_output_dir = joinpath($(esc(profile_dir)), "plugins", "profile")
        date = maximum(readdir(trace_output_dir))
        traces_path = joinpath(trace_output_dir, date)

        filename = first(f for f in readdir(traces_path) if endswith(f, ".xplane.pb"))
        data = $(parse_xprof_profile_data)(joinpath(traces_path, filename))
    end
end

end # module Profiler
