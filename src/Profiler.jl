module Profiler

import ..Reactant
using Sockets: Sockets

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
    metadata::Union{Dict{String,Any},Nothing}=nothing,
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

end # module Profiler
