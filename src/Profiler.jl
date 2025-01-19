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
with_profiler("./traces/") do
    compiled_func = @compile myfunc(x, y, z)
    compiled_func(x, y, z)
end
```
"""
function with_profiler(
    f,
    trace_output_dir::String;
    trace_device=true,
    trace_host=true,
    create_perfetto_link=false,
)
    # TODO: we should be able to inject traces from Julia to fill in the blank spots in the trace.

    device_tracer_level = UInt32(trace_device ? 1 : 0)
    host_tracer_level = UInt32(trace_host ? 2 : 0)
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

export with_profiler

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

end # module Profiler
