"""
    with_profiler(f, trace_output_dir::String; trace_device=true, trace_host=true)

Runs the provided function under a profiler for XLA (similar to [JAX's profiler](https://jax.readthedocs.io/en/latest/profiling.html)).
The traces will be exported in the provided folder and can be seen
using tools like [perfetto.dev](https://ui.perfetto.dev). It will return the return values
from the function.

```julia
with_profiler("./traces/") do
    compiled_func = @compile myfunc(x, y, z)
    compiled_func(x, y, z)
end
```
"""
function with_profiler(f, trace_output_dir::String; trace_device=true, trace_host=true)
    # TODO: we should be able to inject traces from Julia to fill in the blank spots in the trace.

    device_tracer_level = UInt32(trace_device ? 1 : 0)
    host_tracer_level = UInt32(trace_host ? 2 : 0)
    profiler = @ccall Reactant.MLIR.API.mlir_c.CreateProfilerSession(
        device_tracer_level::UInt32, host_tracer_level::UInt32
    )::Ptr{Cvoid}
    try
        f()
    finally
        @ccall Reactant.MLIR.API.mlir_c.ProfilerSessionCollectData(
            profiler::Ptr{Cvoid}, trace_output_dir::Cstring
        )::Cvoid
        @ccall Reactant.MLIR.API.mlir_c.ProfilerSessionDelete(profiler::Ptr{Cvoid})::Cvoid
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
