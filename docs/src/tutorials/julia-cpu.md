# [Julia as backend](@id julia-cpu)

`julia-cpu` is a Reactant backend written entirely in Julia. Like the `cpu`, `cuda` and `tpu` backends, it is a PJRT
plugin, so Reactant talks to it through the same PJRT C API. But it doesn't compile programs to machine code: it
translates each StableHLO op of a program into a call to a Julia method, and runs these calls one after the other on the
CPU.

This makes it useful for:

- **Debugging**: Every op runs as ordinary Julia code, so you can inspect intermediate values and profile a program with
  Julia's own tools.
- **Writing new backends**: It is the reference implementation of [`Reactant.XLA.PJRT.Plugin`](@ref pjrt-plugin-api),
  the Julia interface for writing PJRT plugins without touching the C API.

!!! warning "Experimental"

    The JuliaCPU backend is experimental. It runs on a single CPU device, supports a subset of StableHLO, and is much
    slower than the `cpu` backend. Don't use it for performance.

## Selecting the backend

`julia-cpu` is registered with the other backends when Reactant is loaded. It has the same priority as `cpu`, which is
registered first, so `cpu` stays the default and you have to select `julia-cpu` explicitly. The simplest way is to make
it the default backend:

```julia
using Reactant

Reactant.set_default_backend("julia-cpu")
@show Reactant.XLA.platform_name(Reactant.XLA.default_backend()) # "julia-cpu"
```

To keep the default backend unchanged, place the arrays on `julia-cpu` instead. [`@jit`](@ref) and
[`@compile`](@ref) run a function on the device of its arguments:

```julia
client = Reactant.XLA.client("julia-cpu")

x = Reactant.to_rarray(rand(Float32, 4, 3); client)
y = Reactant.to_rarray(rand(Float32, 4, 3); client)

@jit f(x, y) # runs on julia-cpu
```

All the arguments of a call must be on the same device, so you can't mix `julia-cpu` and `cpu` arrays in one call.

To run Reactant's test suite on `julia-cpu`, set `REACTANT_BACKEND_GROUP=julia-cpu`. Test files that need features the
backend doesn't support are skipped (see `test/runtests.jl`).

## Limitations

- **Unsupported operations**: Every StableHLO op that isn't deprecated has a handler, except these:
  - Multi-device: `all_gather`, `all_reduce`, `all_to_all`, `collective_broadcast`, `collective_permute`,
    `collective_reduce`, `reduce_scatter`, `partition_id`, `replica_id`
  - Host communication: `send`, `recv`, `infeed`, `outfeed`
  - Asynchronous execution: `async_start`, `async_done`
  - Quantization: `uniform_quantize`, `uniform_dequantize`
  - Other: `custom_call`, `select_and_scatter`

  Compiling a program with any of them fails with an `UNIMPLEMENTED` error such as
  ``"`stablehlo.replica_id` is not supported by `JuliaClient`"``.
- **StableHLO only**: Besides StableHLO, the backend only supports `func.call`. Programs with ops from other dialects
  (e.g. `enzymexla`) never reach the backend: XLA fails to serialize them into a StableHLO portable artifact, with an
  error like

  ```
  Failed to serialize StableHLO with mixed dialects to plugin version ...; found unstable op: ...
  ````

- **Single device and process** There is no sharding and no multi-host support. Programs with more than one replica
  or partition are rejected.
- **No custom calls (yet)**: This rules out [`Reactant.Ops.julia_callback`](@ref), KernelAbstractions and CUDA kernels,
  Triton, and the linear algebra that Reactant lowers into BLAS and LAPACK calls for `cpu` (e.g. most factorizations
  and solvers).
- **No executable serialization or fingerprints, and no topology description (yet)**: Compiled executables can't be
  serialized, and programs can't be compiled ahead of time without a client.
- **Dense host data only**: Host buffers must be dense, in major-to-minor order. Strided host layouts are rejected.
- **No buffer donation (yet)**: Outputs are always new buffers, copied from the handlers' results, since they may alias
  inputs or constants embedded in the generated code.
- **Slow (yet)**: Ops run one at a time with no fusion, most handlers allocate their result, and some are plain reference
  implementations (e.g. `fft` falls back to an O(n²) transform for odd lengths). Also, each compiled program defines a
  new method of `execute`, which stays alive for the rest of the session.
- **No device trace when profiling (yet)**: The plugin implements the PJRT profiler extension, so profiling sessions (e.g. [`Reactant.Profiler.with_profiler`](@ref)) work with `julia-cpu`. But the plugin doesn't record its activity yet, so the profiles have no device trace, FLOPs or device time for it.

Many of these limitations may be fixed in the future.

## Under the hood: the Plugin interface

`Reactant.XLA.PJRT.Plugin` implements the PJRT C API in Julia. Each function of the C API is a thin wrapper that
converts its C arguments, calls a Julia interface method, and converts the result back. A plugin like JuliaCPU
never deals with the C API: it only defines subtypes of the abstract types of `Plugin` and implements their interface
functions. The [PJRT Plugin API](@ref pjrt-plugin-api) page documents every interface.

| Abstract type       | What the interface expects                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `AbstractPlugin`    | The entry point: creates clients and topologies, and reports the plugin's attributes              |
| `Client`            | Reports the platform, devices and memories; creates buffers, and compiles and loads programs      |
| `DeviceDescription` | Static information of a device: id, kind, process index and attributes                            |
| `Device`            | A device at runtime: its description, its memories (and the default one), and memory stats        |
| `Memory`            | A memory space: its id, kind, and the devices that can address it                                 |
| `Buffer`            | An array on a memory: element type, size, layout, and copies to the host and other devices        |
| `Executable`        | A device-independent compiled program: name, replicas/partitions, outputs and serialization       |
| `LoadedExecutable`  | A program loaded on devices, returned by `compile`: its `Executable`, devices and `execute`       |
| `Profiler`          | Only with `ProfilerExtension`: one per profiling session, which is started, stopped and collected |

`Error` and `Event` are provided by `Plugin` and shared by all plugins, so JuliaCPU doesn't reimplement them.

### Memory ownership

`Plugin` relies on Julia's GC to manage the memory of the objects it hands to C: they are rooted globally, and passed to
C with `pointer_from_objref` and back with `unsafe_pointer_to_objref`. This keeps the code simple and easy to extend
from Julia, and the GC keeps track of the memory, at the cost of a dynamic dispatch per call. This design has some
consequences for plugin authors:

- **Objects handed to C must be `mutable struct`s**, since only they have a stable address. The C wrappers root them
  when they are handed to C and unroot them when C destroys them, so plugins never root objects themselves.
- **Plugin types have no layout requirements**: Some PJRT objects start with a pointer to their function table (e.g.
  `PJRT_Error` and `PJRT_Memory`). `Plugin` represents them with its own types (`Error`, and a shim around each
  `Memory`), which share a single function table that dispatches in Julia.
- **Strings and lists of objects must be the same on every call**: C keeps pointers to the strings and lists that
  plugins return (e.g. `platform_name` and `devices`). The wrappers keep a copy of each one for as long as the object
  that returned it is alive.
  - For lists of objects, the copy holds pointers to the objects, not the objects. A plugin must therefore keep the
    objects alive itself (e.g. the client owns its devices and memories), and return the same objects every time, since
    C identifies them by pointer.
- **No moving GC**: A moving GC (e.g. MMTk) would invalidate the pointers returned by `pointer_from_objref`.

### Building the API table

[`make_api`](@ref Reactant.XLA.PJRT.Plugin.make_api) builds the `PJRT_Api` function table of a plugin, which is what
XLA loads. Here's how JuliaCPU sets itself up (abridged from `src/accelerators/JuliaCPU/JuliaCPU.jl`):

```julia
struct JuliaPlugin <: AbstractPlugin end

create_client(::JuliaPlugin, ::Dict{String,Any}) = JuliaClient()

const plugin_api = Ref{PluginApi}()

function make_pjrt_client(; kwargs...)
    errstr = Ref{Cstring}()
    api = unsafe_convert(Ptr{CAPI.PJRT_Api}, plugin_api[])
    ptr = API.MakeClientFromApi(api, "julia-cpu", "JULIA-CPU", errstr)
    ptr == C_NULL && error(unsafe_string(errstr[]))
    return Reactant.XLA.PJRT.Client(ptr)
end

function __init__()
    plugin_api[] = make_api(JuliaPlugin(); extensions=[ProfilerExtension])

    Reactant.Accelerators.Registration.register_backend(
        "julia-cpu";
        priority=100,
        pjrt_initialize_function=make_pjrt_client,
        ifrt_initialize_function=make_ifrt_client,
    )
    return nothing
end
```

- **The plugin type must be stateless**, i.e. a `struct` without fields. Functions that have no object to dispatch on
  (e.g. client creation) dispatch on the plugin type. Plugin-wide state belongs in globals of the plugin's module.
- **Each plugin gets its own table**: the entries that dispatch on the plugin type (e.g. client creation) are
  `@cfunction`s of singleton callables parametrized on it. Julia compiles these to plain function pointers, not
  closures, so `make_api` also works where closure `@cfunction`s are not supported (e.g. AArch64).
- **The returned `PluginApi` must stay alive** for as long as any client created through it exists. Store it in a
  global set from `__init__`, as above.
- **Extensions are opt-in**: XLA only uses an extension that is in the table, so pass only the ones whose interface
  the plugin implements: [`ProfilerExtension`](@ref Reactant.XLA.PJRT.Plugin.ProfilerExtension),
  [`MemoryDescriptionsExtension`](@ref Reactant.XLA.PJRT.Plugin.MemoryDescriptionsExtension),
  [`FFIExtension`](@ref Reactant.XLA.PJRT.Plugin.FFIExtension) or
  [`LayoutsExtension`](@ref Reactant.XLA.PJRT.Plugin.LayoutsExtension).

`MakeClientFromApi` registers the table with XLA under the given device type and creates a client from it. XLA wraps
that client like any other PJRT plugin's, so the rest of Reactant, including the IFRT runtime, uses `julia-cpu` like any
other backend.

To write a backend of your own, the files in `src/accelerators/JuliaCPU/` are a good starting point: each one implements
one interface of `Plugin`.
