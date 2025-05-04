```@meta
CollapsedDocStrings = true
```

# Core Reactant API

## Compile API

```@docs
Reactant.@compile
Reactant.@jit
```

## ReactantCore API

```@docs
within_compile
```

```@docs
@trace
```

## Inspect Generated HLO

```@docs
@code_hlo
@code_mhlo
@code_xla
@mlir_visualize
```

## Profile XLA

Reactant can hook into XLA's profiler to generate compilation and execution traces.
See the [profiling tutorial](@ref profiling) for more details.

```@docs
Reactant.Profiler.with_profiler
Reactant.Profiler.annotate
Reactant.Profiler.@annotate
```

## Devices

```@docs
Reactant.devices
Reactant.addressable_devices
```

## Internal utils

```@docs
ReactantCore.materialize_traced_array
```
