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
```

## Tracing customization

```@docs
Reactant.@skip_rewrite_func
Reactant.@skip_rewrite_type
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

## Differentiation Specific API

```@docs
Reactant.ignore_derivatives
```

## Internal utils

```@docs
ReactantCore.materialize_traced_array
```
