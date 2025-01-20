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
@trace
```

## Inspect Generated HLO

```@docs
@code_hlo
```

## Profile XLA

Reactant can hook into XLA's profiler to generate compilation and execution traces.
See the [profiling tutorial](@ref profiling) for more details.

```@docs
Reactant.Profiler.with_profiler
Reactant.Profiler.annotate
Reactant.Profiler.@annotate
```
