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

```@raw html
<br>
```

# Internal Functionality

!!! danger "Private"

    These functions are not part of the public API and are subject to change at any time.

```@docs
Reactant.Compiler.codegen_unflatten!
Reactant.Compiler.codegen_flatten!
Reactant.Compiler.codegen_xla_call
```
