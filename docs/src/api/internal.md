```@meta
CollapsedDocStrings = true
```

# Internal API

!!! danger "Private"

    These functions are not part of the public API and are subject to change at any time.

```@docs
Reactant.REDUB_ARGUMENTS_NAME
Reactant.Compiler.codegen_unflatten!
Reactant.Compiler.codegen_flatten!
Reactant.Compiler.codegen_xla_call
Reactant.synchronize
```

## Other Docstrings

!!! warning "Private"

    These docstrings are present here to prevent missing docstring warnings. For official
    Enzyme documentation checkout https://enzymead.github.io/Enzyme.jl/stable/.

```@autodocs
Modules = [EnzymeCore, EnzymeCore.EnzymeRules]
```
