```@meta
CollapsedDocStrings = true
```

# [PJRT Plugin API](@id pjrt-plugin-api)

`Reactant.XLA.PJRT.Plugin` implements the PJRT C API in Julia, so that a PJRT plugin can be written by implementing the
Julia interfaces below instead of the C API. `Reactant.Accelerators.JuliaCPU` is the reference implementation. See the
[Julia as backend](@ref julia-cpu) tutorial for an overview of how they fit together.

!!! warning

    This API is experimental and may change without deprecation.

## Plugin

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Plugin/Plugin.jl", "Plugin/AbstractPlugin.jl"]
```

## Error

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Plugin/Error.jl"]
```

## Event

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Plugin/Event.jl"]
```

## Client

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Plugin/Client.jl"]
```

## Devices and memories

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Plugin/DeviceDescription.jl", "Plugin/Device.jl", "Plugin/Memory.jl"]
```

## Buffers

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Plugin/Buffer.jl", "Plugin/MemoryLayout.jl"]
```

## Executables

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Plugin/Executable.jl", "Plugin/LoadedExecutable.jl", "Plugin/ExecuteContext.jl"]
```

## Topology

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Plugin/TopologyDescription.jl"]
```

## Host-to-device transfers

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Plugin/AsyncHostToDeviceTransferManager.jl", "Plugin/CopyToDeviceStream.jl"]
```

## Extensions

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Extensions/Extension.jl"]
```

### Memory descriptions

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Extensions/MemoryDescriptions.jl"]
```

### FFI

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Extensions/FFI.jl"]
```

### Layouts

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Extensions/Layouts.jl"]
```

### Profiler

```@autodocs
Modules = [Reactant.XLA.PJRT.Plugin]
Pages = ["Extensions/Profiler.jl"]
```

## JuliaCPU

```@autodocs
Modules = [Reactant.Accelerators.JuliaCPU]
```
