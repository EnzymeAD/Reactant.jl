# [Persistent Compilation Cache](@id persistent_compile_cache)

Reactant.jl supports a persistent compilation cache that caches compiled and autotuned
kernels on disk. We use [XLA's persisted autotuning](https://openxla.org/xla/persisted_autotuning)
for this purpose. By default, the autotuning cache is enabled.

## Preferences

- `persistent_cache_enabled`: Whether to enable the persistent compilation cache. Defaults
  to `false`.
- `persistent_cache_directory`: The base directory to use for the persistent compilation
  cache. Note that it is recommended to not set this preference, as Reactant will create
  a unique directory corresponding to XLA and Reactant_jll's version. If the user sets
  this preference, it is the user's responsibility to ensure that the directory exists
  and is writable and needs to be segregated based on XLA and Reactant_jll's version.
  Defaults to `""`.
- `persistent_kernel_cache_enabled`: Whether to enable the kernel cache. Defaults to `false`.
- `persistent_autotune_cache_enabled`: Whether to enable the autotuning cache. Defaults to
  `true`.

## Clearing the cache

To clear the cache, you can use [`Reactant.clear_compilation_cache!`](@ref):

```julia
using Reactant
clear_compilation_cache!()
```
