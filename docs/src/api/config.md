```@meta
CollapsedDocStrings = true
```

# Configuration Options

## Scoped Values

!!! warning

    Currently options are scattered in the form of global variables and scoped values. We
    are in the process of migrating all of them into scoped values.

```@docs
Reactant.with_config
```

### Values

- `DOT_GENERAL_PRECISION`: Controls the `precision_config` for `stablehlo.dot_general`.
- `DOT_GENERAL_ALGORITHM`: Controls the `algorithm` for `stablehlo.dot_general`.
- `LOWER_PARTIALSORT_TO_APPROX_TOP_K`: Whether to lower `partialsort` to `Ops.approx_top_k`.
  Note that XLA only supports lowering `ApproxTopK` for TPUs unless
  `FALLBACK_APPROX_TOP_K_LOWERING` is set to `true`. Defaults to `false`.
- `FALLBACK_APPROX_TOP_K_LOWERING`: Whether to fallback to lowering `ApproxTopK` to
  `stablehlo.top_k` if the XLA backend doesn't support `ApproxTopK`. Defaults to `true`.

### DotGeneral

```@docs
Reactant.DotGeneralAlgorithmPreset
Reactant.PrecisionConfig
Reactant.DotGeneralAlgorithm
```

### Zygote Overlay

- `OVERLAY_ZYGOTE_CALLS`: Whether to overlay `Zygote.gradient` calls with `Enzyme.autodiff`
  calls.

## Environment Variables

The following environment variables can be used to configure Reactant.

### GPU Configuration

- `XLA_REACTANT_GPU_MEM_FRACTION`: The fraction of GPU memory to use for XLA. Defaults to
  `0.75`.
- `XLA_REACTANT_GPU_PREALLOCATE`: Whether to preallocate GPU memory. Defaults to `true`.
- `REACTANT_VISIBLE_GPU_DEVICES`: A comma-separated list of GPU device IDs to use. Defaults
  to all visible GPU devices. Preferably use `CUDA_VISIBLE_DEVICES` instead.

### TPU Configuration

- `TPU_LIBRARY_PATH`: The path to the libtpu.so library. If not provided, we download and
  use Scratch.jl to save the library.

### Distributed Setup

- `REACTANT_COORDINATOR_BIND_ADDRESS`: The address to bind the coordinator to. If not
  provided, we try to automatically infer it from the environment.
