


# Configuration Options {#Configuration-Options}

## Scoped Values {#Scoped-Values}

::: warning Warning

Currently options are scattered in the form of global variables and scoped values. We are in the process of migrating all of them into scoped values.

:::
<details class='jldocstring custom-block' >
<summary><a id='Reactant.with_config' href='#Reactant.with_config'><span class="jlbinding">Reactant.with_config</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
with_config(f; kwargs...)
```


Run the function `f` within a dynamic scope such that all uses of the config within this scope will use the provided values.

**Extended Help**

**Configuration Options**

**Lowering**
- `lower_partialsort_to_approx_top_k`: Whether to lower `partialsort` and `partialsortperm` to `Ops.approx_top_k`. Note that XLA only supports lowering `ApproxTopK` for TPUs unless `fallback_approx_top_k_lowering` is set to `true`.
  
- `fallback_approx_top_k_lowering`: Whether to lower `Ops.approx_top_k` to `stablehlo.top_k` if the XLA backend doesn&#39;t support `ApproxTopK`. Defaults to `true`.
  

**DotGeneral**
- `dot_general_algorithm`: Algorithm preset for `stablehlo.dot_general`. Can be `nothing`, [`DotGeneralAlgorithm`](/api/config#Reactant.DotGeneralAlgorithm) or [`DotGeneralAlgorithmPreset`](/api/config#Reactant.DotGeneralAlgorithmPreset). Defaults to `DotGeneralAlgorithmPreset.DEFAULT`.
  
- `dot_general_precision`: Precision for `stablehlo.dot_general`. Can be `nothing`, or [`DotGeneralPrecision`](/api/config#Reactant.DotGeneralPrecision). Defaults to `DotGeneralPrecision.DEFAULT`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Configuration.jl#L6-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Values {#Values}
- `DOT_GENERAL_PRECISION`: Controls the `precision_config` for `stablehlo.dot_general`.
  
- `DOT_GENERAL_ALGORITHM`: Controls the `algorithm` for `stablehlo.dot_general`.
  
- `LOWER_PARTIALSORT_TO_APPROX_TOP_K`: Whether to lower `partialsort` to `Ops.approx_top_k`. Note that XLA only supports lowering `ApproxTopK` for TPUs unless `FALLBACK_APPROX_TOP_K_LOWERING` is set to `true`. Defaults to `false`.
  
- `FALLBACK_APPROX_TOP_K_LOWERING`: Whether to fallback to lowering `ApproxTopK` to `stablehlo.top_k` if the XLA backend doesn&#39;t support `ApproxTopK`. Defaults to `true`.
  

### DotGeneral {#DotGeneral}
<details class='jldocstring custom-block' >
<summary><a id='Reactant.DotGeneralAlgorithmPreset' href='#Reactant.DotGeneralAlgorithmPreset'><span class="jlbinding">Reactant.DotGeneralAlgorithmPreset</span></a> <Badge type="info" class="jlObjectType jlModule" text="Module" /></summary>



```julia
DotGeneralAlgorithmPreset
```


Controls the `precision_config` for `stablehlo.dot_general`. Valid values are:
- `DEFAULT`
  
- `ANY_F8_ANY_F8_F32`
  
- `ANY_F8_ANY_F8_F32_FAST_ACCUM`
  
- `ANY_F8_ANY_F8_ANY`
  
- `ANY_F8_ANY_F8_ANY_FAST_ACCUM`
  
- `F16_F16_F16`
  
- `F16_F16_F32`
  
- `BF16_BF16_BF16`
  
- `BF16_BF16_F32`
  
- `BF16_BF16_F32_X3`
  
- `BF16_BF16_F32_X6`
  
- `BF16_BF16_F32_X9`
  
- `F32_F32_F32`
  
- `F64_F64_F64`
  

The following functions are available:

`supported_lhs_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)`   `supported_rhs_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)`   `accumulation_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T)`   `supported_output_eltype(dot_algorithm_preset::DotGeneralAlgorithmPreset.T, T1, T2)`   `MLIR.IR.Attribute(dot_algorithm_preset::DotGeneralAlgorithmPreset.T, T1, T2)`


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Configuration.jl#L167-L194" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.DotGeneralPrecision' href='#Reactant.DotGeneralPrecision'><span class="jlbinding">Reactant.DotGeneralPrecision</span></a> <Badge type="info" class="jlObjectType jlModule" text="Module" /></summary>



```julia
DotGeneralPrecision
```


Controls the `precision_config` for `stablehlo.dot_general`. Valid values are:
- `DEFAULT`
  
- `HIGH`
  
- `HIGHEST`
  

The following functions are available:

`MLIR.IR.Attribute(precision::DotGeneralPrecision.T)`


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Configuration.jl#L65-L77" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.DotGeneralAlgorithm' href='#Reactant.DotGeneralAlgorithm'><span class="jlbinding">Reactant.DotGeneralAlgorithm</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DotGeneralAlgorithm(
    ::Type{lhsT}, ::Type{rhsT}, ::Type{accumT},
    rhs_component_count::Int, lhs_component_count::Int, num_primitive_operations::Int,
    allow_imprecise_accumulation::Bool
)
DotGeneralAlgorithm{lhsT,rhsT,accumT}(
    lhs_component_count::Int, rhs_component_count::Int, num_primitive_operations::Int,
    allow_imprecise_accumulation::Bool
)
```


Represents the configuration of the `stablehlo.dot_general` operation.

**Arguments**
- `lhsT`: The type of the left-hand side operand.
  
- `rhsT`: The type of the right-hand side operand.
  
- `accumT`: The type of the accumulation operand.
  
- `lhs_component_count`: The number of components in the left-hand side operand.
  
- `rhs_component_count`: The number of components in the right-hand side operand.
  
- `num_primitive_operations`: The number of primitive operations in the `stablehlo.dot_general` operation.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Configuration.jl#L103-L125" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Environment Variables {#Environment-Variables}

The following environment variables can be used to configure Reactant.

### GPU Configuration {#GPU-Configuration}
- `XLA_REACTANT_GPU_MEM_FRACTION`: The fraction of GPU memory to use for XLA. Defaults to `0.75`.
  
- `XLA_REACTANT_GPU_PREALLOCATE`: Whether to preallocate GPU memory. Defaults to `true`.
  
- `REACTANT_VISIBLE_GPU_DEVICES`: A comma-separated list of GPU device IDs to use. Defaults to all visible GPU devices. Preferably use `CUDA_VISIBLE_DEVICES` instead.
  

### TPU Configuration {#TPU-Configuration}
- `TPU_LIBRARY_PATH`: The path to the libtpu.so library. If not provided, we download and use Scratch.jl to save the library.
  

### Distributed Setup {#Distributed-Setup}
- `REACTANT_COORDINATOR_BIND_ADDRESS`: The address to bind the coordinator to. If not provided, we try to automatically infer it from the environment.
  
