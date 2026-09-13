


# XLA {#XLA}
<details class='jldocstring custom-block' >
<summary><a id='Reactant.XLA.AllocatorStats' href='#Reactant.XLA.AllocatorStats'><span class="jlbinding">Reactant.XLA.AllocatorStats</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



AllocatorStats()

Contains the following fields:
- `num_allocs`
  
- `bytes_in_use`
  
- `peak_bytes_in_use`
  
- `largest_alloc_size`
  
- `bytes_limit`
  
- `bytes_reserved`
  
- `peak_bytes_reserved`
  
- `bytes_reservable_limit`
  
- `largest_free_block_bytes`
  
- `pool_bytes`
  
- `peak_pool_bytes`
  

It should be constructed using the [`allocatorstats`](/api/xla#Reactant.XLA.allocatorstats) function.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/xla/Stats.jl#L16-L33" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.XLA.allocatorstats' href='#Reactant.XLA.allocatorstats'><span class="jlbinding">Reactant.XLA.allocatorstats</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



allocatorstats([device])

Return an [`AllocatorStats`](/api/xla#Reactant.XLA.AllocatorStats) instance with information about the device specific allocator.

::: warning Warning

This method is currently not implemented for the CPU device.

:::


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/xla/Stats.jl#L69-L77" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.XLA.cost_analysis' href='#Reactant.XLA.cost_analysis'><span class="jlbinding">Reactant.XLA.cost_analysis</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
cost_analysis(::AbstractLoadedExecutable)
cost_analysis(::Reactant.Thunk)
```


Returns a HloCostAnalysisProperties object with the cost analysis of the loaded executable.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/xla/Stats.jl#L134-L139" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.XLA.device_ordinal' href='#Reactant.XLA.device_ordinal'><span class="jlbinding">Reactant.XLA.device_ordinal</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
device_ordinal(device::Device)
```


Given the device, return the corresponding global device ordinal in the client.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/xla/Device.jl#L15-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

