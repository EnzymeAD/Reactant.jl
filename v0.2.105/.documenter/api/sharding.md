


# Sharding API {#sharding-api}

`Reactant.Sharding` module provides a high-level API to construct MLIR operations with support for sharding.

Currently we haven&#39;t documented all the functions in `Reactant.Sharding`.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.Sharding.DimsSharding' href='#Reactant.Sharding.DimsSharding'><span class="jlbinding">Reactant.Sharding.DimsSharding</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DimsSharding(
    mesh::Mesh,
    dims::NTuple{D,Int},
    partition_spec;
    is_closed::NTuple{D,Bool}=ntuple(Returns(true), D),
    priority::NTuple{D,Int}=ntuple(i -> -1, D),
)
```


Similar to [`NamedSharding`](/api/sharding#Reactant.Sharding.NamedSharding) but works for a arbitrary dimensional array. Dimensions not specified in `dims` are replicated. If any dimension in `dims` is greater than the total number of dimensions in the array, the corresponding `partition_spec`, `is_closed` and `priority` are ignored. Additionally for any negative dimensions in `dims`, the true dims are calculated as `ndims(x) - dim + 1`. A dims value of `0` will throw an error.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Sharding.jl#L629-L643" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Sharding.Mesh' href='#Reactant.Sharding.Mesh'><span class="jlbinding">Reactant.Sharding.Mesh</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Mesh(devices::AbstractArray{XLA.AbstractDevice}, axis_names)
```


Construct a `Mesh` from an array of devices and a tuple of axis names. The size of the i-th axis is given by `size(devices, i)`. All the axis names must be unique, and cannot be nothing.

**Examples**

Assuming that we have a total of 8 devices, we can construct a mesh with the following:

```julia
julia> devices = Reactant.devices();

julia> mesh = Mesh(reshape(devices, 2, 2, 2), (:x, :y, :z));

julia> mesh = Mesh(reshape(devices, 4, 2), (:x, :y));
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Sharding.jl#L6-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Sharding.NamedSharding' href='#Reactant.Sharding.NamedSharding'><span class="jlbinding">Reactant.Sharding.NamedSharding</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NamedSharding(
    mesh::Mesh, partition_spec::Tuple;
    is_closed::NTuple{N,Bool}=ntuple(Returns(true), length(partition_spec)),
    priority::NTuple{N,Int}=ntuple(i -> -1, length(partition_spec)),
)
```


Sharding annotation that indicates that the array is sharded along the given `partition_spec`. For details on the sharding representation see the [Shardy documentation](https://openxla.org/shardy/sharding_representation).

**Arguments**
- `mesh`: [`Sharding.Mesh`](/api/sharding#Reactant.Sharding.Mesh) that describes the mesh of the devices.
  
- `partition_spec`: Must be equal to the ndims of the array being sharded. Each element can be:
  1. `nothing`: indicating the corresponding dimension is replicated along the axis.
    
  2. A tuple of axis names indicating the axis names that the corresponding dimension is sharded along.
    
  3. A single axis name indicating the axis name that the corresponding dimension is sharded along.
    
  

**Keyword Arguments**
- `is_closed`: A tuple of booleans indicating whether the corresponding dimension is closed along the axis. Defaults to `true` for all dimensions.
  
- `priority`: A tuple of integers indicating the priority of the corresponding dimension. Defaults to `-1` for all dimensions. A negative priority means that the priority is not considered by shardy.
  

**Examples**

```julia
julia> devices = Reactant.devices();

julia> mesh = Mesh(reshape(devices, 2, 2, 2), (:x, :y, :z));

julia> sharding = NamedSharding(mesh, (:x, :y, nothing)); # 3D Array sharded along x and y on dim 1 and 2 respectively, while dim 3 is replicated

julia> sharding = NamedSharding(mesh, ((:x, :y), nothing, nothing)); # 3D Array sharded along x and y on dim 1, 2 and 3 are replicated

julia> sharding = NamedSharding(mesh, (nothing, nothing)); # fully replicated Matrix
```


See also: [`Sharding.NoSharding`](/api/sharding#Reactant.Sharding.NoSharding)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Sharding.jl#L215-L259" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Sharding.NoSharding' href='#Reactant.Sharding.NoSharding'><span class="jlbinding">Reactant.Sharding.NoSharding</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NoSharding()
```


Sharding annotation that indicates that the array is not sharded.

See also: [`Sharding.NamedSharding`](/api/sharding#Reactant.Sharding.NamedSharding)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Sharding.jl#L181-L187" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Sharding.Replicated' href='#Reactant.Sharding.Replicated'><span class="jlbinding">Reactant.Sharding.Replicated</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Replicated(mesh::Mesh)
```


Sharding annotation that indicates that the array is fully replicated along all dimensions.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Sharding.jl#L723-L727" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Sharding.ShardyPropagationOptions' href='#Reactant.Sharding.ShardyPropagationOptions'><span class="jlbinding">Reactant.Sharding.ShardyPropagationOptions</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ShardyPropagationOptions
```


Fine-grained control over the sharding propagation pipeline.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Sharding.jl#L1113-L1117" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Sharding.is_sharded-Tuple{Reactant.Sharding.NoSharding}' href='#Reactant.Sharding.is_sharded-Tuple{Reactant.Sharding.NoSharding}'><span class="jlbinding">Reactant.Sharding.is_sharded</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
is_sharded(sharding)
is_sharded(x::AbstractArray)
```


Checks whether the given sharding refers to no sharding.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Sharding.jl#L1045-L1050" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Sharding.sharding_to_array_slices' href='#Reactant.Sharding.sharding_to_array_slices'><span class="jlbinding">Reactant.Sharding.sharding_to_array_slices</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sharding_to_array_slices(
    sharding, size_x; client=nothing, return_updated_sharding=Val(false)
)
```


Given a sharding and an array size, returns the device to array slices mapping. If `return_updated_sharding` is `Val(true)`, the updated sharding is returned as well (for inputs requiring padding).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Sharding.jl#L170-L178" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Sharding.unwrap_shardinfo-Tuple{Reactant.Sharding.AbstractSharding}' href='#Reactant.Sharding.unwrap_shardinfo-Tuple{Reactant.Sharding.AbstractSharding}'><span class="jlbinding">Reactant.Sharding.unwrap_shardinfo</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
unwrap_shardinfo(x)
```


Unwraps a sharding info object, returning the sharding object itself.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Sharding.jl#L1065-L1069" target="_blank" rel="noreferrer">source</a></Badge>

</details>


# Distributed API {#distributed-api}

`Reactant.Distributed` module provides a high-level API to run reactant on multiple hosts.

Currently we haven&#39;t documented all the functions in `Reactant.Distributed`.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.Distributed.is_initialized-Tuple{}' href='#Reactant.Distributed.is_initialized-Tuple{}'><span class="jlbinding">Reactant.Distributed.is_initialized</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
is_initialized()
```


Returns `true` if the distributed environment has been initialized.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Distributed.jl#L22-L26" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Distributed.local_rank-Tuple{}' href='#Reactant.Distributed.local_rank-Tuple{}'><span class="jlbinding">Reactant.Distributed.local_rank</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
local_rank()
```


Returns the local rank of the current process.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Distributed.jl#L8-L12" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Distributed.num_processes-Tuple{}' href='#Reactant.Distributed.num_processes-Tuple{}'><span class="jlbinding">Reactant.Distributed.num_processes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
num_processes()
```


Returns the number of processes.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Distributed.jl#L15-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

