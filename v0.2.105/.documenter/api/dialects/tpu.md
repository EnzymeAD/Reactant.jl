


# TPU Dialect {#TPU-Dialect}

Refer to the [official documentation](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/tpu.td) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tpu.broadcast_in_sublanes-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tpu.broadcast_in_sublanes-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tpu.broadcast_in_sublanes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_in_sublanes`

For each sublane `i`, broadcasts the value in lane `lane + i` along the entire sublane. If `lane + i` is not in [0, lane_count), then the value in sublane `i` is not defined (can be anything).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/TPU.jl#L136-L142" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tpu.create_subelement_mask-Tuple{}' href='#Reactant.MLIR.Dialects.tpu.create_subelement_mask-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.tpu.create_subelement_mask</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`create_subelement_mask`

The &quot;half-sublanes&quot;, &quot;quarter-sublanes&quot;, etc. (unit is determined by the type of `output`) of the mask are masked in the range specified by `from` and `to`.
- If `from <= to`, the range `[from, to)` is set and the rest is unset.
  
- If `to <= from`, the range `[to, from)` is unset and the rest is set.
  

All lanes are set identically.

**Example**

```mlir
%msk = tpu.create_subelement_mask 3, 9 : vector<8x128x2xi1>
```


This creates a mask `%msk` where, for all `lane`s, `%msk[*][lane][*]` is:

```
[[0, 0], [0, 1], [1, 1], [1, 1], [1, 0], [0, 0], [0, 0], [0, 0]]
```


It is currently only supported:
- In TPU v4, for `num_subelems` of 1 and 2.
  
- In TPU v5, for `num_subelems` of 1, 2, and 4.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/TPU.jl#L204-L231" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tpu.rotate-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tpu.rotate-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tpu.rotate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`rotate`

Rotates the given vector by the given amount in the given dimension, i.e., for a 2D vector of shape (m, n), rotating dim 0 by `amount` will shift a row at index `i` to index `(i + amount) % m`


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/TPU.jl#L934-L940" target="_blank" rel="noreferrer">source</a></Badge>

</details>

