


# Shardy Dialect {#Shardy-Dialect}

Refer to the [official documentation](https://openxla.org/shardy) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.all_gather-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.sdy.all_gather-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.all_gather</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`all_gather`

Gathers chunks of a tensor along axes specified in `gathering_axes`.

The `gathering_axes` is a list of lists of axes. The outer list is over the dimensions of the tensor. Each inner list specifies the axes along which a separate gather should be performed on the respective dimension. It will be applied to the sharding of the operand (`tensor`) to obtain the sharding of the result (`out_sharding`).

Note that `out_sharding` is not used to determine the sharding of the result. Instead, the sharding of the result is determined by the sharding of the operand and the `gathering_axes`, and `out_sharding` must match this inferred sharding.

**Example**

```mlir
%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b", "c"}, {}, {"d"}\]>]>} : tensor<8x8x8xf32>
%2 = sdy.all_gather [{"b", "c"}, {}, {"d"}\] %1 out_sharding=<@mesh, [{"a"}, {}, {}\]> : tensor<8x8x8xf32>
```


**Constraints:**
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
  
- Elements in `gathering_axes` must satisfy the constraints listed in `AxisRefListAttr`.
  
- Applying `gathering_axes` to the operand sharding gets `out_sharding`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L16-L43" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.all_reduce-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.sdy.all_reduce-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.all_reduce</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`all_reduce`

Reduces chunks of a tensor along axes specified in `reduction_axes`. The order of `reduction_axes` is not important for the result, but can affect the order of the corresponding replica groups.

**Constraints:**
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
  
- `reduction_axes` must satisfy the constraints listed in `AxisRefListAttr`;
  
- `reduction_axes` must not overlap with the operand sharding axes;
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L73-L84" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.all_slice-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.sdy.all_slice-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.all_slice</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`all_slice`

Slices chunks of a tensor along axes specified in `slicing_axes`. There is an algebric duality between `sdy.all_slice` and `sdy.all_gather`.

The `slicing_axes` is a list of lists of axes. The outer list is over the dimensions of the tensor. Each inner list specifies the axes along which a slice should be performed on the respective dimension. It will be applied to the sharding of the operand (`tensor`) to obtain the sharding of the result (`out_sharding`).

Note that `out_sharding` is not used to determine the sharding of the result. Instead, the sharding of the result is determined by the sharding of the operand and the `slicing_axes`, and `out_sharding` must match this inferred sharding.

**Example**

```mlir
%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}, {}\]>]>} : tensor<8x8x8xf32>
%2 = sdy.all_slice [{"b", "c"}, {}, {"d"}\] %1 out_sharding=<@mesh, [{"a", "b", "c"}, {}, {"d"}\]> : tensor<8x8x8xf32>
```


**Constraints:**
- Elements in `slicing_axes` must satisfy the constraints listed in `AxisRefListAttr`.
  
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
  
- Applying `slicing_axes` to the operand sharding gets `out_sharding`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L114-L142" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.all_to_all-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.sdy.all_to_all-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.all_to_all</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`all_to_all`

For each (axes, src_dim, tgt_dim) tuple in the parameter list, this operation slices chunks of a tensor along dimension `tgt_dim` and axes specified in `axes`, scatteres those chunks along the axes, and concatenates them along dimension `src_dim`.

This operation is essentially a combination of an all-gather along `src_dim` and `axes`, followed by an all-slice along `tgt_dim` and `axes`, i.e., a suffix of the axes sharding dimension `src_dim` on the input tensor is appended to the axes sharding dimension `tgt_dim` on the output tensor.

The all-to-all will be applied to the sharding of the operand (`tensor`) to obtain the sharding of the result (`out_sharding`).

Note that `out_sharding` is not used to determine the sharding of the result. Instead, the sharding of the result is determined by the sharding of the operand, `src_dim`, `tgt_dim`, and `axes`, and `out_sharding` must match this inferred sharding.

**Example**

```mlir
%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b"}, {"c"}, {}, {}\]>]>} : tensor<8x8x4x4x32>
%2 = sdy.all_to_all [{"b"}: 0->2, {"c"}: 1->3] %1 out_sharding=<@mesh, [{"a"}, {}, {"b"}, {"c"}\]> : tensor<8x8x4x4x32>
```


**Constraints:**
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
  
- The parameter list must not be empty.
  
- For each parameter in `params`:
  - Elements in `axes` must satisfy the constraints of `AxisRefAttr`.
    
  - `src_dim` and `tgt_dim` must be valid dimensions (non-negative and less
    
  than rank of tensor).
  - Any `src_dim` or `tgt_dim` must be unique across all parameters.
    
  - `src_dim` must be sorted in ascending order across all parameters.
    
  
- Moving `axes` from `src_dim` to `tgt_dim` in the operand sharding gets `out_sharding`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L172-L210" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.collective_permute-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.sdy.collective_permute-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.collective_permute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`collective_permute`

Sends a chunk of the input tensor from each device to another to reorder/replace the axes that shard the tensor.

A collective permute can transform the input sharding such that each dimension must be as sharded as it was before, i.e., it must be sharded along axes whose product of sizes matches that of the axes that previously sharded the tensor.

This is useful for reordering axes in a single dimension or across different dimensions, and swapping sharded axes with replicated ones.

In the below example, the sharded tensor size is `tensor<1x4x2xf32>`, and that is preserved by the collective permute.

**Example**

```mlir
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=4, "d"=2, "e"=2, "f"=2]>
%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "c"}, {"f"}, {"d", "e"}\]>]>} : tensor<8x8x8xf32>
%2 = sdy.collective_permute %1 out_sharding=<@mesh, [{"c":(1)2, "b", "f"}, {"a"}, {"e", "d"}\]> : tensor<8x8x8xf32>
```


**Constraints:**
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
  
- If input and output sharding have different meshes, then those meshes must have exactly the same axes and different order of device ids.
  
- For each dimension, the product of sharding axis sizes in `out_sharding` must match that of the corresponding operand dimension sharding.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L239-L269" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.constant-Tuple{}' href='#Reactant.MLIR.Dialects.sdy.constant-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.constant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`constant`

Produces an `output` tensor from a constant `value`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant

NOTE: SDY defines its own constant op that isn&#39;t ConstantLike and doesn&#39;t have a folder, so that we&#39;ll be able to duplicate constants without any greedy pattern rewriter folding them back into a single constant. In this way, constants can be sharded differently for every use, and no propagation is done between constants (or constant expressions).

**Example**

```mlir
%output = sdy.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L292-L310" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.data_flow_edge-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.sdy.data_flow_edge-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.data_flow_edge</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`data_flow_edge`

A data flow edge of some op X defines a bridge between a set of sources (each is either an operand of X or an operand of X&#39;s block terminator) and a set of targets (each is either a result of X or a block argument of X), such that all sources and targets should be sharded in the same way.

An op can have multiple data flow edges that are orthogonal to one another.

For example:

```mlir
  y_0, ..., y_n = while (x_0, ..., x_n)
                  ((pred_arg_0,... , pred_arg_n) { ... })
                  ((body_arg_0,..., body_arg_n) {
                    ...
                    return return_value_0, ..., return_value_n
                  })
```


This while op has n data flow edges, the i-th data flow edges is between sources `x_i`, `return_value_i` and targets `y_i`, `pred_arg_i`, `body_arg_i`.

An `sdy.data_flow_edge` takes as input the owner of an edge (can be any of the targets, but preferably an op result rather than a block argument), which shouldn&#39;t have any other uses. This op isn&#39;t pure because it can take an input that originally didn&#39;t have any uses.

The `sdy.data_flow_edge` also holds an optional sharding for all targets of the edge, and that sharding should be updated instead of the targets&#39; sharding (if can be attached) during propagation. This is useful when an op has many edges, as it&#39;s much more efficient to:
- propagate through each edge separately.
  
- update the sharding of each edge separately instead of all targets at once (e.g. an op has a single immutable `TensorShardingPerValueAttr` for result shardings).
  
- add each edge to the worklist separately when the sharding of a source has changed.
  

Propagation will propagate shardings between all sources and targets of a `sdy.data_flow_edge` as if it was a regular op with the sources as operands and targets as results, and an identity `sdy.op_sharding_rule`. That means that forward propagation is from sources to targets and backwards propagation is from targets to sources.

We don&#39;t allow the input of a `sdy.data_flow_edge` to be defined by an `SdyDialect` op, so we can assume that it&#39;s defined by an op that has unregistered `sdy.sharding` attribute.

NOTE: it&#39;s NOT the responsibility of the `sdy.data_flow_edge` to link between sources and targets, it&#39;s simply attached to the owner of the edge. The op that this edge is bound to (while in the example above) is responsible for providing this information.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L331-L386" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.manual_computation-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.sdy.manual_computation-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.manual_computation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`manual_computation`

Jump into a region written in terms of per-device local code with explicit collectives, where logical shapes match local per-device physical buffer shapes and collectives correspond exactly to physical cross-device communication.

The body is local wrt the manual_axes. Propagation will occur through the body on any free axes - those not in the manual_axes list.

**Constraints:**
- Elements in `in_shardings` and `out_shardings` must satisfy the constraints listed in `TensorShardingAttr`.
  
- The number of global and local tensor inputs/outputs of the op region must match.
  
- The manual axes must come before any free axes in each dim sharding.
  
- The manual axes cannot introduce padding. Namely, the dimension size must be divisible by the corresponding manual axes size.
  
- The global and local shapes of the op regions arguments/results must match.
  
- No manual axes are split.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L413-L431" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.mesh-Tuple{}' href='#Reactant.MLIR.Dialects.sdy.mesh-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.mesh</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mesh`

Defines a new named mesh. All meshes in a module must have the same number of devices (except for meshes with a single device_id). The mesh is a `Symbol` operation that appears in the module&#39;s `SymbolTable` and can be referenced by its `name`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L463-L470" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.named_computation-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.sdy.named_computation-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.named_computation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`named_computation`

Groups a computation, i.e. a block of operations, and gives it a name. Propagation will flow in/out of the region as if everything was inlined.

This can be used to handle propagating through call instructions to other functions. Any users of Shardy should write an import/export pass that converts their call ops to `sdy.named_computation` ops, duplicating/copying the body of the called function into the body of the `named_computation`.

The type of each block arguments and returned values in the region must be the same as the type of the operands and results type of the op.

**Example**

```mlir
%1 = sdy.named_computation<"foo">(%0) (%arg1: tensor<16x32xf32>) {
  sdy.return %arg1 : tensor<16x32xf32>
} : (tensor<16x32xf32>) -> tensor<16x32xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L492-L513" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.propagation_barrier-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.sdy.propagation_barrier-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.propagation_barrier</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`propagation_barrier`

This op operates like an identity op, outputting the same value it took as input. But in terms of propagation, this will only allow propagation to flow through it in a certain direction.

This prevents shardings from being propagated between the uses of the result of the barrier op and its operand.
- `FORWARD` means shardings can only flow from the operand to the result.
  
- `BACKWARD` means shardings can only flow from the result to the operand.
  
- `NONE` means no sharding can propagate through this op.
  
- Cannot specify `BOTH`, as this op would be redundant.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L545-L559" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.reshard-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.sdy.reshard-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.reshard</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`reshard`

Reshards the input tensor with the specified sharding, which is different from the input tensor&#39;s existing sharding.

Both ShardingConstraintOp and ReshardOp attach a sharding to a tensor. Their lifespan is:
1. Before sharding propagation, ShardingConstraintOp is added by users.
  
2. Sharding propagation consumes ShardingConstraintOp. There is no ShardingConstraintOp in the results of sharding propagation. Instead, ReshardOp may be added if needed.
  
3. A partitioner converts a ReshardOp into a collective op (or an identity op). There should be no ReshardOp in the results of the partitioner.
  

// TODO(b/331680067). Add a canonicalization pattern to remove redundant   // reshard ops.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L585-L602" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.sharding_constraint-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.sdy.sharding_constraint-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.sharding_constraint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`sharding_constraint`

Attaches a sharding to an intermediate tensor (e.g. the result of a matmul) to indicate that this is how that tensor, or a subset of its uses, should be sharded.

If the sharding has open dimensions and unconstraint axes, it means the tensor can be further sharded along the open dimensions.

This op can either:
- Have no uses (dangling) - which means the attached sharding is how the input tensor itself should be sharded.
  
- Have uses - which means the attached sharding is how the uses of the sharding constraint op should be sharded, while other uses of the input tensor might have a different sharding (if the input tensor has no other uses then the behavior is the same as the no uses case).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L644-L661" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.sdy.sharding_group-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.sdy.sharding_group-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.sdy.sharding_group</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`sharding_group`

This op provides an interface to assign tensors to sharding groups ( groups of tensors that will be enforced to have identical shardings). During propagation, as soon as one group element is sharded, all other members will be sharded in exactly the same way. This operation takes the argument group ID and returns no result, but instead modifies the internal sharding group representation to add the input tensor to the group with the given ID.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Shardy.jl#L684-L694" target="_blank" rel="noreferrer">source</a></Badge>

</details>

