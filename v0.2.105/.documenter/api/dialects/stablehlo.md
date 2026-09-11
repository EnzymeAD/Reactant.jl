


# StableHLO Dialect {#StableHLO-Dialect}

Refer to the [official documentation](https://openxla.org/stablehlo) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.abs-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.abs-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.abs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`abs`

Performs element-wise abs operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs

**Example**

```mlir
%result = stablehlo.abs %operand : tensor<3xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L16-L29" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.add-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.add-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.add</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`add`

Performs element-wise addition of two tensors `lhs` and `rhs` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add

**Example**

```mlir
%result = stablehlo.add %lhs, %rhs : tensor<2x2xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L50-L63" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.after_all-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.after_all-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.after_all</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`after_all`

Ensures that the operations producing the `inputs` are executed before any operations that depend on `result`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#after_all

**Example**

```mlir
%result = stablehlo.after_all %input0, %input1 : !stablehlo.token
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L86-L99" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.all_gather-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.all_gather-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.all_gather</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`all_gather`

Within each process group in the process grid, concatenates the values of the `operand` tensor from each process along `all_gather_dim` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_gather

**Example**

```mlir
%result:2 = "stablehlo.all_gather"(%operand0, %operand1) {
  all_gather_dim = 1 : i64,
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L122-L140" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.all_reduce-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.all_reduce-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.all_reduce</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`all_reduce`

Within each process group in the process grid, applies a reduction function `computation` to the values of the `operand` tensor from each process and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_reduce

**Example**

```mlir
%result:2 = "stablehlo.all_reduce"(%operand0, %operand0) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  "stablehlo.return"(%0) : (tensor<i64>) -> ()
}) {
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L175-L196" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.all_to_all-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.all_to_all-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.all_to_all</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`all_to_all`

Within each process group in the process grid, splits the values of the `operand` tensor along `split_dimension` into parts, scatters the split parts between the processes, concatenates the scattered parts along `concat_dimension` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_to_all

**Example**

```mlir
%result:2 = "stablehlo.all_to_all"(%operand1, %operand2) {
  split_dimension = 1 : i64,
  concat_dimension = 0 : i64,
  split_count = 2 : i64,
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
} : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L228-L248" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.and-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.and-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.and</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`and`

Performs element-wise AND of two tensors `lhs` and `rhs` and produces a `result` tensor

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#and

**Example**

```mlir
%result = stablehlo.and %lhs, %rhs : tensor<2x2xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L285-L298" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.atan2-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.atan2-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.atan2</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`atan2`

Performs element-wise atan2 operation on `lhs` and `rhs` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#atan2

**Example**

```mlir
%result = stablehlo.atan2 %lhs, %rhs : tensor<3xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L321-L334" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.batch_norm_grad-NTuple{5, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.batch_norm_grad-NTuple{5, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.batch_norm_grad</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`batch_norm_grad`

Computes gradients of several inputs of BatchNormTrainingOp backpropagating from `grad_output`, and produces `grad_operand`, `grad_scale` and `grad_offset` tensors.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#batch_norm_grad

**Example**

```mlir
%grad_operand, %grad_scale, %grad_offset =
"stablehlo.batch_norm_grad"(%operand, %scale, %mean, %variance, %grad_output) {
  epsilon = 0.0 : f32,
  feature_index = 2 : i64
} : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>,
     tensor<2x2x2xf64>) -> (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L357-L376" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.batch_norm_inference-NTuple{5, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.batch_norm_inference-NTuple{5, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.batch_norm_inference</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`batch_norm_inference`

Normalizes the `operand` tensor across all dimensions except for the `feature_index` dimension and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#batch_norm_inference

**Example**

```mlir
%result = "stablehlo.batch_norm_inference"(%operand, %scale, %offset, %mean, %variance) {
  epsilon = 0.0 : f32,
  feature_index = 2 : i64
} : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<2x2x2xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L413-L429" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.batch_norm_training-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.batch_norm_training-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.batch_norm_training</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`batch_norm_training`

Computes mean and variance across batch and spatial dimensions and normalizes the `operand` tensor, for each feature in the `feature_index` dimension and produces `output`, `batch_mean` and `batch_var` tensors.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#batch_norm_training

**Example**

```mlir
%output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%operand, %scale, %offset) {
  epsilon = 0.0 : f32,
  feature_index = 2 : i64
} : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>) ->
    (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L462-L480" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.bitcast_convert-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.bitcast_convert-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.bitcast_convert</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`bitcast_convert`

Performs a bitcast operation on `operand` tensor and produces a `result` tensor where the bits of the entire `operand` tensor are reinterpreted using the type of the `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#bitcast_convert

**Example**

```mlir
%result = stablehlo.bitcast_convert %operand : (tensor<f64>) -> tensor<4xf16>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L515-L529" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.broadcast-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.broadcast-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.broadcast</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast`

This operation is on its way out of StableHLO, so it is not included in the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

Informally, this operation does the same thing as XLA&#39;s Broadcast: https://www.tensorflow.org/xla/operation_semantics#broadcast

**Example**

```mlir
%result = stablehlo.broadcast %operand, sizes = [1, 2] : (tensor<3xi32>) -> tensor<1x2x3xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L586-L599" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.broadcast_in_dim-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.broadcast_in_dim-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.broadcast_in_dim</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_in_dim`

Expands the dimensions and/or rank of an input tensor by duplicating the data in the `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim

**Example**

```mlir
%result = stablehlo.broadcast_in_dim %operand, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L549-L562" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.case-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.case-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.case</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`case`

Produces the output from executing exactly one `function` from `branches` depending on the value of `index`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case

**Example**

```mlir
%result0, %result1 = "stablehlo.case"(%index) ({
  stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
}, {
  stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
}) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L625-L642" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.cbrt-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.cbrt-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.cbrt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cbrt`

Performs element-wise cubic root operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cbrt

**Example**

```mlir
%result = stablehlo.cbrt %operand : tensor<4xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L664-L677" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.ceil-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.ceil-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.ceil</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`ceil`

Performs element-wise ceil of `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#ceil

**Example**

```mlir
%result = stablehlo.ceil %operand : tensor<5xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L705-L717" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.cholesky-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.cholesky-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.cholesky</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cholesky`

Computes the Cholesky decomposition of a batch of matrices.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cholesky

**Example**

```mlir
%result = stablehlo.cholesky %a, lower = true : tensor<3x3xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L738-L750" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.clamp-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.clamp-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.clamp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`clamp`

Clamps every element of the `operand` tensor between a minimum and maximum value and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#clamp

**Example**

```mlir
%result = stablehlo.clamp %min, %operand, %max : tensor<3xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L774-L787" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.collective_broadcast-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.collective_broadcast-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.collective_broadcast</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`collective_broadcast`

Within each process group in the process grid, send the value of the `operand` tensor from the source process to the target processes and produce a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#collective_broadcast

**Example**

```mlir
%result = "stablehlo.collective_broadcast"(%operand) {
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<1x2xi64>) -> tensor<1x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L850-L867" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.collective_permute-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.collective_permute-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.collective_permute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`collective_permute`

Within each process group in the process grid, sends the value of the `operand` tensor from the source process to the target process and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#collective_permute

**Example**

```mlir
%result = "stablehlo.collective_permute"(%operand) {
  source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<2x2xi64>) -> tensor<2x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L896-L913" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.compare-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.compare-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.compare</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`compare`

Performs element-wise comparison of `lhs` and `rhs` tensors according to `comparison_direction` and `compare_type`, and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare

**Example**

```mlir
%result = stablehlo.compare LT, %lhs, %rhs, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L942-L955" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.complex-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.complex-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.complex</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`complex`

Performs element-wise conversion to a complex value from a pair of real and imaginary values, `lhs` and `rhs`, and produces a `result` tensor. See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#complex

**Example**

```mlir
%result = stablehlo.complex %lhs, %rhs : tensor<2xcomplex<f64>>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L987-L998" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.composite-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.composite-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.composite</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`composite`

Encapsulates an operation made up (composed) of other StableHLO operations, taking `inputs` and `composite_attributes` and producing `results`. The semantics of the op are implemented by the `decomposition` attribute. The `composite` op can be replaced with its decomposition without changing program semantics. In cases where inlining the decomposition does not provide the same op semantics, prefer using `custom_call`.

The `version` field (defaults to `0`) is used to denote when a composite&#39;s semantics change.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#composite

**Example**

```mlir
%results = stablehlo.composite "my.op" %input0, %input1 {
  composite_attributes = {
    my_attribute = "my_value"
  },
  decomposition = @my_op,
  version = 1 : i32
} : (tensor<f32>, tensor<f32>) -> tensor<f32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1021-L1047" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.concatenate-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.concatenate-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.concatenate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`concatenate`

Concatenates a variadic number of tensors in `inputs` along `dimension` dimension in the same order as the given arguments and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#concatenate

**Example**

```mlir
%result = stablehlo.concatenate %input0, %input1, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1080-L1094" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.constant-Tuple{}' href='#Reactant.MLIR.Dialects.stablehlo.constant-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.constant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`constant`

Produces an `output` tensor from a constant `value`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant

**Example**

```mlir
%output = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1120-L1132" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.convert-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.convert-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.convert</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`convert`

Performs an element-wise conversion from one element type to another on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convert

**Example**

```mlir
%result = stablehlo.convert %operand : (tensor<3xi64>) -> tensor<3xcomplex<f64>>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1153-L1166" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.convolution-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.convolution-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.convolution</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`convolution`

Computes dot products between windows of `lhs` and slices of `rhs` and produces `result`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution

**Example**

```mlir
%result = stablehlo.convolution(%lhs, %rhs)
  dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
  window = {
    stride = [4, 4],
    pad = [[0, 0], [0, 0]],
    lhs_dilate = [2, 2],
    rhs_dilate = [1, 1],
    reverse = [0, 0]
  } {
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } :
(tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1186-L1212" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.cosine-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.cosine-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.cosine</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cosine`

Performs element-wise cosine operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cosine

**Example**

```mlir
%result = stablehlo.cosine %operand : tensor<2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1261-L1274" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.count_leading_zeros-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.count_leading_zeros-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.count_leading_zeros</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`count_leading_zeros`

Performs element-wise count of the number of leading zero bits in the `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#count_leading_zeros

**Example**

```mlir
%result = stablehlo.count_leading_zeros %operand : tensor<2x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L814-L827" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.create_token-Tuple{}' href='#Reactant.MLIR.Dialects.stablehlo.create_token-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.create_token</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`create_token`

This operation is on its way out of StableHLO, so it is not included in the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

Informally, this operation does the same thing as AfterAllOp with 0 inputs: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#after_all

**Example**

```mlir
%output = stablehlo.create_token : !stablehlo.token
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1302-L1315" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.cross_replica_sum-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.cross_replica_sum-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.cross_replica_sum</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cross_replica_sum`

This operation is on its way out of StableHLO, so it is not included in the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

Informally, this operation does the same thing as AllReduceOp with `channel_id = 0`, `use_global_device_ids = false` and `computation` implementing addition: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_reduce

**Example**

```mlir
%result = "stablehlo.cross-replica-sum"(%operand) {
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
} : (tensor<4xf32>) -> tensor<4xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1336-L1353" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.custom_call-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.custom_call-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.custom_call</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`custom_call`

Encapsulates an implementation-defined operation `call_target_name` that takes `inputs` and `called_computations` and produces `results`.

Depending on the API version there are two ways to pass extra bits of static information to the external function:
1. Use `API_VERSION_TYPED_FFI` which allows passing a dictionary attribute.
  
2. Use a previous API version with a StringAttr to encode backend config.
  

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#custom_call

**Example**

```mlir
%results = stablehlo.custom_call @foo(%input0) {
  backend_config = {bar = 42 : i32},
  api_version = 4 : i32,
  called_computations = [@foo]
} : (tensor<f64>) -> tensor<f64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1379-L1401" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.divide-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.divide-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.divide</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`divide`

Performs element-wise division of dividend `lhs` and divisor `rhs` tensors and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#divide

**Example**

```mlir
%result = stablehlo.divide %lhs, %rhs : tensor<4xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1446-L1459" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.dot-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.dot-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.dot</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dot`

This operation is on its way out of StableHLO, so it is not included in the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

Informally, this operation does the same thing as XLA&#39;s Dot: https://www.tensorflow.org/xla/operation_semantics#dot

**Example**

```mlir
%0 = stablehlo.dot %arg0, %arg1 : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<1x1xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1533-L1546" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.dot_general-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.dot_general-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.dot_general</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dot_general`

Computes dot products between slices of `lhs` and slices of `rhs` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general

**Example**

```mlir
%result = stablehlo.dot_general %lhs, %rhs,
  batching_dims = [0] x [0],
  contracting_dims = [2] x [1],
  precision = [DEFAULT, DEFAULT],
  algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
  : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1482-L1500" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.dynamic_broadcast_in_dim-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.dynamic_broadcast_in_dim-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.dynamic_broadcast_in_dim</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dynamic_broadcast_in_dim`

This operation is functionally identical to [broadcast_in_dim](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim) op, but the result shape is specified dynamically via `output_dimensions`.

It also accepts optional attributes to express static knowledge about the expanding behavior of dimensions. If not specified, all dimensions are assumed to be possibly expanding. The sets of dimensions that are known to be expanding and the set of dimensions that are known to be non-expanding must be disjoint and they must be a subset of the operand&#39;s dimensions.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_broadcast_in_dim

**Example**

```mlir
%operand = stablehlo.constant dense<[[1, 2, 3]]> : tensor<1x3xi64>
%output_dimensions = stablehlo.constant dense<[2, 3, 2]> : tensor<3xi64>
%result = "stablehlo.dynamic_broadcast_in_dim"(%operand, %output_dimensions) {
  broadcast_dimensions = array<i64: 2, 1>,
  known_expanding_dimensions = array<i64: 0>,
  known_nonexpanding_dimensions = array<i64: 1>
} : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1570-L1595" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.dynamic_conv-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.dynamic_conv-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.dynamic_conv</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dynamic_conv`

This operation is functionally identical to [convolution](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution) op, but the padding is specified dynamically via `padding`.

**Example**

```mlir
%padding = stablehlo.constant dense<2> : tensor<2x2xi64>
%result = "stablehlo.dynamic_conv"(%lhs, %rhs, %padding) {
  window_strides = array<i64: 4, 4>,
  lhs_dilation = array<i64: 2, 2>,
  rhs_dilation = array<i64: 1, 1>,
  window_reversal = array<i1: false, false>,
  dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
  batch_group_count = 1 : i64,
  feature_group_count = 1 : i64,
  precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
} : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>, tensor<2x2xi64>) -> tensor<1x2x2x1xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1633-L1654" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.dynamic_gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.dynamic_gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.dynamic_gather</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dynamic_gather`

This operation is functionally identical to [gather](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather) op, with the `slice_sizes` specified dynamically as an operand.

**Example**

```mlir
%slice_sizes = stablehlo.constant dense<[1, 2, 2]> : tensor<3xi64>
%result = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
  dimension_numbers = #stablehlo.gather<
    offset_dims = [2, 3],
    collapsed_slice_dims = [0],
    start_index_map = [0, 2],
    index_vector_dim = 2>,
  indices_are_sorted = false
} : (tensor<3x4x2xi64>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1702-L1721" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.dynamic_iota-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.dynamic_iota-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.dynamic_iota</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dynamic_iota`

This operation is functionally identical to [iota](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota) op, but the result shape is specified dynamically via `output_shape`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_iota

**Example**

```mlir
%output_shape = stablehlo.constant dense<[4, 5]> : tensor<2xi64>
%0 = stablehlo.dynamic_iota %output_shape, dim = 0 : (tensor<2xi64>) -> tensor<4x5xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1752-L1767" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.dynamic_pad-NTuple{5, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.dynamic_pad-NTuple{5, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.dynamic_pad</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dynamic_pad`

This operation is functionally identical to [pad](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad) https://github.com/openxla/stablehlo/pull/2306#discussion_r1595669709 op, but with `edge_padding_low`,`edge_padding_high`and`interior_padding` specified dynamically as values.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_pad

**Example**

```mlir
%edge_padding_low = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
%edge_padding_high = stablehlo.constant dense<[2, 1]> : tensor<2xi32>
%interior_padding = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
%result = stablehlo.dynamic_pad %operand, %padding_value,
            %edge_padding_low, %edge_padding_high, %interior_padding
            : (tensor<2x3xi64>, tensor<i64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<5x9xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1789-L1809" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.dynamic_reshape-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.dynamic_reshape-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.dynamic_reshape</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dynamic_reshape`

This operation is functionally identical to [reshape](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape) op, but the result shape is specified dynamically via `output_shape`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_reshape

**Example**

```mlir
%output_shape = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
%result = stablehlo.dynamic_reshape %operand, %output_shape : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<3x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1839-L1854" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.dynamic_slice-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.dynamic_slice-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.dynamic_slice</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dynamic_slice`

Extracts a slice from the `operand` using dynamically-computed starting indices and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_slice

**Example**

```mlir
%result = stablehlo.dynamic_slice %operand, %start_indices0, %start_indices1, sizes = [2, 2]
  : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1876-L1890" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.dynamic_update_slice-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.dynamic_update_slice-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.dynamic_update_slice</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dynamic_update_slice`

Produces a `result` tensor which is equal to the `operand` tensor except that the slice starting at `start_indices` is updated with the values in `update`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_update_slice

**Example**

```mlir
%result = stablehlo.dynamic_update_slice %operand, %update, %start_indices0, %start_indices1
  : (tensor<4x4xi32>, tensor<2x2xi32>, tensor<i64>, tensor<i64>) -> tensor<4x4xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1917-L1932" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.einsum-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.einsum-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.einsum</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`einsum`

This operation is on its way out of StableHLO, so it is not included in the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

Informally, this operation does the same thing as TF&#39;s einsum: https://www.tensorflow.org/api_docs/python/tf/einsum

**Example**

```mlir
%result = "stablehlo.einsum"(%lhs, %rhs) {
  einsum_config = "ab,bc->ac"
} : (tensor<4x16xf32>, tensor<16x4xf32>) -> tensor<4x4xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1959-L1974" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.exponential-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.exponential-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.exponential</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`exponential`

Performs element-wise exponential operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential

**Example**

```mlir
%result = stablehlo.exponential %operand : tensor<2x2xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L1996-L2009" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.exponential_minus_one-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.exponential_minus_one-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.exponential_minus_one</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`exponential_minus_one`

Performs element-wise exponential minus one operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential_minus_one

**Example**

```mlir
%result = stablehlo.exponential_minus_one %operand : tensor<2xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2037-L2050" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.fft-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.fft-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.fft</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`fft`

Performs the forward and inverse Fourier transforms for real and complex inputs/outputs.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#fft

**Example**

```mlir
%result = stablehlo.fft %operand, type = FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2078-L2091" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.floor-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.floor-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.floor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`floor`

Performs element-wise floor of `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#floor

**Example**

```mlir
%result = stablehlo.floor %operand : tensor<2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2120-L2133" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.gather</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`gather`

Gathers slices from `operand` tensor from offsets specified in `start_indices` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather

**Example**

```mlir
%result = "stablehlo.gather"(%operand, %start_indices) {
  dimension_numbers = #stablehlo.gather<
    offset_dims = [3, 4],
    collapsed_slice_dims = [1],
    operand_batching_dims = [0],
    start_indices_batching_dims = [1],
    start_index_map = [2, 1],
    index_vector_dim = 3>,
  slice_sizes = array<i64: 1, 1, 2, 2>,
  indices_are_sorted = false
} : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2154-L2177" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.get_dimension_size-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.get_dimension_size-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.get_dimension_size</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`get_dimension_size`

Produces the size of the given `dimension` of the `operand`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_dimension_size

**Example**

```mlir
%result = stablehlo.get_dimension_size %operand, dim = 1 : (tensor<2x3xi64>) -> tensor<i32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2211-L2223" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.get_tuple_element-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.get_tuple_element-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.get_tuple_element</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`get_tuple_element`

Extracts element at `index` position of the `operand` tuple and produces a `result`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_tuple_element

**Example**

```mlir
%result = stablehlo.get_tuple_element %operand[0] : (tuple<tensor<2xf64>, tuple<tensor<i64>>>) -> tensor<2xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2246-L2259" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.if_-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.if_-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.if_</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`if_`

Produces the output from executing exactly one branch from `true_branch` or `false_branch` depending on the value of `pred`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#if

**Example**

%result = &quot;stablehlo.if&quot;(%pred) ({   &quot;stablehlo.return&quot;(%result_true_branch) : (tensor&lt;i32&gt;) -&gt; () }, {   &quot;stablehlo.return&quot;(%result_false_branch) : (tensor&lt;i32&gt;) -&gt; () }) : (tensor&lt;i1&gt;) -&gt; tensor&lt;i32&gt;


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2282-L2297" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.imag-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.imag-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.imag</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`imag`

Extracts the imaginary part, element-wise, from the `operand` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#imag

**Example**

```mlir
%result = stablehlo.imag %operand : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2323-L2336" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.infeed-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.infeed-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.infeed</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`infeed`

Reads data from the infeed and produces `results`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#infeed

**Example**

```mlir
%results0:2 = "stablehlo.infeed"(%token) :
    (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2357-L2370" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.iota-Tuple{}' href='#Reactant.MLIR.Dialects.stablehlo.iota-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.iota</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`iota`

Fills an `output` tensor with values in increasing order starting from zero along the `iota_dimension` dimension.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota

**Example**

```mlir
%output = stablehlo.iota dim = 0 : tensor<4x5xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2399-L2412" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.is_finite-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.is_finite-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.is_finite</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`is_finite`

Performs element-wise check whether the value in `x` is finite (i.e. is neither +Inf, -Inf, nor NaN) and produces a `y` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#is_finite

**Example**

```mlir
%y = stablehlo.is_finite %x : (tensor<7xf64>) -> tensor<7xi1>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2432-L2445" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.log-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.log-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.log</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`log`

Performs element-wise logarithm operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log

**Example**

```mlir
%result = stablehlo.log %operand : tensor<2x2xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2507-L2520" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.log_plus_one-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.log_plus_one-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.log_plus_one</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`log_plus_one`

Performs element-wise logarithm plus one operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log_plus_one

**Example**

```mlir
%result = stablehlo.log_plus_one %operand : tensor<5xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2466-L2479" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.logistic-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.logistic-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.logistic</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`logistic`

Performs element-wise logistic operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#logistic

**Example**

```mlir
%result = stablehlo.logistic %operand : tensor<2x2xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2548-L2561" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.map-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.map-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.map</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`map`

Applies a map function `computation` to `inputs` along the `dimensions` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#map

**Example**

```mlir
%result = "stablehlo.map"(%input0, %input1) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<i64>
    stablehlo.return %0 : tensor<i64>
}) {
  dimensions = array<i64: 0, 1>
} : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2589-L2608" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.maximum-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.maximum-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.maximum</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`maximum`

Performs element-wise max operation on tensors `lhs` and `rhs` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#maximum

**Example**

```mlir
%result = stablehlo.maximum %lhs, %rhs : tensor<4xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2634-L2647" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.minimum-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.minimum-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.minimum</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`minimum`

Performs element-wise min operation on tensors `lhs` and `rhs` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#minimum

**Example**

```mlir
%result = stablehlo.minimum %lhs, %rhs : tensor<4xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2670-L2683" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.multiply-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.multiply-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.multiply</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`multiply`

Performs element-wise product of two tensors `lhs` and `rhs` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#multiply

**Example**

```mlir
%result = stablehlo.multiply %lhs, %rhs : tensor<2xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2706-L2719" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.negate-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.negate-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.negate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`negate`

Performs element-wise negation of `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#negate

**Example**

```mlir
%result = stablehlo.negate %operand : tensor<2x3xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2742-L2755" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.not-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.not-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.not</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`not`

Performs element-wise NOT of tensor `operand` of type integer and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not

**Example**

```mlir
%result = stablehlo.not %operand : tensor<5x3x1xi1>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2776-L2789" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.optimization_barrier-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.optimization_barrier-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.optimization_barrier</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`optimization_barrier`

Ensures that the operations that produce the `operand` are executed before any operations that depend on the `result` and prevents compiler transformations from moving operations across the barrier. Other than that, the operation is an identity, i.e. `result` = `operand`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#optimization_barrier

**Example**

```mlir
%result0, %result1 = stablehlo.optimization_barrier %operand0, %operand1 : tensor<f32>, tensor<f32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2810-L2825" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.or-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.or-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.or</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`or`

Performs element-wise OR of two tensors `lhs` and `rhs` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#or

**Example**

```mlir
%result = stablehlo.or %lhs, %rhs : tensor<2xi1>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2850-L2863" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.outfeed-Tuple{Vector{Reactant.MLIR.IR.Value}, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.outfeed-Tuple{Vector{Reactant.MLIR.IR.Value}, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.outfeed</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`outfeed`

Writes `inputs` to the outfeed and produces a `result` token.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#outfeed

**Example**

```mlir
%result = "stablehlo.outfeed"(%input0, %token) :
    (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2886-L2899" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.pad-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.pad-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.pad</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`pad`

Expands `operand` by padding around the tensor as well as between the elements of the tensor with the given `padding_value`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad

**Example**

```mlir
%0 = stablehlo.pad %arg0, %arg1, low = [0, 1], high = [2, 1], interior = [1, 2]
  : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2928-L2942" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.partition_id-Tuple{}' href='#Reactant.MLIR.Dialects.stablehlo.partition_id-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.partition_id</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`partition_id`

Produces `partition_id` of the current process.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#partition_id

**Example**

```mlir
%result = stablehlo.partition_id : tensor<ui32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L2975-L2987" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.popcnt-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.popcnt-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.popcnt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`popcnt`

Performs element-wise count of the number of bits set in the `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#popcnt

**Example**

```mlir
%result = stablehlo.popcnt %operand : tensor<4xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3008-L3021" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.power-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.power-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.power</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`power`

Performs element-wise exponentiation of `lhs` tensor by `rhs` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#power

**Example**

```mlir
%result = stablehlo.power %lhs, %rhs : tensor<6xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3042-L3055" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.real-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.real-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.real</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`real`

Extracts the real part, element-wise, from the `operand` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#real

**Example**

```mlir
%result = stablehlo.real %operand : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3121-L3134" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.real_dynamic_slice-NTuple{4, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.real_dynamic_slice-NTuple{4, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.real_dynamic_slice</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`real_dynamic_slice`

This operation is a work in progress, so it is not yet included in the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

Informally, this operation does the same thing as SliceOp except that `start_indices`, `limit_indices` and `strides` are specified dynamically: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

**Example**

```mlir
%result = stablehlo.real_dynamic_slice %operand,
            %start_indices, %limit_indices, %strides
       : (tensor<256x?xf32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<256x?xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3078-L3094" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.recv-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.recv-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.recv</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`recv`

Receives data from a channel with `channel_id` and produces `results`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#recv

**Example**

```mlir
%results:2 = "stablehlo.recv"(%token) {
  channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>,
  is_host_transfer = true
} : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3155-L3170" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.reduce-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.reduce-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.reduce</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`reduce`

Applies a reduction function `body` to `inputs` and `init_values` along the `dimensions` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce

**Example**

```mlir
%result = "stablehlo.reduce"(%input, %init_value) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
    stablehlo.return %0 : tensor<i64>
}) {
  dimensions = array<i64: 1>
} : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3198-L3217" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.reduce_precision-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.reduce_precision-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.reduce_precision</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`reduce_precision`

Performs element-wise conversion of `operand` to another floating-point type that uses `exponent_bits` and `mantissa_bits` and back to the original floating-point type and produces an `output` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_precision

**Example**

```mlir
%output = stablehlo.reduce_precision %operand, format = e5m10 : tensor<6xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3244-L3258" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.reduce_scatter-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.reduce_scatter-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.reduce_scatter</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`reduce_scatter`

Within each process group in the process grid, performs reduction, using `computations`, over the values of the `operand` tensor from each process, splits the reduction result along `scatter_dimension` into parts, and scatters the split parts between the processes to produce the `result`.

````
See:
https://github.com/openxla/stablehlo/blob/main/docs/spec#reduce_scatter

Example:
```mlir
%result = "stablehlo.reduce_scatter"(%operand) ({
````


^bb0(%arg0: tensor&lt;i64&gt;, %arg1: tensor&lt;i64&gt;):  %0 = stablehlo.add %arg0, %arg1 : tensor&lt;i64&gt;  stablehlo.return %0 : tensor&lt;i64&gt;     }) {  scatter_dimension = 1 : i64,  replica_groups = dense&lt;[[0, 1]]&gt; : tensor&lt;1x2xi64&gt;,  channel_handle = #stablehlo.channel_handle&lt;handle = 0, type = 0&gt;     } : (tensor&lt;2x4xi64&gt;) -&gt; tensor&lt;2x2xi64&gt;     ```


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3288-L3311" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.reduce_window-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.reduce_window-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.reduce_window</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`reduce_window`

Applies a reduction function `body` to windows of `inputs` and `init_values` and produces `results`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_window

**Example**

```mlir
%result = "stablehlo.reduce_window"(%input, %init_value) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
    stablehlo.return %0 : tensor<i64>
}) {
  window_dimensions = array<i64: 2, 1>,
  window_strides = array<i64: 4, 1>,
  base_dilations = array<i64: 2, 1>,
  window_dilations = array<i64: 3, 1>,
  padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
} : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3347-L3370" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.remainder-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.remainder-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.remainder</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`remainder`

Performs element-wise remainder of dividend `lhs` and divisor `rhs` tensors and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#remainder

**Example**

```mlir
%result = stablehlo.remainder %lhs, %rhs : tensor<4xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3408-L3421" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.replica_id-Tuple{}' href='#Reactant.MLIR.Dialects.stablehlo.replica_id-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.replica_id</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`replica_id`

Produces `replica_id` of the current process.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#replica_id

**Example**

```mlir
%result = stablehlo.replica_id : tensor<ui32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3444-L3456" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.reshape-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.reshape-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.reshape</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`reshape`

Performs reshape of `operand` tensor to a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape

**Example**

```mlir
%result = stablehlo.reshape %operand : (tensor<2xf32>) -> tensor<1x2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3477-L3489" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.reverse-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.reverse-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.reverse</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`reverse`

Reverses the order of elements in the `operand` along the specified `dimensions` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reverse

**Example**

```mlir
%result = stablehlo.reverse %operand, dims = [1] : tensor<3x2xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3528-L3541" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.rng-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.rng-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.rng</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`rng`

Generates random numbers using the `rng_distribution` algorithm and produces a `result` tensor of a given shape `shape`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rng

**Example**

```mlir
%result = stablehlo.rng %a, %b, %shape, distribution = NORMAL : (tensor<i32>, tensor<i32>, tensor<2xi64>) -> tensor<3x3xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3604-L3617" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.rng_bit_generator-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.rng_bit_generator-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.rng_bit_generator</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`rng_bit_generator`

Returns an `output` filled with uniform random data and an updated output state `output_state` given an initial state `initial_state` using the pseudorandom number generator algorithm `rng_algorithm`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rng_bit_generator

**Example**

```mlir
%output_state, %output = stablehlo.rng_bit_generator %initial_state, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x2xui64>)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3564-L3578" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.round_nearest_afz-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.round_nearest_afz-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.round_nearest_afz</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`round_nearest_afz`

Performs element-wise rounding towards the nearest integer, breaking ties away from zero, on the `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_afz

**Example**

```mlir
%result = stablehlo.round_nearest_afz %operand : tensor<5xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3682-L3695" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.round_nearest_even-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.round_nearest_even-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.round_nearest_even</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`round_nearest_even`

Performs element-wise rounding towards the nearest integer, breaking ties towards the even integer, on the `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_even

**Example**

```mlir
%result = stablehlo.round_nearest_even %operand : tensor<5xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3645-L3659" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.rsqrt-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.rsqrt-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.rsqrt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`rsqrt`

Performs element-wise reciprocal square root operation on `operand` tensor and produces a `result` tensor, implementing the `rSqrt` operation from the IEEE-754 specification.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rsqrt

**Example**

```mlir
%result = stablehlo.rsqrt %operand : tensor<2x2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3718-L3732" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.scatter-Tuple{Vector{Reactant.MLIR.IR.Value}, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.scatter-Tuple{Vector{Reactant.MLIR.IR.Value}, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.scatter</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`scatter`

Produces `results` tensors which are equal to `inputs` tensors except that several slices specified by `scatter_indices` are updated with the values `updates` using `update_computation`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter

Example:    `mlir    %result = "stablehlo.scatter"(%input, %scatter_indices, %update) ({  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):    %0 = stablehlo.add %arg0, %arg1 : tensor<i64>    stablehlo.return %0 : tensor<i64>    }) {  scatter_dimension_numbers = #stablehlo.scatter<    update_window_dims = [3, 4],    inserted_window_dims = [1],    input_batching_dims = [0],    scatter_indices_batching_dims = [1],    scatter_dims_to_operand_dims = [2, 1],    index_vector_dim = 3>,  indices_are_sorted = false,  unique_indices = false    } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>`


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3760-L3788" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.select-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.select-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.select</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`select`

Produces a `result` tensor where each element is selected from `on_true` or `on_false` tensor based on the value of the corresponding element of `pred`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select

**Example**

```mlir
%result = stablehlo.select %pred, %on_true, %on_false : tensor<2x2xi1>, tensor<2x2xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3886-L3899" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.select_and_scatter-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.select_and_scatter-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.select_and_scatter</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`select_and_scatter`

Scatters the values from the `source` tensor using `scatter` based on the outcome of `reduce_window` of the `input` tensor using `select` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select_and_scatter

**Example**

```mlir
%result = "stablehlo.select_and_scatter"(%operand, %source, %init_value) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %0 : tensor<i1>
}, {
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
    stablehlo.return %0 : tensor<i64>
}) {
  window_dimensions = array<i64: [3, 1]>,
  window_strides = array<i64: [2, 1]>,
  padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>
} : (tensor<4x2xi64>, tensor<2x2xi64>, tensor<i64>) -> tensor<4x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3824-L3850" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.send-Tuple{Vector{Reactant.MLIR.IR.Value}, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.send-Tuple{Vector{Reactant.MLIR.IR.Value}, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.send</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`send`

Sends `inputs` to a channel `channel_id` and produces a `result` token.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#send

**Example**

```mlir
%result = "stablehlo.send"(%operand, %token) {
  channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
  is_host_transfer = true
} : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3926-L3941" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.set_dimension_size-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.set_dimension_size-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.set_dimension_size</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`set_dimension_size`

This operation is a work in progress, so it is not yet included in the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

Informally, this operation does the same thing as XLA&#39;s SetDimensionSize: https://www.tensorflow.org/xla/operation_semantics#setdimensionsize

**Example**

```mlir
%0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<4x2xf32>, tensor<i32>) -> tensor<4x2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L3971-L3984" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.shift_left-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.shift_left-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.shift_left</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`shift_left`

Performs element-wise left-shift operation on the `lhs` tensor by `rhs` number of bits and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_left

**Example**

```mlir
%result = stablehlo.shift_left %lhs, %rhs : tensor<3xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4011-L4024" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.shift_right_arithmetic-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.shift_right_arithmetic-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.shift_right_arithmetic</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`shift_right_arithmetic`

Performs element-wise arithmetic right-shift operation on the `lhs` tensor by `rhs` number of bits and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_arithmetic

**Example**

```mlir
%result = stablehlo.shift_right_arithmetic %lhs, %rhs : tensor<3xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4047-L4060" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.shift_right_logical-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.shift_right_logical-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.shift_right_logical</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`shift_right_logical`

Performs element-wise logical right-shift operation on the `lhs` tensor by `rhs` number of bits and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_logical

**Example**

```mlir
%result = stablehlo.shift_right_logical %lhs, %rhs : tensor<3xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4083-L4096" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.sign-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.sign-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.sign</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`sign`

Returns the sign of the `operand` element-wise and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sign

**Example**

```mlir
%result = stablehlo.sign %operand : tensor<5xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4119-L4132" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.sine-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.sine-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.sine</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`sine`

Performs element-wise sine operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sine

**Example**

```mlir
%result = stablehlo.sine %operand : tensor<2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4153-L4166" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.slice-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.slice-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.slice</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`slice`

Extracts a slice from the `operand` using statically-computed starting indices and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

**Example**

```mlir
%result = stablehlo.slice %operand [1:3, 4:8:2]
   : (tensor<3x8xi64>) -> tensor<2x2xi64>

// Same in generic form: the `1:3` above is mapped to the first entry in
// `start_indices` and `limit_indices`, while `strides` is implicitly 1.
// The `4:8:2` above is parsed into the second entry of `start_indices`,
// `limit_indices` and `strides` respectively.
%result = "stablehlo.slice" (%operand) {
  start_indices = array<i64: 1, 4>,
  limit_indices = array<i64: 3, 8>,
  strides = array<i64: 1, 2>
} : (tensor<3x8xi64>) -> tensor<2x2xi64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4194-L4218" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.sort-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.sort-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.sort</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`sort`

Sorts a variadic number of tensors in `inputs` together, according to a custom `comparator`, along the given `dimension` and produces a variadic number of tensors as `results`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sort

**Example**

```mlir %result0, %result1 = &quot;stablehlo.sort&quot;(%input0, %input1) ({   ^bb0(%arg0: tensor&lt;i64&gt;, %arg1: tensor&lt;i64&gt;, %arg2: tensor&lt;i64&gt;, %arg3: tensor&lt;i64&gt;):     %predicate = stablehlo.compare GT, %arg0, %arg1 : (tensor&lt;i64&gt;, tensor&lt;i64&gt;) -&gt; tensor&lt;i1&gt;     stablehlo.return %predicate : tensor&lt;i1&gt; }) {   dimension = 0 : i64,   is_stable = true } : (tensor&lt;2x3xi64&gt;, tensor&lt;2x3xi64&gt;) -&gt; (tensor&lt;2x3xi64&gt;, tensor&lt;2x3xi64&gt;)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4250-L4270" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.sqrt-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.sqrt-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.sqrt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`sqrt`

Performs element-wise square root operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sqrt

**Example**

```mlir
%result = stablehlo.sqrt %operand : tensor<2x2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4299-L4312" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.subtract-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.subtract-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.subtract</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`subtract`

Performs element-wise subtraction of two tensors `lhs` and `rhs` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#subtract

**Example**

```mlir
%result = stablehlo.subtract %lhs, %rhs : tensor<2xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4340-L4353" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.tan-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.tan-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.tan</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tan`

Performs element-wise tangent operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tan

**Example**

```mlir
%result = stablehlo.tan %operand : tensor<2x2xf64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4376-L4389" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.tanh-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.tanh-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.tanh</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tanh`

Performs element-wise hyperbolic tangent operation on `operand` tensor and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tanh

**Example**

```mlir
%result = stablehlo.tanh %operand : tensor<2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4417-L4430" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.torch_index_select-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.torch_index_select-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.torch_index_select</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`torch_index_select`

This operation is on its way out of StableHLO, so it is not included in the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

Informally, this operation does the same thing as PyTorch&#39;s index_select, augmented with support for batch dimensions: https://pytorch.org/docs/stable/generated/torch.index_select.html.

The `batch_dims` attribute specifies the number of major batch dimensions (0 or more) that act like a multidimensional loop over both the operand and the index.

**Example**

```mlir
%result = "stablehlo.torch_index_select"(%operand, %index) {
  dim = 2 : i64,
  batch_dims = 1 : i64
} : (tensor<8x128x3072x64xf32>, tensor<8x16x1024xi32>) -> tensor<8x128x16x1024x64xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4458-L4479" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.transpose-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.transpose-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.transpose</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`transpose`

Permutes the dimensions of `operand` tensor using `permutation` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose

**Example**

```mlir
%0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<1x2x3xi32>) -> tensor<3x2x1xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4503-L4516" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.triangular_solve-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.triangular_solve-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.triangular_solve</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`triangular_solve`

Solves batches of systems of linear equations with lower or upper triangular coefficient matrices.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#triangular_solve

**Example**

```mlir
%result = "stablehlo.triangular_solve"(%a, %b) {
  left_side = true,
  lower = true,
  unit_diagonal = false,
  transpose_a = #stablehlo<transpose NO_TRANSPOSE>
} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4539-L4557" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.tuple-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.tuple-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.tuple</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tuple`

Produces a `result` tuple from values `val`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tuple

**Example**

```mlir
%result = stablehlo.tuple %val0, %val1 : tuple<tensor<2xf64>, tuple<tensor<i64>>>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4592-L4604" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.unary_einsum-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.unary_einsum-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.unary_einsum</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`unary_einsum`

This operation is on its way out of StableHLO, so it is not included in the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

Informally, this operation does the same thing as TF&#39;s einsum: https://www.tensorflow.org/api_docs/python/tf/einsum

**Example**

```mlir
%result = "stablehlo.unary_einsum"(%operand) {
  einsum_config = "ab->a"
} : (tensor<4x16xf32>) -> tensor<4xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4627-L4642" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.uniform_dequantize-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.uniform_dequantize-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.uniform_dequantize</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`uniform_dequantize`

Performs element-wise conversion of quantized tensor `operand` to a floating-point tensor `result` according to the quantization parameters defined by the `operand` type.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#uniform_dequantize

**Example**

```mlir
%result = stablehlo.uniform_dequantize %operand : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>) -> tensor<2xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4662-L4676" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.uniform_quantize-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.uniform_quantize-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.uniform_quantize</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`uniform_quantize`

Performs element-wise conversion of floating-point tensor or quantized tensor `operand` to a quantized tensor `result` according to the quantization parameters defined by the `result` type.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#uniform_quantize

**Example**

```mlir
%result = stablehlo.uniform_quantize %operand : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4699-L4713" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.while_-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.stablehlo.while_-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.while_</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`while_`

Produces the output from executing `body` function 0 or more times while the `cond` function outputs `true`.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while

**Example**

```mlir
%results0, %results1 = stablehlo.while(%arg0 = %init_i, %arg1 = %init_sum) : tensor<i64>, tensor<i64>
cond {
  %cond = stablehlo.compare LT, %arg0, %ten : (tensor<i64>, tensor<i64>) -> tensor<i1>
  stablehlo.return %cond : tensor<i1>
} do {
  %new_sum = stablehlo.add %arg1, %one : tensor<i64>
  %new_i = stablehlo.add %arg0, %one : tensor<i64>
  stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4733-L4754" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.stablehlo.xor-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.stablehlo.xor-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.stablehlo.xor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`xor`

Performs element-wise XOR of two tensors `lhs` and `rhs` and produces a `result` tensor.

See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#xor

**Example**

```mlir
%result = stablehlo.xor %lhs, %rhs : tensor<2xi32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/StableHLO.jl#L4780-L4793" target="_blank" rel="noreferrer">source</a></Badge>

</details>

