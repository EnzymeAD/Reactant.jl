module sdy
using ...IR
import ...IR:
    NamedAttribute,
    Value,
    Location,
    Block,
    Region,
    Attribute,
    create_operation,
    context,
    IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

"""
`all_gather`

Gathers chunks of a tensor along axes specified in `gathering_axes`.

The `gathering_axes` is a list of lists of axes. The outer list is over the
dimensions of the tensor. Each inner list specifies the axes along which a
separate gather should be performed on the respective dimension. It will be
applied to the sharding of the operand (`tensor`) to obtain the sharding of
the result (`out_sharding`).

Note that `out_sharding` is not used to determine the sharding of the
result. Instead, the sharding of the result is determined by the sharding of
the operand and the `gathering_axes`, and `out_sharding` must match this
inferred sharding.

# Example
```mlir
%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{\"a\", \"b\", \"c\"}, {}, {\"d\"}\\]>]>} : tensor<8x8x8xf32>
%2 = sdy.all_gather [{\"b\", \"c\"}, {}, {\"d\"}\\] %1 out_sharding=<@mesh, [{\"a\"}, {}, {}\\]> : tensor<8x8x8xf32>
```

**Constraints:**
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
- Elements in `gathering_axes` must satisfy the constraints listed in
  `AxisRefListAttr`.
- Applying `gathering_axes` to the operand sharding gets `out_sharding`.
"""
function all_gather(
    tensor::Value;
    result=nothing::Union{Nothing,IR.Type},
    gathering_axes,
    out_sharding,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("gathering_axes", gathering_axes),
        namedattribute("out_sharding", out_sharding),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "sdy.all_gather",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`all_reduce`

Reduces chunks of a tensor along axes specified in `reduction_axes`.
The order of `reduction_axes` is not important for the result, but can
affect the order of the corresponding replica groups.

**Constraints:**
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
- `reduction_axes` must satisfy the constraints listed in `AxisRefListAttr`.
- `reduction_axes` must be sorted w.r.t. the mesh.
- The operand sharding and `out_sharding` must have equivalent dimension
  shardings.
- `reduction_axes` must not overlap with the operand dimension sharding and
  replicated axes (it can overlap with unreduced axes).
- `reduction_axes` must not overlap with the unreduced axes of
  `out_sharding`. In other words, `out_sharding` must be be replicated along
  `reduction_axes` (implicitly or explicitly).
"""
function all_reduce(
    tensor::Value;
    result=nothing::Union{Nothing,IR.Type},
    reduction_axes,
    out_sharding,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("reduction_axes", reduction_axes),
        namedattribute("out_sharding", out_sharding),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "sdy.all_reduce",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`all_slice`

Slices chunks of a tensor along axes specified in `slicing_axes`. There is
an algebric duality between `sdy.all_slice` and `sdy.all_gather`.

The `slicing_axes` is a list of lists of axes. The outer list is over the
dimensions of the tensor. Each inner list specifies the axes along which a
slice should be performed on the respective dimension. It will be applied to
the sharding of the operand (`tensor`) to obtain the sharding of the result
(`out_sharding`).

Note that `out_sharding` is not used to determine the sharding of the
result. Instead, the sharding of the result is determined by the sharding of
the operand and the `slicing_axes`, and `out_sharding` must match this
inferred sharding.

# Example
```mlir
%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{\"a\"}, {}, {}\\]>]>} : tensor<8x8x8xf32>
%2 = sdy.all_slice [{\"b\", \"c\"}, {}, {\"d\"}\\] %1 out_sharding=<@mesh, [{\"a\", \"b\", \"c\"}, {}, {\"d\"}\\]> : tensor<8x8x8xf32>
```

**Constraints:**
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
- Elements in `slicing_axes` must satisfy the constraints listed in
  `AxisRefListAttr`.
- Applying `slicing_axes` to the operand sharding gets `out_sharding`.
"""
function all_slice(
    tensor::Value;
    result=nothing::Union{Nothing,IR.Type},
    slicing_axes,
    out_sharding,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("slicing_axes", slicing_axes),
        namedattribute("out_sharding", out_sharding),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "sdy.all_slice",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`all_to_all`

For each (axes, src_dim, tgt_dim) tuple in the parameter list, this
operation slices chunks of a tensor along dimension `tgt_dim` and axes
specified in `axes`, scatteres those chunks along the axes, and concatenates
them along dimension `src_dim`.

This operation is essentially a combination of an all-gather along `src_dim`
and `axes`, followed by an all-slice along `tgt_dim` and `axes`, i.e., a
suffix of the axes sharding dimension `src_dim` on the input tensor is
appended to the axes sharding dimension `tgt_dim` on the output tensor.

The all-to-all will be applied to the sharding of the operand (`tensor`) to
obtain the sharding of the result (`out_sharding`).

Note that `out_sharding` is not used to determine the sharding of the
result. Instead, the sharding of the result is determined by the sharding of
the operand, `src_dim`, `tgt_dim`, and `axes`, and `out_sharding` must match
this inferred sharding.

# Example
```mlir
%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{\"a\", \"b\"}, {\"c\"}, {}, {}\\]>]>} : tensor<8x8x4x4x32>
%2 = sdy.all_to_all [{\"b\"}: 0->2, {\"c\"}: 1->3] %1 out_sharding=<@mesh, [{\"a\"}, {}, {\"b\"}, {\"c\"}\\]> : tensor<8x8x4x4x32>
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
- Moving `axes` from `src_dim` to `tgt_dim` in the operand sharding gets
  `out_sharding`.
"""
function all_to_all(
    tensor::Value;
    result=nothing::Union{Nothing,IR.Type},
    params,
    out_sharding,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("params", params), namedattribute("out_sharding", out_sharding)
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "sdy.all_to_all",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`collective_permute`

Sends a chunk of the input tensor from each device to another to
reorder/replace the axes that shard the tensor.

A collective permute can transform the input sharding such that each
dimension must be as sharded as it was before, i.e., it must be sharded
along axes whose product of sizes matches that of the axes that previously
sharded the tensor.

This is useful for reordering axes in a single dimension or across different
dimensions, and swapping sharded axes with replicated ones.

In the below example, the sharded tensor size is `tensor<1x4x2xf32>`, and
that is preserved by the collective permute.

# Example
```mlir
sdy.mesh @mesh = <[\"a\"=2, \"b\"=2, \"c\"=4, \"d\"=2, \"e\"=2, \"f\"=2]>
%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{\"a\", \"c\"}, {\"f\"}, {\"d\", \"e\"}\\]>]>} : tensor<8x8x8xf32>
%2 = sdy.collective_permute %1 out_sharding=<@mesh, [{\"c\":(1)2, \"b\", \"f\"}, {\"a\"}, {\"e\", \"d\"}\\]> : tensor<8x8x8xf32>
```

**Constraints:**
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
- If input and output sharding have different meshes, then those meshes must
  have exactly the same axes and different order of device ids.
- For each dimension, the product of sharding axis sizes in `out_sharding`
  must match that of the corresponding operand dimension sharding.
"""
function collective_permute(
    tensor::Value; result=nothing::Union{Nothing,IR.Type}, out_sharding, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("out_sharding", out_sharding),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "sdy.collective_permute",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`constant`

Produces an `output` tensor from a constant `value`.

See:
https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant

NOTE: SDY defines its own constant op that isn\'t ConstantLike and doesn\'t
have a folder, so that we\'ll be able to duplicate constants without any
greedy pattern rewriter folding them back into a single constant. In this
way, constants can be sharded differently for every use, and no propagation
is done between constants (or constant expressions).

# Example
```mlir
%output = sdy.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
```
"""
function constant(; output=nothing::Union{Nothing,IR.Type}, value, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]
    !isnothing(output) && push!(op_ty_results, output)

    return create_operation(
        "sdy.constant",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`data_flow_edge`

A data flow edge of some op X defines a bridge between a set of sources
(each is either an operand of X or an operand of X\'s block terminator) and
a set of targets (each is either a result of X or a block argument of X),
such that all sources and targets should be sharded in the same way.

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

This while op has n data flow edges, the i-th data flow edges is between
sources `x_i`, `return_value_i` and targets `y_i`, `pred_arg_i`,
`body_arg_i`.

An `sdy.data_flow_edge` takes as input the owner of an edge (can be
any of the targets, but preferably an op result rather than a block
argument), which shouldn\'t have any other uses. This op isn\'t pure because
it can take an input that originally didn\'t have any uses.

The `sdy.data_flow_edge` also holds an optional sharding for all targets of
the edge, and that sharding should be updated instead of the targets\'
sharding (if can be attached) during propagation. This is useful when an op
has many edges, as it\'s much more efficient to:
- propagate through each edge separately.
- update the sharding of each edge separately instead of all targets at once
  (e.g. an op has a single immutable `TensorShardingPerValueAttr` for result
  shardings).
- add each edge to the worklist separately when the sharding of a source has
  changed.

Propagation will propagate shardings between all sources and targets of a
`sdy.data_flow_edge` as if it was a regular op with the sources as operands
and targets as results, and an identity `sdy.op_sharding_rule`. That means
that forward propagation is from sources to targets and backwards
propagation is from targets to sources.

We don\'t allow the input of a `sdy.data_flow_edge` to be defined by an
`SdyDialect` op, so we can assume that it\'s defined by an op that has
unregistered `sdy.sharding` attribute.

NOTE: it\'s NOT the responsibility of the `sdy.data_flow_edge` to link
between sources and targets, it\'s simply attached to the owner of the edge.
The op that this edge is bound to (while in the example above) is
responsible for providing this information.
"""
function data_flow_edge(
    input::Value;
    result=nothing::Union{Nothing,IR.Type},
    sharding=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(sharding) && push!(attributes, namedattribute("sharding", sharding))

    return create_operation(
        "sdy.data_flow_edge",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`manual_computation`

Jump into a region written in terms of per-device local code with explicit
collectives, where logical shapes match local per-device physical buffer
shapes and collectives correspond exactly to physical cross-device
communication.

The body is local wrt the manual_axes. Propagation will occur through
the body on any free axes - those not in the manual_axes list.

Note that any unranked tensors are expected to have a sharding with rank 0,
i.e. fully replicated.

**Constraints:**
- Elements in `in_shardings` and `out_shardings` must satisfy the constraints listed in `TensorShardingAttr`.
- The number of global and local tensor inputs/outputs of the op region must match.
- The manual axes must come before any free axes in each dim sharding.
- The manual axes cannot introduce padding. Namely, the dimension size must be divisible by the corresponding manual axes size.
- The global and local shapes of the op regions arguments/results must match.
"""
function manual_computation(
    tensors::Vector{Value};
    results::Vector{IR.Type},
    in_shardings,
    out_shardings,
    manual_axes,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[tensors...,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("in_shardings", in_shardings),
        namedattribute("out_shardings", out_shardings),
        namedattribute("manual_axes", manual_axes),
    ]

    return create_operation(
        "sdy.manual_computation",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`mesh`

Defines a new named mesh. All meshes in a module must have the same number
of devices (except for meshes with a single device_id).
The mesh is a `Symbol` operation that appears in the module\'s
`SymbolTable` and can be referenced by its `name`.
"""
function mesh(; sym_name, mesh, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("mesh", mesh)
    ]

    return create_operation(
        "sdy.mesh",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`named_computation`

Groups a computation, i.e. a block of operations, and gives it a name.
Propagation will flow in/out of the region as if everything was inlined.

This can be used to handle propagating through call instructions to other
functions. Any users of Shardy should write an import/export pass that
converts their call ops to `sdy.named_computation` ops, duplicating/copying
the body of the called function into the body of the `named_computation`.

The type of each block arguments and returned values in the region must be
the same as the type of the operands and results type of the op.

# Example

```mlir
%1 = sdy.named_computation<\"foo\">(%0) (%arg1: tensor<16x32xf32>) {
  sdy.return %arg1 : tensor<16x32xf32>
} : (tensor<16x32xf32>) -> tensor<16x32xf32>
```
"""
function named_computation(
    operands::Vector{Value};
    result_0::Vector{IR.Type},
    name,
    in_shardings=nothing,
    out_shardings=nothing,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result_0...,]
    operands = Value[operands...,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name),]
    !isnothing(in_shardings) &&
        push!(attributes, namedattribute("in_shardings", in_shardings))
    !isnothing(out_shardings) &&
        push!(attributes, namedattribute("out_shardings", out_shardings))

    return create_operation(
        "sdy.named_computation",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`propagation_barrier`

This op operates like an identity op, outputting the same value it took as
input. But in terms of propagation, this will only allow propagation to flow
through it in a certain direction.

This prevents shardings from being propagated between the uses of the result
of the barrier op and its operand.

- `FORWARD` means shardings can only flow from the operand to the result.
- `BACKWARD` means shardings can only flow from the result to the operand.
- `NONE` means no sharding can propagate through this op.
- Cannot specify `BOTH`, as this op would be redundant.
"""
function propagation_barrier(
    input::Value;
    result=nothing::Union{Nothing,IR.Type},
    allowed_direction,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("allowed_direction", allowed_direction),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "sdy.propagation_barrier",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`reduce_scatter`

Reduces chunks of a tensor along axes specified in `reduce_scatter_axes`,
and then scatters the result along the same axes. This operation is
essentially a combination of an `sdy.all_reduce` followed by an
`sdy.all_slice` along the same `reduce_scatter_axes`.

**Constraints:**
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
- Elements in `reduce_scatter_axes` must satisfy the constraints listed in
  `AxisRefListAttr`.
- Applying `reduce_scatter_axes` to the operand sharding gets
  `out_sharding`.
"""
function reduce_scatter(
    tensor::Value;
    result=nothing::Union{Nothing,IR.Type},
    reduce_scatter_axes,
    out_sharding,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("reduce_scatter_axes", reduce_scatter_axes),
        namedattribute("out_sharding", out_sharding),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "sdy.reduce_scatter",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`reshard`

Reshards the input tensor with the specified sharding, which is different
from the input tensor\'s existing sharding.

Both ShardingConstraintOp and ReshardOp attach a sharding to a tensor. Their
lifespan is:
1. Before sharding propagation, ShardingConstraintOp is added by users.
2. Sharding propagation consumes ShardingConstraintOp. There is no
   ShardingConstraintOp in the results of sharding propagation. Instead,
   ReshardOp may be added if needed.
3. A partitioner converts a ReshardOp into a collective op (or an identity
   op). There should be no ReshardOp in the results of the partitioner.

  // TODO(b/331680067). Add a canonicalization pattern to remove redundant
  // reshard ops.
"""
function reshard(
    input::Value; result=nothing::Union{Nothing,IR.Type}, sharding, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sharding", sharding),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "sdy.reshard",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function return_(results::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[results...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "sdy.return",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`sharded_to_unreduced`

The `axes` should be used to shard the operand. This operation makes them
unreduced in the result. We have the following relationship:

all-gather(x, axes) = all-reduce(sharded-to-unreduced(x, axes), axes), where
all-gather, sharded-to-unreduced, all-reduce are applied on the same axes.

# Example
```mlir
%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{\"a\", \"b\", \"c\"}, {}, {\"d\"}\\], unreduced = [\"e\"]>]>} : tensor<8x8x8xf32>
%2 = sdy.sharded_to_unreduced [{\"b\", \"c\"}, {}, {\"d\"}\\] %1 out_sharding=<@mesh, [{\"a\"}, {}, {}\\], unreduced = [\"b\", \"c\", \"d\", \"e\"]> : tensor<8x8x8xf32>
```

**Constraints:**
- Must satisfy the constraints listed in `Sdy_CollectiveOpInterface`.
- Elements in `axes` must satisfy the constraints listed in `AxisRefListAttr`.
- Applying `axes` to the operand sharding gets `out_sharding`.
"""
function sharded_to_unreduced(
    tensor::Value;
    result=nothing::Union{Nothing,IR.Type},
    axes,
    out_sharding,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("axes", axes), namedattribute("out_sharding", out_sharding)
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "sdy.sharded_to_unreduced",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`sharding_constraint`

Attaches a sharding to an intermediate tensor (e.g. the result of a matmul)
to indicate that this is how that tensor, or a subset of its uses, should be
sharded.

If the sharding has open dimensions and unconstraint axes, it means the
tensor can be further sharded along the open dimensions.

This op can either:
- Have no uses (dangling) - which means the attached sharding is how the
  input tensor itself should be sharded.
- Have uses - which means the attached sharding is how the uses of the
  sharding constraint op should be sharded, while other uses of the input
  tensor might have a different sharding (if the input tensor has no other
  uses then the behavior is the same as the no uses case).
"""
function sharding_constraint(
    input::Value; result=nothing::Union{Nothing,IR.Type}, sharding, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sharding", sharding),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "sdy.sharding_constraint",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`sharding_group`

This op provides an interface to assign tensors to sharding groups (
groups of tensors that will be enforced to have identical shardings).
During propagation, as soon as one group element is sharded, all other
members will be sharded in exactly the same way. This operation takes the
argument group ID and returns no result, but instead modifies the internal
sharding group representation to add the input tensor to the group with the
given ID.
"""
function sharding_group(input::Value; group_id, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("group_id", group_id),]

    return create_operation(
        "sdy.sharding_group",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

end # sdy
