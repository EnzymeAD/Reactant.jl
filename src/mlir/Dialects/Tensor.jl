module tensor
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
`bitcast`

Bitcast a tensor from one type to another type of equivalent element width.
If both are ranked, then the rank should be the same and static dimensions
should match.

# Example

```mlir
// Bitcast from unsigned to signed or signless integer.
%2 = tensor.bitcast %1 : tensor<4xui32> to tensor<4xi32>
```
"""
function bitcast(source::Value; dest::IR.Type, location=Location())
    op_ty_results = IR.Type[dest,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tensor.bitcast",
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
`cast`

Convert a tensor from one type to an equivalent type without changing any
data elements. The source and destination types must both be tensor types
with the same element type. If both are ranked, then the rank should be the
same and static dimensions should match. The operation is invalid if
converting to a mismatching constant dimension.

# Example

```mlir
// Convert from unknown rank to rank 2 with unknown dimension sizes.
%2 = tensor.cast %1 : tensor<*xf32> to tensor<?x?xf32>

// Convert to a type with more known dimensions.
%3 = tensor.cast %2 : tensor<?x?xf32> to tensor<4x?xf32>

// Discard static dimension and rank information.
%4 = tensor.cast %3 : tensor<4x?xf32> to tensor<?x?xf32>
%5 = tensor.cast %4 : tensor<?x?xf32> to tensor<*xf32>
```
"""
function cast(source::Value; dest::IR.Type, location=Location())
    op_ty_results = IR.Type[dest,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tensor.cast",
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
`collapse_shape`

The `tensor.collapse_shape` op produces a new tensor of lower (or equal)
rank whose dimension sizes are a reassociation of the original `src` dimensions.

A reassociation is defined as a continuous grouping of dimensions and is
represented by an array of DenseI64ArrayAttr attribute. The reassociation
maps are applied to the operand shape to obtain the result shape.


# Example

```mlir
// Dimension collapse (i, j) -> i\' and k -> k\'
%b = tensor.collapse_shape %a [[0, 1], [2]]
    : tensor<?x?x?xf32> into tensor<?x?xf32>
```
"""
function collapse_shape(src::Value; result::IR.Type, reassociation, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("reassociation", reassociation),]

    return create_operation(
        "tensor.collapse_shape",
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
`concat`

The \"concat\" operation constructs a tensor out of a variadic list of input
tensors, concatenated along a static dimension number. All inputs and the
result type must share the same rank.

`dim` specifies the dimension along which to concatenate. The size of the
concatenated dimension in the result must be equal to the sum of the sizes
of the inputs along that dimension. All other dimensions in both the inputs
and result must be the same size.

# Example

```mlir
%0 = tensor.concat dim(0) %0, %1, %2 :
    (tensor<3x6xf32>, tensor<3x6xf32>, tensor<1x6xf32) -> tensor<7x6xf32>

// Dynamic + dynamic -> static
%0 = tensor.concat dim(1) %0, %1, %2 :
    (tensor<3x?xf32>, tensor<3x2xf32>, tensor<3x?xf32) -> tensor<3x10xf32>
```
"""
function concat(inputs::Vector{Value}; result::IR.Type, dim, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dim", dim),]

    return create_operation(
        "tensor.concat",
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
`dim`

The `tensor.dim` operation takes a tensor and a dimension operand of type
`index`. It returns the size of the requested dimension of the given
tensor. If the dimension index is out of bounds, the behavior is undefined.

The specified tensor type is that of the first operand.

# Example

```mlir
// Always returns 4, can be constant folded:
%c0 = arith.constant 0 : index
%x = tensor.dim %A, %c0 : tensor<4x?xf32>

// Return the dynamic dimension of %A.
%c1 = arith.constant 1 : index
%y = tensor.dim %A, %c1 : tensor<4x?xf32>

// Equivalent generic form:
%x = \"tensor.dim\"(%A, %c0) : (tensor<4x?xf32>, index) -> index
%y = \"tensor.dim\"(%A, %c1) : (tensor<4x?xf32>, index) -> index
```
"""
function dim(
    source::Value, index::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[source, index]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tensor.dim",
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
`empty`

`tensor.empty` is an operation that defines a tensor of a particular shape.
The shape could be dynamic or static. The contents of the tensor are
unspecified and the only purpose of the op result is to materialize the
specified shape in IR and make it available to other transformations.

`tensor.empty` is useful in transformations that expect destination style
ops. I.e., ops that implement `DestinationStyleOpInterface`. Ops that are
not in destination style can be made compatible with such transformations
with a `tensor.empty` destination.

Note: This op can be lowered to a `bufferization.alloc_tensor`, at which
point it turns into an explicit buffer allocation.
"""
function empty(dynamicSizes::Vector{Value}; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[dynamicSizes...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tensor.empty",
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
`expand_shape`

The `tensor.expand_shape` op produces a tensor of higher (or equal)
rank than the operand `src` whose dimension sizes are a reassociation of
`src`.

A reassociation is defined as a continuous grouping of dimensions and is
represented with an array of DenseI64ArrayAttr attribute.  The reassociation
maps applied to the result tensor with the higher rank must result in the
operand tensor with the smaller rank.

The representation for the output shape supports a partially-static
specification via attributes specified through the `static_output_shape`
argument.  A special sentinel value `ShapedType::kDynamic` encodes that the
corresponding entry has a dynamic value.  There must be exactly as many SSA
inputs in `output_shape` as there are `ShapedType::kDynamic` entries in
`static_output_shape`.

# Example

```mlir
// Dimension expansion i -> (i\', j\') and (k) -> (k\')
%b = tensor.expand_shape %a [[0, 1], [2]] output_shape [%sz0, %sz1, 32]
    : tensor<?x32xf32> into tensor<?x?x32xf32>
```
"""
function expand_shape(
    src::Value,
    output_shape::Vector{Value};
    result::IR.Type,
    reassociation,
    static_output_shape,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[src, output_shape...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("reassociation", reassociation),
        namedattribute("static_output_shape", static_output_shape),
    ]

    return create_operation(
        "tensor.expand_shape",
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
`extract`

The `tensor.extract` op reads a ranked tensor and returns one element as
specified by the given indices. The result of the op is a value with the
same type as the elements of the tensor. The arity of indices must match
the rank of the accessed value. All indices should all be of `index` type.

# Example

```mlir
%4 = tensor.extract %t[%1, %2] : tensor<4x4xi32>
%5 = tensor.extract %rt[%1, %2] : tensor<?x?xi32>
```
"""
function extract(
    tensor::Value,
    indices::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tensor, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tensor.extract",
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
`extract_slice`

The \"extract_slice\" operation extract a tensor from another tensor as
specified by the operation\'s offsets, sizes and strides arguments.

The extract_slice operation supports the following arguments:

* source: the \"base\" tensor from which to extract a slice.
* offsets: tensor-rank number of offsets into the \"base\" tensor from which
           to extract the slice.
* sizes: tensor-rank number of sizes which specify the sizes of the result
         tensor type.
* strides: tensor-rank number of strides specifying subsampling in each
           dimension.

The representation based on offsets, sizes and strides support a
partially-static specification via attributes specified through the
`static_offsets`, `static_sizes` and `static_strides` arguments. A special
sentinel value ShapedType::kDynamic encodes that the corresponding entry has
a dynamic value.

After buffer allocation, the \"extract_slice\" op is expected to lower into a
memref.subview op.

An extract_slice operation may additionally reduce the rank of the resulting
tensor by removing dimensions that are statically known to be of size 1.
This rank-reduction behavior is not required by the op semantics: this
flexibility allows to progressively drop unit dimensions while lowering
between different flavors of ops on that operate on tensors.

#### Verification vs Inference in the rank-reduced case

Note that there may be multiple ways to infer a resulting rank-reduced type.
  e.g. 1x6x1 could potentially rank-reduce to either 1x6 or 6x1 2-D shapes.

To disambiguate, the inference helpers `inferCanonicalRankReducedResultType`
only drop the first unit dimensions, in order:
  e.g. 1x6x1 rank-reduced to 2-D will infer the 6x1 2-D shape, but not 1x6.

Verification however has access to result type and does not need to infer.
The verifier calls `isRankReducedType(getSource(), getResult())` to
determine whether the result type is rank-reduced from the source type.
This computes a so-called rank-reduction mask, consisting of dropped unit
dims, to map the rank-reduced type to the source type by dropping ones:
  e.g. 1x6 is a rank-reduced version of 1x6x1 by mask {2}
       6x1 is a rank-reduced version of 1x6x1 by mask {0}
       1x2x1x4 is a rank-reduced version of 1x1x2x1x1x4x1 by mask {1, 4, 6}
         (remaining common 1 dimensions are matched eagerly)

# Example

```mlir
// Rank-reducing extract_slice.
%1 = tensor.extract_slice %0[0, 0, 0][1, 16, 4][1, 1, 1] :
  tensor<8x16x4xf32> to tensor<16x4xf32>
%3 = tensor.extract_slice %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1] :
  tensor<8x16x4xf32> to tensor<1x?xf32>
```
"""
function extract_slice(
    source::Value,
    offsets::Vector{Value},
    sizes::Vector{Value},
    strides::Vector{Value};
    result::IR.Type,
    static_offsets,
    static_sizes,
    static_strides,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[source, offsets..., sizes..., strides...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("static_offsets", static_offsets),
        namedattribute("static_sizes", static_sizes),
        namedattribute("static_strides", static_strides),
    ]
    push!(
        attributes,
        operandsegmentsizes([1, length(offsets), length(sizes), length(strides)]),
    )

    return create_operation(
        "tensor.extract_slice",
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
`from_elements`

Create a N-D tensor from a range of same-type arguments. The number of
provided `elements` should equal to the number of the elements in the
result type. The `elements` correspond to a flattened tensor.

# Example

```mlir
tensor.from_elements %a, %b, %c, %d, %e, %f :  tensor<2x3xindex>
```

will result in a tensor

[[%a, %b, %c]
 [%d, %e, %f]]
"""
function from_elements(elements::Vector{Value}; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[elements...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tensor.from_elements",
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
`gather`

The `gather` operation extracts a subset of the elements from a `source`
tensor at the given indices.

In its most general form, the tensor of indices specifies all the coordinates
of every element to extract (i.e. COO format, without the payload).
The indices are expected to be confined to coordinate values that fit the
range of the `source` tensor, otherwise the behavior is undefined.

The leading dimensions of the index tensor give the result tensor its leading
dimensions. The trailing dimensions of the result tensor are obtained from
the source tensor by omitting the dimensions specified in `gather_dims`
(rank-reducing semantics) or setting them to `1` (rank-preserving semantics)
(see examples).
The trailing dimension of the index tensor contains the coordinates and is
expected to have its size equal to the number of dimensions being gathered.
This convention allows an idiomatic specification and lowering of \"gathering
multiple N-D slices from the source tensor\".

Note: in the examples below, we separate out the indexing part of the tensor
type by a whitespace for readability purposes.

# Example

```mlir
    // For each 1x2 triple of coordinates in %indices, extract the
    // element (i.e. 0-D subset) at the coordinates triple in %source.
    //
    %out = tensor.gather %source[%indices] gather_dims([0, 1, 2]) :
      (tensor<4x4x4xf32>, tensor<1x2x 3xindex>) -> tensor<1x2x 1x1x1xf32>

    // Note: result type may be further rank-reduced to tensor<1x2x f32>.
```

A slice variant is provided to allow specifying whole slices of the source
tensor.

# Example

```mlir
    // For each 5x6 singleton of coordinates in %indices, extract the 2-D
    // slice %source[*, %indices[...]:%indices[...] + 1, *] with the indices
    // corresponding to the `gather_dims` attribute specified by %indices.
    //
    %out = tensor.gather %source[%indices] gather_dims([1]) :
      (tensor<3x4x5xf32>, tensor<6x7x 1xindex>) -> tensor<6x7x 3x1x5xf32>

    // Note: result type may be further rank-reduced to tensor<6x7x 3x5xf32>.
```

The dimensions specified in the gather_dims attribute are ones for which the
result tensor has size `1`.
I.e. if the source type is `axbxcxd` and the coordinates are [1, 3], then
the shape suffix is `ax1xcx1`.
Gather also allows rank-reducing semantics where the shape `ax1xcx1` can be
further simplified to `axc`.

The elemental type of the indices tensor can be any integer type.
In the absence of target-specific or problem specific information the default
type one should use is `index`.

This operation does not support unranked tensors.

An optional `unique` unit attribute may be specified to indicate that the
coordinates in `indices` are statically guaranteed to be unique at runtime.
Incorrectly setting the `unique` attribute when the coordinates are not truly
unique is undefined behavior.

Only full slices are meant to be supported by this op, if one desires
partial slices (e.g. strided windows) one should compose this op with other
tensor ops (e.g. tensor.extract_slice). This is to avoid a slippery slope of
complexity that would make the op unusable in practice.

At the tensor-level, the index tensor is specified in an AoS form (i.e.
coordinate tuple is the most minor). It is the responsibility of further
lowerings and bufferization to implement various concrete layouts.

Note: As currently specified, the operation must lower to an abstraction that
performs copies to the output tensor. This is because the buffer type system
is currently not rich enough to allow multiple non-contiguous views in the
same type. This is visible more clearly in a notional buffer version of the
op:

```mlir
    // memref<?x4x1xf32> is a contiguous buffer of ?x4x1 elements.
    // gather from random source slices must copy to the contiguous output.
    %out = memref.gather %source[%indices] gather_dims([1]) :
      (memref<4x4xf32>, memref<?x 1xindex>) -> memref<?x 4x1xf32>

    // Nested buffer support would allow gather to directly index into the
    // source buffer (i.e. represent a jagged view into the source).
    %out = memref.gather %source[%indices] gather_dims([1]) :
      (memref<4x4xf32>, memref<?x 1xindex>) -> memref<? x memref<4x1xf32>>
```
"""
function gather(
    source::Value,
    indices::Value;
    result::IR.Type,
    gather_dims,
    unique=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[source, indices]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("gather_dims", gather_dims),]
    !isnothing(unique) && push!(attributes, namedattribute("unique", unique))

    return create_operation(
        "tensor.gather",
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
`generate`

This operation creates a dynamically sized tensor with elements of any type.
It expects one index operand per dynamic extent of the result tensor.

The body region defines the tensor\'s elements. It takes index operands as
its region arguments that span the index space. The element at the given
position is yielded with the `yield` operation (see `YieldOp`). There is
no defined ordering to the invocations of the body. It is conceptually
a \"parallel map\" operation.

# Example

```mlir
  %tnsr = tensor.generate %m, %n {
  ^bb0(%i : index, %j : index, %k : index):
    ...
    yield %elem : f32
  } : tensor<?x3x?f32>
```
"""
function generate(
    dynamicExtents::Vector{Value}; result::IR.Type, body::Region, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[dynamicExtents...,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tensor.generate",
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
`insert`

The `tensor.insert` op inserts a scalar into a ranked tensor `dest` as
specified by the operation\'s indices.

It returns a copy of `dest` with the indexed position updated to the value
of `scalar`.

The arity of `indices `must match the rank of the tensor `dest`. All
indices should be of `index` type.

# Example

```mlir
%4 = tensor.insert %t into %dest[%1, %2] : tensor<4x4xi32>
%5 = tensor.insert %rt into %dest[%1, %2] : tensor<?x?xi32>
```
"""
function insert(
    scalar::Value,
    dest::Value,
    indices::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[scalar, dest, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tensor.insert",
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
`insert_slice`

The \"insert_slice\" operation insert a tensor `source` into another
tensor `dest` as specified by the operation\'s offsets, sizes and strides
arguments.

It returns a copy of `dest` with the proper slice updated with the value
of `source`.

The insert_slice operation supports the following arguments:

* source: the tensor that is inserted.
* dest: the tensor into which the source tensor is inserted.
* offsets: tensor-rank number of offsets into the `dest` tensor into which
           the slice is inserted.
* sizes: tensor-rank number of sizes which specify the sizes of the source
         tensor type.
* strides: tensor-rank number of strides that specify subsampling in each
           dimension.

The representation based on offsets, sizes and strides support a
partially-static specification via attributes specified through the
`static_offsets`, `static_sizes` and `static_strides` arguments. A special
sentinel value ShapedType::kDynamic encodes that the corresponding entry has
a dynamic value.

After buffer allocation, the \"insert_slice\" op is expected to lower into a
memref.subview op.

An insert_slice operation may additionally specify insertion into a tensor
of higher rank than the source tensor, along dimensions that are statically
known to be of size 1.
This rank-altering behavior is not required by the op semantics: this
flexibility allows to progressively drop unit dimensions while lowering
between different flavors of ops on that operate on tensors.
The rank-altering behavior of tensor.insert_slice matches the rank-reducing
behavior of tensor.extract_slice.

#### Verification in the rank-reduced case

The same verification discussion and mechanisms apply as for ExtractSliceOp.
Unlike ExtractSliceOp however, there is no need for a specific inference.

# Example

```mlir
// Rank-altering insert_slice.
%1 = tensor.insert_slice %t into %0[0, 0, 0][1, 16, 4][1, 1, 1] :
  tensor<16x4xf32> into tensor<8x16x4xf32>
%3 = tensor.insert_slice %tt into %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1] :
  tensor<1x?xf32> into tensor<8x16x4xf32>
```
"""
function insert_slice(
    source::Value,
    dest::Value,
    offsets::Vector{Value},
    sizes::Vector{Value},
    strides::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    static_offsets,
    static_sizes,
    static_strides,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source, dest, offsets..., sizes..., strides...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("static_offsets", static_offsets),
        namedattribute("static_sizes", static_sizes),
        namedattribute("static_strides", static_strides),
    ]
    push!(
        attributes,
        operandsegmentsizes([1, 1, length(offsets), length(sizes), length(strides)]),
    )
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tensor.insert_slice",
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
`pad`

`tensor.pad` is an operation that pads the `source` tensor
with given `low` and `high` padding config.

The PadOp operation supports the following arguments:

* source: the \"base\" tensor on which to pad.
* low: A list contains the padding along the start of each
       dimension, i.e., how many padded values are prepended
       to the beginning of the tensor in each dimension.
* high: A list contains the padding along the end of each
        dimension, i.e., how many padded values are appended
        to the end of the tensor in each dimension.
* nofold: indicates that the operation should not be folded when source and
          result types are equal.

The result tensor dimensions are `low[i]` + `dim[i]` + `high[i]` for each
dimension `i`. The number of elements of `low` and `high` must match the
rank of the input tensor. They can be either a constant or a dynamic value.

The region of the `tensor.pad` operation returns the value to use
for the padding. The arguments of the region represent the index
of the source being accessed. There should be as many arguments as
the rank of the `source` tensor. The value `yield`-ed by the
region is used as the value of the view at the given position.

If `nofold` is set, the padding operation will not be folded away even
if the source type and the padded type have the same static shape. This can
be used, e.g., for packing or promotion to faster memory.

Example 1: add 3 zeros to the beginning and 5 zeros to the end of a 1D
tensor.

```mlir
  %arg0 = ... : tensor<10xi32>
  %c0_i32 = arith.constant 0 : i32
  %padded = tensor.pad %arg0 low[3] high[5] {
  ^bb0(%arg1: index):
    tensor.yield %c0_i32 : i32
  } : tensor<10xi32> to tensor<18xi32>
```

Example 2: add 1 value to the beginning of dimension 0, 2 values to the end
of dimension 0, 2 values to the start of dimension 1, and 3 values to the
end of dimension 1.

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %0 low[1, 2] high[2, 3] {
  ^bb0(%arg0 : index, %arg1 : index):
    tensor.yield %pad_value : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
```

Example 3:

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %arg0 low[2, %arg1, 3, 3] high[3, 3, %arg1, 2] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %pad_value : f32
  } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
```

Example 4:

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %arg0 low[0, 0] high[%ub0, %ub1] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %pad_value : f32
  } : tensor<2x3xf32> to tensor<?x?xf32>
```

Example 5: Force a padded value to be always exist with `nofold`, even
though the padding config specifies that no new elements will be added to
the tensor.

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %arg0 nofold low[0, 0] high[0, 0] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %pad_value : f32
  } : tensor<2x3xf32> to tensor<2x3xf32>
```
"""
function pad(
    source::Value,
    low::Vector{Value},
    high::Vector{Value};
    result::IR.Type,
    static_low,
    static_high,
    nofold=nothing,
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[source, low..., high...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("static_low", static_low), namedattribute("static_high", static_high)
    ]
    push!(attributes, operandsegmentsizes([1, length(low), length(high)]))
    !isnothing(nofold) && push!(attributes, namedattribute("nofold", nofold))

    return create_operation(
        "tensor.pad",
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
`parallel_insert_slice`

The `parallel_insert_slice` yields a subset tensor value to its parent
InParallelOpInterface. These subset tensor values are aggregated to
in some unspecified order into a full tensor value returned by the parent
parallel iterating op.
The `parallel_insert_slice` is one such op allowed in the
InParallelOpInterface op.

Conflicting writes result in undefined semantics, in that the indices written
to by multiple parallel updates might contain data from any of the updates,
or even a malformed bit pattern.

If an index is updated exactly once, the value contained at that index
in the resulting tensor will be equal to the value at a corresponding index
of a slice that was used for the updated. If an index is not updated at all,
its value will be equal to the one in the original tensor.

This op does not create a new value, which allows maintaining a clean
separation between the subset and full tensor.

Note that we cannot mark this operation as pure (Pures), even
though it has no side effects, because it will get DCEd during
canonicalization.

The parallel_insert_slice operation supports the following arguments:

* source: the tensor that is inserted.
* dest: the tensor into which the source tensor is inserted.
* offsets: tensor-rank number of offsets into the `dest` tensor into which
           the slice is inserted.
* sizes: tensor-rank number of sizes which specify the sizes of the source
         tensor type.
* strides: tensor-rank number of strides that specify subsampling in each
           dimension.

The representation based on offsets, sizes and strides support a
partially-static specification via attributes specified through the
`static_offsets`, `static_sizes` and `static_strides` arguments. A special
sentinel value ShapedType::kDynamic encodes that the corresponding entry has
a dynamic value.

After buffer allocation, the \"parallel_insert_slice\" op is expected to lower
into a memref.subview op.

A parallel_insert_slice operation may additionally specify insertion into a
tensor of higher rank than the source tensor, along dimensions that are
statically known to be of size 1.
This rank-altering behavior is not required by the op semantics: this
flexibility allows to progressively drop unit dimensions while lowering
between different flavors of ops on that operate on tensors.
The rank-altering behavior of tensor.parallel_insert_slice matches the
rank-reducing behavior of tensor.insert_slice and tensor.extract_slice.

#### Verification in the rank-reduced case

The same verification discussion and mechanisms apply as for ExtractSliceOp.
Unlike ExtractSliceOp however, there is no need for a specific inference.
"""
function parallel_insert_slice(
    source::Value,
    dest::Value,
    offsets::Vector{Value},
    sizes::Vector{Value},
    strides::Vector{Value};
    static_offsets,
    static_sizes,
    static_strides,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source, dest, offsets..., sizes..., strides...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("static_offsets", static_offsets),
        namedattribute("static_sizes", static_sizes),
        namedattribute("static_strides", static_strides),
    ]
    push!(
        attributes,
        operandsegmentsizes([1, 1, length(offsets), length(sizes), length(strides)]),
    )

    return create_operation(
        "tensor.parallel_insert_slice",
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
`rank`

The `tensor.rank` operation takes a tensor operand and returns its rank.

# Example

```mlir
%0 = tensor.rank %arg0 : tensor<*xf32>
%1 = tensor.rank %arg1 : tensor<?x?xf32>
```
"""
function rank(tensor::Value; result_0=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "tensor.rank",
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
`reshape`

The `reshape` operation converts a tensor from one type to an equivalent
type with a provided shape. The source and destination types are compatible
if both have the same element type, same number of elements. The following
combinations are possible:

a. Source type is ranked or unranked. Shape argument has static size.
Result type is ranked.

```mlir
// Reshape statically-shaped tensor.
%dst = tensor.reshape %src(%shape)
         : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
%dst0 = tensor.reshape %src(%shape0)
         : (tensor<4x1xf32>, tensor<2xi32>) -> tensor<2x2xf32>
// Flatten unranked tensor.
%dst = tensor.reshape %src(%shape)
         : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
```

b. Source type is ranked or unranked. Shape argument has dynamic size.
Result type is unranked.

```mlir
// Reshape dynamically-shaped 1D tensor.
%dst = tensor.reshape %src(%shape)
         : (tensor<?xf32>, tensor<?xi32>) -> tensor<*xf32>
// Reshape unranked tensor.
%dst = tensor.reshape %src(%shape)
         : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
```
"""
function reshape(source::Value, shape::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source, shape]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tensor.reshape",
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
`scatter`

The `scatter` operation inserts a `source` tensor into a `dest` tensor at
the given indices.

In its most general form, the tensor of indices specifies all the coordinates
of every element to insert (i.e. COO format, without the payload).
The indices are expected to be confined to coordinate values that fit the
range of the `dest` tensor, otherwise the behavior is undefined.

The leading dimensions of the index tensor must match that of the dest
tensor. The trailing dimensions of the dest tensor must match those of the
source tensor by omitting the dimensions specified in scatter_dims
(rank-reducing semantics) or setting them to `1` (rank-preserving semantics)
(see examples).
This convention allows an idiomatic specification and lowering of
\"scattering multiple N-D slices into the dest tensor\".
The result type must match the type of the dest tensor.

Note: in the examples below, we separate out the indexing part of the tensor
type by a whitespace for readability purposes.

# Example

```mlir
    // For each 1x2 triple of coordinates in %indices, insert the
    // element (i.e. 0-D subset) at the coordinates triple in %dest.
    //
    %out = tensor.scatter %source into %dest[%indices]
        scatter_dims([0, 1, 2]) unique :
      (tensor<1x2x 1x1x1xf32>, tensor<4x4x4xf32>, tensor<1x2x 3xindex>)
        -> tensor<4x4x4xf32>

    // Note: source type may be further rank-reduced to tensor<1x2x f32>.
```

A slice variant is provided to allow specifying insertion of whole tensor
slices into the `dest` tensor.

# Example

```mlir
    // For each 3 singleton of coordinates in %indices, insert the 2-D
    // slice into %dest[*, %indices[...]:%indices[...] + 1, *] with the
    // indices corresponding to the scatter_dims attribute specified by
    // %indices.
    //
    %out = tensor.scatter %source into %dest[%indices] scatter_dims([1]) unique :
      (tensor<3x 4x1x6xf32>, tensor<4x5x6xf32>, tensor<3x 1xindex>)
        -> tensor<4x5x6xf32>
```

The dimensions specified in the scatter_dims attribute are ones for which the
source tensor has size `1`.
I.e. if the dest type is `axbxcxd` and the coordinates are [1, 3], then
the source type suffix is `ax1xcx1`.
Scatter also allows rank-reducing semantics where the shape `ax1xcx1` can be
further simplified to `axc`.

The elemental type of the indices tensor can be any integer type.
In the absence of target-specific or problem specific information the default
type one should use is `index`.

This operation does not support unranked tensors.

A `unique` unit attribute must be be specified to indicate that the
coordinates are statically guaranteed to be unique at runtime. If coordinates
are not truly unique at runtime, the behavior is undefined.

Only full slices are meant to be supported by this op, if one desires
partial slices (e.g. strided windows) one should compose this op with other
tensor ops (e.g. tensor.insert_slice). This is to avoid a slippery slope of
complexity that would make the op unusable in practice.

At the tensor-level, the index tensor is specified in an AoS form (i.e.
coordinate tuple is the most minor). It is the responsibility of further
lowerings and bufferization to implement various concrete layouts.

Note: As currently specified, the operation must lower to an abstraction that
performs copies to the output tensor. This is because the buffer type system
is currently not rich enough to allow multiple non-contiguous views in the
same type. This is visible more clearly in a notional buffer version of the
op:

```mlir
    // memref<?x 4xf32> is a contiguous buffer of ?x4 elements, scatter into
    // random dest slices must copy to the contiguous dest.
    //
    some_side_effecting_op_writing_into %source, ...: memref<3x 4xf32>
    memref.scatter %source into %dest[%indices] scatter_dims([1]) unique :
      (memref<3x 4xf32>, memref<?x 4xf32>, memref<?x 1xindex>)

    // Nested buffer support in the producing op would allow writing directly
    // into the dest buffer.
    %v = some_nested_buffer_view_op %dest[%indices] scatter_dims([1]) unique :
      memref<? x memref<4xf32>>
    some_side_effecting_op_writing_into %v, ...: memref<? x memref<4xf32>>
```
"""
function scatter(
    source::Value,
    dest::Value,
    indices::Value;
    result::IR.Type,
    scatter_dims,
    unique=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[source, dest, indices]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("scatter_dims", scatter_dims),]
    !isnothing(unique) && push!(attributes, namedattribute("unique", unique))

    return create_operation(
        "tensor.scatter",
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
`splat`

Broadcast the operand to all elements of the result tensor.

An additional argument of type `index` must be provided for each dynamic
dimension present in the result type.

Example for a statically shaped tensor:

```mlir
%s = arith.constant 1.0 : f32
%t = tensor.splat %s : tensor<8x16xf32>
```

Example for a tensor containing dynamic dimensions:

```mlir
// Broadcasts %s to a 3D dynamically shaped tensor, with %m and %n binding
// to dimensions 0 and 2 of the resulting tensor, respectively.
%m = arith.constant 10 : index
%n = arith.constant 30 : index
%t = tensor.splat %s[%m, %n] : tensor<?x20x?xf32>
```
"""
function splat(
    input::Value, dynamicSizes::Vector{Value}; aggregate::IR.Type, location=Location()
)
    op_ty_results = IR.Type[aggregate,]
    operands = Value[input, dynamicSizes...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tensor.splat",
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
`yield`

This operation is used to yield a single value from a within a region. It
is used to create dynamically sized tensors
(see `tensor.generate` and `tensor.pad` ops).
"""
function yield(value::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tensor.yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # tensor
