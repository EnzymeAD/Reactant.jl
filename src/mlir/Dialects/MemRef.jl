module memref
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
`assume_alignment`

The `assume_alignment` operation takes a memref and an integer of alignment
value, and internally annotates the buffer with the given alignment. If
the buffer isn\'t aligned to the given alignment, the behavior is undefined.

This operation doesn\'t affect the semantics of a correct program. It\'s for
optimization only, and the optimization is best-effort.
"""
function assume_alignment(memref::Value; alignment, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[memref,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("alignment", alignment),]

    return create_operation(
        "memref.assume_alignment",
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
`atomic_rmw`

The `memref.atomic_rmw` operation provides a way to perform a read-modify-write
sequence that is free from data races. The kind enumeration specifies the
modification to perform. The value operand represents the new value to be
applied during the modification. The memref operand represents the buffer
that the read and write will be performed against, as accessed by the
specified indices. The arity of the indices is the rank of the memref. The
result represents the latest value that was stored.

# Example

```mlir
%x = memref.atomic_rmw \"addf\" %value, %I[%i] : (f32, memref<10xf32>) -> f32
```
"""
function atomic_rmw(
    value::Value,
    memref::Value,
    indices::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    kind,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kind", kind),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "memref.atomic_rmw",
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
`atomic_yield`

\"memref.atomic_yield\" yields an SSA value from a
GenericAtomicRMWOp region.
"""
function atomic_yield(result::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[result,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.atomic_yield",
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
`copy`

Copies the data from the source to the destination memref.

Usage:

```mlir
memref.copy %arg0, %arg1 : memref<?xf32> to memref<?xf32>
```

Source and destination are expected to have the same element type and shape.
Otherwise, the result is undefined. They may have different layouts.
"""
function copy(source::Value, target::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source, target]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.copy",
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
`generic_atomic_rmw`

The `memref.generic_atomic_rmw` operation provides a way to perform a
read-modify-write sequence that is free from data races. The memref operand
represents the buffer that the read and write will be performed against, as
accessed by the specified indices. The arity of the indices is the rank of
the memref. The result represents the latest value that was stored. The
region contains the code for the modification itself. The entry block has
a single argument that represents the value stored in `memref[indices]`
before the write is performed. No side-effecting ops are allowed in the
body of `GenericAtomicRMWOp`.

# Example

```mlir
%x = memref.generic_atomic_rmw %I[%i] : memref<10xf32> {
  ^bb0(%current_value : f32):
    %c1 = arith.constant 1.0 : f32
    %inc = arith.addf %c1, %current_value : f32
    memref.atomic_yield %inc : f32
}
```
"""
function generic_atomic_rmw(
    memref::Value,
    indices::Vector{Value};
    result::IR.Type,
    atomic_body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[memref, indices...]
    owned_regions = Region[atomic_body,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.generic_atomic_rmw",
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
`load`

The `load` op reads an element from a memref specified by an index list. The
output of load is a new value with the same type as the elements of the
memref. The arity of indices is the rank of the memref (i.e., if the memref
loaded from is of rank 3, then 3 indices are required for the load following
the memref identifier).

In an `affine.if` or `affine.for` body, the indices of a load are restricted
to SSA values bound to surrounding loop induction variables,
[symbols](Affine.md/#dimensions-and-symbols), results of a
constant operations, or the result of an
`affine.apply` operation that can in turn take as arguments all of the
aforementioned SSA values or the recursively result of such an
`affine.apply` operation.

# Example

```mlir
%1 = affine.apply affine_map<(d0, d1) -> (3*d0)> (%i, %j)
%2 = affine.apply affine_map<(d0, d1) -> (d1+1)> (%i, %j)
%12 = memref.load %A[%1, %2] : memref<8x?xi32, #layout, memspace0>

// Example of an indirect load (treated as non-affine)
%3 = affine.apply affine_map<(d0) -> (2*d0 + 1)>(%12)
%13 = memref.load %A[%3, %2] : memref<4x?xi32, #layout, memspace0>
```

**Context:** The `load` and `store` operations are specifically crafted to
fully resolve a reference to an element of a memref, and (in affine
`affine.if` and `affine.for` operations) the compiler can follow use-def
chains (e.g. through [`affine.apply`](Affine.md/#affineapply-affineapplyop)
operations) to precisely analyze references at compile-time using polyhedral
techniques. This is possible because of the
[restrictions on dimensions and symbols](Affine.md/#restrictions-on-dimensions-and-symbols)
in these contexts.
"""
function load(
    memref::Value,
    indices::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    nontemporal=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))

    return create_operation(
        "memref.load",
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
`alloc`

The `alloc` operation allocates a region of memory, as specified by its
memref type.

# Example

```mlir
%0 = memref.alloc() : memref<8x64xf32, 1>
```

The optional list of dimension operands are bound to the dynamic dimensions
specified in its memref type. In the example below, the ssa value \'%d\' is
bound to the second dimension of the memref (which is dynamic).

```mlir
%0 = memref.alloc(%d) : memref<8x?xf32, 1>
```

The optional list of symbol operands are bound to the symbols of the
memrefs affine map. In the example below, the ssa value \'%s\' is bound to
the symbol \'s0\' in the affine map specified in the allocs memref type.

```mlir
%0 = memref.alloc()[%s] : memref<8x64xf32,
                          affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
```

This operation returns a single ssa value of memref type, which can be used
by subsequent load and store operations.

The optional `alignment` attribute may be specified to ensure that the
region of memory that will be indexed is aligned at the specified byte
boundary.

```mlir
%0 = memref.alloc()[%s] {alignment = 8} :
  memref<8x64xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
```
"""
function alloc(
    dynamicSizes::Vector{Value},
    symbolOperands::Vector{Value};
    memref::IR.Type,
    alignment=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[memref,]
    operands = Value[dynamicSizes..., symbolOperands...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(dynamicSizes), length(symbolOperands)]))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))

    return create_operation(
        "memref.alloc",
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
`alloca`

The `alloca` operation allocates memory on the stack, to be automatically
released when control transfers back from the region of its closest
surrounding operation with an
[`AutomaticAllocationScope`](../Traits.md/#automaticallocationscope) trait.
The amount of memory allocated is specified by its memref and additional
operands. For example:

```mlir
%0 = memref.alloca() : memref<8x64xf32>
```

The optional list of dimension operands are bound to the dynamic dimensions
specified in its memref type. In the example below, the SSA value \'%d\' is
bound to the second dimension of the memref (which is dynamic).

```mlir
%0 = memref.alloca(%d) : memref<8x?xf32>
```

The optional list of symbol operands are bound to the symbols of the
memref\'s affine map. In the example below, the SSA value \'%s\' is bound to
the symbol \'s0\' in the affine map specified in the allocs memref type.

```mlir
%0 = memref.alloca()[%s] : memref<8x64xf32,
                           affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>>
```

This operation returns a single SSA value of memref type, which can be used
by subsequent load and store operations. An optional alignment attribute, if
specified, guarantees alignment at least to that boundary. If not specified,
an alignment on any convenient boundary compatible with the type will be
chosen.
"""
function alloca(
    dynamicSizes::Vector{Value},
    symbolOperands::Vector{Value};
    memref::IR.Type,
    alignment=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[memref,]
    operands = Value[dynamicSizes..., symbolOperands...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(dynamicSizes), length(symbolOperands)]))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))

    return create_operation(
        "memref.alloca",
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
`alloca_scope`

The `memref.alloca_scope` operation represents an explicitly-delimited
scope for the alloca allocations. Any `memref.alloca` operations that are
used within this scope are going to be cleaned up automatically once
the control-flow exits the nested region. For example:

```mlir
memref.alloca_scope {
  %myalloca = memref.alloca(): memref<4x3xf32>
  ...
}
```

Here, `%myalloca` memref is valid within the explicitly delimited scope
and is automatically deallocated at the end of the given region. Conceptually,
`memref.alloca_scope` is a passthrough operation with
`AutomaticAllocationScope` that spans the body of the region within the operation.

`memref.alloca_scope` may also return results that are defined in the nested
region. To return a value, one should use `memref.alloca_scope.return`
operation:

```mlir
%result = memref.alloca_scope {
  ...
  memref.alloca_scope.return %value
}
```

If `memref.alloca_scope` returns no value, the `memref.alloca_scope.return ` can
be left out, and will be inserted implicitly.
"""
function alloca_scope(; results::Vector{IR.Type}, bodyRegion::Region, location=Location())
    op_ty_results = IR.Type[results...,]
    operands = Value[]
    owned_regions = Region[bodyRegion,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.alloca_scope",
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
`alloca_scope_return`

`memref.alloca_scope.return` operation returns zero or more SSA values
from the region within `memref.alloca_scope`. If no values are returned,
the return operation may be omitted. Otherwise, it has to be present
to indicate which values are going to be returned. For example:

```mlir
memref.alloca_scope.return %value
```
"""
function alloca_scope_return(results::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[results...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.alloca_scope.return",
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

The `memref.cast` operation converts a memref from one type to an equivalent
type with a compatible shape. The source and destination types are
compatible if:

a. Both are ranked memref types with the same element type, address space,
and rank and:
  1. Both have the same layout or both have compatible strided layouts.
  2. The individual sizes (resp. offset and strides in the case of strided
     memrefs) may convert constant dimensions to dynamic dimensions and
     vice-versa.

If the cast converts any dimensions from an unknown to a known size, then it
acts as an assertion that fails at runtime if the dynamic dimensions
disagree with resultant destination size.

# Example

```mlir
// Assert that the input dynamic shape matches the destination static shape.
%2 = memref.cast %1 : memref<?x?xf32> to memref<4x4xf32>
// Erase static shape information, replacing it with dynamic information.
%3 = memref.cast %1 : memref<4xf32> to memref<?xf32>

// The same holds true for offsets and strides.

// Assert that the input dynamic shape matches the destination static stride.
%4 = memref.cast %1 : memref<12x4xf32, strided<[?, ?], offset: ?>> to
                      memref<12x4xf32, strided<[4, 1], offset: 5>>
// Erase static offset and stride information, replacing it with
// dynamic information.
%5 = memref.cast %1 : memref<12x4xf32, strided<[4, 1], offset: 5>> to
                      memref<12x4xf32, strided<[?, ?], offset: ?>>
```

b. Either or both memref types are unranked with the same element type, and
address space.

# Example

```mlir
Cast to concrete shape.
    %4 = memref.cast %1 : memref<*xf32> to memref<4x?xf32>

Erase rank information.
    %5 = memref.cast %1 : memref<4x?xf32> to memref<*xf32>
```
"""
function cast(source::Value; dest::IR.Type, location=Location())
    op_ty_results = IR.Type[dest,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.cast",
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

The `memref.collapse_shape` op produces a new view with a smaller rank
whose sizes are a reassociation of the original `view`. The operation is
limited to such reassociations, where subsequent, contiguous dimensions are
collapsed into a single dimension. Such reassociations never require
additional allocs or copies.

Collapsing non-contiguous dimensions is undefined behavior. When a group of
dimensions can be statically proven to be non-contiguous, collapses of such
groups are rejected in the verifier on a best-effort basis. In the general
case, collapses of dynamically-sized dims with dynamic strides cannot be
proven to be contiguous or non-contiguous due to limitations in the memref
type.

A reassociation is defined as a continuous grouping of dimensions and is
represented with an array of DenseI64ArrayAttr attribute.

Note: Only the dimensions within a reassociation group must be contiguous.
The remaining dimensions may be non-contiguous.

The result memref type can be zero-ranked if the source memref type is
statically shaped with all dimensions being unit extent. In such a case, the
reassociation indices must be empty.

Examples:

```mlir
// Dimension collapse (i, j) -> i\' and k -> k\'
%1 = memref.collapse_shape %0 [[0, 1], [2]] :
    memref<?x?x?xf32, stride_spec> into memref<?x?xf32, stride_spec_2>
```

For simplicity, this op may not be used to cast dynamicity of dimension
sizes and/or strides. I.e., a result dimension must be dynamic if and only
if at least one dimension in the corresponding reassociation group is
dynamic. Similarly, the stride of a result dimension must be dynamic if and
only if the corresponding start dimension in the source type is dynamic.

Note: This op currently assumes that the inner strides are of the
source/result layout map are the faster-varying ones.
"""
function collapse_shape(src::Value; result::IR.Type, reassociation, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("reassociation", reassociation),]

    return create_operation(
        "memref.collapse_shape",
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
`dealloc`

The `dealloc` operation frees the region of memory referenced by a memref
which was originally created by the `alloc` operation.
The `dealloc` operation should not be called on memrefs which alias an
alloc\'d memref (e.g. memrefs returned by `view` operations).

# Example

```mlir
%0 = memref.alloc() : memref<8x64xf32, affine_map<(d0, d1) -> (d0, d1), 1>>
memref.dealloc %0 : memref<8x64xf32,  affine_map<(d0, d1) -> (d0, d1), 1>>
```
"""
function dealloc(memref::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[memref,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.dealloc",
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

The `dim` operation takes a memref and a dimension operand of type `index`.
It returns the size of the requested dimension of the given memref.
If the dimension index is out of bounds the behavior is undefined.

The specified memref type is that of the first operand.

# Example

```mlir
// Always returns 4, can be constant folded:
%c0 = arith.constant 0 : index
%x = memref.dim %A, %c0 : memref<4 x ? x f32>

// Returns the dynamic dimension of %A.
%c1 = arith.constant 1 : index
%y = memref.dim %A, %c1 : memref<4 x ? x f32>

// Equivalent generic form:
%x = \"memref.dim\"(%A, %c0) : (memref<4 x ? x f32>, index) -> index
%y = \"memref.dim\"(%A, %c1) : (memref<4 x ? x f32>, index) -> index
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
        "memref.dim",
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
`dma_start`

# Syntax

```
operation ::= `memref.dma_start` ssa-use`[`ssa-use-list`]` `,`
               ssa-use`[`ssa-use-list`]` `,` ssa-use `,`
               ssa-use`[`ssa-use-list`]` (`,` ssa-use `,` ssa-use)?
              `:` memref-type `,` memref-type `,` memref-type
```

DmaStartOp starts a non-blocking DMA operation that transfers data from a
source memref to a destination memref. The source and destination memref
need not be of the same dimensionality, but need to have the same elemental
type. The operands include the source and destination memref\'s each followed
by its indices, size of the data transfer in terms of the number of elements
(of the elemental type of the memref), a tag memref with its indices, and
optionally at the end, a stride and a number_of_elements_per_stride
arguments. The tag location is used by a DmaWaitOp to check for completion.
The indices of the source memref, destination memref, and the tag memref
have the same restrictions as any load/store. The optional stride arguments
should be of \'index\' type, and specify a stride for the slower memory space
(memory space with a lower memory space id), transferring chunks of
number_of_elements_per_stride every stride until %num_elements are
transferred. Either both or no stride arguments should be specified. If the
source and destination locations overlap the behavior of this operation is
not defined.

For example, a DmaStartOp operation that transfers 256 elements of a memref
\'%src\' in memory space 0 at indices [%i, %j] to memref \'%dst\' in memory
space 1 at indices [%k, %l], would be specified as follows:

```mlir
%num_elements = arith.constant 256
%idx = arith.constant 0 : index
%tag = memref.alloc() : memref<1 x i32, affine_map<(d0) -> (d0)>, 4>
dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx] :
  memref<40 x 128 x f32>, affine_map<(d0) -> (d0)>, 0>,
  memref<2 x 1024 x f32>, affine_map<(d0) -> (d0)>, 1>,
  memref<1 x i32>, affine_map<(d0) -> (d0)>, 2>
```

If %stride and %num_elt_per_stride are specified, the DMA is expected to
transfer %num_elt_per_stride elements every %stride elements apart from
memory space 0 until %num_elements are transferred.

```mlir
dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx], %stride,
          %num_elt_per_stride :
```

* TODO: add additional operands to allow source and destination striding, and
multiple stride levels.
* TODO: Consider replacing src/dst memref indices with view memrefs.
"""
function dma_start(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.dma_start",
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
`dma_wait`

DmaWaitOp blocks until the completion of a DMA operation associated with the
tag element \'%tag[%index]\'. %tag is a memref, and %index has to be an index
with the same restrictions as any load/store index. %num_elements is the
number of elements associated with the DMA operation.

# Example

```mlir
 dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%index] :
   memref<2048 x f32>, affine_map<(d0) -> (d0)>, 0>,
   memref<256 x f32>, affine_map<(d0) -> (d0)>, 1>
   memref<1 x i32>, affine_map<(d0) -> (d0)>, 2>
 ...
 ...
 dma_wait %tag[%index], %num_elements : memref<1 x i32, affine_map<(d0) -> (d0)>, 2>
 ```
"""
function dma_wait(
    tagMemRef::Value, tagIndices::Vector{Value}, numElements::Value; location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[tagMemRef, tagIndices..., numElements]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.dma_wait",
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

The `memref.expand_shape` op produces a new view with a higher rank whose
sizes are a reassociation of the original `view`. The operation is limited
to such reassociations, where a dimension is expanded into one or multiple
contiguous dimensions. Such reassociations never require additional allocs
or copies.

A reassociation is defined as a grouping of dimensions and is represented
with an array of DenseI64ArrayAttr attributes.

# Example

```mlir
%r = memref.expand_shape %0 [[0, 1], [2]] output_shape [%sz0, %sz1, 32]
    : memref<?x32xf32> into memref<?x?x32xf32>
```

If an op can be statically proven to be invalid (e.g, an expansion from
`memref<10xf32>` to `memref<2x6xf32>`), it is rejected by the verifier. If
it cannot statically be proven invalid (e.g., the full example above; it is
unclear whether the first source dimension is divisible by 5), the op is
accepted by the verifier. However, if the op is in fact invalid at runtime,
the behavior is undefined.

The source memref can be zero-ranked. In that case, the reassociation
indices must be empty and the result shape may only consist of unit
dimensions.

For simplicity, this op may not be used to cast dynamicity of dimension
sizes and/or strides. I.e., if and only if a source dimension is dynamic,
there must be a dynamic result dimension in the corresponding reassociation
group. Same for strides.

The representation for the output shape supports a partially-static
specification via attributes specified through the `static_output_shape`
argument.  A special sentinel value `ShapedType::kDynamic` encodes that the
corresponding entry has a dynamic value.  There must be exactly as many SSA
inputs in `output_shape` as there are `ShapedType::kDynamic` entries in
`static_output_shape`.

Note: This op currently assumes that the inner strides are of the
source/result layout map are the faster-varying ones.
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
        "memref.expand_shape",
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
`extract_aligned_pointer_as_index`

Extracts the underlying aligned pointer as an index.

This operation is useful for lowering to lower-level dialects while still
avoiding the need to define a pointer type in higher-level dialects such as
the memref dialect.

This operation is intended solely as step during lowering, it has no side
effects. A reverse operation that creates a memref from an index interpreted
as a pointer is explicitly discouraged.

# Example

```
  %0 = memref.extract_aligned_pointer_as_index %arg : memref<4x4xf32> -> index
  %1 = arith.index_cast %0 : index to i64
  %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
  call @foo(%2) : (!llvm.ptr) ->()
```
"""
function extract_aligned_pointer_as_index(
    source::Value; aligned_pointer=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(aligned_pointer) && push!(op_ty_results, aligned_pointer)

    return create_operation(
        "memref.extract_aligned_pointer_as_index",
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
`extract_strided_metadata`

Extracts a base buffer, offset and strides. This op allows additional layers
of transformations and foldings to be added as lowering progresses from
higher-level dialect to lower-level dialects such as the LLVM dialect.

The op requires a strided memref source operand. If the source operand is not
a strided memref, then verification fails.

This operation is also useful for completeness to the existing memref.dim op.
While accessing strides, offsets and the base pointer independently is not
available, this is useful for composing with its natural complement op:
`memref.reinterpret_cast`.

Intended Use Cases:

The main use case is to expose the logic for manipulate memref metadata at a
higher level than the LLVM dialect.
This makes lowering more progressive and brings the following benefits:
  - not all users of MLIR want to lower to LLVM and the information to e.g.
    lower to library calls---like libxsmm---or to SPIR-V was not available.
  - foldings and canonicalizations can happen at a higher level in MLIR:
    before this op existed, lowering to LLVM would create large amounts of
    LLVMIR. Even when LLVM does a good job at folding the low-level IR from
    a performance perspective, it is unnecessarily opaque and inefficient to
    send unkempt IR to LLVM.

# Example

```mlir
  %base, %offset, %sizes:2, %strides:2 =
    memref.extract_strided_metadata %memref :
      memref<10x?xf32>, index, index, index, index, index

  // After folding, the type of %m2 can be memref<10x?xf32> and further
  // folded to %memref.
  %m2 = memref.reinterpret_cast %base to
      offset: [%offset],
      sizes: [%sizes#0, %sizes#1],
      strides: [%strides#0, %strides#1]
    : memref<f32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
```
"""
function extract_strided_metadata(
    source::Value;
    base_buffer=nothing::Union{Nothing,IR.Type},
    offset=nothing::Union{Nothing,IR.Type},
    sizes=nothing::Union{Nothing,Vector{IR.Type}},
    strides=nothing::Union{Nothing,Vector{IR.Type}},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(base_buffer) && push!(op_ty_results, base_buffer)
    !isnothing(offset) && push!(op_ty_results, offset)
    !isnothing(sizes) && push!(op_ty_results, sizes...)
    !isnothing(strides) && push!(op_ty_results, strides...)

    return create_operation(
        "memref.extract_strided_metadata",
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
`get_global`

The `memref.get_global` operation retrieves the memref pointing to a
named global variable. If the global variable is marked constant, writing
to the result memref (such as through a `memref.store` operation) is
undefined.

# Example

```mlir
%x = memref.get_global @foo : memref<2xf32>
```
"""
function get_global(; result::IR.Type, name, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name),]

    return create_operation(
        "memref.get_global",
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
`global_`

The `memref.global` operation declares or defines a named global memref
variable. The backing memory for the variable is allocated statically and is
described by the type of the variable (which should be a statically shaped
memref type). The operation is a declaration if no `initial_value` is
specified, else it is a definition. The `initial_value` can either be a unit
attribute to represent a definition of an uninitialized global variable, or
an elements attribute to represent the definition of a global variable with
an initial value. The global variable can also be marked constant using the
`constant` unit attribute. Writing to such constant global variables is
undefined.

The global variable can be accessed by using the `memref.get_global` to
retrieve the memref for the global variable. Note that the memref
for such global variable itself is immutable (i.e., memref.get_global for a
given global variable will always return the same memref descriptor).

# Example

```mlir
// Private variable with an initial value.
memref.global \"private\" @x : memref<2xf32> = dense<0.0,2.0>

// Private variable with an initial value and an alignment (power of 2).
memref.global \"private\" @x : memref<2xf32> = dense<0.0,2.0> {alignment = 64}

// Declaration of an external variable.
memref.global \"private\" @y : memref<4xi32>

// Uninitialized externally visible variable.
memref.global @z : memref<3xf16> = uninitialized

// Externally visible constant variable.
memref.global constant @c : memref<2xi32> = dense<1, 4>
```
"""
function global_(;
    sym_name,
    sym_visibility=nothing,
    type,
    initial_value=nothing,
    constant=nothing,
    alignment=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("type", type)
    ]
    !isnothing(sym_visibility) &&
        push!(attributes, namedattribute("sym_visibility", sym_visibility))
    !isnothing(initial_value) &&
        push!(attributes, namedattribute("initial_value", initial_value))
    !isnothing(constant) && push!(attributes, namedattribute("constant", constant))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))

    return create_operation(
        "memref.global",
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
`memory_space_cast`

This operation casts memref values between memory spaces.
The input and result will be memrefs of the same types and shape that alias
the same underlying memory, though, for some casts on some targets,
the underlying values of the pointer stored in the memref may be affected
by the cast.

The input and result must have the same shape, element type, rank, and layout.

If the source and target address spaces are the same, this operation is a noop.

# Example

```mlir
// Cast a GPU private memory attribution into a generic pointer
%2 = memref.memory_space_cast %1 : memref<?xf32, 5> to memref<?xf32>
// Cast a generic pointer to workgroup-local memory
%4 = memref.memory_space_cast %3 : memref<5x4xi32> to memref<5x34xi32, 3>
// Cast between two non-default memory spaces
%6 = memref.memory_space_cast %5
  : memref<*xmemref<?xf32>, 5> to memref<*xmemref<?xf32>, 3>
```
"""
function memory_space_cast(source::Value; dest::IR.Type, location=Location())
    op_ty_results = IR.Type[dest,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.memory_space_cast",
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
`prefetch`

The \"prefetch\" op prefetches data from a memref location described with
subscript indices similar to memref.load, and with three attributes: a
read/write specifier, a locality hint, and a cache type specifier as shown
below:

```mlir
memref.prefetch %0[%i, %j], read, locality<3>, data : memref<400x400xi32>
```

The read/write specifier is either \'read\' or \'write\', the locality hint
ranges from locality<0> (no locality) to locality<3> (extremely local keep
in cache). The cache type specifier is either \'data\' or \'instr\'
and specifies whether the prefetch is performed on data cache or on
instruction cache.
"""
function prefetch(
    memref::Value,
    indices::Vector{Value};
    isWrite,
    localityHint,
    isDataCache,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("isWrite", isWrite),
        namedattribute("localityHint", localityHint),
        namedattribute("isDataCache", isDataCache),
    ]

    return create_operation(
        "memref.prefetch",
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

The `memref.rank` operation takes a memref operand and returns its rank.

# Example

```mlir
%0 = memref.rank %arg0 : memref<*xf32>
%1 = memref.rank %arg1 : memref<?x?xf32>
```
"""
function rank(memref::Value; result_0=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[memref,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "memref.rank",
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
`realloc`

The `realloc` operation changes the size of a memory region. The memory
region is specified by a 1D source memref and the size of the new memory
region is specified by a 1D result memref type and an optional dynamic Value
of `Index` type. The source and the result memref must be in the same memory
space and have the same element type.

The operation may move the memory region to a new location. In this case,
the content of the memory block is preserved up to the lesser of the new
and old sizes. If the new size if larger, the value of the extended memory
is undefined. This is consistent with the ISO C realloc.

The operation returns an SSA value for the memref.

# Example

```mlir
%0 = memref.realloc %src : memref<64xf32> to memref<124xf32>
```

The source memref may have a dynamic shape, in which case, the compiler will
generate code to extract its size from the runtime data structure for the
memref.

```mlir
%1 = memref.realloc %src : memref<?xf32> to memref<124xf32>
```

If the result memref has a dynamic shape, a result dimension operand is
needed to spefify its dynamic dimension. In the example below, the ssa value
\'%d\' specifies the unknown dimension of the result memref.

```mlir
%2 = memref.realloc %src(%d) : memref<?xf32> to memref<?xf32>
```

An optional `alignment` attribute may be specified to ensure that the
region of memory that will be indexed is aligned at the specified byte
boundary.  This is consistent with the fact that memref.alloc supports such
an optional alignment attribute. Note that in ISO C standard, neither alloc
nor realloc supports alignment, though there is aligned_alloc but not
aligned_realloc.

```mlir
%3 = memref.realloc %src {alignment = 8} : memref<64xf32> to memref<124xf32>
```

Referencing the memref through the old SSA value after realloc is undefined
behavior.

```mlir
%new = memref.realloc %old : memref<64xf32> to memref<124xf32>
%4 = memref.load %new[%index]   // ok
%5 = memref.load %old[%index]   // undefined behavior
```
"""
function realloc(
    source::Value,
    dynamicResultSize=nothing::Union{Nothing,Value};
    result_0::IR.Type,
    alignment=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result_0,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dynamicResultSize) && push!(operands, dynamicResultSize)
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))

    return create_operation(
        "memref.realloc",
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
`reinterpret_cast`

Modify offset, sizes and strides of an unranked/ranked memref.

# Example
```mlir
memref.reinterpret_cast %ranked to
  offset: [0],
  sizes: [%size0, 10],
  strides: [1, %stride1]
: memref<?x?xf32> to memref<?x10xf32, strided<[1, ?], offset: 0>>

memref.reinterpret_cast %unranked to
  offset: [%offset],
  sizes: [%size0, %size1],
  strides: [%stride0, %stride1]
: memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
```

This operation creates a new memref descriptor using the base of the
source and applying the input arguments to the other metadata.
In other words:
```mlir
%dst = memref.reinterpret_cast %src to
  offset: [%offset],
  sizes: [%sizes],
  strides: [%strides]
```
means that `%dst`\'s descriptor will be:
```mlir
%dst.base = %src.base
%dst.aligned = %src.aligned
%dst.offset = %offset
%dst.sizes = %sizes
%dst.strides = %strides
```
"""
function reinterpret_cast(
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
        "memref.reinterpret_cast",
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
`reshape`

The `reshape` operation converts a memref from one type to an
equivalent type with a provided shape. The data is never copied or
modified. The source and destination types are compatible if both have the
same element type, same number of elements, address space and identity
layout map. The following combinations are possible:

a. Source type is ranked or unranked. Shape argument has static size.
Result type is ranked.

```mlir
// Reshape statically-shaped memref.
%dst = memref.reshape %src(%shape)
         : (memref<4x1xf32>, memref<1xi32>) to memref<4xf32>
%dst0 = memref.reshape %src(%shape0)
         : (memref<4x1xf32>, memref<2xi32>) to memref<2x2xf32>
// Flatten unranked memref.
%dst = memref.reshape %src(%shape)
         : (memref<*xf32>, memref<1xi32>) to memref<?xf32>
```

b. Source type is ranked or unranked. Shape argument has dynamic size.
Result type is unranked.

```mlir
// Reshape dynamically-shaped 1D memref.
%dst = memref.reshape %src(%shape)
         : (memref<?xf32>, memref<?xi32>) to memref<*xf32>
// Reshape unranked memref.
%dst = memref.reshape %src(%shape)
         : (memref<*xf32>, memref<?xi32>) to memref<*xf32>
```
"""
function reshape(source::Value, shape::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source, shape]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.reshape",
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
`store`

Store a value to a memref location given by indices. The value stored should
have the same type as the elemental type of the memref. The number of
arguments provided within brackets need to match the rank of the memref.

In an affine context, the indices of a store are restricted to SSA values
bound to surrounding loop induction variables,
[symbols](Affine.md/#restrictions-on-dimensions-and-symbols), results of a
`constant` operation, or the result of an
[`affine.apply`](Affine.md/#affineapply-affineapplyop) operation that can in
turn take as arguments all of the aforementioned SSA values or the
recursively result of such an `affine.apply` operation.

# Example

```mlir
memref.store %100, %A[%1, 1023] : memref<4x?xf32, #layout, memspace0>
```

**Context:** The `load` and `store` operations are specifically crafted to
fully resolve a reference to an element of a memref, and (in polyhedral
`affine.if` and `affine.for` operations) the compiler can follow use-def
chains (e.g. through [`affine.apply`](Affine.md/#affineapply-affineapplyop)
operations) to precisely analyze references at compile-time using polyhedral
techniques. This is possible because of the
[restrictions on dimensions and symbols](Affine.md/#restrictions-on-dimensions-and-symbols)
in these contexts.
"""
function store(
    value::Value,
    memref::Value,
    indices::Vector{Value};
    nontemporal=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))

    return create_operation(
        "memref.store",
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
`transpose`

The `transpose` op produces a strided memref whose sizes and strides
are a permutation of the original `in` memref. This is purely a metadata
transformation.

# Example

```mlir
%1 = memref.transpose %0 (i, j) -> (j, i) : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>>
```
"""
function transpose(in::Value; result_0::IR.Type, permutation, location=Location())
    op_ty_results = IR.Type[result_0,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("permutation", permutation),]

    return create_operation(
        "memref.transpose",
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
`view`

The \"view\" operation extracts an N-D contiguous memref with empty layout map
with arbitrary element type from a 1-D contiguous memref with empty layout
map of i8 element  type. The ViewOp supports the following arguments:

* A single dynamic byte-shift operand must be specified which represents a
  a shift of the base 1-D memref pointer from which to create the resulting
  contiguous memref view with identity layout.
* A dynamic size operand that must be specified for each dynamic dimension
  in the resulting view memref type.

The \"view\" operation gives a structured indexing form to a flat 1-D buffer.
Unlike \"subview\" it can perform a type change. The type change behavior
requires the op to have special semantics because, e.g. a byte shift of 3
cannot be represented as an offset on f64.
For now, a \"view\" op:

1. Only takes a contiguous source memref with 0 offset and empty layout.
2. Must specify a byte_shift operand (in the future, a special integer
   attribute may be added to support the folded case).
3. Returns a contiguous memref with 0 offset and empty layout.

# Example

```mlir
// Allocate a flat 1D/i8 memref.
%0 = memref.alloc() : memref<2048xi8>

// ViewOp with dynamic offset and static sizes.
%1 = memref.view %0[%offset_1024][] : memref<2048xi8> to memref<64x4xf32>

// ViewOp with dynamic offset and two dynamic size.
%2 = memref.view %0[%offset_1024][%size0, %size1] :
  memref<2048xi8> to memref<?x4x?xf32>
```
"""
function view(
    source::Value,
    byte_shift::Value,
    sizes::Vector{Value};
    result_0::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[result_0,]
    operands = Value[source, byte_shift, sizes...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "memref.view",
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
`subview`

The \"subview\" operation converts a memref type to another memref type
which represents a reduced-size view of the original memref as specified by
the operation\'s offsets, sizes and strides arguments.

The SubView operation supports the following arguments:

* source: the \"base\" memref on which to create a \"view\" memref.
* offsets: memref-rank number of offsets into the \"base\" memref at which to
           create the \"view\" memref.
* sizes: memref-rank number of sizes which specify the sizes of the result
         \"view\" memref type.
* strides: memref-rank number of strides that compose multiplicatively with
           the base memref strides in each dimension.

The representation based on offsets, sizes and strides support a
partially-static specification via attributes specified through the
`static_offsets`, `static_sizes` and `static_strides` arguments. A special
sentinel value ShapedType::kDynamic encodes that the corresponding entry has
a dynamic value.

A subview operation may additionally reduce the rank of the resulting view
by removing dimensions that are statically known to be of size 1.

Example 1:

```mlir
%0 = memref.alloc() : memref<64x4xf32, affine_map<(d0, d1) -> (d0 * 4 + d1)>>

// Create a sub-view of \"base\" memref \'%0\' with offset arguments \'%c0\',
// dynamic sizes for each dimension, and stride arguments \'%c1\'.
%1 = memref.subview %0[%c0, %c0][%size0, %size1][%c1, %c1]
  : memref<64x4xf32, affine_map<(d0, d1) -> (d0 * 4 + d1)>> to
    memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + d1 + s0)>>
```

Example 2:

```mlir
%0 = memref.alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>

// Create a sub-view of \"base\" memref \'%0\' with dynamic offsets, sizes,
// and strides.
// Note that dynamic offsets are represented by the linearized dynamic
// offset symbol \'s0\' in the subview memref layout map, and that the
// dynamic strides operands, after being applied to the base memref
// strides in each dimension, are represented in the view memref layout
// map as symbols \'s1\', \'s2\' and \'s3\'.
%1 = memref.subview %0[%i, %j, %k][%size0, %size1, %size2][%x, %y, %z]
  : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>> to
    memref<?x?x?xf32,
      affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + d1 * s2 + d2 * s3 + s0)>>
```

Example 3:

```mlir
%0 = memref.alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>

// Subview with constant offsets, sizes and strides.
%1 = memref.subview %0[0, 2, 0][4, 4, 4][1, 1, 1]
  : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>> to
    memref<4x4x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 8)>>
```

Example 4:

```mlir
%0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>

// Subview with constant size, but dynamic offsets and
// strides. The resulting memref has a static shape, but if the
// base memref has an affine map to describe the layout, the result
// memref also uses an affine map to describe the layout. The
// strides of the result memref is computed as follows:
//
// Let #map1 represents the layout of the base memref, and #map2
// represents the layout of the result memref. A #mapsubview can be
// constructed to map an index from the result memref to the base
// memref (note that the description below uses more convenient
// naming for symbols, while in affine maps, symbols are
// represented as unsigned numbers that identify that symbol in the
// given affine map.
//
// #mapsubview = (d0, d1)[o0, o1, t0, t1] -> (d0 * t0 + o0, d1 * t1 + o1)
//
// where, o0, o1, ... are offsets, and t0, t1, ... are strides. Then,
//
// #map2 = #map1.compose(#mapsubview)
//
// If the layout map is represented as
//
// #map1 = (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)
//
// then,
//
// #map2 = (d0, d1)[s0, s1, s2, o0, o1, t0, t1] ->
//              (d0 * s1 * t0 + d1 * s2 * t1 + o0 * s1 + o1 * s2 + s0)
//
// Representing this canonically
//
// #map2 = (d0, d1)[r0, r1, r2] -> (d0 * r1 + d1 * r2 + r0)
//
// where, r0 = o0 * s1 + o1 * s2 + s0, r1 = s1 * t0, r2 = s2 * t1.
%1 = memref.subview %0[%i, %j][4, 4][%x, %y] :
  : memref<?x?xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)>> to
    memref<4x4xf32, affine_map<(d0, d1)[r0, r1, r2] -> (d0 * r1 + d1 * r2 + r0)>>

// Note that the subview op does not guarantee that the result
// memref is \"inbounds\" w.r.t to base memref. It is upto the client
// to ensure that the subview is accessed in a manner that is
// in-bounds.
```

Example 5:

```mlir
// Rank-reducing subview.
%1 = memref.subview %0[0, 0, 0][1, 16, 4][1, 1, 1] :
  memref<8x16x4xf32> to memref<16x4xf32>

// Original layout:
// (d0, d1, d2) -> (64 * d0 + 16 * d1 + d2)
// Subviewed layout:
// (d0, d1, d2) -> (64 * (d0 + 3) + 4 * (d1 + 4) + d2 + 2) = (64 * d0 + 4 * d1 + d2 + 210)
// After rank reducing:
// (d0, d1) -> (4 * d0 + d1 + 210)
%3 = memref.subview %2[3, 4, 2][1, 6, 3][1, 1, 1] :
  memref<8x16x4xf32> to memref<6x3xf32, strided<[4, 1], offset: 210>>
```
"""
function subview(
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
        "memref.subview",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # memref
