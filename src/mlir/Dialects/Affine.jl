module affine
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
`apply`

The `affine.apply` operation applies an [affine mapping](#affine-maps)
to a list of SSA values, yielding a single SSA value. The number of
dimension and symbol operands to `affine.apply` must be equal to the
respective number of dimensional and symbolic inputs to the affine mapping;
the affine mapping has to be one-dimensional, and so the `affine.apply`
operation always returns one value. The input operands and result must all
have ‘index’ type.

An operand that is a valid dimension as per the [rules on valid affine
dimensions and symbols](#restrictions-on-dimensions-and-symbols)
cannot be used as a symbolic operand.

# Example

```mlir
#map = affine_map<(d0, d1) -> (d0 floordiv 8 + d1 floordiv 128)>
...
%1 = affine.apply #map (%s, %t)

// Inline example.
%2 = affine.apply affine_map<(i)[s0] -> (i + s0)> (%42)[%n]
```
"""
function apply(
    mapOperands::Vector{Value};
    result_0=nothing::Union{Nothing,IR.Type},
    map,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[mapOperands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("map", map),]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "affine.apply",
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
`delinearize_index`

The `affine.delinearize_index` operation takes a single index value and
calculates the multi-index according to the given basis.

# Example

```
%indices:3 = affine.delinearize_index %linear_index into (%c16, %c224, %c224) : index, index, index
```

In the above example, `%indices:3` conceptually holds the following:

```
#map0 = affine_map<()[s0] -> (s0 floordiv 50176)>
#map1 = affine_map<()[s0] -> ((s0 mod 50176) floordiv 224)>
#map2 = affine_map<()[s0] -> (s0 mod 224)>
%indices_0 = affine.apply #map0()[%linear_index]
%indices_1 = affine.apply #map1()[%linear_index]
%indices_2 = affine.apply #map2()[%linear_index]
```

In other words, `%0:3 = affine.delinearize_index %x into (B, C)` produces
`%0 = {%x / (B * C), (%x mod (B * C)) / C, %x mod C}`.

The basis may either contain `N` or `N-1` elements, where `N` is the number of results.
If there are N basis elements, the first one will not be used during computations,
but may be used during analysis and canonicalization to eliminate terms from
the `affine.delinearize_index` or to enable conclusions about the total size of
`%linear_index`.

If the basis is fully provided, the delinearize_index operation is said to \"have
an outer bound\". The builders assume that an `affine.delinearize_index` has
an outer bound by default, as this is how the operation was initially defined.

That is, the example above could also have been written
```mlir
%0:3 = affine.delinearize_index %linear_index into (244, 244) : index, index
```

Note that, for symmetry with `getPaddedBasis()`, if `hasOuterBound` is `true`
when one of the `OpFoldResult` builders is called but the first element of the
basis is `nullptr`, that first element is ignored and the builder proceeds as if
there was no outer bound.

Due to the constraints of affine maps, all the basis elements must
be strictly positive. A dynamic basis element being 0 or negative causes
undefined behavior.

As with other affine operations, lowerings of delinearize_index may assume
that the underlying computations do not overflow the index type in a signed sense
- that is, the product of all basis elements is positive as an `index` as well.
"""
function delinearize_index(
    linear_index::Value,
    dynamic_basis::Vector{Value};
    multi_index::Vector{IR.Type},
    static_basis,
    location=Location(),
)
    op_ty_results = IR.Type[multi_index...,]
    operands = Value[linear_index, dynamic_basis...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("static_basis", static_basis),]

    return create_operation(
        "affine.delinearize_index",
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
`for_`

# Syntax

```
operation   ::= `affine.for` ssa-id `=` lower-bound `to` upper-bound
                (`step` integer-literal)? `{` op* `}`

lower-bound ::= `max`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
upper-bound ::= `min`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
shorthand-bound ::= ssa-id | `-`? integer-literal
```

The `affine.for` operation represents an affine loop nest. It has one region
containing its body. This region must contain one block that terminates with
[`affine.yield`](#affineyield-mliraffineyieldop). *Note:* when
`affine.for` is printed in custom format, the terminator is omitted. The
block has one argument of [`index`](Builtin.md/#indextype) type that
represents the induction variable of the loop.

The `affine.for` operation executes its body a number of times iterating
from a lower bound to an upper bound by a stride. The stride, represented by
`step`, is a positive constant integer which defaults to \"1\" if not present.
The lower and upper bounds specify a half-open range: the range includes the
lower bound but does not include the upper bound.

The lower and upper bounds of a `affine.for` operation are represented as an
application of an affine mapping to a list of SSA values passed to the map.
The [same restrictions](#restrictions-on-dimensions-and-symbols) hold for
these SSA values as for all bindings of SSA values to dimensions and
symbols.

The affine mappings for the bounds may return multiple results, in which
case the `max`/`min` keywords are required (for the lower/upper bound
respectively), and the bound is the maximum/minimum of the returned values.
There is no semantic ambiguity, but MLIR syntax requires the use of these
keywords to make things more obvious to human readers.

Many upper and lower bounds are simple, so MLIR accepts two custom form
syntaxes: the form that accepts a single \'ssa-id\' (e.g. `%N`) is shorthand
for applying that SSA value to a function that maps a single symbol to
itself, e.g., `()[s]->(s)()[%N]`. The integer literal form (e.g. `-42`) is
shorthand for a nullary mapping function that returns the constant value
(e.g. `()->(-42)()`).

Example showing reverse iteration of the inner loop:

```mlir
#map57 = affine_map<(d0)[s0] -> (s0 - d0 - 1)>

func.func @simple_example(%A: memref<?x?xf32>, %B: memref<?x?xf32>) {
  %N = dim %A, 0 : memref<?x?xf32>
  affine.for %i = 0 to %N step 1 {
    affine.for %j = 0 to %N {   // implicitly steps by 1
      %0 = affine.apply #map57(%j)[%N]
      %tmp = call @F1(%A, %i, %0) : (memref<?x?xf32>, index, index)->(f32)
      call @F2(%tmp, %B, %i, %0) : (f32, memref<?x?xf32>, index, index)->()
    }
  }
  return
}
```
`affine.for` can also operate on loop-carried variables (`iter_args`) and
return the final values after loop termination. The initial values of the
variables are passed as additional SSA operands to the `affine.for`
following the operands for the loop\'s lower and upper bounds. The
operation\'s region has equivalent arguments for each variable representing
the value of the variable at the current iteration.

The region must terminate with an `affine.yield` that passes all the current
iteration variables to the next iteration, or to the `affine.for`\'s results
if at the last iteration. For `affine.for`\'s that execute zero iterations, the
initial values of the loop-carried variables (corresponding to the SSA
operands) will be the op\'s results.

For example, to sum-reduce a memref:

 ```mlir
func.func @reduce(%buffer: memref<1024xf32>) -> (f32) {
  // Initial sum set to 0.
  %sum_0 = arith.constant 0.0 : f32
  // iter_args binds initial values to the loop\'s region arguments.
  %sum = affine.for %i = 0 to 10 step 2
      iter_args(%sum_iter = %sum_0) -> (f32) {
    %t = affine.load %buffer[%i] : memref<1024xf32>
    %sum_next = arith.addf %sum_iter, %t : f32
    // Yield current iteration sum to next iteration %sum_iter or to %sum
    // if final iteration.
    affine.yield %sum_next : f32
  }
  return %sum : f32
}
```

```mlir
%res:2 = affine.for %i = 0 to 128 iter_args(%arg0 = %init0, %arg1 = %init1)
           -> (index, index) {
  %y0 = arith.addi %arg0, %c1 : index
  %y1 = arith.addi %arg1, %c2 : index
  affine.yield %y0, %y1 : index, index
}
```
If the `affine.for` defines any values, a yield terminator must be
explicitly present. The number and types of the \"affine.for\" results must
match the initial values in the `iter_args` binding and the yield operands.
"""
function for_(
    lowerBoundOperands::Vector{Value},
    upperBoundOperands::Vector{Value},
    inits::Vector{Value};
    results::Vector{IR.Type},
    lowerBoundMap,
    upperBoundMap,
    step,
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[lowerBoundOperands..., upperBoundOperands..., inits...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("lowerBoundMap", lowerBoundMap),
        namedattribute("upperBoundMap", upperBoundMap),
        namedattribute("step", step),
    ]
    push!(
        attributes,
        operandsegmentsizes([
            length(lowerBoundOperands), length(upperBoundOperands), length(inits)
        ]),
    )

    return create_operation(
        "affine.for",
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
`if_`

# Syntax

```
operation  ::= `affine.if` if-op-cond `{` op* `}` (`else` `{` op* `}`)?
if-op-cond ::= integer-set-attr dim-and-symbol-use-list
```

The `affine.if` operation restricts execution to a subset of the loop
iteration space defined by an integer set (a conjunction of affine
constraints). A single `affine.if` may end with an optional `else` clause.

The condition of the `affine.if` is represented by an
[integer set](#integer-sets) (a conjunction of affine constraints),
and the SSA values bound to the dimensions and symbols in the integer set.
The [same restrictions](#restrictions-on-dimensions-and-symbols) hold for
these SSA values as for all bindings of SSA values to dimensions and
symbols.

The `affine.if` operation contains two regions for the \"then\" and \"else\"
clauses.  `affine.if` may return results that are defined in its regions.
The values defined are determined by which execution path is taken.  Each
region of the `affine.if` must contain a single block with no arguments,
and be terminated by `affine.yield`.  If `affine.if` defines no values,
the `affine.yield` can be left out, and will be inserted implicitly.
Otherwise, it must be explicit.  If no values are defined, the else block
may be empty (i.e. contain no blocks).

# Example

```mlir
#set = affine_set<(d0, d1)[s0]: (d0 - 10 >= 0, s0 - d0 - 9 >= 0,
                                 d1 - 10 >= 0, s0 - d1 - 9 >= 0)>
func.func @reduced_domain_example(%A, %X, %N) : (memref<10xi32>, i32, i32) {
  affine.for %i = 0 to %N {
     affine.for %j = 0 to %N {
       %0 = affine.apply #map42(%j)
       %tmp = call @S1(%X, %i, %0)
       affine.if #set(%i, %j)[%N] {
          %1 = affine.apply #map43(%i, %j)
          call @S2(%tmp, %A, %i, %1)
       }
    }
  }
  return
}
```

Example with an explicit yield (initialization with edge padding):

```mlir
#interior = affine_set<(i, j) : (i - 1 >= 0, j - 1 >= 0,  10 - i >= 0, 10 - j >= 0)> (%i, %j)
func.func @pad_edges(%I : memref<10x10xf32>) -> (memref<12x12xf32) {
  %O = alloc memref<12x12xf32>
  affine.parallel (%i, %j) = (0, 0) to (12, 12) {
    %1 = affine.if #interior (%i, %j) {
      %2 = load %I[%i - 1, %j - 1] : memref<10x10xf32>
      affine.yield %2
    } else {
      %2 = arith.constant 0.0 : f32
      affine.yield %2 : f32
    }
    affine.store %1, %O[%i, %j] : memref<12x12xf32>
  }
  return %O
}
```
"""
function if_(
    operand_0::Vector{Value};
    results::Vector{IR.Type},
    condition,
    thenRegion::Region,
    elseRegion::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[operand_0...,]
    owned_regions = Region[thenRegion, elseRegion]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("condition", condition),]

    return create_operation(
        "affine.if",
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
`linearize_index`

The `affine.linearize_index` operation takes a sequence of index values and a
basis of the same length and linearizes the indices using that basis.

That is, for indices `%idx_0` to `%idx_{N-1}` and basis elements `b_0`
(or `b_1`) up to `b_{N-1}` it computes

```
sum(i = 0 to N-1) %idx_i * product(j = i + 1 to N-1) B_j
```

In other words, `%0 = affine.linearize_index [%z, %y, %x] by (Z, Y, X)`
gives `%0 = %x + %y * X + %z * X * Y`, or `%0 = %x + X * (%y + Y * (%z))`.

The basis may either have `N` or `N-1` elements, where `N` is the number of
inputs to linearize_index. If `N` inputs are provided, the first one is not used
in computation, but may be used during analysis or canonicalization as a bound
on `%idx_0`.

If all `N` basis elements are provided, the linearize_index operation is said to
\"have an outer bound\".

As a convenience, and for symmetry with `getPaddedBasis()`, if the first
element of a set of `OpFoldResult`s passed to the builders of this operation is
`nullptr`, that element is ignored.

If the `disjoint` property is present, this is an optimization hint that,
for all `i`, `0 <= %idx_i < B_i` - that is, no index affects any other index,
except that `%idx_0` may be negative to make the index as a whole negative.
In addition, `disjoint` is an assertion that all bases elements are non-negative.

Note that the outputs of `affine.delinearize_index` are, by definition, `disjoint`.

As with other affine ops, undefined behavior occurs if the linearization
computation overflows in the signed sense.

# Example

```mlir
%linear_index = affine.linearize_index [%index_0, %index_1, %index_2] by (2, 3, 5) : index
// Same effect
%linear_index = affine.linearize_index [%index_0, %index_1, %index_2] by (3, 5) : index
```

In the above example, `%linear_index` conceptually holds the following:

```mlir
#map = affine_map<()[s0, s1, s2] -> (s0 * 15 + s1 * 5 + s2)>
%linear_index = affine.apply #map()[%index_0, %index_1, %index_2]
```
"""
function linearize_index(
    multi_index::Vector{Value},
    dynamic_basis::Vector{Value};
    linear_index=nothing::Union{Nothing,IR.Type},
    static_basis,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[multi_index..., dynamic_basis...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("static_basis", static_basis),]
    push!(attributes, operandsegmentsizes([length(multi_index), length(dynamic_basis)]))
    !isnothing(linear_index) && push!(op_ty_results, linear_index)

    return create_operation(
        "affine.linearize_index",
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
`load`

# Syntax

```
operation ::= ssa-id `=` `affine.load` ssa-use `[` multi-dim-affine-map-of-ssa-ids `]` `:` memref-type
```

The `affine.load` op reads an element from a memref, where the index
for each memref dimension is an affine expression of loop induction
variables and symbols. The output of `affine.load` is a new value with the
same type as the elements of the memref. An affine expression of loop IVs
and symbols must be specified for each dimension of the memref. The keyword
`symbol` can be used to indicate SSA identifiers which are symbolic.

Example 1:

```mlir
%1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
```

Example 2: Uses `symbol` keyword for symbols `%n` and `%m`.

```mlir
%1 = affine.load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>
```
"""
function load(
    memref::Value, indices::Vector{Value}; result::IR.Type, map, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("map", map),]

    return create_operation(
        "affine.load",
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
`max`

The `affine.max` operation computes the maximum value result from a multi-result
affine map.

# Example

```mlir
%0 = affine.max (d0) -> (1000, d0 + 512) (%i0) : index
```
"""
function max(
    operands::Vector{Value};
    result_0=nothing::Union{Nothing,IR.Type},
    map,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("map", map),]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "affine.max",
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
`min`

# Syntax

```
operation ::= ssa-id `=` `affine.min` affine-map-attribute dim-and-symbol-use-list
```

The `affine.min` operation applies an [affine mapping](#affine-expressions)
to a list of SSA values, and returns the minimum value of all result
expressions. The number of dimension and symbol arguments to `affine.min`
must be equal to the respective number of dimensional and symbolic inputs to
the affine mapping; the `affine.min` operation always returns one value. The
input operands and result must all have \'index\' type.

# Example

```mlir
%0 = affine.min affine_map<(d0)[s0] -> (1000, d0 + 512, s0)> (%arg0)[%arg1]
```
"""
function min(
    operands::Vector{Value};
    result_0=nothing::Union{Nothing,IR.Type},
    map,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("map", map),]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "affine.min",
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
`parallel`

The `affine.parallel` operation represents a hyper-rectangular affine
parallel band, defining zero or more SSA values for its induction variables.
It has one region capturing the parallel band body. The induction variables
are represented as arguments of this region. These SSA values always have
type index, which is the size of the machine word. The strides, represented
by steps, are positive constant integers which defaults to \"1\" if not
present. The lower and upper bounds specify a half-open range: the range
includes the lower bound but does not include the upper bound. The body
region must contain exactly one block that terminates with `affine.yield`.

The lower and upper bounds of a parallel operation are represented as an
application of an affine mapping to a list of SSA values passed to the map.
The same restrictions hold for these SSA values as for all bindings of SSA
values to dimensions and symbols. The list of expressions in each map is
interpreted according to the respective bounds group attribute. If a single
expression belongs to the group, then the result of this expression is taken
as a lower(upper) bound of the corresponding loop induction variable. If
multiple expressions belong to the group, then the lower(upper) bound is the
max(min) of these values obtained from these expressions. The loop band has
as many loops as elements in the group bounds attributes.

Each value yielded by `affine.yield` will be accumulated/reduced via one of
the reduction methods defined in the AtomicRMWKind enum.  The order of
reduction is unspecified, and lowering may produce any valid ordering.
Loops with a 0 trip count will produce as a result the identity value
associated with each reduction (i.e. 0.0 for addf, 1.0 for mulf).  Assign
reductions for loops with a trip count != 1 produces undefined results.

Note: Calling `AffineParallelOp::build` will create the required region and
block, and insert the required terminator if it is trivial (i.e. no values
are yielded).  Parsing will also create the required region, block, and
terminator, even when they are missing from the textual representation.

Example (3x3 valid convolution):

```mlir
func.func @conv_2d(%D : memref<100x100xf32>, %K : memref<3x3xf32>) -> (memref<98x98xf32>) {
  %O = memref.alloc() : memref<98x98xf32>
  affine.parallel (%x, %y) = (0, 0) to (98, 98) {
    %0 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce (\"addf\") -> f32 {
      %1 = affine.load %D[%x + %kx, %y + %ky] : memref<100x100xf32>
      %2 = affine.load %K[%kx, %ky] : memref<3x3xf32>
      %3 = arith.mulf %1, %2 : f32
      affine.yield %3 : f32
    }
    affine.store %0, %O[%x, %y] : memref<98x98xf32>
  }
  return %O : memref<98x98xf32>
}
```

Example (tiling by potentially imperfectly dividing sizes):

```mlir
affine.parallel (%ii, %jj) = (0, 0) to (%N, %M) step (32, 32) {
  affine.parallel (%i, %j) = (%ii, %jj)
                          to (min(%ii + 32, %N), min(%jj + 32, %M)) {
    call @f(%i, %j) : (index, index) -> ()
  }
}
```
"""
function parallel(
    mapOperands::Vector{Value};
    results::Vector{IR.Type},
    reductions,
    lowerBoundsMap,
    lowerBoundsGroups,
    upperBoundsMap,
    upperBoundsGroups,
    steps,
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[mapOperands...,]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("reductions", reductions),
        namedattribute("lowerBoundsMap", lowerBoundsMap),
        namedattribute("lowerBoundsGroups", lowerBoundsGroups),
        namedattribute("upperBoundsMap", upperBoundsMap),
        namedattribute("upperBoundsGroups", upperBoundsGroups),
        namedattribute("steps", steps),
    ]

    return create_operation(
        "affine.parallel",
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

The `affine.prefetch` op prefetches data from a memref location described
with an affine subscript similar to affine.load, and has three attributes:
a read/write specifier, a locality hint, and a cache type specifier as shown
below:

```mlir
affine.prefetch %0[%i, %j + 5], read, locality<3>, data : memref<400x400xi32>
```

The read/write specifier is either \'read\' or \'write\', the locality hint
specifier ranges from locality<0> (no locality) to locality<3> (extremely
local keep in cache). The cache type specifier is either \'data\' or \'instr\'
and specifies whether the prefetch is performed on data cache or on
instruction cache.
"""
function prefetch(
    memref::Value,
    indices::Vector{Value};
    isWrite,
    localityHint,
    isDataCache,
    map,
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
        namedattribute("map", map),
    ]

    return create_operation(
        "affine.prefetch",
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

# Syntax

```
operation ::= `affine.store` ssa-use, ssa-use `[` multi-dim-affine-map-of-ssa-ids `]` `:` memref-type
```

The `affine.store` op writes an element to a memref, where the index
for each memref dimension is an affine expression of loop induction
variables and symbols. The `affine.store` op stores a new value which is the
same type as the elements of the memref. An affine expression of loop IVs
and symbols must be specified for each dimension of the memref. The keyword
`symbol` can be used to indicate SSA identifiers which are symbolic.

Example 1:

```mlir
affine.store %v0, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
```

Example 2: Uses `symbol` keyword for symbols `%n` and `%m`.

```mlir
affine.store %v0, %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>
```
"""
function store(
    value::Value, memref::Value, indices::Vector{Value}; map, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("map", map),]

    return create_operation(
        "affine.store",
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
`vector_load`

The `affine.vector_load` is the vector counterpart of
[affine.load](#affineload-mliraffineloadop). It reads a slice from a
[MemRef](Builtin.md/#memreftype), supplied as its first operand,
into a [vector](Builtin.md/#vectortype) of the same base elemental type.
The index for each memref dimension is an affine expression of loop induction
variables and symbols. These indices determine the start position of the read
within the memref. The shape of the return vector type determines the shape of
the slice read from the memref. This slice is contiguous along the respective
dimensions of the shape. Strided vector loads will be supported in the future.
An affine expression of loop IVs and symbols must be specified for each
dimension of the memref. The keyword `symbol` can be used to indicate SSA
identifiers which are symbolic.

Example 1: 8-wide f32 vector load.

```mlir
%1 = affine.vector_load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>, vector<8xf32>
```

Example 2: 4-wide f32 vector load. Uses `symbol` keyword for symbols `%n` and `%m`.

```mlir
%1 = affine.vector_load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>, vector<4xf32>
```

Example 3: 2-dim f32 vector load.

```mlir
%1 = affine.vector_load %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
```

TODOs:
* Add support for strided vector loads.
* Consider adding a permutation map to permute the slice that is read from memory
(see [vector.transfer_read](../Vector/#vectortransfer_read-mlirvectortransferreadop)).
"""
function vector_load(
    memref::Value, indices::Vector{Value}; result::IR.Type, map, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("map", map),]

    return create_operation(
        "affine.vector_load",
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
`vector_store`

The `affine.vector_store` is the vector counterpart of
[affine.store](#affinestore-mliraffinestoreop). It writes a
[vector](Builtin.md/#vectortype), supplied as its first operand,
into a slice within a [MemRef](Builtin.md/#memreftype) of the same base
elemental type, supplied as its second operand.
The index for each memref dimension is an affine expression of loop
induction variables and symbols. These indices determine the start position
of the write within the memref. The shape of the input vector determines the
shape of the slice written to the memref. This slice is contiguous along the
respective dimensions of the shape. Strided vector stores will be supported
in the future.
An affine expression of loop IVs and symbols must be specified for each
dimension of the memref. The keyword `symbol` can be used to indicate SSA
identifiers which are symbolic.

Example 1: 8-wide f32 vector store.

```mlir
affine.vector_store %v0, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>, vector<8xf32>
```

Example 2: 4-wide f32 vector store. Uses `symbol` keyword for symbols `%n` and `%m`.

```mlir
affine.vector_store %v0, %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>, vector<4xf32>
```

Example 3: 2-dim f32 vector store.

```mlir
affine.vector_store %v0, %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
```

TODOs:
* Add support for strided vector stores.
* Consider adding a permutation map to permute the slice that is written to memory
(see [vector.transfer_write](../Vector/#vectortransfer_write-mlirvectortransferwriteop)).
"""
function vector_store(
    value::Value, memref::Value, indices::Vector{Value}; map, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("map", map),]

    return create_operation(
        "affine.vector_store",
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

The `affine.yield` yields zero or more SSA values from an affine op region and
terminates the region. The semantics of how the values yielded are used
is defined by the parent operation.
If `affine.yield` has any operands, the operands must match the parent
operation\'s results.
If the parent operation defines no values, then the `affine.yield` may be
left out in the custom syntax and the builders will insert one implicitly.
Otherwise, it has to be present in the syntax to indicate which values are
yielded.
"""
function yield(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "affine.yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # affine
