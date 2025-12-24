module arith
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
`addf`

The `addf` operation takes two operands and returns one result, each of
these is required to be the same type. This type may be a floating point
scalar type, a vector whose element type is a floating point type, or a
floating point tensor.

# Example

```mlir
// Scalar addition.
%a = arith.addf %b, %c : f64

// SIMD vector addition, e.g. for Intel SSE.
%f = arith.addf %g, %h : vector<4xf32>

// Tensor addition.
%x = arith.addf %y, %z : tensor<4x?xbf16>
```

TODO: In the distant future, this will accept optional attributes for fast
math, contraction, rounding mode, and other controls.
"""
function addf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.addf",
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
`addi`

Performs N-bit addition on the operands. The operands are interpreted as
unsigned bitvectors. The result is represented by a bitvector containing the
mathematical value of the addition modulo 2^n, where `n` is the bitwidth.
Because `arith` integers use a two\'s complement representation, this operation
is applicable on both signed and unsigned integer operands.

The `addi` operation takes two operands and returns one result, each of
these is required to be the same type. This type may be an integer scalar type,
a vector whose element type is integer, or a tensor of integers.

This op supports `nuw`/`nsw` overflow flags which stands for
\"No Unsigned Wrap\" and \"No Signed Wrap\", respectively. If the `nuw` and/or
`nsw` flags are present, and an unsigned/signed overflow occurs
(respectively), the result is poison.

# Example

```mlir
// Scalar addition.
%a = arith.addi %b, %c : i64

// Scalar addition with overflow flags.
%a = arith.addi %b, %c overflow<nsw, nuw> : i64

// SIMD vector element-wise addition.
%f = arith.addi %g, %h : vector<4xi32>

// Tensor element-wise addition.
%x = arith.addi %y, %z : tensor<4x?xi8>
```
"""
function addi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    overflowFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(overflowFlags) &&
        push!(attributes, namedattribute("overflowFlags", overflowFlags))

    return create_operation(
        "arith.addi",
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
`addui_extended`

Performs (N+1)-bit addition on zero-extended operands. Returns two results:
the N-bit sum (same type as both operands), and the overflow bit
(boolean-like), where `1` indicates unsigned addition overflow, while `0`
indicates no overflow.

# Example

```mlir
// Scalar addition.
%sum, %overflow = arith.addui_extended %b, %c : i64, i1

// Vector element-wise addition.
%d:2 = arith.addui_extended %e, %f : vector<4xi32>, vector<4xi1>

// Tensor element-wise addition.
%x:2 = arith.addui_extended %y, %z : tensor<4x?xi8>, tensor<4x?xi1>
```
"""
function addui_extended(
    lhs::Value, rhs::Value; sum::IR.Type, overflow::IR.Type, location=Location()
)
    op_ty_results = IR.Type[sum, overflow]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "arith.addui_extended",
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
`andi`

The `andi` operation takes two operands and returns one result, each of
these is required to be the same type. This type may be an integer scalar
type, a vector whose element type is integer, or a tensor of integers. It
has no standard attributes.

# Example

```mlir
// Scalar integer bitwise and.
%a = arith.andi %b, %c : i64

// SIMD vector element-wise bitwise integer and.
%f = arith.andi %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer and.
%x = arith.andi %y, %z : tensor<4x?xi8>
```
"""
function andi(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.andi",
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
`bitcast`

Bitcast an integer or floating point value to an integer or floating point
value of equal bit width. When operating on vectors, casts elementwise.

Note that this implements a logical bitcast independent of target
endianness. This allows constant folding without target information and is
consitent with the bitcast constant folders in LLVM (see
https://github.com/llvm/llvm-project/blob/18c19414eb/llvm/lib/IR/ConstantFold.cpp#L168)
For targets where the source and target type have the same endianness (which
is the standard), this cast will also change no bits at runtime, but it may
still require an operation, for example if the machine has different
floating point and integer register files. For targets that have a different
endianness for the source and target types (e.g. float is big-endian and
integer is little-endian) a proper lowering would add operations to swap the
order of words in addition to the bitcast.
"""
function bitcast(in::Value; out::IR.Type, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "arith.bitcast",
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
`ceildivsi`

Signed integer division. Rounds towards positive infinity, i.e. `7 / -2 = -3`.

Divison by zero, or signed division overflow (minimum value divided by -1)
is undefined behavior. When applied to `vector` and `tensor` values, the
behavior is undefined if _any_ of its elements are divided by zero or has a
signed division overflow.

# Example

```mlir
// Scalar signed integer division.
%a = arith.ceildivsi %b, %c : i64
```
"""
function ceildivsi(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.ceildivsi",
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
`ceildivui`

Unsigned integer division. Rounds towards positive infinity. Treats the
leading bit as the most significant, i.e. for `i16` given two\'s complement
representation, `6 / -2 = 6 / (2^16 - 2) = 1`.

Division by zero is undefined behavior. When applied to `vector` and
`tensor` values, the behavior is undefined if _any_ elements are divided by
zero.

# Example

```mlir
// Scalar unsigned integer division.
%a = arith.ceildivui %b, %c : i64
```
"""
function ceildivui(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.ceildivui",
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
`cmpf`

The `cmpf` operation compares its two operands according to the float
comparison rules and the predicate specified by the respective attribute.
The predicate defines the type of comparison: (un)orderedness, (in)equality
and signed less/greater than (or equal to) as well as predicates that are
always true or false.  The operands must have the same type, and this type
must be a float type, or a vector or tensor thereof.  The result is an i1,
or a vector/tensor thereof having the same shape as the inputs. Unlike cmpi,
the operands are always treated as signed. The u prefix indicates
*unordered* comparison, not unsigned comparison, so \"une\" means unordered or
not equal. For the sake of readability by humans, custom assembly form for
the operation uses a string-typed attribute for the predicate.  The value of
this attribute corresponds to lower-cased name of the predicate constant,
e.g., \"one\" means \"ordered not equal\".  The string representation of the
attribute is merely a syntactic sugar and is converted to an integer
attribute by the parser.

# Example

```mlir
%r1 = arith.cmpf oeq, %0, %1 : f32
%r2 = arith.cmpf ult, %0, %1 : tensor<42x42xf64>
%r3 = \"arith.cmpf\"(%0, %1) {predicate: 0} : (f8, f8) -> i1
```
"""
function cmpf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    predicate,
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate),]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.cmpf",
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
`cmpi`

The `cmpi` operation is a generic comparison for integer-like types. Its two
arguments can be integers, vectors or tensors thereof as long as their types
match. The operation produces an i1 for the former case, a vector or a
tensor of i1 with the same shape as inputs in the other cases.

Its first argument is an attribute that defines which type of comparison is
performed. The following comparisons are supported:

-   equal (mnemonic: `\"eq\"`; integer value: `0`)
-   not equal (mnemonic: `\"ne\"`; integer value: `1`)
-   signed less than (mnemonic: `\"slt\"`; integer value: `2`)
-   signed less than or equal (mnemonic: `\"sle\"`; integer value: `3`)
-   signed greater than (mnemonic: `\"sgt\"`; integer value: `4`)
-   signed greater than or equal (mnemonic: `\"sge\"`; integer value: `5`)
-   unsigned less than (mnemonic: `\"ult\"`; integer value: `6`)
-   unsigned less than or equal (mnemonic: `\"ule\"`; integer value: `7`)
-   unsigned greater than (mnemonic: `\"ugt\"`; integer value: `8`)
-   unsigned greater than or equal (mnemonic: `\"uge\"`; integer value: `9`)

The result is `1` if the comparison is true and `0` otherwise. For vector or
tensor operands, the comparison is performed elementwise and the element of
the result indicates whether the comparison is true for the operand elements
with the same indices as those of the result.

Note: while the custom assembly form uses strings, the actual underlying
attribute has integer type (or rather enum class in C++ code) as seen from
the generic assembly form. String literals are used to improve readability
of the IR by humans.

This operation only applies to integer-like operands, but not floats. The
main reason being that comparison operations have diverging sets of
attributes: integers require sign specification while floats require various
floating point-related particularities, e.g., `-ffast-math` behavior,
IEEE754 compliance, etc
([rationale](../Rationale/Rationale.md#splitting-floating-point-vs-integer-operations)).
The type of comparison is specified as attribute to avoid introducing ten
similar operations, taking into account that they are often implemented
using the same operation downstream
([rationale](../Rationale/Rationale.md#specifying-comparison-kind-as-attribute)). The
separation between signed and unsigned order comparisons is necessary
because of integers being signless. The comparison operation must know how
to interpret values with the foremost bit being set: negatives in two\'s
complement or large positives
([rationale](../Rationale/Rationale.md#specifying-sign-in-integer-comparison-operations)).

# Example

```mlir
// Custom form of scalar \"signed less than\" comparison.
%x = arith.cmpi slt, %lhs, %rhs : i32

// Generic form of the same operation.
%x = \"arith.cmpi\"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

// Custom form of vector equality comparison.
%x = arith.cmpi eq, %lhs, %rhs : vector<4xi64>

// Generic form of the same operation.
%x = \"arith.cmpi\"(%lhs, %rhs) {predicate = 0 : i64}
    : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
```
"""
function cmpi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    predicate,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.cmpi",
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

The `constant` operation produces an SSA value equal to some integer or
floating-point constant specified by an attribute. This is the way MLIR
forms simple integer and floating point constants.

# Example

```
// Integer constant
%1 = arith.constant 42 : i32

// Equivalent generic form
%1 = \"arith.constant\"() {value = 42 : i32} : () -> i32
```
"""
function constant(; result=nothing::Union{Nothing,IR.Type}, value, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.constant",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function divf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.divf",
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
`divsi`

Signed integer division. Rounds towards zero. Treats the leading bit as
sign, i.e. `6 / -2 = -3`.

Divison by zero, or signed division overflow (minimum value divided by -1)
is undefined behavior. When applied to `vector` and `tensor` values, the
behavior is undefined if _any_ of its elements are divided by zero or has a
signed division overflow.

If the `exact` attribute is present, the result value is poison if `lhs` is
not a multiple of `rhs`.

# Example

```mlir
// Scalar signed integer division.
%a = arith.divsi %b, %c : i64

// Scalar signed integer division where %b is known to be a multiple of %c.
%a = arith.divsi %b, %c exact : i64

// SIMD vector element-wise division.
%f = arith.divsi %g, %h : vector<4xi32>

// Tensor element-wise integer division.
%x = arith.divsi %y, %z : tensor<4x?xi8>
```
"""
function divsi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    isExact=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(isExact) && push!(attributes, namedattribute("isExact", isExact))

    return create_operation(
        "arith.divsi",
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
`divui`

Unsigned integer division. Rounds towards zero. Treats the leading bit as
the most significant, i.e. for `i16` given two\'s complement representation,
`6 / -2 = 6 / (2^16 - 2) = 0`.

Division by zero is undefined behavior. When applied to `vector` and
`tensor` values, the behavior is undefined if _any_ elements are divided by
zero.

If the `exact` attribute is present, the result value is poison if `lhs` is
not a multiple of `rhs`.

# Example

```mlir
// Scalar unsigned integer division.
%a = arith.divui %b, %c : i64

// Scalar unsigned integer division where %b is known to be a multiple of %c.
%a = arith.divui %b, %c exact : i64

// SIMD vector element-wise division.
%f = arith.divui %g, %h : vector<4xi32>

// Tensor element-wise integer division.
%x = arith.divui %y, %z : tensor<4x?xi8>
```
"""
function divui(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    isExact=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(isExact) && push!(attributes, namedattribute("isExact", isExact))

    return create_operation(
        "arith.divui",
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
`extf`

Cast a floating-point value to a larger floating-point-typed value.
The destination type must to be strictly wider than the source type.
When operating on vectors, casts elementwise.
"""
function extf(in::Value; out::IR.Type, fastmath=nothing, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.extf",
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
`extsi`

The integer sign extension operation takes an integer input of
width M and an integer destination type of width N. The destination
bit-width must be larger than the input bit-width (N > M).
The top-most (N - M) bits of the output are filled with copies
of the most-significant bit of the input.

# Example

```mlir
%1 = arith.constant 5 : i3      // %1 is 0b101
%2 = arith.extsi %1 : i3 to i6  // %2 is 0b111101
%3 = arith.constant 2 : i3      // %3 is 0b010
%4 = arith.extsi %3 : i3 to i6  // %4 is 0b000010

%5 = arith.extsi %0 : vector<2 x i32> to vector<2 x i64>
```
"""
function extsi(in::Value; out::IR.Type, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "arith.extsi",
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
`extui`

The integer zero extension operation takes an integer input of
width M and an integer destination type of width N. The destination
bit-width must be larger than the input bit-width (N > M).
The top-most (N - M) bits of the output are filled with zeros.

# Example

```mlir
  %1 = arith.constant 5 : i3      // %1 is 0b101
  %2 = arith.extui %1 : i3 to i6  // %2 is 0b000101
  %3 = arith.constant 2 : i3      // %3 is 0b010
  %4 = arith.extui %3 : i3 to i6  // %4 is 0b000010

  %5 = arith.extui %0 : vector<2 x i32> to vector<2 x i64>
```
"""
function extui(in::Value; out::IR.Type, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "arith.extui",
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
`fptosi`

Cast from a value interpreted as floating-point to the nearest (rounding
towards zero) signed integer value. When operating on vectors, casts
elementwise.
"""
function fptosi(in::Value; out::IR.Type, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "arith.fptosi",
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
`fptoui`

Cast from a value interpreted as floating-point to the nearest (rounding
towards zero) unsigned integer value. When operating on vectors, casts
elementwise.
"""
function fptoui(in::Value; out::IR.Type, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "arith.fptoui",
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
`floordivsi`

Signed integer division. Rounds towards negative infinity, i.e. `5 / -2 = -3`.

Divison by zero, or signed division overflow (minimum value divided by -1)
is undefined behavior. When applied to `vector` and `tensor` values, the
behavior is undefined if _any_ of its elements are divided by zero or has a
signed division overflow.

# Example

```mlir
// Scalar signed integer division.
%a = arith.floordivsi %b, %c : i64

```
"""
function floordivsi(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.floordivsi",
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
`index_cast`

Casts between scalar or vector integers and corresponding \'index\' scalar or
vectors. Index is an integer of platform-specific bit width. If casting to
a wider integer, the value is sign-extended. If casting to a narrower
integer, the value is truncated.
"""
function index_cast(in::Value; out::IR.Type, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "arith.index_cast",
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
`index_castui`

Casts between scalar or vector integers and corresponding \'index\' scalar or
vectors. Index is an integer of platform-specific bit width. If casting to
a wider integer, the value is zero-extended. If casting to a narrower
integer, the value is truncated.
"""
function index_castui(in::Value; out::IR.Type, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "arith.index_castui",
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
`maxnumf`

Returns the maximum of the two arguments.
If the arguments are -0.0 and +0.0, then the result is either of them.
If one of the arguments is NaN, then the result is the other argument.

# Example

```mlir
// Scalar floating-point maximum.
%a = arith.maxnumf %b, %c : f64
```
"""
function maxnumf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.maxnumf",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function maxsi(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.maxsi",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function maxui(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.maxui",
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
`maximumf`

Returns the maximum of the two arguments, treating -0.0 as less than +0.0.
If one of the arguments is NaN, then the result is also NaN.

# Example

```mlir
// Scalar floating-point maximum.
%a = arith.maximumf %b, %c : f64
```
"""
function maximumf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.maximumf",
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
`minnumf`

Returns the minimum of the two arguments.
If the arguments are -0.0 and +0.0, then the result is either of them.
If one of the arguments is NaN, then the result is the other argument.

# Example

```mlir
// Scalar floating-point minimum.
%a = arith.minnumf %b, %c : f64
```
"""
function minnumf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.minnumf",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function minsi(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.minsi",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function minui(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.minui",
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
`minimumf`

Returns the minimum of the two arguments, treating -0.0 as less than +0.0.
If one of the arguments is NaN, then the result is also NaN.

# Example

```mlir
// Scalar floating-point minimum.
%a = arith.minimumf %b, %c : f64
```
"""
function minimumf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.minimumf",
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
`mulf`

The `mulf` operation takes two operands and returns one result, each of
these is required to be the same type. This type may be a floating point
scalar type, a vector whose element type is a floating point type, or a
floating point tensor.

# Example

```mlir
// Scalar multiplication.
%a = arith.mulf %b, %c : f64

// SIMD pointwise vector multiplication, e.g. for Intel SSE.
%f = arith.mulf %g, %h : vector<4xf32>

// Tensor pointwise multiplication.
%x = arith.mulf %y, %z : tensor<4x?xbf16>
```

TODO: In the distant future, this will accept optional attributes for fast
math, contraction, rounding mode, and other controls.
"""
function mulf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.mulf",
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
`muli`

Performs N-bit multiplication on the operands. The operands are interpreted as
unsigned bitvectors. The result is represented by a bitvector containing the
mathematical value of the multiplication modulo 2^n, where `n` is the bitwidth.
Because `arith` integers use a two\'s complement representation, this operation is
applicable on both signed and unsigned integer operands.

The `muli` operation takes two operands and returns one result, each of
these is required to be the same type. This type may be an integer scalar type,
a vector whose element type is integer, or a tensor of integers.

This op supports `nuw`/`nsw` overflow flags which stands for
\"No Unsigned Wrap\" and \"No Signed Wrap\", respectively. If the `nuw` and/or
`nsw` flags are present, and an unsigned/signed overflow occurs
(respectively), the result is poison.

# Example

```mlir
// Scalar multiplication.
%a = arith.muli %b, %c : i64

// Scalar multiplication with overflow flags.
%a = arith.muli %b, %c overflow<nsw, nuw> : i64

// SIMD vector element-wise multiplication.
%f = arith.muli %g, %h : vector<4xi32>

// Tensor element-wise multiplication.
%x = arith.muli %y, %z : tensor<4x?xi8>
```
"""
function muli(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    overflowFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(overflowFlags) &&
        push!(attributes, namedattribute("overflowFlags", overflowFlags))

    return create_operation(
        "arith.muli",
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
`mulsi_extended`

Performs (2*N)-bit multiplication on sign-extended operands. Returns two
N-bit results: the low and the high halves of the product. The low half has
the same value as the result of regular multiplication `arith.muli` with
the same operands.

# Example

```mlir
// Scalar multiplication.
%low, %high = arith.mulsi_extended %a, %b : i32

// Vector element-wise multiplication.
%c:2 = arith.mulsi_extended %d, %e : vector<4xi32>

// Tensor element-wise multiplication.
%x:2 = arith.mulsi_extended %y, %z : tensor<4x?xi8>
```
"""
function mulsi_extended(
    lhs::Value,
    rhs::Value;
    low=nothing::Union{Nothing,IR.Type},
    high=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(low) && push!(op_ty_results, low)
    !isnothing(high) && push!(op_ty_results, high)

    return create_operation(
        "arith.mulsi_extended",
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
`mului_extended`

Performs (2*N)-bit multiplication on zero-extended operands. Returns two
N-bit results: the low and the high halves of the product. The low half has
the same value as the result of regular multiplication `arith.muli` with
the same operands.

# Example

```mlir
// Scalar multiplication.
%low, %high = arith.mului_extended %a, %b : i32

// Vector element-wise multiplication.
%c:2 = arith.mului_extended %d, %e : vector<4xi32>

// Tensor element-wise multiplication.
%x:2 = arith.mului_extended %y, %z : tensor<4x?xi8>
```
"""
function mului_extended(
    lhs::Value,
    rhs::Value;
    low=nothing::Union{Nothing,IR.Type},
    high=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(low) && push!(op_ty_results, low)
    !isnothing(high) && push!(op_ty_results, high)

    return create_operation(
        "arith.mului_extended",
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
`negf`

The `negf` operation computes the negation of a given value. It takes one
operand and returns one result of the same type. This type may be a float
scalar type, a vector whose element type is float, or a tensor of floats.
It has no standard attributes.

# Example

```mlir
// Scalar negation value.
%a = arith.negf %b : f64

// SIMD vector element-wise negation value.
%f = arith.negf %g : vector<4xf32>

// Tensor element-wise negation value.
%x = arith.negf %y : tensor<4x?xf8>
```
"""
function negf(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.negf",
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
`ori`

The `ori` operation takes two operands and returns one result, each of these
is required to be the same type. This type may be an integer scalar type, a
vector whose element type is integer, or a tensor of integers. It has no
standard attributes.

# Example

```mlir
// Scalar integer bitwise or.
%a = arith.ori %b, %c : i64

// SIMD vector element-wise bitwise integer or.
%f = arith.ori %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer or.
%x = arith.ori %y, %z : tensor<4x?xi8>
```
"""
function ori(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.ori",
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
`remf`

Returns the floating point division remainder.
The remainder has the same sign as the dividend (lhs operand).
"""
function remf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.remf",
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
`remsi`

Signed integer division remainder. Treats the leading bit as sign, i.e. `6 %
-2 = 0`.

Division by zero is undefined behavior. When applied to `vector` and
`tensor` values, the behavior is undefined if _any_ elements are divided by
zero.

# Example

```mlir
// Scalar signed integer division remainder.
%a = arith.remsi %b, %c : i64

// SIMD vector element-wise division remainder.
%f = arith.remsi %g, %h : vector<4xi32>

// Tensor element-wise integer division remainder.
%x = arith.remsi %y, %z : tensor<4x?xi8>
```
"""
function remsi(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.remsi",
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
`remui`

Unsigned integer division remainder. Treats the leading bit as the most
significant, i.e. for `i16`, `6 % -2 = 6 % (2^16 - 2) = 6`.

Division by zero is undefined behavior. When applied to `vector` and
`tensor` values, the behavior is undefined if _any_ elements are divided by
zero.

# Example

```mlir
// Scalar unsigned integer division remainder.
%a = arith.remui %b, %c : i64

// SIMD vector element-wise division remainder.
%f = arith.remui %g, %h : vector<4xi32>

// Tensor element-wise integer division remainder.
%x = arith.remui %y, %z : tensor<4x?xi8>
```
"""
function remui(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.remui",
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
`sitofp`

Cast from a value interpreted as a signed integer to the corresponding
floating-point value. If the value cannot be exactly represented, it is
rounded using the default rounding mode. When operating on vectors, casts
elementwise.
"""
function sitofp(in::Value; out::IR.Type, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "arith.sitofp",
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
`scaling_extf`

This operation upcasts input floating-point values using provided scale
values. It expects both scales and the input operand to be of the same shape,
making the operation elementwise. Scales are usually calculated per block
following the OCP MXFP spec as described in https://arxiv.org/abs/2310.10537.

If scales are calculated per block where blockSize != 1, then scales may
require broadcasting to make this operation elementwise. For example, let\'s
say the input is of shape `<dim1 x dim2 x ... dimN>`. Given blockSize != 1 and
assuming quantization happens on the last axis, the input can be reshaped to
`<dim1 x dim2 x ... (dimN/blockSize) x blockSize>`. Scales will be calculated
per block on the last axis. Therefore, scales will be of shape
`<dim1 x dim2 x ... (dimN/blockSize) x 1>`. Scales could also be of some other
shape as long as it is broadcast compatible with the input, e.g.,
`<1 x 1 x ... (dimN/blockSize) x 1>`.

In this example, before calling into `arith.scaling_extf`, scales must be
broadcasted to `<dim1 x dim2 x dim3 ... (dimN/blockSize) x blockSize>`. Note
that there could be multiple quantization axes. Internally,
`arith.scaling_extf` would perform the following:

```mlir
// Cast scale to result type.
%0 = arith.truncf %1 : f32 to f8E8M0FNU
%1 = arith.extf %0 : f8E8M0FNU to f16

// Cast input to result type.
%2 = arith.extf %3 : f4E2M1FN to f16

// Perform scaling
%3 = arith.mulf %2, %1 : f16
```
It propagates NaN values. Therefore, if either scale or the input element
contains NaN, then the output element value will also be a NaN.

# Example

```mlir
// Upcast from f4E2M1FN to f32.
%a = arith.scaling_extf %b, %c : f4E2M1FN, f8E8M0FNU to f32

// Element-wise upcast with broadcast (blockSize = 32).
%f = vector.broadcast %g : vector<1xf8E8M0FNU> to vector<32xf8E8M0FNU>
%h = arith.scaling_extf %i, %f : vector<32xf4E2M1FN>, vector<32xf8E8M0FNU> to vector<32xbf16>
```
"""
function scaling_extf(
    in::Value, scale::Value; out::IR.Type, fastmath=nothing, location=Location()
)
    op_ty_results = IR.Type[out,]
    operands = Value[in, scale]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.scaling_extf",
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
`scaling_truncf`

This operation downcasts input using the provided scale values. It expects
both scales and the input operand to be of the same shape and, therefore,
makes the operation elementwise. Scales are usually calculated per block
following the OCP MXFP spec as described in https://arxiv.org/abs/2310.10537.
Users are required to normalize and clamp the scales as necessary before calling
passing them to this operation.  OCP MXFP spec also does the flushing of denorms
on the input operand, which should be handled during lowering by passing appropriate
fastMath flag to this operation.

If scales are calculated per block where blockSize != 1, scales may require
broadcasting to make this operation elementwise. For example, let\'s say the
input is of shape `<dim1 x dim2 x ... dimN>`. Given blockSize != 1 and
assuming quantization happens on the last axis, the input can be reshaped to
`<dim1 x dim2 x ... (dimN/blockSize) x blockSize>`. Scales will be calculated
per block on the last axis. Therefore, scales will be of shape
`<dim1 x dim2 x ... (dimN/blockSize) x 1>`. Scales could also be of some other
shape as long as it is broadcast compatible with the input, e.g.,
`<1 x 1 x ... (dimN/blockSize) x 1>`.

In this example, before calling into `arith.scaling_truncf`, scales must be
broadcasted to `<dim1 x dim2 x dim3 ... (dimN/blockSize) x blockSize>`. Note
that there could be multiple quantization axes. Internally,
`arith.scaling_truncf` would perform the following:

```mlir
// Cast scale to input type.
%0 = arith.truncf %1 : f32 to f8E8M0FNU
%1 = arith.extf %0 : f8E8M0FNU to f16

// Perform scaling.
%3 = arith.divf %2, %1 : f16

// Cast to result type.
%4 = arith.truncf %3 : f16 to f4E2M1FN
```

# Example

```mlir
// Downcast from f32 to f4E2M1FN.
%a = arith.scaling_truncf %b, %c : f32, f8E8M0FNU to f4E2M1FN

// Element-wise downcast with broadcast (blockSize = 32).
%f = vector.broadcast %g : vector<1xf8E8M0FNU> to vector<32xf8E8M0FNU>
%h = arith.scaling_truncf %i, %f : vector<32xbf16>, vector<32xf8E8M0FNU> to vector<32xf4E2M1FN>
```
"""
function scaling_truncf(
    in::Value,
    scale::Value;
    out::IR.Type,
    roundingmode=nothing,
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[out,]
    operands = Value[in, scale]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(roundingmode) &&
        push!(attributes, namedattribute("roundingmode", roundingmode))
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.scaling_truncf",
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
`shli`

The `shli` operation shifts the integer value of the first operand to the left
by the integer value of the second operand. The second operand is interpreted as
unsigned. The low order bits are filled with zeros. If the value of the second
operand is greater or equal than the bitwidth of the first operand, then the
operation returns poison.

This op supports `nuw`/`nsw` overflow flags which stands for
\"No Unsigned Wrap\" and \"No Signed Wrap\", respectively. If the `nuw` and/or
`nsw` flags are present, and an unsigned/signed overflow occurs
(respectively), the result is poison.

# Example

```mlir
%1 = arith.constant 5 : i8  // %1 is 0b00000101
%2 = arith.constant 3 : i8
%3 = arith.shli %1, %2 : i8 // %3 is 0b00101000
%4 = arith.shli %1, %2 overflow<nsw, nuw> : i8
```
"""
function shli(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    overflowFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(overflowFlags) &&
        push!(attributes, namedattribute("overflowFlags", overflowFlags))

    return create_operation(
        "arith.shli",
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
`shrsi`

The `shrsi` operation shifts an integer value of the first operand to the right
by the value of the second operand. The first operand is interpreted as signed,
and the second operand is interpreter as unsigned. The high order bits in the
output are filled with copies of the most-significant bit of the shifted value
(which means that the sign of the value is preserved). If the value of the second
operand is greater or equal than bitwidth of the first operand, then the operation
returns poison.

If the `exact` attribute is present, the result value of shrsi is a poison
value if any of the bits shifted out are non-zero.

# Example

```mlir
%1 = arith.constant 160 : i8         // %1 is 0b10100000
%2 = arith.constant 3 : i8
%3 = arith.shrsi %1, %2 exact : i8   // %3 is 0b11110100
%4 = arith.constant 98 : i8          // %4 is 0b01100010
%5 = arith.shrsi %4, %2 : i8         // %5 is 0b00001100
```
"""
function shrsi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    isExact=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(isExact) && push!(attributes, namedattribute("isExact", isExact))

    return create_operation(
        "arith.shrsi",
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
`shrui`

The `shrui` operation shifts an integer value of the first operand to the right
by the value of the second operand. The first operand is interpreted as unsigned,
and the second operand is interpreted as unsigned. The high order bits are always
filled with zeros. If the value of the second operand is greater or equal than the
bitwidth of the first operand, then the operation returns poison.

If the `exact` attribute is present, the result value of shrui is a poison
value if any of the bits shifted out are non-zero.

# Example

```mlir
%1 = arith.constant 160 : i8        // %1 is 0b10100000
%2 = arith.constant 3 : i8
%3 = arith.constant 6 : i8
%4 = arith.shrui %1, %2 exact : i8  // %4 is 0b00010100
%5 = arith.shrui %1, %3 : i8        // %3 is 0b00000010
```
"""
function shrui(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    isExact=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(isExact) && push!(attributes, namedattribute("isExact", isExact))

    return create_operation(
        "arith.shrui",
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
`subf`

The `subf` operation takes two operands and returns one result, each of
these is required to be the same type. This type may be a floating point
scalar type, a vector whose element type is a floating point type, or a
floating point tensor.

# Example

```mlir
// Scalar subtraction.
%a = arith.subf %b, %c : f64

// SIMD vector subtraction, e.g. for Intel SSE.
%f = arith.subf %g, %h : vector<4xf32>

// Tensor subtraction.
%x = arith.subf %y, %z : tensor<4x?xbf16>
```

TODO: In the distant future, this will accept optional attributes for fast
math, contraction, rounding mode, and other controls.
"""
function subf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.subf",
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
`subi`

Performs N-bit subtraction on the operands. The operands are interpreted as unsigned
bitvectors. The result is represented by a bitvector containing the mathematical
value of the subtraction modulo 2^n, where `n` is the bitwidth. Because `arith`
integers use a two\'s complement representation, this operation is applicable on
both signed and unsigned integer operands.

The `subi` operation takes two operands and returns one result, each of
these is required to be the same type. This type may be an integer scalar type,
a vector whose element type is integer, or a tensor of integers.

This op supports `nuw`/`nsw` overflow flags which stands for
\"No Unsigned Wrap\" and \"No Signed Wrap\", respectively. If the `nuw` and/or
`nsw` flags are present, and an unsigned/signed overflow occurs
(respectively), the result is poison.

# Example

```mlir
// Scalar subtraction.
%a = arith.subi %b, %c : i64

// Scalar subtraction with overflow flags.
%a = arith.subi %b, %c overflow<nsw, nuw> : i64

// SIMD vector element-wise subtraction.
%f = arith.subi %g, %h : vector<4xi32>

// Tensor element-wise subtraction.
%x = arith.subi %y, %z : tensor<4x?xi8>
```
"""
function subi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    overflowFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(overflowFlags) &&
        push!(attributes, namedattribute("overflowFlags", overflowFlags))

    return create_operation(
        "arith.subi",
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
`truncf`

Truncate a floating-point value to a smaller floating-point-typed value.
The destination type must be strictly narrower than the source type.
If the value cannot be exactly represented, it is rounded using the
provided rounding mode or the default one if no rounding mode is provided.
When operating on vectors, casts elementwise.
"""
function truncf(
    in::Value; out::IR.Type, roundingmode=nothing, fastmath=nothing, location=Location()
)
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(roundingmode) &&
        push!(attributes, namedattribute("roundingmode", roundingmode))
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))

    return create_operation(
        "arith.truncf",
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
`trunci`

The integer truncation operation takes an integer input of
width M and an integer destination type of width N. The destination
bit-width must be smaller than the input bit-width (N < M).
The top-most (N - M) bits of the input are discarded.

This op supports `nuw`/`nsw` overflow flags which stands for \"No Unsigned
Wrap\" and \"No Signed Wrap\", respectively. If the nuw keyword is present,
and any of the truncated bits are non-zero, the result is a poison value.
If the nsw keyword is present, and any of the truncated bits are not the
same as the top bit of the truncation result, the result is a poison value.

# Example

```mlir
  // Scalar truncation.
  %1 = arith.constant 21 : i5     // %1 is 0b10101
  %2 = arith.trunci %1 : i5 to i4 // %2 is 0b0101
  %3 = arith.trunci %1 : i5 to i3 // %3 is 0b101

  // Vector truncation.
  %4 = arith.trunci %0 : vector<2 x i32> to vector<2 x i16>

  // Scalar truncation with overflow flags.
  %5 = arith.trunci %a overflow<nsw, nuw> : i32 to i16
```
"""
function trunci(in::Value; out::IR.Type, overflowFlags=nothing, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(overflowFlags) &&
        push!(attributes, namedattribute("overflowFlags", overflowFlags))

    return create_operation(
        "arith.trunci",
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
`uitofp`

Cast from a value interpreted as unsigned integer to the corresponding
floating-point value. If the value cannot be exactly represented, it is
rounded using the default rounding mode. When operating on vectors, casts
elementwise.
"""
function uitofp(in::Value; out::IR.Type, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "arith.uitofp",
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
`xori`

The `xori` operation takes two operands and returns one result, each of
these is required to be the same type. This type may be an integer scalar
type, a vector whose element type is integer, or a tensor of integers. It
has no standard attributes.

# Example

```mlir
// Scalar integer bitwise xor.
%a = arith.xori %b, %c : i64

// SIMD vector element-wise bitwise integer xor.
%f = arith.xori %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer xor.
%x = arith.xori %y, %z : tensor<4x?xi8>
```
"""
function xori(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.xori",
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
`select`

The `arith.select` operation chooses one value based on a binary condition
supplied as its first operand.

If the value of the first operand (the condition) is `1`, then the second
operand is returned, and the third operand is ignored, even if it was poison.

If the value of the first operand (the condition) is `0`, then the third
operand is returned, and the second operand is ignored, even if it was poison.

If the value of the first operand (the condition) is poison, then the
operation returns poison.

The operation applies to vectors and tensors elementwise given the _shape_
of all operands is identical. The choice is made for each element
individually based on the value at the same position as the element in the
condition operand. If an i1 is provided as the condition, the entire vector
or tensor is chosen.

# Example

```mlir
// Custom form of scalar selection.
%x = arith.select %cond, %true, %false : i32

// Generic form of the same operation.
%x = \"arith.select\"(%cond, %true, %false) : (i1, i32, i32) -> i32

// Element-wise vector selection.
%vx = arith.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

// Full vector selection.
%vx = arith.select %cond, %vtrue, %vfalse : vector<42xf32>
```
"""
function select(
    condition::Value,
    true_value::Value,
    false_value::Value;
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[condition, true_value, false_value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "arith.select",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

end # arith
