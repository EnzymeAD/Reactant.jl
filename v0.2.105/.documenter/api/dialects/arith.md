


# Arithmetic Dialect {#Arithmetic-Dialect}

Refer to the [official documentation](https://mlir.llvm.org/docs/Dialects/ArithOps/) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.addf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.addf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.addf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`addf`

The `addf` operation takes two operands and returns one result, each of these is required to be the same type. This type may be a floating point scalar type, a vector whose element type is a floating point type, or a floating point tensor.

**Example**

```mlir
// Scalar addition.
%a = arith.addf %b, %c : f64

// SIMD vector addition, e.g. for Intel SSE.
%f = arith.addf %g, %h : vector<4xf32>

// Tensor addition.
%x = arith.addf %y, %z : tensor<4x?xbf16>
```


TODO: In the distant future, this will accept optional attributes for fast math, contraction, rounding mode, and other controls.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L16-L39" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.addi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.addi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.addi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`addi`

Performs N-bit addition on the operands. The operands are interpreted as  unsigned bitvectors. The result is represented by a bitvector containing the  mathematical value of the addition modulo 2^n, where `n` is the bitwidth.  Because `arith` integers use a two&#39;s complement representation, this operation  is applicable on both signed and unsigned integer operands.

The `addi` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type,  a vector whose element type is integer, or a tensor of integers.

This op supports `nuw`/`nsw` overflow flags which stands stand for &quot;No Unsigned Wrap&quot; and &quot;No Signed Wrap&quot;, respectively. If the `nuw` and/or `nsw` flags are present, and an unsigned/signed overflow occurs (respectively), the result is poison.

**Example**

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



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L67-L100" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.addui_extended-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.addui_extended-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.addui_extended</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`addui_extended`

Performs (N+1)-bit addition on zero-extended operands. Returns two results: the N-bit sum (same type as both operands), and the overflow bit (boolean-like), where `1` indicates unsigned addition overflow, while `0` indicates no overflow.

**Example**

```mlir
// Scalar addition.
%sum, %overflow = arith.addui_extended %b, %c : i64, i1

// Vector element-wise addition.
%d:2 = arith.addui_extended %e, %f : vector<4xi32>, vector<4xi1>

// Tensor element-wise addition.
%x:2 = arith.addui_extended %y, %z : tensor<4x?xi8>, tensor<4x?xi1>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L129-L149" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.andi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.andi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.andi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`andi`

The `andi` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers. It has no standard attributes.

**Example**

```mlir
// Scalar integer bitwise and.
%a = arith.andi %b, %c : i64

// SIMD vector element-wise bitwise integer and.
%f = arith.andi %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer and.
%x = arith.andi %y, %z : tensor<4x?xi8>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L171-L191" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.bitcast-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.bitcast-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.bitcast</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`bitcast`

Bitcast an integer or floating point value to an integer or floating point value of equal bit width. When operating on vectors, casts elementwise.

Note that this implements a logical bitcast independent of target endianness. This allows constant folding without target information and is consitent with the bitcast constant folders in LLVM (see https://github.com/llvm/llvm-project/blob/18c19414eb/llvm/lib/IR/ConstantFold.cpp#L168) For targets where the source and target type have the same endianness (which is the standard), this cast will also change no bits at runtime, but it may still require an operation, for example if the machine has different floating point and integer register files. For targets that have a different endianness for the source and target types (e.g. float is big-endian and integer is little-endian) a proper lowering would add operations to swap the order of words in addition to the bitcast.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L214-L231" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.ceildivsi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.ceildivsi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.ceildivsi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`ceildivsi`

Signed integer division. Rounds towards positive infinity, i.e. `7 / -2 = -3`.

Divison by zero, or signed division overflow (minimum value divided by -1)  is undefined behavior. When applied to `vector` and `tensor` values, the  behavior is undefined if _any_ of its elements are divided by zero or has a  signed division overflow.

**Example**

```mlir
// Scalar signed integer division.
%a = arith.ceildivsi %b, %c : i64
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L251-L267" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.ceildivui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.ceildivui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.ceildivui</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`ceildivui`

Unsigned integer division. Rounds towards positive infinity. Treats the leading bit as the most significant, i.e. for `i16` given two&#39;s complement representation, `6 / -2 = 6 / (2^16 - 2) = 1`. 

Division by zero is undefined behavior. When applied to `vector` and  `tensor` values, the behavior is undefined if _any_ elements are divided by  zero.

**Example**

```mlir
// Scalar unsigned integer division.
%a = arith.ceildivui %b, %c : i64
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L290-L307" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.cmpf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.cmpf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.cmpf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cmpf`

The `cmpf` operation compares its two operands according to the float comparison rules and the predicate specified by the respective attribute. The predicate defines the type of comparison: (un)orderedness, (in)equality and signed less/greater than (or equal to) as well as predicates that are always true or false.  The operands must have the same type, and this type must be a float type, or a vector or tensor thereof.  The result is an i1, or a vector/tensor thereof having the same shape as the inputs. Unlike cmpi, the operands are always treated as signed. The u prefix indicates _unordered_ comparison, not unsigned comparison, so &quot;une&quot; means unordered or not equal. For the sake of readability by humans, custom assembly form for the operation uses a string-typed attribute for the predicate.  The value of this attribute corresponds to lower-cased name of the predicate constant, e.g., &quot;one&quot; means &quot;ordered not equal&quot;.  The string representation of the attribute is merely a syntactic sugar and is converted to an integer attribute by the parser.

**Example**

```mlir
%r1 = arith.cmpf oeq, %0, %1 : f32
%r2 = arith.cmpf ult, %0, %1 : tensor<42x42xf64>
%r3 = "arith.cmpf"(%0, %1) {predicate: 0} : (f8, f8) -> i1
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L330-L356" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.cmpi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.cmpi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.cmpi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cmpi`

The `cmpi` operation is a generic comparison for integer-like types. Its two arguments can be integers, vectors or tensors thereof as long as their types match. The operation produces an i1 for the former case, a vector or a tensor of i1 with the same shape as inputs in the other cases.

Its first argument is an attribute that defines which type of comparison is performed. The following comparisons are supported:
- equal (mnemonic: `"eq"`; integer value: `0`)
  
- not equal (mnemonic: `"ne"`; integer value: `1`)
  
- signed less than (mnemonic: `"slt"`; integer value: `2`)
  
- signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
  
- signed greater than (mnemonic: `"sgt"`; integer value: `4`)
  
- signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
  
- unsigned less than (mnemonic: `"ult"`; integer value: `6`)
  
- unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
  
- unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
  
- unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)
  

The result is `1` if the comparison is true and `0` otherwise. For vector or tensor operands, the comparison is performed elementwise and the element of the result indicates whether the comparison is true for the operand elements with the same indices as those of the result.

Note: while the custom assembly form uses strings, the actual underlying attribute has integer type (or rather enum class in C++ code) as seen from the generic assembly form. String literals are used to improve readability of the IR by humans.

This operation only applies to integer-like operands, but not floats. The main reason being that comparison operations have diverging sets of attributes: integers require sign specification while floats require various floating point-related particularities, e.g., `-ffast-math` behavior, IEEE754 compliance, etc ([rationale](../Rationale/Rationale.md#splitting-floating-point-vs-integer-operations)). The type of comparison is specified as attribute to avoid introducing ten similar operations, taking into account that they are often implemented using the same operation downstream ([rationale](../Rationale/Rationale.md#specifying-comparison-kind-as-attribute)). The separation between signed and unsigned order comparisons is necessary because of integers being signless. The comparison operation must know how to interpret values with the foremost bit being set: negatives in two&#39;s complement or large positives ([rationale](../Rationale/Rationale.md#specifying-sign-in-integer-comparison-operations)).

**Example**

```mlir
// Custom form of scalar "signed less than" comparison.
%x = arith.cmpi slt, %lhs, %rhs : i32

// Generic form of the same operation.
%x = "arith.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

// Custom form of vector equality comparison.
%x = arith.cmpi eq, %lhs, %rhs : vector<4xi64>

// Generic form of the same operation.
%x = "arith.cmpi"(%lhs, %rhs) {predicate = 0 : i64}
    : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L385-L449" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.constant-Tuple{}' href='#Reactant.MLIR.Dialects.arith.constant-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.constant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`constant`

The `constant` operation produces an SSA value equal to some integer or floating-point constant specified by an attribute. This is the way MLIR forms simple integer and floating point constants.

**Example**

```
// Integer constant
%1 = arith.constant 42 : i32

// Equivalent generic form
%1 = "arith.constant"() {value = 42 : i32} : () -> i32
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L476-L492" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.divsi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.divsi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.divsi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`divsi`

Signed integer division. Rounds towards zero. Treats the leading bit as sign, i.e. `6 / -2 = -3`.

Divison by zero, or signed division overflow (minimum value divided by -1)  is undefined behavior. When applied to `vector` and `tensor` values, the  behavior is undefined if _any_ of its elements are divided by zero or has a  signed division overflow.

**Example**

```mlir
// Scalar signed integer division.
%a = arith.divsi %b, %c : i64

// SIMD vector element-wise division.
%f = arith.divsi %g, %h : vector<4xi32>

// Tensor element-wise integer division.
%x = arith.divsi %y, %z : tensor<4x?xi8>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L540-L563" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.divui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.divui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.divui</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`divui`

Unsigned integer division. Rounds towards zero. Treats the leading bit as the most significant, i.e. for `i16` given two&#39;s complement representation, `6 / -2 = 6 / (2^16 - 2) = 0`.

Division by zero is undefined behavior. When applied to `vector` and  `tensor` values, the behavior is undefined if _any_ elements are divided by  zero.

**Example**

```mlir
// Scalar unsigned integer division.
%a = arith.divui %b, %c : i64

// SIMD vector element-wise division.
%f = arith.divui %g, %h : vector<4xi32>

// Tensor element-wise integer division.
%x = arith.divui %y, %z : tensor<4x?xi8>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L586-L609" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.extf-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.extf-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.extf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`extf`

Cast a floating-point value to a larger floating-point-typed value. The destination type must to be strictly wider than the source type. When operating on vectors, casts elementwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L632-L638" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.extsi-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.extsi-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.extsi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`extsi`

The integer sign extension operation takes an integer input of width M and an integer destination type of width N. The destination bit-width must be larger than the input bit-width (N &gt; M). The top-most (N - M) bits of the output are filled with copies of the most-significant bit of the input.

**Example**

```mlir
%1 = arith.constant 5 : i3      // %1 is 0b101
%2 = arith.extsi %1 : i3 to i6  // %2 is 0b111101
%3 = arith.constant 2 : i3      // %3 is 0b010
%4 = arith.extsi %3 : i3 to i6  // %4 is 0b000010

%5 = arith.extsi %0 : vector<2 x i32> to vector<2 x i64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L659-L678" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.extui-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.extui-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.extui</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`extui`

The integer zero extension operation takes an integer input of width M and an integer destination type of width N. The destination bit-width must be larger than the input bit-width (N &gt; M). The top-most (N - M) bits of the output are filled with zeros.

**Example**

```mlir
  %1 = arith.constant 5 : i3      // %1 is 0b101
  %2 = arith.extui %1 : i3 to i6  // %2 is 0b000101
  %3 = arith.constant 2 : i3      // %3 is 0b010
  %4 = arith.extui %3 : i3 to i6  // %4 is 0b000010

  %5 = arith.extui %0 : vector<2 x i32> to vector<2 x i64>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L698-L716" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.floordivsi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.floordivsi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.floordivsi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`floordivsi`

Signed integer division. Rounds towards negative infinity, i.e. `5 / -2 = -3`.

Divison by zero, or signed division overflow (minimum value divided by -1)  is undefined behavior. When applied to `vector` and `tensor` values, the  behavior is undefined if _any_ of its elements are divided by zero or has a  signed division overflow.

**Example**

```mlir
// Scalar signed integer division.
%a = arith.floordivsi %b, %c : i64

```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L788-L805" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.fptosi-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.fptosi-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.fptosi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`fptosi`

Cast from a value interpreted as floating-point to the nearest (rounding towards zero) signed integer value. When operating on vectors, casts elementwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L736-L742" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.fptoui-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.fptoui-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.fptoui</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`fptoui`

Cast from a value interpreted as floating-point to the nearest (rounding towards zero) unsigned integer value. When operating on vectors, casts elementwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L762-L768" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.index_cast-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.index_cast-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.index_cast</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`index_cast`

Casts between scalar or vector integers and corresponding &#39;index&#39; scalar or vectors. Index is an integer of platform-specific bit width. If casting to a wider integer, the value is sign-extended. If casting to a narrower integer, the value is truncated.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L828-L835" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.index_castui-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.index_castui-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.index_castui</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`index_castui`

Casts between scalar or vector integers and corresponding &#39;index&#39; scalar or vectors. Index is an integer of platform-specific bit width. If casting to a wider integer, the value is zero-extended. If casting to a narrower integer, the value is truncated.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L855-L862" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.maximumf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.maximumf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.maximumf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`maximumf`

Returns the maximum of the two arguments, treating -0.0 as less than +0.0. If one of the arguments is NaN, then the result is also NaN.

**Example**

```mlir
// Scalar floating-point maximum.
%a = arith.maximumf %b, %c : f64
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L967-L979" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.maxnumf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.maxnumf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.maxnumf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`maxnumf`

Returns the maximum of the two arguments. If the arguments are -0.0 and +0.0, then the result is either of them. If one of the arguments is NaN, then the result is the other argument.

**Example**

```mlir
// Scalar floating-point maximum.
%a = arith.maxnumf %b, %c : f64
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L882-L895" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.minimumf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.minimumf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.minimumf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`minimumf`

Returns the minimum of the two arguments, treating -0.0 as less than +0.0. If one of the arguments is NaN, then the result is also NaN.

**Example**

```mlir
// Scalar floating-point minimum.
%a = arith.minimumf %b, %c : f64
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1092-L1104" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.minnumf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.minnumf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.minnumf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`minnumf`

Returns the minimum of the two arguments. If the arguments are -0.0 and +0.0, then the result is either of them. If one of the arguments is NaN, then the result is the other argument.

**Example**

```mlir
// Scalar floating-point minimum.
%a = arith.minnumf %b, %c : f64
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1007-L1020" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.mulf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.mulf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.mulf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mulf`

The `mulf` operation takes two operands and returns one result, each of these is required to be the same type. This type may be a floating point scalar type, a vector whose element type is a floating point type, or a floating point tensor.

**Example**

```mlir
// Scalar multiplication.
%a = arith.mulf %b, %c : f64

// SIMD pointwise vector multiplication, e.g. for Intel SSE.
%f = arith.mulf %g, %h : vector<4xf32>

// Tensor pointwise multiplication.
%x = arith.mulf %y, %z : tensor<4x?xbf16>
```


TODO: In the distant future, this will accept optional attributes for fast math, contraction, rounding mode, and other controls.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1132-L1155" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.muli-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.muli-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.muli</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`muli`

Performs N-bit multiplication on the operands. The operands are interpreted as unsigned bitvectors. The result is represented by a bitvector containing the mathematical value of the multiplication modulo 2^n, where `n` is the bitwidth. Because `arith` integers use a two&#39;s complement representation, this operation is applicable on both signed and unsigned integer operands.

The `muli` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers.

This op supports `nuw`/`nsw` overflow flags which stands stand for &quot;No Unsigned Wrap&quot; and &quot;No Signed Wrap&quot;, respectively. If the `nuw` and/or `nsw` flags are present, and an unsigned/signed overflow occurs (respectively), the result is poison.

**Example**

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



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1183-L1216" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.mulsi_extended-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.mulsi_extended-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.mulsi_extended</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mulsi_extended`

Performs (2*N)-bit multiplication on sign-extended operands. Returns two N-bit results: the low and the high halves of the product. The low half has the same value as the result of regular multiplication `arith.muli` with the same operands.

**Example**

```mlir
// Scalar multiplication.
%low, %high = arith.mulsi_extended %a, %b : i32

// Vector element-wise multiplication.
%c:2 = arith.mulsi_extended %d, %e : vector<4xi32>

// Tensor element-wise multiplication.
%x:2 = arith.mulsi_extended %y, %z : tensor<4x?xi8>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1245-L1265" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.mului_extended-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.mului_extended-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.mului_extended</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mului_extended`

Performs (2*N)-bit multiplication on zero-extended operands. Returns two N-bit results: the low and the high halves of the product. The low half has the same value as the result of regular multiplication `arith.muli` with the same operands.

**Example**

```mlir
// Scalar multiplication.
%low, %high = arith.mului_extended %a, %b : i32

// Vector element-wise multiplication.
%c:2 = arith.mului_extended %d, %e : vector<4xi32>

// Tensor element-wise multiplication.
%x:2 = arith.mului_extended %y, %z : tensor<4x?xi8>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1293-L1313" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.negf-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.negf-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.negf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`negf`

The `negf` operation computes the negation of a given value. It takes one operand and returns one result of the same type. This type may be a float scalar type, a vector whose element type is float, or a tensor of floats. It has no standard attributes.

**Example**

```mlir
// Scalar negation value.
%a = arith.negf %b : f64

// SIMD vector element-wise negation value.
%f = arith.negf %g : vector<4xf32>

// Tensor element-wise negation value.
%x = arith.negf %y : tensor<4x?xf8>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1341-L1361" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.ori-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.ori-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.ori</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`ori`

The `ori` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers. It has no standard attributes.

**Example**

```mlir
// Scalar integer bitwise or.
%a = arith.ori %b, %c : i64

// SIMD vector element-wise bitwise integer or.
%f = arith.ori %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer or.
%x = arith.ori %y, %z : tensor<4x?xi8>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1388-L1408" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.remf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.remf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.remf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`remf`

Returns the floating point division remainder. The remainder has the same sign as the dividend (lhs operand).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1431-L1436" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.remsi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.remsi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.remsi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`remsi`

Signed integer division remainder. Treats the leading bit as sign, i.e. `6 % -2 = 0`.

Division by zero is undefined behavior. When applied to `vector` and  `tensor` values, the behavior is undefined if _any_ elements are divided by  zero.

**Example**

```mlir
// Scalar signed integer division remainder.
%a = arith.remsi %b, %c : i64

// SIMD vector element-wise division remainder.
%f = arith.remsi %g, %h : vector<4xi32>

// Tensor element-wise integer division remainder.
%x = arith.remsi %y, %z : tensor<4x?xi8>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1464-L1486" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.remui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.remui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.remui</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`remui`

Unsigned integer division remainder. Treats the leading bit as the most significant, i.e. for `i16`, `6 % -2 = 6 % (2^16 - 2) = 6`.

Division by zero is undefined behavior. When applied to `vector` and  `tensor` values, the behavior is undefined if _any_ elements are divided by  zero.

**Example**

```mlir
// Scalar unsigned integer division remainder.
%a = arith.remui %b, %c : i64

// SIMD vector element-wise division remainder.
%f = arith.remui %g, %h : vector<4xi32>

// Tensor element-wise integer division remainder.
%x = arith.remui %y, %z : tensor<4x?xi8>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1509-L1531" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.select-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.select-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.select</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`select`

The `arith.select` operation chooses one value based on a binary condition supplied as its first operand.

If the value of the first operand (the condition) is `1`, then the second operand is returned, and the third operand is ignored, even if it was poison.

If the value of the first operand (the condition) is `0`, then the third operand is returned, and the second operand is ignored, even if it was poison.

If the value of the first operand (the condition) is poison, then the operation returns poison.

The operation applies to vectors and tensors elementwise given the _shape_ of all operands is identical. The choice is made for each element individually based on the value at the same position as the element in the condition operand. If an i1 is provided as the condition, the entire vector or tensor is chosen.

**Example**

```mlir
// Custom form of scalar selection.
%x = arith.select %cond, %true, %false : i32

// Generic form of the same operation.
%x = "arith.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

// Element-wise vector selection.
%vx = arith.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

// Full vector selection.
%vx = arith.select %cond, %vtrue, %vfalse : vector<42xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1967-L2003" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.shli-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.shli-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.shli</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`shli`

The `shli` operation shifts the integer value of the first operand to the left  by the integer value of the second operand. The second operand is interpreted as  unsigned. The low order bits are filled with zeros. If the value of the second  operand is greater or equal than the bitwidth of the first operand, then the operation returns poison.

This op supports `nuw`/`nsw` overflow flags which stands stand for &quot;No Unsigned Wrap&quot; and &quot;No Signed Wrap&quot;, respectively. If the `nuw` and/or `nsw` flags are present, and an unsigned/signed overflow occurs (respectively), the result is poison.

**Example**

```mlir
%1 = arith.constant 5 : i8  // %1 is 0b00000101
%2 = arith.constant 3 : i8
%3 = arith.shli %1, %2 : i8 // %3 is 0b00101000
%4 = arith.shli %1, %2 overflow<nsw, nuw> : i8  
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1581-L1603" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.shrsi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.shrsi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.shrsi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`shrsi`

The `shrsi` operation shifts an integer value of the first operand to the right  by the value of the second operand. The first operand is interpreted as signed,  and the second operand is interpreter as unsigned. The high order bits in the  output are filled with copies of the most-significant bit of the shifted value  (which means that the sign of the value is preserved). If the value of the second  operand is greater or equal than bitwidth of the first operand, then the operation returns poison.

**Example**

```mlir
%1 = arith.constant 160 : i8               // %1 is 0b10100000
%2 = arith.constant 3 : i8
%3 = arith.shrsi %1, %2 : (i8, i8) -> i8   // %3 is 0b11110100
%4 = arith.constant 96 : i8                   // %4 is 0b01100000
%5 = arith.shrsi %4, %2 : (i8, i8) -> i8   // %5 is 0b00001100
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1632-L1652" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.shrui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.shrui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.shrui</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`shrui`

The `shrui` operation shifts an integer value of the first operand to the right  by the value of the second operand. The first operand is interpreted as unsigned, and the second operand is interpreted as unsigned. The high order bits are always  filled with zeros. If the value of the second operand is greater or equal than the bitwidth of the first operand, then the operation returns poison.

**Example**

```mlir
%1 = arith.constant 160 : i8               // %1 is 0b10100000
%2 = arith.constant 3 : i8
%3 = arith.shrui %1, %2 : (i8, i8) -> i8   // %3 is 0b00010100
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1675-L1691" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.sitofp-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.sitofp-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.sitofp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`sitofp`

Cast from a value interpreted as a signed integer to the corresponding floating-point value. If the value cannot be exactly represented, it is rounded using the default rounding mode. When operating on vectors, casts elementwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1554-L1561" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.subf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.subf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.subf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`subf`

The `subf` operation takes two operands and returns one result, each of these is required to be the same type. This type may be a floating point scalar type, a vector whose element type is a floating point type, or a floating point tensor.

**Example**

```mlir
// Scalar subtraction.
%a = arith.subf %b, %c : f64

// SIMD vector subtraction, e.g. for Intel SSE.
%f = arith.subf %g, %h : vector<4xf32>

// Tensor subtraction.
%x = arith.subf %y, %z : tensor<4x?xbf16>
```


TODO: In the distant future, this will accept optional attributes for fast math, contraction, rounding mode, and other controls.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1714-L1737" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.subi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.subi-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.subi</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`subi`

Performs N-bit subtraction on the operands. The operands are interpreted as unsigned bitvectors. The result is represented by a bitvector containing the mathematical value of the subtraction modulo 2^n, where `n` is the bitwidth. Because `arith` integers use a two&#39;s complement representation, this operation is applicable on both signed and unsigned integer operands.

The `subi` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers.

This op supports `nuw`/`nsw` overflow flags which stands stand for &quot;No Unsigned Wrap&quot; and &quot;No Signed Wrap&quot;, respectively. If the `nuw` and/or `nsw` flags are present, and an unsigned/signed overflow occurs (respectively), the result is poison.

**Example**

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



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1765-L1798" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.truncf-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.truncf-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.truncf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`truncf`

Truncate a floating-point value to a smaller floating-point-typed value. The destination type must be strictly narrower than the source type. If the value cannot be exactly represented, it is rounded using the provided rounding mode or the default one if no rounding mode is provided. When operating on vectors, casts elementwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1827-L1835" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.trunci-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.trunci-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.trunci</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`trunci`

The integer truncation operation takes an integer input of width M and an integer destination type of width N. The destination bit-width must be smaller than the input bit-width (N &lt; M). The top-most (N - M) bits of the input are discarded.

**Example**

```mlir
  %1 = arith.constant 21 : i5     // %1 is 0b10101
  %2 = arith.trunci %1 : i5 to i4 // %2 is 0b0101
  %3 = arith.trunci %1 : i5 to i3 // %3 is 0b101

  %5 = arith.trunci %0 : vector<2 x i32> to vector<2 x i16>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1860-L1877" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.uitofp-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.uitofp-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.uitofp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`uitofp`

Cast from a value interpreted as unsigned integer to the corresponding floating-point value. If the value cannot be exactly represented, it is rounded using the default rounding mode. When operating on vectors, casts elementwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1897-L1904" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.arith.xori-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.arith.xori-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.arith.xori</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`xori`

The `xori` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers. It has no standard attributes.

**Example**

```mlir
// Scalar integer bitwise xor.
%a = arith.xori %b, %c : i64

// SIMD vector element-wise bitwise integer xor.
%f = arith.xori %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer xor.
%x = arith.xori %y, %z : tensor<4x?xi8>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Arith.jl#L1924-L1944" target="_blank" rel="noreferrer">source</a></Badge>

</details>

