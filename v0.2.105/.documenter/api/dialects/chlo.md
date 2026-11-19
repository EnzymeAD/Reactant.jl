


# CHLO Dialect {#CHLO-Dialect}

Refer to the [official documentation](https://github.com/openxla/xla/tree/main/xla/mlir_hlo#hlo-client-dialect-chlo) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo._asin_acos_kernel-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo._asin_acos_kernel-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo._asin_acos_kernel</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`_asin_acos_kernel`

Returns `AsinAcosKernel(operand)` element-wise.

```
If
  w = _asin_acos_kernel(z)
  w' = _asin_acos_kernel(I * z)
Then
  asin(z) = complex(atan2(z.real, w.real), sign(z.imag) * w.imag)
  acos(z) = complex(atan2(w.real, z.real), -sign(z.imag) * w.imag)
  asinh(z) = complex(sign(z.real) * w'.imag, atan2(z.imag, w'.real))
  acosh(z) = complex(w.imag, sign(z.imag) * atan2(w.real, z.real))
```


This op is used as an intermediate value in decompositions and should never be constructed directly by frameworks or consumed by backends.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L76-L95" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.acos-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.acos-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.acos</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`acos`

Returns `Acos(operand)` element-wise.

$

\acos(x) = 2 * \atan(\sqrt(1 - x^2) / (1 + x)) if x != -1          = pi                                  if x == -1 $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L16-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.acosh-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.acosh-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.acosh</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`acosh`

Returns `Acosh(operand)` element-wise.

$

\acosh(x) = log(x + sqrt(x^2 - 1))      if x &gt;= -1 \acosh(x) = nan                         if x &lt; -1 $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L46-L55" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.asin-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.asin-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.asin</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`asin`

Returns `Asin(operand)` element-wise.

$

\asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2))) $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L118-L126" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.asinh-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.asinh-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.asinh</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`asinh`

Returns `Asinh(operand)` element-wise.

$

\asinh(x) = log(x + sqrt(x^2 + 1)) $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L147-L155" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.atan-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.atan-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.atan</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`atan`

Returns `Atan(operand)` element-wise.

$

\atan(x) = \atan2(x, 1) $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L176-L184" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.atanh-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.atanh-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.atanh</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`atanh`

Returns `Atanh(operand)` element-wise.

$

\atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) &lt;= 1           = nan                          otherwise $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L205-L214" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.bessel_i1e-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.bessel_i1e-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.bessel_i1e</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`bessel_i1e`

Returns `bessel_i1e(operand)` element-wise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L235-L239" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_add-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_add-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_add</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_add`

Returns `lhs + rhs` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L262-L269" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_and-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_and-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_and</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_and`

Returns `logical_and(lhs, rhs)` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L298-L305" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_atan2-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_atan2-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_atan2</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_atan2`

Returns `atan2(lhs/rhs)` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L334-L341" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_compare-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_compare-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_compare</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_compare`

Compares `lhs` and `rhs` elementwise according to `comparison_direction` and `compare_type`. If unspecified, `compare_type` is FLOAT for float element types, SIGNED for signed element types and UNSIGNED for unsigned element types.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L370-L380" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_complex-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_complex-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_complex</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_complex`

Performs element-wise conversion of a pair of real and imaginary values to a complex value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L415-L420" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_divide-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_divide-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_divide</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_divide`

Returns `lhs / rhs` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L449-L456" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_maximum-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_maximum-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_maximum</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_maximum`

Returns `max(lhs, rhs)` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L485-L492" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_minimum-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_minimum-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_minimum</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_minimum`

Returns `min(lhs, rhs)` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L521-L528" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_multiply-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_multiply-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_multiply</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_multiply`

Returns `lhs * rhs` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L557-L564" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_next_after-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_next_after-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_next_after</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_next_after`

Returns the next representable value of `lhs` in the direction of `rhs`, element-wise. It can also return a subnormal number.

Equivalent to the C++ std::nextafter function.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L593-L600" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_or-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_or-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_or</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_or`

Returns `logical_or(lhs, rhs)` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L629-L636" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_polygamma-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_polygamma-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_polygamma</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_polygamma`

Returns `Polygamma(operand, operand)` element-wise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L665-L669" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_power-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_power-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_power</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_power`

Returns `lhs ^ rhs` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L698-L705" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_remainder-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_remainder-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_remainder</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_remainder`

Returns `lhs % rhs` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L734-L741" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_select-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_select-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_select</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_select`

Constructs an output array from elements of two input arrays, based on the values of a predicate array.

See https://www.tensorflow.org/xla/operation_semantics#select


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L770-L777" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_shift_left-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_shift_left-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_shift_left</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_shift_left`

Returns `lhs << rhs` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L804-L811" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_shift_right_arithmetic-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_shift_right_arithmetic-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_shift_right_arithmetic</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_shift_right_arithmetic`

Returns `lhs >> rhs` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L840-L847" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_shift_right_logical-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_shift_right_logical-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_shift_right_logical</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_shift_right_logical`

Returns `lhs >> rhs` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L876-L883" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_subtract-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_subtract-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_subtract</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_subtract`

Returns `lhs - rhs` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L912-L919" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_xor-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_xor-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_xor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_xor`

Returns `logical_xor(lhs, rhs)` element-wise.

See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L948-L955" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.broadcast_zeta-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.broadcast_zeta-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.broadcast_zeta</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast_zeta`

Returns `Zeta(operand, operand)` element-wise.

$

(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}) $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L984-L992" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.conj-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.conj-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.conj</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`conj`

Returns `Conj(operand)` element-wise.

$

\conj(x) = (\real(x), \neg(\imag(x))) $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1021-L1029" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.constant-Tuple{}' href='#Reactant.MLIR.Dialects.chlo.constant-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.constant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`constant`

Represents a constant value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1077-L1081" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.constant_like-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.constant_like-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.constant_like</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`constant_like`

Returns a splat constant of the same shape as the operand.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1050-L1054" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.cosh-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.cosh-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.cosh</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cosh`

Returns `Cosh(operand)` element-wise.

$

\cosh(x) = (e^x + e^-x) / 2 $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1102-L1110" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.digamma-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.digamma-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.digamma</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`digamma`

Returns `Digamma(operand)` element-wise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1131-L1135" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.erf-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.erf-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.erf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`erf`

Computes the Gauss error function of `x` element-wise.

erf(x) = erf_impl(x)            if |x| &lt; 1        = 1 - erfc_impl(x)       otherwise


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1185-L1192" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.erf_inv-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.erf_inv-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.erf_inv</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`erf_inv`

Returns `ErfInv(operand)` element-wise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1158-L1162" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.erfc-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.erfc-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.erfc</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`erfc`

Computes an approximation of the error function complement (1 - erf(x)).

erfc(x) = erfc_impl(x)           if |x| &gt; 1         = 1 - erf_impl(x)        otherwise


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1213-L1220" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.is_inf-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.is_inf-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.is_inf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`is_inf`

Returns if a value is +/-inf element-wise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1241-L1245" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.is_neg_inf-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.is_neg_inf-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.is_neg_inf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`is_neg_inf`

Returns if a value is -inf element-wise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1266-L1270" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.is_pos_inf-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.is_pos_inf-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.is_pos_inf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`is_pos_inf`

Returns if a value is +inf element-wise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1293-L1297" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.lgamma-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.lgamma-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.lgamma</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`lgamma`

Returns `Lgamma(operand)` element-wise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1320-L1324" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.next_after-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.next_after-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.next_after</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`next_after`

Returns the next representable value of `x` in the direction of `y`, element-wise. It can also return a subnormal number.

Equivalent to the C++ std::nextafter function.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1345-L1352" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.polygamma-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.polygamma-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.polygamma</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`polygamma`

Returns `Polygamma(operand, operand)` element-wise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1375-L1379" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.ragged_dot-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.ragged_dot-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.ragged_dot</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`ragged_dot`

This operation takes three tensor args–-lhs, rhs, and group_sizes–-and a &quot;ragged_dot_dimension_numbers&quot; attribute. Like dot_general, the lhs and rhs are allowed arbitrary batch and contracting dimensions. Additionally, the lhs is required to have one ragged dimension, and the rhs may have at most one group dimension. The op has three modes, depending on the kind of the lhs ragged dimension.

In mode 1, the shape-signature is `[b,m,k], [g,b,k,n], [b,g] -> [b,m,n]`. Here the ragged dimension is an lhs non-contracting dimension (`m`). The dimensions `b` and `k` represent batch and contracting dimensions respectively. The rhs is required to have a group dimension (`g`).

In mode 2, the shape-signature is `[b,m,k], [b,k,n], [b,g] -> [g,b,m,n]`. Here the ragged dimension is an lhs/rhs contracting dimension (`k`).

In mode 3, the shape-signature is `[b,m,k], [b,k,n], [g] -> [b,m,n]`. Here the ragged dimension is an lhs/rhs batch dimension (`b`).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1402-L1423" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.sinh-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.sinh-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.sinh</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`sinh`

Returns `Sinh(operand)` element-wise.

$

\sinh(x) = (e^x - e^-x) / 2                     if |x| &lt; 1          = e^(x + log(1/2)) - e^(-x + log(1/2)) otherwise. $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1455-L1464" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.square-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.square-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.square</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`square`

Returns `Square(operand)` element-wise.

$

\square(x) = complex((x.real - x.imag) * (x.real + x.imag), x.real * x.imag * 2) if x is a complex number            = x * x                                                               otherwise $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1485-L1494" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.tan-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.tan-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.tan</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tan`

Returns `Tan(operand)` element-wise.

$

\tan(x) = \sin(x) / \cos(x) $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1515-L1523" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.top_k-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.top_k-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.top_k</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`top_k`

If the input is a vector (rank-1), finds the `k` largest entries in the vector and outputs their values and indices as vectors.  Thus `values[j]` is the `j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each row (resp. vector along the last dimension).  Thus,

```
values.shape = indices.shape = input.shape[:-1] + [k]
```


If two elements are equal, the lower-index element appears first.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1544-L1559" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.chlo.zeta-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.chlo.zeta-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.chlo.zeta</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`zeta`

Returns `Zeta(operand, operand)` element-wise.

$

(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}) $


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/CHLO.jl#L1587-L1595" target="_blank" rel="noreferrer">source</a></Badge>

</details>

