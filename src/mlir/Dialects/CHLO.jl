module chlo
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
`acos`

Returns `Acos(operand)` element-wise.

\$\$
\\acos(x) = 2 * \\atan(\\sqrt(1 - x^2) / (1 + x)) if x != -1
         = pi                                  if x == -1
\$\$
"""
function acos(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.acos",
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
`acosh`

Returns `Acosh(operand)` element-wise.

\$\$
\\acosh(x) = log(x + sqrt(x^2 - 1))      if x >= -1
\\acosh(x) = nan                         if x < -1
\$\$
"""
function acosh(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.acosh",
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
`_asin_acos_kernel`

Returns `AsinAcosKernel(operand)` element-wise.

```
If
  w = _asin_acos_kernel(z)
  w\' = _asin_acos_kernel(I * z)
Then
  asin(z) = complex(atan2(z.real, w.real), sign(z.imag) * w.imag)
  acos(z) = complex(atan2(w.real, z.real), -sign(z.imag) * w.imag)
  asinh(z) = complex(sign(z.real) * w\'.imag, atan2(z.imag, w\'.real))
  acosh(z) = complex(w.imag, sign(z.imag) * atan2(w.real, z.real))
```

This op is used as an intermediate value in decompositions and
should never be constructed directly by frameworks or consumed by
backends.
"""
function _asin_acos_kernel(
    operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo._asin_acos_kernel",
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
`asin`

Returns `Asin(operand)` element-wise.

\$\$
\\asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
\$\$
"""
function asin(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.asin",
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
`asinh`

Returns `Asinh(operand)` element-wise.

\$\$
\\asinh(x) = log(x + sqrt(x^2 + 1))
\$\$
"""
function asinh(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.asinh",
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
`atan`

Returns `Atan(operand)` element-wise.

\$\$
\\atan(x) = \\atan2(x, 1)
\$\$
"""
function atan(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.atan",
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
`atanh`

Returns `Atanh(operand)` element-wise.

\$\$
\\atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) <= 1
          = nan                          otherwise
\$\$
"""
function atanh(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.atanh",
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
`bessel_i1e`

Returns `bessel_i1e(operand)` element-wise.
"""
function bessel_i1e(
    operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.bessel_i1e",
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
`broadcast_add`

Returns `lhs + rhs` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_add(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_add",
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
`broadcast_and`

Returns `logical_and(lhs, rhs)` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_and(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_and",
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
`broadcast_atan2`

Returns `atan2(lhs/rhs)` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_atan2(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_atan2",
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
`broadcast_compare`

Compares `lhs` and `rhs` elementwise according to `comparison_direction`
and `compare_type`. If unspecified, `compare_type` is FLOAT for float element
types, SIGNED for signed element types and UNSIGNED for unsigned element
types.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.
"""
function broadcast_compare(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    comparison_direction,
    compare_type=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute(
        "comparison_direction", comparison_direction
    ),]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))
    !isnothing(compare_type) &&
        push!(attributes, namedattribute("compare_type", compare_type))

    return create_operation(
        "chlo.broadcast_compare",
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
`broadcast_complex`

Performs element-wise conversion of a pair of real and imaginary values to
a complex value.
"""
function broadcast_complex(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_complex",
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
`broadcast_divide`

Returns `lhs / rhs` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_divide(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_divide",
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
`broadcast_maximum`

Returns `max(lhs, rhs)` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_maximum(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_maximum",
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
`broadcast_minimum`

Returns `min(lhs, rhs)` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_minimum(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_minimum",
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
`broadcast_multiply`

Returns `lhs * rhs` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_multiply(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_multiply",
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
`broadcast_next_after`

Returns the next representable value of `lhs` in the direction of `rhs`,
element-wise. It can also return a subnormal number.

Equivalent to the C++ std::nextafter function.
"""
function broadcast_next_after(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_next_after",
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
`broadcast_or`

Returns `logical_or(lhs, rhs)` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_or(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_or",
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
`broadcast_polygamma`

Returns `Polygamma(operand, operand)` element-wise.
"""
function broadcast_polygamma(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_polygamma",
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
`broadcast_power`

Returns `lhs ^ rhs` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_power(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_power",
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
`broadcast_remainder`

Returns `lhs % rhs` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_remainder(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_remainder",
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
`broadcast_select`

Constructs an output array from elements of two input arrays, based on the
values of a predicate array.

See https://www.tensorflow.org/xla/operation_semantics#select
"""
function broadcast_select(
    pred::Value,
    on_true::Value,
    on_false::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[pred, on_true, on_false]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "chlo.broadcast_select",
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
`broadcast_shift_left`

Returns `lhs << rhs` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_shift_left(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_shift_left",
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
`broadcast_shift_right_arithmetic`

Returns `lhs >> rhs` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_shift_right_arithmetic(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_shift_right_arithmetic",
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
`broadcast_shift_right_logical`

Returns `lhs >> rhs` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_shift_right_logical(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_shift_right_logical",
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
`broadcast_subtract`

Returns `lhs - rhs` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_subtract(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_subtract",
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
`broadcast_xor`

Returns `logical_xor(lhs, rhs)` element-wise.

See
https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
"""
function broadcast_xor(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_xor",
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
`broadcast_zeta`

Returns `Zeta(operand, operand)` element-wise.

\$\$
\\(\\zeta(x, q) = \\sum_{n=0}^{\\infty} (q + n)^{-x}\\)
\$\$
"""
function broadcast_zeta(
    lhs::Value,
    rhs::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    broadcast_dimensions=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(broadcast_dimensions) &&
        push!(attributes, namedattribute("broadcast_dimensions", broadcast_dimensions))

    return create_operation(
        "chlo.broadcast_zeta",
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
`conj`

Returns `Conj(operand)` element-wise.

\$\$
\\conj(x) = (\\real(x), \\neg(\\imag(x)))
\$\$
"""
function conj(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.conj",
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
`constant_like`

Returns a splat constant of the same shape as the operand.
"""
function constant_like(
    operand::Value; result_0=nothing::Union{Nothing,IR.Type}, value, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "chlo.constant_like",
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

Represents a constant value.
"""
function constant(; output=nothing::Union{Nothing,IR.Type}, value, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]
    !isnothing(output) && push!(op_ty_results, output)

    return create_operation(
        "chlo.constant",
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
`cosh`

Returns `Cosh(operand)` element-wise.

\$\$
\\cosh(x) = (e^x + e^-x) / 2
\$\$
"""
function cosh(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.cosh",
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
`digamma`

Returns `Digamma(operand)` element-wise.
"""
function digamma(
    operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.digamma",
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
`erf_inv`

Returns `ErfInv(operand)` element-wise.
"""
function erf_inv(
    operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.erf_inv",
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
`erf`

Computes the Gauss error function of `x` element-wise.

erf(x) = erf_impl(x)            if |x| < 1
       = 1 - erfc_impl(x)       otherwise
"""
function erf(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.erf",
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
`erfc`

Computes an approximation of the error function complement (1 - erf(x)).

erfc(x) = erfc_impl(x)           if |x| > 1
        = 1 - erf_impl(x)        otherwise
"""
function erfc(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.erfc",
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
`is_inf`

Returns if a value is +/-inf element-wise.
"""
function is_inf(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.is_inf",
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
`is_neg_inf`

Returns if a value is -inf element-wise.
"""
function is_neg_inf(
    operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.is_neg_inf",
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
`is_pos_inf`

Returns if a value is +inf element-wise.
"""
function is_pos_inf(
    operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.is_pos_inf",
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
`lgamma`

Returns `Lgamma(operand)` element-wise.
"""
function lgamma(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.lgamma",
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
`next_after`

Returns the next representable value of `x` in the direction of `y`,
element-wise. It can also return a subnormal number.

Equivalent to the C++ std::nextafter function.
"""
function next_after(
    x::Value, y::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[x, y]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.next_after",
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
`polygamma`

Returns `Polygamma(operand, operand)` element-wise.
"""
function polygamma(
    n::Value, x::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[n, x]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.polygamma",
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
`ragged_dot`


This operation takes three tensor args---lhs, rhs, and group_sizes---and
a \"ragged_dot_dimension_numbers\" attribute. Like dot_general, the lhs and
rhs are allowed arbitrary batch and contracting dimensions. Additionally,
the lhs is required to have one ragged dimension, and the rhs may have at
most one group dimension. The op has three modes, depending on the kind of
the lhs ragged dimension.

In mode 1, the shape-signature is `[b,m,k], [g,b,k,n], [b,g] -> [b,m,n]`.
Here the ragged dimension is an lhs non-contracting dimension (`m`). The
dimensions `b` and `k` represent batch and contracting dimensions
respectively. The rhs is required to have a group dimension (`g`).

In mode 2, the shape-signature is `[b,m,k], [b,k,n], [b,g] -> [g,b,m,n]`.
Here the ragged dimension is an lhs/rhs contracting dimension (`k`).

In mode 3, the shape-signature is `[b,m,k], [b,k,n], [g] -> [b,m,n]`. Here
the ragged dimension is an lhs/rhs batch dimension (`b`).
"""
function ragged_dot(
    lhs::Value,
    rhs::Value,
    group_sizes::Value;
    result::IR.Type,
    ragged_dot_dimension_numbers,
    precision_config=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs, group_sizes]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute(
        "ragged_dot_dimension_numbers", ragged_dot_dimension_numbers
    ),]
    !isnothing(precision_config) &&
        push!(attributes, namedattribute("precision_config", precision_config))

    return create_operation(
        "chlo.ragged_dot",
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
`sinh`

Returns `Sinh(operand)` element-wise.

\$\$
\\sinh(x) = (e^x - e^-x) / 2                     if |x| < 1
         = e^(x + log(1/2)) - e^(-x + log(1/2)) otherwise.
\$\$
"""
function sinh(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.sinh",
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
`square`

Returns `Square(operand)` element-wise.

\$\$
\\square(x) = complex((x.real - x.imag) * (x.real + x.imag), x.real * x.imag * 2) if x is a complex number
           = x * x                                                               otherwise
\$\$
"""
function square(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.square",
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
`tan`

Returns `Tan(operand)` element-wise.

\$\$
\\tan(x) = \\sin(x) / \\cos(x)
\$\$
"""
function tan(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.tan",
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
`top_k`

If the input is a vector (rank-1), finds the `k` largest entries in the
vector and outputs their values and indices as vectors.  Thus `values[j]` is
the `j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

```
values.shape = indices.shape = input.shape[:-1] + [k]
```

If two elements are equal, the lower-index element appears first.
"""
function top_k(
    operand::Value;
    values=nothing::Union{Nothing,IR.Type},
    indices=nothing::Union{Nothing,IR.Type},
    k,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("k", k),]
    !isnothing(values) && push!(op_ty_results, values)
    !isnothing(indices) && push!(op_ty_results, indices)

    return create_operation(
        "chlo.top_k",
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
`zeta`

Returns `Zeta(operand, operand)` element-wise.

\$\$
\\(\\zeta(x, q) = \\sum_{n=0}^{\\infty} (q + n)^{-x}\\)
\$\$
"""
function zeta(
    x::Value, q::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[x, q]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "chlo.zeta",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

end # chlo
