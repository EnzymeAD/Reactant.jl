module complex
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
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API

"""
`abs`

The `abs` op takes a single complex number and computes its absolute value.

# Example

```mlir
%a = complex.abs %b : complex<f32>
```
"""
function abs(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.abs",
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
`add`

The `add` operation takes two complex numbers and returns their sum.

# Example

```mlir
%a = complex.add %b, %c : complex<f32>
```
"""
function add(
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
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.add",
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
`angle`

The `angle` op takes a single complex number and computes its argument value with a branch cut along the negative real axis.

# Example

```mlir
     %a = complex.angle %b : complex<f32>
```
"""
function angle(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.angle",
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
`atan2`

For complex numbers it is expressed using complex logarithm
atan2(y, x) = -i * log((x + i * y) / sqrt(x**2 + y**2))

# Example

```mlir
%a = complex.atan2 %b, %c : complex<f32>
```
"""
function atan2(
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
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.atan2",
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


# Example

```mlir
     %a = complex.bitcast %b : complex<f32> -> i64
```
"""
function bitcast(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "complex.bitcast",
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
`conj`

The `conj` op takes a single complex number and computes the
complex conjugate.

# Example

```mlir
%a = complex.conj %b: complex<f32>
```
"""
function conj(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.conj",
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

The `complex.constant` operation creates a constant complex number from an
attribute containing the real and imaginary parts.

# Example

```mlir
%a = complex.constant [0.1, -1.0] : complex<f64>
```
"""
function constant(; complex::IR.Type, value, location=Location())
    op_ty_results = IR.Type[complex,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("value", value),]

    return create_operation(
        "complex.constant",
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
`cos`

The `cos` op takes a single complex number and computes the cosine of
it, i.e. `cos(x)`, where `x` is the input value.

# Example

```mlir
%a = complex.cos %b : complex<f32>
```
"""
function cos(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.cos",
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
`create`

The `complex.create` operation creates a complex number from two
floating-point operands, the real and the imaginary part.

# Example

```mlir
%a = complex.create %b, %c : complex<f32>
```
"""
function create(real::Value, imaginary::Value; complex::IR.Type, location=Location())
    op_ty_results = IR.Type[complex,]
    operands = Value[real, imaginary]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "complex.create",
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
`div`

The `div` operation takes two complex numbers and returns result of their
division:

```mlir
%a = complex.div %b, %c : complex<f32>
```
"""
function div(
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
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.div",
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
`eq`

The `eq` op takes two complex numbers and returns whether they are equal.

# Example

```mlir
%a = complex.eq %b, %c : complex<f32>
```
"""
function eq(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "complex.eq",
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
`exp`

The `exp` op takes a single complex number and computes the exponential of
it, i.e. `exp(x)` or `e^(x)`, where `x` is the input value.
`e` denotes Euler\'s number and is approximately equal to 2.718281.

# Example

```mlir
%a = complex.exp %b : complex<f32>
```
"""
function exp(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.exp",
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
`expm1`

complex.expm1(x) := complex.exp(x) - 1

# Example

```mlir
%a = complex.expm1 %b : complex<f32>
```
"""
function expm1(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.expm1",
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
`im`

The `im` op takes a single complex number and extracts the imaginary part.

# Example

```mlir
%a = complex.im %b : complex<f32>
```
"""
function im(
    complex::Value;
    imaginary=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(imaginary) && push!(op_ty_results, imaginary)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.im",
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
`log1p`

The `log` op takes a single complex number and computes the natural
logarithm of one plus the given value, i.e. `log(1 + x)` or `log_e(1 + x)`,
where `x` is the input value. `e` denotes Euler\'s number and is
approximately equal to 2.718281.

# Example

```mlir
%a = complex.log1p %b : complex<f32>
```
"""
function log1p(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.log1p",
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
`log`

The `log` op takes a single complex number and computes the natural
logarithm of it, i.e. `log(x)` or `log_e(x)`, where `x` is the input value.
`e` denotes Euler\'s number and is approximately equal to 2.718281.

# Example

```mlir
%a = complex.log %b : complex<f32>
```
"""
function log(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.log",
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
`mul`

The `mul` operation takes two complex numbers and returns their product:

```mlir
%a = complex.mul %b, %c : complex<f32>
```
"""
function mul(
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
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.mul",
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
`neg`

The `neg` op takes a single complex number `complex` and returns `-complex`.

# Example

```mlir
%a = complex.neg %b : complex<f32>
```
"""
function neg(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.neg",
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
`neq`

The `neq` op takes two complex numbers and returns whether they are not
equal.

# Example

```mlir
%a = complex.neq %b, %c : complex<f32>
```
"""
function neq(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "complex.neq",
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
`pow`

The `pow` operation takes a complex number raises it to the given complex
exponent.

# Example

```mlir
%a = complex.pow %b, %c : complex<f32>
```
"""
function pow(
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
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.pow",
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
`powi`

The `powi` operation takes a `base` operand of complex type and a `power`
operand of signed integer type and returns one result of the same type
as `base`. The result is `base` raised to the power of `power`.

# Example

```mlir
%a = complex.powi %b, %c : complex<f32>, i32
```
"""
function powi(
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
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.powi",
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
`re`

The `re` op takes a single complex number and extracts the real part.

# Example

```mlir
%a = complex.re %b : complex<f32>
```
"""
function re(
    complex::Value;
    real=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(real) && push!(op_ty_results, real)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.re",
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
`rsqrt`

The `rsqrt` operation computes reciprocal of square root.

# Example

```mlir
%a = complex.rsqrt %b : complex<f32>
```
"""
function rsqrt(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.rsqrt",
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
`sign`

The `sign` op takes a single complex number and computes the sign of
it, i.e. `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

# Example

```mlir
%a = complex.sign %b : complex<f32>
```
"""
function sign(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.sign",
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
`sin`

The `sin` op takes a single complex number and computes the sine of
it, i.e. `sin(x)`, where `x` is the input value.

# Example

```mlir
%a = complex.sin %b : complex<f32>
```
"""
function sin(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.sin",
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
`sqrt`

The `sqrt` operation takes a complex number and returns its square root.

# Example

```mlir
%a = complex.sqrt %b : complex<f32>
```
"""
function sqrt(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.sqrt",
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
`sub`

The `sub` operation takes two complex numbers and returns their difference.

# Example

```mlir
%a = complex.sub %b, %c : complex<f32>
```
"""
function sub(
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
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.sub",
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

The `tan` op takes a single complex number and computes the tangent of
it, i.e. `tan(x)`, where `x` is the input value.

# Example

```mlir
%a = complex.tan %b : complex<f32>
```
"""
function tan(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.tan",
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
`tanh`

The `tanh` operation takes a complex number and returns its hyperbolic
tangent.

# Example

```mlir
%a = complex.tanh %b : complex<f32>
```
"""
function tanh(
    complex::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[complex,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(fastmath) && push!(attributes, NamedAttribute("fastmath", fastmath))

    return create_operation(
        "complex.tanh",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

end # complex
