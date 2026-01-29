module shape
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
import ..Dialects: NamedAttribute, operandsegmentsizes, resultsegmentsizes
import ...API

"""
`add`

Adds two sizes or indices. If either operand is an error it will be
propagated to the result. The operands can be of type `size` or `index`. If
at least one of the operands can hold an error, i.e. if it is of type
`size`, the result must be of type `size`. If error propagation is not
possible because both operands are of type `index` then the result may be
of type `size` or `index`.
"""
function add(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.add",
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
`any`

This operation takes multiple input shapes or extent tensors and returns
some combination of their dimensions. This can be best seen with examples
below.

The result is undefined, but still side-effect free, in cases where the
inputs have differing ranks or differ in extents of shared dimensions.

# Example
```mlir
%s0 = shape.any [2,?], [?,3] // [2,3]
%s1 = shape.any [?,?], [1,2] // [1,2]
```
"""
function any(inputs::Vector{Value}; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.any",
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
`assuming_all`

Used to simplify constraints as any single failing precondition is enough
to prevent execution.

\"assuming\" operations represent an execution order restriction to the
compiler, information for dependent code to rely on (by assuming), and
nothing else. They should not exist after a program is fully lowered and
ready to execute.

# Example
```mlir
%w0 = shape.cstr_broadcastable [2,2], [3,1,2] // Passing
%w1 = shape.cstr_broadcastable [2,2], [3,2] // Failure
%w2 = shape.cstr_eq [1,2], [1,2], [1,2] // Passing
%wf = shape.assuming_all %w0, %w1 // Failure
%wt = shape.assuming_all %w0, %w2 // Passing
```
"""
function assuming_all(
    inputs::Vector{Value}; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.assuming_all",
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
`assuming`

Executes the region assuming all witnesses are true.

\"assuming\" operations represent an execution order restriction to the
compiler, information for dependent code to rely on (by assuming), and
nothing else. They should not exist after a program is fully lowered and
ready to execute.
"""
function assuming(
    witness::Value; results::Vector{IR.Type}, doRegion::Region, location=Location()
)
    op_ty_results = IR.Type[results...,]
    operands = Value[witness,]
    owned_regions = Region[doRegion,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.assuming",
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
`assuming_yield`

This yield operation represents a return operation within the
`shape.assuming` operation region. The operation takes variable number of
operands and produces no results. The operand number and types must match
the number and types of parent `shape.assuming` results.
"""
function assuming_yield(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.assuming_yield",
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
`broadcast`

Returns the broadcasted shape for input shapes or extent tensors. The rest
of this description is simplified for the 2 input case but can be extended
to more inputs. Both operands can be of type `shape.shape` or
`tensor<?xindex>`. The result is of type `shape.shape` and, if both
operands are tensors, may be of type `tensor<?xindex>`.

If the two operand shapes are of different rank the smaller one is padded
with 1\'s from the left. The resulting broadcasted shape is then defined as

    result[i] = lhs[i] if lhs[i] == rhs[i]
              = lhs[i] if rhs[i] == 1
              = rhs[i] if lhs[i] == 1.

In case the resulting shape is undefined, i.e. if corresponding extents are
different from each other but none is 1, the result is an error shape.
Likewise error values are propagated if any of the operands holds an error
value. If the result type is an extent tensor (and can therefore not hold
the error value) the behavior may be undefined. The optional string
attribute can be used to describe the error case.
"""
function broadcast(
    shapes::Vector{Value}; result::IR.Type, error=nothing, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[shapes...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(error) && push!(attributes, NamedAttribute("error", error))

    return create_operation(
        "shape.broadcast",
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

Creates a shape whose dimensions consist of first the dimensions from `lhs`
followed by the dimensions of `rhs`.

# Example
concat([2,3], [4,5]) -> [2,3,4,5]
concat([], []) -> []
concat([], [4,5,6]) -> [4,5,6]
"""
function concat(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.concat",
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
`const_shape`

Creates a constant shape or extent tensor. The individual extents are given
as the `shape` attribute. The number of these values equals the shape\'s
rank.

```mlir
%0 = shape.const_shape [] : !shape.shape
%1 = shape.const_shape [1, 2, 3] : !shape.shape
%2 = shape.const_shape [4, 5, 6] : tensor<3xindex>
```
"""
function const_shape(; result=nothing::Union{Nothing,IR.Type}, shape, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("shape", shape),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.const_shape",
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
`const_size`

Creates a `shape.size` type representing the constant size given by `value`.

```mlir
%x = shape.const_size 10
```
"""
function const_size(; result=nothing::Union{Nothing,IR.Type}, value, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("value", value),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.const_size",
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
`const_witness`

This operation represents a statically known witness result. This can be
often used to canonicalize/fold constraint and assuming code that will always
pass.

```mlir
%0 = shape.const_shape [1,2,3]
%1 = shape.const_shape [1,2,3]
%w0 = shape.cstr_eq(%0, %1) // Can be folded to \"const_witness true\"
%w1 = shape.const_witness true
%w2 = shape.assuming_all(%w0, %w2) // Can be folded to \"const_witness true\"
```
"""
function const_witness(;
    result=nothing::Union{Nothing,IR.Type}, passing, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("passing", passing),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.const_witness",
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
`cstr_broadcastable`

Given input shapes or extent tensors, return a witness specifying if they
are broadcastable. This broadcastable follows the same logic as what
shape.broadcast documents.

\"cstr\" operations represent runtime assertions.

# Example
```mlir
%w0 = shape.cstr_broadcastable [2,2], [3,1,2] // Passing
%w1 = shape.cstr_broadcastable [2,2], [3,2] // Failure
```
"""
function cstr_broadcastable(
    shapes::Vector{Value}; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[shapes...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.cstr_broadcastable",
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
`cstr_eq`

Given 1 or more input shapes, determine if all shapes are the exact same.

\"cstr\" operations represent runtime assertions.

# Example
```mlir
%w0 = shape.cstr_eq [1,2], [1,2], [1,2] // Passing
%w1 = shape.cstr_eq [2,2], [1,2] // Failure
```
"""
function cstr_eq(
    shapes::Vector{Value}; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[shapes...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.cstr_eq",
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
`cstr_require`

Represents a runtime assertion that an i1 is true. It returns a
!shape.witness to order this assertion.

For simplicity, prefer using other cstr_* ops if they are available for a
given constraint.

# Example
```mlir
%bool = ...
%w0 = shape.cstr_require %bool, \"msg\" // Passing if `%bool` is true.
```

Since this op can be used to express many different possible assertions
(depending on whatever computation calculated `pred`), the `msg`
should clarify the nature of the assertion for users.
"""
function cstr_require(
    pred::Value; result=nothing::Union{Nothing,IR.Type}, msg, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[pred,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("msg", msg),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.cstr_require",
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
`debug_print`

Prints the input dim or shape and passes through input.

Note: This is intended for testing and debugging only.
"""
function debug_print(input::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.debug_print",
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

Gets the extent indexed by `dim` from the shape of the `value` operand. If
the index is error or out-of-bound then it returns an invalid size if the
return type carries error information else the behavior is undefined.

This is a convenience op that performs the equivalent of getting the extent
of a shape (e.g., `dim(x, i) == get_extent(shape_of(x), i)`).
"""
function dim(
    value::Value, index::Value; extent=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[value, index]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(extent) && push!(op_ty_results, extent)

    return create_operation(
        "shape.dim",
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
`div`

Divides two sizes or indices. If either operand is an error it will be
propagated to the result. The operands can be of type `size` or `index`.
If at least one of the operands can hold an error, i.e. if it is of type
`size`, the result must be of type `size`. If error propagation is not
possible because both operands are of type `index` then the result may be
of type  `size` or `index`. If both operands and result are of type
`index`, their runtime values could be negative. The result is rounded
toward negative infinity, i.e. floor(lhs / rhs), such that

    div(lhs, rhs) * rhs + mod(lhs, rhs) = lhs

always holds. If any of the values is of type `size`, the behavior for
negative value is undefined.
"""
function div(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.div",
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
`from_extent_tensor`

Creates a shape from a 1D integral tensor of extents. The rank of the
resulting shape equals the number of elements in the tensor, and the
extents match the values of the elements.
"""
function from_extent_tensor(
    input::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.from_extent_tensor",
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
`from_extents`

Creates a shape from multiple SSA values representing the extents of
the shape.

```mlir
// Rank 2 shape.
%s0 = shape.from_extents %a, %b
// Rank 0 shape.
%s1 = shape.from_extents
```
"""
function from_extents(
    extents::Vector{Value}; shape=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[extents...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(shape) && push!(op_ty_results, shape)

    return create_operation(
        "shape.from_extents",
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
`func`

An operation with a name containing a single `SSACFG` region which
represents a shape transfer function or helper function for shape transfer
function.
"""
function func(;
    sym_name,
    function_type,
    arg_attrs=nothing,
    res_attrs=nothing,
    sym_visibility=nothing,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("sym_name", sym_name), NamedAttribute("function_type", function_type)
    ]
    !isnothing(arg_attrs) && push!(attributes, NamedAttribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, NamedAttribute("res_attrs", res_attrs))
    !isnothing(sym_visibility) &&
        push!(attributes, NamedAttribute("sym_visibility", sym_visibility))

    return create_operation(
        "shape.func",
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
`function_library`

Represents a list of shape functions and the ops whose shape transfer
functions they represent.

# Example

```mlir
shape.function_library {
  func @same_result_shape(%arg: !shape.value_shape) -> !shape.shape {
    %0 = shape_of %arg : !shape.value_shape -> !shape.shape
    return %0 : !shape.shape
  }
} mapping {
  std.atan = @same_result_shape
}
```
"""
function function_library(;
    sym_name, sym_visibility=nothing, mapping, body::Region, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("sym_name", sym_name), NamedAttribute("mapping", mapping)
    ]
    !isnothing(sym_visibility) &&
        push!(attributes, NamedAttribute("sym_visibility", sym_visibility))

    return create_operation(
        "shape.function_library",
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
`get_extent`

Gets the extent indexed by `dim` from the `shape` operand. If the shape is
an error then it returns an invalid size.
"""
function get_extent(
    shape::Value, dim::Value; extent=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[shape, dim]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(extent) && push!(op_ty_results, extent)

    return create_operation(
        "shape.get_extent",
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
`index_to_size`

Converts a standard index to a `shape.size`. This operation and its
inverse, `size_to_index`, facilitate index conversion between the standard
and the shape dialect.

The behavior is undefined for negative indices.
"""
function index_to_size(
    arg::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.index_to_size",
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
`is_broadcastable`

Given multiple input shapes or extent tensors, return a predicate
specifying if they are broadcastable. This broadcastable follows the same
logic as what shape.broadcast documents.

Concretely, shape.is_broadcastable returning true implies that
shape.broadcast will not give an error, and shape.cstr_broadcastable will
not result in an assertion failure. Similarly, false implies an error or
assertion failure.

# Example
```mlir
%true = shape.is_broadcastable [2,2], [3,1,2]
%false = shape.is_broadcastable [2,2], [3,2]
```
"""
function is_broadcastable(
    shapes::Vector{Value}; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[shapes...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.is_broadcastable",
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
`max`

Computes the elementwise maximum of two sizes or shapes with equal ranks.
If either operand is an error, then an error will be propagated to the
result. If the input types mismatch or the ranks do not match, then the
result is an error.
"""
function max(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.max",
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
`meet`

An operation that computes the least general shape or dim of input operands.
This effectively asserts that corresponding static dimensions are equal.
The behavior is to match each element of the shape/size and propagate the
most restrictive information, returning an invalid shape if there are
contradictory requirements. E.g., using pseudo code

```
shape.meet([*], [*]) -> [*]
shape.meet([*], [1, ?]) -> [1, ?]
shape.meet([1, 2], [1, ?]) -> [1, 2]
shape.meet([*], [1, 2]) -> [1, 2]
shape.meet([], []) -> []
shape.meet([], [*]) -> []
shape.meet([], [?, ?]) -> [invalid]
shape.meet([1, ?], [2, ?, ?]) -> [invalid]
```

`shape.meet` also allows specifying an optional error string, that may be
used to return an error to the user upon mismatch of dimensions.

```mlir
%c = shape.meet %a, %b, error=\"<reason>\" : !shape.shape, !shape.shape -> !shape.shape
```
"""
function meet(
    arg0::Value,
    arg1::Value;
    result=nothing::Union{Nothing,IR.Type},
    error=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[arg0, arg1]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(error) && push!(attributes, NamedAttribute("error", error))

    return create_operation(
        "shape.meet",
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

Computes the elementwise minimum of two sizes or shapes with equal ranks.
If either operand is an error, then an error will be propagated to the
result. If the input types mismatch or the ranks do not match, then the
result is an error.
"""
function min(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.min",
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

Multiplies two sizes or indices. If either operand is an error it will be
propagated to the result. The operands can be of type `size` or `index`. If
at least one of the operands can hold an error, i.e. if it is of type
`size`, the result must be of type `size`. If error propagation is not
possible because both operands are of type `index` then the result may be
of type `size` or `index`.
"""
function mul(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.mul",
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
`num_elements`

Returns the number of elements for a given shape which is the product of
its extents. If the argument is of type `shape` then the result will be of
type `size` and potential errors will be propagated. Otherwise, if the
argument is and extent tensor `tensor<?xindex>` then the result will be of
type `index`.
"""
function num_elements(
    shape::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[shape,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.num_elements",
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
`rank`

Returns the rank of the shape or extent tensor, i.e. the number of extents.
"""
function rank(shape::Value; rank=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[shape,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(rank) && push!(op_ty_results, rank)

    return create_operation(
        "shape.rank",
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
`reduce`

An operation that takes as input a shape or extent tensor, and a number of
initial values. This operation has a region that is applied repeatedly for
every extent of the input. Starting with the initial values, the individual
extents are then aggregated as defined by the associated region.

Conceptually this op performs the following reduction:

```
res[] = init;
for (int i = 0, i < shape.rank(); i++) {
  res = reduce(i, shape[i], res[0], ..., res[n]);
}
```

Where `reduce` represents the region attached and the result of the reduce
op is the last computed output of the reduce region. As an example, the
number of elements can be computed as follows:

```mlir
func.func @reduce(%shape : !shape.shape, %init : !shape.size) ->
    !shape.size {
  %num_elements = shape.reduce(%shape, %init) -> !shape.size  {
    ^bb0(%index: index, %dim: !shape.size, %acc: !shape.size):
      %updated_acc = \"shape.mul\"(%acc, %dim) :
        (!shape.size, !shape.size) -> !shape.size
      shape.yield %updated_acc : !shape.size
  }
  return %num_elements : !shape.size
}
```
"""
function reduce(
    shape::Value,
    initVals::Vector{Value};
    result::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result...,]
    operands = Value[shape, initVals...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.reduce",
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
`return_`

The `shape.return` operation represents a return operation within a
function.  The operation takes variable number of operands and produces no
results.
"""
function return_(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.return",
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
`shape_eq`

Takes one or more shape or extent tensor operands and determines whether
they are equal. When extent tensors are compared to shapes they are
regarded as their equivalent non-error shapes. Error shapes can be tested
for equality like any other shape value, meaning that the error value is
equal to itself.
"""
function shape_eq(
    shapes::Vector{Value}; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[shapes...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.shape_eq",
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
`shape_of`

The operation takes a value or a shaped operand as an argument and it
returns a shape or extent tensor.
"""
function shape_of(arg::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.shape_of",
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
`size_to_index`

Converts a `shape.size` to a standard index. This operation and its
inverse, `index_to_size`, facilitate index conversion between the standard
and the shape dialect. The behavior is undefined for unknown and invalid
arguments.
"""
function size_to_index(
    arg::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.size_to_index",
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
`split_at`

Splits a shape at a given dimension `index`, returning two shapes. If
`index` is negative, it is treated as indexing from the back of the shape.
This negative-handling behavior is important when handling unranked shapes,
where the positive index is not necessarily knowable due to a dynamic
number of leading dimensions. If the result is in extent tensor form out of
bounds indices result in undefined behavior.

Examples:
- split_at([4,5,6], index=0) -> [], [4,5,6]
- split_at([4,5,6], index=1) -> [4], [5,6]
- split_at([4,5,6], index=2) -> [4,5], [6]
- split_at([4,5,6], index=3) -> [4,5,6], []
- split_at([4,5,6], index=4) -> error
- split_at([4,5,6], index=-1) -> [4,5], [6]
- split_at([4,5,6], index=-2) -> [4], [5,6]
- split_at([4,5,6], index=-3) -> [], [4,5,6]
- split_at([4,5,6], index=-4) -> error

Requires:
- `index` is in the range [-rank(operand),rank(operand)]
"""
function split_at(
    operand::Value, index::Value; head::IR.Type, tail::IR.Type, location=Location()
)
    op_ty_results = IR.Type[head, tail]
    operands = Value[operand, index]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.split_at",
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
`to_extent_tensor`

Converts a shape to a 1D integral tensor of extents. The number of elements
in the tensor equals the rank of the shape, and the elements equal the
extents of the shape.

If the shape represents an error, this op\'s behavior is undefined.
"""
function to_extent_tensor(input::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.to_extent_tensor",
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
`value_as_shape`

The operations takes a ValueShape and returns a Shape corresponding to the
value.  If the input value cannot be shape (e.g., not a 1D tensor of
integral value representing sizes) then this propagages the error shape.
E.g.,

```mlir
// The following
%0 = arith.constant dense<[1,2]> : tensor<2xi32>
%shape = shape.value_as_shape %0 : tensor<2xi32> -> !shape.shape
// is equivalent to
%shape\' = shape.const_shape [1, 2] : !shape.shape
```

This operation is the complement of `shape_of` wrt ValueShape values.
"""
function value_as_shape(arg::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.value_as_shape",
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
`value_of`

The operation takes !shape.value_shape, a.k.a. (value, shape) tuple as an
argument, and returns its value. The behavior is undefined for unknown and
invalid arguments.
"""
function value_of(arg::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.value_of",
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
`with_shape`

Returns ValueShape with the shape updated to match the shape operand. That
is a new ValueShape tuple is created with value equal to `operand`\'s
value and shape equal to `shape`. If the ValueShape and given `shape` are
non-conformant, then the returned ValueShape will represent an error of
this mismatch. Similarly if either inputs are in an error state, then an
error is propagated.

Usage:
  %0 = shape.with_shape %1, %2 : tensor<...>, !shape.shape

This is used, for example, where one combines shape function calculations
and/or call one shape function from another. E.g.,

```mlir
func.func @shape_foobah(%a: !shape.value_shape,
                   %b: !shape.value_shape,
                   %c: !shape.value_shape) -> !shape.shape {
  %0 = call @shape_foo(%a, %b) :
    (!shape.value_shape, !shape.value_shape) -> !shape.shape
  %1 = shape.with_shape %b, %0 : !shape.value_shape, !shape.shape
  %2 = call @shape_bah(%c, %1) :
    (!shape.value_shape, !shape.value_shape) -> !shape.shape
  return %2 : !shape.shape
}
```

This op need not be a refinement of the shape. In non-error cases the input
ValueShape\'s value and shape are conformant and so too for the output, but
the result may be less specified than `operand`\'s shape as `shape` is
merely used to construct the new ValueShape. If join behavior is desired
then a join op should be used.
"""
function with_shape(
    operand::Value,
    shape::Value;
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand, shape]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "shape.with_shape",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function yield(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "shape.yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # shape
