module func
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
`call_indirect`

The `func.call_indirect` operation represents an indirect call to a value
of function type. The operands and result types of the call must match the
specified function type.

Function values can be created with the
[`func.constant` operation](#funcconstant-constantop).

# Example

```mlir
%func = func.constant @my_func : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
%result = func.call_indirect %func(%0, %1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
```
"""
function call_indirect(
    callee::Value,
    callee_operands::Vector{Value};
    results::Vector{IR.Type},
    arg_attrs=nothing,
    res_attrs=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[callee, callee_operands...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))

    return create_operation(
        "func.call_indirect",
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
`call`

The `func.call` operation represents a direct call to a function that is
within the same symbol scope as the call. The operands and result types of
the call must match the specified function type. The callee is encoded as a
symbol reference attribute named \"callee\".

# Example

```mlir
%2 = func.call @my_add(%0, %1) : (f32, f32) -> f32
```
"""
function call(
    operands::Vector{Value};
    result_0::Vector{IR.Type},
    callee,
    arg_attrs=nothing,
    res_attrs=nothing,
    no_inline=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result_0...,]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("callee", callee),]
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(no_inline) && push!(attributes, namedattribute("no_inline", no_inline))

    return create_operation(
        "func.call",
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
`constant`

The `func.constant` operation produces an SSA value from a symbol reference
to a `func.func` operation

# Example

```mlir
// Reference to function @myfn.
%2 = func.constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>

// Equivalent generic forms
%2 = \"func.constant\"() { value = @myfn } : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
```

MLIR does not allow direct references to functions in SSA operands because
the compiler is multithreaded, and disallowing SSA values to directly
reference a function simplifies this
([rationale](../Rationale/Rationale.md#multithreading-the-compiler)).
"""
function constant(; result_0::IR.Type, value, location=Location())
    op_ty_results = IR.Type[result_0,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]

    return create_operation(
        "func.constant",
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
`func_`

Operations within the function cannot implicitly capture values defined
outside of the function, i.e. Functions are `IsolatedFromAbove`. All
external references must use function arguments or attributes that establish
a symbolic connection (e.g. symbols referenced by name via a string
attribute like SymbolRefAttr). An external function declaration (used when
referring to a function declared in some other module) has no body. While
the MLIR textual form provides a nice inline syntax for function arguments,
they are internally represented as “block arguments” to the first block in
the region.

Only dialect attribute names may be specified in the attribute dictionaries
for function arguments, results, or the function itself.

# Example

```mlir
// External function definitions.
func.func private @abort()
func.func private @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

// A function that returns its argument twice:
func.func @count(%x: i64) -> (i64, i64)
  attributes {fruit = \"banana\"} {
  return %x, %x: i64, i64
}

// A function with an argument attribute
func.func private @example_fn_arg(%x: i32 {swift.self = unit})

// A function with a result attribute
func.func private @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

// A function with an attribute
func.func private @example_fn_attr() attributes {dialectName.attrName = false}
```
"""
function func_(;
    sym_name,
    function_type,
    sym_visibility=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
    no_inline=nothing,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("function_type", function_type)
    ]
    !isnothing(sym_visibility) &&
        push!(attributes, namedattribute("sym_visibility", sym_visibility))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(no_inline) && push!(attributes, namedattribute("no_inline", no_inline))

    return create_operation(
        "func.func",
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

The `func.return` operation represents a return operation within a function.
The operation takes variable number of operands and produces no results.
The operand number and types must match the signature of the function
that contains the operation.

# Example

```mlir
func.func @foo() -> (i32, f8) {
  ...
  return %0, %1 : i32, f8
}
```
"""
function return_(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "func.return",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # func
