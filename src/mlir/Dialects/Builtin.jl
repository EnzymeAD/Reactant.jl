module builtin
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
`module_`

A `module` represents a top-level container operation. It contains a single
[graph region](../LangRef.md#control-flow-and-ssacfg-regions) containing a single block
which can contain any operations and does not have a terminator. Operations
within this region cannot implicitly capture values defined outside the module,
i.e. Modules are [IsolatedFromAbove](../Traits#isolatedfromabove). Modules have
an optional [symbol name](../SymbolsAndSymbolTables.md) which can be used to refer
to them in operations.

# Example

```mlir
module {
  func.func @foo()
}
```
"""
function module_(;
    sym_name=nothing, sym_visibility=nothing, bodyRegion::Region, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[bodyRegion,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(sym_name) && push!(attributes, namedattribute("sym_name", sym_name))
    !isnothing(sym_visibility) &&
        push!(attributes, namedattribute("sym_visibility", sym_visibility))

    return create_operation(
        "builtin.module",
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
`unrealized_conversion_cast`

An `unrealized_conversion_cast` operation represents an unrealized
conversion from one set of types to another, that is used to enable the
inter-mixing of different type systems. This operation should not be
attributed any special representational or execution semantics, and is
generally only intended to be used to satisfy the temporary intermixing of
type systems during the conversion of one type system to another.

This operation may produce results of arity 1-N, and accept as input
operands of arity 0-N.

# Example

```mlir
// An unrealized 0-1 conversion. These types of conversions are useful in
// cases where a type is removed from the type system, but not all uses have
// been converted. For example, imagine we have a tuple type that is
// expanded to its element types. If only some uses of an empty tuple type
// instance are converted we still need an instance of the tuple type, but
// have no inputs to the unrealized conversion.
%result = unrealized_conversion_cast to !bar.tuple_type<>

// An unrealized 1-1 conversion.
%result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type

// An unrealized 1-N conversion.
%results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type<!foo.type, !foo.type> to !foo.type, !foo.type

// An unrealized N-1 conversion.
%result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type<!foo.type, !foo.type>
```
"""
function unrealized_conversion_cast(
    inputs::Vector{Value}; outputs::Vector{IR.Type}, location=Location()
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "builtin.unrealized_conversion_cast",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # builtin
