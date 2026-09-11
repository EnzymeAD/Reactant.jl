


# Builtin Dialect {#Builtin-Dialect}

Refer to the [official documentation](https://mlir.llvm.org/docs/Dialects/Builtin/) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.builtin.module_-Tuple{}' href='#Reactant.MLIR.Dialects.builtin.module_-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.builtin.module_</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`module_`

A `module` represents a top-level container operation. It contains a single [graph region](../LangRef.md#control-flow-and-ssacfg-regions) containing a single block which can contain any operations and does not have a terminator. Operations within this region cannot implicitly capture values defined outside the module, i.e. Modules are [IsolatedFromAbove](../Traits.md#isolatedfromabove). Modules have an optional [symbol name](../SymbolsAndSymbolTables.md) which can be used to refer to them in operations.

**Example**

```mlir
module {
  func.func @foo()
}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Builtin.jl#L16-L34" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.builtin.unrealized_conversion_cast-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.builtin.unrealized_conversion_cast-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.builtin.unrealized_conversion_cast</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`unrealized_conversion_cast`

An `unrealized_conversion_cast` operation represents an unrealized conversion from one set of types to another, that is used to enable the inter-mixing of different type systems. This operation should not be attributed any special representational or execution semantics, and is generally only intended to be used to satisfy the temporary intermixing of type systems during the conversion of one type system to another.

This operation may produce results of arity 1-N, and accept as input operands of arity 0-N.

**Example**

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



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Builtin.jl#L59-L92" target="_blank" rel="noreferrer">source</a></Badge>

</details>

