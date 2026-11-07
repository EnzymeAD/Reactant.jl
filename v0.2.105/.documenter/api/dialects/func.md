


# Func Dialect {#Func-Dialect}

Refer to the [official documentation](https://mlir.llvm.org/docs/Dialects/Func/) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.func.call-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.func.call-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.func.call</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`call`

The `func.call` operation represents a direct call to a function that is within the same symbol scope as the call. The operands and result types of the call must match the specified function type. The callee is encoded as a symbol reference attribute named &quot;callee&quot;.

**Example**

```mlir
%2 = func.call @my_add(%0, %1) : (f32, f32) -> f32
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Func.jl#L61-L74" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.func.call_indirect-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.func.call_indirect-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.func.call_indirect</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`call_indirect`

The `func.call_indirect` operation represents an indirect call to a value of function type. The operands and result types of the call must match the specified function type.

Function values can be created with the [`func.constant` operation](/api/dialects/func#funcconstant-constantop).

**Example**

```mlir
%func = func.constant @my_func : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
%result = func.call_indirect %func(%0, %1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Func.jl#L16-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.func.constant-Tuple{}' href='#Reactant.MLIR.Dialects.func.constant-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.func.constant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`constant`

The `func.constant` operation produces an SSA value from a symbol reference to a `func.func` operation

**Example**

```mlir
// Reference to function @myfn.
%2 = func.constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>

// Equivalent generic forms
%2 = "func.constant"() { value = @myfn } : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
```


MLIR does not allow direct references to functions in SSA operands because the compiler is multithreaded, and disallowing SSA values to directly reference a function simplifies this ([rationale](../Rationale/Rationale.md#multithreading-the-compiler)).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Func.jl#L105-L125" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.func.func_-Tuple{}' href='#Reactant.MLIR.Dialects.func.func_-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.func.func_</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`func_`

Operations within the function cannot implicitly capture values defined outside of the function, i.e. Functions are `IsolatedFromAbove`. All external references must use function arguments or attributes that establish a symbolic connection (e.g. symbols referenced by name via a string attribute like SymbolRefAttr). An external function declaration (used when referring to a function declared in some other module) has no body. While the MLIR textual form provides a nice inline syntax for function arguments, they are internally represented as “block arguments” to the first block in the region.

Only dialect attribute names may be specified in the attribute dictionaries for function arguments, results, or the function itself.

**Example**

```mlir
// External function definitions.
func.func private @abort()
func.func private @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

// A function that returns its argument twice:
func.func @count(%x: i64) -> (i64, i64)
  attributes {fruit = "banana"} {
  return %x, %x: i64, i64
}

// A function with an argument attribute
func.func private @example_fn_arg(%x: i32 {swift.self = unit})

// A function with a result attribute
func.func private @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

// A function with an attribute
func.func private @example_fn_attr() attributes {dialectName.attrName = false}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Func.jl#L145-L183" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.func.return_-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.func.return_-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.func.return_</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`return_`

The `func.return` operation represents a return operation within a function. The operation takes variable number of operands and produces no results. The operand number and types must match the signature of the function that contains the operation.

**Example**

```mlir
func.func @foo() -> (i32, f8) {
  ...
  return %0, %1 : i32, f8
}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Func.jl#L219-L235" target="_blank" rel="noreferrer">source</a></Badge>

</details>

