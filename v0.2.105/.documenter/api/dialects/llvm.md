


# LLVM Dialect {#LLVM-Dialect}

Refer to the [official documentation](https://mlir.llvm.org/docs/Dialects/LLVM/) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.blockaddress-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.blockaddress-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.blockaddress</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`blockaddress`

Creates an SSA value containing a pointer to a basic block. The block address information (function and block) is given by the `BlockAddressAttr` attribute. This operation assumes an existing `llvm.blocktag` operation identifying an existing MLIR block within a function. Example:

```mlir
llvm.mlir.global private @g() : !llvm.ptr {
  %0 = llvm.blockaddress <function = @fn, tag = <id = 0>> : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func @fn() {
  llvm.br ^bb1
^bb1:  // pred: ^bb0
  llvm.blocktag <id = 0>
  llvm.return
}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L381-L402" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.blocktag-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.blocktag-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.blocktag</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`blocktag`

This operation uses a `tag` to uniquely identify an MLIR block in a function. The same tag is used by `llvm.blockaddress` in order to compute the target address.

A given function should have at most one `llvm.blocktag` operation with a given `tag`. This operation cannot be used as a terminator.

**Example**

```mlir
llvm.func @f() -> !llvm.ptr {
  %addr = llvm.blockaddress <function = @f, tag = <id = 1>> : !llvm.ptr
  llvm.br ^bb1
^bb1:
  llvm.blocktag <id = 1>
  llvm.return %addr : !llvm.ptr
}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L422-L443" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.call-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.llvm.call-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.call</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`call`

In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect implements this behavior by providing a variadic `call` operation for 0- and 1-result functions. Even though MLIR supports multi-result functions, LLVM IR dialect disallows them.

The `call` instruction supports both direct and indirect calls. Direct calls start with a function name (`@`-prefixed) and indirect calls start with an SSA value (`%`-prefixed). The direct callee, if present, is stored as a function attribute `callee`. For indirect calls, the callee is of `!llvm.ptr` type and is stored as the first value in `callee_operands`. If and only if the callee is a variadic function, the `var_callee_type` attribute must carry the variadic LLVM function type. The trailing type list contains the optional indirect callee type and the MLIR function type, which differs from the LLVM function type that uses an explicit void type to model functions that do not return a value.

If this operatin has the `no_inline` attribute, then this specific function call  will never be inlined. The opposite behavior will occur if the call has `always_inline`  attribute. The `inline_hint` attribute indicates that it is desirable to inline  this function call.

Examples:

```mlir
// Direct call without arguments and with one result.
%0 = llvm.call @foo() : () -> (f32)

// Direct call with arguments and without a result.
llvm.call @bar(%0) : (f32) -> ()

// Indirect call with an argument and without a result.
%1 = llvm.mlir.addressof @foo : !llvm.ptr
llvm.call %1(%0) : !llvm.ptr, (f32) -> ()

// Direct variadic call.
llvm.call @printf(%0, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

// Indirect variadic call
llvm.call %1(%0) vararg(!llvm.func<void (...)>) : !llvm.ptr, (i32) -> ()
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L532-L575" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.call_intrinsic-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.llvm.call_intrinsic-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.call_intrinsic</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`call_intrinsic`

Call the specified llvm intrinsic. If the intrinsic is overloaded, use the MLIR function type of this op to determine which intrinsic to call.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L486-L491" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.comdat-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.comdat-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.comdat</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`comdat`

Provides access to object file COMDAT section/group functionality.

Examples:

```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L656-L668" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.comdat_selector-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.comdat_selector-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.comdat_selector</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`comdat_selector`

Provides access to object file COMDAT section/group functionality.

Examples:

```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L688-L700" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.dso_local_equivalent-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.dso_local_equivalent-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.dso_local_equivalent</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dso_local_equivalent`

Creates an SSA value containing a pointer to a global value (function or alias to function). It represents a function which is functionally equivalent to a given function, but is always defined in the current linkage unit. The target function may not have `extern_weak` linkage.

Examples:

```mlir
llvm.mlir.global external constant @const() : i64 {
  %0 = llvm.mlir.addressof @const : !llvm.ptr
  %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
  %2 = llvm.dso_local_equivalent @func : !llvm.ptr
  %4 = llvm.ptrtoint %2 : !llvm.ptr to i64
  llvm.return %4 : i64
}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L823-L842" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.func-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.func-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.func</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`func`

MLIR functions are defined by an operation that is not built into the IR itself. The LLVM dialect provides an `llvm.func` operation to define functions compatible with LLVM IR. These functions have LLVM dialect function type but use MLIR syntax to express it. They are required to have exactly one result type. LLVM function operation is intended to capture additional properties of LLVM functions, such as linkage and calling convention, that may be modeled differently by the built-in MLIR function.

```mlir
// The type of @bar is !llvm<"i64 (i64)">
llvm.func @bar(%arg0: i64) -> i64 {
  llvm.return %arg0 : i64
}

// Type type of @foo is !llvm<"void (i64)">
// !llvm.void type is omitted
llvm.func @foo(%arg0: i64) {
  llvm.return
}

// A function with `internal` linkage.
llvm.func internal @internal_func() {
  llvm.return
}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L1785-L1813" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.getelementptr-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.llvm.getelementptr-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.getelementptr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`getelementptr`

This operation mirrors LLVM IRs &#39;getelementptr&#39; operation that is used to perform pointer arithmetic.

Like in LLVM IR, it is possible to use both constants as well as SSA values as indices. In the case of indexing within a structure, it is required to either use constant indices directly, or supply a constant SSA value.

The no-wrap flags can be used to specify the low-level pointer arithmetic overflow behavior that LLVM uses after lowering the operation to LLVM IR. Valid options include &#39;inbounds&#39; (pointer arithmetic must be within object bounds), &#39;nusw&#39; (no unsigned signed wrap), and &#39;nuw&#39; (no unsigned wrap). Note that &#39;inbounds&#39; implies &#39;nusw&#39; which is ensured by the enum definition. The flags can be set individually or in combination.

Examples:

```mlir
// GEP with an SSA value offset
%0 = llvm.getelementptr %1[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32

// GEP with a constant offset and the inbounds attribute set
%0 = llvm.getelementptr inbounds %1[3] : (!llvm.ptr) -> !llvm.ptr, f32

// GEP with constant offsets into a structure
%0 = llvm.getelementptr %1[0, 1]
   : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f32)>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L1215-L1245" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.indirectbr-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.llvm.indirectbr-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.indirectbr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`indirectbr`

Transfer control flow to address in `$addr`. A list of possible target blocks in `$successors` can be provided and maybe used as a hint in LLVM:

```mlir
...
llvm.func @g(...
  %dest = llvm.blockaddress <function = @g, tag = <id = 0>> : !llvm.ptr
  llvm.indirectbr %dest : !llvm.ptr, [
    ^head
  ]
^head:
  llvm.blocktag <id = 0>
  llvm.return %arg0 : i32
  ...
```


It also supports a list of operands that can be passed to a target block:

```mlir
  llvm.indirectbr %dest : !llvm.ptr, [
    ^head(%arg0 : i32),
    ^tail(%arg1, %arg0 : i32, i32)
  ]
^head(%r0 : i32):
  llvm.return %r0 : i32
^tail(%r1 : i32, %r2 : i32):
  ...
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L1543-L1574" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.inline_asm-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.llvm.inline_asm-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.inline_asm</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`inline_asm`

The InlineAsmOp mirrors the underlying LLVM semantics with a notable exception: the embedded `asm_string` is not allowed to define or reference any symbol or any global variable: only the operands of the op may be read, written, or referenced. Attempting to define or reference any symbol or any global behavior is considered undefined behavior at this time.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L1602-L1611" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.linker_options-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.linker_options-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.linker_options</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`linker_options`

Pass the given options to the linker when the resulting object file is linked. This is used extensively on Windows to determine the C runtime that the object files should link against.

Examples:

```mlir
// Link against the MSVC static threaded CRT.
llvm.linker_options ["/DEFAULTLIB:", "libcmt"]

// Link against aarch64 compiler-rt builtins
llvm.linker_options ["-l", "clang_rt.builtins-aarch64"]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L2016-L2031" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.load-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.llvm.load-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.load</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`load`

The `load` operation is used to read from memory. A load may be marked as atomic, volatile, and/or nontemporal, and takes a number of optional attributes that specify aliasing information.

An atomic load only supports a limited set of pointer, integer, and floating point types, and requires an explicit alignment.

Examples:

```mlir
// A volatile load of a float variable.
%0 = llvm.load volatile %ptr : !llvm.ptr -> f32

// A nontemporal load of a float variable.
%0 = llvm.load %ptr {nontemporal} : !llvm.ptr -> f32

// An atomic load of an integer variable.
%0 = llvm.load %ptr atomic monotonic {alignment = 8 : i64}
    : !llvm.ptr -> i64
```


See the following link for more details: https://llvm.org/docs/LangRef.html#load-instruction


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L2051-L2076" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.mlir_addressof-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.mlir_addressof-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.mlir_addressof</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mlir_addressof`

Creates an SSA value containing a pointer to a global value (function, variable or alias). The global value can be defined after its first referenced. If the global value is a constant, storing into it is not allowed.

Examples:

```mlir
func @foo() {
  // Get the address of a global variable.
  %0 = llvm.mlir.addressof @const : !llvm.ptr

  // Use it as a regular pointer.
  %1 = llvm.load %0 : !llvm.ptr -> i32

  // Get the address of a function.
  %2 = llvm.mlir.addressof @foo : !llvm.ptr

  // The function address can be used for indirect calls.
  llvm.call %2() : !llvm.ptr, () -> ()

  // Get the address of an aliased global.
  %3 = llvm.mlir.addressof @const_alias : !llvm.ptr
}

// Define the global.
llvm.mlir.global @const(42 : i32) : i32

// Define an alias.
llvm.mlir.alias @const_alias : i32 {
  %0 = llvm.mlir.addressof @const : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L84-L121" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.mlir_alias-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.mlir_alias-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.mlir_alias</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mlir_alias`

`llvm.mlir.alias` is a top level operation that defines a global alias for global variables and functions. The operation is always initialized by using a initializer region which could be a direct map to another global value or contain some address computation on top of it.

It uses a symbol for its value, which will be uniqued by the module with respect to other symbols in it.

Similarly to functions and globals, they can also have a linkage attribute. This attribute is placed between `llvm.mlir.alias` and the symbol name. If the attribute is omitted, `external` linkage is assumed by default.

Examples:

```mlir
// Global alias use @-identifiers.
llvm.mlir.alias external @foo_alias {addr_space = 0 : i32} : !llvm.ptr {
  %0 = llvm.mlir.addressof @some_function : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// More complex initialization.
llvm.mlir.alias linkonce_odr hidden @glob
{addr_space = 0 : i32, dso_local} : !llvm.array<32 x i32> {
  %0 = llvm.mlir.constant(1234 : i64) : i64
  %1 = llvm.mlir.addressof @glob.private : !llvm.ptr
  %2 = llvm.ptrtoint %1 : !llvm.ptr to i64
  %3 = llvm.add %2, %0 : i64
  %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
  llvm.return %4 : !llvm.ptr
}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L141-L176" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.mlir_constant-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.mlir_constant-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.mlir_constant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mlir_constant`

Unlike LLVM IR, MLIR does not have first-class constant values. Therefore, all constants must be created as SSA values before being used in other operations. `llvm.mlir.constant` creates such values for scalars, vectors, strings, and structs. It has a mandatory `value` attribute whose type depends on the type of the constant value. The type of the constant value must correspond to the attribute type converted to LLVM IR type.

When creating constant scalars, the `value` attribute must be either an integer attribute or a floating point attribute. The type of the attribute may be omitted for `i64` and `f64` types that are implied.

When creating constant vectors, the `value` attribute must be either an array attribute, a dense attribute, or a sparse attribute that contains integers or floats. The number of elements in the result vector must match the number of elements in the attribute.

When creating constant strings, the `value` attribute must be a string attribute. The type of the constant must be an LLVM array of `i8`s, and the length of the array must match the length of the attribute.

When creating constant structs, the `value` attribute must be an array attribute that contains integers or floats. The type of the constant must be an LLVM struct type. The number of fields in the struct must match the number of elements in the attribute, and the type of each LLVM struct field must correspond to the type of the corresponding attribute element converted to LLVM IR.

Examples:

```mlir
// Integer constant, internal i32 is mandatory
%0 = llvm.mlir.constant(42 : i32) : i32

// It's okay to omit i64.
%1 = llvm.mlir.constant(42) : i64

// Floating point constant.
%2 = llvm.mlir.constant(42.0 : f32) : f32

// Splat dense vector constant.
%3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L758-L803" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.mlir_global-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.mlir_global-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.mlir_global</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mlir_global`

Since MLIR allows for arbitrary operations to be present at the top level, global variables are defined using the `llvm.mlir.global` operation. Both global constants and variables can be defined, and the value may also be initialized in both cases.

There are two forms of initialization syntax. Simple constants that can be represented as MLIR attributes can be given in-line:

```mlir
llvm.mlir.global @variable(32.0 : f32) : f32
```


This initialization and type syntax is similar to `llvm.mlir.constant` and may use two types: one for MLIR attribute and another for the LLVM value. These types must be compatible.

More complex constants that cannot be represented as MLIR attributes can be given in an initializer region:

```mlir
// This global is initialized with the equivalent of:
//   i32* getelementptr (i32* @g2, i32 2)
llvm.mlir.global constant @int_gep() : !llvm.ptr {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr, i32) -> !llvm.ptr, i32
  // The initializer region must end with `llvm.return`.
  llvm.return %2 : !llvm.ptr
}
```


Only one of the initializer attribute or initializer region may be provided.

`llvm.mlir.global` must appear at top-level of the enclosing module. It uses an @-identifier for its value, which will be uniqued by the module with respect to other @-identifiers in it.

Examples:

```mlir
// Global values use @-identifiers.
llvm.mlir.global constant @cst(42 : i32) : i32

// Non-constant values must also be initialized.
llvm.mlir.global @variable(32.0 : f32) : f32

// Strings are expected to be of wrapped LLVM i8 array type and do not
// automatically include the trailing zero.
llvm.mlir.global @string("abc") : !llvm.array<3 x i8>

// For strings globals, the trailing type may be omitted.
llvm.mlir.global constant @no_trailing_type("foo bar")

// A complex initializer is constructed with an initializer region.
llvm.mlir.global constant @int_gep() : !llvm.ptr {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr, i32) -> !llvm.ptr, i32
  llvm.return %2 : !llvm.ptr
}
```


Similarly to functions, globals have a linkage attribute. In the custom syntax, this attribute is placed between `llvm.mlir.global` and the optional `constant` keyword. If the attribute is omitted, `external` linkage is assumed by default.

Examples:

```mlir
// A constant with internal linkage will not participate in linking.
llvm.mlir.global internal constant @cst(42 : i32) : i32

// By default, "external" linkage is assumed and the global participates in
// symbol resolution at link-time.
llvm.mlir.global @glob(0 : f32) : f32

// Alignment is optional
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) : !llvm.array<8 x f32>
```


Like global variables in LLVM IR, globals can have an (optional) alignment attribute using keyword `alignment`. The integer value of the alignment must be a positive integer that is a power of 2.

Examples:

```mlir
// Alignment is optional
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) { alignment = 32 : i64 } : !llvm.array<8 x f32>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L1364-L1460" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.mlir_global_ctors-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.mlir_global_ctors-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.mlir_global_ctors</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mlir_global_ctors`

Specifies a list of constructor functions, priorities, and associated data. The functions referenced by this array will be called in ascending order of priority (i.e. lowest first) when the module is loaded. The order of functions with the same priority is not defined. This operation is translated to LLVM&#39;s global_ctors global variable. The initializer functions are run at load time. However, if the associated data is not `#llvm.zero`, functions only run if the data is not discarded.

Examples:

```mlir
llvm.func @ctor() {
  ...
  llvm.return
}
llvm.mlir.global_ctors ctors = [@ctor], priorities = [0],
                               data = [#llvm.zero]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L1275-L1296" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.mlir_global_dtors-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.mlir_global_dtors-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.mlir_global_dtors</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mlir_global_dtors`

Specifies a list of destructor functions and priorities. The functions referenced by this array will be called in descending order of priority (i.e. highest first) when the module is unloaded. The order of functions with the same priority is not defined. This operation is translated to LLVM&#39;s global_dtors global variable. The destruction functions are run at load time. However, if the associated data is not `#llvm.zero`, functions only run if the data is not discarded.

Examples:

```mlir
llvm.func @dtor() {
  llvm.return
}
llvm.mlir.global_dtors dtors = [@dtor], priorities = [0],
                               data = [#llvm.zero]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L1320-L1340" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.mlir_none-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.mlir_none-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.mlir_none</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mlir_none`

Unlike LLVM IR, MLIR does not have first-class token values. They must be explicitly created as SSA values using `llvm.mlir.none`. This operation has no operands or attributes, and returns a none token value of a wrapped LLVM IR pointer type.

Examples:

```mlir
%0 = llvm.mlir.none : !llvm.token
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L2185-L2198" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.mlir_poison-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.mlir_poison-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.mlir_poison</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mlir_poison`

Unlike LLVM IR, MLIR does not have first-class poison values. Such values must be created as SSA values using `llvm.mlir.poison`. This operation has no operands or attributes. It creates a poison value of the specified LLVM IR dialect type.

**Example**

```mlir
// Create a poison value for a structure with a 32-bit integer followed
// by a float.
%0 = llvm.mlir.poison : !llvm.struct<(i32, f32)>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L2246-L2261" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.mlir_undef-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.mlir_undef-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.mlir_undef</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mlir_undef`

Unlike LLVM IR, MLIR does not have first-class undefined values. Such values must be created as SSA values using `llvm.mlir.undef`. This operation has no operands or attributes. It creates an undefined value of the specified LLVM IR dialect type.

**Example**

```mlir
// Create a structure with a 32-bit integer followed by a float.
%0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L2716-L2730" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.mlir_zero-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.mlir_zero-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.mlir_zero</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mlir_zero`

Unlike LLVM IR, MLIR does not have first-class zero-initialized values. Such values must be created as SSA values using `llvm.mlir.zero`. This operation has no operands or attributes. It creates a zero-initialized value of the specified LLVM IR dialect type.

**Example**

```mlir
// Create a zero-initialized value for a structure with a 32-bit integer
// followed by a float.
%0 = llvm.mlir.zero : !llvm.struct<(i32, f32)>
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L2830-L2845" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.module_flags-Tuple{}' href='#Reactant.MLIR.Dialects.llvm.module_flags-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.module_flags</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`module_flags`

Represents the equivalent in MLIR for LLVM&#39;s `llvm.module.flags` metadata, which requires a list of metadata triplets. Each triplet entry is described by a `ModuleFlagAttr`.

**Example**

```mlir
llvm.module.flags [
  #llvm.mlir.module_flag<error, "wchar_size", 4>,
  #llvm.mlir.module_flag<max, "PIC Level", 2>
]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L2129-L2143" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.llvm.store-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.llvm.store-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.llvm.store</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`store`

The `store` operation is used to write to memory. A store may be marked as atomic, volatile, and/or nontemporal, and takes a number of optional attributes that specify aliasing information.

An atomic store only supports a limited set of pointer, integer, and floating point types, and requires an explicit alignment.

Examples:

```mlir
// A volatile store of a float variable.
llvm.store volatile %val, %ptr : f32, !llvm.ptr

// A nontemporal store of a float variable.
llvm.store %val, %ptr {nontemporal} : f32, !llvm.ptr

// An atomic store of an integer variable.
llvm.store %val, %ptr atomic monotonic {alignment = 8 : i64}
    : i64, !llvm.ptr
```


See the following link for more details: https://llvm.org/docs/LangRef.html#store-instruction


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Llvm.jl#L2496-L2521" target="_blank" rel="noreferrer">source</a></Badge>

</details>

