module llvm
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

function ashr(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    isExact=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(isExact) && push!(attributes, namedattribute("isExact", isExact))

    return create_operation(
        "llvm.ashr",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function add(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.add",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function addrspacecast(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.addrspacecast",
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
`mlir_addressof`

Creates an SSA value containing a pointer to a global value (function,
variable or alias). The global value can be defined after its first
referenced. If the global value is a constant, storing into it is not
allowed.

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
"""
function mlir_addressof(; res::IR.Type, global_name, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("global_name", global_name),]

    return create_operation(
        "llvm.mlir.addressof",
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
`mlir_alias`

`llvm.mlir.alias` is a top level operation that defines a global alias for
global variables and functions. The operation is always initialized by
using a initializer region which could be a direct map to another global
value or contain some address computation on top of it.

It uses a symbol for its value, which will be uniqued by the module
with respect to other symbols in it.

Similarly to functions and globals, they can also have a linkage attribute.
This attribute is placed between `llvm.mlir.alias` and the symbol name. If
the attribute is omitted, `external` linkage is assumed by default.

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
"""
function mlir_alias(;
    alias_type,
    sym_name,
    linkage,
    dso_local=nothing,
    thread_local_=nothing,
    unnamed_addr=nothing,
    visibility_=nothing,
    initializer::Region,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initializer,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("alias_type", alias_type),
        namedattribute("sym_name", sym_name),
        namedattribute("linkage", linkage),
    ]
    !isnothing(dso_local) && push!(attributes, namedattribute("dso_local", dso_local))
    !isnothing(thread_local_) &&
        push!(attributes, namedattribute("thread_local_", thread_local_))
    !isnothing(unnamed_addr) &&
        push!(attributes, namedattribute("unnamed_addr", unnamed_addr))
    !isnothing(visibility_) && push!(attributes, namedattribute("visibility_", visibility_))

    return create_operation(
        "llvm.mlir.alias",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function alloca(
    arraySize::Value;
    res::IR.Type,
    alignment=nothing,
    elem_type,
    inalloca=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[res,]
    operands = Value[arraySize,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("elem_type", elem_type),]
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(inalloca) && push!(attributes, namedattribute("inalloca", inalloca))

    return create_operation(
        "llvm.alloca",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function and(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.and",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function cmpxchg(
    ptr::Value,
    cmp::Value,
    val::Value;
    res=nothing::Union{Nothing,IR.Type},
    success_ordering,
    failure_ordering,
    syncscope=nothing,
    alignment=nothing,
    weak=nothing,
    volatile_=nothing,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[ptr, cmp, val]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("success_ordering", success_ordering),
        namedattribute("failure_ordering", failure_ordering),
    ]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(syncscope) && push!(attributes, namedattribute("syncscope", syncscope))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(weak) && push!(attributes, namedattribute("weak", weak))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return create_operation(
        "llvm.cmpxchg",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function atomicrmw(
    ptr::Value,
    val::Value;
    res=nothing::Union{Nothing,IR.Type},
    bin_op,
    ordering,
    syncscope=nothing,
    alignment=nothing,
    volatile_=nothing,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[ptr, val]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("bin_op", bin_op), namedattribute("ordering", ordering)
    ]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(syncscope) && push!(attributes, namedattribute("syncscope", syncscope))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return create_operation(
        "llvm.atomicrmw",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function bitcast(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.bitcast",
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
`blockaddress`

Creates an SSA value containing a pointer to a basic block. The block
address information (function and block) is given by the `BlockAddressAttr`
attribute. This operation assumes an existing `llvm.blocktag` operation
identifying an existing MLIR block within a function. Example:

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
"""
function blockaddress(; res::IR.Type, block_addr, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("block_addr", block_addr),]

    return create_operation(
        "llvm.blockaddress",
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
`blocktag`

This operation uses a `tag` to uniquely identify an MLIR block in a
function. The same tag is used by `llvm.blockaddress` in order to compute
the target address.

A given function should have at most one `llvm.blocktag` operation with a
given `tag`. This operation cannot be used as a terminator.

# Example

```mlir
llvm.func @f() -> !llvm.ptr {
  %addr = llvm.blockaddress <function = @f, tag = <id = 1>> : !llvm.ptr
  llvm.br ^bb1
^bb1:
  llvm.blocktag <id = 1>
  llvm.return %addr : !llvm.ptr
}
```
"""
function blocktag(; tag, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("tag", tag),]

    return create_operation(
        "llvm.blocktag",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function br(
    destOperands::Vector{Value}; loop_annotation=nothing, dest::Block, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[destOperands...,]
    owned_regions = Region[]
    successors = Block[dest,]
    attributes = NamedAttribute[]
    !isnothing(loop_annotation) &&
        push!(attributes, namedattribute("loop_annotation", loop_annotation))

    return create_operation(
        "llvm.br",
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
`call_intrinsic`

Call the specified llvm intrinsic. If the intrinsic is overloaded, use
the MLIR function type of this op to determine which intrinsic to call.
"""
function call_intrinsic(
    args::Vector{Value},
    op_bundle_operands::Vector{Value};
    results=nothing::Union{Nothing,IR.Type},
    intrin,
    fastmathFlags=nothing,
    op_bundle_sizes,
    op_bundle_tags=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[args..., op_bundle_operands...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("intrin", intrin), namedattribute("op_bundle_sizes", op_bundle_sizes)
    ]
    push!(attributes, operandsegmentsizes([length(args), length(op_bundle_operands)]))
    !isnothing(results) && push!(op_ty_results, results)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    !isnothing(op_bundle_tags) &&
        push!(attributes, namedattribute("op_bundle_tags", op_bundle_tags))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))

    return create_operation(
        "llvm.call_intrinsic",
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

In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect
implements this behavior by providing a variadic `call` operation for 0- and
1-result functions. Even though MLIR supports multi-result functions, LLVM
IR dialect disallows them.

The `call` instruction supports both direct and indirect calls. Direct calls
start with a function name (`@`-prefixed) and indirect calls start with an
SSA value (`%`-prefixed). The direct callee, if present, is stored as a
function attribute `callee`. For indirect calls, the callee is of `!llvm.ptr` type
and is stored as the first value in `callee_operands`. If and only if the
callee is a variadic function, the `var_callee_type` attribute must carry
the variadic LLVM function type. The trailing type list contains the
optional indirect callee type and the MLIR function type, which differs from
the LLVM function type that uses an explicit void type to model functions
that do not return a value.

If this operatin has the `no_inline` attribute, then this specific function call
will never be inlined. The opposite behavior will occur if the call has `always_inline`
attribute. The `inline_hint` attribute indicates that it is desirable to inline
this function call.

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
"""
function call(
    callee_operands::Vector{Value},
    op_bundle_operands::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    var_callee_type=nothing,
    callee=nothing,
    fastmathFlags=nothing,
    CConv=nothing,
    TailCallKind=nothing,
    memory_effects=nothing,
    convergent=nothing,
    no_unwind=nothing,
    will_return=nothing,
    op_bundle_sizes,
    op_bundle_tags=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
    no_inline=nothing,
    always_inline=nothing,
    inline_hint=nothing,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[callee_operands..., op_bundle_operands...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("op_bundle_sizes", op_bundle_sizes),]
    push!(
        attributes,
        operandsegmentsizes([length(callee_operands), length(op_bundle_operands)]),
    )
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(var_callee_type) &&
        push!(attributes, namedattribute("var_callee_type", var_callee_type))
    !isnothing(callee) && push!(attributes, namedattribute("callee", callee))
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    !isnothing(CConv) && push!(attributes, namedattribute("CConv", CConv))
    !isnothing(TailCallKind) &&
        push!(attributes, namedattribute("TailCallKind", TailCallKind))
    !isnothing(memory_effects) &&
        push!(attributes, namedattribute("memory_effects", memory_effects))
    !isnothing(convergent) && push!(attributes, namedattribute("convergent", convergent))
    !isnothing(no_unwind) && push!(attributes, namedattribute("no_unwind", no_unwind))
    !isnothing(will_return) && push!(attributes, namedattribute("will_return", will_return))
    !isnothing(op_bundle_tags) &&
        push!(attributes, namedattribute("op_bundle_tags", op_bundle_tags))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(no_inline) && push!(attributes, namedattribute("no_inline", no_inline))
    !isnothing(always_inline) &&
        push!(attributes, namedattribute("always_inline", always_inline))
    !isnothing(inline_hint) && push!(attributes, namedattribute("inline_hint", inline_hint))
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return create_operation(
        "llvm.call",
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
`comdat`

Provides access to object file COMDAT section/group functionality.

Examples:
```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```
"""
function comdat(; sym_name, body::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name),]

    return create_operation(
        "llvm.comdat",
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
`comdat_selector`

Provides access to object file COMDAT section/group functionality.

Examples:
```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```
"""
function comdat_selector(; sym_name, comdat, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("comdat", comdat)
    ]

    return create_operation(
        "llvm.comdat_selector",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cond_br(
    condition::Value,
    trueDestOperands::Vector{Value},
    falseDestOperands::Vector{Value};
    branch_weights=nothing,
    loop_annotation=nothing,
    trueDest::Block,
    falseDest::Block,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[condition, trueDestOperands..., falseDestOperands...]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest]
    attributes = NamedAttribute[]
    push!(
        attributes,
        operandsegmentsizes([1, length(trueDestOperands), length(falseDestOperands)]),
    )
    !isnothing(branch_weights) &&
        push!(attributes, namedattribute("branch_weights", branch_weights))
    !isnothing(loop_annotation) &&
        push!(attributes, namedattribute("loop_annotation", loop_annotation))

    return create_operation(
        "llvm.cond_br",
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
`mlir_constant`

Unlike LLVM IR, MLIR does not have first-class constant values. Therefore,
all constants must be created as SSA values before being used in other
operations. `llvm.mlir.constant` creates such values for scalars, vectors,
strings, structs, and array of structs. It has a mandatory `value` attribute
whose type depends on the type of the constant value. The type of the constant
value must correspond to the attribute type converted to LLVM IR type.

When creating constant scalars, the `value` attribute must be either an
integer attribute or a floating point attribute. The type of the attribute
may be omitted for `i64` and `f64` types that are implied.

When creating constant vectors, the `value` attribute must be either an
array attribute, a dense attribute, or a sparse attribute that contains
integers or floats. The number of elements in the result vector must match
the number of elements in the attribute.

When creating constant strings, the `value` attribute must be a string
attribute. The type of the constant must be an LLVM array of `i8`s, and the
length of the array must match the length of the attribute.

When creating constant structs, the `value` attribute must be an array
attribute that contains integers or floats. The type of the constant must be
an LLVM struct type. The number of fields in the struct must match the
number of elements in the attribute, and the type of each LLVM struct field
must correspond to the type of the corresponding attribute element converted
to LLVM IR.

When creating an array of structs, the `value` attribute must be an array
attribute, itself containing zero, or undef, or array attributes for each
potential nested array type, and the elements of the leaf array attributes
for must match the struct element types or be zero or undef attributes.

Examples:

```mlir
// Integer constant, internal i32 is mandatory
%0 = llvm.mlir.constant(42 : i32) : i32

// It\'s okay to omit i64.
%1 = llvm.mlir.constant(42) : i64

// Floating point constant.
%2 = llvm.mlir.constant(42.0 : f32) : f32

// Splat dense vector constant.
%3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
```
"""
function mlir_constant(; res::IR.Type, value, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]

    return create_operation(
        "llvm.mlir.constant",
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
`dso_local_equivalent`

Creates an SSA value containing a pointer to a global value (function or
alias to function). It represents a function which is functionally
equivalent to a given function, but is always defined in the current
linkage unit. The target function may not have `extern_weak` linkage.

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
"""
function dso_local_equivalent(; res::IR.Type, function_name, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("function_name", function_name),]

    return create_operation(
        "llvm.dso_local_equivalent",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function extractelement(
    vector::Value, position::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[vector, position]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.extractelement",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function extractvalue(container::Value; res::IR.Type, position, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[container,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("position", position),]

    return create_operation(
        "llvm.extractvalue",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fadd(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return create_operation(
        "llvm.fadd",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function fcmp(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    predicate,
    fastmathFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate),]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return create_operation(
        "llvm.fcmp",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function fdiv(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return create_operation(
        "llvm.fdiv",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function fmul(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return create_operation(
        "llvm.fmul",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function fneg(
    operand::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return create_operation(
        "llvm.fneg",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function fpext(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.fpext",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fptosi(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.fptosi",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fptoui(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.fptoui",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fptrunc(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.fptrunc",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function frem(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return create_operation(
        "llvm.frem",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function fsub(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return create_operation(
        "llvm.fsub",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function fence(; ordering, syncscope=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("ordering", ordering),]
    !isnothing(syncscope) && push!(attributes, namedattribute("syncscope", syncscope))

    return create_operation(
        "llvm.fence",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function freeze(val::Value; res=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.freeze",
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
`getelementptr`

This operation mirrors LLVM IRs \'getelementptr\' operation that is used to
perform pointer arithmetic.

Like in LLVM IR, it is possible to use both constants as well as SSA values
as indices. In the case of indexing within a structure, it is required to
either use constant indices directly, or supply a constant SSA value.

The no-wrap flags can be used to specify the low-level pointer arithmetic
overflow behavior that LLVM uses after lowering the operation to LLVM IR.
Valid options include \'inbounds\' (pointer arithmetic must be within object
bounds), \'nusw\' (no unsigned signed wrap), and \'nuw\' (no unsigned wrap).
Note that \'inbounds\' implies \'nusw\' which is ensured by the enum
definition. The flags can be set individually or in combination.

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
"""
function getelementptr(
    base::Value,
    dynamicIndices::Vector{Value};
    res::IR.Type,
    rawConstantIndices,
    elem_type,
    location=Location(),
)
    op_ty_results = IR.Type[res,]
    operands = Value[base, dynamicIndices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("rawConstantIndices", rawConstantIndices),
        namedattribute("elem_type", elem_type),
    ]

    return create_operation(
        "llvm.getelementptr",
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
`mlir_global_ctors`

Specifies a list of constructor functions, priorities, and associated data.
The functions referenced by this array will be called in ascending order
of priority (i.e. lowest first) when the module is loaded. The order of
functions with the same priority is not defined. This operation is
translated to LLVM\'s global_ctors global variable. The initializer
functions are run at load time. However, if the associated data is not
`#llvm.zero`, functions only run if the data is not discarded.

Examples:

```mlir
llvm.func @ctor() {
  ...
  llvm.return
}
llvm.mlir.global_ctors ctors = [@ctor], priorities = [0],
                               data = [#llvm.zero]
```
"""
function mlir_global_ctors(; ctors, priorities, data, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("ctors", ctors),
        namedattribute("priorities", priorities),
        namedattribute("data", data),
    ]

    return create_operation(
        "llvm.mlir.global_ctors",
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
`mlir_global_dtors`

Specifies a list of destructor functions and priorities. The functions
referenced by this array will be called in descending order of priority
(i.e. highest first) when the module is unloaded. The order of functions
with the same priority is not defined. This operation is translated to
LLVM\'s global_dtors global variable. The destruction functions are run at
load time. However, if the associated data is not `#llvm.zero`, functions
only run if the data is not discarded.

Examples:

```mlir
llvm.func @dtor() {
  llvm.return
}
llvm.mlir.global_dtors dtors = [@dtor], priorities = [0],
                               data = [#llvm.zero]
```
"""
function mlir_global_dtors(; dtors, priorities, data, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("dtors", dtors),
        namedattribute("priorities", priorities),
        namedattribute("data", data),
    ]

    return create_operation(
        "llvm.mlir.global_dtors",
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
`mlir_global`

Since MLIR allows for arbitrary operations to be present at the top level,
global variables are defined using the `llvm.mlir.global` operation. Both
global constants and variables can be defined, and the value may also be
initialized in both cases.

There are two forms of initialization syntax. Simple constants that can be
represented as MLIR attributes can be given in-line:

```mlir
llvm.mlir.global @variable(32.0 : f32) : f32
```

This initialization and type syntax is similar to `llvm.mlir.constant` and
may use two types: one for MLIR attribute and another for the LLVM value.
These types must be compatible.

More complex constants that cannot be represented as MLIR attributes can be
given in an initializer region:

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

`llvm.mlir.global` must appear at top-level of the enclosing module. It uses
an @-identifier for its value, which will be uniqued by the module with
respect to other @-identifiers in it.

Examples:

```mlir
// Global values use @-identifiers.
llvm.mlir.global constant @cst(42 : i32) : i32

// Non-constant values must also be initialized.
llvm.mlir.global @variable(32.0 : f32) : f32

// Strings are expected to be of wrapped LLVM i8 array type and do not
// automatically include the trailing zero.
llvm.mlir.global @string(\"abc\") : !llvm.array<3 x i8>

// For strings globals, the trailing type may be omitted.
llvm.mlir.global constant @no_trailing_type(\"foo bar\")

// A complex initializer is constructed with an initializer region.
llvm.mlir.global constant @int_gep() : !llvm.ptr {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr, i32) -> !llvm.ptr, i32
  llvm.return %2 : !llvm.ptr
}
```

Similarly to functions, globals have a linkage attribute. In the custom
syntax, this attribute is placed between `llvm.mlir.global` and the optional
`constant` keyword. If the attribute is omitted, `external` linkage is
assumed by default.

Examples:

```mlir
// A constant with internal linkage will not participate in linking.
llvm.mlir.global internal constant @cst(42 : i32) : i32

// By default, \"external\" linkage is assumed and the global participates in
// symbol resolution at link-time.
llvm.mlir.global @glob(0 : f32) : f32

// Alignment is optional
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) : !llvm.array<8 x f32>
```

Like global variables in LLVM IR, globals can have an (optional)
alignment attribute using keyword `alignment`. The integer value of the
alignment must be a positive integer that is a power of 2.

Examples:

```mlir
// Alignment is optional
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) { alignment = 32 : i64 } : !llvm.array<8 x f32>
```

The `target_specific_attrs` attribute provides a mechanism to preserve
target-specific LLVM IR attributes that are not explicitly modeled in the
LLVM dialect.

The attribute is an array containing either string attributes or
two-element array attributes of strings. The value of a standalone string
attribute is interpreted as the name of an LLVM IR attribute on the global.
A two-element array is interpreted as a key-value pair.

# Example

```mlir
llvm.mlir.global external @example() {
  target_specific_attrs = [\"value-less-attr\", [\"int-attr\", \"4\"], [\"string-attr\", \"string\"]]} : f64
```
"""
function mlir_global(;
    global_type,
    constant=nothing,
    sym_name,
    linkage,
    dso_local=nothing,
    thread_local_=nothing,
    externally_initialized=nothing,
    value=nothing,
    alignment=nothing,
    addr_space=nothing,
    unnamed_addr=nothing,
    section=nothing,
    comdat=nothing,
    dbg_exprs=nothing,
    visibility_=nothing,
    target_specific_attrs=nothing,
    initializer::Region,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initializer,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("global_type", global_type),
        namedattribute("sym_name", sym_name),
        namedattribute("linkage", linkage),
    ]
    !isnothing(constant) && push!(attributes, namedattribute("constant", constant))
    !isnothing(dso_local) && push!(attributes, namedattribute("dso_local", dso_local))
    !isnothing(thread_local_) &&
        push!(attributes, namedattribute("thread_local_", thread_local_))
    !isnothing(externally_initialized) &&
        push!(attributes, namedattribute("externally_initialized", externally_initialized))
    !isnothing(value) && push!(attributes, namedattribute("value", value))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(addr_space) && push!(attributes, namedattribute("addr_space", addr_space))
    !isnothing(unnamed_addr) &&
        push!(attributes, namedattribute("unnamed_addr", unnamed_addr))
    !isnothing(section) && push!(attributes, namedattribute("section", section))
    !isnothing(comdat) && push!(attributes, namedattribute("comdat", comdat))
    !isnothing(dbg_exprs) && push!(attributes, namedattribute("dbg_exprs", dbg_exprs))
    !isnothing(visibility_) && push!(attributes, namedattribute("visibility_", visibility_))
    !isnothing(target_specific_attrs) &&
        push!(attributes, namedattribute("target_specific_attrs", target_specific_attrs))

    return create_operation(
        "llvm.mlir.global",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function icmp(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    predicate,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate),]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.icmp",
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
`mlir_ifunc`

`llvm.mlir.ifunc` is a top level operation that defines a global ifunc.
It defines a new symbol and takes a symbol refering to a resolver function.
IFuncs can be called as regular functions. The function type is the same
as the IFuncType. The symbol is resolved at runtime by calling a resolver
function.

Examples:

```mlir
// IFuncs resolve a symbol at runtime using a resovler function.
llvm.mlir.ifunc external @foo: !llvm.func<f32 (i64)>, !llvm.ptr @resolver

llvm.func @foo_1(i64) -> f32
llvm.func @foo_2(i64) -> f32

llvm.func @resolve_foo() -> !llvm.ptr attributes {
  %0 = llvm.mlir.addressof @foo_2 : !llvm.ptr
  %1 = llvm.mlir.addressof @foo_1 : !llvm.ptr

  // ... Logic selecting from foo_{1, 2}

  // Return function pointer to the selected function
  llvm.return %7 : !llvm.ptr
}

llvm.func @use_foo() {
  // IFuncs are called as regular functions
  %res = llvm.call @foo(%value) : i64 -> f32
}
```
"""
function mlir_ifunc(;
    sym_name,
    i_func_type,
    resolver,
    resolver_type,
    linkage,
    dso_local=nothing,
    address_space=nothing,
    unnamed_addr=nothing,
    visibility_=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("sym_name", sym_name),
        namedattribute("i_func_type", i_func_type),
        namedattribute("resolver", resolver),
        namedattribute("resolver_type", resolver_type),
        namedattribute("linkage", linkage),
    ]
    !isnothing(dso_local) && push!(attributes, namedattribute("dso_local", dso_local))
    !isnothing(address_space) &&
        push!(attributes, namedattribute("address_space", address_space))
    !isnothing(unnamed_addr) &&
        push!(attributes, namedattribute("unnamed_addr", unnamed_addr))
    !isnothing(visibility_) && push!(attributes, namedattribute("visibility_", visibility_))

    return create_operation(
        "llvm.mlir.ifunc",
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
`indirectbr`

Transfer control flow to address in `\$addr`. A list of possible target
blocks in `\$successors` can be provided and maybe used as a hint in LLVM:

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
"""
function indirectbr(
    addr::Value,
    succOperands::Vector{Value};
    indbr_operand_segments,
    successors::Vector{Block},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[addr, succOperands...]
    owned_regions = Region[]
    successors = Block[successors...,]
    attributes = NamedAttribute[namedattribute(
        "indbr_operand_segments", indbr_operand_segments
    ),]

    return create_operation(
        "llvm.indirectbr",
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
`inline_asm`

The InlineAsmOp mirrors the underlying LLVM semantics with a notable
exception: the embedded `asm_string` is not allowed to define or reference
any symbol or any global variable: only the operands of the op may be read,
written, or referenced.
Attempting to define or reference any symbol or any global behavior is
considered undefined behavior at this time.
If `tail_call_kind` is used, the operation behaves like the specified
tail call kind. The `musttail` kind it\'s not available for this operation,
since it isn\'t supported by LLVM\'s inline asm.
"""
function inline_asm(
    operands::Vector{Value};
    res=nothing::Union{Nothing,IR.Type},
    asm_string,
    constraints,
    has_side_effects=nothing,
    is_align_stack=nothing,
    tail_call_kind=nothing,
    asm_dialect=nothing,
    operand_attrs=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("asm_string", asm_string), namedattribute("constraints", constraints)
    ]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(has_side_effects) &&
        push!(attributes, namedattribute("has_side_effects", has_side_effects))
    !isnothing(is_align_stack) &&
        push!(attributes, namedattribute("is_align_stack", is_align_stack))
    !isnothing(tail_call_kind) &&
        push!(attributes, namedattribute("tail_call_kind", tail_call_kind))
    !isnothing(asm_dialect) && push!(attributes, namedattribute("asm_dialect", asm_dialect))
    !isnothing(operand_attrs) &&
        push!(attributes, namedattribute("operand_attrs", operand_attrs))

    return create_operation(
        "llvm.inline_asm",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function insertelement(
    vector::Value,
    value::Value,
    position::Value;
    res=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[vector, value, position]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.insertelement",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function insertvalue(
    container::Value,
    value::Value;
    res=nothing::Union{Nothing,IR.Type},
    position,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[container, value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("position", position),]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.insertvalue",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function inttoptr(arg::Value; res::IR.Type, dereferenceable=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dereferenceable) &&
        push!(attributes, namedattribute("dereferenceable", dereferenceable))

    return create_operation(
        "llvm.inttoptr",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function invoke(
    callee_operands::Vector{Value},
    normalDestOperands::Vector{Value},
    unwindDestOperands::Vector{Value},
    op_bundle_operands::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    var_callee_type=nothing,
    callee=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
    branch_weights=nothing,
    CConv=nothing,
    op_bundle_sizes,
    op_bundle_tags=nothing,
    normalDest::Block,
    unwindDest::Block,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[
        callee_operands...,
        normalDestOperands...,
        unwindDestOperands...,
        op_bundle_operands...,
    ]
    owned_regions = Region[]
    successors = Block[normalDest, unwindDest]
    attributes = NamedAttribute[namedattribute("op_bundle_sizes", op_bundle_sizes),]
    push!(
        attributes,
        operandsegmentsizes([
            length(callee_operands),
            length(normalDestOperands),
            length(unwindDestOperands),
            length(op_bundle_operands),
        ]),
    )
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(var_callee_type) &&
        push!(attributes, namedattribute("var_callee_type", var_callee_type))
    !isnothing(callee) && push!(attributes, namedattribute("callee", callee))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(branch_weights) &&
        push!(attributes, namedattribute("branch_weights", branch_weights))
    !isnothing(CConv) && push!(attributes, namedattribute("CConv", CConv))
    !isnothing(op_bundle_tags) &&
        push!(attributes, namedattribute("op_bundle_tags", op_bundle_tags))

    return create_operation(
        "llvm.invoke",
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
`func`

MLIR functions are defined by an operation that is not built into the IR
itself. The LLVM dialect provides an `llvm.func` operation to define
functions compatible with LLVM IR. These functions have LLVM dialect
function type but use MLIR syntax to express it. They are required to have
exactly one result type. LLVM function operation is intended to capture
additional properties of LLVM functions, such as linkage and calling
convention, that may be modeled differently by the built-in MLIR function.

```mlir
// The type of @bar is !llvm<\"i64 (i64)\">
llvm.func @bar(%arg0: i64) -> i64 {
  llvm.return %arg0 : i64
}

// Type type of @foo is !llvm<\"void (i64)\">
// !llvm.void type is omitted
llvm.func @foo(%arg0: i64) {
  llvm.return
}

// A function with `internal` linkage.
llvm.func internal @internal_func() {
  llvm.return
}
```
"""
function func(;
    sym_name,
    sym_visibility=nothing,
    function_type,
    linkage=nothing,
    dso_local=nothing,
    CConv=nothing,
    comdat=nothing,
    convergent=nothing,
    personality=nothing,
    garbageCollector=nothing,
    passthrough=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
    function_entry_count=nothing,
    memory_effects=nothing,
    visibility_=nothing,
    arm_streaming=nothing,
    arm_locally_streaming=nothing,
    arm_streaming_compatible=nothing,
    arm_new_za=nothing,
    arm_in_za=nothing,
    arm_out_za=nothing,
    arm_inout_za=nothing,
    arm_preserves_za=nothing,
    section=nothing,
    unnamed_addr=nothing,
    alignment=nothing,
    vscale_range=nothing,
    frame_pointer=nothing,
    target_cpu=nothing,
    tune_cpu=nothing,
    reciprocal_estimates=nothing,
    prefer_vector_width=nothing,
    target_features=nothing,
    no_infs_fp_math=nothing,
    no_nans_fp_math=nothing,
    no_signed_zeros_fp_math=nothing,
    denormal_fp_math=nothing,
    denormal_fp_math_f32=nothing,
    fp_contract=nothing,
    instrument_function_entry=nothing,
    instrument_function_exit=nothing,
    no_inline=nothing,
    always_inline=nothing,
    inline_hint=nothing,
    no_unwind=nothing,
    will_return=nothing,
    optimize_none=nothing,
    vec_type_hint=nothing,
    work_group_size_hint=nothing,
    reqd_work_group_size=nothing,
    intel_reqd_sub_group_size=nothing,
    uwtable_kind=nothing,
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
    !isnothing(linkage) && push!(attributes, namedattribute("linkage", linkage))
    !isnothing(dso_local) && push!(attributes, namedattribute("dso_local", dso_local))
    !isnothing(CConv) && push!(attributes, namedattribute("CConv", CConv))
    !isnothing(comdat) && push!(attributes, namedattribute("comdat", comdat))
    !isnothing(convergent) && push!(attributes, namedattribute("convergent", convergent))
    !isnothing(personality) && push!(attributes, namedattribute("personality", personality))
    !isnothing(garbageCollector) &&
        push!(attributes, namedattribute("garbageCollector", garbageCollector))
    !isnothing(passthrough) && push!(attributes, namedattribute("passthrough", passthrough))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(function_entry_count) &&
        push!(attributes, namedattribute("function_entry_count", function_entry_count))
    !isnothing(memory_effects) &&
        push!(attributes, namedattribute("memory_effects", memory_effects))
    !isnothing(visibility_) && push!(attributes, namedattribute("visibility_", visibility_))
    !isnothing(arm_streaming) &&
        push!(attributes, namedattribute("arm_streaming", arm_streaming))
    !isnothing(arm_locally_streaming) &&
        push!(attributes, namedattribute("arm_locally_streaming", arm_locally_streaming))
    !isnothing(arm_streaming_compatible) && push!(
        attributes, namedattribute("arm_streaming_compatible", arm_streaming_compatible)
    )
    !isnothing(arm_new_za) && push!(attributes, namedattribute("arm_new_za", arm_new_za))
    !isnothing(arm_in_za) && push!(attributes, namedattribute("arm_in_za", arm_in_za))
    !isnothing(arm_out_za) && push!(attributes, namedattribute("arm_out_za", arm_out_za))
    !isnothing(arm_inout_za) &&
        push!(attributes, namedattribute("arm_inout_za", arm_inout_za))
    !isnothing(arm_preserves_za) &&
        push!(attributes, namedattribute("arm_preserves_za", arm_preserves_za))
    !isnothing(section) && push!(attributes, namedattribute("section", section))
    !isnothing(unnamed_addr) &&
        push!(attributes, namedattribute("unnamed_addr", unnamed_addr))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(vscale_range) &&
        push!(attributes, namedattribute("vscale_range", vscale_range))
    !isnothing(frame_pointer) &&
        push!(attributes, namedattribute("frame_pointer", frame_pointer))
    !isnothing(target_cpu) && push!(attributes, namedattribute("target_cpu", target_cpu))
    !isnothing(tune_cpu) && push!(attributes, namedattribute("tune_cpu", tune_cpu))
    !isnothing(reciprocal_estimates) &&
        push!(attributes, namedattribute("reciprocal_estimates", reciprocal_estimates))
    !isnothing(prefer_vector_width) &&
        push!(attributes, namedattribute("prefer_vector_width", prefer_vector_width))
    !isnothing(target_features) &&
        push!(attributes, namedattribute("target_features", target_features))
    !isnothing(no_infs_fp_math) &&
        push!(attributes, namedattribute("no_infs_fp_math", no_infs_fp_math))
    !isnothing(no_nans_fp_math) &&
        push!(attributes, namedattribute("no_nans_fp_math", no_nans_fp_math))
    !isnothing(no_signed_zeros_fp_math) && push!(
        attributes, namedattribute("no_signed_zeros_fp_math", no_signed_zeros_fp_math)
    )
    !isnothing(denormal_fp_math) &&
        push!(attributes, namedattribute("denormal_fp_math", denormal_fp_math))
    !isnothing(denormal_fp_math_f32) &&
        push!(attributes, namedattribute("denormal_fp_math_f32", denormal_fp_math_f32))
    !isnothing(fp_contract) && push!(attributes, namedattribute("fp_contract", fp_contract))
    !isnothing(instrument_function_entry) && push!(
        attributes,
        namedattribute("instrument_function_entry", instrument_function_entry),
    )
    !isnothing(instrument_function_exit) && push!(
        attributes, namedattribute("instrument_function_exit", instrument_function_exit)
    )
    !isnothing(no_inline) && push!(attributes, namedattribute("no_inline", no_inline))
    !isnothing(always_inline) &&
        push!(attributes, namedattribute("always_inline", always_inline))
    !isnothing(inline_hint) && push!(attributes, namedattribute("inline_hint", inline_hint))
    !isnothing(no_unwind) && push!(attributes, namedattribute("no_unwind", no_unwind))
    !isnothing(will_return) && push!(attributes, namedattribute("will_return", will_return))
    !isnothing(optimize_none) &&
        push!(attributes, namedattribute("optimize_none", optimize_none))
    !isnothing(vec_type_hint) &&
        push!(attributes, namedattribute("vec_type_hint", vec_type_hint))
    !isnothing(work_group_size_hint) &&
        push!(attributes, namedattribute("work_group_size_hint", work_group_size_hint))
    !isnothing(reqd_work_group_size) &&
        push!(attributes, namedattribute("reqd_work_group_size", reqd_work_group_size))
    !isnothing(intel_reqd_sub_group_size) && push!(
        attributes,
        namedattribute("intel_reqd_sub_group_size", intel_reqd_sub_group_size),
    )
    !isnothing(uwtable_kind) &&
        push!(attributes, namedattribute("uwtable_kind", uwtable_kind))

    return create_operation(
        "llvm.func",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function lshr(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    isExact=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(isExact) && push!(attributes, namedattribute("isExact", isExact))

    return create_operation(
        "llvm.lshr",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function landingpad(
    operand_0::Vector{Value}; res::IR.Type, cleanup=nothing, location=Location()
)
    op_ty_results = IR.Type[res,]
    operands = Value[operand_0...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(cleanup) && push!(attributes, namedattribute("cleanup", cleanup))

    return create_operation(
        "llvm.landingpad",
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
`linker_options`

Pass the given options to the linker when the resulting object file is linked.
This is used extensively on Windows to determine the C runtime that the object
files should link against.

Examples:
```mlir
// Link against the MSVC static threaded CRT.
llvm.linker_options [\"/DEFAULTLIB:\", \"libcmt\"]

// Link against aarch64 compiler-rt builtins
llvm.linker_options [\"-l\", \"clang_rt.builtins-aarch64\"]
```
"""
function linker_options(; options, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("options", options),]

    return create_operation(
        "llvm.linker_options",
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
`load`

The `load` operation is used to read from memory. A load may be marked as
atomic, volatile, and/or nontemporal, and takes a number of optional
attributes that specify aliasing information.

An atomic load only supports a limited set of pointer, integer, and
floating point types, and requires an explicit alignment.

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

See the following link for more details:
https://llvm.org/docs/LangRef.html#load-instruction
"""
function load(
    addr::Value;
    res::IR.Type,
    alignment=nothing,
    volatile_=nothing,
    nontemporal=nothing,
    invariant=nothing,
    invariantGroup=nothing,
    ordering=nothing,
    syncscope=nothing,
    dereferenceable=nothing,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[res,]
    operands = Value[addr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))
    !isnothing(invariant) && push!(attributes, namedattribute("invariant", invariant))
    !isnothing(invariantGroup) &&
        push!(attributes, namedattribute("invariantGroup", invariantGroup))
    !isnothing(ordering) && push!(attributes, namedattribute("ordering", ordering))
    !isnothing(syncscope) && push!(attributes, namedattribute("syncscope", syncscope))
    !isnothing(dereferenceable) &&
        push!(attributes, namedattribute("dereferenceable", dereferenceable))
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return create_operation(
        "llvm.load",
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
`module_flags`

Represents the equivalent in MLIR for LLVM\'s `llvm.module.flags` metadata,
which requires a list of metadata triplets. Each triplet entry is described
by a `ModuleFlagAttr`.

# Example
```mlir
llvm.module.flags [
  #llvm.mlir.module_flag<error, \"wchar_size\", 4>,
  #llvm.mlir.module_flag<max, \"PIC Level\", 2>
]
```
"""
function module_flags(; flags, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("flags", flags),]

    return create_operation(
        "llvm.module_flags",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mul(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.mul",
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
`mlir_none`

Unlike LLVM IR, MLIR does not have first-class token values. They must be
explicitly created as SSA values using `llvm.mlir.none`. This operation has
no operands or attributes, and returns a none token value of a wrapped LLVM IR
pointer type.

Examples:

```mlir
%0 = llvm.mlir.none : !llvm.token
```
"""
function mlir_none(; res=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.mlir.none",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function or(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    isDisjoint=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(isDisjoint) && push!(attributes, namedattribute("isDisjoint", isDisjoint))

    return create_operation(
        "llvm.or",
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
`mlir_poison`

Unlike LLVM IR, MLIR does not have first-class poison values. Such values
must be created as SSA values using `llvm.mlir.poison`. This operation has
no operands or attributes. It creates a poison value of the specified LLVM
IR dialect type.

# Example

```mlir
// Create a poison value for a structure with a 32-bit integer followed
// by a float.
%0 = llvm.mlir.poison : !llvm.struct<(i32, f32)>
```
"""
function mlir_poison(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.mlir.poison",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function ptrtoint(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.ptrtoint",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function resume(value::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.resume",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function return_(arg=nothing::Union{Nothing,Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(arg) && push!(operands, arg)

    return create_operation(
        "llvm.return",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sdiv(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    isExact=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(isExact) && push!(attributes, namedattribute("isExact", isExact))

    return create_operation(
        "llvm.sdiv",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function sext(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.sext",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sitofp(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.sitofp",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function srem(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.srem",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function select(
    condition::Value,
    trueValue::Value,
    falseValue::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[condition, trueValue, falseValue]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return create_operation(
        "llvm.select",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function shl(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.shl",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function shufflevector(v1::Value, v2::Value; res::IR.Type, mask, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[v1, v2]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("mask", mask),]

    return create_operation(
        "llvm.shufflevector",
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
`store`

The `store` operation is used to write to memory. A store may be marked as
atomic, volatile, and/or nontemporal, and takes a number of optional
attributes that specify aliasing information.

An atomic store only supports a limited set of pointer, integer, and
floating point types, and requires an explicit alignment.

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

See the following link for more details:
https://llvm.org/docs/LangRef.html#store-instruction
"""
function store(
    value::Value,
    addr::Value;
    alignment=nothing,
    volatile_=nothing,
    nontemporal=nothing,
    invariantGroup=nothing,
    ordering=nothing,
    syncscope=nothing,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[value, addr]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))
    !isnothing(invariantGroup) &&
        push!(attributes, namedattribute("invariantGroup", invariantGroup))
    !isnothing(ordering) && push!(attributes, namedattribute("ordering", ordering))
    !isnothing(syncscope) && push!(attributes, namedattribute("syncscope", syncscope))
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return create_operation(
        "llvm.store",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sub(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.sub",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function switch(
    value::Value,
    defaultOperands::Vector{Value},
    caseOperands::Vector{Value};
    case_values=nothing,
    case_operand_segments,
    branch_weights=nothing,
    defaultDestination::Block,
    caseDestinations::Vector{Block},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[value, defaultOperands..., caseOperands...]
    owned_regions = Region[]
    successors = Block[defaultDestination, caseDestinations...]
    attributes = NamedAttribute[namedattribute(
        "case_operand_segments", case_operand_segments
    ),]
    push!(
        attributes, operandsegmentsizes([1, length(defaultOperands), length(caseOperands)])
    )
    !isnothing(case_values) && push!(attributes, namedattribute("case_values", case_values))
    !isnothing(branch_weights) &&
        push!(attributes, namedattribute("branch_weights", branch_weights))

    return create_operation(
        "llvm.switch",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function trunc(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.trunc",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function udiv(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    isExact=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(isExact) && push!(attributes, namedattribute("isExact", isExact))

    return create_operation(
        "llvm.udiv",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function uitofp(arg::Value; res::IR.Type, nonNeg=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(nonNeg) && push!(attributes, namedattribute("nonNeg", nonNeg))

    return create_operation(
        "llvm.uitofp",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function urem(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.urem",
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
`mlir_undef`

Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
must be created as SSA values using `llvm.mlir.undef`. This operation has no
operands or attributes. It creates an undefined value of the specified LLVM
IR dialect type.

# Example

```mlir
// Create a structure with a 32-bit integer followed by a float.
%0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>
```
"""
function mlir_undef(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.mlir.undef",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function unreachable(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.unreachable",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function va_arg(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.va_arg",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function xor(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "llvm.xor",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function zext(arg::Value; res::IR.Type, nonNeg=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(nonNeg) && push!(attributes, namedattribute("nonNeg", nonNeg))

    return create_operation(
        "llvm.zext",
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
`mlir_zero`

Unlike LLVM IR, MLIR does not have first-class zero-initialized values.
Such values must be created as SSA values using `llvm.mlir.zero`. This
operation has no operands or attributes. It creates a zero-initialized
value of the specified LLVM IR dialect type.

# Example

```mlir
// Create a zero-initialized value for a structure with a 32-bit integer
// followed by a float.
%0 = llvm.mlir.zero : !llvm.struct<(i32, f32)>
```
"""
function mlir_zero(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "llvm.mlir.zero",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # llvm
