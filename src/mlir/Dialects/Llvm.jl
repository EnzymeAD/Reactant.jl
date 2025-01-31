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
import ..Dialects: namedattribute, operandsegmentsizes, c
import ...API
using EnumX

"""
`AtomicOrdering`
Atomic ordering for LLVM\'s memory model
"""
@enumx AtomicOrdering not_atomic = 0 unordered = 1 monotonic = 2 acquire = 4 release = 5 acq_rel =
    6 seq_cst = 7

IR.Attribute(e::AtomicOrdering.T) = Int(e)

"""
`AtomicBinOp`
llvm.atomicrmw binary operations
"""
@enumx AtomicBinOp xchg = 0 add = 1 sub = 2 _and = 3 nand = 4 _or = 5 _xor = 6 max = 7 min =
    8 umax = 9 umin = 10 fadd = 11 fsub = 12 fmax = 13 fmin = 14 uinc_wrap = 15 udec_wrap =
    16 usub_cond = 17 usub_sat = 18

IR.Attribute(e::AtomicBinOp.T) = Int(e)

"""
`FastmathFlags`
LLVM fastmath flags
"""
@enumx FastmathFlags none nnan ninf nsz arcp contract afn reassoc fast
FastmathFlagsStorage = [
    "none", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc", "fast"
]

function IR.Attribute(e::FastmathFlags.T)
    return parse(Attribute, "#llvm<fastmath <$(FastmathFlagsStorage[Int(e)+1])>>")
end

"""
`Comdat`
LLVM Comdat Types
"""
@enumx Comdat Any = 0 ExactMatch = 1 Largest = 2 NoDeduplicate = 3 SameSize = 4

IR.Attribute(e::Comdat.T) = Int(e)

"""
`FCmpPredicate`
llvm.fcmp comparison predicate
"""
@enumx FCmpPredicate _false = 0 oeq = 1 ogt = 2 oge = 3 olt = 4 ole = 5 one = 6 ord = 7 ueq =
    8 ugt = 9 uge = 10 ult = 11 ule = 12 une = 13 uno = 14 _true = 15

IR.Attribute(e::FCmpPredicate.T) = Int(e)

"""
`UnnamedAddr`
LLVM GlobalValue UnnamedAddr
"""
@enumx UnnamedAddr None = 0 Local = 1 Global = 2

IR.Attribute(e::UnnamedAddr.T) = Int(e)

"""
`Visibility`
LLVM GlobalValue Visibility
"""
@enumx Visibility Default = 0 Hidden = 1 Protected = 2

IR.Attribute(e::Visibility.T) = Int(e)

"""
`ICmpPredicate`
lvm.icmp comparison predicate
"""
@enumx ICmpPredicate eq = 0 ne = 1 slt = 2 sle = 3 sgt = 4 sge = 5 ult = 6 ule = 7 ugt = 8 uge =
    9

IR.Attribute(e::ICmpPredicate.T) = Int(e)

"""
`AsmDialect`
ATT (0) or Intel (1) asm dialect
"""
@enumx AsmDialect AD_ATT = 0 AD_Intel = 1

IR.Attribute(e::AsmDialect.T) = Int(e)

function ashr(
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    isExact::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function add(
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function addrspacecast(arg::Value; res::IR.Type, location::Location=Location())
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

Creates an SSA value containing a pointer to a global variable or constant
defined by `llvm.mlir.global`. The global value can be defined after its
first referenced. If the global value is a constant, storing into it is not
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
}

// Define the global.
llvm.mlir.global @const(42 : i32) : i32
```
"""
function mlir_addressof(;
    res::IR.Type, global_name::IR.FlatSymbol, location::Location=Location()
)
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

function alloca(
    arraySize::Value;
    res::IR.Type,
    alignment::Union{Int64,Nothing}=nothing,
    elem_type::IR.Type,
    inalloca::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
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
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function cmpxchg(
    ptr::Value,
    cmp::Value,
    val::Value;
    res::Union{Nothing,IR.Type}=nothing,
    success_ordering::AtomicOrdering.T,
    failure_ordering::AtomicOrdering.T,
    syncscope::Union{String,Nothing}=nothing,
    alignment::Union{Int64,Nothing}=nothing,
    weak::Union{Bool,Nothing}=nothing,
    volatile_::Union{Bool,Nothing}=nothing,
    access_groups::Union{Vector{Any},Nothing}=nothing,
    alias_scopes::Union{Vector{Any},Nothing}=nothing,
    noalias_scopes::Union{Vector{Any},Nothing}=nothing,
    tbaa::Union{Vector{Any},Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function atomicrmw(
    ptr::Value,
    val::Value;
    res::Union{Nothing,IR.Type}=nothing,
    bin_op::AtomicBinOp.T,
    ordering::AtomicOrdering.T,
    syncscope::Union{String,Nothing}=nothing,
    alignment::Union{Int64,Nothing}=nothing,
    volatile_::Union{Bool,Nothing}=nothing,
    access_groups::Union{Vector{Any},Nothing}=nothing,
    alias_scopes::Union{Vector{Any},Nothing}=nothing,
    noalias_scopes::Union{Vector{Any},Nothing}=nothing,
    tbaa::Union{Vector{Any},Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function bitcast(arg::Value; res::IR.Type, location::Location=Location())
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

function br(
    destOperands::Vector{Value};
    loop_annotation=nothing,
    dest::Block,
    location::Location=Location(),
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
    results::Union{Nothing,IR.Type}=nothing,
    intrin::String,
    fastmathFlags::Union{FastmathFlags.T,Nothing}=nothing,
    op_bundle_sizes::Vector{Int32},
    op_bundle_tags::Union{Vector{Attribute},Nothing}=nothing,
    location::Location=Location(),
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
    result::Union{Nothing,IR.Type}=nothing,
    var_callee_type=nothing,
    callee::Union{IR.FlatSymbol,Nothing}=nothing,
    fastmathFlags::Union{FastmathFlags.T,Nothing}=nothing,
    branch_weights::Union{Vector{Int32},Nothing}=nothing,
    CConv=nothing,
    TailCallKind=nothing,
    memory_effects=nothing,
    convergent::Union{Bool,Nothing}=nothing,
    no_unwind::Union{Bool,Nothing}=nothing,
    will_return::Union{Bool,Nothing}=nothing,
    op_bundle_sizes::Vector{Int32},
    op_bundle_tags::Union{Vector{Attribute},Nothing}=nothing,
    access_groups::Union{Vector{Any},Nothing}=nothing,
    alias_scopes::Union{Vector{Any},Nothing}=nothing,
    noalias_scopes::Union{Vector{Any},Nothing}=nothing,
    tbaa::Union{Vector{Any},Nothing}=nothing,
    location::Location=Location(),
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
    !isnothing(branch_weights) &&
        push!(attributes, namedattribute("branch_weights", branch_weights))
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
function comdat(; sym_name::String, body::Region, location::Location=Location())
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
function comdat_selector(;
    sym_name::String, comdat::Comdat.T, location::Location=Location()
)
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
    branch_weights::Union{Vector{Int32},Nothing}=nothing,
    loop_annotation=nothing,
    trueDest::Block,
    falseDest::Block,
    location::Location=Location(),
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
strings, and structs. It has a mandatory `value` attribute whose type
depends on the type of the constant value. The type of the constant value
must correspond to the attribute type converted to LLVM IR type.

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
function mlir_constant(; res::IR.Type, value::IR.Attribute, location::Location=Location())
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

function extractelement(
    vector::Value,
    position::Value;
    res::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function extractvalue(
    container::Value; res::IR.Type, position::Vector{Int64}, location::Location=Location()
)
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
    res::Union{Nothing,IR.Type}=nothing,
    fastmathFlags::Union{FastmathFlags.T,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function fcmp(
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    predicate::FCmpPredicate.T,
    fastmathFlags::Union{FastmathFlags.T,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function fdiv(
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    fastmathFlags::Union{FastmathFlags.T,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function fmul(
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    fastmathFlags::Union{FastmathFlags.T,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function fneg(
    operand::Value;
    res::Union{Nothing,IR.Type}=nothing,
    fastmathFlags::Union{FastmathFlags.T,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function fpext(arg::Value; res::IR.Type, location::Location=Location())
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

function fptosi(arg::Value; res::IR.Type, location::Location=Location())
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

function fptoui(arg::Value; res::IR.Type, location::Location=Location())
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

function fptrunc(arg::Value; res::IR.Type, location::Location=Location())
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
    res::Union{Nothing,IR.Type}=nothing,
    fastmathFlags::Union{FastmathFlags.T,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function fsub(
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    fastmathFlags::Union{FastmathFlags.T,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function fence(;
    ordering::AtomicOrdering.T,
    syncscope::Union{String,Nothing}=nothing,
    location::Location=Location(),
)
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

function freeze(
    val::Value; res::Union{Nothing,IR.Type}=nothing, location::Location=Location()
)
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

"""
`getelementptr`

This operation mirrors LLVM IRs \'getelementptr\' operation that is used to
perform pointer arithmetic.

Like in LLVM IR, it is possible to use both constants as well as SSA values
as indices. In the case of indexing within a structure, it is required to
either use constant indices directly, or supply a constant SSA value.

An optional \'inbounds\' attribute specifies the low-level pointer arithmetic
overflow behavior that LLVM uses after lowering the operation to LLVM IR.

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
    rawConstantIndices::Vector{Int32},
    elem_type::IR.Type,
    inbounds::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[res,]
    operands = Value[base, dynamicIndices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("rawConstantIndices", rawConstantIndices),
        namedattribute("elem_type", elem_type),
    ]
    !isnothing(inbounds) && push!(attributes, namedattribute("inbounds", inbounds))

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

Specifies a list of constructor functions and priorities. The functions
referenced by this array will be called in ascending order of priority (i.e.
lowest first) when the module is loaded. The order of functions with the
same priority is not defined. This operation is translated to LLVM\'s
global_ctors global variable. The initializer functions are run at load
time. The `data` field present in LLVM\'s global_ctors variable is not
modeled here.

Examples:

```mlir
llvm.mlir.global_ctors {@ctor}

llvm.func @ctor() {
  ...
  llvm.return
}
```
"""
function mlir_global_ctors(;
    ctors::Vector{IR.FlatSymbol}, priorities::Vector{Int32}, location::Location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("ctors", ctors), namedattribute("priorities", priorities)
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
referenced by this array will be called in descending order of priority (i.e.
highest first) when the module is unloaded. The order of functions with the
same priority is not defined. This operation is translated to LLVM\'s
global_dtors global variable. The `data` field present in LLVM\'s
global_dtors variable is not modeled here.

Examples:

```mlir
llvm.func @dtor() {
  llvm.return
}
llvm.mlir.global_dtors {@dtor}
```
"""
function mlir_global_dtors(;
    dtors::Vector{IR.FlatSymbol}, priorities::Vector{Int32}, location::Location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("dtors", dtors), namedattribute("priorities", priorities)
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
"""
function mlir_global(;
    global_type::IR.Type,
    constant::Union{Bool,Nothing}=nothing,
    sym_name::String,
    linkage,
    dso_local::Union{Bool,Nothing}=nothing,
    thread_local_::Union{Bool,Nothing}=nothing,
    externally_initialized::Union{Bool,Nothing}=nothing,
    value::Union{IR.Attribute,Nothing}=nothing,
    alignment::Union{Int64,Nothing}=nothing,
    addr_space::Union{Int32,Nothing}=nothing,
    unnamed_addr::Union{UnnamedAddr.T,Nothing}=nothing,
    section::Union{String,Nothing}=nothing,
    comdat=nothing,
    dbg_exprs::Union{Vector{Any},Nothing}=nothing,
    visibility_::Union{Visibility.T,Nothing}=nothing,
    initializer::Region,
    location::Location=Location(),
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
    res::Union{Nothing,IR.Type}=nothing,
    predicate::ICmpPredicate.T,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
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
"""
function inline_asm(
    operands::Vector{Value};
    res::Union{Nothing,IR.Type}=nothing,
    asm_string::String,
    constraints::String,
    has_side_effects::Union{Bool,Nothing}=nothing,
    is_align_stack::Union{Bool,Nothing}=nothing,
    asm_dialect::Union{AsmDialect.T,Nothing}=nothing,
    operand_attrs::Union{Vector{Attribute},Nothing}=nothing,
    location::Location=Location(),
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
    res::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function insertvalue(
    container::Value,
    value::Value;
    res::Union{Nothing,IR.Type}=nothing,
    position::Vector{Int64},
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function inttoptr(arg::Value; res::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

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
    result::Union{Nothing,IR.Type}=nothing,
    var_callee_type=nothing,
    callee::Union{IR.FlatSymbol,Nothing}=nothing,
    branch_weights::Union{Vector{Int32},Nothing}=nothing,
    CConv=nothing,
    op_bundle_sizes::Vector{Int32},
    op_bundle_tags::Union{Vector{Attribute},Nothing}=nothing,
    normalDest::Block,
    unwindDest::Block,
    location::Location=Location(),
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
    sym_name::String,
    sym_visibility::Union{String,Nothing}=nothing,
    function_type,
    linkage=nothing,
    dso_local::Union{Bool,Nothing}=nothing,
    CConv=nothing,
    comdat=nothing,
    convergent::Union{Bool,Nothing}=nothing,
    personality::Union{IR.FlatSymbol,Nothing}=nothing,
    garbageCollector::Union{String,Nothing}=nothing,
    passthrough::Union{Vector{Attribute},Nothing}=nothing,
    arg_attrs::Union{Vector{Any},Nothing}=nothing,
    res_attrs::Union{Vector{Any},Nothing}=nothing,
    function_entry_count::Union{Int64,Nothing}=nothing,
    memory_effects=nothing,
    visibility_::Union{Visibility.T,Nothing}=nothing,
    arm_streaming::Union{Bool,Nothing}=nothing,
    arm_locally_streaming::Union{Bool,Nothing}=nothing,
    arm_streaming_compatible::Union{Bool,Nothing}=nothing,
    arm_new_za::Union{Bool,Nothing}=nothing,
    arm_in_za::Union{Bool,Nothing}=nothing,
    arm_out_za::Union{Bool,Nothing}=nothing,
    arm_inout_za::Union{Bool,Nothing}=nothing,
    arm_preserves_za::Union{Bool,Nothing}=nothing,
    section::Union{String,Nothing}=nothing,
    unnamed_addr::Union{UnnamedAddr.T,Nothing}=nothing,
    alignment::Union{Int64,Nothing}=nothing,
    vscale_range=nothing,
    frame_pointer=nothing,
    target_cpu::Union{String,Nothing}=nothing,
    tune_cpu::Union{String,Nothing}=nothing,
    target_features=nothing,
    unsafe_fp_math::Union{Bool,Nothing}=nothing,
    no_infs_fp_math::Union{Bool,Nothing}=nothing,
    no_nans_fp_math::Union{Bool,Nothing}=nothing,
    approx_func_fp_math::Union{Bool,Nothing}=nothing,
    no_signed_zeros_fp_math::Union{Bool,Nothing}=nothing,
    denormal_fp_math::Union{String,Nothing}=nothing,
    denormal_fp_math_f32::Union{String,Nothing}=nothing,
    fp_contract::Union{String,Nothing}=nothing,
    no_inline::Union{Bool,Nothing}=nothing,
    always_inline::Union{Bool,Nothing}=nothing,
    no_unwind::Union{Bool,Nothing}=nothing,
    will_return::Union{Bool,Nothing}=nothing,
    optimize_none::Union{Bool,Nothing}=nothing,
    vec_type_hint=nothing,
    work_group_size_hint::Union{Vector{Int32},Nothing}=nothing,
    reqd_work_group_size::Union{Vector{Int32},Nothing}=nothing,
    intel_reqd_sub_group_size::Union{Int32,Nothing}=nothing,
    body::Region,
    location::Location=Location(),
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
    !isnothing(target_features) &&
        push!(attributes, namedattribute("target_features", target_features))
    !isnothing(unsafe_fp_math) &&
        push!(attributes, namedattribute("unsafe_fp_math", unsafe_fp_math))
    !isnothing(no_infs_fp_math) &&
        push!(attributes, namedattribute("no_infs_fp_math", no_infs_fp_math))
    !isnothing(no_nans_fp_math) &&
        push!(attributes, namedattribute("no_nans_fp_math", no_nans_fp_math))
    !isnothing(approx_func_fp_math) &&
        push!(attributes, namedattribute("approx_func_fp_math", approx_func_fp_math))
    !isnothing(no_signed_zeros_fp_math) && push!(
        attributes, namedattribute("no_signed_zeros_fp_math", no_signed_zeros_fp_math)
    )
    !isnothing(denormal_fp_math) &&
        push!(attributes, namedattribute("denormal_fp_math", denormal_fp_math))
    !isnothing(denormal_fp_math_f32) &&
        push!(attributes, namedattribute("denormal_fp_math_f32", denormal_fp_math_f32))
    !isnothing(fp_contract) && push!(attributes, namedattribute("fp_contract", fp_contract))
    !isnothing(no_inline) && push!(attributes, namedattribute("no_inline", no_inline))
    !isnothing(always_inline) &&
        push!(attributes, namedattribute("always_inline", always_inline))
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
    res::Union{Nothing,IR.Type}=nothing,
    isExact::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function landingpad(
    operand_0::Vector{Value};
    res::IR.Type,
    cleanup::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
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
function linker_options(; options::Vector{String}, location::Location=Location())
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
    alignment::Union{Int64,Nothing}=nothing,
    volatile_::Union{Bool,Nothing}=nothing,
    nontemporal::Union{Bool,Nothing}=nothing,
    invariant::Union{Bool,Nothing}=nothing,
    invariantGroup::Union{Bool,Nothing}=nothing,
    ordering::Union{AtomicOrdering.T,Nothing}=nothing,
    syncscope::Union{String,Nothing}=nothing,
    access_groups::Union{Vector{Any},Nothing}=nothing,
    alias_scopes::Union{Vector{Any},Nothing}=nothing,
    noalias_scopes::Union{Vector{Any},Nothing}=nothing,
    tbaa::Union{Vector{Any},Nothing}=nothing,
    location::Location=Location(),
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

function mul(
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
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
function mlir_none(; res::Union{Nothing,IR.Type}=nothing, location::Location=Location())
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function or(
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    isDisjoint::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
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
function mlir_poison(; res::IR.Type, location::Location=Location())
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

function ptrtoint(arg::Value; res::IR.Type, location::Location=Location())
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

function resume(value::Value; location::Location=Location())
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

function return_(arg::Union{Nothing,Value}=nothing; location::Location=Location())
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
    res::Union{Nothing,IR.Type}=nothing,
    isExact::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function sext(arg::Value; res::IR.Type, location::Location=Location())
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

function sitofp(arg::Value; res::IR.Type, location::Location=Location())
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
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function select(
    condition::Value,
    trueValue::Value,
    falseValue::Value;
    res::Union{Nothing,IR.Type}=nothing,
    fastmathFlags::Union{FastmathFlags.T,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function shl(
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function shufflevector(
    v1::Value, v2::Value; res::IR.Type, mask::Vector{Int32}, location::Location=Location()
)
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
    alignment::Union{Int64,Nothing}=nothing,
    volatile_::Union{Bool,Nothing}=nothing,
    nontemporal::Union{Bool,Nothing}=nothing,
    invariantGroup::Union{Bool,Nothing}=nothing,
    ordering::Union{AtomicOrdering.T,Nothing}=nothing,
    syncscope::Union{String,Nothing}=nothing,
    access_groups::Union{Vector{Any},Nothing}=nothing,
    alias_scopes::Union{Vector{Any},Nothing}=nothing,
    noalias_scopes::Union{Vector{Any},Nothing}=nothing,
    tbaa::Union{Vector{Any},Nothing}=nothing,
    location::Location=Location(),
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
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function switch(
    value::Value,
    defaultOperands::Vector{Value},
    caseOperands::Vector{Value};
    case_values::Union{IR.DenseElements{Int64},Nothing}=nothing,
    case_operand_segments::Vector{Int32},
    branch_weights::Union{Vector{Int32},Nothing}=nothing,
    defaultDestination::Block,
    caseDestinations::Vector{Block},
    location::Location=Location(),
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

function trunc(arg::Value; res::IR.Type, location::Location=Location())
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
    res::Union{Nothing,IR.Type}=nothing,
    isExact::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function uitofp(
    arg::Value;
    res::IR.Type,
    nonNeg::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
)
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
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
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
function mlir_undef(; res::IR.Type, location::Location=Location())
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

function unreachable(; location::Location=Location())
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

function va_arg(arg::Value; res::IR.Type, location::Location=Location())
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
    lhs::Value,
    rhs::Value;
    res::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
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
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function zext(
    arg::Value;
    res::IR.Type,
    nonNeg::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
)
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
function mlir_zero(; res::IR.Type, location::Location=Location())
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
