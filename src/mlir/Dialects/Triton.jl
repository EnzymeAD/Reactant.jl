module tt
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
`call`

The `tt.call` operation represents a direct call to a function that is
within the same symbol scope as the call. The operands and result types of
the call must match the specified function type. The callee is encoded as a
symbol reference attribute named \"callee\".

# Example

```mlir
%2 = tt.call @my_add(%0, %1) : (f32, f32) -> f32
```
"""
function call(
    operands::Vector{Value}; result_0::Vector{IR.Type}, callee, location=Location()
)
    op_ty_results = IR.Type[result_0...,]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("callee", callee),]

    return create_operation(
        "tt.call",
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
tt.func @abort()
tt.func @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

// A function that returns its argument twice:
tt.func @count(%x: i64) -> (i64, i64)
  attributes {fruit: \"banana\"} {
  return %x, %x: i64, i64
}

// A function with an argument attribute
tt.func @example_fn_arg(%x: i32 {swift.self = unit})

// A function with a result attribute
tt.func @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

// A function with an attribute
tt.func @example_fn_attr() attributes {dialectName.attrName = false}
```
"""
function func(;
    sym_name,
    function_type,
    sym_visibility=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
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

    return create_operation(
        "tt.func",
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
`reinterpret_tensor_descriptor`

This Op exists to help the transition from untyped raw TMA objects to typed Tensor descriptor objects.
Ideally, we can remove this once the APIs are fully fleshed out.
"""
function reinterpret_tensor_descriptor(rawDesc::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[rawDesc,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.reinterpret_tensor_descriptor",
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

The `tt.return` operation represents a return operation within a function.
The operation takes variable number of operands and produces no results.
The operand number and types must match the signature of the function
that contains the operation.

# Example

```mlir
tt.func @foo() : (i32, f8) {
  ...
  tt.return %0, %1 : i32, f8
}
```
"""
function return_(srcs::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[srcs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.return",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function addptr(ptr::Value, offset::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[ptr, offset]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.addptr",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function advance(ptr::Value, offsets::Vector{Value}; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[ptr, offsets...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.advance",
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
`assert`

`tt.assert` takes a condition tensor and a message string.
If the condition is false, the message is printed, and the program is aborted.
"""
function assert(condition::Value; message, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[condition,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("message", message),]

    return create_operation(
        "tt.assert",
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
`atomic_cas`

compare \$cmp with data \$old at location \$ptr,

if \$old == \$cmp, store \$val to \$ptr,

else store \$old to \$ptr,

return \$old
"""
function atomic_cas(
    ptr::Value, cmp::Value, val::Value; result::IR.Type, sem, scope, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[ptr, cmp, val]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sem", sem), namedattribute("scope", scope)]

    return create_operation(
        "tt.atomic_cas",
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
`atomic_rmw`

load data at \$ptr, do \$rmw_op with \$val, and store result to \$ptr.

return old value at \$ptr
"""
function atomic_rmw(
    ptr::Value,
    val::Value,
    mask=nothing::Union{Nothing,Value};
    result::IR.Type,
    atomic_rmw_op,
    sem,
    scope,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[ptr, val]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("atomic_rmw_op", atomic_rmw_op),
        namedattribute("sem", sem),
        namedattribute("scope", scope),
    ]
    !isnothing(mask) && push!(operands, mask)

    return create_operation(
        "tt.atomic_rmw",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function bitcast(src::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.bitcast",
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

For a given tensor, broadcast changes one or more dimensions with size 1
to a new size, e.g. tensor<1x32x1xf32> -> tensor<2x32x4xf32>.  You cannot
change the size of a non-1 dimension.
"""
function broadcast(src::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.broadcast",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cat(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.cat",
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
`clampf`

Clamp operation for floating point types.

The operation takes three arguments: x, min, and max. It returns a tensor of the same shape as x with its values clamped to the range [min, max].
"""
function clampf(
    x::Value,
    min::Value,
    max::Value;
    result=nothing::Union{Nothing,IR.Type},
    propagateNan,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[x, min, max]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("propagateNan", propagateNan),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tt.clampf",
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
`dot`

\$d = matrix_multiply(\$a, \$b) + \$c. \$inputPrecision describes how to exercise the TC
when the inputs are f32. It can be one of: tf32, tf32x3, ieee.
tf32: use TC with tf32 ops.
tf32x3: implement the 3xTF32 trick. For more info see the pass in F32DotTC.cpp
ieee: don\'t use TC, implement dot in software.
If the GPU does not have Tensor cores or the inputs are not f32, this flag is ignored.
"""
function dot(
    a::Value,
    b::Value,
    c::Value;
    d=nothing::Union{Nothing,IR.Type},
    inputPrecision=nothing,
    maxNumImpreciseAcc=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[a, b, c]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(d) && push!(op_ty_results, d)
    !isnothing(inputPrecision) &&
        push!(attributes, namedattribute("inputPrecision", inputPrecision))
    !isnothing(maxNumImpreciseAcc) &&
        push!(attributes, namedattribute("maxNumImpreciseAcc", maxNumImpreciseAcc))

    return create_operation(
        "tt.dot",
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
`dot_scaled`

\$d = matrix_multiply(scale(\$lhs, \$lhs_scale), scale(\$rhs, \$rhs_scale)) + \$c.
Where scale(x, s) is a function that applies the scale per block following microscaling spec.
"""
function dot_scaled(
    lhs::Value,
    rhs::Value,
    c::Value,
    lhs_scale=nothing::Union{Nothing,Value};
    rhs_scale=nothing::Union{Nothing,Value},
    d::IR.Type,
    lhs_type,
    rhs_type,
    location=Location(),
)
    op_ty_results = IR.Type[d,]
    operands = Value[lhs, rhs, c]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("lhs_type", lhs_type), namedattribute("rhs_type", rhs_type)
    ]
    !isnothing(lhs_scale) && push!(operands, lhs_scale)
    !isnothing(rhs_scale) && push!(operands, rhs_scale)
    push!(attributes, operandsegmentsizes([
        1,
        1,
        1,
        if (lhs_scale == nothing)
            0
        elseif 1(rhs_scale == nothing)
            0
        else
            1
        end,
    ]))

    return create_operation(
        "tt.dot_scaled",
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
`elementwise_inline_asm`

Runs an inline asm block to generate one or more tensors.

The asm block is given `packed_element` elements at a time.  Exactly which
elems it receives is unspecified.
"""
function elementwise_inline_asm(
    args::Vector{Value};
    result::Vector{IR.Type},
    asm_string,
    constraints,
    pure,
    packed_element,
    location=Location(),
)
    op_ty_results = IR.Type[result...,]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("asm_string", asm_string),
        namedattribute("constraints", constraints),
        namedattribute("pure", pure),
        namedattribute("packed_element", packed_element),
    ]

    return create_operation(
        "tt.elementwise_inline_asm",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function expand_dims(
    src::Value; result=nothing::Union{Nothing,IR.Type}, axis, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tt.expand_dims",
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
`experimental_descriptor_load`

This operation will be lowered to Nvidia TMA load operation on targets supporting it.
`desc` is a tensor descriptor object.
The destination tensor type and shape must match the descriptor otherwise the result is undefined.

This is an escape hatch and is only there for testing/experimenting.
This op will be removed in the future.
"""
function experimental_descriptor_load(
    desc::Value,
    indices::Vector{Value};
    result::IR.Type,
    cache=nothing,
    evict=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[desc, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(cache) && push!(attributes, namedattribute("cache", cache))
    !isnothing(evict) && push!(attributes, namedattribute("evict", evict))

    return create_operation(
        "tt.experimental_descriptor_load",
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
`experimental_descriptor_store`

This operation will be lowered to Nvidia TMA store operation on targets supporting it.
`desc` is a tensor descriptor object.
The shape and types of `src` must match the descriptor otherwise the result is undefined.

This is an escape hatch and is only there for testing/experimenting.
This op will be removed in the future.
"""
function experimental_descriptor_store(
    desc::Value, src::Value, indices::Vector{Value}; location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[desc, src, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.experimental_descriptor_store",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function experimental_tensormap_create(
    desc_ptr::Value,
    global_address::Value,
    box_dim::Vector{Value},
    global_dim::Vector{Value},
    global_stride::Vector{Value},
    element_stride::Vector{Value};
    elem_type,
    interleave_layout,
    swizzle_mode,
    fill_mode,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[
        desc_ptr,
        global_address,
        box_dim...,
        global_dim...,
        global_stride...,
        element_stride...,
    ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("elem_type", elem_type),
        namedattribute("interleave_layout", interleave_layout),
        namedattribute("swizzle_mode", swizzle_mode),
        namedattribute("fill_mode", fill_mode),
    ]
    push!(
        attributes,
        operandsegmentsizes([
            1,
            1,
            length(box_dim),
            length(global_dim),
            length(global_stride),
            length(element_stride),
        ]),
    )

    return create_operation(
        "tt.experimental_tensormap_create",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function experimental_tensormap_fenceproxy_acquire(desc_ptr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[desc_ptr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.experimental_tensormap_fenceproxy_acquire",
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
`extern_elementwise`

call an external function \$symbol implemented in \$libpath/\$libname with \$args
return \$libpath/\$libname:\$symbol(\$args...)
"""
function extern_elementwise(
    srcs::Vector{Value};
    result::IR.Type,
    libname,
    libpath,
    symbol,
    pure,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[srcs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("libname", libname),
        namedattribute("libpath", libpath),
        namedattribute("symbol", symbol),
        namedattribute("pure", pure),
    ]

    return create_operation(
        "tt.extern_elementwise",
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
`fp_to_fp`

Floating point casting for custom types (F8), and non-default rounding modes.

F8 <-> FP16, BF16, FP32, FP64
"""
function fp_to_fp(src::Value; result::IR.Type, rounding=nothing, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(rounding) && push!(attributes, namedattribute("rounding", rounding))

    return create_operation(
        "tt.fp_to_fp",
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
`gather`

Gather elements from the input tensor using the indices tensor along a
single specified axis. The output tensor has the same shape as the indices
tensor. The input and indices tensors must have the same number of
dimension, and each dimension of the indices tensor that is not the gather
dimension cannot be greater than the corresponding dimension in the input
tensor.
"""
function gather(
    src::Value,
    indices::Value;
    result=nothing::Union{Nothing,IR.Type},
    axis,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[src, indices]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tt.gather",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function get_num_programs(;
    result=nothing::Union{Nothing,IR.Type}, axis, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tt.get_num_programs",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function get_program_id(; result=nothing::Union{Nothing,IR.Type}, axis, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tt.get_program_id",
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
`histogram`

Return the histogram of the input tensor. The number of bins is equal to
the dimension of the output tensor. Each bins has a width of 1 and bins
start at 0.
"""
function histogram(src::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.histogram",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function int_to_ptr(src::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.int_to_ptr",
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
`join`

For example, if the two input tensors are 4x8xf32, returns a tensor of
shape 4x8x2xf32.

Because Triton tensors always have a power-of-two number of elements,
the two input tensors must have the same shape.
"""
function join(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tt.join",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function load(
    ptr::Value,
    mask=nothing::Union{Nothing,Value};
    other=nothing::Union{Nothing,Value},
    result=nothing::Union{Nothing,IR.Type},
    boundaryCheck=nothing,
    padding=nothing,
    cache=nothing,
    evict=nothing,
    isVolatile=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[ptr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(mask) && push!(operands, mask)
    !isnothing(other) && push!(operands, other)
    push!(attributes, operandsegmentsizes([
        1,
        if (mask == nothing)
            0
        elseif 1(other == nothing)
            0
        else
            1
        end,
    ]))
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(boundaryCheck) &&
        push!(attributes, namedattribute("boundaryCheck", boundaryCheck))
    !isnothing(padding) && push!(attributes, namedattribute("padding", padding))
    !isnothing(cache) && push!(attributes, namedattribute("cache", cache))
    !isnothing(evict) && push!(attributes, namedattribute("evict", evict))
    !isnothing(isVolatile) && push!(attributes, namedattribute("isVolatile", isVolatile))

    return create_operation(
        "tt.load",
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
`make_range`

Returns an 1D int32 tensor.

Values span from \$start to \$end (exclusive), with step = 1
"""
function make_range(; result::IR.Type, start, end_, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("start", start), namedattribute("end", end_)]

    return create_operation(
        "tt.make_range",
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
`make_tensor_descriptor`

`tt.make_tensor_descriptor` takes both meta information of the parent tensor and the block size,
and returns a descriptor object which can be used to load/store from the tensor in global memory.
"""
function make_tensor_descriptor(
    base::Value,
    shape::Vector{Value},
    strides::Vector{Value};
    result::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[base, shape..., strides...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.make_tensor_descriptor",
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
`make_tensor_ptr`

`tt.make_tensor_ptr` takes both meta information of the parent tensor and the block tensor, then it returns a
pointer to the block tensor, e.g. returns a type of `tt.ptr<tensor<8x8xf16>>`.
"""
function make_tensor_ptr(
    base::Value,
    shape::Vector{Value},
    strides::Vector{Value},
    offsets::Vector{Value};
    result::IR.Type,
    order,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[base, shape..., strides..., offsets...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("order", order),]

    return create_operation(
        "tt.make_tensor_ptr",
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
`mulhiui`

Most significant N bits of the 2N-bit product of two integers.
"""
function mulhiui(
    x::Value, y::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[x, y]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tt.mulhiui",
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
`precise_divf`

Precise div for floating point types.
"""
function precise_divf(
    x::Value, y::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[x, y]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tt.precise_divf",
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
`precise_sqrt`

Precise sqrt for floating point types.
"""
function precise_sqrt(x::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[x,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tt.precise_sqrt",
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
`print`

`tt.print` takes a literal string prefix and an arbitrary number of scalar or tensor arguments that should be printed.
format are generated automatically from the arguments.
"""
function print(args::Vector{Value}; prefix, hex, isSigned, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("prefix", prefix),
        namedattribute("hex", hex),
        namedattribute("isSigned", isSigned),
    ]

    return create_operation(
        "tt.print",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function ptr_to_int(src::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.ptr_to_int",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function reduce(
    srcs::Vector{Value};
    result::Vector{IR.Type},
    axis,
    combineOp::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result...,]
    operands = Value[srcs...,]
    owned_regions = Region[combineOp,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis),]

    return create_operation(
        "tt.reduce",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function reduce_return(result::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[result...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.reduce.return",
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
`reshape`

reinterpret a tensor to a different shape.

If allow_reorder is set the compiler is free to change the order of
elements to generate more efficient code.

If efficient_layout is set, this is a hint that the destination layout should be kept for performance reason.
The compiler is still free to change it for better performance.
"""
function reshape(
    src::Value;
    result::IR.Type,
    allow_reorder=nothing,
    efficient_layout=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(allow_reorder) &&
        push!(attributes, namedattribute("allow_reorder", allow_reorder))
    !isnothing(efficient_layout) &&
        push!(attributes, namedattribute("efficient_layout", efficient_layout))

    return create_operation(
        "tt.reshape",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function scan(
    srcs::Vector{Value};
    result::Vector{IR.Type},
    axis,
    reverse,
    combineOp::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result...,]
    operands = Value[srcs...,]
    owned_regions = Region[combineOp,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("axis", axis), namedattribute("reverse", reverse)
    ]

    return create_operation(
        "tt.scan",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function scan_return(result::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[result...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.scan.return",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function splat(src::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tt.splat",
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
`split`

The input must be a tensor whose last dimension has size 2.  Returns two
tensors, src[..., 0] and src[..., 1].

For example, if the input shape is 4x8x2xf32, returns two tensors of
shape 4x8xf32.
"""
function split(
    src::Value;
    outLHS=nothing::Union{Nothing,IR.Type},
    outRHS=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(outLHS) && push!(op_ty_results, outLHS)
    !isnothing(outRHS) && push!(op_ty_results, outRHS)

    return create_operation(
        "tt.split",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function store(
    ptr::Value,
    value::Value,
    mask=nothing::Union{Nothing,Value};
    boundaryCheck=nothing,
    cache=nothing,
    evict=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[ptr, value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(mask) && push!(operands, mask)
    !isnothing(boundaryCheck) &&
        push!(attributes, namedattribute("boundaryCheck", boundaryCheck))
    !isnothing(cache) && push!(attributes, namedattribute("cache", cache))
    !isnothing(evict) && push!(attributes, namedattribute("evict", evict))

    return create_operation(
        "tt.store",
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
`trans`

For example, given a tensor x with shape [1,2,4], transpose(x) with
order=[2,0,1] rearranges the tensor to have shape [4,1,2].

Although this op is called \"trans\", it implements both tl.trans() and
tl.permute().  (\"permute\" might be a better name, but it\'s called \"trans\"
because originally it only supported 2D tensors.)

## Implementation note on encodings:

In the TritonGPU dialect (and probably others), an encoding is chosen for
this op\'s output so it\'s a nop from the perspective of code generation.

For example, suppose tensor x has an encoding such that GPU thread [i,j,k]
has a register containing element [i,j,k] of the tensor.  Now we transpose
x with order [2,1,0], i.e. we reverse the order of its dimensions.  In
TritonGPU, we will choose a layout for the output of the transpose so that
GPU thread [i,j,k] has element [k,j,i] of transpose(x).  But this is the
same element it had before!  All we\'ve done is \"rename\" the element that
thread [i,j,k] has.

The \"real\" transpose -- i.e. moving data between GPU threads -- occurs in
convertLayout ops that appear before and/or after the operation.

We do this so that you can chain multiple data-movement ops (e.g.
transpose+reshape+concat) without going to shared memory after each one.
"""
function trans(
    src::Value; result=nothing::Union{Nothing,IR.Type}, order, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("order", order),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tt.trans",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

end # tt
