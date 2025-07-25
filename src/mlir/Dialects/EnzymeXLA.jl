module enzymexla
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

function scope(
    operands::Vector{Value}; results::Vector{IR.Type}, region::Region, location=Location()
)
    op_ty_results = IR.Type[results...,]
    operands = Value[operands...,]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.scope",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function alternatives(; regions::Vector{Region}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[regions...,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.alternatives",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function barrier(indices::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[indices...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.barrier",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function comm_region(; result_0::Vector{IR.Type}, body::Region, location=Location())
    op_ty_results = IR.Type[result_0...,]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.comm_region",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function extend(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    lhs,
    rhs,
    dimension,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("lhs", lhs),
        namedattribute("rhs", rhs),
        namedattribute("dimension", dimension),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.extend",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function gpu_block(
    blockIndexX::Value,
    blockIndexY::Value,
    blockIndexZ::Value;
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[blockIndexX, blockIndexY, blockIndexZ]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.gpu_block",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function gpu_error(; result::IR.Type, region::Region, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.gpu_error",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function gpu_thread(
    threadIndexX::Value,
    threadIndexY::Value,
    threadIndexZ::Value;
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[threadIndexX, threadIndexY, threadIndexZ]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.gpu_thread",
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
`gpu_wrapper`

The optional arguments to this operation are suggestions about what block
dimensions this gpu kernel should have - usually taken from kernel launch
params
"""
function gpu_wrapper(
    blockDims::Vector{Value}; result::IR.Type, region::Region, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[blockDims...,]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.gpu_wrapper",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function ml_gelu(
    input::Value;
    result=nothing::Union{Nothing,IR.Type},
    gelu_approximation,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("gelu_approximation", gelu_approximation),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.ml.gelu",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function get_stream(; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.get_stream",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function jit_call(
    inputs::Vector{Value};
    result_0::Vector{IR.Type},
    fn,
    backend_config=nothing,
    operand_layouts=nothing,
    result_layouts=nothing,
    output_operand_aliases=nothing,
    xla_side_effect_free=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result_0...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]
    !isnothing(backend_config) &&
        push!(attributes, namedattribute("backend_config", backend_config))
    !isnothing(operand_layouts) &&
        push!(attributes, namedattribute("operand_layouts", operand_layouts))
    !isnothing(result_layouts) &&
        push!(attributes, namedattribute("result_layouts", result_layouts))
    !isnothing(output_operand_aliases) &&
        push!(attributes, namedattribute("output_operand_aliases", output_operand_aliases))
    !isnothing(xla_side_effect_free) &&
        push!(attributes, namedattribute("xla_side_effect_free", xla_side_effect_free))

    return create_operation(
        "enzymexla.jit_call",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function kernel_call(
    gridx::Value,
    gridy::Value,
    gridz::Value,
    blockx::Value,
    blocky::Value,
    blockz::Value,
    shmem::Value,
    inputs::Vector{Value};
    result_0::Vector{IR.Type},
    fn,
    backend_config=nothing,
    operand_layouts=nothing,
    result_layouts=nothing,
    output_operand_aliases=nothing,
    xla_side_effect_free=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result_0...,]
    operands = Value[gridx, gridy, gridz, blockx, blocky, blockz, shmem, inputs...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]
    !isnothing(backend_config) &&
        push!(attributes, namedattribute("backend_config", backend_config))
    !isnothing(operand_layouts) &&
        push!(attributes, namedattribute("operand_layouts", operand_layouts))
    !isnothing(result_layouts) &&
        push!(attributes, namedattribute("result_layouts", result_layouts))
    !isnothing(output_operand_aliases) &&
        push!(attributes, namedattribute("output_operand_aliases", output_operand_aliases))
    !isnothing(xla_side_effect_free) &&
        push!(attributes, namedattribute("xla_side_effect_free", xla_side_effect_free))

    return create_operation(
        "enzymexla.kernel_call",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function linalg_lu(
    input::Value;
    output::IR.Type,
    pivots::IR.Type,
    permutation::IR.Type,
    info::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[output, pivots, permutation, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.linalg.lu",
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
`memcpy`

The `gpu.memcpy` operation copies the content of one memref to another.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

# Example

```mlir
%token = gpu.memcpy async [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
```
"""
function memcpy(
    asyncDependencies::Vector{Value},
    target::Value,
    source::Value,
    size::Value;
    asyncToken=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[asyncDependencies..., target, source, size]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(op_ty_results, asyncToken)

    return create_operation(
        "enzymexla.memcpy",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function memref2pointer(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.memref2pointer",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function noop(blockDims::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[blockDims...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.noop",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function pointer2memref(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.pointer2memref",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function polygeist_yield(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.polygeist_yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function linalg_qr(
    input::Value; output::IR.Type, tau::IR.Type, info::IR.Type, location=Location()
)
    op_ty_results = IR.Type[output, tau, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.linalg.qr",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function ml_relu(input::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.ml.relu",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function rotate(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    amount,
    dimension,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("amount", amount), namedattribute("dimension", dimension)
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.rotate",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function linalg_svd(
    input::Value;
    U::IR.Type,
    S::IR.Type,
    Vt::IR.Type,
    info::IR.Type,
    full=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[U, S, Vt, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(full) && push!(attributes, namedattribute("full", full))

    return create_operation(
        "enzymexla.linalg.svd",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function stream2token(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.stream2token",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function wrap(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    lhs,
    rhs,
    dimension,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("lhs", lhs),
        namedattribute("rhs", rhs),
        namedattribute("dimension", dimension),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.wrap",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function xla_wrapper(inputs::Vector{Value}; fn, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]

    return create_operation(
        "enzymexla.xla_wrapper",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # enzymexla
