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

end # enzymexla
