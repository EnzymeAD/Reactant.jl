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
import ..Dialects: namedattribute, operandsegmentsizes, c
import ...API
using EnumX

function kernel_call(
    gridx::Value,
    gridy::Value,
    gridz::Value,
    blockx::Value,
    blocky::Value,
    blockz::Value,
    shmem::Value,
    inputs::Vector{Value};
    result::Union{Vector{IR.Type},Tuple{Vararg{IR.Type}}},
    fn::IR.FlatSymbol,
    backend_config::Union{String,Nothing}=nothing,
    operand_layouts::Union{IR.Attribute,Nothing}=nothing,
    result_layouts::Union{IR.Attribute,Nothing}=nothing,
    output_operand_aliases::Union{Vector{Attribute},Nothing}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[result...,]
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

end # enzymexla
