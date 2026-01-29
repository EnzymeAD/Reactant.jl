module enzymexla_tt_ext
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
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API

function call(
    gridx::Value,
    gridy::Value,
    gridz::Value,
    clusterx::Value,
    clustery::Value,
    clusterz::Value,
    inputs::Vector{Value};
    result_0::Vector{IR.Type},
    fn,
    backend_config=nothing,
    operand_layouts=nothing,
    result_layouts=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
    output_operand_aliases=nothing,
    xla_side_effect_free=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result_0...,]
    operands = Value[gridx, gridy, gridz, clusterx, clustery, clusterz, inputs...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("fn", fn),]
    !isnothing(backend_config) &&
        push!(attributes, NamedAttribute("backend_config", backend_config))
    !isnothing(operand_layouts) &&
        push!(attributes, NamedAttribute("operand_layouts", operand_layouts))
    !isnothing(result_layouts) &&
        push!(attributes, NamedAttribute("result_layouts", result_layouts))
    !isnothing(arg_attrs) && push!(attributes, NamedAttribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, NamedAttribute("res_attrs", res_attrs))
    !isnothing(output_operand_aliases) &&
        push!(attributes, NamedAttribute("output_operand_aliases", output_operand_aliases))
    !isnothing(xla_side_effect_free) &&
        push!(attributes, NamedAttribute("xla_side_effect_free", xla_side_effect_free))

    return create_operation(
        "enzymexla_tt_ext.call",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function module_(; sym_name, bodyRegion::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[bodyRegion,]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name),]

    return create_operation(
        "enzymexla_tt_ext.module",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # enzymexla_tt_ext
