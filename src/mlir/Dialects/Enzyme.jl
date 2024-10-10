module enzyme
using ...IR
import ...IR: NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`addTo`

TODO
"""
function addTo(values::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[values..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "enzyme.addTo", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function autodiff(inputs::Vector{Value}; outputs::Vector{IR.Type}, fn, activity, ret_activity, location=Location())
    op_ty_results = IR.Type[outputs..., ]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn), namedattribute("activity", activity), namedattribute("ret_activity", ret_activity), ]
    
    create_operation(
        "enzyme.autodiff", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function batch(inputs::Vector{Value}; outputs::Vector{IR.Type}, fn, batch_shape, location=Location())
    op_ty_results = IR.Type[outputs..., ]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn), namedattribute("batch_shape", batch_shape), ]
    
    create_operation(
        "enzyme.batch", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function fwddiff(inputs::Vector{Value}; outputs::Vector{IR.Type}, fn, activity, ret_activity, location=Location())
    op_ty_results = IR.Type[outputs..., ]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn), namedattribute("activity", activity), namedattribute("ret_activity", ret_activity), ]
    
    create_operation(
        "enzyme.fwddiff", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function genericAdjoint(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{IR.Type}, indexing_maps, iterator_types, doc=nothing, library_call=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("indexing_maps", indexing_maps), namedattribute("iterator_types", iterator_types), ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    !isnothing(doc) && push!(attributes, namedattribute("doc", doc))
    !isnothing(library_call) && push!(attributes, namedattribute("library_call", library_call))
    
    create_operation(
        "enzyme.genericAdjoint", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function get(gradient::Value; result_0::IR.Type, location=Location())
    op_ty_results = IR.Type[result_0, ]
    operands = Value[gradient, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "enzyme.get", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function init(; result_0::IR.Type, location=Location())
    op_ty_results = IR.Type[result_0, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "enzyme.init", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function placeholder(; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "enzyme.placeholder", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function pop(cache::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[cache, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "enzyme.pop", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function push(cache::Value, value::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[cache, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "enzyme.push", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function set(gradient::Value, value::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[gradient, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "enzyme.set", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

end # enzyme
