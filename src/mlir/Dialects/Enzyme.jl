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


function autodiff(inputs::Vector{Value}; outputs::Vector{IR.Type}, fn, activity, ret_activity, width=nothing, location=Location())
    op_ty_results = IR.Type[outputs..., ]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn), namedattribute("activity", activity), namedattribute("ret_activity", ret_activity), ]
    !isnothing(width) && push!(attributes, namedattribute("width", width))
    
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

"""
`broadcast`

Broadcast the operand by adding extra dimensions with sizes provided by the `shape` attribute to the front.
For scalar operands, ranked tensor is created.

NOTE: Only works for scalar and *ranked* tensor operands for now.
"""
function broadcast(input::Value; output::IR.Type, shape, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("shape", shape), ]
    
    create_operation(
        "enzyme.broadcast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function fwddiff(inputs::Vector{Value}; outputs::Vector{IR.Type}, fn, activity, ret_activity, width=nothing, location=Location())
    op_ty_results = IR.Type[outputs..., ]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn), namedattribute("activity", activity), namedattribute("ret_activity", ret_activity), ]
    !isnothing(width) && push!(attributes, namedattribute("width", width))
    
    create_operation(
        "enzyme.fwddiff", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`generate`

Generate a sample from a probabilistic function by replacing all SampleOps with distribution calls.
"""
function generate(inputs::Vector{Value}; outputs::Vector{IR.Type}, fn, name=nothing, location=Location())
    op_ty_results = IR.Type[outputs..., ]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn), ]
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    create_operation(
        "enzyme.generate", location;
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


function sample(inputs::Vector{Value}; outputs::Vector{IR.Type}, fn, name=nothing, location=Location())
    op_ty_results = IR.Type[outputs..., ]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn), ]
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    create_operation(
        "enzyme.sample", location;
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

"""
`simulate`

Simulate a probabilistic function to generate execution trace
by replacing all SampleOps with distribution calls and inserting
sampled values into the choice map.
"""
function simulate(inputs::Vector{Value}; trace::IR.Type, fn, name=nothing, location=Location())
    op_ty_results = IR.Type[trace, ]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn), ]
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    create_operation(
        "enzyme.simulate", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`trace`

Execute a probabilistic function specified by a symbol reference using the provided arguments,
and a set of constraints on the sampled variables (if provided). Return the execution trace
(if provided) and the log-likelihood of the execution trace.
"""
function trace(inputs::Vector{Value}, oldTrace=nothing::Union{Nothing, Value}; constraints=nothing::Union{Nothing, Value}, newTrace::IR.Type, weights::Vector{IR.Type}, fn, name=nothing, location=Location())
    op_ty_results = IR.Type[newTrace, weights..., ]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn), ]
    !isnothing(oldTrace) && push!(operands, oldTrace)
    !isnothing(constraints) && push!(operands, constraints)
    push!(attributes, operandsegmentsizes([length(inputs), (oldTrace==nothing) ? 0 : 1(constraints==nothing) ? 0 : 1]))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    create_operation(
        "enzyme.trace", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`addSampleToTrace`

Add a sampled value into the execution trace.
"""
function addSampleToTrace(trace::Value, sample::Value; name=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[trace, sample, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    create_operation(
        "enzyme.addSampleToTrace", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`insertChoiceToMap`

Insert a constraint on a sampled variable into the choice map.
"""
function insertChoiceToMap(choiceMap::Value, choice::Value; newChoiceMap::IR.Type, name=nothing, location=Location())
    op_ty_results = IR.Type[newChoiceMap, ]
    operands = Value[choiceMap, choice, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    create_operation(
        "enzyme.insertChoiceToMap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

end # enzyme
