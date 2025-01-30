module enzyme
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
`Activity`
Possible activity states for variables
"""
@enumx Activity enzyme_active enzyme_dup enzyme_const enzyme_dupnoneed enzyme_activenoneed enzyme_constnoneed
ActivityStorage = [
    "enzyme_active",
    "enzyme_dup",
    "enzyme_const",
    "enzyme_dupnoneed",
    "enzyme_activenoneed",
    "enzyme_constnoneed",
]

function IR.Attribute(e::Activity.T)
    return parse(Attribute, "#enzyme<activity $(ActivityStorage[Int(e)+1])>")
end

"""
`addTo`

TODO
"""
function addTo(values::Vector{Value}; location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[values...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.addTo",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function autodiff(
    inputs::Vector{Value};
    outputs::Union{Vector{IR.Type},Tuple{Vararg{IR.Type}}},
    fn::IR.FlatSymbol,
    activity::Vector{Activity.T},
    ret_activity::Vector{Activity.T},
    width::Union{Int64,Nothing}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("fn", fn),
        namedattribute("activity", activity),
        namedattribute("ret_activity", ret_activity),
    ]
    !isnothing(width) && push!(attributes, namedattribute("width", width))

    return create_operation(
        "enzyme.autodiff",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function batch(
    inputs::Vector{Value};
    outputs::Union{Vector{IR.Type},Tuple{Vararg{IR.Type}}},
    fn::IR.FlatSymbol,
    batch_shape::Vector{Int64},
    location::Location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("fn", fn), namedattribute("batch_shape", batch_shape)
    ]

    return create_operation(
        "enzyme.batch",
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

Broadcast the operand by adding extra dimensions with sizes provided by the `shape` attribute to the front.
For scalar operands, ranked tensor is created.

NOTE: Only works for scalar and *ranked* tensor operands for now.
"""
function broadcast(
    input::Value; output::IR.Type, shape::Vector{Int64}, location::Location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("shape", shape),]

    return create_operation(
        "enzyme.broadcast",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fwddiff(
    inputs::Vector{Value};
    outputs::Union{Vector{IR.Type},Tuple{Vararg{IR.Type}}},
    fn::IR.FlatSymbol,
    activity::Vector{Activity.T},
    ret_activity::Vector{Activity.T},
    width::Union{Int64,Nothing}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("fn", fn),
        namedattribute("activity", activity),
        namedattribute("ret_activity", ret_activity),
    ]
    !isnothing(width) && push!(attributes, namedattribute("width", width))

    return create_operation(
        "enzyme.fwddiff",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function genericAdjoint(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Union{Vector{IR.Type},Tuple{Vararg{IR.Type}}},
    indexing_maps::Vector{Any},
    iterator_types::Vector{Attribute},
    doc::Union{String,Nothing}=nothing,
    library_call::Union{String,Nothing}=nothing,
    region::Region,
    location::Location=Location(),
)
    op_ty_results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("indexing_maps", indexing_maps),
        namedattribute("iterator_types", iterator_types),
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(doc) && push!(attributes, namedattribute("doc", doc))
    !isnothing(library_call) &&
        push!(attributes, namedattribute("library_call", library_call))

    return create_operation(
        "enzyme.genericAdjoint",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function get(gradient::Value; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[gradient,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.get",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function init(; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.init",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function placeholder(; output::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.placeholder",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function pop(cache::Value; output::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[cache,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.pop",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function push(cache::Value, value::Value; location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[cache, value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.push",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function set(gradient::Value, value::Value; location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[gradient, value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.set",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # enzyme
