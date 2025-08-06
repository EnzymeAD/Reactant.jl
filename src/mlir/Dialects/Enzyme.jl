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
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

"""
`addRetvalToTrace`

Add the function\'s return value(s) into the execution trace.
"""
function addRetvalToTrace(
    trace::Value, retval::Vector{Value}; updated_trace::IR.Type, location=Location()
)
    op_ty_results = IR.Type[updated_trace,]
    operands = Value[trace, retval...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.addRetvalToTrace",
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
`addSampleToTrace`

Add a sampled value into the execution trace.
"""
function addSampleToTrace(
    trace::Value, sample::Vector{Value}; updated_trace::IR.Type, symbol, location=Location()
)
    op_ty_results = IR.Type[updated_trace,]
    operands = Value[trace, sample...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("symbol", symbol),]

    return create_operation(
        "enzyme.addSampleToTrace",
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
`addSubtrace`

Insert a subtrace into a parent trace.
"""
function addSubtrace(
    subtrace::Value, trace::Value; updated_trace::IR.Type, symbol, location=Location()
)
    op_ty_results = IR.Type[updated_trace,]
    operands = Value[subtrace, trace]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("symbol", symbol),]

    return create_operation(
        "enzyme.addSubtrace",
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
`addTo`

TODO
"""
function addTo(values::Vector{Value}; location=Location())
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

"""
`addWeightToTrace`

Add the aggregated log-probability weight to the execution trace.
"""
function addWeightToTrace(
    trace::Value, weight::Value; updated_trace::IR.Type, location=Location()
)
    op_ty_results = IR.Type[updated_trace,]
    operands = Value[trace, weight]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.addWeightToTrace",
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
    outputs::Vector{IR.Type},
    fn,
    activity,
    ret_activity,
    width=nothing,
    strong_zero=nothing,
    location=Location(),
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
    !isnothing(strong_zero) && push!(attributes, namedattribute("strong_zero", strong_zero))

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
    inputs::Vector{Value}; outputs::Vector{IR.Type}, fn, batch_shape, location=Location()
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
function broadcast(input::Value; output::IR.Type, shape, location=Location())
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
    outputs::Vector{IR.Type},
    fn,
    activity,
    ret_activity,
    width=nothing,
    strong_zero=nothing,
    location=Location(),
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
    !isnothing(strong_zero) && push!(attributes, namedattribute("strong_zero", strong_zero))

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

"""
`generate`

Generate an execution trace and weight from a probabilistic function.
If a `constraint` dict is provided AND the sample op\'s `symbol` is in the
`constrained_symbols` array, we will use the corresponding constraint value
instead of generating new samples from the probabilistic function.
By convention, the 0th operand in `inputs` or `outputs` is the initial RNG
state (seed).
"""
function generate(
    inputs::Vector{Value},
    constraint::Value;
    trace::IR.Type,
    weight::IR.Type,
    outputs::Vector{IR.Type},
    fn,
    constrained_addresses,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[trace, weight, outputs...]
    operands = Value[inputs..., constraint]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("fn", fn),
        namedattribute("constrained_addresses", constrained_addresses),
    ]
    !isnothing(name) && push!(attributes, namedattribute("name", name))

    return create_operation(
        "enzyme.generate",
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
    result_tensors::Vector{IR.Type},
    indexing_maps,
    iterator_types,
    doc=nothing,
    library_call=nothing,
    region::Region,
    location=Location(),
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

function get(gradient::Value; result_0::IR.Type, location=Location())
    op_ty_results = IR.Type[result_0,]
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

"""
`getSampleFromConstraint`

Get sampled values from a constraint for a given symbol.
"""
function getSampleFromConstraint(
    constraint::Value; outputs::Vector{IR.Type}, symbol, location=Location()
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[constraint,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("symbol", symbol),]

    return create_operation(
        "enzyme.getSampleFromConstraint",
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
`getSubconstraint`

Get a subconstraint from a constraint for a given symbol.
"""
function getSubconstraint(
    constraint::Value; subconstraint::IR.Type, symbol, location=Location()
)
    op_ty_results = IR.Type[subconstraint,]
    operands = Value[constraint,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("symbol", symbol),]

    return create_operation(
        "enzyme.getSubconstraint",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function ignore_derivatives(input::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.ignore_derivatives",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function init(; result_0::IR.Type, location=Location())
    op_ty_results = IR.Type[result_0,]
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

"""
`initTrace`

Initialize an execution trace for a probabilistic function.
"""
function initTrace(; trace::IR.Type, location=Location())
    op_ty_results = IR.Type[trace,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.initTrace",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function placeholder(; output::IR.Type, location=Location())
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

function pop(cache::Value; output::IR.Type, location=Location())
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

function push(cache::Value, value::Value; location=Location())
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

"""
`sample`

Sample from a distribution. By convention, the 0th operand in `inputs`
or `outputs` is the initial RNG state (seed).
"""
function sample(
    inputs::Vector{Value};
    outputs::Vector{IR.Type},
    fn,
    logpdf=nothing,
    symbol=nothing,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]
    !isnothing(logpdf) && push!(attributes, namedattribute("logpdf", logpdf))
    !isnothing(symbol) && push!(attributes, namedattribute("symbol", symbol))
    !isnothing(name) && push!(attributes, namedattribute("name", name))

    return create_operation(
        "enzyme.sample",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function set(gradient::Value, value::Value; location=Location())
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

"""
`simulate`

Simulate a probabilistic function to generate execution trace
by replacing all SampleOps with distribution calls and recording
all sampled values into the trace. This op returns the trace, the weight
(accumulated log-probability), and the other outputs. By convention,
the 0th operand in `inputs` or `outputs` is the initial RNG state (seed).
"""
function simulate(
    inputs::Vector{Value};
    trace::IR.Type,
    weight::IR.Type,
    outputs::Vector{IR.Type},
    fn,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[trace, weight, outputs...]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]
    !isnothing(name) && push!(attributes, namedattribute("name", name))

    return create_operation(
        "enzyme.simulate",
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
`untracedCall`

Call a probabilistic function without tracing. By convention, the 0th operand in `inputs`
or `outputs` is the initial RNG state (seed).
"""
function untracedCall(
    inputs::Vector{Value}; outputs::Vector{IR.Type}, fn, name=nothing, location=Location()
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]
    !isnothing(name) && push!(attributes, namedattribute("name", name))

    return create_operation(
        "enzyme.untracedCall",
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
