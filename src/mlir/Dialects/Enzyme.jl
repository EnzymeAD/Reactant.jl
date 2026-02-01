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
import ..Dialects: operandsegmentsizes, resultsegmentsizes
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
    attributes = NamedAttribute[NamedAttribute("symbol", symbol),]

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
    attributes = NamedAttribute[NamedAttribute("symbol", symbol),]

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

function affine_atomic_rmw(
    value::Value,
    memref::Value,
    indices::Vector{Value};
    result::IR.Type,
    kind,
    map,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[value, memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), NamedAttribute("map", map)]

    return create_operation(
        "enzyme.affine_atomic_rmw",
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
        NamedAttribute("fn", fn),
        NamedAttribute("activity", activity),
        NamedAttribute("ret_activity", ret_activity),
    ]
    !isnothing(width) && push!(attributes, NamedAttribute("width", width))
    !isnothing(strong_zero) && push!(attributes, NamedAttribute("strong_zero", strong_zero))

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

function autodiff_region(
    inputs::Vector{Value};
    outputs::Vector{IR.Type},
    activity,
    ret_activity,
    width=nothing,
    strong_zero=nothing,
    fn=nothing,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("activity", activity), NamedAttribute("ret_activity", ret_activity)
    ]
    !isnothing(width) && push!(attributes, NamedAttribute("width", width))
    !isnothing(strong_zero) && push!(attributes, NamedAttribute("strong_zero", strong_zero))
    !isnothing(fn) && push!(attributes, NamedAttribute("fn", fn))

    return create_operation(
        "enzyme.autodiff_region",
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
        NamedAttribute("fn", fn), NamedAttribute("batch_shape", batch_shape)
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
    attributes = NamedAttribute[NamedAttribute("shape", shape),]

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

"""
`cholesky`

Computes the Cholesky decomposition of a symmetric positive definite matrix A.
Returns L such that A = L @ L^T (if lower=true) or A = U^T @ U (if lower=false).
"""
function cholesky(input::Value; result::IR.Type, lower=nothing, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(lower) && push!(attributes, NamedAttribute("lower", lower))

    return create_operation(
        "enzyme.cholesky",
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
`concat`

Concat list of input arguments into a generic value
"""
function concat(inputs::Vector{Value}; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.concat",
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
`dot`

Computes a general dot product operation. To be lowered to `stablehlo.dot_general`.
"""
function dot(
    lhs::Value,
    rhs::Value;
    result::IR.Type,
    lhs_batching_dimensions,
    rhs_batching_dimensions,
    lhs_contracting_dimensions,
    rhs_contracting_dimensions,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("lhs_batching_dimensions", lhs_batching_dimensions),
        NamedAttribute("rhs_batching_dimensions", rhs_batching_dimensions),
        NamedAttribute("lhs_contracting_dimensions", lhs_contracting_dimensions),
        NamedAttribute("rhs_contracting_dimensions", rhs_contracting_dimensions),
    ]

    return create_operation(
        "enzyme.dot",
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
`dump`

Debug operation that dumps a tensor value with a label.
"""
function dump(value::Value; output::IR.Type, label, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("label", label),]

    return create_operation(
        "enzyme.dump",
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
`dynamic_extract`

Extracts a slice at the specified dynamic index along the first dimension.
"""
function dynamic_extract(input::Value, index::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[input, index]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.dynamic_extract",
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
`dynamic_update`

Returns a new tensor with the slice at the specified dynamic index replaced.
"""
function dynamic_update(
    input::Value, index::Value, value::Value; result::IR.Type, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[input, index, value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.dynamic_update",
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
`extract`

Extract value from batched operand at index
"""
function extract(input::Value; output::IR.Type, index, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("index", index),]

    return create_operation(
        "enzyme.extract",
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
`for_loop`

A counted loop operation that iterates from `lowerBound` to `upperBound`
by `step`, carrying `iter_args` through each iteration. The iteration
variable and iter_args are passed to the body region.
"""
function for_loop(
    lowerBound::Value,
    upperBound::Value,
    step::Value,
    initArgs::Vector{Value};
    results::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[lowerBound, upperBound, step, initArgs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.for_loop",
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
        NamedAttribute("fn", fn),
        NamedAttribute("activity", activity),
        NamedAttribute("ret_activity", ret_activity),
    ]
    !isnothing(width) && push!(attributes, NamedAttribute("width", width))
    !isnothing(strong_zero) && push!(attributes, NamedAttribute("strong_zero", strong_zero))

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

function fwddiff_region(
    inputs::Vector{Value};
    outputs::Vector{IR.Type},
    activity,
    ret_activity,
    width=nothing,
    strong_zero=nothing,
    fn=nothing,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("activity", activity), NamedAttribute("ret_activity", ret_activity)
    ]
    !isnothing(width) && push!(attributes, NamedAttribute("width", width))
    !isnothing(strong_zero) && push!(attributes, NamedAttribute("strong_zero", strong_zero))
    !isnothing(fn) && push!(attributes, NamedAttribute("fn", fn))

    return create_operation(
        "enzyme.fwddiff_region",
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
        NamedAttribute("fn", fn),
        NamedAttribute("constrained_addresses", constrained_addresses),
    ]
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

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
        NamedAttribute("indexing_maps", indexing_maps),
        NamedAttribute("iterator_types", iterator_types),
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(doc) && push!(attributes, NamedAttribute("doc", doc))
    !isnothing(library_call) &&
        push!(attributes, NamedAttribute("library_call", library_call))

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

"""
`getFlattenedSamplesFromTrace`

Get sampled values for multiple addresses from an execution trace and
flatten them into a single position vector for HMC.
"""
function getFlattenedSamplesFromTrace(
    trace::Value; position::IR.Type, selection, location=Location()
)
    op_ty_results = IR.Type[position,]
    operands = Value[trace,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("selection", selection),]

    return create_operation(
        "enzyme.getFlattenedSamplesFromTrace",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function get(gradient::Value; result::IR.Type, location=Location())
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
    attributes = NamedAttribute[NamedAttribute("symbol", symbol),]

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
`getSampleFromTrace`

Get the sampled value for a given symbol from an execution trace.
"""
function getSampleFromTrace(
    trace::Value; sample::Vector{IR.Type}, symbol, location=Location()
)
    op_ty_results = IR.Type[sample...,]
    operands = Value[trace,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("symbol", symbol),]

    return create_operation(
        "enzyme.getSampleFromTrace",
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
    attributes = NamedAttribute[NamedAttribute("symbol", symbol),]

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

"""
`getSubtrace`

Get a subtrace from a trace for a given symbol.
"""
function getSubtrace(trace::Value; subtrace::IR.Type, symbol, location=Location())
    op_ty_results = IR.Type[subtrace,]
    operands = Value[trace,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("symbol", symbol),]

    return create_operation(
        "enzyme.getSubtrace",
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
`getWeightFromTrace`

Get the accumulated log-probability weight from an execution trace.
"""
function getWeightFromTrace(trace::Value; weight::IR.Type, location=Location())
    op_ty_results = IR.Type[weight,]
    operands = Value[trace,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.getWeightFromTrace",
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
`if_`

A conditional operation that executes exactly one of two branches based on a
boolean predicate.
"""
function if_(
    predicate::Value;
    results::Vector{IR.Type},
    trueBranch::Region,
    falseBranch::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[predicate,]
    owned_regions = Region[trueBranch, falseBranch]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.if",
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

function load(cache::Value, indices::Vector{Value}; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[cache, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.load",
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
`log_add_exp`

Computes log(exp(x) + exp(y)).
"""
function log_add_exp(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.log_add_exp",
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
`logistic`

Computes the logistic (sigmoid) function: 1 / (1 + exp(-x)).
"""
function logistic(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.logistic",
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
`mcmc`

Performs MCMC inference on a probabilistic function.

By default (num_warmup=0, num_samples=1, thinning=1), performs a single MCMC step
and returns the resulting trace - backwards compatible with previous behavior.

When num_samples > 1, runs a full MCMC chain and returns a trace where
sampled values have an additional batch dimension (first dim = num_samples / thinning).

Thinning controls the fraction of samples retained: when thinning=k, every k-th
sample is kept, resulting in num_samples/k output samples.

Warmup iterations are discarded and used only for adaptation.

By convention, the 0th operand in inputs is the initial RNG state
and the 0th operand in results is the updated RNG state.
"""
function mcmc(
    inputs::Vector{Value},
    original_trace::Value,
    inverse_mass_matrix=nothing::Union{Nothing,Value};
    step_size=nothing::Union{Nothing,Value},
    new_trace::IR.Type,
    accepted::IR.Type,
    output_rng_state::IR.Type,
    fn,
    selection,
    num_warmup=nothing,
    num_samples=nothing,
    thinning=nothing,
    hmc_config=nothing,
    nuts_config=nothing,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[new_trace, accepted, output_rng_state]
    operands = Value[inputs..., original_trace]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn), NamedAttribute("selection", selection)
    ]
    !isnothing(inverse_mass_matrix) && push!(operands, inverse_mass_matrix)
    !isnothing(step_size) && push!(operands, step_size)
    push!(
        attributes,
        operandsegmentsizes([
            length(inputs),
            1,
            Int(!isnothing(inverse_mass_matrix)),
            Int(!isnothing(step_size)),
        ]),
    )
    !isnothing(num_warmup) && push!(attributes, NamedAttribute("num_warmup", num_warmup))
    !isnothing(num_samples) && push!(attributes, NamedAttribute("num_samples", num_samples))
    !isnothing(thinning) && push!(attributes, NamedAttribute("thinning", thinning))
    !isnothing(hmc_config) && push!(attributes, NamedAttribute("hmc_config", hmc_config))
    !isnothing(nuts_config) && push!(attributes, NamedAttribute("nuts_config", nuts_config))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

    return create_operation(
        "enzyme.mcmc",
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
`mh`

Perform a Metropolis-Hastings step on a probabilistic function.
This operation proposes a new trace by regenerating selected addresses,
computes the acceptance probability, and returns the updated trace.
By convention, the 0th operand in inputs is the initial RNG state
and the 0th operand in results is the updated RNG state.
"""
function mh(
    inputs::Vector{Value},
    original_trace::Value;
    new_trace::IR.Type,
    accepted::IR.Type,
    output_rng_state::IR.Type,
    fn,
    selection,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[new_trace, accepted, output_rng_state]
    operands = Value[inputs..., original_trace]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn), NamedAttribute("selection", selection)
    ]
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

    return create_operation(
        "enzyme.mh",
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

"""
`popcount`

Returns the number of 1-bits elementwise.
"""
function popcount(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.popcount",
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
`random`

Generates random numbers using the rng_distribution algorithm and produces
a result tensor.

If rng_distribution = UNIFORM, then the random numbers are generated following
the uniform distribution over the interval [a, b). If a >= b, the behavior is
undefined.

If rng_distribution = NORMAL, then the random numbers are generated following
the normal distribution with mean = a and standard deviation = b. If b < 0,
the behavior is undefined.

If rng_distribution = MULTINORMAL, then the random numbers are generated
following the multivariate normal distribution with mean = a (scalar or vector)
and covariance matrix = b. The parameter b should be a positive definite matrix.

By convention, the 0th operand in inputs is the initial RNG state and the
0th operand in results is the updated RNG state.
"""
function random(
    rng_state::Value,
    a::Value,
    b::Value;
    output_rng_state::IR.Type,
    result::IR.Type,
    rng_distribution,
    location=Location(),
)
    op_ty_results = IR.Type[output_rng_state, result]
    operands = Value[rng_state, a, b]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rng_distribution", rng_distribution),]

    return create_operation(
        "enzyme.random",
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
`randomSplit`

Splits an RNG state into multiple independent RNG states.
Reference: https://github.com/jax-ml/jax/blob/c25e095fcec9678a4ce5f723afce0c6a3c48a5e7/jax/_src/random.py#L281-L294
"""
function randomSplit(
    rng_state::Value; output_rng_states::Vector{IR.Type}, location=Location()
)
    op_ty_results = IR.Type[output_rng_states...,]
    operands = Value[rng_state,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.randomSplit",
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
`recover_sample`

Recover a sample from a flattened position vector.
"""
function recover_sample(position::Value; result::IR.Type, offset, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[position,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("offset", offset),]

    return create_operation(
        "enzyme.recover_sample",
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
`regenerate`

Regenerate selected addresses in a probabilistic function while keeping
other addresses fixed to their values in the given trace.
By convention, the 0th operand in inputs is the initial RNG state
and the 0th operand in results is the updated RNG state.
"""
function regenerate(
    inputs::Vector{Value},
    original_trace::Value;
    trace::IR.Type,
    weight::IR.Type,
    output_rng_state::IR.Type,
    fn,
    selection,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[trace, weight, output_rng_state]
    operands = Value[inputs..., original_trace]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn), NamedAttribute("selection", selection)
    ]
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

    return create_operation(
        "enzyme.regenerate",
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
    support=nothing,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("fn", fn),]
    !isnothing(logpdf) && push!(attributes, NamedAttribute("logpdf", logpdf))
    !isnothing(symbol) && push!(attributes, NamedAttribute("symbol", symbol))
    !isnothing(support) && push!(attributes, NamedAttribute("support", support))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

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

"""
`select`

Extended select operation that supports:
- `tensor<i1>` conditions with differently-sized operands
- `!enzyme.Trace` operands
- standard cases supported by `arith.select`
"""
function select(
    condition::Value,
    true_value::Value,
    false_value::Value;
    result::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[condition, true_value, false_value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.select",
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
    attributes = NamedAttribute[NamedAttribute("fn", fn),]
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

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

function store(value::Value, cache::Value, indices::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, cache, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.store",
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
`triangular_solve`

Solves a system of linear equations with a triangular coefficient matrix.
If left_side=true, solves op(A) @ X = B for X.
If left_side=false, solves X @ op(A) = B for X.
op(A) is determined by transpose_a: NO_TRANSPOSE, TRANSPOSE, or ADJOINT.
"""
function triangular_solve(
    a::Value,
    b::Value;
    result::IR.Type,
    left_side=nothing,
    lower=nothing,
    unit_diagonal=nothing,
    transpose_a=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[a, b]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(left_side) && push!(attributes, NamedAttribute("left_side", left_side))
    !isnothing(lower) && push!(attributes, NamedAttribute("lower", lower))
    !isnothing(unit_diagonal) &&
        push!(attributes, NamedAttribute("unit_diagonal", unit_diagonal))
    !isnothing(transpose_a) && push!(attributes, NamedAttribute("transpose_a", transpose_a))

    return create_operation(
        "enzyme.triangular_solve",
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
    attributes = NamedAttribute[NamedAttribute("fn", fn),]
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

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

"""
`update`

Update selected addresses in a trace with new values from a position vector,
re-evaluate the probabilistic function, and return the updated trace with
the new weight (log probability) and updated RNG state.
By convention, the 0th operand in inputs is the initial RNG state.
"""
function update(
    inputs::Vector{Value},
    original_trace::Value,
    position::Value;
    updated_trace::IR.Type,
    weight::IR.Type,
    output_rng_state::IR.Type,
    fn,
    selection,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[updated_trace, weight, output_rng_state]
    operands = Value[inputs..., original_trace, position]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn), NamedAttribute("selection", selection)
    ]
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

    return create_operation(
        "enzyme.update",
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
`while_loop`

A while loop operation that continues iterating as long as the condition
evaluates to true. Intended to be lowered to `stablehlo.while`.
"""
function while_loop(
    initArgs::Vector{Value};
    results::Vector{IR.Type},
    conditionRegion::Region,
    bodyRegion::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[initArgs...,]
    owned_regions = Region[conditionRegion, bodyRegion]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.while_loop",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function yield(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.yield",
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
