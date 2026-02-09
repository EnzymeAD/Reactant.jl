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
`dynamic_slice`

Extract a slice from a tensor at dynamic start indices.
"""
function dynamic_slice(
    operand::Value,
    start_indices::Vector{Value};
    result::IR.Type,
    slice_sizes,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, start_indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("slice_sizes", slice_sizes),]

    return create_operation(
        "enzyme.dynamic_slice",
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
`dynamic_update_slice`

Update a slice in a tensor at dynamic start indices.
"""
function dynamic_update_slice(
    operand::Value,
    update::Value,
    start_indices::Vector{Value};
    result::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, update, start_indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.dynamic_update_slice",
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

Generates from a generative function with some addresses constrained.
The constraint tensor contains flattened constrained values in the order
specified by constrained_addresses.

Returns: (trace, weight, rng, retvals...)
"""
function generate(
    inputs::Vector{Value},
    constraint::Value;
    trace::IR.Type,
    weight::IR.Type,
    outputs::Vector{IR.Type},
    fn,
    selection,
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
        NamedAttribute("selection", selection),
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

Runs MCMC inference on selected addresses.

The `selection` attribute determines which addresses to sample via HMC/NUTS.
All sample addresses are included in the trace tensor for consistency.

Returns: (trace, diagnostics, rng)
- trace: tensor<num_samples x position_size x f64>
- diagnostics: tensor<num_samples x i1> - placeholder for future expansion
- rng: updated RNG state
"""
function mcmc(
    inputs::Vector{Value},
    original_trace::Value,
    inverse_mass_matrix=nothing::Union{Nothing,Value};
    step_size=nothing::Union{Nothing,Value},
    trace::IR.Type,
    diagnostics::IR.Type,
    output_rng_state::IR.Type,
    fn,
    selection,
    all_addresses,
    num_warmup=nothing,
    num_samples=nothing,
    thinning=nothing,
    hmc_config=nothing,
    nuts_config=nothing,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[trace, diagnostics, output_rng_state]
    operands = Value[inputs..., original_trace]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn),
        NamedAttribute("selection", selection),
        NamedAttribute("all_addresses", all_addresses),
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

Performs one MH step: regenerates selected addresses and accepts/rejects
based on weight ratio.
"""
function mh(
    original_trace::Value,
    original_weight::Value,
    inputs::Vector{Value};
    new_trace::IR.Type,
    new_weight::IR.Type,
    accepted::IR.Type,
    output_rng::IR.Type,
    fn,
    selection,
    regenerate_addresses,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[new_trace, new_weight, accepted, output_rng]
    operands = Value[original_trace, original_weight, inputs...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn),
        NamedAttribute("selection", selection),
        NamedAttribute("regenerate_addresses", regenerate_addresses),
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
`regenerate`

Regenerates selected addresses while keeping others fixed.
Used internally by MH.

Takes explicit old_trace and returns new trace with weight.

Returns: (new_trace, weight, retvals...)
- new_trace: tensor<1 x position_size x f64> - flattened samples
- weight: tensor<f64> - accumulated log probability
- retvals: original function return values
"""
function regenerate(
    inputs::Vector{Value},
    original_trace::Value;
    new_trace::IR.Type,
    weight::IR.Type,
    outputs::Vector{IR.Type},
    fn,
    selection,
    regenerate_addresses,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[new_trace, weight, outputs...]
    operands = Value[inputs..., original_trace]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn),
        NamedAttribute("selection", selection),
        NamedAttribute("regenerate_addresses", regenerate_addresses),
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

function reshape(input::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.reshape",
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

Simulates a generative function, building a trace tensor containing all
sampled values and computing the accumulated log probability weight.

The `selection` attribute specifies all sample addresses in order,
determining the trace tensor layout.

Returns: (trace, weight, rng, retvals...)
- trace: tensor<1 x position_size x f64> - flattened samples
- weight: tensor<f64> - accumulated log probability
- rng: updated RNG state
- retvals: original function return values
"""
function simulate(
    inputs::Vector{Value};
    trace::IR.Type,
    weight::IR.Type,
    outputs::Vector{IR.Type},
    fn,
    selection,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[trace, weight, outputs...]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn), NamedAttribute("selection", selection)
    ]
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

"""
`slice`

Extract a static slice from a tensor.
"""
function slice(
    operand::Value;
    result::IR.Type,
    start_indices,
    limit_indices,
    strides,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("start_indices", start_indices),
        NamedAttribute("limit_indices", limit_indices),
        NamedAttribute("strides", strides),
    ]

    return create_operation(
        "enzyme.slice",
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
