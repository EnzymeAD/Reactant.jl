module impulse
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
        "impulse.cholesky",
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
        "impulse.dot",
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
        "impulse.dynamic_slice",
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
        "impulse.dynamic_update_slice",
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
`for_`

A counted loop operation that iterates from `lowerBound` to `upperBound`
by `step`, carrying `iter_args` through each iteration. The iteration
variable and iter_args are passed to the body region.
"""
function for_(
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
        "impulse.for",
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
        "impulse.generate",
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
        "impulse.if",
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
`infer`

Runs MCMC inference on selected addresses.

Two modes of operation:
1. Trace-based mode: `fn` and `original_trace` are provided. The model
   function with `impulse.sample` ops defines the density.
2. Custom logpdf mode: `logpdf_fn` and `init_position` are provided.
   The logpdf function maps position -> scalar log-density directly.

The `selection` attribute determines which addresses to sample via HMC/NUTS.
All sample addresses are included in the trace tensor for consistency.

Resumable algorithm state (adapted step size, inverse mass matrix, dual
averaging / Welford accumulators, warmup progress, and the final
position/gradient/potential energy) is carried opaquely as a bundle:
`state_in` warm-starts a run (empty = cold start), `state_out` returns it,
and `state_schema` names each slot. `trace`, `diagnostics`, and
`output_rng_state` are the only universal results.
"""
function infer(
    inputs::Vector{Value},
    state_in::Vector{Value},
    original_trace=nothing::Union{Nothing,Value};
    constraint=nothing::Union{Nothing,Value},
    init_inverse_mass_matrix=nothing::Union{Nothing,Value},
    init_step_size=nothing::Union{Nothing,Value},
    init_position=nothing::Union{Nothing,Value},
    init_scale=nothing::Union{Nothing,Value},
    trace::IR.Type,
    diagnostics::IR.Type,
    output_rng_state::IR.Type,
    state_out::Vector{IR.Type},
    fn=nothing,
    selection,
    all_addresses,
    num_warmup=nothing,
    num_samples=nothing,
    thinning=nothing,
    num_chains=nothing,
    total_warmup=nothing,
    hmc_config=nothing,
    nuts_config=nothing,
    vi_config=nothing,
    mh_config=nothing,
    constrained_addresses=nothing,
    logpdf_fn=nothing,
    state_schema=nothing,
    autodiff_attrs=nothing,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[trace, diagnostics, output_rng_state, state_out...]
    operands = Value[inputs..., state_in...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("selection", selection),
        NamedAttribute("all_addresses", all_addresses),
    ]
    !isnothing(original_trace) && push!(operands, original_trace)
    !isnothing(constraint) && push!(operands, constraint)
    !isnothing(init_inverse_mass_matrix) && push!(operands, init_inverse_mass_matrix)
    !isnothing(init_step_size) && push!(operands, init_step_size)
    !isnothing(init_position) && push!(operands, init_position)
    !isnothing(init_scale) && push!(operands, init_scale)
    push!(
        attributes,
        operandsegmentsizes([
            length(inputs),
            length(state_in),
            Int(!isnothing(original_trace)),
            Int(!isnothing(constraint)),
            Int(!isnothing(init_inverse_mass_matrix)),
            Int(!isnothing(init_step_size)),
            Int(!isnothing(init_position)),
            Int(!isnothing(init_scale)),
        ]),
    )
    !isnothing(fn) && push!(attributes, NamedAttribute("fn", fn))
    !isnothing(num_warmup) && push!(attributes, NamedAttribute("num_warmup", num_warmup))
    !isnothing(num_samples) && push!(attributes, NamedAttribute("num_samples", num_samples))
    !isnothing(thinning) && push!(attributes, NamedAttribute("thinning", thinning))
    !isnothing(num_chains) && push!(attributes, NamedAttribute("num_chains", num_chains))
    !isnothing(total_warmup) &&
        push!(attributes, NamedAttribute("total_warmup", total_warmup))
    !isnothing(hmc_config) && push!(attributes, NamedAttribute("hmc_config", hmc_config))
    !isnothing(nuts_config) && push!(attributes, NamedAttribute("nuts_config", nuts_config))
    !isnothing(vi_config) && push!(attributes, NamedAttribute("vi_config", vi_config))
    !isnothing(mh_config) && push!(attributes, NamedAttribute("mh_config", mh_config))
    !isnothing(constrained_addresses) &&
        push!(attributes, NamedAttribute("constrained_addresses", constrained_addresses))
    !isnothing(logpdf_fn) && push!(attributes, NamedAttribute("logpdf_fn", logpdf_fn))
    !isnothing(state_schema) &&
        push!(attributes, NamedAttribute("state_schema", state_schema))
    !isnothing(autodiff_attrs) &&
        push!(attributes, NamedAttribute("autodiff_attrs", autodiff_attrs))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

    return create_operation(
        "impulse.infer",
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
`infer_region`

Region-form of `impulse.infer`, produced by `inline-infer-regions` and
consumed directly by `expand-impulse`:

- `sampler`: the model computation (with `sample_region` ops), the target of
  sampler-region SICM and the source for proposal-based algorithms.
- `logpdf`: the joint logpdf, constructed by the `sicm` pass (after
  sampler-region SICM, so it is born optimized). Its block arguments are
  the position values, one per non-RNG sample result of each selected
  address in `selection` order; everything else it references is captured
  from the enclosing scope and is sample-invariant by construction
  (hoisted invariants, model inputs, and pinned trace values of unselected
  sites). SICM runs a second time on this region; `expand-impulse` then
  clones it as the potential-energy body inside the materialized
  HMC/NUTS loop.

Mirrors `impulse.infer`\'s operands/results/attributes (created by a
wholesale attribute copy) plus the two regions and the
`num_position_args`/`position_size` bookkeeping for the joint logpdf.
"""
function infer_region(
    inputs::Vector{Value},
    state_in::Vector{Value},
    original_trace=nothing::Union{Nothing,Value};
    constraint=nothing::Union{Nothing,Value},
    init_inverse_mass_matrix=nothing::Union{Nothing,Value},
    init_step_size=nothing::Union{Nothing,Value},
    init_position=nothing::Union{Nothing,Value},
    init_scale=nothing::Union{Nothing,Value},
    trace::IR.Type,
    diagnostics::IR.Type,
    output_rng_state::IR.Type,
    state_out::Vector{IR.Type},
    fn=nothing,
    selection,
    all_addresses,
    num_warmup=nothing,
    num_samples=nothing,
    thinning=nothing,
    num_chains=nothing,
    total_warmup=nothing,
    hmc_config=nothing,
    nuts_config=nothing,
    vi_config=nothing,
    mh_config=nothing,
    constrained_addresses=nothing,
    logpdf_fn=nothing,
    state_schema=nothing,
    autodiff_attrs=nothing,
    name=nothing,
    num_position_args=nothing,
    position_size=nothing,
    sampler::Region,
    logpdf::Region,
    location=Location(),
)
    op_ty_results = IR.Type[trace, diagnostics, output_rng_state, state_out...]
    operands = Value[inputs..., state_in...]
    owned_regions = Region[sampler, logpdf]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("selection", selection),
        NamedAttribute("all_addresses", all_addresses),
    ]
    !isnothing(original_trace) && push!(operands, original_trace)
    !isnothing(constraint) && push!(operands, constraint)
    !isnothing(init_inverse_mass_matrix) && push!(operands, init_inverse_mass_matrix)
    !isnothing(init_step_size) && push!(operands, init_step_size)
    !isnothing(init_position) && push!(operands, init_position)
    !isnothing(init_scale) && push!(operands, init_scale)
    push!(
        attributes,
        operandsegmentsizes([
            length(inputs),
            length(state_in),
            Int(!isnothing(original_trace)),
            Int(!isnothing(constraint)),
            Int(!isnothing(init_inverse_mass_matrix)),
            Int(!isnothing(init_step_size)),
            Int(!isnothing(init_position)),
            Int(!isnothing(init_scale)),
        ]),
    )
    !isnothing(fn) && push!(attributes, NamedAttribute("fn", fn))
    !isnothing(num_warmup) && push!(attributes, NamedAttribute("num_warmup", num_warmup))
    !isnothing(num_samples) && push!(attributes, NamedAttribute("num_samples", num_samples))
    !isnothing(thinning) && push!(attributes, NamedAttribute("thinning", thinning))
    !isnothing(num_chains) && push!(attributes, NamedAttribute("num_chains", num_chains))
    !isnothing(total_warmup) &&
        push!(attributes, NamedAttribute("total_warmup", total_warmup))
    !isnothing(hmc_config) && push!(attributes, NamedAttribute("hmc_config", hmc_config))
    !isnothing(nuts_config) && push!(attributes, NamedAttribute("nuts_config", nuts_config))
    !isnothing(vi_config) && push!(attributes, NamedAttribute("vi_config", vi_config))
    !isnothing(mh_config) && push!(attributes, NamedAttribute("mh_config", mh_config))
    !isnothing(constrained_addresses) &&
        push!(attributes, NamedAttribute("constrained_addresses", constrained_addresses))
    !isnothing(logpdf_fn) && push!(attributes, NamedAttribute("logpdf_fn", logpdf_fn))
    !isnothing(state_schema) &&
        push!(attributes, NamedAttribute("state_schema", state_schema))
    !isnothing(autodiff_attrs) &&
        push!(attributes, NamedAttribute("autodiff_attrs", autodiff_attrs))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    !isnothing(num_position_args) &&
        push!(attributes, NamedAttribute("num_position_args", num_position_args))
    !isnothing(position_size) &&
        push!(attributes, NamedAttribute("position_size", position_size))

    return create_operation(
        "impulse.infer_region",
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
        "impulse.log_add_exp",
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
        "impulse.logistic",
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

An optional `constraint` tensor (with `constrained_addresses`) supplies
observed values, forwarded verbatim to the internal `regenerate`.
"""
function mh(
    original_trace::Value,
    original_weight::Value,
    inputs::Vector{Value},
    constraint=nothing::Union{Nothing,Value};
    new_trace::IR.Type,
    new_weight::IR.Type,
    accepted::IR.Type,
    output_rng::IR.Type,
    fn,
    selection,
    regenerate_addresses,
    constrained_addresses=nothing,
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
    !isnothing(constraint) && push!(operands, constraint)
    push!(
        attributes, operandsegmentsizes([1, 1, length(inputs), Int(!isnothing(constraint))])
    )
    !isnothing(constrained_addresses) &&
        push!(attributes, NamedAttribute("constrained_addresses", constrained_addresses))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

    return create_operation(
        "impulse.mh",
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
        "impulse.popcount",
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
        "impulse.random",
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
        "impulse.randomSplit",
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

An optional `constraint` tensor (with `constrained_addresses` describing its
layout) supplies observed values for constrained sample sites. Constrained
sites are always scored/written from the constraint value when their branch
is taken, rather than reused from the old trace slot. This is what keeps MH
correct when a proposed value flips which `impulse.if` branch executes: a
site that was previously not taken (its trace slot is stale) is refreshed
from the constraint instead of the stale slot.

Returns: (new_trace, weight, retvals...)
- new_trace: tensor<1 x position_size x f64> - flattened samples
- weight: tensor<f64> - accumulated log probability
- retvals: original function return values
"""
function regenerate(
    inputs::Vector{Value},
    original_trace::Value,
    constraint=nothing::Union{Nothing,Value};
    new_trace::IR.Type,
    weight::IR.Type,
    outputs::Vector{IR.Type},
    fn,
    selection,
    regenerate_addresses,
    constrained_addresses=nothing,
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
    !isnothing(constraint) && push!(operands, constraint)
    push!(attributes, operandsegmentsizes([length(inputs), 1, Int(!isnothing(constraint))]))
    !isnothing(constrained_addresses) &&
        push!(attributes, NamedAttribute("constrained_addresses", constrained_addresses))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

    return create_operation(
        "impulse.regenerate",
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
        "impulse.reshape",
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
        "impulse.sample",
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
`sample_region`

Region-form of `impulse.sample`: the sampler and logpdf function bodies are
inlined as nested regions, enabling sample-invariant code motion (SICM) to
analyze across sample sites. The `logpdf` region is cleared once its body has
been merged into the enclosing `infer_region`\'s joint logpdf.

The `address` attribute is the site\'s full address as an array of symbols;
submodel dissolution prepends the enclosing site\'s address, so entries of
the `infer_region`\'s `selection`/`all_addresses` match by attribute equality.
"""
function sample_region(
    inputs::Vector{Value};
    outputs::Vector{IR.Type},
    fn=nothing,
    logpdf_fn=nothing,
    address=nothing,
    support=nothing,
    name=nothing,
    sampler::Region,
    logpdf::Region,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[sampler, logpdf]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fn) && push!(attributes, NamedAttribute("fn", fn))
    !isnothing(logpdf_fn) && push!(attributes, NamedAttribute("logpdf_fn", logpdf_fn))
    !isnothing(address) && push!(attributes, NamedAttribute("address", address))
    !isnothing(support) && push!(attributes, NamedAttribute("support", support))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))

    return create_operation(
        "impulse.sample_region",
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
        "impulse.select",
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
        "impulse.simulate",
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
        "impulse.slice",
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
        "impulse.triangular_solve",
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
        "impulse.untracedCall",
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
`while_`

A while loop operation that continues iterating as long as the condition
evaluates to true. Intended to be lowered to `stablehlo.while`.
"""
function while_(
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
        "impulse.while",
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
        "impulse.yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # impulse
