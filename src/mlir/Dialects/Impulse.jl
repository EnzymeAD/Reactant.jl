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

Computes the Cholesky factorization of the symmetric positive-definite
matrix in the trailing two dimensions of `input` (any leading dimensions are
treated as a batch). If `lower` is true, returns the lower-triangular L with
A = L @ L^T; if false, returns the upper-triangular U with A = U^T @ U. The
strictly-opposite triangle of the result is zero. The input must be symmetric
positive definite; the result is otherwise unspecified. Result shape equals
the input shape.

Note: lowered 1:1 to `stablehlo.cholesky` with the same `lower` flag.
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

Generalized tensor contraction of `lhs` and `rhs`. Contracts the paired
`lhs_contracting_dimensions`/`rhs_contracting_dimensions` (which must have
equal size in each pair) by summing the elementwise products over them, and
batches over the paired `lhs_batching_dimensions`/`rhs_batching_dimensions`
(also equal-sized per pair). The result shape is, in order: the batch
dimensions, the remaining (free) `lhs` dimensions, then the remaining (free)
`rhs` dimensions. A dimension index may appear at most once across the
batching and contracting lists for a given operand.

Note: lowered 1:1 to `stablehlo.dot_general` with the same dimension numbers.
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

Extracts a sub-tensor of static shape `slice_sizes` from `operand`, starting
at the runtime `start_indices` (one scalar index tensor per operand
dimension). Before use, each start index is clamped into
`[0, operand_dim - slice_sizes[d]]` so the whole slice stays in bounds, so
out-of-range indices shift the window to the nearest fully-in-bounds position
rather than reading out of bounds. The result shape is `slice_sizes`.

Note: lowered 1:1 to `stablehlo.dynamic_slice`, whose clamping semantics this
matches.
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

Produces a copy of `operand` with `update` written in starting at the runtime
`start_indices` (one scalar index tensor per operand dimension). Before use,
each start index is clamped into `[0, operand_dim - update_dim[d]]` so the
entire `update` lands in bounds, so out-of-range indices shift the write
window to the nearest fully-in-bounds position rather than writing out of
bounds. The result has the shape of `operand`.

Note: lowered 1:1 to `stablehlo.dynamic_update_slice`, whose clamping
semantics this matches.
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
and `state_schema` names each slot. `trace`, `diagnostics`,
`log_densities`, and `output_rng_state` are the only universal results.
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
    log_densities::IR.Type,
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
    op_ty_results = IR.Type[
        trace, diagnostics, log_densities, output_rng_state, state_out...
    ]
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
    log_densities::IR.Type,
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
    op_ty_results = IR.Type[
        trace, diagnostics, log_densities, output_rng_state, state_out...
    ]
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
`lgamma`

Elementwise log|Gamma(x)|. Lowered to `chlo.lgamma` in Enzyme-JAX; emitted by
the distribution builder for logpdf normalizers that need it (Gamma, StudentT,
Poisson).
"""
function lgamma(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "impulse.lgamma",
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

Elementwise `log(exp(lhs) + exp(rhs))`, computed in a numerically stable way
as `max(lhs, rhs) + log1p(exp(-|lhs - rhs|))`. When the difference is not
finite (e.g. an infinite operand) the result falls back to `lhs + rhs`.
`lhs` and `rhs` share the result shape and element type.

Note: there is no single StableHLO op; this lowers to the max/log1p
expansion above.
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

Elementwise logistic (sigmoid) function `1 / (1 + exp(-x))` over `operand`.
Result and operand share shape and element type.

Note: lowered 1:1 to `stablehlo.logistic`.
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

Elementwise population count: each result element is the number of set bits
(1-bits) in the corresponding integer `operand` element. Operand and result
are integer tensors of the same shape and element type.

Note: lowered 1:1 to `stablehlo.popcnt`.
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

Draws a result tensor of random numbers from `rng_state` and returns the
advanced RNG state alongside the samples. Uniform bits are produced by a
counter-based bit generator keyed by `rng_state`; the parametric distributions
below are constructed deterministically from those bits, so the same input
state reproduces the same samples.

If rng_distribution = UNIFORM, samples are drawn from the uniform distribution
over [a, b). A base draw in [0, 1) is formed from the raw bits (assembling the
mantissa of a float in [1, 2) and subtracting 1) and affine-mapped to
[a, b). If a >= b, the behavior is undefined.

If rng_distribution = NORMAL, samples follow the normal distribution with
mean = a and standard deviation = b, formed from a uniform draw u in [0, 1)
via the probit `sqrt(2) * erfinv(2u - 1)`, then scaled by b and shifted by a.
If b < 0, the behavior is undefined.

If rng_distribution = MULTINORMAL, samples follow the multivariate normal with
mean = a (scalar or vector) and covariance matrix = b: a standard-normal vector
is drawn as above and mapped through the Cholesky factor of b, then shifted by
a. The parameter b must be positive definite.

By convention, the 0th operand in inputs is the initial RNG state and the
0th operand in results is the updated RNG state.

Note: lowered using `stablehlo.rng_bit_generator` (DEFAULT algorithm) for the
bits, with `chlo.erf_inv` for the probit and `stablehlo.cholesky` for
MULTINORMAL.
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

Derives N independent RNG states (N = number of results) from a single input
`rng_state` by hashing it with the Threefry2x32 counter-based PRF (five
rounds), matching JAX\'s `random.split`. Each output state is statistically
independent and deterministically reproducible from the input state.
Reference: https://github.com/jax-ml/jax/blob/c25e095fcec9678a4ce5f723afce0c6a3c48a5e7/jax/_src/random.py#L281-L294

Note: lowered to the inlined Threefry2x32 round sequence over StableHLO
integer ops (no single StableHLO op).
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

The trace TYPE selects the weight semantics. A bare trace
(tensor<1 x position_size x f64>) yields the full joint log probability.
A score-tail trace (width = position_size + one memoized score scalar per
site) yields the accept-ratio DELTA sum_dirty(logp_new - stored_score): a
density term is dirty iff its site is regenerated or its parameters are
dataflow-reachable from a regenerated site\'s value; clean terms cancel in
the Metropolis--Hastings ratio and are never emitted, and old scores are
read from the tail instead of recomputed. The caller maintains the carried
weight as `w + delta` on accept, and the returned trace carries refreshed
scores.

Returns: (new_trace, weight, retvals...)
- new_trace: same type as the given trace - flattened samples
  (plus refreshed site scores for score-tail traces)
- weight: tensor<f64> - full joint or delta, per the trace type
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

"""
`reshape`

Reinterprets `input` as a tensor of the result type\'s static shape,
preserving the linear (row-major, last-dimension-fastest) order of elements.
The result shape is fully static and its total element count must equal that
of `input`; no elements are added, dropped, or reordered.

Note: lowered 1:1 to `stablehlo.reshape`.
"""
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

The sampler symbol `fn` is optional: a site that carries a `#impulse.dist`
attribute derives its sampler and logpdf from the shared distribution builder,
so it needs no traced `fn`/`logpdf` symbols and its result types are the ones
the frontend constructs directly from the distribution\'s shape. A site with a
traced sampler (the rejection-loop natives, generative sub-calls, or the
custom-logpdf escape hatch) carries `fn`; a custom-logpdf site additionally
carries `logpdf`. The verifier requires either `fn` or `dist` to be present.

Round-trip limitation: with `fn` absent the pretty form prints as
`impulse.sample(%inputs) ...`, which the generic MLIR parser cannot round-trip
(the leading `(` after the op name is read as the start of a type). Passes
operate on the in-memory op, so this affects only textual `.mlir` inputs; write
an fn-absent site in generic op form (`\"impulse.sample\"(...) {...} : ...`) when
a test must parse one from text.
"""
function sample(
    inputs::Vector{Value};
    outputs::Vector{IR.Type},
    fn=nothing,
    logpdf=nothing,
    symbol=nothing,
    support=nothing,
    dist=nothing,
    name=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fn) && push!(attributes, NamedAttribute("fn", fn))
    !isnothing(logpdf) && push!(attributes, NamedAttribute("logpdf", logpdf))
    !isnothing(symbol) && push!(attributes, NamedAttribute("symbol", symbol))
    !isnothing(support) && push!(attributes, NamedAttribute("support", support))
    !isnothing(dist) && push!(attributes, NamedAttribute("dist", dist))
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
    dist=nothing,
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
    !isnothing(dist) && push!(attributes, NamedAttribute("dist", dist))
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

Elementwise selection between `true_value` and `false_value`: each result
element is the corresponding `true_value` element where `condition` is true,
otherwise the `false_value` element. As an extension over `arith.select`, a
rank-0 `tensor<i1>` `condition` selects between the two operands as whole
tensors, so the operands may have a shape that differs from a per-element
condition. When `condition` matches the operand shape, selection is per
element. `true_value` and `false_value` share the result\'s shape and element
type.

Note: lowered 1:1 to `stablehlo.select`.
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

Extracts a statically-shaped sub-tensor of `operand`. In each dimension `d`
it takes the elements at indices `start_indices[d] + k*strides[d]` for
`k = 0, 1, ...` while the index stays below `limit_indices[d]`, yielding
`ceil((limit_indices[d] - start_indices[d]) / strides[d])` elements. All
three index arrays are static and have one entry per operand dimension; the
result shape follows from them and must be in bounds
(`0 <= start <= limit <= dim_size`, `strides >= 1`).

Note: lowered 1:1 to `stablehlo.slice`.
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

Solves a linear system whose coefficient matrix `a` is triangular, returning
X with the shape of `b`. Only the triangle of `a` selected by `lower` is
read (lower-triangular if `lower` is true, else upper); when `unit_diagonal`
is true the diagonal of `a` is taken to be all ones and its stored diagonal
is ignored. `op(a)` is `a`, `a^T`, or `a^H` according to `transpose_a`
(NO_TRANSPOSE, TRANSPOSE, ADJOINT). If `left_side` is true, solves
`op(a) @ X = b`; if false, solves `X @ op(a) = b`. `a` operates on the
trailing two dimensions; leading dimensions are batched.

Note: lowered 1:1 to `stablehlo.triangular_solve` with the same attributes.
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
