module ProbProg

using ..Reactant: MLIR, TracedUtils, AbstractRNG

include("Types.jl")
include("distributions/distributions.jl")

# Re-export the new Reactant-friendly distributions from the `Distributions`
# submodule into ProbProg so existing code can keep using
# `ProbProg.Normal`, `ProbProg.Exponential`, etc.
using .Distributions:
    AbstractDistribution,
    StdNormal,
    StdExponential,
    StdUniform,
    StdInverseGamma,
    StdTDist,
    AffineDistribution,
    TransformedDistribution,
    AbstractTransform,
    ConstantJacobianTransform,
    IdentityTransform,
    AffineTransform,
    LogTransform,
    LogitTransform,
    ComposeTransform,
    forward,
    inverse,
    logabsdetjac,
    Normal,
    Exponential,
    Uniform,
    InverseGamma,
    TDist,
    LogNormal,
    LogitNormal,
    Bernoulli,
    logpdf,
    unnormed_logpdf,
    lognorm,
    insupport,
    params,
    cdf,
    quantile,
    AbstractSupport,
    RealSupport,
    PositiveSupport,
    UnitIntervalSupport,
    IntervalSupport,
    GreaterThanSupport,
    LessThanSupport,
    SimplexSupport,
    LowerCholeskySupport

# `Distribution` is the legacy alias for the abstract supertype that
# `Modeling.jl` and other probprog code dispatches on (`<:Distribution`).
const Distribution = AbstractDistribution

# Trait API used by `Modeling.jl:118-124` — `sampler` / `logpdf_fn` /
# `support` / `bounds`. Function stubs and defaults are declared in
# `distributions/distributions.jl`; per-distribution implementations live
# in each `std_*.jl`; the trait dispatch (which needs both
# `TransformedDistribution`/`AffineDistribution` and the base type) lives
# at the bottom of `distributions/transformed.jl`.
using .Distributions: sampler, logpdf_fn, support, bounds

include("FFI.jl")
include("Modeling.jl")
include("Display.jl")
include("Stats.jl")
include("MH.jl")
include("MCMC.jl")

# Types.
export Trace, Constraint, Selection, Address, TraceEntry, TracedTrace, MCMCState

# Distributions.
export Distribution,
    AbstractDistribution,
    StdNormal,
    StdExponential,
    StdUniform,
    StdInverseGamma,
    StdTDist,
    AffineDistribution,
    TransformedDistribution,
    AbstractTransform,
    ConstantJacobianTransform,
    IdentityTransform,
    AffineTransform,
    LogTransform,
    LogitTransform,
    ComposeTransform,
    forward,
    inverse,
    logabsdetjac,
    Normal,
    Exponential,
    Uniform,
    InverseGamma,
    TDist,
    LogNormal,
    LogitNormal,
    Bernoulli,
    logpdf,
    unnormed_logpdf,
    lognorm,
    insupport,
    params,
    cdf,
    quantile,
    AbstractSupport,
    RealSupport,
    PositiveSupport,
    UnitIntervalSupport,
    IntervalSupport,
    GreaterThanSupport,
    LessThanSupport,
    SimplexSupport,
    LowerCholeskySupport

# Utility functions.
export get_choices,
    select, unflatten_trace, filter_entries_by_selection, with_trace, flatten_constraint

# MCMC Statistics.
export mcmc_summary

# Core MLIR ops.
export sample,
    untraced_call,
    simulate,
    generate,
    mh,
    mcmc,
    mcmc_logpdf,
    save_state,
    load_state,
    run_chain

# Gen-like helper functions.
export simulate_, generate_

# Debug utilities.
export clear_dump_buffer!, show_dumps

end
