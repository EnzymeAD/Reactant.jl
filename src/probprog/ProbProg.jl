module ProbProg

using ..Reactant: MLIR, TracedUtils, AbstractRNG, Ops
using ReactantCore: @trace

# F2 spike flag (attr-level, default OFF): when true, distributions that the
# shared bijector/distribution library can emit (currently Normal as
# StdNormal + Affine) attach a `#impulse.dist` attribute to their sample site,
# routing lowering through the C++ builder's emitted sampler/logpdf instead of
# the traced sampler/logpdf symbols.
const EMIT_DISTRIBUTIONS = Ref(false)

include("Types.jl")
include("Distributions.jl")
include("FFI.jl")
include("Modeling.jl")
include("Display.jl")
include("Stats.jl")
include("MH.jl")
include("MCMC.jl")
include("VI.jl")

# Types.
export Trace, Constraint, Selection, Address, TraceEntry, TracedTrace, MCMCState, VIState

# Distributions.
export Distribution,
    Normal,
    Exponential,
    HalfNormal,
    LogNormal,
    Bernoulli,
    Poisson,
    MultivariateNormal,
    MultiNormal,
    Gamma,
    StudentT,
    HalfCauchy,
    Dirichlet,
    DiscreteUniform

# Utility functions.
export get_choices,
    select, unflatten_trace, filter_entries_by_selection, with_trace, flatten_constraint

# MCMC Statistics.
export mcmc_summary

# Core MLIR ops.
export sample,
    if_,
    untraced_call,
    simulate,
    generate,
    mh,
    mcmc,
    mcmc_logpdf,
    svi,
    save_state,
    load_state,
    run_chain

# Gen-like helper functions.
export simulate_, generate_

# Debug utilities.
export clear_dump_buffer!, show_dumps

end
