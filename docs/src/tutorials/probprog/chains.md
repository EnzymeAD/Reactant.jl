# [Running and resuming chains](@id probprog-chains)

The [`mcmc`](@ref) and [`mcmc_logpdf`](@ref) ops give you one compiled kernel
that takes `num_samples` and runs all of them. That is the right granularity
when you want the tightest compiled loop; it is less convenient when you
want a progress bar, a very long chain you do not want to re-JIT after every
hang-up, or a resumable inference session.

[`run_chain`](@ref), [`MCMCState`](@ref), [`save_state`](@ref), and
[`load_state`](@ref) are the higher-level drivers for those cases.

!!! note

    On the current branch, [`run_chain`](@ref) drives a user-supplied
    log-density via [`mcmc_logpdf`](@ref). A trace-based mode is tracked in
    `#2619` and not yet exposed. For trace-based chains today, call
    [`mcmc`](@ref) with an appropriate `num_samples` and unflatten the
    resulting tensor yourself (see [the MCMC tutorial](@ref probprog-mcmc)).

## `run_chain`

Drives a chain in fixed-size chunks, yielding intermediate `MCMCState`s
for checkpointing.

```julia
using Reactant
using Reactant: ProbProg, ReactantRNG

seed = Reactant.to_rarray(UInt64[1, 5])
rng  = ReactantRNG(seed)

initial_position = Reactant.to_rarray(zeros(d))

samples, state = ProbProg.run_chain(
    rng, logpdf_fn, initial_position;
    algorithm         = :NUTS,
    num_warmup        = 1000,
    num_samples       = 5000,
    chunk_size        = 500,
    step_size         = nothing,               # let adaptation pick
    inverse_mass_matrix = nothing,             # identity to start
    adapt_step_size   = true,
    adapt_mass_matrix = true,
    progress_bar      = true,
    max_tree_depth    = 10,
    max_delta_energy  = 1000.0,
    thinning          = 1,
    trajectory_length = 2π,
)
```

`samples` is a plain `Array{Float64,2}` of shape
`(num_samples ÷ thinning, d)`. `state` is an [`MCMCState`](@ref) you can
save, ship to another process, or feed back in to continue sampling.

### What chunking does

`run_chain` splits the post-warmup samples into chunks of `chunk_size`. The
first chunk also carries the warmup; each later chunk is a pure sampling
kernel with adaptation turned off (so warmup only happens once, at the
start). Between chunks `run_chain` prints a progress bar with the current
step size and running acceptance rate. For production runs where you want
a single fused kernel and no Julia-level printing, set `progress_bar =
false`; the driver then compiles one monolithic function.

## `MCMCState`

Every inference entry point returns an [`MCMCState`](@ref): sampler states need to resume inference.
It packs everything needed from exactly where
the previous call left off:

```julia
state.position           # last accepted position (1 × d)
state.gradient           # gradient of the log-density at that position
state.potential_energy   # -log density at that position (scalar)
state.step_size          # current (or adapted-final) leapfrog step size
state.inverse_mass_matrix
state.rng                # RNG seed snapshot
```

The state-based call forms take it as the first argument and reuse its
fields as defaults, with warmup/adaptation disabled by default:

```julia
samples, state = ProbProg.run_chain(state, logpdf_fn;
    algorithm   = :NUTS,
    num_samples = 10_000,
    chunk_size  = 1000,
)

# Or trace-based:
trace, diag, _, state = ProbProg.mcmc(
    state, trace, model, args...;
    selection = ProbProg.select(ProbProg.Address(:θ)),
    num_samples = 500,
)

# Or logpdf-based:
samples, diag, _, state = ProbProg.mcmc_logpdf(
    state, logpdf_fn, args...;
    num_samples = 500,
)
```

In every state-based form, `num_warmup` defaults to `0` and both adaptation
flags default to `false`. Override explicitly if you want another warmup
window (e.g. after changing the model).

## Checkpointing

`MCMCState` serialises to disk with [`save_state`](@ref) and reloads with
[`load_state`](@ref):

```julia
ProbProg.save_state("chain.jls", state)

# … in a fresh session …
using Reactant
using Reactant: ProbProg

state = ProbProg.load_state("chain.jls")
samples, state = ProbProg.run_chain(state, logpdf_fn;
    num_samples = 5000, chunk_size = 500,
)
```

Under the hood this uses Julia's `Serialization` module on a dictionary of
arrays, so it is cross-session but version-sensitive to Reactant itself.
For long-term archival, save the raw `position`, `inverse_mass_matrix`,
`step_size`, and `rng` fields yourself.

### `save_state`

Serialises an `MCMCState` to disk via Julia's `Serialization` module.

### `load_state`

Reloads an `MCMCState` previously written with `save_state`.

## `mcmc_summary`

[`mcmc_summary`](@ref) takes either a sample matrix or a [`Trace`](@ref) and
produces a per-parameter summary table: mean, std, median, 5%/95% quantiles,
effective sample size, and `r_hat`.

From a sample matrix:

```julia
ProbProg.mcmc_summary(samples)

# or with explicit names:
ProbProg.mcmc_summary(samples; names = ["β0", "β1", "σ"])
```

From a `Trace` produced by [`mcmc`](@ref) + `unflatten_trace`:

```julia
ProbProg.mcmc_summary(trace)
```

When printed, the result renders as an aligned table via `PrettyTables`:

```
             mean    std   median   5.0%   95.0%   n_eff   r_hat
    β0       1.02   0.15    1.02    0.77    1.26    487    1.00
    β1      -0.47   0.08   -0.47   -0.60   -0.34    511    1.00
    σ        0.94   0.04    0.94    0.88    1.01    623    1.00
```

Indexing by name is supported: `summary[:β0]` returns a
`ParameterSummary` with fields `mean`, `std`, `median`, `q5`, `q95`,
`n_eff`, `r_hat`.

## Putting it together

A full logistic-regression workflow might look like this:

```julia
using Reactant
using Reactant: ProbProg, ReactantRNG

seed = Reactant.to_rarray(UInt64[1, 5])
rng  = ReactantRNG(seed)

function logpdf(θ, X, y)
    logits = X * θ
    ll = sum(y .* logits .- max.(logits, 0.0) .- log1p.(exp.(.-abs.(logits))))
    pr = -0.5 * sum(θ.^2)
    return ll + pr
end

d = size(X, 2)
initial_position = Reactant.to_rarray(zeros(d))

samples, state = ProbProg.run_chain(
    rng, logpdf, initial_position, X, y;
    algorithm         = :NUTS,
    num_warmup        = 1000,
    num_samples       = 5000,
    adapt_step_size   = true,
    adapt_mass_matrix = true,
)

ProbProg.save_state("logistic.jls", state)

names = ["β$i" for i in 0:d-1]
ProbProg.mcmc_summary(samples; names=names)
```
