# [Probabilistic Programming API](@id probprog-api)

Exports from `Reactant.ProbProg`. Names are qualified as `ProbProg.<name>`;
`using Reactant: ProbProg` unqualifies them. Usage shown in the
[tutorials](@ref probprog).

## Types

### `Trace`

[Gen-style trace](https://www.gen.dev/docs/stable/ref/core/gfi/#Traces) of one model execution:

```julia
mutable struct Trace
    choices::Dict{Symbol, Any}
    retval::Any
    weight::Any
    subtraces::Dict{Symbol, Any}
end

ProbProg.Trace()
```

- `choices`: per-address values; first axis indexes samples.
- `subtraces`: nested `Trace`s keyed by outer `symbol`.
- `retval`: model return.
- `weight`: prior log-density (`simulate`) or importance weight
  (`generate`).

See [`get_choices`](@ref), [`unflatten_trace`](@ref).

### `Address`

Paths of symbols used to index into the trace.

```julia
ProbProg.Address(:slope)
ProbProg.Address(:outer, :inner, :x) # nested
```

### `Selection`

[Gen-style selection](https://www.gen.dev/docs/stable/ref/core/selections/) of addresses that is used to specify which random choices should be included in the inference operation. Constructed via [`select`](@ref).

### `Constraint`

`Address`-keyed dict of observations. Similar to Gen's [`ChoiceMap`](https://www.gen.dev/docs/stable/ref/core/choice_maps/).

```julia
obs = ProbProg.Constraint(
    :y     => [0.1, 0.2, 0.3],
    :outer => :inner => [1.0],
)
```

### `TraceEntry`

Layout metadata for one sampled site, used to reconstruct the trace from a flat tensor representation.

```julia
struct TraceEntry
    symbol::Symbol
    shape::Tuple
    num_elements::Int
    offset::Int
    parent_path::Vector{Symbol}
end
```

Auto-generated during tracing. Consumed by [`unflatten_trace`](@ref) and
[`filter_entries_by_selection`](@ref).

### `TracedTrace`

Per-trace context. Returned by [`with_trace`](@ref).

### `MCMCState`

Sampler states needed to resume probabilistic inference from a previous state.

```julia
mutable struct MCMCState
    position
    gradient
    potential_energy
    step_size
    inverse_mass_matrix
    rng
end
```

Used by [`mcmc`](@ref), [`mcmc_logpdf`](@ref),
[`run_chain`](@ref). Serialized and deserialized by [`save_state`](@ref) / [`load_state`](@ref).

## Distributions

Built-in distributions:

| Constructor | Default |
|-------------|---------|
| `ProbProg.Normal(μ, σ, shape)`     | `Normal()` → `μ=0, σ=1, shape=(1,)` |
| `ProbProg.Exponential(λ, shape)`   | `Exponential()` → `λ=1, shape=(1,)` |
| `ProbProg.LogNormal(μ, σ, shape)`  | `LogNormal()` → `μ=0, σ=1, shape=(1,)` |
| `ProbProg.Bernoulli(logits, shape)`| — |

Each registers a sampler and log-density; `ProbProg.sample(rng, dist)`
needs no extra arguments.

One can also define custom distributions by implementing the following methods:
- `sampler(::Type{<:Distribution})`
- `logpdf_fn(::Type{<:Distribution})`
- `params(::Distribution)`
- `support(::Type{<:Distribution})` (for HMC/NUTS automatic constraint transformation)
- `bounds(::Type{<:Distribution})` (for HMC/NUTS automatic constraint transformation)

See the [documentation](@ref distributions) for more details.

## Modeling

### `sample`

```julia
# Distribution
ProbProg.sample(rng, dist; symbol=gensym("sample"))

# Custom sampler
ProbProg.sample(
    rng, f, args...;
    symbol  = gensym("sample"),
    logpdf  = nothing,  # assumes (sample, args...) -> scalar
    support = :real,
    bounds  = (nothing, nothing),
)
```

Returns `(updated_rng, value)`. `symbol` becomes the trace address.

### `untraced_call`

Call a probabilistic function without recording its choices:

```julia
ProbProg.untraced_call(rng, f, args...)
```

### `simulate`

Forward simulation.

```julia
trace_tensor, weight, retval = ProbProg.simulate(rng, f, args...)
```

Default dimension of `trace_tensor`: `(1, position_size)`. Default dimension of `weight`: scalar tensor. Needs to be embedded in an `@compile` / `@jit` for compilation.

### `simulate_`

Compile + run wrapper returning an unflattened trace. Handles `@compile` / `@jit` context. Similar to `simulate`, but the trace is returned in an unflattened form.

```julia
trace, weight = ProbProg.simulate_(rng, f, args...)
```

### `generate`

Generate a trace conditioned on a set of observed random choices, returning the trace and the log importance weight.

```julia
trace_tensor, weight, retval = ProbProg.generate(
    rng, constraint_tensor, f, args...;
    constrained_addresses::Set{Address},
)
```

Default dimension of `constraint_tensor`: `(1, total_constrained_size)`, values in `extract_addresses(constraint)` order. Default dimension of `trace_tensor`: `(1, position_size)`. Default dimension of `weight`: scalar tensor. Needs to be embedded in an `@compile` / `@jit` for compilation.

### `generate_`

Compile + run wrapper returning an unflattened trace. Handles `@compile` / `@jit` context, and builds the constraint tensor from a [`Constraint`](@ref). Similar to `generate`, but the trace is returned in an unflattened form.

```julia
trace, weight = ProbProg.generate_(rng, constraint, f, args...)
```

## Inference

### `mh`

One MH step:

```julia
new_trace, new_weight, accepted, _ = ProbProg.mh(
    rng, trace, weight, f, args...;
    selection::Selection,
)
```

Selected sites regenerate from the prior. `accepted` is a scalar `Bool`
tensor. Alias: `ProbProg.metropolis_hastings`.

### `mcmc`

Trace-based HMC / NUTS:

```julia
# Fresh chain
new_trace, diagnostics, retval, state = ProbProg.mcmc(
    rng, original_trace, f, args...;
    selection::Selection,
    algorithm            = :HMC,   # or :NUTS
    inverse_mass_matrix  = nothing,
    step_size            = nothing,
    trajectory_length    = 2π,
    max_tree_depth       = 10,
    max_delta_energy     = 1000.0,
    num_warmup           = 0,
    num_samples          = 1,
    thinning             = 1,
    adapt_step_size      = true,
    adapt_mass_matrix    = true,
)

# Resume
new_trace, diagnostics, retval, state = ProbProg.mcmc(
    state::MCMCState, original_trace, f, args...;
    selection::Selection,
    kwargs...,
)
```

Returns:
- `new_trace`: `(num_samples ÷ thinning, selected_position_size)`.
- `diagnostics`: per-iteration `Bool` (scalar if
  `num_samples ÷ thinning == 1`).
- `retval`: model return.
- `state`: [`MCMCState`](@ref).

### `mcmc_logpdf`

HMC / NUTS over a user provided log-density function. Mirrors [`mcmc`](@ref):

```julia
# Fresh chain
samples, diagnostics, retval, state = ProbProg.mcmc_logpdf(
    rng, logdensity_fn, initial_position, args...;
    algorithm                 = :NUTS,
    inverse_mass_matrix       = nothing,
    step_size                 = nothing,
    initial_gradient          = nothing,
    initial_potential_energy  = nothing,
    max_tree_depth            = 10,
    max_delta_energy          = 1000.0,
    num_warmup                = 0,
    num_samples               = 1,
    thinning                  = 1,
    adapt_step_size           = true,
    adapt_mass_matrix         = true,
    trajectory_length         = 2π,
    strong_zero               = false,
)

# Resume
samples, diagnostics, retval, state = ProbProg.mcmc_logpdf(
    state::MCMCState, logdensity_fn, args...; kwargs...,
)
```

`logdensity_fn`: `(position, args...) -> scalar`. `strong_zero = true`
treats zero paths as strong zeros, avoiding NaN gradients through inactive
branches.

### `run_chain`

Chunked chain driver:

```julia
samples::Array, state::MCMCState = ProbProg.run_chain(
    rng, logpdf_fn, initial_position, args...;
    algorithm           = :NUTS,
    num_warmup          = 0,
    num_samples         = 1000,
    chunk_size          = 100,
    step_size           = nothing,
    inverse_mass_matrix = nothing,
    progress_bar        = true,
    max_tree_depth      = 10,
    max_delta_energy    = 1000.0,
    adapt_step_size     = true,
    adapt_mass_matrix   = true,
    thinning            = 1,
    trajectory_length   = 2π,
)

# Resume
samples, state = ProbProg.run_chain(state::MCMCState, logpdf_fn; kwargs...)
```

`samples`: host `Array{Float64,2}`. `progress_bar=false` compiles one
monolithic function; `true` compiles warmup and sampling kernels and
invokes the latter chunk by chunk.

## State persistence

### `save_state`

Serialize a [`MCMCState`](@ref) to a file.

```julia
ProbProg.save_state(filename::String, state::MCMCState)
```

### `load_state`

Deserialize a [`MCMCState`](@ref) from a saved state.

```julia
state::MCMCState = ProbProg.load_state(filename::String)
```

## Utilities

### `select`

Construct a [`Selection`](@ref) from a list of addresses.

```julia
ProbProg.select(
    ProbProg.Address(:slope),
    ProbProg.Address(:intercept),
)
```

### `get_choices`

Return the choices of a `Trace`.

### `unflatten_trace`

Rebuild a dictionary-like `Trace` from a flat tensor with layout metadata.

```julia
trace = ProbProg.unflatten_trace(trace_tensor, weight, tt.entries, retval)
```

### `mcmc_summary`

Per-parameter summary: mean, std, median, quantiles, `n_eff`, `r_hat`.
Accepts a sample matrix or a [`Trace`](@ref):

```julia
ProbProg.mcmc_summary(samples; names=["β0", "β1"])
ProbProg.mcmc_summary(trace)
```
