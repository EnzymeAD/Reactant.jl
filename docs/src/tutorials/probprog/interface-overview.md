# [Interface overview](@id probprog-interface-overview)

The public surface of `Reactant.ProbProg` is partitioned into four groups:
modeling primitives, inference drivers, persistent state, and data types.
The full signatures and semantics are documented in the
[API reference](@ref probprog-api); the tables below are a quick index.

## Modeling primitives

| Symbol | Signature | Role |
|--------|-----------|------|
| [`sample`](@ref)          | `sample(rng, dist; symbol)` &nbsp;/&nbsp; `sample(rng, f, args...; symbol, logpdf, support, bounds)` | Record a random choice from a distribution or user-defined sampler. |
| [`untraced_call`](@ref)   | `untraced_call(rng, f, args...)` | Execute a probabilistic function without recording its choices. |
| [`simulate`](@ref) &nbsp;/&nbsp; [`simulate_`](@ref) | `simulate(rng, f, args...)` | Forward-simulate a model; return its trace and prior log-density. |
| [`generate`](@ref) &nbsp;/&nbsp; [`generate_`](@ref) | `generate(rng, constraint, f, args...)` | Execute a model conditioned on observations; return the trace and log importance weight. |

## Inference

| Symbol | Signature | Role |
|--------|-----------|------|
| [`mh`](@ref)           | `mh(rng, trace, weight, f, args...; selection)` | One Metropolis-Hastings step regenerating the selected addresses. |
| [`mcmc`](@ref)         | `mcmc(rng, trace, f, args...; selection, algorithm, num_samples, ...)` | Trace-based HMC or NUTS. |
| [`mcmc_logpdf`](@ref)  | `mcmc_logpdf(rng, logdensity_fn, initial_position, args...; algorithm, ...)` | HMC or NUTS over a user-supplied log-density. |
| [`run_chain`](@ref)    | `run_chain(rng, logpdf_fn, initial_position, args...; num_warmup, num_samples, chunk_size, ...)` | Chunked chain driver with progress bar and resumable state. |

## State and persistence

| Symbol | Signature | Role |
|--------|-----------|------|
| [`MCMCState`](@ref)   | *(struct)* `position, gradient, potential_energy, step_size, inverse_mass_matrix, rng` | Resume token returned by every inference entry point. |
| [`save_state`](@ref)  | `save_state(filename, state)` | Serialise an `MCMCState` to disk. |
| [`load_state`](@ref)  | `load_state(filename)` | Reload an `MCMCState` from disk. |

## Data types and distributions

| Symbol | Role |
|--------|------|
| [`Trace`](@ref)       | Executed-model record: `choices`, `subtraces`, `retval`, `weight`. |
| [`Constraint`](@ref probprog-conditioning)  | Dict-like `Address → value` mapping for observations. |
| [`Address`](@ref)     | Immutable symbol path identifying a sample site. |
| [`Selection`](@ref)   | Ordered set of addresses; constructed with [`select`](@ref). |
| [`Normal`](@ref) &nbsp;/&nbsp; [`Exponential`](@ref) &nbsp;/&nbsp; [`LogNormal`](@ref) &nbsp;/&nbsp; [`Bernoulli`](@ref) | Built-in distributions, each carrying a static `shape::Tuple` and registered sampler / log-density. |
