# [Probabilistic Programming API](@id probprog-api)

This page is a flat index of the public exports of `Reactant.ProbProg`,
Reactant's probabilistic-programming module. Names are qualified as
`ProbProg.<name>`; `using Reactant: ProbProg` unqualifies them.

For walkthroughs and runnable end-to-end programs, follow the
[Probabilistic Programming tutorials](@ref probprog):

- [Overview](@ref probprog) — Bayesian linear and logistic regression.
- [Interface overview](@ref probprog-interface-overview) — quick tabular
  summary of the public surface.
- [Sampling and distributions](@ref probprog-sampling) — `sample`, built-in
  distributions, custom samplers, supports.
- [Traces and constrained inference](@ref probprog-traces) — `Trace`,
  `Address`, `Constraint`, `Selection`, `simulate` and `generate`.
- [Trace representation](@ref probprog-trace-representation) — how
  Reactant carries a trace as a flat position-vector tensor during
  compilation, and the helpers that expand it back into a
  tree-shaped `Trace`.
- [MCMC: MH, HMC, NUTS](@ref probprog-mcmc) — `mh`, `mcmc`, `mcmc_logpdf`.
- [Running and resuming chains](@ref probprog-chains) — `run_chain`,
  `MCMCState`, checkpointing, `mcmc_summary`.

The conceptual vocabulary (traces, choice maps, selections, generative
functions) is the same as in [Gen.jl](https://www.gen.dev/); readers
familiar with Gen can map `Constraint` onto Gen's `ChoiceMap`, `select`
onto `Gen.select`, and so on.

## Types

| Symbol | Signature | See |
|--------|-----------|-----|
| `Trace`        | `mutable struct Trace; choices, retval, weight, subtraces; end` | [Traces and constrained inference](@ref probprog-traces) |
| `Address`      | `Address(syms::Symbol...)` | [Traces and constrained inference](@ref probprog-traces) |
| `Selection`    | `const Selection = OrderedSet{Address}`; built via `select` | [Traces and constrained inference](@ref probprog-traces) |
| `Constraint`   | `Constraint(pairs::Pair...)` | [Traces and constrained inference](@ref probprog-traces) |
| `MCMCState`    | `mutable struct MCMCState; position, gradient, potential_energy, step_size, inverse_mass_matrix, rng; end` | [Running and resuming chains](@ref probprog-chains) |

## Distributions

`ProbProg.sample(rng, dist; symbol=:x)` works on any `Distribution`
subtype. Built-in distributions (`Normal`, `Exponential`, `LogNormal`,
`Bernoulli`) and the recipe for defining custom ones are described in
the [Sampling and distributions](@ref probprog-sampling) tutorial.

## Modeling

| Symbol | Signature | See |
|--------|-----------|-----|
| `sample`        | `sample(rng, dist; symbol)` &nbsp;/&nbsp; `sample(rng, f, args...; symbol, logpdf, support, bounds)` | [Sampling and distributions](@ref probprog-sampling) |
| `untraced_call` | `untraced_call(rng, f, args...)` | [Sampling and distributions](@ref probprog-sampling) |
| `simulate`      | `simulate(rng, f, args...) -> (trace_tensor, weight, retval)` | [Traces and constrained inference](@ref probprog-traces) |
| `simulate_`     | `simulate_(rng, f, args...) -> (trace::Trace, weight)` | [Traces and constrained inference](@ref probprog-traces) |
| `generate`      | `generate(rng, constraint_tensor, f, args...; constrained_addresses)` | [Traces and constrained inference](@ref probprog-traces) |
| `generate_`     | `generate_(rng, constraint, f, args...) -> (trace::Trace, weight)` | [Traces and constrained inference](@ref probprog-traces) |

## Inference

| Symbol | Signature | See |
|--------|-----------|-----|
| `mh`          | `mh(rng, trace, weight, f, args...; selection)` | [MCMC: MH, HMC, NUTS](@ref probprog-mcmc) |
| `mcmc`        | `mcmc(rng, original_trace, f, args...; selection, algorithm, step_size, inverse_mass_matrix, num_warmup, num_samples, ...)` &nbsp;/&nbsp; `mcmc(state::MCMCState, ...)` | [MCMC: MH, HMC, NUTS](@ref probprog-mcmc) |
| `mcmc_logpdf` | `mcmc_logpdf(rng, logdensity_fn, initial_position, args...; algorithm, step_size, inverse_mass_matrix, ...)` &nbsp;/&nbsp; `mcmc_logpdf(state::MCMCState, ...)` | [MCMC: MH, HMC, NUTS](@ref probprog-mcmc) |
| `run_chain`   | `run_chain(rng, logpdf_fn, initial_position, args...; num_warmup, num_samples, chunk_size, ...)` &nbsp;/&nbsp; `run_chain(state::MCMCState, ...)` | [Running and resuming chains](@ref probprog-chains) |

NUTS currently requires an explicit `step_size`; the default
`step_size = nothing` is rejected by the pass implementation
(`find_reasonable_step_size` is not yet implemented). Pass
`Reactant.ConcreteRNumber(0.1)` (or similar) and rely on dual averaging
when `adapt_step_size = true`.

## State persistence

| Symbol | Signature | See |
|--------|-----------|-----|
| `save_state` | `save_state(filename::String, state::MCMCState)` | [Running and resuming chains](@ref probprog-chains) |
| `load_state` | `load_state(filename::String) -> MCMCState` | [Running and resuming chains](@ref probprog-chains) |

## Utilities

| Symbol | Signature | See |
|--------|-----------|-----|
| `select`                       | `select(addrs::Address...) -> Selection` | [Traces and constrained inference](@ref probprog-traces) |
| `get_choices`                  | `get_choices(trace::Trace) -> Dict{Symbol,Any}` | [Traces and constrained inference](@ref probprog-traces) |
| `with_trace`                   | `with_trace(f) -> (f_result, tt)` (installs the Impulse tracing context for the duration of `f()`; collects the layout metadata `unflatten_trace` needs) | [Traces and constrained inference](@ref probprog-traces) |
| `unflatten_trace`              | `unflatten_trace(trace_tensor, weight, entries, retval) -> Trace` | [Traces and constrained inference](@ref probprog-traces) |
| `filter_entries_by_selection`  | `filter_entries_by_selection(entries, selection)` | [MCMC: MH, HMC, NUTS](@ref probprog-mcmc) |
| `extract_addresses`            | `extract_addresses(constraint::Constraint) -> Set{Address}` | [Traces and constrained inference](@ref probprog-traces) |
| `flatten_constraint`           | `flatten_constraint(constraint::Constraint) -> ConcreteRArray` | [Traces and constrained inference](@ref probprog-traces) |
| `mcmc_summary`                 | `mcmc_summary(samples; names)` &nbsp;/&nbsp; `mcmc_summary(trace::Trace)` | [Running and resuming chains](@ref probprog-chains) |
