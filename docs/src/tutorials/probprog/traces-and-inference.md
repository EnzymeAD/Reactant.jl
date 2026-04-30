# [Traces and constrained inference](@id probprog-traces)

A *trace* records every random choice in one model execution, along with
its log-density and the model's return value. There are two ways to produce a trace:

- [`simulate`](@ref) runs forward; each `sample` draws from the prior.
- [`generate`](@ref) conditions on observed choices and returns an
  importance weight.

Each has a low-level form returning MLIR values, and a helper that
compiles, runs, and returns an unflattened [`Trace`](@ref).

## `Trace`

| Field | Contents |
|-------|----------|
| `choices`   | `Dict{Symbol,Any}` at this level |
| `subtraces` | `Dict{Symbol,Any}` for submodels |
| `retval`    | Model return |
| `weight`    | Importance weight (`generate`) or prior log-density (`simulate`) |

Each value in `choices` is an array indexed on its first axis by the sample's symbol.

## [Internal Trace Representation](@id probprog-trace-representation)

`@compile optimize = :probprog` returns Impulse's internal tensor-based representation of traces. 

The flat form contains every random choice the model would make,
flattened and concatenated into a single row, with the layout fixed at
trace time. With `num_samples` rows (e.g., from a NUTS run with
`num_samples = 12`), the trace tensor becomes a
`(num_samples, position_size)` tensor where `position_size` is the
total number of elements across all sampled sites. There are no symbol
lookups in this representation; per-site offsets and shapes are baked
in at compile time, and the bridge helpers below carry that layout
metadata so the tensor can be reconstituted into a tree.

### Helpers that bridge the two

Reactant frontend provides convenience helpers that handle the conversion in either direction.

- [`simulate_`](@ref) and [`generate_`](@ref) compile, run the model,
  and immediately convert the result back to a tree-shaped
  [`Trace`](@ref). The flat form never surfaces.
- [`unflatten_trace`](@ref) does the explicit tensor → tree conversion,
  given a trace tensor and layout metadata
  (per-site offset, shape, address-path) collected by the Impulse tracing context.
- [`with_trace`](@ref) installs the Impulse tracing context that collects the
  layout metadata while a compiled program is being built.
  your `@compile` call in it.

## `simulate_`

```@example probprog_traces
using Reactant
using Reactant: ProbProg, ReactantRNG

function model(rng, μ, σ, shape)
    _, s = ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape); symbol=:s)
    _, t = ProbProg.sample(rng, ProbProg.Normal(s, σ, shape); symbol=:t)
    return t
end

seed = Reactant.to_rarray(UInt64[1, 4])
rng  = ReactantRNG(seed)
μ    = Reactant.ConcreteRNumber(0.0)
σ    = Reactant.ConcreteRNumber(1.0)

trace, weight = ProbProg.simulate_(rng, model, μ, σ, (3,))

trace.choices[:s]   # (1, 3)
trace.choices[:t]   # (1, 3)
trace.retval[1]
trace.weight
```

`simulate_` calls `@compile optimize=:probprog` on
`ProbProg.simulate(rng, model, args...)` and reshapes the flat position
tensor using layout metadata collected during tracing.

### Submodels

```@example probprog_traces
function pair(rng, μ, σ, shape)
    _, a = ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape); symbol=:a)
    _, b = ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape); symbol=:b)
    return a .* b
end

function outer(rng, μ, σ, shape)
    _, s = ProbProg.sample(rng, pair, μ, σ, shape; symbol=:s)
    _, t = ProbProg.sample(rng, pair, s,  σ, shape; symbol=:t)
    return t
end

trace, _ = ProbProg.simulate_(rng, outer, μ, σ, (3, 3, 3))
trace.subtraces[:s].choices[:a]
trace.subtraces[:t].choices[:b]
```

## Low-level `simulate`

Inside a larger compiled program, [`simulate`](@ref) returns MLIR-traced
values:

```julia
trace_tensor, weight, retval = ProbProg.simulate(rng, model, μ, σ, shape)
```

`trace_tensor` is rank-2 shape `(1, position_size)`. Wrap with
[`with_trace`](@ref) to install the tracing context:

```julia
code, tt = ProbProg.with_trace() do
    @code_hlo optimize=:probprog begin
        ProbProg.simulate(rng, model, μ, σ, shape)
    end
end
```

Rebuild a `Trace` with [`unflatten_trace`](@ref):

```julia
trace = ProbProg.unflatten_trace(trace_tensor, weight, tt.entries, retval)
```

## Conditioning

A [`Constraint`](@ref) pins addresses to observed values:

```@example probprog_traces
obs = ProbProg.Constraint(
    :param_a => [0.0],
    :param_b => [0.0],
    :ys_a    => [-2.3, -1.6, -0.4, 0.6, 1.4],
    :ys_b    => [-2.6, -1.4, -0.6, 0.4, 1.6],
)
```

Nested addresses: `Constraint(:outer => :inner => value)`.

[`generate_`](@ref) returns a trace whose `weight` is the log importance
weight:

```julia
trace, weight = ProbProg.generate_(rng, obs, model, xs)
```

For embedding inside a compiled function, flatten manually and call
[`generate`](@ref):

```julia
constrained_addresses = ProbProg.extract_addresses(obs)
obs_flat = Float64[]
for addr in constrained_addresses
    append!(obs_flat, vec(obs[addr]))
end
obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

trace_tensor, weight, _ = ProbProg.generate(
    rng, obs_tensor, model, xs; constrained_addresses,
)
```

Append values in `extract_addresses(obs)` order. `generate_` handles this
automatically.

## Addresses

An [`Address`](@ref) is a path of symbols:

```@example probprog_traces
ProbProg.Address(:slope)
ProbProg.Address(:outer, :inner, :x)
ProbProg.Address([:outer, :inner, :x])
```

Equality is path equality. A [`Selection`](@ref) is an
`OrderedSet{Address}`:

```@example probprog_traces
ProbProg.select(
    ProbProg.Address(:slope),
    ProbProg.Address(:intercept),
)
```

MCMC kernels use selections to choose which sites to update.

## Summary

```text
simulate(_)   forward sampling, prior log-density
generate(_)   observations applied, importance weight
Trace         choices + subtraces + retval + weight
Constraint    Address => observed value
Selection     OrderedSet{Address}
```

Next: [MH, HMC, NUTS](@ref probprog-mcmc).
