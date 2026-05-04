# [Sampling and distributions](@id probprog-sampling)

A random choice uses [`sample`](@ref). Two call forms: a built-in
[`Distribution`](@ref), or a user sampler function with optional log-density.

## Built-in distribution

```julia
using Reactant
using Reactant: ProbProg, ReactantRNG

seed = Reactant.to_rarray(UInt64[1, 4])
rng  = ReactantRNG(seed)

_, x = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (10,)); symbol=:x)
```

Returns `(updated_rng, value)`. `symbol` is the trace address. Omitting it
produces a `gensym` name that cannot be constrained or inspected later.

### RNG

Counter-based, seeded by a length-2 `UInt64` vector:

```julia
seed = Reactant.to_rarray(UInt64[1, 4])
rng  = ReactantRNG(seed)
```

Same seed reproduces the same trajectory. `rng.seed` updates in place after
each compiled call.

### Distributions

| Type | Constructor | `support` |
|------|-------------|-----------|
| [`Normal`](@ref)       | `Normal(μ, σ, shape)`     | `:real` |
| [`Exponential`](@ref)  | `Exponential(λ, shape)`   | `:positive` |
| [`LogNormal`](@ref)    | `LogNormal(μ, σ, shape)`  | `:positive` |
| [`Bernoulli`](@ref)    | `Bernoulli(logits, shape)`| `:real` (logit scale) |

`shape` is a non-empty tuple. Parameters broadcast against `shape`, so `μ`
and `σ` can be scalars, `ConcreteRNumber`s, or arrays.

```julia
ProbProg.Normal()               # μ=0, σ=1, shape=(1,)
ProbProg.Normal(0.0, 1.0)       # shape=(1,)
ProbProg.Normal(0.0, 1.0, (5,))
ProbProg.Exponential(2.0, (3,))
ProbProg.LogNormal(0.0, 0.5, (2, 2))
ProbProg.Bernoulli(logits, (4,))
```

## Custom sampler

```julia
ProbProg.sample(
    rng, my_sampler, args...;
    symbol  = :my_site,
    logpdf  = my_logpdf,            # (sample, args...) -> scalar
    support = :real,
    bounds  = (nothing, nothing),
)
```

- `my_sampler(rng, args...)` must be traceable by Reactant.
- With `logpdf`, the site contributes to the model weight and is an
  inference target. Without it, the site is traced but contributes no
  log-density.
- `logpdf` is called with the sampled value and the original `args`, no
  `rng`.

Example:

```@example probprog_sampling
normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -length(x) * log(σ) - length(x)/2 * log(2π) -
           sum((x .- μ).^2 ./ (2 .* σ.^2))
end

function model(rng, xs)
    _, slope = ProbProg.sample(
        rng, normal, 0.0, 2.0, (1,);
        symbol=:slope, logpdf=normal_logpdf,
    )
    _, intercept = ProbProg.sample(
        rng, normal, 0.0, 10.0, (1,);
        symbol=:intercept, logpdf=normal_logpdf,
    )
    _, ys = ProbProg.sample(
        rng, normal, slope .* xs .+ intercept, 1.0, (length(xs),);
        symbol=:ys, logpdf=normal_logpdf,
    )
    return ys
end
```

### `support` and `bounds`

HMC and NUTS unconstrain sites based on `support` before proposing:

| `support` | Meaning |
|-----------|---------|
| `:real`            | Unconstrained (default) |
| `:positive`        | \$x > 0\$ |
| `:unit_interval`   | \$x \in (0, 1)\$ |
| `:interval`        | \$x \in (\text{lower}, \text{upper})\$ via `bounds` |
| `:greater_than`    | \$x > \text{lower}\$ |
| `:less_than`       | \$x < \text{upper}\$ |
| `:simplex`         | Probability simplex |
| `:lower_cholesky`  | Lower-triangular Cholesky factor |

Pass `bounds = (lower, upper)` for interval supports; either endpoint can
be `nothing`.

```julia
ProbProg.sample(
    rng, my_sampler, args...;
    symbol=:θ, logpdf=my_logpdf,
    support=:interval, bounds=(0.0, 1.0),
)

ProbProg.sample(
    rng, my_sampler, args...;
    symbol=:τ, logpdf=my_logpdf,
    support=:greater_than, bounds=(0.5, nothing),
)
```

Built-in distributions set `support` automatically.

## Submodels

A sampler that itself calls `ProbProg.sample` yields nested traces. Inner
sites become child addresses under the outer `symbol`:

```@example probprog_sampling
function inner(rng, μ, σ, shape)
    _, a = ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape); symbol=:a)
    _, b = ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape); symbol=:b)
    return a .* b
end

function outer(rng, μ, σ, shape)
    _, s = ProbProg.sample(rng, inner, μ, σ, shape; symbol=:s)
    _, t = ProbProg.sample(rng, inner, s, σ, shape; symbol=:t)
    return t
end
```

The resulting trace exposes `trace.subtraces[:s].choices[:a]`, etc.

[`untraced_call`](@ref) calls a probabilistic function without recording its
choices:

```julia
ProbProg.untraced_call(rng, inner, μ, σ, shape)
```

## Inspecting IR

Unoptimised form shows raw `impulse.sample` ops:

```julia
@code_hlo optimize=false ProbProg.sample(rng, ProbProg.Normal(μ, σ, (10,)))
```

Lowered form:

```julia
@code_hlo optimize=:probprog ProbProg.untraced_call(rng, model, μ, σ, (10,))
```

## Symbol reference

### `sample`

Draw a value from a distribution at a named address.

### `untraced_call`

Call a probabilistic function without recording its choices in the parent
trace.

### `Distribution`

Abstract supertype for built-in distributions. A subtype defines a
constructor and a `support`, which together determine how `sample`
interacts with the site at inference time.

### `Normal`

Gaussian distribution. `Normal(μ, σ, shape)`; scalar or array parameters broadcast
against `shape`. `support = :real`.

### `Exponential`

Exponential rate distribution. `Exponential(λ, shape)`. `support = :positive`.

### `LogNormal`

Log-normal distribution. `LogNormal(μ, σ, shape)`. `support = :positive`.

### `Bernoulli`

Bernoulli on logit scale. `Bernoulli(logits, shape)`. `support = :real`
(logits are unconstrained).

!!! todo "Distributions.jl tracing support"
    Tracing support for
    [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) (independent of Impulse) is
    planned but not yet implemented. Once landed, distributions from 
    `Distributions.Distribution` (e.g., `Distributions.Normal(0.0, 1.0)`, `Distributions.MvNormal(μ, Σ)`) will be usable as the distribution argument to `sample` directly.

Next: [traces and constrained inference](@ref probprog-traces).
