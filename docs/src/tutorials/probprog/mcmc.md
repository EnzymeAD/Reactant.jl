# [MCMC: MH, HMC, NUTS](@id probprog-mcmc)

Four MCMC entry points, all compiled end-to-end via `optimize = :probprog`:

| Function | Algorithm | Proposal |
|----------|-----------|----------|
| [`mh`](@ref)          | Metropolis-Hastings  | Regenerate selected sites from the prior |
| [`mcmc`](@ref) (`:HMC`)  | HMC  | Gradient-based, fixed trajectory length |
| [`mcmc`](@ref) (`:NUTS`) | NUTS | Adaptive tree-doubling HMC |
| [`mcmc_logpdf`](@ref) | HMC or NUTS | Over a user log-density |

Inner loops run on device without Julia round-trips.

## `mh`

One MH step over a [`Selection`](@ref). Selected sites resample from the
prior; the rest stay fixed.

```@example probprog_mcmc
using Reactant
using Reactant: ProbProg, ReactantRNG

xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]

obs                   = ProbProg.Constraint(:ys => ys)
constrained_addresses = ProbProg.extract_addresses(obs)
obs_flat              = Float64[]
for addr in constrained_addresses
    append!(obs_flat, vec(obs[addr]))
end
obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

seed = Reactant.to_rarray(UInt64[1, 5])
rng  = ReactantRNG(seed)
```

```@example probprog_mcmc
function model(rng, xs)
    _, slope = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:slope,
    )
    _, intercept = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 10.0, (1,)); symbol=:intercept,
    )
    _, ys = ProbProg.sample(
        rng,
        ProbProg.Normal(slope .* xs .+ intercept, 1.0, (length(xs),));
        symbol=:ys,
    )
    return ys
end

function mh_program(rng, model, xs, num_iters, obs_tensor, constrained_addresses)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, model, xs; constrained_addresses,
    )

    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(
            rng, trace, weight, model, xs;
            selection = ProbProg.select(ProbProg.Address(:slope)),
        )
        trace, weight, _ = ProbProg.mh(
            rng, trace, weight, model, xs;
            selection = ProbProg.select(ProbProg.Address(:intercept)),
        )
    end

    return trace, weight
end
```

`@trace for` unrolls the loop inside the compiled function; `num_iters` is
passed as `ConcreteRNumber`. Use one `select` for a joint update instead of
alternating.

### Compile and run

```@example probprog_mcmc
num_iters = Reactant.ConcreteRNumber(1000)

compiled_fn, tt = ProbProg.with_trace() do
    @compile optimize=:probprog mh_program(
        rng, model, xs, num_iters, obs_tensor, constrained_addresses,
    )
end

trace_tensor, weight_val = compiled_fn(
    rng, model, xs, num_iters, obs_tensor, constrained_addresses,
)

trace = ProbProg.unflatten_trace(trace_tensor, weight_val, tt.entries, ())
```

## `mcmc`

```julia
trace, diagnostics, _, state = ProbProg.mcmc(
    rng,
    initial_trace,
    model,
    args...;
    selection           = ProbProg.select(ProbProg.Address(:param_a),
                                          ProbProg.Address(:param_b)),
    algorithm           = :NUTS,                # or :HMC
    step_size           = Reactant.ConcreteRNumber(0.1),
    inverse_mass_matrix = Reactant.ConcreteRArray([0.5 0.0; 0.0 0.5]),
    max_tree_depth      = 10,                   # NUTS
    max_delta_energy    = 1000.0,               # NUTS
    trajectory_length   = 2π,                   # HMC
    num_warmup          = 200,
    num_samples         = 500,
    thinning            = 1,
    adapt_step_size     = true,
    adapt_mass_matrix   = true,
)
```

### `Selection`

A `Selection` names the sites an inference program updates. Only selected sites
enter the position vector; others stay at their initial-trace values. Output `trace` has shape
`(num_samples ÷ thinning, selected_dim)`.

Rebuild a tree `Trace` with `filter_entries_by_selection` and
`unflatten_trace`:

```julia
selected_entries = ProbProg.filter_entries_by_selection(tt.entries, selection)
trace = ProbProg.unflatten_trace(trace_tensor, 0.0, selected_entries, nothing)
```

### Inverse mass matrix

Rank determines structure:

| Rank | Interpretation |
|------|----------------|
| 1 (length `d`)  | Diagonal |
| 2 (`d × d`)     | Dense |
| `nothing`       | Identity |

Shape must match the selected position size.

### Step size and adaptation

`step_size` is the leapfrog step. With `adapt_step_size = true`, warmup
tunes it and writes the adapted value to `state.step_size`.

`adapt_mass_matrix = true` enables warmup-window mass-matrix adaptation.
Verify against a reference implementation for pathological problems.

### HMC vs NUTS

- `:HMC`: `trajectory_length / step_size` leapfrog steps per iteration,
  MH accept or reject.
- `:NUTS`: tree doubling to depth `max_tree_depth`; divergence above
  `max_delta_energy` aborts.

### Diagnostics

Per-iteration `Bool`, rank-1 of length `num_samples ÷ thinning` (scalar
if 1). HMC: MH acceptance. NUTS: `true` if no divergence.

## `mcmc_logpdf`

Skip the trace machinery when a closed-form log-density exists:

```julia
function logpdf(θ, xs, ys)
    # θ: position vector; xs, ys: data
    ...
end

initial_position = Reactant.to_rarray(zeros(d))

samples, diagnostics, _, state = ProbProg.mcmc_logpdf(
    rng, logpdf, initial_position, xs, ys;
    algorithm         = :NUTS,
    step_size         = Reactant.ConcreteRNumber(0.1),
    num_warmup        = 500,
    num_samples       = 1000,
    adapt_step_size   = true,
    adapt_mass_matrix = true,
)
```

`initial_position` is 1D or 2D; 1D is reshaped to `(1, length(p))`.
`samples` has shape `(num_samples ÷ thinning, d)`.

Optional `initial_gradient` and `initial_potential_energy` seed chain
resume. `strong_zero = true` sets an autodiff attribute that treats
zero paths as strong zeros, avoiding NaN gradients through inactive
branches.

Next: [Running and resuming chains](@ref probprog-chains).
