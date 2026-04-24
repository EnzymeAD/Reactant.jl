# [Probabilistic Programming](@id probprog)

`Reactant.ProbProg` is Reactant's probabilistic programming module, built on the `impulse` dialect.

A summary of exported symbols is provided in the [Interface Overview](@ref probprog-interface-overview). Please refer to the [API Reference](@ref probprog-api) for documentation of exported symbols.

## Example Usage

A generative function can be constructed using the Gen-style modeling language or 
*trace-based mode*, where the generative function is expressed with
[`ProbProg.sample`](@ref) calls and the inference kernel walks the trace;
and *custom logpdf mode*, where the user supplies a log-density function directly.
Each is illustrated below with a canonical example — Bayesian
linear regression for the trace-based route, and Bayesian logistic
regression for the custom logpdf route.

### Trace-based mode: Bayesian linear regression

With the generative function written as ordinary Julia code and each
random choice named via `symbol`, [`generate`](@ref) folds observations
into the trace and [`mcmc`](@ref) updates the unobserved addresses via
NUTS. Generation and inference are fused into a single compiled program:

```julia
using Reactant, Statistics
using Reactant: ProbProg, ReactantRNG

# slope     ~ Normal(0,  2)
# intercept ~ Normal(0, 10)
# yᵢ | slope, intercept ~ Normal(slope · xᵢ + intercept, 1)
function linreg(rng, xs)
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

xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]

# Fold observations into a flat constraint tensor in canonical address order.
obs                   = ProbProg.Constraint(:ys => ys)
constrained_addresses = ProbProg.extract_addresses(obs)
obs_flat              = Float64[]
for addr in constrained_addresses
    append!(obs_flat, vec(obs[addr]))
end
obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

# `generate` conditions on observations; `mcmc` explores slope/intercept via NUTS.
function program(rng, model, xs, obs_tensor, constrained_addresses)
    trace, _, _ = ProbProg.generate(
        rng, obs_tensor, model, xs;
        constrained_addresses=constrained_addresses,
    )
    trace, diag, _, _ = ProbProg.mcmc(
        rng, trace, model, xs;
        selection   = ProbProg.select(
            ProbProg.Address(:slope),
            ProbProg.Address(:intercept),
        ),
        algorithm   = :NUTS,
        num_warmup  = 200,
        num_samples = 500,
    )
    return trace, diag
end

seed = Reactant.to_rarray(UInt64[1, 5])
rng  = ReactantRNG(seed)

compiled_fn, tt = ProbProg.with_trace() do
    @compile optimize=:probprog program(rng, linreg, xs, obs_tensor, constrained_addresses)
end

trace_tensor, _ = compiled_fn(rng, linreg, xs, obs_tensor, constrained_addresses)

selection        = ProbProg.select(
    ProbProg.Address(:slope),
    ProbProg.Address(:intercept),
)
selected_entries = ProbProg.filter_entries_by_selection(tt.entries, selection)
trace            = ProbProg.unflatten_trace(trace_tensor, 0.0, selected_entries, nothing)

(
    posterior_mean_slope     = mean(trace.choices[:slope][:, 1]),
    posterior_mean_intercept = mean(trace.choices[:intercept][:, 1]),
)
```

The data were generated from `slope = -2`, `intercept = 10`; NUTS
recovers both posterior means.

### Custom logpdf mode: Bayesian logistic regression

When a closed-form log-density is available, [`mcmc_logpdf`](@ref) skips
the trace machinery. Below, a standard Normal prior on the weight vector
is combined with a logistic-regression likelihood written in the
numerically stable form of the binary cross-entropy:

```julia
# β ~ Normal(0, I);  yᵢ | β ~ Bernoulli(σ(xᵢ · β))
# log p(β | X, y) = -½ ‖β‖²  +  Σᵢ [ yᵢ (xᵢ·β) − log(1 + exp(xᵢ·β)) ]
function logdensity(β, X, y)
    logits = X * β
    ll     = sum(y .* logits .- max.(logits, 0.0) .- log1p.(exp.(.-abs.(logits))))
    pr     = -0.5 * sum(β .^ 2)
    return ll + pr
end

# Design matrix with an intercept column and one real-valued feature.
X_data = Float64[
    1.0 -0.5
    1.0  0.3
    1.0  0.8
    1.0 -0.2
    1.0  1.4
    1.0 -1.1
]
y_data = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]

X                = Reactant.to_rarray(X_data)
y                = Reactant.to_rarray(y_data)
initial_position = Reactant.to_rarray(zeros(2))

seed_lr = Reactant.to_rarray(UInt64[2, 7])
rng_lr  = ReactantRNG(seed_lr)

samples, _, _, state = ProbProg.mcmc_logpdf(
    rng_lr, logdensity, initial_position, X, y;
    algorithm         = :NUTS,
    num_warmup        = 200,
    num_samples       = 500,
    adapt_step_size   = true,
    adapt_mass_matrix = true,
)

(
    posterior_mean_β    = vec(mean(Array(samples); dims=1)),  # (intercept, slope)
    adapted_step_size   = Array(state.step_size)[],
)
```

Trace-based mode is preferable when the model is naturally expressed as
a generative function and the same definition should drive forward
simulation, conditioning, and inference; custom logpdf mode is
preferable when a log-density implementation is already available or when
integrating with an external log-density library.

## Further reading

- [Sampling and distributions](@ref probprog-sampling) — semantics of
  [`sample`](@ref), the built-in [`Distribution`](@ref) hierarchy, custom
  samplers with user-supplied `logpdf`, and constrained supports.
- [Traces and constrained inference](@ref probprog-traces) — [`simulate`](@ref)
  versus [`generate`](@ref), [`Constraint`](@ref) / [`Address`](@ref)
  construction, and the interpretation of importance weights.
- [MCMC: MH, HMC, NUTS](@ref probprog-mcmc) — Metropolis-Hastings over a
  [`Selection`](@ref), gradient-based chains via [`mcmc`](@ref), and
  log-density-driven chains via [`mcmc_logpdf`](@ref).
- [Running and resuming chains](@ref probprog-chains) — [`run_chain`](@ref),
  warmup and checkpointing through [`MCMCState`](@ref),
  [`save_state`](@ref) / [`load_state`](@ref), and posterior summaries with
  [`mcmc_summary`](@ref).
