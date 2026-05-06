# [Probabilistic Programming](@id probprog)

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) compiles ordinary Julia
code through [MLIR](https://mlir.llvm.org/), running it on CPU, GPU, or TPU
without rewriting. You write a normal Julia function, wrap a call to it in
[`@compile`](@ref), and Reactant traces the function, lowers it through MLIR,
and hands you back a callable compiled program. Array inputs are staged
through [`ConcreteRArray`](@ref) (created with [`Reactant.to_rarray`](@ref))
so they live on the target device.

`Reactant.ProbProg` is the Julia front-end for the `impulse` dialect,
implemented across [Enzyme](https://github.com/EnzymeAD/Enzyme)
(dialect definition and inference materialization passes) and
[Enzyme-JAX](https://github.com/EnzymeAD/Enzyme-JAX) (backend-specific lowering). The `impulse` dialect provides high-level MLIR ops for describing
probabilistic modeling and inference, materializes inference computation
through compiler passes, and applies 
general-purpose and probabilistic-programming-specific optimizations
during lowering.

!!! todo "`optimize = :probprog` opt-in required"
    For now, `@compile` needs an explicit `optimize = :probprog` argument
    on probabilistic programs to enable the impulse-specific MLIR passes
    (you'll see this in every `@compile` call below). Merging those passes
    into the default `@compile` pipeline is work in progress; once it
    lands, the explicit opt-in will no longer be required.

Next, we walk through two operating modes of `Reactant.ProbProg`: a trace-based mode built around a generative function, and a custom log-density mode that takes a custom log-density function.

## [Trace-based mode](@id probprog-trace-based)

We describe a Bayesian linear regression question:

```math
\begin{aligned}
\text{slope}     &\sim \mathcal{N}(0, 2) \\
\text{intercept} &\sim \mathcal{N}(0, 10) \\
y_i \mid \text{slope}, \text{intercept} &\sim \mathcal{N}(\text{slope} \cdot x_i + \text{intercept},\, 1)
\end{aligned}
```

Both regression coefficients are given Gaussian priors, tighter on `slope`
(standard deviation 2) and looser on `intercept` (standard deviation 10).
Each observation `y_i` is then drawn from a Gaussian centered on the
fitted value `slope · x_i + intercept` with fixed noise (standard deviation 1).

### The data

Synthetic data drawn from `slope = -2, intercept = 10`. The `xs` /
`ys` pair is what we'll condition on.

```@example probprog_index
xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
nothing # hide
```

### Describing the Model

We describe this model in `Reactant.ProbProg` as follows:

```@example probprog_index
using Reactant, Statistics            # hide
using Reactant: ReactantRNG           # hide
using Reactant: ProbProg

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
```

Each random choice is introduced by a `ProbProg.sample(rng, dist; symbol=...)`
call that takes a random number generator (RNG) and a distribution function. The `symbol` keyword names
the sample site used for conditioning and specifying parameters to infer.

As a calling convention, `ProbProg.sample` returns `(rng, value)`; the first element (omitted with `_` above) is the updated RNG. In the current implementation `rng` is a traced [`ReactantRNG`](https://github.com/EnzymeAD/Reactant.jl/blob/main/src/Types.jl#L648) whose state corresponds to a `tensor<2xui64>` RNG state in the generated MLIR. We don't thread it through manually because Reactant tracing handles the input/output threading at the IR level, and `ReactantRNG`'s internal state is updated via Julia mutability (see  [here](https://github.com/EnzymeAD/Reactant.jl/blob/main/src/stdlibs/Random.jl#L62) for details).

### Describing Inference

We condition on the observed `ys` with a [`Constraint`](@ref probprog-conditioning) object:

```@example probprog_index
obs = ProbProg.Constraint(:ys => ys)
```

The current implementation requires a bit of boilerplate to flatten the
`Constraint` into a tensor representation and to extract its address set
before passing them to the `@compile`'d function below (see
[Traces and constrained inference](@ref probprog-traces) for details):

```@example probprog_index
obs_tensor = ProbProg.flatten_constraint(obs)
```

```@example probprog_index
constrained_addresses = ProbProg.extract_addresses(obs)
```

We then specify what parameters to infer:

```@example probprog_index
selection = ProbProg.select(
    ProbProg.Address(:slope),
    ProbProg.Address(:intercept),
)
```

We express inference in a single function that conditions the model on the
constraint with [`generate`](@ref) and then runs NUTS over the selected sites
with [`mcmc`](@ref).

```@example probprog_index
function infer(rng, xs, obs_tensor, step_size, inverse_mass_matrix)
    trace, = ProbProg.generate(
        rng, obs_tensor, model, xs; constrained_addresses,
    )
    trace, = ProbProg.mcmc(
        rng, trace, model, xs;
        selection, algorithm=:NUTS,
        step_size, inverse_mass_matrix,
        num_warmup=200, num_samples=500,
    )
    return trace
end
```

The returned `trace` contains the sampling result as a 2D tensor: each row is the concatenation of all selected sites' flattened values for one post-warmup sample. (We will show a possible trace for this example problem below.)


#### Compiling with `@compile`

We compile `infer` with Reactant's [`@compile`](@ref) for 
compiler-optimized probabilistic inference:

```@example probprog_index
rng                 = ReactantRNG()
step_size           = Reactant.ConcreteRNumber(0.1)
inverse_mass_matrix = Reactant.ConcreteRArray([1.0 0.0; 0.0 1.0])

compiled_fn = @compile optimize=:probprog infer(
    rng, xs, obs_tensor, step_size, inverse_mass_matrix,
)
```

!!! note "Defaults"
    It is often sufficient to start with `step_size = 1.0` and an
    identity `inverse_mass_matrix`. With the default `adapt_step_size =
    true` and `adapt_mass_matrix = true`, [`mcmc`](@ref) adaptively
    selects appropriate values during the warmup iterations.

The `compiled_fn` is a callable object that takes the same arguments as
`infer` and returns the inference result. We can execute the compiled inference program any number of times by calling it: 

```@example probprog_index
trace_tensor = compiled_fn(rng, xs, obs_tensor, step_size, inverse_mass_matrix)
```

Each row is one post-warmup sample; columns hold the sampled values for each selected site:

```text
            :intercept   :slope
sample 1:      ...         ...
sample 2:      ...         ...
   ⋮            ⋮           ⋮
sample N:      ...         ...
```

From the sampler output, the posterior mean is:

```@example probprog_index
(
    posterior_mean_intercept = mean(trace_tensor[:, 1]),
    posterior_mean_slope     = mean(trace_tensor[:, 2]),
)
```

The data were generated from `slope = -2, intercept = 10`; NUTS recovers
both posterior means.

## Custom logpdf mode

In larger applications, it is often infeasible to express the model in
a PPL modeling language as we showed in the [trace-based mode](@ref probprog-trace-based)
above. We can use `Reactant.ProbProg` to compile and run its inference algorithms directly on a hand-written log-density function via the custom logpdf
mode.

For example, we can write the log-density function of the previous Bayesian linear regression model directly:
```@example probprog_index
function logdensity(θ, xs, ys)
    X = hcat(xs, ones(length(xs)))
    residuals = ys .- X * θ
    pr = -0.5 * sum(θ .^ 2 ./ [4.0, 100.0])
    ll = -0.5 * sum(residuals .^ 2)
    return ll + pr
end
```

We pass `logdensity` to the [`mcmc_logpdf`](@ref) interface along with
an initial [position](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) vector
(the parameter values the chain starts from):

```@example probprog_index
function infer_logpdf(rng, θ0, xs, ys, step_size, inverse_mass_matrix)
    trace, = ProbProg.mcmc_logpdf(rng, logdensity, θ0, xs, ys;
        algorithm=:NUTS,
        step_size, inverse_mass_matrix,
        num_warmup=200, num_samples=500,
    )
    return trace
end

θ0 = Reactant.to_rarray(reshape([0.0, 0.0], 1, 2))
compiled_logpdf = @compile optimize=:probprog infer_logpdf(
    rng, θ0, xs, ys, step_size, inverse_mass_matrix,
)
trace = compiled_logpdf(rng, θ0, xs, ys, step_size, inverse_mass_matrix)
```

We get similar inference results

```@example probprog_index
(
    posterior_mean_slope     = mean(trace[:, 1]),
    posterior_mean_intercept = mean(trace[:, 2]),
)
```

NUTS recovers both posterior means here too.

## More Explanations

- [Sampling and distributions](@ref probprog-sampling) — semantics of
  [`sample`](@ref), the built-in [`Distribution`](@ref) hierarchy, custom
  samplers with user-supplied `logpdf`, and constrained supports.
- [Traces and constrained inference](@ref probprog-traces) —
  [`simulate`](@ref) versus [`generate`](@ref), [`Constraint`](@ref probprog-conditioning) /
  [`Address`](@ref) construction, and the trace round-trip between flat
  position vector and tree-shaped [`Trace`](@ref).
- [MCMC: MH, HMC, NUTS](@ref probprog-mcmc) — Metropolis-Hastings over a
  [`Selection`](@ref), gradient-based chains via [`mcmc`](@ref), and
  log-density-driven chains via [`mcmc_logpdf`](@ref).
- [Running and resuming chains](@ref probprog-chains) —
  [`run_chain`](@ref), warmup and checkpointing through
  [`MCMCState`](@ref), [`save_state`](@ref) / [`load_state`](@ref), and
  posterior summaries with [`mcmc_summary`](@ref).
