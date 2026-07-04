#!/usr/bin/env julia
#=
Linear State-Space Model with Shared Transition Kernel
=======================================================

Model: A D-dimensional latent state evolves via a discrete-time linear
dynamical system:
    x_{t+1} = K(θ) · x_t

where K(θ) = θ · W'W + I is a positive-definite transition matrix.
  - W is a known D×D spatial coupling basis
  - θ (scalar) controls the coupling strength
  - I ensures positive definiteness and bounded eigenvalues

Given an initial state x₀ and noisy observation of the final state y_T,
the log-likelihood under Gaussian noise is:
    log p(y_T | θ) = -1/2 · ||K(θ)^T · x₀ - y_T||²

Computing K(θ)^T · x₀ requires T sequential matrix–vector products
(a genuine recurrence that cannot be vectorized).

K(θ) is loop-invariant (depends on θ, not t), so pre-AD LICM can hoist
the D×D matrix construction K = θ·W'W + I outside the T-step loop before
differentiation. Without this, Enzyme caches K per iteration (O(T·D²) memory)
and the backward pass computes the chain rule dK→dW→dθ per iteration
(O(T·D³) extra FLOPs).

Usage (from Reactant.jl root):
    JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/forward_model.jl probprog
    JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/forward_model.jl probprog_no_opt
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg

const D_STATE = 100    # state dimension
const T_STEPS = 100    # propagation steps (discrete time horizon)

function state_space_logpdf(position, W_basis, x0)
    @allowscalar begin
        theta = position[1, 1]       # dynamics coupling strength
        I_d = Matrix{Float64}(I, D_STATE, D_STATE)

        # State propagation: x_{t+1} = K(θ) · x_t
        x = x0   # 1 × D
        @trace track_numbers=false for t in 0:(T_STEPS-1)
            # Transition kernel (loop-invariant: depends on θ, not t)
            W = theta .* W_basis                         # D × D
            K = permutedims(W) * W .+ I_d                # D × D, PD
            # State update (loop-variant recurrence)
            x = x * K                                    # 1 × D
        end

        # Observation log-likelihood: -½ ||x_T||²
        # (equivalent to observing y_T = 0 with unit noise)
        total = -0.5 * sum(x .* x)
        return total
    end
end

function run_mcmc(rng, pos, ss, W, x0)
    ProbProg.mcmc_logpdf(rng, state_space_logpdf, pos, W, x0;
        algorithm=:NUTS, step_size=ss, max_tree_depth=5,
        num_warmup=0, num_samples=20,
        adapt_step_size=false, adapt_mass_matrix=false)
end

Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = true

pipeline = Symbol(ARGS[1])
println("=== State-Space Model: $pipeline, D=$D_STATE, T=$T_STEPS ===")

Random.seed!(42)
# Spatial coupling basis (small norm for stable dynamics)
W_r = Reactant.to_rarray(randn(D_STATE, D_STATE) * 0.01)
# Initial state
x0_r = Reactant.to_rarray(ones(1, D_STATE) * 0.01)
# Initial position (coupling strength)
pos = Reactant.to_rarray(ones(1, 1) * 0.5)
ss = Reactant.ConcreteRNumber(0.1)
rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))

compile_time = @elapsed begin
    compiled_fn = @compile sync=true optimize=pipeline run_mcmc(rng, pos, ss, W_r, x0_r)
end
println("  Compile: $(round(compile_time, digits=1))s")

# Warmup (1st call)
rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
compiled_fn(rng1, pos, ss, W_r, x0_r)

# Timed runs (post-warmup)
times = Float64[]
for i in 1:5
    rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100+i, 0]))
    t = @elapsed compiled_fn(rng_i, pos, ss, W_r, x0_r)
    push!(times, t)
end
median_t = sort(times)[3]
println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
println("  All: $(round.(times .* 1000, digits=1))ms")

# IR stats
hlo = @code_hlo optimize=pipeline run_mcmc(rng, pos, ss, W_r, x0_r)
hlo_str = sprint(show, hlo)
n_dot = count("dot_general", hlo_str)
n_while = count("stablehlo.while", hlo_str)
n_dslice = count("dynamic_slice", hlo_str)
println("  IR: $(count("\n", hlo_str)) lines, $n_dot dot_generals, $n_while while-loops, $n_dslice dynamic_slices")
