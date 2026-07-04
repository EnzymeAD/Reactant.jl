#!/usr/bin/env julia
#=
Multi-Parameter Linear Dynamics (LICM with multiple loop-invariant expressions)
================================================================================

Model: A D-dimensional latent state evolves via:
    x_{t+1} = M(θ) · x_t

where M(θ) = θ₁·A₁ + θ₂·A₂ + θ₃·A₃ + I is a transition matrix formed from
three fixed basis matrices A₁, A₂, A₃ weighted by sampled parameters.

All θᵢ are sampled, but M(θ) is loop-invariant (depends on θ, not t).
Pre-AD LICM hoists the O(D²) matrix construction AND the three scaled
matrix additions outside the loop before differentiation.

Comparison with forward_model.jl:
  forward_model:  1 parameter, 1 basis matrix (θ·W'W + I)
  this model:     3 parameters, 3 basis matrices (θ₁·A₁ + θ₂·A₂ + θ₃·A₃ + I)
  Demonstrates LICM benefit with multiple parameters → higher-dimensional
  posterior (position_size=3 → deeper NUTS trees → more leapfrog steps).

Usage:
    CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/multi_param_dynamics.jl probprog
    CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/multi_param_dynamics.jl probprog_no_opt
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg

const D_STATE = 100    # state dimension
const T_STEPS = 100    # propagation steps

function multi_param_logpdf(position, A1, A2, A3, x0)
    @allowscalar begin
        theta1 = position[1, 1]
        theta2 = position[1, 2]
        theta3 = position[1, 3]

        I_d = Matrix{Float64}(I, D_STATE, D_STATE)

        x = x0   # 1 × D
        @trace track_numbers=false for t in 0:(T_STEPS-1)
            # Transition matrix (LOOP-INVARIANT: depends on θ₁,θ₂,θ₃, not t)
            M = theta1 .* A1 .+ theta2 .* A2 .+ theta3 .* A3 .+ I_d  # D × D
            x = x * M                                                   # 1 × D
        end

        # Observation log-likelihood: -½ ||x_T||²
        total = -0.5 * sum(x .* x)
        return total
    end
end

function run_mcmc(rng, pos, ss, A1, A2, A3, x0)
    ProbProg.mcmc_logpdf(rng, multi_param_logpdf, pos, A1, A2, A3, x0;
        algorithm=:NUTS, step_size=ss, max_tree_depth=5,
        num_warmup=0, num_samples=20,
        adapt_step_size=false, adapt_mass_matrix=false)
end

Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = true

pipeline = Symbol(ARGS[1])
println("=== Multi-Param Dynamics: $pipeline, D=$D_STATE, T=$T_STEPS ===")

Random.seed!(42)
# Three orthogonal basis matrices (small norm for stability)
A1_r = Reactant.to_rarray(randn(D_STATE, D_STATE) * 0.005)
A2_r = Reactant.to_rarray(randn(D_STATE, D_STATE) * 0.005)
A3_r = Reactant.to_rarray(randn(D_STATE, D_STATE) * 0.005)
# Initial state
x0_r = Reactant.to_rarray(ones(1, D_STATE) * 0.01)
# Initial position (3 coupling strengths)
pos = Reactant.to_rarray(ones(1, 3) * 0.5)
ss = Reactant.ConcreteRNumber(0.1)
rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))

compile_time = @elapsed begin
    compiled_fn = @compile sync=true optimize=pipeline run_mcmc(rng, pos, ss, A1_r, A2_r, A3_r, x0_r)
end
println("  Compile: $(round(compile_time, digits=1))s")

# Warmup
rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
compiled_fn(rng1, pos, ss, A1_r, A2_r, A3_r, x0_r)

# Timed runs
times = Float64[]
for i in 1:5
    rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100+i, 0]))
    t = @elapsed compiled_fn(rng_i, pos, ss, A1_r, A2_r, A3_r, x0_r)
    push!(times, t)
end
median_t = sort(times)[3]
println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
println("  All: $(round.(times .* 1000, digits=1))ms")

# IR stats
hlo = @code_hlo optimize=pipeline run_mcmc(rng, pos, ss, A1_r, A2_r, A3_r, x0_r)
hlo_str = sprint(show, hlo)
n_dot = count("dot_general", hlo_str)
n_while = count("stablehlo.while", hlo_str)
n_dslice = count("dynamic_slice", hlo_str)
println("  IR: $(count("\n", hlo_str)) lines, $n_dot dot_generals, $n_while while-loops, $n_dslice dynamic_slices")
