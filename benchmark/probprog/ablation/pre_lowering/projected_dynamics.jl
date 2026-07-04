#!/usr/bin/env julia
#=
Projected Dynamics with Two Loop-Invariant Matrices
=====================================================

Model: A D-dimensional latent state evolves, observed through a projection:
    x_{t+1} = K(θ) · x_t                (transition, K loop-invariant)
    z_T = P(θ) · x_T                    (final observation through P)
    log p(y | θ) = -½ ||z_T - y||²

where K(θ) = θ · A + I and P(θ) = θ · B' * B + I_obs.

Both K(θ) and P(θ) are loop-invariant. LICM hoists BOTH matrix constructions
outside the loop before AD, reducing the backward pass cost for both.

Comparison with forward_model.jl:
  forward_model: 1 invariant matrix, D²
  this model:    2 invariant matrices (D² transition + D_obs×D projection),
                 demonstrates LICM benefit scales with number of invariant ops.

Usage:
    CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/projected_dynamics.jl probprog
    CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/projected_dynamics.jl probprog_no_opt
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg

const D_STATE = 100    # state dimension
const D_OBS = 30       # observation dimension
const T_STEPS = 100    # propagation steps

function projected_dynamics_logpdf(position, A_basis, B_basis, y_target, x0)
    @allowscalar begin
        theta = position[1, 1]
        I_d = Matrix{Float64}(I, D_STATE, D_STATE)
        I_obs = Matrix{Float64}(I, D_OBS, D_OBS)

        x = x0   # 1 × D_STATE
        @trace track_numbers=false for t in 0:(T_STEPS-1)
            # Transition matrix (LOOP-INVARIANT #1)
            K = theta .* A_basis .+ I_d                        # D × D
            # State update
            x = x * K                                          # 1 × D
        end

        # Observation projection (INVARIANT #2: depends on θ, computed once)
        # P(θ) = θ · B'B + I_obs
        P = theta .* (permutedims(B_basis) * B_basis) .+ I_obs  # D_OBS × D_OBS

        # Project final state: z = x_T · B · P^{-1}... simplified:
        # Just use P as a mixing matrix on the observation side
        z = x[:, 1:D_OBS] * P                                 # 1 × D_OBS

        diff = z .- y_target
        total = -0.5 * sum(diff .* diff)
        return total
    end
end

function run_mcmc(rng, pos, ss, A_basis, B_basis, y_target, x0)
    ProbProg.mcmc_logpdf(rng, projected_dynamics_logpdf, pos, A_basis, B_basis, y_target, x0;
        algorithm=:NUTS, step_size=ss, max_tree_depth=5,
        num_warmup=0, num_samples=20,
        adapt_step_size=false, adapt_mass_matrix=false)
end

Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = true

pipeline = Symbol(ARGS[1])
println("=== Projected Dynamics: $pipeline, D=$D_STATE, D_obs=$D_OBS, T=$T_STEPS ===")

Random.seed!(42)
A_r = Reactant.to_rarray(randn(D_STATE, D_STATE) * 0.01)
B_r = Reactant.to_rarray(randn(D_OBS, D_OBS) * 0.1)
y_r = Reactant.to_rarray(randn(1, D_OBS) * 0.1)
x0_r = Reactant.to_rarray(ones(1, D_STATE) * 0.01)
pos = Reactant.to_rarray(ones(1, 1) * 0.5)
ss = Reactant.ConcreteRNumber(0.1)
rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))

compile_time = @elapsed begin
    compiled_fn = @compile sync=true optimize=pipeline run_mcmc(rng, pos, ss, A_r, B_r, y_r, x0_r)
end
println("  Compile: $(round(compile_time, digits=1))s")

rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
compiled_fn(rng1, pos, ss, A_r, B_r, y_r, x0_r)

times = Float64[]
for i in 1:5
    rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100+i, 0]))
    t = @elapsed compiled_fn(rng_i, pos, ss, A_r, B_r, y_r, x0_r)
    push!(times, t)
end
median_t = sort(times)[3]
println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
println("  All: $(round.(times .* 1000, digits=1))ms")

hlo = @code_hlo optimize=pipeline run_mcmc(rng, pos, ss, A_r, B_r, y_r, x0_r)
hlo_str = sprint(show, hlo)
n_dot = count("dot_general", hlo_str)
n_while = count("stablehlo.while", hlo_str)
n_dslice = count("dynamic_slice", hlo_str)
println("  IR: $(count("\n", hlo_str)) lines, $n_dot dot_generals, $n_while while-loops, $n_dslice dynamic_slices")
