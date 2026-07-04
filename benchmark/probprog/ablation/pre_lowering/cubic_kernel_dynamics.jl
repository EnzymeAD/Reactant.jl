#!/usr/bin/env julia
#=
Cubic-Cost Kernel Dynamics (LICM with O(D³) loop-invariant construction)
=========================================================================

Model: Same recurrence as forward_model but with a more expensive kernel:
    x_{t+1} = K(θ) · x_t

where K(θ) = (θ · W)' * (θ · W) + I = θ² · W'W + I.

The difference from forward_model: we do NOT precompute W'W. Instead, the
matrix-matrix product (θ·W)' * (θ·W) is computed INSIDE the loop.
This is an O(D³) operation (vs O(D²) element-wise scaling in forward_model).

Pre-AD LICM hoists the O(D³) matrix multiply outside the loop. Without LICM,
the backward pass pays O(T · D³) for the matmul adjoint per iteration.

Usage:
    CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/cubic_kernel_dynamics.jl probprog
    CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/cubic_kernel_dynamics.jl probprog_no_opt
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg

const D_STATE = 60     # state dimension (matmul cost = O(D³))
const T_STEPS = 50     # propagation steps

function cubic_kernel_logpdf(position, W_basis, x0)
    @allowscalar begin
        theta = position[1, 1]
        I_d = Matrix{Float64}(I, D_STATE, D_STATE)

        x = x0   # 1 × D
        @trace track_numbers=false for t in 0:(T_STEPS-1)
            # Kernel construction: O(D³) matrix-matrix multiply (LOOP-INVARIANT)
            W_scaled = theta .* W_basis                        # D × D, O(D²)
            K = permutedims(W_scaled) * W_scaled .+ I_d        # D × D, O(D³)
            # State update (loop-variant recurrence)
            x = x * K                                          # 1 × D, O(D²)
        end

        total = -0.5 * sum(x .* x)
        return total
    end
end

function run_mcmc(rng, pos, ss, W, x0)
    ProbProg.mcmc_logpdf(rng, cubic_kernel_logpdf, pos, W, x0;
        algorithm=:NUTS, step_size=ss, max_tree_depth=5,
        num_warmup=0, num_samples=20,
        adapt_step_size=false, adapt_mass_matrix=false)
end

Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = true

pipeline = Symbol(ARGS[1])
println("=== Cubic Kernel Dynamics: $pipeline, D=$D_STATE, T=$T_STEPS ===")

Random.seed!(42)
W_r = Reactant.to_rarray(randn(D_STATE, D_STATE) * 0.01)
x0_r = Reactant.to_rarray(ones(1, D_STATE) * 0.01)
pos = Reactant.to_rarray(ones(1, 1) * 0.5)
ss = Reactant.ConcreteRNumber(0.1)
rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))

compile_time = @elapsed begin
    compiled_fn = @compile sync=true optimize=pipeline run_mcmc(rng, pos, ss, W_r, x0_r)
end
println("  Compile: $(round(compile_time, digits=1))s")

rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
compiled_fn(rng1, pos, ss, W_r, x0_r)

times = Float64[]
for i in 1:5
    rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100+i, 0]))
    t = @elapsed compiled_fn(rng_i, pos, ss, W_r, x0_r)
    push!(times, t)
end
median_t = sort(times)[3]
println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
println("  All: $(round.(times .* 1000, digits=1))ms")

hlo = @code_hlo optimize=pipeline run_mcmc(rng, pos, ss, W_r, x0_r)
hlo_str = sprint(show, hlo)
n_dot = count("dot_general", hlo_str)
n_while = count("stablehlo.while", hlo_str)
n_dslice = count("dynamic_slice", hlo_str)
println("  IR: $(count("\n", hlo_str)) lines, $n_dot dot_generals, $n_while while-loops, $n_dslice dynamic_slices")
