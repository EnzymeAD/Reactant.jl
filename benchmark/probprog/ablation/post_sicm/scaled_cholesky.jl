#!/usr/bin/env julia
#=
Post-lowering SICM ablation: Scaled Cholesky model
===================================================

Model: log p(y|σ) = -0.5 * y' * (σ²·C + I)^{-1} * y
  σ is the inferred parameter, C is fixed data covariance.

The logpdf contains cholesky(σ²·C + I) which SICM decomposes via
CholeskyEigenLift (or CholeskyScale for the simpler σ²·C case).

Three pipeline configs:
  1. :probprog (pre-SICM): decomposes before AD → O(N²) gradient
  2. :probprog_no_opt (no SICM): full O(N³) cholesky per gradient
  3. :probprog_no_opt + POST_SICM: decomposes forward cholesky after AD,
     but backward pass is already O(N³) → partial benefit only

Usage:
  julia --project=../.. scaled_cholesky.jl probprog
  julia --project=../.. scaled_cholesky.jl probprog_no_opt
  julia --project=../.. scaled_cholesky.jl probprog_no_opt --post-sicm
=#
using Random, LinearAlgebra
using Reactant
using Reactant: @allowscalar, @compile
using Reactant.ProbProg

const D = 40

function scaled_chol_logpdf(position, C_base, y_obs)
    @allowscalar begin
        sigma = position[1, 1]
        sigma_sq = sigma * sigma
        I_d = Matrix{Float64}(I, D, D)

        # Scaled covariance: Q = σ² * C + I
        Q = sigma_sq .* C_base .+ I_d

        # Cholesky factorization (this is the SICM target)
        C_factor = cholesky(Q)

        # Mahalanobis distance: y' * Q^{-1} * y via cholesky solve
        alpha = C_factor \ permutedims(y_obs)  # D × N_obs
        mahal = sum(alpha .* alpha)

        return -0.5 * mahal
    end
end

function run_mcmc(rng, pos, ss, C, y)
    ProbProg.mcmc_logpdf(rng, scaled_chol_logpdf, pos, C, y;
        algorithm=:NUTS, step_size=ss, max_tree_depth=5,
        num_warmup=0, num_samples=50,
        adapt_step_size=false, adapt_mass_matrix=false)
end

pipeline = Symbol(ARGS[1])
use_post_sicm = "--post-sicm" in ARGS
if use_post_sicm
    Reactant.Compiler.DEBUG_PROBPROG_POST_SICM[] = true
end
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = (pipeline == :probprog_no_opt)

label = if pipeline == :probprog
    "Pre-SICM (full)"
elseif use_post_sicm
    "Post-lowering SICM"
else
    "No SICM"
end
println("=== $label: $pipeline, D=$D ===")

Random.seed!(42)
A = randn(D, D) * 0.2
C_r = Reactant.to_rarray(A' * A + 0.1 * I(D))
y_r = Reactant.to_rarray(randn(1, D))
pos = Reactant.to_rarray(ones(1, 1) * 0.5)
ss = Reactant.ConcreteRNumber(0.1)
rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))

compile_time = @elapsed begin
    compiled_fn = @compile sync=true optimize=pipeline run_mcmc(rng, pos, ss, C_r, y_r)
end
println("  Compile: $(round(compile_time, digits=1))s")

rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
compiled_fn(rng1, pos, ss, C_r, y_r)

times = Float64[]
for i in 1:5
    rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100+i, 0]))
    t = @elapsed compiled_fn(rng_i, pos, ss, C_r, y_r)
    push!(times, t)
end
median_t = sort(times)[3]
println("  Median (50 NUTS): $(round(median_t * 1000, digits=1))ms")
println("  All: $(round.(times .* 1000, digits=1))ms")
