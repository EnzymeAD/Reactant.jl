#!/usr/bin/env julia
#=
Post-lowering SICM ablation: Scale-Family MVN
==============================================
Model: y ~ MVN(0, τ² * R) where R is a fixed correlation matrix.
SICM decomposes: chol(τ²R) → τ·chol(R), hoists chol(R).

Configs:
  probprog                    → pre-SICM: O(1) per iteration
  probprog_no_opt             → no SICM: O(N³) per iteration
  probprog_no_opt --post-sicm → post-lowering algebraic rewrite on forward only

Usage:
  julia --project=../.. scale_family_mvn.jl probprog
  julia --project=../.. scale_family_mvn.jl probprog_no_opt
  julia --project=../.. scale_family_mvn.jl probprog_no_opt -- --post-sicm
=#
using Random, LinearAlgebra
using Reactant
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

const N = 60

function model_scale_mvn(rng, R_data, y)
    n = Base.length(y)
    _, tau = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:tau)
    tau_2d = reshape(tau, 1, 1)
    K = (tau_2d .* tau_2d) .* R_data
    _, obs = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(n), K, (n,)); symbol=:y)
    return obs
end

function run_bench(rng, model, R, y, step_size, obs_tensor, constrained_addresses, selection, inv_mass)
    trace, _, _ = ProbProg.generate(rng, obs_tensor, model, R, y; constrained_addresses)
    trace, diag, _, _ = ProbProg.mcmc(
        rng, trace, model, R, y;
        selection, algorithm=:NUTS, inverse_mass_matrix=inv_mass,
        step_size, max_tree_depth=10, num_warmup=500, num_samples=500,
        adapt_step_size=true, adapt_mass_matrix=true,
    )
    return trace, diag
end

pipeline = Symbol(ARGS[1])
use_post_sicm = "--post-sicm" in ARGS
if use_post_sicm
    Reactant.Compiler.DEBUG_PROBPROG_POST_SICM[] = true
end
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = (pipeline == :probprog_no_opt)

label = pipeline == :probprog ? "Pre-SICM (full)" :
        use_post_sicm ? "Post-lowering SICM" : "No SICM"
println("=== $label, N=$N ===")

Random.seed!(42)
A = randn(N, N) * 0.3
R = A' * A + I(N)
R = R ./ sqrt.(diag(R) * diag(R)')  # normalize to correlation
y_true = randn(N)

R_r = Reactant.to_rarray(R)
obs = ProbProg.Constraint(:tau => [1.0], :y => y_true)
ca = ProbProg.extract_addresses(obs)
cf = Float64[]; for a in ca; append!(cf, vec(obs[a])); end
ct = Reactant.to_rarray(reshape(cf, 1, :))
selection = ProbProg.select(ProbProg.Address(:tau))
rng = ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
ss = ConcreteRNumber(0.1)
im = ConcreteRArray(ones(1))
y_r = Reactant.to_rarray(y_true)

compile_time = @elapsed begin
    compiled_fn, tt = ProbProg.with_trace() do
        @compile sync=true optimize=pipeline run_bench(rng, model_scale_mvn, R_r, y_r, ss, ct, ca, selection, im)
    end
end
println("  Compile: $(round(compile_time, digits=1))s")

# Warmup + timed runs
compiled_fn(rng, model_scale_mvn, R_r, y_r, ss, ct, ca, selection, im)
times = Float64[]
for i in 1:5
    rng_i = ReactantRNG(Reactant.to_rarray(UInt64[UInt64(100+i), UInt64(0)]))
    t = @elapsed compiled_fn(rng_i, model_scale_mvn, R_r, y_r, ss, ct, ca, selection, im)
    push!(times, t)
end
median_t = sort(times)[3]
println("  Median (500w+500s): $(round(median_t * 1000, digits=1))ms")
println("  All: $(round.(times .* 1000, digits=1))ms")
