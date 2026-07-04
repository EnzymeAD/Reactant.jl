#!/usr/bin/env julia
#=
Scaled Multi-Sensor Fusion (DotGeneralScale standalone benchmark)
==================================================================

Model: K sensors observe a shared latent state through known measurement
matrices M_k, each scaled by a common gain parameter σ:

    z_k = (σ · M_k) · s_k       for k = 1..K

where M_k (D_obs × D_state) are known sensor-specific measurement matrices,
s_k (D_state) are known calibration signals, and σ is the unknown gain.

Log-likelihood:  -½ Σ_k ||z_k - y_k||²

Without DotGeneralScale, SICM sees `σ · M_k · s_k` as sample-dependent
(σ is the sampled parameter) and cannot hoist any matrix operations.

With DotGeneralScale:
  (σ · M_k) · s_k  →  σ · (M_k · s_k)
  Now M_k · s_k is sample-invariant → hoisted outside the MCMC loop.

This eliminates K × O(D²) matrix-vector products per leapfrog step,
replacing them with K × O(D) scalar-vector multiplications.

Usage:
    CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/scaled_multi_sensor.jl probprog
    CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/scaled_multi_sensor.jl probprog_no_opt
=#
using Random
using Reactant, LinearAlgebra
using Reactant: @allowscalar, @compile
using Reactant.ProbProg

const D_STATE = 300    # state dimension (matvec cost = O(D²))
const D_OBS = 300      # observation dimension
const K_SENSORS = 10   # number of sensors

function multi_sensor_logpdf(
    position,
    M1, M2, M3, M4, M5, M6, M7, M8, M9, M10,
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10
)
    @allowscalar begin
        sigma = position[1, 1]

        # Each sensor: z_k = (sigma * M_k) @ s_k
        # DotGeneralScale: → sigma * (M_k @ s_k), hoisting M_k @ s_k
        z1 = (sigma .* M1) * s1
        z2 = (sigma .* M2) * s2
        z3 = (sigma .* M3) * s3
        z4 = (sigma .* M4) * s4
        z5 = (sigma .* M5) * s5
        z6 = (sigma .* M6) * s6
        z7 = (sigma .* M7) * s7
        z8 = (sigma .* M8) * s8
        z9 = (sigma .* M9) * s9
        z10 = (sigma .* M10) * s10

        # Observation log-likelihood: -½ Σ_k ||z_k - y_k||²
        loglik = -0.5 * (
            sum((z1 .- y1) .* (z1 .- y1)) +
            sum((z2 .- y2) .* (z2 .- y2)) +
            sum((z3 .- y3) .* (z3 .- y3)) +
            sum((z4 .- y4) .* (z4 .- y4)) +
            sum((z5 .- y5) .* (z5 .- y5)) +
            sum((z6 .- y6) .* (z6 .- y6)) +
            sum((z7 .- y7) .* (z7 .- y7)) +
            sum((z8 .- y8) .* (z8 .- y8)) +
            sum((z9 .- y9) .* (z9 .- y9)) +
            sum((z10 .- y10) .* (z10 .- y10))
        )
        return loglik
    end
end

function run_mcmc(
    rng, pos, ss,
    M1, M2, M3, M4, M5, M6, M7, M8, M9, M10,
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10
)
    ProbProg.mcmc_logpdf(
        rng, multi_sensor_logpdf, pos,
        M1, M2, M3, M4, M5, M6, M7, M8, M9, M10,
        s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
        y1, y2, y3, y4, y5, y6, y7, y8, y9, y10;
        algorithm=:NUTS, step_size=ss, max_tree_depth=5,
        num_warmup=0, num_samples=20,
        adapt_step_size=false, adapt_mass_matrix=false
    )
end

pipeline = Symbol(ARGS[1])
println("=== Scaled Multi-Sensor: $pipeline, D=$D_STATE, K=$K_SENSORS ===")

Random.seed!(42)
# Generate K sensor matrices (D_obs × D_state) and calibration signals (D_state × 1)
Ms = [Reactant.to_rarray(randn(D_OBS, D_STATE) * 0.01) for _ in 1:K_SENSORS]
ss_vecs = [Reactant.to_rarray(randn(D_STATE, 1) * 1.0) for _ in 1:K_SENSORS]
# Generate synthetic observations (y_k = true_sigma * M_k * s_k + noise)
true_sigma = 2.0
ys = [Reactant.to_rarray(true_sigma * (randn(D_OBS, D_STATE) * 0.01) * (randn(D_STATE, 1)) + randn(D_OBS, 1) * 0.1) for _ in 1:K_SENSORS]

pos = Reactant.to_rarray(ones(1, 1) * 1.5)
ss = Reactant.ConcreteRNumber(0.1)
rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))

compile_time = @elapsed begin
    compiled_fn = @compile sync=true optimize=pipeline run_mcmc(
        rng, pos, ss,
        Ms[1], Ms[2], Ms[3], Ms[4], Ms[5], Ms[6], Ms[7], Ms[8], Ms[9], Ms[10],
        ss_vecs[1], ss_vecs[2], ss_vecs[3], ss_vecs[4], ss_vecs[5],
        ss_vecs[6], ss_vecs[7], ss_vecs[8], ss_vecs[9], ss_vecs[10],
        ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7], ys[8], ys[9], ys[10]
    )
end
println("  Compile: $(round(compile_time, digits=1))s")

# Warmup
rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
compiled_fn(
    rng1, pos, ss,
    Ms[1], Ms[2], Ms[3], Ms[4], Ms[5], Ms[6], Ms[7], Ms[8], Ms[9], Ms[10],
    ss_vecs[1], ss_vecs[2], ss_vecs[3], ss_vecs[4], ss_vecs[5],
    ss_vecs[6], ss_vecs[7], ss_vecs[8], ss_vecs[9], ss_vecs[10],
    ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7], ys[8], ys[9], ys[10]
)

# Timed runs
times = Float64[]
for i in 1:5
    rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100+i, 0]))
    t = @elapsed compiled_fn(
        rng_i, pos, ss,
        Ms[1], Ms[2], Ms[3], Ms[4], Ms[5], Ms[6], Ms[7], Ms[8], Ms[9], Ms[10],
        ss_vecs[1], ss_vecs[2], ss_vecs[3], ss_vecs[4], ss_vecs[5],
        ss_vecs[6], ss_vecs[7], ss_vecs[8], ss_vecs[9], ss_vecs[10],
        ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7], ys[8], ys[9], ys[10]
    )
    push!(times, t)
end
median_t = sort(times)[3]
println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
println("  All: $(round.(times .* 1000, digits=1))ms")

hlo = @code_hlo optimize=pipeline run_mcmc(
    rng, pos, ss,
    Ms[1], Ms[2], Ms[3], Ms[4], Ms[5], Ms[6], Ms[7], Ms[8], Ms[9], Ms[10],
    ss_vecs[1], ss_vecs[2], ss_vecs[3], ss_vecs[4], ss_vecs[5],
    ss_vecs[6], ss_vecs[7], ss_vecs[8], ss_vecs[9], ss_vecs[10],
    ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7], ys[8], ys[9], ys[10]
)
hlo_str = sprint(show, hlo)
n_dot = count("dot_general", hlo_str)
n_while = count("stablehlo.while", hlo_str)
println("  IR: $(count("\n", hlo_str)) lines, $n_dot dot_generals, $n_while while-loops")
