#!/usr/bin/env julia
#=
State-Space Model with Loop-Invariant Cholesky (O(D³) hoisting)
================================================================

Model: A D-dimensional latent state evolves via a noisy linear dynamical system:
    x_{t+1} = A_fixed · x_t + L(σ) · ε_t

where L(σ) = cholesky(σ² · C_base + I) is the noise Cholesky factor.
  - A_fixed is a known D×D stable transition matrix (spectral radius < 1)
  - C_base is a known D×D spatial correlation matrix
  - σ (scalar) controls the process noise scale
  - ε_t = randn(D) is white noise (absorbed into the deterministic recurrence
    by fixing the noise sequence for differentiability)

The Cholesky L(σ) is loop-invariant: it depends on σ but not on t or x_t.
Pre-AD LICM hoists the O(D³) Cholesky outside the T-step loop before
differentiation. Without this, Enzyme caches L per iteration and the backward
pass pays O(T · D³) for the Cholesky adjoint chain.

Comparison with forward_model.jl:
  forward_model: loop-invariant O(D²) matrix construction (W'W scaling)
  this model:    loop-invariant O(D³) Cholesky decomposition
  Expected LICM savings are larger here due to cubic vs quadratic cost.

Usage:
    CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/cholesky_recurrence.jl probprog
    CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/cholesky_recurrence.jl probprog_no_opt
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg

const D_STATE = 40     # state dimension (Cholesky cost = O(D³))
const T_STEPS = 50     # propagation steps

function cholesky_recurrence_logpdf(position, A_fixed, C_base, eps_seq, x0)
    @allowscalar begin
        sigma = position[1, 1]    # process noise scale

        I_d = Matrix{Float64}(I, D_STATE, D_STATE)

        x = x0   # 1 × D
        @trace track_numbers=false for t in 0:(T_STEPS-1)
            # Noise covariance Cholesky (LOOP-INVARIANT: depends on σ, not t)
            Sigma = (sigma * sigma) .* C_base .+ I_d    # D × D
            L = cholesky(Sigma)                          # O(D³), loop-invariant

            # Deterministic transition + noise injection
            x_pred = x * A_fixed                         # 1 × D
            noise = reshape(eps_seq[t+1, :], 1, D_STATE) * L.factors  # 1 × D
            x = x_pred .+ noise
        end

        # Observation log-likelihood: -½ ||x_T||²
        total = -0.5 * sum(x .* x)
        return total
    end
end

function run_mcmc(rng, pos, ss, A_fixed, C_base, eps_seq, x0)
    ProbProg.mcmc_logpdf(rng, cholesky_recurrence_logpdf, pos, A_fixed, C_base, eps_seq, x0;
        algorithm=:NUTS, step_size=ss, max_tree_depth=5,
        num_warmup=0, num_samples=20,
        adapt_step_size=false, adapt_mass_matrix=false)
end

Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = true

pipeline = Symbol(ARGS[1])
println("=== Cholesky Recurrence: $pipeline, D=$D_STATE, T=$T_STEPS ===")

Random.seed!(42)
# Stable transition matrix (spectral radius < 1)
A_raw = randn(D_STATE, D_STATE) * 0.3 / sqrt(D_STATE)
A_fixed_np = A_raw - Diagonal(diag(A_raw)) + 0.5 * I  # diagonal dominance → stable
A_r = Reactant.to_rarray(A_fixed_np)

# Spatial correlation base (positive definite)
V = randn(D_STATE, D_STATE) * 0.1
C_base_np = V' * V + 0.1 * I
C_r = Reactant.to_rarray(C_base_np)

# Fixed noise sequence (T × D)
eps_r = Reactant.to_rarray(randn(T_STEPS, D_STATE) * 0.01)

# Initial state
x0_r = Reactant.to_rarray(ones(1, D_STATE) * 0.01)
# Initial position (noise scale)
pos = Reactant.to_rarray(ones(1, 1) * 0.5)
ss = Reactant.ConcreteRNumber(0.1)
rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))

compile_time = @elapsed begin
    compiled_fn = @compile sync=true optimize=pipeline run_mcmc(rng, pos, ss, A_r, C_r, eps_r, x0_r)
end
println("  Compile: $(round(compile_time, digits=1))s")

# Warmup
rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
compiled_fn(rng1, pos, ss, A_r, C_r, eps_r, x0_r)

# Timed runs
times = Float64[]
for i in 1:5
    rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100+i, 0]))
    t = @elapsed compiled_fn(rng_i, pos, ss, A_r, C_r, eps_r, x0_r)
    push!(times, t)
end
median_t = sort(times)[3]
println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
println("  All: $(round.(times .* 1000, digits=1))ms")

# IR stats
hlo = @code_hlo optimize=pipeline run_mcmc(rng, pos, ss, A_r, C_r, eps_r, x0_r)
hlo_str = sprint(show, hlo)
n_chol = count("cholesky", hlo_str)
n_dot = count("dot_general", hlo_str)
n_while = count("stablehlo.while", hlo_str)
println("  IR: $(count("\n", hlo_str)) lines, $n_chol choleskys, $n_dot dot_generals, $n_while while-loops")
