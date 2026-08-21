#!/usr/bin/env julia
#=
Mauna Loa CO2 GP regression -- SICM (CholeskyEigenLift) ablation benchmark
=========================================================================

Textbook Gaussian-process regression (Rasmussen & Williams, GPML 2006, Ch. 5.4)
on the Scripps/NOAA Mauna Loa monthly CO2 record. A quadratic secular trend is
removed offline; the residual (dominated by the 12-month seasonal cycle plus
medium-term wiggles) is modeled with a FIXED-shape kernel matrix K_base (periodic
+ RBF, computed once by maunaloa_gp_prep.py, so it is sample-invariant). We infer
only the overall signal amplitude and the observation-noise scale:

    z_f ~ Normal(0, 1),  sigma_f = exp(z_f)          (log signal amplitude)
    z_n ~ Normal(0, 1),  sigma_n = exp(z_n)          (log noise scale)
    y ~ MVN(0, sigma_f^2 * K_base + sigma_n^2 * I)

The covariance is exactly the s*A + t*I form that SICM's CholeskyEigenLift
targets: A = K_base is sample-invariant, s = sigma_f^2 and t = sigma_n^2 are
sample-dependent. Per NUTS gradient:
  SICM off: cholesky(sigma_f^2 K_base + sigma_n^2 I) inside the sampler loop, O(N^3)
  SICM on : eig(K_base) hoisted once (enzymexla.syevd, outside all loops),
            leaving an O(N) eigenvalue shift + O(N^2) solves per gradient.

Model form: custom logpdf via ProbProg.mcmc_logpdf with the MVN density spelled
out (same vehicle as ablation/pre_lowering/heat_inverse.jl), so density/trajectory
parity against a NumPyro potential_fn is exact. The MVN density here is identical
(same gradients) to the generative `y ~ MultiNormal(0, K)` form used by the
sibling EigenLift models (sicm/gp_fixed_lengthscale.jl, sicm/eigenlift_solve.jl);
that generative form is the paper-facing snippet.

Usage (from Reactant.jl root):
  JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
    benchmark/probprog/sicm/maunaloa/maunaloa_gp.jl <sicm|nosicm> <N> [mode]
  N    : subsample size (data file maunaloa_data_N<N>.json must exist), e.g. 400
  mode : bench (default) | parity | point | ir
=#
using Random
using Reactant, LinearAlgebra
using Reactant: @allowscalar, @compile
using Reactant.ProbProg
using JSON3

# ── CLI ──────────────────────────────────────────────────────────────────────
const ARM = length(ARGS) >= 1 ? lowercase(ARGS[1]) : "sicm"
const NPTS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 400
const MODE = length(ARGS) >= 3 ? ARGS[3] : "bench"
@assert ARM in ("sicm", "nosicm") "arm must be sicm|nosicm (got $ARM)"
const DISABLE_SICM = (ARM == "nosicm")

const DATA_DIR = get(ENV, "MAUNALOA_DATA_DIR", "/tmp/sbrantq-heatinv/maunaloa")
const PS = parse(Float64, get(ENV, "MAUNALOA_PRIOR_SD", "1.0"))   # LogNormal(0, PS)
const STEP_SIZE = parse(Float64, get(ENV, "MAUNALOA_STEP", "0.02"))
const N_SAMPLES = parse(Int, get(ENV, "MAUNALOA_NSAMPLES", "20"))
const MAX_TREE_DEPTH = parse(Int, get(ENV, "MAUNALOA_MTD", "5"))
const LOG2PI = log(2 * pi)

# ── Load data ────────────────────────────────────────────────────────────────
const DATA_PATH = joinpath(DATA_DIR, "maunaloa_data_N$(NPTS).json")
@assert isfile(DATA_PATH) "missing data file $DATA_PATH (run maunaloa_gp_prep.py)"
const DATA = JSON3.read(read(DATA_PATH, String))
const N = Int(DATA["attrs"]["n"])
const KBASE = [Float64(DATA["K_base"][i][j]) for i in 1:N, j in 1:N]
const YVEC = Float64.(DATA["Y"])

# ── Model: log joint density (returns a log-density scalar) ──────────────────
# position = [z_f, z_n]; sigma_f = exp(z_f), sigma_n = exp(z_n).
# K = sigma_f^2 K_base + sigma_n^2 I  (the s*A + t*I EigenLift target).
function maunaloa_logpdf(position, K_base, ycol)
    z_f = @allowscalar position[1]
    z_n = @allowscalar position[2]
    sf2 = exp(2.0 * z_f)      # sigma_f^2
    sn2 = exp(2.0 * z_n)      # sigma_n^2

    I_n = Matrix{Float64}(I, N, N)
    K = sf2 .* K_base .+ sn2 .* I_n     # N x N, SPD, sample-dependent
    C = cholesky(K)                      # chol(s*A + t*I) -> EigenLift target
    sol = C \ ycol                       # K^-1 y (full solve chain, 2 tri-solves)
    quad = sum(ycol .* sol)              # y^T K^-1 y
    # log|K| = 2*sum(log(diag(U))). Extract diag(U) via multiply-by-identity so
    # SICM's CholeskyEigenLift recognizes it (isDiagonalExtraction matches
    # multiply(L, I)); a strided-slice diag would leave the cholesky un-erased.
    diagU = vec(sum(C.factors .* I_n; dims=1))   # column sums -> U[j,j]
    logdet = 2.0 * sum(log.(diagU))              # log|K|

    loglik = -0.5 * (N * LOG2PI + logdet + quad)
    logprior = -0.5 * (z_f * z_f + z_n * z_n) / (PS * PS) - LOG2PI - 2.0 * log(PS)
    return logprior + loglik
end

function run_mcmc(rng, pos, ss, imm, K_base, ycol; num_samples=N_SAMPLES)
    ProbProg.mcmc_logpdf(rng, maunaloa_logpdf, pos, K_base, ycol;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm,
        max_tree_depth=MAX_TREE_DEPTH, num_warmup=0, num_samples=num_samples,
        adapt_step_size=false, adapt_mass_matrix=false)
end

# ── IR analysis: count ops split by stablehlo.while-body membership. ──────────
# (Identical bookkeeping to heat_inverse.jl.)
function count_in_out_loops(hlo_str::AbstractString, needle::AbstractString)
    region_is_while = Bool[]
    pending_while = false
    n_in = 0
    n_out = 0
    for raw in split(hlo_str, '\n')
        line = strip(raw)
        if startswith(line, "}") && !isempty(region_is_while)
            pop!(region_is_while)
        end
        if occursin(needle, line)
            c = count(needle, line)
            if any(region_is_while)
                n_in += c
            else
                n_out += c
            end
        end
        occursin("stablehlo.while", line) && (pending_while = true)
        if endswith(line, "{")
            is_while_body = (occursin("do {", line) || endswith(line, "cond {")) &&
                            pending_while
            push!(region_is_while, is_while_body)
            occursin("do {", line) && (pending_while = false)
        end
    end
    return (in=n_in, out=n_out, total=n_in + n_out)
end

function ir_report(hlo_str::AbstractString)
    needles = ["stablehlo.cholesky", "triangular_solve", "syevd",
               "dot_general", "stablehlo.while", "custom_call"]
    println("  --- IR op counts (in-loop / out-of-loop / total) ---")
    for nd in needles
        r = count_in_out_loops(hlo_str, nd)
        println("    $(rpad(nd, 22)) in=$(r.in)  out=$(r.out)  total=$(r.total)")
    end
    println("    $(rpad("lines", 22)) $(count(==('\n'), hlo_str))")
end

# ── Setup ────────────────────────────────────────────────────────────────────
Reactant.set_default_backend("cpu")
# Region path (required for SICM); toggle SICM only.
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = DISABLE_SICM

println("=== Mauna Loa GP: arm $ARM (SICM ", DISABLE_SICM ? "OFF" : "ON",
        "), N=$N, step=$STEP_SIZE, mtd=$MAX_TREE_DEPTH, samples=$N_SAMPLES, mode=$MODE ===")
println("  data: $DATA_PATH (resid_std=$(round(Float64(DATA["attrs"]["resid_std"]), sigdigits=4)))")

K_r = Reactant.to_rarray(KBASE)
y_r = Reactant.to_rarray(reshape(YVEC, N, 1))          # N x 1
imm_r = Reactant.to_rarray(Matrix{Float64}(I, 2, 2))   # dense identity mass
pos0 = Reactant.to_rarray(reshape([0.0, 0.0], 1, 2))   # z-space zeros
ss = Reactant.ConcreteRNumber(STEP_SIZE)

# ── Modes ────────────────────────────────────────────────────────────────────
if MODE == "point"
    zpts = [[0.0, 0.0], [0.7, -0.7], [0.3, -0.2], [-0.4, 0.5]]
    eval_lp(z, K, y) = maunaloa_logpdf(z, K, y)
    println("  compiling point evaluator (optimize=:probprog)...")
    z_r0 = Reactant.to_rarray(zpts[1])
    compiled_lp = @compile optimize = :probprog eval_lp(z_r0, K_r, y_r)
    results = Dict{String,Float64}()
    for z in zpts
        z_r = Reactant.to_rarray(z)
        val = Float64(compiled_lp(z_r, K_r, y_r))
        results["$(z)"] = val
        println("    logpdf(z=$z) = $(val)")
    end
    open(joinpath(DATA_DIR, "maunaloa_logpdf_julia_N$(N).json"), "w") do io
        JSON3.write(io, results)
    end
    println("  wrote julia logpdf values.")

elseif MODE == "parity"
    seed = UInt64[1, 5]
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    println("  compiling parity run (arm $ARM)...")
    compiled_fn = @compile sync = true optimize = :probprog run_mcmc(
        rng, pos0, ss, imm_r, K_r, y_r; num_samples=N_SAMPLES)
    rng2 = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    samples, diagnostics, _, _ = compiled_fn(rng2, pos0, ss, imm_r, K_r, y_r)
    samples_arr = Array(samples)
    diag_arr = Array(diagnostics)
    println("  samples (z_f, z_n):")
    for j in 1:size(samples_arr, 1)
        println("    [$j] $(samples_arr[j, 1])  $(samples_arr[j, 2])   accepted=$(diag_arr[j])")
    end
    out = Dict("seed" => Int.(seed), "step_size" => STEP_SIZE,
        "max_tree_depth" => MAX_TREE_DEPTH,
        "samples" => [collect(samples_arr[j, :]) for j in 1:size(samples_arr, 1)],
        "accepted" => Int.(diag_arr))
    open(joinpath(DATA_DIR, "maunaloa_samples_julia_N$(N)_$(ARM).json"), "w") do io
        JSON3.write(io, out)
    end
    println("  wrote julia samples. n_accepted=$(sum(diag_arr))")

elseif MODE == "ir"
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    hlo = @code_hlo optimize = :probprog run_mcmc(rng, pos0, ss, imm_r, K_r, y_r)
    hlo_str = sprint(show, hlo)
    ir_report(hlo_str)
    hlo_path = joinpath(DATA_DIR, "maunaloa_hlo_$(ARM)_N$(N).mlir")
    open(io -> print(io, hlo_str), hlo_path, "w")
    println("  saved HLO: $hlo_path")

else  # bench
    println("  compiling (optimize=:probprog, disable_sicm=$DISABLE_SICM)...")
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compile_time = @elapsed begin
        compiled_fn = @compile sync = true optimize = :probprog run_mcmc(
            rng, pos0, ss, imm_r, K_r, y_r)
    end
    println("  Compile: $(round(compile_time, digits=1))s")

    rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compiled_fn(rng1, pos0, ss, imm_r, K_r, y_r)   # warmup

    n_runs = parse(Int, get(ENV, "MAUNALOA_RUNS", "5"))
    times = Float64[]
    for i in 1:n_runs
        rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100 + i, 0]))
        t = @elapsed compiled_fn(rng_i, pos0, ss, imm_r, K_r, y_r)
        push!(times, t)
    end
    sorted_t = sort(times)
    median_t = sorted_t[(length(sorted_t) + 1) ÷ 2]
    println("  Median ($(N_SAMPLES) NUTS samples): $(round(median_t * 1000, digits=1))ms")
    println("  All: $(round.(times .* 1000, digits=1))ms")

    hlo = @code_hlo optimize = :probprog run_mcmc(rng, pos0, ss, imm_r, K_r, y_r)
    hlo_str = sprint(show, hlo)
    ir_report(hlo_str)
    hlo_path = joinpath(DATA_DIR, "maunaloa_hlo_$(ARM)_N$(N).mlir")
    open(io -> print(io, hlo_str), hlo_path, "w")
    println("  saved HLO: $hlo_path")
end
