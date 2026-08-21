#!/usr/bin/env julia
#=
Mauna Loa CO2 GP regression -- GENERATIVE form (paper-facing snippet).

Same model as maunaloa_gp.jl but expressed generatively with sample sites and an
`enzyme.sample` MultiNormal observation, mirroring the sibling EigenLift models
(sicm/gp_fixed_lengthscale.jl, sicm/eigenlift_solve.jl):

    z_f ~ Normal(0, PS),  sigma_f = exp(z_f)
    z_n ~ Normal(0, PS),  sigma_n = exp(z_n)
    y ~ MultiNormal(0, sigma_f^2 * K_base + sigma_n^2 * I)

Purpose: confirm CholeskyEigenLift fully fires on the generative MultiNormal form
(the form shown in the paper), and that its NUTS posterior matches the custom
logpdf form of maunaloa_gp.jl. Driven directly (generate + mcmc), not through the
PPLBench subprocess harness, so it is comparable to the custom-logpdf driver.

Usage:
  JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
    benchmark/probprog/sicm/maunaloa/maunaloa_gp_generative.jl <sicm|nosicm> <N> [mode]
  mode : ir (default) | bench | parity
=#
using Random
using Reactant, LinearAlgebra
using Reactant: @allowscalar, @compile
using Reactant.ProbProg
using JSON3

const ARM = length(ARGS) >= 1 ? lowercase(ARGS[1]) : "sicm"
const NPTS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 400
const MODE = length(ARGS) >= 3 ? ARGS[3] : "ir"
@assert ARM in ("sicm", "nosicm")
const DISABLE_SICM = (ARM == "nosicm")

const DATA_DIR = get(ENV, "MAUNALOA_DATA_DIR", "/tmp/sbrantq-heatinv/maunaloa")
const PS = parse(Float64, get(ENV, "MAUNALOA_PRIOR_SD", "1.0"))
const STEP_SIZE = parse(Float64, get(ENV, "MAUNALOA_STEP", "0.02"))
const N_SAMPLES = parse(Int, get(ENV, "MAUNALOA_NSAMPLES", "20"))
const MAX_TREE_DEPTH = parse(Int, get(ENV, "MAUNALOA_MTD", "5"))
# Protocol knobs to match the tab:sicm-ablation PPLBench configs
# (pplbench_configs/sicm_ablation/*.json: warmup with both adaptations on).
const N_WARMUP = parse(Int, get(ENV, "MAUNALOA_WARMUP", "0"))
const ADAPT = get(ENV, "MAUNALOA_ADAPT", "0") == "1"

const DATA_PATH = joinpath(DATA_DIR, "maunaloa_data_N$(NPTS).json")
@assert isfile(DATA_PATH) "missing $DATA_PATH"
const DATA = JSON3.read(read(DATA_PATH, String))
const N = Int(DATA["attrs"]["n"])
const KBASE = [Float64(DATA["K_base"][i][j]) for i in 1:N, j in 1:N]
const YVEC = Float64.(DATA["Y"])

# ── Generative model ─────────────────────────────────────────────────────────
function model_maunaloa_gp(rng, K_base, y)
    n = Base.length(y)
    _, z_f = ProbProg.sample(rng, ProbProg.Normal(0.0, PS, (1,)); symbol=:z_f)
    _, z_n = ProbProg.sample(rng, ProbProg.Normal(0.0, PS, (1,)); symbol=:z_n)

    sf2 = exp.(2.0 .* z_f)     # sigma_f^2
    sn2 = exp.(2.0 .* z_n)     # sigma_n^2

    I_n = Matrix{Float64}(I, n, n)
    K = reshape(sf2, 1, 1) .* K_base .+ reshape(sn2, 1, 1) .* I_n

    _, obs = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(n), K, (n,)); symbol=:y)
    return obs
end

const SELECTION = ProbProg.select(ProbProg.Address(:z_f), ProbProg.Address(:z_n))

function flatten_constraint(constraint::ProbProg.Constraint)
    constrained_addresses = ProbProg.extract_addresses(constraint)
    flat = Float64[]
    for addr in constrained_addresses
        append!(flat, vec(constraint[addr]))
    end
    return Reactant.to_rarray(reshape(flat, 1, :)), constrained_addresses
end

function nuts_program(rng, K_r, y_r, ss, imm, ct, ca)
    t, _, _ = ProbProg.generate(rng, ct, model_maunaloa_gp, K_r, y_r;
        constrained_addresses=ca)
    t, diag, _, _ = ProbProg.mcmc(rng, t, model_maunaloa_gp, K_r, y_r;
        selection=SELECTION, algorithm=:NUTS, inverse_mass_matrix=imm,
        step_size=ss, max_tree_depth=MAX_TREE_DEPTH, num_warmup=N_WARMUP,
        num_samples=N_SAMPLES, adapt_step_size=ADAPT, adapt_mass_matrix=ADAPT)
    return t, diag
end

# IR op counting (identical to maunaloa_gp.jl)
function count_in_out_loops(hlo_str, needle)
    region_is_while = Bool[]; pending_while = false; n_in = 0; n_out = 0
    for raw in split(hlo_str, '\n')
        line = strip(raw)
        startswith(line, "}") && !isempty(region_is_while) && pop!(region_is_while)
        if occursin(needle, line)
            c = count(needle, line)
            any(region_is_while) ? (n_in += c) : (n_out += c)
        end
        occursin("stablehlo.while", line) && (pending_while = true)
        if endswith(line, "{")
            isw = (occursin("do {", line) || endswith(line, "cond {")) && pending_while
            push!(region_is_while, isw)
            occursin("do {", line) && (pending_while = false)
        end
    end
    return (in=n_in, out=n_out, total=n_in + n_out)
end
function ir_report(s)
    println("  --- IR op counts (in-loop / out-of-loop / total) ---")
    for nd in ["stablehlo.cholesky", "triangular_solve", "syevd", "dot_general",
               "stablehlo.while", "custom_call"]
        r = count_in_out_loops(s, nd)
        println("    $(rpad(nd, 22)) in=$(r.in)  out=$(r.out)  total=$(r.total)")
    end
end

Reactant.set_default_backend("cpu")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = DISABLE_SICM
println("=== Mauna Loa GP GENERATIVE: arm $ARM (SICM ", DISABLE_SICM ? "OFF" : "ON",
        "), N=$N, step=$STEP_SIZE, mtd=$MAX_TREE_DEPTH, warmup=$N_WARMUP, ",
        "samples=$N_SAMPLES, adapt=$ADAPT, mode=$MODE ===")

K_r = Reactant.to_rarray(KBASE)
y_r = Reactant.to_rarray(YVEC)
imm_r = Reactant.to_rarray(Matrix{Float64}(I, 2, 2))
ss = Reactant.ConcreteRNumber(STEP_SIZE)
# Constraint: z_f, z_n initialized to 0; y fixed to data.
constraint = ProbProg.Constraint(:z_f => [0.0], :z_n => [0.0], :y => YVEC)
ct, ca = flatten_constraint(constraint)

if MODE == "ir"
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    hlo = @code_hlo optimize = :probprog nuts_program(rng, K_r, y_r, ss, imm_r, ct, ca)
    s = sprint(show, hlo)
    ir_report(s)
    open(io -> print(io, s), joinpath(DATA_DIR, "maunaloa_gen_hlo_$(ARM)_N$(N).mlir"), "w")
    println("  saved HLO.")
elseif MODE == "parity"
    seed = UInt64[1, 5]
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    compiled = @compile sync = true optimize = :probprog nuts_program(
        rng, K_r, y_r, ss, imm_r, ct, ca)
    rng2 = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    samples, diag = compiled(rng2, K_r, y_r, ss, imm_r, ct, ca)
    sarr = Array(samples)
    # tt layout: extract selected columns via unflatten is overkill; the trace
    # tensor rows are samples, cols are [z_f, z_n] in selection order.
    println("  gen samples (first col, second col):")
    for j in 1:size(sarr, 1)
        println("    [$j] $(sarr[j, 1])  $(sarr[j, 2])")
    end
    out = Dict("samples" => [collect(sarr[j, :]) for j in 1:size(sarr, 1)])
    open(io -> JSON3.write(io, out),
        joinpath(DATA_DIR, "maunaloa_gen_samples_N$(N)_$(ARM).json"), "w")
    println("  wrote generative samples.")
else  # bench
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    ct_c = Array(ct)
    compiled = @compile sync = true optimize = :probprog nuts_program(
        rng, K_r, y_r, ss, imm_r, ct, ca)
    rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compiled(rng1, K_r, y_r, ss, imm_r, ct, ca)
    n_runs = parse(Int, get(ENV, "MAUNALOA_RUNS", "5"))
    times = Float64[]
    for i in 1:n_runs
        rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100 + i, 0]))
        push!(times, @elapsed compiled(rng_i, K_r, y_r, ss, imm_r, ct, ca))
    end
    st = sort(times)
    println("  Median: $(round(st[(length(st)+1)÷2]*1000, digits=1))ms  All: $(round.(times.*1000,digits=1))ms")
end
