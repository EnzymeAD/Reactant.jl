#!/usr/bin/env julia
#=
Heat-Equation Bayesian Inverse Problem (v2: generative form)
============================================================

The paper-facing flagship model. Same physics, data, and hyperparameters as v1
(`heat_inverse.jl`), but written in natural PPL form with `ProbProg.sample`
sites: the two log-space parameters are latent sites at top level, and the
observation `y_j ~ Normal(H x_{t_j}, sigma)` is ONE observed site placed inside
a `@trace for` observation-window loop (WS-A Stage 1: an observed in-loop site
stacks to one address of value shape (S, M_obs), NumPyro-scan semantics).

    z_kappa ~ Normal(0,1),  kappa = exp(z_kappa)
    z_sigma ~ Normal(0,1),  sigma = exp(z_sigma)
    M(kappa) = I + dt*kappa*L,   M x_{t+1} = x_t     (cholesky + 2 solves/step)
    for j = 0..M_obs-1:  <advance OBS_STEP implicit-Euler steps>;  y_j ~ Normal(H x, sigma)

v2 runs on the FLAT expand path (the region/SICM path does not yet support
in-loop sample sites). v1 owns the ablation ladder; v2's job is posterior
identity plus being the paper-facing model code. The joint density is identical
to v1's custom logpdf up to the same constant set (full Normal normalization).

Usage (from Reactant.jl root):
  JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
    benchmark/probprog/ablation/pre_lowering/heat_inverse_v2.jl <scale> [mode] [num_samples]
  scale: 0 (smoke D=20,T=10), 1 (D=100,T=100), 2 (D=300,T=30)
  mode : point (density identity, default) | parity (mcmc samples) | bench | ir
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray
using JSON3

include(joinpath(@__DIR__, "heat_inverse_data.jl"))

# Yield-based wait so the enzyme.dump FFI callback can fire during synchronous
# execution; gated on the dump flag (as in harness.jl) so timing runs keep the
# stock blocking wait.
import Reactant.XLA.PJRT: Future
if Reactant.Compiler.DEBUG_PROBPROG_DUMP_VALUE[]
    function Base.wait(future::Future)::Nothing
        GC.@preserve future begin
            while !isready(future)
                GC.safepoint()
                yield()
            end
        end
        return nothing
    end
end

# ── CLI ──────────────────────────────────────────────────────────────────────
const SCALE = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
const MODE = length(ARGS) >= 2 ? ARGS[2] : "point"
const NUM_SAMPLES = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20

# ── Config / constants (baked into the traced model) ─────────────────────────
const CFG = heat_config(SCALE)
const D = CFG.D
const T = CFG.T
const DT = CFG.dt
const M_OBS = CFG.M_obs
const S = CFG.S
const OBS_STEP = CFG.obs_step        # coarse steps between observations
const SENSOR_IDX = CFG.sensor_idx    # Vector{Int}, length S

# ── Path selection ───────────────────────────────────────────────────────────
Reactant.set_default_backend("cpu")
# Flat expand path (region path does not support in-loop sample sites yet).
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = true

# ── v2 generative model ──────────────────────────────────────────────────────
#
# Latent priors are top-level sites; the observation site sits directly in the
# outer `@trace for` observation-window loop (one stacked `:y` address of value
# shape (S, M_obs)). The inner sample-free `@trace for` advances the implicit-
# Euler recurrence OBS_STEP steps with the cholesky + two triangular solves in
# the loop body (identical chunked structure to v1). 0-based loops: the flat
# reverse-AD while cache removal requires start-0 step-1 loops.
function heat_inverse_v2(rng, L, x0col)
    _, z_kappa = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:z_kappa)
    _, z_sigma = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:z_sigma)
    kappa = exp(@allowscalar z_kappa[1])
    sigma = exp(@allowscalar z_sigma[1])

    I_d = Matrix{Float64}(I, D, D)
    xcol = x0col .+ 0.0                        # fresh buffer (D x 1 state)
    # track_numbers=false: keep captured plain constants (SENSOR_IDX::Vector{Int})
    # untraced so the sensor gather stays a static slice.
    @trace track_numbers = false for j in 0:(M_OBS - 1)
        # Advance OBS_STEP implicit-Euler steps to the next observation time.
        @trace track_numbers = false for t in 0:(OBS_STEP - 1)
            M = I_d .+ (DT * kappa) .* L        # D x D, SPD, loop-invariant
            F = cholesky(M)                      # O(D^3), loop-invariant
            xcol = F \ xcol                      # 2 triangular solves, D x 1
        end
        obs_j = xcol[SENSOR_IDX, 1]              # length S sensor readings (H x)
        _, y = ProbProg.sample(rng, ProbProg.Normal(obs_j, sigma, (S,)); symbol=:y)
    end
    return xcol
end

# The stacked `:y` constraint has value shape (shape..., T) = (S, M_obs).
# v1's y matrix is (M_obs, S); permutedims -> (S, M_obs) so column j (window j)
# holds that window's S sensor values, matching the time-major stacked layout.
y_stacked_matrix(y_data) = permutedims(y_data)     # (S, M_obs)

const SELECTION = ProbProg.select(ProbProg.Address(:z_kappa), ProbProg.Address(:z_sigma))

# ── Setup ────────────────────────────────────────────────────────────────────
cfg_data = heat_generate_data(CFG)
data_path = heat_data_path(SCALE)
heat_save_data(data_path, merge(CFG, cfg_data), cfg_data)
println("=== Heat Inverse v2 (generative, flat path), scale $SCALE, ",
        "D=$D T=$T M_obs=$M_OBS S=$S, mode=$MODE ===")
println("  data: kappa_true=$(CFG.kappa_true) sigma_true=$(CFG.sigma_true) dt=$(round(DT, sigdigits=4))")
println("  data file: $data_path")

L_r = Reactant.to_rarray(heat_laplacian(D))
Y_STACKED = y_stacked_matrix(cfg_data.y)                 # (S, M_obs)
imm_r = Reactant.to_rarray(Matrix{Float64}(I, 2, 2))     # dense identity mass
const STEP_SIZE = parse(Float64, get(ENV, "HEATINV_STEP", "0.02"))

if MODE == "point"
    # ── Density identity: generate_ with ALL sites constrained returns weight =
    # full joint log-density. Compare to v1's logpdf value at the same z point. ──
    zpts = [[0.0, 0.0], [0.3, -0.2], [-0.5, 0.1], [0.2, 0.5]]
    v1_ref_path = joinpath(dirname(data_path), "heat_logpdf_julia_scale$(SCALE).json")
    v1_ref = isfile(v1_ref_path) ? JSON3.read(read(v1_ref_path, String)) : nothing

    results = Dict{String,Float64}()
    worst_rel = Ref(0.0)
    for z in zpts
        rng = ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
        x0_fresh = Reactant.to_rarray(reshape(cfg_data.x0_coarse, D, 1))
        local cons = ProbProg.Constraint(
            :z_kappa => [z[1]], :z_sigma => [z[2]], :y => Y_STACKED
        )
        _, weight = ProbProg.generate_(rng, cons, heat_inverse_v2, L_r, x0_fresh)
        val = Float64(weight)
        results["$(z)"] = val
        refstr = ""
        if v1_ref !== nothing && haskey(v1_ref, Symbol("$(z)"))
            ref = Float64(v1_ref[Symbol("$(z)")])
            rel = abs(val - ref) / max(abs(ref), 1e-300)
            worst_rel[] = max(worst_rel[], rel)
            refstr = "   v1=$(ref)   rel=$(rel)"
        end
        println("    logpdf_v2(z=$z) = $(val)$refstr"); flush(stdout)
    end
    println("  worst rel diff vs v1 = $(worst_rel[])"); flush(stdout)
    open(joinpath(dirname(data_path), "heat_logpdf_v2_scale$(SCALE).json"), "w") do io
        JSON3.write(io, results)
    end

elseif MODE == "parity" || MODE == "stat"
    # ── Trace-based NUTS on v2.
    # trace0 is HAND-BUILT ([z_kappa=0, z_sigma=0, y-stacked] in declaration
    # order) instead of taken from `generate`: the :probprog opt pipeline
    # corrupts the stacked trace block written by a draw>1 in-loop site (keeps
    # only the first element of each iteration's draw; weight unaffected). See
    # /tmp/sbrantq-heatinv/phase3/repro_stacked_draw2.jl. The mcmc density path
    # READS the trace correctly, so a correct trace0 yields the true posterior.
    # mcmc gets the raw seed, exactly like v1's mcmc_logpdf run. ──
    num_samples = MODE == "stat" ? NUM_SAMPLES : 20
    seed = UInt64[1, 5]
    trace0_r = Reactant.to_rarray(reshape([0.0; 0.0; vec(Y_STACKED)], 1, :))

    function v2_program(rng_mcmc, trace0, L, x0col, sel, ss, imm)
        trace, diagnostics, _, _ = ProbProg.mcmc(
            rng_mcmc, trace0, heat_inverse_v2, L, x0col;
            selection=sel, algorithm=:NUTS, inverse_mass_matrix=imm, step_size=ss,
            max_tree_depth=5, num_warmup=0, num_samples=num_samples,
            adapt_step_size=false, adapt_mass_matrix=false)
        return trace, diagnostics
    end

    println("  compiling v2 trace-based NUTS ($num_samples samples)...")
    ss = ConcreteRNumber(STEP_SIZE)
    x0_r = Reactant.to_rarray(reshape(cfg_data.x0_coarse, D, 1))
    rng_mcmc = ReactantRNG(Reactant.to_rarray(copy(seed)))

    compiled_fn, tt = ProbProg.with_trace() do
        @compile sync = true optimize = :probprog v2_program(
            rng_mcmc, trace0_r, L_r, x0_r, SELECTION, ss, imm_r)
    end

    rng_mcmc2 = ReactantRNG(Reactant.to_rarray(copy(seed)))
    x0_r2 = Reactant.to_rarray(reshape(cfg_data.x0_coarse, D, 1))
    trace0_r2 = Reactant.to_rarray(reshape([0.0; 0.0; vec(Y_STACKED)], 1, :))
    trace_tensor, diagnostics = compiled_fn(
        rng_mcmc2, trace0_r2, L_r, x0_r2, SELECTION, ss, imm_r)
    trace_tensor = Array(trace_tensor)                # num_samples x 2
    diag_arr = Array(diagnostics)

    # Samples matrix columns follow the alphabetical selection order
    # (z_kappa, z_sigma), matching v1's position layout.
    println("  samples (z_kappa, z_sigma):")
    for j in 1:size(trace_tensor, 1)
        println("    [$j] $(trace_tensor[j, 1])  $(trace_tensor[j, 2])   accepted=$(diag_arr[j])")
    end
    out = Dict("seed" => Int.(seed), "step_size" => STEP_SIZE, "max_tree_depth" => 5,
        "num_samples" => num_samples,
        "samples" => [collect(trace_tensor[j, :]) for j in 1:size(trace_tensor, 1)],
        "accepted" => Int.(diag_arr))
    suffix = MODE == "stat" ? "_stat$(num_samples)" : ""
    open(joinpath(dirname(data_path), "heat_samples_v2$(suffix)_scale$(SCALE).json"), "w") do io
        JSON3.write(io, out)
    end
    println("  wrote v2 samples. n_accepted=$(sum(diag_arr))/$(length(diag_arr))")

elseif MODE == "ir"
    seed = UInt64[1, 5]
    ss = ConcreteRNumber(STEP_SIZE)
    x0_r = Reactant.to_rarray(reshape(cfg_data.x0_coarse, D, 1))
    trace0_r = Reactant.to_rarray(reshape([0.0; 0.0; vec(Y_STACKED)], 1, :))
    rng_mcmc = ReactantRNG(Reactant.to_rarray(copy(seed)))

    function v2_program(rng_mcmc, trace0, L, x0col, sel, ss, imm)
        trace, diagnostics, _, _ = ProbProg.mcmc(
            rng_mcmc, trace0, heat_inverse_v2, L, x0col;
            selection=sel, algorithm=:NUTS, inverse_mass_matrix=imm, step_size=ss,
            max_tree_depth=5, num_warmup=0, num_samples=20,
            adapt_step_size=false, adapt_mass_matrix=false)
        return trace, diagnostics
    end

    hlo = ProbProg.with_trace() do
        @code_hlo optimize = :probprog v2_program(
            rng_mcmc, trace0_r, L_r, x0_r, SELECTION, ss, imm_r)
    end
    hlo_str = sprint(show, hlo[1])
    hlo_path = joinpath(dirname(data_path), "heat_v2_hlo_scale$(SCALE).mlir")
    open(io -> print(io, hlo_str), hlo_path, "w")
    println("  saved HLO: $hlo_path  ($(count(==('\n'), hlo_str)) lines)")

else  # bench
    num_samples = 20
    seed = UInt64[1, 5]
    ss = ConcreteRNumber(STEP_SIZE)

    function v2_program(rng_mcmc, trace0, L, x0col, sel, ss, imm)
        trace, diagnostics, _, _ = ProbProg.mcmc(
            rng_mcmc, trace0, heat_inverse_v2, L, x0col;
            selection=sel, algorithm=:NUTS, inverse_mass_matrix=imm, step_size=ss,
            max_tree_depth=5, num_warmup=0, num_samples=num_samples,
            adapt_step_size=false, adapt_mass_matrix=false)
        return trace, diagnostics
    end

    mkinputs() = (
        ReactantRNG(Reactant.to_rarray(copy(seed))),
        Reactant.to_rarray(reshape([0.0; 0.0; vec(Y_STACKED)], 1, :)),
        L_r,
        Reactant.to_rarray(reshape(cfg_data.x0_coarse, D, 1)),
        SELECTION, ss, imm_r,
    )
    println("  compiling (flat path)...")
    compile_time = @elapsed begin
        compiled_fn = @compile sync = true optimize = :probprog v2_program(mkinputs()...)
    end
    println("  Compile: $(round(compile_time, digits=1))s")
    compiled_fn(mkinputs()...)   # warmup

    n_runs = 5
    times = Float64[]
    for i in 1:n_runs
        inp = mkinputs()
        t = @elapsed compiled_fn(inp...)
        push!(times, t)
    end
    sorted_t = sort(times)
    median_t = sorted_t[(length(sorted_t) + 1) ÷ 2]
    println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
    println("  All: $(round.(times .* 1000, digits=1))ms")
end
