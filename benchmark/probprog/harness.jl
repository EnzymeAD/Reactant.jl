#!/usr/bin/env julia
#
# Generic Impulse (Reactant.jl) benchmark harness.
#
# Usage:
#   julia --project=<benchmark_dir> harness.jl \
#       --model standard/logistic_regression \
#       --data input.json --output output.json \
#       --num-warmup 500 --num-samples 1000 --seed 42 \
#       --step-size 0.1 --max-tree-depth 10 \
#       --adapt-step-size --adapt-mass-matrix --server

ENV["JULIA_DEBUG"] = "Reactant,Reactant_jll"

using Reactant, Random, Statistics, LinearAlgebra, JSON3, Printf, Dates
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

Reactant.Compiler.DEBUG_PROBPROG_DUMP_VALUE[] = get(ENV, "PPLBENCH_DUMP_DA", "") == "1"

# When probprog dumps are active, the enzyme.dump FFI callback runs on the host
# during synchronous execution; a plain blocking Base.wait deadlocks against it.
# Yield-based wait (per CLAUDE.md Debug Utilities) lets the callback fire.
# Gated on the dump flag so normal timing runs keep the fast blocking wait.
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

const DUMP_MLIR = haskey(ENV, "PPLBENCH_DUMP_MLIR")
const PROFILE_XLA = haskey(ENV, "PPLBENCH_PROFILE")
const PROFILE_BREAKDOWN_ENV = haskey(ENV, "PPLBENCH_PROFILE_BREAKDOWN")

if DUMP_MLIR
    println("MLIR dump enabled (via DEBUG_PROBPROG_DUMP_PASSES)")
end

# =============================================================================
# Shared NUTS infrastructure
# =============================================================================

function nuts_program(
    rng, model, model_args, step_size, max_tree_depth,
    num_warmup, num_samples, constraint_tensor, constrained_addresses,
    selection, adapt_step_size, adapt_mass_matrix, inverse_mass_matrix,
)
    t, _, _ = ProbProg.generate(
        rng, constraint_tensor, model, model_args...; constrained_addresses
    )
    t, diagnostics, _, _ = ProbProg.mcmc(
        rng, t, model, model_args...;
        selection, algorithm=:NUTS, inverse_mass_matrix,
        step_size, max_tree_depth, num_warmup, num_samples,
        adapt_step_size, adapt_mass_matrix,
    )
    return t, diagnostics
end

# Frozen-adapted-state protocol (2026-08-25), ADDITIVE: identical to
# `nuts_program` except that it also returns the post-run `MCMCState`'s adapted
# step size and inverse mass matrix, which is what a warmup run has to hand back
# so both measured arms can later be started from the same adapted state with
# adaptation off. `ProbProg.mcmc` already returns that state as its 5th value;
# nothing in the sampler or the compiler changes. Used only by `--save-state`;
# the measured path still goes through `nuts_program` untouched.
function nuts_state_program(
    rng, model, model_args, step_size, max_tree_depth,
    num_warmup, num_samples, constraint_tensor, constrained_addresses,
    selection, adapt_step_size, adapt_mass_matrix, inverse_mass_matrix,
)
    t, _, _ = ProbProg.generate(
        rng, constraint_tensor, model, model_args...; constrained_addresses
    )
    t, diagnostics, _, _, state = ProbProg.mcmc(
        rng, t, model, model_args...;
        selection, algorithm=:NUTS, inverse_mass_matrix,
        step_size, max_tree_depth, num_warmup, num_samples,
        adapt_step_size, adapt_mass_matrix,
    )
    return t, diagnostics, state.step_size, state.inverse_mass_matrix
end

# Convert `extract_samples` output (per-address draws) into the flat
# `init_params` shape that every model's `build_constraint` already consumes,
# taking the LAST draw. Reusing that path means the injected position travels
# the same route pplbench's per-trial init_params already travel, in constrained
# space, so no unconstrained/constrained conversion is introduced here.
function last_draw_init_params(samples::AbstractDict)
    out = Dict{String,Any}()
    for (k, v) in samples
        isempty(v) && continue
        last = v[end]
        out[k] = last isa AbstractVector ? collect(Float64, last) : Float64[last]
    end
    return out
end

function flatten_constraint(constraint::ProbProg.Constraint)
    constrained_addresses = ProbProg.extract_addresses(constraint)
    constraint_flat = Float64[]
    for addr in constrained_addresses
        append!(constraint_flat, vec(constraint[addr]))
    end
    constraint_tensor = Reactant.to_rarray(reshape(constraint_flat, 1, :))
    return constraint_tensor, constrained_addresses
end

# Genuine multi-chain NUTS by concurrent thread dispatch of the compiled
# single-chain kernel (mirrors ProbProg.run_chain(:parallel), which cannot be
# used here because the generic model harness has no hand-written logdensity
# function). `split_chain_rngs` reproduces `jax.random.split(key, num_chains)`
# bitwise, so chain c matches NumPyro/Stan chain c; each chain owns private
# copies of every donated concrete input, so the shared compiled thunk never
# races on a donated buffer. `:sequential` runs the same chains serially.
# Returns (chain_major_output_dict, elapsed_seconds); chains are pooled for PLL
# and kept per-chain for split-Rhat / pooled-ESS downstream (arviz in ppl_helper).
function run_nuts_multichain(
    run_one, rng, num_chains::Int, chain_method::Symbol
)
    chain_keys = ProbProg.split_chain_rngs(rng, num_chains)
    results = Vector{Dict{String,Any}}(undef, num_chains)
    elapsed = @elapsed begin
        if chain_method == :parallel
            tasks = [Threads.@spawn run_one(chain_keys[c]) for c in 1:num_chains]
            for c in 1:num_chains
                results[c] = fetch(tasks[c])
            end
        else
            for c in 1:num_chains
                results[c] = run_one(chain_keys[c])
            end
        end
    end
    # Chain-major nesting: a Vector of per-chain values (each itself the
    # extract_samples value, a Vector or Vector-of-Vectors), so JSON3 serializes
    # nested arrays that the subprocess reads as (num_chains, draw, ...).
    output = Dict{String,Any}()
    for var in keys(results[1])
        output[var] = [results[c][var] for c in 1:num_chains]
    end
    return output, elapsed
end

# =============================================================================
# Shared SVI infrastructure
# =============================================================================

# `guide` / `optimizer` are compile-time Symbols; `num_steps`, `num_particles`,
# `init_scale` are compile-time literals baked into the VIConfigAttr. Only the
# rng, learning rate, constraint tensor and initial guide loc are runtime
# operands, so a single compilation serves every server trial.
function svi_program(
    rng, model, model_args, step_size, num_steps, num_particles, init_scale,
    guide, optimizer, constraint_tensor, constrained_addresses, selection,
    initial_position,
)
    trace0, _, _ = ProbProg.generate(
        rng, constraint_tensor, model, model_args...; constrained_addresses
    )
    _, elbo, _, state = ProbProg.svi(
        rng, trace0, model, model_args...;
        selection, guide, optimizer, step_size, num_steps, num_particles,
        init_scale, initial_position,
    )
    return elbo, state.loc, state.scale
end

# Draw `K` posterior samples from a fitted AutoNormal/AutoDelta guide, mirroring
# NumPyro's `Predictive` over the guide: eps ~ N(0,I), z = loc + scale*eps in the
# unconstrained latent space, then the per-site constraining bijector. `scale` is
# the guide's constrained (softplus) std for AutoNormal, zeros for AutoDelta.
# Host RNG (seeded per trial): predictive draws are a post-fit diagnostic, not
# part of the SICM-optimized SVI loop.
function svi_guide_draws(loc, scale, selected_entries, K::Int, seed::Integer)
    d = size(loc, 2)
    gen = Random.MersenneTwister(seed)
    eps = randn(gen, K, d)
    z = loc .+ scale .* eps
    constrained = copy(z)
    for entry in selected_entries
        cols = (entry.offset + 1):(entry.offset + entry.num_elements)
        if entry.support === :positive
            @views constrained[:, cols] .= exp.(z[:, cols])
        elseif entry.support === :real
            # identity bijector
        else
            error(
                "SVI guide draw: unsupported support :$(entry.support) for site " *
                ":$(entry.symbol) (only :real and :positive are handled)",
            )
        end
    end
    return constrained
end

# Post-fit column layout for the VI guide. The C++ VI lowering lays the fitted
# `loc`/`scale` (and per-step trace) out in model DECLARATION order among the
# selected sites (`collectSupportInfoForSelection` with
# `positionDeclarationOrder=true`, matching `build_selection_attr`'s
# all_addresses order) so the guide's per-site RNG draw order matches NumPyro.
# `ProbProg.filter_entries_by_selection` instead accumulates offsets in the
# alphabetized `select()` order used by the NUTS I/O boundary. Using the latter
# here would apply the per-site constraining bijector (`exp` for :positive) and
# the column->site mapping to the wrong columns whenever declaration order
# differs from alphabetical order (e.g. n_schools, whose HalfCauchy scales are
# declared first but sort last). Rebuild the entries by walking `tt.entries` in
# declaration order with declaration-order offsets so `svi_guide_draws` and
# `unflatten_trace` agree with the guide's actual column layout.
function svi_selected_entries(entries, selection)
    filtered = ProbProg.TraceEntry[]
    new_offset = 0
    for entry in entries
        entry_path = [entry.parent_path..., entry.symbol]
        for address in selection
            if entry_path == collect(address.path)
                push!(
                    filtered,
                    ProbProg.TraceEntry(
                        entry.symbol,
                        entry.shape,
                        entry.num_elements,
                        new_offset,
                        entry.parent_path,
                        entry.support,
                        entry.event_size,
                    ),
                )
                new_offset += entry.num_elements
                break
            end
        end
    end
    return filtered
end

# Per-site fitted guide params in UNCONSTRAINED latent space, keyed by site name.
# `loc`/`scale` are 1×d laid out in model DECLARATION order (matching the C++ VI
# lowering, see `svi_selected_entries`); slice each site by its declaration-order
# offset so the two harnesses compare by NAME, not column position (avoids the
# declaration-vs-alphabetical mismatch, e.g. n_schools sigma_* declared first).
function _svi_loc_scale_by_site(loc, scale, selected_entries)
    loc_by_site = Dict{String,Vector{Float64}}()
    scale_by_site = Dict{String,Vector{Float64}}()
    for entry in selected_entries
        cols = (entry.offset + 1):(entry.offset + entry.num_elements)
        loc_by_site[String(entry.symbol)] = Float64.(vec(loc[:, cols]))
        scale_by_site[String(entry.symbol)] = Float64.(vec(scale[:, cols]))
    end
    return loc_by_site, scale_by_site
end

function _svi_write_diagnostics(model_name, seed, elbo_traj, per_step, num_steps;
                                loc_by_site=nothing, scale_by_site=nothing)
    out_dir = get(ENV, "PPLBENCH_OUTPUT_DIR", "outputs")
    diag_dir = joinpath(out_dir, "svi_diagnostics")
    mkpath(diag_dir)
    safe = replace(String(model_name), r"[^A-Za-z0-9_]" => "_")
    path = joinpath(diag_dir, "$(safe)_seed$(seed).json")
    payload = Dict{String,Any}(
        "model" => String(model_name),
        "seed" => seed,
        "num_steps" => num_steps,
        "per_step_s" => per_step,
        "final_elbo" => isempty(elbo_traj) ? NaN : Float64(elbo_traj[end]),
        "elbo_trajectory" => Float64.(elbo_traj),
    )
    # Fitted guide loc/scale (validate correctness compares the FIT, not draws).
    if loc_by_site !== nothing
        payload["loc"] = loc_by_site
        payload["scale"] = scale_by_site
    end
    open(path, "w") do io
        JSON3.write(io, payload)
    end
    return path
end

# =============================================================================
# Profiling helpers
# =============================================================================

function _collect_op_entries(node, entries)
    children = get(node, :children, nothing)
    children === nothing && return
    for child in children
        name = string(get(child, :name, "?"))
        metrics = get(child, :metrics, nothing)
        metrics === nothing && continue
        time_ps = Float64(get(metrics, :rawTime, 0.0))
        occurrences = Int(get(metrics, :occurrences, 0))
        sub = get(child, :children, nothing)
        if sub !== nothing && !isempty(sub)
            _collect_op_entries(child, entries)
        else
            push!(entries, (name, time_ps, occurrences))
        end
    end
end

function print_op_profile_breakdown(xplane_file::String)
    raw = Reactant.Profiler.xspace_to_tools_data([xplane_file], "op_profile")[1]
    data = JSON3.read(raw)

    root = nothing
    for key in (:byCategory, :byCategoryExcludeIdle, :byProgram, :byProgramExcludeIdle)
        candidate = get(data, key, nothing)
        candidate === nothing && continue
        children = get(candidate, :children, nothing)
        if children !== nothing && !isempty(children)
            root = candidate
            println("  [using op_profile root: $key]")
            break
        end
    end

    if root === nothing
        println("  [no op_profile children found]")
        return
    end

    entries = Tuple{String, Float64, Int}[]
    _collect_op_entries(root, entries)

    if isempty(entries)
        println("  [collected 0 entries from op_profile]")
        return
    end

    sort!(entries; by=x -> -x[2])
    total_ps = sum(e[2] for e in entries)

    println("\n", "="^80)
    println("  OP PROFILE BREAKDOWN")
    println("="^80)
    println(rpad("Operation", 40), rpad("Time", 15), rpad("% Total", 10), "Occurrences")
    println("-"^80)
    for (name, time_ps, occ) in entries
        pct = total_ps > 0 ? time_ps / total_ps * 100 : 0.0
        time_str = if time_ps >= 1e12
            @sprintf("%.2f s", time_ps / 1e12)
        elseif time_ps >= 1e9
            @sprintf("%.2f ms", time_ps / 1e9)
        elseif time_ps >= 1e6
            @sprintf("%.2f μs", time_ps / 1e6)
        else
            @sprintf("%.0f ps", time_ps)
        end
        println(rpad(name, 40), rpad(time_str, 15), rpad(@sprintf("%.1f%%", pct), 10), occ)
    end
    println("-"^80)
    total_str = if total_ps >= 1e12
        @sprintf("%.2f s", total_ps / 1e12)
    elseif total_ps >= 1e9
        @sprintf("%.2f ms", total_ps / 1e9)
    else
        @sprintf("%.2f μs", total_ps / 1e6)
    end
    println(rpad("TOTAL", 40), total_str)
    println("="^80)
end

# =============================================================================
# Parse --model early and include at top level (avoids world-age issues)
# =============================================================================

function _parse_model_arg()
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--model" && i < length(ARGS)
            return ARGS[i+1]
        end
        i += 1
    end
    error("Missing --model argument")
end

# Canonical matched-init value for a REAL-support site whose value feeds a
# reference log-prob that is non-differentiable at the origin. MUST stay
# bit-identical to `MATCHED_INIT_REAL` in
# pplbench/pplbench/ppls/numpyro/base_numpyro_impl.py, which carries the full
# rationale: numpyro's Bernoulli(logits=x).log_prob reports a gradient that is
# wrong by exactly 0.5 per observation at x == 0, and a logit that is a linear
# map of zero-initialised real sites is exactly 0 there. Both arms must pin the
# same offset or --validate stops comparing the same starting point.
const MATCHED_INIT_REAL = 1e-6

const MODEL_PATH = _parse_model_arg()
include(joinpath(@__DIR__, MODEL_PATH * ".jl"))

# =============================================================================
# SVI entry point
# =============================================================================

function run_svi(
    data, spec, output_path, seed, step_size, num_steps, num_particles,
    init_scale, guide, optimizer, num_draws, opt_pipeline, server_mode,
)
    println("Impulse SVI: $(spec.model_name)")
    println("  guide=$guide, optimizer=$optimizer, step_size=$step_size, num_steps=$num_steps")
    println("  num_particles=$num_particles, init_scale=$init_scale, num_draws=$num_draws")
    println("  pipeline=$opt_pipeline")

    pos = spec.position_size
    # Deterministic unconstrained guide loc at 0 (constrained: 1 for positive
    # sites, 0 for real), matching the D3 reference. Trials vary only by rng seed.
    initial_position = Reactant.to_rarray(zeros(1, pos))
    step_size_rnum = ConcreteRNumber(step_size)

    constraint = build_constraint(data, get(data, "init_params", nothing))
    constraint_tensor, constrained_addresses = flatten_constraint(constraint)

    rng = ReactantRNG(Reactant.to_rarray(UInt64[UInt64(seed), UInt64(0)]))

    print("Compiling... ")
    compile_time = @elapsed begin
        compiled_fn, tt = ProbProg.with_trace() do
            @compile sync = true optimize = opt_pipeline svi_program(
                rng, spec.model_fn, spec.model_args, step_size_rnum, num_steps,
                num_particles, init_scale, guide, optimizer, constraint_tensor,
                constrained_addresses, spec.selection, initial_position,
            )
        end
    end
    println("$(round(compile_time * 1000, digits=2)) ms")
    selected_entries = svi_selected_entries(tt.entries, spec.selection)

    function fit_and_sample(r, ct, ip, this_seed)
        run_time = @elapsed begin
            elbo_t, loc_t, scale_t = compiled_fn(
                r, spec.model_fn, spec.model_args, step_size_rnum, num_steps,
                num_particles, init_scale, guide, optimizer, ct,
                constrained_addresses, spec.selection, ip,
            )
            elbo = Array(elbo_t)
            loc = Array(loc_t)
            scale = Array(scale_t)
        end
        constrained = svi_guide_draws(loc, scale, selected_entries, num_draws, this_seed)
        trace = ProbProg.unflatten_trace(constrained, 0.0, selected_entries, nothing)
        out = extract_samples(trace)
        per_step = run_time / num_steps
        loc_by_site, scale_by_site = _svi_loc_scale_by_site(loc, scale, selected_entries)
        _svi_write_diagnostics(
            spec.model_name, this_seed, vec(elbo), per_step, num_steps;
            loc_by_site=loc_by_site, scale_by_site=scale_by_site,
        )
        return out, run_time, vec(elbo)
    end

    print("Fitting... ")
    output, run_time, elbo = fit_and_sample(rng, constraint_tensor, initial_position, seed)
    final_elbo = isempty(elbo) ? NaN : elbo[end]
    println("$(round(run_time * 1000, digits=2)) ms  final_elbo=$final_elbo")

    output["compile_time"] = compile_time
    output["run_time"] = run_time
    open(output_path, "w") do f
        JSON3.write(f, output)
    end
    println("Output written to $output_path")
    println("  compile_time: $(round(compile_time, digits=3))s")
    println("  run_time: $(round(run_time, digits=3))s")

    if server_mode
        flush(stdout)
        println("###READY###")
        flush(stdout)

        while !eof(stdin)
            line = readline(stdin)
            stripped = strip(line)
            isempty(stripped) && continue
            stripped == "EXIT" && break

            req = JSON3.read(stripped)
            new_seed = Int(req["seed"])
            new_output_path = String(req["output"])
            println("Server trial: seed=$new_seed, output=$new_output_path")
            flush(stdout)

            new_rng = ReactantRNG(Reactant.to_rarray(UInt64[UInt64(new_seed), UInt64(0)]))
            new_output, new_run_time, _ = fit_and_sample(
                new_rng, constraint_tensor, initial_position, new_seed
            )
            new_output["compile_time"] = 0.0
            new_output["run_time"] = new_run_time
            open(new_output_path, "w") do f
                JSON3.write(f, new_output)
            end

            println("###DONE### $(round(new_run_time * 1000, digits=2)) ms")
            flush(stdout)
        end
    end
end

# =============================================================================
# Main
# =============================================================================

function main()
    data_path = ""
    output_path = ""
    num_warmup = 500
    num_samples = 1000
    seed = 42
    step_size = 0.1
    max_tree_depth = 10
    adapt_step_size = false
    adapt_mass_matrix = false
    disable_opt = false
    disable_sicm = false
    disable_region = false
    no_opt = false
    post_sicm = false
    disable_licm = false
    # Frozen-adapted-state protocol (additive; both default off = old behaviour).
    save_state_path = ""
    load_state_path = ""
    repeat_runs = 1
    dump_passes = false
    profile_breakdown = false
    server_mode = false
    method = "nuts"
    num_chains = 1
    chain_method = "parallel"
    num_steps = 300
    num_particles = 1
    init_scale = 0.1
    guide = "AutoNormal"
    optimizer = "Adam"
    device = get(ENV, "CUDA_VISIBLE_DEVICES", "") == "" ? "cpu" : "cuda"

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--model"
            i += 1  # skip (already parsed)
        elseif arg == "--data"
            i += 1; data_path = ARGS[i]
        elseif arg == "--output"
            i += 1; output_path = ARGS[i]
        elseif arg == "--method"
            i += 1; method = ARGS[i]
        elseif arg == "--num-chains"
            i += 1; num_chains = parse(Int, ARGS[i])
        elseif arg == "--chain-method"
            i += 1; chain_method = ARGS[i]
        elseif arg == "--num-steps"
            i += 1; num_steps = parse(Int, ARGS[i])
        elseif arg == "--num-particles"
            i += 1; num_particles = parse(Int, ARGS[i])
        elseif arg == "--init-scale"
            i += 1; init_scale = parse(Float64, ARGS[i])
        elseif arg == "--guide"
            i += 1; guide = ARGS[i]
        elseif arg == "--optimizer"
            i += 1; optimizer = ARGS[i]
        elseif arg == "--num-warmup"
            i += 1; num_warmup = parse(Int, ARGS[i])
        elseif arg == "--num-samples"
            i += 1; num_samples = parse(Int, ARGS[i])
        elseif arg == "--seed"
            i += 1; seed = parse(Int, ARGS[i])
        elseif arg == "--step-size"
            i += 1; step_size = parse(Float64, ARGS[i])
        elseif arg == "--max-tree-depth"
            i += 1; max_tree_depth = parse(Int, ARGS[i])
        elseif arg == "--adapt-step-size"
            adapt_step_size = true
        elseif arg == "--adapt-mass-matrix"
            adapt_mass_matrix = true
        elseif arg == "--disable-opt"
            disable_opt = true
        elseif arg == "--disable-sicm"
            disable_sicm = true
        elseif arg == "--disable-region"
            disable_region = true
        elseif arg == "--no-opt"
            no_opt = true
        elseif arg == "--repeat"
            i += 1; repeat_runs = parse(Int, ARGS[i])
        elseif arg == "--save-state"
            i += 1; save_state_path = ARGS[i]
        elseif arg == "--load-state"
            i += 1; load_state_path = ARGS[i]
        elseif arg == "--disable-licm"
            # Strip the `*_licm` pattern subset from the standard passes
            # (CompileOptions.disable_licm_optimization_passes). Combined with
            # --no-opt this is the "no pre-lowering optimization" ladder arm.
            disable_licm = true
        elseif arg == "--post-sicm"
            post_sicm = true
        elseif arg == "--dump-passes"
            dump_passes = true
        elseif arg == "--profile-breakdown"
            profile_breakdown = true
        elseif arg == "--server"
            server_mode = true
        else
            @warn "Unknown argument: $arg"
        end
        i += 1
    end

    model_path = MODEL_PATH
    @assert !isempty(data_path) "Missing --data argument"
    @assert !isempty(output_path) "Missing --output argument"

    # The ablation "No opt" arm (`--no-opt`) is the fully-unoptimized baseline:
    # it selects the flat expand-impulse lowering, i.e. exactly the same runtime
    # mechanism as `--disable-opt`. (The legacy `:probprog_no_opt` named pipeline
    # it used to select no longer exists in the unified `:probprog` pipeline.)
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = disable_opt || no_opt
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = disable_sicm
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_REGION[] = disable_region
    if post_sicm
        error(
            "--post-sicm (post-lowering SICM) is not a supported arm in the " *
            "unified :probprog pipeline; it selected the removed " *
            ":probprog_post_sicm pipeline. Use --disable-sicm (No SICM) or " *
            "--no-opt (No opt) instead.",
        )
    end
    if dump_passes || DUMP_MLIR
        Reactant.Compiler.DEBUG_PROBPROG_DUMP_PASSES[] = true
        dump_passes_dir = joinpath(get(ENV, "PPLBENCH_OUTPUT_DIR", "outputs"), "sicm_dumps")
        mkpath(dump_passes_dir)
        Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dump_passes_dir
        println("SICM pass dumps enabled → $dump_passes_dir")
    end
    if get(ENV, "PPLBENCH_DUMP_MLIR_ALWAYS", "") == "1"
        # Dump every compiled module (pre/post optimization), independent of
        # the SICM-pass dumper, which only fires on the region pipeline. Used
        # for IR evidence on flat-path (--no-opt) arms.
        always_dir = joinpath(get(ENV, "PPLBENCH_OUTPUT_DIR", "outputs"), "mlir_dumps")
        mkpath(always_dir)
        Reactant.MLIR.IR.DUMP_MLIR_DIR[] = always_dir
        Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
        println("MLIR dumps (always) → $always_dir")
    end

    if device == "cuda"
        Reactant.set_default_backend("gpu")
    else
        Reactant.set_default_backend("cpu")
    end

    # Load data
    data = JSON3.read(read(data_path, String))

    # Model-specific setup
    spec = setup(data)

    if method == "svi"
        opt_pipeline = :probprog
        run_svi(
            data, spec, output_path, seed, step_size, num_steps, num_particles,
            init_scale, Symbol(guide), Symbol(optimizer), num_samples,
            opt_pipeline, server_mode,
        )
        return
    end

    println("Impulse Benchmark: $(spec.model_name)")
    println("  device=$device, model=$model_path")
    println("  num_warmup=$num_warmup, num_samples=$num_samples, seed=$seed")
    println("  step_size=$step_size, max_tree_depth=$max_tree_depth")
    println("  adapt_step_size=$adapt_step_size, adapt_mass_matrix=$adapt_mass_matrix")

    # Build initial constraint. Under `--load-state` the initial position comes
    # from the saved post-warmup draw instead of the caller's init_params, and
    # the adapted step size / inverse mass replace the CLI values with
    # adaptation forced off, so both arms start from one shared adapted state.
    init_override = get(data, "init_params", nothing)
    loaded_inverse_mass = nothing
    if !isempty(load_state_path)
        st = JSON3.read(read(load_state_path, String))
        init_override = st["post_warmup_init_params"]
        step_size = Float64(st["adapted_step_size"])
        imm = st["adapted_inverse_mass"]
        rank_saved = haskey(st, "inverse_mass_rank") ? Int(st["inverse_mass_rank"]) :
                     (length(imm) > 0 && imm[1] isa AbstractVector ? 2 : 1)
        loaded_inverse_mass = if rank_saved == 2
            n_r = length(imm)
            imm[1] isa AbstractVector ||
                error("--load-state: inverse_mass_rank=2 but the saved matrix is " *
                      "flat (length $(length(imm))). Re-save with a harness that " *
                      "writes nested rows; a flat array would be silently read " *
                      "as a diagonal-mass vector.")
            [Float64(imm[r][c]) for r in 1:n_r, c in 1:length(imm[1])]
        else
            collect(Float64, imm)
        end
        # Hard shape gate: a mis-shaped inverse mass corrupts every measurement
        # downstream and does so silently, so fail loudly here instead.
        let P = spec.position_size, sz = size(loaded_inverse_mass)
            (rank_saved == 2 ? sz == (P, P) : sz == (P,)) || error(
                "--load-state: inverse mass $(sz) does not match position_size $P")
        end
        adapt_step_size = false
        adapt_mass_matrix = false
        num_warmup = 0
        println("Loaded adapted state from $load_state_path")
        println("  adapted step_size = $step_size, inverse mass ",
                size(loaded_inverse_mass), ", adaptation forced OFF")
    end
    constraint = build_constraint(data, init_override)
    constraint_tensor, constrained_addresses = flatten_constraint(constraint)

    # RNG
    rng = ReactantRNG(Reactant.to_rarray(UInt64[UInt64(seed), UInt64(0)]))

    # Inverse mass matrix. A model's `setup` may return an explicit
    # `inverse_mass_matrix` (a length-`position_size` vector selects the
    # diagonal-mass path); otherwise default to a dense identity.
    step_size_rnum = ConcreteRNumber(step_size)
    inverse_mass_matrix = Reactant.to_rarray(
        loaded_inverse_mass !== nothing ? loaded_inverse_mass :
        hasproperty(spec, :inverse_mass_matrix) ? spec.inverse_mass_matrix :
        Matrix{Float64}(I, spec.position_size, spec.position_size),
    )

    # `--save-state`: run the warmup once with adaptation ON and write the
    # adapted state (step size, inverse mass, post-warmup draw) to JSON, then
    # exit. Uses `nuts_state_program`, which differs from the measured path only
    # in also returning the state; the measured path below is untouched.
    if !isempty(save_state_path)
        println("Saving adapted state after $num_warmup warmup iterations ...")
        compiled_state_fn, tt_s = ProbProg.with_trace() do
            @compile sync = true optimize = :probprog nuts_state_program(
                rng, spec.model_fn, spec.model_args, step_size_rnum,
                max_tree_depth, num_warmup, 1,
                constraint_tensor, constrained_addresses, spec.selection,
                true, true, inverse_mass_matrix,
            )
        end
        trace_tensor_s, _, ss_out, imm_out = compiled_state_fn(
            rng, spec.model_fn, spec.model_args, step_size_rnum,
            max_tree_depth, num_warmup, 1,
            constraint_tensor, constrained_addresses, spec.selection,
            true, true, inverse_mass_matrix,
        )
        sel_s = ProbProg.filter_entries_by_selection(tt_s.entries, spec.selection)
        trace_s = ProbProg.unflatten_trace(
            Array(trace_tensor_s), 0.0, sel_s, nothing
        )
        samples_s = extract_samples(trace_s)
        adapted_ss = Float64(Array(ss_out)[])
        adapted_imm = Array(imm_out)
        state_out = Dict{String,Any}(
            "protocol" => "frozen-adapted-state: warm up once with adaptation ON, " *
                          "then both arms measure adaptation-OFF from this shared state",
            "model" => MODEL_PATH,
            "created" => string(Dates.now()),
            "num_warmup" => num_warmup,
            "warmup_seed" => seed,
            "initial_step_size" => step_size,
            "max_tree_depth" => max_tree_depth,
            "adapted_step_size" => adapted_ss,
            # Serialize a 2D inverse mass as nested ROWS. JSON3 flattens a Julia
            # Matrix to a column-major vector, which a reader would then take for
            # a diagonal-mass vector: a silent, measurement-corrupting shape bug.
            # Nested rows round-trip the rank explicitly.
            "adapted_inverse_mass" => (
                ndims(adapted_imm) == 2 ?
                [collect(Float64, adapted_imm[r, :]) for r in 1:size(adapted_imm, 1)] :
                collect(Float64, adapted_imm)
            ),
            "inverse_mass_rank" => ndims(adapted_imm),
            "inverse_mass_shape" => collect(size(adapted_imm)),
            "position_size" => spec.position_size,
            "post_warmup_init_params" => last_draw_init_params(samples_s),
        )
        open(save_state_path, "w") do io
            JSON3.write(io, state_out)
        end
        println("  adapted step_size = $adapted_ss (from $step_size), " *
                "inverse mass $(size(adapted_imm))")
        println("  wrote $save_state_path")
        return
    end

    # Compile
    opt_pipeline = :probprog
    # NOTE: when `compile_options` is given, @compile ignores its other
    # kwargs (incl. `sync`), so sync must live inside CompileOptions.
    licm_off_opts = Reactant.CompileOptions(;
        optimization_passes=opt_pipeline,
        disable_licm_optimization_passes=true,
        sync=true,
    )
    print("Compiling... ")
    compile_time = @elapsed begin
        compiled_fn, tt = ProbProg.with_trace() do
            if disable_licm
                @compile compile_options = licm_off_opts nuts_program(
                    rng, spec.model_fn, spec.model_args, step_size_rnum,
                    max_tree_depth, num_warmup, num_samples,
                    constraint_tensor, constrained_addresses, spec.selection,
                    adapt_step_size, adapt_mass_matrix, inverse_mass_matrix,
                )
            else
                @compile sync = true optimize = opt_pipeline nuts_program(
                    rng, spec.model_fn, spec.model_args, step_size_rnum,
                    max_tree_depth, num_warmup, num_samples,
                    constraint_tensor, constrained_addresses, spec.selection,
                    adapt_step_size, adapt_mass_matrix, inverse_mass_matrix,
                )
            end
        end
    end
    println("$(round(compile_time * 1000, digits=2)) ms")

    # MLIR dumps are handled by DEBUG_PROBPROG_DUMP_PASSES (intermediate IR at each stage)
    # Output goes to sicm_dumps/ directory when DUMP_MLIR or --dump-passes is set.

    # Run
    selected_entries = ProbProg.filter_entries_by_selection(tt.entries, spec.selection)

    function run_compiled(r, ct)
        Reactant.Compiler.DEBUG_PROBPROG_DUMP_VALUE[] && ProbProg.clear_dump_buffer!()
        trace_tensor, diagnostics = compiled_fn(
            r, spec.model_fn, spec.model_args, step_size_rnum,
            max_tree_depth, num_warmup, num_samples,
            ct, constrained_addresses, spec.selection,
            adapt_step_size, adapt_mass_matrix, inverse_mass_matrix,
        )
        Reactant.Compiler.DEBUG_PROBPROG_DUMP_VALUE[] && ProbProg.show_dumps()
        return trace_tensor, diagnostics
    end

    # One chain's samples from private copies of every donated concrete input,
    # so the shared compiled thunk is safe to run concurrently across chains.
    function run_one_chain(key_c, ct)
        tt, _ = compiled_fn(
            key_c, spec.model_fn, spec.model_args, copy(step_size_rnum),
            max_tree_depth, num_warmup, num_samples,
            copy(ct), constrained_addresses, spec.selection,
            adapt_step_size, adapt_mass_matrix, copy(inverse_mass_matrix),
        )
        return extract_samples(
            ProbProg.unflatten_trace(Array(tt), 0.0, selected_entries, nothing)
        )
    end

    chain_method_sym = Symbol(chain_method)
    multichain = num_chains > 1
    if multichain
        println("  num_chains=$num_chains, chain_method=$chain_method_sym")
    end

    print("Running... ")
    run_time = @elapsed begin
        if multichain
            output, _ = run_nuts_multichain(
                k -> run_one_chain(k, constraint_tensor), rng, num_chains,
                chain_method_sym,
            )
        elseif profile_breakdown || PROFILE_BREAKDOWN_ENV
            println("profiling with op_profile breakdown...")
            result = Reactant.Profiler.profile_with_xprof(
                compiled_fn, rng, spec.model_fn, spec.model_args, step_size_rnum,
                max_tree_depth, num_warmup, num_samples,
                constraint_tensor, constrained_addresses, spec.selection,
                adapt_step_size, adapt_mass_matrix, inverse_mass_matrix,
            )
            trace_tensor, diagnostics = result.val
            print_op_profile_breakdown(result.xplane_file)
        elseif PROFILE_XLA
            trace_dir = joinpath(get(ENV, "PPLBENCH_OUTPUT_DIR", "outputs"), "traces", "impulse")
            mkpath(trace_dir)
            println("XLA profiling → $trace_dir")
            trace_tensor, diagnostics = Reactant.Profiler.with_profiler(trace_dir) do
                run_compiled(rng, constraint_tensor)
            end
        else
            trace_tensor, diagnostics = run_compiled(rng, constraint_tensor)
        end
        if !multichain
            trace_tensor = Array(trace_tensor)
            diagnostics = Array(diagnostics)
            trace = ProbProg.unflatten_trace(trace_tensor, 0.0, selected_entries, nothing)
            output = extract_samples( trace)
        end
    end
    println("$(round(run_time * 1000, digits=2)) ms")

    output["compile_time"] = compile_time
    output["run_time"] = run_time

    # `--repeat N`: additional timed executions of the SAME compiled thunk in
    # this process. The first execution above pays first-call warm-up, which on
    # short cells dominates and dilutes an arm ratio; pplbench avoids that by
    # discarding trial 0 in its server, and the pre-lowering harnesses by taking
    # medians of repeated runs. `repeat_times_s` therefore excludes the first
    # call and is the steady-state series to summarize.
    if repeat_runs > 1 && !multichain
        rts = Float64[]
        for _ in 2:repeat_runs
            t = @elapsed begin
                tt_r, _ = run_compiled(rng, constraint_tensor)
                Array(tt_r)
            end
            push!(rts, t)
        end
        output["repeat_times_s"] = rts
        println("  repeats (steady state, s): ",
                join([string(round(t; digits=5)) for t in rts], ", "))
    end

    open(output_path, "w") do f
        JSON3.write(f, output)
    end

    println("Output written to $output_path")
    println("  compile_time: $(round(compile_time, digits=3))s")
    println("  run_time: $(round(run_time, digits=3))s")

    # Server mode
    if server_mode
        flush(stdout)
        println("###READY###")
        flush(stdout)

        while !eof(stdin)
            line = readline(stdin)
            stripped = strip(line)
            isempty(stripped) && continue
            stripped == "EXIT" && break

            req = JSON3.read(stripped)
            new_seed = Int(req["seed"])
            new_output_path = String(req["output"])
            println("Server trial: seed=$new_seed, output=$new_output_path")
            flush(stdout)

            new_rng = ReactantRNG(Reactant.to_rarray(UInt64[UInt64(new_seed), UInt64(0)]))

            new_constraint_tensor = constraint_tensor
            if haskey(req, "init_params") && req["init_params"] !== nothing
                new_constraint = build_constraint( data, req["init_params"])
                new_constraint_tensor, _ = flatten_constraint(new_constraint)
            end

            profile_dir = haskey(req, "profile") && req["profile"] !== nothing ?
                String(req["profile"]) : nothing

            new_run_time = @elapsed begin
                if multichain
                    new_output, _ = run_nuts_multichain(
                        k -> run_one_chain(k, new_constraint_tensor), new_rng,
                        num_chains, chain_method_sym,
                    )
                elseif profile_dir !== nothing
                    mkpath(profile_dir)
                    println("Profiling -> $profile_dir")
                    flush(stdout)
                    new_trace_tensor, new_diagnostics = Reactant.Profiler.with_profiler(profile_dir) do
                        run_compiled(new_rng, new_constraint_tensor)
                    end
                    new_trace_tensor = Array(new_trace_tensor)
                    new_diagnostics = Array(new_diagnostics)
                    new_trace = ProbProg.unflatten_trace(new_trace_tensor, 0.0, selected_entries, nothing)
                    new_output = extract_samples( new_trace)
                else
                    new_trace_tensor, new_diagnostics = run_compiled(new_rng, new_constraint_tensor)
                    new_trace_tensor = Array(new_trace_tensor)
                    new_diagnostics = Array(new_diagnostics)
                    new_trace = ProbProg.unflatten_trace(new_trace_tensor, 0.0, selected_entries, nothing)
                    new_output = extract_samples( new_trace)
                end
            end
            new_output["compile_time"] = 0.0
            new_output["run_time"] = new_run_time

            open(new_output_path, "w") do f
                JSON3.write(f, new_output)
            end

            println("###DONE### $(round(new_run_time * 1000, digits=2)) ms")
            flush(stdout)
        end
    end
end

main()
