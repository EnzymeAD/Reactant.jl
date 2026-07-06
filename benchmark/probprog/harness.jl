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

using Reactant, Random, Statistics, LinearAlgebra, JSON3, Printf
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

Reactant.Compiler.DEBUG_PROBPROG_DUMP_VALUE[] = false

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

function flatten_constraint(constraint::ProbProg.Constraint)
    constrained_addresses = ProbProg.extract_addresses(constraint)
    constraint_flat = Float64[]
    for addr in constrained_addresses
        append!(constraint_flat, vec(constraint[addr]))
    end
    constraint_tensor = Reactant.to_rarray(reshape(constraint_flat, 1, :))
    return constraint_tensor, constrained_addresses
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

function _svi_write_diagnostics(model_name, seed, elbo_traj, per_step, num_steps)
    out_dir = get(ENV, "PPLBENCH_OUTPUT_DIR", "outputs")
    diag_dir = joinpath(out_dir, "svi_diagnostics")
    mkpath(diag_dir)
    safe = replace(String(model_name), r"[^A-Za-z0-9_]" => "_")
    path = joinpath(diag_dir, "$(safe)_seed$(seed).json")
    open(path, "w") do io
        JSON3.write(io, Dict(
            "model" => String(model_name),
            "seed" => seed,
            "num_steps" => num_steps,
            "per_step_s" => per_step,
            "final_elbo" => isempty(elbo_traj) ? NaN : Float64(elbo_traj[end]),
            "elbo_trajectory" => Float64.(elbo_traj),
        ))
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
    selected_entries = ProbProg.filter_entries_by_selection(tt.entries, spec.selection)

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
        _svi_write_diagnostics(spec.model_name, this_seed, vec(elbo), per_step, num_steps)
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
    dump_passes = false
    profile_breakdown = false
    server_mode = false
    method = "nuts"
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

    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = disable_opt
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = disable_sicm
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_REGION[] = disable_region
    if dump_passes || DUMP_MLIR
        Reactant.Compiler.DEBUG_PROBPROG_DUMP_PASSES[] = true
        dump_passes_dir = joinpath(get(ENV, "PPLBENCH_OUTPUT_DIR", "outputs"), "sicm_dumps")
        mkpath(dump_passes_dir)
        Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dump_passes_dir
        println("SICM pass dumps enabled → $dump_passes_dir")
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
        opt_pipeline = post_sicm ? :probprog_post_sicm : no_opt ? :probprog_no_opt : :probprog
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

    # Build initial constraint
    constraint = build_constraint( data, get(data, "init_params", nothing))
    constraint_tensor, constrained_addresses = flatten_constraint(constraint)

    # RNG
    rng = ReactantRNG(Reactant.to_rarray(UInt64[UInt64(seed), UInt64(0)]))

    # Inverse mass matrix. A model's `setup` may return an explicit
    # `inverse_mass_matrix` (a length-`position_size` vector selects the
    # diagonal-mass path); otherwise default to a dense identity.
    step_size_rnum = ConcreteRNumber(step_size)
    inverse_mass_matrix = Reactant.to_rarray(
        hasproperty(spec, :inverse_mass_matrix) ? spec.inverse_mass_matrix :
        Matrix{Float64}(I, spec.position_size, spec.position_size),
    )

    # Compile
    opt_pipeline = post_sicm ? :probprog_post_sicm : no_opt ? :probprog_no_opt : :probprog
    print("Compiling... ")
    compile_time = @elapsed begin
        compiled_fn, tt = ProbProg.with_trace() do
            @compile sync = true optimize = opt_pipeline nuts_program(
                rng, spec.model_fn, spec.model_args, step_size_rnum,
                max_tree_depth, num_warmup, num_samples,
                constraint_tensor, constrained_addresses, spec.selection,
                adapt_step_size, adapt_mass_matrix, inverse_mass_matrix,
            )
        end
    end
    println("$(round(compile_time * 1000, digits=2)) ms")

    # MLIR dumps are handled by DEBUG_PROBPROG_DUMP_PASSES (intermediate IR at each stage)
    # Output goes to sicm_dumps/ directory when DUMP_MLIR or --dump-passes is set.

    # Run
    selected_entries = ProbProg.filter_entries_by_selection(tt.entries, spec.selection)

    function run_compiled(r, ct)
        trace_tensor, diagnostics = compiled_fn(
            r, spec.model_fn, spec.model_args, step_size_rnum,
            max_tree_depth, num_warmup, num_samples,
            ct, constrained_addresses, spec.selection,
            adapt_step_size, adapt_mass_matrix, inverse_mass_matrix,
        )
        return trace_tensor, diagnostics
    end

    print("Running... ")
    run_time = @elapsed begin
        if profile_breakdown || PROFILE_BREAKDOWN_ENV
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
        trace_tensor = Array(trace_tensor)
        diagnostics = Array(diagnostics)
        trace = ProbProg.unflatten_trace(trace_tensor, 0.0, selected_entries, nothing)
        output = extract_samples( trace)
    end
    println("$(round(run_time * 1000, digits=2)) ms")

    output["compile_time"] = compile_time
    output["run_time"] = run_time

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
                if profile_dir !== nothing
                    mkpath(profile_dir)
                    println("Profiling -> $profile_dir")
                    flush(stdout)
                    new_trace_tensor, new_diagnostics = Reactant.Profiler.with_profiler(profile_dir) do
                        run_compiled(new_rng, new_constraint_tensor)
                    end
                else
                    new_trace_tensor, new_diagnostics = run_compiled(new_rng, new_constraint_tensor)
                end
                new_trace_tensor = Array(new_trace_tensor)
                new_diagnostics = Array(new_diagnostics)
                new_trace = ProbProg.unflatten_trace(new_trace_tensor, 0.0, selected_entries, nothing)
                new_output = extract_samples( new_trace)
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
