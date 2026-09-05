#!/usr/bin/env julia
#
# Generic Turing.jl benchmark harness.
#
# Usage:
#   julia --project=turing_env turing_harness.jl \
#       --model standard/logistic_regression \
#       --data input.json --output output.json \
#       --num-warmup 500 --num-samples 1000 --seed 42 \
#       --step-size 0.1 --max-tree-depth 10 \
#       --adapt-step-size --adapt-mass-matrix --server

using Turing, Random, LinearAlgebra, JSON3
using Turing: NUTS, sample, DynamicPPL
using ReverseDiff  # backend for AutoReverseDiff (--adtype reversediff)
using AdvancedHMC: DenseEuclideanMetric

# Turing.Variational re-exports the AdvancedVI machinery. AdvancedVI is only a
# transitive dependency (not in turing_env/Project.toml), so it is reached
# through the Turing.Variational namespace rather than a direct `import`.
const TV = Turing.Variational
const AVI = Turing.Variational.AdvancedVI

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
# pplbench/pplbench/ppls/base_ppl_impl.py, which carries the full rationale:
# numpyro's Bernoulli(logits=x).log_prob reports a gradient that is wrong by
# exactly 0.5 per observation at x == 0, and a logit that is a linear map of
# zero-initialised real sites is exactly 0 there. Turing's own Bernoulli is fine
# at 0; this arm takes the offset purely so the four-way matched-init protocol
# keeps comparing a shared starting point.
const MATCHED_INIT_REAL = 1e-6

const MODEL_PATH = _parse_model_arg()
include(joinpath(@__DIR__, "turing", MODEL_PATH * ".jl"))

# =============================================================================
# Run NUTS via Turing
# =============================================================================

"""Resolve an `--adtype` CLI name to an ADTypes backend.

`forwarddiff` is Turing's own default (`Turing.DEFAULT_ADTYPE`), best for
low-dimensional models; `reversediff` is compiled-tape ReverseDiff, the backend
Turing's performance documentation recommends for high-dimensional models.
`compile=true` is safe for every benchmark model here: all are fixed-structure
with no parameter-value-dependent control flow.
"""
function resolve_adtype(adtype_name)
    if adtype_name == "forwarddiff"
        return AutoForwardDiff()
    elseif adtype_name == "reversediff"
        return AutoReverseDiff(compile=true)
    else
        error("Unknown --adtype: $adtype_name (expected forwarddiff|reversediff)")
    end
end

function run_turing_nuts(
    model, seed, num_warmup, num_samples, step_size, max_tree_depth,
    adapt_step_size, adapt_mass_matrix, init_params, adtype_name,
)
    rng = Xoshiro(seed)
    total_samples = num_warmup + num_samples
    adtype = resolve_adtype(adtype_name)

    if adapt_step_size || adapt_mass_matrix
        n_adapts = num_warmup
        sampler = NUTS(n_adapts, 0.8; max_depth=max_tree_depth, init_ϵ=step_size, metricT=DenseEuclideanMetric, adtype=adtype)
    else
        sampler = NUTS(0, 0.8; max_depth=max_tree_depth, init_ϵ=step_size, metricT=DenseEuclideanMetric, adtype=adtype)
    end

    sample_kwargs = Dict{Symbol,Any}(
        :discard_initial => 0,
        :progress => false,
    )

    if init_params !== nothing
        sample_kwargs[:initial_params] = DynamicPPL.InitFromParams(init_params)
    end

    chain = sample(rng, model, sampler, total_samples; sample_kwargs...)

    return chain
end

# =============================================================================
# Run ADVI (mean-field) via Turing.Variational / AdvancedVI
# =============================================================================
#
# Mirrors the Impulse / NumPyro SVI protocol as closely as this Turing version
# allows:
#   * mean-field Gaussian guide (== NumPyro AutoNormal), initialized at
#     unconstrained loc = 0, diagonal scale = init_scale;
#   * reparameterization-gradient ELBO with `n_particles` MC samples per step
#     (== num_particles);
#   * Optimisers.Adam(lr) with lr = step_size, `num_steps` iterations;
#   * NoAveraging (return the last iterate, matching Adam's final params rather
#     than a Polyak average);
#   * ClipScale operator (AdvancedVI's recommended safeguard against a singular
#     scale for MvLocationScale families; not an optimizer change).
# AdvancedVI 0.6's ADVI == KLMinRepGradDescent. The prox-descent variant that
# Turing's `vi` uses by default only accepts DoG/DoWG/Descent, so Adam requires
# KLMinRepGradDescent explicitly.

function svi_draw_and_extract(model, q, vi_linked, rng, K)
    # q is a Bijectors.TransformedDistribution wrapping the unconstrained
    # MvLocationScale (q.dist) in the model's canonical linked ordering. Draw in
    # unconstrained space, then unflatten + invlink each draw to constrained
    # space and read named sites, so the output layout matches the MCMC
    # extractor (one JSON key per model site).
    qd = q.dist
    zs = rand(rng, qd, K)  # d x K, unconstrained
    scalar_acc = Dict{String,Vector{Float64}}()
    vector_acc = Dict{String,Vector{Vector{Float64}}}()
    for j in 1:K
        z = zs[:, j]
        vc = DynamicPPL.invlink(DynamicPPL.unflatten(vi_linked, z), model)
        for (vn, val) in DynamicPPL.values_as(vc, Dict)
            name = string(vn)
            if val isa AbstractArray
                push!(get!(vector_acc, name, Vector{Vector{Float64}}()),
                      Float64.(vec(val)))
            else
                push!(get!(scalar_acc, name, Float64[]), Float64(val))
            end
        end
    end
    out = Dict{String,Any}()
    for (nm, v) in scalar_acc
        out[nm] = v
    end
    for (nm, v) in vector_acc
        out[nm] = v  # draw-major: K entries, each a per-draw vector
    end
    return out
end

function run_turing_advi(
    model, seed, num_steps, lr, n_particles, init_scale, K, guide, optimizer,
)
    @assert guide == "AutoNormal" "Turing ADVI arm only supports the AutoNormal (mean-field Gaussian) guide; got $guide"
    @assert optimizer == "Adam" "Turing ADVI arm only supports the Adam optimizer; got $optimizer"

    rng = Xoshiro(seed)
    # Forward-mode AD, deliberately: the SVI rows run only the three continuous
    # models (d <= ~57), where forward mode is the appropriate default. The NUTS
    # path selects its backend per model via --adtype (ReverseDiff is in the env
    # since 2026-08-24); switching the VI backend is a separate decision.
    adtype = Turing.DEFAULT_ADTYPE  # AutoForwardDiff()

    vi_linked = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    d = length(vi_linked[:])

    q0 = TV.q_meanfield_gaussian(
        model; location=zeros(d), scale=Diagonal(fill(init_scale, d))
    )
    alg = TV.KLMinRepGradDescent(
        adtype;
        optimizer = AVI.Optimisers.Adam(lr),
        n_samples = n_particles,
        averager = AVI.NoAveraging(),
        operator = AVI.ClipScale(),
    )

    local q, info
    fit_time = @elapsed begin
        q, info, _ = TV.vi(
            rng, model, q0, num_steps;
            algorithm=alg, adtype=adtype, show_progress=false,
        )
    end
    elbo_traj = Float64[Float64(nt.elbo) for nt in info]

    out = svi_draw_and_extract(model, q, vi_linked, rng, K)
    return out, elbo_traj, fit_time
end

function svi_write_diagnostics(model_path, seed, elbo_traj, fit_time, num_steps)
    diag_dir = joinpath(
        get(ENV, "PPLBENCH_OUTPUT_DIR", "outputs"), "svi_diagnostics"
    )
    try
        mkpath(diag_dir)
        safe = replace(String(model_path), r"[^A-Za-z0-9_]" => "_")
        path = joinpath(diag_dir, "turing_$(safe)_seed$(seed).json")
        open(path, "w") do f
            JSON3.write(f, Dict(
                "model" => String(model_path),
                "seed" => Int(seed),
                "num_steps" => Int(num_steps),
                "per_step_s" => num_steps > 0 ? fit_time / num_steps : NaN,
                "final_elbo" => isempty(elbo_traj) ? NaN : elbo_traj[end],
                "elbo_trajectory" => elbo_traj,
            ))
        end
    catch e
        @warn "Failed to write SVI diagnostics" exception=e
    end
end

function svi_main(
    spec, data, output_path, seed, num_steps, lr, n_particles, init_scale, K,
    guide, optimizer, server_mode,
)
    model = spec.turing_model
    println("Turing.jl ADVI: $(spec.model_name)")
    println("  model=$MODEL_PATH")
    println("  num_steps=$num_steps, lr(step_size)=$lr, num_particles=$n_particles")
    println("  init_scale=$init_scale, K(draws)=$K, guide=$guide, optimizer=$optimizer")

    # JIT warmup: a short fit compiles the ADVI / AD / draw code paths.
    print("JIT warmup... ")
    compile_time = @elapsed begin
        _ = run_turing_advi(model, seed, 2, lr, n_particles, init_scale, 2, guide, optimizer)
    end
    println("$(round(compile_time * 1000, digits=2)) ms")

    print("Running ADVI fit... ")
    out, elbo_traj, fit_time = run_turing_advi(
        model, seed, num_steps, lr, n_particles, init_scale, K, guide, optimizer
    )
    final_elbo = isempty(elbo_traj) ? NaN : elbo_traj[end]
    println("$(round(fit_time * 1000, digits=2)) ms  final_elbo=$(round(final_elbo, digits=6))")
    svi_write_diagnostics(MODEL_PATH, seed, elbo_traj, fit_time, num_steps)

    out["compile_time"] = compile_time
    out["run_time"] = fit_time
    open(output_path, "w") do f
        JSON3.write(f, out)
    end
    println("Output written to $output_path")

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
            println("Server ADVI trial: seed=$new_seed, output=$new_output_path")
            flush(stdout)

            new_out, new_elbo, new_fit_time = run_turing_advi(
                model, new_seed, num_steps, lr, n_particles, init_scale, K,
                guide, optimizer,
            )
            svi_write_diagnostics(MODEL_PATH, new_seed, new_elbo, new_fit_time, num_steps)
            new_out["compile_time"] = 0.0
            new_out["run_time"] = new_fit_time
            open(new_output_path, "w") do f
                JSON3.write(f, new_out)
            end
            println("###DONE### $(round(new_fit_time * 1000, digits=2)) ms")
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
    server_mode = false
    method = "nuts"
    num_steps = 300
    num_particles = 1
    init_scale = 0.1
    guide = "AutoNormal"
    optimizer = "Adam"
    matched_init = false
    adtype_name = "forwarddiff"

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--model"
            i += 1  # skip (already parsed)
        elseif arg == "--data"
            i += 1; data_path = ARGS[i]
        elseif arg == "--output"
            i += 1; output_path = ARGS[i]
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
        elseif arg == "--matched-init"
            matched_init = true
        elseif arg == "--adtype"
            i += 1; adtype_name = ARGS[i]
        elseif arg == "--server"
            server_mode = true
        else
            @warn "Unknown argument: $arg"
        end
        i += 1
    end

    @assert !isempty(data_path) "Missing --data argument"
    @assert !isempty(output_path) "Missing --output argument"

    # Load data
    data = JSON3.read(read(data_path, String))

    # Model-specific setup
    spec = setup(data)

    # SVI (ADVI) path: separate from the NUTS path below (which is unchanged).
    if method == "svi"
        svi_main(
            spec, data, output_path, seed, num_steps, step_size, num_particles,
            init_scale, num_samples, guide, optimizer, server_mode,
        )
        return
    end

    println("Turing.jl Benchmark: $(spec.model_name)")
    println("  model=$MODEL_PATH")
    println("  num_warmup=$num_warmup, num_samples=$num_samples, seed=$seed")
    println("  step_size=$step_size, max_tree_depth=$max_tree_depth")
    println("  adapt_step_size=$adapt_step_size, adapt_mass_matrix=$adapt_mass_matrix")
    println("  adtype=$adtype_name")

    # Build initial params
    init_params = get_init_params(data, get(data, "init_params", nothing))

    # Matched-init control experiment: deterministic uniform-simplex init,
    # identical to the Impulse arm's NUTS start. DynamicPPL.InitFromParams treats
    # these as constrained (original-space) values. Only the start point changes;
    # the NUTS kernel is otherwise unchanged vs the standard protocol.
    if matched_init && isdefined(@__MODULE__, :matched_init_params)
        init_params = matched_init_params(data)
        println("Turing.jl: --matched-init constrained point:")
        for (nm, val) in pairs(init_params)
            v = vec(collect(val))
            println("  $nm = ", v[1:min(length(v), 3)], " ... (len $(length(v)))")
        end
        try
            lj = DynamicPPL.logjoint(spec.turing_model, init_params)
            println("  logjoint at matched-init point = ", lj)
        catch e
            @warn "logjoint check skipped" exception=e
        end
    end

    # JIT warmup: run a minimal sample() to trigger Julia compilation of all
    # NUTS internals (leapfrog, tree doubling, AD, adaptation).
    # Julia JIT compiles based on type signatures, not runtime values, so this
    # 2-iteration warmup compiles the same code paths as the real call.
    print("JIT warmup... ")
    compile_time = @elapsed begin
        _ = run_turing_nuts(
            spec.turing_model, seed, 0, 2, step_size, max_tree_depth,
            false, false, init_params, adtype_name,
        )
    end
    println("$(round(compile_time * 1000, digits=2)) ms")

    # First trial (JIT already warmed up)
    print("Running... ")
    run_time = @elapsed begin
        chain = run_turing_nuts(
            spec.turing_model, seed, num_warmup, num_samples,
            step_size, max_tree_depth, adapt_step_size, adapt_mass_matrix,
            init_params, adtype_name,
        )
    end
    println("$(round(run_time * 1000, digits=2)) ms")

    # Extract and write output
    output = extract_samples(chain, num_samples)
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

            # Matched init (when set) wins over any per-trial driver init: keep
            # the fixed matched init_params for EVERY trial.
            new_init_params = init_params
            if !matched_init && haskey(req, "init_params") && req["init_params"] !== nothing
                new_init_params = get_init_params(data, req["init_params"])
            end

            new_run_time = @elapsed begin
                new_chain = run_turing_nuts(
                    spec.turing_model, new_seed, num_warmup, num_samples,
                    step_size, max_tree_depth, adapt_step_size, adapt_mass_matrix,
                    new_init_params, adtype_name,
                )
            end

            new_output = extract_samples(new_chain, num_samples)
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
