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
using AdvancedHMC: DenseEuclideanMetric

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
include(joinpath(@__DIR__, "turing", MODEL_PATH * ".jl"))

# =============================================================================
# Run NUTS via Turing
# =============================================================================

function run_turing_nuts(
    model, seed, num_warmup, num_samples, step_size, max_tree_depth,
    adapt_step_size, adapt_mass_matrix, init_params,
)
    rng = Xoshiro(seed)
    total_samples = num_warmup + num_samples

    if adapt_step_size || adapt_mass_matrix
        n_adapts = num_warmup
        sampler = NUTS(n_adapts, 0.8; max_depth=max_tree_depth, init_ϵ=step_size, metricT=DenseEuclideanMetric)
    else
        sampler = NUTS(0, 0.8; max_depth=max_tree_depth, init_ϵ=step_size, metricT=DenseEuclideanMetric)
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

    println("Turing.jl Benchmark: $(spec.model_name)")
    println("  model=$MODEL_PATH")
    println("  num_warmup=$num_warmup, num_samples=$num_samples, seed=$seed")
    println("  step_size=$step_size, max_tree_depth=$max_tree_depth")
    println("  adapt_step_size=$adapt_step_size, adapt_mass_matrix=$adapt_mass_matrix")

    # Build initial params
    init_params = get_init_params(data, get(data, "init_params", nothing))

    # JIT warmup: run a minimal sample() to trigger Julia compilation of all
    # NUTS internals (leapfrog, tree doubling, AD, adaptation).
    # Julia JIT compiles based on type signatures, not runtime values, so this
    # 2-iteration warmup compiles the same code paths as the real call.
    print("JIT warmup... ")
    compile_time = @elapsed begin
        _ = run_turing_nuts(
            spec.turing_model, seed, 0, 2, step_size, max_tree_depth,
            false, false, init_params,
        )
    end
    println("$(round(compile_time * 1000, digits=2)) ms")

    # First trial (JIT already warmed up)
    print("Running... ")
    run_time = @elapsed begin
        chain = run_turing_nuts(
            spec.turing_model, seed, num_warmup, num_samples,
            step_size, max_tree_depth, adapt_step_size, adapt_mass_matrix,
            init_params,
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

            new_init_params = init_params
            if haskey(req, "init_params") && req["init_params"] !== nothing
                new_init_params = get_init_params(data, req["init_params"])
            end

            new_run_time = @elapsed begin
                new_chain = run_turing_nuts(
                    spec.turing_model, new_seed, num_warmup, num_samples,
                    step_size, max_tree_depth, adapt_step_size, adapt_mass_matrix,
                    new_init_params,
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
