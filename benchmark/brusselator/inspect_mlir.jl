using Printf
using Reactant

include("runbenchmarks.jl")

function inspect_options(args)
    options = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || throw(ArgumentError("expected --name=value, got $arg"))
        key_value = split(arg[3:end], "="; limit=2)
        length(key_value) == 2 || throw(ArgumentError("expected --name=value, got $arg"))
        options[key_value[1]] = key_value[2]
    end
    N = parse(Int, get(options, "n", "3"))
    Ks = Tuple(parse.(Int, split(get(options, "ks", "1,2,4,8,12"), ',')))
    output_dir = abspath(get(options, "output-dir", joinpath(@__DIR__, "results", "mlir")))
    all(K -> K in SUPPORTED_CHUNKS, Ks) ||
        throw(ArgumentError("Ks must be selected from $SUPPORTED_CHUNKS"))
    return (; N, Ks, output_dir)
end

function chunk_arguments(N, K)
    problem = brusselator_problem(N)
    state = split_state(problem.u)
    seeds = make_tangent_seeds(state, K; kind=:dense)
    compressed = zero_compressed_jacobian(state, K)
    return Reactant.to_rarray((compressed, state, seeds, problem.coordinates, problem.p))
end

function chunk_mlir(wrapper, args; compile_options=nothing)
    module_ = if isnothing(compile_options)
        Reactant.@code_hlo optimize = :none wrapper(args...)
    else
        Reactant.@code_hlo compile_options = compile_options wrapper(args...)
    end
    return sprint(show, module_)
end

function pass_prefix(compile_options)
    common = (; recognize_comms=true, lower_comms=true, backend="GPU", is_sharded=false)
    first = Reactant.Compiler.optimization_passes(compile_options; sroa=true, common...)
    second = Reactant.Compiler.optimization_passes(compile_options; sroa=false, common...)
    return join(("mark-func-memory-effects", first, "enzyme-batch", second), ',')
end

function post_enzyme_optimization_pipeline(compile_options)
    common = (; recognize_comms=true, lower_comms=true, backend="GPU", is_sharded=false)
    optimization = Reactant.Compiler.optimization_passes(
        compile_options; sroa=false, common...
    )
    return join(
        (
            optimization,
            "canonicalize",
            "remove-unnecessary-enzyme-ops",
            "enzyme-simplify-math",
        ),
        ',',
    )
end

function run_pipeline(source, pipeline)
    return string(Reactant.Compiler.run_pass_pipeline_on_source(source, pipeline))
end

function occurrences(needle, haystack)
    return length(findall(needle, haystack))
end

function fwddiff_details(source)
    lines = filter(line -> occursin("enzyme.fwddiff", line), split(source, '\n'))
    widths = map(lines) do line
        match_ = match(r"width\s*=\s*(\d+)", line)
        return isnothing(match_) ? 1 : parse(Int, only(match_.captures))
    end
    callees = map(lines) do line
        match_ = match(r"enzyme\.fwddiff\s+@\"([^\"]+)\"", line)
        return isnothing(match_) ? "<unknown>" : only(match_.captures)
    end
    return (; count=length(lines), widths, callees)
end

function stage_summary(source)
    fwddiff = fwddiff_details(source)
    return (;
        fwddiff,
        enzyme_concat=occurrences("enzyme.concat", source),
        enzyme_extract=occurrences("enzyme.extract", source),
        stablehlo_concatenate=occurrences("stablehlo.concatenate", source),
        stablehlo_slice=occurrences("stablehlo.slice", source),
    )
end

function print_summary(K, name, source)
    summary = stage_summary(source)
    @printf(
        "K=%2d %-26s fwddiff=%2d widths=%-16s concat/extract=%d/%d stablehlo.concat/slice=%d/%d\n",
        K,
        name,
        summary.fwddiff.count,
        string(summary.fwddiff.widths),
        summary.enzyme_concat,
        summary.enzyme_extract,
        summary.stablehlo_concatenate,
        summary.stablehlo_slice,
    )
    return summary
end

function save_stage(output_dir, K, name, source)
    directory = joinpath(output_dir, "k$K")
    mkpath(directory)
    path = joinpath(directory, "$name.mlir")
    write(path, source)
    return path
end

function inspect_brusselator_mlir(; N=3, Ks=(1, 2, 4, 8, 12), output_dir)
    off_options = brusselator_compile_options(false)
    on_options = brusselator_compile_options(true)
    prefix = pass_prefix(off_options)
    off_post_enzyme_optimization = post_enzyme_optimization_pipeline(off_options)
    on_post_enzyme_optimization = post_enzyme_optimization_pipeline(on_options)

    println("Brusselator differentiation-batching MLIR inspection")
    println("  N:          ", N)
    println("  K:          ", join(Ks, ','))
    println("  output dir: ", output_dir)
    println()

    results = map(Ks) do K
        wrapper = chunk_function(K)
        initial = chunk_mlir(wrapper, chunk_arguments(N, K))
        save_stage(output_dir, K, "initial", initial)
        initial_summary = print_summary(K, "initial", initial)

        before_diff_batch = run_pipeline(initial, prefix)
        save_stage(output_dir, K, "before_diff_batch", before_diff_batch)
        before_diff_batch_summary =
            print_summary(K, "before_diff_batch", before_diff_batch)

        off_after_core_enzyme = run_pipeline(
            before_diff_batch, Reactant.Compiler.enzyme_pass
        )
        save_stage(output_dir, K, "off_after_core_enzyme", off_after_core_enzyme)
        off_after_core_enzyme_summary =
            print_summary(K, "off_after_core_enzyme", off_after_core_enzyme)

        off_after_post_optimization = run_pipeline(
            off_after_core_enzyme, off_post_enzyme_optimization
        )
        save_stage(
            output_dir, K, "off_after_post_optimization", off_after_post_optimization
        )
        off_after_post_optimization_summary = print_summary(
            K, "off_after_post_optimization", off_after_post_optimization
        )

        after_diff_batch = run_pipeline(before_diff_batch, "enzyme-diff-batch")
        save_stage(output_dir, K, "after_diff_batch", after_diff_batch)
        after_diff_batch_summary = print_summary(K, "after_diff_batch", after_diff_batch)

        after_batch_legalization =
            run_pipeline(after_diff_batch, "enzyme-batch-to-stablehlo")
        save_stage(output_dir, K, "after_batch_legalization", after_batch_legalization)
        after_batch_legalization_summary =
            print_summary(K, "after_batch_legalization", after_batch_legalization)

        after_core_enzyme = run_pipeline(
            after_batch_legalization, Reactant.Compiler.enzyme_pass
        )
        save_stage(output_dir, K, "after_core_enzyme", after_core_enzyme)
        after_core_enzyme_summary =
            print_summary(K, "after_core_enzyme", after_core_enzyme)

        # This is a diagnostic cleanup of helpers emitted by core Enzyme, not part of
        # the production differentiation-batching pipeline. Keeping it explicit here
        # lets us inspect and lower the resulting StableHLO independently.
        after_post_batch_legalization =
            run_pipeline(after_core_enzyme, "enzyme-batch-to-stablehlo")
        save_stage(
            output_dir,
            K,
            "after_post_batch_legalization",
            after_post_batch_legalization,
        )
        after_post_batch_legalization_summary = print_summary(
            K, "after_post_batch_legalization", after_post_batch_legalization
        )

        after_post_optimization = run_pipeline(
            after_post_batch_legalization, on_post_enzyme_optimization
        )
        save_stage(output_dir, K, "after_post_optimization", after_post_optimization)
        after_post_optimization_summary =
            print_summary(K, "after_post_optimization", after_post_optimization)

        full_off = chunk_mlir(wrapper, chunk_arguments(N, K); compile_options=off_options)
        save_stage(output_dir, K, "full_off", full_off)
        full_off_summary = print_summary(K, "full_off", full_off)

        full_on = chunk_mlir(wrapper, chunk_arguments(N, K); compile_options=on_options)
        save_stage(output_dir, K, "full_on", full_on)
        full_on_summary = print_summary(K, "full_on", full_on)

        println()
        return (;
            initial=initial_summary,
            before_diff_batch=before_diff_batch_summary,
            off_after_core_enzyme=off_after_core_enzyme_summary,
            off_after_post_optimization=off_after_post_optimization_summary,
            after_diff_batch=after_diff_batch_summary,
            after_batch_legalization=after_batch_legalization_summary,
            after_core_enzyme=after_core_enzyme_summary,
            after_post_batch_legalization=after_post_batch_legalization_summary,
            after_post_optimization=after_post_optimization_summary,
            full_off=full_off_summary,
            full_on=full_on_summary,
        )
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = inspect_options(ARGS)
    inspect_brusselator_mlir(; options...)
end
