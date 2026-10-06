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
    N = parse(Int, get(options, "n", "16"))
    Ks = Tuple(parse.(Int, split(get(options, "ks", "1,2,4,8,12"), ',')))
    output_dir = abspath(get(options, "output-dir", joinpath(@__DIR__, "results", "mlir")))
    backend = get(options, "backend", "gpu")
    backend in ("cpu", "gpu") || throw(ArgumentError("backend must be cpu or gpu"))
    all(K -> K in SUPPORTED_CHUNKS, Ks) ||
        throw(ArgumentError("Ks must be selected from $SUPPORTED_CHUNKS"))
    return (; N, Ks, output_dir, backend)
end

function chunk_arguments(N, K)
    problem = brusselator_problem(N)
    state = split_state(problem.u)
    seeds = make_tangent_seeds(state, K; kind=:dense)
    compressed = zero_compressed_jacobian(state, K)
    return Reactant.to_rarray((compressed, state, seeds, problem.coordinates, problem.p))
end

function fwddiff_details(source)
    lines = filter(line -> occursin("enzyme.fwddiff", line), split(source, '\n'))
    widths = map(lines) do line
        match_ = match(r"width\s*=\s*(\d+)", line)
        return isnothing(match_) ? 1 : parse(Int, only(match_.captures))
    end
    callees = map(lines) do line
        match_ = match(r"enzyme\.fwddiff\s+@(?:\"([^\"]+)\"|([^\s(]+))", line)
        return isnothing(match_) ? "<unknown>" : something(match_[1], match_[2])
    end
    return (; count=length(lines), widths, callees)
end

function save_stage(directory, name, source)
    path = joinpath(directory, "$name.mlir")
    write(path, source)
    details = fwddiff_details(source)
    println(name, ": fwddiff=", details.count, " widths=", details.widths)
    return details
end

function run_pipeline(source, pipeline)
    return string(Reactant.Compiler.run_pass_pipeline_on_source(source, pipeline))
end

# Read the exact pipeline emitted by the real :all compilation, including its backend
# options. Do not reconstruct the historical optimization_passes(...) API by hand.
function production_pipeline(source)
    lines = split(source, '\n'; limit=3)
    @assert lines[1] == "// Pass pipeline:"
    pipeline = strip(chopprefix(lines[2], "// "))
    for anchor in ("any(", "builtin.module(")
        if startswith(pipeline, anchor)
            pipeline = chop(chopprefix(pipeline, anchor))
            break
        end
    end
    return pipeline
end

function top_level_passes(pipeline)
    passes = String[]
    depth = 0
    quoted = false
    escaped = false
    start = firstindex(pipeline)
    for (i, char) in pairs(pipeline)
        if escaped
            escaped = false
        elseif char == '\\' && quoted
            escaped = true
        elseif char == '"'
            quoted = !quoted
        elseif !quoted
            if char in ('(', '{', '[')
                depth += 1
            elseif char in (')', '}', ']')
                depth -= 1
            elseif char == ',' && depth == 0
                push!(passes, strip(pipeline[start:prevind(pipeline, i)]))
                start = nextind(pipeline, i)
            end
        end
    end
    @assert depth == 0 && !quoted
    push!(passes, strip(pipeline[start:end]))
    return passes
end

function capture_production(wrapper, args, options, directory)
    mkpath(directory)
    dump_always = Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[]
    dump_dir = Reactant.MLIR.IR.DUMP_MLIR_DIR[]
    try
        Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
        Reactant.MLIR.IR.DUMP_MLIR_DIR[] = directory
        compiled = Reactant.compile(wrapper, args; compile_options=options)
        compiled(args...)
    finally
        Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = dump_always
        Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dump_dir
    end
    source_path = only(
        filter(path -> endswith(path, "_pre_all_pm.mlir"), readdir(directory; join=true))
    )
    source = read(source_path, String)
    return source, production_pipeline(source), Array(args[1])
end

function inspect_brusselator_mlir(; N=16, Ks=SUPPORTED_CHUNKS, output_dir, backend="gpu")
    Reactant.set_default_backend(backend)
    mkpath(output_dir)
    summaries = []
    for K in Ks
        outputs = Dict{Bool,Matrix{Float64}}()
        for diff_batch in (false, true)
            directory = joinpath(output_dir, "k$K", diff_batch ? "on" : "off")
            # Refuse stale captures: every file in this directory must come from this run.
            isdir(directory) &&
                !isempty(readdir(directory)) &&
                error("Use an empty output directory: $directory")
            options = brusselator_compile_options(diff_batch)
            source, pipeline, output = capture_production(
                chunk_function(K), chunk_arguments(N, K), options, directory
            )
            outputs[diff_batch] = output
            @assert size(output) == (2 * N^2, K)
            @assert all(isfinite, output)
            write(joinpath(directory, "production-pipeline.txt"), pipeline * "\n")
            initial = save_stage(directory, "production-input", source)

            # Replay only an exact prefix of the captured production pipeline to expose
            # the intermediate explicit AD requests. The full pipeline above also ran
            # through XLA and executed on the selected device.
            stages = top_level_passes(pipeline)
            boundary = diff_batch ? "enzyme-diff-batch" : "enzyme{"
            position = findfirst(stage -> startswith(stage, boundary), stages)
            @assert !isnothing(position)
            prefix = join(stages[1:(position - 1)], ',')
            before = run_pipeline(source, prefix)
            before_details = save_stage(directory, "before-differentiation", before)
            @assert before_details.count == K
            @assert before_details.widths == fill(1, K)
            @assert length(unique(before_details.callees)) == 1
            @assert only(unique(before_details.callees)) != "<unknown>"
            if diff_batch
                after = run_pipeline(before, stages[position])
                after_details = save_stage(directory, "after-diff-batch", after)
                @assert after_details.count == 1
                @assert after_details.widths == [K]
                @assert startswith(stages[position + 1], "enzyme-batch-to-stablehlo")
                legal = run_pipeline(after, stages[position + 1])
                save_stage(directory, "after-batch-legalization", legal)
            end
            push!(
                summaries,
                (;
                    K,
                    diff_batch,
                    initial_count=initial.count,
                    before_count=before_details.count,
                    final_count=diff_batch ? 1 : K,
                    width=diff_batch ? K : 1,
                ),
            )
        end
        @assert passes(outputs[true], outputs[false]; atol=1e-10, rtol=1e-10)
        println("K=$K production execution and off/on equivalence: PASS")
    end
    open(joinpath(output_dir, "summary.tsv"), "w") do io
        println(
            io, "K\tdiff_batch\tinitial_requests\tbefore_requests\tafter_requests\twidth"
        )
        for s in summaries
            println(io, join(Tuple(s), '\t'))
        end
    end
    return summaries
end

if abspath(PROGRAM_FILE) == @__FILE__
    inspect_brusselator_mlir(; inspect_options(ARGS)...)
end
