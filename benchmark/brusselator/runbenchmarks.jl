using Enzyme
using Printf
using Reactant

include(joinpath(@__DIR__, "workload.jl"))
using .BrusselatorWorkload

zero_state(state) = map(component -> zeros(eltype(component), size(component)), state)
host_state(state) = map(Array, state)
logical_array(state::Tuple) = stack_state(host_state(state))
logical_array(array::AbstractArray) = Array(array)

function error_metrics(actual, reference; relative_floor=nothing)
    actual_array = logical_array(actual)
    reference_array = logical_array(reference)
    relative_floor === nothing && (relative_floor = eps(eltype(reference_array)))
    difference = abs.(actual_array .- reference_array)
    scale = max.(abs.(actual_array), abs.(reference_array), relative_floor)
    return (;
        max_abs=maximum(difference),
        max_rel=maximum(difference ./ scale),
        finite=all(isfinite, actual_array),
        shape=size(actual_array),
    )
end

function passes(actual, reference; atol, rtol)
    actual_array = logical_array(actual)
    reference_array = logical_array(reference)
    return all(isapprox.(actual_array, reference_array; atol, rtol)) &&
           all(isfinite, actual_array)
end

function buffers_do_not_alias(arrays)
    for i in eachindex(arrays)
        for j in (i + 1):length(arrays)
            Base.mightalias(arrays[i], arrays[j]) && return false
        end
    end
    return true
end

function collect_arrays!(arrays, value)
    if value isa AbstractArray
        push!(arrays, value)
    elseif value isa Tuple
        foreach(item -> collect_arrays!(arrays, item), value)
    end
    return arrays
end

function flatten_states(states)
    arrays = Any[]
    collect_arrays!(arrays, states)
    return arrays
end

function brusselator_compile_options(diff_batch::Bool)
    ad_optimization_passes = if diff_batch
        Reactant.ADOptimizationOptions(; diff_batch=true)
    else
        false
    end
    return Reactant.CompileOptions(; sync=true, ad_optimization_passes)
end

function compile_timed(f, args, compile_options::Reactant.CompileOptions)
    compiled = Ref{Any}()
    seconds = @elapsed compiled[] = Reactant.compile(f, args; compile_options)
    return compiled[], seconds
end

function execute_timed(compiled, args)
    return @elapsed compiled(args...)
end

function steady_timings(compiled, args, samples)
    times = [execute_timed(compiled, args) for _ in 1:samples]
    sorted = sort(times)
    middle = (length(sorted) + 1) ÷ 2
    median =
        isodd(length(sorted)) ? sorted[middle] : (sorted[middle] + sorted[middle + 1]) / 2
    return (; minimum=minimum(times), median, samples=times)
end

function package_version(package)
    version = Base.pkgversion(package)
    return version === nothing ? "unknown" : string(version)
end

function repository_sha()
    try
        repository_root = normpath(joinpath(@__DIR__, "..", ".."))
        return readchomp(`git -C $repository_root rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function print_metrics(label, metrics, correct)
    @printf(
        "%-38s max_abs=%10.3e  max_rel=%10.3e  finite=%-5s shape=%-12s %s\n",
        label,
        metrics.max_abs,
        metrics.max_rel,
        string(metrics.finite),
        string(metrics.shape),
        correct ? "PASS" : "FAIL",
    )
end

function print_timing(label, timing)
    @printf(
        "%-24s compile=%9.4f s  first_exec=%9.6f s  compile+first=%9.4f s  steady_median=%9.6f s  steady_min=%9.6f s\n",
        label,
        timing.compile,
        timing.first_execution,
        timing.compile + timing.first_execution,
        timing.steady.median,
        timing.steady.minimum,
    )
end

function print_performance_timing(label, timing, work_items; directions=1)
    milliseconds = 1.0e3 * timing.steady.median
    giga_items_per_second = work_items / timing.steady.median / 1.0e9
    @printf(
        "%-24s compile=%9.4f s  first_exec=%9.6f s  steady_median=%9.6f s (%8.3f ms)  steady_min=%8.3f ms  throughput=%8.3f Gcell-dir/s  per_direction=%8.3f ms\n",
        label,
        timing.compile,
        timing.first_execution,
        timing.steady.median,
        milliseconds,
        1.0e3 * timing.steady.minimum,
        giga_items_per_second,
        milliseconds / directions,
    )
end

function run_brusselator_validation(;
    N::Int=16,
    Ks::Tuple=(1, 2, 4),
    seed_kind::Symbol=:dense,
    epsilon::Float64=1.0e-4,
    samples::Int=5,
    primal_atol::Float64=1.0e-10,
    primal_rtol::Float64=1.0e-10,
    jvp_atol::Float64=2.0e-6,
    jvp_rtol::Float64=2.0e-7,
    diff_batch::Bool=false,
)
    isempty(Ks) && throw(ArgumentError("Ks must not be empty"))
    all(K -> K in SUPPORTED_CHUNKS, Ks) ||
        throw(ArgumentError("Ks must be selected from $SUPPORTED_CHUNKS"))
    samples > 0 || throw(ArgumentError("samples must be positive"))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive"))
    compile_options = brusselator_compile_options(diff_batch)

    problem = brusselator_problem(N)
    state = split_state(problem.u)
    max_K = maximum(Ks)
    seeds_3d = make_tangent_seeds(problem.u, max_K; kind=seed_kind)
    seeds = map(split_state, seeds_3d)

    reference_primal = similar(problem.u)
    brusselator_2d_reference!(reference_primal, problem.u, problem.coordinates, problem.p)
    julia_primal = zero_state(state)
    brusselator_2d_loop!(julia_primal, state, problem.coordinates, problem.p)
    julia_primal_metrics = error_metrics(julia_primal, reference_primal)
    julia_primal_ok = passes(
        julia_primal, reference_primal; atol=primal_atol, rtol=primal_rtol
    )

    primal_args = Reactant.to_rarray((
        zero_state(state), state, problem.coordinates, problem.p
    ))
    primal_compiled, primal_compile_seconds = compile_timed(
        brusselator_2d_loop!, primal_args, compile_options
    )
    primal_first_seconds = execute_timed(primal_compiled, primal_args)
    primal_steady = steady_timings(primal_compiled, primal_args, samples)
    reactant_primal = host_state(primal_args[1])
    reactant_primal_metrics = error_metrics(reactant_primal, reference_primal)
    reactant_primal_ok = passes(
        reactant_primal, reference_primal; atol=primal_atol, rtol=primal_rtol
    )
    primal_timing = (;
        compile=primal_compile_seconds,
        first_execution=primal_first_seconds,
        steady=primal_steady,
    )

    fd_jvps = map(seeds_3d) do seed
        return finite_difference_jvp(
            problem.u, seed, problem.coordinates, problem.p; epsilon
        )
    end

    native_jvps = map(seeds) do seed
        output = zero_state(state)
        residual_jvp!(output, state, seed, problem.coordinates, problem.p)
        return output
    end
    native_jvp_metrics = error_metrics(native_jvps[1], fd_jvps[1])
    native_jvp_ok = passes(native_jvps[1], fd_jvps[1]; atol=jvp_atol, rtol=jvp_rtol)

    single_args = Reactant.to_rarray((
        zero_state(state),
        state,
        seeds[1],
        problem.coordinates,
        problem.p,
    ))
    single_compiled, single_compile_seconds = compile_timed(
        residual_jvp!, single_args, compile_options
    )
    single_first_seconds = execute_timed(single_compiled, single_args)
    single_steady = steady_timings(single_compiled, single_args, samples)
    reactant_single = host_state(single_args[1])
    single_native_metrics = error_metrics(reactant_single, native_jvps[1])
    single_fd_metrics = error_metrics(reactant_single, fd_jvps[1])
    single_native_ok = passes(
        reactant_single, native_jvps[1]; atol=primal_atol, rtol=primal_rtol
    )
    single_fd_ok = passes(reactant_single, fd_jvps[1]; atol=jvp_atol, rtol=jvp_rtol)
    single_timing = (;
        compile=single_compile_seconds,
        first_execution=single_first_seconds,
        steady=single_steady,
    )

    individual_reactant_jvps = map(seeds) do seed
        args = Reactant.to_rarray((
            zero_state(state),
            state,
            seed,
            problem.coordinates,
            problem.p,
        ))
        single_compiled(args...)
        return host_state(args[1])
    end

    chunk_results = Dict{Int,Any}()
    chunks_ok = true
    for K in Ks
        outputs = ntuple(_ -> zero_state(state), K)
        chunk_seeds = ntuple(k -> seeds[k], K)
        alias_arrays = flatten_states((state, chunk_seeds, outputs))
        alias_ok = buffers_do_not_alias(alias_arrays)

        chunk_args = Reactant.to_rarray((
            outputs, state, chunk_seeds, problem.coordinates, problem.p
        ))
        wrapper = chunk_function(K)
        chunk_compiled, chunk_compile_seconds = compile_timed(
            wrapper, chunk_args, compile_options
        )
        chunk_first_seconds = execute_timed(chunk_compiled, chunk_args)
        chunk_steady = steady_timings(chunk_compiled, chunk_args, samples)
        chunk_outputs = map(host_state, chunk_args[1])

        individual_metrics = ntuple(
            k -> error_metrics(chunk_outputs[k], individual_reactant_jvps[k]), K
        )
        fd_metrics = ntuple(k -> error_metrics(chunk_outputs[k], fd_jvps[k]), K)
        individual_ok = all(1:K) do k
            return passes(
                chunk_outputs[k],
                individual_reactant_jvps[k];
                atol=primal_atol,
                rtol=primal_rtol,
            )
        end
        fd_ok = all(1:K) do k
            return passes(chunk_outputs[k], fd_jvps[k]; atol=jvp_atol, rtol=jvp_rtol)
        end
        chunk_ok = alias_ok && individual_ok && fd_ok
        chunks_ok &= chunk_ok

        chunk_results[K] = (;
            outputs=chunk_outputs,
            alias_ok,
            individual_metrics,
            fd_metrics,
            individual_ok,
            fd_ok,
            correct=chunk_ok,
            timing=(;
                compile=chunk_compile_seconds,
                first_execution=chunk_first_seconds,
                steady=chunk_steady,
            ),
        )
    end

    correct =
        julia_primal_ok &&
        reactant_primal_ok &&
        native_jvp_ok &&
        single_native_ok &&
        single_fd_ok &&
        chunks_ok

    println("Brusselator Reactant validation")
    println("  Julia:    ", VERSION)
    println("  Reactant: ", package_version(Reactant))
    println("  Enzyme:   ", package_version(Enzyme))
    println("  repo SHA: ", repository_sha())
    println("  N:        ", N)
    println("  K:        ", join(Ks, ","))
    println("  seeds:    ", seed_kind)
    println("  epsilon:  ", epsilon)
    println("  diff batch: ", diff_batch)
    println("  logical output shape: ", size(reference_primal))
    println()
    print_metrics(
        "ordinary component primal vs loop", julia_primal_metrics, julia_primal_ok
    )
    print_metrics(
        "Reactant primal vs ordinary loop", reactant_primal_metrics, reactant_primal_ok
    )
    print_metrics("ordinary Enzyme JVP vs finite diff", native_jvp_metrics, native_jvp_ok)
    print_metrics(
        "Reactant JVP vs ordinary Enzyme", single_native_metrics, single_native_ok
    )
    print_metrics("Reactant JVP vs finite diff", single_fd_metrics, single_fd_ok)
    for K in Ks
        result = chunk_results[K]
        max_individual_abs = maximum(metric.max_abs for metric in result.individual_metrics)
        max_individual_rel = maximum(metric.max_rel for metric in result.individual_metrics)
        max_fd_abs = maximum(metric.max_abs for metric in result.fd_metrics)
        max_fd_rel = maximum(metric.max_rel for metric in result.fd_metrics)
        @printf(
            "chunk K=%-2d vs individual: max_abs=%10.3e max_rel=%10.3e; vs FD: max_abs=%10.3e max_rel=%10.3e; alias=%-5s %s\n",
            K,
            max_individual_abs,
            max_individual_rel,
            max_fd_abs,
            max_fd_rel,
            string(result.alias_ok),
            result.correct ? "PASS" : "FAIL",
        )
    end
    println()
    println("Synchronous timings")
    print_timing("primal", primal_timing)
    print_timing("single JVP", single_timing)
    for K in Ks
        print_timing("chunk K=$K", chunk_results[K].timing)
    end
    println()
    println("overall correctness: ", correct ? "PASS" : "FAIL")

    correct || error("Brusselator validation failed")
    return (;
        correct,
        N,
        Ks,
        seed_kind,
        diff_batch,
        primal=(;
            julia_metrics=julia_primal_metrics,
            reactant_metrics=reactant_primal_metrics,
            timing=primal_timing,
        ),
        single_jvp=(;
            native_metrics=native_jvp_metrics,
            reactant_native_metrics=single_native_metrics,
            finite_difference_metrics=single_fd_metrics,
            timing=single_timing,
        ),
        chunks=chunk_results,
    )
end

"""
    run_brusselator_performance(; N=4096, Ks=(1, 2, 4, 8, 12), ...)

Run a single-device, synchronized performance benchmark at a scale large enough to avoid
measuring only launch overhead. Numerical finite-difference checks deliberately remain in
`run_brusselator_validation`, whose smaller grid avoids cancellation from the `O(N^2)`
diffusion coefficient. This path still checks finite outputs and independent buffers.
"""
function run_brusselator_performance(;
    N::Int=4096,
    Ks::Tuple=(1, 2, 4, 8, 12),
    seed_kind::Symbol=:onehot,
    samples::Int=30,
    diff_batch::Bool=false,
)
    isempty(Ks) && throw(ArgumentError("Ks must not be empty"))
    all(K -> K in SUPPORTED_CHUNKS, Ks) ||
        throw(ArgumentError("Ks must be selected from $SUPPORTED_CHUNKS"))
    samples > 0 || throw(ArgumentError("samples must be positive"))
    compile_options = brusselator_compile_options(diff_batch)

    problem = brusselator_problem(N)
    state = split_state(problem.u)
    max_K = maximum(Ks)
    seeds = make_tangent_seeds(state, max_K; kind=seed_kind)
    logical_state_bytes = 2 * N^2 * sizeof(eltype(problem.u))

    primal_args = Reactant.to_rarray((
        zero_state(state), state, problem.coordinates, problem.p
    ))
    primal_compiled, primal_compile_seconds = compile_timed(
        brusselator_2d_loop!, primal_args, compile_options
    )
    primal_first_seconds = execute_timed(primal_compiled, primal_args)
    primal_steady = steady_timings(primal_compiled, primal_args, samples)
    primal_finite = all(isfinite, logical_array(primal_args[1]))
    primal_timing = (;
        compile=primal_compile_seconds,
        first_execution=primal_first_seconds,
        steady=primal_steady,
    )
    primal_args = nothing
    primal_compiled = nothing
    GC.gc(true)

    single_args = Reactant.to_rarray((
        zero_state(state),
        state,
        seeds[1],
        problem.coordinates,
        problem.p,
    ))
    single_compiled, single_compile_seconds = compile_timed(
        residual_jvp!, single_args, compile_options
    )
    single_first_seconds = execute_timed(single_compiled, single_args)
    single_steady = steady_timings(single_compiled, single_args, samples)
    single_finite = all(isfinite, logical_array(single_args[1]))
    single_timing = (;
        compile=single_compile_seconds,
        first_execution=single_first_seconds,
        steady=single_steady,
    )
    single_args = nothing
    single_compiled = nothing
    GC.gc(true)

    chunk_results = Dict{Int,Any}()
    chunks_ok = true
    for K in Ks
        outputs = ntuple(_ -> zero_state(state), K)
        chunk_seeds = ntuple(k -> seeds[k], K)
        alias_arrays = flatten_states((state, chunk_seeds, outputs))
        alias_ok = buffers_do_not_alias(alias_arrays)

        chunk_args = Reactant.to_rarray((
            outputs, state, chunk_seeds, problem.coordinates, problem.p
        ))
        chunk_compiled, chunk_compile_seconds = compile_timed(
            chunk_function(K), chunk_args, compile_options
        )
        chunk_first_seconds = execute_timed(chunk_compiled, chunk_args)
        chunk_steady = steady_timings(chunk_compiled, chunk_args, samples)
        first_output_finite = all(isfinite, logical_array(chunk_args[1][1]))
        chunk_ok = alias_ok && first_output_finite
        chunks_ok &= chunk_ok
        chunk_results[K] = (;
            alias_ok,
            finite=first_output_finite,
            argument_gib=(1 + 2 * K) * logical_state_bytes / 2.0^30,
            timing=(;
                compile=chunk_compile_seconds,
                first_execution=chunk_first_seconds,
                steady=chunk_steady,
            ),
        )
        chunk_args = nothing
        chunk_compiled = nothing
        GC.gc(true)
    end

    correct = primal_finite && single_finite && chunks_ok
    println("Brusselator Reactant performance benchmark")
    println("  Julia:    ", VERSION)
    println("  Reactant: ", package_version(Reactant))
    println("  Enzyme:   ", package_version(Enzyme))
    println("  repo SHA: ", repository_sha())
    println("  N:        ", N, " (", N^2, " cells)")
    println("  K:        ", join(Ks, ","))
    println("  seeds:    ", seed_kind)
    println("  samples:  ", samples)
    println("  diff batch: ", diff_batch)
    @printf("  one logical state: %.3f GiB\n", logical_state_bytes / 2.0^30)
    println("  execution is synchronous; no distributed sharding is requested")
    println()
    print_performance_timing("primal", primal_timing, N^2)
    print_performance_timing("single JVP", single_timing, N^2)
    for K in Ks
        result = chunk_results[K]
        print_performance_timing("chunk K=$K", result.timing, K * N^2; directions=K)
        @printf(
            "  K=%-2d argument buffers=%6.3f GiB  finite=%-5s alias=%-5s\n",
            K,
            result.argument_gib,
            string(result.finite),
            string(result.alias_ok),
        )
    end
    println()
    println("performance sanity checks: ", correct ? "PASS" : "FAIL")

    correct || error("Brusselator performance sanity checks failed")
    return (;
        correct,
        N,
        Ks,
        seed_kind,
        diff_batch,
        primal=(; finite=primal_finite, timing=primal_timing),
        single_jvp=(; finite=single_finite, timing=single_timing),
        chunks=chunk_results,
    )
end

function parse_command_line(args)
    options = Dict{String,String}()
    for argument in args
        startswith(argument, "--") ||
            throw(ArgumentError("expected --name=value, got $argument"))
        pieces = split(argument[3:end], "="; limit=2)
        length(pieces) == 2 || throw(ArgumentError("expected --name=value, got $argument"))
        options[pieces[1]] = pieces[2]
    end
    mode = Symbol(get(options, "mode", "validation"))
    mode in (:validation, :performance) ||
        throw(ArgumentError("mode must be validation or performance"))
    default_N = mode === :validation ? "16" : "4096"
    default_Ks = mode === :validation ? "1,2,4" : "1,2,4,8,12"
    default_seed = mode === :validation ? "dense" : "onehot"
    default_samples = mode === :validation ? "5" : "30"
    N = parse(Int, get(options, "n", default_N))
    Ks = Tuple(parse.(Int, split(get(options, "ks", default_Ks), ",")))
    seed_kind = Symbol(get(options, "seed", default_seed))
    epsilon = parse(Float64, get(options, "epsilon", "1e-4"))
    samples = parse(Int, get(options, "samples", default_samples))
    diff_batch = parse(Bool, get(options, "diff-batch", "false"))
    return (; mode, N, Ks, seed_kind, epsilon, samples, diff_batch)
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = parse_command_line(ARGS)
    if options.mode === :validation
        run_brusselator_validation(;
            N=options.N,
            Ks=options.Ks,
            seed_kind=options.seed_kind,
            epsilon=options.epsilon,
            samples=options.samples,
            diff_batch=options.diff_batch,
        )
    else
        run_brusselator_performance(;
            N=options.N,
            Ks=options.Ks,
            seed_kind=options.seed_kind,
            samples=options.samples,
            diff_batch=options.diff_batch,
        )
    end
end
