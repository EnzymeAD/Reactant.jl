using Printf
using Reactant

include("runbenchmarks.jl")

function profile_options(args)
    options = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || throw(ArgumentError("expected --name=value, got $arg"))
        key_value = split(arg[3:end], "="; limit=2)
        length(key_value) == 2 || throw(ArgumentError("expected --name=value, got $arg"))
        options[key_value[1]] = key_value[2]
    end
    N = parse(Int, get(options, "n", "4096"))
    samples = parse(Int, get(options, "samples", "10"))
    warmup = parse(Int, get(options, "warmup", "3"))
    diff_batch = parse(Bool, get(options, "diff-batch", "false"))
    profile_dir = get(options, "profile-dir", "")
    N > 1 || throw(ArgumentError("n must be greater than one"))
    samples > 0 || throw(ArgumentError("samples must be positive"))
    warmup > 0 || throw(ArgumentError("warmup must be positive"))
    return (; N, samples, warmup, diff_batch, profile_dir)
end

function save_profile_reports(profile_dir, xplane_file)
    kernel_stats = Reactant.Profiler.get_kernel_stats(xplane_file)
    framework_stats = Reactant.Profiler.get_framework_op_stats(xplane_file)

    open(joinpath(profile_dir, "kernel-stats.txt"), "w") do io
        Reactant.Profiler.print_kernel_report(kernel_stats; io)
    end
    open(joinpath(profile_dir, "framework-op-stats.txt"), "w") do io
        Reactant.Profiler.print_framework_op_stats(framework_stats; io)
    end
    open(joinpath(profile_dir, "kernel-stats.tsv"), "w") do io
        println(
            io,
            "name\toccurrences\ttotal_ns\taverage_ns\tregisters_per_thread\t" *
            "static_shared_bytes\tdynamic_shared_bytes\tblock\tgrid\toccupancy_pct",
        )
        for report in kernel_stats.reports
            println(
                io,
                join(
                    (
                        report.name,
                        report.occurrences,
                        report.total_duration_ns,
                        report.total_duration_ns ÷ report.occurrences,
                        report.registers_per_thread,
                        report.static_shmem_bytes,
                        report.dynamic_shmem_bytes,
                        join(report.block_dim, 'x'),
                        join(report.grid_dim, 'x'),
                        report.occupancy_pct,
                    ),
                    '\t',
                ),
            )
        end
    end
    return kernel_stats, framework_stats
end

function profile_k8(; N, samples, warmup, diff_batch, profile_dir)
    K = 8
    problem = brusselator_problem(N)
    state = split_state(problem.u)
    seeds = make_tangent_seeds(state, K; kind=:onehot)
    outputs = ntuple(_ -> zero_state(state), K)
    args = Reactant.to_rarray((outputs, state, seeds, problem.coordinates, problem.p))
    compile_options = brusselator_compile_options(diff_batch)
    compiled, compile_seconds = compile_timed(chunk_function(K), args, compile_options)

    for _ in 1:warmup
        compiled(args...)
    end
    timings = steady_timings(compiled, args, samples)

    @printf(
        "K=8 N=%d diff_batch=%s compile=%.6f s median=%.6f ms minimum=%.6f ms\n",
        N,
        string(diff_batch),
        compile_seconds,
        1.0e3 * timings.median,
        1.0e3 * timings.minimum,
    )

    if !isempty(profile_dir)
        mkpath(profile_dir)
        profile = Reactant.Profiler.profile_and_get_xplane_file(
            compiled,
            args...;
            nrepeat=samples,
            warmup=warmup,
            profile_dir,
        )
        kernel_stats, framework_stats = save_profile_reports(
            profile_dir, profile.xplane_file
        )
        println("xplane: ", profile.xplane_file)
        println("kernel reports: ", length(kernel_stats.reports))
        println("framework reports: ", length(framework_stats))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    profile_k8(; profile_options(ARGS)...)
end
