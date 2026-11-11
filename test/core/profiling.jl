using Reactant, Test

const RunningOnCPU = Reactant.XLA.platform_name(Reactant.XLA.default_backend()) == "cpu"
const RunningOnCUDA = Reactant.XLA.platform_name(Reactant.XLA.default_backend()) == "cuda"

x = Reactant.to_rarray(randn(Float32, 100, 2))
W = Reactant.to_rarray(randn(Float32, 10, 100))
b = Reactant.to_rarray(randn(Float32, 10))

linear(x, W, b) = (W * x) .+ b

@testset "Profiling" begin
    # Run the profiling/timing tools and print
    if !Sys.iswindows()
        fn = @compile linear(x, W, b)
        @test_throws AssertionError Reactant.Profiler.profile_and_get_xplane_file(
            fn, x, W, b; nrepeat=10
        )

        fn = @compile sync = true linear(x, W, b)
        file =
            Reactant.Profiler.profile_and_get_xplane_file(
                fn, x, W, b; nrepeat=10
            ).xplane_file
        @test isfile(file)

        kernel_stats = Reactant.Profiler.get_kernel_stats(file)
        if RunningOnCUDA
            @test length(kernel_stats.reports) > 0
        end

        framework_stats = Reactant.Profiler.get_framework_op_stats(file)
        @test length(framework_stats) > 0

        metrics = Reactant.Profiler.get_aggregate_metrics(file, 10)
        @test metrics isa Reactant.Proto.tensorflow.profiler.op_profile.Metrics
        @test metrics.raw_time > 0
        @test metrics.raw_flops_rate > 0
        @test metrics.bf16_flops_rate > 0
        @test isfinite(metrics.raw_flops_rate)
        @test isfinite(metrics.bf16_flops_rate)
        @test metrics.raw_flops_rate ≈ metrics.raw_flops / (metrics.raw_time * 1e-12)
        @test metrics.bf16_flops_rate ≈ metrics.bf16_flops / (metrics.raw_time * 1e-12)
        if RunningOnCPU
            expected_step_flops = Reactant.XLA.cost_analysis(fn).flops
            @test metrics.raw_flops == 10 * expected_step_flops
            @test metrics.occurrences == 10
        end

        # Zero-FLOP and zero-time edge cases must never produce NaN rates
        zero_metrics = Reactant.Proto.tensorflow.profiler.op_profile.Metrics(
            0.0, 0.0, Float64[], 0.0, 0.0, 0.0, 0.0, Float64[], UInt32(0), 0.0
        )
        @test zero_metrics.raw_flops_rate == 0.0
        @test zero_metrics.bf16_flops_rate == 0.0

        noop_fn = @compile sync = true identity(x)
        noop_prof = Reactant.Profiler.profile_with_xprof(noop_fn, x; nrepeat=4, warmup=1)
        @test noop_prof.profiling_result.runtime_ns > 0
        if noop_prof.profiling_result.metrics_data !== nothing
            @test isfinite(noop_prof.profiling_result.metrics_data.raw_flops_rate)
            @test isfinite(noop_prof.profiling_result.metrics_data.bf16_flops_rate)
        end

        function loop_fma(a)
            acc = copy(a)
            @trace for _ in 1:5
                acc = muladd.(acc, a, a)
            end
            return acc
        end
        loop_fn = @compile sync = true loop_fma(x)
        loop_prof = Reactant.Profiler.profile_with_xprof(loop_fn, x; nrepeat=3, warmup=1)
        @test loop_prof.profiling_result.metrics_data !== nothing
        @test loop_prof.profiling_result.metrics_data.raw_time > 0
        @test loop_prof.profiling_result.metrics_data.raw_flops >= 3 * 5 * 2 * length(x)
        @test isfinite(loop_prof.profiling_result.metrics_data.raw_flops_rate)
        @test loop_prof.profiling_result.metrics_data.raw_flops_rate > 0

        big_x = Reactant.to_rarray(ones(Float32, 256, 256))
        big_loop_fn = @compile sync = true loop_fma(big_x)
        big_loop_prof = Reactant.Profiler.profile_with_xprof(
            big_loop_fn, big_x; nrepeat=2, warmup=1
        )
        @test big_loop_prof.profiling_result.metrics_data !== nothing
        @test big_loop_prof.profiling_result.metrics_data.raw_time > 0
        @test big_loop_prof.profiling_result.metrics_data.raw_flops >=
            2 * 5 * 2 * length(big_x)
        @test isfinite(big_loop_prof.profiling_result.metrics_data.raw_flops_rate)
        @test big_loop_prof.profiling_result.metrics_data.raw_flops_rate > 0

        Reactant.@timed nrepeat = 32 linear(x, W, b)
        Reactant.@time nrepeat = 32 linear(x, W, b)
        Reactant.@profile nrepeat = 32 linear(x, W, b)
        Reactant.@profile nrepeat = 32 compile_options = Reactant.DefaultXLACompileOptions() linear(
            x, W, b
        )
    end
end

@testset "Compile timings" begin
    if !Sys.iswindows()
        # At the default precision a TPU does this Float32 matmul in bfloat16, and
        # the result is off by ~1e-3 relative; the test is about the compile
        # timings, not the accelerator's default precision.
        fn, timings = Reactant.with_config(;
            dot_general_precision=Reactant.PrecisionConfig.HIGHEST
        ) do
            Reactant.Profiler.@timed_compile linear(x, W, b)
        end
        @test Array(fn(x, W, b)) ≈ linear(Array(x), Array(W), Array(b))

        @test timings isa Reactant.Profiler.CompileTimings
        @test timings.total_compile_time_μs > 0
        @test timings.tracing_time_μs > 0
        @test timings.xla_time_μs > 0
        @test timings.tracing_time_μs <= timings.total_compile_time_μs
        @test timings.mlir_time_μs <= timings.total_compile_time_μs
        @test timings.xla_time_μs <= timings.total_compile_time_μs
        @test !isempty(timings.traces)
    end

    # Lines (threads) sharing a name must not overwrite each other
    XPB = Reactant.Proto.tensorflow.profiler
    event(duration_ps) =
        XPB.XEvent(1, XPB.OneOf(:offset_ps, Int64(0)), duration_ps, XPB.XStat[])
    line(id, duration_ps) =
        XPB.XLine(id, id, "worker", "", 0, duration_ps, [event(duration_ps)])
    plane = XPB.XPlane(
        1,
        "/host:CPU",
        [line(1, 1_000_000), line(2, 2_000_000)],
        Dict(1 => XPB.XEventMetadata(1, "compile f", "", UInt8[], XPB.XStat[], Int64[])),
        Dict{Int64,XPB.XStatMetadata}(),
        XPB.XStat[],
    )
    traces = Reactant.Profiler.trace_trees(
        XPB.XSpace([plane], String[], String[], String[])
    )
    @test length(traces["/host:CPU"]) == 2
    @test Reactant.Profiler._total_duration(traces, "compile ") == 3
end
