using Reactant, Test

const RunningOnCPU = contains(string(Reactant.devices()[1]), "CPU")
const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

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
        if !RunningOnCPU
            @test length(framework_stats) > 0
        end

        metrics = Reactant.Profiler.get_aggregate_metrics(file, 1)
        if !RunningOnCPU
            @test metrics isa Reactant.Proto.tensorflow.profiler.op_profile.Metrics
            @test metrics.raw_flops_rate > 0
            @test metrics.bf16_flops_rate > 0
            @test metrics.raw_flops_rate ≈ metrics.raw_flops / (metrics.raw_time * 1e-12)
            @test metrics.bf16_flops_rate ≈ metrics.bf16_flops / (metrics.raw_time * 1e-12)
        end

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
        fn, timings = Reactant.Profiler.@timed_compile linear(x, W, b)
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
