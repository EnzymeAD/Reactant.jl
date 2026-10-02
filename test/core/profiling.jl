using Reactant, FileCheck

const RunningOnCPU = contains(string(Reactant.devices()[1]), "CPU")
const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

x = Reactant.to_rarray(randn(Float32, 100, 2))
W = Reactant.to_rarray(randn(Float32, 10, 100))
b = Reactant.to_rarray(randn(Float32, 10))

linear(x, W, b) = (W * x) .+ b

Reactant.Profiler.@annotate function annotated_add_one(x)
    return x .+ 1
end

function trace_ends_in_batch(x)
    id = Reactant.Profiler.profiler_activity_start(
        "trace_ends_in_batch", Reactant.Profiler.TRACE_ME_LEVEL_CRITICAL
    )
    return Reactant.Ops.batch(x, [1]) do y
        Reactant.Profiler.profiler_activity_end(id)
        y .+ 1
    end
end

@testset "Profiling" begin
    profiler_hlo = Reactant.Profiler.ScopedValues.with(
        Reactant.Profiler.ENABLE_RUNTIME_TRACING => true
    ) do
        repr(@code_hlo optimize = false annotated_add_one(x))
    end
    @test @filecheck begin
        @check_dag "ProfilerActivityStart"
        @check_dag "ProfilerActivityEnd"
        @check "enzymexla.jit_call"
        @check "enzyme.init"
        @check "enzyme.push"
        @check "enzyme.pop"
        @check "enzymexla.jit_call"
        @check_not "enzymexla.jit_call"
        profiler_hlo
    end
    @test Array(@jit annotated_add_one(x)) ≈ Array(x) .+ 1

    batched_profiler_hlo = @test_warn r"start and end are in a different function body" Reactant.Profiler.ScopedValues.with(
        Reactant.Profiler.ENABLE_RUNTIME_TRACING => true
    ) do
        repr(@code_hlo optimize = false trace_ends_in_batch(x))
    end
    @test @filecheck begin
        @check_not "enzymexla.jit_call"
        @check_not "enzyme.init"
        @check_not "enzyme.push"
        @check_not "enzyme.pop"
        batched_profiler_hlo
    end

    disabled_profiler_hlo = Reactant.Profiler.ScopedValues.with(
        Reactant.Profiler.ENABLE_RUNTIME_TRACING => false
    ) do
        repr(@code_hlo optimize = false annotated_add_one(x))
    end
    @test @filecheck begin
        @check_not "ProfilerActivityStart"
        @check_not "ProfilerActivityEnd"
        @check_not "enzymexla.jit_call"
        @check_not "enzyme.init"
        @check_not "enzyme.push"
        @check_not "enzyme.pop"
        disabled_profiler_hlo
    end

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
