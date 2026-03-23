using Reactant

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

@testset "Advanced config profiling" begin
    if !Sys.iswindows()
        fn = @compile sync = true linear(x, W, b)

        # Test with_profiler accepting advanced_config dict
        mktempdir() do profile_dir
            Reactant.Profiler.with_profiler(
                profile_dir; advanced_config=Dict{String,String}()
            ) do
                fn(x, W, b)
            end
            traces_path = joinpath(profile_dir, "plugins", "profile")
            @test isdir(traces_path)
        end

        # Test profile_and_get_xplane_file with advanced_config
        result = Reactant.Profiler.profile_and_get_xplane_file(
            fn, x, W, b; nrepeat=3, advanced_config=Dict{String,String}()
        )
        @test isfile(result.xplane_file)

        if RunningOnCUDA
            # Test with PM counters on GPU
            # PM counters may fail due to permissions (CUPTI_ERROR_INSUFFICIENT_PRIVILEGES),
            # so we test that the API accepts them without error. The profiler gracefully
            # degrades if permissions are missing.
            mktempdir() do profile_dir
                Reactant.Profiler.with_profiler(
                    profile_dir;
                    pm_counters=Reactant.Profiler.default_pm_counters(),
                ) do
                    fn(x, W, b)
                end
                traces_path = joinpath(profile_dir, "plugins", "profile")
                @test isdir(traces_path)
            end

            # Test profile_and_get_xplane_file with pm_counters
            pm_result = Reactant.Profiler.profile_and_get_xplane_file(
                fn, x, W, b;
                nrepeat=3,
                pm_counters=Reactant.Profiler.default_pm_counters(),
            )
            @test isfile(pm_result.xplane_file)

            # framework_op_stats may be empty if CUPTI permissions are restricted
            pm_stats = Reactant.Profiler.get_framework_op_stats(pm_result.xplane_file)
            @test pm_stats isa Vector
        end
    end
end
