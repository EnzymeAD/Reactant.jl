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
            @test length(kernel_stats) > 0
        end

        framework_stats = Reactant.Profiler.get_framework_op_stats(file)
        if !RunningOnCPU
            @test length(framework_stats) > 0
        end

        flops = Reactant.Profiler.get_aggregate_flops_statistics(file, 1)
        if !RunningOnCPU
            @test flops isa Reactant.Profiler.FlopsSummary
            @test flops.RawFlopsRate > 0
            @test flops.BF16FlopsRate > 0
            @test flops.RawFlopsRate ≈ flops.RawFlops / (flops.RawTime * 1e-12)
            @test flops.BF16FlopsRate ≈ flops.BF16Flops / (flops.RawTime * 1e-12)
        end

        Reactant.@timed nrepeat = 32 linear(x, W, b)
        Reactant.@time nrepeat = 32 linear(x, W, b)
        Reactant.@profile nrepeat = 32 linear(x, W, b)
        Reactant.@profile nrepeat = 32 compile_options = Reactant.DefaultXLACompileOptions() linear(
            x, W, b
        )
    end
end
