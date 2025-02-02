using Reactant, SafeTestsets, Test

@testset "Reactant.jl Tests" begin
    if REACTANT_TEST_GROUP == "all" || REACTANT_TEST_GROUP == "core"
        @safetestset "Layout" include("layout.jl")
        @safetestset "Tracing" include("tracing.jl")
        @safetestset "Basic" include("basic.jl")
        @safetestset "Autodiff" include("autodiff.jl")
        @safetestset "Complex" include("complex.jl")
        @safetestset "Broadcast" include("bcast.jl")
        @safetestset "Struct" include("struct.jl")
        @safetestset "Closure" include("closure.jl")
        @safetestset "Compile" include("compile.jl")
        @safetestset "Buffer Donation" include("buffer_donation.jl")
        @safetestset "Shortcuts to MLIR ops" include("ops.jl")
        @safetestset "Wrapped Arrays" include("wrapped_arrays.jl")
        @safetestset "Control Flow" include("control_flow.jl")
        @safetestset "Sorting" include("sorting.jl")
        @safetestset "Indexing" include("indexing.jl")
        if !Sys.isapple()
            @safetestset "Custom Number Types" include("custom_number_types.jl")
        end
        @safetestset "Sharding" include("sharding.jl")
    end

    if REACTANT_TEST_GROUP == "all" || REACTANT_TEST_GROUP == "integration"
        @safetestset "CUDA" include("integration/cuda.jl")
        @safetestset "KernelAbstractions" include("integration/kernelabstractions.jl")
        @safetestset "Linear Algebra" include("integration/linear_algebra.jl")
        @safetestset "OffsetArrays" include("integration/offsetarrays.jl")
        @safetestset "AbstractFFTs" include("integration/fft.jl")
        @safetestset "SpecialFunctions" include("integration/special_functions.jl")
        @safetestset "Random" include("integration/random.jl")
        @safetestset "Python" include("integration/python.jl")
    end

    if REACTANT_TEST_GROUP == "all" || REACTANT_TEST_GROUP == "neural_networks"
        @testset "Neural Networks" begin
            @safetestset "NNlib Primitives" include("nn/nnlib.jl")
            @safetestset "Flux.jl Integration" include("nn/flux.jl")
            if Sys.islinux()
                @safetestset "LuxLib Primitives" include("nn/luxlib.jl")
                @safetestset "Lux Integration" include("nn/lux.jl")
            end
        end
    end
end
