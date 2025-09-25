using Reactant, SafeTestsets, Test

if lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "all")) == "gpu"
    Reactant.set_default_backend("gpu")
end

const REACTANT_TEST_GROUP = lowercase(get(ENV, "REACTANT_TEST_GROUP", "all"))

@testset "Reactant.jl Tests" begin
    if REACTANT_TEST_GROUP == "all" || REACTANT_TEST_GROUP == "core"
        if Sys.isapple() && haskey(Reactant.XLA.global_backend_state.clients, "metal")
            @safetestset "Metal Plugin" include("plugins/metal.jl")
        end

        @safetestset "Layout" include("layout.jl")
        @safetestset "Tracing" include("tracing.jl")
        @safetestset "Basic" include("basic.jl")
        @safetestset "Constructor" include("constructor.jl")
        @safetestset "Autodiff" include("autodiff.jl")
        @safetestset "Complex" include("complex.jl")
        @safetestset "Broadcast" include("bcast.jl")
        @safetestset "Struct" include("struct.jl")
        @safetestset "Closure" include("closure.jl")
        @safetestset "Compile" include("compile.jl")
        @safetestset "IR" include("ir.jl")
        @safetestset "Buffer Donation" include("buffer_donation.jl")
        @safetestset "Wrapped Arrays" include("wrapped_arrays.jl")
        @safetestset "Control Flow" include("control_flow.jl")
        @safetestset "Sorting" include("sorting.jl")
        @safetestset "Shortcuts to MLIR ops" include("ops.jl")
        @safetestset "Indexing" include("indexing.jl")
        @safetestset "Ranges" include("ranges.jl")
        if !Sys.isapple()
            @safetestset "Custom Number Types" include("custom_number_types.jl")
        end
        @safetestset "Sharding" include("sharding.jl")
        @safetestset "Comm Optimization" include("optimize_comm.jl")
        @safetestset "Cluster Detection" include("cluster_detector.jl")
        @safetestset "Config" include("config.jl")
        @safetestset "Batching" include("batching.jl")
        @safetestset "QA" include("qa.jl")
    end

    if REACTANT_TEST_GROUP == "all" || REACTANT_TEST_GROUP == "integration"
        @safetestset "CUDA" include("integration/cuda.jl")
        @safetestset "KernelAbstractions" include("integration/kernelabstractions.jl")
        @safetestset "Linear Algebra" include("integration/linear_algebra.jl")
        @safetestset "OffsetArrays" include("integration/offsetarrays.jl")
        @safetestset "OneHotArrays" include("integration/onehotarrays.jl")
        @safetestset "AbstractFFTs" include("integration/fft.jl")
        @safetestset "SpecialFunctions" include("integration/special_functions.jl")
        @safetestset "Random" include("integration/random.jl")
        @safetestset "Python" include("integration/python.jl")
        @safetestset "Optimisers" include("integration/optimisers.jl")
        @safetestset "FillArrays" include("integration/fillarrays.jl")
        @safetestset "MPI" begin
            using MPI
            nranks = 2
            run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) integration/mpi.jl`)
        end

        # Zygote is not supported on 1.12 https://github.com/FluxML/Zygote.jl/issues/1580
        if VERSION < v"1.12-"
            @safetestset "Zygote" include("integration/zygote.jl")
        end

        @testset "Triton" begin
            @safetestset "vector_add" include("integration/triton/vector_add.jl")
            @safetestset "softmax" include("integration/triton/softmax.jl")
            # @safetestset "matmul" include("integration/triton/matmul.jl") # XXX
            @safetestset "low_memory_dropout" include(
                "integration/triton/low_memory_dropout.jl"
            )
            @safetestset "layer norm" include("integration/triton/layer_norm.jl")
            # @safetestset "attention" include("integration/triton/attention.jl")
            @safetestset "libdevice" include("integration/triton/libdevice.jl")
            # @safetestset "grouped gemm" include("integration/triton/grouped_gemm.jl")
            # @safetestset "persistant matmul" include(
            #     "integration/triton/persistant_matmul.jl"
            # )
            # @safetestset "block scaled matmul" include(
            #     "integration/triton/block_scaled_matmul.jl"
            # )
        end
    end

    if REACTANT_TEST_GROUP == "all" || REACTANT_TEST_GROUP == "neural_networks"
        @safetestset "NNlib Primitives" include("nn/nnlib.jl")
        @safetestset "Flux.jl Integration" include("nn/flux.jl")
        @safetestset "LuxLib Primitives" include("nn/luxlib.jl")
        @safetestset "Lux Integration" include("nn/lux.jl")
    end
end
