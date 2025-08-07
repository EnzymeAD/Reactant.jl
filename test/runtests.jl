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
        @info "Layout tests finished"
        @safetestset "Tracing" include("tracing.jl")
        @info "Tracing tests finished"
        @safetestset "Basic" include("basic.jl")
        @info "Basic tests finished"
        @safetestset "Constructor" include("constructor.jl")
        @info "Constructor tests finished"
        @safetestset "Autodiff" include("autodiff.jl")
        @info "Autodiff tests finished"
        @safetestset "Complex" include("complex.jl")
        @info "Complex tests finished"
        @safetestset "Broadcast" include("bcast.jl")
        @info "Broadcast tests finished"
        @safetestset "Struct" include("struct.jl")
        @info "Struct tests finished"
        @safetestset "Closure" include("closure.jl")
        @info "Closure tests finished"
        @safetestset "Compile" include("compile.jl")
        @info "Compile tests finished"
        @safetestset "IR" include("ir.jl")
        @info "IR tests finished"
        @safetestset "Buffer Donation" include("buffer_donation.jl")
        @info "Buffer Donation tests finished"
        @safetestset "Shortcuts to MLIR ops" include("ops.jl")
        @info "Shortcuts to MLIR ops tests finished"
        @safetestset "Wrapped Arrays" include("wrapped_arrays.jl")
        @info "Wrapped Arrays tests finished"
        @safetestset "Control Flow" include("control_flow.jl")
        @info "Control Flow tests finished"
        @safetestset "Sorting" include("sorting.jl")
        @info "Sorting tests finished"
        @safetestset "Indexing" include("indexing.jl")
        @info "Indexing tests finished"
        if !Sys.isapple()
            @safetestset "Custom Number Types" include("custom_number_types.jl")
            @info "Custom Number Types tests finished"
        end
        @safetestset "Sharding" include("sharding.jl")
        @info "Sharding tests finished"
        @safetestset "Comm Optimization" include("optimize_comm.jl")
        @info "Comm Optimization tests finished"
        @safetestset "Cluster Detection" include("cluster_detector.jl")
        @info "Cluster Detection tests finished"
        @safetestset "Config" include("config.jl")
        @info "Config tests finished"
        @safetestset "Batching" include("batching.jl")
        @info "Batching tests finished"
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
    end

    if REACTANT_TEST_GROUP == "all" || REACTANT_TEST_GROUP == "neural_networks"
        @safetestset "NNlib Primitives" include("nn/nnlib.jl")
        @safetestset "Flux.jl Integration" include("nn/flux.jl")
        if Sys.islinux()
            @safetestset "LuxLib Primitives" include("nn/luxlib.jl")
            @safetestset "Lux Integration" include("nn/lux.jl")
        end
    end
end
