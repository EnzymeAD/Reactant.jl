using Reactant, SafeTestsets, Test

const REACTANT_BACKEND_GROUP = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "all"))

if REACTANT_BACKEND_GROUP == "gpu"
    Reactant.set_default_backend("gpu")
elseif REACTANT_BACKEND_GROUP == "tpu"
    Reactant.set_default_backend("tpu")
elseif REACTANT_BACKEND_GROUP == "cpu"
    Reactant.set_default_backend("cpu")
end

@testset "Reactant.jl Tests" begin
    @testset "Core" begin
        @safetestset "Layout" include("core/layout.jl")
        @safetestset "Tracing" include("core/tracing.jl")
        @safetestset "Basic" include("core/basic.jl")
        @safetestset "Constructor" include("core/constructor.jl")
        @safetestset "Autodiff" include("core/autodiff.jl")
        @safetestset "Complex" include("core/complex.jl")
        @safetestset "Broadcast" include("core/bcast.jl")
        @safetestset "Struct" include("core/struct.jl")
        @safetestset "Closure" include("core/closure.jl")
        @safetestset "Compile" include("core/compile.jl")
        @safetestset "IR" include("core/ir.jl")
        @safetestset "Buffer Donation" include("core/buffer_donation.jl")
        @safetestset "Wrapped Arrays" include("core/wrapped_arrays.jl")
        @safetestset "Control Flow" include("core/control_flow.jl")
        @safetestset "Sorting" include("core/sorting.jl")
        @safetestset "Shortcuts to MLIR ops" include("core/ops.jl")
        @safetestset "Indexing" include("core/indexing.jl")
        @safetestset "Config" include("core/config.jl")
        @safetestset "Batching" include("core/batching.jl")

        if !Sys.isapple()
            @safetestset "Custom Number Types" include("core/custom_number_types.jl")
        end
    end

    @testset "Distributed" begin
        @safetestset "Sharding" include("distributed/sharding.jl")
        @safetestset "Comm Optimization" include("distributed/optimize_comm.jl")
        @safetestset "Cluster Detection" include("distributed/cluster_detector.jl")
    end

    @testset "Plugins" begin
        if Sys.isapple() && haskey(Reactant.XLA.global_backend_state.clients, "metal")
            @safetestset "Metal Plugin" include("plugins/metal.jl")
        end
    end

    @testset "Standard Libraries" begin
        @safetestset "Linear Algebra" include("stdlibs/linear_algebra.jl")
        @safetestset "Random" include("stdlibs/random.jl")
    end

    @testset "Common Integrations" begin
        # most integrations are tested in the integration tests except deps that are
        # very common
        @safetestset "SpecialFunctions" include("common_integration/special_functions.jl")
    end

    @testset "Quality Assurance" begin
        @safetestset "QA" include("core/qa.jl")
    end
end
