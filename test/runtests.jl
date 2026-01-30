using Reactant, ParallelTestRunner, CondaPkg, Test

const BACKEND = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "auto"))

const REACTANT_TEST_GROUP = lowercase(get(ENV, "REACTANT_TEST_GROUP", "all"))
@assert REACTANT_TEST_GROUP ∈ ("all", "core", "integration", "neural_networks")

const ENZYMEJAX_INSTALLED = Ref(false)

# Install specific packages. Pkg.test doesn't pick up CondaPkg.toml in test folder
if REACTANT_TEST_GROUP == "all" || REACTANT_TEST_GROUP == "integration"
    CondaPkg.add_pip("jax"; version="==0.5")
    try
        CondaPkg.add_pip("enzyme_ad"; version=">=0.0.9")
        ENZYMEJAX_INSTALLED[] = true
    catch
    end
end

testsuite = find_tests(@__DIR__)

if REACTANT_TEST_GROUP == "core"
    for k in keys(testsuite)
        !(startswith(k, "core/") || startswith(k, "plugins/")) && delete!(testsuite, k)
    end
elseif REACTANT_TEST_GROUP == "integration"
    for k in keys(testsuite)
        !startswith(k, "integration/") && delete!(testsuite, k)
    end
elseif REACTANT_TEST_GROUP == "neural_networks"
    for k in keys(testsuite)
        !startswith(k, "nn/") && delete!(testsuite, k)
    end
end

delete!(testsuite, "plugins/metal") # Currently completely non functional

if Sys.isapple()
    delete!(testsuite, "core/custom_number_types")
end

# Zygote is not supported on 1.12 https://github.com/FluxML/Zygote.jl/issues/1580
if VERSION ≥ v"1.12-"
    delete!(testsuite, "integration/zygote")
end

# This is run in a special way
delete!(testsuite, "integration/mpi")

if !ENZYMEJAX_INSTALLED[]
    delete!(testsuite, "integration/enzymejax")
end

total_jobs = min(ParallelTestRunner.default_njobs(), length(keys(testsuite)))

@testset "Reactant Tests" begin
    withenv(
        "XLA_REACTANT_GPU_MEM_FRACTION" => 1 / (total_jobs + 0.1),
        "XLA_REACTANT_GPU_PREALLOCATE" => false,
    ) do
        runtests(
            Reactant,
            ARGS;
            testsuite,
            init_code=quote
                using Reactant
                $(BACKEND) != "auto" && Reactant.set_default_backend($(BACKEND))
            end,
        )
    end

    if REACTANT_TEST_GROUP == "integration" || REACTANT_TEST_GROUP == "all"
        @testset "MPI" begin
            using MPI
            nranks = 2
            run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) integration/mpi.jl`)
        end
    end
end
