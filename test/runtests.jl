using Reactant, ParallelTestRunner, CondaPkg, Test

const BACKEND = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "auto"))
BACKEND != "auto" && Reactant.set_default_backend(BACKEND)

const REACTANT_TEST_GROUP = lowercase(get(ENV, "REACTANT_TEST_GROUP", "all"))
@assert REACTANT_TEST_GROUP ∈ ("all", "core", "integration", "nn")

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

if REACTANT_TEST_GROUP == "core"
    testsuite = merge(find_tests("plugins/"), find_tests("core/"))
elseif REACTANT_TEST_GROUP == "integration"
    testsuite = find_tests("integration/")
elseif REACTANT_TEST_GROUP == "nn"
    testsuite = find_tests("nn/")
else
    testsuite = find_tests(@__DIR__)
end

if !(Sys.isapple() && haskey(Reactant.XLA.global_backend_state.clients, "metal"))
    delete!(testsuite, "plugins/metal")
end
if Sys.isapple()
    delete!(testsuite, "core/custom_number_types")
end

# Zygote is not supported on 1.12 https://github.com/FluxML/Zygote.jl/issues/1580
if VERSION ≥ v"1.12-"
    delete!(testsuite, "integration/zygote")
end

# This is run in a special way
delete!(testsuite, "integration/mpi")

runtests(Reactant, ARGS; testsuite)

if REACTANT_TEST_GROUP == "integration" || REACTANT_TEST_GROUP == "all"
    @testset "MPI" begin
        using MPI
        nranks = 2
        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) integration/mpi.jl`)
    end
end
