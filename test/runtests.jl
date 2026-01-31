using Reactant, ParallelTestRunner, CondaPkg, Test

const BACKEND = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "auto"))

parsed_args = parse_args(ARGS)

const ENZYMEJAX_INSTALLED = Ref(false)
const NUMPYRO_INSTALLED = Ref(false)

# Install specific packages. Pkg.test doesn't pick up CondaPkg.toml in test folder
if (
    isempty(parsed_args.positionals) ||
    "integration" ∈ parsed_args.positionals ||
    "integration/enzymejax" ∈ parsed_args.positionals
)
    CondaPkg.add_pip("jax"; version="==0.5")
    try
        CondaPkg.add_pip("enzyme_ad"; version=">=0.0.9")
        ENZYMEJAX_INSTALLED[] = true
    catch
    end
end

if (
    isempty(parsed_args.positionals) ||
    "integration" ∈ parsed_args.positionals ||
    "integration/numpyro" ∈ parsed_args.positionals
)
    try
        CondaPkg.add_pip("numpyro")
        NUMPYRO_INSTALLED[] = true
    catch
    end
end

testsuite = find_tests(@__DIR__)

filter_tests!(testsuite, parsed_args)

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

if !NUMPYRO_INSTALLED[]
    delete!(testsuite, "integration/numpyro")
end

total_jobs = min(
    something(parsed_args.jobs, ParallelTestRunner.default_njobs()), length(keys(testsuite))
)

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

    if (
        isempty(parsed_args.positionals) ||
        "integration" ∈ parsed_args.positionals ||
        "integration/mpi" ∈ parsed_args.positionals
    )
        @testset "MPI" begin
            using MPI
            nranks = 2
            run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) integration/mpi.jl`)
        end
    end
end
