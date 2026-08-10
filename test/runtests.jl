using Reactant, ParallelTestRunner, CondaPkg, Test, Setfield

const BACKEND = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "auto"))

parsed_args = parse_args(ARGS)

const ENZYMEJAX_INSTALLED = Ref(false)
const NUMPYRO_INSTALLED = Ref(false)

function is_python_module_installed(mod)
    python = CondaPkg.which("python")
    python === nothing && return false
    return success(pipeline(`$(python) -c "import $(mod)"`; stdout=devnull, stderr=devnull))
end

# `CondaPkg.add_pip` can silently do nothing (it skips resolving based on file timestamps),
# so check that the module is actually importable instead of trusting that no error was
# thrown. Otherwise the corresponding test is run against a missing python package.
function add_pip_and_check(pkg, mod; version="")
    msg = "Could not install the python package `$(pkg)`, skipping the tests using it"
    try
        CondaPkg.add_pip(pkg; version)
        is_python_module_installed(mod) && return true
        @warn "`import $(mod)` failed after adding `$(pkg)`, forcing a re-resolve"
        CondaPkg.resolve(; force=true)
        is_python_module_installed(mod) && return true
        @warn msg
        return false
    catch e
        @warn msg exception = e
        return false
    end
end

# Install specific packages. Pkg.test doesn't pick up CondaPkg.toml in test folder
if (
    isempty(parsed_args.positionals) ||
    "integration" ∈ parsed_args.positionals ||
    "integration/enzymejax" ∈ parsed_args.positionals
)
    CondaPkg.add_pip("jax"; version=">=0.9")
    if !Sys.iswindows() && !(Sys.isapple() && Sys.ARCH === :x86_64)
        ENZYMEJAX_INSTALLED[] = add_pip_and_check(
            "enzyme_ad", "enzyme_ad"; version=">=0.0.15"
        )
    end
end

if (
    isempty(parsed_args.positionals) ||
    "integration" ∈ parsed_args.positionals ||
    "integration/numpyro" ∈ parsed_args.positionals
)
    NUMPYRO_INSTALLED[] = add_pip_and_check("numpyro", "numpyro"; version=">=0.21")
end

testsuite = find_tests(@__DIR__)

filter_tests!(testsuite, parsed_args)

delete!(testsuite, "test_helpers")

delete!(testsuite, "plugins/metal") # Currently completely non functional

if Sys.isapple()
    delete!(testsuite, "core/custom_number_types")
end

if Sys.iswindows() || (Sys.isapple() && Sys.ARCH === :x86_64)
    delete!(testsuite, "integration/enzymejax")
end

# Zygote is not supported on 1.12 https://github.com/FluxML/Zygote.jl/issues/1580
if VERSION ≥ v"1.12-"
    delete!(testsuite, "integration/zygote")
end

# This is run in a special way
delete!(testsuite, "integration/mpi")
delete!(testsuite, "core/qa")
delete!(testsuite, "core/sharding/sharding")
delete!(testsuite, "core/sharding/optimize_comm")

# ProbProg tests require a compatible combination of NumPyro and JAX.
filter!(((k, _),) -> !startswith(k, "probprog"), testsuite)

if !ENZYMEJAX_INSTALLED[]
    delete!(testsuite, "integration/enzymejax")
end

if !NUMPYRO_INSTALLED[]
    delete!(testsuite, "integration/numpyro")
end

include("test_helpers.jl")

custom_test_worker = false

if NTPUs > 0 || BACKEND == "tpu"
    custom_test_worker = true
    @set! parsed_args.jobs = Some(NTPUs)
end

total_jobs = min(
    something(parsed_args.jobs, ParallelTestRunner.default_njobs()), length(keys(testsuite))
)

test_worker = custom_test_worker ? tpu_custom_worker_launcher : Returns(nothing)

@testset "Reactant Tests" begin
    withenv(
        "XLA_REACTANT_GPU_MEM_FRACTION" => 1 / (total_jobs + 0.1),
        "XLA_REACTANT_GPU_PREALLOCATE" => false,
    ) do
        runtests(
            Reactant,
            parsed_args;
            testsuite,
            test_worker,
            init_code=quote
                using Reactant
                $(BACKEND) != "auto" && Reactant.set_default_backend($(BACKEND))
            end,
        )
    end

    if (
        BACKEND in ("auto", "cpu", "cuda", "gpu") && (
            isempty(parsed_args.positionals) ||
            "integration" ∈ parsed_args.positionals ||
            "integration/mpi" ∈ parsed_args.positionals
        )
    )
        @testset "MPI" begin
            using MPI
            nranks = 2
            run(
                `$(mpiexec()) -n $nranks $(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(@__DIR__, "integration", "mpi.jl"))`,
            )
        end
    end

    if (isempty(parsed_args.positionals) || "probprog" ∈ parsed_args.positionals)
        @testset "ProbProg" begin
            run(
                `$(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(@__DIR__, "probprog", "runtests.jl"))`,
            )
        end
    end

    # Important to run these last, since they will capture the TPU and any
    # processes triggered in multi-process tests will not be able to use TPUs
    # These tests require multiple devices
    run_specific_test("core/sharding/sharding", parsed_args)
    run_specific_test("core/sharding/optimize_comm", parsed_args)

    # qa tests run a separate process
    if BACKEND == "cpu" || BACKEND == "auto"
        run_specific_test("core/qa", parsed_args)
    end
end
