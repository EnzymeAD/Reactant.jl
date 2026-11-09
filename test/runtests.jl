using Reactant, ParallelTestRunner, CondaPkg, Test, Setfield

const BACKEND = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "auto"))

parsed_args = parse_args(ARGS)

function active_python_exe()
    exe = get(ENV, "JULIA_PYTHONCALL_EXE", "")
    !isempty(exe) && return exe
    if CondaPkg.backend() !== :Null
        return CondaPkg.which("python")
    end
    return Sys.which("python")
end

function is_python_module_installed(mod)
    python = active_python_exe()
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

# The julia PJRT backend (`src/accelerators/Julia.jl`) runs on a single CPU device and only supports the StableHLO ops
# with a `handle` method, so tests that need anything else are skipped
if BACKEND == "julia"
    for name in (
        # `enzymexla.lapack`/`enzymexla.linalg` ops (`lu`, `inv`, `\`, `det`, `svd`, ...)
        "integration/factorizations",
        "integration/linear_algebra",
        "integration/linearsolve",
        # custom calls (Julia callbacks, kernels, Python)
        "core/julia_callback",
        "integration/cuda",
        "integration/kernelabstractions",
        "integration/nfft_callback",
        "integration/enzymejax",
        # profiler and allocator statistics are not implemented
        "core/profiling",
        "core/compile",
    )
        delete!(testsuite, name)
    end
    filter!(((k, _),) -> !startswith(k, "integration/triton"), testsuite)
end

const RUN_PYTHON_TEST =
    haskey(testsuite, "integration/python") && !(Sys.isapple() && Sys.ARCH === :x86_64)
const RUN_ENZYMEJAX_TEST = haskey(testsuite, "integration/enzymejax")
const RUN_NUMPYRO_TEST = haskey(testsuite, "integration/numpyro")

# Install specific Python packages needed by the selected integration tests.
# Pkg.test doesn't pick up CondaPkg.toml in test folder.
if RUN_PYTHON_TEST || RUN_ENZYMEJAX_TEST || RUN_NUMPYRO_TEST
    if CondaPkg.backend() === :Null
        if RUN_ENZYMEJAX_TEST && !is_python_module_installed("enzyme_ad")
            delete!(testsuite, "integration/enzymejax")
        end
        if RUN_NUMPYRO_TEST && !is_python_module_installed("numpyro")
            delete!(testsuite, "integration/numpyro")
        end
    else
        if RUN_PYTHON_TEST
            # TensorFlow wheels (used by integration/python to validate exported SavedModels)
            # currently support up to Python 3.13. The environment may already have been
            # resolved with a newer Python while precompiling (PythonCall extensions of our
            # dependencies trigger a resolve). Downgrading the interpreter in place leaves
            # stale `cp314` wheels behind on some platforms (notably Windows), so start from
            # a clean environment whenever the installed Python is too new.
            python = CondaPkg.which("python")
            if python !== nothing
                installed_py = try
                    pycmd = "import sys; print('%d.%d' % sys.version_info[:2])"
                    VersionNumber(readchomp(`$(python) -c $(pycmd)`))
                catch
                    nothing
                end
                if installed_py === nothing || installed_py >= v"3.14"
                    env_dir = joinpath(dirname(CondaPkg.cur_deps_file()), ".CondaPkg")
                    isdir(env_dir) && rm(env_dir; recursive=true, force=true)
                end
            end
            CondaPkg.add("python"; version="<3.14,>=3.9,<4", resolve=false)
            CondaPkg.add_pip("tensorflow"; version=">=2.17", resolve=false)
        end
        if RUN_ENZYMEJAX_TEST || RUN_NUMPYRO_TEST
            CondaPkg.add_pip("jax"; version=">=0.9", resolve=false)
        end
        CondaPkg.resolve(; force=true)

        if RUN_ENZYMEJAX_TEST &&
            !add_pip_and_check("enzyme_ad", "enzyme_ad"; version=">=0.0.15")
            delete!(testsuite, "integration/enzymejax")
        end

        if RUN_NUMPYRO_TEST && !add_pip_and_check("numpyro", "numpyro"; version=">=0.21")
            delete!(testsuite, "integration/numpyro")
        end

        CondaPkg.activate!(ENV)
        if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
            python = CondaPkg.which("python")
            if python !== nothing
                ENV["JULIA_PYTHONCALL_EXE"] = python
            end
        end
    end
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
            # Determine backend before test to avoid illegal xla client creation before nccl initialization
            # MPI is only supported on CPU and CUDA
            BACKEND != "auto" && Reactant.set_default_backend(BACKEND)
            mpi_backend = lowercase(
                Reactant.XLA.platform_name(Reactant.XLA.default_backend())
            )
            if mpi_backend in ("cpu", "cuda")
                using MPI
                nranks = 2
                if mpi_backend == "cuda"
                    # NCCL pins exactly one GPU per MPI rank (see
                    # `Reactant.initialize_nccl_comm!`), so we can't launch more ranks
                    # than there are GPUs on this host.
                    ngpus = length(
                        Reactant.XLA.addressable_devices(Reactant.XLA.default_backend())
                    )
                    if ngpus < nranks
                        @warn "Only $ngpus GPU(s) available; running the GPU MPI tests \
                               with $ngpus rank(s) instead of $nranks"
                        nranks = ngpus
                    end
                end
                cmd = `$(mpiexec()) -n $nranks $(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(@__DIR__, "integration", "mpi.jl"))`
                run(addenv(cmd, "REACTANT_MPI_BACKEND" => mpi_backend))
            end
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
    # These tests require multiple devices, which the "julia" backend doesn't have
    if BACKEND != "julia"
        run_specific_test("core/sharding/sharding", parsed_args)
        run_specific_test("core/sharding/optimize_comm", parsed_args)
    end

    # qa tests run a separate process
    if BACKEND == "cpu" || BACKEND == "auto"
        run_specific_test("core/qa", parsed_args)
    end
end
