using Reactant, ParallelTestRunner, CondaPkg, Test

const BACKEND = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "auto"))

CondaPkg.add_pip("jax"; version=">=0.9")
CondaPkg.add_pip("numpyro"; version=">=0.21")
CondaPkg.resolve()

# The environment is resolved once, here. Each worker's `using PythonCall` would
# otherwise resolve it again from PythonCall's `__init__` (CondaPkg.resolve takes
# the environment's lock file before it decides there is nothing to do), so all
# the workers race for one lock at startup. On Windows that race can end in
# EACCES rather than EEXIST from Pidfile.tryopen_exclusive, which is not
# retried, and the worker fails to load PythonCall. Hand the workers the Python
# resolved above and the activated environment, so they never touch CondaPkg.
CondaPkg.activate!(ENV)
ENV["JULIA_PYTHONCALL_EXE"] = CondaPkg.which("python")

testsuite = find_tests(@__DIR__)
delete!(testsuite, "common")

include("../test_helpers.jl")

if NTPUs > 0 || BACKEND == "tpu"
    empty!(testsuite)
end

custom_test_worker = NTPUs > 0 || BACKEND == "tpu"

jobs = min(
    something(custom_test_worker ? NTPUs : nothing, ParallelTestRunner.default_njobs()),
    length(keys(testsuite)),
)

@testset "ProbProg" begin
    withenv(
        "XLA_REACTANT_GPU_MEM_FRACTION" => 1 / (jobs + 0.1),
        "XLA_REACTANT_GPU_PREALLOCATE" => false,
    ) do
        runtests(
            Reactant,
            String["--jobs=$(jobs)"];
            testsuite,
            test_worker=custom_test_worker ? tpu_custom_worker_launcher : Returns(nothing),
            init_code=quote
                using Reactant
                $(BACKEND) != "auto" && Reactant.set_default_backend($(BACKEND))
            end,
        )
    end
end
