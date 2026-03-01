using Reactant, ParallelTestRunner, CondaPkg, Test

const BACKEND = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "auto"))

CondaPkg.add_pip("jax"; version="==0.9.0")
CondaPkg.add_pip("numpyro"; version="==0.19.0")
CondaPkg.resolve()

testsuite = find_tests(@__DIR__)
delete!(testsuite, "common")

jobs = min(
    something(
        Reactant.Accelerators.TPU.has_tpu() || BACKEND == "tpu" ? 1 : nothing,
        ParallelTestRunner.default_njobs(),
    ),
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
            init_code=quote
                using Reactant
                $(BACKEND) != "auto" && Reactant.set_default_backend($(BACKEND))
            end,
        )
    end
end
