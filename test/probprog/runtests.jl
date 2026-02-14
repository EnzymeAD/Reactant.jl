using Reactant, ParallelTestRunner, CondaPkg, Test

const BACKEND = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "auto"))
BACKEND != "auto" && Reactant.set_default_backend(BACKEND)

CondaPkg.add_pip("jax"; version="==0.9.0")
CondaPkg.add_pip("numpyro"; version="==0.19.0")
CondaPkg.resolve()

testsuite = find_tests(@__DIR__)
delete!(testsuite, "common")

@testset "ProbProg" begin
    runtests(
        Reactant,
        String[];
        testsuite,
        init_code=quote
            using Reactant
            $(BACKEND) != "auto" && Reactant.set_default_backend($(BACKEND))
        end,
    )
end
