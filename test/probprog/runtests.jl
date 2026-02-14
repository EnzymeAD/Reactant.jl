using Reactant, ParallelTestRunner, Test

const BACKEND = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "auto"))
BACKEND != "auto" && Reactant.set_default_backend(BACKEND)

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
