using Reactant, Test, Random, StableRNGs, Statistics
using Reactant: ProbProg

normal(rng, mean, stddev) = mean .+ stddev .* randn(rng, 10000)

function model(rng, mean, stddev)
    s = ProbProg.sample(normal, rng, mean, stddev)
    t = ProbProg.sample(normal, rng, s, stddev)
    return t
end

@testset "ProbProg" begin
    @testset "normal_hlo" begin
        rng = StableRNG(0)
        before = @code_hlo optimize = :none ProbProg.generate(
            model, rng, Reactant.to_rarray(0.0), Reactant.to_rarray(1.0)
        )
        @test contains(repr(before), "enzyme.generate")
        @test contains(repr(before), "enzyme.sample")

        # println("Before")
        # println(repr(before))

        after = @code_hlo optimize = :all ProbProg.generate(
            model, rng, Reactant.to_rarray(0.0), Reactant.to_rarray(1.0)
        )
        @test !contains(repr(after), "enzyme.generate")
        @test !contains(repr(after), "enzyme.sample")

        # println("After")
        # println(repr(after))
    end

    @testset "normal_generate" begin
        rng = StableRNG(1)
        X = Array(
            @jit optimize = :all ProbProg.generate(
                model, rng, Reactant.to_rarray(0.0), Reactant.to_rarray(1.0)
            )
        )
        @test mean(X) ≈ 0.0 atol = 0.05 rtol = 0.05
    end
end
