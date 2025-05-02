using Reactant, Test, Random, StableRNGs, Statistics
using Reactant: ProbProg

normal(rng, mean, stddev) = mean .+ stddev .* randn(rng, 10000)

function model(mean, stddev)
    s = ProbProg.sample(normal, StableRNG(0), mean, stddev)
    t = ProbProg.sample(normal, StableRNG(0), s, stddev)
    return t
end

@testset "ProbProg" begin
    @testset "normal_hlo" begin
        hlo = @code_hlo ProbProg.generate(
            model, Reactant.to_rarray(0.0), Reactant.to_rarray(1.0)
        )
        @test contains(repr(hlo), "enzyme.generate")
        @test contains(repr(hlo), "enzyme.sample")
        # println(hlo)

        lowered = Reactant.Compiler.run_pass_pipeline_on_source(repr(hlo), "probprog")
        println(lowered)
    end

    @testset "normal_generate" begin
        X = Array(
            @jit optimize = :probprog ProbProg.generate(
                model, Reactant.to_rarray(0.0), Reactant.to_rarray(1.0)
            )
        )
        @test mean(X) ≈ 0.0 atol = 0.05 rtol = 0.05
    end
end
