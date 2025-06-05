using Reactant, Test, Random, StableRNGs, Statistics
using Reactant: ProbProg
using Libdl: Libdl

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function blr(seed, xs)
    function model(seed, xs)
        rng = Random.default_rng()
        Random.seed!(rng, seed)
        slope = ProbProg.sample!(normal, rng, 0, 2, (1,); symbol=:slope)
        intercept = ProbProg.sample!(normal, rng, 0, 10, (1,); symbol=:intercept)
        for (i, x) in enumerate(xs)
            ProbProg.sample!(normal, rng, slope * x + intercept, 1, (1,); symbol=Symbol("y-$i"))
        end
        return intercept
    end

    return ProbProg.simulate(model, seed, xs)
end

@testset "BLR" begin
    xs = [1, 2, 3, 4, 5]
    seed = Reactant.to_rarray(UInt64[1, 4])
    X = ProbProg.getTrace(@jit optimize = :probprog blr(seed, xs))
    @test X[:_integrity_check] == 0x123456789abcdef
    @show X
end
