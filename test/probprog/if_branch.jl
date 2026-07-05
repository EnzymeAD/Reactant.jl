using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

function branch_sim(rng, cond)
    return ProbProg.if_(
        rng,
        cond,
        () -> ProbProg.sample(rng, ProbProg.Normal(1.0, 0.5, (1,)); symbol=:ea)[2],
        () -> ProbProg.sample(rng, ProbProg.Normal(5.0, 0.5, (1,)); symbol=:eb)[2],
    )
end

# The uniform switch prior and constant-log-prob nuisances cancel, so MH targets p(z | y).
function mh_toy(rng)
    _, z = ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, 1.0); symbol=:z)
    cond = sum(z) > 0.5
    ProbProg.if_(
        rng,
        cond,
        () -> ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, 1.0); symbol=:ea)[2],
        () -> ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, 1.0); symbol=:eb)[2],
    )
    logit = 2.234 .* z .- 0.847
    _, y = ProbProg.sample(rng, ProbProg.Bernoulli(logit, (1,)); symbol=:y)
    return y
end

function mh_program(rng, model, obs_tensor, constrained, num_iters)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, model; constrained_addresses=constrained
    )
    zsum = trace[1:1, 1:1] .* 0.0
    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(
            rng, trace, weight, model; selection=ProbProg.select(ProbProg.Address(:z))
        )
        zsum = zsum .+ trace[1:1, 1:1]
    end
    return zsum
end

@testset "if_branch" begin
    @testset "DiscreteUniform" begin
        function du_model(rng)
            return ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, 3.0); symbol=:z)[2]
        end
        rng = ReactantRNG(Reactant.to_rarray(UInt64[7, 9]))
        trace, weight = ProbProg.simulate_(rng, du_model)
        @test weight ≈ -log(4) atol = 1e-12

        function du_draw(rng)
            return ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, 3.0); symbol=:z)[2]
        end
        draws = Float64[]
        for k in 1:400
            r = ReactantRNG(Reactant.to_rarray(UInt64[UInt64(k), 0x5]))
            tr, _ = ProbProg.simulate_(r, du_draw)
            push!(draws, ProbProg.get_choices(tr)[:z][1, 1])
        end
        @test all(d -> d in (0.0, 1.0, 2.0, 3.0), draws)
        @test 0.15 < mean(draws) / 3.0 < 0.85
    end

    @testset "simulate_branch_slots" begin
        rngT = ReactantRNG(Reactant.to_rarray(UInt64[1, 4]))
        trT, _ = ProbProg.simulate_(rngT, branch_sim, Reactant.ConcreteRNumber(true))
        choicesT = ProbProg.get_choices(trT)
        @test choicesT[:eb][1, 1] == 0.0
        @test choicesT[:ea][1, 1] != 0.0
        @test isapprox(choicesT[:ea][1, 1], 1.0; atol=2.0)

        rngF = ReactantRNG(Reactant.to_rarray(UInt64[1, 4]))
        trF, _ = ProbProg.simulate_(rngF, branch_sim, Reactant.ConcreteRNumber(false))
        choicesF = ProbProg.get_choices(trF)
        @test choicesF[:ea][1, 1] == 0.0
        @test choicesF[:eb][1, 1] != 0.0
        @test isapprox(choicesF[:eb][1, 1], 5.0; atol=2.0)
    end

    @testset "mh_posterior" begin
        rng = ReactantRNG(Reactant.to_rarray(UInt64[3, 7]))
        obs = ProbProg.Constraint(:y => [1.0])
        constrained = ProbProg.extract_addresses(obs)
        obs_tensor = ProbProg.flatten_constraint(obs)
        N = 40000

        compiled, _ = ProbProg.with_trace() do
            @compile optimize = :probprog mh_program(
                rng, mh_toy, obs_tensor, constrained, ConcreteRNumber(N)
            )
        end
        zsum = compiled(rng, mh_toy, obs_tensor, constrained, ConcreteRNumber(N))
        phat = Array(zsum)[1] / N

        sig(t) = 1 / (1 + exp(-t))
        exact = sig(2.234 - 0.847) / (sig(2.234 - 0.847) + sig(-0.847))
        @test isapprox(phat, exact; atol=0.02)
    end
end
