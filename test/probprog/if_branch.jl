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

# The observation lives inside the taken branch; a flip must re-read it from the constraint.
_pois_sampler(rng, rate) = Reactant.Ops.fill(0.0, 1)
_pois_logpdf(y, rate) = sum(y .* log.(rate) .- rate .- Reactant.Ops.lgamma(y .+ 1.0))

function coal1(rng)
    _, s = ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, 1.0); symbol=:s)
    _, l1 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l1)
    _, l2 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l2)
    cond = sum(s) > 0.0
    ProbProg.if_(
        rng,
        cond,
        () -> ProbProg.sample(rng, _pois_sampler, l1; symbol=:ye_0, logpdf=_pois_logpdf)[2],
        () -> ProbProg.sample(rng, _pois_sampler, l2; symbol=:yl_0, logpdf=_pois_logpdf)[2],
    )
    return nothing
end

function coal1_mh(rng, model, obs_tensor, constrained, num_iters)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, model; constrained_addresses=constrained
    )
    acc = trace[1:1, 1:1] .* 0.0
    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model;
            selection=ProbProg.select(ProbProg.Address(:s)),
            constraint=obs_tensor,
            constrained_addresses=constrained,
        )
        acc = acc .+ trace[1:1, 1:1]
    end
    return acc
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

    @testset "coal_branch_flip_mh" begin
        rng = ReactantRNG(Reactant.to_rarray(UInt64[7, 9]))
        obs = ProbProg.Constraint(
            :l1 => [3.0], :l2 => [0.9], :ye_0 => [5.0], :yl_0 => [5.0]
        )
        constrained = ProbProg.extract_addresses(obs)
        obs_tensor = ProbProg.flatten_constraint(obs)
        N = 20000

        compiled, _ = ProbProg.with_trace() do
            @compile optimize = :probprog coal1_mh(
                rng, coal1, obs_tensor, constrained, ConcreteRNumber(N)
            )
        end
        acc = compiled(rng, coal1, obs_tensor, constrained, ConcreteRNumber(N))
        phat = Array(acc)[1] / N

        # Analytic p(s=1 | y) = σ(logpmf(5|3) - logpmf(5|0.9)) = 0.9806.
        @test 0.97 <= phat <= 0.99
    end
end

# MH must read top-level latents from the trace, not from the constraint that lists them.
const _COALIF_Y = [5.0, 4.0, 1.0, 0.0, 1.0]
const _COALIF_T = length(_COALIF_Y)

function _coalif_model(rng)
    _, s = ProbProg.sample(
        rng, ProbProg.DiscreteUniform(0.0, Float64(_COALIF_T)); symbol=:s
    )
    _, l1 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l1)
    _, l2 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l2)
    for t in 0:(_COALIF_T - 1)
        ProbProg.if_(
            rng,
            sum(s) > Float64(t),
            () -> ProbProg.sample(
                rng, _pois_sampler, l1; symbol=Symbol("ye_", t), logpdf=_pois_logpdf
            )[2],
            () -> ProbProg.sample(
                rng, _pois_sampler, l2; symbol=Symbol("yl_", t), logpdf=_pois_logpdf
            )[2],
        )
    end
    return nothing
end

function _coalif_chain(rng, model, bins, ot, constrained, n)
    trace, weight, _ = ProbProg.generate(rng, ot, model; constrained_addresses=constrained)
    hist = zero.(bins)
    a1 = zero.(bins[1:1, 1:1])
    a2 = zero.(a1)
    q1 = zero.(a1)
    q2 = zero.(a1)
    @trace for _ in 1:n
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model;
            selection=ProbProg.select(ProbProg.Address(:s)),
            constraint=ot,
            constrained_addresses=constrained,
        )
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model;
            selection=ProbProg.select(ProbProg.Address(:l1)),
            constraint=ot,
            constrained_addresses=constrained,
        )
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model;
            selection=ProbProg.select(ProbProg.Address(:l2)),
            constraint=ot,
            constrained_addresses=constrained,
        )
        sv = sum(trace[1:1, 1:1])
        hist = hist .+ ifelse.(abs.(bins .- sv) .< 0.5, 1.0, 0.0)
        x = trace[1:1, 2:2]
        y = trace[1:1, 3:3]
        a1 = a1 .+ x
        a2 = a2 .+ y
        q1 = q1 .+ x .* x
        q2 = q2 .+ y .* y
    end
    return hist, a1, a2, q1, q2
end

# Exact posterior by Gamma-Poisson conjugacy: p(s | y), E and sd of l1, l2.
function _coalif_exact()
    a, b = 3.0, 1.0
    T, Y = _COALIF_T, _COALIF_Y
    lg(x) = sum(log, 1.0:1.0:(x - 1.0); init=0.0)  # log Gamma(x), integer x
    logG(S, n) = lg(a + S) - lg(a) - (a + S) * log(b + n)
    lp = zeros(T + 1)
    m1 = zeros(T + 1)
    m2 = zeros(T + 1)
    v1 = zeros(T + 1)
    v2 = zeros(T + 1)
    for s in 0:T
        S1, n1 = sum(Y[1:s]), s
        S2, n2 = sum(Y[(s + 1):end]), T - s
        lp[s + 1] = logG(S1, n1) + logG(S2, n2)
        m1[s + 1] = (a + S1) / (b + n1)
        m2[s + 1] = (a + S2) / (b + n2)
        v1[s + 1] = (a + S1) / (b + n1)^2
        v2[s + 1] = (a + S2) / (b + n2)^2
    end
    p = exp.(lp .- maximum(lp))
    p ./= sum(p)
    E1, E2 = sum(p .* m1), sum(p .* m2)
    sd1 = sqrt(sum(p .* (v1 .+ m1 .^ 2)) - E1^2)
    sd2 = sqrt(sum(p .* (v2 .+ m2 .^ 2)) - E2^2)
    return p, E1, sd1, E2, sd2
end

@testset "coal_if_branch_exact_posterior" begin
    p, E1, sd1, E2, sd2 = _coalif_exact()
    pairs = Pair{Symbol,Any}[:l1 => [3.0], :l2 => [0.9]]
    for t in 0:(_COALIF_T - 1)
        push!(pairs, Symbol("ye_", t) => [_COALIF_Y[t + 1]])
        push!(pairs, Symbol("yl_", t) => [_COALIF_Y[t + 1]])
    end
    obs = ProbProg.Constraint(pairs...)
    constrained = ProbProg.extract_addresses(obs)
    ot = ProbProg.flatten_constraint(obs)
    bins = Reactant.to_rarray(reshape(Float64.(0:_COALIF_T), 1, _COALIF_T + 1))
    N = 200000
    rng = ReactantRNG(Reactant.to_rarray(UInt64[11, 12]))
    compiled, _ = ProbProg.with_trace() do
        @compile optimize = :probprog _coalif_chain(
            rng, _coalif_model, bins, ot, constrained, ConcreteRNumber(N)
        )
    end
    hist, a1, a2, q1, q2 = compiled(
        rng, _coalif_model, bins, ot, constrained, ConcreteRNumber(N)
    )
    num(x) = x isa Number ? Float64(x) : sum(Array(x))
    m1, m2 = num(a1) / N, num(a2) / N
    s1, s2 = sqrt(num(q1) / N - m1^2), sqrt(num(q2) / N - m2^2)
    q = vec(Array(hist)) ./ N
    @test 0.5 * sum(abs.(q .- p)) < 0.03
    @test m1 ≈ E1 atol = 0.08
    @test m2 ≈ E2 atol = 0.06
    @test 0.9 < s1 / sd1 < 1.1
    @test 0.9 < s2 / sd2 < 1.1
end
