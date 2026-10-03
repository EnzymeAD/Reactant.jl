using Reactant, Test, Random, FileCheck
using Reactant: ProbProg, ReactantRNG

# Reference: https://www.gen.dev/docs/stable/getting_started/linear_regression/

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -length(x) * log(σ) - length(x) / 2 * log(2π) -
           sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end

function model(rng, xs)
    _, slope = ProbProg.sample(
        rng, normal, 0.0, 2.0, (1,); symbol=:slope, logpdf=normal_logpdf
    )
    _, intercept = ProbProg.sample(
        rng, normal, 0.0, 10.0, (1,); symbol=:intercept, logpdf=normal_logpdf
    )

    _, ys = ProbProg.sample(
        rng,
        normal,
        slope .* xs .+ intercept,
        1.0,
        (length(xs),);
        symbol=:ys,
        logpdf=normal_logpdf,
    )

    return ys
end

function mh_program(rng, model, xs, num_iters, constraint_tensor, constrained_addresses)
    trace, weight, _ = ProbProg.generate(
        rng, constraint_tensor, model, xs; constrained_addresses=constrained_addresses
    )

    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            xs;
            selection=ProbProg.select(ProbProg.Address(:slope)),
        )
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            xs;
            selection=ProbProg.select(ProbProg.Address(:intercept)),
        )
    end

    return trace, weight
end

function mh_program_flat(
    rng, model, xs, num_iters, constraint_tensor, constrained_addresses
)
    trace, weight, _ = ProbProg.generate(
        rng, constraint_tensor, model, xs; constrained_addresses=constrained_addresses
    )
    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            xs;
            selection=ProbProg.select(ProbProg.Address(:slope)),
            region=false,
        )
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            xs;
            selection=ProbProg.select(ProbProg.Address(:intercept)),
            region=false,
        )
    end
    return trace, weight
end

@testset "linear_regression" begin
    @testset "simulate" begin
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)

        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        xs_r = Reactant.to_rarray(xs)

        trace, _ = ProbProg.simulate_(rng, model, xs_r)

        @test haskey(trace.choices, :slope)
        @test haskey(trace.choices, :intercept)
        @test haskey(trace.choices, :ys)
    end

    @testset "inference" begin
        seed = Reactant.to_rarray(UInt64[1, 5])
        rng = ReactantRNG(seed)

        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
        obs = ProbProg.Constraint(:ys => ys)
        num_iters = ConcreteRNumber(10000)
        constrained_addresses = ProbProg.extract_addresses(obs)
        obs_tensor = ProbProg.flatten_constraint(obs)

        code, _ = ProbProg.with_trace() do
            @code_hlo optimize = :probprog mh_program(
                rng, model, xs, num_iters, obs_tensor, constrained_addresses
            )
        end
        @test @filecheck begin
            @check_not "impulse.mh"
            repr(code)
        end

        num_iters = ConcreteRNumber(1000)
        compiled_fn, tt = ProbProg.with_trace() do
            @compile optimize = :probprog mh_program(
                rng, model, xs, num_iters, obs_tensor, constrained_addresses
            )
        end

        trace_tensor, weight_val = compiled_fn(
            rng, model, xs, num_iters, obs_tensor, constrained_addresses
        )
        trace = ProbProg.unflatten_trace(trace_tensor, weight_val, tt.entries, ())

        slope = trace.choices[:slope][1, 1]
        intercept = trace.choices[:intercept][1, 1]
        @test slope ≈ -2.0 rtol = 0.1
        @test intercept ≈ 10.0 rtol = 0.1
    end

    @testset "region_vs_flat" begin
        seed = Reactant.to_rarray(UInt64[1, 5])
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
        obs = ProbProg.Constraint(:ys => ys)
        constrained_addresses = ProbProg.extract_addresses(obs)
        obs_tensor = ProbProg.flatten_constraint(obs)
        num_iters = ConcreteRNumber(200)

        rngF = ReactantRNG(Reactant.to_rarray(copy(Array(seed))))
        cF, ttF = ProbProg.with_trace() do
            @compile optimize = :probprog mh_program_flat(
                rngF, model, xs, num_iters, obs_tensor, constrained_addresses
            )
        end
        trF, wF = cF(rngF, model, xs, num_iters, obs_tensor, constrained_addresses)
        traceF = ProbProg.unflatten_trace(trF, wF, ttF.entries, ())

        rngR = ReactantRNG(Reactant.to_rarray(copy(Array(seed))))
        cR, ttR = ProbProg.with_trace() do
            @compile optimize = :probprog mh_program(
                rngR, model, xs, num_iters, obs_tensor, constrained_addresses
            )
        end
        trR, wR = cR(rngR, model, xs, num_iters, obs_tensor, constrained_addresses)
        traceR = ProbProg.unflatten_trace(trR, wR, ttR.entries, ())

        @test traceR.choices[:slope][1, 1] ≈ traceF.choices[:slope][1, 1] atol = 1e-12
        @test traceR.choices[:intercept][1, 1] ≈ traceF.choices[:intercept][1, 1] atol =
            1e-12
        @test Array(wR)[] ≈ Array(wF)[] atol = 1e-12
    end
end

# SICM rewrites chol(tau^2 R), so weights differ by ULPs but the chains must match.
function model_scale_family_mvn(rng, R_data, y)
    n = Base.length(y)
    _, tau = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:tau)
    tau_2d = reshape(tau, 1, 1)
    K = (tau_2d .* tau_2d) .* R_data
    _, obs = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(n), K, (n,)); symbol=:y)
    return obs
end

function mvn_mh_region(rng, model, R, y, obs_tensor, constrained, num_iters)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, model, R, y; constrained_addresses=constrained
    )
    acc = weight .* 0.0
    @trace for _ in 1:num_iters
        trace, weight, accepted, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            R,
            y;
            selection=ProbProg.select(ProbProg.Address(:tau)),
            region=true,
        )
        acc = acc .+ ifelse.(accepted, 1.0, 0.0)
    end
    return trace, weight, acc
end
function mvn_mh_flat(rng, model, R, y, obs_tensor, constrained, num_iters)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, model, R, y; constrained_addresses=constrained
    )
    acc = weight .* 0.0
    @trace for _ in 1:num_iters
        trace, weight, accepted, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            R,
            y;
            selection=ProbProg.select(ProbProg.Address(:tau)),
            region=false,
        )
        acc = acc .+ ifelse.(accepted, 1.0, 0.0)
    end
    return trace, weight, acc
end

@testset "sicm_scale_family_parity" begin
    n = 8
    R = [i == j ? 1.0 : 0.3^abs(i - j) for i in 1:n, j in 1:n]
    y = collect(range(0.4, -0.4; length=n))
    Rr = Reactant.to_rarray(R)
    yr = Reactant.to_rarray(y)
    obs = ProbProg.Constraint(:y => y)
    constrained = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)
    N = ConcreteRNumber(100)

    rngF = ReactantRNG(Reactant.to_rarray(UInt64[2, 11]))
    cF, ttF = ProbProg.with_trace() do
        @compile optimize = :probprog mvn_mh_flat(
            rngF, model_scale_family_mvn, Rr, yr, obs_tensor, constrained, N
        )
    end
    trF, wF, accF = cF(rngF, model_scale_family_mvn, Rr, yr, obs_tensor, constrained, N)
    traceF = ProbProg.unflatten_trace(trF, wF, ttF.entries, ())

    old_opt = Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[]
    old_sicm = Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[]
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = false
    local trR, wR, accR, traceR, hloR
    try
        rngC = ReactantRNG(Reactant.to_rarray(UInt64[2, 11]))
        codeR, _ = ProbProg.with_trace() do
            @code_hlo optimize = :probprog mvn_mh_region(
                rngC, model_scale_family_mvn, Rr, yr, obs_tensor, constrained, N
            )
        end
        hloR = repr(codeR)

        rngR = ReactantRNG(Reactant.to_rarray(UInt64[2, 11]))
        cR, ttR = ProbProg.with_trace() do
            @compile optimize = :probprog mvn_mh_region(
                rngR, model_scale_family_mvn, Rr, yr, obs_tensor, constrained, N
            )
        end
        trR, wR, accR = cR(rngR, model_scale_family_mvn, Rr, yr, obs_tensor, constrained, N)
        traceR = ProbProg.unflatten_trace(trR, wR, ttR.entries, ())
    finally
        Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = old_opt
        Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = old_sicm
    end

    # The rewritten sqrt(tau^2) * chol(R) appears in the loop; the flat path has no sqrt.
    widx = findfirst("stablehlo.while", hloR)
    body = hloR[first(widx):end]
    @test length(collect(eachmatch(r"stablehlo\.sqrt", body))) >= 1

    @test Float64(accF) > 0
    @test Float64(accR) == Float64(accF)

    @test traceR.choices[:tau][1, 1] ≈ traceF.choices[:tau][1, 1] atol = 1e-12
    @test maximum(abs.(Array(trR) .- Array(trF))) <= 1e-12
    @test Array(wR)[] ≈ Array(wF)[] atol = 1e-12
end

# Without the Hastings q terms, prior-regeneration MH targets prior^2 * likelihood.

using SpecialFunctions: loggamma

function _hmh_with_region_pipeline(f, on::Bool)
    old = Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[]
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = !on
    try
        return f()
    finally
        Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = old
    end
end

function _hmh_gamma_model(rng)
    _, _ = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l)
    return nothing
end

function _hmh_gamma_branch_model(rng)
    _, l = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l)
    ProbProg.if_(
        rng,
        sum(l) > 3.0,
        () -> ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:na)[2],
        () -> ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, 1.0); symbol=:nb)[2],
    )
    return nothing
end

for R in (false, true)
    @eval function _hmh_prior_chain(rng, model, ot, constrained, n, ::Val{$R})
        trace, weight, _ = ProbProg.generate(
            rng, ot, model; constrained_addresses=constrained
        )
        s1 = weight .* 0.0
        s2 = weight .* 0.0
        nacc = weight .* 0.0
        @trace for _ in 1:n
            trace, weight, accepted, _ = ProbProg.mh(
                rng,
                trace,
                weight,
                model;
                selection=ProbProg.select(ProbProg.Address(:l)),
                region=$R,
            )
            v = sum(trace[1:1, 1:1])
            s1 = s1 .+ v
            s2 = s2 .+ v .* v
            nacc = nacc .+ ifelse.(accepted, 1.0, 0.0)
        end
        return s1, s2, nacc
    end
end

_num(x) = x isa Number ? Float64(x) : sum(Array(x))

@testset "mh_hastings_prior_invariance" begin
    N = 40000
    # (label, model, region keyword, region pipeline on)
    cases = (
        ("flat mh, score-tail trace", _hmh_gamma_model, false, false),
        ("infer mh, score-tail trace, flat lowering", _hmh_gamma_model, true, false),
        ("infer mh, score-tail trace, region form", _hmh_gamma_model, true, true),
        ("flat mh, bare trace with branch flips", _hmh_gamma_branch_model, false, false),
        ("infer mh, bare trace with branch flips", _hmh_gamma_branch_model, true, true),
    )
    for (i, (label, model, region, pipeline)) in enumerate(cases)
        @testset "$label" begin
            obs = ProbProg.Constraint(:l => [3.0])
            constrained = ProbProg.extract_addresses(obs)
            ot = ProbProg.flatten_constraint(obs)
            rng = ReactantRNG(Reactant.to_rarray(UInt64[12345, i]))
            s1, s2, nacc = _hmh_with_region_pipeline(pipeline) do
                compiled, _ = ProbProg.with_trace() do
                    @compile optimize = :probprog _hmh_prior_chain(
                        rng, model, ot, constrained, ConcreteRNumber(N), Val(region)
                    )
                end
                compiled(rng, model, ot, constrained, ConcreteRNumber(N), Val(region))
            end
            m = _num(s1) / N
            v = _num(s2) / N - m^2
            @test m ≈ 3.0 atol = 0.1
            @test v ≈ 3.0 atol = 0.3
            # Nothing observed: every prior-regeneration move is accepted.
            @test _num(nacc) / N > 0.999
        end
    end
end

# A :z flip draws the newly live latent fresh; both latents' densities leave the acceptance.
_hmh_obs_sampler(rng, mu) = zero.(mu)
_hmh_obs_logpdf(y, mu) = sum(-0.5 * log(2π) .- (y .- mu) .^ 2 ./ 2)

function _hmh_switch_model(rng)
    _, z = ProbProg.sample(rng, ProbProg.Bernoulli(0.8, (1,)); symbol=:z)
    v = ProbProg.if_(
        rng,
        sum(z) > 0.5,
        () -> ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, 3.0); symbol=:a)[2],
        () -> ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, 2.0); symbol=:b)[2],
    )
    _, y = ProbProg.sample(rng, _hmh_obs_sampler, v; symbol=:y, logpdf=_hmh_obs_logpdf)
    return y
end

# Bins: (z=1, a) -> a, (z=0, b) -> 4 + b; trace columns z, a, b, y.
for R in (false, true)
    @eval function _hmh_switch_chain(rng, model, bins, ot, constrained, n, ::Val{$R})
        trace, weight, _ = ProbProg.generate(
            rng, ot, model; constrained_addresses=constrained
        )
        hist = zero.(bins)
        @trace for _ in 1:n
            trace, weight, _ = ProbProg.mh(
                rng,
                trace,
                weight,
                model;
                selection=ProbProg.select(ProbProg.Address(:z)),
                region=$R,
            )
            trace, weight, _ = ProbProg.mh(
                rng,
                trace,
                weight,
                model;
                selection=ProbProg.select(ProbProg.Address(:a)),
                region=$R,
            )
            trace, weight, _ = ProbProg.mh(
                rng,
                trace,
                weight,
                model;
                selection=ProbProg.select(ProbProg.Address(:b)),
                region=$R,
            )
            zv = sum(trace[1:1, 1:1])
            av = sum(trace[1:1, 2:2])
            bv = sum(trace[1:1, 3:3])
            st = zv .* av .+ (1.0 .- zv) .* (4.0 .+ bv)
            hist = hist .+ ifelse.(abs.(bins .- st) .< 0.5, 1.0, 0.0)
        end
        return hist
    end
end

function _hmh_switch_exact(y)
    pz = 1 / (1 + exp(-0.8))
    lik(m) = exp(-0.5 * (y - m)^2)
    w = vcat(
        [pz * (1 / 4) * lik(a) for a in 0:3], [(1 - pz) * (1 / 3) * lik(b) for b in 0:2]
    )
    return w ./ sum(w)
end

@testset "mh_hastings_branch_flip_exact" begin
    y = 2.2
    exact = _hmh_switch_exact(y)
    # The uncorrected kernel's target is 0.093 away in total variation.
    N = 30000
    obs = ProbProg.Constraint(:y => [y])
    constrained = ProbProg.extract_addresses(obs)
    ot = ProbProg.flatten_constraint(obs)
    bins = Reactant.to_rarray(reshape(collect(0.0:6.0), 1, 7))
    for (i, region) in enumerate((false, true))
        @testset "region=$region" begin
            rng = ReactantRNG(Reactant.to_rarray(UInt64[777, i]))
            compiled, _ = ProbProg.with_trace() do
                @compile optimize = :probprog _hmh_switch_chain(
                    rng,
                    _hmh_switch_model,
                    bins,
                    ot,
                    constrained,
                    ConcreteRNumber(N),
                    Val(region),
                )
            end
            hist = vec(
                Array(
                    compiled(
                        rng,
                        _hmh_switch_model,
                        bins,
                        ot,
                        constrained,
                        ConcreteRNumber(N),
                        Val(region),
                    ),
                ),
            )
            @test sum(hist) == N
            tv = 0.5 * sum(abs.(hist ./ N .- exact))
            @test tv < 0.03
        end
    end
end

# Jarrett coal-mining series 1851..1961, the two missing years dropped.
const _HMH_COAL_COUNTS = [
    4,
    5,
    4,
    0,
    1,
    4,
    3,
    4,
    0,
    6,
    3,
    3,
    4,
    0,
    2,
    6,
    3,
    3,
    5,
    4,
    5,
    3,
    1,
    4,
    4,
    1,
    5,
    5,
    3,
    4,
    2,
    5,
    2,
    2,
    3,
    4,
    2,
    1,
    3,
    nothing,
    2,
    1,
    1,
    1,
    1,
    3,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    3,
    1,
    0,
    3,
    2,
    2,
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    2,
    1,
    0,
    0,
    0,
    1,
    1,
    0,
    2,
    3,
    3,
    1,
    nothing,
    2,
    1,
    1,
    1,
    1,
    2,
    4,
    2,
    0,
    0,
    1,
    4,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    1,
]
const _HMH_COAL_T = length(_HMH_COAL_COUNTS)
const _HMH_COAL_TIDX = Float64[
    i - 1 for (i, c) in enumerate(_HMH_COAL_COUNTS) if c !== nothing
]
const _HMH_COAL_Y = Float64[c for c in _HMH_COAL_COUNTS if c !== nothing]

_hmh_coal_sampler(rng, s, l1, l2, tvec, lg) = zero.(tvec)
function _hmh_coal_logpdf(yv, s, l1, l2, tvec, lg)
    lam = ifelse.(tvec .< sum(s), sum(l1), sum(l2))
    return sum(yv .* log.(lam) .- lam .- lg)
end
function _hmh_coal_model(rng, tvec, lg)
    _, s = ProbProg.sample(
        rng, ProbProg.DiscreteUniform(0.0, Float64(_HMH_COAL_T)); symbol=:s
    )
    _, l1 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l1)
    _, l2 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l2)
    _, _ = ProbProg.sample(
        rng, _hmh_coal_sampler, s, l1, l2, tvec, lg; symbol=:y, logpdf=_hmh_coal_logpdf
    )
    return nothing
end

function _hmh_coal_chain(rng, model, tvec, lg, bins, ot, constrained, n)
    trace, weight, _ = ProbProg.generate(
        rng, ot, model, tvec, lg; constrained_addresses=constrained
    )
    hist = zero.(bins)
    sl1 = weight .* 0.0
    sl2 = weight .* 0.0
    @trace for _ in 1:n
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            tvec,
            lg;
            selection=ProbProg.select(ProbProg.Address(:s)),
        )
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            tvec,
            lg;
            selection=ProbProg.select(ProbProg.Address(:l1)),
        )
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            tvec,
            lg;
            selection=ProbProg.select(ProbProg.Address(:l2)),
        )
        s_v = sum(trace[1:1, 1:1])
        hist = hist .+ ifelse.(abs.(bins .- s_v) .< 0.5, 1.0, 0.0)
        sl1 = sl1 .+ sum(trace[1:1, 2:2])
        sl2 = sl2 .+ sum(trace[1:1, 3:3])
    end
    return hist, sl1, sl2
end

function _hmh_coal_exact(; a=3.0, b=1.0)
    T = _HMH_COAL_T
    t = _HMH_COAL_TIDX
    y = _HMH_COAL_Y
    logG(S, n) = loggamma(a + S) - loggamma(a) - (a + S) * log(b + n)
    lp = zeros(T + 1)
    m1 = zeros(T + 1)
    m2 = zeros(T + 1)
    for s in 0:T
        e = t .< s
        S1, n1 = sum(y[e]), count(e)
        S2, n2 = sum(y[.!e]), count(.!e)
        lp[s + 1] = logG(S1, n1) + logG(S2, n2)
        m1[s + 1] = (a + S1) / (b + n1)
        m2[s + 1] = (a + S2) / (b + n2)
    end
    p = exp.(lp .- maximum(lp))
    p ./= sum(p)
    return p, sum(p .* m1), sum(p .* m2)
end

@testset "mh_hastings_coal_jarrett_exact" begin
    p, El1, El2 = _hmh_coal_exact()
    @test El1 ≈ 3.1432 atol = 1e-4
    @test El2 ≈ 0.9623 atol = 1e-4

    lg_data = [loggamma(v + 1.0) for v in _HMH_COAL_Y]
    tvec = Reactant.to_rarray(_HMH_COAL_TIDX)
    lg = Reactant.to_rarray(lg_data)
    bins = Reactant.to_rarray(reshape(Float64.(0:_HMH_COAL_T), 1, _HMH_COAL_T + 1))
    obs = ProbProg.Constraint(:y => _HMH_COAL_Y, :l1 => [3.0], :l2 => [0.9])
    constrained = ProbProg.extract_addresses(obs)
    ot = ProbProg.flatten_constraint(obs)
    N = 200000
    rng = ReactantRNG(Reactant.to_rarray(UInt64[1851, 1961]))
    compiled, _ = ProbProg.with_trace() do
        @compile optimize = :probprog _hmh_coal_chain(
            rng, _hmh_coal_model, tvec, lg, bins, ot, constrained, ConcreteRNumber(N)
        )
    end
    hist, sl1, sl2 = compiled(
        rng, _hmh_coal_model, tvec, lg, bins, ot, constrained, ConcreteRNumber(N)
    )
    q = vec(Array(hist)) ./ N
    @test 0.5 * sum(abs.(q .- p)) < 0.05
    # Tolerances are half the gap to the uncorrected kernel's target.
    @test _num(sl1) / N ≈ El1 atol = 0.014
    @test _num(sl2) / N ≈ El2 atol = 0.007
end

# Gen tutorial line-vs-sine data; _LS_E_SLOPE is the exact posterior mean by quadrature.
# Region ops capture slope etc. from above, so a move on them must dirty the observation.

const _LS_X = [
    0.0,
    0.2759,
    0.5517,
    0.8276,
    1.1034,
    1.3793,
    1.6552,
    1.931,
    2.2069,
    2.4828,
    2.7586,
    3.0345,
    3.3103,
    3.5862,
    3.8621,
    4.1379,
    4.4138,
    4.6897,
    4.9655,
    5.2414,
    5.5172,
    5.7931,
    6.069,
    6.3448,
    6.6207,
    6.8966,
    7.1724,
    7.4483,
    7.7241,
    8.0,
]
const _LS_Y = [
    -0.4453,
    -0.8217,
    -0.4477,
    -0.3201,
    -0.9794,
    -0.1994,
    0.5287,
    0.1443,
    1.2569,
    1.8079,
    1.0433,
    1.9606,
    2.2105,
    3.1492,
    2.9465,
    3.6189,
    2.5726,
    3.3957,
    3.6323,
    4.1488,
    5.7783,
    4.2277,
    5.3055,
    5.5581,
    5.9808,
    5.735,
    6.3052,
    6.6012,
    5.8466,
    6.8494,
]
const _LS_E_SLOPE = 1.0217303713623416

_ls_unif_sampler(rng, shape) = Float64(2π) .* rand(rng, shape...)
_ls_unif_logpdf(x, _) = -log(2π) .+ 0.0 .* sum(x)
_ls_y_sampler(rng, mu) = zero.(mu)
_ls_y_logpdf(yv, mu) = sum(.-log(0.5) .- 0.5 * log(2π) .- (yv .- mu) .^ 2 ./ (2 * 0.5^2))

function _ls_sites(rng)
    _, m = ProbProg.sample(rng, ProbProg.Bernoulli(0.0, (1,)); symbol=:m)
    _, slope = ProbProg.sample(rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:slope)
    _, intercept = ProbProg.sample(rng, ProbProg.Normal(0.0, 3.0, (1,)); symbol=:intercept)
    _, amp = ProbProg.sample(rng, ProbProg.Normal(0.0, 3.0, (1,)); symbol=:amp)
    _, phase = ProbProg.sample(
        rng, _ls_unif_sampler, (1,); symbol=:phase, logpdf=_ls_unif_logpdf
    )
    return m, slope, intercept, amp, phase
end

# Index 0 = sine, 1 = line.
function _ls_mean_case(m, slope, intercept, amp, phase, xs)
    return Reactant.Ops.case(
        convert(Reactant.TracedRNumber{Int32}, sum(m)),
        [
            (xv, sl, ic, am, ph) -> sum(am) .* sin.(Float64(2π) .* xv ./ 4.0 .+ sum(ph)),
            (xv, sl, ic, am, ph) -> sum(sl) .* xv .+ sum(ic),
        ],
        xs,
        slope,
        intercept,
        amp,
        phase;
        track_numbers=Number,
    )
end

# Literal 2π/4: @trace if passes every name its branches read, and π cannot be traced.
function _ls_mean_traceif(m, slope, intercept, amp, phase, xs)
    @trace if sum(m) > 0.5
        mu = sum(slope) .* xs .+ sum(intercept)
    else
        mu = sum(amp) .* sin.(xs .* 1.5707963267948966 .+ sum(phase))
    end
    return mu
end

for enc in (:case, :traceif)
    meanfn = Symbol(:_ls_mean_, enc)
    @eval function $(Symbol(:_ls_model_, enc))(rng, xs)
        mu = $meanfn(_ls_sites(rng)..., xs)
        _, _ = ProbProg.sample(rng, _ls_y_sampler, mu; symbol=:y, logpdf=_ls_y_logpdf)
        return nothing
    end
    @eval function $(Symbol(:_ls_prior_, enc))(rng, xs)
        $meanfn(_ls_sites(rng)..., xs)
        return nothing
    end
end

function _ls_chain(rng, model, xs, ot, constrained, n, burn)
    trace, weight, _ = ProbProg.generate(
        rng, ot, model, xs; constrained_addresses=constrained
    )
    # Distinct objects: each accumulator is its own loop-carried value.
    a_m, a_sl, a_ic, a_am, a_ph = ntuple(_ -> weight .* 0.0, 5)
    s1, s2, i1, i2 = ntuple(_ -> weight .* 0.0, 4)
    @trace for it in 1:n
        trace, weight, acc, _ = ProbProg.mh(
            rng, trace, weight, model, xs; selection=ProbProg.select(ProbProg.Address(:m))
        )
        a_m = a_m .+ ifelse.(acc, 1.0, 0.0)
        trace, weight, acc, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            xs;
            selection=ProbProg.select(ProbProg.Address(:slope)),
        )
        a_sl = a_sl .+ ifelse.(acc, 1.0, 0.0)
        trace, weight, acc, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            xs;
            selection=ProbProg.select(ProbProg.Address(:intercept)),
        )
        a_ic = a_ic .+ ifelse.(acc, 1.0, 0.0)
        trace, weight, acc, _ = ProbProg.mh(
            rng, trace, weight, model, xs; selection=ProbProg.select(ProbProg.Address(:amp))
        )
        a_am = a_am .+ ifelse.(acc, 1.0, 0.0)
        trace, weight, acc, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            xs;
            selection=ProbProg.select(ProbProg.Address(:phase)),
        )
        a_ph = a_ph .+ ifelse.(acc, 1.0, 0.0)
        keep = ifelse(it > burn, 1.0, 0.0)
        sl = sum(trace[1:1, 2:2])
        ic = sum(trace[1:1, 3:3])
        s1 = s1 .+ keep .* sl
        s2 = s2 .+ keep .* sl .* sl
        i1 = i1 .+ keep .* ic
        i2 = i2 .+ keep .* ic .* ic
    end
    return a_m, a_sl, a_ic, a_am, a_ph, s1, s2, i1, i2
end

@testset "mh_delta_region_op_mean" begin
    xs = Reactant.to_rarray(_LS_X)
    for (k, enc) in enumerate((:case, :traceif))
        @testset "$enc posterior" begin
            model = getfield(@__MODULE__, Symbol(:_ls_model_, enc))
            obs = ProbProg.Constraint(:y => _LS_Y)
            constrained = ProbProg.extract_addresses(obs)
            ot = ProbProg.flatten_constraint(obs)
            N, burn = 5000, 1000
            rng = ReactantRNG(Reactant.to_rarray(UInt64[4242, k]))
            compiled, _ = ProbProg.with_trace() do
                @compile optimize = :probprog _ls_chain(
                    rng,
                    model,
                    xs,
                    ot,
                    constrained,
                    ConcreteRNumber(N),
                    ConcreteRNumber(burn),
                )
            end
            a_m, a_sl, a_ic, a_am, a_ph, s1, s2, i1, i2 = compiled(
                rng, model, xs, ot, constrained, ConcreteRNumber(N), ConcreteRNumber(burn)
            )
            # A chain ignoring the observation accepts all of these (correct: ~2%, 5%).
            @test 0 < _num(a_sl) / N < 0.5
            @test 0 < _num(a_ic) / N < 0.5
            kept = N - burn
            @test _num(s1) / kept ≈ _LS_E_SLOPE atol = 0.1
        end
    end
    for (k, enc) in enumerate((:case, :traceif))
        @testset "$enc prior invariance" begin
            model = getfield(@__MODULE__, Symbol(:_ls_prior_, enc))
            obs = ProbProg.Constraint(:slope => [0.0])
            constrained = ProbProg.extract_addresses(obs)
            ot = ProbProg.flatten_constraint(obs)
            N = 20000
            rng = ReactantRNG(Reactant.to_rarray(UInt64[777, k]))
            compiled, _ = ProbProg.with_trace() do
                @compile optimize = :probprog _ls_chain(
                    rng,
                    model,
                    xs,
                    ot,
                    constrained,
                    ConcreteRNumber(N),
                    ConcreteRNumber(0),
                )
            end
            a_m, a_sl, a_ic, a_am, a_ph, s1, s2, i1, i2 = compiled(
                rng, model, xs, ot, constrained, ConcreteRNumber(N), ConcreteRNumber(0)
            )
            # Nothing observed: every move is accepted; slope, intercept keep their priors.
            for a in (a_m, a_sl, a_ic, a_am, a_ph)
                @test _num(a) / N > 0.999
            end
            ms, mi = _num(s1) / N, _num(i1) / N
            @test ms ≈ 0.0 atol = 0.1
            @test _num(s2) / N - ms^2 ≈ 4.0 atol = 0.3
            @test mi ≈ 0.0 atol = 0.15
            @test _num(i2) / N - mi^2 ≈ 9.0 atol = 0.6
        end
    end
end
