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
