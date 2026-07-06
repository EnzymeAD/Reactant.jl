# Alternating single-site mh moves: SICM IR checks and per-step timing, SICM on vs off.
using Reactant, Random, LinearAlgebra, Printf, Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

const OUT = get(
    ENV,
    "E2_OUT",
    "/tmp/claude-140061798/-mnt-sbrantq-probprog/9007adff-5119-4ba4-8ced-2cd50c62b05e/scratchpad/sdd/e2-paper-artifacts",
)
mkpath(OUT)

_force(x) = x isa AbstractArray ? Array(x) : Float64(x)

import Reactant.XLA.PJRT: Future
function Base.wait(future::Future)::Nothing
    GC.@preserve future begin
        while !isready(future)
            GC.safepoint()
            yield()
        end
    end
    return nothing
end

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)
function normal_logpdf(x, μ, σ, _)
    return -length(x) * log(σ) - length(x) / 2 * log(2π) -
           sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end
function model_linreg(rng, xs)
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

function programmable_mh_linreg(rng, xs, num_iters, obs_tensor, constrained)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, model_linreg, xs; constrained_addresses=constrained
    )
    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model_linreg,
            xs;
            selection=ProbProg.select(ProbProg.Address(:slope)),
        )
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model_linreg,
            xs;
            selection=ProbProg.select(ProbProg.Address(:intercept)),
        )
    end
    return trace, weight
end

function model_scale_linreg_mvn(rng, xs, R_data)
    n = Base.length(xs)
    _, slope = ProbProg.sample(rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:slope)
    _, tau = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:tau)
    tau_2d = reshape(tau, 1, 1)
    K = (tau_2d .* tau_2d) .* R_data
    mu = slope .* xs
    _, obs = ProbProg.sample(rng, ProbProg.MultiNormal(mu, K, (n,)); symbol=:y)
    return obs
end

function programmable_mh_cov(rng, xs, R, num_iters, obs_tensor, constrained)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, model_scale_linreg_mvn, xs, R; constrained_addresses=constrained
    )
    acc = weight .* 0.0
    @trace for _ in 1:num_iters
        trace, weight, a1, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model_scale_linreg_mvn,
            xs,
            R;
            selection=ProbProg.select(ProbProg.Address(:slope)),
        )
        acc = acc .+ ifelse.(a1, 1.0, 0.0)
        trace, weight, a2, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model_scale_linreg_mvn,
            xs,
            R;
            selection=ProbProg.select(ProbProg.Address(:tau)),
        )
        acc = acc .+ ifelse.(a2, 1.0, 0.0)
    end
    return trace, weight, acc
end

function data_linreg()
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
    return (xs=xs, ys=ys)
end
function data_cov(n)
    xs = collect(range(-1.0, 1.0; length=n))
    R = [i == j ? 1.0 : 0.3^abs(i - j) for i in 1:n, j in 1:n]
    slope_true = -2.0
    y = slope_true .* xs .+ 0.15 .* collect(range(0.4, -0.4; length=n))
    return (xs=xs, R=R, y=y)
end

count_matches(re, s) = length(collect(eachmatch(re, s)))

function while_metrics(hlo)
    idx = findlast("stablehlo.while", hlo)
    pre = idx === nothing ? hlo : hlo[1:first(idx)]
    body = idx === nothing ? "" : hlo[first(idx):end]
    return (
        chol_pre=count("cholesky", pre),
        chol_body=count("cholesky", body),
        tri_pre=count("triangular", pre),
        tri_body=count("triangular", body),
        sqrt_body=count_matches(r"stablehlo\.sqrt", body),
        has_while=idx !== nothing,
    )
end

function evidence_rebuttal()
    d = data_linreg()
    xs = d.xs
    obs = ProbProg.Constraint(:ys => d.ys)
    constrained = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)
    N = ConcreteRNumber(2)

    code_inline, _ = ProbProg.with_trace() do
        Reactant.@code_hlo optimize = "inline-infer-regions" programmable_mh_linreg(
            rng, xs, N, obs_tensor, constrained
        )
    end
    s_inline = repr(code_inline)
    n_region = count_matches(r"impulse\.infer_region", s_inline)
    n_flat = count_matches(r"impulse\.infer\b", s_inline)
    n_flat = n_region == 0 ? n_flat : count_matches(r"impulse\.infer[^_]", s_inline)
    write(joinpath(OUT, "rebuttal_inline_infer_regions.mlir"), s_inline)

    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = false
    hlo_on = repr(
        Reactant.@code_hlo optimize = :probprog programmable_mh_linreg(
            rng, xs, N, obs_tensor, constrained
        )
    )
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = true
    hlo_off = repr(
        Reactant.@code_hlo optimize = :probprog programmable_mh_linreg(
            rng, xs, N, obs_tensor, constrained
        )
    )
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = false
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = true
    write(joinpath(OUT, "rebuttal_hlo_sicm_on.mlir"), hlo_on)
    write(joinpath(OUT, "rebuttal_hlo_sicm_off.mlir"), hlo_off)

    function op_counts(hlo)
        idx = findlast("stablehlo.while", hlo)
        pre = idx === nothing ? hlo : hlo[1:first(idx)]
        body = idx === nothing ? "" : hlo[first(idx):end]
        pre_add = count_matches(r"stablehlo\.(add|subtract|multiply|divide)", pre)
        body_add = count_matches(r"stablehlo\.(add|subtract|multiply|divide)", body)
        return (
            pre_arith=pre_add,
            body_arith=body_add,
            total=count_matches(r"stablehlo\.\w+", hlo),
        )
    end
    on = op_counts(hlo_on)
    off = op_counts(hlo_off)

    @info "[rebuttal] inline-infer-regions: infer_region=$n_region flat-infer=$n_flat"
    @info "[rebuttal] final HLO SICM ON : body-arith=$(on.body_arith) prologue-arith=$(on.pre_arith) total-hlo-ops=$(on.total)"
    @info "[rebuttal] final HLO SICM OFF: body-arith=$(off.body_arith) prologue-arith=$(off.pre_arith) total-hlo-ops=$(off.total)"
    sicm_neutral = (on.body_arith == off.body_arith) && (on.total == off.total)
    @info "[rebuttal] SICM-neutral (on==off in/around loop)? $sicm_neutral  (region-pipeline win here is LICM/CSE)"
    return (n_region=n_region, n_flat=n_flat, on=on, off=off, sicm_neutral=sicm_neutral)
end

function evidence_cov(n)
    d = data_cov(n)
    xs = Reactant.to_rarray(d.xs)
    R = Reactant.to_rarray(d.R)
    obs = ProbProg.Constraint(:y => d.y)
    constrained = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)
    seed = Reactant.to_rarray(UInt64[2, 11])
    rng = ReactantRNG(seed)
    N = ConcreteRNumber(2)

    # `lower-impulse-to-stablehlo{linalg-only}`, the elementwise tail via
    _co = Reactant.Compiler.CompileOptions()
    _op = Reactant.Compiler.optimization_passes(_co; sroa=true, backend="cpu")
    _op2 = Reactant.Compiler.optimization_passes(_co; sroa=false, backend="cpu")
    region_on = join(
        [
            "mark-func-memory-effects",
            _op,
            "enzyme-batch",
            _op2,
            "inline-infer-regions",
            "canonicalize,cse",
            "lower-impulse-to-stablehlo{linalg-only=true backend=cpu}",
            "arith-raise{stablehlo=true}",
            "canonicalize,cse",
            "sicm",
            "canonicalize,cse",
        ],
        ",",
    )
    # Counts chol above infer_region; XLA LICM lifts it above the while later.
    chol_above_region = -1
    try
        ir_on = repr(
            Reactant.@code_hlo optimize = region_on programmable_mh_cov(
                rng, xs, R, N, obs_tensor, constrained
            )
        )
        write(joinpath(OUT, "cov_region_sicm_on.mlir"), ir_on)
        ridx = findfirst("impulse.infer_region", ir_on)
        chol_above_region =
            ridx === nothing ? -1 : count("stablehlo.cholesky", ir_on[1:first(ridx)])
    catch e
        @warn "cov region-on IR capture failed" exception = (e,)
    end

    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = false
    hlo_on = repr(
        Reactant.@code_hlo optimize = :probprog programmable_mh_cov(
            rng, xs, R, N, obs_tensor, constrained
        )
    )
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = true
    hlo_off = repr(
        Reactant.@code_hlo optimize = :probprog programmable_mh_cov(
            rng, xs, R, N, obs_tensor, constrained
        )
    )
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = false
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = true
    write(joinpath(OUT, "cov_hlo_sicm_on.mlir"), hlo_on)
    write(joinpath(OUT, "cov_hlo_sicm_off.mlir"), hlo_off)

    mon = while_metrics(hlo_on)
    moff = while_metrics(hlo_off)
    @info "[cov] region-level: cholesky hoisted OUT of the density region (SICM on) = $chol_above_region"
    @info "[cov] final-HLO SICM ON : cholesky before-while=$(mon.chol_pre) in-while=$(mon.chol_body) | triangular in-while=$(mon.tri_body) | sqrt in-while=$(mon.sqrt_body)"
    @info "[cov] final-HLO SICM OFF: cholesky before-while=$(moff.chol_pre) in-while=$(moff.chol_body) | triangular in-while=$(moff.tri_body) | sqrt in-while=$(moff.sqrt_body)"
    return (on=mon, off=moff, chol_above_region=chol_above_region)
end

function run_arm(
    prog, rng, args...; disable_opt::Bool, disable_sicm::Bool, num_steps::Int, reps::Int=8
)
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = disable_opt
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = disable_sicm
    compiled, tt = ProbProg.with_trace() do
        @compile optimize = :probprog prog(rng, args...)
    end
    out = compiled(rng, args...)
    for o in out
        _force(o)
    end
    best = Inf
    for _ in 1:reps
        t = @elapsed begin
            o = compiled(rng, args...)
            for x in o
                _force(x)
            end
        end
        best = min(best, t)
    end
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = true
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = false
    return (wall=best, per_step=best / num_steps, out=out, tt=tt)
end

function benchmark_cov(n, num_iters)
    d = data_cov(n)
    xs = Reactant.to_rarray(d.xs)
    R = Reactant.to_rarray(d.R)
    obs = ProbProg.Constraint(:y => d.y)
    constrained = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)
    N = ConcreteRNumber(num_iters)
    num_steps = 2 * num_iters   # two mh moves per iteration

    seed() = ReactantRNG(Reactant.to_rarray(UInt64[2, 11]))

    sicm = run_arm(
        programmable_mh_cov,
        seed(),
        xs,
        R,
        N,
        obs_tensor,
        constrained;
        disable_opt=false,
        disable_sicm=false,
        num_steps=num_steps,
    )
    region_nosicm = run_arm(
        programmable_mh_cov,
        seed(),
        xs,
        R,
        N,
        obs_tensor,
        constrained;
        disable_opt=false,
        disable_sicm=true,
        num_steps=num_steps,
    )
    noopt = run_arm(
        programmable_mh_cov,
        seed(),
        xs,
        R,
        N,
        obs_tensor,
        constrained;
        disable_opt=true,
        disable_sicm=false,
        num_steps=num_steps,
    )

    trS, wS, accS = sicm.out
    trN, wN, accN = noopt.out
    traceS = ProbProg.unflatten_trace(trS, wS, sicm.tt.entries, ())
    traceN = ProbProg.unflatten_trace(trN, wN, noopt.tt.entries, ())
    tauS = traceS.choices[:tau][1, 1]
    tauN = traceN.choices[:tau][1, 1]
    slopeS = traceS.choices[:slope][1, 1]
    slopeN = traceN.choices[:slope][1, 1]

    return (
        n=n,
        num_iters=num_iters,
        num_steps=num_steps,
        sicm=sicm,
        region_nosicm=region_nosicm,
        noopt=noopt,
        accS=Float64(accS),
        accN=Float64(accN),
        tauS=tauS,
        tauN=tauN,
        slopeS=slopeS,
        slopeN=slopeN,
    )
end

println("==== E2: programmable-MH evidence + benchmark ====")
reb = evidence_rebuttal()
covir = evidence_cov(8)
bench = benchmark_cov(8, 500)

sp_flat = bench.noopt.per_step / bench.sicm.per_step
sp_isol = bench.region_nosicm.per_step / bench.sicm.per_step

open(joinpath(OUT, "mh_benchmark.json"), "w") do io
    println(io, "{")
    println(io, "  \"rebuttal\": {")
    println(
        io,
        "    \"infer_region_count\": $(reb.n_region), \"flat_infer_count\": $(reb.n_flat),",
    )
    println(io, "    \"sicm_neutral\": $(reb.sicm_neutral),")
    println(
        io,
        "    \"body_arith_sicm_on\": $(reb.on.body_arith), \"body_arith_sicm_off\": $(reb.off.body_arith),",
    )
    println(io, "    \"prologue_arith_sicm_on\": $(reb.on.pre_arith)")
    println(io, "  },")
    println(io, "  \"cov_ir\": {")
    println(
        io,
        "    \"region_chol_hoisted_out_of_density_sicm_on\": $(covir.chol_above_region),",
    )
    println(
        io,
        "    \"chol_in_while_sicm_on\": $(covir.on.chol_body), \"chol_in_while_sicm_off\": $(covir.off.chol_body),",
    )
    println(
        io,
        "    \"chol_before_while_sicm_on\": $(covir.on.chol_pre), \"chol_before_while_sicm_off\": $(covir.off.chol_pre),",
    )
    println(
        io,
        "    \"triangular_in_while_sicm_on\": $(covir.on.tri_body), \"triangular_in_while_sicm_off\": $(covir.off.tri_body),",
    )
    println(
        io,
        "    \"sqrt_in_while_sicm_on\": $(covir.on.sqrt_body), \"sqrt_in_while_sicm_off\": $(covir.off.sqrt_body)",
    )
    println(io, "  },")
    println(io, "  \"benchmark\": {")
    println(
        io,
        "    \"n\": $(bench.n), \"num_iters\": $(bench.num_iters), \"mh_steps\": $(bench.num_steps),",
    )
    println(
        io, "    \"impulse_sicm_per_step_us\": $(round(bench.sicm.per_step*1e6, digits=4)),"
    )
    println(
        io,
        "    \"impulse_region_nosicm_per_step_us\": $(round(bench.region_nosicm.per_step*1e6, digits=4)),",
    )
    println(
        io,
        "    \"impulse_noopt_flat_per_step_us\": $(round(bench.noopt.per_step*1e6, digits=4)),",
    )
    println(io, "    \"speedup_vs_noopt_flat\": $(round(sp_flat, digits=3)),")
    println(io, "    \"speedup_vs_region_nosicm\": $(round(sp_isol, digits=3)),")
    println(
        io,
        "    \"accept_count_sicm\": $(bench.accS), \"accept_count_noopt\": $(bench.accN),",
    )
    println(io, "    \"tau_sicm\": $(bench.tauS), \"tau_noopt\": $(bench.tauN),")
    println(io, "    \"slope_sicm\": $(bench.slopeS), \"slope_noopt\": $(bench.slopeN)")
    println(io, "  }")
    println(io, "}")
end

open(joinpath(OUT, "mh_benchmark_table.md"), "w") do io
    println(io, "| model | arm | per-step (µs) | speedup | notes |")
    println(io, "|---|---|---:|---:|---|")
    ps(x) = @sprintf("%.3f", x.per_step * 1e6)
    println(
        io,
        "| scale_linreg_mvn (n=$(bench.n)) | Impulse (SICM) | $(ps(bench.sicm)) | 1.00× | chol in-while: $(covir.on.chol_body); sqrt in-while: $(covir.on.sqrt_body) |",
    )
    println(
        io,
        "| scale_linreg_mvn (n=$(bench.n)) | Impulse (region, no SICM) | $(ps(bench.region_nosicm)) | $(@sprintf("%.2f", sp_isol))× | isolates SICM algebra |",
    )
    println(
        io,
        "| scale_linreg_mvn (n=$(bench.n)) | Impulse (no opt / flat) | $(ps(bench.noopt)) | $(@sprintf("%.2f", sp_flat))× | legacy impulse.mh; chol in-while: $(covir.off.chol_body) |",
    )
end

println("\n==== E2 SUMMARY ====")
@printf(
    "rebuttal: infer_region=%d flat=%d  SICM-neutral=%s\n",
    reb.n_region,
    reb.n_flat,
    reb.sicm_neutral
)
@printf(
    "cov IR  : chol in-while on=%d off=%d ; before-while on=%d off=%d ; sqrt in-while on=%d off=%d\n",
    covir.on.chol_body,
    covir.off.chol_body,
    covir.on.chol_pre,
    covir.off.chol_pre,
    covir.on.sqrt_body,
    covir.off.sqrt_body
)
@printf(
    "bench   : SICM=%.3fus  region-no-sicm=%.3fus  no-opt-flat=%.3fus  speedup(vs flat)=%.2fx (vs region-no-sicm)=%.2fx\n",
    bench.sicm.per_step * 1e6,
    bench.region_nosicm.per_step * 1e6,
    bench.noopt.per_step * 1e6,
    sp_flat,
    sp_isol
)
@printf(
    "posterior sanity: accepts SICM=%.0f no-opt=%.0f | tau SICM=%.10f no-opt=%.10f | slope SICM=%.10f no-opt=%.10f\n",
    bench.accS,
    bench.accN,
    bench.tauS,
    bench.tauN,
    bench.slopeS,
    bench.slopeN
)
@printf(
    "  |Δtau|=%.2e |Δslope|=%.2e |Δaccept|=%.0f\n",
    abs(bench.tauS - bench.tauN),
    abs(bench.slopeS - bench.slopeN),
    abs(bench.accS - bench.accN)
)
println("Artifacts in ", OUT)

using Test
@testset "E2 programmable-MH" begin
    @test reb.n_region == 2
    @test reb.n_flat == 0
    @test reb.sicm_neutral                       # plain-Normal density has no SICM target
    @test covir.on.sqrt_body >= 1
    @test covir.off.sqrt_body == 0
    @test covir.chol_above_region >= 1
    @test bench.accS == bench.accN
    @test bench.accS > 0
    @test abs(bench.tauS - bench.tauN) <= 1e-12
    @test abs(bench.slopeS - bench.slopeN) <= 1e-12
    @test bench.sicm.per_step <= bench.region_nosicm.per_step * 1.1
end
