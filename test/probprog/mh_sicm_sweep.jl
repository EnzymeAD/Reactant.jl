# n-sweep of mh_sicm_benchmark.jl: SICM makes the per-move density O(n^2) instead of O(n^3).
using Reactant, Random, LinearAlgebra, Printf, Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

const OUT = get(ENV, "SWEEP_OUT", "/mnt/sbrantq/probprog/oopsla/evidence/programmable_mh")
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

function data_cov(n)
    xs = collect(range(-1.0, 1.0; length=n))
    R = [i == j ? 1.0 : 0.3^abs(i - j) for i in 1:n, j in 1:n]
    slope_true = -2.0
    y = slope_true .* xs .+ 0.15 .* collect(range(0.4, -0.4; length=n))
    return (xs=xs, R=R, y=y)
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
    return (wall=best, per_step=best / num_steps, out=out)
end

const NUM_ITERS = 500
const NUM_STEPS = 2 * NUM_ITERS

results = Dict{Int,Any}()
for n in (8, 32, 64, 128, 256)
    d = data_cov(n)
    xs = Reactant.to_rarray(d.xs)
    R = Reactant.to_rarray(d.R)
    obs = ProbProg.Constraint(:y => d.y)
    constrained = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)
    N = ConcreteRNumber(NUM_ITERS)
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
        num_steps=NUM_STEPS,
    )
    nosicm = run_arm(
        programmable_mh_cov,
        seed(),
        xs,
        R,
        N,
        obs_tensor,
        constrained;
        disable_opt=false,
        disable_sicm=true,
        num_steps=NUM_STEPS,
    )
    accS = Float64(sicm.out[3])
    results[n] = (
        sicm_us=sicm.per_step * 1e6, nosicm_us=nosicm.per_step * 1e6, accepts=accS
    )
    @printf(
        "n=%-4d SICM %10.3f us | no-SICM %10.3f us | ratio %.2fx | accepts %.0f/%d\n",
        n,
        sicm.per_step * 1e6,
        nosicm.per_step * 1e6,
        nosicm.per_step / sicm.per_step,
        accS,
        NUM_STEPS
    )
end

open(joinpath(OUT, "impulse_sweep.json"), "w") do io
    println(io, "{")
    entries = String[]
    for n in sort(collect(keys(results)))
        r = results[n]
        push!(
            entries,
            """  "$n": {"sicm_per_move_us": $(round(r.sicm_us, digits=4)), "nosicm_per_move_us": $(round(r.nosicm_us, digits=4)), "accepts": $(r.accepts)}""",
        )
    end
    println(io, join(entries, ",\n"))
    println(io, "}")
end
println("Artifacts in ", OUT)
