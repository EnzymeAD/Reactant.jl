# SVI on the Scotland lip-cancer Leroux CAR, SICM on/off, vs NumPyro SVI.
using Reactant, Random, LinearAlgebra, Printf, Test, Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber
include(joinpath(@__DIR__, "common.jl"))

const BENCH = joinpath(@__DIR__, "..", "..", "benchmark", "probprog")
const OUT = get(ENV, "D3_OUT", joinpath(tempdir(), "sbrantq-svi-real-data"))
mkpath(OUT)
const MODEL_KEY = get(ENV, "SVI_REAL_DATA_MODEL", "car_leroux_scotland")

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

include(joinpath(BENCH, "sicm", "car_leroux.jl"))

function load_json(path)
    json = pyimport("json")
    return pyconvert(Any, json.loads(read(path, String)))
end

function spec_car_leroux_scotland()
    d = load_json(
        joinpath(BENCH, "ablation", "sicm_roster", "data", "car_leroux_scotland.json")
    )
    s = setup(d)
    n = Int(d["attrs"]["n"])
    R = Array(s.model_args[1])
    log_E = Float64.(d["log_E"])
    y = Float64.(d["Y"])
    return (
        name="car_leroux_scotland",
        n=n,
        model=s.model_fn,
        args=s.model_args,
        selection=s.selection,
        constraint=build_constraint(d, nothing),   # alpha 0, sigma_phi 1, rho 0.5, phi 0
        init_loc=Reactant.to_rarray(zeros(1, s.position_size)),
        pos=s.position_size,
        data=(R=R, log_E=log_E, y=y),
        numpyro_param=[:alpha_auto_loc, :alpha_auto_scale],
    )
end

function make_svi_program(spec, num_steps)
    cons_addrs = ProbProg.extract_addresses(spec.constraint)
    obs_flat = Float64[]
    for a in cons_addrs
        append!(obs_flat, vec(spec.constraint[a]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))
    function prog(rng, step_size)
        trace0, _, _ = ProbProg.generate(
            rng, obs_tensor, spec.model, spec.args...; constrained_addresses=cons_addrs
        )
        trace, elbo, _, state = ProbProg.svi(
            rng,
            trace0,
            spec.model,
            spec.args...;
            selection=spec.selection,
            guide=:AutoNormal,
            optimizer=:Adam,
            step_size,
            num_steps,
            num_particles=1,
            init_scale=0.1,
            initial_position=spec.init_loc,
        )
        return trace, elbo, state.loc, state.scale
    end
    return prog
end

function capture_ir(spec, prog, rng, lr)
    region_on = "inline-infer-regions,canonicalize,cse,sicm"
    region_off = "inline-infer-regions,canonicalize,cse,sicm{max-iterations=0}"
    try
        write(
            joinpath(OUT, "$(spec.name)_region_sicm_on.mlir"),
            repr(Reactant.@code_hlo optimize = region_on prog(rng, lr)),
        )
        write(
            joinpath(OUT, "$(spec.name)_region_sicm_off.mlir"),
            repr(Reactant.@code_hlo optimize = region_off prog(rng, lr)),
        )
    catch e
        @warn "region-level IR capture failed for $(spec.name)" exception = (e,)
    end
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = false
    hlo_on = repr(Reactant.@code_hlo optimize = :probprog prog(rng, lr))
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = true
    hlo_off = repr(Reactant.@code_hlo optimize = :probprog prog(rng, lr))
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = false
    write(joinpath(OUT, "$(spec.name)_hlo_sicm_on.mlir"), hlo_on)
    write(joinpath(OUT, "$(spec.name)_hlo_sicm_off.mlir"), hlo_off)
    function while_metrics(hlo)
        idx = findfirst("stablehlo.while", hlo)
        pre = idx === nothing ? hlo : hlo[1:first(idx)]
        body = idx === nothing ? "" : hlo[first(idx):end]
        return (
            cpre=count("cholesky", pre),
            cbody=count("cholesky", body),
            epre=count("syevd", pre),
            ebody=count("syevd", body),
        )
    end
    mon = while_metrics(hlo_on)
    moff = while_metrics(hlo_off)
    @info "[$(spec.name)] final-HLO SICM ON : cholesky before-while=$(mon.cpre) in-while=$(mon.cbody) | syevd before=$(mon.epre) in=$(mon.ebody)"
    @info "[$(spec.name)] final-HLO SICM OFF: cholesky before-while=$(moff.cpre) in-while=$(moff.cbody) | syevd before=$(moff.epre) in=$(moff.ebody)"
    return (on=mon, off=moff)
end

function run_impulse(spec, prog, rng, lr, num_steps; disable_sicm::Bool)
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = disable_sicm
    compiled = @compile optimize = :probprog prog(rng, lr)
    # Fresh rng: the thunk advances the seed in place.
    rng_fresh = ReactantRNG(Reactant.to_rarray(UInt64[0, 0]))
    _, elbo0, loc0, scale0 = compiled(rng_fresh, lr)
    elbo = Array(elbo0)
    times = Float64[]
    for _ in 1:8
        t = @elapsed begin
            out = compiled(rng, lr)
            Array(out[2])
            Array(out[3])                  # force sync
        end
        push!(times, t)
    end
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = false
    best = minimum(times)
    return (
        elbo=vec(elbo),
        wall=best,
        per_step=best / num_steps,
        times=times,
        loc=vec(Array(loc0)),
        scale=vec(Array(scale0)),
    )
end

function build_numpyro_model(spec)
    dist = pyimport("numpyro.distributions")
    numpyro_mod = pyimport("numpyro")
    jnp = pyimport("jax.numpy")
    n = spec.n
    R = jnp.array(spec.data.R)
    log_E = jnp.array(spec.data.log_E)
    y = jnp.array(spec.data.y)
    return pyfunc(
        function ()
            alpha = numpyro_mod.sample("alpha", dist.Normal(0.0, 10.0))
            sigma_phi = numpyro_mod.sample("sigma_phi", dist.HalfNormal(1.0))
            rho = numpyro_mod.sample("rho", dist.Uniform(0.0, 1.0))
            tau = jnp.divide(1.0, jnp.power(sigma_phi, 2))
            P = jnp.multiply(
                tau,
                jnp.add(
                    jnp.multiply(rho, R), jnp.multiply(jnp.subtract(1.0, rho), jnp.eye(n))
                ),
            )
            phi = numpyro_mod.sample(
                "phi", dist.MultivariateNormal(; loc=jnp.zeros(n), precision_matrix=P)
            )
            numpyro_mod.sample(
                "y", dist.Poisson(jnp.exp(jnp.add(jnp.add(log_E, alpha), phi))); obs=y
            )
            return nothing
        end;
        name="model",
    )
end

function run_numpyro(spec, num_steps, lr)
    model = build_numpyro_model(spec)
    rng_key = seed_to_rbg_key(UInt64[0, 0])
    jnp = pyimport("jax.numpy")
    # Constrained values whose unconstrained images are Impulse's loc = 0.
    init_vals = Dict(
        :alpha => 0.0, :sigma_phi => 1.0, :rho => 0.5, :phi => jnp.array(zeros(spec.n))
    )
    kw = (;
        model=model,
        rng_key=rng_key,
        model_args=(),
        init_loc_values=init_vals,
        init_scale=0.1,
        num_steps=num_steps,
        step_size=lr,
        optimizer=:Adam,
        num_particles=1,
        param_names=spec.numpyro_param,
    )
    py1 = run_numpyro_svi(; kw...)
    best = Inf
    for _ in 1:4
        best = min(best, @elapsed run_numpyro_svi(; kw...))
    end
    return (elbo=py1.elbo_traj, wall=best, per_step=best / num_steps)
end

const NUM_STEPS = parse(Int, get(ENV, "SVI_REAL_DATA_STEPS", "300"))
spec = if MODEL_KEY == "car_leroux_scotland"
    spec_car_leroux_scotland()
else
    error("unknown SVI_REAL_DATA_MODEL=$MODEL_KEY")
end
@info "==== $(spec.name) (n=$(spec.n), pos=$(spec.pos)), $(NUM_STEPS) steps ===="
seed = Reactant.to_rarray(UInt64[0, 0]);
rng = ReactantRNG(seed);
lr = ConcreteRNumber(0.01)
prog = make_svi_program(spec, NUM_STEPS)

ir = if get(ENV, "SVI_REAL_DATA_SKIP_IR", "0") == "1"
    nothing
else
    capture_ir(spec, prog, rng, lr)
end
imp_on = run_impulse(spec, prog, rng, lr, NUM_STEPS; disable_sicm=false)
imp_off = run_impulse(spec, prog, rng, lr, NUM_STEPS; disable_sicm=true)

np = nothing
if get(ENV, "SVI_REAL_DATA_SKIP_NUMPYRO", "0") != "1" && check_numpyro_available()
    try
        global np = run_numpyro(spec, NUM_STEPS, 0.01)
    catch e
        @warn "NumPyro SVI failed for $(spec.name)" exception = (e, catch_backtrace())
    end
end
npelbo = np === nothing ? nothing : Float64.(collect(np.elbo))
elbo_diff = npelbo === nothing ? NaN : maximum(abs.(imp_on.elbo .- npelbo))
elbo_rel = npelbo === nothing ? NaN : maximum(abs.(imp_on.elbo .- npelbo) ./ abs.(npelbo))
onoff_diff = maximum(abs.(imp_on.elbo .- imp_off.elbo))

open(joinpath(OUT, "svi_benchmark.json"), "w") do io
    fmtv(v) = "[" * join(string.(v), ", ") * "]"
    println(io, "{\"$(spec.name)\": {")
    println(
        io,
        "  \"n\": $(spec.n), \"position_size\": $(spec.pos), \"num_steps\": $(NUM_STEPS),",
    )
    println(io, "  \"impulse_sicm_per_step_us\": $(round(imp_on.per_step*1e6, digits=3)),")
    println(
        io, "  \"impulse_noopt_per_step_us\": $(round(imp_off.per_step*1e6, digits=3)),"
    )
    println(io, "  \"impulse_sicm_call_times_s\": $(fmtv(imp_on.times)),")
    println(io, "  \"impulse_noopt_call_times_s\": $(fmtv(imp_off.times)),")
    println(
        io,
        "  \"numpyro_per_step_us\": $(np === nothing ? "null" : round(np.per_step*1e6, digits=3)),",
    )
    println(
        io,
        "  \"impulse_speedup_vs_noopt\": $(round(imp_off.per_step/imp_on.per_step, digits=3)),",
    )
    println(io, "  \"impulse_final_elbo\": $(imp_on.elbo[end]),")
    println(io, "  \"numpyro_final_elbo\": $(npelbo === nothing ? "null" : npelbo[end]),")
    println(io, "  \"elbo_max_abs_diff\": $(elbo_diff),")
    println(io, "  \"elbo_max_rel_diff\": $(elbo_rel),")
    println(io, "  \"elbo_sicm_on_vs_off_max_abs_diff\": $(onoff_diff),")
    if ir !== nothing
        println(
            io,
            "  \"hlo_chol_in_while_sicm_on\": $(ir.on.cbody), \"hlo_chol_in_while_sicm_off\": $(ir.off.cbody),",
        )
        println(
            io,
            "  \"hlo_syevd_before_while_sicm_on\": $(ir.on.epre), \"hlo_syevd_in_while_sicm_on\": $(ir.on.ebody),",
        )
    end
    println(io, "  \"impulse_elbo\": $(fmtv(imp_on.elbo)),")
    println(io, "  \"impulse_noopt_elbo\": $(fmtv(imp_off.elbo)),")
    println(io, "  \"numpyro_elbo\": $(npelbo === nothing ? "null" : fmtv(npelbo))")
    println(io, "}}")
end

println("\n==== SVI SICM REAL-DATA BENCHMARK ====")
@printf(
    "%-22s Impulse(SICM)=%.2fus  Impulse(noopt)=%.2fus  speedup=%.2fx  NumPyro=%s us  ELBO|d|=%.3e (rel %.3e)  on/off|d|=%.3e\n",
    spec.name,
    imp_on.per_step * 1e6,
    imp_off.per_step * 1e6,
    imp_off.per_step / imp_on.per_step,
    np === nothing ? "NA" : @sprintf("%.2f", np.per_step * 1e6),
    elbo_diff,
    elbo_rel,
    onoff_diff
)
println("Artifacts in ", OUT)
