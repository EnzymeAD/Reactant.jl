# SVI with SICM: region IR and final HLO dumps, and timing/ELBO vs NumPyro SVI.
using Reactant, Random, LinearAlgebra, Printf, Test, Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber
include(joinpath(@__DIR__, "common.jl"))

const OUT = get(
    ENV,
    "D3_OUT",
    "/tmp/claude-140061798/-mnt-sbrantq-probprog/9007adff-5119-4ba4-8ced-2cd50c62b05e/scratchpad/sdd/d3-evidence",
)
mkpath(OUT)

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

function model_scale_family_mvn(rng, R_data, y)
    n = Base.length(y)
    _, tau = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:tau)
    tau_2d = reshape(tau, 1, 1)
    K = (tau_2d .* tau_2d) .* R_data
    _, obs = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(n), K, (n,)); symbol=:y)
    return obs
end

function model_gp_pois_regr(rng, x, k, rho)
    n = Base.length(x)
    _, alpha = ProbProg.sample(rng, ProbProg.HalfNormal(2.0, (1,)); symbol=:alpha)
    _, f_tilde = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (n,)); symbol=:f_tilde)
    x_col = reshape(x, n, 1)
    x_row = reshape(x, 1, n)
    delta = x_col .- x_row
    rho_2d = reshape(rho, 1, 1)
    K_base = exp.(-0.5 .* (delta ./ rho_2d) .^ 2)
    alpha_2d = reshape(alpha, 1, 1)
    cov = (alpha_2d .* alpha_2d) .* K_base
    C = cholesky(cov)
    f = reshape(C.factors' * reshape(f_tilde, n, 1), n)
    _, obs = ProbProg.sample(rng, ProbProg.Poisson(exp.(f), (n,)); symbol=:k)
    return obs
end

_eye(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

function data_mvn(n)
    R = _eye(n)
    for i in 1:n, j in 1:n
        i != j && (R[i, j] = 0.3^abs(i - j))
    end
    y = collect(range(0.4, -0.4; length=n))
    return (R=R, y=y)
end
function data_gp(n)
    x = collect(range(1.0, Float64(n); length=n))
    rho = [1.0]
    k = Float64[(i % 3) + 1 for i in 1:n]
    return (x=x, rho=rho, k=k)
end

function spec_mvn(n)
    d = data_mvn(n)
    return (
        name="scale_family_mvn",
        n=n,
        model=model_scale_family_mvn,
        args=(Reactant.to_rarray(d.R), Reactant.to_rarray(d.y)),
        selection=ProbProg.select(ProbProg.Address(:tau)),
        constraint=ProbProg.Constraint(:tau => [1.0], :y => d.y),
        init_loc=Reactant.to_rarray(zeros(1, 1)),
        pos=1,
        data=d,
        numpyro_param=[:tau_auto_loc, :tau_auto_scale],
    )
end
function spec_gp(n)
    d = data_gp(n)
    return (
        name="gp_pois_regr",
        n=n,
        model=model_gp_pois_regr,
        args=(Reactant.to_rarray(d.x), Reactant.to_rarray(d.k), Reactant.to_rarray(d.rho)),
        selection=ProbProg.select(ProbProg.Address(:alpha), ProbProg.Address(:f_tilde)),
        constraint=ProbProg.Constraint(:alpha => [1.0], :f_tilde => zeros(n), :k => d.k),
        init_loc=Reactant.to_rarray(zeros(1, 1 + n)),
        pos=1 + n,
        data=d,
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
        ir_on = repr(Reactant.@code_hlo optimize = region_on prog(rng, lr))
        ir_off = repr(Reactant.@code_hlo optimize = region_off prog(rng, lr))
        write(joinpath(OUT, "$(spec.name)_region_sicm_on.mlir"), ir_on)
        write(joinpath(OUT, "$(spec.name)_region_sicm_off.mlir"), ir_off)
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
        cpre = count("cholesky", pre)
        cbody = count("cholesky", body)
        tpre = count("triangular", pre)
        tbody = count("triangular", body)
        return (cpre=cpre, cbody=cbody, tpre=tpre, tbody=tbody)
    end
    mon = while_metrics(hlo_on)
    moff = while_metrics(hlo_off)
    @info "[$(spec.name)] final-HLO SICM ON : cholesky before-while=$(mon.cpre) in-while=$(mon.cbody) | triangular before=$(mon.tpre) in=$(mon.tbody)"
    @info "[$(spec.name)] final-HLO SICM OFF: cholesky before-while=$(moff.cpre) in-while=$(moff.cbody) | triangular before=$(moff.tpre) in=$(moff.tbody)"
    return (on=mon, off=moff)
end

function run_impulse(spec, prog, rng, lr, num_steps; disable_sicm::Bool)
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = disable_sicm
    compiled = @compile optimize = :probprog prog(rng, lr)
    _, elbo0, loc0, _ = compiled(rng, lr)
    elbo = Array(elbo0)
    best = Inf
    for _ in 1:8
        t = @elapsed begin
            out = compiled(rng, lr)
            Array(out[2])
            Array(out[3])                  # force sync
        end
        best = min(best, t)
    end
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = false
    return (elbo=vec(elbo), wall=best, per_step=best / num_steps)
end

function build_numpyro_model(spec)
    dist = pyimport("numpyro.distributions")
    numpyro_mod = pyimport("numpyro")
    jnp = pyimport("jax.numpy")
    if spec.name == "scale_family_mvn"
        Rpy = jnp.array(spec.data.R)
        ypy = jnp.array(spec.data.y)
        n = spec.n
        return pyfunc(
            function ()
                tau = numpyro_mod.sample("tau", dist.HalfNormal(1.0))
                cov = jnp.multiply(jnp.multiply(tau, tau), Rpy)
                numpyro_mod.sample(
                    "y",
                    dist.MultivariateNormal(jnp.zeros(n); covariance_matrix=cov);
                    obs=ypy,
                )
                return nothing
            end;
            name="model",
        )
    else
        xpy = jnp.array(spec.data.x)
        kpy = jnp.array(spec.data.k)
        rho = spec.data.rho[1]
        n = spec.n
        xcol = jnp.reshape(xpy, pytuple((n, 1)))
        xrow = jnp.reshape(xpy, pytuple((1, n)))
        return pyfunc(
            function ()
                alpha = numpyro_mod.sample("alpha", dist.HalfNormal(2.0))
                f_tilde = numpyro_mod.sample(
                    "f_tilde", dist.Normal(0, 1).expand(pytuple((n,)))
                )
                delta = jnp.subtract(xcol, xrow)
                Kb = jnp.exp(jnp.multiply(-0.5, jnp.power(jnp.divide(delta, rho), 2)))
                cov = jnp.multiply(jnp.multiply(alpha, alpha), Kb)
                L = jnp.linalg.cholesky(cov)
                f = jnp.matmul(L, f_tilde)
                numpyro_mod.sample("k", dist.Poisson(; rate=jnp.exp(f)); obs=kpy)
                return nothing
            end;
            name="model",
        )
    end
end

function run_numpyro(spec, num_steps, lr)
    model = build_numpyro_model(spec)
    rng_key = seed_to_rbg_key(UInt64[0, 0])
    init_vals = if spec.name == "scale_family_mvn"
        Dict(:tau => 1.0)
    else
        Dict(:alpha => 1.0, :f_tilde => zeros(spec.n))
    end
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
        t = @elapsed run_numpyro_svi(; kw...)
        best = min(best, t)
    end
    return (elbo=py1.elbo_traj, wall=best, per_step=best / num_steps)
end

const NUM_STEPS = 300
rows = Any[]
for spec in (spec_mvn(8), spec_gp(8))
    @info "==== $(spec.name) (n=$(spec.n), pos=$(spec.pos)) ===="
    seed = Reactant.to_rarray(UInt64[0, 0])
    rng = ReactantRNG(seed)
    lr = ConcreteRNumber(0.01)
    prog = make_svi_program(spec, NUM_STEPS)

    ir = capture_ir(spec, prog, rng, lr)
    imp_on = run_impulse(spec, prog, rng, lr, NUM_STEPS; disable_sicm=false)
    imp_off = run_impulse(spec, prog, rng, lr, NUM_STEPS; disable_sicm=true)

    np = nothing
    if check_numpyro_available()
        try
            np = run_numpyro(spec, NUM_STEPS, 0.01)
        catch e
            @warn "NumPyro SVI failed for $(spec.name)" exception = (e, catch_backtrace())
        end
    end

    elbo_diff = np === nothing ? NaN : maximum(abs.(imp_on.elbo .- Float64.(np.elbo)))
    push!(
        rows,
        (
            name=spec.name,
            n=spec.n,
            ir=ir,
            imp_on=imp_on,
            imp_off=imp_off,
            np=np,
            elbo_diff=elbo_diff,
        ),
    )
    @info "[$(spec.name)] ELBO agreement (Impulse vs NumPyro) max|Δ| = $elbo_diff"
    @info "[$(spec.name)] final ELBO: impulse=$(imp_on.elbo[end]) numpyro=$(np === nothing ? NaN : np.elbo[end])"
end

open(joinpath(OUT, "svi_benchmark.json"), "w") do io
    println(io, "{")
    for (i, r) in enumerate(rows)
        npq(f) = r.np === nothing ? "null" : string(f(r.np))
        print(
            io,
            "  \"$(r.name)\": {",
            "\"n\": $(r.n), ",
            "\"impulse_sicm_per_step_us\": $(round(r.imp_on.per_step*1e6, digits=3)), ",
            "\"impulse_noopt_per_step_us\": $(round(r.imp_off.per_step*1e6, digits=3)), ",
            "\"numpyro_per_step_us\": $(npq(x->round(x.per_step*1e6, digits=3))), ",
            "\"impulse_speedup_vs_noopt\": $(round(r.imp_off.per_step/r.imp_on.per_step, digits=3)), ",
            "\"impulse_final_elbo\": $(r.imp_on.elbo[end]), ",
            "\"numpyro_final_elbo\": $(npq(x->x.elbo[end])), ",
            "\"elbo_max_abs_diff\": $(r.elbo_diff), ",
            "\"hlo_chol_in_while_sicm_on\": $(r.ir.on.cbody), ",
            "\"hlo_chol_in_while_sicm_off\": $(r.ir.off.cbody), ",
            "\"hlo_chol_before_while_sicm_on\": $(r.ir.on.cpre)",
            "}",
            i < length(rows) ? "," : "",
            "\n",
        )
    end
    println(io, "}")
end

open(joinpath(OUT, "svi_benchmark_table.md"), "w") do io
    println(io, "| model | framework | per-step (µs) | final ELBO | notes |")
    println(io, "|---|---|---:|---:|---|")
    for r in rows
        pon = @sprintf("%.2f", r.imp_on.per_step * 1e6)
        poff = @sprintf("%.2f", r.imp_off.per_step * 1e6)
        sp = @sprintf("%.2f", r.imp_off.per_step / r.imp_on.per_step)
        eon = @sprintf("%.4f", r.imp_on.elbo[end])
        println(
            io,
            "| $(r.name) | Impulse (SICM) | $pon | $eon | chol in-while: on=$(r.ir.on.cbody) off=$(r.ir.off.cbody) |",
        )
        println(io, "| $(r.name) | Impulse (no opt) | $poff | $eon | $(sp)× slower/step |")
        if r.np !== nothing
            pnp = @sprintf("%.2f", r.np.per_step * 1e6)
            enp = @sprintf("%.4f", r.np.elbo[end])
            ed = @sprintf("%.2e", r.elbo_diff)
            println(
                io,
                "| $(r.name) | NumPyro SVI | $pnp | $enp | ELBO max|Δ| vs Impulse = $ed |",
            )
        end
    end
end

println("\n==== D3 SVI BENCHMARK ====")
for r in rows
    @printf(
        "%-20s Impulse(SICM)=%.2fus  Impulse(noopt)=%.2fus  speedup=%.2fx  NumPyro=%s us  ELBO|Δ|=%.2e\n",
        r.name,
        r.imp_on.per_step * 1e6,
        r.imp_off.per_step * 1e6,
        r.imp_off.per_step / r.imp_on.per_step,
        r.np === nothing ? "NA" : @sprintf("%.2f", r.np.per_step * 1e6),
        r.elbo_diff
    )
end
println("Artifacts in ", OUT)
