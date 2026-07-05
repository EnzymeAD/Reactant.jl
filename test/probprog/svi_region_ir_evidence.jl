# Region-level IR after sicm, before expand-impulse, for the two SICM VI models.
using Reactant, Random, LinearAlgebra
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

const OUT = get(
    ENV,
    "D3_OUT",
    "/tmp/claude-140061798/-mnt-sbrantq-probprog/9007adff-5119-4ba4-8ced-2cd50c62b05e/scratchpad/sdd/d3-evidence",
)
mkpath(OUT)

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

function specs(n)
    R = _eye(n)
    for i in 1:n, j in 1:n
        i != j && (R[i, j] = 0.3^abs(i - j))
    end
    y = collect(range(0.4, -0.4; length=n))
    mvn = (
        name="scale_family_mvn",
        model=model_scale_family_mvn,
        args=(Reactant.to_rarray(R), Reactant.to_rarray(y)),
        sel=ProbProg.select(ProbProg.Address(:tau)),
        cons=ProbProg.Constraint(:tau => [1.0], :y => y),
        init=Reactant.to_rarray(zeros(1, 1)),
    )
    x = collect(range(1.0, Float64(n); length=n))
    rho = [1.0]
    k = Float64[(i % 3) + 1 for i in 1:n]
    gp = (
        name="gp_pois_regr",
        model=model_gp_pois_regr,
        args=(Reactant.to_rarray(x), Reactant.to_rarray(k), Reactant.to_rarray(rho)),
        sel=ProbProg.select(ProbProg.Address(:alpha), ProbProg.Address(:f_tilde)),
        cons=ProbProg.Constraint(:alpha => [1.0], :f_tilde => zeros(n), :k => k),
        init=Reactant.to_rarray(zeros(1, 1 + n)),
    )
    return (mvn, gp)
end

function prog_of(spec, num_steps)
    cons_addrs = ProbProg.extract_addresses(spec.cons)
    obs_flat = Float64[]
    for a in cons_addrs
        append!(obs_flat, vec(spec.cons[a]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))
    function prog(rng, step_size)
        trace0, _, _ = ProbProg.generate(
            rng, obs_tensor, spec.model, spec.args...; constrained_addresses=cons_addrs
        )
        t, e, _, s = ProbProg.svi(
            rng,
            trace0,
            spec.model,
            spec.args...;
            selection=spec.sel,
            guide=:AutoNormal,
            optimizer=:Adam,
            step_size,
            num_steps,
            num_particles=1,
            init_scale=0.1,
            initial_position=spec.init,
        )
        return t, e, s.loc, s.scale
    end
    return prog
end

const PREFIX = "mark-func-memory-effects,enzyme-batch,inline-infer-regions,canonicalize,cse"
const P_ON = PREFIX * ",sicm"
const P_OFF = PREFIX * ",sicm{max-iterations=0}"

for spec in specs(8)
    seed = Reactant.to_rarray(UInt64[0, 0])
    rng = ReactantRNG(seed)
    lr = ConcreteRNumber(0.01)
    prog = prog_of(spec, 300)
    for (tag, pl) in (("on", P_ON), ("off", P_OFF))
        try
            ir = repr(Reactant.@code_hlo optimize = pl prog(rng, lr))
            write(joinpath(OUT, "$(spec.name)_region_sicm_$tag.mlir"), ir)
            ri = findfirst("impulse.infer_region", ir)
            pre = ri === nothing ? ir : ir[1:first(ri)]
            in_region = ri === nothing ? "" : ir[first(ri):end]
            @info "[$(spec.name) region SICM $tag] chol above-infer_region=$(count("cholesky", pre)) inside=$(count("cholesky", in_region)) | tri above=$(count("triangular", pre)) inside=$(count("triangular", in_region)) | bytes=$(length(ir))"
        catch e
            @warn "region IR $(spec.name) $tag failed" exception = (e,)
        end
    end
end
println("region IR done -> ", OUT)
