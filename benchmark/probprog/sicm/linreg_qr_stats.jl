# linreg_qr_stats.jl
#
# Standalone parity + timing harness for the GaussianSufficientStats (QR) SICM
# rule. Bayesian linear regression with a Gaussian likelihood
#   w     ~ Normal(0, 1, (D,))
#   sigma ~ LogNormal(0, 1)
#   y[i]  ~ Normal((X w)[i], sigma),  i = 1..N
# run under NUTS. The likelihood emits the sum-of-squares residual chain the QR
# rule targets, so SICM-on == QR-rule-on for this model (no covariance-centric
# rule fires; the prior is a plain Normal).
#
# Usage (from benchmark/probprog):
#   JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=. sicm/linreg_qr_stats.jl <mode> <N> <D>
#     mode = parity | bench   (default parity)
#
# parity: adapt off, num_warmup=0; dumps post-optimization HLO to confirm the QR
#   op is emitted and hoisted, then compares the SICM-on / SICM-off sample
#   trajectories pointwise (benign-reassociation roundoff class) and against
#   NumPyro if available.
# bench: realistic warmup/samples; reports median wall-clock for SICM on vs off.

using Reactant, Random, Statistics, Printf
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray
Reactant.set_default_backend("cpu")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false  # enable region+SICM pipeline
# Default (QR) form: the rule emits enzymexla.linalg.qr, which executes on CPU
# via the Fortran LAPACK dgeqrf_/dorgqr_ runtime symbols registered in Reactant's
# TracedLinearAlgebra.__init__ (same mechanism syevd uses). The Gram fallback is
# left off.
Reactant.Compiler.DEBUG_PROBPROG_GRAM_STATS[] = false


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
function linreg_model(rng, X, D, N)
    _, w = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (D,)); symbol=:w)
    _, sigma = ProbProg.sample(rng, ProbProg.LogNormal(0.0, 1.0); symbol=:sigma)
    yhat = X * w
    _, _y = ProbProg.sample(rng, ProbProg.Normal(yhat, sigma, (N,)); symbol=:y)
    return (w, sigma)
end

function gen_data(N, D; seed=42)
    rng = Random.MersenneTwister(seed)
    X = randn(rng, N, D)
    w_true = randn(rng, D)
    sigma_true = 0.3
    y = X * w_true .+ sigma_true .* randn(rng, N)
    return X, y
end

# ---------------------------------------------------------------------------
# NUTS program
# ---------------------------------------------------------------------------
function nuts_program(rng, model, X, D, N, step_size, max_tree_depth, num_warmup,
                      num_samples, inv_mass, constraint_tensor,
                      constrained_addresses, selection, adapt_step_size,
                      adapt_mass_matrix)
    trace, _, _ = ProbProg.generate(rng, constraint_tensor, model, X, D, N;
                                    constrained_addresses)
    trace, diagnostics, _, _ = ProbProg.mcmc(
        rng, trace, model, X, D, N;
        selection, algorithm=:NUTS, inverse_mass_matrix=inv_mass, step_size,
        max_tree_depth, num_warmup, num_samples, adapt_step_size,
        adapt_mass_matrix,
    )
    return trace, diagnostics
end

function build(N, D; num_warmup, num_samples, adapt)
    X_DATA, Y_DATA = gen_data(N, D)
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)

    obs = ProbProg.Constraint(
        :w => zeros(Float64, D), :sigma => [1.0], :y => Y_DATA,
    )
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)
    selection = ProbProg.select(ProbProg.Address(:w), ProbProg.Address(:sigma))
    X = Reactant.to_rarray(X_DATA)
    step_size = ConcreteRNumber(0.05)
    inv_mass = nothing  # default diagonal placeholder

    args = (rng, linreg_model, X, D, N, step_size, 10, num_warmup, num_samples,
            inv_mass, obs_tensor, constrained_addresses, selection, adapt, adapt)
    return X_DATA, Y_DATA, args
end

function run_once(N, D; num_warmup, num_samples, adapt, disable_sicm, dump_ir=false)
    _, _, args = build(N, D; num_warmup, num_samples, adapt)
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = disable_sicm

    if dump_ir
        hlo = @code_hlo optimize = :probprog nuts_program(args...)
        return string(hlo)
    end

    compiled, tt = ProbProg.with_trace() do
        @compile optimize = :probprog nuts_program(args...)
    end
    t = @elapsed begin
        trace_tensor, _ = compiled(args...)
        trace_tensor = Array(trace_tensor)
    end
    selected = ProbProg.filter_entries_by_selection(tt.entries,
                                                    args[13])
    trace = ProbProg.unflatten_trace(trace_tensor, 0.0, selected, nothing)
    return trace, t
end

function parity(N, D)
    @info "=== PARITY  N=$N D=$D (QR form) ==="
    # 1. IR evidence: dump SICM-on and SICM-off HLO. The rule fired iff the O(N D)
    #    forward mean contraction (N-vector dot) is gone from the loop and the
    #    invariant Gram statistics (D x D) appear.
    ir_on = run_once(N, D; num_warmup=0, num_samples=2, adapt=false,
                     disable_sicm=false, dump_ir=true)
    ir_off = run_once(N, D; num_warmup=0, num_samples=2, adapt=false,
                      disable_sicm=true, dump_ir=true)
    for (nm, ir) in (("on", ir_on), ("off", ir_off))
        open("/tmp/sbrantq-heatinv/qr-rule/linreg_ir_N$(N)_D$(D)_$(nm).mlir", "w") do io
            write(io, ir)
        end
    end
    has_qr(ir) = occursin("linalg.qr", ir) || occursin("geqrf", ir) ||
                 occursin("orgqr", ir)
    fired = has_qr(ir_on) && !has_qr(ir_off)
    n_bigdot_on = count(l -> occursin("dot_general", l) && occursin("$(N)x$(D)", l),
                        split(ir_on, '\n'))
    n_bigdot_off = count(l -> occursin("dot_general", l) && occursin("$(N)x$(D)", l),
                         split(ir_off, '\n'))
    @info "IR: QR factorization present on=$(has_qr(ir_on)) off=$(has_qr(ir_off)); rule fired = $fired"
    @info "IR: (NxD) dot_general count  on=$n_bigdot_on  off=$n_bigdot_off"

    # 2. SICM on vs off trajectory agreement (adapt off).
    tr_on, _ = run_once(N, D; num_warmup=0, num_samples=5, adapt=false,
                        disable_sicm=false)
    tr_off, _ = run_once(N, D; num_warmup=0, num_samples=5, adapt=false,
                         disable_sicm=true)
    won = Array(tr_on.choices[:w]); woff = Array(tr_off.choices[:w])
    rel = maximum(abs.(won .- woff) ./ (abs.(woff) .+ 1e-30))
    absd = maximum(abs.(won .- woff))
    @info "SICM on/off w trajectory: max_rel=$rel  max_abs=$absd"
    return (; fired, rel, absd)
end

function bench_side(N, D; num_warmup, num_samples, disable_sicm, reps)
    # adapt off so on/off run the same NUTS trajectory (identical gradient-eval
    # count to roundoff); the wall-clock ratio then reflects per-gradient cost.
    _, _, args = build(N, D; num_warmup, num_samples, adapt=false)
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = disable_sicm
    compiled, _ = ProbProg.with_trace() do
        @compile optimize = :probprog nuts_program(args...)
    end
    compiled(args...)  # warmup / discard
    ts = Float64[]
    for _ in 1:reps
        push!(ts, @elapsed (r = compiled(args...); Array(r[1])))
    end
    return median(ts)
end

function bench(N, D; num_warmup=500, num_samples=500, reps=5)
    @info "=== BENCH  N=$N D=$D ==="
    ton = bench_side(N, D; num_warmup, num_samples, disable_sicm=false, reps)
    toff = bench_side(N, D; num_warmup, num_samples, disable_sicm=true, reps)
    @info @sprintf("N=%d D=%d  SICM-on=%.4fs  SICM-off=%.4fs  speedup=%.2fx",
                   N, D, ton, toff, toff / ton)
    return (; N, D, ton, toff, speedup=toff / ton)
end

# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    mode = length(ARGS) >= 1 ? ARGS[1] : "parity"
    N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1000
    D = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
    mkpath("/tmp/sbrantq-heatinv/qr-rule")
    if mode == "parity"
        parity(N, D)
    else
        bench(N, D)
    end
end
