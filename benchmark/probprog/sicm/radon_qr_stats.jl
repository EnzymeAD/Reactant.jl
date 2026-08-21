# radon_qr_stats.jl
#
# Radon multilevel model (varying county intercept + floor slope) as the
# recognizable real-data anchor for the GaussianSufficientStats (QR) SICM rule.
#
# The hierarchical likelihood mean alpha[county[i]] + beta*x[i] is written as a
# SINGLE contraction of the concatenated design [x Z] (Z the 919x85 county
# one-hot) against the concatenated parameter vector [beta; alpha], exactly the
# "QR of the concatenated [X Z]" form the design note (Section 5.2/7.2)
# prescribes. That single mean dot is what the single-block QR rule targets:
#   mu   = Xtot * params,  Xtot = [x Z]  (919 x 86, invariant, hoisted via QR)
#   y    ~ Normal(mu, sigma, (919,))
# so SICM-on == QR-rule-on for this model (no covariance-centric rule fires).
#
# Usage (from benchmark/probprog):
#   JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=. sicm/radon_qr_stats.jl <mode>
#     mode = parity | bench

using Reactant, Random, Statistics, Printf, JSON3, LinearAlgebra
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray
Reactant.set_default_backend("cpu")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false  # enable region+SICM pipeline
# Default (QR) form: enzymexla.linalg.qr now executes on CPU via the Fortran
# LAPACK dgeqrf_/dorgqr_ symbols registered in TracedLinearAlgebra.__init__.
Reactant.Compiler.DEBUG_PROBPROG_GRAM_STATS[] = false


const RADON_JSON = "/home/sbrantq/anaconda3/envs/probprog/lib/python3.14/site-packages/arviz/data/example_data/code/radon/radon.json"

function load_radon()
    d = JSON3.read(read(RADON_JSON, String))
    N = Int(d.N); J = Int(d.J)
    y = Float64.(collect(d.y))
    x = Float64.(collect(d.x))                 # floor indicator
    county = Int.(collect(d.county)) .+ 1      # 1-based county index
    # Concatenated design [x Z] : (N, 1 + J)
    Xtot = zeros(Float64, N, 1 + J)
    Xtot[:, 1] .= x
    for i in 1:N
        Xtot[i, 1 + county[i]] = 1.0
    end
    return N, J, y, Xtot
end

# params = [beta; alpha_1..alpha_J]  (length 1 + J), prior Normal(0, 5)
function radon_model(rng, Xtot, P, N)
    _, params = ProbProg.sample(rng, ProbProg.Normal(0.0, 5.0, (P,)); symbol=:params)
    _, sigma = ProbProg.sample(rng, ProbProg.LogNormal(0.0, 1.0); symbol=:sigma)
    mu = Xtot * params
    _, _y = ProbProg.sample(rng, ProbProg.Normal(mu, sigma, (N,)); symbol=:y)
    return (params, sigma)
end

function nuts_program(rng, model, Xtot, P, N, step_size, num_warmup, num_samples,
                      constraint_tensor, constrained_addresses, selection, adapt)
    trace, _, _ = ProbProg.generate(rng, constraint_tensor, model, Xtot, P, N;
                                    constrained_addresses)
    trace, diagnostics, _, _ = ProbProg.mcmc(
        rng, trace, model, Xtot, P, N;
        selection, algorithm=:NUTS, inverse_mass_matrix=nothing, step_size,
        max_tree_depth=10, num_warmup, num_samples,
        adapt_step_size=adapt, adapt_mass_matrix=adapt,
    )
    return trace, diagnostics
end

function build(; num_warmup, num_samples, adapt)
    N, J, y, Xtot = load_radon()
    P = 1 + J
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)
    obs = ProbProg.Constraint(:params => zeros(P), :sigma => [1.0], :y => y)
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)
    selection = ProbProg.select(ProbProg.Address(:params), ProbProg.Address(:sigma))
    Xt = Reactant.to_rarray(Xtot)
    step_size = ConcreteRNumber(0.01)
    args = (rng, radon_model, Xt, P, N, step_size, num_warmup, num_samples,
            obs_tensor, constrained_addresses, selection, adapt)
    return N, P, args
end

function run_once(; num_warmup, num_samples, adapt, disable_sicm, dump_ir=false)
    _, _, args = build(; num_warmup, num_samples, adapt)
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = disable_sicm
    if dump_ir
        return string(@code_hlo optimize = :probprog nuts_program(args...))
    end
    compiled, tt = ProbProg.with_trace() do
        @compile optimize = :probprog nuts_program(args...)
    end
    t = @elapsed begin
        trace_tensor, _ = compiled(args...)
        trace_tensor = Array(trace_tensor)
    end
    return t
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath("/tmp/sbrantq-heatinv/qr-rule")
    mode = length(ARGS) >= 1 ? ARGS[1] : "parity"
    if mode == "parity"
        ir = run_once(; num_warmup=0, num_samples=2, adapt=false,
                      disable_sicm=false, dump_ir=true)
        has_qr = occursin("linalg.qr", ir) || occursin("geqrf", ir)
        @info "radon IR: QR present = $has_qr"
        open("/tmp/sbrantq-heatinv/qr-rule/radon_ir.mlir", "w") do io
            write(io, ir)
        end
    else
        ton = median([run_once(; num_warmup=500, num_samples=500, adapt=true,
                                disable_sicm=false) for _ in 1:3])
        toff = median([run_once(; num_warmup=500, num_samples=500, adapt=true,
                                 disable_sicm=true) for _ in 1:3])
        @info @sprintf("radon SICM-on=%.3fs SICM-off=%.3fs speedup=%.2fx",
                       ton, toff, toff / ton)
    end
end
