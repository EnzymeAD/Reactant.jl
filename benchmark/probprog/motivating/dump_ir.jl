#!/usr/bin/env julia
#
# Dump pass-by-pass IR for gp_pois_regr model into motivating/dump/
#
# Usage: CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog benchmark/probprog/motivating/dump_ir.jl

ENV["JULIA_DEBUG"] = "Reactant,Reactant_jll"

using Reactant, Random, Statistics, LinearAlgebra
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

# Point dumps to our target directory
const DUMP_DIR = joinpath(@__DIR__, "dump")
mkpath(DUMP_DIR)
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = DUMP_DIR

# Enable pass-by-pass IR dumping for probprog pipeline
Reactant.Compiler.DEBUG_PROBPROG_DUMP_PASSES[] = true
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
Reactant.Compiler.DEBUG_PROBPROG_DUMP_VALUE[] = false

# === Model (same as benchmark) ===

function model_gp_pois_regr(rng, x, k, rho)
    n = Base.length(x)

    _, alpha = ProbProg.sample(
        rng, ProbProg.HalfNormal(2.0, (1,)); symbol=:alpha
    )

    _, f_tilde = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (n,)); symbol=:f_tilde
    )

    x_col = reshape(x, n, 1)
    x_row = reshape(x, 1, n)
    delta = x_col .- x_row

    rho_2d = reshape(rho, 1, 1)
    K_base = exp.(-0.5 .* (delta ./ rho_2d) .^ 2)

    alpha_2d = reshape(alpha, 1, 1)
    cov = (alpha_2d .* alpha_2d) .* K_base

    C = cholesky(cov)
    f = reshape(C.factors' * reshape(f_tilde, n, 1), n)

    _, obs = ProbProg.sample(
        rng, ProbProg.Poisson(exp.(f), (n,)); symbol=:k
    )
    return obs
end

# === NUTS program ===

function nuts_program(
    rng, model, model_args, step_size, max_tree_depth,
    num_warmup, num_samples, constraint_tensor,
    constrained_addresses, selection, adapt_step_size,
    adapt_mass_matrix, inverse_mass_matrix,
)
    t, _, _ = ProbProg.generate(
        rng, constraint_tensor, model, model_args...; constrained_addresses
    )
    t, diagnostics, _, _ = ProbProg.mcmc(
        rng, t, model, model_args...;
        selection=selection, algorithm=:NUTS,
        inverse_mass_matrix=inverse_mass_matrix,
        step_size=step_size, max_tree_depth=max_tree_depth,
        num_warmup=num_warmup, num_samples=num_samples,
        adapt_step_size=adapt_step_size, adapt_mass_matrix=adapt_mass_matrix,
    )
    return t, diagnostics
end

# === Setup (minimal, n=11) ===

function flatten_constraint(constraint)
    constrained_addresses = ProbProg.extract_addresses(constraint)
    constraint_flat = Float64[]
    for addr in constrained_addresses
        append!(constraint_flat, vec(constraint[addr]))
    end
    constraint_tensor = Reactant.to_rarray(reshape(constraint_flat, 1, :))
    return constraint_tensor, constrained_addresses
end

n = 11
rho = 6.0
x = collect(range(0.0, 1.0, length=n))
k = ones(n)  # dummy observations

constraint = ProbProg.Constraint(
    :alpha => [1.0],
    :f_tilde => zeros(n),
    :k => k,
)
selection = ProbProg.select(ProbProg.Address(:alpha), ProbProg.Address(:f_tilde))
position_size = 1 + n

rng = ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
constraint_tensor, constrained_addresses = flatten_constraint(constraint)
x_rarray = Reactant.to_rarray(x)
rho_rarray = Reactant.to_rarray([rho])
model_args = (x_rarray, k, rho_rarray)
step_size_rnum = ConcreteRNumber(0.1)
inverse_mass_matrix = Reactant.to_rarray(Matrix{Float64}(I, position_size, position_size))

# === Dump IR ===

println("Compiling with pass-by-pass IR dumping...")
println("Dumps will be written to: $DUMP_DIR")
println()

hlo_module = ProbProg.with_trace() do
    @code_hlo optimize = :probprog nuts_program(
        rng, model_gp_pois_regr, model_args, step_size_rnum,
        10, 0, 5,  # max_tree_depth=10, num_warmup=0, num_samples=5
        constraint_tensor, constrained_addresses, selection,
        false, false,  # no adaptation (simpler IR)
        inverse_mass_matrix,
    )
end

println()
println("Done. Dumps written to: $DUMP_DIR")
println("Files:")
for f in sort(readdir(DUMP_DIR))
    sz = filesize(joinpath(DUMP_DIR, f))
    println("  $f  ($(sz ÷ 1024) KB)")
end
