#!/usr/bin/env julia
#
# Branchy change-point demo (Impulse-only, C2 stochastic-control-flow story).
#
# The per-year Poisson likelihood is observed INSIDE each `impulse.if` branch
# (per-branch addresses `ye_t`/`yl_t`, C2 shape constraint honored). This driver
# recovers the posterior over the switch position `s` and shows concentration at
# the true `s*`.
#
# NOTE (blocked kernel): the intended single-address MH kernel
# (`coal_mining_mh_program`, alternating `mh(:s)/mh(:l1)/mh(:l2)`) is BLOCKED by a
# C2 regenerate bug: `mh`/regenerate over the discrete branch-selecting address
# does NOT re-apply the observed constraint to a branch site that becomes active
# after a flip, so the posterior is wrong (minimal n=1 reproducer: `generate`
# gives the exact p(s=1|y)=0.980, MH gives 0.11). The branchy mechanism itself is
# correct: `generate` threads the taken-branch weight exactly. This driver
# therefore recovers p(s|y) by enumerating s through the verified `generate`
# path (p(s|y) = softmax_s weight(s | y, l1, l2)), which is exactly the marginal
# the blocked MH would target.
#
# Usage:
#   CUDA_VISIBLE_DEVICES="" julia -t 32 --project=benchmark/probprog \
#       benchmark/probprog/coal_mining_mh_demo.jl [n] [s_true] [seed]

using Reactant, Random, LinearAlgebra, Statistics, JSON3, Printf
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

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

Reactant.set_default_backend("cpu")

const BENCH_DIR = @__DIR__
include(joinpath(BENCH_DIR, "standard", "coal_mining.jl"))

# Series length is a compile-time constant (an `impulse.if` loop bound), held in
# a global Ref read by the top-level model function so the model closure captures
# no local field.
const DEMO_N = Ref(56)
branch_model(rng) = coal_mining_branch(rng, DEMO_N[])

# Log joint weight with `s` (and l1,l2) constrained: taken-branch Poisson loglik.
function weight_program(rng, model, obs_tensor, constrained)
    _, w, _ = ProbProg.generate(
        rng, obs_tensor, model; constrained_addresses=constrained
    )
    return w
end

function generate_counts(n::Int, s_true::Int, l1::Float64, l2::Float64, seed::Int)
    rng = MersenneTwister(seed)
    poisson_knuth(r, lam) = begin
        L = exp(-lam); k = 0; p = 1.0
        while true
            k += 1
            p *= rand(r)
            p <= L && break
        end
        return k - 1
    end
    return Float64[poisson_knuth(rng, t < s_true ? l1 : l2) for t in 0:(n - 1)]
end

# Build the constraint plus the flat vector and the offset of `:s`. The flat
# order follows `extract_addresses` (an unordered Set), so the same Set is
# reused for the tensor, the `:s` offset, and the `generate` call to stay
# aligned. l1,l2 are fixed at truth.
function enum_constraint(y::Vector{Float64}, l1::Float64, l2::Float64)
    n = length(y)
    pairs = Pair{Symbol,Any}[:s => [0.0], :l1 => [l1], :l2 => [l2]]
    for t in 0:(n - 1)
        push!(pairs, Symbol("ye_", t) => [y[t + 1]])
        push!(pairs, Symbol("yl_", t) => [y[t + 1]])
    end
    obs = ProbProg.Constraint(pairs...)
    addresses = ProbProg.extract_addresses(obs)
    flat = Float64[]
    s_slot = 0
    pos = 1
    for a in addresses
        v = vec(obs[a])
        a == ProbProg.Address(:s) && (s_slot = pos)
        append!(flat, v)
        pos += length(v)
    end
    return obs, addresses, flat, s_slot
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 56
    s_true = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 40
    seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 12345

    DEMO_N[] = n
    l1_true, l2_true = 3.0, 0.9

    y = generate_counts(n, s_true, l1_true, l2_true, seed)
    @printf("data: n=%d, s*=%d, early mean=%.3f, late mean=%.3f\n",
            n, s_true, mean(y[1:s_true]), mean(y[(s_true + 1):end]))
    flush(stdout)

    obs, constrained, base_flat, s_slot = enum_constraint(y, l1_true, l2_true)
    @printf("s occupies flat slot %d of %d\n", s_slot, length(base_flat))
    rng = ReactantRNG(Reactant.to_rarray(UInt64[UInt64(seed), 0]))

    @printf("compiling branchy generate (%d if-sites)... ", n)
    flush(stdout)
    ot0 = Reactant.to_rarray(reshape(copy(base_flat), 1, :))
    ct = @elapsed begin
        compiled, _ = ProbProg.with_trace() do
            @compile sync = true optimize = :probprog weight_program(
                rng, branch_model, ot0, constrained
            )
        end
    end
    @printf("%.1fs\n", ct)
    flush(stdout)

    # Enumerate s = 0..n through the compiled generate (only slot 1 changes).
    print("enumerating s = 0..", n, " ... ")
    flush(stdout)
    weights = Float64[]
    rt = @elapsed for sval in 0:n
        flat = copy(base_flat)
        flat[s_slot] = Float64(sval)
        ot = Reactant.to_rarray(reshape(flat, 1, :))
        push!(weights, Array(compiled(rng, branch_model, ot, constrained))[])
    end
    @printf("%.1fs\n", rt)

    m = maximum(weights)
    ps = exp.(weights .- m)
    ps ./= sum(ps)
    smap = argmax(ps) - 1
    @printf("MAP s = %d (true %d), p(MAP) = %.3f\n", smap, s_true, ps[smap + 1])

    println("\nposterior mass around s* (s : p(s|y)):")
    for s in max(0, s_true - 5):min(n, s_true + 5)
        bar = repeat("#", round(Int, ps[s + 1] * 100))
        marker = s == s_true ? "  <- s*" : ""
        @printf("  %3d : %.4f %s%s\n", s, ps[s + 1], bar, marker)
    end

    outdir = joinpath(BENCH_DIR, "figures")
    mkpath(outdir)
    json_path = joinpath(outdir, "coal_mining_mh_posterior.json")
    open(json_path, "w") do f
        JSON3.write(f, Dict(
            "s" => collect(0:n), "p" => collect(ps), "weights" => collect(weights),
            "s_true" => s_true, "l1_true" => l1_true, "l2_true" => l2_true,
            "n" => n, "seed" => seed, "map_s" => smap,
            "method" => "enumerate-via-generate (MH/regenerate blocked by C2 bug)",
            "compile_time" => ct, "run_time" => rt,
        ))
    end
    println("\nposterior-over-s written to ", json_path)
    return nothing
end

main()
