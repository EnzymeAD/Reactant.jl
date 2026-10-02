# Impulse joint log density of the diamonds regression model at given points:
# the weight of a fully constrained generate (intercept, b, sigma, y), compiled
# once, the counterpart of diamonds_numpyro_reference.py.
#
# Usage: julia --project=. ablation/sicm_roster/diamonds_joint_density.jl \
#            <data.json> <points.json> <out.json>
using Reactant, JSON3, LinearAlgebra
using Reactant: ProbProg, ReactantRNG

include(joinpath(@__DIR__, "..", "..", "sicm", "diamonds_regression.jl"))

function main(data_path, points_path, out_path)
    data = JSON3.read(read(data_path, String), Dict{String,Any})
    points = JSON3.read(read(points_path, String), Vector{Dict{String,Any}})
    s = setup(data)
    function params(p)
        return Dict{String,Any}(
            "beta" => Float64.(p["beta"]), "sigma" => Float64.(p["sigma"])
        )
    end
    c0 = build_constraint(data, params(points[1]))
    ca = ProbProg.extract_addresses(c0)
    rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 4]))
    function prog(rng, ot, a...)
        return ProbProg.generate(rng, ot, s.model_fn, a...; constrained_addresses=ca)
    end
    compiled = @compile optimize = :probprog prog(
        rng, ProbProg.flatten_constraint(c0), s.model_args...
    )
    ld = Float64[]
    for p in points
        c = build_constraint(data, params(p))
        @assert ProbProg.extract_addresses(c) == ca
        _, w, _ = compiled(rng, ProbProg.flatten_constraint(c), s.model_args...)
        push!(ld, Float64(Array(w)[]))
    end
    open(out_path, "w") do io
        JSON3.write(io, Dict("log_density" => ld))
    end
end

main(ARGS...)
