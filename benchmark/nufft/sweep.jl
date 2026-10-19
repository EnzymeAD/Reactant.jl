#==============================================================================
Standalone parameter sweep for the Reactant type-2 NUFFT.

Unlike `runbenchmarks.jl` (which records the standard suite results consumed by
CI), this script takes the problem size on the command line and saves raw
`BenchmarkTools` trials and Reactant profiles to a JLD2 file, which
`process.jl` aggregates into a table.

Usage:

    julia --project=benchmark/nufft benchmark/nufft/sweep.jl 10000000 512 2 \
        --opt all --out bench-1e7-512-all.jld

REPL usage:

    julia --project=benchmark/nufft
    julia> include("benchmark/nufft/common.jl")
    julia> setup = setup_nufft(; M = 100_000, N = 128, D = 2);
    julia> setup.comp_type2(setup.args_type2...)
==============================================================================#

using ArgParse
using BenchmarkTools
using FileIO
using JLD2

include("common.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "M"
    help = "number of non-uniform points"
    arg_type = Int
    default = 100_000
    "N"
    help = "number of uniform modes per dimension"
    arg_type = Int
    default = 128
    "D"
    help = "number of dimensions"
    arg_type = Int
    default = 2
    "--eps"
    help = "NUFFT accuracy tolerance"
    arg_type = Float64
    default = 1e-6
    "--eltype"
    help = "floating-point type (Float32 or Float64)"
    arg_type = String
    default = "Float64"
    "--iflag"
    help = "NUFFT sign flag (1 or -1)"
    arg_type = Int
    default = -1
    "--seed"
    help = "random number generator seed for reproducibility"
    arg_type = Int
    default = 42
    "--opt"
    help = "optimization pass list (:all, :after_enzyme, ...)"
    arg_type = String
    default = "all"
    "--backend"
    help = "select Reactant backend (cpu, cuda, ...)"
    arg_type = String
    "--out"
    help = "save benchmarks to file"
    arg_type = String
end
parsed_args = parse_args(ARGS, s)

M = parsed_args["M"]
N = parsed_args["N"]
D = parsed_args["D"]
eps = parsed_args["eps"]
T = eval(Meta.parse(parsed_args["eltype"]))
iflag = parsed_args["iflag"]
seed = parsed_args["seed"]
optimize = Symbol(parsed_args["opt"])
backend = parsed_args["backend"]
out = parsed_args["out"]

if !isnothing(backend)
    Reactant.set_default_backend(backend)
end

setup = setup_nufft(; M, N, D, eps, T, iflag, seed, optimize)

@info "Executing (prepared + compiled; setpts excluded)…"
println("\n--- type 2 ---")
b_type2 = @benchmark $(setup.comp_type2)($(setup.args_type2)...)
display(b_type2)

println("\n--- reverse-mode VJP ---")
b_reverse = @benchmark $(setup.comp_reverse)($(setup.args_reverse)...)
display(b_reverse)

println("\n--- analytic adjoint (type 1) ---")
b_adjoint = @benchmark $(setup.comp_adjoint)($(setup.args_adjoint)...)
display(b_adjoint)
println()

@info "Profiling..."
prof_type2 = Reactant.@timed setup.comp_type2(setup.args_type2...)
prof_reverse = Reactant.@timed setup.comp_reverse(setup.args_reverse...)
prof_adjoint = Reactant.@timed setup.comp_adjoint(setup.args_adjoint...)

if !isnothing(out)
    jldsave(
        out;
        M,
        N,
        D,
        eps,
        eltype=T,
        iflag,
        seed,
        optimize,
        bench_primal_type_2=b_type2,
        bench_revdiff_enzyme=b_reverse,
        bench_revdiff_analytical_type_1=b_adjoint,
        prof_primal_type_2=prof_type2,
        prof_revdiff_enzyme=prof_reverse,
        prof_revdiff_analytical_type_1=prof_adjoint,
    )
end
