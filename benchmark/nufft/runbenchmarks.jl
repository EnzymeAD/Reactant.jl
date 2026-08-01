#==============================================================================
Micro-benchmark for the Reactant type-2 NUFFT in VLBISkyModelsReactantExt.

The benchmark measures:
  * type-2 execution,
  * its Enzyme reverse-mode VJP with respect to the uniform modes, and
  * the equivalent analytic adjoint, evaluated with a type-1 NUFFT.

Planning and `set_nufft_points` happen before compilation and are not included
in any benchmark trial.

REPL usage (start Julia with the test project so Reactant + BenchmarkTools
are available):

    julia --project=test
    julia> include("bench/bench_nufft.jl")
    julia> res = run_nufft_bench(; M = 100_000, N = 128, D = 2, eps = 1e-6);
    julia> res.type2          # type-2 forward execution
    julia> res.reverse        # Enzyme reverse-mode VJP
    julia> res.adjoint        # analytic VJP via type 1

`res` also carries the compiled thunks, prepared point sets, and inputs so you
can inspect them directly in the REPL.
==============================================================================#

using Reactant
using Enzyme
using Random
using BenchmarkTools
using VLBISkyModels
using ArgParse
using FileIO
using JLD2

# The transforms are defined inside the weak-dep extension, not exported.
const _EXT = Base.get_extension(VLBISkyModels, :VLBISkyModelsReactantExt)
_EXT === nothing && error("VLBISkyModelsReactantExt not loaded — is Reactant available?")
const execute_nufft = _EXT.execute_nufft
const execute_nufft! = _EXT.execute_nufft!
const plan_nufft = _EXT.plan_nufft
const set_nufft_points = _EXT.set_nufft_points

"""
    type2_vjp!(dfk, out, out_shadow, prep, fk, cotangent)

Evaluate a seeded reverse-mode derivative of a prepared type-2 NUFFT with
respect to `fk`, writing the result into `dfk`. The prepared point set is
constant. The primal output and its shadow are explicit buffers so `cotangent`
can seed the vector-Jacobian product without reducing the NUFFT output to a
scalar loss.
"""
function type2_vjp!(dfk, out, out_shadow, prep, fk, cotangent)
    fill!(dfk, zero(eltype(dfk)))
    copyto!(out_shadow, cotangent)
    Enzyme.autodiff(
        Enzyme.Reverse,
        Enzyme.Const(execute_nufft!),
        Enzyme.Const,
        Enzyme.Duplicated(out, out_shadow),
        Enzyme.Const(prep),
        Enzyme.Duplicated(fk, dfk),
    )
    return dfk
end

"""
    run_nufft_bench(; M=100_000, N=128, D=2, eps=1e-6, T=Float64, iflag=-1, seed=42)

Prepare random non-uniform points once, then compile and benchmark a type-2
NUFFT, its Enzyme reverse-mode VJP with respect to the uniform modes, and its
analytic adjoint (a type-1 NUFFT with the opposite sign). Compilation and point
setup are reported separately and excluded from the benchmark trials.
"""
function run_nufft_bench(;
    M=100_000, N=128, D=2, eps=1.0e-6, T=Float64, iflag=-1, seed=42, optimize=:all
)
    nmodes = ntuple(_ -> N, D)
    @info "NUFFT benchmark" D M nmodes eps eltype = T optimize

    rng = Random.MersenneTwister(seed)
    # Non-uniform points in [-pi, pi) per dimension.
    points = ntuple(_ -> Reactant.to_rarray(T(2π) .* rand(rng, T, M) .- T(π)), D)
    # Type-2 modes and an arbitrary output cotangent for the VJP.
    fk = Reactant.to_rarray(randn(rng, complex(T), nmodes...))
    cotangent = Reactant.to_rarray(randn(rng, complex(T), M))

    # Type 2 uses `iflag`; its analytic adjoint is type 1 with the opposite sign.
    plan2 = Reactant.to_rarray(plan_nufft(T, 2, nmodes; iflag, eps))
    plan1 = Reactant.to_rarray(plan_nufft(T, 1, nmodes; iflag=-iflag, eps))

    @info "Preparing points (excluded from benchmark timings)…"
    t_setpts2 = @elapsed prep2 = @jit set_nufft_points(plan2, points)
    t_setpts1 = @elapsed prep1 = @jit set_nufft_points(plan1, points)
    println("  type-2 setpts: $(round(t_setpts2; digits = 2)) s")
    println("  type-1 setpts: $(round(t_setpts1; digits = 2)) s")

    @info "Compiling…"
    out = Reactant.to_rarray(zeros(complex(T), M))
    t_type2 = @elapsed comp_type2 = @compile(
        optimize = optimize,
        sync = true,
        assert_nonallocating = true,
        execute_nufft!(out, prep2, fk),
    )
    dfk = Reactant.to_rarray(zeros(complex(T), nmodes...))
    out_shadow = Reactant.to_rarray(zeros(complex(T), M))
    t_reverse = @elapsed comp_reverse = @compile(
        optimize = optimize,
        sync = true,
        assert_nonallocating = true,
        type2_vjp!(dfk, out, out_shadow, prep2, fk, cotangent),
    )
    out_adjoint = Reactant.to_rarray(zeros(complex(T), nmodes...))
    t_adjoint = @elapsed comp_adjoint = @compile(
        optimize = optimize,
        sync = true,
        assert_nonallocating = true,
        execute_nufft!(out_adjoint, prep1, cotangent),
    )
    println("  type-2 compile:           $(round(t_type2; digits = 2)) s")
    println("  reverse-mode compile:     $(round(t_reverse; digits = 2)) s")
    println("  analytic adjoint compile: $(round(t_adjoint; digits = 2)) s")

    # Warm up, synchronize, and verify the differentiated VJP against the
    # analytic type-1 adjoint. None of this is part of a benchmark trial.
    @info "Warming up..."
    c_out = comp_type2(out, prep2, fk)
    dfk_reverse = comp_reverse(dfk, out, out_shadow, prep2, fk, cotangent)
    dfk_adjoint = comp_adjoint(out_adjoint, prep1, cotangent)

    reverse_host = Array(dfk_reverse)
    adjoint_host = Array(dfk_adjoint)
    max_error = maximum(abs, reverse_host .- adjoint_host)
    relative_error = max_error / max(maximum(abs, adjoint_host), Base.eps(T))
    println("  type-2 out: $(typeof(c_out)) size $(size(c_out))")
    println("  reverse/adjoint max error: $max_error (relative $relative_error)")

    @info "Executing (prepared + compiled; setpts excluded)…"
    println("\n--- type 2 ---")
    b_type2 = @benchmark $comp_type2($out, $prep2, $fk)
    display(b_type2)

    println("\n--- reverse-mode VJP ---")
    b_reverse = @benchmark $comp_reverse($dfk, $out, $out_shadow, $prep2, $fk, $cotangent)
    display(b_reverse)

    println("\n--- analytic adjoint (type 1) ---")
    b_adjoint = @benchmark $comp_adjoint($out_adjoint, $prep1, $cotangent)
    display(b_adjoint)
    println()

    @info "Profiling..."
    prof_type2 = Reactant.@timed comp_type2(out, prep2, fk)
    prof_reverse = Reactant.@timed comp_reverse(dfk, out, out_shadow, prep2, fk, cotangent)
    prof_adjoint = Reactant.@timed comp_adjoint(out_adjoint, prep1, cotangent)

    return (;
        type2=b_type2,
        reverse=b_reverse,
        adjoint=b_adjoint,
        comp_type2,
        comp_reverse,
        comp_adjoint,
        prof_type2,
        prof_reverse,
        prof_adjoint,
        prep2,
        prep1,
        points,
        fk,
        dfk,
        out,
        out_shadow,
        out_adjoint,
        cotangent,
        nmodes,
    )
end

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

data = run_nufft_bench(; M, N, D, eps, T, iflag, seed, optimize)

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
        bench_primal_type_2=data.type2,
        bench_revdiff_enzyme=data.reverse,
        bench_revdiff_analytical_type_1=data.adjoint,
        prof_primal_type_2=data.prof_type2,
        prof_revdiff_enzyme=data.prof_reverse,
        prof_revdiff_analytical_type_1=data.prof_adjoint,
    )
end
