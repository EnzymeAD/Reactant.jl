#==============================================================================
Shared setup for the Reactant type-2 NUFFT benchmarks in
VLBISkyModelsReactantExt.

Three kernels are prepared:
  * type-2 execution,
  * its Enzyme reverse-mode VJP with respect to the uniform modes, and
  * the equivalent analytic adjoint, evaluated with a type-1 NUFFT.

Planning and `set_nufft_points` happen before compilation and are excluded from
every timed region.

`runbenchmarks.jl` uses this to record the standard suite results; `sweep.jl`
uses it for the parameter sweep consumed by `process.jl`.
==============================================================================#

using Printf: @sprintf
using Random: Random
using Reactant: Reactant, @compile, @jit
using Enzyme: Enzyme
using VLBISkyModels: VLBISkyModels

include("../utils.jl")

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
    default_eps(T) -> Float64

Accuracy tolerance that the NUFFT kernel can actually reach in `T`.
"""
default_eps(::Type{Float64}) = 1.0e-6
default_eps(::Type{Float32}) = 1.0e-5

"""
    setup_nufft(; M=100_000, N=128, D=2, T=Float64, eps=default_eps(T), iflag=-1,
                seed=42, optimize=:all, assert_nonallocating=true)

Prepare random non-uniform points once, then compile a type-2 NUFFT, its Enzyme
reverse-mode VJP with respect to the uniform modes, and its analytic adjoint (a
type-1 NUFFT with the opposite sign).

Returns a named tuple holding the three compiled thunks together with the
argument tuples they are called with, the prepared point sets, and the accuracy
of the differentiated VJP relative to the analytic adjoint. Compilation and
point setup are reported but excluded from any benchmark timing.
"""
function setup_nufft(;
    M=100_000,
    N=128,
    D=2,
    T=Float64,
    eps=default_eps(T),
    iflag=-1,
    seed=42,
    optimize=:all,
    assert_nonallocating=true,
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
    # The plans stay on the host: they only carry kernel coefficients, which are
    # traced as constants by the compiled point setup and execution.
    plan2 = plan_nufft(T, 2, nmodes; iflag, eps)
    plan1 = plan_nufft(T, 1, nmodes; iflag=-iflag, eps)

    @info "Preparing points (excluded from benchmark timings)…"
    t_setpts2 = @elapsed prep2 = @jit set_nufft_points(plan2, points)
    t_setpts1 = @elapsed prep1 = @jit set_nufft_points(plan1, points)
    println("  type-2 setpts: $(round(t_setpts2; digits = 2)) s")
    println("  type-1 setpts: $(round(t_setpts1; digits = 2)) s")

    @info "Compiling…"
    out = Reactant.to_rarray(zeros(complex(T), M))
    args_type2 = (out, prep2, fk)
    t_type2 = @elapsed comp_type2 = @compile(
        optimize = optimize,
        sync = true,
        assert_nonallocating = assert_nonallocating,
        execute_nufft!(args_type2...),
    )
    dfk = Reactant.to_rarray(zeros(complex(T), nmodes...))
    out_shadow = Reactant.to_rarray(zeros(complex(T), M))
    args_reverse = (dfk, out, out_shadow, prep2, fk, cotangent)
    t_reverse = @elapsed comp_reverse = @compile(
        optimize = optimize,
        sync = true,
        assert_nonallocating = assert_nonallocating,
        type2_vjp!(args_reverse...),
    )
    out_adjoint = Reactant.to_rarray(zeros(complex(T), nmodes...))
    args_adjoint = (out_adjoint, prep1, cotangent)
    t_adjoint = @elapsed comp_adjoint = @compile(
        optimize = optimize,
        sync = true,
        assert_nonallocating = assert_nonallocating,
        execute_nufft!(args_adjoint...),
    )
    println("  type-2 compile:           $(round(t_type2; digits = 2)) s")
    println("  reverse-mode compile:     $(round(t_reverse; digits = 2)) s")
    println("  analytic adjoint compile: $(round(t_adjoint; digits = 2)) s")

    # Warm up, synchronize, and verify the differentiated VJP against the
    # analytic type-1 adjoint. None of this is part of a benchmark trial.
    @info "Warming up..."
    c_out = comp_type2(args_type2...)
    dfk_reverse = comp_reverse(args_reverse...)
    dfk_adjoint = comp_adjoint(args_adjoint...)

    reverse_host = Array(dfk_reverse)
    adjoint_host = Array(dfk_adjoint)
    max_error = maximum(abs, reverse_host .- adjoint_host)
    relative_error = max_error / max(maximum(abs, adjoint_host), Base.eps(T))
    println("  type-2 out: $(typeof(c_out)) size $(size(c_out))")
    println("  reverse/adjoint max error: $max_error (relative $relative_error)")
    if !(relative_error ≤ 100 * eps)
        @warn "Reverse-mode VJP disagrees with the analytic adjoint" max_error relative_error eps
    end

    return (;
        comp_type2,
        args_type2,
        comp_reverse,
        args_reverse,
        comp_adjoint,
        args_adjoint,
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
        max_error,
        relative_error,
    )
end

"""
    profile_nufft_kernel!(results, benchmark_name, thunk, args; nrepeat=3, warmup=1)

Profile one compiled kernel and record its runtime and TFLOP/s under
`benchmark_name` in the standard results dictionary.

The repeat count is deliberately small: one type-1 execution takes tens of
seconds on a GPU and minutes on the CPU backend, so a larger `nrepeat` would
push the job past the workflow timeout.
"""
function profile_nufft_kernel!(
    results::Dict, benchmark_name::String, thunk, args::Tuple; nrepeat::Int=3, warmup::Int=1
)
    if !haskey(results, "Runtime (s)")
        results["Runtime (s)"] = Dict{String,Float64}()
    end
    if !haskey(results, "TFLOP/s")
        results["TFLOP/s"] = Dict{String,Float64}()
    end
    @assert !haskey(results["Runtime (s)"], benchmark_name) "Benchmark already exists: \
                                                             $(benchmark_name)"

    prof_result = Reactant.Profiler.profile_with_xprof(thunk, args...; nrepeat, warmup)

    runtime = prof_result.profiling_result.runtime_ns / 1e9
    if runtime ≤ 0
        # xprof has no step time for this kernel (the type-1 execution is one
        # such case). Time it directly instead of recording the sentinel: the
        # thunks are compiled with `sync = true`, so this is a real device time.
        runtime = @elapsed thunk(args...)
        @warn "No xprof runtime; using a wall-clock sample" benchmark_name runtime
    end

    # A metrics entry with no measured device time gives a non-finite rate, which
    # JSON cannot represent; report it as unknown like a missing entry.
    tflops = if prof_result.profiling_result.metrics_data === nothing
        -1.0
    else
        prof_result.profiling_result.metrics_data.raw_flops_rate / 1e12
    end
    isfinite(tflops) || (tflops = -1.0)

    results["Runtime (s)"][benchmark_name] = runtime
    results["TFLOP/s"][benchmark_name] = tflops

    GC.gc(true)

    print_stmt = @sprintf(
        "%100s     :     %.5gs    %.5g TFLOP/s",
        benchmark_name,
        results["Runtime (s)"][benchmark_name],
        results["TFLOP/s"][benchmark_name]
    )
    @info print_stmt
    return nothing
end

"""
    run_nufft_benchmark!(results, backend; M, N, D, T, kwargs...)

Compile and profile the three NUFFT kernels for one problem size, storing the
results under names of the form
`NUFFT M=… N=…^D [T]/<mode>/<backend>/<implementation>`.
"""
function run_nufft_benchmark!(
    results::Dict, backend::String; M::Int, N::Int, D::Int, T::Type, kwargs...
)
    setup = setup_nufft(; M, N, D, T, kwargs...)
    benchmark_name = "NUFFT M=$(M) N=$(N)^$(D) [$(T)]"

    profile_nufft_kernel!(
        results,
        string(benchmark_name, "/primal/", backend, "/type 2"),
        setup.comp_type2,
        setup.args_type2,
    )
    profile_nufft_kernel!(
        results,
        string(benchmark_name, "/reverse/", backend, "/Enzyme"),
        setup.comp_reverse,
        setup.args_reverse,
    )
    profile_nufft_kernel!(
        results,
        string(benchmark_name, "/reverse/", backend, "/analytic type 1"),
        setup.comp_adjoint,
        setup.args_adjoint,
    )

    return nothing
end
