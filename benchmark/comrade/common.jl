using Printf: @sprintf
using Reactant: Reactant, @compile
using Enzyme: Enzyme, Const
using Random: Random

include("../utils.jl")

logdensityofref(tpostr, xr) = logdensityof(tpostr[], xr)
function gradref(tpostr, xr)
    derivs, val = (Enzyme.gradient(ReverseWithPrimal, logdensityofref, Ref(tpostr), xr))
    return last(derivs), val
end

function run_comrade_benchmark!(
    results::Dict, benchmark_name::String, backend::String, tpostr
)
    # TODO which Enzyme passes do I want to enable and disable
    run_benchmark!(results, benchmark_name, backend, tpostr, "forward", "Default")
    run_benchmark!(results, benchmark_name, backend, tpostr, "backward", "Default")
    return nothing
end

function run_benchmark!(
    results::Dict,
    benchmark_name::String,
    backend::String,
    tpost,
    fwd_or_bwd::String,
    tag::String,
)
    if !haskey(results, "TFLOP/s")
        results["TFLOP/s"] = Dict{String,Float64}()
    end
    if !haskey(results, "Runtime (s)")
        results["Runtime (s)"] = Dict{String,Float64}()
    end

    prim_or_rev = fwd_or_bwd == "forward" ? "primal" : "reverse"
    full_benchmark_name = string(benchmark_name, "/", prim_or_rev, "/", backend, "/", tag)
    @assert !haskey(results["Runtime (s)"], full_benchmark_name) "Benchmark already \
                                                                  exists: \
                                                                  $(full_benchmark_name)"
    @assert !haskey(results["TFLOP/s"], full_benchmark_name) "Benchmark already exists: \
                                                              $(full_benchmark_name)"

    rng = Random.default_rng()  # don't use any other rng
    Random.seed!(rng, 0)

    Ts = if backend == "TPU"
        Float32
    else
        Float64
    end

    x = Reactant.to_rarray(randn(rng, Ts, dimension(tpost)))

    @info typeof(x)
    @info typeof(tpost)

    fn = if fwd_or_bwd == "forward"
        logdensityof
    elseif fwd_or_bwd == "backward"
        gradref
    else
        error("Unknown fwd_or_bwd: $(fwd_or_bwd)")
    end

    prof_result = Reactant.Profiler.profile_with_xprof(fn, tpost, x; nrepeat=10, warmup=3)

    @code_hlo fn(tpost, x)

    results["Runtime (s)"][full_benchmark_name] =
        prof_result.profiling_result.runtime_ns / 1e9
    results["TFLOP/s"][full_benchmark_name] =
        if prof_result.profiling_result.flops_data === nothing
            -1
        else
            prof_result.profiling_result.flops_data.RawFlopsRate / 1e12
        end

    GC.gc(true)

    print_stmt = @sprintf(
        "%100s     :     %.5gs    %.5g TFLOP/s",
        full_benchmark_name,
        results["Runtime (s)"][full_benchmark_name],
        results["TFLOP/s"][full_benchmark_name]
    )
    @info print_stmt
    return nothing
end
