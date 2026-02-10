using Reactant: Reactant, @compile, @jit
using LinearAlgebra: norm
using Chairmarks: @b
using Printf: @sprintf

include("../utils.jl")

function recursive_check(x::AbstractArray, y::AbstractArray; kwargs...)
    res = isapprox(x, y; norm=Base.Fix2(norm, Inf), kwargs...)
    if !res
        x_arr = Array(x)
        y_arr = Array(y)
        diff = abs.(x_arr .- y_arr)
        # @show findall(diff .> 1e-2)
        @show maximum(diff)
    end
    return res
end
function recursive_check(gt::Tuple, res::Tuple; kwargs...)
    return all(recursive_check(gt[i], res[i]; kwargs...) for i in 1:length(gt))
end

function run_benchmark!(
    results::Dict,
    benchmark_name::String,
    backend::String,
    fn::F,
    args::Tuple,
    ground_truth_fn::GT=nothing; # pass in a function that is equivalent to fn but faster
    track_numbers::Bool=false,
    skip_manual_vectorized_bench::Bool=false,
) where {F,GT}
    compile_modes = [("Default", Reactant.CompileOptions())]
    # NOTE: extremely slow to benchmark
    # if backend == "CPU"
    #     push!(compile_modes, ("NoOpt", Reactant.DefaultXLACompileOptions()))
    #     push!(
    #         compile_modes,
    #         ("NoRaising", Reactant.CompileOptions(; disable_loop_raising_passes=true)),
    #     )
    # end

    if !haskey(results, "TFLOP/s")
        results["TFLOP/s"] = Dict{String,Float64}()
    end
    if !haskey(results, "Runtime (s)")
        results["Runtime (s)"] = Dict{String,Float64}()
    end

    gt_provided = ground_truth_fn !== nothing
    if !gt_provided
        ground_truth_fn = fn
    end

    fn_splat = splat(fn)
    ground_truth_fn_splat = splat(ground_truth_fn)

    args_copied = map(copy, args)
    gt_res = ground_truth_fn_splat(args_copied)
    benchmark_name = string(benchmark_name, "/primal")

    if backend == "CPU"
        full_benchmark_name = string(benchmark_name, "/CPU/Julia")

        args_copied = map(copy, args)
        bench = @b fn_splat($args_copied) seconds = 15 evals = 1 samples = 10

        results["Runtime (s)"][full_benchmark_name] = bench.time
        results["TFLOP/s"][full_benchmark_name] = -1 # TODO: compute FLOP/s using LIKWID??

        print_stmt = @sprintf "%100s     :     %.5gs" full_benchmark_name bench.time
        @info print_stmt
        GC.gc(true)
    end

    fn_list = Any[(fn, "")]
    if gt_provided && !skip_manual_vectorized_bench
        push!(fn_list, (ground_truth_fn, "_manual_vectorized"))
    end

    for (tag, compile_options) in compile_modes
        for (f′, add_tag) in fn_list
            full_benchmark_name = string(benchmark_name, "/", backend, "/", tag, add_tag)

            args_ra = map(x -> Reactant.to_rarray(x; track_numbers), args)
            res_ra = @jit f′(args_ra...)
            @assert recursive_check(res_ra, gt_res, atol=5e-2, rtol=5e-2) "Result does not \
                                                                           match ground \
                                                                           truth"

            prof_result = Reactant.Profiler.profile_with_xprof(
                f′, args_ra...; nrepeat=10, compile_options=compile_options
            )
            results["Runtime (s)"][full_benchmark_name] =
                prof_result.profiling_result.runtime_ns / 1e9
            results["TFLOP/s"][full_benchmark_name] =
                if prof_result.profiling_result.flops_data === nothing
                    -1
                else
                    prof_result.profiling_result.flops_data.RawFlopsRate / 1e12
                end

            print_stmt = @sprintf(
                "%100s     :     %.5gs\t%.5g TFLOP/s",
                full_benchmark_name,
                results["Runtime (s)"][full_benchmark_name],
                results["TFLOP/s"][full_benchmark_name]
            )
            @info print_stmt
            GC.gc(true)
        end
    end

    return nothing
end
