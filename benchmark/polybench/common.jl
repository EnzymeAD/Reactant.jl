using Reactant: Reactant, @compile
using LinearAlgebra: norm
using Chairmarks: @b
using Printf: @sprintf

function get_backend()
    # To run benchmarks on a specific backend
    BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", nothing)

    if BENCHMARK_GROUP == "CUDA"
        Reactant.set_default_backend("gpu")
        @info "Running CUDA benchmarks" maxlog = 1
    elseif BENCHMARK_GROUP == "TPU"
        Reactant.set_default_backend("tpu")
    elseif BENCHMARK_GROUP == "CPU"
        Reactant.set_default_backend("cpu")
        @info "Running CPU benchmarks" maxlog = 1
    else
        BENCHMARK_GROUP = String(split(string(first(Reactant.devices())), ":")[1])
        @info "Running $(BENCHMARK_GROUP) benchmarks" maxlog = 1
    end
    return BENCHMARK_GROUP
end

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

        results[full_benchmark_name] = bench.time

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
                                                                        match ground truth"

            time = Reactant.Profiler.profile_with_xprof(
                f′, args_ra...; nrepeat=10, compile_options=compile_options
            )
            time = time.profiling_result.runtime_ns / 1e9
            results[full_benchmark_name] = time

            print_stmt = @sprintf "%100s     :     %.5gs" full_benchmark_name time
            @info print_stmt
            GC.gc(true)
        end
    end

    return nothing
end
