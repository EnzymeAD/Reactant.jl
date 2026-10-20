include("../misc/common.jl")
include("workload.jl")
using .BrusselatorWorkload

function brusselator_compile_options(ad_options::Reactant.ADOptimizationOptions)
    return Reactant.CompileOptions(; sync=true, ad_optimization_passes=ad_options)
end

function ad_configurations(choice)
    configurations = (
        "Default" => Reactant.ADOptimizationOptions(),
        "Activity" => Reactant.ADOptimizationOptions(; activity=true),
        "DiffBatch" => Reactant.ADOptimizationOptions(; diff_batch=true),
        "RegionHoist" => Reactant.ADOptimizationOptions(; region_hoist=true),
        "AllAD" => Reactant.ADOptimizationOptions(;
            activity=true, diff_batch=true, region_hoist=true
        ),
    )
    choice == "compare" && return configurations
    choices = ("default", "activity", "batch", "hoist", "all")
    index = findfirst(==(choice), choices)
    index === nothing &&
        throw(ArgumentError("ad must be default, activity, batch, hoist, all, or compare"))
    return (configurations[index],)
end

function parse_command_line(args; default_mode="performance")
    options = Dict{String,String}()
    allowed = ("mode", "n", "samples", "ad", "backend", "output-dir", "damping")
    for argument in args
        pieces = split(argument, "="; limit=2)
        startswith(argument, "--") && length(pieces) == 2 ||
            throw(ArgumentError("expected --name=value, got $argument"))
        name = pieces[1][3:end]
        name in allowed || throw(ArgumentError("unknown option --$name"))
        options[name] = pieces[2]
    end
    mode = get(options, "mode", default_mode)
    mode in ("validation", "performance") ||
        throw(ArgumentError("mode must be validation or performance"))
    N = parse(Int, get(options, "n", mode == "validation" ? "4" : "8"))
    samples = parse(Int, get(options, "samples", "30"))
    damping = parse(Float64, get(options, "damping", "1.0"))
    configurations = ad_configurations(get(options, "ad", "default"))
    backend = get(options, "backend", "auto")
    backend in ("auto", "cpu", "gpu", "tpu") ||
        throw(ArgumentError("backend must be auto, cpu, gpu, or tpu"))
    N >= 3 ||
        throw(ArgumentError("n must be at least 3; the N=2 initial Jacobian is singular"))
    samples > 0 || throw(ArgumentError("samples must be positive"))
    isfinite(damping) && 0 < damping <= 1 ||
        throw(ArgumentError("damping must be finite and in (0, 1]"))
    output_dir = abspath(get(options, "output-dir", joinpath(@__DIR__, "results")))
    return (; mode, N, samples, damping, configurations, backend, output_dir)
end

function select_backend(backend)
    if backend != "auto"
        ENV["BENCHMARK_GROUP"] = backend == "gpu" ? "CUDA" : uppercase(backend)
    end
    return get_backend()
end
