using Reactant
using Reactant: ProbProg, ReactantRNG
using JSON3
using ArgParse

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -length(x) * log(σ) - length(x) / 2 * log(2π) -
           sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end

function model(rng, xs)
    _, param_a = ProbProg.sample(
        rng, normal, 0.0, 5.0, (1,); symbol=:param_a, logpdf=normal_logpdf
    )
    _, param_b = ProbProg.sample(
        rng, normal, 0.0, 5.0, (1,); symbol=:param_b, logpdf=normal_logpdf
    )

    _, ys_a = ProbProg.sample(
        rng, normal, param_a .+ xs[1:5], 0.5, (5,); symbol=:ys_a, logpdf=normal_logpdf
    )

    _, ys_b = ProbProg.sample(
        rng, normal, param_b .+ xs[6:10], 0.5, (5,); symbol=:ys_b, logpdf=normal_logpdf
    )

    return vcat(ys_a, ys_b)
end

struct BenchmarkResult
    name::String
    framework::String
    algorithm::String
    compile_time_s::Float64
    run_time_s::Float64
    param_a_final::Float64
    param_b_final::Float64
end

function hmc_program(
    rng,
    model,
    xs,
    step_size,
    num_steps::Int,
    inverse_mass_matrix,
    constraint,
    constrained_addresses,
)
    t, _, _ = ProbProg.generate(rng, constraint, model, xs; constrained_addresses)

    t, accepted, _ = ProbProg.mcmc(
        rng,
        t,
        model,
        xs;
        selection=ProbProg.select(ProbProg.Address(:param_a), ProbProg.Address(:param_b)),
        algorithm=:HMC,
        inverse_mass_matrix,
        step_size,
        num_steps,
    )

    return t, accepted
end

function nuts_program(
    rng,
    model,
    xs,
    step_size,
    max_tree_depth::Int,
    inverse_mass_matrix,
    constraint,
    constrained_addresses,
)
    t, _, _ = ProbProg.generate(rng, constraint, model, xs; constrained_addresses)

    t, accepted, _ = ProbProg.mcmc(
        rng,
        t,
        model,
        xs;
        selection=ProbProg.select(ProbProg.Address(:param_a), ProbProg.Address(:param_b)),
        algorithm=:NUTS,
        inverse_mass_matrix,
        step_size,
        max_tree_depth,
    )

    return t, accepted
end

function benchmark_hmc()
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)

    xs = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    ys_a = [-2.3, -1.6, -0.4, 0.6, 1.4]
    ys_b = [-2.6, -1.4, -0.6, 0.4, 1.6]

    constraint = ProbProg.Constraint(
        :param_a => ([0.0],),
        :param_b => ([0.0],),
        :ys_a => (ys_a,),
        :ys_b => (ys_b,),
    )
    constrained_addresses = ProbProg.extract_addresses(constraint)

    step_size = ConcreteRNumber(0.01)
    num_steps = 3
    inverse_mass_matrix = nothing

    seed_buffer = only(rng.seed.data).buffer

    compile_time = @elapsed begin
        compiled_fn = @compile optimize = :probprog hmc_program(
            rng,
            model,
            xs,
            step_size,
            num_steps,
            inverse_mass_matrix,
            constraint,
            constrained_addresses,
        )
    end

    local trace
    run_time = @elapsed begin
        GC.@preserve seed_buffer constraint begin
            trace, _ = compiled_fn(
                rng,
                model,
                xs,
                step_size,
                num_steps,
                inverse_mass_matrix,
                constraint,
                constrained_addresses,
            )
            trace = ProbProg.ProbProgTrace(trace)
        end
    end

    param_a_final = only(trace.choices[:param_a])[1]
    param_b_final = only(trace.choices[:param_b])[1]

    return BenchmarkResult(
        "NormalNormal/HMC",
        "Reactant",
        "HMC",
        compile_time,
        run_time,
        param_a_final,
        param_b_final,
    )
end

function benchmark_nuts()
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)

    xs = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    ys_a = [-2.3, -1.6, -0.4, 0.6, 1.4]
    ys_b = [-2.6, -1.4, -0.6, 0.4, 1.6]

    constraint = ProbProg.Constraint(
        :param_a => ([0.0],),
        :param_b => ([0.0],),
        :ys_a => (ys_a,),
        :ys_b => (ys_b,),
    )
    constrained_addresses = ProbProg.extract_addresses(constraint)

    step_size = ConcreteRNumber(0.001)
    max_tree_depth = 10
    inverse_mass_matrix = nothing

    seed_buffer = only(rng.seed.data).buffer

    compile_time = @elapsed begin
        compiled_fn = @compile optimize = :probprog nuts_program(
            rng,
            model,
            xs,
            step_size,
            max_tree_depth,
            inverse_mass_matrix,
            constraint,
            constrained_addresses,
        )
    end

    local trace
    run_time = @elapsed begin
        GC.@preserve seed_buffer constraint begin
            trace, _ = compiled_fn(
                rng,
                model,
                xs,
                step_size,
                max_tree_depth,
                inverse_mass_matrix,
                constraint,
                constrained_addresses,
            )
            trace = ProbProg.ProbProgTrace(trace)
        end
    end

    param_a_final = only(trace.choices[:param_a])[1]
    param_b_final = only(trace.choices[:param_b])[1]

    return BenchmarkResult(
        "NormalNormal/NUTS",
        "Reactant",
        "NUTS",
        compile_time,
        run_time,
        param_a_final,
        param_b_final,
    )
end

function run_benchmarks(test::String)
    results = BenchmarkResult[]

    println("=" ^ 70)
    println("Reactant Benchmark (matching test configuration)")
    println("=" ^ 70)
    println("Reactant version: ", pkgversion(Reactant))
    println("=" ^ 70)

    if test in ["hmc", "all"]
        print("\n[HMC] step_size=0.01, num_steps=3")
        result = benchmark_hmc()
        println("  compile=$(round(result.compile_time_s, digits=3))s, run=$(round(result.run_time_s, digits=3))s")
        println("  param_a=$(result.param_a_final), param_b=$(result.param_b_final)")
        push!(results, result)
    end

    if test in ["nuts", "all"]
        print("\n[NUTS] step_size=0.001")
        result = benchmark_nuts()
        println("  compile=$(round(result.compile_time_s, digits=3))s, run=$(round(result.run_time_s, digits=3))s")
        println("  param_a=$(result.param_a_final), param_b=$(result.param_b_final)")
        push!(results, result)
    end

    println("\n" * "=" ^ 70)
    println("Benchmark Complete")
    println("=" ^ 70)

    return results
end

function parse_args()
    s = ArgParseSettings(description="Reactant Benchmark")

    @add_arg_table! s begin
        "--test"
            arg_type = String
            default = "all"
            help = "Which test to run: hmc, nuts, or all"
        "--output"
            arg_type = String
            default = "reactant_results.json"
            help = "Output JSON file path"
    end

    return ArgParse.parse_args(s)
end

function main()
    args = parse_args()

    output_dir = dirname(args["output"])
    if !isempty(output_dir)
        mkpath(output_dir)
    end

    results = run_benchmarks(args["test"])

    results_json = [
        Dict(
            "name" => r.name,
            "framework" => r.framework,
            "algorithm" => r.algorithm,
            "compile_time_s" => r.compile_time_s,
            "run_time_s" => r.run_time_s,
            "param_a_final" => r.param_a_final,
            "param_b_final" => r.param_b_final,
        )
        for r in results
    ]

    output_data = Dict(
        "framework" => "Reactant",
        "reactant_version" => string(pkgversion(Reactant)),
        "results" => results_json,
    )

    open(args["output"], "w") do io
        JSON3.pretty(io, output_data)
    end

    println("\nResults saved to $(args["output"])")
end

main()
