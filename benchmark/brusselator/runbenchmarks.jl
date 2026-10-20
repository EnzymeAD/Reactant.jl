include("common.jl")
include("brusselator.jl")
include("validate.jl")

function run_all_benchmarks(backend::String, options)
    results = Dict{String,Dict{String,Float64}}()
    run_brusselator_benchmark!(
        results,
        backend;
        N=options.N,
        damping=options.damping,
        nrepeat=options.samples,
        configurations=options.configurations,
    )
    return results
end

function main(args=ARGS)
    options = parse_command_line(args)
    @info sprint(io -> versioninfo(io; verbose=true))
    backend = select_backend(options.backend)
    if options.mode == "validation"
        for (name, ad_options) in options.configurations
            @info "Validate Brusselator" configuration = name
            validate_brusselator(; N=options.N, damping=options.damping, ad_options)
        end
        return nothing
    end
    results = run_all_benchmarks(backend, options)
    save_results(results, options.output_dir, "brusselator", backend)
    pretty_print_results(results, "brusselator", backend)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
elseif abspath(PROGRAM_FILE) == normpath(joinpath(@__DIR__, "..", "runbenchmarks.jl"))
    # The suite driver consumes its own backend and suite arguments.
    main(String[])
end
