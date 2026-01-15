const REACTANT_ROOT = dirname(dirname(@__DIR__))
const DEFAULT_OUTPUT_DIR = joinpath(REACTANT_ROOT, "src", "xla", "PJRT")

const bazel_cmd = if !isnothing(Sys.which("bazelisk"))
    "bazelisk"
elseif !isnothing(Sys.which("bazel"))
    "bazel"
else
    error("Could not find `bazel` or `bazelisk` in PATH!")
end

function main(; output_dir::String=DEFAULT_OUTPUT_DIR, force::Bool=false)
    bazel_target = "julia_pjrt_capi_bindings"

    run(
        Cmd(
            `$(bazel_cmd) build
            --action_env=JULIA=$(Base.julia_cmd().exec[1])
            --jobs=$(Threads.nthreads())
            --action_env=JULIA_DEPOT_PATH=$(Base.DEPOT_PATH)
            --repo_env HERMETIC_PYTHON_VERSION="3.10"
            --check_visibility=false
            --verbose_failures
            //:$bazel_target
            `; dir=@__DIR__,
        ),
    )

    Base.Filesystem.cp(
        joinpath(@__DIR__, "bazel-bin", "CAPI.jl"), joinpath(output_dir, "CAPI.jl"); force
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    force = "--force" in ARGS || "-f" in ARGS

    # Filter out flags to get positional args
    positional_args = filter(a -> !startswith(a, "-"), ARGS)
    output_dir = isempty(positional_args) ? DEFAULT_OUTPUT_DIR : positional_args[1]

    main(; output_dir=output_dir, force=force)
end
