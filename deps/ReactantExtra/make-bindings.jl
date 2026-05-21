const bazel_cmd = if !isnothing(Sys.which("bazelisk"))
    "bazelisk"
elseif !isnothing(Sys.which("bazel"))
    "bazel"
else
    error("Could not find `bazel` or `bazelisk` in PATH!")
end

src_dir = joinpath(dirname(dirname(@__DIR__)), "src")

files = ["libMLIR_h.jl"]
bazel_targets = ["//:$f" for f in files]

# Build all targets simultaneously
run(
    Cmd(
        `$(bazel_cmd) build --action_env=JULIA=$(Base.julia_cmd().exec[1]) --jobs=$(Threads.nthreads()) --action_env=JULIA_DEPOT_PATH=$(Base.DEPOT_PATH) --repo_env HERMETIC_PYTHON_VERSION="3.12" --check_visibility=false --verbose_failures $bazel_targets`;
        dir=@__DIR__,
    ),
)

# Copy built files to their destinations
for file in other_files
    Base.Filesystem.cp(
        joinpath(@__DIR__, "bazel-bin", file), joinpath(src_dir, "mlir", file); force=true
    )
end
