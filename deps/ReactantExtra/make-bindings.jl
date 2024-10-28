JULIA_CMD = Base.julia_cmd().exec[1]
if haskey(ENV, "JULIA_DEPOT_PATH")
    JULIA_CMD = "JULIA_DEPOT_PATH=$(ENV["JULIA_DEPOT_PATH"]) $(JULIA_CMD)"
end

function build_file(output_path)
    file = basename(output_path)
    run(
        Cmd(
            `bazel build --action_env=JULIA=\"$JULIA_CMD\" --repo_env HERMETIC_PYTHON_VERSION="3.10" --check_visibility=false --verbose_failures //:$file`;
            dir=@__DIR__,
        ),
    )
    Base.Filesystem.cp(
        joinpath(@__DIR__, "bazel-bin", file),
        output_path;
        force=true,
    )
end

src_dir = joinpath(dirname(dirname(@__DIR__)), "src")

for file in [
    "Builtin.jl",
    "Arith.jl",
    "Affine.jl",
    "Func.jl",
    "Enzyme.jl",
    "StableHLO.jl",
    "CHLO.jl",
    "VHLO.jl",
]
    build_file(joinpath(src_dir, "mlir", "Dialects", file))
end

build_file(joinpath(src_dir, "mlir", "libMLIR_h.jl"))
